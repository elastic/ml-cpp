/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTree.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>

#include <maths/CBoostedTreeImpl.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>

#include <maths/CMathsFuncs.h>

#include <sstream>
#include <utility>

namespace ml {
namespace maths {
namespace {
const std::string LEFT_CHILD_TAG{"left_child"};
const std::string RIGHT_CHILD_TAG{"right_child"};
const std::string SPLIT_FEATURE_TAG{"split_feature"};
const std::string ASSIGN_MISSING_TO_LEFT_TAG{"assign_missing_to_left "};
const std::string NODE_VALUE_TAG{"node_value"};
const std::string SPLIT_VALUE_TAG{"split_value"};
}

namespace boosted_tree_detail {

CArgMinLossImpl::CArgMinLossImpl(double lambda) : m_Lambda{lambda} {
}

double CArgMinLossImpl::lambda() const {
    return m_Lambda;
}

CArgMinMseImpl::CArgMinMseImpl(double lambda) : CArgMinLossImpl{lambda} {
}

std::unique_ptr<CArgMinLossImpl> CArgMinMseImpl::clone() const {
    return std::make_unique<CArgMinMseImpl>(*this);
}

bool CArgMinMseImpl::nextPass() {
    return false;
}

void CArgMinMseImpl::add(double prediction, double actual) {
    m_MeanError.add(actual - prediction);
}

void CArgMinMseImpl::merge(const CArgMinLossImpl& other) {
    const auto* mse = dynamic_cast<const CArgMinMseImpl*>(&other);
    if (mse != nullptr) {
        m_MeanError += mse->m_MeanError;
    }
}

double CArgMinMseImpl::value() const {
    double count{CBasicStatistics::count(m_MeanError)};
    return count == 0.0
               ? 0.0
               : count / (count + this->lambda()) * CBasicStatistics::mean(m_MeanError);
}

CArgMinLogisticImpl::CArgMinLogisticImpl(double lambda)
    : CArgMinLossImpl{lambda}, m_CategoryCounts{0},
      m_BucketCategoryCounts(128, TSizeVector{0}) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinLogisticImpl::clone() const {
    return std::make_unique<CArgMinLogisticImpl>(*this);
}

bool CArgMinLogisticImpl::nextPass() {
    m_CurrentPass += this->bucketWidth() > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinLogisticImpl::add(double prediction, double actual) {
    switch (m_CurrentPass) {
    case 0: {
        m_MinMaxPrediction.add(prediction);
        ++m_CategoryCounts(static_cast<std::size_t>(actual));
        break;
    }
    case 1: {
        auto& count = m_BucketCategoryCounts[this->bucket(prediction)];
        ++count(static_cast<std::size_t>(actual));
        break;
    }
    default:
        break;
    }
}

void CArgMinLogisticImpl::merge(const CArgMinLossImpl& other) {
    const auto* logistic = dynamic_cast<const CArgMinLogisticImpl*>(&other);
    if (logistic != nullptr) {
        switch (m_CurrentPass) {
        case 0:
            m_MinMaxPrediction += logistic->m_MinMaxPrediction;
            m_CategoryCounts += logistic->m_CategoryCounts;
            break;
        case 1:
            for (std::size_t i = 0; i < m_BucketCategoryCounts.size(); ++i) {
                m_BucketCategoryCounts[i] += logistic->m_BucketCategoryCounts[i];
            }
            break;
        default:
            break;
        }
    }
}

double CArgMinLogisticImpl::value() const {

    std::function<double(double)> objective;
    double minWeight;
    double maxWeight;

    if (this->bucketWidth() == 0.0) {
        objective = [this](double weight) {
            double p{CTools::logisticFunction(weight)};
            std::size_t c0{m_CategoryCounts(0)};
            std::size_t c1{m_CategoryCounts(1)};
            return this->lambda() * CTools::pow2(weight) -
                   static_cast<double>(c0) * CTools::fastLog(1.0 - p) -
                   static_cast<double>(c1) * CTools::fastLog(p);
        };

        // Weight shrinkage means the optimal weight will be somewhere
        // between the logit of the empirical probability and zero.
        std::size_t c0{m_CategoryCounts(0) + 1};
        std::size_t c1{m_CategoryCounts(1) + 1};
        double p{static_cast<double>(c1) / static_cast<double>(c0 + c1)};
        minWeight = p < 0.5 ? std::log(p / (1.0 - p)) : 0.0;
        maxWeight = p < 0.5 ? 0.0 : std::log(p / (1.0 - p));

    } else {
        objective = [this](double weight) {
            double loss{0.0};
            for (std::size_t i = 0; i < m_BucketCategoryCounts.size(); ++i) {
                double bucketPrediction{this->bucketCentre(i)};
                double p{CTools::logisticFunction(bucketPrediction + weight)};
                std::size_t c0{m_BucketCategoryCounts[i](0)};
                std::size_t c1{m_BucketCategoryCounts[i](1)};
                loss -= static_cast<double>(c0) * CTools::fastLog(1.0 - p) +
                        static_cast<double>(c1) * CTools::fastLog(p);
            }
            return loss + this->lambda() * CTools::pow2(weight);
        };

        // Choose a weight interval in which all probabilites vary from close to
        // zero to close to one.
        minWeight = -m_MinMaxPrediction.max() - 2.0;
        maxWeight = -m_MinMaxPrediction.min() + 2.0;
    }

    if (minWeight == maxWeight) {
        return minWeight;
    }

    double minimum;
    double objectiveAtMinimum;
    std::size_t maxIterations{10};
    CSolvers::minimize(minWeight, maxWeight, objective(minWeight), objective(maxWeight),
                       objective, 1e-3, maxIterations, minimum, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimum << " objective(minimum) = " << objectiveAtMinimum);

    return minimum;
}
}

using namespace boosted_tree_detail;
namespace boosted_tree {

CArgMinLoss::CArgMinLoss(const CArgMinLoss& other)
    : m_Impl{other.m_Impl->clone()} {
}

CArgMinLoss& CArgMinLoss::operator=(const CArgMinLoss& other) {
    if (this != &other) {
        m_Impl = other.m_Impl->clone();
    }
    return *this;
}

bool CArgMinLoss::nextPass() const {
    return m_Impl->nextPass();
}

void CArgMinLoss::add(double prediction, double actual) {
    return m_Impl->add(prediction, actual);
}

void CArgMinLoss::merge(CArgMinLoss& other) {
    return m_Impl->merge(*other.m_Impl);
}

double CArgMinLoss::value() const {
    return m_Impl->value();
}

CArgMinLoss::CArgMinLoss(const CArgMinLossImpl& impl) : m_Impl{impl.clone()} {
}

CArgMinLoss CLoss::makeMinimizer(const boosted_tree_detail::CArgMinLossImpl& impl) const {
    return {impl};
}

std::unique_ptr<CLoss> CMse::clone() const {
    return std::make_unique<CMse>(*this);
}

double CMse::value(double prediction, double actual) const {
    return CTools::pow2(prediction - actual);
}

double CMse::gradient(double prediction, double actual) const {
    return 2.0 * (prediction - actual);
}

double CMse::curvature(double /*prediction*/, double /*actual*/) const {
    return 2.0;
}

bool CMse::isCurvatureConstant() const {
    return true;
}

CArgMinLoss CMse::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinMseImpl{lambda});
}

const std::string& CMse::name() const {
    return NAME;
}

const std::string CMse::NAME{"mse"};

std::unique_ptr<CLoss> CLogistic::clone() const {
    return std::make_unique<CLogistic>(*this);
}

double CLogistic::value(double prediction, double actual) const {
    // Cross entropy
    prediction = CTools::logisticFunction(prediction);
    return -((1.0 - actual) * CTools::fastLog(1.0 - prediction) +
             actual * CTools::fastLog(prediction));
}

double CLogistic::gradient(double prediction, double actual) const {
    prediction = CTools::logisticFunction(prediction);
    return prediction - actual;
}

double CLogistic::curvature(double prediction, double /*actual*/) const {
    prediction = CTools::logisticFunction(prediction);
    return prediction * (1.0 - prediction);
}

bool CLogistic::isCurvatureConstant() const {
    return false;
}

CArgMinLoss CLogistic::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinLogisticImpl{lambda});
}

const std::string& CLogistic::name() const {
    return NAME;
}

const std::string CLogistic::NAME{"logistic"};
}

std::size_t CBoostedTreeNode::leafIndex(const CEncodedDataFrameRowRef& row,
                                        const TNodeVec& tree,
                                        std::int32_t index) const {
    if (this->isLeaf()) {
        return index;
    }
    double value{row[m_SplitFeature]};
    bool missing{CDataFrameUtils::isMissing(value)};
    return (missing && m_AssignMissingToLeft) || (missing == false && value < m_SplitValue)
               ? tree[m_LeftChild].leafIndex(row, tree, m_LeftChild)
               : tree[m_RightChild].leafIndex(row, tree, m_RightChild);
}

CBoostedTreeNode::TSizeSizePr CBoostedTreeNode::split(std::size_t splitFeature,
                                                      double splitValue,
                                                      bool assignMissingToLeft,
                                                      double gain,
                                                      double curvature,
                                                      TNodeVec& tree) {
    m_SplitFeature = splitFeature;
    m_SplitValue = splitValue;
    m_AssignMissingToLeft = assignMissingToLeft;
    m_LeftChild = static_cast<std::int32_t>(tree.size());
    m_RightChild = static_cast<std::int32_t>(tree.size() + 1);
    m_Gain = gain;
    m_Curvature = curvature;
    tree.resize(tree.size() + 2);
    return {m_LeftChild, m_RightChild};
}

CBoostedTreeNode::TPackedBitVectorPackedBitVectorBoolTr
CBoostedTreeNode::childrenRowMasks(std::size_t numberThreads,
                                   const core::CDataFrame& frame,
                                   const CDataFrameCategoryEncoder& encoder,
                                   core::CPackedBitVector rowMask) const {

    LOG_TRACE(<< "Splitting feature '" << m_SplitFeature << "' @ " << m_SplitValue);
    LOG_TRACE(<< "# rows in node = " << rowMask.manhattan());
    LOG_TRACE(<< "row mask = " << rowMask);

    using TRowItr = core::CDataFrame::TRowItr;

    auto result = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](auto& state, TRowItr beginRows, TRowItr endRows) {
                core::CPackedBitVector& leftRowMask{std::get<0>(state)};
                std::size_t& leftChildNumberRows{std::get<1>(state)};
                std::size_t& rightChildNumberRows{std::get<2>(state)};
                for (auto row = beginRows; row != endRows; ++row) {
                    std::size_t index{row->index()};
                    double value{encoder.encode(*row)[m_SplitFeature]};
                    bool missing{CDataFrameUtils::isMissing(value)};
                    if ((missing && m_AssignMissingToLeft) ||
                        (missing == false && value < m_SplitValue)) {
                        leftRowMask.extend(false, index - leftRowMask.size());
                        leftRowMask.extend(true);
                        ++leftChildNumberRows;
                    } else {
                        ++rightChildNumberRows;
                    }
                }
            },
            std::make_tuple(core::CPackedBitVector{}, std::size_t{0}, std::size_t{0})),
        &rowMask);
    auto& masks = result.first;

    for (auto& mask_ : masks) {
        auto& mask = std::get<0>(mask_.s_FunctionState);
        mask.extend(false, rowMask.size() - mask.size());
    }

    core::CPackedBitVector leftRowMask;
    std::size_t leftChildNumberRows;
    std::size_t rightChildNumberRows;
    std::tie(leftRowMask, leftChildNumberRows, rightChildNumberRows) =
        std::move(masks[0].s_FunctionState);
    for (std::size_t i = 1; i < masks.size(); ++i) {
        leftRowMask |= std::get<0>(masks[i].s_FunctionState);
        leftChildNumberRows += std::get<1>(masks[i].s_FunctionState);
        rightChildNumberRows += std::get<2>(masks[i].s_FunctionState);
    }
    LOG_TRACE(<< "# rows in left node = " << leftRowMask.manhattan());
    LOG_TRACE(<< "left row mask = " << leftRowMask);

    core::CPackedBitVector rightRowMask{std::move(rowMask)};
    rightRowMask ^= leftRowMask;
    LOG_TRACE(<< "# rows in right node = " << rightRowMask.manhattan());
    LOG_TRACE(<< "left row mask = " << rightRowMask);

    return std::make_tuple(std::move(leftRowMask), std::move(rightRowMask),
                           leftChildNumberRows < rightChildNumberRows);
}

void CBoostedTreeNode::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(LEFT_CHILD_TAG, m_LeftChild, inserter);
    core::CPersistUtils::persist(RIGHT_CHILD_TAG, m_RightChild, inserter);
    core::CPersistUtils::persist(SPLIT_FEATURE_TAG, m_SplitFeature, inserter);
    core::CPersistUtils::persist(ASSIGN_MISSING_TO_LEFT_TAG, m_AssignMissingToLeft, inserter);
    core::CPersistUtils::persist(NODE_VALUE_TAG, m_NodeValue, inserter);
    core::CPersistUtils::persist(SPLIT_VALUE_TAG, m_SplitValue, inserter);
}

bool CBoostedTreeNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(LEFT_CHILD_TAG,
                core::CPersistUtils::restore(LEFT_CHILD_TAG, m_LeftChild, traverser))
        RESTORE(RIGHT_CHILD_TAG,
                core::CPersistUtils::restore(RIGHT_CHILD_TAG, m_RightChild, traverser))
        RESTORE(SPLIT_FEATURE_TAG,
                core::CPersistUtils::restore(SPLIT_FEATURE_TAG, m_SplitFeature, traverser))
        RESTORE(ASSIGN_MISSING_TO_LEFT_TAG,
                core::CPersistUtils::restore(ASSIGN_MISSING_TO_LEFT_TAG,
                                             m_AssignMissingToLeft, traverser))
        RESTORE(NODE_VALUE_TAG,
                core::CPersistUtils::restore(NODE_VALUE_TAG, m_NodeValue, traverser))
        RESTORE(SPLIT_VALUE_TAG,
                core::CPersistUtils::restore(SPLIT_VALUE_TAG, m_SplitValue, traverser))
    } while (traverser.next());
    return true;
}

std::string CBoostedTreeNode::print(const TNodeVec& tree) const {
    std::ostringstream result;
    return this->doPrint("", tree, result).str();
}

std::ostringstream& CBoostedTreeNode::doPrint(std::string pad,
                                              const TNodeVec& tree,
                                              std::ostringstream& result) const {
    result << "\n" << pad;
    if (this->isLeaf()) {
        result << m_NodeValue;
    } else {
        result << "split feature '" << m_SplitFeature << "' @ " << m_SplitValue;
        tree[m_LeftChild].doPrint(pad + "  ", tree, result);
        tree[m_RightChild].doPrint(pad + "  ", tree, result);
    }
    return result;
}

CBoostedTree::CBoostedTree(core::CDataFrame& frame,
                           TProgressCallback recordProgress,
                           TMemoryUsageCallback recordMemoryUsage,
                           TTrainingStateCallback recordTrainingState,
                           TImplUPtr&& impl)
    : CDataFrameRegressionModel{frame, std::move(recordProgress),
                                std::move(recordMemoryUsage),
                                std::move(recordTrainingState)},
      m_Impl{std::move(impl)} {
}

CBoostedTree::~CBoostedTree() = default;

void CBoostedTree::train() {
    m_Impl->train(this->frame(), this->progressRecorder(),
                  this->memoryUsageRecorder(), this->trainingStateRecorder());
}

void CBoostedTree::predict() const {
    m_Impl->predict(this->frame(), this->progressRecorder());
}

void CBoostedTree::write(core::CRapidJsonConcurrentLineWriter& writer) const {
    m_Impl->write(writer);
}

const CBoostedTree::TDoubleVec& CBoostedTree::featureWeights() const {
    return m_Impl->featureWeights();
}

std::size_t CBoostedTree::columnHoldingDependentVariable() const {
    return m_Impl->columnHoldingDependentVariable();
}

std::size_t CBoostedTree::columnHoldingPrediction(std::size_t numberColumns) const {
    return predictionColumn(numberColumns);
}

const CBoostedTree::TNodeVecVec& CBoostedTree::trainedModel() const {
    return m_Impl->trainedModel();
}

const std::string& CBoostedTree::bestHyperparametersName() {
    return CBoostedTreeImpl::bestHyperparametersName();
}

const std::string& CBoostedTree::bestRegularizationHyperparametersName() {
    return CBoostedTreeImpl::bestRegularizationHyperparametersName();
}

CBoostedTree::TStrVec CBoostedTree::bestHyperparameterNames() {
    return CBoostedTreeImpl::bestHyperparameterNames();
}

bool CBoostedTree::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CBoostedTree::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}
}
}
