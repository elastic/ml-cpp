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

#include <limits>
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
const std::string NUMBER_SAMPLES_TAG{"number_samples"};

double LOG_EPSILON{std::log(100.0 * std::numeric_limits<double>::epsilon())};

double logOneMinusLogistic(double logOdds) {
    // For large x logistic(x) = 1 - e^(-x) + O(e^(-2x))
    if (logOdds > -LOG_EPSILON) {
        return -logOdds;
    }
    return CTools::fastLog(1.0 - CTools::logisticFunction(logOdds));
}

double logLogistic(double logOdds) {
    // For small x logistic(x) = e^(x) + O(e^(2x))
    if (logOdds < LOG_EPSILON) {
        return logOdds;
    }
    return CTools::fastLog(CTools::logisticFunction(logOdds));
}
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

void CArgMinMseImpl::add(const TMemoryMappedFloatVector& prediction, double actual, double weight) {
    m_MeanError.add(actual - prediction(0), weight);
}

void CArgMinMseImpl::merge(const CArgMinLossImpl& other) {
    const auto* mse = dynamic_cast<const CArgMinMseImpl*>(&other);
    if (mse != nullptr) {
        m_MeanError += mse->m_MeanError;
    }
}

CArgMinMseImpl::TDoubleVector CArgMinMseImpl::value() const {

    // We searching for the value x which minimises
    //
    //    x^* = argmin_x{ sum_i{(a_i - (p_i + x))^2} + lambda * x^2 }
    //
    // This is convex so there is one minimum where the derivative w.r.t. x is zero
    // and x^* = 1 / (n + lambda) sum_i{ a_i - p_i }. Denoting the mean prediction
    // error m = 1/n sum_i{ a_i - p_i } we have x^* = n / (n + lambda) m.

    double count{CBasicStatistics::count(m_MeanError)};
    double meanError{CBasicStatistics::mean(m_MeanError)};

    TDoubleVector result(1);
    result(0) = count == 0.0 ? 0.0 : count / (count + this->lambda()) * meanError;
    return result;
}

CArgMinLogisticImpl::CArgMinLogisticImpl(double lambda)
    : CArgMinLossImpl{lambda}, m_CategoryCounts{0},
      m_BucketCategoryCounts(128, TDoubleVector2x1{0.0}) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinLogisticImpl::clone() const {
    return std::make_unique<CArgMinLogisticImpl>(*this);
}

bool CArgMinLogisticImpl::nextPass() {
    m_CurrentPass += this->bucketWidth() > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinLogisticImpl::add(const TMemoryMappedFloatVector& prediction,
                              double actual,
                              double weight) {
    switch (m_CurrentPass) {
    case 0: {
        m_PredictionMinMax.add(prediction(0));
        m_CategoryCounts(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    case 1: {
        auto& count = m_BucketCategoryCounts[this->bucket(prediction(0))];
        count(static_cast<std::size_t>(actual)) += weight;
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
            m_PredictionMinMax += logistic->m_PredictionMinMax;
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

CArgMinLogisticImpl::TDoubleVector CArgMinLogisticImpl::value() const {

    std::function<double(double)> objective;
    double minWeight;
    double maxWeight;

    // This is true if and only if all the predictions were identical. In this
    // case we only need one pass over the data and can compute the optimal
    // value from the counts of the two categories.
    if (this->bucketWidth() == 0.0) {
        objective = [this](double weight) {
            double c0{m_CategoryCounts(0)};
            double c1{m_CategoryCounts(1)};
            return this->lambda() * CTools::pow2(weight) -
                   c0 * logOneMinusLogistic(weight) - c1 * logLogistic(weight);
        };

        // Weight shrinkage means the optimal weight will be somewhere between
        // the logit of the empirical probability and zero.
        double c0{m_CategoryCounts(0) + 1.0};
        double c1{m_CategoryCounts(1) + 1.0};
        double empiricalProbabilityC1{c1 / (c0 + c1)};
        double empiricalLogOddsC1{
            std::log(empiricalProbabilityC1 / (1.0 - empiricalProbabilityC1))};
        minWeight = empiricalProbabilityC1 < 0.5 ? empiricalLogOddsC1 : 0.0;
        maxWeight = empiricalProbabilityC1 < 0.5 ? 0.0 : empiricalLogOddsC1;

    } else {
        objective = [this](double weight) {
            double loss{0.0};
            for (std::size_t i = 0; i < m_BucketCategoryCounts.size(); ++i) {
                double logOdds{this->bucketCentre(i) + weight};
                double c0{m_BucketCategoryCounts[i](0)};
                double c1{m_BucketCategoryCounts[i](1)};
                loss -= c0 * logOneMinusLogistic(logOdds) + c1 * logLogistic(logOdds);
            }
            return loss + this->lambda() * CTools::pow2(weight);
        };

        // Choose a weight interval in which all probabilites vary from close to
        // zero to close to one. In particular, the idea is to minimize the leaf
        // weight on an interval [a, b] where if we add "a" the log-odds for all
        // rows <= -5, i.e. max prediction + a = -5, and if we add "b" the log-odds
        // for all rows >= 5, i.e. min prediction + a = 5.
        minWeight = -m_PredictionMinMax.max() - 5.0;
        maxWeight = -m_PredictionMinMax.min() + 5.0;
    }

    TDoubleVector result(1);

    if (minWeight == maxWeight) {
        result(0) = minWeight;
        return result;
    }

    double minimum;
    double objectiveAtMinimum;
    std::size_t maxIterations{10};
    CSolvers::minimize(minWeight, maxWeight, objective(minWeight), objective(maxWeight),
                       objective, 1e-3, maxIterations, minimum, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimum << " objective(minimum) = " << objectiveAtMinimum);

    result(0) = minimum;
    return result;
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

void CArgMinLoss::add(const TMemoryMappedFloatVector& prediction, double actual, double weight) {
    return m_Impl->add(prediction, actual, weight);
}

void CArgMinLoss::merge(CArgMinLoss& other) {
    return m_Impl->merge(*other.m_Impl);
}

CArgMinLoss::TDoubleVector CArgMinLoss::value() const {
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

std::size_t CMse::numberParameters() const {
    return 1;
}

double CMse::value(const TMemoryMappedFloatVector& prediction, double actual, double weight) const {
    return weight * CTools::pow2(prediction(0) - actual);
}

void CMse::gradient(const TMemoryMappedFloatVector& prediction,
                    double actual,
                    TWriter writer,
                    double weight) const {
    writer(0, 2.0 * weight * (prediction(0) - actual));
}

void CMse::curvature(const TMemoryMappedFloatVector& /*prediction*/,
                     double /*actual*/,
                     TWriter writer,
                     double weight) const {
    writer(0, 2.0 * weight);
}

bool CMse::isCurvatureConstant() const {
    return true;
}

CMse::TDoubleVector CMse::transform(const TMemoryMappedFloatVector& prediction) const {
    return TDoubleVector{prediction};
}

CArgMinLoss CMse::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinMseImpl{lambda});
}

const std::string& CMse::name() const {
    return NAME;
}

const std::string CMse::NAME{"mse"};

std::unique_ptr<CLoss> CBinomialLogistic::clone() const {
    return std::make_unique<CBinomialLogistic>(*this);
}

std::size_t CBinomialLogistic::numberParameters() const {
    return 1;
}

double CBinomialLogistic::value(const TMemoryMappedFloatVector& prediction,
                                double actual,
                                double weight) const {
    return -weight * ((1.0 - actual) * logOneMinusLogistic(prediction(0)) +
                      actual * logLogistic(prediction(0)));
}

void CBinomialLogistic::gradient(const TMemoryMappedFloatVector& prediction,
                                 double actual,
                                 TWriter writer,
                                 double weight) const {
    if (prediction(0) > -LOG_EPSILON && actual == 1.0) {
        writer(0, -weight * std::exp(-prediction(0)));
    }
    writer(0, weight * (CTools::logisticFunction(prediction(0)) - actual));
}

void CBinomialLogistic::curvature(const TMemoryMappedFloatVector& prediction,
                                  double /*actual*/,
                                  TWriter writer,
                                  double weight) const {
    if (prediction(0) > -LOG_EPSILON) {
        writer(0, weight * std::exp(-prediction(0)));
    }
    double probability{CTools::logisticFunction(prediction(0))};
    writer(0, weight * probability * (1.0 - probability));
}

bool CBinomialLogistic::isCurvatureConstant() const {
    return false;
}

CBinomialLogistic::TDoubleVector
CBinomialLogistic::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{prediction};
    result(0) = CTools::logisticFunction(result(0));
    return result;
}

CArgMinLoss CBinomialLogistic::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinLogisticImpl{lambda});
}

const std::string& CBinomialLogistic::name() const {
    return NAME;
}

const std::string CBinomialLogistic::NAME{"binomial_logistic"};
}

CBoostedTreeNode::TNodeIndex CBoostedTreeNode::leafIndex(const CEncodedDataFrameRowRef& row,
                                                         const TNodeVec& tree,
                                                         TNodeIndex index) const {
    if (this->isLeaf()) {
        return index;
    }
    double value{row[m_SplitFeature]};
    bool missing{CDataFrameUtils::isMissing(value)};
    return (missing && m_AssignMissingToLeft) || (missing == false && value < m_SplitValue)
               ? tree[m_LeftChild.get()].leafIndex(row, tree, m_LeftChild.get())
               : tree[m_RightChild.get()].leafIndex(row, tree, m_RightChild.get());
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
    m_LeftChild = static_cast<TNodeIndex>(tree.size());
    m_RightChild = static_cast<TNodeIndex>(tree.size() + 1);
    m_Gain = gain;
    m_Curvature = curvature;
    TSizeSizePr result{m_LeftChild.get(), m_RightChild.get()};
    // Don't access members after calling resize because this object is likely an element of the vector being resized.
    tree.resize(tree.size() + 2);
    return result;
}

void CBoostedTreeNode::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(LEFT_CHILD_TAG, m_LeftChild, inserter);
    core::CPersistUtils::persist(RIGHT_CHILD_TAG, m_RightChild, inserter);
    core::CPersistUtils::persist(SPLIT_FEATURE_TAG, m_SplitFeature, inserter);
    core::CPersistUtils::persist(ASSIGN_MISSING_TO_LEFT_TAG, m_AssignMissingToLeft, inserter);
    core::CPersistUtils::persist(NODE_VALUE_TAG, m_NodeValue, inserter);
    core::CPersistUtils::persist(SPLIT_VALUE_TAG, m_SplitValue, inserter);
    core::CPersistUtils::persist(NUMBER_SAMPLES_TAG, m_NumberSamples, inserter);
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
        RESTORE(NUMBER_SAMPLES_TAG,
                core::CPersistUtils::restore(NUMBER_SAMPLES_TAG, m_NumberSamples, traverser))
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
        tree[m_LeftChild.get()].doPrint(pad + "  ", tree, result);
        tree[m_RightChild.get()].doPrint(pad + "  ", tree, result);
    }
    return result;
}

void CBoostedTreeNode::accept(CVisitor& visitor) const {
    visitor.addNode(m_SplitFeature, m_SplitValue, m_AssignMissingToLeft, m_NodeValue,
                    m_Gain, m_NumberSamples, m_LeftChild, m_RightChild);
}

void CBoostedTreeNode::numberSamples(std::size_t numberSamples) {
    m_NumberSamples = numberSamples;
}

std::size_t CBoostedTreeNode::numberSamples() const {
    return m_NumberSamples;
}

CBoostedTree::CBoostedTree(core::CDataFrame& frame,
                           TTrainingStateCallback recordTrainingState,
                           TImplUPtr&& impl)
    : CDataFramePredictiveModel{frame, std::move(recordTrainingState)}, m_Impl{std::move(impl)} {
}

CBoostedTree::~CBoostedTree() = default;

void CBoostedTree::train() {
    m_Impl->train(this->frame(), this->trainingStateRecorder());
}

void CBoostedTree::predict() const {
    m_Impl->predict(this->frame());
}

void CBoostedTree::computeShapValues() {
    m_Impl->computeShapValues(this->frame());
}

std::size_t CBoostedTree::columnHoldingDependentVariable() const {
    return m_Impl->columnHoldingDependentVariable();
}

std::size_t CBoostedTree::columnHoldingPrediction() const {
    return predictionColumn(m_Impl->numberInputColumns());
}

double CBoostedTree::probabilityAtWhichToAssignClassOne() const {
    return m_Impl->probabilityAtWhichToAssignClassOne();
}

const CBoostedTree::TNodeVecVec& CBoostedTree::trainedModel() const {
    return m_Impl->trainedModel();
}

const CBoostedTree::TDoubleVec& CBoostedTree::featureWeightsForTraining() const {
    return m_Impl->featureSampleProbabilities();
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

void CBoostedTree::accept(CBoostedTree::CVisitor& visitor) const {
    m_Impl->accept(visitor);
}

const CBoostedTreeHyperparameters& CBoostedTree::bestHyperparameters() const {
    return m_Impl->bestHyperparameters();
}

CBoostedTree::TSizeRange CBoostedTree::columnsHoldingShapValues() const {
    return m_Impl->columnsHoldingShapValues();
}

std::size_t CBoostedTree::topShapValues() const {
    if (m_Impl) {
        return m_Impl->topShapValues();
    }
    return 0;
}
}
}
