/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTree.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>

#include <maths/CBoostedTreeImpl.h>

#include <sstream>
#include <utility>

namespace ml {
namespace maths {
namespace boosted_tree_detail {

std::unique_ptr<CArgMinLossImpl> CArgMinMseImpl::clone() const {
    return std::make_unique<CArgMinMseImpl>(*this);
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
    return CBasicStatistics::mean(m_MeanError);
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

CArgMinLoss CMse::minimizer() const {
    return this->makeMinimizer(CArgMinMseImpl{});
}

const std::string& CMse::name() const {
    return NAME;
}

const std::string CMse::NAME{"mse"};
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

namespace {
const std::string BOOSTED_TREE_IMPL_TAG{"boosted_tree_impl"};
}

bool CBoostedTree::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CBoostedTree::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}
}
}
