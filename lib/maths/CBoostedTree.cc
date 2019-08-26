/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTree.h>

#include <core/CDataFrame.h>

#include <maths/CBoostedTreeImpl.h>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;

namespace boosted_tree {

void CArgMinLoss::add(double prediction, double actual) {
    std::unique_lock<std::mutex> lock{m_Mutex};
    this->addImpl(prediction, actual);
}

double CMse::value(double prediction, double actual) const {
    return CTools::pow2(prediction - actual);
}

double CMse::gradient(double prediction, double actual) const {
    return prediction - actual;
}

double CMse::curvature(double /*prediction*/, double /*actual*/) const {
    return 2.0;
}

bool CMse::isCurvatureConstant() const {
    return true;
}

const std::string CMse::NAME{"mse"};

const std::string& CMse::name() const {
    return NAME;
}

void CArgMinMse::addImpl(double prediction, double actual) {
    m_MeanError.add(actual - prediction);
}

CMse::TArgMinLossUPtr CMse::minimizer() const {
    return std::make_unique<CArgMinMse>();
}

double CArgMinMse::value() const {
    return CBasicStatistics::mean(m_MeanError);
}
}

CBoostedTree::CBoostedTree(core::CDataFrame& frame, TProgressCallback recordProgress, TImplUPtr&& impl)
    : CDataFrameRegressionModel{frame, std::move(recordProgress)}, m_Impl{std::move(impl)} {
}

CBoostedTree::~CBoostedTree() = default;

void CBoostedTree::train() {
    m_Impl->train(this->frame(), this->progressRecorder());
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
    do {
        const std::string& name = traverser.name();
        RESTORE(BOOSTED_TREE_IMPL_TAG,
                core::CPersistUtils::restore(BOOSTED_TREE_IMPL_TAG, *m_Impl, traverser))
    } while (traverser.next());
    return true;
}

void CBoostedTree::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(BOOSTED_TREE_IMPL_TAG, *m_Impl, inserter);
}
}
}
