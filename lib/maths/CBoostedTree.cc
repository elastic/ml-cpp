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

CBoostedTree::CBoostedTree(core::CDataFrame& frame, TImplUPtr& impl)
    : CDataFrameRegressionModel(frame), m_Impl{std::move(impl)} {
}

CBoostedTree::~CBoostedTree() = default;

void CBoostedTree::train(TProgressCallback recordProgress) {
    m_Impl->train(this->frame(), recordProgress);
}

void CBoostedTree::predict(TProgressCallback recordProgress) const {
    m_Impl->predict(this->frame(), recordProgress);
}

void CBoostedTree::write(core::CRapidJsonConcurrentLineWriter& writer) const {
    m_Impl->write(writer);
}

CBoostedTree::TDoubleVec CBoostedTree::featureWeights() const {
    return m_Impl->featureWeights();
}

std::size_t CBoostedTree::columnHoldingPrediction(std::size_t numberColumns) const {
    return predictionColumn(numberColumns);
}
}
}
