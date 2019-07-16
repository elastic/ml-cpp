/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTree.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CPackedBitVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/operators.hpp>
#include <boost/optional.hpp>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

namespace {
std::size_t predictionColumn(std::size_t numberColumns) {
    return numberColumns - 3;
}
}

CBoostedTree::CBoostedTree(std::size_t numberThreads, std::size_t dependentVariable, TLossFunctionUPtr loss)
    : m_Impl{std::make_unique<CBoostedTreeImpl>(numberThreads, dependentVariable, std::move(loss))} {
}

CBoostedTree::~CBoostedTree() {
}

CBoostedTree& CBoostedTree::numberFolds(std::size_t folds) {
    m_Impl->numberFolds(folds);
    return *this;
}

CBoostedTree& CBoostedTree::lambda(double lambda) {
    m_Impl->lambda(lambda);
    return *this;
}

CBoostedTree& CBoostedTree::gamma(double gamma) {
    m_Impl->gamma(gamma);
    return *this;
}

CBoostedTree& CBoostedTree::eta(double eta) {
    m_Impl->eta(eta);
    return *this;
}

CBoostedTree& CBoostedTree::maximumNumberTrees(std::size_t maximumNumberTrees) {
    m_Impl->maximumNumberTrees(maximumNumberTrees);
    return *this;
}

CBoostedTree& CBoostedTree::rowsPerFeature(std::size_t rowsPerFeature) {
    m_Impl->rowsPerFeature(rowsPerFeature);
    return *this;
}

CBoostedTree& CBoostedTree::featureBagFraction(double featureBagFraction) {
    m_Impl->featureBagFraction(featureBagFraction);
    return *this;
}

CBoostedTree& CBoostedTree::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    m_Impl->maximumOptimisationRoundsPerHyperparameter(rounds);
    return *this;
}

void CBoostedTree::train(core::CDataFrame& frame, TProgressCallback recordProgress) {
    m_Impl->train(frame, recordProgress);
}

void CBoostedTree::predict(core::CDataFrame& frame, TProgressCallback recordProgress) const {
    m_Impl->predict(frame, recordProgress);
}

void CBoostedTree::write(core::CRapidJsonConcurrentLineWriter& writer) const {
    m_Impl->write(writer);
}

CBoostedTree::TDoubleVec CBoostedTree::featureWeights() const {
    return m_Impl->featureWeights();
}

std::size_t CBoostedTree::numberExtraColumnsForTrain() const {
    return m_Impl->numberExtraColumnsForTrain();
}

std::size_t CBoostedTree::columnHoldingPrediction(std::size_t numberColumns) const {
    return predictionColumn(numberColumns);
}
std::size_t CBoostedTree::estimateMemoryUsage(std::size_t numberRows,
                                              std::size_t numberColumns) const {
    return this->m_Impl->estimateMemoryUsage(numberRows, numberColumns);
}
}
}
