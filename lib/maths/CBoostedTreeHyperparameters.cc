/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeHyperparameters.h>

#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

namespace ml {
namespace maths {

const std::string CBoostedTreeHyperparameters::HYPERPARAM_DOWNSAMPLE_FACTOR_TAG{
    "hyperparam_downsample_factor"};
const std::string CBoostedTreeHyperparameters::HYPERPARAM_ETA_TAG{"hyperparam_eta"};
const std::string CBoostedTreeHyperparameters::HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG{
    "hyperparam_eta_growth_rate_per_tree"};
const std::string CBoostedTreeHyperparameters::HYPERPARAM_FEATURE_BAG_FRACTION_TAG{
    "hyperparam_feature_bag_fraction"};
const std::string CBoostedTreeHyperparameters::HYPERPARAM_MAXIMUM_NUMBER_TREES_TAG{
    "hyperparam_maximum_number_trees"};
const std::string CBoostedTreeHyperparameters::HYPERPARAM_REGULARIZATION_TAG{
    "hyperparam_regularization"};

void CBoostedTreeHyperparameters::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(HYPERPARAM_DOWNSAMPLE_FACTOR_TAG,
                                 m_DownsampleFactor, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_TAG, m_Eta, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                 m_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(HYPERPARAM_MAXIMUM_NUMBER_TREES_TAG,
                                 m_MaximumNumberTrees, inserter);
    core::CPersistUtils::persist(HYPERPARAM_REGULARIZATION_TAG, m_Regularization, inserter);
}

bool CBoostedTreeHyperparameters::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(HYPERPARAM_ETA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_TAG, m_Eta, traverser))
        RESTORE(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                             m_EtaGrowthRatePerTree, traverser))
        RESTORE(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                             m_FeatureBagFraction, traverser))
        RESTORE(HYPERPARAM_MAXIMUM_NUMBER_TREES_TAG,
                core::CPersistUtils::restore(HYPERPARAM_MAXIMUM_NUMBER_TREES_TAG,
                                             m_MaximumNumberTrees, traverser))
        RESTORE(HYPERPARAM_REGULARIZATION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_REGULARIZATION_TAG,
                                             m_Regularization, traverser))
    } while (traverser.next());
    return true;
}

const CBoostedTreeHyperparameters::TRegularization&
CBoostedTreeHyperparameters::regularization() const {
    return m_Regularization;
}

double CBoostedTreeHyperparameters::downsampleFactor() const {
    return m_DownsampleFactor;
}

double CBoostedTreeHyperparameters::eta() const {
    return m_Eta;
}

double CBoostedTreeHyperparameters::etaGrowthRatePerTree() const {
    return m_EtaGrowthRatePerTree;
}

std::size_t CBoostedTreeHyperparameters::maximumNumberTrees() const {
    return m_MaximumNumberTrees;
}

double CBoostedTreeHyperparameters::featureBagFraction() const {
    return m_FeatureBagFraction;
}

void CBoostedTreeHyperparameters::regularization(const TRegularization& regularization) {
    m_Regularization = regularization;
}

void CBoostedTreeHyperparameters::downsampleFactor(double downsampleFactor) {
    m_DownsampleFactor = downsampleFactor;
}

void CBoostedTreeHyperparameters::eta(double eta) {
    m_Eta = eta;
}

void CBoostedTreeHyperparameters::etaGrowthRatePerTree(double etaGrowthRatePerTree) {
    m_EtaGrowthRatePerTree = etaGrowthRatePerTree;
}

void CBoostedTreeHyperparameters::maximumNumberTrees(std::size_t maximumNumberTrees) {
    m_MaximumNumberTrees = maximumNumberTrees;
}

void CBoostedTreeHyperparameters::featureBagFraction(double featureBagFraction) {
    m_FeatureBagFraction = featureBagFraction;
}

CBoostedTreeHyperparameters::CBoostedTreeHyperparameters(
    const CBoostedTreeHyperparameters::TRegularization& regularization,
    double downsampleFactor,
    double eta,
    double etaGrowthRatePerTree,
    std::size_t maximumNumberTrees,
    double featureBagFraction)
    : m_Regularization{regularization}, m_DownsampleFactor{downsampleFactor}, m_Eta{eta},
      m_EtaGrowthRatePerTree{etaGrowthRatePerTree},
      m_MaximumNumberTrees{maximumNumberTrees}, m_FeatureBagFraction{featureBagFraction} {
}
}
}
