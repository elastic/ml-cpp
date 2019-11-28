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
const std::string CBoostedTreeHyperparameters::HYPERPARAM_REGULARIZATION_TAG{
    "hyperparam_regularization"};

void CBoostedTreeHyperparameters::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(HYPERPARAM_DOWNSAMPLE_FACTOR_TAG,
                                 m_downsampleFactor, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_TAG, m_eta, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_etaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                 m_featureBagFraction, inserter);
    core::CPersistUtils::persist(HYPERPARAM_REGULARIZATION_TAG, m_Regularization, inserter);
}

bool CBoostedTreeHyperparameters::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(HYPERPARAM_ETA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_TAG, m_eta, traverser))
        RESTORE(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                             m_etaGrowthRatePerTree, traverser))
        RESTORE(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                             m_featureBagFraction, traverser))
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
    return m_downsampleFactor;
}

double CBoostedTreeHyperparameters::eta() const {
    return m_eta;
}

double CBoostedTreeHyperparameters::etaGrowthRatePerTree() const {
    return m_etaGrowthRatePerTree;
}

double CBoostedTreeHyperparameters::featureBagFraction() const {
    return m_featureBagFraction;
}

void CBoostedTreeHyperparameters::regularization(const TRegularization& regularization) {
    m_Regularization = regularization;
}

void CBoostedTreeHyperparameters::downsampleFactor(double downsampleFactor) {
    m_downsampleFactor = downsampleFactor;
}

void CBoostedTreeHyperparameters::eta(double eta) {
    m_eta = eta;
}

void CBoostedTreeHyperparameters::etaGrowthRatePerTree(double etaGrowthRatePerTree) {
    m_etaGrowthRatePerTree = etaGrowthRatePerTree;
}

void CBoostedTreeHyperparameters::featureBagFraction(double featureBagFraction) {
    m_featureBagFraction = featureBagFraction;
}

CBoostedTreeHyperparameters::CBoostedTreeHyperparameters(
    const CBoostedTreeHyperparameters::TRegularization& regularization,
    double downsampleFactor,
    double eta,
    double etaGrowthRatePerTree,
    double featureBagFraction)
    : m_Regularization{regularization}, m_downsampleFactor{downsampleFactor}, m_eta{eta},
      m_etaGrowthRatePerTree{etaGrowthRatePerTree}, m_featureBagFraction{featureBagFraction} {
}
}
}
