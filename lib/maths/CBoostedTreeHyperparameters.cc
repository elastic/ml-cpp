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




const std::string  CBoostedTreeHyperparameters::HYPERPARAM_DOWNSAMPLE_FACTOR_TAG{"hyperparam_downsample_factor"};
const std::string  CBoostedTreeHyperparameters::HYPERPARAM_ETA_TAG{"hyperparam_eta"};
const std::string  CBoostedTreeHyperparameters::HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG{"hyperparam_eta_growth_rate_per_tree"};
const std::string  CBoostedTreeHyperparameters::HYPERPARAM_FEATURE_BAG_FRACTION_TAG{"hyperparam_feature_bag_fraction"};
const std::string  CBoostedTreeHyperparameters::HYPERPARAM_REGULARIZATION_TAG{"hyperparam_regularization"};

void CBoostedTreeHyperparameters::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    core::CPersistUtils::persist(HYPERPARAM_DOWNSAMPLE_FACTOR_TAG,
                                 m_BestDownsampleFactor, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_TAG, m_BestEta, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_BestEtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                 m_BestFeatureBagFraction, inserter);
    core::CPersistUtils::persist(HYPERPARAM_REGULARIZATION_TAG, m_BestRegularization, inserter);
}

bool CBoostedTreeHyperparameters::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(HYPERPARAM_ETA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_TAG, m_BestEta, traverser))
        RESTORE(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                             m_BestEtaGrowthRatePerTree, traverser))
        RESTORE(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                             m_BestFeatureBagFraction, traverser))
        RESTORE(HYPERPARAM_REGULARIZATION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_REGULARIZATION_TAG,
                                             m_BestRegularization, traverser))
    } while (traverser.next());
    return true;
}

const CBoostedTreeHyperparameters::TRegularization &CBoostedTreeHyperparameters::regularization() const {
    return m_BestRegularization;
}

double CBoostedTreeHyperparameters::downsampleFactor() const {
    return m_BestDownsampleFactor;
}

double CBoostedTreeHyperparameters::eta() const {
    return m_BestEta;
}

double CBoostedTreeHyperparameters::etaGrowthRatePerTree() const {
    return m_BestEtaGrowthRatePerTree;
}

double CBoostedTreeHyperparameters::featureBagFraction() const {
    return m_BestFeatureBagFraction;
}

void
CBoostedTreeHyperparameters::regularization(const TRegularization &regularization) {
    m_BestRegularization = regularization;
}

void CBoostedTreeHyperparameters::downsampleFactor(double downsampleFactor) {
    m_BestDownsampleFactor = downsampleFactor;
}

void CBoostedTreeHyperparameters::eta(double eta) {
    m_BestEta = eta;
}

void CBoostedTreeHyperparameters::etaGrowthRatePerTree(double etaGrowthRatePerTree) {
    m_BestEtaGrowthRatePerTree = etaGrowthRatePerTree;
}

void CBoostedTreeHyperparameters::featureBagFraction(double featureBagFraction) {
    m_BestFeatureBagFraction = featureBagFraction;
}

CBoostedTreeHyperparameters::CBoostedTreeHyperparameters(const TRegularization &regularization,
                                                     double downsampleFactor, double eta, double etaGrowthRatePerTree,
                                                     double featureBagFraction): m_BestRegularization{regularization}, m_BestDownsampleFactor{downsampleFactor},
                                                                                 m_BestEta{eta}, m_BestEtaGrowthRatePerTree{etaGrowthRatePerTree}{

}

template<typename T>
void CRegularization<T>::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
                                 m_DepthPenaltyMultiplier, inserter);
    core::CPersistUtils::persist(REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                                 m_TreeSizePenaltyMultiplier, inserter);
    core::CPersistUtils::persist(REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                                 m_LeafWeightPenaltyMultiplier, inserter);
    core::CPersistUtils::persist(REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
                                 m_SoftTreeDepthLimit, inserter);
    core::CPersistUtils::persist(REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
                                 m_SoftTreeDepthTolerance, inserter);
}

template<typename T>
bool CRegularization<T>::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
                                             m_DepthPenaltyMultiplier, traverser))
        RESTORE(REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                                             m_TreeSizePenaltyMultiplier, traverser))
        RESTORE(REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                                             m_LeafWeightPenaltyMultiplier, traverser))
        RESTORE(REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
                core::CPersistUtils::restore(REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
                                             m_SoftTreeDepthLimit, traverser))
        RESTORE(REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
                core::CPersistUtils::restore(REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
                                             m_SoftTreeDepthTolerance, traverser))
    } while (traverser.next());
    return true;
}

}
}
