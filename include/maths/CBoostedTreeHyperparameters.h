/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeHyperparameters_h
#define INCLUDED_ml_maths_CBoostedTreeHyperparameters_h

#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <boost/optional.hpp>

#include <cmath>

namespace ml {
namespace maths {

//! \brief Holds the parameters associated with the different types of regularizer
//! terms available.
template<typename T>
class CBoostedTreeRegularization final {
public:
    //! Set the multiplier of the tree depth penalty.
    CBoostedTreeRegularization& depthPenaltyMultiplier(double depthPenaltyMultiplier) {
        m_DepthPenaltyMultiplier = depthPenaltyMultiplier;
        return *this;
    }

    //! Set the multiplier of the tree size penalty.
    CBoostedTreeRegularization& treeSizePenaltyMultiplier(double treeSizePenaltyMultiplier) {
        m_TreeSizePenaltyMultiplier = treeSizePenaltyMultiplier;
        return *this;
    }

    //! Set the multiplier of the square leaf weight penalty.
    CBoostedTreeRegularization& leafWeightPenaltyMultiplier(double leafWeightPenaltyMultiplier) {
        m_LeafWeightPenaltyMultiplier = leafWeightPenaltyMultiplier;
        return *this;
    }

    //! Set the soft depth tree depth limit.
    CBoostedTreeRegularization& softTreeDepthLimit(double softTreeDepthLimit) {
        m_SoftTreeDepthLimit = softTreeDepthLimit;
        return *this;
    }

    //! Set the tolerance in the depth tree depth limit.
    CBoostedTreeRegularization& softTreeDepthTolerance(double softTreeDepthTolerance) {
        m_SoftTreeDepthTolerance = softTreeDepthTolerance;
        return *this;
    }

    //! Count the number of parameters which have their default values.
    std::size_t countNotSet() const {
        return (m_DepthPenaltyMultiplier == T{} ? 1 : 0) +
               (m_TreeSizePenaltyMultiplier == T{} ? 1 : 0) +
               (m_LeafWeightPenaltyMultiplier == T{} ? 1 : 0) +
               (m_SoftTreeDepthLimit == T{} ? 1 : 0) +
               (m_SoftTreeDepthTolerance == T{} ? 1 : 0);
    }

    //! Multiplier of the tree depth penalty.
    T depthPenaltyMultiplier() const { return m_DepthPenaltyMultiplier; }

    //! Multiplier of the tree size penalty.
    T treeSizePenaltyMultiplier() const { return m_TreeSizePenaltyMultiplier; }

    //! Multiplier of the square leaf weight penalty.
    T leafWeightPenaltyMultiplier() const {
        return m_LeafWeightPenaltyMultiplier;
    }

    //! Soft depth tree depth limit.
    T softTreeDepthLimit() const { return m_SoftTreeDepthLimit; }

    //! Soft depth tree depth limit tolerance.
    T softTreeDepthTolerance() const { return m_SoftTreeDepthTolerance; }

    //! Get the penalty which applies to a leaf at depth \p depth.
    T penaltyForDepth(std::size_t depth) const {
        return std::exp((static_cast<double>(depth) / m_SoftTreeDepthLimit - 1.0) /
                        m_SoftTreeDepthTolerance);
    }

    //! Get description of the regularization parameters.
    std::string print() const {
        return "(depth penalty multiplier = " + toString(m_DepthPenaltyMultiplier) +
               ", soft depth limit = " + toString(m_SoftTreeDepthLimit) +
               ", soft depth tolerance = " + toString(m_SoftTreeDepthTolerance) +
               ", tree size penalty multiplier = " + toString(m_TreeSizePenaltyMultiplier) +
               ", leaf weight penalty multiplier = " +
               toString(m_LeafWeightPenaltyMultiplier) + ")";
    }

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
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

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
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

public:
    static const std::string REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG;
    static const std::string REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG;
    static const std::string REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG;
    static const std::string REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG;
    static const std::string REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG;

private:
    using TOptionalDouble = boost::optional<double>;

private:
    static std::string toString(double x) { return std::to_string(x); }
    static std::string toString(TOptionalDouble x) {
        return x != boost::none ? toString(*x) : "null";
    }

private:
    T m_DepthPenaltyMultiplier = T{};
    T m_TreeSizePenaltyMultiplier = T{};
    T m_LeafWeightPenaltyMultiplier = T{};
    T m_SoftTreeDepthLimit = T{};
    T m_SoftTreeDepthTolerance = T{};
};

template<typename T>
const std::string CBoostedTreeRegularization<T>::REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG{
    "regularization_depth_penalty_multiplier"};
template<typename T>
const std::string CBoostedTreeRegularization<T>::REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG{
    "regularization_tree_size_penalty_multiplier"};
template<typename T>
const std::string CBoostedTreeRegularization<T>::REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG{
    "regularization_leaf_weight_penalty_multiplier"};
template<typename T>
const std::string CBoostedTreeRegularization<T>::REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG{
    "regularization_soft_tree_depth_limit"};
template<typename T>
const std::string CBoostedTreeRegularization<T>::REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG{
    "regularization_soft_tree_depth_tolerance"};

//! \brief The algorithm parameters we'll directly optimise to improve test error.
class MATHS_EXPORT CBoostedTreeHyperparameters {
public:
    using TRegularization = CBoostedTreeRegularization<double>;

public:
    CBoostedTreeHyperparameters() = default;

    CBoostedTreeHyperparameters(const TRegularization& regularization,
                                double downsampleFactor,
                                double eta,
                                double etaGrowthRatePerTree,
                                double featureBagFraction);
    //! The regularisation parameters.
    void regularization(const TRegularization& regularization);
    //! The regularisation parameters.
    const TRegularization& regularization() const;
    //! The downsample factor.
    void downsampleFactor(double downsampleFactor);
    //! The downsample factor.
    double downsampleFactor() const;
    //! Shrinkage.
    void eta(double eta);
    //! Shrinkage.
    double eta() const;
    //! Rate of growth of shrinkage in the training loop.
    void etaGrowthRatePerTree(double etaGrowthRatePerTree);
    //! Rate of growth of shrinkage in the training loop.
    double etaGrowthRatePerTree() const;
    //! The fraction of features we use per bag.
    void featureBagFraction(double featureBagFraction);
    //! The fraction of features we use per bag.
    double featureBagFraction() const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

public:
    static const std::string HYPERPARAM_DOWNSAMPLE_FACTOR_TAG;
    static const std::string HYPERPARAM_ETA_TAG;
    static const std::string HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG;
    static const std::string HYPERPARAM_FEATURE_BAG_FRACTION_TAG;
    static const std::string HYPERPARAM_REGULARIZATION_TAG;

private:
    //! The regularisation parameters.
    TRegularization m_Regularization;

    //! The downsample factor.
    double m_downsampleFactor;

    //! Shrinkage.
    double m_eta;

    //! Rate of growth of shrinkage in the training loop.
    double m_etaGrowthRatePerTree;

    //! The fraction of features we use per bag.
    double m_featureBagFraction;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeHyperparameters_h
