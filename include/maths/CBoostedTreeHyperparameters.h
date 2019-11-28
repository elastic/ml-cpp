/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeHyperparameters_h
#define INCLUDED_ml_maths_CBoostedTreeHyperparameters_h

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <boost/optional.hpp>

#include <cmath>

namespace ml {
namespace maths {

//! \brief Holds the parameters associated with the different types of regularizer
//! terms available.
template<typename T>
class CRegularization final {
public:
    //! Set the multiplier of the tree depth penalty.
    CRegularization& depthPenaltyMultiplier(double depthPenaltyMultiplier) {
        m_DepthPenaltyMultiplier = depthPenaltyMultiplier;
        return *this;
    }

    //! Set the multiplier of the tree size penalty.
    CRegularization& treeSizePenaltyMultiplier(double treeSizePenaltyMultiplier) {
        m_TreeSizePenaltyMultiplier = treeSizePenaltyMultiplier;
        return *this;
    }

    //! Set the multiplier of the square leaf weight penalty.
    CRegularization& leafWeightPenaltyMultiplier(double leafWeightPenaltyMultiplier) {
        m_LeafWeightPenaltyMultiplier = leafWeightPenaltyMultiplier;
        return *this;
    }

    //! Set the soft depth tree depth limit.
    CRegularization& softTreeDepthLimit(double softTreeDepthLimit) {
        m_SoftTreeDepthLimit = softTreeDepthLimit;
        return *this;
    }

    //! Set the tolerance in the depth tree depth limit.
    CRegularization& softTreeDepthTolerance(double softTreeDepthTolerance) {
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
    T treeSizePenaltyMultiplier() const {
        return m_TreeSizePenaltyMultiplier;
    }

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
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

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
const std::string CRegularization<T>::REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG{"regularization_depth_penalty_multiplier"};
template<typename T>
const std::string CRegularization<T>::REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG{
        "regularization_tree_size_penalty_multiplier"};
template<typename T>
const std::string CRegularization<T>::REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG{
        "regularization_leaf_weight_penalty_multiplier"};
template<typename T>
const std::string CRegularization<T>::REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG{"regularization_soft_tree_depth_limit"};
template<typename T>
const std::string CRegularization<T>::REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG{
        "regularization_soft_tree_depth_tolerance"};

//! \brief The algorithm parameters we'll directly optimise to improve test error.
class CBoostedTreeHyperparameters {
public:
    using TRegularization = CRegularization<double>;

public:
    CBoostedTreeHyperparameters() = default;

    CBoostedTreeHyperparameters(const TRegularization& regularization, double downsampleFactor, double eta,
                                double etaGrowthRatePerTree, double featureBagFraction);

    void regularization(const TRegularization &regularization);

    void downsampleFactor(double downsampleFactor);

    void eta(double eta);

    void etaGrowthRatePerTree(double etaGrowthRatePerTree);

    void featureBagFraction(double featureBagFraction);

    const TRegularization &regularization() const;

    double downsampleFactor() const;

    double eta() const;

    double etaGrowthRatePerTree() const;

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
    TRegularization m_BestRegularization;

    //! The downsample factor.
    double m_BestDownsampleFactor;

    //! Shrinkage.
    double m_BestEta;

    //! Rate of growth of shrinkage in the training loop.
    double m_BestEtaGrowthRatePerTree;

    //! The fraction of features we use per bag.
    double m_BestFeatureBagFraction;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeHyperparameters_h
