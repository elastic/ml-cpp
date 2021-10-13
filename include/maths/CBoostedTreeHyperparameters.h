/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeHyperparameters_h
#define INCLUDED_ml_maths_CBoostedTreeHyperparameters_h

#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CChecksum.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <utility>

namespace ml {
namespace maths {
class CDataFrameTrainBoostedTreeInstrumentationInterface;
template<typename T>
class CScopeBoostedTreeParameterOverrides;

//! \brief Encapsulates a boosted tree parameter.
//!
//! DESCRIPTION:\n
//! This provides the ability to save and load, persist and restore and fix a
//! parameter. Fixed parameters are not optimised. This is typically as a result
//! of a user override but we can also choose to fix a parameter if we can't
//! determine a good search range.
template<typename T>
class CBoostedTreeParameter final {
public:
    explicit CBoostedTreeParameter(T value) : m_Value{value} {}

    //! Set to \p value.
    //!
    //! \note Has no effect if the parameter is fixed.
    void set(T value) {
        if (m_Fixed == false) {
            m_Value = value;
        }
    }

    //! Fix to \p value.
    void fixTo(T value) {
        m_Value = value;
        m_Fixed = true;
    }

    //! Fix the current value.
    void fix() { m_Fixed = true; }

    //! Check if the
    bool fixed() const { return m_Fixed; }

    //! Save the current value.
    void save() { m_SavedValue = m_Value; }
    //! Load the saved value.
    void load() { m_Value = m_SavedValue; }

    //! Get the value.
    T value() const { return m_Value; }

    //! Persist writing to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        core::CPersistUtils::persist(VALUE_TAG, m_Value, inserter);
        core::CPersistUtils::persist(SAVED_VALUE_TAG, m_SavedValue, inserter);
        core::CPersistUtils::persist(FIXED_TAG, m_Fixed, inserter);
    }

    //! Restore reading from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name = traverser.name();
            RESTORE(VALUE_TAG, core::CPersistUtils::restore(VALUE_TAG, m_Value, traverser))
            RESTORE(SAVED_VALUE_TAG,
                    core::CPersistUtils::restore(SAVED_VALUE_TAG, m_SavedValue, traverser))
            RESTORE(FIXED_TAG, core::CPersistUtils::restore(FIXED_TAG, m_Fixed, traverser))
        } while (traverser.next());
        return true;
    }

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        seed = CChecksum::calculate(seed, m_Value);
        seed = CChecksum::calculate(seed, m_Fixed);
        return CChecksum::calculate(seed, m_SavedValue);
    }

    //! Print for debug.
    std::string print() const {
        return (m_Fixed ? "fixed:" : "") + std::to_string(m_Value);
    }

private:
    static const std::string VALUE_TAG;
    static const std::string FIXED_TAG;
    static const std::string SAVED_VALUE_TAG;

private:
    T m_Value{};
    T m_SavedValue{};
    bool m_Fixed{false};

    template<typename>
    friend class CScopeBoostedTreeParameterOverrides;
};

template<typename T>
const std::string CBoostedTreeParameter<T>::VALUE_TAG{"value"};
template<typename T>
const std::string CBoostedTreeParameter<T>::FIXED_TAG{"fixed"};
template<typename T>
const std::string CBoostedTreeParameter<T>::SAVED_VALUE_TAG{"saved_value"};

//! \brief Simple RAII type to force override a collection of parameter values
//! for the object lifetime.
template<typename T>
class CScopeBoostedTreeParameterOverrides {
public:
    CScopeBoostedTreeParameterOverrides() = default;
    ~CScopeBoostedTreeParameterOverrides() {
        // Undo changes in reverse order to which they were applied.
        for (std::size_t i = m_Parameters.size(); i > 0; --i) {
            m_Parameters[i - 1]->m_Value = m_ValuesToRestore[i - 1];
        }
    }

    CScopeBoostedTreeParameterOverrides(const CScopeBoostedTreeParameterOverrides&) = delete;
    CScopeBoostedTreeParameterOverrides&
    operator=(const CScopeBoostedTreeParameterOverrides&) = delete;

    void apply(CBoostedTreeParameter<T>& parameter, T value, bool undo = true) {
        if (undo) {
            m_ValuesToRestore.push_back(parameter.value());
            m_Parameters.push_back(&parameter);
        }
        parameter.m_Value = value;
    }

private:
    using TVec = std::vector<T>;
    using TParameterPtrVec = std::vector<CBoostedTreeParameter<T>*>;

private:
    TParameterPtrVec m_Parameters;
    TVec m_ValuesToRestore;
};

//! \name The hyperparameters for boosted tree training.
//!
//! DESCRIPTION:\n
//! This stores and manages persistence of all the boosted tree training and
//! incremental training hyperparameters. It also provides functionality for
//! optimizing these using a combination of random (Sobolov sequence) search
//! and Bayesian Optimisation.
class CBoostedTreeHyperparameters {
public:
    using TDoubleDoublePrVec = std::vector<std::pair<double, double>>;
    using TStrVec = std::vector<std::string>;
    using TDoubleParameter = CBoostedTreeParameter<double>;
    using TSizeParameter = CBoostedTreeParameter<std::size_t>;
    using TAddInitialRangeFunc =
        std::function<void(boosted_tree_detail::EHyperparameter, TDoubleDoublePrVec&)>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using THyperparameterImportanceVec =
        std::vector<boosted_tree_detail::SHyperparameterImportance>;

public:
    static const std::string BAYESIAN_OPTIMIZATION_TAG;
    static const std::string BEST_FOREST_LOSS_GAP_TAG;
    static const std::string BEST_FOREST_TEST_LOSS_TAG;
    static const std::string CURRENT_ROUND_TAG;
    static const std::string DEPTH_PENALTY_MULTIPLIER_TAG;
    static const std::string DOWNSAMPLE_FACTOR_TAG;
    static const std::string ETA_GROWTH_RATE_PER_TREE_TAG;
    static const std::string ETA_TAG;
    static const std::string FEATURE_BAG_FRACTION_TAG;
    static const std::string LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG;
    static const std::string MAXIMUM_NUMBER_TREES_TAG;
    static const std::string MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG;
    static const std::string MEAN_FOREST_SIZE_ACCUMULATOR_TAG;
    static const std::string MEAN_TEST_LOSS_ACCUMULATOR_TAG;
    static const std::string NUMBER_FOLDS_TAG;
    static const std::string NUMBER_ROUNDS_TAG;
    static const std::string PREDICTION_CHANGE_COST_TAG;
    static const std::string RETRAINED_TREE_ETA_TAG;
    static const std::string SOFT_TREE_DEPTH_LIMIT_TAG;
    static const std::string SOFT_TREE_DEPTH_TOLERANCE_TAG;
    static const std::string STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG;
    static const std::string TRAIN_FRACTION_PER_FOLD_TAG;
    static const std::string TREE_SIZE_PENALTY_MULTIPLIER_TAG;
    static const std::string TREE_TOPOLOGY_CHANGE_PENALTY_TAG;

    //! We prefer smaller models if it costs little in test accuracy.
    static constexpr double RELATIVE_SIZE_PENALTY{0.01};

public:
    CBoostedTreeHyperparameters();

    //! Set if we're incremental training.
    void incrementalTraining(bool value) { m_IncrementalTraining = value; }
    //! \return True if we are incremental training.
    bool incrementalTraining() const { return m_IncrementalTraining; }

    //! Get the writeable multiplier of the tree depth penalty.
    TDoubleParameter& depthPenaltyMultiplier() {
        return m_DepthPenaltyMultiplier;
    }
    //! Get the multiplier of the tree depth penalty.
    const TDoubleParameter& depthPenaltyMultiplier() const {
        return m_DepthPenaltyMultiplier;
    }

    //! Get the writeable multiplier of the tree size penalty.
    TDoubleParameter& treeSizePenaltyMultiplier() {
        return m_TreeSizePenaltyMultiplier;
    }
    //! Get the multiplier of the tree size penalty.
    const TDoubleParameter& treeSizePenaltyMultiplier() const {
        return m_TreeSizePenaltyMultiplier;
    }

    //! Get the writeable multiplier of the square leaf weight penalty.
    TDoubleParameter& leafWeightPenaltyMultiplier() {
        return m_LeafWeightPenaltyMultiplier;
    }
    //! Get the multiplier of the square leaf weight penalty.
    const TDoubleParameter& leafWeightPenaltyMultiplier() const {
        return m_LeafWeightPenaltyMultiplier;
    }

    //! Get the writable soft depth tree depth limit.
    TDoubleParameter& softTreeDepthLimit() { return m_SoftTreeDepthLimit; }
    //! Get the soft depth tree depth limit.
    const TDoubleParameter& softTreeDepthLimit() const {
        return m_SoftTreeDepthLimit;
    }

    //! Get the writable tolerance in the depth tree depth limit.
    TDoubleParameter& softTreeDepthTolerance() {
        return m_SoftTreeDepthTolerance;
    }
    //! Get the soft depth tree depth limit tolerance.
    const TDoubleParameter& softTreeDepthTolerance() const {
        return m_SoftTreeDepthTolerance;
    }
    //! Get the penalty which applies to a leaf at depth \p depth.
    double penaltyForDepth(std::size_t depth) const;

    //! Get the writeable tolerance in the depth tree depth limit.
    TDoubleParameter& treeTopologyChangePenalty() {
        return m_TreeTopologyChangePenalty;
    }
    //! Get the tolerance in the depth tree depth limit.
    const TDoubleParameter& treeTopologyChangePenalty() const {
        return m_TreeTopologyChangePenalty;
    }

    //! Get the writeable data downsample factor when computing loss derivatives.
    TDoubleParameter& downsampleFactor() { return m_DownsampleFactor; }
    //! Get the data downsample factor when computing loss derivatives.
    const TDoubleParameter& downsampleFactor() const {
        return m_DownsampleFactor;
    }

    //! Get the writeable fraction of features which are selected for training a tree.
    TDoubleParameter& featureBagFraction() { return m_FeatureBagFraction; }
    //! Get the fraction of features which are selected for training a tree.
    const TDoubleParameter& featureBagFraction() const {
        return m_FeatureBagFraction;
    }

    //! Get the writeable weight shinkage.
    TDoubleParameter& eta() { return m_Eta; }
    //! Get the weight shinkage.
    const TDoubleParameter& eta() const { return m_Eta; }

    //! Get the writeable growth in weight shrinkage per tree which is added.
    TDoubleParameter& etaGrowthRatePerTree() { return m_EtaGrowthRatePerTree; }
    //! Get the growth in weight shrinkage per tree which is added.
    const TDoubleParameter& etaGrowthRatePerTree() const {
        return m_EtaGrowthRatePerTree;
    }

    //! Get the writeable weight shrinkage to use when retraining a tree.
    TDoubleParameter& retrainedTreeEta() { return m_RetrainedTreeEta; }
    //! Get the weight shrinkage to use when retraining a tree.
    const TDoubleParameter& retrainedTreeEta() const {
        return m_RetrainedTreeEta;
    }

    //! Get the writeable multiplier of prediction change penalty.
    TDoubleParameter& predictionChangeCost() { return m_PredictionChangeCost; }
    //! Get the multiplier of prediction change penalty.
    const TDoubleParameter& predictionChangeCost() const {
        return m_PredictionChangeCost;
    }

    //! Set the maximum number of trees to use.
    TSizeParameter& maximumNumberTrees() { return m_MaximumNumberTrees; }
    //! Get the maximum number of trees to use.
    const TSizeParameter& maximumNumberTrees() const {
        return m_MaximumNumberTrees;
    }

    //! Scale the multipliers of the regularisation terms in the loss function by \p scale.
    void scaleRegularizationMultipliers(double scale,
                                        CScopeBoostedTreeParameterOverrides<double>& overrides,
                                        bool undo = true);

    //! \name Optimisation
    //@{
    //! Set the number of search rounds to use per hyperparameter which is being tuned.
    void maximumOptimisationRoundsPerHyperparameter(std::size_t rounds);

    //! Set whether to stop hyperparameter optimization early.
    void stopHyperparameterOptimizationEarly(bool enable);

    //! Set the maximum number of restarts to use internally in Bayesian Optimisation.
    void bayesianOptimisationRestarts(std::size_t restarts);

    //! Get the number of hyperparameters to tune.
    std::size_t numberToTune() const;

    //! Reset search state.
    void resetSearch();

    //! Initialize the search for best values of tunable hyperparameters.
    void initializeSearch(const TAddInitialRangeFunc& addInitialRange);

    //! Initialize a search for the best hyperparameters.
    void startSearch();

    //! Check if the search for the best hyperparameter values has finished.
    bool searchNotFinished() const { return m_CurrentRound < m_NumberRounds; }

    //! Start a new round of hyperparameter search.
    void startNextSearchRound() { ++m_CurrentRound; }

    //! Get the current round of the search for the best hyperparameters.
    std::size_t currentRound() const { return m_CurrentRound; }

    //! Get the maximum number of rounds used to search for the best hyperparameters.
    std::size_t numberRounds() const { return m_NumberRounds; }

    //! Get the best forest test loss.
    double bestForestTestLoss() const { return m_BestForestTestLoss; }

    //! Get the gap between the train and test loss for the best forest.
    double bestForestLossGap() const { return m_BestForestLossGap; }

    //! Update with the statistics for the current round.
    void addRoundStats(const TMeanAccumulator& meanForestSizeAccumulator, double meanTestLoss);

    //! Choose the next set of hyperparameters to test.
    bool selectNext(const TMeanVarAccumulator& testLossMoments,
                    double explainedVariance = 0.0);

    //! Capture the current hyperparameters if they're the best we've seen so far.
    void captureBest(const TMeanVarAccumulator& testLossMoments,
                     double meanLossGap,
                     double numberKeptNodes,
                     double numberNewNodes,
                     std::size_t numberTrees);

    //! The penalty to apply based on the model size.
    double modelSizePenalty(double numberKeptNodes, double numberNewNodes) const;

    //! Restore the best saved hyperparameters.
    void restoreSaved();

    //! Compute the loss at \p n standard deviations of \p lossMoments above
    //! the mean.
    static double lossAtNSigma(double n, const TMeanVarAccumulator& lossMoments) {
        return CBasicStatistics::mean(lossMoments) +
               n * std::sqrt(CBasicStatistics::variance(lossMoments));
    }

    //! Get the vector of hyperparameter importances.
    THyperparameterImportanceVec importances() const;
    //@}

    //! Write the current hyperparameters to \p instrumentation.
    void output(CDataFrameTrainBoostedTreeInstrumentationInterface& instrumentation) const;

    //! Check invariants which are assumed to hold in order to optimize hyperparameters.
    void checkSearchInvariants() const;

    //! Check the invariants which should hold after restoring.
    void checkRestoredInvariants(bool expectOptimizerInitialized) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Compute a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Get description of the parameters.
    std::string print() const;

    //! \name Test Only
    //@{
    //! A list of the names of the best individual hyperparameters in the state document.
    static TStrVec names();
    //@}

private:
    using TBayesinOptimizationUPtr = std::unique_ptr<CBayesianOptimisation>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using THyperparametersVec = std::vector<boosted_tree_detail::EHyperparameter>;
    using TOptionalSize = boost::optional<std::size_t>;

private:
    void initializeTunableHyperparameters();
    void saveCurrent();

private:
    bool m_IncrementalTraining{false};

    //! \name Hyperparameters
    //@{
    TDoubleParameter m_DepthPenaltyMultiplier{0.0};
    TDoubleParameter m_TreeSizePenaltyMultiplier{0.0};
    TDoubleParameter m_LeafWeightPenaltyMultiplier{0.0};
    TDoubleParameter m_SoftTreeDepthLimit{0.0};
    TDoubleParameter m_SoftTreeDepthTolerance{1.0};
    TDoubleParameter m_TreeTopologyChangePenalty{0.0};
    TDoubleParameter m_DownsampleFactor{0.5};
    TDoubleParameter m_FeatureBagFraction{0.5};
    TDoubleParameter m_Eta{0.1};
    TDoubleParameter m_EtaGrowthRatePerTree{1.05};
    TDoubleParameter m_RetrainedTreeEta{1.0};
    TDoubleParameter m_PredictionChangeCost{0.5};
    TSizeParameter m_MaximumNumberTrees{20};
    //@}

    //@ \name Hyperparameter Optimisation
    //@{
    bool m_StopHyperparameterOptimizationEarly{true};
    std::size_t m_MaximumOptimisationRoundsPerHyperparameter{2};
    TOptionalSize m_BayesianOptimisationRestarts;
    THyperparametersVec m_TunableHyperparameters;
    TDoubleVecVec m_HyperparameterSamples;
    TBayesinOptimizationUPtr m_BayesianOptimization;
    std::size_t m_NumberRounds{1};
    std::size_t m_CurrentRound{0};
    double m_BestForestTestLoss{boosted_tree_detail::INF};
    double m_BestForestLossGap{0.0};
    TMeanAccumulator m_MeanForestSizeAccumulator;
    TMeanAccumulator m_MeanTestLossAccumulator;
    //@}
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeHyperparameters_h
