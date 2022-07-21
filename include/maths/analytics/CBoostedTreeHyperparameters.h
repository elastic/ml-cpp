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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeHyperparameters_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeHyperparameters_h

#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBayesianOptimisation.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/CTools.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace ml {
namespace core {
class CDataFrame;
}
namespace maths {
namespace analytics {
class CBoostedTreeImpl;
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
    using TVector = common::CVectorNx1<T, 3>;

    enum ESearchType { E_LinearSearch, E_LogSearch };

public:
    explicit CBoostedTreeParameter(T value, ESearchType logSearch = E_LinearSearch)
        : m_Value{value}, m_MinValue{value}, m_MaxValue{value}, m_LogSearch{logSearch == E_LogSearch} {
    }

    //! Get the value.
    T value() const { return m_Scale * m_Value; }

    //! Set to \p value.
    //!
    //! \note Has no effect if the parameter is fixed.
    CBoostedTreeParameter& set(T value) {
        if (m_FixedToRange) {
            value = common::CTools::truncate(value, m_MinValue, m_MaxValue);
        }
        m_Value = value;
        return *this;
    }

    //! Get the multiplier which is applied when the parameter is read.
    T scale() const { return m_Scale; }

    //! Set the multiplier which is applied when the parameter is read to \p scale.
    CBoostedTreeParameter& scale(T scale) {
        m_Scale = scale;
        return *this;
    }

    //! Convert the scale to a multiplier of the parameter.
    //!
    //! \warning Handle with care since this also applies the scaling to the range
    //! limits. This is consistent with the behaviour of scaling which can move
    //! values outside this interval.
    CBoostedTreeParameter& captureScale() {
        m_Value *= m_Scale;
        m_MinValue *= m_Scale;
        m_MaxValue *= m_Scale;
        m_Scale = T{1};
        return *this;
    }

    //! Set to the midpoint of the range or \p value if the parameters isn't
    //! fixed to a range.
    //!
    //! \note Has no effect if the parameter is fixed.
    void setToRangeMidpointOr(T value) {
        if (this->fixed() == false) {
            m_Value = m_FixedToRange
                          ? this->fromSearchValue((this->toSearchValue(m_MinValue) +
                                                   this->toSearchValue(m_MaxValue)) /
                                                  T{2})
                          : value;
        }
    }

    //! Fix to \p value.
    void fixTo(T value) {
        m_Value = value;
        m_MinValue = value;
        m_MaxValue = value;
        m_Scale = T{1};
        m_FixedToRange = true;
    }

    //! Fix to the range [ \p minValue, \p maxValue ].
    //!
    //! \note Has no effect if the parameter is fixed.
    void fixToRange(T minValue, T maxValue) {
        m_Value = common::CTools::truncate(m_Value, minValue, maxValue);
        m_MinValue = minValue;
        m_MaxValue = maxValue;
        m_Scale = T{1};
        m_FixedToRange = true;
    }

    //! Fix to either a single value or range depending on the length of \p value.
    void fixTo(const std::vector<T>& value) {
        if (value.empty()) {
            return;
        }
        if (value.size() == 1) {
            this->fixTo(value[0]);
            return;
        }
        if (value[0] > value[1]) {
            this->fixToRange(value[1], value[0]);
        }
        this->fixToRange(value[0], value[1]);
    }

    //! Fix the current value.
    void fix() { this->fixTo(m_Value); }

    //! Check if the value is fixed.
    bool fixed() const { return m_FixedToRange && m_MinValue == m_MaxValue; }

    //! Check if the range is fixed.
    bool rangeFixed() const { return m_FixedToRange; }

    //! Check if \p value is valid for this parameter.
    bool valid(double value) const {
        return m_LogSearch == false || value > 0.0;
    }

    //! Get the unscaled value converted to a search value.
    double toSearchValue() const { return this->toSearchValue(m_Value); }

    //! Convert \p value to the value used by BO for fine tuning.
    double toSearchValue(T value) const {
        return m_LogSearch ? common::CTools::stableLog(static_cast<double>(value))
                           : static_cast<double>(value);
    }

    //! Convert \p value from its value used by BO for fine tuning.
    T fromSearchValue(double value) const {
        return static_cast<T>(m_LogSearch ? common::CTools::stableExp(value) : value);
    }

    //! Get the value range.
    std::pair<T, T> searchRange() const {
        return {this->toSearchValue(m_MinValue), this->toSearchValue(m_MaxValue)};
    }

    //! Save the current value.
    void save() {
        m_SavedValue = m_Value;
        m_SavedMinValue = m_MinValue;
        m_SavedMaxValue = m_MaxValue;
        m_SavedScale = m_Scale;
    }

    //! Load the saved value.
    CBoostedTreeParameter& load() {
        m_Value = m_SavedValue;
        m_MinValue = m_SavedMinValue;
        m_MaxValue = m_SavedMaxValue;
        m_Scale = m_SavedScale;
        return *this;
    }

    //! Persist writing to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        core::CPersistUtils::persist(FIXED_TO_RANGE_TAG, m_FixedToRange, inserter);
        core::CPersistUtils::persist(LOG_SEARCH_TAG, m_LogSearch, inserter);
        core::CPersistUtils::persist(MAX_VALUE_TAG, m_MaxValue, inserter);
        core::CPersistUtils::persist(MIN_VALUE_TAG, m_MinValue, inserter);
        core::CPersistUtils::persist(SAVED_MAX_VALUE_TAG, m_SavedMaxValue, inserter);
        core::CPersistUtils::persist(SAVED_MIN_VALUE_TAG, m_SavedMinValue, inserter);
        core::CPersistUtils::persist(SAVED_SCALE_TAG, m_SavedScale, inserter);
        core::CPersistUtils::persist(SAVED_VALUE_TAG, m_SavedValue, inserter);
        core::CPersistUtils::persist(SCALE_TAG, m_Scale, inserter);
        core::CPersistUtils::persist(VALUE_TAG, m_Value, inserter);
    }

    //! Restore reading from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name{traverser.name()};
            RESTORE(FIXED_TO_RANGE_TAG,
                    core::CPersistUtils::restore(FIXED_TO_RANGE_TAG, m_FixedToRange, traverser))
            RESTORE(LOG_SEARCH_TAG,
                    core::CPersistUtils::restore(LOG_SEARCH_TAG, m_LogSearch, traverser))
            RESTORE(MAX_VALUE_TAG,
                    core::CPersistUtils::restore(MAX_VALUE_TAG, m_MaxValue, traverser))
            RESTORE(MIN_VALUE_TAG,
                    core::CPersistUtils::restore(MIN_VALUE_TAG, m_MinValue, traverser))
            RESTORE(SAVED_MAX_VALUE_TAG,
                    core::CPersistUtils::restore(SAVED_MAX_VALUE_TAG, m_SavedMaxValue, traverser))
            RESTORE(SAVED_MIN_VALUE_TAG,
                    core::CPersistUtils::restore(SAVED_MIN_VALUE_TAG, m_SavedMinValue, traverser))
            RESTORE(SAVED_SCALE_TAG,
                    core::CPersistUtils::restore(SAVED_SCALE_TAG, m_SavedScale, traverser))
            RESTORE(SAVED_VALUE_TAG,
                    core::CPersistUtils::restore(SAVED_VALUE_TAG, m_SavedValue, traverser))
            RESTORE(SCALE_TAG, core::CPersistUtils::restore(SCALE_TAG, m_Scale, traverser))
            RESTORE(VALUE_TAG, core::CPersistUtils::restore(VALUE_TAG, m_Value, traverser))
        } while (traverser.next());
        return true;
    }

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        seed = common::CChecksum::calculate(seed, m_FixedToRange);
        seed = common::CChecksum::calculate(seed, m_LogSearch);
        seed = common::CChecksum::calculate(seed, m_MaxValue);
        seed = common::CChecksum::calculate(seed, m_MinValue);
        seed = common::CChecksum::calculate(seed, m_SavedMaxValue);
        seed = common::CChecksum::calculate(seed, m_SavedMinValue);
        seed = common::CChecksum::calculate(seed, m_SavedScale);
        seed = common::CChecksum::calculate(seed, m_SavedValue);
        seed = common::CChecksum::calculate(seed, m_Scale);
        return common::CChecksum::calculate(seed, m_Value);
    }

    //! Print for debug.
    std::string print() const {
        return std::to_string(m_Value) +
               (m_Scale != T{1} ? " scaled by " + std::to_string(m_Scale) : "") +
               (m_FixedToRange ? " fixed to [" + std::to_string(m_MinValue) +
                                     "," + std::to_string(m_MaxValue) + "]"
                               : "");
    }

private:
    static const std::string FIXED_TO_RANGE_TAG;
    static const std::string LOG_SEARCH_TAG;
    static const std::string MIN_VALUE_TAG;
    static const std::string MAX_VALUE_TAG;
    static const std::string SAVED_MAX_VALUE_TAG;
    static const std::string SAVED_MIN_VALUE_TAG;
    static const std::string SAVED_SCALE_TAG;
    static const std::string SAVED_VALUE_TAG;
    static const std::string SCALE_TAG;
    static const std::string VALUE_TAG;

private:
    T m_Value{};
    T m_MinValue{};
    T m_MaxValue{};
    T m_Scale{1};
    T m_SavedValue{};
    T m_SavedMinValue{};
    T m_SavedMaxValue{};
    T m_SavedScale{1};
    bool m_FixedToRange{false};
    bool m_LogSearch{false};

    template<typename>
    friend class CScopeBoostedTreeParameterOverrides;
};

template<typename T>
const std::string CBoostedTreeParameter<T>::FIXED_TO_RANGE_TAG{"fixed_to_range"};
template<typename T>
const std::string CBoostedTreeParameter<T>::LOG_SEARCH_TAG{"log_search"};
template<typename T>
const std::string CBoostedTreeParameter<T>::MIN_VALUE_TAG{"max_value"};
template<typename T>
const std::string CBoostedTreeParameter<T>::MAX_VALUE_TAG{"min_value"};
template<typename T>
const std::string CBoostedTreeParameter<T>::SAVED_MAX_VALUE_TAG{"saved_max_value"};
template<typename T>
const std::string CBoostedTreeParameter<T>::SAVED_MIN_VALUE_TAG{"saved_min_value"};
template<typename T>
const std::string CBoostedTreeParameter<T>::SAVED_SCALE_TAG{"saved_scale_tag"};
template<typename T>
const std::string CBoostedTreeParameter<T>::SAVED_VALUE_TAG{"saved_value"};
template<typename T>
const std::string CBoostedTreeParameter<T>::SCALE_TAG{"scale_tag"};
template<typename T>
const std::string CBoostedTreeParameter<T>::VALUE_TAG{"value"};

//! \brief Simple RAII type to force override a collection of parameter values
//! for the object lifetime.
template<typename T>
class CScopeBoostedTreeParameterOverrides {
public:
    CScopeBoostedTreeParameterOverrides() noexcept = default;
    ~CScopeBoostedTreeParameterOverrides() {
        // Undo changes in reverse order to which they were applied.
        for (std::size_t i = m_Parameters.size(); i > 0; --i) {
            m_Parameters[i - 1]->m_Value = m_ValuesToRestore[i - 1];
        }
    }

    CScopeBoostedTreeParameterOverrides(const CScopeBoostedTreeParameterOverrides&) = delete;
    CScopeBoostedTreeParameterOverrides&
    operator=(const CScopeBoostedTreeParameterOverrides&) = delete;

    void apply(CBoostedTreeParameter<T>& parameter, T value) {
        m_ValuesToRestore.push_back(parameter.value());
        m_Parameters.push_back(&parameter);
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
//! optimizing these using a combination of random (Sobolev sequence) search
//! and Bayesian Optimisation.
class MATHS_ANALYTICS_EXPORT CBoostedTreeHyperparameters {
public:
    using TStrVec = std::vector<std::string>;
    using TOptionalSize = std::optional<std::size_t>;
    using TOptionalDoubleSizePr = std::optional<std::pair<double, std::size_t>>;
    using TDoubleParameter = CBoostedTreeParameter<double>;
    using TSizeParameter = CBoostedTreeParameter<std::size_t>;
    using TVector = common::CDenseVector<double>;
    using TVector3x1 = common::CVectorNx1<double, 3>;
    using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using THyperparameterImportanceVec =
        std::vector<boosted_tree_detail::SHyperparameterImportance>;

    //! \brief The arguments to the initial search we perform for each parameter.
    class CInitializeFineTuneArguments {
    public:
        using TUpdateParameter = std::function<bool(CBoostedTreeImpl&, double)>;
        using TTruncateParameter = std::function<void(TVector3x1&)>;
        using TAdjustTestLoss = std::function<double(double, double, double)>;

    public:
        CInitializeFineTuneArguments(core::CDataFrame& frame,
                                     CBoostedTreeImpl& tree,
                                     double maxValue,
                                     double searchInterval,
                                     TUpdateParameter updateParameter)
            : m_UpdateParameter{std::move(updateParameter)}, m_Frame{frame}, m_Tree{tree},
              m_MaxValue{maxValue}, m_SearchInterval{searchInterval} {}

        CInitializeFineTuneArguments(const CInitializeFineTuneArguments&) = delete;
        CInitializeFineTuneArguments& operator=(const CInitializeFineTuneArguments&) = delete;

        CInitializeFineTuneArguments& adjustLoss(TAdjustTestLoss adjustLoss) {
            m_AdjustLoss = std::move(adjustLoss);
            return *this;
        }

        CInitializeFineTuneArguments& truncateParameter(TTruncateParameter truncateParameter) {
            m_TruncateParameter = std::move(truncateParameter);
            return *this;
        }

        core::CDataFrame& frame() const { return m_Frame; }
        CBoostedTreeImpl& tree() const { return m_Tree; }
        double maxValue() const { return m_MaxValue; }
        double searchInterval() const { return m_SearchInterval; }
        const TUpdateParameter& updateParameter() const {
            return m_UpdateParameter;
        }
        const TAdjustTestLoss& adjustLoss() const { return m_AdjustLoss; }
        const TTruncateParameter& truncateParameter() const {
            return m_TruncateParameter;
        }

    private:
        static void noopTruncateParameter(TVector3x1&) {}
        static double noopAdjustTestLoss(double, double, double testLoss) {
            return testLoss;
        }

    private:
        TUpdateParameter m_UpdateParameter;
        TTruncateParameter m_TruncateParameter{noopTruncateParameter};
        TAdjustTestLoss m_AdjustLoss{noopAdjustTestLoss};
        core::CDataFrame& m_Frame;
        CBoostedTreeImpl& m_Tree;
        double m_MaxValue;
        double m_SearchInterval;
    };

public:
    //! We prefer smaller models if it costs little in test accuracy.
    static constexpr double RELATIVE_SIZE_PENALTY{0.01};

public:
    CBoostedTreeHyperparameters();

    //! Set if we're incremental training.
    void incrementalTraining(bool value) { m_IncrementalTraining = value; }
    //! \return True if we are incremental training.
    bool incrementalTraining() const { return m_IncrementalTraining; }

    //! Set whether we should scale regularisation hyperaparameters as we adjust
    //! the downsample factor.
    void disableScaling(bool disabled) { m_ScalingDisabled = disabled; }
    //! \return True if we are not to scaling regularisation hyperaparameters as
    //! we adjust the downsample factor.
    bool scalingDisabled() const { return m_ScalingDisabled; }

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

    //! Get the writeable weight shrinkage.
    TDoubleParameter& eta() { return m_Eta; }
    //! Get the weight shrinkage.
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

    //! Compute the learn rate for the tree at \p index.
    double etaForTreeAtPosition(std::size_t index) const;

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

    //! \name Optimisation
    //@{
    //! Set the number of search rounds to use per hyperparameter which is being tuned.
    void maximumOptimisationRoundsPerHyperparameter(std::size_t rounds);

    //! Set whether to stop hyperparameter optimization early.
    void stopHyperparameterOptimizationEarly(bool enable);

    //! Set the maximum number of restarts to use internally in Bayesian Optimisation.
    void bayesianOptimisationRestarts(std::size_t restarts);

    //! Get the maximum number of iterations used in testLossLineSearch.
    static std::size_t maxLineSearchIterations() { return 10; }

    //! Get the number of hyperparameters to tune.
    std::size_t numberToTune() const;

    //! Reset search state.
    void resetFineTuneSearch();

    //! Compute the fine tune search interval for \p parameter.
    //!
    //! \return The best number of trees to use for the current hyperparameter settings.
    TOptionalDoubleSizePr
    initializeFineTuneSearchInterval(const CInitializeFineTuneArguments& args,
                                     TDoubleParameter& parameter) const;

    //! Initialize the search for best values of tunable hyperparameters.
    void initializeFineTuneSearch(double lossGap, std::size_t numberTrees);

    //! Check if search is making no progress improving the test loss.
    bool optimisationMakingNoProgress() const;

    //! Initialize a search for the best hyperparameters.
    void startFineTuneSearch();

    //! Check if the search for the best hyperparameter values has finished.
    bool fineTuneSearchNotFinished() const {
        return m_StopHyperparameterOptimizationEarly == false && m_CurrentRound < m_NumberRounds;
    }

    //! Start a new round of hyperparameter search.
    void startNextRound() { ++m_CurrentRound; }

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

    //! Capture the current hyperparameters if they're the best we've seen.
    //!
    //! \return True if we they are the best hyperparameters.
    bool captureBest(const TMeanVarAccumulator& testLossMoments,
                     double meanLossGap,
                     double numberKeptNodes,
                     double numberNewNodes,
                     std::size_t numberTrees);

    //! Restore the hyperparameters saved by captureBest.
    void restoreBest();

    //! Capture any scaling which has been applied to the hyperparameters.
    void captureScale();

    //! The penalty to apply based on the model size.
    double modelSizePenalty(double numberKeptNodes, double numberNewNodes) const;

    //! Get the vector of hyperparameter importances.
    THyperparameterImportanceVec importances() const;
    //@}

    //! Write the current hyperparameters to \p instrumentation.
    void recordHyperparameters(CDataFrameTrainBoostedTreeInstrumentationInterface& instrumentation) const;

    //! Check invariants which are assumed to hold in order to optimize hyperparameters.
    void checkSearchInvariants() const;

    //! Check the invariants which should hold after restoring.
    void checkRestoredInvariants(bool expectOptimizerInitialized) const;

    //! Estimate the memory that this object will use.
    std::size_t estimateMemoryUsage() const;

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
    using TBayesinOptimizationUPtr = std::unique_ptr<common::CBayesianOptimisation>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TDoubleDoubleDoubleSizeTupleVec =
        std::vector<std::tuple<double, double, double, std::size_t>>;
    using TOptionalVector3x1 = std::optional<TVector3x1>;
    using TIndexVec = std::vector<TVector::TIndexType>;
    using TOptionalVector3x1DoubleSizeTr = std::tuple<TOptionalVector3x1, double, std::size_t>;
    using TVectorDoubleDoubleTr = std::tuple<TVector, double, double>;
    using TVectorDoubleDoubleTrVec = std::vector<TVectorDoubleDoubleTr>;
    using THyperparametersVec = std::vector<boosted_tree_detail::EHyperparameter>;

private:
    void initializeTunableHyperparameters();
    void initialTestLossLineSearch(const CInitializeFineTuneArguments& args,
                                   double intervalLeftEnd,
                                   double intervalRightEnd,
                                   TDoubleDoubleDoubleSizeTupleVec& testLosses) const;
    TOptionalVector3x1DoubleSizeTr testLossLineSearch(const CInitializeFineTuneArguments& args,
                                                      double intervalLeftEnd,
                                                      double intervalRightEnd) const;
    void fineTuneTestLoss(const CInitializeFineTuneArguments& args,
                          double intervalLeftEnd,
                          double intervalRightEnd,
                          TDoubleDoubleDoubleSizeTupleVec& testLosses) const;
    TOptionalVector3x1DoubleSizeTr
    minimizeTestLoss(double intervalLeftEnd,
                     double intervalRightEnd,
                     TDoubleDoubleDoubleSizeTupleVec testLosses) const;
    void checkIfCanSkipFineTuneSearch();
    void captureHyperparametersAndLoss(const TMeanVarAccumulator& loss);
    TVector currentParametersVector() const;
    void setHyperparameterValues(TVector parameters);
    void saveCurrent();
    template<typename F>
    void foreachTunableParameter(const F& f) const;

private:
    bool m_IncrementalTraining{false};

    //! \name Hyperparameters
    //@{
    TDoubleParameter m_DepthPenaltyMultiplier{0.0, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_TreeSizePenaltyMultiplier{0.0, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_LeafWeightPenaltyMultiplier{0.0, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_SoftTreeDepthLimit{0.0, TDoubleParameter::E_LinearSearch};
    TDoubleParameter m_SoftTreeDepthTolerance{1.0, TDoubleParameter::E_LinearSearch};
    TDoubleParameter m_TreeTopologyChangePenalty{0.0, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_DownsampleFactor{0.5, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_FeatureBagFraction{0.5, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_Eta{0.1, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_EtaGrowthRatePerTree{1.05, TDoubleParameter::E_LinearSearch};
    TDoubleParameter m_RetrainedTreeEta{1.0, TDoubleParameter::E_LogSearch};
    TDoubleParameter m_PredictionChangeCost{0.5, TDoubleParameter::E_LogSearch};
    TSizeParameter m_MaximumNumberTrees{20, TSizeParameter::E_LinearSearch};
    //@}

    //! \name Hyperparameter Optimisation
    //@{
    bool m_EarlyHyperparameterOptimizationStoppingEnabled{true};
    bool m_StopHyperparameterOptimizationEarly{false};
    bool m_ScalingDisabled{false};
    std::size_t m_MaximumOptimisationRoundsPerHyperparameter{2};
    TOptionalSize m_BayesianOptimisationRestarts;
    THyperparametersVec m_TunableHyperparameters;
    TDoubleVecVec m_HyperparameterSamples;
    TBayesinOptimizationUPtr m_BayesianOptimization;
    std::size_t m_NumberRounds{1};
    std::size_t m_CurrentRound{0};
    double m_BestForestTestLoss{boosted_tree_detail::INF};
    double m_BestForestNumberKeptNodes{0.0};
    double m_BestForestNumberNewNodes{0.0};
    double m_BestForestLossGap{0.0};
    TMeanAccumulator m_MeanForestSizeAccumulator;
    TMeanAccumulator m_MeanTestLossAccumulator;
    TIndexVec m_LineSearchRelevantParameters;
    TVectorDoubleDoubleTrVec m_LineSearchHyperparameterLosses;
    //@}
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeHyperparameters_h
