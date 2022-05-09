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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionDetail_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionDetail_h

#include <core/CSmallVector.h>
#include <core/CStateMachine.h>
#include <core/CoreTypes.h>

#include <maths/time_series/CCalendarComponent.h>
#include <maths/time_series/CCalendarCyclicTest.h>
#include <maths/time_series/CExpandingWindow.h>
#include <maths/time_series/CSeasonalComponent.h>
#include <maths/time_series/CSeasonalTime.h>
#include <maths/time_series/CTimeSeriesDecompositionInterface.h>
#include <maths/time_series/CTimeSeriesTestForChange.h>
#include <maths/time_series/CTimeSeriesTestForSeasonality.h>
#include <maths/time_series/CTrendComponent.h>
#include <maths/time_series/ImportExport.h>

#include <boost/circular_buffer.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace CTimeSeriesDecompositionTest {
class CNanInjector;
}

namespace ml {
namespace maths {
namespace time_series {
class CExpandingWindow;
class CTimeSeriesDecomposition;

//! \brief Utilities for computing the decomposition.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesDecompositionDetail
    : private CTimeSeriesDecompositionTypes {
public:
    using TDoubleVec = std::vector<double>;
    using TMakePredictor = std::function<TPredictor()>;
    using TMakeFilteredPredictor = std::function<TFilteredPredictor()>;
    using TChangePointUPtr = std::unique_ptr<CChangePoint>;

    // clang-format off
    using TMakeTestForSeasonality =
        std::function<CTimeSeriesTestForSeasonality(const CExpandingWindow&,
                                                    core_t::TTime,
                                                    std::size_t,
                                                    const TFilteredPredictor&,
                                                    double)>;
    // clang-format on

    class CMediator;

    //! \brief The base message passed.
    struct MATHS_TIME_SERIES_EXPORT SMessage {
        SMessage(core_t::TTime time, core_t::TTime lastTime);

        //! The message time.
        core_t::TTime s_Time;
        //! The last update time.
        core_t::TTime s_LastTime;
    };

    //! \brief The message passed to add a point.
    struct MATHS_TIME_SERIES_EXPORT SAddValue : public SMessage {
        SAddValue(core_t::TTime time,
                  core_t::TTime lastTime,
                  core_t::TTime timeShift,
                  double value,
                  const maths_t::TDoubleWeightsAry& weights,
                  double occupancy,
                  core_t::TTime firstValueTime,
                  double trend,
                  double seasonal,
                  double calendar,
                  CTimeSeriesDecomposition& decomposition,
                  const TMakePredictor& makePredictor,
                  const TMakeFilteredPredictor& makeSeasonalityTestPreconditioner,
                  const TMakeTestForSeasonality& makeTestForSeasonality);
        SAddValue(const SAddValue&) = delete;
        SAddValue& operator=(const SAddValue&) = delete;

        //! The time shift being applied.
        core_t::TTime s_TimeShift;
        //! The value to add.
        double s_Value;
        //! The weights of associated with the value.
        const maths_t::TDoubleWeightsAry& s_Weights;
        //! The proportion of non-empty buckets.
        double s_Occupancy;
        //! The time of the first value added to the decomposition.
        core_t::TTime s_FirstValueTime;
        //! The trend component prediction at the value's time.
        double s_Trend;
        //! The seasonal component prediction at the value's time.
        double s_Seasonal;
        //! The calendar component prediction at the value's time.
        double s_Calendar;
        //! The time series decomposition.
        CTimeSeriesDecomposition* s_Decomposition;
        //! Makes the predictor to use in the change detector test.
        TMakePredictor s_MakePredictor;
        //! Makes the preconditioner to use for seasonality testing. This removes
        //! components which won't be explicitly tested.
        TMakeFilteredPredictor s_MakeSeasonalityTestPreconditioner;
        //! A factory function to create the test for seasonal components.
        TMakeTestForSeasonality s_MakeTestForSeasonality;
    };

    //! \brief The message passed to indicate periodic components have been
    //! detected.
    struct MATHS_TIME_SERIES_EXPORT SDetectedSeasonal : public SMessage {
        SDetectedSeasonal(core_t::TTime time, core_t::TTime lastTime, CSeasonalDecomposition components);

        //! The components found.
        CSeasonalDecomposition s_Components;
    };

    //! \brief The message passed to indicate calendar components have been
    //! detected.
    struct MATHS_TIME_SERIES_EXPORT SDetectedCalendar : public SMessage {
        SDetectedCalendar(core_t::TTime time,
                          core_t::TTime lastTime,
                          CCalendarFeature feature,
                          core_t::TTime timeZoneOffset);

        //! The calendar feature found.
        CCalendarFeature s_Feature;
        //! The time zone offset which applies to the feature.
        core_t::TTime s_TimeZoneOffset;
    };

    //! \brief The message passed to indicate the trend is being used for prediction.
    struct MATHS_TIME_SERIES_EXPORT SDetectedTrend : public SMessage {
        SDetectedTrend(const TPredictor& predictor,
                       const TComponentChangeCallback& componentChangeCallback);

        TPredictor s_Predictor;
        TComponentChangeCallback s_ComponentChangeCallback;
    };

    //! \brief The message passed to indicate a sudden change has occurred.
    struct MATHS_TIME_SERIES_EXPORT SDetectedChangePoint : public SMessage {
        SDetectedChangePoint(core_t::TTime time, core_t::TTime lastTime, TChangePointUPtr change);

        //! The change description.
        TChangePointUPtr s_Change;
    };

    //! \brief The basic interface for one aspect of the modeling of a time
    //! series decomposition.
    class MATHS_TIME_SERIES_EXPORT CHandler {
    public:
        CHandler() = default;
        virtual ~CHandler() = default;
        CHandler(const CHandler&) = delete;
        CHandler& operator=(const CHandler&) = delete;

        //! Add a value.
        virtual void handle(const SAddValue& message);

        //! Handle when a diurnal component is detected.
        virtual void handle(const SDetectedSeasonal& message);

        //! Handle when a calendar component is detected.
        virtual void handle(const SDetectedCalendar& message);

        //! Handle when a new component is being modeled.
        virtual void handle(const SDetectedTrend& message);

        //! Handle when a new sudden change is detected.
        virtual void handle(const SDetectedChangePoint& message);

        //! Set the mediator.
        void mediator(CMediator* mediator);

        //! Get the mediator.
        CMediator* mediator() const;

    private:
        //! The controller responsible for forwarding messages.
        CMediator* m_Mediator = nullptr;
    };

    //! \brief Manages communication between handlers.
    class MATHS_TIME_SERIES_EXPORT CMediator {
    public:
        CMediator() = default;
        CMediator(const CMediator&) = delete;
        CMediator& operator=(const CMediator&) = delete;

        //! Forward \p message to all registered models.
        template<typename M>
        void forward(const M& message) const;

        //! Register \p handler.
        void registerHandler(CHandler& handler);

        //! Debug the memory used by this object.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using THandlerRef = std::reference_wrapper<CHandler>;
        using THandlerRefVec = std::vector<THandlerRef>;

    private:
        //! The handlers which have added by registration.
        THandlerRefVec m_Handlers;
    };

    //! \brief Checks for sudden change or change events.
    class MATHS_TIME_SERIES_EXPORT CChangePointTest : public CHandler {
    public:
        static constexpr double CHANGE_COUNT_WEIGHT = 0.1;
        static constexpr core_t::TTime MINIMUM_WINDOW_BUCKET_LENGTH = core::constants::HOUR;

    public:
        CChangePointTest(double decayRate, core_t::TTime bucketLength);
        CChangePointTest(const CChangePointTest& other, bool isForForecast = false);
        CChangePointTest& operator=(const CChangePointTest&) = delete;

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Efficiently swap the state of this and \p other.
        void swap(CChangePointTest& other);

        //! Update the test with a new value.
        void handle(const SAddValue& message) override;

        //! Reset residual distribution moments.
        void handle(const SDetectedSeasonal&) override;

        //! Get the count weight to apply to samples.
        double countWeight(core_t::TTime time) const;

        //! Get the derate to apply to the Winsorisation weight.
        double winsorisationDerate(core_t::TTime time) const;

        //! Age the test to account for the interval \p end - \p start elapsed time.
        void propagateForwards(core_t::TTime start, core_t::TTime end);

        //! Get a checksum for this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TFloatMeanAccumulatorCBuf = boost::circular_buffer<TFloatMeanAccumulator>;

    private:
        //! Handle \p symbol.
        void apply(std::size_t symbol);

        //! Update the total count weight statistics.
        void updateTotalCountWeights(const SAddValue& message);

        //! Update the fraction of recent large errors.
        void testForCandidateChange(const SAddValue& message);

        //! Test if any change has occurred.
        void testForChange(const SAddValue& message);

        //! Test whether to undo the last change which was applied.
        void testUndoLastChange(const SAddValue& message);

        //! True if the time series may be experiencing a change point.
        bool mayHaveChanged() const;

        //! The magnitude of a large error.
        double largeError() const;

        //! Check if we should test for a change.
        bool shouldTest(core_t::TTime time, double occupancy) const;

        //! The minimum time a change has to last.
        core_t::TTime minimumChangeLength(double occupancy) const;

        //! The length of time in which we expect to detect a change.
        core_t::TTime maximumIntervalToDetectChange(double occupancy) const;

        //! The start of the sliding window of buckets given the time is now \p time.
        core_t::TTime bucketsStartTime(core_t::TTime time, core_t::TTime bucketsLength) const;

        //! The start of the values given the buckets start at \p bucketStartTime.
        core_t::TTime valuesStartTime(core_t::TTime bucketsStartTime) const;

        //! The start time of the window bucket containing \p time.
        core_t::TTime startOfWindowBucket(core_t::TTime time) const;

        //! The length of the window.
        core_t::TTime windowLength() const;

        //! The length of the window buckets.
        core_t::TTime windowBucketLength() const;

        //! Get the window size to use.
        std::size_t windowSize() const;

    private:
        //! The state machine.
        core::CStateMachine m_Machine;

        //! Controls the rate at which information is lost.
        double m_DecayRate;

        //! The raw data bucketing interval.
        core_t::TTime m_BucketLength;

        //! The window tested for changes.
        TFloatMeanAccumulatorCBuf m_Window;

        //! The average offset of the values time w.r.t. the start of the buckets.
        TFloatMeanAccumulator m_MeanOffset;

        //! The mean and variance of the prediction residuals.
        TMeanVarAccumulator m_ResidualMoments;

        //! The proportion of recent values with significantly prediction error.
        double m_LargeErrorFraction = 0.0;

        //! The total adjustment applied to the count weight.
        double m_TotalCountWeightAdjustment = 0.0;

        //! The minimum permitted total adjustment applied to the count weight.
        double m_MinimumTotalCountWeightAdjustment = 0.0;

        //! The last test time.
        core_t::TTime m_LastTestTime;

        //! The last time a change point was detected.
        core_t::TTime m_LastChangePointTime;

        //! The time the last candidate change point occurred.
        core_t::TTime m_LastCandidateChangePointTime;

        //! The last change which was made, if it hasn't been committed, in a form
        //! which can be undone.
        TChangePointUPtr m_UndoableLastChange;
    };

    //! \brief Scans through increasingly low frequencies looking for significant
    //! seasonal components.
    class MATHS_TIME_SERIES_EXPORT CSeasonalityTest : public CHandler {
    public:
        //! Test types (categorised as short and long period tests).
        enum ETest { E_Short, E_Long };

    public:
        CSeasonalityTest(double decayRate, core_t::TTime bucketLength);
        CSeasonalityTest(const CSeasonalityTest& other, bool isForForecast = false);
        CSeasonalityTest& operator=(const CSeasonalityTest&) = delete;

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Efficiently swap the state of this and \p other.
        void swap(CSeasonalityTest& other);

        //! Update the test with a new value.
        void handle(const SAddValue& message) override;

        //! Sample the prediction residuals.
        void handle(const SDetectedTrend& message) override;

        //! Shift the start of the tests' expanding windows by \p shift at \p time.
        void shiftTime(core_t::TTime time, core_t::TTime shift);

        //! Age the test to account for the interval \p end - \p start
        //! elapsed time.
        void propagateForwards(core_t::TTime start, core_t::TTime end);

        //! Get the window minus the predictions of \p predictor.
        TFloatMeanAccumulatorVec residuals(const TPredictor& predictor) const;

        //! Get a checksum for this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TExpandingWindowUPtr = std::unique_ptr<CExpandingWindow>;
        using TExpandingWindowPtrAry = std::array<TExpandingWindowUPtr, 2>;

    private:
        //! Handle \p symbol.
        void apply(std::size_t symbol, const SMessage& message);

        //! Test to see whether any seasonal components are present.
        void test(const SAddValue& message);

        //! Check if we should run the periodicity test on \p window.
        bool shouldTest(ETest test, core_t::TTime time) const;

        //! Get a new \p test. (Warning: this is owned by the caller.)
        TExpandingWindowUPtr newWindow(ETest test, bool deflate = true) const;

        //! Account for memory that is not allocated by initialisation.
        std::size_t extraMemoryOnInitialization() const;

    private:
        //! The state machine.
        core::CStateMachine m_Machine;

        //! Controls the rate at which information is lost.
        double m_DecayRate;

        //! The raw data bucketing interval.
        core_t::TTime m_BucketLength;

        //! Expanding windows on the "recent" time series values.
        TExpandingWindowPtrAry m_Windows;
    };

    //! \brief Tests for cyclic calendar components explaining large prediction
    //! errors.
    class MATHS_TIME_SERIES_EXPORT CCalendarTest : public CHandler {
    public:
        CCalendarTest(double decayRate, core_t::TTime bucketLength);
        CCalendarTest(const CCalendarTest& other, bool isForForecast = false);
        CCalendarTest& operator=(const CCalendarTest&) = delete;

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Efficiently swap the state of this and \p other.
        void swap(CCalendarTest& other);

        //! Update the test with a new value.
        void handle(const SAddValue& message) override;

        //! Reset the test.
        void handle(const SDetectedSeasonal& message) override;

        //! Test to see whether any seasonal components are present.
        void test(const SMessage& message);

        //! Age the test to account for the interval \p end - \p start
        //! elapsed time.
        void propagateForwards(core_t::TTime start, core_t::TTime end);

        //! Get a checksum for this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TCalendarCyclicTestPtr = std::unique_ptr<CCalendarCyclicTest>;

    private:
        //! Handle \p symbol.
        void apply(std::size_t symbol, const SMessage& message);

        //! Check if we should run a test.
        bool shouldTest(core_t::TTime time);

        //! Get the month of \p time.
        int month(core_t::TTime time) const;

        //! Account for memory that is not yet allocated
        //! during the initial state
        std::size_t extraMemoryOnInitialization() const;

    private:
        //! The state machine.
        core::CStateMachine m_Machine;

        //! Controls the rate at which information is lost.
        double m_DecayRate;

        //! The last month for which the test was run.
        int m_LastMonth;

        //! The test for arbitrary periodic components.
        TCalendarCyclicTestPtr m_Test;
    };

    //! \brief Holds and updates the components of the decomposition.
    class MATHS_TIME_SERIES_EXPORT CComponents : public CHandler {
    public:
        class CScopeAttachComponentChangeCallback {
        public:
            CScopeAttachComponentChangeCallback(CComponents& components,
                                                TComponentChangeCallback componentChangeCallback,
                                                maths_t::TModelAnnotationCallback modelAnnotationCallback);
            ~CScopeAttachComponentChangeCallback();
            CScopeAttachComponentChangeCallback(const CScopeAttachComponentChangeCallback&) = delete;
            CScopeAttachComponentChangeCallback&
            operator=(const CScopeAttachComponentChangeCallback&) = delete;

        private:
            CComponents& m_Components;
        };

    public:
        CComponents(double decayRate, core_t::TTime bucketLength, std::size_t seasonalComponentSize);
        CComponents(const CComponents& other);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(const common::SDistributionRestoreParams& params,
                                    core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Efficiently swap the state of this and \p other.
        void swap(CComponents& other);

        //! Update the components with a new value.
        void handle(const SAddValue& message) override;

        //! Create new seasonal components.
        void handle(const SDetectedSeasonal& message) override;

        //! Create a new calendar component.
        void handle(const SDetectedCalendar& message) override;

        //! Update the components with a sudden change.
        void handle(const SDetectedChangePoint& message) override;

        //! Maybe re-interpolate the components.
        void interpolateForForecast(core_t::TTime time);

        //! Set the data type.
        void dataType(maths_t::EDataType dataType);

        //! Set the decay rate.
        void decayRate(double decayRate);

        //! Get the decay rate.
        double decayRate() const;

        //! Age the components to account for the interval \p end - \p start
        //! elapsed time.
        void propagateForwards(core_t::TTime start, core_t::TTime end);

        //! Check if the decomposition has any initialized components.
        bool initialized() const;

        //! Get the long term trend.
        const CTrendComponent& trend() const;

        //! Get the seasonal components.
        const maths_t::TSeasonalComponentVec& seasonal() const;

        //! Get the calendar components.
        const maths_t::TCalendarComponentVec& calendar() const;

        //! Return true if we're using the trend for prediction.
        bool usingTrendForPrediction() const;

        //! Start using the trend for prediction.
        void useTrendForPrediction();

        //! Get a factory for the seasonal components test.
        TMakeTestForSeasonality makeTestForSeasonality(const TFilteredPredictor& predictor) const;

        //! Get the mean value of the baseline in the vicinity of \p time.
        double meanValue(core_t::TTime time) const;

        //! Get the mean variance of the baseline.
        double meanVariance() const;

        //! Get the mean error variance scale for the components.
        double meanVarianceScale() const;

        //! Get a checksum for this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TSeasonalComponentPtrVec = std::vector<CSeasonalComponent*>;
        using TCalendarComponentPtrVec = std::vector<CCalendarComponent*>;

        //! \brief Manages the setting of the error gain when updating
        //! the components with a value.
        //!
        //! DESCRIPTION:\n
        //! The gain is the scale applied to the error in the prediction
        //! when updating the components with a new value. If we think it
        //! is safe, we use a large gain since this improves prediction
        //! accuracy. However, this can also lead to instability if, for
        //! example, the seasonal components present in the time series
        //! suddenly change. When instability occurs it manifests as the
        //! amplitude of all the components growing.
        //!
        //! This object therefore monitors the sum of the absolute component
        //! amplitudes and decreases the gain when it detects that this is
        //! significantly increasing.
        class MATHS_TIME_SERIES_EXPORT CGainController {
        public:
            //! Initialize by reading state from \p traverser.
            bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

            //! Persist state by passing information to \p inserter.
            void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

            //! Clear all state.
            void clear();

            //! Get the gain to use when updating the components with a new value.
            double gain() const;

            //! Add seed predictions \p predictions.
            void seed(const TDoubleVec& predictions);

            //! Add the predictions \p predictions at \p time.
            void add(core_t::TTime time, const TDoubleVec& predictions);

            //! Age by \p factor.
            void age(double factor);

            //! Shift the mean prediction error regression model time origin
            //! to \p time.
            void shiftOrigin(core_t::TTime time);

            //! Get a checksum for this object.
            std::uint64_t checksum(std::uint64_t seed) const;

        private:
            using TRegression = common::CLeastSquaresOnlineRegression<1>;

        private:
            //! The origin for the mean prediction error regression model.
            core_t::TTime m_RegressionOrigin = 0;
            //! The sum of the absolute component predictions w.r.t. their means.
            TFloatMeanAccumulator m_MeanSumAmplitudes;
            //! A regression model for the absolute component predictions.
            TRegression m_MeanSumAmplitudesTrend;
        };

        //! \brief Tracks prediction errors with and without components.
        //!
        //! DESCRIPTION:\n
        //! This tracks the prediction errors with and without seasonal and
        //! calendar periodic components and tests to see if including the
        //! component is worthwhile.
        class MATHS_TIME_SERIES_EXPORT CComponentErrors {
        public:
            //! Initialize from a delimited string.
            bool fromDelimited(const std::string& str);

            //! Convert to a delimited string.
            std::string toDelimited() const;

            //! Update the errors.
            //!
            //! \param[in] referenceError The reference error with no components.
            //! \param[in] error The prediction error.
            //! \param[in] prediction The prediction from the component.
            //! \param[in] varianceIncrease The increase in predicted variance
            //! due to the component.
            //! \param[in] weight The weight of \p error.
            void add(double referenceError,
                     double error,
                     double prediction,
                     double varianceIncrease,
                     double weight);

            //! Clear the error statistics.
            void clear();

            //! Check if we should discard the component.
            bool remove(core_t::TTime bucketLength, core_t::TTime period) const;

            //! Age the errors by \p factor.
            void age(double factor);

            //! Get a checksum for this object.
            std::uint64_t checksum(std::uint64_t seed) const;

        private:
            using TMaxAccumulator = common::CBasicStatistics::SMax<double>::TAccumulator;
            using TVector = common::CVectorNx1<common::CFloatStorage, 3>;
            using TVectorMeanAccumulator =
                common::CBasicStatistics::SSampleMean<TVector>::TAccumulator;

        private:
            //! Truncate large, i.e. more than 6 sigma, errors.
            TVector winsorise(const TVector& squareError) const;

        private:
            //! The vector mean errors:
            //! <pre>
            //! | excluding all components from the prediction |
            //! |  including the component in the prediction   |
            //! | excluding the component from the prediction  |
            //! </pre>
            TVectorMeanAccumulator m_MeanErrors;

            //! The maximum increase in variance due to the component.
            TMaxAccumulator m_MaxVarianceIncrease;
        };

        using TComponentErrorsVec = std::vector<CComponentErrors>;
        using TComponentErrorsPtrVec = std::vector<CComponentErrors*>;

        //! \brief The seasonal components of the decomposition.
        class MATHS_TIME_SERIES_EXPORT CSeasonal {
        public:
            //! Initialize by reading state from \p traverser.
            bool acceptRestoreTraverser(double decayRate,
                                        core_t::TTime bucketLength,
                                        core::CStateRestoreTraverser& traverser);

            //! Persist state by passing information to \p inserter.
            void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

            //! Set the decay rate.
            void decayRate(double decayRate);

            //! Age the seasonal components to account for the interval \p end
            //! - \p start elapsed time.
            void propagateForwards(core_t::TTime start, core_t::TTime end);

            //! Clear the components' prediction errors.
            void clearPredictionErrors();

            //! Get the combined size of the seasonal components.
            std::size_t size() const;

            //! Get the components.
            const maths_t::TSeasonalComponentVec& components() const;
            //! Get the components.
            maths_t::TSeasonalComponentVec& components();

            //! Get the state to update.
            void componentsErrorsAndDeltas(core_t::TTime time,
                                           TSeasonalComponentPtrVec& components,
                                           TComponentErrorsPtrVec& errors,
                                           TDoubleVec& deltas);

            //! Append the predictions at \p time.
            void appendPredictions(core_t::TTime time, TDoubleVec& predictions) const;

            //! Check if we need to interpolate any of the components.
            bool shouldInterpolate(core_t::TTime time) const;

            //! Interpolate the components at \p time.
            void interpolate(core_t::TTime time, bool refine);

            //! Check if any of the components has been initialized.
            bool initialized() const;

            //! Add and initialize a new component.
            void add(const CSeasonalTime& seasonalTime,
                     std::size_t size,
                     double decayRate,
                     double bucketLength,
                     core_t::TTime maxTimeShiftPerPeriod,
                     common::CSplineTypes::EBoundaryCondition boundaryCondition,
                     core_t::TTime startTime,
                     core_t::TTime endTime,
                     const TFloatMeanAccumulatorVec& values);

            //! Apply \p change to the components.
            void apply(const CChangePoint& change);

            //! Refresh state after adding new components.
            void refreshForNewComponents();

            //! Remove all components masked by \p removeComponentsMask.
            bool remove(const TBoolVec& removeComponentsMask);

            //! Remove any components with invalid values
            bool removeComponentsWithBadValues(core_t::TTime);

            //! Remove low value components
            bool prune(core_t::TTime time, core_t::TTime bucketLength);

            //! Shift the components' time origin to \p time.
            void shiftOrigin(core_t::TTime time);

            //! Get a checksum for this object.
            std::uint64_t checksum(std::uint64_t seed = 0) const;

            //! Debug the memory used by this object.
            void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

            //! Get the memory used by this object.
            std::size_t memoryUsage() const;

        private:
            //! The components.
            maths_t::TSeasonalComponentVec m_Components;

            //! The components' prediction errors.
            TComponentErrorsVec m_PredictionErrors;

            //! Befriend a helper class used by the unit tests
            friend class CTimeSeriesDecompositionTest::CNanInjector;
        };

        using TSeasonalPtr = std::unique_ptr<CSeasonal>;

        //! \brief Calendar periodic components of the decomposition.
        class MATHS_TIME_SERIES_EXPORT CCalendar {
        public:
            //! Initialize by reading state from \p traverser.
            bool acceptRestoreTraverser(double decayRate,
                                        core_t::TTime bucketLength,
                                        core::CStateRestoreTraverser& traverser);

            //! Persist state by passing information to \p inserter.
            void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

            //! Set the decay rate.
            void decayRate(double decayRate);

            //! Age the calendar components to account for the interval \p end
            //! - \p start elapsed time.
            void propagateForwards(core_t::TTime start, core_t::TTime end);

            //! Clear the components' prediction errors.
            void clearPredictionErrors();

            //! Get the combined size of the seasonal components.
            std::size_t size() const;

            //! Get the components.
            const maths_t::TCalendarComponentVec& components() const;

            //! Check if there is already a component for \p feature.
            bool haveComponent(CCalendarFeature feature) const;

            //! Get the state to update.
            void componentsAndErrors(core_t::TTime time,
                                     TCalendarComponentPtrVec& components,
                                     TComponentErrorsPtrVec& errors);

            //! Append the predictions at \p time.
            void appendPredictions(core_t::TTime time, TDoubleVec& predictions) const;

            //! Check if we need to interpolate any of the components.
            bool shouldInterpolate(core_t::TTime time) const;

            //! Interpolate the components at \p time.
            void interpolate(core_t::TTime time, bool refine);

            //! Check if any of the components has been initialized.
            bool initialized() const;

            //! Add and initialize a new component.
            void add(const CCalendarFeature& feature,
                     core_t::TTime timeZoneOffset,
                     std::size_t size,
                     double decayRate,
                     double bucketLength);

            //! Apply \p change to the components.
            void apply(const CChangePoint& change);

            //! Remove low value components.
            bool prune(core_t::TTime time, core_t::TTime bucketLength);

            //! Get a checksum for this object.
            std::uint64_t checksum(std::uint64_t seed = 0) const;

            //! Debug the memory used by this object.
            void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

            //! Get the memory used by this object.
            std::size_t memoryUsage() const;

            //! Remove any components with invalid values
            bool removeComponentsWithBadValues(core_t::TTime time);

        private:
            //! The calendar components.
            maths_t::TCalendarComponentVec m_Components;

            //! The components' prediction errors.
            TComponentErrorsVec m_PredictionErrors;
        };

        using TCalendarPtr = std::unique_ptr<CCalendar>;

    private:
        //! Get the total size of the components.
        std::size_t size() const;

        //! Get the maximum permitted size of the components.
        std::size_t maxSize() const;

        //! Add new seasonal components.
        void addSeasonalComponents(const CSeasonalDecomposition& components);

        //! Add a new calendar component.
        void addCalendarComponent(const CCalendarFeature& feature, core_t::TTime timeZoneOffset);

        //! Fit the trend component \p component to \p values.
        void fitTrend(core_t::TTime startTime,
                      core_t::TTime dt,
                      const TFloatMeanAccumulatorVec& values,
                      CTrendComponent& trend) const;

        //! Clear all component error statistics.
        void clearComponentErrors();

        //! Handle \p symbol.
        void apply(std::size_t symbol, const SMessage& message);

        //! Test to see if using the trend improves prediction accuracy.
        bool shouldUseTrendForPrediction();

        //! Check if we should interpolate.
        bool shouldInterpolate(core_t::TTime time);

        //! Maybe re-interpolate the components.
        void interpolate(const SMessage& message);

        //! Shift the various regression model time origins to \p time.
        void shiftOrigin(core_t::TTime time);

        //! Get the components in canonical form.
        //!
        //! This standardizes the level and gradient across the various
        //! components. In particular, common offsets and gradients are
        //! shifted into the long term trend or in the absence of that
        //! the shortest component.
        void canonicalize(core_t::TTime time);

    private:
        //! The state machine.
        core::CStateMachine m_Machine;

        //! Controls the rate at which information is lost.
        double m_DecayRate;

        //! The raw data bucketing interval.
        core_t::TTime m_BucketLength;

        //! Sets the gain used when updating with a new value.
        //!
        //! \see CGainController for more details.
        CGainController m_GainController;

        //! The number of buckets to use to estimate a periodic component.
        std::size_t m_SeasonalComponentSize;

        //! The number of buckets to use to estimate a periodic component.
        std::size_t m_CalendarComponentSize;

        //! The long term trend.
        CTrendComponent m_Trend;

        //! The seasonal components.
        TSeasonalPtr m_Seasonal;

        //! The calendar components.
        TCalendarPtr m_Calendar;

        //! The mean error variance scale for the components.
        TFloatMeanAccumulator m_MeanVarianceScale;

        //! The moments of the error in the predictions excluding the trend.
        TMeanVarAccumulator m_PredictionErrorWithoutTrend;

        //! The moments of the error in the predictions including the trend.
        TMeanVarAccumulator m_PredictionErrorWithTrend;

        //! Supplied with the prediction residuals if a component is added.
        TComponentChangeCallback m_ComponentChangeCallback;

        //! Supplied with an annotation if a component is added.
        maths_t::TModelAnnotationCallback m_ModelAnnotationCallback;

        //! Set to true if the trend model should be used for prediction.
        bool m_UsingTrendForPrediction = false;

        //! Befriend a helper class used by the unit tests
        friend class CTimeSeriesDecompositionTest::CNanInjector;
    };
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CSeasonalityTest& lhs,
                 CTimeSeriesDecompositionDetail::CSeasonalityTest& rhs) {
    lhs.swap(rhs);
}

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CCalendarTest& lhs,
                 CTimeSeriesDecompositionDetail::CCalendarTest& rhs) {
    lhs.swap(rhs);
}

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CComponents& lhs,
                 CTimeSeriesDecompositionDetail::CComponents& rhs) {
    lhs.swap(rhs);
}
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionDetail_h
