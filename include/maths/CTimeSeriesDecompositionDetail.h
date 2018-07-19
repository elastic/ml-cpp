/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionDetail_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionDetail_h

#include <core/CSmallVector.h>
#include <core/CStateMachine.h>
#include <core/CoreTypes.h>

#include <maths/CCalendarComponent.h>
#include <maths/CCalendarCyclicTest.h>
#include <maths/CExpandingWindow.h>
#include <maths/CPeriodicityHypothesisTests.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/CTrendComponent.h>
#include <maths/ImportExport.h>

#include <boost/ref.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace ml {
namespace maths {
class CExpandingWindow;
class CTimeSeriesDecomposition;

//! \brief Utilities for computing the decomposition.
class MATHS_EXPORT CTimeSeriesDecompositionDetail {
public:
    using TPredictor = std::function<double(core_t::TTime)>;
    using TDoubleVec = std::vector<double>;
    using TTimeVec = std::vector<core_t::TTime>;
    class CMediator;

    //! \brief The base message passed.
    struct MATHS_EXPORT SMessage {
        SMessage(core_t::TTime time, core_t::TTime lastTime);

        //! The message time.
        core_t::TTime s_Time;
        //! The last update time.
        core_t::TTime s_LastTime;
    };

    //! \brief The message passed to add a point.
    struct MATHS_EXPORT SAddValue : public SMessage {
        SAddValue(core_t::TTime time,
                  core_t::TTime lastTime,
                  double value,
                  const maths_t::TDoubleWeightsAry& weights,
                  double trend,
                  double seasonal,
                  double calendar,
                  const TPredictor& predictor,
                  const CPeriodicityHypothesisTestsConfig& periodicityTestConfig);
        SAddValue(const SAddValue&) = delete;
        SAddValue& operator=(const SAddValue&) = delete;

        //! The value to add.
        double s_Value;
        //! The weights of associated with the value.
        const maths_t::TDoubleWeightsAry& s_Weights;
        //! The trend component prediction at the value's time.
        double s_Trend;
        //! The seasonal component prediction at the value's time.
        double s_Seasonal;
        //! The calendar component prediction at the value's time.
        double s_Calendar;
        //! The predictor for value.
        TPredictor s_Predictor;
        //! The periodicity test configuration.
        CPeriodicityHypothesisTestsConfig s_PeriodicityTestConfig;
    };

    //! \brief The message passed to indicate periodic components have
    //! been detected.
    struct MATHS_EXPORT SDetectedSeasonal : public SMessage {
        SDetectedSeasonal(core_t::TTime time,
                          core_t::TTime lastTime,
                          const CPeriodicityHypothesisTestsResult& result,
                          const CExpandingWindow& window,
                          const TPredictor& predictor);

        //! The components found.
        CPeriodicityHypothesisTestsResult s_Result;
        //! The window tested.
        const CExpandingWindow& s_Window;
        //! The predictor for window values.
        TPredictor s_Predictor;
    };

    //! \brief The message passed to indicate calendar components have
    //! been detected.
    struct MATHS_EXPORT SDetectedCalendar : public SMessage {
        SDetectedCalendar(core_t::TTime time, core_t::TTime lastTime, CCalendarFeature feature);

        //! The calendar feature found.
        CCalendarFeature s_Feature;
    };

    //! \brief The message passed to indicate new components are being
    //! modeled.
    struct MATHS_EXPORT SNewComponents : public SMessage {
        enum EComponent {
            E_DiurnalSeasonal,
            E_GeneralSeasonal,
            E_CalendarCyclic
        };

        SNewComponents(core_t::TTime time, core_t::TTime lastTime, EComponent component);

        //! The type of component.
        EComponent s_Component;
    };

    //! \brief The basic interface for one aspect of the modeling of a time
    //! series decomposition.
    class MATHS_EXPORT CHandler {
    public:
        CHandler();
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
        virtual void handle(const SNewComponents& message);

        //! Set the mediator.
        void mediator(CMediator* mediator);

        //! Get the mediator.
        CMediator* mediator() const;

    private:
        //! The controller responsible for forwarding messages.
        CMediator* m_Mediator;
    };

    //! \brief Manages communication between handlers.
    class MATHS_EXPORT CMediator {
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
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using THandlerRef = boost::reference_wrapper<CHandler>;
        using THandlerRefVec = std::vector<THandlerRef>;

    private:
        //! The handlers which have added by registration.
        THandlerRefVec m_Handlers;
    };

    //! \brief Scans through increasingly low frequencies looking for custom
    //! diurnal and any other large amplitude seasonal components.
    class MATHS_EXPORT CPeriodicityTest : public CHandler {
    public:
        //! Test types (categorised as short and long period tests).
        enum ETest { E_Short, E_Long };

    public:
        CPeriodicityTest(double decayRate, core_t::TTime bucketLength);
        CPeriodicityTest(const CPeriodicityTest& other, bool isForForecast = false);
        CPeriodicityTest& operator=(const CPeriodicityTest&) = delete;

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Efficiently swap the state of this and \p other.
        void swap(CPeriodicityTest& other);

        //! Update the test with a new value.
        virtual void handle(const SAddValue& message);

        //! Reset the test.
        virtual void handle(const SNewComponents& message);

        //! Test to see whether any seasonal components are present.
        void test(const SAddValue& message);

        //! Age the test to account for the interval \p end - \p start
        //! elapsed time.
        void propagateForwards(core_t::TTime start, core_t::TTime end);

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TTimeAry = boost::array<core_t::TTime, 2>;
        using TExpandingWindowPtr = std::unique_ptr<CExpandingWindow>;
        using TExpandingWindowPtrAry = boost::array<TExpandingWindowPtr, 2>;

    private:
        //! The bucket lengths to use to test for short period components.
        static const TTimeVec SHORT_BUCKET_LENGTHS;

        //! The bucket lengths to use to test for long period components.
        static const TTimeVec LONG_BUCKET_LENGTHS;

    private:
        //! Handle \p symbol.
        void apply(std::size_t symbol, const SMessage& message);

        //! Check if we should run the periodicity test on \p window.
        bool shouldTest(ETest test, core_t::TTime time) const;

        //! Get a new \p test. (Warning: this is owned by the caller.)
        CExpandingWindow* newWindow(ETest test, bool deflate = true) const;

        //! Account for memory that is not yet allocated
        //! during the initial state
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
    class MATHS_EXPORT CCalendarTest : public CHandler {
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
        virtual void handle(const SAddValue& message);

        //! Reset the test.
        virtual void handle(const SNewComponents& message);

        //! Test to see whether any seasonal components are present.
        void test(const SMessage& message);

        //! Age the test to account for the interval \p end - \p start
        //! elapsed time.
        void propagateForwards(core_t::TTime start, core_t::TTime end);

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

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
    class MATHS_EXPORT CComponents : public CHandler {
    public:
        CComponents(double decayRate, core_t::TTime bucketLength, std::size_t seasonalComponentSize);
        CComponents(const CComponents& other);

        //! \brief Watches to see if the seasonal components state changes.
        class MATHS_EXPORT CScopeNotifyOnStateChange {
        public:
            CScopeNotifyOnStateChange(CComponents& components);
            ~CScopeNotifyOnStateChange();
            CScopeNotifyOnStateChange(const CScopeNotifyOnStateChange&) = delete;
            CScopeNotifyOnStateChange& operator=(const CScopeNotifyOnStateChange&) = delete;

            //! Check if the seasonal component's state changed.
            bool changed() const;

        private:
            //! The seasonal components this is watching.
            CComponents& m_Components;

            //! The flag used to watch for changes.
            bool m_Watcher;
        };

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                    core_t::TTime lastValueTime,
                                    core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Efficiently swap the state of this and \p other.
        void swap(CComponents& other);

        //! Update the components with a new value.
        virtual void handle(const SAddValue& message);

        //! Create new seasonal components.
        virtual void handle(const SDetectedSeasonal& message);

        //! Create a new calendar component.
        virtual void handle(const SDetectedCalendar& message);

        //! Start using the trend for prediction.
        void useTrendForPrediction();

        //! Test to see if using the trend improves prediction accuracy.
        bool shouldUseTrendForPrediction();

        //! Apply \p shift to the level at \p time and \p value.
        void shiftLevel(core_t::TTime time, double value, double shift);

        //! Apply a linear scale of \p scale.
        void linearScale(core_t::TTime time, double scale);

        //! Maybe re-interpolate the components.
        void interpolate(const SMessage& message);

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

        //! Get configuration for the periodicity test.
        CPeriodicityHypothesisTestsConfig periodicityTestConfig() const;

        //! Get the mean value of the baseline in the vicinity of \p time.
        double meanValue(core_t::TTime time) const;

        //! Get the mean variance of the baseline.
        double meanVariance() const;

        //! Get the mean error variance scale for the components.
        double meanVarianceScale() const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TOptionalDouble = boost::optional<double>;
        using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TSeasonalComponentPtrVec = std::vector<CSeasonalComponent*>;
        using TCalendarComponentPtrVec = std::vector<CCalendarComponent*>;
        using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;

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
        class MATHS_EXPORT CGainController {
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
            uint64_t checksum(uint64_t seed) const;

        private:
            using TRegression = CRegression::CLeastSquaresOnline<1>;

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
        class MATHS_EXPORT CComponentErrors {
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
            uint64_t checksum(uint64_t seed) const;

        private:
            using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;
            using TVector = CVectorNx1<CFloatStorage, 3>;
            using TVectorMeanAccumulator = CBasicStatistics::SSampleMean<TVector>::TAccumulator;

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
        class MATHS_EXPORT CSeasonal {
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
            bool shouldInterpolate(core_t::TTime time, core_t::TTime last) const;

            //! Interpolate the components at \p time.
            void interpolate(core_t::TTime time, core_t::TTime last, bool refine);

            //! Check if any of the components has been initialized.
            bool initialized() const;

            //! Add and initialize a new component.
            void add(const CSeasonalTime& seasonalTime,
                     std::size_t size,
                     double decayRate,
                     double bucketLength,
                     CSplineTypes::EBoundaryCondition boundaryCondition,
                     core_t::TTime startTime,
                     core_t::TTime endTime,
                     const TFloatMeanAccumulatorVec& values);

            //! Refresh state after adding new components.
            void refreshForNewComponents();

            //! Remove all components excluded by adding the component corresponding
            //! to \p time.
            void removeExcludedComponents(const CSeasonalTime& time);

            //! Remove low value components
            bool prune(core_t::TTime time, core_t::TTime bucketLength);

            //! Shift the components' time origin to \p time.
            void shiftOrigin(core_t::TTime time);

            //! Linearly scale the components' by \p scale.
            void linearScale(core_t::TTime time, double scale);

            //! Get a checksum for this object.
            uint64_t checksum(uint64_t seed = 0) const;

            //! Debug the memory used by this object.
            void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

            //! Get the memory used by this object.
            std::size_t memoryUsage() const;

        private:
            //! The components.
            maths_t::TSeasonalComponentVec m_Components;

            //! The components' prediction errors.
            TComponentErrorsVec m_PredictionErrors;
        };

        using TSeasonalPtr = std::unique_ptr<CSeasonal>;

        //! \brief Calendar periodic components of the decomposition.
        class MATHS_EXPORT CCalendar {
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
            bool shouldInterpolate(core_t::TTime time, core_t::TTime last) const;

            //! Interpolate the components at \p time.
            void interpolate(core_t::TTime time, core_t::TTime last, bool refine);

            //! Check if any of the components has been initialized.
            bool initialized() const;

            //! Add and initialize a new component.
            void add(const CCalendarFeature& feature, std::size_t size, double decayRate, double bucketLength);

            //! Remove low value components.
            bool prune(core_t::TTime time, core_t::TTime bucketLength);

            //! Linearly scale the components' by \p scale.
            void linearScale(core_t::TTime time, double scale);

            //! Get a checksum for this object.
            uint64_t checksum(uint64_t seed = 0) const;

            //! Debug the memory used by this object.
            void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

            //! Get the memory used by this object.
            std::size_t memoryUsage() const;

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

        //! Add new seasonal components to \p components.
        bool addSeasonalComponents(const CPeriodicityHypothesisTestsResult& result,
                                   const CExpandingWindow& window,
                                   const TPredictor& predictor);

        //! Add a new calendar component to \p components.
        bool addCalendarComponent(const CCalendarFeature& feature, core_t::TTime time);

        //! Reweight the outlier values in \p values.
        //!
        //! These are the values with largest error w.r.t. \p predictor.
        void reweightOutliers(core_t::TTime startTime,
                              core_t::TTime dt,
                              TPredictor predictor,
                              TFloatMeanAccumulatorVec& values) const;

        //! Fit the trend component \p component to \p values.
        void fit(core_t::TTime startTime,
                 core_t::TTime dt,
                 const TFloatMeanAccumulatorVec& values,
                 CTrendComponent& trend) const;

        //! Clear all component error statistics.
        void clearComponentErrors();

        //! Handle \p symbol.
        void apply(std::size_t symbol, const SMessage& message);

        //! Check if we should interpolate.
        bool shouldInterpolate(core_t::TTime time, core_t::TTime last);

        //! Shift the various regression model time origins to \p time.
        void shiftOrigin(core_t::TTime time);

        //! Get the components in canonical form.
        //!
        //! This standardizes the level and gradient across the various
        //! components. In particular, common offsets and gradients are
        //! shifted into the long term trend or in the absence of that
        //! the shortest component.
        void canonicalize(core_t::TTime time);

        //! Set a watcher for state changes.
        void notifyOnNewComponents(bool* watcher);

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

        //! Set to true if the trend model should be used for prediction.
        bool m_UsingTrendForPrediction = false;

        //! Set to true if non-null when the seasonal components change.
        bool* m_Watcher = nullptr;
    };
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CPeriodicityTest& lhs,
                 CTimeSeriesDecompositionDetail::CPeriodicityTest& rhs) {
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

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionDetail_h
