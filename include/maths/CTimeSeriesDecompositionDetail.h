/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionDetail_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionDetail_h

#include <core/CSmallVector.h>
#include <core/CStateMachine.h>
#include <core/CoreTypes.h>

#include <maths/CCalendarComponent.h>
#include <maths/CPeriodicityHypothesisTests.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/CTrendComponent.h>
#include <maths/CTrendTests.h>
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
    struct MATHS_EXPORT SAddValue : public SMessage, private core::CNonCopyable {
        SAddValue(core_t::TTime time,
                  core_t::TTime lastTime,
                  double value,
                  const maths_t::TDoubleWeightsAry& weights,
                  double trend,
                  double seasonal,
                  double calendar,
                  const TPredictor& predictor,
                  const CPeriodicityHypothesisTestsConfig& periodicityTestConfig);

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
    class MATHS_EXPORT CHandler : core::CNonCopyable {
    public:
        CHandler();
        virtual ~CHandler();

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
    class MATHS_EXPORT CMediator : core::CNonCopyable {
    public:
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
        CPeriodicityTest(double decayRate, core_t::TTime bucketLength);
        CPeriodicityTest(const CPeriodicityTest& other);

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
        using TExpandingWindowPtr = std::shared_ptr<CExpandingWindow>;
        using TExpandingWindowPtrAry = boost::array<TExpandingWindowPtr, 2>;

        //! Test types (categorised as short and long period tests).
        enum ETest { E_Short, E_Long };

    private:
        //! The bucket lengths to use to test for short period components.
        static const TTimeVec SHORT_BUCKET_LENGTHS;

        //! The bucket lengths to use to test for long period components.
        static const TTimeVec LONG_BUCKET_LENGTHS;

    private:
        //! Handle \p symbol.
        void apply(std::size_t symbol, const SMessage& message);

        //! Check if we should run the periodicity test on \p window.
        bool shouldTest(const TExpandingWindowPtr& window, core_t::TTime time) const;

        //! Get a new \p test. (Warning owned by the caller.)
        CExpandingWindow* newWindow(ETest test) const;

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
        CCalendarTest(const CCalendarTest& other);

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
        using TCalendarCyclicTestPtr = std::shared_ptr<CCalendarCyclicTest>;

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
        class MATHS_EXPORT CScopeNotifyOnStateChange : core::CNonCopyable {
        public:
            CScopeNotifyOnStateChange(CComponents& components);
            ~CScopeNotifyOnStateChange();

            //! Check if the seasonal component's state changed.
            bool changed() const;

        private:
            //! The seasonal components this is watching.
            CComponents& m_Components;

            //! The flag used to watch for changes.
            bool m_Watcher;
        };

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

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

        //! Maybe re-interpolate the components.
        void interpolate(const SMessage& message, bool refine = true);

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
            //! \param[in] error The prediction error.
            //! \param[in] prediction The prediction from the component.
            //! \param[in] weight The weight of \p error.
            void add(double error, double prediction, double weight);

            //! Clear the error statistics.
            void clear();

            //! Check if we should discard \p seasonal.
            bool remove(core_t::TTime bucketLength, CSeasonalComponent& seasonal) const;

            //! Check if we should discard \p calendar.
            bool remove(core_t::TTime bucketLength, CCalendarComponent& calendar) const;

            //! Age the errors by \p factor.
            void age(double factor);

            //! Get a checksum for this object.
            uint64_t checksum(uint64_t seed) const;

        private:
            //! Truncate large, i.e. more than 6 sigma, errors.
            static double winsorise(double squareError, const TFloatMeanAccumulator& variance);

        private:
            //! The mean prediction error in the window.
            TFloatMeanAccumulator m_MeanErrorWithComponent;

            //! The mean prediction error in the window without the component.
            TFloatMeanAccumulator m_MeanErrorWithoutComponent;
        };

        using TComponentErrorsVec = std::vector<CComponentErrors>;
        using TComponentErrorsPtrVec = std::vector<CComponentErrors*>;

        //! \brief The seasonal components of the decomposition.
        struct MATHS_EXPORT SSeasonal {
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

            //! Get the combined size of the seasonal components.
            std::size_t size() const;

            //! Get the state to update.
            void componentsErrorsAndDeltas(core_t::TTime time,
                                           TSeasonalComponentPtrVec& components,
                                           TComponentErrorsPtrVec& errors,
                                           TDoubleVec& deltas);

            //! Check if we need to interpolate any of the components.
            bool shouldInterpolate(core_t::TTime time, core_t::TTime last) const;

            //! Interpolate the components at \p time.
            void interpolate(core_t::TTime time, core_t::TTime last, bool refine);

            //! Check if any of the components has been initialized.
            bool initialized() const;

            //! Remove low value components
            bool prune(core_t::TTime time, core_t::TTime bucketLength);

            //! Shift the components' time origin to \p time.
            void shiftOrigin(core_t::TTime time);

            //! Get a checksum for this object.
            uint64_t checksum(uint64_t seed = 0) const;

            //! Debug the memory used by this object.
            void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

            //! Get the memory used by this object.
            std::size_t memoryUsage() const;

            //! The seasonal components.
            maths_t::TSeasonalComponentVec s_Components;

            //! The prediction errors relating to the component.
            TComponentErrorsVec s_PredictionErrors;
        };

        using TSeasonalPtr = std::shared_ptr<SSeasonal>;

        //! \brief Calendar periodic components of the decomposition.
        struct MATHS_EXPORT SCalendar {
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

            //! Get the combined size of the seasonal components.
            std::size_t size() const;

            //! Check if there is already a component for \p feature.
            bool haveComponent(CCalendarFeature feature) const;

            //! Get the state to update.
            void componentsAndErrors(core_t::TTime time,
                                     TCalendarComponentPtrVec& components,
                                     TComponentErrorsPtrVec& errors);

            //! Check if we need to interpolate any of the components.
            bool shouldInterpolate(core_t::TTime time, core_t::TTime last) const;

            //! Interpolate the components at \p time.
            void interpolate(core_t::TTime time, core_t::TTime last, bool refine);

            //! Check if any of the components has been initialized.
            bool initialized() const;

            //! Remove low value components.
            bool prune(core_t::TTime time, core_t::TTime bucketLength);

            //! Get a checksum for this object.
            uint64_t checksum(uint64_t seed = 0) const;

            //! Debug the memory used by this object.
            void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

            //! Get the memory used by this object.
            std::size_t memoryUsage() const;

            //! The calendar components.
            maths_t::TCalendarComponentVec s_Components;

            //! The prediction errors after removing the component.
            TComponentErrorsVec s_PredictionErrors;
        };

        using TCalendarPtr = std::shared_ptr<SCalendar>;

    private:
        //! Get the total size of the components.
        std::size_t size() const;

        //! Get the maximum permitted size of the components.
        std::size_t maxSize() const;

        //! Add new seasonal components to \p components.
        bool addSeasonalComponents(const CPeriodicityHypothesisTestsResult& result,
                                   const CExpandingWindow& window,
                                   const TPredictor& predictor,
                                   CTrendComponent& trend,
                                   maths_t::TSeasonalComponentVec& components,
                                   TComponentErrorsVec& errors) const;

        //! Add a new calendar component to \p components.
        bool addCalendarComponent(const CCalendarFeature& feature,
                                  core_t::TTime time,
                                  maths_t::TCalendarComponentVec& components,
                                  TComponentErrorsVec& errors) const;

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

        //! The moments of the values added.
        TMeanVarAccumulator m_Moments;

        //! The moments of the values added after subtracting a trend.
        TMeanVarAccumulator m_MomentsMinusTrend;

        //! Set to true if the trend model should be used for prediction.
        bool m_UsingTrendForPrediction;

        //! Set to true if non-null when the seasonal components change.
        bool* m_Watcher;
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
