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

#include <core/CoreTypes.h>
#include <core/CSmallVector.h>
#include <core/CStateMachine.h>

#include <maths/CCalendarComponent.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/CTrendTests.h>
#include <maths/ImportExport.h>

#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>

#include <cstddef>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Utilities for computing the decomposition.
class MATHS_EXPORT CTimeSeriesDecompositionDetail
{
    public:
        using TDoubleVec = std::vector<double>;
        using TTimeVec = std::vector<core_t::TTime>;
        using TRegression = CRegression::CLeastSquaresOnline<3, double>;
        using TRegressionParameterProcess = CRegression::CLeastSquaresOnlineParameterProcess<4, double>;
        class CMediator;

        //! \brief The base message passed.
        struct MATHS_EXPORT SMessage
        {
            SMessage(void);
            SMessage(core_t::TTime time, core_t::TTime lastTime);

            //! The message time.
            core_t::TTime s_Time;

            //! The last update time.
            core_t::TTime s_LastTime;
        };

        //! \brief The message passed to add a point.
        struct MATHS_EXPORT SAddValue : public SMessage
        {
            SAddValue(core_t::TTime time,
                      core_t::TTime lastTime,
                      double value,
                      const maths_t::TWeightStyleVec &weightStyles,
                      const maths_t::TDouble4Vec &weights,
                      double trend,
                      double nonDiurnal,
                      double seasonal,
                      double calendar);
            //! The value to add.
            double s_Value;
            //! The styles of the weights. Both the count and the Winsorisation
            //! weight styles have an effect. See maths_t::ESampleWeightStyle
            //! for more details.
            const maths_t::TWeightStyleVec &s_WeightStyles;
            //! The weights of associated with the value. The smaller the count
            //! weight the less influence the value has on the trend and it's
            //! local variance.
            const maths_t::TDouble4Vec &s_Weights;
            //! The trend component prediction at the value's time.
            double s_Trend;
            //! The non daily/weekly seasonal components' prediction at the
            //! value's time.
            double s_NonDiurnal;
            //! The seasonal component prediction at the value's time.
            double s_Seasonal;
            //! The calendar component prediction at the value's time.
            double s_Calendar;
        };

        //! \brief The message passed to indicate a trend has been detected.
        struct MATHS_EXPORT SDetectedTrend : public SMessage
        {
            SDetectedTrend(core_t::TTime time,
                           core_t::TTime lastTime,
                           const CTrendTest &test);
            const CTrendTest &s_Test;
        };

        //! \brief The message passed to indicate diurnal periodic components
        //! have been detected.
        struct MATHS_EXPORT SDetectedDiurnal : public SMessage
        {
            SDetectedDiurnal(core_t::TTime time,
                             core_t::TTime lastTime,
                             const CPeriodicityTestResult &result,
                             const CDiurnalPeriodicityTest &test);
            CPeriodicityTestResult s_Result;
            const CDiurnalPeriodicityTest &s_Test;
        };

        //! \brief The message passed to indicate general periodic components
        //! have been detected.
        struct MATHS_EXPORT SDetectedNonDiurnal : public SMessage
        {
            SDetectedNonDiurnal(core_t::TTime time,
                                core_t::TTime lastTime,
                                bool discardLongTermTrend,
                                const CPeriodicityTestResult &result,
                                const CGeneralPeriodicityTest &test);
            bool s_DiscardLongTermTrend;
            CPeriodicityTestResult s_Result;
            const CGeneralPeriodicityTest &s_Test;
        };

        //! \brief The mssage passed to indicate calendar components have been
        //! detected.
        struct MATHS_EXPORT SDetectedCalendar : public SMessage
        {
            SDetectedCalendar(core_t::TTime time,
                              core_t::TTime lastTime,
                              CCalendarFeature feature);
            CCalendarFeature s_Feature;
        };

        //! \brief The message passed to indicate new diurnal components are
        //! being modeled.
        struct MATHS_EXPORT SNewComponents : public SMessage
        {
            enum EComponent
            {
                E_Trend,
                E_DiurnalSeasonal,
                E_GeneralSeasonal,
                E_CalendarCyclic
            };

            SNewComponents(core_t::TTime time,
                           core_t::TTime lastTime,
                           EComponent component);

            //! The type of component.
            EComponent s_Component;
        };

        //! \brief The basic interface for one aspect of the modeling of a time
        //! series decomposition.
        class MATHS_EXPORT CHandler : core::CNonCopyable
        {
            public:
                CHandler(void);
                virtual ~CHandler(void);

                //! Add a value.
                virtual void handle(const SAddValue &message);

                //! Handle when a trend is detected.
                virtual void handle(const SDetectedTrend &message);

                //! Handle when a diurnal component is detected.
                virtual void handle(const SDetectedDiurnal &message);

                //! Handle when a non-diurnal seasonal component is detected.
                virtual void handle(const SDetectedNonDiurnal &message);

                //! Handle when a calendar component is detected.
                virtual void handle(const SDetectedCalendar &message);

                //! Handle when a new component is being modeled.
                virtual void handle(const SNewComponents &message);

                //! Set the mediator.
                void mediator(CMediator *mediator);

                //! Get the mediator.
                CMediator *mediator(void) const;

            private:
                //! The controller responsible for forwarding messages.
                CMediator *m_Mediator;
        };

        //! \brief Manages communication between handlers.
        class MATHS_EXPORT CMediator : core::CNonCopyable
        {
            public:
                //! Forward \p message to all registered models.
                template<typename M>
                void forward(const M &message) const;

                //! Register \p handler.
                void registerHandler(CHandler &handler);

                //! Debug the memory used by this object.
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this object.
                std::size_t memoryUsage(void) const;

            private:
                using THandlerRef = boost::reference_wrapper<CHandler>;
                using THandlerRefVec = std::vector<THandlerRef>;

            private:
                //! The handlers which have added by registration.
                THandlerRefVec m_Handlers;
        };

        //! \brief Tests for a long term trend.
        class MATHS_EXPORT CLongTermTrendTest : public CHandler
        {
            public:
                CLongTermTrendTest(double decayRate);
                CLongTermTrendTest(const CLongTermTrendTest &other);

                //! Initialize by reading state from \p traverser.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to \p inserter.
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Efficiently swap the state of this and \p other.
                void swap(CLongTermTrendTest &other);

                //! Update the test with a new value.
                virtual void handle(const SAddValue &message);

                //! Reset the test if still testing.
                virtual void handle(const SNewComponents &message);

                //! Check if the time series has shifted level.
                void test(const SMessage &message);

                //! Set the decay rate.
                void decayRate(double decayRate);

                //! Age the test to account for the interval \p end - \p start
                //! elapsed time.
                void propagateForwards(core_t::TTime start, core_t::TTime end);

                //! Roll time forwards by \p skipInterval.
                void skipTime(core_t::TTime skipInterval);

                //! Get a checksum for this object.
                uint64_t checksum(uint64_t seed = 0) const;

                //! Debug the memory used by this object.
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this object.
                std::size_t memoryUsage(void) const;

            private:
                using TTrendTestPtr = boost::shared_ptr<CTrendTest>;

            private:
                //! Handle \p symbol.
                void apply(std::size_t symbol, const SMessage &message);

                //! Check if we should run a test.
                bool shouldTest(core_t::TTime time);

                //! Get the interval between tests.
                core_t::TTime testInterval(void) const;

            private:
                //! The state machine.
                core::CStateMachine m_Machine;

                //! The maximum rate at which information is lost.
                double m_MaximumDecayRate;

                //! The next time to test for a long term trend.
                core_t::TTime m_NextTestTime;

                //! The test for a long term trend.
                TTrendTestPtr m_Test;
        };

        //! \brief Tests for daily and weekly periodic components and weekend
        //! weekday splits of the time series.
        class MATHS_EXPORT CDiurnalTest : public CHandler
        {
            public:
                CDiurnalTest(double decayRate, core_t::TTime bucketLength);
                CDiurnalTest(const CDiurnalTest &other);

                //! Initialize by reading state from \p traverser.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to \p inserter.
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Efficiently swap the state of this and \p other.
                void swap(CDiurnalTest &other);

                //! Update the test with a new value.
                virtual void handle(const SAddValue &message);

                //! Reset the test.
                virtual void handle(const SNewComponents &message);

                //! Test to see whether any seasonal components are present.
                void test(const SMessage &message);

                //! Age the test to account for the interval \p end - \p start
                //! elapsed time.
                void propagateForwards(core_t::TTime start, core_t::TTime end);

                //! Roll time forwards by \p skipInterval.
                void skipTime(core_t::TTime skipInterval);

                //! Get a checksum for this object.
                uint64_t checksum(uint64_t seed = 0) const;

                //! Debug the memory used by this object.
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this object.
                std::size_t memoryUsage(void) const;

            private:
                using TRandomizedPeriodicityTestPtr = boost::shared_ptr<CRandomizedPeriodicityTest>;
                using TPeriodicityTestPtr = boost::shared_ptr<CDiurnalPeriodicityTest>;

            private:
                //! Handle \p symbol.
                void apply(std::size_t symbol, const SMessage &message);

                //! Check if we should run a test.
                bool shouldTest(core_t::TTime time);

                //! Get the interval between tests.
                core_t::TTime testInterval(void) const;

                //! Get the time at which to time out the regular test.
                core_t::TTime timeOutRegularTest(void) const;

                //! Account for memory that is not yet allocated
                //! during the initial state
                std::size_t extraMemoryOnInitialization(void) const;

            private:
                //! The state machine.
                core::CStateMachine m_Machine;

                //! Controls the rate at which information is lost.
                double m_DecayRate;

                //! The raw data bucketing interval.
                core_t::TTime m_BucketLength;

                //! The next time to test for periodic components.
                core_t::TTime m_NextTestTime;

                //! The time at which we began regular testing.
                core_t::TTime m_StartedRegularTest;

                //! The time at which we switch to the small test to save memory.
                core_t::TTime m_TimeOutRegularTest;

                //! The test for periodic components.
                TPeriodicityTestPtr m_RegularTest;

                //! A small but slower test for periodic components that is used
                //! after a while if the regular test is inconclusive.
                TRandomizedPeriodicityTestPtr m_SmallTest;

                //! The result of the last test for periodic components.
                CPeriodicityTestResult m_Periods;
        };

        //! \brief Scans through increasingly low frequencies looking for the
        //! the largest amplitude seasonal components.
        class MATHS_EXPORT CNonDiurnalTest : public CHandler
        {
            public:
                CNonDiurnalTest(double decayRate, core_t::TTime bucketLength);
                CNonDiurnalTest(const CNonDiurnalTest &other);

                //! Initialize by reading state from \p traverser.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to \p inserter.
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Efficiently swap the state of this and \p other.
                void swap(CNonDiurnalTest &other);

                //! Update the test with a new value.
                virtual void handle(const SAddValue &message);

                //! Reset the test.
                virtual void handle(const SNewComponents &message);

                //! Test to see whether any seasonal components are present.
                void test(const SMessage &message);

                //! Age the test to account for the interval \p end - \p start
                //! elapsed time.
                void propagateForwards(core_t::TTime start, core_t::TTime end);

                //! Roll time forwards by \p skipInterval.
                void skipTime(core_t::TTime skipInterval);

                //! Get a checksum for this object.
                uint64_t checksum(uint64_t seed = 0) const;

                //! Debug the memory used by this object.
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this object.
                std::size_t memoryUsage(void) const;

            private:
                using TTimeAry = boost::array<core_t::TTime, 2>;
                using TScanningPeriodicityTestPtr = boost::shared_ptr<CScanningPeriodicityTest>;
                using TScanningPeriodicityTestPtrAry = boost::array<TScanningPeriodicityTestPtr, 2>;

                //! Test types (categorised as short and long period tests).
                enum ETest { E_Short, E_Long };

            private:
                //! The size of the periodicity test.
                static const std::size_t TEST_SIZE;

                //! The bucket lengths to use to test for short period components.
                static const TTimeVec SHORT_BUCKET_LENGTHS;

                //! The bucket lengths to use to test for long period components.
                static const TTimeVec LONG_BUCKET_LENGTHS;

            private:
                //! Handle \p symbol.
                void apply(std::size_t symbol,
                           const SMessage &message,
                           const TTimeAry &offsets = {{345600, 345600}}); // 4 * DAY

                //! Get a new \p test. (Warning owned by the caller.)
                CScanningPeriodicityTest *newTest(ETest test) const;

                //! Account for memory that is not yet allocated
                //! during the initial state
                std::size_t extraMemoryOnInitialization(void) const;

            private:
                //! The state machine.
                core::CStateMachine m_Machine;

                //! Controls the rate at which information is lost.
                double m_DecayRate;

                //! The raw data bucketing interval.
                core_t::TTime m_BucketLength;

                //! The test for arbitrary periodic components.
                TScanningPeriodicityTestPtrAry m_Tests;
        };

        //! \brief Tests for cyclic calendar components explaining large prediction
        //! errors.
        class MATHS_EXPORT CCalendarTest : public CHandler
        {
            public:
                CCalendarTest(double decayRate, core_t::TTime bucketLength);
                CCalendarTest(const CCalendarTest &other);

                //! Initialize by reading state from \p traverser.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to \p inserter.
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Efficiently swap the state of this and \p other.
                void swap(CCalendarTest &other);

                //! Update the test with a new value.
                virtual void handle(const SAddValue &message);

                //! Reset the test.
                virtual void handle(const SNewComponents &message);

                //! Test to see whether any seasonal components are present.
                void test(const SMessage &message);

                //! Age the test to account for the interval \p end - \p start
                //! elapsed time.
                void propagateForwards(core_t::TTime start, core_t::TTime end);

                //! Roll time forwards to \p time.
                void advanceTimeTo(core_t::TTime time);

                //! Get a checksum for this object.
                uint64_t checksum(uint64_t seed = 0) const;

                //! Debug the memory used by this object.
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this object.
                std::size_t memoryUsage(void) const;

            private:
                using TCalendarCyclicTestPtr = boost::shared_ptr<CCalendarCyclicTest>;

            private:
                //! Handle \p symbol.
                void apply(std::size_t symbol, const SMessage &message);

                //! Check if we should run a test.
                bool shouldTest(core_t::TTime time);

                //! Get the month of \p time.
                int month(core_t::TTime time) const;

                //! Account for memory that is not yet allocated
                //! during the initial state
                std::size_t extraMemoryOnInitialization(void) const;
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

        //! \brief A reference to the long term trend.
        class MATHS_EXPORT CTrendCRef
        {
            public:
                using TMatrix = CSymmetricMatrixNxN<double, 4>;

            public:
                CTrendCRef(void);
                CTrendCRef(const TRegression &regression,
                           double variance,
                           core_t::TTime timeOrigin,
                           core_t::TTime lastUpdate,
                           const TRegressionParameterProcess &process);

                //! Check if the trend has been initialized.
                bool initialized(void) const;

                //! Get the count of values added to the trend.
                double count(void) const;

                //! Predict the long term trend at \p time with confidence
                //! interval \p confidence.
                maths_t::TDoubleDoublePr prediction(core_t::TTime time, double confidence) const;

                //! Get the variance about the long term trend.
                double variance(void) const;

                //! Get the covariance matrix of the regression parameters'
                //! at \p time.
                //!
                //! \param[out] result Filled in with the regression parameters'
                //! covariance matrix.
                bool covariances(TMatrix &result) const;

                //! Get the variance in the prediction due to drift in the
                //! regression model parameters expected by \p time.
                double varianceDueToParameterDrift(core_t::TTime time) const;

                //! Get the time at which to evaluate the regression model
                //! of the trend.
                double time(core_t::TTime time) const;

            private:
                //! The regression model of the trend.
                const TRegression *m_Trend;

                //! The variance of the prediction residuals.
                double m_Variance;

                //! The origin of the time coordinate system.
                core_t::TTime m_TimeOrigin;

                //! The time of the last update of the regression model.
                core_t::TTime m_LastUpdate;

                //! The Wiener process which describes the evolution of the
                //! regression model parameters.
                const TRegressionParameterProcess *m_ParameterProcess;
        };

        //! \brief Holds and updates the components of the decomposition.
        class MATHS_EXPORT CComponents : public CHandler
        {
            public:
                CComponents(double decayRate,
                            core_t::TTime bucketLength,
                            std::size_t seasonalComponentSize);
                CComponents(const CComponents &other);

                //! \brief Watches to see if the seasonal components state changes.
                class MATHS_EXPORT CScopeNotifyOnStateChange : core::CNonCopyable
                {
                    public:
                        CScopeNotifyOnStateChange(CComponents &components);
                        ~CScopeNotifyOnStateChange(void);

                        //! Check if the seasonal component's state changed.
                        bool changed(void) const;

                    private:
                        //! The seasonal components this is watching.
                        CComponents &m_Components;

                        //! The flag used to watch for changes.
                        bool m_Watcher;
                };

                //! Initialize by reading state from \p traverser.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to \p inserter.
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Efficiently swap the state of this and \p other.
                void swap(CComponents &other);

                //! Update the components with a new value.
                virtual void handle(const SAddValue &message);

                //! Create a new trend component.
                virtual void handle(const SDetectedTrend &message);

                //! Create new diurnal components.
                virtual void handle(const SDetectedDiurnal &message);

                //! Create new general seasonal component.
                virtual void handle(const SDetectedNonDiurnal &message);

                //! Create a new calendar component.
                virtual void handle(const SDetectedCalendar &message);

                //! Maybe re-interpolate the components.
                void interpolate(const SMessage &message);

                //! Set the decay rate.
                void decayRate(double decayRate);

                //! Age the components to account for the interval \p end - \p start
                //! elapsed time.
                void propagateForwards(core_t::TTime start, core_t::TTime end);

                //! Check if we're forecasting.
                bool forecasting(void) const;

                //! Start forecasting.
                void forecast(void);

                //! Check if the decomposition has any initialized components.
                bool initialized(void) const;

                //! Get the long term trend.
                CTrendCRef trend(void) const;

                //! Get the components.
                const maths_t::TSeasonalComponentVec &seasonal(void) const;

                //! Get the calendar components.
                const maths_t::TCalendarComponentVec &calendar(void) const;

                //! Get the mean value of the baseline in the vicinity of \p time.
                double meanValue(core_t::TTime time) const;

                //! Get the mean variance of the baseline.
                double meanVariance(void) const;

                //! Get a checksum for this object.
                uint64_t checksum(uint64_t seed = 0) const;

                //! Debug the memory used by this object.
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this object.
                std::size_t memoryUsage(void) const;

            private:
                using TOptionalDouble = boost::optional<double>;
                using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
                using TSeasonalComponentPtrVec = std::vector<CSeasonalComponent*>;
                using TCalendarComponentPtrVec = std::vector<CCalendarComponent*>;
                using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
                using TVector = CVectorNx1<double, 4>;

                //! \brief Tracks prediction errors with and without components.
                //!
                //! DESCRIPTION:\n
                //! This tracks the prediction errors with and without seasonal and
                //! calendar periodic components and tests to see if including the
                //! component is worthwhile.
                class MATHS_EXPORT CComponentErrors
                {
                    public:
                        //! Initialize from a delimited string.
                        bool fromDelimited(const std::string &str);

                        //! Convert to a delimited string.
                        std::string toDelimited(void) const;

                        //! Update the errors.
                        //!
                        //! \param[in] error The prediction error.
                        //! \param[in] prediction The prediction from the component.
                        //! \param[in] weight The weight of \p error.
                        void add(double error, double prediction, double weight);

                        //! Clear the error statistics.
                        void clear(void);

                        //! Check if we should discard \p seasonal.
                        bool remove(core_t::TTime bucketLength, CSeasonalComponent &seasonal) const;

                        //! Check if we should discard \p calendar.
                        bool remove(core_t::TTime bucketLength, CCalendarComponent &calendar) const;

                        //! Age the errors by \p factor.
                        void age(double factor);

                        //! Get a checksum for this object.
                        uint64_t checksum(uint64_t seed) const;

                    private:
                        //! Truncate large, i.e. more than 6 sigma, errors.
                        static double winsorise(double squareError,
                                                const TFloatMeanAccumulator &variance);

                    private:
                        //! The mean prediction error in the window.
                        TFloatMeanAccumulator m_MeanErrorWithComponent;

                        //! The mean prediction error in the window without the component.
                        TFloatMeanAccumulator m_MeanErrorWithoutComponent;
                };

                using TComponentErrorsVec = std::vector<CComponentErrors>;
                using TComponentErrorsPtrVec = std::vector<CComponentErrors*>;

                //! \brief The long term trend.
                struct MATHS_EXPORT STrend
                {
                    STrend(void);

                    //! Initialize by reading state from \p traverser.
                    bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                    //! Persist state by passing information to \p inserter.
                    void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                    //! Get a reference to this trend.
                    CTrendCRef reference(void) const;

                    //! Shift the regression's time origin to \p time.
                    void shiftOrigin(core_t::TTime time);

                    //! Get a checksum for this object.
                    uint64_t checksum(uint64_t seed = 0) const;

                    //! The regression model of the trend.
                    TRegression s_Regression;

                    //! The variance of the trend.
                    double s_Variance;

                    //! The origin of the time coordinate system.
                    core_t::TTime s_TimeOrigin;

                    //! The time of the last update of the regression model.
                    core_t::TTime s_LastUpdate;

                    //! The Wiener process which describes the evolution of the
                    //! regression model parameters.
                    TRegressionParameterProcess s_ParameterProcess;
                };

                using TTrendPtr = boost::shared_ptr<STrend>;

                //! \brief The seasonal components of the decomposition.
                struct MATHS_EXPORT SSeasonal
                {
                    //! Initialize by reading state from \p traverser.
                    bool acceptRestoreTraverser(double decayRate,
                                                core_t::TTime bucketLength,
                                                core::CStateRestoreTraverser &traverser);

                    //! Persist state by passing information to \p inserter.
                    void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                    //! Set the decay rate.
                    void decayRate(double decayRate);

                    //! Age the seasonal components to account for the interval \p end
                    //! - \p start elapsed time.
                    void propagateForwards(core_t::TTime start, core_t::TTime end);

                    //! Get the combined size of the seasonal components.
                    std::size_t size(void) const;

                    //! Check if there is already a component with \p period.
                    bool haveComponent(core_t::TTime period) const;

                    //! Get the state to update.
                    void componentsErrorsAndDeltas(core_t::TTime time,
                                                   TSeasonalComponentPtrVec &components,
                                                   TComponentErrorsPtrVec &errors,
                                                   TDoubleVec &deltas);

                    //! Check if we need to interpolate any of the components.
                    bool shouldInterpolate(core_t::TTime time, core_t::TTime last) const;

                    //! Interpolate the components at \p time.
                    void interpolate(core_t::TTime time, core_t::TTime last, bool refine);

                    //! Check if any of the components has been initialized.
                    bool initialized(void) const;

                    //! Remove low value components
                    bool prune(core_t::TTime time, core_t::TTime bucketLength);

                    //! Shift the components' time origin to \p time.
                    void shiftOrigin(core_t::TTime time);

                    //! Get a checksum for this object.
                    uint64_t checksum(uint64_t seed = 0) const;

                    //! Debug the memory used by this object.
                    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                    //! Get the memory used by this object.
                    std::size_t memoryUsage(void) const;

                    //! The seasonal components.
                    maths_t::TSeasonalComponentVec s_Components;

                    //! The prediction errors relating to the component.
                    TComponentErrorsVec s_PredictionErrors;
                };

                using TSeasonalPtr = boost::shared_ptr<SSeasonal>;

                //! \brief Calendar periodic components of the decomposition.
                struct MATHS_EXPORT SCalendar
                {
                    //! Initialize by reading state from \p traverser.
                    bool acceptRestoreTraverser(double decayRate,
                                                core_t::TTime bucketLength,
                                                core::CStateRestoreTraverser &traverser);

                    //! Persist state by passing information to \p inserter.
                    void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                    //! Set the decay rate.
                    void decayRate(double decayRate);

                    //! Age the calendar components to account for the interval \p end
                    //! - \p start elapsed time.
                    void propagateForwards(core_t::TTime start, core_t::TTime end);

                    //! Get the combined size of the seasonal components.
                    std::size_t size(void) const;

                    //! Check if there is already a component for \p feature.
                    bool haveComponent(CCalendarFeature feature) const;

                    //! Get the state to update.
                    void componentsAndErrors(core_t::TTime time,
                                             TCalendarComponentPtrVec &components,
                                             TComponentErrorsPtrVec &errors);

                    //! Check if we need to interpolate any of the components.
                    bool shouldInterpolate(core_t::TTime time, core_t::TTime last) const;

                    //! Interpolate the components at \p time.
                    void interpolate(core_t::TTime time, core_t::TTime last, bool refine);

                    //! Check if any of the components has been initialized.
                    bool initialized(void) const;

                    //! Remove low value components.
                    bool prune(core_t::TTime time, core_t::TTime bucketLength);

                    //! Get a checksum for this object.
                    uint64_t checksum(uint64_t seed = 0) const;

                    //! Debug the memory used by this object.
                    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                    //! Get the memory used by this object.
                    std::size_t memoryUsage(void) const;

                    //! The calendar components.
                    maths_t::TCalendarComponentVec s_Components;

                    //! The prediction errors after removing the component.
                    TComponentErrorsVec s_PredictionErrors;
                };

                using TCalendarPtr = boost::shared_ptr<SCalendar>;

            private:
                //! Get the total size of the components.
                std::size_t size(void) const;

                //! Get the maximum permitted size of the components.
                std::size_t maxSize(void) const;

                //! Add new seasonal components to \p components.
                void addSeasonalComponents(const CPeriodicityTest &test,
                                           const CPeriodicityTestResult &result,
                                           core_t::TTime time,
                                           maths_t::TSeasonalComponentVec &components,
                                           TComponentErrorsVec &errors) const;

                //! Add a new calendar component to \p components.
                void addCalendarComponent(const CCalendarFeature &feature,
                                          core_t::TTime time,
                                          maths_t::TCalendarComponentVec &components,
                                          TComponentErrorsVec &errors) const;

                //! Clear all component error statistics.
                void clearComponentErrors(void);

                //! Handle \p symbol.
                void apply(std::size_t symbol, const SMessage &message);

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
                void notifyOnNewComponents(bool *watcher);

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
                TTrendPtr m_Trend;

                //! The seasonal components.
                TSeasonalPtr m_Seasonal;

                //! The calendar components.
                TCalendarPtr m_Calendar;

                //! Set to true if non-null when the seasonal components change.
                bool *m_Watcher;
        };
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CLongTermTrendTest &lhs,
                 CTimeSeriesDecompositionDetail::CLongTermTrendTest &rhs)
{
    lhs.swap(rhs);
}

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CDiurnalTest &lhs,
                 CTimeSeriesDecompositionDetail::CDiurnalTest &rhs)
{
    lhs.swap(rhs);
}

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CNonDiurnalTest &lhs,
                 CTimeSeriesDecompositionDetail::CNonDiurnalTest &rhs)
{
    lhs.swap(rhs);
}

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CCalendarTest &lhs,
                 CTimeSeriesDecompositionDetail::CCalendarTest &rhs)
{
    lhs.swap(rhs);
}

//! Create a free function which will be found by Koenig lookup.
inline void swap(CTimeSeriesDecompositionDetail::CComponents &lhs,
                 CTimeSeriesDecompositionDetail::CComponents &rhs)
{
    lhs.swap(rhs);
}

}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionDetail_h
