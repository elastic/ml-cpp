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

#ifndef INCLUDED_ml_api_CForecastRunner_h
#define INCLUDED_ml_api_CForecastRunner_h

#include <core/CConcurrentWrapper.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/common/CModel.h>

#include <model/CAnomalyDetector.h>
#include <model/CForecastDataSink.h>
#include <model/CResourceMonitor.h>

#include <api/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace CForecastRunnerTest {
struct testPopulation;
struct testRare;
struct testInsufficientData;
struct testValidateDefaultExpiry;
struct testValidateNoExpiry;
struct testValidateInvalidExpiry;
struct testValidateBrokenMessage;
struct testValidateMissingId;
struct testValidateProvidedMinDiskSpace;
struct testValidateProvidedMaxMemoryLimit;
struct testValidateProvidedTooLargeMaxMemoryLimit;
struct testSufficientDiskSpace;
}

namespace ml {
namespace api {

//! \brief
//! Forecast Worker to create forecasts of timeseries/ml models.
//!
//! DESCRIPTION:\n
//! Executes forecast jobs async to the main thread
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses only 1 thread as worker.
//!
//! The forecast runs in parallel to the main thread, this has
//! various consequences:
//!
//! (math) models are cloned for forecasting at the time the request
//! is made, as models would otherwise continue changing in the main
//! thread.
//! For the same reason, any field values are copied as they might get
//! pruned in the main thread.
//! Cloning also happens beforehand as the forecast job might hang in
//! the queue for a while
class API_EXPORT CForecastRunner final : private core::CNonCopyable {
public:
    //! max open forecast requests
    //! if you change this, also change the ERROR_TOO_MANY_JOBS message accordingly
    static const std::size_t MAX_FORECAST_JOBS_IN_QUEUE = 3;

    //! default expiry time
    static const std::size_t DEFAULT_EXPIRY_TIME = 14 * core::constants::DAY;

    //! max memory allowed to use for forecast models
    //! (not defined inline because we need its address)
    static const std::size_t DEFAULT_MAX_FORECAST_MODEL_MEMORY;

    //! Note: This value measures the size in memory, not the size of the persistence,
    //! which is likely higher and would be hard to calculate upfront
    //! max memory allowed to use for forecast models persisting to disk
    static const std::size_t MAX_FORECAST_MODEL_PERSISTANCE_MEMORY = 524288000ull; // 500MB

    //! Note: This value is lower than in the ML Java code to prevent side-effects.
    //! If you change this value also change the limit in the ML Java code.
    //! The purpose of this value is to guard the rest of the system against
    //! running out of disk space.
    //! minimum disk space required for disk persistence
    //! (not defined inline because we need its address)
    static const std::size_t DEFAULT_MIN_FORECAST_AVAILABLE_DISK_SPACE;

    //! minimum time between stat updates to prevent to many updates in a short time
    static const std::uint64_t MINIMUM_TIME_ELAPSED_FOR_STATS_UPDATE = 3000ul; // 3s

private:
    static const std::string ERROR_FORECAST_REQUEST_FAILED_TO_PARSE;
    static const std::string ERROR_NO_FORECAST_ID;
    static const std::string ERROR_TOO_MANY_JOBS;
    static const std::string ERROR_NO_MODELS;
    static const std::string ERROR_NO_DATA_PROCESSED;
    static const std::string ERROR_NO_CREATE_TIME;
    static const std::string ERROR_BAD_MEMORY_STATUS;
    static const std::string ERROR_BAD_MODEL_MEMORY_LIMIT;
    static const std::string ERROR_MEMORY_LIMIT_DISK;
    static const std::string ERROR_MEMORY_LIMIT_DISKSPACE;
    static const std::string ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS;
    static const std::string ERROR_NO_SUPPORTED_FUNCTIONS;
    static const std::string WARNING_INVALID_EXPIRY;
    static const std::string INFO_DEFAULT_DURATION;
    static const std::string INFO_DEFAULT_EXPIRY;
    static const std::string INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST;

public:
    using TOStreamConcurrentWrapper = core::CConcurrentWrapper<std::ostream>;
    using TOStreamConcurrentWrapperPtr = std::shared_ptr<TOStreamConcurrentWrapper>;

    using TAnomalyDetectorPtr = std::shared_ptr<model::CAnomalyDetector>;
    using TAnomalyDetectorPtrVec = std::vector<TAnomalyDetectorPtr>;

    using TForecastModelWrapper = model::CForecastDataSink::CForecastModelWrapper;
    using TForecastResultSeries = model::CForecastDataSink::SForecastResultSeries;
    using TForecastResultSeriesVec = std::vector<TForecastResultSeries>;
    using TMathsModelPtr = std::unique_ptr<maths::common::CModel>;

    using TStrUSet = boost::unordered_set<std::string>;

public:
    //! Initialize and start the forecast runner thread
    //! \p jobId The job ID
    //! \p strmOut The output stream to write forecast results to
    CForecastRunner(const std::string& jobId,
                    core::CJsonOutputStreamWrapper& strmOut,
                    model::CResourceMonitor& resourceMonitor);

    //! Destructor, cancels all queued forecast requests, finishes a running forecast.
    //! To finish all remaining forecasts call finishForecasts() first.
    ~CForecastRunner();

    //! Enqueue a forecast job that will execute the requested forecast
    //!
    //! Parses and verifies the controlMessage and creates an internal job object which
    //! contains the required detectors (reference) as well as start and end date.
    //! The forecast itself isn't executed but might start later depending on the workers
    //! load.
    //!
    //! Validation fails if the message is invalid and/or the too many jobs are in the
    //! queue.
    //!
    //! \param controlMessage The control message retrieved.
    //! \param detectors vector of detectors (shallow copy)
    //! \return true if the forecast request passed validation
    bool pushForecastJob(const std::string& controlMessage,
                         const TAnomalyDetectorPtrVec& detectors,
                         const core_t::TTime lastResultsTime);

    //! Blocks and waits until all queued forecasts are done
    void finishForecasts();

    //! Deletes all pending forecast requests
    void deleteAllForecastJobs();

private:
    struct API_EXPORT SForecast {
        SForecast() = default;

        SForecast(SForecast&& other) = default;
        SForecast& operator=(SForecast&& other) = default;

        SForecast(const SForecast& that) = delete;
        SForecast& operator=(const SForecast&) = delete;

        //! reset the struct, important to e.g. clean up reference counts
        void reset();

        //! get the the end time
        core_t::TTime forecastEnd() const;

        //! The forecast ID
        std::string s_ForecastId;

        //! The forecast alias
        std::string s_ForecastAlias;

        //! Vector of models/series selected for forecasting (cloned for forecasting)
        TForecastResultSeriesVec s_ForecastSeries;

        //! Forecast create time
        core_t::TTime s_CreateTime{0};

        //! Forecast start time
        core_t::TTime s_StartTime{0};

        //! Forecast duration
        core_t::TTime s_Duration{0};

        //! Expiration of the forecast (for automatic deletion)
        core_t::TTime s_ExpiryTime{0};

        //! Forecast bounds
        double s_BoundsPercentile{maths::common::CModel::DEFAULT_BOUNDS_PERCENTILE};

        //! total number of models
        std::size_t s_NumberOfModels{0};

        //! total number of models able to forecast
        std::size_t s_NumberOfForecastableModels{0};

        //! total memory required for this forecasting job (only the models)
        std::size_t s_MemoryUsage{0};

        //! maximum allowed memory (in bytes) that this forecast can use
        std::size_t s_MaxForecastModelMemory{DEFAULT_MAX_FORECAST_MODEL_MEMORY};

        //! minimum free disk space (in bytes) for a forecast to use disk
        std::size_t s_MinForecastAvailableDiskSpace{DEFAULT_MIN_FORECAST_AVAILABLE_DISK_SPACE};

        //! A collection storing important messages from forecasting
        TStrUSet s_Messages;

        //! A directory to persist models on disk
        std::string s_TemporaryFolder;
    };

private:
    using TErrorFunc =
        std::function<void(const SForecast& forecastJob, const std::string& message)>;

private:
    //! The worker loop
    void forecastWorker();

    //! Check for new jobs, blocks while waiting
    bool tryGetJob(SForecast& forecastJob);

    //! pushes new jobs into the internal 'queue' (thread boundary)
    bool push(SForecast& forecastJob);

    //! send a scheduled message
    void sendScheduledMessage(const SForecast& forecastJob) const;

    //! send an error message
    void sendErrorMessage(const SForecast& forecastJob, const std::string& message) const;

    //! send a final message
    void sendFinalMessage(const SForecast& forecastJob, const std::string& message) const;

    //! send a message using \p write
    template<typename WRITE>
    void sendMessage(WRITE write, const SForecast& forecastJob, const std::string& message) const;

    //! parse and validate a forecast request and turn it into a forecast job
    static bool parseAndValidateForecastRequest(
        const std::string& controlMessage,
        SForecast& forecastJob,
        const core_t::TTime lastResultsTime,
        std::size_t jobBytesSizeLimit = std::numeric_limits<std::size_t>::max() / 2,
        const TErrorFunc& errorFunction = TErrorFunc());

private:
    //! This job ID
    std::string m_JobId;

    //! the output stream to write results to
    core::CJsonOutputStreamWrapper& m_ConcurrentOutputStream;

    //! The resource monitor by reference (owned by CAnomalyJob)
    //! note: we use the resource monitor only for checks at the moment
    model::CResourceMonitor& m_ResourceMonitor;

    //! thread for the worker
    std::thread m_Worker;

    //! indicator for worker
    std::atomic_bool m_Shutdown;

    //! The 'queue' of forecast jobs to be executed
    std::list<SForecast> m_ForecastJobs;

    //! Mutex
    std::mutex m_Mutex;

    //! Condition variable for the requests queue
    std::condition_variable m_WorkAvailableCondition;

    //! Condition variable for notifications on done requests
    std::condition_variable m_WorkCompleteCondition;

    friend struct CForecastRunnerTest::testPopulation;
    friend struct CForecastRunnerTest::testRare;
    friend struct CForecastRunnerTest::testInsufficientData;
    friend struct CForecastRunnerTest::testValidateDefaultExpiry;
    friend struct CForecastRunnerTest::testValidateNoExpiry;
    friend struct CForecastRunnerTest::testValidateInvalidExpiry;
    friend struct CForecastRunnerTest::testValidateBrokenMessage;
    friend struct CForecastRunnerTest::testValidateMissingId;
    friend struct CForecastRunnerTest::testValidateProvidedMinDiskSpace;
    friend struct CForecastRunnerTest::testValidateProvidedMaxMemoryLimit;
    friend struct CForecastRunnerTest::testValidateProvidedTooLargeMaxMemoryLimit;
    friend struct CForecastRunnerTest::testSufficientDiskSpace;
};
}
}

#endif // INCLUDED_ml_api_CForecastRunner_h
