/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CForecastRunner_h
#define INCLUDED_ml_api_CForecastRunner_h

#include <api/ImportExport.h>

#include <core/CConcurrentWrapper.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CModel.h>

#include <model/CAnomalyDetector.h>
#include <model/CForecastDataSink.h>
#include <model/CResourceMonitor.h>

#include <boost/filesystem.hpp>
#include <boost/unordered_set.hpp>

#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <stdint.h>

class CForecastRunnerTest;

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
    static const size_t MAX_FORECAST_JOBS_IN_QUEUE = 3;

    //! default expiry time
    static const size_t DEFAULT_EXPIRY_TIME = 14 * core::constants::DAY;

    //! max memory allowed to use for forecast models
    static const size_t MAX_FORECAST_MODEL_MEMORY = 20971520ull; // 20MB

    //! Note: This value measures the size in memory, not the size of the persistence,
    //! which is likely higher and would be hard to calculate upfront
    //! max memory allowed to use for forecast models persisting to disk
    static const size_t MAX_FORECAST_MODEL_PERSISTANCE_MEMORY = 524288000ull; // 500MB

    //! Note: This value is lower than in the ML Java code to prevent side-effects.
    //! If you change this value also change the limit in the ML Java code.
    //! The purpose of this value is to guard the rest of the system against
    //! running out of disk space.
    //! minimum disk space required for disk persistence
    static const size_t MIN_FORECAST_AVAILABLE_DISK_SPACE = 4294967296ull; // 4GB

    //! minimum time between stat updates to prevent to many updates in a short time
    static const uint64_t MINIMUM_TIME_ELAPSED_FOR_STATS_UPDATE = 3000ul; // 3s

private:
    static const std::string ERROR_FORECAST_REQUEST_FAILED_TO_PARSE;
    static const std::string ERROR_NO_FORECAST_ID;
    static const std::string ERROR_TOO_MANY_JOBS;
    static const std::string ERROR_NO_MODELS;
    static const std::string ERROR_NO_DATA_PROCESSED;
    static const std::string ERROR_NO_CREATE_TIME;
    static const std::string ERROR_BAD_MEMORY_STATUS;
    static const std::string ERROR_MEMORY_LIMIT;
    static const std::string ERROR_MEMORY_LIMIT_DISK;
    static const std::string ERROR_MEMORY_LIMIT_DISKSPACE;
    static const std::string ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS;
    static const std::string ERROR_NO_SUPPORTED_FUNCTIONS;
    static const std::string WARNING_DURATION_LIMIT;
    static const std::string WARNING_INVALID_EXPIRY;
    static const std::string INFO_DEFAULT_DURATION;
    static const std::string INFO_DEFAULT_EXPIRY;
    static const std::string INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST;

public:
    using TOStreamConcurrentWrapper = core::CConcurrentWrapper<std::ostream>;
    using TOStreamConcurrentWrapperPtr = std::shared_ptr<TOStreamConcurrentWrapper>;

    using TAnomalyDetectorPtr = std::shared_ptr<model::CAnomalyDetector>;
    using TAnomalyDetectorPtrVec = std::vector<TAnomalyDetectorPtr>;

    using TForecastModelWrapper = model::CForecastDataSink::SForecastModelWrapper;
    using TForecastResultSeries = model::CForecastDataSink::SForecastResultSeries;
    using TForecastResultSeriesVec = std::vector<TForecastResultSeries>;
    using TMathsModelPtr = std::unique_ptr<maths::CModel>;

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
        SForecast();

        SForecast(SForecast&& other);
        SForecast& operator=(SForecast&& other);

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
        core_t::TTime s_CreateTime;

        //! Forecast start time
        core_t::TTime s_StartTime;

        //! Forecast duration
        core_t::TTime s_Duration;

        //! Expiration of the forecast (for automatic deletion)
        core_t::TTime s_ExpiryTime;

        //! Forecast bounds
        double s_BoundsPercentile;

        //! total number of models
        size_t s_NumberOfModels;

        //! total number of models able to forecast
        size_t s_NumberOfForecastableModels;

        //! total memory required for this forecasting job (only the models)
        size_t s_MemoryUsage;

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

    //! check for sufficient disk space
    bool sufficientAvailableDiskSpace(const boost::filesystem::path& path);

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
    static bool
    parseAndValidateForecastRequest(const std::string& controlMessage,
                                    SForecast& forecastJob,
                                    const core_t::TTime lastResultsTime,
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
    volatile bool m_Shutdown;

    //! The 'queue' of forecast jobs to be executed
    std::list<SForecast> m_ForecastJobs;

    //! Mutex
    std::mutex m_Mutex;

    //! Condition variable for the requests queue
    std::condition_variable m_WorkAvailableCondition;

    //! Condition variable for notifications on done requests
    std::condition_variable m_WorkCompleteCondition;

    friend class ::CForecastRunnerTest;
};
}
}

#endif // INCLUDED_ml_api_CForecastRunner_h
