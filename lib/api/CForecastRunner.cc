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

#include <api/CForecastRunner.h>

#include <core/CLogger.h>
#include <core/CStopWatch.h>
#include <core/CTimeUtils.h>

#include <model/CForecastDataSink.h>
#include <model/CForecastModelPersist.h>
#include <model/ModelTypes.h>

#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/system/error_code.hpp>

#include <sstream>

namespace ml {
namespace api {

namespace {
bool sufficientAvailableDiskSpaceForPath(std::size_t minForecastAvailableDiskSpace,
                                         const boost::filesystem::path& path) {
    boost::system::error_code errorCode;
    auto spaceInfo = boost::filesystem::space(path, errorCode);

    if (errorCode) {
        LOG_ERROR(<< "Failed to retrieve disk information for " << path
                  << " error " << errorCode.message());
        return false;
    }

    if (spaceInfo.available < minForecastAvailableDiskSpace) {
        LOG_WARN(<< "Checked disk space for " << path << " - required: " << minForecastAvailableDiskSpace
                 << ", available: " << spaceInfo.available);
        return false;
    }

    return true;
}

const std::string EMPTY_STRING;
}

const std::size_t CForecastRunner::DEFAULT_MAX_FORECAST_MODEL_MEMORY{20971520}; // 20MB
const std::size_t CForecastRunner::DEFAULT_MIN_FORECAST_AVAILABLE_DISK_SPACE{4294967296ull}; // 4GB

const std::string CForecastRunner::ERROR_FORECAST_REQUEST_FAILED_TO_PARSE("Failed to parse forecast request: ");
const std::string CForecastRunner::ERROR_NO_FORECAST_ID("forecast ID must be specified and non empty");
const std::string CForecastRunner::ERROR_TOO_MANY_JOBS("Forecast cannot be executed due to queue limit. Please wait for requests to finish and try again");
const std::string CForecastRunner::ERROR_NO_MODELS("Forecast cannot be executed as model is not yet established. Job requires more time to learn");
const std::string CForecastRunner::ERROR_NO_DATA_PROCESSED(
    "Forecast cannot be executed as job requires data to have been processed and modeled");
const std::string CForecastRunner::ERROR_NO_CREATE_TIME("Forecast create time must be specified and non zero");
const std::string CForecastRunner::ERROR_BAD_MEMORY_STATUS("Forecast cannot be executed as model memory status is not OK");
const std::string CForecastRunner::ERROR_BAD_MODEL_MEMORY_LIMIT(
    "Forecast max_model_memory must be below 500MB and must not exceed 40% of the job's configured model memory limit.");
const std::string CForecastRunner::ERROR_MEMORY_LIMIT_DISK(
    "Forecast cannot be executed as forecast memory usage is predicted to exceed 500MB");
const std::string CForecastRunner::ERROR_MEMORY_LIMIT_DISKSPACE(
    "Forecast cannot be executed as models exceed internal memory limit and available disk space is insufficient");
const std::string CForecastRunner::ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS("Forecast is not supported for population analysis");
const std::string CForecastRunner::ERROR_NO_SUPPORTED_FUNCTIONS("Forecast is not supported for the used functions");
const std::string CForecastRunner::WARNING_INVALID_EXPIRY("Forecast expires_in invalid, setting to 14 days");
const std::string CForecastRunner::INFO_DEFAULT_DURATION("Forecast duration not specified, setting to 1 day");
const std::string CForecastRunner::INFO_DEFAULT_EXPIRY("Forecast expires_in not specified, setting to 14 days");
const std::string CForecastRunner::INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST("Insufficient history to forecast for all models");

CForecastRunner::CForecastRunner(const std::string& jobId,
                                 core::CJsonOutputStreamWrapper& strmOut,
                                 model::CResourceMonitor& resourceMonitor)
    : m_JobId{jobId}, m_ConcurrentOutputStream{strmOut},
      m_ResourceMonitor{resourceMonitor}, m_Shutdown{false} {
    m_Worker = std::thread([this] { this->forecastWorker(); });
}

CForecastRunner::~CForecastRunner() {
    // shutdown
    m_Shutdown.store(true);
    // signal the worker
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_WorkAvailableCondition.notify_all();
    }
    m_Worker.join();
}

void CForecastRunner::finishForecasts() {
    std::unique_lock<std::mutex> lock(m_Mutex);
    // note: forecast could still be active
    while (m_Shutdown.load() == false && m_ForecastJobs.empty() == false) {
        // items in the queue, wait
        m_WorkCompleteCondition.wait(lock);
    }
}

void CForecastRunner::forecastWorker() {
    SForecast forecastJob;
    while (m_Shutdown.load() == false) {
        if (this->tryGetJob(forecastJob)) {
            LOG_INFO(<< "Start forecasting from "
                     << core::CTimeUtils::toIso8601(forecastJob.s_StartTime) << " to "
                     << core::CTimeUtils::toIso8601(forecastJob.forecastEnd()));

            core::CStopWatch timer(true);
            std::uint64_t lastStatsUpdate = 0;

            LOG_TRACE(<< "about to create sink");
            model::CForecastDataSink sink(
                m_JobId, forecastJob.s_ForecastId, forecastJob.s_ForecastAlias,
                forecastJob.s_CreateTime, forecastJob.s_StartTime,
                forecastJob.forecastEnd(), forecastJob.s_ExpiryTime,
                forecastJob.s_MemoryUsage, m_ConcurrentOutputStream);

            std::string message;

            // collecting the runtime messages first and sending it in 1 go
            TStrUSet messages(forecastJob.s_Messages);
            double processedModels = 0;
            double totalNumberOfForecastableModels =
                static_cast<double>(forecastJob.s_NumberOfForecastableModels);
            std::size_t failedForecasts = 0;
            sink.writeStats(0.0, 0, forecastJob.s_Messages);

            // while loops allow us to free up memory for every model right after each forecast is done
            while (!forecastJob.s_ForecastSeries.empty()) {
                TForecastResultSeries& series = forecastJob.s_ForecastSeries.back();
                std::unique_ptr<model::CForecastModelPersist::CRestore> modelRestore;

                // initialize persistence restore exactly once
                if (!series.s_ToForecastPersisted.empty()) {
                    modelRestore = std::make_unique<model::CForecastModelPersist::CRestore>(
                        series.s_ModelParams, series.s_MinimumSeasonalVarianceScale,
                        series.s_ToForecastPersisted);
                }

                while (series.s_ToForecast.empty() == false || modelRestore != nullptr) {
                    // check if we should backfill from persistence
                    if (series.s_ToForecast.empty()) {
                        TMathsModelPtr model;
                        core_t::TTime firstDataTime;
                        core_t::TTime lastDataTime;
                        model_t::EFeature feature;
                        std::string byFieldValue;

                        if (modelRestore->nextModel(model, firstDataTime, lastDataTime,
                                                    feature, byFieldValue)) {
                            series.s_ToForecast.emplace_back(
                                feature, byFieldValue, std::move(model),
                                firstDataTime, lastDataTime);
                        } else {
                            // restorer exhausted, no need for further restoring
                            modelRestore.reset();
                            break;
                        }
                    }

                    const TForecastModelWrapper& model{series.s_ToForecast.back()};
                    bool success{model.forecast(
                        series, forecastJob.s_StartTime, forecastJob.forecastEnd(),
                        forecastJob.s_BoundsPercentile, sink, message)};
                    series.s_ToForecast.pop_back();

                    if (success == false) {
                        LOG_DEBUG(<< "Detector " << series.s_DetectorIndex
                                  << " failed to forecast");
                        ++failedForecasts;
                    }

                    if (message.empty() == false) {
                        messages.insert("Detector[" + std::to_string(series.s_DetectorIndex) +
                                        "]: " + message);
                        message.clear();
                    }

                    ++processedModels;

                    if (processedModels != totalNumberOfForecastableModels) {
                        std::uint64_t elapsedTime = timer.lap();
                        if (elapsedTime - lastStatsUpdate > MINIMUM_TIME_ELAPSED_FOR_STATS_UPDATE) {
                            sink.writeStats(processedModels / totalNumberOfForecastableModels,
                                            elapsedTime, forecastJob.s_Messages);
                            lastStatsUpdate = elapsedTime;
                        }
                    }
                }
                forecastJob.s_ForecastSeries.pop_back();
            }
            // write final message
            sink.writeStats(1.0, timer.stop(), messages,
                            failedForecasts != forecastJob.s_NumberOfForecastableModels);

            // important: reset the structure to decrease shared pointer reference counts
            forecastJob.reset();
            LOG_INFO(<< "Finished forecasting, wrote "
                     << sink.numRecordsWritten() << " records");

            // signal that job is done
            m_WorkCompleteCondition.notify_all();

            // cleanup
            if (!forecastJob.s_TemporaryFolder.empty()) {
                boost::filesystem::path temporaryFolder(forecastJob.s_TemporaryFolder);
                boost::system::error_code errorCode;
                boost::filesystem::remove_all(temporaryFolder, errorCode);
                if (errorCode) {
                    // not an error: there is also cleanup code on the Java side
                    LOG_WARN(<< "Failed to cleanup temporary data from: "
                             << forecastJob.s_TemporaryFolder << " error "
                             << errorCode.message());
                }
            }
        }
    }

    // clear any queued forecast jobs (paranoia, this should not happen)
    this->deleteAllForecastJobs();
}

void CForecastRunner::deleteAllForecastJobs() {
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_ForecastJobs.clear();
    m_WorkAvailableCondition.notify_all();
}

bool CForecastRunner::tryGetJob(SForecast& forecastJob) {
    std::unique_lock<std::mutex> lock(m_Mutex);

    if (!m_ForecastJobs.empty()) {
        std::swap(forecastJob, m_ForecastJobs.front());
        m_ForecastJobs.pop_front();
        return true;
    }

    // m_Shutdown might have been set meanwhile
    if (m_Shutdown.load()) {
        return false;
    }

    m_WorkAvailableCondition.wait(lock);
    return false;
}

bool CForecastRunner::pushForecastJob(const std::string& controlMessage,
                                      const TAnomalyDetectorPtrVec& detectors,
                                      const core_t::TTime lastResultsTime) {
    SForecast forecastJob;
    if (parseAndValidateForecastRequest(
            controlMessage, forecastJob, lastResultsTime,
            m_ResourceMonitor.getBytesMemoryLimit(),
            std::bind(&CForecastRunner::sendErrorMessage, this,
                      std::placeholders::_1, std::placeholders::_2)) == false) {
        return false;
    }

    if (m_ResourceMonitor.memoryStatus() != model_t::E_MemoryStatusOk) {
        this->sendErrorMessage(forecastJob, ERROR_BAD_MEMORY_STATUS);
        return false;
    }

    std::size_t totalNumberOfModels = 0;
    std::size_t totalNumberOfForecastModels = 0;
    bool atLeastOneNonPopulationModel = false;
    bool atLeastOneSupportedFunction = false;
    std::size_t totalMemoryUsage = 0;

    // 1st loop over the detectors to check prerequisites
    for (const auto& detector : detectors) {
        if (detector.get() == nullptr) {
            LOG_ERROR(<< "Unexpected empty detector found");
            continue;
        }

        model::CForecastDataSink::SForecastModelPrerequisites prerequisites =
            detector->getForecastPrerequisites();

        totalNumberOfModels += prerequisites.s_NumberOfModels;
        totalNumberOfForecastModels += prerequisites.s_NumberOfForecastableModels;
        atLeastOneNonPopulationModel = atLeastOneNonPopulationModel ||
                                       !prerequisites.s_IsPopulation;
        atLeastOneSupportedFunction = atLeastOneSupportedFunction ||
                                      prerequisites.s_IsSupportedFunction;
        totalMemoryUsage += prerequisites.s_MemoryUsageForDetector;

        if (totalMemoryUsage >= forecastJob.s_MaxForecastModelMemory &&
            forecastJob.s_TemporaryFolder.empty()) {
            this->sendErrorMessage(
                forecastJob, "Forecast cannot be executed as forecast memory usage is predicted to exceed " +
                                 std::to_string(forecastJob.s_MaxForecastModelMemory) +
                                 " bytes while disk space is exceeded");
            return false;
        }
    }

    if (totalMemoryUsage >= MAX_FORECAST_MODEL_PERSISTANCE_MEMORY) {
        this->sendErrorMessage(forecastJob, ERROR_MEMORY_LIMIT_DISK);
        return false;
    }

    if (atLeastOneNonPopulationModel == false) {
        this->sendErrorMessage(forecastJob, ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS);
        return false;
    }

    if (atLeastOneSupportedFunction == false) {
        this->sendErrorMessage(forecastJob, ERROR_NO_SUPPORTED_FUNCTIONS);
        return false;
    }

    if (totalNumberOfForecastModels == 0) {
        this->sendFinalMessage(forecastJob, INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST);
        return false;
    }

    forecastJob.s_NumberOfModels = totalNumberOfModels;
    forecastJob.s_NumberOfForecastableModels = totalNumberOfForecastModels;
    forecastJob.s_MemoryUsage = totalMemoryUsage;

    // send a notification that job has been scheduled
    this->sendScheduledMessage(forecastJob);

    // 2nd loop over the detectors to clone models for forecasting
    bool persistOnDisk = false;
    if (totalMemoryUsage >= forecastJob.s_MaxForecastModelMemory) {
        boost::filesystem::path temporaryFolder(forecastJob.s_TemporaryFolder);

        if (sufficientAvailableDiskSpaceForPath(forecastJob.s_MinForecastAvailableDiskSpace,
                                                temporaryFolder) == false) {
            this->sendErrorMessage(forecastJob, ERROR_MEMORY_LIMIT_DISKSPACE);
            return false;
        }

        LOG_WARN(<< "Forecast [" << forecastJob.s_ForecastId << "] memory usage exceeds configured byte limit ["
                 << std::to_string(forecastJob.s_MaxForecastModelMemory) << "] (requires "
                 << std::to_string(1 + (totalMemoryUsage >> 20)) << " MB), using disk.");

        // create a subdirectory using the unique forecast id
        temporaryFolder /= forecastJob.s_ForecastId;
        forecastJob.s_TemporaryFolder = temporaryFolder.string();

        boost::system::error_code errorCode;
        boost::filesystem::create_directories(temporaryFolder, errorCode);
        if (errorCode) {
            this->sendErrorMessage(
                forecastJob,
                "Forecast internal error, failed to create temporary folder " +
                    temporaryFolder.string() + " error: " + errorCode.message());
            return false;
        }

        LOG_DEBUG(<< "Persisting to: " << temporaryFolder.string());
        persistOnDisk = true;
    } else {
        forecastJob.s_TemporaryFolder.clear();
    }

    for (const auto& detector : detectors) {
        if (detector.get() == nullptr) {
            LOG_ERROR(<< "Unexpected empty detector found");
            continue;
        }

        forecastJob.s_ForecastSeries.emplace_back(detector->getForecastModels(
            persistOnDisk, forecastJob.s_TemporaryFolder));
    }

    return this->push(forecastJob);
}

bool CForecastRunner::push(SForecast& forecastJob) {
    std::unique_lock<std::mutex> lock(m_Mutex);

    if (m_ForecastJobs.size() == MAX_FORECAST_JOBS_IN_QUEUE) {
        this->sendErrorMessage(forecastJob, ERROR_TOO_MANY_JOBS);
        return false;
    }

    if (forecastJob.s_NumberOfModels == 0) {
        this->sendErrorMessage(forecastJob, ERROR_NO_MODELS);
        return false;
    }

    m_ForecastJobs.push_back(std::move(forecastJob));

    lock.unlock();
    m_WorkAvailableCondition.notify_all();
    return true;
}

bool CForecastRunner::parseAndValidateForecastRequest(const std::string& controlMessage,
                                                      SForecast& forecastJob,
                                                      const core_t::TTime lastResultsTime,
                                                      std::size_t jobBytesSizeLimit,
                                                      const TErrorFunc& errorFunction) {
    std::istringstream stringStream(controlMessage.substr(1));
    forecastJob.s_StartTime = lastResultsTime;

    core_t::TTime expiresIn = 0;
    boost::property_tree::ptree properties;
    try {
        boost::property_tree::read_json(stringStream, properties);

        forecastJob.s_ForecastId = properties.get<std::string>("forecast_id", EMPTY_STRING);
        forecastJob.s_ForecastAlias =
            properties.get<std::string>("forecast_alias", EMPTY_STRING);
        forecastJob.s_Duration = properties.get<core_t::TTime>("duration", 0);
        forecastJob.s_CreateTime = properties.get<core_t::TTime>("create_time", 0);
        forecastJob.s_MaxForecastModelMemory = properties.get<std::size_t>(
            "max_model_memory", DEFAULT_MAX_FORECAST_MODEL_MEMORY);
        forecastJob.s_MinForecastAvailableDiskSpace = properties.get<std::size_t>(
            "min_available_disk_space", DEFAULT_MIN_FORECAST_AVAILABLE_DISK_SPACE);

        // tmp storage if available
        forecastJob.s_TemporaryFolder = properties.get<std::string>("tmp_storage", EMPTY_STRING);
        // use -1 as default to allow 0 as 'never expires'
        expiresIn = properties.get<core_t::TTime>("expires_in", -1l);

        // note: this is not exposed on the Java side
        forecastJob.s_BoundsPercentile = properties.get<double>(
            "boundspercentile", maths::common::CModel::DEFAULT_BOUNDS_PERCENTILE);
    } catch (const std::exception& e) {
        LOG_ERROR(<< ERROR_FORECAST_REQUEST_FAILED_TO_PARSE << e.what());
        return false;
    }

    if (forecastJob.s_ForecastId.empty()) {
        LOG_ERROR(<< ERROR_NO_FORECAST_ID);
        return false;
    }

    // from now we have a forecast ID and can send error messages
    if (forecastJob.s_MaxForecastModelMemory != DEFAULT_MAX_FORECAST_MODEL_MEMORY &&
        (forecastJob.s_MaxForecastModelMemory >= MAX_FORECAST_MODEL_PERSISTANCE_MEMORY ||
         forecastJob.s_MaxForecastModelMemory >=
             static_cast<std::size_t>(jobBytesSizeLimit * 0.40))) {
        errorFunction(forecastJob, ERROR_BAD_MODEL_MEMORY_LIMIT);
        return false;
    }

    if (lastResultsTime == 0) {
        errorFunction(forecastJob, ERROR_NO_DATA_PROCESSED);
        return false;
    }

    if (forecastJob.s_CreateTime == 0) {
        errorFunction(forecastJob, ERROR_NO_CREATE_TIME);
        return false;
    }

    if (forecastJob.s_Duration == 0) {
        // only log
        forecastJob.s_Duration = core::constants::DAY;
        LOG_INFO(<< INFO_DEFAULT_DURATION);
    }

    if (expiresIn < -1) {
        // only log
        expiresIn = DEFAULT_EXPIRY_TIME;
        LOG_INFO(<< WARNING_INVALID_EXPIRY);
    } else if (expiresIn == -1) {
        // only log
        expiresIn = DEFAULT_EXPIRY_TIME;
        LOG_DEBUG(<< INFO_DEFAULT_EXPIRY);
    }

    forecastJob.s_ExpiryTime = forecastJob.s_CreateTime + expiresIn;

    return true;
}

void CForecastRunner::sendScheduledMessage(const SForecast& forecastJob) const {
    LOG_DEBUG(<< "job passed forecast validation, scheduled for forecasting");
    model::CForecastDataSink sink(
        m_JobId, forecastJob.s_ForecastId, forecastJob.s_ForecastAlias,
        forecastJob.s_CreateTime, forecastJob.s_StartTime, forecastJob.forecastEnd(),
        forecastJob.s_ExpiryTime, forecastJob.s_MemoryUsage, m_ConcurrentOutputStream);
    sink.writeScheduledMessage();
}

void CForecastRunner::sendErrorMessage(const SForecast& forecastJob,
                                       const std::string& message) const {
    LOG_ERROR(<< message);
    this->sendMessage(&model::CForecastDataSink::writeErrorMessage, forecastJob, message);
}

void CForecastRunner::sendFinalMessage(const SForecast& forecastJob,
                                       const std::string& message) const {
    this->sendMessage(&model::CForecastDataSink::writeFinalMessage, forecastJob, message);
}

template<typename WRITE>
void CForecastRunner::sendMessage(WRITE write,
                                  const SForecast& forecastJob,
                                  const std::string& message) const {
    model::CForecastDataSink sink(
        m_JobId, forecastJob.s_ForecastId, forecastJob.s_ForecastAlias,
        forecastJob.s_CreateTime, forecastJob.s_StartTime, forecastJob.forecastEnd(),
        // in an error case use the default expiry time
        forecastJob.s_CreateTime + DEFAULT_EXPIRY_TIME,
        forecastJob.s_MemoryUsage, m_ConcurrentOutputStream);
    (sink.*write)(message);
}

bool CForecastRunner::sufficientAvailableDiskSpace(std::size_t minForecastAvailableDiskSpace,
                                                   const char* path) {
    return sufficientAvailableDiskSpaceForPath(minForecastAvailableDiskSpace, path);
}

void CForecastRunner::SForecast::reset() {
    // clean up all non-simple types
    s_ForecastSeries.clear();
}

core_t::TTime CForecastRunner::SForecast::forecastEnd() const {
    return s_StartTime + s_Duration;
}
}
}
