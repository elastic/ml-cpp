/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <api/CForecastRunner.h>

#include <core/CLogger.h>
#include <core/CStopWatch.h>
#include <core/CTimeUtils.h>

#include <model/CForecastDataSink.h>
#include <model/ModelTypes.h>

#include <boost/bind.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <sstream>

namespace ml {
namespace api {

namespace {
const std::string EMPTY_STRING;
}

const std::string CForecastRunner::ERROR_FORECAST_REQUEST_FAILED_TO_PARSE("Failed to parse forecast request: ");
const std::string CForecastRunner::ERROR_NO_FORECAST_ID("forecast ID must be specified and non empty");
const std::string CForecastRunner::ERROR_TOO_MANY_JOBS(
    "Forecast cannot be executed due to queue limit. Please wait for requests to finish and try again");
const std::string CForecastRunner::ERROR_NO_MODELS(
    "Forecast cannot be executed as model is not yet established. Job requires more time to learn");
const std::string CForecastRunner::ERROR_NO_DATA_PROCESSED(
    "Forecast cannot be executed as job requires data to have been processed and modeled");
const std::string CForecastRunner::ERROR_NO_CREATE_TIME("Forecast create time must be specified and non zero");
const std::string
    CForecastRunner::ERROR_BAD_MEMORY_STATUS("Forecast cannot be executed as model memory status is not OK");
const std::string CForecastRunner::ERROR_MEMORY_LIMIT(
    "Forecast cannot be executed as forecast memory usage is predicted to exceed 20MB");
const std::string
    CForecastRunner::ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS("Forecast is not supported for population analysis");
const std::string CForecastRunner::ERROR_NO_SUPPORTED_FUNCTIONS("Forecast is not supported for the used functions");
const std::string
    CForecastRunner::WARNING_DURATION_LIMIT("Forecast duration exceeds internal limit, setting to 8 weeks");
const std::string CForecastRunner::WARNING_INVALID_EXPIRY("Forecast expires_in invalid, setting to 14 days");
const std::string CForecastRunner::INFO_DEFAULT_DURATION("Forecast duration not specified, setting to 1 day");
const std::string CForecastRunner::INFO_DEFAULT_EXPIRY("Forecast expires_in not specified, setting to 14 days");
const std::string
    CForecastRunner::INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST("Insufficient history to forecast for all models");

CForecastRunner::SForecast::SForecast()
    : s_ForecastId(),
      s_ForecastAlias(),
      s_ForecastSeries(),
      s_CreateTime(0),
      s_StartTime(0),
      s_Duration(0),
      s_ExpiryTime(0),
      s_BoundsPercentile(0),
      s_NumberOfModels(0),
      s_NumberOfForecastableModels(0),
      s_MemoryUsage(0),
      s_Messages() {
}

CForecastRunner::SForecast::SForecast(SForecast&& other)
    : s_ForecastId(std::move(other.s_ForecastId)),
      s_ForecastAlias(std::move(other.s_ForecastAlias)),
      s_ForecastSeries(std::move(other.s_ForecastSeries)),
      s_CreateTime(other.s_CreateTime),
      s_StartTime(other.s_StartTime),
      s_Duration(other.s_Duration),
      s_ExpiryTime(other.s_ExpiryTime),
      s_BoundsPercentile(other.s_BoundsPercentile),
      s_NumberOfModels(other.s_NumberOfModels),
      s_NumberOfForecastableModels(other.s_NumberOfForecastableModels),
      s_MemoryUsage(other.s_MemoryUsage),
      s_Messages(other.s_Messages) {
}

CForecastRunner::SForecast& CForecastRunner::SForecast::operator=(SForecast&& other) {
    s_ForecastId = std::move(other.s_ForecastId);
    s_ForecastAlias = std::move(other.s_ForecastAlias);
    s_ForecastSeries = std::move(other.s_ForecastSeries);
    s_CreateTime = other.s_CreateTime;
    s_StartTime = other.s_StartTime;
    s_Duration = other.s_Duration;
    s_ExpiryTime = other.s_ExpiryTime;
    s_BoundsPercentile = other.s_BoundsPercentile;
    s_NumberOfModels = other.s_NumberOfModels;
    s_NumberOfForecastableModels = other.s_NumberOfForecastableModels;
    s_MemoryUsage = other.s_MemoryUsage;
    s_Messages = other.s_Messages;

    return *this;
}

CForecastRunner::CForecastRunner(const std::string& jobId,
                                 core::CJsonOutputStreamWrapper& strmOut,
                                 model::CResourceMonitor& resourceMonitor)
    : m_JobId(jobId), m_ConcurrentOutputStream(strmOut), m_ResourceMonitor(resourceMonitor), m_Shutdown(false) {
    m_Worker = std::thread([this] { this->forecastWorker(); });
}

CForecastRunner::~CForecastRunner() {
    // shutdown
    m_Shutdown = true;
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
    while (!m_Shutdown && !m_ForecastJobs.empty()) {
        // items in the queue, wait
        m_WorkCompleteCondition.wait(lock);
    }
}

void CForecastRunner::forecastWorker() {
    SForecast forecastJob;
    while (!m_Shutdown) {
        if (this->tryGetJob(forecastJob)) {
            LOG_INFO("Start forecasting from " << core::CTimeUtils::toIso8601(forecastJob.s_StartTime) << " to "
                                               << core::CTimeUtils::toIso8601(forecastJob.forecastEnd()));

            core::CStopWatch timer(true);
            uint64_t lastStatsUpdate = 0;

            LOG_TRACE("about to create sink");
            model::CForecastDataSink sink(m_JobId,
                                          forecastJob.s_ForecastId,
                                          forecastJob.s_ForecastAlias,
                                          forecastJob.s_CreateTime,
                                          forecastJob.s_StartTime,
                                          forecastJob.forecastEnd(),
                                          forecastJob.s_ExpiryTime,
                                          forecastJob.s_MemoryUsage,
                                          m_ConcurrentOutputStream);

            std::string message;

            // collecting the runtime messages first and sending it in 1 go
            TStrUSet messages(forecastJob.s_Messages);
            double processedModels = 0;
            double totalNumberOfForecastableModels = static_cast<double>(forecastJob.s_NumberOfForecastableModels);
            size_t failedForecasts = 0;
            sink.writeStats(0.0, 0, forecastJob.s_Messages);

            // while loops allow us to free up memory for every model right after each forecast is done
            while (!forecastJob.s_ForecastSeries.empty()) {
                TForecastResultSeries& series = forecastJob.s_ForecastSeries.back();

                while (!series.s_ToForecast.empty()) {
                    const TForecastModelWrapper& model = series.s_ToForecast.back();
                    model_t::TDouble1VecDouble1VecPr support = model_t::support(model.s_Feature);
                    bool success = model.s_ForecastModel->forecast(forecastJob.s_StartTime,
                                                                   forecastJob.forecastEnd(),
                                                                   forecastJob.s_BoundsPercentile,
                                                                   support.first,
                                                                   support.second,
                                                                   boost::bind(&model::CForecastDataSink::push,
                                                                               &sink,
                                                                               _1,
                                                                               model_t::print(model.s_Feature),
                                                                               series.s_PartitionFieldName,
                                                                               series.s_PartitionFieldValue,
                                                                               series.s_ByFieldName,
                                                                               model.s_ByFieldValue,
                                                                               series.s_DetectorIndex),
                                                                   message);
                    series.s_ToForecast.pop_back();

                    if (success == false) {
                        LOG_DEBUG("Detector " << series.s_DetectorIndex << " failed to forecast");
                        ++failedForecasts;
                    }

                    if (message.empty() == false) {
                        messages.insert("Detector[" + std::to_string(series.s_DetectorIndex) + "]: " + message);
                        message.clear();
                    }

                    ++processedModels;

                    if (processedModels != totalNumberOfForecastableModels) {
                        uint64_t elapsedTime = timer.lap();
                        if (elapsedTime - lastStatsUpdate > MINIMUM_TIME_ELAPSED_FOR_STATS_UPDATE) {
                            sink.writeStats(
                                processedModels / totalNumberOfForecastableModels, elapsedTime, forecastJob.s_Messages);
                            lastStatsUpdate = elapsedTime;
                        }
                    }
                }
                forecastJob.s_ForecastSeries.pop_back();
            }
            // write final message
            sink.writeStats(1.0, timer.stop(), messages, failedForecasts != forecastJob.s_NumberOfForecastableModels);

            // important: reset the structure to decrease shared pointer reference counts
            forecastJob.reset();
            LOG_INFO("Finished forecasting, wrote " << sink.numRecordsWritten() << " records");

            // signal that job is done
            m_WorkCompleteCondition.notify_all();
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
    if (m_Shutdown) {
        return false;
    }

    m_WorkAvailableCondition.wait(lock);
    return false;
}

bool CForecastRunner::pushForecastJob(const std::string& controlMessage,
                                      const TAnomalyDetectorPtrVec& detectors,
                                      const core_t::TTime lastResultsTime) {
    SForecast forecastJob;
    if (parseAndValidateForecastRequest(controlMessage,
                                        forecastJob,
                                        lastResultsTime,
                                        boost::bind(&CForecastRunner::sendErrorMessage, this, _1, _2)) == false) {
        return false;
    }

    if (m_ResourceMonitor.getMemoryStatus() != model_t::E_MemoryStatusOk) {
        this->sendErrorMessage(forecastJob, ERROR_BAD_MEMORY_STATUS);
        return false;
    }

    size_t totalNumberOfModels = 0;
    size_t totalNumberOfForecastModels = 0;
    bool atLeastOneNonPopulationModel = false;
    bool atLeastOneSupportedFunction = false;
    size_t totalMemoryUsage = 0;

    // 1st loop over the detectors to check prerequisites
    for (const auto& detector : detectors) {
        if (detector.get() == nullptr) {
            LOG_ERROR("Unexpected empty detector found");
            continue;
        }

        model::CForecastDataSink::SForecastModelPrerequisites prerequisites = detector->getForecastPrerequisites();

        totalNumberOfModels += prerequisites.s_NumberOfModels;
        totalNumberOfForecastModels += prerequisites.s_NumberOfForecastableModels;
        atLeastOneNonPopulationModel = atLeastOneNonPopulationModel || !prerequisites.s_IsPopulation;
        atLeastOneSupportedFunction = atLeastOneSupportedFunction || prerequisites.s_IsSupportedFunction;
        totalMemoryUsage += prerequisites.s_MemoryUsageForDetector;

        if (totalMemoryUsage >= MAX_FORECAST_MODEL_MEMORY) {
            // note: for now MAX_FORECAST_MODEL_MEMORY is a static limit, a user can not change it
            this->sendErrorMessage(forecastJob, ERROR_MEMORY_LIMIT);
            return false;
        }
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
    TForecastResultSeriesVec s;

    for (const auto& detector : detectors) {
        if (detector.get() == nullptr) {
            LOG_ERROR("Unexpected empty detector found");
            continue;
        }

        forecastJob.s_ForecastSeries.emplace_back(detector->getForecastModels());
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
                                                      const TErrorFunc& errorFunction) {
    std::istringstream stringStream(controlMessage.substr(1));
    forecastJob.s_StartTime = lastResultsTime;

    core_t::TTime expiresIn = 0l;
    boost::property_tree::ptree properties;
    try {
        boost::property_tree::read_json(stringStream, properties);

        forecastJob.s_ForecastId = properties.get<std::string>("forecast_id", EMPTY_STRING);
        forecastJob.s_ForecastAlias = properties.get<std::string>("forecast_alias", EMPTY_STRING);
        forecastJob.s_Duration = properties.get<core_t::TTime>("duration", 0);
        forecastJob.s_CreateTime = properties.get<core_t::TTime>("create_time", 0);

        // use -1 as default to allow 0 as 'never expires'
        expiresIn = properties.get<core_t::TTime>("expires_in", -1l);

        // note: this is not exposed on x-pack side
        forecastJob.s_BoundsPercentile = properties.get<double>("boundspercentile", 95.0);
    } catch (const std::exception& e) {
        LOG_ERROR(ERROR_FORECAST_REQUEST_FAILED_TO_PARSE << e.what());
        return false;
    }

    if (forecastJob.s_ForecastId.empty()) {
        LOG_ERROR(ERROR_NO_FORECAST_ID);
        return false;
    }

    // from now we have a forecast ID and can send error messages
    if (lastResultsTime == 0l) {
        errorFunction(forecastJob, ERROR_NO_DATA_PROCESSED);
        return false;
    }

    if (forecastJob.s_CreateTime == 0) {
        errorFunction(forecastJob, ERROR_NO_CREATE_TIME);
        return false;
    }

    // Limit the forecast end time to 8 weeks after the last result
    // to be replaced by https://github.com/elastic/machine-learning-cpp/issues/443
    // TODO this is a temporary fix to prevent the analysis blowing up
    // if you change this value, also change the log string
    // todo: refactor validation out from here
    core_t::TTime maxDuration = 8 * core::constants::WEEK;
    if (forecastJob.s_Duration > maxDuration) {
        LOG_INFO(WARNING_DURATION_LIMIT);
        forecastJob.s_Messages.insert(WARNING_DURATION_LIMIT);
        forecastJob.s_Duration = maxDuration;
    }

    if (forecastJob.s_Duration == 0) {
        // only log
        forecastJob.s_Duration = core::constants::DAY;
        LOG_INFO(INFO_DEFAULT_DURATION);
    }

    if (expiresIn < -1l) {
        // only log
        expiresIn = DEFAULT_EXPIRY_TIME;
        LOG_INFO(WARNING_INVALID_EXPIRY);
    } else if (expiresIn == -1l) {
        // only log
        expiresIn = DEFAULT_EXPIRY_TIME;
        LOG_DEBUG(INFO_DEFAULT_EXPIRY);
    }

    forecastJob.s_ExpiryTime = forecastJob.s_CreateTime + expiresIn;

    return true;
}

void CForecastRunner::sendScheduledMessage(const SForecast& forecastJob) const {
    LOG_DEBUG("job passed forecast validation, scheduled for forecasting");
    model::CForecastDataSink sink(m_JobId,
                                  forecastJob.s_ForecastId,
                                  forecastJob.s_ForecastAlias,
                                  forecastJob.s_CreateTime,
                                  forecastJob.s_StartTime,
                                  forecastJob.forecastEnd(),
                                  forecastJob.s_ExpiryTime,
                                  forecastJob.s_MemoryUsage,
                                  m_ConcurrentOutputStream);
    sink.writeScheduledMessage();
}

void CForecastRunner::sendErrorMessage(const SForecast& forecastJob, const std::string& message) const {
    LOG_ERROR(message);
    this->sendMessage(&model::CForecastDataSink::writeErrorMessage, forecastJob, message);
}

void CForecastRunner::sendFinalMessage(const SForecast& forecastJob, const std::string& message) const {
    this->sendMessage(&model::CForecastDataSink::writeFinalMessage, forecastJob, message);
}

template<typename WRITE>
void CForecastRunner::sendMessage(WRITE write, const SForecast& forecastJob, const std::string& message) const {
    model::CForecastDataSink sink(m_JobId,
                                  forecastJob.s_ForecastId,
                                  forecastJob.s_ForecastAlias,
                                  forecastJob.s_CreateTime,
                                  forecastJob.s_StartTime,
                                  forecastJob.forecastEnd(),
                                  // in an error case use the default expiry time
                                  forecastJob.s_CreateTime + DEFAULT_EXPIRY_TIME,
                                  forecastJob.s_MemoryUsage,
                                  m_ConcurrentOutputStream);
    (sink.*write)(message);
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
