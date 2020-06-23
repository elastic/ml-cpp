/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CForecastDataSink_h
#define INCLUDED_ml_model_CForecastDataSink_h

#include <core/CConcurrentWrapper.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CoreTypes.h>

#include <maths/CModel.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>
#include <model/SModelParams.h>

#include <rapidjson/allocators.h>
#include <rapidjson/fwd.h>

#include <boost/unordered_set.hpp>

#include <iosfwd>
#include <memory>
#include <string>

#include <stdint.h>

namespace ml {
namespace model {

//! \brief
//! Sink for data created from forecasting
//!
//! NOTE: Except for push, this is a stub implementation and going
//! to change (e.g. the json writing should not happen in this class).
class MODEL_EXPORT CForecastDataSink final : private core::CNonCopyable {
public:
    using TMathsModelPtr = std::shared_ptr<maths::CModel>;
    using TStrUMap = boost::unordered_set<std::string>;
    struct SForecastResultSeries;

    //! \brief Wrapper which supports creating a forecast for a single
    //! time series model.
    class MODEL_EXPORT CForecastModelWrapper {
    public:
        CForecastModelWrapper(model_t::EFeature feature,
                              const std::string& byFieldValue,
                              TMathsModelPtr&& forecastModel,
                              core_t::TTime firstDataTime,
                              core_t::TTime lastDataTime);

        CForecastModelWrapper(CForecastModelWrapper&& other) = default;

        CForecastModelWrapper(const CForecastModelWrapper&) = delete;
        CForecastModelWrapper& operator=(const CForecastModelWrapper&) = delete;

        bool forecast(const SForecastResultSeries& series,
                      core_t::TTime startTime,
                      core_t::TTime endTime,
                      double boundsPercentile,
                      CForecastDataSink& sink,
                      std::string& message) const;

    private:
        model_t::EFeature m_Feature;
        std::string m_ByFieldValue;
        TMathsModelPtr m_ForecastModel;
        core_t::TTime m_FirstDataTime;
        core_t::TTime m_LastDataTime;
    };

    //! Everything that defines 1 series of forecasts
    struct MODEL_EXPORT SForecastResultSeries {
        SForecastResultSeries(const SModelParams& modelParams);

        SForecastResultSeries(SForecastResultSeries&&) = default;

        SForecastResultSeries(const SForecastResultSeries&) = delete;
        SForecastResultSeries& operator=(const SForecastResultSeries&) = delete;

        SModelParams s_ModelParams;
        int s_DetectorIndex;
        std::vector<CForecastModelWrapper> s_ToForecast;
        std::string s_ToForecastPersisted;
        std::string s_PartitionFieldName;
        std::string s_PartitionFieldValue;
        std::string s_ByFieldName;
        double s_MinimumSeasonalVarianceScale;
    };

    //! \brief Data describing prerequisites prior predictions
    struct MODEL_EXPORT SForecastModelPrerequisites {
        std::size_t s_NumberOfModels;
        std::size_t s_NumberOfForecastableModels;
        std::size_t s_MemoryUsageForDetector;
        bool s_IsPopulation;
        bool s_IsSupportedFunction;
    };

private:
    static const std::string JOB_ID;
    static const std::string FORECAST_ID;
    static const std::string FORECAST_ALIAS;
    static const std::string DETECTOR_INDEX;
    static const std::string MODEL_FORECAST;
    static const std::string MODEL_FORECAST_STATS;
    static const std::string PARTITION_FIELD_NAME;
    static const std::string PARTITION_FIELD_VALUE;
    static const std::string FEATURE;
    static const std::string BY_FIELD_NAME;
    static const std::string BY_FIELD_VALUE;
    static const std::string LOWER;
    static const std::string UPPER;
    static const std::string PREDICTION;
    static const std::string BUCKET_SPAN;
    static const std::string PROCESSED_RECORD_COUNT;
    static const std::string CREATE_TIME;
    static const std::string TIMESTAMP;
    static const std::string START_TIME;
    static const std::string END_TIME;
    static const std::string EXPIRY_TIME;
    static const std::string MEMORY_USAGE;
    static const std::string MESSAGES;
    static const std::string PROCESSING_TIME_MS;
    static const std::string PROGRESS;
    static const std::string STATUS;

public:
    //! Create a DataSink instance
    CForecastDataSink(const std::string& jobId,
                      const std::string& forecastId,
                      const std::string& forecastAlias,
                      core_t::TTime createTime,
                      core_t::TTime startTime,
                      core_t::TTime endTime,
                      core_t::TTime expiryTime,
                      size_t memoryUsage,
                      core::CJsonOutputStreamWrapper& outStream);

    //! Push a forecast datapoint
    //! Note: No forecasting for models with over field, therefore no over field
    void push(const maths::SErrorBar errorBar,
              const std::string& feature,
              const std::string& partitionFieldName,
              const std::string& partitionFieldValue,
              const std::string& byFieldName,
              const std::string& byFieldValue,
              int detectorIndex);

    //! Write Statistics about the forecast, also marks the ending
    void writeStats(const double progress,
                    uint64_t runtime,
                    const TStrUMap& messages,
                    bool successful = true);

    //! Write a scheduled message to signal that validation was successful
    void writeScheduledMessage();

    //! Write an error message to signal a problem with forecasting
    void writeErrorMessage(const std::string& message);

    //! Write a message to signal that forecasting is complete
    //!
    //! This is used when exiting early but not as a result of an error
    void writeFinalMessage(const std::string& message);

    //! get the number of forecast records written
    uint64_t numRecordsWritten() const;

private:
    void writeCommonStatsFields(rapidjson::Value& doc);
    void push(bool flush, rapidjson::Value& doc);

private:
    //! The job ID
    std::string m_JobId;

    //! The forecast ID
    std::string m_ForecastId;

    //! The forecast alias
    std::string m_ForecastAlias;

    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;

    //! count of how many records written
    uint64_t m_NumRecordsWritten;

    //! Forecast create time
    core_t::TTime m_CreateTime;

    //! Forecast start time
    core_t::TTime m_StartTime;

    //! Forecast end time
    core_t::TTime m_EndTime;

    //! Forecast expiry time
    core_t::TTime m_ExpiryTime;

    //! Forecast memory usage for models
    size_t m_MemoryUsage;
};

} /* namespace model  */
} /* namespace ml */

#endif /* INCLUDED_ml_model_CForecastDataSink_h */
