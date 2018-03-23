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
#include <model/CModelParams.h>

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
        using TMathsModelPtr = boost::shared_ptr<maths::CModel>;
    using TStrUMap = boost::unordered_set<std::string>;

    //! Wrapper for 1 timeseries model, its feature and by Field
    struct MODEL_EXPORT SForecastModelWrapper {
        SForecastModelWrapper(model_t::EFeature feature, TMathsModelPtr&& forecastModel, const std::string& byFieldValue);

        SForecastModelWrapper(SForecastModelWrapper&& other);

        SForecastModelWrapper(const SForecastModelWrapper& that) = delete;
        SForecastModelWrapper& operator=(const SForecastModelWrapper&) = delete;

        model_t::EFeature s_Feature;
        TMathsModelPtr s_ForecastModel;
        std::string s_ByFieldValue;
    };

    //! Everything that defines 1 series of forecasts
    struct MODEL_EXPORT SForecastResultSeries {
            SForecastResultSeries(const SModelParams &modelParams);

        SForecastResultSeries(SForecastResultSeries&& other);

        SForecastResultSeries(const SForecastResultSeries& that) = delete;
        SForecastResultSeries& operator=(const SForecastResultSeries&) = delete;

            SModelParams                            s_ModelParams;
        int s_DetectorIndex;
        std::vector<SForecastModelWrapper> s_ToForecast;
            std::string                             s_ToForecastPersisted;
        std::string s_PartitionFieldName;
        std::string s_PartitionFieldValue;
        std::string s_ByFieldName;
            double                                  s_MinimumSeasonalVarianceScale;
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
    void writeStats(const double progress, uint64_t runtime, const TStrUMap& messages, bool successful = true);

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
