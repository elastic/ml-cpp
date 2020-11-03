/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CAnomalyJobConfig_h
#define INCLUDED_ml_api_CAnomalyJobConfig_h

#include <core/CLogger.h>

#include <api/CDetectionRulesJsonParser.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief A parser to convert JSON configuration of an anomaly job JSON into an object
class API_EXPORT CAnomalyJobConfig {
public:
    class CAnalysisConfig {
    public:
        class CDetectorConfig {
        public:
            static const std::string FUNCTION;
            static const std::string FIELD_NAME;
            static const std::string BY_FIELD_NAME;
            static const std::string OVER_FIELD_NAME;
            static const std::string PARTITION_FIELD_NAME;
            static const std::string DETECTOR_DESCRIPTION;
            static const std::string EXCLUDE_FREQUENT;
            static const std::string CUSTOM_RULES;
            static const std::string USE_NULL;

        public:
            CDetectorConfig() {}

            void parse(const rapidjson::Value& json);

            std::string function() const { return m_Function; }
            std::string fieldName() const { return m_FieldName; }
            std::string byFieldName() const { return m_ByFieldName; }
            std::string overFieldName() const { return m_OverFieldName; }
            std::string partitionFieldName() const {
                return m_PartitionFieldName;
            }
            std::string excludeFrequent() const { return m_ExcludeFrequent; }
            std::string detectorDescription() const {
                return m_DetectorDescription;
            }
            CDetectionRulesJsonParser::TDetectionRuleVec customRules() const {
                return m_CustomRules;
            }
            bool useNull() const { return m_UseNull; }

        private:
            std::string m_Function{};
            std::string m_FieldName{};
            std::string m_ByFieldName{};
            std::string m_OverFieldName{};
            std::string m_PartitionFieldName{};
            std::string m_ExcludeFrequent{};
            std::string m_DetectorDescription{};
            CDetectionRulesJsonParser::TDetectionRuleVec m_CustomRules{};
            bool m_UseNull{false};
        };

    public:
        static const std::string BUCKET_SPAN;
        static const std::string SUMMARY_COUNT_FIELD_NAME;
        static const std::string CATEGORIZATION_FIELD_NAME;
        static const std::string CATEGORIZATION_FILTERS;
        static const std::string DETECTORS;
        static const std::string INFLUENCERS;
        static const std::string LATENCY;
        static const std::string PER_PARTITION_CATEGORIZATION;
        static const std::string ENABLED;
        static const std::string STOP_ON_WARN;

        static const std::size_t DEFAULT_BUCKET_SPAN;

    public:
        using TStrVec = std::vector<std::string>;
        using TDetctorConfigVec = std::vector<CDetectorConfig>;

    public:
        CAnalysisConfig() {}

        void parse(const rapidjson::Value& json);

        std::size_t bucketSpan() const { return m_BucketSpan; }

        std::string summaryCountFieldName() const {
            return m_SummaryCountFieldName;
        }
        std::string categorizationFieldName() const {
            return m_CategorizationFieldName;
        }
        const TStrVec& categorizationFilters() const {
            return m_CategorizationFilters;
        }
        bool perPartitionCategorizationEnabled() const {
            return m_PerPartitionCategorizationEnabled;
        }
        bool perPartitionCategorizationStopOnWarn() const {
            return m_PerPartitionCategorizationStopOnWarn;
        }
        const TDetctorConfigVec& detectorsConfig() const { return m_Detectors; }
        const TStrVec& influencers() const { return m_Influencers; }
        std::string latency() const { return m_Latency; }

    private:
        std::size_t m_BucketSpan{300}; // 5m
        std::string m_SummaryCountFieldName{};
        std::string m_CategorizationFieldName{};
        TStrVec m_CategorizationFilters{};
        bool m_PerPartitionCategorizationEnabled{false};
        bool m_PerPartitionCategorizationStopOnWarn{false};
        TDetctorConfigVec m_Detectors{};
        TStrVec m_Influencers{};
        std::string m_Latency{};
    };

    class CDataDescription {
    public:
        static const std::string TIME_FIELD; // unused in production
        static const std::string TIME_FORMAT;

    public:
        //! Default constructor
        CDataDescription() {}

        void parse(const rapidjson::Value& json);

        std::string timeField() const { return m_TimeField; }
        std::string timeFormat() const { return m_TimeFormat; }

    private:
        std::string m_TimeField;  // e.g. timestamp
        std::string m_TimeFormat; // e.g. epoch_ms
    };

    class CModelPlotConfig {
    public:
        static const std::string ANNOTATIONS_ENABLED;
        static const std::string ENABLED;
        static const std::string TERMS;

    public:
        //! Default constructor
        CModelPlotConfig() {}

        void parse(const rapidjson::Value& modelPlotConfig);

        bool annotationsEnabled() const { return m_AnnotationsEnabled; }
        bool enabled() const { return m_Enabled; }
        std::string terms() const { return m_Terms; }

    private:
        bool m_AnnotationsEnabled{true};
        bool m_Enabled{false};
        std::string m_Terms;
    };

    class CAnalysisLimits {
    public:
        static const std::string MODEL_MEMORY_LIMIT;
        static const std::string CATEGORIZATION_EXAMPLES_LIMIT;

        static const std::size_t DEFAULT_MEMORY_LIMIT;

    public:
        //! Default constructor
        CAnalysisLimits() {}

        void parse(const rapidjson::Value& analysLimits);

        //! Size of the memory limit for the resource monitor
        //! as a whole number of MB.
        std::size_t modelMemoryLimit() const { return m_ModelMemoryLimit; }

        long categorizationExamplesLimit() const {
            return m_CategorizationExamplesLimit;
        }

    private:
        long m_CategorizationExamplesLimit{4};
        std::size_t m_ModelMemoryLimit{};
    };

public:
    static const std::string JOB_ID;
    static const std::string JOB_TYPE;
    static const std::string ANALYSIS_CONFIG;
    static const std::string DATA_DESCRIPTION;
    static const std::string MODEL_PLOT_CONFIG;
    static const std::string ANALYSIS_LIMITS;

public:
    //! Default constructor
    CAnomalyJobConfig() {}

    bool parse(const std::string& json);

    std::string jobId() const { return m_JobId; }
    std::string jobType() const { return m_JobType; }
    const CAnalysisConfig& analysisConfig() const { return m_AnalysisConfig; }
    const CDataDescription& dataDescription() const {
        return m_DataDescription;
    }
    const CModelPlotConfig& modelPlotConfig() const { return m_ModelConfig; }
    const CAnalysisLimits& analysisLimits() const { return m_AnalysisLimits; }

private:
    std::string m_JobId;
    std::string m_JobType;
    CAnalysisConfig m_AnalysisConfig;
    CDataDescription m_DataDescription;
    CModelPlotConfig m_ModelConfig;
    CAnalysisLimits m_AnalysisLimits;
};
}
}
#endif // INCLUDED_ml_api_CAnomalyJobConfig_h
