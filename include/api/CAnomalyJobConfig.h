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

#include <model/CLimits.h>
#include <model/FunctionTypes.h>

#include <rapidjson/document.h>

#include <string>
#include <vector>

class CTestAnomalyJob;

namespace ml {
namespace model {
class CAnomalyDetectorModelConfig;
}

namespace api {

//! \brief A parser to convert JSON configuration of an anomaly job JSON into an object
class API_EXPORT CAnomalyJobConfig {
public:
    using TStrDetectionRulePr = std::pair<std::string, model::CDetectionRule>;
    using TStrDetectionRulePrVec = std::vector<TStrDetectionRulePr>;

public:
    class API_EXPORT CAnalysisConfig {
    public:
        class API_EXPORT CDetectorConfig {
        public:
            static const std::string FUNCTION;
            static const std::string FIELD_NAME;
            static const std::string BY_FIELD_NAME;
            static const std::string OVER_FIELD_NAME;
            static const std::string PARTITION_FIELD_NAME;
            static const std::string DETECTOR_DESCRIPTION;
            static const std::string DETECTOR_INDEX;
            static const std::string EXCLUDE_FREQUENT;
            static const std::string CUSTOM_RULES;
            static const std::string USE_NULL;
            static const std::string ALL_TOKEN;
            static const std::string BY_TOKEN;
            static const std::string NONE_TOKEN;
            static const std::string OVER_TOKEN;

            //! Strings that define the type of analysis to run
            static const std::string FUNCTION_COUNT;
            static const std::string FUNCTION_COUNT_ABBREV;
            static const std::string FUNCTION_LOW_COUNT;
            static const std::string FUNCTION_LOW_COUNT_ABBREV;
            static const std::string FUNCTION_HIGH_COUNT;
            static const std::string FUNCTION_HIGH_COUNT_ABBREV;
            static const std::string FUNCTION_DISTINCT_COUNT;
            static const std::string FUNCTION_DISTINCT_COUNT_ABBREV;
            static const std::string FUNCTION_LOW_DISTINCT_COUNT;
            static const std::string FUNCTION_LOW_DISTINCT_COUNT_ABBREV;
            static const std::string FUNCTION_HIGH_DISTINCT_COUNT;
            static const std::string FUNCTION_HIGH_DISTINCT_COUNT_ABBREV;
            static const std::string FUNCTION_NON_ZERO_COUNT;
            static const std::string FUNCTION_NON_ZERO_COUNT_ABBREV;
            static const std::string FUNCTION_RARE_NON_ZERO_COUNT;
            static const std::string FUNCTION_RARE_NON_ZERO_COUNT_ABBREV;
            static const std::string FUNCTION_RARE;
            static const std::string FUNCTION_RARE_COUNT;
            static const std::string FUNCTION_FREQ_RARE;
            static const std::string FUNCTION_FREQ_RARE_ABBREV;
            static const std::string FUNCTION_FREQ_RARE_COUNT;
            static const std::string FUNCTION_FREQ_RARE_COUNT_ABBREV;
            static const std::string FUNCTION_LOW_NON_ZERO_COUNT;
            static const std::string FUNCTION_LOW_NON_ZERO_COUNT_ABBREV;
            static const std::string FUNCTION_HIGH_NON_ZERO_COUNT;
            static const std::string FUNCTION_HIGH_NON_ZERO_COUNT_ABBREV;
            static const std::string FUNCTION_INFO_CONTENT;
            static const std::string FUNCTION_LOW_INFO_CONTENT;
            static const std::string FUNCTION_HIGH_INFO_CONTENT;
            static const std::string FUNCTION_METRIC;
            static const std::string FUNCTION_AVERAGE;
            static const std::string FUNCTION_MEAN;
            static const std::string FUNCTION_LOW_MEAN;
            static const std::string FUNCTION_HIGH_MEAN;
            static const std::string FUNCTION_LOW_AVERAGE;
            static const std::string FUNCTION_HIGH_AVERAGE;
            static const std::string FUNCTION_MEDIAN;
            static const std::string FUNCTION_LOW_MEDIAN;
            static const std::string FUNCTION_HIGH_MEDIAN;
            static const std::string FUNCTION_MIN;
            static const std::string FUNCTION_MAX;
            static const std::string FUNCTION_VARIANCE;
            static const std::string FUNCTION_LOW_VARIANCE;
            static const std::string FUNCTION_HIGH_VARIANCE;
            static const std::string FUNCTION_SUM;
            static const std::string FUNCTION_LOW_SUM;
            static const std::string FUNCTION_HIGH_SUM;
            static const std::string FUNCTION_NON_NULL_SUM;
            static const std::string FUNCTION_NON_NULL_SUM_ABBREV;
            static const std::string FUNCTION_LOW_NON_NULL_SUM;
            static const std::string FUNCTION_LOW_NON_NULL_SUM_ABBREV;
            static const std::string FUNCTION_HIGH_NON_NULL_SUM;
            static const std::string FUNCTION_HIGH_NON_NULL_SUM_ABBREV;
            static const std::string FUNCTION_TIME_OF_DAY;
            static const std::string FUNCTION_TIME_OF_WEEK;
            static const std::string FUNCTION_LAT_LONG;
            static const std::string FUNCTION_MAX_VELOCITY;
            static const std::string FUNCTION_MIN_VELOCITY;
            static const std::string FUNCTION_MEAN_VELOCITY;
            static const std::string FUNCTION_SUM_VELOCITY;

        public:
            CDetectorConfig() {}

            // Convenience ctor intended for use by the unit tests only
            CDetectorConfig(const std::string& functionName,
                            const std::string& fieldName,
                            const std::string& byFieldName,
                            const std::string& overFieldName,
                            const std::string& partitionFieldName)
                : m_FunctionName(functionName),
                  m_FieldName(fieldName), m_ByFieldName{byFieldName},
                  m_OverFieldName{overFieldName}, m_PartitionFieldName{partitionFieldName} {
                this->determineFunction(false);
                this->decipherExcludeFrequentSetting();
            }

            void parse(const rapidjson::Value& detectorConfig,
                       const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
                       bool haveSummaryCountField,
                       int detectorIndex,
                       CDetectionRulesJsonParser::TDetectionRuleVec& detectionRules);

            int detectorIndex() const { return m_DetectorIndex; }
            std::string functionName() const { return m_FunctionName; }
            model::function_t::EFunction function() const { return m_Function; }
            std::string fieldName() const { return m_FieldName; }
            std::string byFieldName() const { return m_ByFieldName; }
            std::string overFieldName() const { return m_OverFieldName; }
            std::string partitionFieldName() const {
                return m_PartitionFieldName;
            }

            model_t::EExcludeFrequent excludeFrequent() const;

            std::string detectorDescription() const {
                return m_DetectorDescription;
            }
            bool useNull() const { return m_UseNull; }
            bool isPopulation() const {
                return m_OverFieldName.empty() == false;
            }

        private:
            bool determineFunction(bool haveSummaryCountField);

            bool decipherExcludeFrequentSetting();

        private:
            std::string m_FunctionName{};
            model::function_t::EFunction m_Function;
            std::string m_FieldName{};
            std::string m_ByFieldName{};
            std::string m_OverFieldName{};
            std::string m_PartitionFieldName{};
            std::string m_ExcludeFrequent{};
            bool m_ByHasExcludeFrequent{false};
            bool m_OverHasExcludeFrequent{false};
            std::string m_DetectorDescription{};
            int m_DetectorIndex{};
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
        static const std::string MULTIVARIATE_BY_FIELDS;
        static const std::string PER_PARTITION_CATEGORIZATION;
        static const std::string ENABLED;
        static const std::string STOP_ON_WARN;

        static const core_t::TTime DEFAULT_BUCKET_SPAN;
        static const core_t::TTime DEFAULT_LATENCY;

        static const std::string CLEAR;
        static const char SUFFIX_SEPARATOR;
        static const std::string SCHEDULED_EVENT_PREFIX;
        static const std::string DESCRIPTION_SUFFIX;
        static const std::string RULES_SUFFIX;

    public:
        using TStrVec = std::vector<std::string>;
        using TDetectorConfigVec = std::vector<CDetectorConfig>;

        using TIntDetectionRuleVecUMap =
            boost::unordered_map<int, CDetectionRulesJsonParser::TDetectionRuleVec>;

        //! Used to maintain a list of all unique config keys
        using TIntSet = std::set<int>;

    public:
        //! Default constructor
        CAnalysisConfig() {}

        //! Construct with just a categorization field.  (In the case of a
        //! categorization job, this is all that is needed for this config.)
        CAnalysisConfig(const std::string& categorizationFieldName)
            : m_CategorizationFieldName{categorizationFieldName} {}

        //! Constructor taking a map of detector rule filters keyed by filter_id &
        //! a vector of scheduled events data
        CAnalysisConfig(const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
                        const TStrDetectionRulePrVec& scheduledEvents)
            : m_RuleFilters(ruleFilters), m_ScheduledEvents(scheduledEvents) {}

        void init(const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
                  const TStrDetectionRulePrVec& scheduledEvents) {
            m_RuleFilters = ruleFilters;
            m_ScheduledEvents = scheduledEvents;
        }

        void parse(const rapidjson::Value& json);

        bool updateFilters(const boost::property_tree::ptree& propTree);

        bool updateScheduledEvents(const boost::property_tree::ptree& propTree);

        core_t::TTime bucketSpan() const { return m_BucketSpan; }

        std::string summaryCountFieldName() const {
            return m_SummaryCountFieldName;
        }
        std::string categorizationFieldName() const {
            return m_CategorizationFieldName;
        }
        std::string categorizationPartitionFieldName() const {
            return m_CategorizationPartitionFieldName;
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
        const TDetectorConfigVec& detectorsConfig() const {
            return m_Detectors;
        }
        const TStrVec& influencers() const { return m_Influencers; }
        core_t::TTime latency() const { return m_Latency; }

        bool multivariateByFields() const { return m_MultivariateByFields; }

        const TIntDetectionRuleVecUMap& detectionRules() const {
            return m_DetectorRules;
        }

        const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters() const {
            return m_RuleFilters;
        }

        //! Get the scheduled events
        const TStrDetectionRulePrVec& scheduledEvents() const {
            return m_ScheduledEvents;
        }

        ml::model::CAnomalyDetectorModelConfig makeModelConfig() const;

        static core_t::TTime durationSeconds(const std::string& durationString,
                                             core_t::TTime defaultDuration);

        bool parseRules(int detectorIndex, const std::string& rules);

    private:
        // Convenience method intended for use by the unit tests only
        void addDetector(const std::string& functionName,
                         const std::string& fieldName,
                         const std::string& byFieldName,
                         const std::string& overFieldName,
                         const std::string& partitionFieldName,
                         const TStrVec& influencers,
                         const std::string& summaryCountFieldName) {
            m_Influencers = influencers;
            m_SummaryCountFieldName = summaryCountFieldName;
            m_Detectors.emplace_back(functionName, fieldName, byFieldName,
                                     overFieldName, partitionFieldName);
        }

        bool processFilter(const std::string& key, const std::string& value);

        //! Process and store a scheduled event
        bool processScheduledEvent(const boost::property_tree::ptree& propTree,
                                   const std::string& key,
                                   const std::string& value,
                                   TIntSet& handledScheduledEvents);

        bool parseRules(CDetectionRulesJsonParser::TDetectionRuleVec& detectionRules,
                        const std::string& rules);

    private:
        core_t::TTime m_BucketSpan{DEFAULT_BUCKET_SPAN};
        std::string m_SummaryCountFieldName{};
        std::string m_CategorizationFieldName{};
        std::string m_CategorizationPartitionFieldName{};
        TStrVec m_CategorizationFilters{};
        bool m_PerPartitionCategorizationEnabled{false};
        bool m_PerPartitionCategorizationStopOnWarn{false};
        TDetectorConfigVec m_Detectors{};
        TStrVec m_Influencers{};
        core_t::TTime m_Latency{DEFAULT_LATENCY};
        bool m_MultivariateByFields{false};

        //! The detection rules per detector index.
        TIntDetectionRuleVecUMap m_DetectorRules;

        //! The filters per id used by categorical rule conditions.
        CDetectionRulesJsonParser::TStrPatternSetUMap m_RuleFilters{};

        //! The scheduled events (events apply to all detectors).
        //! Events consist of a description and a detection rule
        TStrDetectionRulePrVec m_ScheduledEvents{};

        friend class ::CTestAnomalyJob;
    };

    class API_EXPORT CDataDescription {
    public:
        static const std::string TIME_FIELD;

        //  The time format present in the job config is in Java format and
        // should be ignored in the C++ code. In any case, in production, the
        // time field is always specified in seconds since epoch.
        static const std::string TIME_FORMAT;

    public:
        //! Default constructor
        CDataDescription() {}

        void parse(const rapidjson::Value& json);

        std::string timeField() const { return m_TimeField; }

    private:
        std::string m_TimeField;  // e.g. timestamp
        std::string m_TimeFormat; // e.g. epoch_ms
    };

    class API_EXPORT CModelPlotConfig {
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

        // The terms string is a comma separated list of partition or
        // by field values, but with no form of escaping.
        // TODO improve this to be a more robust format
        std::string terms() const { return m_Terms; }

    private:
        bool m_AnnotationsEnabled{false};
        bool m_Enabled{false};
        std::string m_Terms;
    };

    class API_EXPORT CAnalysisLimits {
    public:
        static const std::string MODEL_MEMORY_LIMIT;
        static const std::string CATEGORIZATION_EXAMPLES_LIMIT;

        static const std::size_t DEFAULT_MEMORY_LIMIT_BYTES;

    public:
        //! Default constructor
        CAnalysisLimits() {}

        void parse(const rapidjson::Value& analysLimits);

        //! Size of the memory limit for the resource monitor
        //! as a whole number of MB.
        std::size_t modelMemoryLimitMb() const { return m_ModelMemoryLimitMb; }

        std::size_t categorizationExamplesLimit() const {
            return m_CategorizationExamplesLimit;
        }

        static std::size_t modelMemoryLimitMb(const std::string& memoryLimitStr);

    private:
        std::size_t m_CategorizationExamplesLimit{model::CLimits::DEFAULT_RESULTS_MAX_EXAMPLES};
        std::size_t m_ModelMemoryLimitMb{model::CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB};
    };

    class API_EXPORT CFilterConfig {
    public:
        static const std::string FILTER_ID;
        static const std::string ITEMS;

    public:
        CFilterConfig() {}

        void parse(const rapidjson::Value& filterConfig,
                   CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters);

        std::string name() const { return m_FilterName; }
        CAnomalyJobConfig::CAnalysisConfig::TStrVec filterList() const {
            return m_FilterList;
        }

    private:
        using TStrVec = std::vector<std::string>;

        std::string m_FilterName;
        TStrVec m_FilterList;
    };

    class API_EXPORT CEventConfig {
    public:
        static const std::string DESCRIPTION;
        static const std::string RULES;

    public:
        void parse(const rapidjson::Value& filterConfig,
                   const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
                   TStrDetectionRulePrVec& scheduledEvents);

    private:
        std::string m_Description;
        CDetectionRulesJsonParser::TDetectionRuleVec m_DetectionRules;
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

    explicit CAnomalyJobConfig(const std::string& categorizationFieldName)
        : m_AnalysisConfig(categorizationFieldName) {}

    // This one is only needed for historical reasons. Once The Java side sends the
    // scheduled events and filters config as JSON this can go.
    CAnomalyJobConfig(const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
                      const TStrDetectionRulePrVec& scheduledEvents)
        : m_AnalysisConfig(ruleFilters, scheduledEvents) {}

    bool readFile(const std::string& fileName, std::string& fileContents);

    bool initFromFile(const std::string& configFile);

    bool initFromFiles(const std::string& configFile,
                       const std::string& filtersConfigFile,
                       const std::string& eventsConfigFile);

    bool parse(const std::string& json);
    bool parseFilterConfig(const std::string& json);
    bool parseEventConfig(const std::string& json);

    std::string jobId() const { return m_JobId; }
    std::string jobType() const { return m_JobType; }
    const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters() const {
        return m_RuleFilters;
    }
    CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters() {
        return m_RuleFilters;
    }
    const TStrDetectionRulePrVec& scheduledEvents() const {
        return m_ScheduledEvents;
    }
    CAnalysisConfig& analysisConfig() { return m_AnalysisConfig; }
    const CAnalysisConfig& analysisConfig() const { return m_AnalysisConfig; }
    const CDataDescription& dataDescription() const {
        return m_DataDescription;
    }
    const CModelPlotConfig& modelPlotConfig() const { return m_ModelConfig; }
    const CAnalysisLimits& analysisLimits() const { return m_AnalysisLimits; }
    bool isInitialized() const { return m_IsInitialized; }

private:
    std::string m_JobId;
    std::string m_JobType;
    bool m_IsInitialized{false};
    CDetectionRulesJsonParser::TStrPatternSetUMap m_RuleFilters;
    TStrDetectionRulePrVec m_ScheduledEvents;
    CAnalysisConfig m_AnalysisConfig;
    CDataDescription m_DataDescription;
    CModelPlotConfig m_ModelConfig;
    CAnalysisLimits m_AnalysisLimits;

    std::vector<CFilterConfig> m_Filters{};
    std::vector<CEventConfig> m_Events{};
};
}
}
#endif // INCLUDED_ml_api_CAnomalyJobConfig_h
