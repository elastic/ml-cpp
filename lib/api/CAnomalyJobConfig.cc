/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CAnomalyJobConfig.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>
#include <core/Constants.h>

#include <api/CAnomalyJobConfigReader.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>
#include <model/FunctionTypes.h>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#ifdef Windows
// rapidjson::Writer<rapidjson::StringBuffer> gets instantiated in the core
// library, and on Windows it gets exported too, because
// CRapidJsonConcurrentLineWriter inherits from it and is also exported.
// To avoid breaching the one-definition rule we must reuse this exported
// instantiation, as deduplication of template instantiations doesn't work
// across DLLs.  To make this even more confusing, this is only strictly
// necessary when building without optimisation, because with optimisation
// enabled the instantiation in this library gets inlined to the extent that
// there are no clashing symbols.
template class CORE_EXPORT rapidjson::Writer<rapidjson::StringBuffer>;
#endif

namespace ml {
namespace api {

const std::string CAnomalyJobConfig::JOB_ID{"job_id"};
const std::string CAnomalyJobConfig::JOB_TYPE{"job_type"};
const std::string CAnomalyJobConfig::ANALYSIS_CONFIG{"analysis_config"};
const std::string CAnomalyJobConfig::ANALYSIS_LIMITS{"analysis_limits"};
const std::string CAnomalyJobConfig::DATA_DESCRIPTION{"data_description"};
const std::string CAnomalyJobConfig::MODEL_PLOT_CONFIG{"model_plot_config"};
const std::string CAnomalyJobConfig::FILTERS{"filters"};
const std::string CAnomalyJobConfig::EVENTS{"events"};

const std::string CAnomalyJobConfig::CAnalysisConfig::BUCKET_SPAN{"bucket_span"};
const std::string CAnomalyJobConfig::CAnalysisConfig::SUMMARY_COUNT_FIELD_NAME{
    "summary_count_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CATEGORIZATION_FIELD_NAME{
    "categorization_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CATEGORIZATION_FILTERS{"categorization_filters"};
const std::string CAnomalyJobConfig::CAnalysisConfig::DETECTORS{"detectors"};
const std::string CAnomalyJobConfig::CAnalysisConfig::INFLUENCERS{"influencers"};
const std::string CAnomalyJobConfig::CAnalysisConfig::PER_PARTITION_CATEGORIZATION{
    "per_partition_categorization"};
const std::string CAnomalyJobConfig::CAnalysisConfig::ENABLED{"enabled"};
const std::string CAnomalyJobConfig::CAnalysisConfig::STOP_ON_WARN{"stop_on_warn"};
const std::string CAnomalyJobConfig::CAnalysisConfig::LATENCY{"latency"};
const std::string CAnomalyJobConfig::CAnalysisConfig::MULTIVARIATE_BY_FIELDS{"multivariate_by_fields"};

const core_t::TTime CAnomalyJobConfig::CAnalysisConfig::DEFAULT_BUCKET_SPAN{300};
const core_t::TTime CAnomalyJobConfig::CAnalysisConfig::DEFAULT_LATENCY{0};

const std::string CAnomalyJobConfig::CAnalysisConfig::CLEAR{"clear"};
const char CAnomalyJobConfig::CAnalysisConfig::SUFFIX_SEPARATOR{'.'};
const std::string CAnomalyJobConfig::CAnalysisConfig::SCHEDULED_EVENT_PREFIX("scheduledevent.");
const std::string CAnomalyJobConfig::CAnalysisConfig::DESCRIPTION_SUFFIX(".description");
const std::string CAnomalyJobConfig::CAnalysisConfig::RULES_SUFFIX(".rules");

const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION{"function"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FIELD_NAME{"field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::BY_FIELD_NAME{"by_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::OVER_FIELD_NAME{
    "over_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::PARTITION_FIELD_NAME{
    "partition_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_DESCRIPTION{
    "detector_description"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_INDEX{"detector_index"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::EXCLUDE_FREQUENT{
    "exclude_frequent"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::USE_NULL{"use_null"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::CUSTOM_RULES{"custom_rules"};

const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::ALL_TOKEN("all");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::BY_TOKEN("by");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::NONE_TOKEN("none");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::OVER_TOKEN("over");

// Event rate functions
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_COUNT("count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_COUNT_ABBREV("c");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_COUNT("low_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_COUNT_ABBREV("low_c");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_COUNT("high_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_COUNT_ABBREV("high_c");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_DISTINCT_COUNT("distinct_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_DISTINCT_COUNT_ABBREV("dc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_DISTINCT_COUNT("low_distinct_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_DISTINCT_COUNT_ABBREV("low_dc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_DISTINCT_COUNT("high_distinct_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_DISTINCT_COUNT_ABBREV("high_dc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_NON_ZERO_COUNT("non_zero_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_NON_ZERO_COUNT_ABBREV("nzc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_RARE_NON_ZERO_COUNT("rare_non_zero_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_RARE_NON_ZERO_COUNT_ABBREV("rnzc");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_RARE("rare");
// No abbreviation for "rare" as "r" is a little too obscure
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_RARE_COUNT("rare_count");
// No abbreviation for "rare_count" as "rc" is sometimes used as an abbreviation
// for "return code"
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_FREQ_RARE("freq_rare");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_FREQ_RARE_ABBREV("fr");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_FREQ_RARE_COUNT("freq_rare_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_FREQ_RARE_COUNT_ABBREV("frc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_NON_ZERO_COUNT("low_non_zero_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_NON_ZERO_COUNT_ABBREV("low_nzc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_NON_ZERO_COUNT("high_non_zero_count");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_NON_ZERO_COUNT_ABBREV("high_nzc");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_INFO_CONTENT("info_content");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_INFO_CONTENT("low_info_content");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_INFO_CONTENT("high_info_content");

// Metric functions
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_METRIC("metric");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_AVERAGE("avg");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MEAN("mean");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_MEAN("low_mean");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_MEAN("high_mean");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_AVERAGE("low_avg");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_AVERAGE("high_avg");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MEDIAN("median");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_MEDIAN("low_median");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_MEDIAN("high_median");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MIN("min");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MAX("max");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_VARIANCE("varp");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_VARIANCE("low_varp");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_VARIANCE("high_varp");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_SUM("sum");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_SUM("low_sum");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_SUM("high_sum");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_NON_NULL_SUM("non_null_sum");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_NON_NULL_SUM_ABBREV("nns");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_NON_NULL_SUM("low_non_null_sum");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LOW_NON_NULL_SUM_ABBREV("low_nns");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_NON_NULL_SUM("high_non_null_sum");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_HIGH_NON_NULL_SUM_ABBREV("high_nns");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_TIME_OF_DAY("time_of_day");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_TIME_OF_WEEK("time_of_week");
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_LAT_LONG("lat_long");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MAX_VELOCITY("max_velocity");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MIN_VELOCITY("min_velocity");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_MEAN_VELOCITY("mean_velocity");
const std::string
    CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION_SUM_VELOCITY("sum_velocity");

const std::string CAnomalyJobConfig::CModelPlotConfig::ANNOTATIONS_ENABLED{"annotations_enabled"};
const std::string CAnomalyJobConfig::CModelPlotConfig::ENABLED{"enabled"};
const std::string CAnomalyJobConfig::CModelPlotConfig::TERMS{"terms"};

const std::string CAnomalyJobConfig::CAnalysisLimits::CATEGORIZATION_EXAMPLES_LIMIT{
    "categorization_examples_limit"};
const std::string CAnomalyJobConfig::CAnalysisLimits::MODEL_MEMORY_LIMIT{"model_memory_limit"};

const std::size_t CAnomalyJobConfig::CAnalysisLimits::DEFAULT_MEMORY_LIMIT_BYTES{
    1024ULL * 1024 * 1024};

const std::string CAnomalyJobConfig::CDataDescription::TIME_FIELD{"time_field"};
const std::string CAnomalyJobConfig::CDataDescription::TIME_FORMAT{"time_format"};

const std::string CAnomalyJobConfig::CEventConfig::DESCRIPTION{"description"};
const std::string CAnomalyJobConfig::CEventConfig::RULES{"rules"};

const std::string CAnomalyJobConfig::CFilterConfig::FILTER_ID{"filter_id"};
const std::string CAnomalyJobConfig::CFilterConfig::ITEMS{"items"};

namespace {
const std::string EMPTY_STRING;

std::string toString(const rapidjson::Value& value) {
    rapidjson::StringBuffer strbuf;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    value.Accept(writer);
    return strbuf.GetString();
};

const CAnomalyJobConfigReader FILTERS_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CFilterConfig::FILTER_ID,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::CFilterConfig::ITEMS,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader EVENTS_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CEventConfig::DESCRIPTION,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::CEventConfig::RULES,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::JOB_ID,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::JOB_TYPE,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::ANALYSIS_CONFIG,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::ANALYSIS_LIMITS,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::MODEL_PLOT_CONFIG,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::DATA_DESCRIPTION,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader ANALYSIS_CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::BUCKET_SPAN,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::SUMMARY_COUNT_FIELD_NAME,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CATEGORIZATION_FIELD_NAME,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CATEGORIZATION_FILTERS,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::PER_PARTITION_CATEGORIZATION,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::DETECTORS,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::INFLUENCERS,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::LATENCY,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader DETECTOR_CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FIELD_NAME,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::BY_FIELD_NAME,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::OVER_FIELD_NAME,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::PARTITION_FIELD_NAME,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_DESCRIPTION,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_INDEX,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::EXCLUDE_FREQUENT,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::USE_NULL,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::CUSTOM_RULES,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader PPC_CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::ENABLED,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::STOP_ON_WARN,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader MODEL_PLOT_CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CModelPlotConfig::ANNOTATIONS_ENABLED,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CModelPlotConfig::ENABLED,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CModelPlotConfig::TERMS,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader ANALYSIS_LIMITS_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CAnalysisLimits::CATEGORIZATION_EXAMPLES_LIMIT,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::CAnalysisLimits::MODEL_MEMORY_LIMIT,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader DATA_DESCRIPTION_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CDataDescription::TIME_FIELD,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::CDataDescription::TIME_FORMAT,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    return theReader;
}()};
}

bool CAnomalyJobConfig::initFromFile(const std::string& configFile) {
    std::string anomalyJobConfigJson;
    bool couldReadConfigFile;
    std::tie(anomalyJobConfigJson, couldReadConfigFile) =
        ml::core::CStringUtils::readFileToString(configFile);
    if (couldReadConfigFile == false) {
        LOG_ERROR(<< "Failed to read config file '" << configFile << "'");
        return false;
    }

    if (this->parse(anomalyJobConfigJson) == false) {
        LOG_ERROR(<< "Failed to parse anomaly job config: '" << anomalyJobConfigJson << "'");
        return false;
    }

    return true;
}

bool CAnomalyJobConfig::readFile(const std::string& fileName, std::string& fileContents) {
    bool couldReadFile;
    std::tie(fileContents, couldReadFile) = ml::core::CStringUtils::readFileToString(fileName);
    if (couldReadFile == false) {
        LOG_ERROR(<< "Failed to read file '" << fileName << "'");
        return false;
    }

    return true;
}

bool CAnomalyJobConfig::initFromFiles(const std::string& configFile,
                                      const std::string& filtersConfigFile,
                                      const std::string& eventsConfigFile) {

    std::string filtersConfigJson;
    if (filtersConfigFile.empty() == false &&
        this->readFile(filtersConfigFile, filtersConfigJson)) {
        if (this->parseFilterConfig(filtersConfigJson) == false) {
            LOG_ERROR(<< "Failed to parse filters job config: '" << filtersConfigJson << "'");
            return false;
        }
    }

    std::string eventsConfigJson;
    if (eventsConfigFile.empty() == false &&
        this->readFile(eventsConfigFile, eventsConfigJson)) {
        if (this->parseEventConfig(eventsConfigJson) == false) {
            LOG_ERROR(<< "Failed to parse scheduled events config: '"
                      << eventsConfigJson << "'");
            return false;
        }
    }

    m_AnalysisConfig.init(m_RuleFilters, m_ScheduledEvents);

    std::string anomalyJobConfigJson;
    if (this->readFile(configFile, anomalyJobConfigJson) == false) {
        // error logged by readFile
        return false;
    }

    if (this->parse(anomalyJobConfigJson) == false) {
        LOG_ERROR(<< "Failed to parse anomaly job config: '" << anomalyJobConfigJson << "'");
        return false;
    }

    return true;
}

bool CAnomalyJobConfig::parseEventConfig(const std::string& json) {
    rapidjson::Document doc;
    if (doc.Parse<0>(json).HasParseError()) {
        LOG_ERROR(<< "An error occurred while parsing scheduled event config from JSON: "
                  << doc.GetParseError());
        return false;
    }

    if (doc.Empty()) {
        return true;
    }

    try {
        if (doc.HasMember(EVENTS) == false || doc[EVENTS].IsArray() == false) {
            LOG_ERROR(<< "Missing expected array field '" << EVENTS << "'. JSON: " << json);
            return false;
        }

        const rapidjson::Value& value = doc[EVENTS];

        m_Events.resize(value.Size());
        for (unsigned int i = 0; i < value.Size(); ++i) {
            if (value[i].IsObject() == false) {
                LOG_ERROR(<< "Could not parse scheduled events: expected events array to contain objects. JSON: "
                          << json);
                return false;
            }

            m_Events[i].parse(value[i], m_RuleFilters, m_ScheduledEvents);
        }

    } catch (CAnomalyJobConfigReader::CParseError& e) {
        LOG_ERROR(<< "Error parsing events config: " << e.what());
        return false;
    }

    return true;
}

void CAnomalyJobConfig::CEventConfig::parse(const rapidjson::Value& filterConfig,
                                            const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
                                            TStrDetectionRulePrVec& scheduledEvents) {
    auto parameters = EVENTS_READER.read(filterConfig);

    m_Description = parameters[DESCRIPTION].as<std::string>();

    auto eventRules = parameters[RULES].jsonObject();
    if (eventRules != nullptr) {
        std::string errorString;
        CDetectionRulesJsonParser rulesParser(ruleFilters);
        if (rulesParser.parseRules(*eventRules, m_DetectionRules, errorString) == false) {
            LOG_ERROR(<< errorString << toString(*eventRules));
            throw CAnomalyJobConfigReader::CParseError(
                "Error parsing scheduled event rules: " + toString(*eventRules));
        }
    }

    if (m_DetectionRules.size() != 1) {
        throw CAnomalyJobConfigReader::CParseError(
            "Scheduled events must have exactly 1 rule: " + toString(*eventRules));
    }

    scheduledEvents.emplace_back(m_Description, m_DetectionRules[0]);
}

bool CAnomalyJobConfig::parseFilterConfig(const std::string& json) {
    rapidjson::Document doc;
    if (doc.Parse<0>(json).HasParseError()) {
        LOG_ERROR(<< "An error occurred while parsing filter config from JSON: "
                  << doc.GetParseError());
        return false;
    }

    if (doc.Empty()) {
        return true;
    }

    try {
        if (doc.HasMember(FILTERS) == false || doc[FILTERS].IsArray() == false) {
            LOG_ERROR(<< "Missing expected array field '" << FILTERS << "'. JSON: " << json);
            return false;
        }

        const rapidjson::Value& value = doc[FILTERS];
        m_Filters.resize(value.Size());
        for (unsigned int i = 0; i < value.Size(); ++i) {
            if (value[i].IsObject() == false) {
                LOG_ERROR(<< "Could not parse filters: expected filters array to contain objects. JSON: "
                          << toString(value[i]));
                return false;
            }

            m_Filters[i].parse(value[i], m_RuleFilters);
        }
    } catch (CAnomalyJobConfigReader::CParseError& e) {
        LOG_ERROR(<< "Error parsing filter config: " << e.what());
        return false;
    }

    return true;
}

void CAnomalyJobConfig::CFilterConfig::parse(const rapidjson::Value& filterConfig,
                                             CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters) {
    auto parameters = FILTERS_READER.read(filterConfig);

    m_FilterName = parameters[FILTER_ID].as<std::string>();
    m_FilterList = parameters[ITEMS].fallback(TStrVec{});

    core::CPatternSet& filter = ruleFilters[m_FilterName];
    if (filter.initFromPatternList(m_FilterList) == false) {
        throw CAnomalyJobConfigReader::CParseError("Error building filter rules: " +
                                                   toString(filterConfig));
    }
}

bool CAnomalyJobConfig::parse(const std::string& json) {
    rapidjson::Document doc;
    if (doc.Parse<0>(json).HasParseError()) {
        LOG_ERROR(<< "An error occurred while parsing anomaly job config from JSON: "
                  << doc.GetParseError());
        return false;
    }

    try {
        auto parameters = CONFIG_READER.read(doc);

        m_JobId = parameters[JOB_ID].as<std::string>();
        m_JobType = parameters[JOB_TYPE].fallback(EMPTY_STRING);

        auto analysisConfig = parameters[ANALYSIS_CONFIG].jsonObject();
        if (analysisConfig != nullptr) {
            m_AnalysisConfig.parse(*analysisConfig);
        }

        auto analysisLimits = parameters[ANALYSIS_LIMITS].jsonObject();
        if (analysisLimits != nullptr) {
            m_AnalysisLimits.parse(*analysisLimits);
        }

        auto description = parameters[DATA_DESCRIPTION].jsonObject();
        if (description != nullptr) {
            m_DataDescription.parse(*description);
        }

        auto modelPlotConfig = parameters[MODEL_PLOT_CONFIG].jsonObject();
        if (modelPlotConfig != nullptr) {
            m_ModelConfig.parse(*modelPlotConfig);
        }
    } catch (CAnomalyJobConfigReader::CParseError& e) {
        LOG_ERROR(<< "Error parsing anomaly job config: " << e.what());
        return false;
    }

    m_IsInitialized = true;

    return true;
}

void CAnomalyJobConfig::CModelPlotConfig::parse(const rapidjson::Value& modelPlotConfig) {
    auto parameters = MODEL_PLOT_CONFIG_READER.read(modelPlotConfig);

    m_AnnotationsEnabled = parameters[ANNOTATIONS_ENABLED].fallback(false);
    m_Enabled = parameters[ENABLED].fallback(false);
    m_Terms = parameters[TERMS].fallback(EMPTY_STRING);
}

void CAnomalyJobConfig::CAnalysisLimits::parse(const rapidjson::Value& analysisLimits) {
    auto parameters = ANALYSIS_LIMITS_READER.read(analysisLimits);

    m_CategorizationExamplesLimit = parameters[CATEGORIZATION_EXAMPLES_LIMIT].fallback(
        model::CLimits::DEFAULT_RESULTS_MAX_EXAMPLES);

    const std::string memoryLimitStr{parameters[MODEL_MEMORY_LIMIT].as<std::string>()};
    m_ModelMemoryLimitMb = CAnomalyJobConfig::CAnalysisLimits::modelMemoryLimitMb(memoryLimitStr);
}

std::size_t CAnomalyJobConfig::CAnalysisLimits::modelMemoryLimitMb(const std::string& memoryLimitStr) {
    // We choose to ignore any errors here parsing the model memory limit string
    // as we assume that it has already been validated by ES. In the event that any
    // error _does_ occur an error is logged and a default value used.
    std::size_t memoryLimitBytes{0};
    std::tie(memoryLimitBytes, std::ignore) = core::CStringUtils::memorySizeStringToBytes(
        memoryLimitStr, DEFAULT_MEMORY_LIMIT_BYTES);

    std::size_t memoryLimitMb{memoryLimitBytes / core::constants::BYTES_IN_MEGABYTES};

    if (memoryLimitMb == 0) {
        LOG_ERROR(<< "Invalid limit value " << memoryLimitStr << ". Limit must have a minimum value of 1mb."
                  << " Using default memory limit value "
                  << DEFAULT_MEMORY_LIMIT_BYTES / core::constants::BYTES_IN_MEGABYTES);
        memoryLimitMb = DEFAULT_MEMORY_LIMIT_BYTES / core::constants::BYTES_IN_MEGABYTES;
    }

    return memoryLimitMb;
}

void CAnomalyJobConfig::CDataDescription::parse(const rapidjson::Value& analysisLimits) {
    auto parameters = DATA_DESCRIPTION_READER.read(analysisLimits);

    m_TimeField = parameters[TIME_FIELD].as<std::string>();
    m_TimeFormat = parameters[TIME_FORMAT].fallback(EMPTY_STRING); // Ignore
}

void CAnomalyJobConfig::CAnalysisConfig::parse(const rapidjson::Value& analysisConfig) {
    auto parameters = ANALYSIS_CONFIG_READER.read(analysisConfig);
    // We choose to ignore any errors here parsing the time duration string as
    // we assume that it has already been validated by ES. In the event that any
    // error _does_ occur an error is logged and a default value used.
    const std::string& bucketSpanString{parameters[BUCKET_SPAN].fallback(EMPTY_STRING)};
    m_BucketSpan = CAnomalyJobConfig::CAnalysisConfig::durationSeconds(
        bucketSpanString, DEFAULT_BUCKET_SPAN);

    m_SummaryCountFieldName = parameters[SUMMARY_COUNT_FIELD_NAME].fallback(EMPTY_STRING);
    m_CategorizationFieldName = parameters[CATEGORIZATION_FIELD_NAME].fallback(EMPTY_STRING);
    m_CategorizationFilters = parameters[CATEGORIZATION_FILTERS].fallback(TStrVec{});

    auto ppc = parameters[PER_PARTITION_CATEGORIZATION].jsonObject();
    if (ppc != nullptr) {
        auto ppcParameters = PPC_CONFIG_READER.read(*ppc);
        m_PerPartitionCategorizationEnabled = ppcParameters[ENABLED].fallback(false);
        m_PerPartitionCategorizationStopOnWarn = ppcParameters[STOP_ON_WARN].fallback(false);
    }

    auto detectorsConfig = parameters[DETECTORS].jsonObject();

    // The Job config has already been validated by Java before being passed to
    // the C++ backend. So we can safely assume that the detector config is a
    // non-null array - hence this check isn't strictly necessary.
    if (detectorsConfig != nullptr && detectorsConfig->IsArray()) {
        m_Detectors.resize(detectorsConfig->Size());
        int fallbackDetectorIndex{0};
        for (std::size_t i = 0; i < detectorsConfig->Size(); ++i) {
            m_Detectors[i].parse((*detectorsConfig)[fallbackDetectorIndex], m_RuleFilters,
                                 (m_SummaryCountFieldName.empty() == false), fallbackDetectorIndex,
                                 m_DetectorRules[fallbackDetectorIndex]);

            if (m_PerPartitionCategorizationEnabled) {
                if (m_CategorizationFieldName.empty()) {
                    throw CAnomalyJobConfigReader::CParseError(
                        "per_partition_categorization enabled without a categorization field");
                }
                if (m_Detectors[i].partitionFieldName().empty()) {
                    throw CAnomalyJobConfigReader::CParseError(
                        "per_partition_categorization enabled without a partition field");
                }
                if (m_CategorizationPartitionFieldName.empty()) {
                    m_CategorizationPartitionFieldName = m_Detectors[i].partitionFieldName();
                } else {
                    if (m_CategorizationPartitionFieldName !=
                        m_Detectors[i].partitionFieldName()) {
                        throw CAnomalyJobConfigReader::CParseError(
                            "per_partition_categorization enabled when partition "
                            "field varies between detectors");
                    }
                }
            }
            ++fallbackDetectorIndex;
        }
    }

    m_Influencers = parameters[INFLUENCERS].fallback(TStrVec{});

    const std::string& latencyString{parameters[LATENCY].fallback(EMPTY_STRING)};
    if (latencyString.empty() == false) {
        m_Latency = CAnomalyJobConfig::CAnalysisConfig::durationSeconds(
            latencyString, DEFAULT_LATENCY);
    }

    m_MultivariateByFields = parameters[MULTIVARIATE_BY_FIELDS].fallback(false);
}

// TODO: Process updates as JSON
bool CAnomalyJobConfig::CAnalysisConfig::updateFilters(const boost::property_tree::ptree& propTree) {
    for (const auto& filterEntry : propTree) {
        const std::string& key = filterEntry.first;
        const std::string& value = filterEntry.second.data();
        if (this->processFilter(key, value) == false) {
            return false;
        }
    }
    return true;
}

// TODO: Process updates as JSON
bool CAnomalyJobConfig::CAnalysisConfig::processFilter(const std::string& key,
                                                       const std::string& value) {
    // expected format is filter.<filterId>=[json, array]
    std::size_t sepPos{key.find(SUFFIX_SEPARATOR)};
    if (sepPos == std::string::npos) {
        LOG_ERROR(<< "Unrecognised filter key: " + key);
        return false;
    }
    std::string filterId = key.substr(sepPos + 1);
    core::CPatternSet& filter = m_RuleFilters[filterId];
    return filter.initFromJson(value);
}

// TODO: Process updates as JSON
bool CAnomalyJobConfig::CAnalysisConfig::updateScheduledEvents(const boost::property_tree::ptree& propTree) {
    m_ScheduledEvents.clear();

    bool isClear = propTree.get(CLEAR, false);
    if (isClear) {
        return true;
    }

    TIntSet handledScheduledEvents;

    for (const auto& scheduledEventEntry : propTree) {
        const std::string& key = scheduledEventEntry.first;
        const std::string& value = scheduledEventEntry.second.data();
        if (this->processScheduledEvent(propTree, key, value, handledScheduledEvents) == false) {
            return false;
        }
    }
    return true;
}

// TODO: Process updates as JSON
bool CAnomalyJobConfig::CAnalysisConfig::processScheduledEvent(
    const boost::property_tree::ptree& propTree,
    const std::string& key,
    const std::string& value,
    TIntSet& handledScheduledEvents) {
    // Here we pull out the "1" in "scheduledevent.1.description"
    // description may contain a '.'
    std::size_t sepPos{key.find(SUFFIX_SEPARATOR, SCHEDULED_EVENT_PREFIX.length() + 1)};
    if (sepPos == std::string::npos || sepPos == key.length() - 1) {
        LOG_ERROR(<< "Unrecognised configuration option " << key << " = " << value);
        return false;
    }

    std::string indexString{key, SCHEDULED_EVENT_PREFIX.length(),
                            sepPos - SCHEDULED_EVENT_PREFIX.length()};
    int indexKey;
    if (core::CStringUtils::stringToType(indexString, indexKey) == false) {
        LOG_ERROR(<< "Cannot convert config key to integer: " << indexString);
        return false;
    }

    // Check if we've already seen this key
    if (handledScheduledEvents.insert(indexKey).second == false) {
        // Not an error
        return true;
    }

    std::string description{propTree.get(
        boost::property_tree::ptree::path_type{
            SCHEDULED_EVENT_PREFIX + indexString + DESCRIPTION_SUFFIX, '\t'},
        EMPTY_STRING)};

    std::string rules{propTree.get(
        boost::property_tree::ptree::path_type{
            SCHEDULED_EVENT_PREFIX + indexString + RULES_SUFFIX, '\t'},
        EMPTY_STRING)};

    CDetectionRulesJsonParser::TDetectionRuleVec detectionRules;
    if (this->parseRules(detectionRules, rules) == false) {
        // parseRules() will have logged the error
        return false;
    }

    if (detectionRules.size() != 1) {
        LOG_ERROR(<< "Scheduled events must have exactly 1 rule");
        return false;
    }

    m_ScheduledEvents.emplace_back(description, detectionRules[0]);

    return true;
}

// TODO: Process updates as JSON
bool CAnomalyJobConfig::CAnalysisConfig::parseRules(int detectorIndex,
                                                    const std::string& rules) {
    return parseRules(m_DetectorRules[detectorIndex], rules);
}

// TODO: Process updates as JSON
bool CAnomalyJobConfig::CAnalysisConfig::parseRules(CDetectionRulesJsonParser::TDetectionRuleVec& detectionRules,
                                                    const std::string& rules) {
    if (rules.empty()) {
        return true;
    }

    CDetectionRulesJsonParser rulesParser{m_RuleFilters};
    return rulesParser.parseRules(rules, detectionRules);
}

ml::model::CAnomalyDetectorModelConfig
CAnomalyJobConfig::CAnalysisConfig::makeModelConfig() const {
    model_t::ESummaryMode summaryMode{
        m_SummaryCountFieldName.empty() ? model_t::E_None : model_t::E_Manual};

    model::CAnomalyDetectorModelConfig modelConfig{model::CAnomalyDetectorModelConfig::defaultConfig(
        m_BucketSpan, summaryMode, m_SummaryCountFieldName, m_Latency, m_MultivariateByFields)};

    modelConfig.detectionRules(
        model::CAnomalyDetectorModelConfig::TIntDetectionRuleVecUMapCRef(m_DetectorRules));

    modelConfig.scheduledEvents(
        model::CAnomalyDetectorModelConfig::TStrDetectionRulePrVecCRef(m_ScheduledEvents));

    return modelConfig;
}

core_t::TTime
CAnomalyJobConfig::CAnalysisConfig::durationSeconds(const std::string& durationString,
                                                    core_t::TTime defaultDuration) {
    core_t::TTime durationSeconds{0};
    std::tie(durationSeconds, std::ignore) =
        core::CTimeUtils::timeDurationStringToSeconds(durationString, defaultDuration);

    if (durationSeconds == 0) {
        LOG_ERROR(<< "Invalid duration value " << durationString
                  << ". Duration must have a minimum value of 1s. "
                     "Using default duration value "
                  << defaultDuration);
        durationSeconds = defaultDuration;
    }

    return durationSeconds;
}

void CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::parse(
    const rapidjson::Value& detectorConfig,
    const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters,
    bool haveSummaryCountField,
    int fallbackDetectorIndex,
    CDetectionRulesJsonParser::TDetectionRuleVec& detectionRules) {

    auto parameters = DETECTOR_CONFIG_READER.read(detectorConfig);

    m_FunctionName = parameters[FUNCTION].as<std::string>();
    m_FieldName = parameters[FIELD_NAME].fallback(EMPTY_STRING);
    m_ByFieldName = parameters[BY_FIELD_NAME].fallback(EMPTY_STRING);
    m_OverFieldName = parameters[OVER_FIELD_NAME].fallback(EMPTY_STRING);
    m_ByFieldName = parameters[BY_FIELD_NAME].fallback(EMPTY_STRING);
    m_PartitionFieldName = parameters[PARTITION_FIELD_NAME].fallback(EMPTY_STRING);
    m_ExcludeFrequent = parameters[EXCLUDE_FREQUENT].fallback(EMPTY_STRING);
    m_DetectorDescription = parameters[DETECTOR_DESCRIPTION].fallback(EMPTY_STRING);

    // The detector index is of type int for historical reasons
    // and for consistency across the code base.
    m_DetectorIndex = parameters[DETECTOR_INDEX].fallback(fallbackDetectorIndex);

    auto customRules = parameters[CUSTOM_RULES].jsonObject();
    if (customRules != nullptr) {
        std::string errorString;
        CDetectionRulesJsonParser rulesParser(ruleFilters);
        if (rulesParser.parseRules(*customRules, detectionRules, errorString) == false) {
            LOG_ERROR(<< errorString << toString(*customRules));
            throw CAnomalyJobConfigReader::CParseError(
                "Error parsing custom rules: " + toString(*customRules));
        }
    }

    m_UseNull = parameters[USE_NULL].fallback(false);

    if (this->determineFunction(haveSummaryCountField) == false) {
        throw CAnomalyJobConfigReader::CParseError("Error determining function");
    }
    if (this->decipherExcludeFrequentSetting() == false) {
        throw CAnomalyJobConfigReader::CParseError("Error deciphering exclude frequent setting");
    }
}

bool CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::determineFunction(bool haveSummaryCountField) {

    bool isPopulation{m_OverFieldName.empty() == false};
    bool hasByField{m_ByFieldName.empty() == false};

    // Some functions must take a field, some mustn't and for the rest it's
    // optional.  Validate this based on the contents of these flags after
    // determining the function.  Similarly for by fields.
    // TODO: Check how much validation is required here (if any) if parsing JSON job config.
    bool fieldRequired{false};
    bool fieldInvalid{false};
    bool byFieldRequired{false};
    bool byFieldInvalid{false};

    if (m_FunctionName.empty()) {
        LOG_ERROR(<< "No function specified");
        return false;
    }

    if (m_FunctionName == FUNCTION_COUNT || m_FunctionName == FUNCTION_COUNT_ABBREV) {
        m_Function = isPopulation ? model::function_t::E_PopulationCount
                                  : model::function_t::E_IndividualRareCount;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_DISTINCT_COUNT ||
               m_FunctionName == FUNCTION_DISTINCT_COUNT_ABBREV) {
        m_Function = isPopulation ? model::function_t::E_PopulationDistinctCount
                                  : model::function_t::E_IndividualDistinctCount;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_DISTINCT_COUNT ||
               m_FunctionName == FUNCTION_LOW_DISTINCT_COUNT_ABBREV) {
        m_Function = isPopulation ? model::function_t::E_PopulationLowDistinctCount
                                  : model::function_t::E_IndividualLowDistinctCount;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_DISTINCT_COUNT ||
               m_FunctionName == FUNCTION_HIGH_DISTINCT_COUNT_ABBREV) {
        m_Function = isPopulation ? model::function_t::E_PopulationHighDistinctCount
                                  : model::function_t::E_IndividualHighDistinctCount;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_NON_ZERO_COUNT ||
               m_FunctionName == FUNCTION_NON_ZERO_COUNT_ABBREV) {
        m_Function = model::function_t::E_IndividualNonZeroCount;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_RARE_NON_ZERO_COUNT ||
               m_FunctionName == FUNCTION_RARE_NON_ZERO_COUNT_ABBREV) {
        m_Function = model::function_t::E_IndividualRareNonZeroCount;
        fieldInvalid = true;
        byFieldRequired = true;
    } else if (m_FunctionName == FUNCTION_RARE) {
        m_Function = isPopulation ? model::function_t::E_PopulationRare
                                  : model::function_t::E_IndividualRare;
        fieldInvalid = true;
        byFieldRequired = true;
    } else if (m_FunctionName == FUNCTION_RARE_COUNT) {
        m_Function = model::function_t::E_PopulationRareCount;
        fieldInvalid = true;
        byFieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_COUNT || m_FunctionName == FUNCTION_LOW_COUNT_ABBREV) {
        m_Function = isPopulation ? model::function_t::E_PopulationLowCounts
                                  : model::function_t::E_IndividualLowCounts;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_HIGH_COUNT ||
               m_FunctionName == FUNCTION_HIGH_COUNT_ABBREV) {
        m_Function = isPopulation ? model::function_t::E_PopulationHighCounts
                                  : model::function_t::E_IndividualHighCounts;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_LOW_NON_ZERO_COUNT ||
               m_FunctionName == FUNCTION_LOW_NON_ZERO_COUNT_ABBREV) {
        m_Function = model::function_t::E_IndividualLowNonZeroCount;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_HIGH_NON_ZERO_COUNT ||
               m_FunctionName == FUNCTION_HIGH_NON_ZERO_COUNT_ABBREV) {
        m_Function = model::function_t::E_IndividualHighNonZeroCount;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_FREQ_RARE || m_FunctionName == FUNCTION_FREQ_RARE_ABBREV) {
        m_Function = model::function_t::E_PopulationFreqRare;
        fieldInvalid = true;
        byFieldRequired = true;
    } else if (m_FunctionName == FUNCTION_FREQ_RARE_COUNT ||
               m_FunctionName == FUNCTION_FREQ_RARE_COUNT_ABBREV) {
        m_Function = model::function_t::E_PopulationFreqRareCount;
        fieldInvalid = true;
        byFieldRequired = true;
    } else if (m_FunctionName == FUNCTION_INFO_CONTENT) {
        m_Function = isPopulation ? model::function_t::E_PopulationInfoContent
                                  : model::function_t::E_IndividualInfoContent;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_INFO_CONTENT) {
        m_Function = isPopulation ? model::function_t::E_PopulationLowInfoContent
                                  : model::function_t::E_IndividualLowInfoContent;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_INFO_CONTENT) {
        m_Function = isPopulation ? model::function_t::E_PopulationHighInfoContent
                                  : model::function_t::E_IndividualHighInfoContent;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_METRIC) {
        if (haveSummaryCountField) {
            LOG_ERROR(<< "Function " << m_FunctionName
                      << "() cannot be used with a summary count field");
            return false;
        }

        m_Function = isPopulation ? model::function_t::E_PopulationMetric
                                  : model::function_t::E_IndividualMetric;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_AVERAGE || m_FunctionName == FUNCTION_MEAN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricMean
                                  : model::function_t::E_IndividualMetricMean;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_AVERAGE || m_FunctionName == FUNCTION_LOW_MEAN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricLowMean
                                  : model::function_t::E_IndividualMetricLowMean;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_AVERAGE || m_FunctionName == FUNCTION_HIGH_MEAN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricHighMean
                                  : model::function_t::E_IndividualMetricHighMean;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_MEDIAN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricMedian
                                  : model::function_t::E_IndividualMetricMedian;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_MEDIAN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricLowMedian
                                  : model::function_t::E_IndividualMetricLowMedian;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_MEDIAN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricHighMedian
                                  : model::function_t::E_IndividualMetricHighMedian;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_MIN) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricMin
                                  : model::function_t::E_IndividualMetricMin;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_MAX) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricMax
                                  : model::function_t::E_IndividualMetricMax;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_VARIANCE) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricVariance
                                  : model::function_t::E_IndividualMetricVariance;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_VARIANCE) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricLowVariance
                                  : model::function_t::E_IndividualMetricLowVariance;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_VARIANCE) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricHighVariance
                                  : model::function_t::E_IndividualMetricHighVariance;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_SUM) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricSum
                                  : model::function_t::E_IndividualMetricSum;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_SUM) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricLowSum
                                  : model::function_t::E_IndividualMetricLowSum;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_SUM) {
        m_Function = isPopulation ? model::function_t::E_PopulationMetricHighSum
                                  : model::function_t::E_IndividualMetricHighSum;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_NON_NULL_SUM ||
               m_FunctionName == FUNCTION_NON_NULL_SUM_ABBREV) {
        m_Function = model::function_t::E_IndividualMetricNonNullSum;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_LOW_NON_NULL_SUM ||
               m_FunctionName == FUNCTION_LOW_NON_NULL_SUM_ABBREV) {
        m_Function = model::function_t::E_IndividualMetricLowNonNullSum;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_HIGH_NON_NULL_SUM ||
               m_FunctionName == FUNCTION_HIGH_NON_NULL_SUM_ABBREV) {
        m_Function = model::function_t::E_IndividualMetricHighNonNullSum;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_TIME_OF_DAY) {
        m_Function = isPopulation ? model::function_t::E_PopulationTimeOfDay
                                  : model::function_t::E_IndividualTimeOfDay;
        fieldRequired = false;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_TIME_OF_WEEK) {
        m_Function = isPopulation ? model::function_t::E_PopulationTimeOfWeek
                                  : model::function_t::E_IndividualTimeOfWeek;
        fieldRequired = false;
        fieldInvalid = true;
    } else if (m_FunctionName == FUNCTION_LAT_LONG) {
        m_Function = isPopulation ? model::function_t::E_PopulationLatLong
                                  : model::function_t::E_IndividualLatLong;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_MAX_VELOCITY) {
        m_Function = isPopulation ? model::function_t::E_PopulationMaxVelocity
                                  : model::function_t::E_IndividualMaxVelocity;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_MIN_VELOCITY) {
        m_Function = isPopulation ? model::function_t::E_PopulationMinVelocity
                                  : model::function_t::E_IndividualMinVelocity;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_MEAN_VELOCITY) {
        m_Function = isPopulation ? model::function_t::E_PopulationMeanVelocity
                                  : model::function_t::E_IndividualMeanVelocity;
        fieldRequired = true;
    } else if (m_FunctionName == FUNCTION_SUM_VELOCITY) {
        m_Function = isPopulation ? model::function_t::E_PopulationSumVelocity
                                  : model::function_t::E_IndividualSumVelocity;
        fieldRequired = true;
    } else {
        LOG_ERROR(<< "Invalid function " << m_FunctionName << " specified");
        return false;
    }

    // Validate
    if (model::function_t::isPopulation(m_Function) && isPopulation == false) {
        LOG_ERROR(<< "Function " << m_FunctionName << " requires an 'over' field");
        return false;
    }

    if (isPopulation && model::function_t::isPopulation(m_Function) == false) {
        LOG_ERROR(<< "Function " << m_FunctionName << " cannot be used with an 'over' field");
        return false;
    }

    if (byFieldRequired && hasByField == false) {
        LOG_ERROR(<< "Function " << m_FunctionName << " requires a 'by' field");
        return false;
    }

    if (byFieldInvalid && hasByField) {
        LOG_ERROR(<< "Function " << m_FunctionName << " cannot be used with a 'by' field");
        return false;
    }

    if (fieldRequired && m_FieldName.empty()) {
        LOG_ERROR(<< "Function " << m_FunctionName << " requires a field");
        return false;
    }

    if (fieldInvalid && m_FieldName.empty() == false) {
        LOG_ERROR(<< "Function " << m_FunctionName << " does not work on a field");
        return false;
    }

    return true;
}

model_t::EExcludeFrequent
CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::excludeFrequent() const {
    if (m_OverHasExcludeFrequent) {
        if (m_ByHasExcludeFrequent) {
            return model_t::E_XF_Both;
        } else {
            return model_t::E_XF_Over;
        }
    } else {
        if (m_ByHasExcludeFrequent) {
            return model_t::E_XF_By;
        }
    }
    return model_t::E_XF_None;
}

bool CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::decipherExcludeFrequentSetting() {

    bool hasByField{m_ByFieldName.empty() == false};
    bool isPopulation{m_OverFieldName.empty() == false};

    if (m_ExcludeFrequent.empty() == false) {
        if (m_ExcludeFrequent == ALL_TOKEN) {
            m_ByHasExcludeFrequent = hasByField;
            m_OverHasExcludeFrequent = isPopulation;
        } else if (m_ExcludeFrequent == BY_TOKEN) {
            m_ByHasExcludeFrequent = hasByField;
        } else if (m_ExcludeFrequent == OVER_TOKEN) {
            m_OverHasExcludeFrequent = isPopulation;
        } else {
            if (m_ExcludeFrequent != NONE_TOKEN) {
                LOG_ERROR(<< "Unexpected exclude_frequent value = " << m_ExcludeFrequent);
                return false;
            }
        }
    }
    return true;
}
}
}
