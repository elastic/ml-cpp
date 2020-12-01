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

#include <model/CLimits.h>

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

const core_t::TTime CAnomalyJobConfig::CAnalysisConfig::DEFAULT_BUCKET_SPAN{300};

const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FUNCTION{"function"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::FIELD_NAME{"field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::BY_FIELD_NAME{"by_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::OVER_FIELD_NAME{
    "over_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::PARTITION_FIELD_NAME{
    "partition_field_name"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_DESCRIPTION{
    "detector_description"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::EXCLUDE_FREQUENT{
    "exclude_frequent"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::USE_NULL{"use_null"};
const std::string CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::CUSTOM_RULES{"custom_rules"};

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

namespace {
const std::string EMPTY_STRING;

std::string toString(const rapidjson::Value& value) {
    rapidjson::StringBuffer strbuf;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    value.Accept(writer);
    return strbuf.GetString();
};

const CAnomalyJobConfigReader CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::JOB_ID,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::JOB_TYPE,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::ANALYSIS_CONFIG,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    theReader.addParameter(CAnomalyJobConfig::ANALYSIS_LIMITS,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::MODEL_PLOT_CONFIG,
                           CAnomalyJobConfigReader::E_OptionalParameter);
    theReader.addParameter(CAnomalyJobConfig::DATA_DESCRIPTION,
                           CAnomalyJobConfigReader::E_RequiredParameter);
    return theReader;
}()};

const CAnomalyJobConfigReader ANALYSIS_CONFIG_READER{[] {
    CAnomalyJobConfigReader theReader;
    theReader.addParameter(CAnomalyJobConfig::CAnalysisConfig::BUCKET_SPAN,
                           CAnomalyJobConfigReader::E_RequiredParameter);
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

bool CAnomalyJobConfig::parse(const std::string& json) {
    LOG_DEBUG(<< "Parsing anomaly job config");

    rapidjson::Document doc;
    if (doc.Parse<0>(json).HasParseError()) {
        LOG_ERROR(<< "An error occurred while parsing anomaly job config from JSON: "
                  << doc.GetParseError());
        return false;
    }

    try {
        auto parameters = CONFIG_READER.read(doc);

        m_JobId = parameters[JOB_ID].as<std::string>();
        m_JobType = parameters[JOB_TYPE].as<std::string>();

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

    std::size_t memoryLimitMb = memoryLimitBytes / core::constants::BYTES_IN_MEGABYTE;

    if (memoryLimitMb == 0) {
        LOG_ERROR(<< "Invalid limit value " << memoryLimitStr << ". Limit must have a minimum value of 1mb."
                  << " Using default memory limit value "
                  << DEFAULT_MEMORY_LIMIT_BYTES / core::constants::BYTES_IN_MEGABYTE);
        memoryLimitMb = DEFAULT_MEMORY_LIMIT_BYTES / core::constants::BYTES_IN_MEGABYTE;
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
    const std::string bucketSpanString{parameters[BUCKET_SPAN].as<std::string>()};
    m_BucketSpan = CAnomalyJobConfig::CAnalysisConfig::bucketSpanSeconds(bucketSpanString);

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
        for (std::size_t i = 0; i < detectorsConfig->Size(); ++i) {
            m_Detectors[i].parse((*detectorsConfig)[static_cast<int>(i)], m_RuleFilters);
        }
    }

    m_Influencers = parameters[INFLUENCERS].fallback(TStrVec{});
    m_Latency = parameters[LATENCY].fallback(EMPTY_STRING);
}

core_t::TTime CAnomalyJobConfig::CAnalysisConfig::bucketSpanSeconds(const std::string& bucketSpanString) {
    core_t::TTime bucketSpanSeconds{0};
    std::tie(bucketSpanSeconds, std::ignore) =
        core::CTimeUtils::timeDurationStringToSeconds(bucketSpanString, DEFAULT_BUCKET_SPAN);

    if (bucketSpanSeconds == 0) {
        LOG_ERROR(<< "Invalid bucket span value " << bucketSpanString
                  << ". Duration must have a minimum value of 1s. "
                     "Using default bucket span value "
                  << DEFAULT_BUCKET_SPAN);
        bucketSpanSeconds = DEFAULT_BUCKET_SPAN;
    }

    return bucketSpanSeconds;
}

void CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::parse(
    const rapidjson::Value& detectorConfig,
    const CDetectionRulesJsonParser::TStrPatternSetUMap& ruleFilters) {
    auto parameters = DETECTOR_CONFIG_READER.read(detectorConfig);

    m_Function = parameters[FUNCTION].as<std::string>();
    m_FieldName = parameters[FIELD_NAME].fallback(EMPTY_STRING);
    m_ByFieldName = parameters[BY_FIELD_NAME].fallback(EMPTY_STRING);
    m_OverFieldName = parameters[OVER_FIELD_NAME].fallback(EMPTY_STRING);
    m_ByFieldName = parameters[BY_FIELD_NAME].fallback(EMPTY_STRING);
    m_PartitionFieldName = parameters[PARTITION_FIELD_NAME].fallback(EMPTY_STRING);
    m_ExcludeFrequent = parameters[EXCLUDE_FREQUENT].fallback(EMPTY_STRING);
    m_DetectorDescription = parameters[DETECTOR_DESCRIPTION].fallback(EMPTY_STRING);

    auto customRules = parameters[CUSTOM_RULES].jsonObject();
    if (customRules != nullptr) {
        std::string errorString;
        CDetectionRulesJsonParser rulesParser(ruleFilters);
        if (rulesParser.parseRules(*customRules, m_CustomRules, errorString) == false) {
            LOG_ERROR(<< errorString << toString(*customRules));
            throw CAnomalyJobConfigReader::CParseError(
                "Error parsing custom rules: " + toString(*customRules));
        }
    }

    m_UseNull = parameters[USE_NULL].fallback(false);
}
}
}
