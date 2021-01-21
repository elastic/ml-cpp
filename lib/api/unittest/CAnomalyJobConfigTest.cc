/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <api/CAnomalyJobConfig.h>

#include <model/FunctionTypes.h>

#include <boost/test/unit_test.hpp>

#include <rapidjson/stringbuffer.h>

namespace {
const std::string EMPTY_STRING;
}

BOOST_AUTO_TEST_SUITE(CAnomalyJobConfigTest)

BOOST_AUTO_TEST_CASE(testIntervalStagger) {
    const std::string job1ConfigJson{
        "{\"job_id\":\"job1\",\"description\":\"job number one\", \"analysis_config\":{\"detectors\":[]}}"};

    ml::api::CAnomalyJobConfig job1Config;
    BOOST_TEST_REQUIRE(job1Config.parse(job1ConfigJson));
    BOOST_TEST_REQUIRE(job1Config.isInitialized());
    BOOST_REQUIRE_LE(0, job1Config.intervalStagger());
    BOOST_REQUIRE_GE(3599, job1Config.intervalStagger());

    const std::string job2ConfigJson{
        "{\"job_id\":\"job2\",\"description\":\"job number two\", \"analysis_config\":{\"detectors\":[]}}"};

    ml::api::CAnomalyJobConfig job2Config;
    BOOST_TEST_REQUIRE(job2Config.parse(job2ConfigJson));
    BOOST_TEST_REQUIRE(job2Config.isInitialized());
    BOOST_REQUIRE_LE(0, job1Config.intervalStagger());
    BOOST_REQUIRE_GE(3599, job1Config.intervalStagger());

    const std::string job3ConfigJson{
        "{\"job_id\":\"job1\",\"description\":\"job number three has same jobId as job number one\", \"analysis_config\":{\"detectors\":[]}}"};

    ml::api::CAnomalyJobConfig job3Config;
    BOOST_TEST_REQUIRE(job3Config.parse(job3ConfigJson));
    BOOST_TEST_REQUIRE(job3Config.isInitialized());
    BOOST_REQUIRE_LE(0, job1Config.intervalStagger());
    BOOST_REQUIRE_GE(3599, job1Config.intervalStagger());

    BOOST_REQUIRE_NE(job1Config.intervalStagger(), job2Config.intervalStagger());
    BOOST_REQUIRE_NE(job2Config.intervalStagger(), job3Config.intervalStagger());
    BOOST_REQUIRE_EQUAL(job3Config.intervalStagger(), job1Config.intervalStagger());
}

BOOST_AUTO_TEST_CASE(testParse) {

    using TAnalysisConfig = ml::api::CAnomalyJobConfig::CAnalysisConfig;
    using TDataDescription = ml::api::CAnomalyJobConfig::CDataDescription;
    using TDetectorConfigVec = ml::api::CAnomalyJobConfig::CAnalysisConfig::TDetectorConfigVec;

    using TStrVec = ml::api::CAnomalyJobConfig::CAnalysisConfig::TStrVec;
    using TAnalysisLimits = ml::api::CAnomalyJobConfig::CAnalysisLimits;
    using TModelPlotConfig = ml::api::CAnomalyJobConfig::CModelPlotConfig;
    {
        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string inValidModelMemoryLimitBytes{
            "[{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"1048076b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}]"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(inValidModelMemoryLimitBytes));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string inValidModelMemoryLimitKiloBytes{
            "[{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"1004kb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}]"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(inValidModelMemoryLimitKiloBytes));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string inValidAnomalyJobConfig{
            "[{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"11mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}]"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(inValidAnomalyJobConfig));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string inValidBucketSpanType{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":1800,\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"11mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(inValidBucketSpanType));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string missingRequiredJobId{
            "{\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"11mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(missingRequiredJobId));
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"all\",\"by_field_name\":\"customer_id\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"background_persist_interval\":\"3h\",\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true,\"terms\":\"customer_id\"},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());
        BOOST_REQUIRE_EQUAL(10800, jobConfig.persistInterval());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Both, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
        BOOST_REQUIRE_EQUAL("customer_id", modelPlotConfig.terms());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"all\",\"by_field_name\":\"customer_id\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":false,\"annotations_enabled\":true,\"terms\":\"customer_id\"},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());
        const ml::core_t::TTime expectedPersistInterval{
            ml::api::CAnomalyJobConfig::DEFAULT_BASE_PERSIST_INTERVAL +
            jobConfig.intervalStagger()};
        BOOST_REQUIRE_EQUAL(expectedPersistInterval, jobConfig.persistInterval());

        const ml::core_t::TTime expectedQuantilePersistInterval{
            ml::api::CAnomalyJobConfig::BASE_MAX_QUANTILE_INTERVAL +
            jobConfig.intervalStagger()};
        BOOST_REQUIRE_EQUAL(expectedQuantilePersistInterval,
                            jobConfig.quantilePersistInterval());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(false, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
        BOOST_REQUIRE_EQUAL("customer_id", modelPlotConfig.terms());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"all\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"background_persist_interval\":\"1d\",\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());
        BOOST_REQUIRE_EQUAL(86400, jobConfig.persistInterval());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Over, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
        BOOST_REQUIRE_EQUAL(EMPTY_STRING, modelPlotConfig.terms());
    }
    {
        const std::string invalidAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"whatever\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(invalidAnomalyJobConfig));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\",\"multivariate_by_fields\":true,"
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"by\",\"by_field_name\":\"customer_id\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true,\"terms\":\"customer_id,category.keyword\"},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());

        const ml::core_t::TTime expectedPersistInterval{
            ml::api::CAnomalyJobConfig::DEFAULT_BASE_PERSIST_INTERVAL +
            jobConfig.intervalStagger()};
        BOOST_REQUIRE_EQUAL(expectedPersistInterval, jobConfig.persistInterval());

        const ml::core_t::TTime expectedQuantilePersistInterval{
            ml::api::CAnomalyJobConfig::BASE_MAX_QUANTILE_INTERVAL +
            jobConfig.intervalStagger()};
        BOOST_REQUIRE_EQUAL(expectedQuantilePersistInterval,
                            jobConfig.quantilePersistInterval());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());
        BOOST_REQUIRE_EQUAL(true, analysisConfig.multivariateByFields());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
        BOOST_REQUIRE_EQUAL("customer_id,category.keyword", modelPlotConfig.terms());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"over\",\"by_field_name\":\"customer_id\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());
        BOOST_REQUIRE_EQUAL(false, analysisConfig.multivariateByFields());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Over, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\",\"multivariate_by_fields\":false,"
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"by\",\"by_field_name\":\"customer_id\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());
        BOOST_REQUIRE_EQUAL(false, analysisConfig.multivariateByFields());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"none\",\"by_field_name\":\"customer_id\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"over\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Over, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfig{
            "{\"job_id\":\"flight_event_rate\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603110779167,"
            "\"description\":\"\",\"analysis_config\":{\"bucket_span\":\"30m\",\"summary_count_field_name\":\"doc_count\","
            "\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"exclude_frequent\":\"none\",\"over_field_name\":\"category.keyword\",\"detector_index\":0}],\"influencers\":[]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"4195304b\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,"
            "\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("flight_event_rate", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(1800, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("doc_count", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();
        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(4, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfigWithMultipleInfluencers{
            "{\"job_id\":\"logs_max_bytes_by_geo\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603290557883,\"description\":\"\","
            "\"analysis_config\":{\"bucket_span\":\"900ms\",\"detectors\":[{\"detector_description\":\"max(bytes) by \\\"geo.src\\\" partitionfield=\\\"host.keyword\\\"\","
            "\"function\":\"max\",\"field_name\":\"bytes\",\"by_field_name\":\"geo.src\",\"partition_field_name\":\"host.keyword\",\"detector_index\":0}],"
            "\"influencers\":[\"geo.src\",\"host.keyword\"]},\"analysis_limits\":{\"model_memory_limit\":\"5140KB\",\"categorization_examples_limit\":4},"
            "\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,"
            "\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfigWithMultipleInfluencers),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("logs_max_bytes_by_geo", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        // When the configured bucket span equates to less than 1s expect the default value
        // to be used.
        BOOST_REQUIRE_EQUAL(ml::api::CAnomalyJobConfig::CAnalysisConfig::DEFAULT_BUCKET_SPAN,
                            analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();

        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("max(bytes) by \"geo.src\" partitionfield=\"host.keyword\"",
                            detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("max", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualMetricMax,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("bytes", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("geo.src", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("host.keyword", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(2, influencers.size());
        BOOST_REQUIRE_EQUAL("geo.src", influencers[0]);
        BOOST_REQUIRE_EQUAL("host.keyword", influencers[1]);

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());

        // Expect the model memory limit to be rounded down to the nearest whole number of megabytes
        BOOST_REQUIRE_EQUAL(5, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfigWithMultipleDetectors{
            "{\"job_id\":\"ecommerce_population\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603290153486,"
            "\"description\":\"ecommerce population job based on category and split by user\","
            "\"analysis_config\":{\"bucket_span\":\"800000micros\","
            "\"detectors\":["
            "{\"detector_description\":\"distinct_count(\\\"category.keyword\\\") by customer_id over \\\"category.keyword\\\"\",\"function\":\"distinct_count\",\"field_name\":\"category.keyword\",\"by_field_name\":\"customer_id\",\"over_field_name\":\"category.keyword\",\"detector_index\":0},"
            "{\"detector_description\":\"count over \\\"category.keyword\\\"\",\"function\":\"count\",\"over_field_name\":\"category.keyword\",\"detector_index\":1}],"
            "\"influencers\":[\"category.keyword\",\"customer_id\"]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"17mb\",\"categorization_examples_limit\":4},"
            "\"data_description\":{\"time_field\":\"order_date\",\"time_format\":\"epoch_ms\"},"
            "\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfigWithMultipleDetectors),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("ecommerce_population", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        // When the configured bucket span equates to less than 1s expect the default value
        // to be used.
        BOOST_REQUIRE_EQUAL(ml::api::CAnomalyJobConfig::CAnalysisConfig::DEFAULT_BUCKET_SPAN,
                            analysisConfig.bucketSpan());
        BOOST_REQUIRE_EQUAL("", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("order_date", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();

        BOOST_REQUIRE_EQUAL(2, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("distinct_count(\"category.keyword\") by customer_id over \"category.keyword\"",
                            detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("distinct_count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationDistinctCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("customer_id", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        BOOST_REQUIRE_EQUAL("count over \"category.keyword\"",
                            detectorsConfig[1].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[1].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount,
                            detectorsConfig[1].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[1].fieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[1].byFieldName());
        BOOST_REQUIRE_EQUAL("category.keyword", detectorsConfig[1].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[1].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[1].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(1).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[1].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(2, influencers.size());
        BOOST_REQUIRE_EQUAL("category.keyword", influencers[0]);
        BOOST_REQUIRE_EQUAL("customer_id", influencers[1]);

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());
        BOOST_REQUIRE_EQUAL(17, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfigWithCustomRule{
            "{\"job_id\":\"count_with_range\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1603901206253,\"description\":\"\","
            "\"analysis_config\":{\"bucket_span\":\"700000000nanos\",\"detectors\":[{\"detector_description\":\"count\",\"function\":\"count\",\"custom_rules\":[{\"actions\":[\"skip_result\"],\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"gt\",\"value\":30.0},{\"applies_to\":\"actual\",\"operator\":\"lt\",\"value\":50.0}]}],\"detector_index\":0}],"
            "\"influencers\":[]},\"analysis_limits\":{\"model_memory_limit\":\"11mb\",\"categorization_examples_limit\":5},"
            "\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfigWithCustomRule),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("count_with_range", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        // When the configured bucket span equates to less than 1s expect the default value
        // to be used.
        BOOST_REQUIRE_EQUAL(ml::api::CAnomalyJobConfig::CAnalysisConfig::DEFAULT_BUCKET_SPAN,
                            analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();

        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(1, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(0, influencers.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(5, analysisLimits.categorizationExamplesLimit());
        BOOST_REQUIRE_EQUAL(11, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(false, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(false, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validMultiMetricCategorizationJobConfig{
            "{\"job_id\":\"categorize_message\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1604311804567,\"description\":\"\","
            "\"analysis_config\":{\"bucket_span\":\"15m\",\"categorization_field_name\":\"message\",\"per_partition_categorization\":{\"enabled\":true,\"stop_on_warn\":false},\"detectors\":["
            "{\"detector_description\":\"count by mlcategory partitionfield=\\\"agent.keyword\\\"\",\"function\":\"count\",\"by_field_name\":\"mlcategory\",\"partition_field_name\":\"agent.keyword\",\"detector_index\":0},"
            "{\"detector_description\":\"rare by mlcategory partitionfield=\\\"agent.keyword\\\"\",\"function\":\"rare\",\"by_field_name\":\"mlcategory\",\"partition_field_name\":\"agent.keyword\",\"detector_index\":1}"
            "],\"influencers\":[\"mlcategory\",\"agent.keyword\",\"message.keyword\"]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"45mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validMultiMetricCategorizationJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("categorize_message", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(900, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();

        BOOST_REQUIRE_EQUAL(2, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count by mlcategory partitionfield=\"agent.keyword\"",
                            detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("mlcategory", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("agent.keyword", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        BOOST_REQUIRE_EQUAL("rare by mlcategory partitionfield=\"agent.keyword\"",
                            detectorsConfig[1].detectorDescription());
        BOOST_REQUIRE_EQUAL("rare", detectorsConfig[1].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRare,
                            detectorsConfig[1].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[1].fieldName());
        BOOST_REQUIRE_EQUAL("mlcategory", detectorsConfig[1].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[1].overFieldName());
        BOOST_REQUIRE_EQUAL("agent.keyword", detectorsConfig[1].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[1].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(1).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[1].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(3, influencers.size());
        BOOST_REQUIRE_EQUAL("mlcategory", influencers[0]);
        BOOST_REQUIRE_EQUAL("agent.keyword", influencers[1]);
        BOOST_REQUIRE_EQUAL("message.keyword", influencers[2]);

        BOOST_REQUIRE_EQUAL("message", analysisConfig.categorizationFieldName());
        const TStrVec& categorizationFilters = analysisConfig.categorizationFilters();
        BOOST_REQUIRE_EQUAL(0, categorizationFilters.size());

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());
        BOOST_REQUIRE_EQUAL(45, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validMultiMetricCategorizationJobConfig{
            "{\"job_id\":\"categorize_message\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1604311804567,\"description\":\"\","
            "\"analysis_config\":{\"bucket_span\":\"15m\",\"categorization_field_name\":\"message\",\"per_partition_categorization\":{\"enabled\":true,\"stop_on_warn\":false},\"detectors\":["
            "{\"detector_description\":\"count by mlcategory partitionfield=\\\"agent.keyword\\\"\",\"function\":\"count\",\"by_field_name\":\"mlcategory\",\"partition_field_name\":\"agent.keyword\",\"detector_index\":0},"
            "{\"detector_description\":\"rare by mlcategory partitionfield=\\\"message.keyword\\\"\",\"function\":\"rare\",\"by_field_name\":\"mlcategory\",\"partition_field_name\":\"message.keyword\",\"detector_index\":1}"
            "],\"influencers\":[\"mlcategory\",\"agent.keyword\",\"message.keyword\"]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"45mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(validMultiMetricCategorizationJobConfig));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string validMultiMetricCategorizationJobConfig{
            "{\"job_id\":\"categorize_message\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1604311804567,\"description\":\"\","
            "\"analysis_config\":{\"bucket_span\":\"15m\",\"categorization_field_name\":\"message\",\"per_partition_categorization\":{\"enabled\":true,\"stop_on_warn\":false},\"detectors\":["
            "{\"detector_description\":\"count by mlcategory partitionfield=\\\"agent.keyword\\\"\",\"function\":\"count\",\"by_field_name\":\"mlcategory\",\"detector_index\":0}"
            "],\"influencers\":[\"mlcategory\",\"agent.keyword\",\"message.keyword\"]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"45mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(validMultiMetricCategorizationJobConfig));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string validMultiMetricCategorizationJobConfig{
            "{\"job_id\":\"categorize_message\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1604311804567,\"description\":\"\","
            "\"analysis_config\":{\"bucket_span\":\"15m\",\"per_partition_categorization\":{\"enabled\":true,\"stop_on_warn\":false},\"detectors\":["
            "{\"detector_description\":\"count by mlcategory partitionfield=\\\"agent.keyword\\\"\",\"function\":\"count\",\"by_field_name\":\"mlcategory\",\"partition_field_name\":\"agent.keyword\",\"detector_index\":0},"
            "{\"detector_description\":\"rare by mlcategory partitionfield=\\\"agent.keyword\\\"\",\"function\":\"rare\",\"by_field_name\":\"mlcategory\",\"partition_field_name\":\"agent.keyword\",\"detector_index\":1}"
            "],\"influencers\":[\"mlcategory\",\"agent.keyword\",\"message.keyword\"]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"45mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},\"model_plot_config\":{\"enabled\":true,\"annotations_enabled\":true},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parse(validMultiMetricCategorizationJobConfig));
        BOOST_TEST_REQUIRE(!jobConfig.isInitialized());
    }
    {
        const std::string validCategorizationJobConfig{
            "{\"job_id\":\"unusual_message_counts\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1604311804567,\"custom_settings\":{\"created_by\":\"categorization-wizard\"},\"description\":\"Unusual message counts\","
            "\"analysis_config\":{\"bucket_span\":\"15m\",\"categorization_field_name\":\"message\",\"categorization_filters\":[\"foo.*\",\"bar.*\"],\"per_partition_categorization\":{\"enabled\":false},\"detectors\":[{\"detector_description\":\"count by mlcategory\",\"function\":\"count\",\"by_field_name\":\"mlcategory\",\"detector_index\":0}],\"influencers\":[\"mlcategory\"]},"
            "\"analysis_limits\":{\"model_memory_limit\":\"26mb\",\"categorization_examples_limit\":4},\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},\"model_plot_config\":{\"enabled\":false,\"annotations_enabled\":false},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validCategorizationJobConfig),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());

        BOOST_REQUIRE_EQUAL("anomaly_detector", jobConfig.jobType());
        BOOST_REQUIRE_EQUAL("unusual_message_counts", jobConfig.jobId());

        const TAnalysisConfig& analysisConfig = jobConfig.analysisConfig();

        BOOST_REQUIRE_EQUAL(900, analysisConfig.bucketSpan());

        BOOST_REQUIRE_EQUAL("", analysisConfig.summaryCountFieldName());

        const TDataDescription& dataDescription = jobConfig.dataDescription();

        BOOST_REQUIRE_EQUAL("timestamp", dataDescription.timeField());

        const TDetectorConfigVec& detectorsConfig = analysisConfig.detectorsConfig();

        BOOST_REQUIRE_EQUAL(1, detectorsConfig.size());
        BOOST_REQUIRE_EQUAL("count by mlcategory", detectorsConfig[0].detectorDescription());
        BOOST_REQUIRE_EQUAL("count", detectorsConfig[0].functionName());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount,
                            detectorsConfig[0].function());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].fieldName());
        BOOST_REQUIRE_EQUAL("mlcategory", detectorsConfig[0].byFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].overFieldName());
        BOOST_REQUIRE_EQUAL("", detectorsConfig[0].partitionFieldName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, detectorsConfig[0].excludeFrequent());
        BOOST_REQUIRE_EQUAL(0, analysisConfig.detectionRules().at(0).size());
        BOOST_REQUIRE_EQUAL(false, detectorsConfig[0].useNull());

        const TStrVec& influencers = analysisConfig.influencers();
        BOOST_REQUIRE_EQUAL(1, influencers.size());
        BOOST_REQUIRE_EQUAL("mlcategory", influencers[0]);

        BOOST_REQUIRE_EQUAL("message", analysisConfig.categorizationFieldName());
        const TStrVec& categorizationFilters = analysisConfig.categorizationFilters();
        BOOST_REQUIRE_EQUAL(2, categorizationFilters.size());
        BOOST_REQUIRE_EQUAL("foo.*", categorizationFilters[0]);
        BOOST_REQUIRE_EQUAL("bar.*", categorizationFilters[1]);

        const TAnalysisLimits& analysisLimits = jobConfig.analysisLimits();
        BOOST_REQUIRE_EQUAL(4, analysisLimits.categorizationExamplesLimit());
        BOOST_REQUIRE_EQUAL(26, analysisLimits.modelMemoryLimitMb());

        const TModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
        BOOST_REQUIRE_EQUAL(false, modelPlotConfig.enabled());
        BOOST_REQUIRE_EQUAL(false, modelPlotConfig.annotationsEnabled());
    }
    {
        const std::string validAnomalyJobConfigWithCustomRuleFilter{
            "{\"job_id\":\"mean_bytes_by_clientip\",\"job_type\":\"anomaly_detector\",\"job_version\":\"8.0.0\",\"create_time\":1604671135245,\"description\":\"mean bytes by clientip\","
            "\"analysis_config\":{\"bucket_span\":\"3h\",\"detectors\":[{\"detector_description\":\"mean(bytes) by clientip\",\"function\":\"mean\",\"field_name\":\"bytes\",\"by_field_name\":\"clientip\","
            "\"custom_rules\":[{\"actions\":[\"skip_result\"],\"scope\":{\"clientip\":{\"filter_id\":\"safe_ips\",\"filter_type\":\"include\"}},\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"lt\",\"value\":10.0}]}],"
            "\"detector_index\":0}],\"influencers\":[\"clientip\"]},\"analysis_limits\":{\"model_memory_limit\":\"42mb\",\"categorization_examples_limit\":4},"
            "\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"},\"model_plot_config\":{\"enabled\":false,\"annotations_enabled\":false},"
            "\"model_snapshot_retention_days\":10,\"daily_model_snapshot_retention_after_days\":1,\"results_index_name\":\"shared\",\"allow_lazy_open\":false}"};

        // Expect parsing to fail if the filter referenced by the custom rule cannot be found
        ml::api::CAnomalyJobConfig jobConfigEmptyFilterMap;
        BOOST_TEST_REQUIRE(!jobConfigEmptyFilterMap.parse(validAnomalyJobConfigWithCustomRuleFilter));
        BOOST_TEST_REQUIRE(!jobConfigEmptyFilterMap.isInitialized());

        // Expect parsing to succeed if the filter referenced by the custom rule can be found in the filter map.
        const std::string filterConfigJson{"{\"filters\":[{\"filter_id\":\"safe_ips\",\"items\":[]}]}"};
        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.parseFilterConfig(filterConfigJson));

        const std::string validScheduledEventsConfigJson{"{\"events\":["
                                                         "]}"};

        BOOST_TEST_REQUIRE(jobConfig.parseEventConfig(validScheduledEventsConfigJson));

        jobConfig.analysisConfig().init(jobConfig.ruleFilters(),
                                        jobConfig.scheduledEvents());

        BOOST_REQUIRE_MESSAGE(jobConfig.parse(validAnomalyJobConfigWithCustomRuleFilter),
                              "Cannot parse JSON job config!");
        BOOST_TEST_REQUIRE(jobConfig.isInitialized());
    }
}

BOOST_AUTO_TEST_CASE(testParseFilterConfig) {
    {
        const std::string validFilterConfigJson{
            "{\"filters\":[{\"filter_id\":\"safe_ips\", \"items\":[\"127.0.0.1\",\"192.168.0.1\"]}]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.parseFilterConfig(validFilterConfigJson));

        BOOST_REQUIRE_EQUAL(1, jobConfig.ruleFilters().size());

        BOOST_TEST_REQUIRE(jobConfig.ruleFilters()["safe_ips"].contains("127.0.0.1"));
        BOOST_TEST_REQUIRE(jobConfig.ruleFilters()["safe_ips"].contains("192.168.0.1"));
    }
    {
        const std::string validFilterConfigJson{
            "{\"filters\":[{\"filter_id\":\"safe_ips\", \"items\":[\"127.0.0.1\",\"192.168.0.1\"]},{\"filter_id\":\"safe_domains\", \"items\":[\"elastic.*\",\"*.co.nz\"]}]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.parseFilterConfig(validFilterConfigJson));

        BOOST_REQUIRE_EQUAL(2, jobConfig.ruleFilters().size());

        BOOST_TEST_REQUIRE(jobConfig.ruleFilters()["safe_ips"].contains("127.0.0.1"));
        BOOST_TEST_REQUIRE(jobConfig.ruleFilters()["safe_ips"].contains("192.168.0.1"));

        BOOST_TEST_REQUIRE(jobConfig.ruleFilters()["safe_domains"].contains("elastic.*"));
        BOOST_TEST_REQUIRE(jobConfig.ruleFilters()["safe_domains"].contains("*.co.nz"));
    }
    {
        const std::string invalidFilterConfigJson{
            "{\"filters\":{\"filter_id\":\"safe_ips\", \"items\":[\"127.0.0.1\",\"192.168.0.1\"]},{\"filter_id\":\"safe_domains\", \"items\":[\"elastic.*\",\"*.co.nz\"]}}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parseFilterConfig(invalidFilterConfigJson));
    }
    {
        const std::string invalidFilterConfigJson{
            "{\"filters\":[{\"filter_id\":[\"127.0.0.1\",\"192.168.0.1\"]},{\"filter_id\":\"safe_domains\", \"items\":[\"elastic.*\",\"*.co.nz\"]}]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parseFilterConfig(invalidFilterConfigJson));
    }
    {
        const std::string invalidFilterConfigJson{
            "{\"filters\":[{\"filter_id\":\"safe_ips\", \"items\":\"127.0.0.1\"}]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parseFilterConfig(invalidFilterConfigJson));
    }
}

BOOST_AUTO_TEST_CASE(testParseScheduledEvents) {

    {
        const std::string validScheduledEventsConfigJson{
            "{\"events\":["
            "{\"description\":\"christmas\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6088544E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6089408E9}]}]},"
            "{\"description\":\"black_friday\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6286364E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6290684E9}]}]}"
            "]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.parseEventConfig(validScheduledEventsConfigJson));

        BOOST_REQUIRE_EQUAL(2, jobConfig.scheduledEvents().size());

        BOOST_REQUIRE_EQUAL("christmas", jobConfig.scheduledEvents()[0].first);
        BOOST_REQUIRE_EQUAL("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1608854400.000000 AND TIME < 1608940800.000000",
                            jobConfig.scheduledEvents()[0].second.print());

        BOOST_REQUIRE_EQUAL("black_friday", jobConfig.scheduledEvents()[1].first);
        BOOST_REQUIRE_EQUAL("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1628636400.000000 AND TIME < 1629068400.000000",
                            jobConfig.scheduledEvents()[1].second.print());
    }
    {
        const std::string invalidScheduledEventsConfigJson{
            "{\"events\":["
            "{\"description\":\"christmas\", \"rules\":[]},"
            "{\"description\":\"black_friday\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6286364E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6290684E9}]}]}"
            "]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parseEventConfig(invalidScheduledEventsConfigJson));
    }
    {
        const std::string invalidScheduledEventsConfigJson{
            "{\"events\":["
            "{\"description\":\"christmas\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6088544E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6089408E9}]}]},"
            "{\"event_rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6286364E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6290684E9}]}]}"
            "]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parseEventConfig(invalidScheduledEventsConfigJson));
    }
    {
        const std::string validScheduledEventsConfigJson{
            "{\"events\":["
            "{\"description\":\"christmas\", \"rules\":[{\"actions\":[\"skip_whatever\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6088544E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6089408E9}]}]},"
            "{\"description\":\"black_friday\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\":1.6286364E9},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\":1.6290684E9}]}]}"
            "]}"};

        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(!jobConfig.parseEventConfig(validScheduledEventsConfigJson));
    }
}

BOOST_AUTO_TEST_SUITE_END()
