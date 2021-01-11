/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDetectionRule.h>

#include <api/CAnomalyJobConfig.h>
#include <api/CConfigUpdater.h>

#include <boost/test/unit_test.hpp>

#include <sstream>
#include <string>

using TStrSet = ml::model::CAnomalyDetectorModelConfig::TStrSet;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TStrSet::iterator)

BOOST_AUTO_TEST_SUITE(CConfigUpdaterTest)

using namespace ml;
using namespace api;

BOOST_AUTO_TEST_CASE(testUpdateGivenUpdateCannotBeParsed) {
    CAnomalyJobConfig jobConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(jobConfig, modelConfig);
    BOOST_TEST_REQUIRE(configUpdater.update("this is invalid") == false);
}

BOOST_AUTO_TEST_CASE(testUpdateGivenUnknownStanzas) {
    CAnomalyJobConfig jobConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(jobConfig, modelConfig);
    BOOST_TEST_REQUIRE(configUpdater.update("[unknown1]\na = 1\n[unknown2]\nb = 2\n") == false);
}

BOOST_AUTO_TEST_CASE(testUpdateGivenModelPlotConfig) {
    CAnomalyJobConfig jobConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    modelConfig.modelPlotBoundsPercentile(95.0);
    TStrSet terms;
    terms.insert(std::string("a"));
    terms.insert(std::string("b"));
    modelConfig.modelPlotTerms(terms);

    std::string configUpdate("[modelPlotConfig]\nboundspercentile = 83.5\nterms = c,d\nannotations_enabled = false\n");

    CConfigUpdater configUpdater(jobConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate));
    BOOST_REQUIRE_EQUAL(83.5, modelConfig.modelPlotBoundsPercentile());

    terms = modelConfig.modelPlotTerms();
    BOOST_REQUIRE_EQUAL(std::size_t(2), terms.size());
    BOOST_TEST_REQUIRE(terms.find(std::string("c")) != terms.end());
    BOOST_TEST_REQUIRE(terms.find(std::string("d")) != terms.end());
}

BOOST_AUTO_TEST_CASE(testUpdateGivenDetectorRules) {
    CAnomalyJobConfig jobConfig;
    std::string originalRules0("[{\"actions\":[\"skip_result\"],");
    originalRules0 += "\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"lt\",\"value\": 5.0}]}]";
    std::string originalRules1("[{\"actions\":[\"skip_result\"],");
    originalRules1 += "\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"gt\",\"value\": 5.0}]}]";
    jobConfig.analysisConfig().parseRules(0, originalRules0);
    jobConfig.analysisConfig().parseRules(1, originalRules1);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    std::string configUpdate0("[detectorRules]\ndetectorIndex = 0\nrulesJson = []\n");
    std::string configUpdate1("[detectorRules]\ndetectorIndex = 1\nrulesJson = "
                              "[{\"actions\":[\"skip_result\"],\"conditions\":[{\"applies_to\":\"typical\","
                              "\"operator\":\"lt\",\"value\": 15.0}]}]");

    CConfigUpdater configUpdater(jobConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate0));
    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate1));

    CAnomalyJobConfig::CAnalysisConfig::TIntDetectionRuleVecUMap::const_iterator itr =
        jobConfig.analysisConfig().detectionRules().find(0);
    BOOST_TEST_REQUIRE(itr->second.empty());
    itr = jobConfig.analysisConfig().detectionRules().find(1);
    BOOST_REQUIRE_EQUAL(std::size_t(1), itr->second.size());
    BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT IF TYPICAL < 15.000000"),
                        itr->second[0].print());
}

BOOST_AUTO_TEST_CASE(testUpdateGivenRulesWithInvalidDetectorIndex) {
    CAnomalyJobConfig jobConfig;
    const std::string validAnomalyJobConfigWithCustomRule{
        "{\"job_id\":\"count_with_range\","
        "\"analysis_config\":{\"detectors\":[{\"function\":\"count\","
        "\"custom_rules\":[{\"actions\":[\"skip_result\"],\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"lt\",\"value\": 5.0}]}],"
        "\"detector_index\":0}]},\"analysis_limits\":{\"model_memory_limit\":\"11mb\",\"categorization_examples_limit\":5},"
        "\"data_description\":{\"time_field\":\"timestamp\",\"time_format\":\"epoch_ms\"}}"};

    BOOST_TEST_REQUIRE(jobConfig.parse(validAnomalyJobConfigWithCustomRule));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    std::string configUpdate("[detectorRules]\ndetectorIndex = invalid\nrulesJson = []\n");

    CConfigUpdater configUpdater(jobConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate) == false);
}

BOOST_AUTO_TEST_CASE(testUpdateGivenFilters) {
    CAnomalyJobConfig jobConfig;

    const std::string jsonFilterConfig{
        "{\"filters\":[{\"filter_id\":\"filter_1\", \"items\":[\"aaa\",\"bbb\"]}, {\"filter_id\":\"filter_2\", \"items\":[\"ccc\",\"ddd\"]}]}"};

    BOOST_TEST_REQUIRE(jobConfig.parseFilterConfig(jsonFilterConfig));
    jobConfig.initRuleFilters();

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    auto ruleFilters = jobConfig.ruleFilters();
    BOOST_REQUIRE_EQUAL(std::size_t(2), ruleFilters.size());

    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("aaa"));
    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("bbb"));
    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("ccc") == false);
    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("ddd") == false);

    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("aaa") == false);
    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("bbb") == false);
    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("ccc"));
    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("ddd"));

    // Update existing ones
    std::string configUpdate("[filters]\nfilter.filter_1=[\"ccc\",\"ddd\"]\nfilter.filter_2=[\"aaa\",\"bbb\"]\n");

    CConfigUpdater configUpdater(jobConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate));

    ruleFilters = jobConfig.analysisConfig().ruleFilters();
    BOOST_REQUIRE_EQUAL(std::size_t(2), ruleFilters.size());

    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("aaa") == false);
    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("bbb") == false);
    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("ccc"));
    BOOST_TEST_REQUIRE(ruleFilters["filter_1"].contains("ddd"));

    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("aaa"));
    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("bbb"));
    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("ccc") == false);
    BOOST_TEST_REQUIRE(ruleFilters["filter_2"].contains("ddd") == false);

    // Now update by adding a new filter
    configUpdate = "[filters]\nfilter.filter_3=[\"new\"]\n";
    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate));

    ruleFilters = jobConfig.analysisConfig().ruleFilters();
    BOOST_REQUIRE_EQUAL(std::size_t(3), ruleFilters.size());
    BOOST_TEST_REQUIRE(ruleFilters["filter_3"].contains("new"));
}

BOOST_AUTO_TEST_CASE(testUpdateGivenScheduledEvents) {
    const std::string validScheduledEvents{
        "{\"events\":["
        "{\"description\":\"old_event_1\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\": 1.0},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\": 2.0}]}]},"
        "{\"description\":\"old_event_2\", \"rules\":[{\"actions\":[\"skip_result\",\"skip_model_update\"],\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\": 3.0},{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\": 4.0}]}]}"
        "]}"};

    CAnomalyJobConfig jobConfig;

    // Set up some events
    {
        BOOST_TEST_REQUIRE(jobConfig.parseEventConfig(validScheduledEvents));
        jobConfig.initScheduledEvents();

        const auto& events = jobConfig.analysisConfig().scheduledEvents();
        BOOST_REQUIRE_EQUAL(std::size_t(2), events.size());
        BOOST_REQUIRE_EQUAL(std::string("old_event_1"), events[0].first);
        BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1.000000 AND TIME < 2.000000"),
                            events[0].second.print());
        BOOST_REQUIRE_EQUAL(std::string("old_event_2"), events[1].first);
        BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 3.000000 AND TIME < 4.000000"),
                            events[1].second.print());
    }

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(jobConfig, modelConfig);

    // Test an update that replaces the events
    {
        std::string validRule1 =
            "[{\"actions\":[\"skip_result\",\"skip_model_update\"],"
            "\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\": 1.0},"
            "{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\": 2.0}]}]";
        std::string validRule2 =
            "[{\"actions\":[\"skip_result\",\"skip_model_update\"],"
            "\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\": 3.0},"
            "{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\": 4.0}]}]";

        std::stringstream configUpdate;
        configUpdate << "[scheduledEvents]"
                     << "\n";
        configUpdate << "scheduledevent.0.description = new_event_1"
                     << "\n";
        configUpdate << "scheduledevent.0.rules = " << validRule2 << "\n";
        configUpdate << "scheduledevent.1.description = new_event_2"
                     << "\n";
        configUpdate << "scheduledevent.1.rules = " << validRule1 << "\n";

        BOOST_TEST_REQUIRE(configUpdater.update(configUpdate.str()));

        const auto& events = jobConfig.analysisConfig().scheduledEvents();
        BOOST_REQUIRE_EQUAL(std::size_t(2), events.size());
        BOOST_REQUIRE_EQUAL(std::string("new_event_1"), events[0].first);
        BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 3.000000 AND TIME < 4.000000"),
                            events[0].second.print());
        BOOST_REQUIRE_EQUAL(std::string("new_event_2"), events[1].first);
        BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1.000000 AND TIME < 2.000000"),
                            events[1].second.print());
    }

    // Now test an update that clears the events
    {
        std::stringstream configUpdate;
        configUpdate << "[scheduledEvents]"
                     << "\n";
        configUpdate << "clear = true"
                     << "\n";

        BOOST_TEST_REQUIRE(configUpdater.update(configUpdate.str()));

        const auto& events = jobConfig.analysisConfig().scheduledEvents();
        BOOST_TEST_REQUIRE(events.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
