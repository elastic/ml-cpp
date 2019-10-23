/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDetectionRule.h>

#include <api/CConfigUpdater.h>
#include <api/CFieldConfig.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/test/unit_test.hpp>

#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CConfigUpdaterTest)

using namespace ml;
using namespace api;

BOOST_AUTO_TEST_CASE(testUpdateGivenUpdateCannotBeParsed) {
    CFieldConfig fieldConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(fieldConfig, modelConfig);
    BOOST_TEST_REQUIRE(configUpdater.update("this is invalid") == false);
}

BOOST_AUTO_TEST_CASE(testUpdateGivenUnknownStanzas) {
    CFieldConfig fieldConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(fieldConfig, modelConfig);
    BOOST_TEST_REQUIRE(configUpdater.update("[unknown1]\na = 1\n[unknown2]\nb = 2\n") == false);
}

BOOST_AUTO_TEST_CASE(testUpdateGivenModelPlotConfig) {
    using TStrSet = model::CAnomalyDetectorModelConfig::TStrSet;

    CFieldConfig fieldConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    modelConfig.modelPlotBoundsPercentile(95.0);
    TStrSet terms;
    terms.insert(std::string("a"));
    terms.insert(std::string("b"));
    modelConfig.modelPlotTerms(terms);

    std::string configUpdate("[modelPlotConfig]\nboundspercentile = 83.5\nterms = c,d\n");

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate));
    BOOST_REQUIRE_EQUAL(83.5, modelConfig.modelPlotBoundsPercentile());

    terms = modelConfig.modelPlotTerms();
    BOOST_REQUIRE_EQUAL(std::size_t(2), terms.size());
    BOOST_TEST_REQUIRE(terms.find(std::string("c")) != terms.end());
    BOOST_TEST_REQUIRE(terms.find(std::string("d")) != terms.end());
}

BOOST_AUTO_TEST_CASE(testUpdateGivenDetectorRules) {
    CFieldConfig fieldConfig;
    std::string originalRules0("[{\"actions\":[\"skip_result\"],");
    originalRules0 += "\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"lt\",\"value\": 5.0}]}]";
    std::string originalRules1("[{\"actions\":[\"skip_result\"],");
    originalRules1 += "\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"gt\",\"value\": 5.0}]}]";
    fieldConfig.parseRules(0, originalRules0);
    fieldConfig.parseRules(1, originalRules1);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    std::string configUpdate0("[detectorRules]\ndetectorIndex = 0\nrulesJson = []\n");
    std::string configUpdate1("[detectorRules]\ndetectorIndex = 1\nrulesJson = "
                              "[{\"actions\":[\"skip_result\"],\"conditions\":[{\"applies_to\":\"typical\","
                              "\"operator\":\"lt\",\"value\": 15.0}]}]");

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate0));
    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate1));

    CFieldConfig::TIntDetectionRuleVecUMap::const_iterator itr =
        fieldConfig.detectionRules().find(0);
    BOOST_TEST_REQUIRE(itr->second.empty());
    itr = fieldConfig.detectionRules().find(1);
    BOOST_REQUIRE_EQUAL(std::size_t(1), itr->second.size());
    BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT IF TYPICAL < 15.000000"),
                      itr->second[0].print());
}

BOOST_AUTO_TEST_CASE(testUpdateGivenRulesWithInvalidDetectorIndex) {
    CFieldConfig fieldConfig;
    std::string originalRules("[{\"actions\":[\"skip_result\"],");
    originalRules += "\"conditions\":[{\"applies_to\":\"actual\",\"operator\":\"lt\",\"value\": 5.0}]}]";
    fieldConfig.parseRules(0, originalRules);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    std::string configUpdate("[detectorRules]\ndetectorIndex = invalid\nrulesJson = []\n");

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate) == false);
}

BOOST_AUTO_TEST_CASE(testUpdateGivenFilters) {
    CFieldConfig fieldConfig;
    fieldConfig.processFilter("filter.filter_1", "[\"aaa\",\"bbb\"]");
    fieldConfig.processFilter("filter.filter_2", "[\"ccc\",\"ddd\"]");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    auto ruleFilters = fieldConfig.ruleFilters();
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

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    BOOST_TEST_REQUIRE(configUpdater.update(configUpdate));

    ruleFilters = fieldConfig.ruleFilters();
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

    ruleFilters = fieldConfig.ruleFilters();
    BOOST_REQUIRE_EQUAL(std::size_t(3), ruleFilters.size());
    BOOST_TEST_REQUIRE(ruleFilters["filter_3"].contains("new"));
}

BOOST_AUTO_TEST_CASE(testUpdateGivenScheduledEvents) {
    std::string validRule1 =
        "[{\"actions\":[\"skip_result\",\"skip_model_update\"],"
        "\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\": 1.0},"
        "{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\": 2.0}]}]";
    std::string validRule2 =
        "[{\"actions\":[\"skip_result\",\"skip_model_update\"],"
        "\"conditions\":[{\"applies_to\":\"time\",\"operator\":\"gte\",\"value\": 3.0},"
        "{\"applies_to\":\"time\",\"operator\":\"lt\",\"value\": 4.0}]}]";

    CFieldConfig fieldConfig;

    // Set up some events
    {
        boost::property_tree::ptree propTree;
        propTree.put(boost::property_tree::ptree::path_type("scheduledevent.0.description", '\t'),
                     "old_event_1");
        propTree.put(boost::property_tree::ptree::path_type("scheduledevent.0.rules", '\t'),
                     validRule1);
        propTree.put(boost::property_tree::ptree::path_type("scheduledevent.1.description", '\t'),
                     "old_event_2");
        propTree.put(boost::property_tree::ptree::path_type("scheduledevent.1.rules", '\t'),
                     validRule2);
        fieldConfig.updateScheduledEvents(propTree);

        const auto& events = fieldConfig.scheduledEvents();
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
    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    // Test an update that replaces the events
    {
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

        const auto& events = fieldConfig.scheduledEvents();
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

        const auto& events = fieldConfig.scheduledEvents();
        BOOST_TEST_REQUIRE(events.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
