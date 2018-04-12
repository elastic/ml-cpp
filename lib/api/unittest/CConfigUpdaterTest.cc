/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CConfigUpdaterTest.h"

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDetectionRule.h>

#include <api/CConfigUpdater.h>
#include <api/CFieldConfig.h>

#include <boost/property_tree/ptree.hpp>

#include <sstream>
#include <string>

using namespace ml;
using namespace api;

CppUnit::Test* CConfigUpdaterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CConfigUpdaterTest");
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenUpdateCannotBeParsed",
        &CConfigUpdaterTest::testUpdateGivenUpdateCannotBeParsed));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenUnknownStanzas",
        &CConfigUpdaterTest::testUpdateGivenUnknownStanzas));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenModelPlotConfig",
        &CConfigUpdaterTest::testUpdateGivenModelPlotConfig));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenDetectorRules",
        &CConfigUpdaterTest::testUpdateGivenDetectorRules));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenRulesWithInvalidDetectorIndex",
        &CConfigUpdaterTest::testUpdateGivenRulesWithInvalidDetectorIndex));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenFilters", &CConfigUpdaterTest::testUpdateGivenFilters));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConfigUpdaterTest>(
        "CConfigUpdaterTest::testUpdateGivenScheduledEvents",
        &CConfigUpdaterTest::testUpdateGivenScheduledEvents));
    return suiteOfTests;
}

void CConfigUpdaterTest::testUpdateGivenUpdateCannotBeParsed() {
    CFieldConfig fieldConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(fieldConfig, modelConfig);
    CPPUNIT_ASSERT(configUpdater.update("this is invalid") == false);
}

void CConfigUpdaterTest::testUpdateGivenUnknownStanzas() {
    CFieldConfig fieldConfig;
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    CConfigUpdater configUpdater(fieldConfig, modelConfig);
    CPPUNIT_ASSERT(configUpdater.update("[unknown1]\na = 1\n[unknown2]\nb = 2\n") == false);
}

void CConfigUpdaterTest::testUpdateGivenModelPlotConfig() {
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

    CPPUNIT_ASSERT(configUpdater.update(configUpdate));
    CPPUNIT_ASSERT_EQUAL(83.5, modelConfig.modelPlotBoundsPercentile());

    terms = modelConfig.modelPlotTerms();
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), terms.size());
    CPPUNIT_ASSERT(terms.find(std::string("c")) != terms.end());
    CPPUNIT_ASSERT(terms.find(std::string("d")) != terms.end());
}

void CConfigUpdaterTest::testUpdateGivenDetectorRules() {
    CFieldConfig fieldConfig;
    std::string originalRules0("[{\"actions\":[\"filter_results\"],\"conditions_connective\":\"or\",");
    originalRules0 += "\"conditions\":[{\"type\":\"numerical_actual\","
                      "\"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}]}]";
    std::string originalRules1("[{\"actions\":[\"filter_results\"],\"conditions_connective\":\"or\",");
    originalRules1 += "\"conditions\":[{\"type\":\"numerical_actual\","
                      "\"condition\":{\"operator\":\"gt\",\"value\":\"5\"}}]}]";
    fieldConfig.parseRules(0, originalRules0);
    fieldConfig.parseRules(1, originalRules1);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    std::string configUpdate0("[detectorRules]\ndetectorIndex = 0\nrulesJson = []\n");
    std::string configUpdate1(
        "[detectorRules]\ndetectorIndex = 1\nrulesJson = "
        "[{\"actions\":[\"filter_results\"],\"conditions_connective\":\"or\","
        "\"conditions\":[{\"type\":\"numerical_"
        "typical\",\"condition\":{\"operator\":\"lt\",\"value\":\"15\"}}]}]");

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    CPPUNIT_ASSERT(configUpdater.update(configUpdate0));
    CPPUNIT_ASSERT(configUpdater.update(configUpdate1));

    CFieldConfig::TIntDetectionRuleVecUMap::const_iterator itr =
        fieldConfig.detectionRules().find(0);
    CPPUNIT_ASSERT(itr->second.empty());
    itr = fieldConfig.detectionRules().find(1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), itr->second.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF TYPICAL < 15.000000"),
                         itr->second[0].print());
}

void CConfigUpdaterTest::testUpdateGivenRulesWithInvalidDetectorIndex() {
    CFieldConfig fieldConfig;
    std::string originalRules("[{\"actions\":[\"filter_results\"],\"conditions_connective\":\"or\",");
    originalRules += "\"conditions\":[{\"type\":\"numerical_actual\","
                     "\"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}]}]";
    fieldConfig.parseRules(0, originalRules);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    std::string configUpdate("[detectorRules]\ndetectorIndex = invalid\nrulesJson = []\n");

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    CPPUNIT_ASSERT(configUpdater.update(configUpdate) == false);
}

void CConfigUpdaterTest::testUpdateGivenFilters() {
    CFieldConfig fieldConfig;
    fieldConfig.processFilter("filter.filter_1", "[\"aaa\",\"bbb\"]");
    fieldConfig.processFilter("filter.filter_2", "[\"ccc\",\"ddd\"]");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    auto ruleFilters = fieldConfig.ruleFilters();
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), ruleFilters.size());

    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("aaa"));
    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("bbb"));
    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("ccc") == false);
    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("ddd") == false);

    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("aaa") == false);
    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("bbb") == false);
    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("ccc"));
    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("ddd"));

    // Update existing ones
    std::string configUpdate("[filters]\nfilter.filter_1=[\"ccc\",\"ddd\"]"
                             "\nfilter.filter_2=[\"aaa\",\"bbb\"]\n");

    CConfigUpdater configUpdater(fieldConfig, modelConfig);

    CPPUNIT_ASSERT(configUpdater.update(configUpdate));

    ruleFilters = fieldConfig.ruleFilters();
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), ruleFilters.size());

    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("aaa") == false);
    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("bbb") == false);
    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("ccc"));
    CPPUNIT_ASSERT(ruleFilters["filter_1"].contains("ddd"));

    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("aaa"));
    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("bbb"));
    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("ccc") == false);
    CPPUNIT_ASSERT(ruleFilters["filter_2"].contains("ddd") == false);

    // Now update by adding a new filter
    configUpdate = "[filters]\nfilter.filter_3=[\"new\"]\n";
    CPPUNIT_ASSERT(configUpdater.update(configUpdate));

    ruleFilters = fieldConfig.ruleFilters();
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), ruleFilters.size());
    CPPUNIT_ASSERT(ruleFilters["filter_3"].contains("new"));
}

void CConfigUpdaterTest::testUpdateGivenScheduledEvents() {
    std::string validRule1 = "[{\"actions\":[\"filter_results\",\"skip_"
                             "sampling\"],\"conditions_connective\":\"and\","
                             "\"conditions\":[{\"type\":\"time\",\"condition\":"
                             "{\"operator\":\"gte\",\"value\":\"1\"}},"
                             "{\"type\":\"time\",\"condition\":{\"operator\":"
                             "\"lt\",\"value\":\"2\"}}]}]";
    std::string validRule2 = "[{\"actions\":[\"filter_results\",\"skip_"
                             "sampling\"],\"conditions_connective\":\"and\","
                             "\"conditions\":[{\"type\":\"time\",\"condition\":"
                             "{\"operator\":\"gte\",\"value\":\"3\"}},"
                             "{\"type\":\"time\",\"condition\":{\"operator\":"
                             "\"lt\",\"value\":\"4\"}}]}]";

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
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), events.size());
        CPPUNIT_ASSERT_EQUAL(std::string("old_event_1"), events[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF TIME >= 1.000000 "
                                         "AND TIME < 2.000000"),
                             events[0].second.print());
        CPPUNIT_ASSERT_EQUAL(std::string("old_event_2"), events[1].first);
        CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF TIME >= 3.000000 "
                                         "AND TIME < 4.000000"),
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

        CPPUNIT_ASSERT(configUpdater.update(configUpdate.str()));

        const auto& events = fieldConfig.scheduledEvents();
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), events.size());
        CPPUNIT_ASSERT_EQUAL(std::string("new_event_1"), events[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF TIME >= 3.000000 "
                                         "AND TIME < 4.000000"),
                             events[0].second.print());
        CPPUNIT_ASSERT_EQUAL(std::string("new_event_2"), events[1].first);
        CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF TIME >= 1.000000 "
                                         "AND TIME < 2.000000"),
                             events[1].second.print());
    }

    // Now test an update that clears the events
    {
        std::stringstream configUpdate;
        configUpdate << "[scheduledEvents]"
                     << "\n";
        configUpdate << "clear = true"
                     << "\n";

        CPPUNIT_ASSERT(configUpdater.update(configUpdate.str()));

        const auto& events = fieldConfig.scheduledEvents();
        CPPUNIT_ASSERT(events.empty());
    }
}
