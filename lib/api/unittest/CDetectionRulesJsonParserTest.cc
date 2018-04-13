/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#include "CDetectionRulesJsonParserTest.h"

#include <core/CLogger.h>
#include <core/CPatternSet.h>

#include <api/CDetectionRulesJsonParser.h>

#include <boost/unordered_map.hpp>

using namespace ml;
using namespace api;

namespace {
using TStrPatternSetUMap = CDetectionRulesJsonParser::TStrPatternSetUMap;
TStrPatternSetUMap EMPTY_VALUE_FILTER_MAP;
}

CppUnit::Test* CDetectionRulesJsonParserTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDetectionRulesJsonParserTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenEmptyString",
        &CDetectionRulesJsonParserTest::testParseRulesGivenEmptyString));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenEmptyArray",
        &CDetectionRulesJsonParserTest::testParseRulesGivenEmptyArray));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenArrayContainsStrings",
        &CDetectionRulesJsonParserTest::testParseRulesGivenArrayContainsStrings));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMissingRuleAction",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMissingRuleAction));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenRuleActionIsNotArray",
        &CDetectionRulesJsonParserTest::testParseRulesGivenRuleActionIsNotArray));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenInvalidRuleAction",
        &CDetectionRulesJsonParserTest::testParseRulesGivenInvalidRuleAction));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionsConnective",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionsConnective));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionsConnective",
        &CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionsConnective));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMissingRuleConditions",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMissingRuleConditions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsNotArray",
        &CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsNotArray));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionOperator",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionOperator));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionOperator",
        &CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionOperator));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenNumericalActualRuleWithConnectiveOr",
        &CDetectionRulesJsonParserTest::testParseRulesGivenNumericalActualRuleWithConnectiveOr));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenNumericalTypicalAndDiffAbsRuleWithConnectiveAnd",
        &CDetectionRulesJsonParserTest::testParseRulesGivenNumericalTypicalAndDiffAbsRuleWithConnectiveAnd));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMultipleRules",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMultipleRules));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenCategoricalMatchRule",
        &CDetectionRulesJsonParserTest::testParseRulesGivenCategoricalMatchRule));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenCategoricalComplementRule",
        &CDetectionRulesJsonParserTest::testParseRulesGivenCategoricalComplementRule));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenTimeRule",
        &CDetectionRulesJsonParserTest::testParseRulesGivenTimeRule));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenDifferentActions",
        &CDetectionRulesJsonParserTest::testParseRulesGivenDifferentActions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenOldStyleCategoricalRule",
        &CDetectionRulesJsonParserTest::testParseRulesGivenOldStyleCategoricalRule));
    return suiteOfTests;
}

void CDetectionRulesJsonParserTest::testParseRulesGivenEmptyString() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);

    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenEmptyArray() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenArrayContainsStrings() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[\"a\", \"b\"]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);

    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMissingRuleAction() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"condition+type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenRuleActionIsNotArray() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":\"not_array\",";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenInvalidRuleAction() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"something_invalid\"],";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionsConnective() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionsConnective() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions_connective\":\"XOR\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMissingRuleConditions() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsNotArray() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": {}";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionOperator() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionOperator() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"ha\",\"value\":\"5\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenNumericalActualRuleWithConnectiveOr() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}},";
    rulesJson += "    {\"type\":\"numerical_actual\", \"field_name\":\"metric\", \"condition\":{\"operator\":\"lte\",\"value\":\"2.3\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF ACTUAL < 5.000000 OR ACTUAL(metric) <= 2.300000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenNumericalTypicalAndDiffAbsRuleWithConnectiveAnd() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"and\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_typical\", \"condition\":{\"operator\":\"gt\",\"value\":\"5\"}},";
    rulesJson += "    {\"type\":\"numerical_diff_abs\", \"field_name\":\"metric\", "
                 "\"field_value\":\"cpu\",\"condition\":{\"operator\":\"gte\",\"value\":\"2.3\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF TYPICAL > 5.000000 AND DIFF_ABS(metric:cpu) >= 2.300000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMultipleRules() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"target_field_name\":\"id\",";
    rulesJson += "  \"target_field_value\":\"foo\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"1\"}}";
    rulesJson += "  ]";
    rulesJson += "},";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_sampling\"],";
    rulesJson += "  \"conditions_connective\":\"and\",";
    rulesJson += "  \"target_field_name\":\"id\",";
    rulesJson += "  \"target_field_value\":\"42\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"2\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS (id:foo) IF ACTUAL < 1.000000"),
                         rules[0].print());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_SAMPLING (id:42) IF ACTUAL < 2.000000"),
                         rules[1].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenCategoricalMatchRule() {
    TStrPatternSetUMap filtersById;
    core::CPatternSet filter;
    filter.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"categorical_match\", \"field_name\":\"foo\", \"filter_id\":\"filter1\"}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF (foo) IN FILTER"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenOldStyleCategoricalRule() {
    // Tests that the rule type can be parsed as categorical_match
    // when the type is categorical

    TStrPatternSetUMap filtersById;
    core::CPatternSet filter;
    filter.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"categorical\", \"field_name\":\"foo\", \"filter_id\":\"filter1\"}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF (foo) IN FILTER"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenCategoricalComplementRule() {
    TStrPatternSetUMap filtersById;
    core::CPatternSet filter;
    filter.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"or\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"categorical_complement\", \"field_name\":\"foo\", \"filter_id\":\"filter1\"}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF (foo) NOT IN FILTER"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenTimeRule() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"filter_results\"],";
    rulesJson += "  \"conditions_connective\":\"and\",";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"type\":\"time\", \"condition\":{\"operator\":\"gte\",\"value\":\"5000\"}}";
    rulesJson += "    ,{\"type\":\"time\", \"condition\":{\"operator\":\"lt\",\"value\":\"10000\"}}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF TIME >= 5000.000000 AND TIME < 10000.000000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenDifferentActions() {
    {
        CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
        CDetectionRulesJsonParser::TDetectionRuleVec rules;
        std::string rulesJson = "[";
        rulesJson += "{";
        rulesJson += "  \"actions\":[\"filter_results\"],";
        rulesJson += "  \"conditions_connective\":\"and\",";
        rulesJson += "  \"conditions\": [";
        rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
        rulesJson += "  ]";
        rulesJson += "}";
        rulesJson += "]";

        CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
        CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS IF ACTUAL < 5.000000"),
                             rules[0].print());
    }

    {
        CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
        CDetectionRulesJsonParser::TDetectionRuleVec rules;
        std::string rulesJson = "[";
        rulesJson += "{";
        rulesJson += "  \"actions\":[\"skip_sampling\"],";
        rulesJson += "  \"conditions_connective\":\"and\",";
        rulesJson += "  \"conditions\": [";
        rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
        rulesJson += "  ]";
        rulesJson += "}";
        rulesJson += "]";

        CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
        CPPUNIT_ASSERT_EQUAL(std::string("SKIP_SAMPLING IF ACTUAL < 5.000000"),
                             rules[0].print());
    }

    {
        CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
        CDetectionRulesJsonParser::TDetectionRuleVec rules;
        std::string rulesJson = "[";
        rulesJson += "{";
        rulesJson += "  \"actions\":[\"skip_sampling\", \"filter_results\"],";
        rulesJson += "  \"conditions_connective\":\"and\",";
        rulesJson += "  \"conditions\": [";
        rulesJson += "    {\"type\":\"numerical_actual\", \"condition\":{\"operator\":\"lt\",\"value\":\"5\"}}";
        rulesJson += "  ]";
        rulesJson += "}";
        rulesJson += "]";

        CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
        CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF ACTUAL < 5.000000"),
                             rules[0].print());
    }
}
