/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
        "CDetectionRulesJsonParserTest::testParseRulesGivenNeitherScopeNorConditions",
        &CDetectionRulesJsonParserTest::testParseRulesGivenNeitherScopeNorConditions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsNotArray",
        &CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsNotArray));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsEmptyArray",
        &CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsEmptyArray));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionOperator",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMissingConditionOperator));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionOperator",
        &CDetectionRulesJsonParserTest::testParseRulesGivenInvalidConditionOperator));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenConditionOnActual",
        &CDetectionRulesJsonParserTest::testParseRulesGivenConditionOnActual));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenConditionsOnTypicalAndDiffFromTypical",
        &CDetectionRulesJsonParserTest::testParseRulesGivenConditionsOnTypicalAndDiffFromTypical));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMultipleRules",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMultipleRules));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenScopeIsEmpty",
        &CDetectionRulesJsonParserTest::testParseRulesGivenScopeIsEmpty));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenIncludeScope",
        &CDetectionRulesJsonParserTest::testParseRulesGivenIncludeScope));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenExcludeScope",
        &CDetectionRulesJsonParserTest::testParseRulesGivenExcludeScope));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenMultipleScopedFields",
        &CDetectionRulesJsonParserTest::testParseRulesGivenMultipleScopedFields));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenScopeAndConditions",
        &CDetectionRulesJsonParserTest::testParseRulesGivenScopeAndConditions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenTimeRule",
        &CDetectionRulesJsonParserTest::testParseRulesGivenTimeRule));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRulesJsonParserTest>(
        "CDetectionRulesJsonParserTest::testParseRulesGivenDifferentActions",
        &CDetectionRulesJsonParserTest::testParseRulesGivenDifferentActions));
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
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0}";
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
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5}";
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
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenNeitherScopeNorConditions() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"]";
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
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": {}";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenRuleConditionsIsEmptyArray() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": []";
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
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"value\": 5.0}";
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
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"ha\",\"value\": 5.0}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenConditionOnActual() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0},";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lte\",\"value\": 2.3}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF ACTUAL < 5.000000 AND ACTUAL <= 2.300000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenConditionsOnTypicalAndDiffFromTypical() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"typical\", \"operator\":\"gt\",\"value\": 5.0},";
    rulesJson += "    {\"applies_to\":\"diff_from_typical\", \"operator\":\"gte\",\"value\": 2.3}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF TYPICAL > 5.000000 AND DIFF_FROM_TYPICAL >= 2.300000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMultipleRules() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 1.0}";
    rulesJson += "  ]";
    rulesJson += "},";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_model_update\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"typical\", \"operator\":\"lt\",\"value\": 2.0}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF ACTUAL < 1.000000"),
                         rules[0].print());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_MODEL_UPDATE IF TYPICAL < 2.000000"),
                         rules[1].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenScopeIsEmpty() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"scope\": {}";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules) == false);
    CPPUNIT_ASSERT(rules.empty());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenIncludeScope() {
    TStrPatternSetUMap filtersById;
    core::CPatternSet filter;
    filter.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"scope\": {";
    rulesJson += "    \"foo\": {\"filter_id\": \"filter1\", \"filter_type\": \"include\"}";
    rulesJson += "  }";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF 'foo' IN FILTER"), rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenExcludeScope() {
    TStrPatternSetUMap filtersById;
    core::CPatternSet filter;
    filter.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"scope\": {";
    rulesJson += "    \"foo\": {\"filter_id\": \"filter1\", \"filter_type\": \"exclude\"}";
    rulesJson += "  }";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF 'foo' NOT IN FILTER"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenMultipleScopedFields() {
    TStrPatternSetUMap filtersById;
    core::CPatternSet filter1;
    filter1.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter1;
    core::CPatternSet filter2;
    filter2.initFromJson("[\"c\"]");
    filtersById["filter2"] = filter2;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"scope\": {";
    rulesJson += "    \"foo\": {\"filter_id\": \"filter1\", \"filter_type\": \"include\"},";
    rulesJson += "    \"bar\": {\"filter_id\": \"filter2\", \"filter_type\": \"exclude\"}";
    rulesJson += "  }";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF 'foo' IN FILTER AND 'bar' NOT IN FILTER"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenScopeAndConditions() {
    TStrPatternSetUMap filtersById;
    core::CPatternSet filter;
    filter.initFromJson("[\"b\", \"a\"]");
    filtersById["filter1"] = filter;

    CDetectionRulesJsonParser parser(filtersById);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"scope\": {";
    rulesJson += "    \"foo\": {\"filter_id\": \"filter1\", \"filter_type\": \"include\"}";
    rulesJson += "  },";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 2.0}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF 'foo' IN FILTER AND ACTUAL < 2.000000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenTimeRule() {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"time\", \"operator\":\"gte\",\"value\": 5000.0},";
    rulesJson += "    {\"applies_to\":\"time\", \"operator\":\"lt\",\"value\": 10000.0}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
    CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF TIME >= 5000.000000 AND TIME < 10000.000000"),
                         rules[0].print());
}

void CDetectionRulesJsonParserTest::testParseRulesGivenDifferentActions() {
    {
        CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
        CDetectionRulesJsonParser::TDetectionRuleVec rules;
        std::string rulesJson = "[";
        rulesJson += "{";
        rulesJson += "  \"actions\":[\"skip_result\"],";
        rulesJson += "  \"conditions\": [";
        rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0}";
        rulesJson += "  ]";
        rulesJson += "}";
        rulesJson += "]";

        CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
        CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT IF ACTUAL < 5.000000"),
                             rules[0].print());
    }

    {
        CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
        CDetectionRulesJsonParser::TDetectionRuleVec rules;
        std::string rulesJson = "[";
        rulesJson += "{";
        rulesJson += "  \"actions\":[\"skip_model_update\"],";
        rulesJson += "  \"conditions\": [";
        rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0}";
        rulesJson += "  ]";
        rulesJson += "}";
        rulesJson += "]";

        CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
        CPPUNIT_ASSERT_EQUAL(std::string("SKIP_MODEL_UPDATE IF ACTUAL < 5.000000"),
                             rules[0].print());
    }

    {
        CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
        CDetectionRulesJsonParser::TDetectionRuleVec rules;
        std::string rulesJson = "[";
        rulesJson += "{";
        rulesJson += "  \"actions\":[\"skip_model_update\", \"skip_result\"],";
        rulesJson += "  \"conditions\": [";
        rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0}";
        rulesJson += "  ]";
        rulesJson += "}";
        rulesJson += "]";

        CPPUNIT_ASSERT(parser.parseRules(rulesJson, rules));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), rules.size());
        CPPUNIT_ASSERT_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF ACTUAL < 5.000000"),
                             rules[0].print());
    }
}
