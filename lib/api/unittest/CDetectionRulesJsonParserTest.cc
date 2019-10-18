/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CPatternSet.h>

#include <api/CDetectionRulesJsonParser.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

BOOST_AUTO_TEST_SUITE(CDetectionRulesJsonParserTest)

using namespace ml;
using namespace api;

namespace {
using TStrPatternSetUMap = CDetectionRulesJsonParser::TStrPatternSetUMap;
TStrPatternSetUMap EMPTY_VALUE_FILTER_MAP;
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenEmptyString) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);

    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenEmptyArray) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[]";

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenArrayContainsStrings) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[\"a\", \"b\"]";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);

    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenMissingRuleAction) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"conditions\": [";
    rulesJson += "    {\"applies_to\":\"actual\", \"operator\":\"lt\",\"value\": 5.0}";
    rulesJson += "  ]";
    rulesJson += "}";
    rulesJson += "]";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenRuleActionIsNotArray) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenInvalidRuleAction) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenNeitherScopeNorConditions) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"]";
    rulesJson += "}";
    rulesJson += "]";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenRuleConditionsIsNotArray) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": {}";
    rulesJson += "}";
    rulesJson += "]";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenRuleConditionsIsEmptyArray) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"conditions\": []";
    rulesJson += "}";
    rulesJson += "]";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenMissingConditionOperator) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenInvalidConditionOperator) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenConditionOnActual) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF ACTUAL < 5.000000 AND ACTUAL <= 2.300000"),
                      rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenConditionsOnTypicalAndDiffFromTypical) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF TYPICAL > 5.000000 AND DIFF_FROM_TYPICAL >= 2.300000"),
                      rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenMultipleRules) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(2), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF ACTUAL < 1.000000"), rules[0].print());
    BOOST_CHECK_EQUAL(std::string("SKIP_MODEL_UPDATE IF TYPICAL < 2.000000"),
                      rules[1].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenScopeIsEmpty) {
    CDetectionRulesJsonParser parser(EMPTY_VALUE_FILTER_MAP);
    CDetectionRulesJsonParser::TDetectionRuleVec rules;
    std::string rulesJson = "[";
    rulesJson += "{";
    rulesJson += "  \"actions\":[\"skip_result\"],";
    rulesJson += "  \"scope\": {}";
    rulesJson += "}";
    rulesJson += "]";

    BOOST_TEST(parser.parseRules(rulesJson, rules) == false);
    BOOST_TEST(rules.empty());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenIncludeScope) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF 'foo' IN FILTER"), rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenExcludeScope) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF 'foo' NOT IN FILTER"),
                      rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenMultipleScopedFields) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF 'foo' IN FILTER AND 'bar' NOT IN FILTER"),
                      rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenScopeAndConditions) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));

    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF 'foo' IN FILTER AND ACTUAL < 2.000000"),
                      rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenTimeRule) {
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

    BOOST_TEST(parser.parseRules(rulesJson, rules));
    BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
    BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF TIME >= 5000.000000 AND TIME < 10000.000000"),
                      rules[0].print());
}

BOOST_AUTO_TEST_CASE(testParseRulesGivenDifferentActions) {
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

        BOOST_TEST(parser.parseRules(rulesJson, rules));

        BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
        BOOST_CHECK_EQUAL(std::string("SKIP_RESULT IF ACTUAL < 5.000000"),
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

        BOOST_TEST(parser.parseRules(rulesJson, rules));

        BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
        BOOST_CHECK_EQUAL(std::string("SKIP_MODEL_UPDATE IF ACTUAL < 5.000000"),
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

        BOOST_TEST(parser.parseRules(rulesJson, rules));

        BOOST_CHECK_EQUAL(std::size_t(1), rules.size());
        BOOST_CHECK_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF ACTUAL < 5.000000"),
                          rules[0].print());
    }
}

BOOST_AUTO_TEST_SUITE_END()
