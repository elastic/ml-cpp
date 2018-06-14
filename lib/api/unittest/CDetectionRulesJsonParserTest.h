/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDetectionRulesJsonParserTest_h
#define INCLUDED_CDetectionRulesJsonParserTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDetectionRulesJsonParserTest : public CppUnit::TestFixture {
public:
    void testParseRulesGivenEmptyString();
    void testParseRulesGivenEmptyArray();
    void testParseRulesGivenArrayContainsStrings();
    void testParseRulesGivenMissingRuleAction();
    void testParseRulesGivenRuleActionIsNotArray();
    void testParseRulesGivenInvalidRuleAction();
    void testParseRulesGivenNeitherScopeNorConditions();
    void testParseRulesGivenRuleConditionsIsNotArray();
    void testParseRulesGivenRuleConditionsIsEmptyArray();
    void testParseRulesGivenMissingConditionOperator();
    void testParseRulesGivenInvalidConditionOperator();
    void testParseRulesGivenConditionOnActual();
    void testParseRulesGivenConditionsOnTypicalAndDiffFromTypical();
    void testParseRulesGivenMultipleRules();
    void testParseRulesGivenScopeIsEmpty();
    void testParseRulesGivenIncludeScope();
    void testParseRulesGivenExcludeScope();
    void testParseRulesGivenMultipleScopedFields();
    void testParseRulesGivenScopeAndConditions();
    void testParseRulesGivenTimeRule();
    void testParseRulesGivenDifferentActions();
    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDetectionRulesJsonParserTest_h
