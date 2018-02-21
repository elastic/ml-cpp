/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDetectionRulesJsonParserTest_h
#define INCLUDED_CDetectionRulesJsonParserTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDetectionRulesJsonParserTest : public CppUnit::TestFixture
{
    public:
        void testParseRulesGivenEmptyString(void);
        void testParseRulesGivenEmptyArray(void);
        void testParseRulesGivenArrayContainsStrings(void);
        void testParseRulesGivenMissingRuleAction(void);
        void testParseRulesGivenRuleActionIsNotArray(void);
        void testParseRulesGivenInvalidRuleAction(void);
        void testParseRulesGivenMissingConditionsConnective(void);
        void testParseRulesGivenInvalidConditionsConnective(void);
        void testParseRulesGivenMissingRuleConditions(void);
        void testParseRulesGivenRuleConditionsIsNotArray(void);
        void testParseRulesGivenMissingConditionOperator(void);
        void testParseRulesGivenInvalidConditionOperator(void);
        void testParseRulesGivenNumericalActualRuleWithConnectiveOr(void);
        void testParseRulesGivenNumericalTypicalAndDiffAbsRuleWithConnectiveAnd(void);
        void testParseRulesGivenMultipleRules(void);
        void testParseRulesGivenCategoricalRule(void);
        void testParseRulesGivenTimeRule(void);
        void testParseRulesGivenDifferentActions(void);
        static CppUnit::Test *suite();
};

#endif // INCLUDED_CDetectionRulesJsonParserTest_h
