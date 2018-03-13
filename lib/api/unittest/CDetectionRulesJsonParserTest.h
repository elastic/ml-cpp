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
#ifndef INCLUDED_CDetectionRulesJsonParserTest_h
#define INCLUDED_CDetectionRulesJsonParserTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDetectionRulesJsonParserTest : public CppUnit::TestFixture {
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

#endif// INCLUDED_CDetectionRulesJsonParserTest_h
