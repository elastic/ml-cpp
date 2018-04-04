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
    void testParseRulesGivenEmptyString();
    void testParseRulesGivenEmptyArray();
    void testParseRulesGivenArrayContainsStrings();
    void testParseRulesGivenMissingRuleAction();
    void testParseRulesGivenRuleActionIsNotArray();
    void testParseRulesGivenInvalidRuleAction();
    void testParseRulesGivenMissingConditionsConnective();
    void testParseRulesGivenInvalidConditionsConnective();
    void testParseRulesGivenMissingRuleConditions();
    void testParseRulesGivenRuleConditionsIsNotArray();
    void testParseRulesGivenMissingConditionOperator();
    void testParseRulesGivenInvalidConditionOperator();
    void testParseRulesGivenNumericalActualRuleWithConnectiveOr();
    void testParseRulesGivenNumericalTypicalAndDiffAbsRuleWithConnectiveAnd();
    void testParseRulesGivenMultipleRules();
    void testParseRulesGivenCategoricalRule();
    void testParseRulesGivenTimeRule();
    void testParseRulesGivenDifferentActions();
    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDetectionRulesJsonParserTest_h
