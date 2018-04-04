/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDetectionRuleTest_h
#define INCLUDED_CDetectionRuleTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CDetectionRuleTest : public CppUnit::TestFixture {
public:
    void testApplyGivenCategoricalCondition();
    void testApplyGivenNumericalActualCondition();
    void testApplyGivenNumericalTypicalCondition();
    void testApplyGivenNumericalDiffAbsCondition();
    void testApplyGivenSingleSeriesModelAndConditionWithField();
    void testApplyGivenNoActualValueAvailable();
    void testApplyGivenDifferentSeriesAndIndividualModel();
    void testApplyGivenDifferentSeriesAndPopulationModel();
    void testApplyGivenMultipleConditionsWithOr();
    void testApplyGivenMultipleConditionsWithAnd();
    void testApplyGivenTargetFieldIsPartitionAndIndividualModel();
    void testApplyGivenTimeCondition();
    void testRuleActions();

    static CppUnit::Test* suite();

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CDetectionRuleTest_h
