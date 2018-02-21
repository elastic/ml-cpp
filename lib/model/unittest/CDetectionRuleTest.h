/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDetectionRuleTest_h
#define INCLUDED_CDetectionRuleTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CDetectionRuleTest : public CppUnit::TestFixture
{
    public:
        void testApplyGivenCategoricalCondition(void);
        void testApplyGivenNumericalActualCondition(void);
        void testApplyGivenNumericalTypicalCondition(void);
        void testApplyGivenNumericalDiffAbsCondition(void);
        void testApplyGivenSingleSeriesModelAndConditionWithField(void);
        void testApplyGivenNoActualValueAvailable(void);
        void testApplyGivenDifferentSeriesAndIndividualModel(void);
        void testApplyGivenDifferentSeriesAndPopulationModel(void);
        void testApplyGivenMultipleConditionsWithOr(void);
        void testApplyGivenMultipleConditionsWithAnd(void);
        void testApplyGivenTargetFieldIsPartitionAndIndividualModel(void);
        void testApplyGivenTimeCondition(void);
        void testRuleActions(void);

        static CppUnit::Test *suite();
    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CDetectionRuleTest_h

