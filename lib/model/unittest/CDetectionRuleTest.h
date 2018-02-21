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

