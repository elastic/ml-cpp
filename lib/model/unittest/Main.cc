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
#include <test/CTestRunner.h>

#include "CAnnotatedProbabilityBuilderTest.h"
#include "CAnomalyDetectorModelConfigTest.h"
#include "CAnomalyScoreTest.h"
#include "CBucketQueueTest.h"
#include "CCountingModelTest.h"
#include "CDetectionRuleTest.h"
#include "CDetectorEqualizerTest.h"
#include "CDynamicStringIdRegistryTest.h"
#include "CEventRateAnomalyDetectorTest.h"
#include "CEventRateDataGathererTest.h"
#include "CEventRateModelTest.h"
#include "CEventRatePopulationDataGathererTest.h"
#include "CEventRatePopulationModelTest.h"
#include "CForecastModelPersistTest.h"
#include "CFunctionTypesTest.h"
#include "CGathererToolsTest.h"
#include "CHierarchicalResultsLevelSetTest.h"
#include "CHierarchicalResultsTest.h"
#include "CInterimBucketCorrectorTest.h"
#include "CLimitsTest.h"
#include "CMemoryUsageEstimatorTest.h"
#include "CMetricAnomalyDetectorTest.h"
#include "CMetricDataGathererTest.h"
#include "CMetricModelTest.h"
#include "CMetricPopulationDataGathererTest.h"
#include "CMetricPopulationModelTest.h"
#include "CModelDetailsViewTest.h"
#include "CModelMemoryTest.h"
#include "CModelToolsTest.h"
#include "CModelTypesTest.h"
#include "CProbabilityAndInfluenceCalculatorTest.h"
#include "CResourceLimitTest.h"
#include "CResourceMonitorTest.h"
#include "CRuleConditionTest.h"
#include "CSampleQueueTest.h"
#include "CStringStoreTest.h"
#include "CToolsTest.h"

int main(int argc, const char** argv) {
    ml::test::CTestRunner runner(argc, argv);
    runner.addTest(CAnnotatedProbabilityBuilderTest::suite());
    runner.addTest(CAnomalyDetectorModelConfigTest::suite());
    runner.addTest(CAnomalyScoreTest::suite());
    runner.addTest(CBucketQueueTest::suite());
    runner.addTest(CCountingModelTest::suite());
    runner.addTest(CDetectionRuleTest::suite());
    runner.addTest(CDetectorEqualizerTest::suite());
    runner.addTest(CDynamicStringIdRegistryTest::suite());
    runner.addTest(CEventRateAnomalyDetectorTest::suite());
    runner.addTest(CEventRateDataGathererTest::suite());
    runner.addTest(CEventRateModelTest::suite());
    runner.addTest(CEventRatePopulationDataGathererTest::suite());
    runner.addTest(CEventRatePopulationModelTest::suite());
    runner.addTest(CFunctionTypesTest::suite());
    runner.addTest(CForecastModelPersistTest::suite());
    runner.addTest(CGathererToolsTest::suite());
    runner.addTest(CHierarchicalResultsTest::suite());
    runner.addTest(CHierarchicalResultsLevelSetTest::suite());
    runner.addTest(CInterimBucketCorrectorTest::suite());
    runner.addTest(CLimitsTest::suite());
    runner.addTest(CMemoryUsageEstimatorTest::suite());
    runner.addTest(CMetricAnomalyDetectorTest::suite());
    runner.addTest(CMetricDataGathererTest::suite());
    runner.addTest(CMetricModelTest::suite());
    runner.addTest(CMetricPopulationDataGathererTest::suite());
    runner.addTest(CMetricPopulationModelTest::suite());
    runner.addTest(CModelDetailsViewTest::suite());
    runner.addTest(CModelMemoryTest::suite());
    runner.addTest(CModelToolsTest::suite());
    runner.addTest(CModelTypesTest::suite());
    runner.addTest(CProbabilityAndInfluenceCalculatorTest::suite());
    runner.addTest(CResourceLimitTest::suite());
    runner.addTest(CResourceMonitorTest::suite());
    runner.addTest(CRuleConditionTest::suite());
    runner.addTest(CSampleQueueTest::suite());
    runner.addTest(CStringStoreTest::suite());
    runner.addTest(CToolsTest::suite());

    return !runner.runTests();
}
