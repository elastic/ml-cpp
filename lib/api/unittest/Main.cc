/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestRunner.h>

#include "CAnomalyJobLimitTest.h"
#include "CAnomalyJobTest.h"
#include "CBackgroundPersisterTest.h"
#include "CBaseTokenListDataTyperTest.h"
#include "CCategoryExamplesCollectorTest.h"
#include "CConfigUpdaterTest.h"
#include "CCsvInputParserTest.h"
#include "CCsvOutputWriterTest.h"
#include "CDetectionRulesJsonParserTest.h"
#include "CFieldConfigTest.h"
#include "CFieldDataTyperTest.h"
#include "CForecastRunnerTest.h"
#include "CIoManagerTest.h"
#include "CJsonOutputWriterTest.h"
#include "CLengthEncodedInputParserTest.h"
#include "CLineifiedJsonInputParserTest.h"
#include "CLineifiedJsonOutputWriterTest.h"
#include "CLineifiedXmlInputParserTest.h"
#include "CModelPlotDataJsonWriterTest.h"
#include "CModelSnapshotJsonWriterTest.h"
#include "CMultiFileDataAdderTest.h"
#include "COutputChainerTest.h"
#include "CRestorePreviousStateTest.h"
#include "CResultNormalizerTest.h"
#include "CSingleStreamDataAdderTest.h"
#include "CStateRestoreStreamFilterTest.h"
#include "CStringStoreTest.h"
#include "CTokenListDataTyperTest.h"
#include "CTokenListReverseSearchCreatorTest.h"

int main(int argc, const char** argv) {
    ml::test::CTestRunner runner(argc, argv);

    runner.addTest(CAnomalyJobLimitTest::suite());
    runner.addTest(CAnomalyJobTest::suite());
    runner.addTest(CBackgroundPersisterTest::suite());
    runner.addTest(CBaseTokenListDataTyperTest::suite());
    runner.addTest(CCategoryExamplesCollectorTest::suite());
    runner.addTest(CConfigUpdaterTest::suite());
    runner.addTest(CCsvInputParserTest::suite());
    runner.addTest(CCsvOutputWriterTest::suite());
    runner.addTest(CDetectionRulesJsonParserTest::suite());
    runner.addTest(CFieldConfigTest::suite());
    runner.addTest(CFieldDataTyperTest::suite());
    runner.addTest(CForecastRunnerTest::suite());
    runner.addTest(CIoManagerTest::suite());
    runner.addTest(CJsonOutputWriterTest::suite());
    runner.addTest(CLengthEncodedInputParserTest::suite());
    runner.addTest(CLineifiedJsonInputParserTest::suite());
    runner.addTest(CLineifiedJsonOutputWriterTest::suite());
    runner.addTest(CLineifiedXmlInputParserTest::suite());
    runner.addTest(CModelPlotDataJsonWriterTest::suite());
    runner.addTest(CModelSnapshotJsonWriterTest::suite());
    runner.addTest(CMultiFileDataAdderTest::suite());
    runner.addTest(COutputChainerTest::suite());
    runner.addTest(CRestorePreviousStateTest::suite());
    runner.addTest(CResultNormalizerTest::suite());
    runner.addTest(CSingleStreamDataAdderTest::suite());
    runner.addTest(CStringStoreTest::suite());
    runner.addTest(CTokenListDataTyperTest::suite());
    runner.addTest(CTokenListReverseSearchCreatorTest::suite());

    return !runner.runTests();
}
