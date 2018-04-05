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

#include "CAutoconfigurerParamsTest.h"
#include "CDataSemanticsTest.h"
#include "CDataSummaryStatisticsTest.h"
#include "CDetectorEnumeratorTest.h"
#include "CReportWriterTest.h"

int main(int argc, const char** argv) {
    ml::test::CTestRunner runner(argc, argv);

    runner.addTest(CAutoconfigurerParamsTest::suite());
    runner.addTest(CDataSemanticsTest::suite());
    runner.addTest(CDataSummaryStatisticsTest::suite());
    runner.addTest(CDetectorEnumeratorTest::suite());
    runner.addTest(CReportWriterTest::suite());

    return !runner.runTests();
}
