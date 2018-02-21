/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <test/CTestRunner.h>

#include "CAutoconfigurerParamsTest.h"
#include "CDataSemanticsTest.h"
#include "CDataSummaryStatisticsTest.h"
#include "CDetectorEnumeratorTest.h"
#include "CReportWriterTest.h"

int main(int argc, const char **argv)
{
    ml::test::CTestRunner runner(argc, argv);

    runner.addTest( CAutoconfigurerParamsTest::suite() );
    runner.addTest( CDataSemanticsTest::suite() );
    runner.addTest( CDataSummaryStatisticsTest::suite() );
    runner.addTest( CDetectorEnumeratorTest::suite() );
    runner.addTest( CReportWriterTest::suite() );

    return !runner.runTests();
}
