/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestRunner.h>

#include "CBlockingCallCancellerThreadTest.h"
#include "CCommandProcessorTest.h"

int main(int argc, const char** argv) {
    ml::test::CTestRunner runner(argc, argv);

    runner.addTest(CBlockingCallCancellerThreadTest::suite());
    runner.addTest(CCommandProcessorTest::suite());

    return !runner.runTests();
}
