/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CProcessPriorityTest.h"

#include <core/CProcessPriority.h>


CppUnit::Test *CProcessPriorityTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CProcessPriorityTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CProcessPriorityTest>(
                                   "CProcessPriorityTest::testReducePriority",
                                   &CProcessPriorityTest::testReducePriority) );

    return suiteOfTests;
}

void CProcessPriorityTest::testReducePriority()
{
    ml::core::CProcessPriority::reducePriority();
}

