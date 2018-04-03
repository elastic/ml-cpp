/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

