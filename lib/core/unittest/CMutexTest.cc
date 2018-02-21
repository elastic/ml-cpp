/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMutexTest.h"

#include <core/CMutex.h>


CppUnit::Test *CMutexTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMutexTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMutexTest>(
                                   "CMutexTest::testRecursive",
                                   &CMutexTest::testRecursive) );

    return suiteOfTests;
}

void CMutexTest::testRecursive(void)
{
    ml::core::CMutex mutex;

    mutex.lock();
    mutex.unlock();

    mutex.lock();
    mutex.lock();
    mutex.unlock();
    mutex.unlock();
}

