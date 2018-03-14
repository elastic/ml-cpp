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
#include "CMutexTest.h"

#include <core/CMutex.h>


CppUnit::Test *CMutexTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMutexTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMutexTest>(
                               "CMutexTest::testRecursive",
                               &CMutexTest::testRecursive) );

    return suiteOfTests;
}

void CMutexTest::testRecursive(void) {
    ml::core::CMutex mutex;

    mutex.lock();
    mutex.unlock();

    mutex.lock();
    mutex.lock();
    mutex.unlock();
    mutex.unlock();
}

