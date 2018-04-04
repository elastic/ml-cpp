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
#include "CThreadPoolTest.h"

#include <core/CLogger.h>

#include <boost/threadpool.hpp>

CppUnit::Test* CThreadPoolTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CThreadPoolTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CThreadPoolTest>("CThreadPoolTest::testPool", &CThreadPoolTest::testPool));

    return suiteOfTests;
}

namespace {

void first_task() {
    LOG_DEBUG("first task is running");
}

void second_task() {
    LOG_DEBUG("second task is running");
}
}

void CThreadPoolTest::testPool() {
    // Create fifo thread pool container with two threads.
    boost::threadpool::pool tp(2);

    // Add some tasks to the pool.
    tp.schedule(&first_task);
    tp.schedule(&second_task);

    // Wait until all tasks are finished.
    tp.wait();
}
