/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
