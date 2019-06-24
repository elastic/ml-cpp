/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CConcurrentWrapperTest.h"

#include <core/CConcurrentWrapper.h>
#include <core/CLogger.h>
#include <core/CMemoryUsage.h>
#include <core/CStaticThreadPool.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

using namespace ml;
using namespace core;

using TOStringStreamConcurrentWrapper = CConcurrentWrapper<std::ostringstream>;

// a low capacity wrapper with only 5 buckets for the queue, the 3 controls the wakeup of threads
using TOStringStreamLowCapacityConcurrentWrapper =
    CConcurrentWrapper<std::ostringstream, 5, 3>;

void CConcurrentWrapperTest::testBasic() {

    std::ostringstream stringStream;
    {
        TOStringStreamConcurrentWrapper wrappedStringStream(stringStream);

        wrappedStringStream([](std::ostream& o) {
            o << "Hello 1";
            o << " world 1\n";
        });
        wrappedStringStream([](std::ostream& o) {
            o << "Hello 2";
            o << " world 2\n";
        });
    }
    CPPUNIT_ASSERT_EQUAL(std::string("Hello 1 world 1\nHello 2 world 2\n"),
                         stringStream.str());
}

namespace {

void task(CConcurrentWrapper<std::ostringstream>& sink, size_t i, std::chrono::microseconds pause) {
    sink([i, pause](std::ostream& o) {
        o << "ta";
        std::this_thread::sleep_for(pause);
        o << "sk ";
        o << std::setw(5);
        o << i;
        o << "\n";
    });
}

void taskLowCapacityQueue(TOStringStreamLowCapacityConcurrentWrapper& sink,
                          size_t i,
                          std::chrono::microseconds pause) {
    sink([i, pause](std::ostream& o) {
        o << "ta";
        std::this_thread::sleep_for(pause);
        o << "sk ";
        o << std::setw(5);
        o << i;
        o << "\n";
    });
}
}

void CConcurrentWrapperTest::testThreads() {
    std::ostringstream stringStream;
    static const size_t MESSAGES(1500);

    {
        TOStringStreamConcurrentWrapper wrappedStringStream(stringStream);
        {
            core::CStaticThreadPool tp(10);
            for (size_t i = 0; i < MESSAGES; ++i) {
                tp.schedule([&wrappedStringStream, i] {
                    task(wrappedStringStream, i, std::chrono::microseconds(0));
                });
            }
        }
    }

    std::string output = stringStream.str();
    size_t numberOfLines = std::count(output.begin(), output.end(), '\n');

    CPPUNIT_ASSERT_EQUAL(MESSAGES, numberOfLines);
    CPPUNIT_ASSERT_EQUAL(11 * MESSAGES, output.size());

    for (size_t i = 0; i < MESSAGES; ++i) {
        CPPUNIT_ASSERT_EQUAL(std::string("task"), output.substr(11 * i, 4));
    }
}

void CConcurrentWrapperTest::testThreadsSlow() {
    std::ostringstream stringStream;
    static const size_t MESSAGES(50);

    {
        TOStringStreamConcurrentWrapper wrappedStringStream(stringStream);
        {
            core::CStaticThreadPool tp(2);
            for (size_t i = 0; i < MESSAGES; ++i) {
                tp.schedule([&wrappedStringStream, i] {
                    task(wrappedStringStream, i, std::chrono::microseconds(50));
                });
            }
        }
    }

    std::string output = stringStream.str();
    size_t numberOfLines = std::count(output.begin(), output.end(), '\n');

    CPPUNIT_ASSERT_EQUAL(MESSAGES, numberOfLines);
    CPPUNIT_ASSERT_EQUAL(11 * MESSAGES, output.size());

    for (size_t i = 0; i < MESSAGES; ++i) {
        CPPUNIT_ASSERT_EQUAL(std::string("task"), output.substr(11 * i, 4));
    }
}

void CConcurrentWrapperTest::testThreadsSlowLowCapacity() {
    std::ostringstream stringStream;
    static const size_t MESSAGES(50);

    {
        TOStringStreamLowCapacityConcurrentWrapper wrappedStringStream(stringStream);
        {
            core::CStaticThreadPool tp(2);
            for (size_t i = 0; i < MESSAGES; ++i) {
                tp.schedule([&wrappedStringStream, i] {
                    taskLowCapacityQueue(wrappedStringStream, i,
                                         std::chrono::microseconds(50));
                });
            }
        }
    }

    std::string output = stringStream.str();
    size_t numberOfLines = std::count(output.begin(), output.end(), '\n');

    CPPUNIT_ASSERT_EQUAL(MESSAGES, numberOfLines);
    CPPUNIT_ASSERT_EQUAL(11 * MESSAGES, output.size());

    for (size_t i = 0; i < MESSAGES; ++i) {
        CPPUNIT_ASSERT_EQUAL(std::string("task"), output.substr(11 * i, 4));
    }
}

void CConcurrentWrapperTest::testThreadsLowCapacity() {
    std::ostringstream stringStream;
    static const size_t MESSAGES(2500);

    {
        TOStringStreamLowCapacityConcurrentWrapper wrappedStringStream(stringStream);
        {
            core::CStaticThreadPool tp(8);
            for (size_t i = 0; i < MESSAGES; ++i) {
                tp.schedule([&wrappedStringStream, i] {
                    taskLowCapacityQueue(wrappedStringStream, i,
                                         std::chrono::microseconds(0));
                });
            }
        }
    }

    std::string output = stringStream.str();
    size_t numberOfLines = std::count(output.begin(), output.end(), '\n');

    CPPUNIT_ASSERT_EQUAL(MESSAGES, numberOfLines);
    CPPUNIT_ASSERT_EQUAL(11 * MESSAGES, output.size());

    for (size_t i = 0; i < MESSAGES; ++i) {
        CPPUNIT_ASSERT_EQUAL(std::string("task"), output.substr(11 * i, 4));
    }
}

void CConcurrentWrapperTest::testMemoryDebug() {
    CMemoryUsage mem;

    std::ostringstream stringStream;
    TOStringStreamConcurrentWrapper wrappedStringStream(stringStream);

    wrappedStringStream.debugMemoryUsage(mem.addChild());
    CPPUNIT_ASSERT_EQUAL(wrappedStringStream.memoryUsage(), mem.usage());
}

CppUnit::Test* CConcurrentWrapperTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CConcurrentWrapperTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentWrapperTest>(
        "CConcurrentWrapperTest::testBasic", &CConcurrentWrapperTest::testBasic));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentWrapperTest>(
        "CConcurrentWrapperTest::testThreads", &CConcurrentWrapperTest::testThreads));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentWrapperTest>(
        "CConcurrentWrapperTest::testThreadsSlow", &CConcurrentWrapperTest::testThreadsSlow));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentWrapperTest>(
        "CConcurrentWrapperTest::testThreadsSlowLowCapacity",
        &CConcurrentWrapperTest::testThreadsSlowLowCapacity));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentWrapperTest>(
        "CConcurrentWrapperTest::testThreadsLowCapacity",
        &CConcurrentWrapperTest::testThreadsLowCapacity));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentWrapperTest>(
        "CConcurrentWrapperTest::testMemoryDebug", &CConcurrentWrapperTest::testMemoryDebug));

    return suiteOfTests;
}
