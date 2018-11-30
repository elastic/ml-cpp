/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CConcurrencyTest.h"

#include <core/CLogger.h>
#include <core/Concurrency.h>

#include <numeric>

using namespace ml;

namespace {
using TIntVec = std::vector<int>;

double throws() {
    throw std::runtime_error("don't run me");
};

[[noreturn]] void throws(int) {
    throw std::runtime_error("don't run me");
};

[[noreturn]] void throws(std::size_t) {
    throw std::runtime_error("don't run me");
};
}

void CConcurrencyTest::testAsyncWithExecutors() {

    core::stopDefaultAsyncExecutor();

    std::string tags[]{"sequential", "parallel"};

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t]);

        auto result =
            core::async(core::defaultAsyncExecutor(), []() { return 42; });
        CPPUNIT_ASSERT_EQUAL(42, result.get());

        result = core::async(core::defaultAsyncExecutor(),
                             [](int i) { return i; }, 43);
        CPPUNIT_ASSERT_EQUAL(43, result.get());

        result = core::async(core::defaultAsyncExecutor(),
                             [](int i, int j) { return i + j; }, 22, 22);
        CPPUNIT_ASSERT_EQUAL(44, result.get());

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CConcurrencyTest::testAsyncWithExecutorsAndExceptions() {

    core::stopDefaultAsyncExecutor();

    std::string tags[]{"sequential", "parallel"};

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t]);

        auto result = core::async(core::defaultAsyncExecutor(),
                                  static_cast<double (*)()>(throws));
        CPPUNIT_ASSERT_THROW_MESSAGE("don't run me", result.get(), std::runtime_error);

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CConcurrencyTest::testParallelForEachWithEmpty() {

    core::stopDefaultAsyncExecutor();

    std::vector<int> values;
    auto result = core::parallel_for_each(0, values.size(),
                                          core::bindRetrievableState(
                                              [&values](double& sum, std::size_t i) {
                                                  sum += static_cast<double>(values[i]);
                                              },
                                              0.0));
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, result.size());
    CPPUNIT_ASSERT_EQUAL(0.0, result[0].s_FunctionState);
}

void CConcurrencyTest::testParallelForEach() {

    core::stopDefaultAsyncExecutor();

    TIntVec values(10000);
    std::iota(values.begin(), values.end(), 0);
    double expected{std::accumulate(values.begin(), values.end(), 0.0)};

    std::string tags[]{"sequential", "parallel"};

    // Test sequential.
    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t] << " indices");

        auto results =
            core::parallel_for_each(0, values.size(),
                                    core::bindRetrievableState(
                                        [&values](double& sum, std::size_t i) {
                                            sum += static_cast<double>(values[i]);
                                        },
                                        0.0));

        if (t == 0) {
            CPPUNIT_ASSERT_EQUAL(std::size_t{1}, results.size());
        }
        double sum{0.0};
        for (const auto& result : results) {
            LOG_TRACE(<< "thread sum = " << result.s_FunctionState);
            sum += result.s_FunctionState;
        }

        LOG_DEBUG(<< "expected " << expected);
        LOG_DEBUG(<< "got      " << sum);
        CPPUNIT_ASSERT_EQUAL(expected, sum);

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t] << " iterators");

        auto results = core::parallel_for_each(
            values.begin(), values.end(),
            core::bindRetrievableState(
                [](double& sum, int value) { sum += static_cast<double>(value); }, 0.0));

        if (t == 0) {
            CPPUNIT_ASSERT_EQUAL(std::size_t{1}, results.size());
        }
        double sum{0.0};
        for (const auto& result : results) {
            LOG_TRACE(<< "thread sum = " << result.s_FunctionState);
            sum += result.s_FunctionState;
        }

        LOG_DEBUG(<< "expected " << expected);
        LOG_DEBUG(<< "got      " << sum);
        CPPUNIT_ASSERT_EQUAL(expected, sum);

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CConcurrencyTest::testParallelForEachWithExceptions() {

    core::stopDefaultAsyncExecutor();

    TIntVec values(1000);

    std::string tags[]{"sequential", "parallel"};

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t] << " indices");

        bool exceptionCaught{false};
        try {
            auto result = core::parallel_for_each(
                0, values.size(), static_cast<void (*)(std::size_t)>(throws));
        } catch (const std::runtime_error& e) {
            LOG_DEBUG(<< "Caught: " << e.what());
            CPPUNIT_ASSERT_EQUAL(std::string{"don't run me"}, std::string{e.what()});
            exceptionCaught = true;
        }

        CPPUNIT_ASSERT(exceptionCaught);

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t] << " iterators");

        bool exceptionCaught{false};
        try {
            auto result = core::parallel_for_each(
                values.begin(), values.end(), static_cast<void (*)(int)>(throws));
        } catch (const std::runtime_error& e) {
            LOG_DEBUG(<< "Caught: " << e.what());
            CPPUNIT_ASSERT_EQUAL(std::string{"don't run me"}, std::string{e.what()});
            exceptionCaught = true;
        }

        CPPUNIT_ASSERT(exceptionCaught);

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

CppUnit::Test* CConcurrencyTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CConcurrencyTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrencyTest>(
        "CConcurrencyTest::testAsyncWithExecutors", &CConcurrencyTest::testAsyncWithExecutors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrencyTest>(
        "CConcurrencyTest::testAsyncWithExecutorsAndExceptions",
        &CConcurrencyTest::testAsyncWithExecutorsAndExceptions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrencyTest>(
        "CConcurrencyTest::testParallelForEachWithEmpty",
        &CConcurrencyTest::testParallelForEachWithEmpty));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrencyTest>(
        "CConcurrencyTest::testParallelForEach", &CConcurrencyTest::testParallelForEach));
    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrencyTest>(
        "CConcurrencyTest::testParallelForEachWithExceptions",
        &CConcurrencyTest::testParallelForEachWithExceptions));

    return suiteOfTests;
}
