/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CConcurrentQueueTest.h"

#include <core/CConcurrentQueue.h>

#include <random>
#include <thread>
#include <vector>

using namespace ml;

namespace {
double work(std::size_t iterations, double x) {
    double result{0.0};
    for (std::size_t i = 0; i < iterations; ++i) {
        result += std::sin(x);
    }
    return result;
}
}

void CConcurrentQueueTest::testStressful() {

    // Test with fuzzing of work performed by the producer and consumer threads
    // between push and pop. (Note this uses random_device so produces different
    // work characteristics in every run.)

    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TQueue = core::CConcurrentQueue<TSizeDoublePr, 50>;
    using TSizeDoublePrVecVec = std::vector<std::vector<TSizeDoublePr>>;
    using TThreadVec = std::vector<std::thread>;

    TQueue queue;

    TThreadVec threads;
    threads.reserve(8);

    for (std::size_t i = 0; i < 4; ++i) {
        threads.emplace_back(
            [&](std::size_t producer) {
                unsigned int seed{static_cast<unsigned int>(producer)};
                try {
                    // Use "non-deterministic" seed if it is available.
                    std::random_device rd;
                    seed = rd();
                } catch (...) {
                    // Carry on regardless.
                }

                std::mt19937 rng{seed};
                std::uniform_int_distribution<int> uniform{10, 500};
                for (std::size_t j = 0; j < 1000; ++j) {
                    double result{work(uniform(rng), 0.1 * static_cast<double>(j))};
                    queue.pushEmplace(1000 * producer + j, result);
                }
            },
            i);
    }

    TSizeDoublePrVecVec results(4);

    for (std::size_t i = 0; i < 4; ++i) {
        threads.emplace_back(
            [&](std::size_t consumer) {
                unsigned int seed{static_cast<unsigned int>(consumer)};
                try {
                    // Use "non-deterministic" seed if it is available.
                    std::random_device rd;
                    seed = rd();
                } catch (...) {
                    // Carry on regardless.
                }

                std::mt19937 rng{seed};
                std::uniform_int_distribution<int> uniform{10, 500};
                for (std::size_t j = 0; j < 1000; ++j) {
                    std::size_t id;
                    double result;
                    std::tie(id, result) = queue.pop();
                    result += work(uniform(rng), 0.1 * static_cast<double>(j));
                    results[consumer].emplace_back(id, result);
                }
            },
            i);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Test we got exactly the work ids we added.

    TSizeVec ids;
    ids.reserve(4000);
    for (const auto& result : results) {
        for (const auto& id : result) {
            ids.push_back(id.first);
        }
    }
    std::sort(ids.begin(), ids.end());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        CPPUNIT_ASSERT_EQUAL(i, ids[i]);
    }
}

CppUnit::Test* CConcurrentQueueTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CConcurrentQueueTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CConcurrentQueueTest>(
        "CConcurrentQueueTest::testStressful", &CConcurrentQueueTest::testStressful));

    return suiteOfTests;
}
