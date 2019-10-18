/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CStaticThreadPool.h>
#include <core/CStopWatch.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <atomic>
#include <chrono>
#include <thread>

BOOST_AUTO_TEST_SUITE(CStaticThreadPoolTest)

using namespace ml;

namespace {
using TTaskVec = std::vector<std::function<void()>>;
void instantTask(std::atomic_uint& counter) {
    ++counter;
}
void fastTask(std::atomic_uint& counter) {
    std::chrono::milliseconds pause{1};
    std::this_thread::sleep_for(pause);
    ++counter;
}
void slowTask(std::atomic_uint& counter) {
    std::chrono::milliseconds pause{100};
    std::this_thread::sleep_for(pause);
    ++counter;
}
}

// ASSERTIONS BASED ON TIMINGS ARE NOT RELIABLE IN OUR VIRTUALISED TEST ENVIRONMENT
// SO ARE COMMENTED OUT. THESE VALUES CONSISTENTLY PASSED AT ONE POINT ON BARE METAL.
// IF YOU MAKE CHANGES TO THE THREAD POOL COMMENT THEM BACK IN AND CHECK THAT YOU
// HAVEN'T DEGRADED PERFORMANCE.

BOOST_AUTO_TEST_CASE(testScheduleDelayMinimisation) {

    // Check we have no delay in scheduling even if one thread is blocked.

    std::atomic_uint counter{0};
    {
        core::CStopWatch timeSchedule{true};

        core::CStaticThreadPool pool{2};

        TTaskVec tasks;
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(200, [&counter] { instantTask(counter); });

        core::CStopWatch watch{true};
        for (auto& task : tasks) {
            pool.schedule(std::move(task));
        }

        // We should get here fast because the unblocked thread should keep
        // draining the instant tasks even though they're added to both queues.

        uint64_t timeToSchedule{watch.stop()};
        LOG_DEBUG(<< "Time to schedule " << timeToSchedule);
        //BOOST_TEST(timeToSchedule <= 1);
    }
    BOOST_CHECK_EQUAL(200u, counter.load());
}

BOOST_AUTO_TEST_CASE(testThroughputStability) {

    // Check for stability of throughput.

    core::CStopWatch totalTimeWatch{true};
    std::atomic_uint counter{0};
    for (std::size_t t = 0; t < 5; ++t) {
        core::CStaticThreadPool pool{2};

        // Arrange for slow tasks to end in the same thread's queue.
        TTaskVec tasks;
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(110, [&counter] { fastTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(190, [&counter] { fastTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(350, [&counter] { fastTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(400, [&counter] { fastTask(counter); });

        core::CStopWatch watch{true};
        for (auto& task : tasks) {
            pool.schedule(std::move(task));
        }

        // The fastest we can get here is 300ms since the last 100 items scheduled
        // contain 200ms of delay and these are the tasks we can hold in the queues.
        // So if we're perfectly balanced we have (800 - 200) / 2 = 300ms of delay
        // to process before we can finish scheduling. We give ourselves a little
        // head room.

        uint64_t timeToSchedule{watch.stop()};
        LOG_DEBUG(<< "Time to schedule " << timeToSchedule);
        //BOOST_TEST(timeToSchedule >= 330);
        //BOOST_TEST(timeToSchedule <= 350);
    }

    BOOST_CHECK_EQUAL(2000u, counter.load());

    // The best we can achieve is 2000ms ignoring all overheads.
    std::uint64_t totalTime{totalTimeWatch.stop()};
    LOG_DEBUG(<< "Total time = " << totalTime);
    //BOOST_TEST(totalTime <= 2400);
}

BOOST_AUTO_TEST_CASE(testManyTasksThroughput) {

    // Check overheads for many instant tasks.

    std::atomic_uint counter{0};
    core::CStopWatch watch{true};
    {
        core::CStaticThreadPool pool{2};

        TTaskVec tasks;
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(2000, [&counter] { instantTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(2500, [&counter] { fastTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(2800, [&counter] { instantTask(counter); });
        tasks.push_back([&counter] { slowTask(counter); });
        tasks.resize(3000, [&counter] { fastTask(counter); });
        tasks.resize(10000, [&counter] { instantTask(counter); });

        for (auto& task : tasks) {
            pool.schedule(std::move(task));
        }
    }

    BOOST_CHECK_EQUAL(10000u, counter.load());

    // We have 1400ms of delays so the best we can achieve here is 700ms elapsed.
    std::uint64_t totalTime{watch.stop()};
    LOG_DEBUG(<< "Total time = " << totalTime);
    //BOOST_TEST(totalTime <= 780);
}

BOOST_AUTO_TEST_CASE(testSchedulingOverhead) {

    // Test the overhead per task is less than 0.7 microseconds.

    core::CStaticThreadPool pool{4};

    core::CStopWatch watch{true};
    for (std::size_t i = 0; i < 2000000; ++i) {
        if (i % 100000 == 0) {
            LOG_DEBUG(<< i);
        }
        pool.schedule([ j = std::size_t{0}, count = i % 1000 ]() mutable {
            for (j = 0; j < count; ++j) {
            }
        });
    }

    double overhead{static_cast<double>(watch.stop()) / 1000.0};
    LOG_DEBUG(<< "Total time = " << overhead);
    //BOOST_TEST(overhead < 1.4);
}

BOOST_AUTO_TEST_CASE(testWithExceptions) {

    // Check we don't deadlock because we don't kill worker threads if we do stupid
    // things.

    std::atomic_uint counter{0};
    {
        core::CStaticThreadPool pool{2};
        for (std::size_t i = 0; i < 5; ++i) {
            core::CStaticThreadPool::TTask null;
            pool.schedule(std::move(null));
            auto throws = []() { throw std::runtime_error{"bad"}; };
            pool.schedule(throws);
        }

        // This would dealock due to lack of consumers if we'd killed our workers.
        for (std::size_t i = 0; i < 200; ++i) {
            pool.schedule([&counter] { instantTask(counter); });
        }
    }

    // We didn't lose any real tasks.
    BOOST_CHECK_EQUAL(200u, counter.load());
}


BOOST_AUTO_TEST_SUITE_END()
