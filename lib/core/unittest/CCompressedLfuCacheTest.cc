/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CCompressedLfuCache.h>
#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStaticThreadPool.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace ml;

BOOST_AUTO_TEST_SUITE(CCompressedLfuCacheTest)

using TDoubleVec = std::vector<double>;
using TDoubleVecStrCache = core::CCompressedLfuCache<TDoubleVec, std::string>;
using TStrStrCache = core::CCompressedLfuCache<std::string, std::string>;
using TConcurrentStrStrCache = core::CConcurrentCompressedLfuCache<std::string, std::string>;

BOOST_AUTO_TEST_CASE(testLookup) {
    // Test we get the cache hits and misses we expect.

    TStrStrCache cache{
        32 * 1024, [](const TStrStrCache::TDictionary& dictionary,
                      const std::string& key) { return dictionary.word(key); }};

    BOOST_REQUIRE_EQUAL(
        false, cache.lookup("key_1", [](std::string key_) { return key_; },
                            [&](const std::string& value) {
                                BOOST_REQUIRE_EQUAL("key_1", value);
                            }));
    BOOST_REQUIRE_EQUAL(1, cache.size());
    BOOST_REQUIRE_EQUAL(0.0, cache.hitFraction());

    BOOST_REQUIRE_EQUAL(true, cache.lookup("key_1",
                                           [](std::string key_) { return key_; },
                                           [&](const std::string& value) {
                                               BOOST_REQUIRE_EQUAL("key_1", value);
                                           }));
    BOOST_REQUIRE_EQUAL(1, cache.size());
    BOOST_REQUIRE_EQUAL(1.0 / 2.0, cache.hitFraction());

    BOOST_REQUIRE_EQUAL(
        false, cache.lookup("key_2", [](std::string key_) { return key_; },
                            [&](const std::string& value) {
                                BOOST_REQUIRE_EQUAL("key_2", value);
                            }));
    BOOST_REQUIRE_EQUAL(2, cache.size());
    BOOST_REQUIRE_EQUAL(1.0 / 3.0, cache.hitFraction());
}

BOOST_AUTO_TEST_CASE(testMemoryUsage) {
    // Test we accurately control memory usage to the specified limit.

    TDoubleVecStrCache cache{32 * 1024, [](const TDoubleVecStrCache::TDictionary& dictionary,
                                           const TDoubleVec& key) {
                                 auto translator = dictionary.translator();
                                 translator.add(key);
                                 return translator.word();
                             }};

    test::CRandomNumbers rng;

    TDoubleVec key;
    for (std::size_t i = 0; i < 500; ++i) {
        rng.generateNormalSamples(0.0, 1.0, 5, key);
        cache.lookup(
            key, // Don't move because we want to retain the value to compare.
            [](TDoubleVec key_) { return core::CContainerPrinter::print(key_); },
            [&](const std::string& value) {
                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(key), value);
            });
        BOOST_REQUIRE_EQUAL(0.0, cache.hitFraction());
        BOOST_TEST_REQUIRE(cache.size() <= i + 1);
        BOOST_TEST_REQUIRE(cache.memoryUsage() < 32 * 1024);
    }
    BOOST_TEST_REQUIRE(cache.memoryUsage() > static_cast<std::size_t>(0.98 * 32 * 1024));
}

BOOST_AUTO_TEST_CASE(testResize) {

    TDoubleVecStrCache cache{32 * 1024, [](const TDoubleVecStrCache::TDictionary& dictionary,
                                           const TDoubleVec& key) {
                                 auto translator = dictionary.translator();
                                 translator.add(key);
                                 return translator.word();
                             }};

    test::CRandomNumbers rng;

    // Add enough values to fill the cache.
    TDoubleVec key;
    for (std::size_t i = 0; i < 500; ++i) {
        rng.generateNormalSamples(0.0, 1.0, 5, key);
        cache.lookup(
            key, // Don't move because we want to retain the value to compare.
            [](TDoubleVec key_) { return core::CContainerPrinter::print(key_); },
            [&](const std::string& value) {
                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(key), value);
            });
    }
    BOOST_TEST_REQUIRE(cache.memoryUsage() > static_cast<std::size_t>(0.98 * 32 * 1024));
    BOOST_TEST_REQUIRE(cache.memoryUsage() < 32 * 1024);
    BOOST_TEST_REQUIRE(cache.size() < 500);

    LOG_DEBUG(<< "32KB size = " << cache.size());
    LOG_DEBUG(<< "32KB memory usage = " << cache.memoryUsage());

    double size32KB{static_cast<double>(cache.size())};

    cache.resize(64 * 1024);

    for (std::size_t i = 0; i < 500; ++i) {
        rng.generateNormalSamples(0.0, 1.0, 5, key);
        cache.lookup(
            key, // Don't move because we want to retain the value to compare.
            [](TDoubleVec key_) { return core::CContainerPrinter::print(key_); },
            [&](const std::string& value) {
                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(key), value);
            });
    }

    LOG_DEBUG(<< "64KB size = " << cache.size());
    LOG_DEBUG(<< "64KB memory usage = " << cache.memoryUsage());

    // Check the cache is around twice as large.
    BOOST_TEST_REQUIRE(cache.memoryUsage() > static_cast<std::size_t>(0.98 * 64 * 1024));
    BOOST_TEST_REQUIRE(cache.memoryUsage() < 64 * 1024);
    BOOST_TEST_REQUIRE(cache.size() > static_cast<std::size_t>(1.95 * size32KB));
    BOOST_TEST_REQUIRE(cache.size() < static_cast<std::size_t>(2.05 * size32KB));

    cache.resize(32 * 1024);

    LOG_DEBUG(<< "32KB size = " << cache.size());
    LOG_DEBUG(<< "32KB memory usage = " << cache.memoryUsage());

    BOOST_TEST_REQUIRE(cache.memoryUsage() > static_cast<std::size_t>(0.98 * 32 * 1024));
    BOOST_TEST_REQUIRE(cache.memoryUsage() < 32 * 1024);
    BOOST_TEST_REQUIRE(cache.size() > static_cast<std::size_t>(0.95 * size32KB));
    BOOST_TEST_REQUIRE(cache.size() < static_cast<std::size_t>(1.05 * size32KB));
}

BOOST_AUTO_TEST_CASE(testConcurrentReadsAndWrites) {

    using TTaskVec = std::vector<std::function<void()>>;

    TConcurrentStrStrCache cache{
        32 * 1024, std::chrono::milliseconds{50},
        [](const TStrStrCache::TDictionary& dictionary,
                      const std::string& key) { return dictionary.word(key); }};

    std::atomic<std::size_t> errorCount{0};

    TTaskVec tasks;
    tasks.resize(10, [&] {
        cache.lookup("a_long_key_0", [](std::string key) { return key; },
                     [&](const std::string& value) {
                         if (value != "a_long_key_0") {
                             ++errorCount;
                         }
                     });
    });
    for (std::size_t i = 1; i < 100; ++i) {
        std::string key{"a_long_key_" + std::to_string((i % 10) + 1)};
        tasks.push_back([&cache, &errorCount, key, i] {
            cache.lookup(key,
                         [&](std::string key_) {
                             std::chrono::milliseconds pause{1 + i / 10};
                             std::this_thread::sleep_for(pause);
                             return key_;
                         },
                         [&](const std::string& value) {
                             if (value != key) {
                                 ++errorCount;
                             }
                         });
        });
    }

    // We destruct the thread pool to make sure we've processed all the tasks.
    {
        core::CStaticThreadPool pool{4};
        for (auto& task : tasks) {
            pool.schedule(std::move(task));
        }
    }

    BOOST_REQUIRE_EQUAL(11, cache.size());
    BOOST_REQUIRE_EQUAL(0, errorCount.load());
    // This should be around 90% but the value depends on timing so don't assert.
    LOG_DEBUG(<< "hit fraction = " << cache.hitFraction());

    for (std::size_t i = 11; i <= 1000; ++i) {
        std::string key{"a_long_key_" + std::to_string(i < 900 ? i : i % 11)};
        tasks.push_back([&cache, &errorCount, key, i] {
            cache.lookup(key,
                         [&](std::string key_) {
                             std::chrono::milliseconds pause{1 + i / 50};
                             std::this_thread::sleep_for(pause);
                             return key_;
                         },
                         [&](const std::string& value) {
                             if (value != key) {
                                 ++errorCount;
                             }
                         });
        });
    }

    {
        core::CStaticThreadPool pool{4};
        for (auto& task : tasks) {
            pool.schedule(std::move(task));
        }
    }

    BOOST_REQUIRE_EQUAL(0, errorCount.load());

    // This should be around 20% but the value depends on timing so don't assert.
    LOG_DEBUG(<< "hit fraction = " << cache.hitFraction());
}

BOOST_AUTO_TEST_CASE(testEvictionStrategyFrequency) {

    // Check that frequent items are retained.

    test::CRandomNumbers rng;
    
    //TDoubleVec 
    //for (std::size_t i = 0; i < 1000; ++i) {

    //}
}

BOOST_AUTO_TEST_CASE(testEvictionStrategySize) {

    // Check that large items get evicted first.

}

BOOST_AUTO_TEST_CASE(testPersist) {
}

BOOST_AUTO_TEST_SUITE_END()
