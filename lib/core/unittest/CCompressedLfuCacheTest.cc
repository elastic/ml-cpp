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
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStaticThreadPool.h>
#include <core/CStopWatch.h>
#include <core/Constants.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace ml;

BOOST_AUTO_TEST_SUITE(CCompressedLfuCacheTest)

namespace {
using TDoubleVec = std::vector<double>;

class CTestValue {
public:
    CTestValue() = default;
    explicit CTestValue(std::size_t size) { m_Buffer.reserve(size); }

    std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(m_Buffer);
    }

private:
    TDoubleVec m_Buffer;
};

using TDoubleVecStrCache = core::CCompressedLfuCache<TDoubleVec, std::string>;
using TStrStrCache = core::CCompressedLfuCache<std::string, std::string>;
using TConcurrentStrStrCache = core::CConcurrentCompressedLfuCache<std::string, std::string>;
using TStrTestValueCache = core::CCompressedLfuCache<std::string, CTestValue>;
}

BOOST_AUTO_TEST_CASE(testLookup) {
    // Test we get the cache hits and misses we expect.

    TStrStrCache cache{32 * core::constants::BYTES_IN_KILOBYTES,
                       [](const TStrStrCache::TDictionary& dictionary, const std::string& key) {
                           return dictionary.word(key);
                       }};

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

    BOOST_TEST_REQUIRE(cache.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testMemoryUsage) {
    // Test we accurately control memory usage to the specified limit.

    TDoubleVecStrCache cache{32 * core::constants::BYTES_IN_KILOBYTES,
                             [](const TDoubleVecStrCache::TDictionary& dictionary,
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
        BOOST_TEST_REQUIRE(cache.memoryUsage() < 32 * core::constants::BYTES_IN_KILOBYTES);
    }
    BOOST_TEST_REQUIRE(cache.memoryUsage() >
                       static_cast<std::size_t>(0.98 * 32 * core::constants::BYTES_IN_KILOBYTES));

    BOOST_TEST_REQUIRE(cache.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testResize) {

    TDoubleVecStrCache cache{32 * core::constants::BYTES_IN_KILOBYTES,
                             [](const TDoubleVecStrCache::TDictionary& dictionary,
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
    BOOST_TEST_REQUIRE(cache.memoryUsage() >
                       static_cast<std::size_t>(0.98 * 32 * core::constants::BYTES_IN_KILOBYTES));
    BOOST_TEST_REQUIRE(cache.memoryUsage() < 32 * core::constants::BYTES_IN_KILOBYTES);
    BOOST_TEST_REQUIRE(cache.size() < 500);

    LOG_DEBUG(<< "32KB size = " << cache.size());
    LOG_DEBUG(<< "32KB memory usage = " << cache.memoryUsage());

    double size32KB{static_cast<double>(cache.size())};

    cache.resize(64 * core::constants::BYTES_IN_KILOBYTES);

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
    BOOST_TEST_REQUIRE(cache.memoryUsage() >
                       static_cast<std::size_t>(0.98 * 64 * core::constants::BYTES_IN_KILOBYTES));
    BOOST_TEST_REQUIRE(cache.memoryUsage() < 64 * core::constants::BYTES_IN_KILOBYTES);
    BOOST_TEST_REQUIRE(cache.size() > static_cast<std::size_t>(1.95 * size32KB));
    BOOST_TEST_REQUIRE(cache.size() < static_cast<std::size_t>(2.05 * size32KB));

    cache.resize(32 * core::constants::BYTES_IN_KILOBYTES);

    LOG_DEBUG(<< "32KB size = " << cache.size());
    LOG_DEBUG(<< "32KB memory usage = " << cache.memoryUsage());

    BOOST_TEST_REQUIRE(cache.memoryUsage() >
                       static_cast<std::size_t>(0.98 * 32 * core::constants::BYTES_IN_KILOBYTES));
    BOOST_TEST_REQUIRE(cache.memoryUsage() < 32 * core::constants::BYTES_IN_KILOBYTES);
    BOOST_TEST_REQUIRE(cache.size() > static_cast<std::size_t>(0.95 * size32KB));
    BOOST_TEST_REQUIRE(cache.size() < static_cast<std::size_t>(1.05 * size32KB));

    BOOST_TEST_REQUIRE(cache.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testConcurrentReadsAndWrites) {

    using TTaskVec = std::vector<std::function<void()>>;

    TConcurrentStrStrCache cache{
        32 * core::constants::BYTES_IN_KILOBYTES, std::chrono::milliseconds{50},
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

    BOOST_TEST_REQUIRE(cache.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testEvictionStrategyFrequency) {

    // Check that frequent items are retained.

    TStrStrCache cache{32 * core::constants::BYTES_IN_KILOBYTES,
                       [](const TStrStrCache::TDictionary& dictionary, const std::string& key) {
                           return dictionary.word(key);
                       }};

    test::CRandomNumbers rng;

    TDoubleVec u01;
    for (std::size_t i = 0; i < 5000; ++i) {
        rng.generateUniformSamples(0.0, 1.0, 1, u01);
        if (u01[0] < 0.2) {
            std::string key{"key_" + std::to_string(i % 10)};
            cache.lookup(key, // Don't move because we want to retain the value to compare.
                         [](std::string key_) { return key_; },
                         [&](const std::string& value) {
                             BOOST_REQUIRE_EQUAL(key, value);
                         });
        } else {
            std::string key{"key_" + std::to_string(i)};
            cache.lookup(key, // Don't move because we want to retain the value to compare.
                         [](std::string key_) { return key_; },
                         [&](const std::string& value) {
                             BOOST_REQUIRE_EQUAL(key, value);
                         });
        }
    }

    // The hit fraction should be around 20%.
    LOG_DEBUG(<< "hit fraction = " << cache.hitFraction());
    BOOST_TEST_REQUIRE(cache.hitFraction() > 0.19);

    // Make sure the cache is full.
    LOG_DEBUG(<< "size = " << cache.size());
    BOOST_TEST_REQUIRE(cache.size() < 5000);

    for (std::size_t i = 0; i < 10; ++i) {
        std::size_t count;
        std::tie(std::ignore, count) = cache.stats("key_" + std::to_string(i % 10));
        // Counts should be around 100 if items never churned.
        BOOST_REQUIRE_CLOSE_ABSOLUTE(std::size_t{100}, count, std::size_t{20});
    }

    BOOST_TEST_REQUIRE(cache.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testEvictionStrategyMemory) {

    // Check that large items get evicted first.

    TStrTestValueCache cache{
        32 * core::constants::BYTES_IN_KILOBYTES,
        [](const TStrStrCache::TDictionary& dictionary,
           const std::string& key) { return dictionary.word(key); }};

    for (std::size_t i = 0; i < 200; ++i) {
        if (i < 100) {
            cache.lookup("large_key_" + std::to_string(i),
                         [i](std::string) { return CTestValue{5 + i}; },
                         [&](const CTestValue&) {});
            auto stats = cache.stats("large_key_" + std::to_string(i));
            BOOST_TEST_REQUIRE(stats.first > 0);
            BOOST_TEST_REQUIRE(stats.second > 0);
        } else {
            cache.lookup("small_key_" + std::to_string(i),
                         [](std::string) { return CTestValue{}; },
                         [&](const CTestValue&) {});
        }
    }

    std::size_t largeHits{0};
    std::size_t smallHits{0};
    for (std::size_t i = 0; i < 200; ++i) {
        if (i < 100) {
            auto stats = cache.stats("large_key_" + std::to_string(i));
            largeHits += stats.second > 0 ? 1 : 0;
        } else {
            auto stats = cache.stats("small_key_" + std::to_string(i));
            smallHits += stats.second > 0 ? 1 : 0;
        }
    }

    LOG_DEBUG(<< "largeHits = " << largeHits << ", smallHits = " << smallHits);

    // Check we have mainly evicted large keys. The way we count hits means we
    // don't immediately only evict large keys.
    BOOST_TEST_REQUIRE(largeHits < 30);
    BOOST_REQUIRE_EQUAL(100, smallHits);

    BOOST_TEST_REQUIRE(cache.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testSingleThreadPerformance) {

    TStrStrCache cache{128 * core::constants::BYTES_IN_KILOBYTES,
                       [](const TStrStrCache::TDictionary& dictionary, const std::string& key) {
                           return dictionary.word(key);
                       }};

    TConcurrentStrStrCache concurrentCache{
        128 * core::constants::BYTES_IN_KILOBYTES, std::chrono::milliseconds{50},
        [](const TStrStrCache::TDictionary& dictionary,
           const std::string& key) { return dictionary.word(key); }};

    test::CRandomNumbers rng;

    TDoubleVec u01;
    rng.generateUniformSamples(0.0, 1.0, 50000, u01);

    core::CStopWatch watch{true};

    for (std::size_t i = 0; i < u01.size(); ++i) {
        if (u01[i] < 0.2) {
            cache.lookup("key_" + std::to_string(i % 10),
                         [](std::string key_) { return key_; },
                         [&](const std::string&) {});
        } else {
            cache.lookup("key_" + std::to_string(i),
                         [](std::string key_) { return key_; },
                         [&](const std::string&) {});
        }
    }

    std::uint64_t durationMs{watch.lap()};

    for (std::size_t i = 0; i < u01.size(); ++i) {
        if (u01[i] < 0.2) {
            concurrentCache.lookup("key_" + std::to_string(i % 10),
                                   [](std::string key_) { return key_; },
                                   [&](const std::string&) {});
        } else {
            concurrentCache.lookup("key_" + std::to_string(i),
                                   [](std::string key_) { return key_; },
                                   [&](const std::string&) {});
        }
    }

    std::uint64_t durationConcurrentMs{watch.stop() - durationMs};

    // No asserts because we don't want to depend on timing but this was about
    // 30% faster on bare metal (32 ms vs 44 ms).
    LOG_DEBUG(<< "duration ms = " << durationMs);
    LOG_DEBUG(<< "duration concurrent ms = " << durationConcurrentMs);
}

BOOST_AUTO_TEST_CASE(testPersist) {

    // Check that persist and restore is idempotent.

    TStrStrCache origCache{
        32 * core::constants::BYTES_IN_KILOBYTES,
        [](const TStrStrCache::TDictionary& dictionary,
           const std::string& key) { return dictionary.word(key); }};

    for (std::size_t i = 0; i < 500; ++i) {
        origCache.lookup("key_" + std::to_string(i),
                         [](std::string key_) { return key_; },
                         [&](const std::string&) {});
    }

    std::stringstream origJsonStream;
    {
        core::CJsonStatePersistInserter inserter{origJsonStream};
        origCache.acceptPersistInserter(inserter);
        origJsonStream.flush();
    }

    LOG_TRACE(<< "JSON representation is: " << origJsonStream.str());

    origJsonStream.seekg(0);
    core::CJsonStateRestoreTraverser traverser{origJsonStream};

    TStrStrCache restoredCache{
        32 * core::constants::BYTES_IN_KILOBYTES,
        [](const TStrStrCache::TDictionary& dictionary,
           const std::string& key) { return dictionary.word(key); }};
    BOOST_TEST_REQUIRE(restoredCache.acceptRestoreTraverser(traverser));

    for (std::size_t i = 0; i < 500; ++i) {
        auto origStats = origCache.stats("key_" + std::to_string(i));
        auto restoredStats = origCache.stats("key_" + std::to_string(i));
        BOOST_REQUIRE_EQUAL(origStats.first, restoredStats.first);
        BOOST_REQUIRE_EQUAL(origStats.second, restoredStats.second);
    }

    std::stringstream restoredJsonStream;
    {
        core::CJsonStatePersistInserter inserter{restoredJsonStream};
        origCache.acceptPersistInserter(inserter);
        restoredJsonStream.flush();
    }

    BOOST_REQUIRE_EQUAL(origJsonStream.str(), restoredJsonStream.str());

    BOOST_TEST_REQUIRE(origCache.checkInvariants());
    BOOST_TEST_REQUIRE(restoredCache.checkInvariants());
}

BOOST_AUTO_TEST_SUITE_END()
