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

#include "CHashingTest.h"

#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CStopWatch.h>

#include <test/CRandomNumbers.h>

#include <boost/bind.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>
#include <boost/unordered_set.hpp>

using namespace ml;
using namespace core;

void CHashingTest::testUniversalHash(void) {
    // We test the definition:
    //   "for all (x,y) in U and x != y P(h(x) = h(y)) <= 1/m"
    //
    // must hold for a randomly selected member of the family. Note that
    // this is subtly different from requiring that this hold for every
    // member of the family, i.e. the probability of collision must be
    // <= 1/m for a sufficiently large proportion of hash functions. We'll
    // test a number of hash functions chosen at random from our family.
    //
    // The hash is also pairwise independent: i.e. for
    //   "for all (x,y) in U and x != y P(h(x) = u and h(y) = v) = 1/m^2
    //
    // must hold for a randomly selected member of the family.

    double tolerances[] = { 1.0, 1.6 };
    uint32_t m[] = { 30, 300 };
    uint32_t u[] = { 1000, 10000 };

    for (size_t i = 0; i < boost::size(m); ++i) {
        CHashing::CUniversalHash::TUInt32HashVec hashes;
        CHashing::CUniversalHash::generateHashes(10u, m[i], hashes);

        double collisionsRandom = 0.0;
        double hashedRandom = 0.0;

        for (std::size_t h = 0u; h < hashes.size(); ++h) {
            LOG_DEBUG("**** Testing hash = " << hashes[h].print() << " ****");

            CHashing::CUniversalHash::CUInt32Hash hash = hashes[h];

            for (size_t j = 0; j < boost::size(u); ++j) {
                uint32_t n = u[j];

                LOG_DEBUG("m = " << m[i] << ", U = [" << n << "]");

                uint32_t collisions = 0u;

                for (uint32_t x = 0; x < n; ++x) {
                    for (uint32_t y = x + 1u; y < n; ++y) {
                        uint32_t hx = hash(x);
                        uint32_t hy = hash(y);
                        if (hx == hy) {
                            ++collisions;
                        }
                    }
                }

                collisionsRandom += static_cast<double>(collisions);
                hashedRandom += static_cast<double>(n * (n - 1)) / 2.0;

                double pc = 2.0 * static_cast<double>(collisions)
                            / static_cast<double>(n * (n - 1));

                LOG_DEBUG("collisions = " << collisions
                          << ", P(collision) = " << pc
                          << ", 1/m = " << (1.0 / static_cast<double>(m[i])));

                // Note that the definition of universality doesn't require
                // the P(collision) <= 1/m for every hash function.
                CPPUNIT_ASSERT(pc < tolerances[i] * (1.0 / static_cast<double>(m[i])));
            }
        }

        double pcRandom = collisionsRandom / hashedRandom;
        LOG_DEBUG("random P(collision) = " << pcRandom);
        CPPUNIT_ASSERT(pcRandom < 1.0 / static_cast<double>(m[i]));
    }

    // We test a large u and m non exhaustively by choosing sufficient
    // different numbers of pairs to hash.

    typedef std::vector<uint32_t> TUInt32Vec;
    typedef std::pair<uint32_t, uint32_t> TUInt32Pr;
    typedef std::set<TUInt32Pr> TUInt32PrSet;
    typedef std::map<TUInt32Pr, unsigned int> TUint32PrUIntMap;
    typedef TUint32PrUIntMap::const_iterator TUint32PrUIntMapCItr;

    {
        LOG_DEBUG("**** m = " << 10000 << ", U = [" << 10000000 << "] ****");

        boost::random::mt11213b generator;
        boost::random::uniform_int_distribution<uint32_t> uniform(0u, 10000000u);

        TUInt32Vec samples;
        std::generate_n(std::back_inserter(samples), 1000u,
                        boost::bind(uniform, boost::ref(generator)));

        CHashing::CUniversalHash::TUInt32HashVec hashes;
        CHashing::CUniversalHash::generateHashes(100u, 10000u, hashes);

        double collisionsRandom = 0.0;
        double hashedRandom = 0.0;

        for (std::size_t h = 0u; h < hashes.size(); ++h) {
            LOG_DEBUG("Testing hash = " << hashes[h].print());

            CHashing::CUniversalHash::CUInt32Hash hash = hashes[h];

            uint32_t collisions = 0u;
            TUInt32PrSet uniquePairs;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                for (std::size_t j = i + 1u; j < samples.size(); ++j) {
                    if (samples[i] != samples[j] &&
                        uniquePairs.insert(TUInt32Pr(samples[i], samples[j])).second) {
                        uint32_t hx = hash(samples[i]);
                        uint32_t hy = hash(samples[j]);
                        if (hx == hy) {
                            ++collisions;
                        }
                    }
                }
            }

            collisionsRandom += static_cast<double>(collisions);
            hashedRandom += static_cast<double>(uniquePairs.size());

            double pc = static_cast<double>(collisions)
                        / static_cast<double>(uniquePairs.size());

            LOG_DEBUG("collisions = " << collisions
                      << ", P(collision) = " << pc
                      << ", 1/m = " << (1.0 / 10000.0));

            // Note that the definition of universality doesn't require
            // the P(collision) <= 1/m for every hash function.
            CPPUNIT_ASSERT(pc < 1.5 / 10000.0);
        }

        double pcRandom = collisionsRandom / hashedRandom;

        LOG_DEBUG("random P(collision) = " << pcRandom);

        CPPUNIT_ASSERT(pcRandom < 1.008 / 10000.0);
    }

    // To test 2-independence we keep counts of unique hash value pairs.
    // Note that this is a very strong property, it requires that all
    // counts of (h(x), h(y)) are equal so we'd only expect this to be
    // true in the limit we use a large number of hash functions. We test
    // this approximately with some tolerance on the counts.
    {
        LOG_DEBUG("Pairwise independence");

        CHashing::CUniversalHash::TUInt32HashVec hashes;
        CHashing::CUniversalHash::generateHashes(50u, 100u, hashes);

        TUint32PrUIntMap uniqueHashedPairs;

        for (std::size_t h = 0u; h < hashes.size(); ++h) {
            LOG_DEBUG("Testing hash = " << hashes[h].print());

            CHashing::CUniversalHash::CUInt32Hash hash = hashes[h];
            for (uint32_t x = 0u; x < 2000; ++x) {
                for (uint32_t y = x + 1u; y < 2000; ++y) {
                    uint32_t hx = hash(x);
                    uint32_t hy = hash(y);
                    ++uniqueHashedPairs[TUInt32Pr(hx, hy)];
                }
            }
        }

        double error = 0.0;

        for (TUint32PrUIntMapCItr i = uniqueHashedPairs.begin();
             i != uniqueHashedPairs.end();
             ++i) {
            double p = 2.0 * static_cast<double>(i->second)
                       / 2000.0 / 1999.0
                       / static_cast<double>(hashes.size());

            if (p > 1.0 / 10000.0) {
                LOG_DEBUG(core::CContainerPrinter::print(*i) << ", p = " << p);
                error += p - 1 / 10000.0;
            }
            CPPUNIT_ASSERT(p < 1.6 / 10000.0);
        }

        LOG_DEBUG("error = " << error);
        CPPUNIT_ASSERT(error < 0.03);
    }
}

void CHashingTest::testMurmurHash(void) {
    {
        std::string key("This is the voice of the Mysterons!");
        uint32_t seed = 0xdead4321;
        uint32_t result = CHashing::murmurHash32(key.c_str(), static_cast<int>(key.size()), seed);
        CPPUNIT_ASSERT_EQUAL(uint32_t(0xEE593473), result);
    }
    {
        std::string key("We know that you can hear us, Earthmen!");
        uint32_t seed = 0xffeeeeff;
        uint32_t result = CHashing::safeMurmurHash32(key.c_str(), static_cast<int>(key.size()), seed);
        CPPUNIT_ASSERT_EQUAL(uint32_t(0x54837c96), result);
    }
    {
        std::string key("Your message has been analysed and it has been decided to allow one member of Spectrum to meet our representative.");
        uint64_t seed = 0xaabbccddffeeeeffULL;
        uint64_t result = CHashing::murmurHash64(key.c_str(), static_cast<int>(key.size()), seed);
        CPPUNIT_ASSERT_EQUAL(uint64_t(14826751455157300659ull), result);
    }
    {
        std::string key("Earthmen, we are peaceful beings and you have tried to destroy us, but you cannot succeed. You and your people will pay for this act of aggression.");
        uint64_t seed = 0x1324fedc9876abdeULL;
        uint64_t result = CHashing::safeMurmurHash64(key.c_str(), static_cast<int>(key.size()), seed);
        CPPUNIT_ASSERT_EQUAL(uint64_t(7291323361835448266ull), result);
    }

    typedef std::vector<std::string> TStrVec;
    typedef std::map<std::size_t, std::size_t> TSizeSizeMap;
    typedef TSizeSizeMap::const_iterator TSizeSizeMapCItr;

    const std::size_t stringSize = 32u;
    const std::size_t numberStrings = 500000u;

    test::CRandomNumbers rng;
    TStrVec testStrings;
    rng.generateWords(stringSize, numberStrings, testStrings);

    core::CStopWatch stopWatch;
    uint64_t defaultInsertTime = 0;
    uint64_t defaultLookupTime = 0;
    uint64_t murmurInsertTime = 0;
    uint64_t murmurLookupTime = 0;
    for (int run = 0; run < 6; ++run) {
        LOG_DEBUG("run = " << run);

        LOG_DEBUG("Starting throughput of boost::unordered_set with default hash");
        {
            boost::unordered_set<std::string> s;
            stopWatch.reset(true);
            for (std::size_t i = 0u; i < testStrings.size(); ++i) {
                s.insert(testStrings[i]);
            }
            defaultInsertTime += stopWatch.stop();
            size_t total(0);
            stopWatch.reset(true);
            for (int i = 0; i < 5; ++i) {
                for (std::size_t j = 0u; j < testStrings.size(); ++j) {
                    total += s.count(testStrings[j]);
                }
            }
            defaultLookupTime += stopWatch.stop();
            CPPUNIT_ASSERT_EQUAL(testStrings.size() * 5, total);
        }
        LOG_DEBUG("Finished throughput of boost::unordered_set with default hash");

        LOG_DEBUG("Starting throughput of boost::unordered_set with murmur hash");
        {
            boost::unordered_set<std::string, CHashing::CMurmurHash2String> s;
            stopWatch.reset(true);
            for (std::size_t i = 0u; i < testStrings.size(); ++i) {
                s.insert(testStrings[i]);
            }
            murmurInsertTime += stopWatch.stop();
            size_t total(0);
            stopWatch.reset(true);
            for (int i = 0; i < 5; ++i) {
                for (std::size_t j = 0u; j < testStrings.size(); ++j) {
                    total += s.count(testStrings[j]);
                }
            }
            murmurLookupTime += stopWatch.stop();
            CPPUNIT_ASSERT_EQUAL(testStrings.size() * 5, total);
        }
        LOG_DEBUG("Finished throughput of boost::unordered_set with murmur hash");
    }

    LOG_DEBUG("default insert runtime = " << defaultInsertTime
              << "ms, murmur insert runtime = " << murmurInsertTime << "ms");
    LOG_DEBUG("default lookup runtime = " << defaultLookupTime
              << "ms, murmur lookup runtime = " << murmurLookupTime << "ms");

    // The benefits of the murmur hash are mainly at lookup time, so just assert
    // on that, but still log a warning for slower insert time
    if (murmurInsertTime > defaultInsertTime) {
        LOG_WARN("murmur insert runtime (" << murmurInsertTime <<
                 "ms) was longer than default insert runtime (" <<
                 defaultInsertTime << "ms)");
    }

    // Most of the times the murmur lookup time will be faster. But it is not
    // always the case. In order to avoid having failing builds but keep guarding
    // the performance, we give some slack (3%) to the comparison.
    CPPUNIT_ASSERT(murmurLookupTime < (defaultLookupTime * 103) / 100);

    // Check the number of collisions.
    TSizeSizeMap uniqueHashes;
    CHashing::CMurmurHash2String h;
    for (std::size_t i = 0u; i < testStrings.size(); ++i) {
        ++uniqueHashes[h(testStrings[i]) % 3000017];
    }

    std::size_t maxCollisions = 0u;
    for (TSizeSizeMapCItr i = uniqueHashes.begin(); i != uniqueHashes.end(); ++i) {
        maxCollisions = std::max(maxCollisions, i->second);
    }

    LOG_DEBUG("# unique \"hash(x) mod 3000017\" = " << uniqueHashes.size());
    LOG_DEBUG("Maximum number of collisions = " << maxCollisions);

    // The number of unique hashes varies for 32 bit and 64 bit versions of
    // this code, so this will need changing if we ever build 32 bit again.
    CPPUNIT_ASSERT_EQUAL(std::size_t(460438) /* for 64 bit code */, uniqueHashes.size());
    CPPUNIT_ASSERT(maxCollisions < 7);
}

void CHashingTest::testHashCombine(void) {
    // Check we get about the same number of collisions using hashCombine
    // verses full hash of string.

    CHashing::CMurmurHash2String hasher;

    typedef std::vector<std::string> TStrVec;
    typedef std::set<uint64_t> TSizeSet;

    const std::size_t stringSize = 32u;
    const std::size_t numberStrings = 2000000u;

    test::CRandomNumbers rng;

    TStrVec testStrings;
    TSizeSet uniqueHashes;
    TSizeSet uniqueHashCombines;
    for (std::size_t i = 0u; i < 5u; ++i) {
        LOG_DEBUG("test " << i);

        // This will overwrite the previous contents of testStrings
        rng.generateWords(stringSize, numberStrings, testStrings);

        uniqueHashes.clear();
        uniqueHashCombines.clear();

        for (std::size_t j = 0u; j < numberStrings; j += 2) {
            uniqueHashes.insert(hasher(testStrings[j] + testStrings[j + 1]));
            uniqueHashCombines.insert(core::CHashing::hashCombine(static_cast<uint64_t>(hasher(testStrings[j])),
                                                                  static_cast<uint64_t>(hasher(testStrings[j + 1]))));
        }

        LOG_DEBUG("# unique hashes          = " << uniqueHashes.size());
        LOG_DEBUG("# unique combined hashes = " << uniqueHashCombines.size());

        CPPUNIT_ASSERT(uniqueHashCombines.size()
                       > static_cast<std::size_t>(0.999 * static_cast<double>(uniqueHashes.size())));
    }
}

void CHashingTest::testConstructors(void) {
    {
        CHashing::CUniversalHash::CUInt32Hash hash(1, 2, 3);
        CPPUNIT_ASSERT_EQUAL(uint32_t(1), hash.m());
        CPPUNIT_ASSERT_EQUAL(uint32_t(2), hash.a());
        CPPUNIT_ASSERT_EQUAL(uint32_t(3), hash.b());
    }
    {
        CHashing::CUniversalHash::CUInt32UnrestrictedHash hash;
        CPPUNIT_ASSERT_EQUAL(uint32_t(1), hash.a());
        CPPUNIT_ASSERT_EQUAL(uint32_t(0), hash.b());
    }
    {
        CHashing::CUniversalHash::CUInt32UnrestrictedHash hash(3, 4);
        CPPUNIT_ASSERT_EQUAL(uint32_t(3), hash.a());
        CPPUNIT_ASSERT_EQUAL(uint32_t(4), hash.b());
        LOG_DEBUG(hash.print());
    }
    {
        CHashing::CUniversalHash::TUInt32Vec a;
        a.push_back(10);
        a.push_back(20);
        a.push_back(30);
        CHashing::CUniversalHash::CUInt32VecHash hash(5, a, 6);
        CPPUNIT_ASSERT_EQUAL(CContainerPrinter::print(a),
                             CContainerPrinter::print(hash.a()));
        CPPUNIT_ASSERT_EQUAL(uint32_t(5), hash.m());
        CPPUNIT_ASSERT_EQUAL(uint32_t(6), hash.b());
        LOG_DEBUG(hash.print());
    }
    {
        CHashing::CUniversalHash::CToString c(':');
        {
            CHashing::CUniversalHash::CUInt32UnrestrictedHash hash(55, 66);
            CPPUNIT_ASSERT_EQUAL(std::string("55:66"), c(hash));
        }
        {
            CHashing::CUniversalHash::CUInt32Hash hash(1111, 2222, 3333);
            CPPUNIT_ASSERT_EQUAL(std::string("1111:2222:3333"), c(hash));
        }
    }
    {
        CHashing::CUniversalHash::CFromString c(',');
        {
            CHashing::CUniversalHash::CUInt32UnrestrictedHash hash;
            CPPUNIT_ASSERT(c("2134,5432", hash));
            CPPUNIT_ASSERT_EQUAL(uint32_t(2134), hash.a());
            CPPUNIT_ASSERT_EQUAL(uint32_t(5432), hash.b());
        }
        {
            CHashing::CUniversalHash::CUInt32Hash hash;
            CPPUNIT_ASSERT(c("92134,54329,00000002", hash));
            CPPUNIT_ASSERT_EQUAL(uint32_t(92134), hash.m());
            CPPUNIT_ASSERT_EQUAL(uint32_t(54329), hash.a());
            CPPUNIT_ASSERT_EQUAL(uint32_t(2), hash.b());
        }
    }
    {
        CHashing::CUniversalHash::TUInt32UnrestrictedHashVec hashVec;
        CHashing::CUniversalHash::generateHashes(5, hashVec);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), hashVec.size());
    }
    {
        CHashing::CUniversalHash::TUInt32VecHashVec hashVec;
        CHashing::CUniversalHash::generateHashes(50, 6666, 7777, hashVec);
        CPPUNIT_ASSERT_EQUAL(std::size_t(50), hashVec.size());
    }
}


CppUnit::Test *CHashingTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CHashingTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CHashingTest>(
                               "CHashingTest::testUniversalHash",
                               &CHashingTest::testUniversalHash) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CHashingTest>(
                               "CHashingTest::testMurmurHash",
                               &CHashingTest::testMurmurHash) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CHashingTest>(
                               "CHashingTest::testHashCombine",
                               &CHashingTest::testHashCombine) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CHashingTest>(
                               "CHashingTest::testConstructors",
                               &CHashingTest::testConstructors) );
    return suiteOfTests;
}
