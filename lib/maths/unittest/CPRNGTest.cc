/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CPRNG.h>
#include <maths/CStatisticalTests.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CPRNGTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testSplitMix64) {
    maths::CPRNG::CSplitMix64 rng1;

    boost::uniform_01<> u01;
    boost::normal_distribution<> norm(4.0, 10.0);

    // Test min and max.
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> min;
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1, std::greater<uint64_t>> max;
    for (std::size_t i = 0; i < 10000; ++i) {
        uint64_t x = rng1();
        min.add(x);
        max.add(x);
    }
    LOG_DEBUG(<< "min = " << min[0] << ", max = " << max[0]);
    BOOST_TEST_REQUIRE(
        min[0] < (maths::CPRNG::CSplitMix64::max() - maths::CPRNG::CSplitMix64::min()) / 2000);
    BOOST_TEST_REQUIRE(max[0] > maths::CPRNG::CSplitMix64::max() -
                                    (maths::CPRNG::CSplitMix64::max() -
                                     maths::CPRNG::CSplitMix64::min()) /
                                        2000);

    // Test generate.
    maths::CPRNG::CSplitMix64 rng2 = rng1;
    uint64_t samples1[50] = {0u};
    rng1.generate(&samples1[0], &samples1[50]);
    uint64_t samples2[50] = {0u};
    for (std::size_t i = 0; i < 50; ++i) {
        samples2[i] = rng2();
    }
    BOOST_TEST_REQUIRE(std::equal(&samples1[0], &samples1[50], &samples2[0]));

    // Test distribution.
    {
        boost::random::mt19937_64 mt;
        double p1[50] = {0.0};
        double p2[50] = {0.0};
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0; t < 50; ++t) {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0; i < 5000; ++i) {
                cvm1.addF(u01(rng1));
                cvm2.addF(u01(mt));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG(<< "p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG(<< "p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG(<< "m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG(<< "m2 = " << maths::CBasicStatistics::mean(m2));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(m1) >
                           0.95 * maths::CBasicStatistics::mean(m2));
    }
    {
        boost::random::mt11213b mt;
        boost::math::normal_distribution<> n410(4.0, 10.0);
        double p1[50] = {0.0};
        double p2[50] = {0.0};
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0; t < 50; ++t) {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0; i < 5000; ++i) {
                cvm1.addF(boost::math::cdf(n410, norm(rng1)));
                cvm2.addF(boost::math::cdf(n410, norm(mt)));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG(<< "p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG(<< "p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG(<< "m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG(<< "m2 = " << maths::CBasicStatistics::mean(m2));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(m1) >
                           0.95 * maths::CBasicStatistics::mean(m2));
    }

    // Test discard.
    maths::CPRNG::CSplitMix64 rng3 = rng1;
    rng1.discard(10);
    for (std::size_t i = 0; i < 10; ++i) {
        rng3();
    }
    for (std::size_t t = 0; t < 500; ++t) {
        BOOST_REQUIRE_EQUAL(rng1(), rng3());
    }

    // Test serialization.
    std::string state = rng1.toString();
    LOG_DEBUG(<< "state = " << state);
    maths::CPRNG::CSplitMix64 rng4;
    BOOST_TEST_REQUIRE(rng4.fromString(state));
    for (std::size_t t = 0; t < 500; ++t) {
        BOOST_REQUIRE_EQUAL(rng1(), rng4());
    }
}

BOOST_AUTO_TEST_CASE(testXorOShiro128Plus) {
    maths::CPRNG::CXorOShiro128Plus rng1;

    boost::uniform_01<> u01;
    boost::normal_distribution<> norm(-4.0, 4.0);

    // Test min and max.
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> min;
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1, std::greater<uint64_t>> max;
    for (std::size_t i = 0; i < 10000; ++i) {
        uint64_t x = rng1();
        min.add(x);
        max.add(x);
    }
    LOG_DEBUG(<< "min = " << min[0] << ", max = " << max[0]);
    BOOST_TEST_REQUIRE(min[0] < (maths::CPRNG::CXorOShiro128Plus::max() -
                                 maths::CPRNG::CXorOShiro128Plus::min()) /
                                    2000);
    BOOST_TEST_REQUIRE(max[0] > maths::CPRNG::CXorOShiro128Plus::max() -
                                    (maths::CPRNG::CXorOShiro128Plus::max() -
                                     maths::CPRNG::CXorOShiro128Plus::min()) /
                                        2000);

    // Test generate.
    maths::CPRNG::CXorOShiro128Plus rng2 = rng1;
    uint64_t samples1[50] = {0u};
    rng1.generate(&samples1[0], &samples1[50]);
    uint64_t samples2[50] = {0u};
    for (std::size_t i = 0; i < 50; ++i) {
        samples2[i] = rng2();
    }
    BOOST_TEST_REQUIRE(std::equal(&samples1[0], &samples1[50], &samples2[0]));

    // Test distribution.
    {
        boost::random::mt11213b mt;
        double p1[50] = {0.0};
        double p2[50] = {0.0};
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0; t < 50; ++t) {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0; i < 5000; ++i) {
                cvm1.addF(u01(rng1));
                cvm2.addF(u01(mt));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG(<< "p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG(<< "p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG(<< "m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG(<< "m2 = " << maths::CBasicStatistics::mean(m2));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(m1) >
                           0.95 * maths::CBasicStatistics::mean(m2));
    }
    {
        boost::random::mt19937_64 mt;
        boost::math::normal_distribution<> nm44(-4.0, 4.0);
        double p1[50] = {0.0};
        double p2[50] = {0.0};
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0; t < 50; ++t) {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0; i < 5000; ++i) {
                cvm1.addF(boost::math::cdf(nm44, norm(rng1)));
                cvm2.addF(boost::math::cdf(nm44, norm(mt)));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG(<< "p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG(<< "p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG(<< "m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG(<< "m2 = " << maths::CBasicStatistics::mean(m2));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(m1) >
                           0.95 * maths::CBasicStatistics::mean(m2));
    }

    // Test discard.
    maths::CPRNG::CXorOShiro128Plus rng3 = rng1;
    rng1.discard(10);
    for (std::size_t i = 0; i < 10; ++i) {
        rng3();
    }
    for (std::size_t t = 0; t < 500; ++t) {
        BOOST_REQUIRE_EQUAL(rng1(), rng3());
    }

    // Test jump. This is difficult to test since the equivalent
    // operation requires us to call operator() 2^64 times. What
    // we do is verify that the shift is at least a consistent
    // offset, i.e. rng + n + jump == rng + jump + n.
    maths::CPRNG::CXorOShiro128Plus rng4(rng1);
    maths::CPRNG::CXorOShiro128Plus rng5(rng1);
    std::size_t steps[] = {10, 3, 19};
    for (std::size_t s = 0; s < boost::size(steps); ++s) {
        rng4.jump();
        rng4.discard(steps[s]);
        rng5.discard(steps[s]);
        rng5.jump();
        for (std::size_t t = 0; t < 20; ++t) {
            BOOST_REQUIRE_EQUAL(rng4(), rng5());
        }
    }

    // Test serialization.
    std::string state = rng1.toString();
    LOG_DEBUG(<< "state = " << state);
    BOOST_TEST_REQUIRE(rng4.fromString(state));
    for (std::size_t t = 0; t < 500; ++t) {
        BOOST_REQUIRE_EQUAL(rng1(), rng4());
    }
}

BOOST_AUTO_TEST_CASE(testXorShift1024Mult) {
    maths::CPRNG::CXorShift1024Mult rng1;

    boost::uniform_01<> u01;
    boost::normal_distribution<> norm(100.0, 8000.0);

    // Test min and max.
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> min;
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1, std::greater<uint64_t>> max;
    for (std::size_t i = 0; i < 10000; ++i) {
        uint64_t x = rng1();
        min.add(x);
        max.add(x);
    }
    LOG_DEBUG(<< "min = " << min[0] << ", max = " << max[0]);
    BOOST_TEST_REQUIRE(min[0] < (maths::CPRNG::CXorShift1024Mult::max() -
                                 maths::CPRNG::CXorShift1024Mult::min()) /
                                    2000);
    BOOST_TEST_REQUIRE(max[0] > maths::CPRNG::CXorShift1024Mult::max() -
                                    (maths::CPRNG::CXorShift1024Mult::max() -
                                     maths::CPRNG::CXorShift1024Mult::min()) /
                                        2000);

    // Test generate.
    maths::CPRNG::CXorShift1024Mult rng2 = rng1;
    uint64_t samples1[50] = {0u};
    rng1.generate(&samples1[0], &samples1[50]);
    uint64_t samples2[50] = {0u};
    for (std::size_t i = 0; i < 50; ++i) {
        samples2[i] = rng2();
    }
    BOOST_TEST_REQUIRE(std::equal(&samples1[0], &samples1[50], &samples2[0]));

    // Test distribution.
    {
        boost::random::mt19937_64 mt;
        double p1[50] = {0.0};
        double p2[50] = {0.0};
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0; t < 50; ++t) {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0; i < 5000; ++i) {
                cvm1.addF(u01(rng1));
                cvm2.addF(u01(mt));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG(<< "p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG(<< "p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG(<< "m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG(<< "m2 = " << maths::CBasicStatistics::mean(m2));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(m1) >
                           0.95 * maths::CBasicStatistics::mean(m2));
    }
    {
        boost::random::mt11213b mt;
        boost::math::normal_distribution<> n1008000(100.0, 8000.0);
        double p1[50] = {0.0};
        double p2[50] = {0.0};
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0; t < 50; ++t) {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0; i < 5000; ++i) {
                cvm1.addF(boost::math::cdf(n1008000, norm(rng1)));
                cvm2.addF(boost::math::cdf(n1008000, norm(mt)));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG(<< "p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG(<< "p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG(<< "m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG(<< "m2 = " << maths::CBasicStatistics::mean(m2));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(m1) >
                           0.95 * maths::CBasicStatistics::mean(m2));
    }

    // Test discard.
    maths::CPRNG::CXorShift1024Mult rng3 = rng1;
    rng1.discard(10);
    for (std::size_t i = 0; i < 10; ++i) {
        rng3();
    }
    for (std::size_t t = 0; t < 500; ++t) {
        BOOST_REQUIRE_EQUAL(rng1(), rng3());
    }

    // Test jump. This is difficult to test since the equivalent
    // operation requires us to call operator() 2^512 times. What
    // we do is verify that the shift is at least a consistent
    // offset, i.e. rng + n + jump == rng + jump + n.
    maths::CPRNG::CXorShift1024Mult rng4(rng1);
    maths::CPRNG::CXorShift1024Mult rng5(rng1);
    std::size_t steps[] = {10, 3, 19};
    for (std::size_t s = 0; s < boost::size(steps); ++s) {
        rng4.jump();
        rng4.discard(steps[s]);
        rng5.discard(steps[s]);
        rng5.jump();
        for (std::size_t t = 0; t < 20; ++t) {
            BOOST_REQUIRE_EQUAL(rng4(), rng5());
        }
    }

    // Test serialization.
    rng1();
    std::string state = rng1.toString();
    LOG_DEBUG(<< "state = " << state);
    BOOST_TEST_REQUIRE(rng4.fromString(state));
    for (std::size_t t = 0; t < 500; ++t) {
        BOOST_REQUIRE_EQUAL(rng1(), rng4());
    }
}

BOOST_AUTO_TEST_SUITE_END()
