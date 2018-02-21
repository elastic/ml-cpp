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

#include "CPRNGTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CPRNG.h>
#include <maths/CStatisticalTests.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

using namespace ml;

void CPRNGTest::testSplitMix64(void)
{
    LOG_DEBUG("+-----------------------------+");
    LOG_DEBUG("|  CPRNGTest::testSplitMix64  |");
    LOG_DEBUG("+-----------------------------+");

    maths::CPRNG::CSplitMix64 rng1;

    boost::uniform_01<> u01;
    boost::normal_distribution<> norm(4.0, 10.0);

    // Test min and max.
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> min;
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1, std::greater<uint64_t> > max;
    for (std::size_t i = 0u; i < 10000; ++i)
    {
        uint64_t x = rng1();
        min.add(x);
        max.add(x);
    }
    LOG_DEBUG("min = " << min[0] << ", max = " << max[0]);
    CPPUNIT_ASSERT(min[0] <  (maths::CPRNG::CSplitMix64::max() - maths::CPRNG::CSplitMix64::min()) / 2000);
    CPPUNIT_ASSERT(max[0] >   maths::CPRNG::CSplitMix64::max()
                           - (maths::CPRNG::CSplitMix64::max() - maths::CPRNG::CSplitMix64::min()) / 2000);

    // Test generate.
    maths::CPRNG::CSplitMix64 rng2 = rng1;
    uint64_t samples1[50] = { 0u };
    rng1.generate(&samples1[0], &samples1[50]);
    uint64_t samples2[50] = { 0u };
    for (std::size_t i = 0u; i < 50; ++i)
    {
        samples2[i] = rng2();
    }
    CPPUNIT_ASSERT(std::equal(&samples1[0], &samples1[50], &samples2[0]));

    // Test distribution.
    {
        boost::random::mt19937_64 mt;
        double p1[50] = { 0.0 };
        double p2[50] = { 0.0 };
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0u; i < 5000; ++i)
            {
                cvm1.addF(u01(rng1));
                cvm2.addF(u01(mt));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG("p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG("p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG("m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG("m2 = " << maths::CBasicStatistics::mean(m2));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(m1) > 0.95 * maths::CBasicStatistics::mean(m2));
    }
    {
        boost::random::mt11213b mt;
        boost::math::normal_distribution<> n410(4.0, 10.0);
        double p1[50] = { 0.0 };
        double p2[50] = { 0.0 };
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0u; i < 5000; ++i)
            {
                cvm1.addF(boost::math::cdf(n410, norm(rng1)));
                cvm2.addF(boost::math::cdf(n410, norm(mt)));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG("p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG("p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG("m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG("m2 = " << maths::CBasicStatistics::mean(m2));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(m1) > 0.95 * maths::CBasicStatistics::mean(m2));
    }

    // Test discard.
    maths::CPRNG::CSplitMix64 rng3 = rng1;
    rng1.discard(10);
    for (std::size_t i = 0u; i < 10; ++i)
    {
        rng3();
    }
    for (std::size_t t = 0u; t < 500; ++t)
    {
        CPPUNIT_ASSERT_EQUAL(rng1(), rng3());
    }

    // Test serialization.
    std::string state = rng1.toString();
    LOG_DEBUG("state = " << state);
    maths::CPRNG::CSplitMix64 rng4;
    CPPUNIT_ASSERT(rng4.fromString(state));
    for (std::size_t t = 0u; t < 500; ++t)
    {
        CPPUNIT_ASSERT_EQUAL(rng1(), rng4());
    }
}

void CPRNGTest::testXorOShiro128Plus(void)
{
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CPRNGTest::testXorOShiro128Plus  |");
    LOG_DEBUG("+-----------------------------------+");

    maths::CPRNG::CXorOShiro128Plus rng1;

    boost::uniform_01<> u01;
    boost::normal_distribution<> norm(-4.0, 4.0);

    // Test min and max.
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> min;
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1, std::greater<uint64_t> > max;
    for (std::size_t i = 0u; i < 10000; ++i)
    {
        uint64_t x = rng1();
        min.add(x);
        max.add(x);
    }
    LOG_DEBUG("min = " << min[0] << ", max = " << max[0]);
    CPPUNIT_ASSERT(min[0] <  (maths::CPRNG::CXorOShiro128Plus::max() - maths::CPRNG::CXorOShiro128Plus::min()) / 2000);
    CPPUNIT_ASSERT(max[0] >   maths::CPRNG::CXorOShiro128Plus::max()
                           - (maths::CPRNG::CXorOShiro128Plus::max() - maths::CPRNG::CXorOShiro128Plus::min()) / 2000);

    // Test generate.
    maths::CPRNG::CXorOShiro128Plus rng2 = rng1;
    uint64_t samples1[50] = { 0u };
    rng1.generate(&samples1[0], &samples1[50]);
    uint64_t samples2[50] = { 0u };
    for (std::size_t i = 0u; i < 50; ++i)
    {
        samples2[i] = rng2();
    }
    CPPUNIT_ASSERT(std::equal(&samples1[0], &samples1[50], &samples2[0]));

    // Test distribution.
    {
        boost::random::mt11213b mt;
        double p1[50] = { 0.0 };
        double p2[50] = { 0.0 };
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0u; i < 5000; ++i)
            {
                cvm1.addF(u01(rng1));
                cvm2.addF(u01(mt));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG("p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG("p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG("m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG("m2 = " << maths::CBasicStatistics::mean(m2));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(m1) > 0.95 * maths::CBasicStatistics::mean(m2));
    }
    {
        boost::random::mt19937_64 mt;
        boost::math::normal_distribution<> nm44(-4.0, 4.0);
        double p1[50] = { 0.0 };
        double p2[50] = { 0.0 };
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0u; i < 5000; ++i)
            {
                cvm1.addF(boost::math::cdf(nm44, norm(rng1)));
                cvm2.addF(boost::math::cdf(nm44, norm(mt)));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG("p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG("p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG("m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG("m2 = " << maths::CBasicStatistics::mean(m2));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(m1) > 0.95 * maths::CBasicStatistics::mean(m2));
    }

    // Test discard.
    maths::CPRNG::CXorOShiro128Plus rng3 = rng1;
    rng1.discard(10);
    for (std::size_t i = 0u; i < 10; ++i)
    {
        rng3();
    }
    for (std::size_t t = 0u; t < 500; ++t)
    {
        CPPUNIT_ASSERT_EQUAL(rng1(), rng3());
    }

    // Test jump. This is difficult to test since the equivalent
    // operation requires us to call operator() 2^64 times. What
    // we do is verify that the shift is at least a consistent
    // offset, i.e. rng + n + jump == rng + jump + n.
    maths::CPRNG::CXorOShiro128Plus rng4(rng1);
    maths::CPRNG::CXorOShiro128Plus rng5(rng1);
    std::size_t steps[] = { 10, 3, 19 };
    for (std::size_t s = 0u; s < boost::size(steps); ++s)
    {
        rng4.jump();
        rng4.discard(steps[s]);
        rng5.discard(steps[s]);
        rng5.jump();
        for (std::size_t t = 0u; t < 20; ++t)
        {
            CPPUNIT_ASSERT_EQUAL(rng4(), rng5());
        }
    }

    // Test serialization.
    std::string state = rng1.toString();
    LOG_DEBUG("state = " << state);
    CPPUNIT_ASSERT(rng4.fromString(state));
    for (std::size_t t = 0u; t < 500; ++t)
    {
        CPPUNIT_ASSERT_EQUAL(rng1(), rng4());
    }
}

void CPRNGTest::testXorShift1024Mult(void)
{
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CPRNGTest::testXorShift1024Mult  |");
    LOG_DEBUG("+-----------------------------------+");

    maths::CPRNG::CXorShift1024Mult rng1;

    boost::uniform_01<> u01;
    boost::normal_distribution<> norm(100.0, 8000.0);

    // Test min and max.
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> min;
    maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1, std::greater<uint64_t> > max;
    for (std::size_t i = 0u; i < 10000; ++i)
    {
        uint64_t x = rng1();
        min.add(x);
        max.add(x);
    }
    LOG_DEBUG("min = " << min[0] << ", max = " << max[0]);
    CPPUNIT_ASSERT(min[0] <  (maths::CPRNG::CXorShift1024Mult::max() - maths::CPRNG::CXorShift1024Mult::min()) / 2000);
    CPPUNIT_ASSERT(max[0] >   maths::CPRNG::CXorShift1024Mult::max()
                           - (maths::CPRNG::CXorShift1024Mult::max() - maths::CPRNG::CXorShift1024Mult::min()) / 2000);

    // Test generate.
    maths::CPRNG::CXorShift1024Mult rng2 = rng1;
    uint64_t samples1[50] = { 0u };
    rng1.generate(&samples1[0], &samples1[50]);
    uint64_t samples2[50] = { 0u };
    for (std::size_t i = 0u; i < 50; ++i)
    {
        samples2[i] = rng2();
    }
    CPPUNIT_ASSERT(std::equal(&samples1[0], &samples1[50], &samples2[0]));

    // Test distribution.
    {
        boost::random::mt19937_64 mt;
        double p1[50] = { 0.0 };
        double p2[50] = { 0.0 };
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0u; i < 5000; ++i)
            {
                cvm1.addF(u01(rng1));
                cvm2.addF(u01(mt));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG("p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG("p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG("m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG("m2 = " << maths::CBasicStatistics::mean(m2));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(m1) > 0.95 * maths::CBasicStatistics::mean(m2));
    }
    {
        boost::random::mt11213b mt;
        boost::math::normal_distribution<> n1008000(100.0, 8000.0);
        double p1[50] = { 0.0 };
        double p2[50] = { 0.0 };
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m1;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m2;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            maths::CStatisticalTests::CCramerVonMises cvm1(50);
            maths::CStatisticalTests::CCramerVonMises cvm2(50);
            for (std::size_t i = 0u; i < 5000; ++i)
            {
                cvm1.addF(boost::math::cdf(n1008000, norm(rng1)));
                cvm2.addF(boost::math::cdf(n1008000, norm(mt)));
            }
            p1[t] = cvm1.pValue();
            p2[t] = cvm2.pValue();
            m1.add(cvm1.pValue());
            m2.add(cvm2.pValue());
        }
        LOG_DEBUG("p1 = " << core::CContainerPrinter::print(p1));
        LOG_DEBUG("p2 = " << core::CContainerPrinter::print(p2));
        LOG_DEBUG("m1 = " << maths::CBasicStatistics::mean(m1));
        LOG_DEBUG("m2 = " << maths::CBasicStatistics::mean(m2));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(m1) > 0.95 * maths::CBasicStatistics::mean(m2));
    }

    // Test discard.
    maths::CPRNG::CXorShift1024Mult rng3 = rng1;
    rng1.discard(10);
    for (std::size_t i = 0u; i < 10; ++i)
    {
        rng3();
    }
    for (std::size_t t = 0u; t < 500; ++t)
    {
        CPPUNIT_ASSERT_EQUAL(rng1(), rng3());
    }

    // Test jump. This is difficult to test since the equivalent
    // operation requires us to call operator() 2^512 times. What
    // we do is verify that the shift is at least a consistent
    // offset, i.e. rng + n + jump == rng + jump + n.
    maths::CPRNG::CXorShift1024Mult rng4(rng1);
    maths::CPRNG::CXorShift1024Mult rng5(rng1);
    std::size_t steps[] = { 10, 3, 19 };
    for (std::size_t s = 0u; s < boost::size(steps); ++s)
    {
        rng4.jump();
        rng4.discard(steps[s]);
        rng5.discard(steps[s]);
        rng5.jump();
        for (std::size_t t = 0u; t < 20; ++t)
        {
            CPPUNIT_ASSERT_EQUAL(rng4(), rng5());
        }
    }

    // Test serialization.
    rng1();
    std::string state = rng1.toString();
    LOG_DEBUG("state = " << state);
    CPPUNIT_ASSERT(rng4.fromString(state));
    for (std::size_t t = 0u; t < 500; ++t)
    {
        CPPUNIT_ASSERT_EQUAL(rng1(), rng4());
    }
}

CppUnit::Test *CPRNGTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CPRNGTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CPRNGTest>(
                                   "CPRNGTest::testSplitMix64",
                                   &CPRNGTest::testSplitMix64) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPRNGTest>(
                                   "CPRNGTest::testXorOShiro128Plus",
                                   &CPRNGTest::testXorOShiro128Plus) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPRNGTest>(
                                   "CPRNGTest::testXorShift1024Mult",
                                   &CPRNGTest::testXorShift1024Mult) );

    return suiteOfTests;
}
