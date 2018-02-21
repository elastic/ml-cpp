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

#include "CTrendTestsTest.h"

#include <core/CoreTypes.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimezone.h>

#include <maths/CTools.h>
#include <maths/CTrendTests.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/bind.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>
#include <boost/scoped_ptr.hpp>

#include <vector>

using namespace ml;

namespace
{

using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TDiurnalPeriodicityTestPtr = boost::shared_ptr<maths::CDiurnalPeriodicityTest>;

class CDiurnalPeriodicityTestInspector : public maths::CDiurnalPeriodicityTest
{
    public:
        CDiurnalPeriodicityTestInspector(const maths::CDiurnalPeriodicityTest &test) :
                maths::CDiurnalPeriodicityTest(test)
        {}

        core_t::TTime bucketLength(void) const
        {
            return this->maths::CDiurnalPeriodicityTest::bucketLength();
        }
};

const core_t::TTime FIVE_MINS = 300;
const core_t::TTime HALF_HOUR = core::constants::HOUR / 2;
const core_t::TTime HOUR = core::constants::HOUR;
const core_t::TTime DAY = core::constants::DAY;
const core_t::TTime WEEK = core::constants::WEEK;

double constant(core_t::TTime /*time*/)
{
    return 0.0;
}

double ramp(core_t::TTime time)
{
    return 0.1 * static_cast<double>(time) / static_cast<double>(WEEK);
}

double markov(core_t::TTime time)
{
    static double state = 0.2;
    if (time % WEEK == 0)
    {
        core::CHashing::CMurmurHash2BT<core_t::TTime> hasher;
        state =  2.0 * static_cast<double>(hasher(time))
                     / static_cast<double>(std::numeric_limits<std::size_t>::max());
    }
    return state;
}

double smoothDaily(core_t::TTime time)
{
    return ::sin(  boost::math::double_constants::two_pi
                 * static_cast<double>(time)
                 / static_cast<double>(DAY));
}

double smoothWeekly(core_t::TTime time)
{
    return ::sin(  boost::math::double_constants::two_pi
                 * static_cast<double>(time)
                 / static_cast<double>(WEEK));
}

double spikeyDaily(core_t::TTime time)
{
    double pattern[] =
        {
            1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1
        };
    return pattern[(time % DAY) / HALF_HOUR];
}

double spikeyWeekly(core_t::TTime time)
{
    double pattern[] =
        {
            1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1
        };
    return pattern[(time % WEEK) / HALF_HOUR];
}

double weekends(core_t::TTime time)
{
    double amplitude[] = { 1.0, 0.9, 0.9, 0.9, 1.0, 0.2, 0.1 };
    return amplitude[(time % WEEK) / DAY]
           * ::sin(  boost::math::double_constants::two_pi
                   * static_cast<double>(time)
                   / static_cast<double>(DAY));
}

const TTimeVec BUCKET_LENGTHS_
    {
        1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400, 172800, 345600, 691200
    };
const maths::CScanningPeriodicityTest::TTimeCRng BUCKET_LENGTHS(BUCKET_LENGTHS_, 7, 16);

}

void CTrendTestsTest::testDiurnalInitialisation(void)
{
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testDiurnalInitialisation  |");
    LOG_DEBUG("+----------------------------------------------+");

    TDiurnalPeriodicityTestPtr test(
            maths::CDiurnalPeriodicityTest::create(DAY * 2, 0.1));
    CPPUNIT_ASSERT(nullptr == test);

    test.reset(maths::CDiurnalPeriodicityTest::create(DAY, 0.1));
    CPPUNIT_ASSERT(nullptr != test);
    CPPUNIT_ASSERT_EQUAL(DAY, CDiurnalPeriodicityTestInspector(*test).bucketLength());

    test.reset(maths::CDiurnalPeriodicityTest::create(600, 0.1));
    CPPUNIT_ASSERT(nullptr != test);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3600),
                         CDiurnalPeriodicityTestInspector(*test).bucketLength());
}

void CTrendTestsTest::testTrend(void)
{
    LOG_DEBUG("+------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testTrend  |");
    LOG_DEBUG("+------------------------------+");

    // Test that there is a high probability that a stationary
    // time series tests negative for having a trend and a high
    // probability that a time series with a trend tests positive
    // for having a trend.
    //
    // For the sake of concreteness, in the following test we
    // assume that the null hypothesis is that the series has
    // no trend. In the first test case the time series have
    // no trend so we are testing probability of type I error.
    // In the second test case the time series have a significant
    // trend so we are testing the probability of type II error.

    test::CRandomNumbers rng;

    double pFalsePositive = 0.0;
    double meanTimeToDetect = 0.0;

    for (std::size_t trend = 0u; trend < 2; ++trend)
    {
        LOG_DEBUG("*** trend = " << trend << " ***");

        int trendCount = 0;
        core_t::TTime timeToDetect = 0;

        for (std::size_t t = 0u; t < 100; ++t)
        {
            TDoubleVec samples;
            rng.generateNormalSamples(1.0, 5.0, 600, samples);
            double scale = 0.03 * ::sqrt(5.0);

            maths::CTrendTest trendTest(0.001);

            core_t::TTime testTimeToDetect = 7200 * samples.size();
            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                core_t::TTime time = static_cast<core_t::TTime>(i) * 7200;
                double x = (trend == 0 ? 0.0 : scale * static_cast<double>(i)) + samples[i];
                trendTest.add(time, x);
                trendTest.captureVariance(time, x);
                trendTest.propagateForwardsByTime(2.0);
                if (trendTest.test())
                {
                    testTimeToDetect = std::min(testTimeToDetect, time);
                    ++trendCount;
                }
            }
            timeToDetect += testTimeToDetect;
            if (trend == 1)
            {
                CPPUNIT_ASSERT_EQUAL(true, trendTest.test());
            }
        }

        if (trend == 0)
        {
            pFalsePositive = trendCount / 60000.0;
        }
        if (trend == 1)
        {
            meanTimeToDetect = timeToDetect / 100;
        }
    }

    LOG_DEBUG("[P(false positive)] = " << pFalsePositive);
    LOG_DEBUG("time to detect = " << meanTimeToDetect);
    CPPUNIT_ASSERT(pFalsePositive < 1e-4);
    CPPUNIT_ASSERT(meanTimeToDetect < 12 * DAY);

    for (std::size_t trend = 0u; trend < 2; ++trend)
    {
        LOG_DEBUG("*** trend = " << trend << " ***");

        std::size_t trendCount = 0u;
        core_t::TTime timeToDetect = 0;

        for (std::size_t t = 0u; t < 100; ++t)
        {
            TDoubleVec samples;
            rng.generateGammaSamples(50.0, 2.0, 600, samples);
            double scale = 0.03 * ::sqrt(200.0);

            maths::CTrendTest trendTest(0.001);

            core_t::TTime testTimeToDetect = 7200 * samples.size();
            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                core_t::TTime time = static_cast<core_t::TTime>(i) * 7200;
                double x = (trend == 0 ? 0.0 : scale * static_cast<double>(i)) + samples[i];
                trendTest.add(time, x);
                trendTest.captureVariance(time, x);
                trendTest.propagateForwardsByTime(2.0);
                if (trendTest.test())
                {
                    testTimeToDetect = std::min(testTimeToDetect, time);
                    ++trendCount;
                }
            }
            timeToDetect += testTimeToDetect;
            if (trend == 1)
            {
                CPPUNIT_ASSERT_EQUAL(true, trendTest.test());
            }
        }

        if (trend == 0)
        {
            pFalsePositive = trendCount / 60000.0;
        }
        if (trend == 1)
        {
            meanTimeToDetect = timeToDetect / 100;
        }
    }

    LOG_DEBUG("[P(false positive)] = " << pFalsePositive);
    LOG_DEBUG("time to detect = " << meanTimeToDetect);
    CPPUNIT_ASSERT(pFalsePositive < 1e-4);
    CPPUNIT_ASSERT(meanTimeToDetect < 12 * DAY);

    for (std::size_t trend = 0u; trend < 2; ++trend)
    {
        LOG_DEBUG("*** trend = " << trend << " ***");

        std::size_t trendCount = 0u;
        core_t::TTime timeToDetect = 0;

        for (std::size_t t = 0u; t < 100; ++t)
        {
            TDoubleVec samples;
            rng.generateLogNormalSamples(2.5, 1.0, 600, samples);
            double scale = 3.0 * ::sqrt(693.20);

            maths::CTrendTest trendTest(0.001);

            core_t::TTime testTimeToDetect = 7200 * samples.size();
            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                core_t::TTime time = static_cast<core_t::TTime>(i) * 7200;
                double x = (trend == 0 ? 0.0 : scale * ::sin(  boost::math::double_constants::two_pi
                                                             * static_cast<double>(i) / 600.0)) + samples[i];
                trendTest.add(time, x);
                trendTest.captureVariance(time, x);
                trendTest.propagateForwardsByTime(2.0);
                if (trendTest.test())
                {
                    testTimeToDetect = std::min(testTimeToDetect, time);
                    ++trendCount;
                }
            }

            timeToDetect += testTimeToDetect;
            if (trend == 1)
            {
                CPPUNIT_ASSERT_EQUAL(true, trendTest.test());
            }
        }

        if (trend == 0)
        {
            pFalsePositive = trendCount / 60000.0;
        }
        if (trend == 1)
        {
            meanTimeToDetect = timeToDetect / 100;
        }
    }

    LOG_DEBUG("[P(false positive)] = " << pFalsePositive);
    LOG_DEBUG("time to detect = " << meanTimeToDetect);
    CPPUNIT_ASSERT(pFalsePositive < 1e-4);
    CPPUNIT_ASSERT(meanTimeToDetect < 23 * DAY);
}

void CTrendTestsTest::testRandomizedPeriodicity(void)
{
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testRandomizedPeriodicity  |");
    LOG_DEBUG("+----------------------------------------------+");

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMaxAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double> >;
    using TFunction = double (*)(core_t::TTime);

    test::CRandomNumbers rng;

    TMeanAccumulator typeI;
    TMeanAccumulator typeII;
    for (std::size_t t = 0u; t < 5; ++t)
    {
        LOG_DEBUG("*** test = " << t << " ***");

        core_t::TTime time = 0;
        core_t::TTime day = 0;

        TDoubleVec samples;
        rng.generateLogNormalSamples(1.0, 4.0, 84000, samples);

        maths::CRandomizedPeriodicityTest::reset();

        maths::CRandomizedPeriodicityTest rtests[8];
        double falsePositives[3] = { 0.0, 0.0, 0.0 };
        double trueNegatives[3]  = { 0.0, 0.0, 0.0 };
        double truePositives[5]  = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double falseNegatives[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        TMeanVarAccumulator timeToDetectionMoments[5];
        TMaxAccumulator timeToDetectionMax[5];
        core_t::TTime lastTruePositive[5] = { time, time, time, time, time };
        TFunction functions[] =
            {
                &constant,
                &ramp,
                &markov,
                &smoothDaily,
                &smoothWeekly,
                &spikeyDaily,
                &spikeyWeekly,
                &weekends
            };

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(functions); ++j)
            {
                rtests[j].add(time, 600.0 * (functions[j])(time) + samples[i]);
            }
            if (time >= day + DAY)
            {
                for (std::size_t j = 0u; j < boost::size(rtests); ++j)
                {
                    if (j < 3)
                    {
                        (rtests[j].test() ? falsePositives[j] : trueNegatives[j]) += 1.0;
                    }
                    else
                    {
                        (rtests[j].test() ? truePositives[j - 3] : falseNegatives[j - 3]) += 1.0;
                        if (rtests[j].test())
                        {
                            timeToDetectionMoments[j - 3].add(time - lastTruePositive[j - 3]);
                            timeToDetectionMax[j - 3].add(static_cast<double>(time - lastTruePositive[j - 3]));
                            lastTruePositive[j - 3] = time;
                        }
                    }
                }
                day += DAY;
            }
            time += HALF_HOUR;
        }

        LOG_DEBUG("falsePositives = " << core::CContainerPrinter::print(falsePositives));
        LOG_DEBUG("trueNegatives = " << core::CContainerPrinter::print(trueNegatives));
        for (std::size_t i = 0u; i < boost::size(falsePositives); ++i)
        {
            CPPUNIT_ASSERT(falsePositives[i] / trueNegatives[i] < 0.1);
            typeI.add(falsePositives[i] / trueNegatives[i]);
        }
        LOG_DEBUG("truePositives = " << core::CContainerPrinter::print(truePositives));
        LOG_DEBUG("falseNegatives = " << core::CContainerPrinter::print(falseNegatives));
        for (std::size_t i = 0u; i < boost::size(falsePositives); ++i)
        {
            CPPUNIT_ASSERT(falseNegatives[i] / truePositives[i] < 0.2);
            typeII.add(falseNegatives[i] / truePositives[i]);
        }

        for (std::size_t i = 0u; i < boost::size(timeToDetectionMoments); ++i)
        {
            LOG_DEBUG("time to detect moments = " << timeToDetectionMoments[i]);
            LOG_DEBUG("maximum time to detect = " << timeToDetectionMax[i][0]);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(timeToDetectionMoments[i]) < 1.5 * DAY);
            CPPUNIT_ASSERT(::sqrt(maths::CBasicStatistics::variance(timeToDetectionMoments[i])) < 5 * DAY);
            CPPUNIT_ASSERT(timeToDetectionMax[i][0] <= 27 * WEEK);
        }
    }
    LOG_DEBUG("type I  = " << maths::CBasicStatistics::mean(typeI));
    LOG_DEBUG("type II = " << maths::CBasicStatistics::mean(typeII));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(typeI) < 0.015);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(typeII) < 0.05);
}

void CTrendTestsTest::testDiurnalPeriodicity(void)
{
    LOG_DEBUG("+-------------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testDiurnalPeriodicity  |");
    LOG_DEBUG("+-------------------------------------------+");

    using TResult = maths::CPeriodicityTestResult;

    test::CRandomNumbers rng;

    LOG_DEBUG("");
    LOG_DEBUG("*** Synthetic: no periods ***");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 5.0, 16128, samples);

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(FIVE_MINS));

        core_t::TTime time = 0;
        core_t::TTime day  = DAY;
        core_t::TTime week = 13 * DAY;

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            if (time > day && time < 13 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
                day += DAY;
            }
            if (time >= week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
                week += WEEK;
            }
            test->add(time, samples[i]);
            time += HALF_HOUR;
        }
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** Ramp ***");
    {
        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));

        for (core_t::TTime time = 0; time < 10 * WEEK; time += HALF_HOUR)
        {
            test->add(time, static_cast<double>(time));

            if (time > DAY && time < 2 * WEEK && time % DAY == 0)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
            }
            if (time > 2 * WEEK && time % (2 * WEEK) == 0)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
            }
        }
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** Synthetic: one period ***");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 5.0, 4032, samples);

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(FIVE_MINS));

        core_t::TTime time = 0;
        core_t::TTime day  = 3 * DAY;
        core_t::TTime week = 13 * DAY;

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            if (time > day && time < 13 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                day += DAY;
            }
            if (time >= week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                week += WEEK;
            }
            double x = 2.0 * ::sqrt(5.0) * sin(  static_cast<double>(i)
                                               / 48.0 * boost::math::double_constants::two_pi);
            test->add(time, x + samples[i]);
            time += HALF_HOUR;
        }

        CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(test->test()));
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** Synthetic: daily weekday/weekend ***");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 5.0, 4032, samples);

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(FIVE_MINS));

        core_t::TTime time = 0;
        core_t::TTime day  = 3 * DAY;
        core_t::TTime week = 13 * DAY;

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            if (time > day && time < 13 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                day += DAY;
            }
            if (time > week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekday daily' }"),
                                     test->print(result));
                week += WEEK;
            }
            double scale = 1.0;
            switch (((time + DAY) % WEEK) / DAY)
            {
            case 0:
            case 1:
                scale = 0.1; break;
            default:
                break;
            }
            double x = 10.0 * ::sqrt(5.0) * sin(  static_cast<double>(i)
                                                / 48.0 * boost::math::double_constants::two_pi);
            test->add(time, scale * (x + samples[i]));
            time += HALF_HOUR;
        }

        CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekday daily' }"),
                             test->print(test->test()));
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** Synthetic: weekly ***");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 5.0, 4032, samples);

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(FIVE_MINS));

        core_t::TTime time = 0;
        core_t::TTime day  = DAY;
        core_t::TTime week = 13 * DAY;
        std::size_t errors = 0u;
        std::size_t calls  = 0u;

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            if (time > day && time < 13 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
                day += DAY;
            }
            if (time >= week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                if (test->print(result) != "{ 'weekly' }")
                {
                    ++errors;
                }
                ++calls;
                week += WEEK;
            }
            double x = 3.0 * ::sqrt(5.0) * sin(  static_cast<double>(i)
                                               / 336.0 * boost::math::double_constants::two_pi);
            test->add(time, x + samples[i]);
            time += HALF_HOUR;
        }
        LOG_DEBUG("errors = " << static_cast<double>(errors) / static_cast<double>(calls));
        CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekly' }"), test->print(test->test()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, static_cast<double>(errors) / static_cast<double>(calls), 0.1);
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** daily ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/spikey_data.csv",
                                                        timeseries,
                                                        startTime,
                                                        endTime,
                                                        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                    timeseries.begin() + 10)
                  << " ...");

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(FIVE_MINS));

        core_t::TTime day  = startTime + 3 * DAY;
        core_t::TTime week = startTime + 13 * DAY;

        for (std::size_t i = 0u; i < timeseries.size(); ++i)
        {
            if (timeseries[i].first > day && timeseries[i].first < startTime + 13 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                day += DAY;
            }
            if (timeseries[i].first >= week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                week += WEEK;
            }
            test->add(timeseries[i].first, timeseries[i].second, 1.0 / 6.0);
        }

        CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(test->test()));
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** daily and weekends ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/diurnal.csv",
                                                        timeseries,
                                                        startTime,
                                                        endTime,
                                                        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                    timeseries.begin() + 10)
                  << " ...");

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(FIVE_MINS));

        core_t::TTime day  = startTime + 5 * DAY;
        core_t::TTime week = startTime + 13 * DAY;

        for (std::size_t i = 0u; i < timeseries.size(); ++i)
        {
            if (timeseries[i].first > 2 * day && timeseries[i].first < startTime + 13 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                day += DAY;
            }
            if (timeseries[i].first > week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekend weekly' 'weekday daily' }"),
                                     test->print(result));
                week += WEEK;
            }
            test->add(timeseries[i].first, timeseries[i].second, 1.0 / 6.0);
        }

        CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekend weekly' 'weekday daily' }"),
                             test->print(test->test()));
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** no periods ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/no_periods.csv",
                                                        timeseries,
                                                        startTime,
                                                        endTime,
                                                        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
                                                        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                    timeseries.begin() + 10)
                  << " ...");

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));

        core_t::TTime day  = startTime + DAY;
        core_t::TTime week = startTime + WEEK;

        for (std::size_t i = 0u; i < timeseries.size(); ++i)
        {
            if (timeseries[i].first > day && timeseries[i].first < startTime + 12 * DAY)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
                day += DAY;
            }
            if (timeseries[i].first > week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT(!result.periodic());
                week += WEEK;
            }
            test->add(timeseries[i].first, timeseries[i].second);
        }
    }

    LOG_DEBUG("");
    LOG_DEBUG("*** daily weekly and weekends ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/thirty_minute_samples.csv",
                                                        timeseries,
                                                        startTime,
                                                        endTime,
                                                        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
                                                        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                    timeseries.begin() + 10)
                  << " ...");

        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));

        core_t::TTime day  = startTime + DAY;
        core_t::TTime week = startTime + 2 * WEEK;

        for (std::size_t i = 0u; i < timeseries.size(); ++i)
        {
            if (timeseries[i].first > day && timeseries[i].first < startTime + 7 * DAY)
            {
                if (timeseries[i].first > startTime + 3 * DAY)
                {
                    TResult result = test->test();
                    LOG_DEBUG("detected = " << test->print(result));
                    CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
                }
                day += DAY;
            }
            if (timeseries[i].first > week)
            {
                TResult result = test->test();
                LOG_DEBUG("detected = " << test->print(result));
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekend weekly' 'weekday daily' }"), test->print(result));
                week += WEEK;
            }
            test->add(timeseries[i].first, timeseries[i].second);
        }

        CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekend weekly' 'weekday daily' }"),
                             test->print(test->test()));
    }
}

void CTrendTestsTest::testDiurnalPeriodicityWithMissingValues(void)
{
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testDiurnalPeriodicityWithMissingValues  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    test::CRandomNumbers rng;

    LOG_DEBUG("Daily Periodic")
    {
        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 7; ++t)
        {
            for (auto value : { 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0})
            {
                if (value > 0.0)
                {
                    test->add(time, value);
                }
                time += HALF_HOUR;
            }
            maths::CPeriodicityTestResult result{test->test()};
            LOG_DEBUG("result = " << test->print(result));
            if (t > 3)
            {
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), test->print(result));
            }
        }
    }
    LOG_DEBUG("Daily Not Periodic")
    {
        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 7; ++t)
        {
            for (auto value : { 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0})
            {
                if (value > 0.0)
                {
                    TDoubleVec rand;
                    rng.generateUniformSamples(-1.0, 1.0, 1, rand);
                    test->add(time, rand[0]);
                }
                time += HALF_HOUR;
            }
            maths::CPeriodicityTestResult result{test->test()};
            LOG_DEBUG("result = " << test->print(result));
            if (t > 3)
            {
                CPPUNIT_ASSERT_EQUAL(std::string("{ }"), test->print(result));
            }
        }
    }
    LOG_DEBUG("Weekly")
    {
        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 4; ++t)
        {
            for (auto value : { 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
            {
                if (value > 0.0)
                {
                    test->add(time, value);
                }
                time += HOUR;
            }
            maths::CPeriodicityTestResult result{test->test()};
            LOG_DEBUG("result = " << test->print(result));
            if (t > 3)
            {
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' 'weekly' }"), test->print(result));
            }
        }
    }
    LOG_DEBUG("Weekly Not Periodic")
    {
        TDiurnalPeriodicityTestPtr test(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 4; ++t)
        {
            for (auto value : { 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
            {
                if (value > 0.0)
                {
                    TDoubleVec rand;
                    rng.generateUniformSamples(-1.0, 1.0, 1, rand);
                    test->add(time, rand[0]);
                }
                time += HOUR;
            }
            maths::CPeriodicityTestResult result{test->test()};
            LOG_DEBUG("result = " << test->print(result));
            if (t > 3)
            {
                CPPUNIT_ASSERT_EQUAL(std::string("{ }"), test->print(result));
            }
        }
    }
}

void CTrendTestsTest::testScanningPeriodicity(void)
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testScanningPeriodicity  |");
    LOG_DEBUG("+--------------------------------------------+");

    using TPeriodicityResultPr = maths::CScanningPeriodicityTest::TPeriodicityResultPr;

    test::CRandomNumbers rng;

    LOG_DEBUG("Smooth")
    {
        TDoubleVec timeseries;
        for (core_t::TTime time = 0; time <= 2 * WEEK; time += HALF_HOUR)
        {
            timeseries.push_back(15.0 + 10.0 * ::sin(0.7 * boost::math::double_constants::two_pi
                                                         * static_cast<double>(time)
                                                         / static_cast<double>(DAY)));
        }

        maths::CScanningPeriodicityTest test(BUCKET_LENGTHS, 240);
        test.initialize(0);

        core_t::TTime time = 0;
        for (std::size_t i = 0u; i < timeseries.size(); ++i, time += HALF_HOUR)
        {
            if (test.needToCompress(time))
            {
                TPeriodicityResultPr result = test.test();
                LOG_DEBUG("time = " << time);
                LOG_DEBUG("periods = " << result.first.print(result.second));
                CPPUNIT_ASSERT(result.second.periodic());
                CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(DAY) / 0.7,
                                             static_cast<double>(result.first.periods()[0]),
                                             900.0);
                break;
            }
            test.add(time, timeseries[i]);
        }
    }

    LOG_DEBUG("Smooth + Noise")
    {
        TDoubleVec timeseries;
        for (core_t::TTime time = 0; time <= 2 * WEEK; time += HALF_HOUR)
        {
            timeseries.push_back(15.0 + 10.0 * ::sin(0.4 * boost::math::double_constants::two_pi
                                                         * static_cast<double>(time)
                                                         / static_cast<double>(DAY)));
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 4.0, timeseries.size(), noise);

        maths::CScanningPeriodicityTest test(BUCKET_LENGTHS, 240);
        test.initialize(0);

        core_t::TTime time = 0;
        std::size_t periodic = 0;
        for (std::size_t i = 0u; i < timeseries.size(); ++i, time += HALF_HOUR)
        {
            if (test.needToCompress(time))
            {
                TPeriodicityResultPr result = test.test();
                if (result.second.periodic())
                {
                    LOG_DEBUG("time = " << time);
                    LOG_DEBUG("periods = " << result.first.print(result.second));
                    CPPUNIT_ASSERT_EQUAL(static_cast<double>(DAY) / 0.4,
                                         static_cast<double>(result.first.periods()[0]));
                    ++periodic;
                    break;
                }
            }
            test.add(time, timeseries[i] + noise[i]);
        }
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), periodic);
    }

    LOG_DEBUG("Long term behaviour")
    {
        TDoubleVec timeseries;
        for (core_t::TTime time = 0; time <= 300 * WEEK; time += HALF_HOUR)
        {
            timeseries.push_back(15.0 + 10.0 * ::sin(0.1 * boost::math::double_constants::two_pi
                                                         * static_cast<double>(time)
                                                         / static_cast<double>(DAY)));
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 4.0, timeseries.size(), noise);

        maths::CScanningPeriodicityTest test(BUCKET_LENGTHS, 240);
        test.initialize(0);

        core_t::TTime time = 0;
        std::size_t periodic = 0;
        for (std::size_t i = 0u; i < timeseries.size(); ++i, time += HALF_HOUR)
        {
            if (test.needToCompress(time))
            {
                TPeriodicityResultPr result = test.test();
                if (result.second.periodic())
                {
                    LOG_DEBUG("time = " << time);
                    LOG_DEBUG("periods = " << result.first.print(result.second));
                    CPPUNIT_ASSERT(result.first.periods()[0] % (10 * DAY) == 0);
                    ++periodic;
                }
            }
            test.add(time, timeseries[i] + noise[i]);
        }
        CPPUNIT_ASSERT_EQUAL(std::size_t(8), periodic);
    }
}

void CTrendTestsTest::testCalendarCyclic(void)
{
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testCalendarCyclic  |");
    LOG_DEBUG("+---------------------------------------+");

    using TOptionalFeature = maths::CCalendarCyclicTest::TOptionalFeature;

    core::CTimezone::instance().timezoneName("GMT");

    test::CRandomNumbers rng;

    LOG_DEBUG("Day of month");
    {
        // Repeated error on the second day of the month.

        core_t::TTime months[] =
            {
                86400,   // 2nd Jan
                2764800, // 2nd Feb
                5184000, // 2nd Mar
                7862400, // 2nd Apr
                10454400 // 2nd May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 30000 && time < months[i - 1] + 50000)
            {
                error[0] *= 5.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("2nd day of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG("Days before end of month");
    {
        // Repeated error on the last day of the month.

        core_t::TTime months[] =
            {
                2592000,  // 31st Jan
                5011200,  // 28th Feb
                7689600,  // 31st Mar
                10281600, // 30th Apr
                12960000  // 31st May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 10000 && time < months[i - 1] + 20000)
            {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("0 days before end of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG("Day of week week of month");
    {
        // Repeated error on first Monday of each month.

        core_t::TTime months[] =
            {
                345600,  // Mon 5th Jan
                2764800, // Mon 2nd Feb
                5184000, // Mon 2nd Mar
                8208000, // Mon 6th Apr
                10627200 // Mon 4th May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000)
            {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("1st Monday of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG("Day of week weeks before end of month");
    {
        // Repeated error on last Friday of each month.
        core_t::TTime months[] =
            {
                2505600, // Fri 30th Jan
                4924800, // Fri 27th Feb
                7344000, // Fri 27th Mar
                9763200, // Fri 24th Apr
                12787200 // Fri 29th May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000)
            {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("0 Fridays before end of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }
}

void CTrendTestsTest::testPersist(void)
{
    LOG_DEBUG("+--------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testPersist  |");
    LOG_DEBUG("+--------------------------------+");

    // Check that persistence is idempotent.

    LOG_DEBUG("Test CTrendTest");
    {
        TDoubleVec timeseries;
        for (core_t::TTime time = 0; time <= 2 * WEEK; time += HALF_HOUR)
        {
            double daily = 15.0 + 10.0 * ::sin(  boost::math::double_constants::two_pi
                                               * static_cast<double>(time)
                                               / static_cast<double>(DAY));
            timeseries.push_back(daily);
        }

        test::CRandomNumbers rng;
        TDoubleVec noise;
        rng.generateNormalSamples(20.0, 16.0, timeseries.size(), noise);

        maths::CTrendTest orig;
        core_t::TTime time = 0;
        for (std::size_t i = 0u; i < timeseries.size(); ++i, time += HALF_HOUR)
        {
            orig.add(time, timeseries[i] + noise[i]);
            orig.captureVariance(time, timeseries[i] + noise[i]);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            orig.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG("XML representation:\n" << origXml);

        maths::CTrendTest restored;
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CTrendTest::acceptRestoreTraverser, &restored, _1)));
        }
        CPPUNIT_ASSERT_EQUAL(orig.checksum(), restored.checksum());

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restored.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }

    LOG_DEBUG("Test CRandomizedPeriodicityTest");
    {
        maths::CRandomizedPeriodicityTest test;
        for (core_t::TTime t = 1400000000; t < 1400050000; t += 5000)
        {
            test.add(t, 0.2);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        std::string origStaticsXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test.staticsAcceptPersistInserter(inserter);
            inserter.toXml(origStaticsXml);
        }

        // Check that the static state is also preserved
        uint64_t origNextRandom = test.ms_Rng();

        LOG_DEBUG("XML representation:\n" << origXml);

        // Restore the XML into a new test
        maths::CRandomizedPeriodicityTest test2;
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CRandomizedPeriodicityTest::acceptRestoreTraverser, &test2, _1)));
        }
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test2.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);

        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                    &maths::CRandomizedPeriodicityTest::staticsAcceptRestoreTraverser));
        }
        std::string newStaticsXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test2.staticsAcceptPersistInserter(inserter);
            inserter.toXml(newStaticsXml);
        }
        CPPUNIT_ASSERT_EQUAL(origStaticsXml, newStaticsXml);

        uint64_t newNextRandom = test2.ms_Rng();
        CPPUNIT_ASSERT_EQUAL(origNextRandom, newNextRandom);
    }

    LOG_DEBUG("Test CDiurnalPeriodicityTest");
    {
        TDoubleVec timeseries;
        for (core_t::TTime time = 0; time <= 2 * WEEK; time += HALF_HOUR)
        {
            double daily = 15.0 + 10.0 * ::sin(  boost::math::double_constants::two_pi
                                               * static_cast<double>(time)
                                               / static_cast<double>(DAY));
            timeseries.push_back(daily);
        }

        test::CRandomNumbers rng;
        TDoubleVec noise;
        rng.generateNormalSamples(20.0, 16.0, timeseries.size(), noise);

        boost::scoped_ptr<maths::CDiurnalPeriodicityTest> orig(maths::CDiurnalPeriodicityTest::create(HALF_HOUR));

        core_t::TTime time = 0;
        for (std::size_t i = 0u; i < timeseries.size(); ++i, time += HALF_HOUR)
        {
            orig->add(time, timeseries[i] + noise[i]);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            orig->acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG("XML representation:\n" << origXml);

        maths::CDiurnalPeriodicityTest restored;
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CDiurnalPeriodicityTest::acceptRestoreTraverser, &restored, _1)));
        }

        CPPUNIT_ASSERT_EQUAL(orig->checksum(), restored.checksum());

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restored.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }

    LOG_DEBUG("Test CScanningPeriodicityTest");
    {
        maths::CScanningPeriodicityTest orig(BUCKET_LENGTHS, 120);
        orig.initialize(0);
        for (core_t::TTime time = 0; time <= 2 * WEEK; time += HALF_HOUR)
        {
            orig.add(time, 15.0 + 10.0 * ::sin(  boost::math::double_constants::two_pi
                                               * static_cast<double>(time)
                                               / static_cast<double>(DAY)));
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            orig.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG("XML representation:\n" << origXml);

        maths::CScanningPeriodicityTest restored(BUCKET_LENGTHS, 10);
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CScanningPeriodicityTest::acceptRestoreTraverser, &restored, _1)));
        }
        CPPUNIT_ASSERT_EQUAL(orig.checksum(), restored.checksum());

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restored.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }

    LOG_DEBUG("Test CCalendarCyclicTest");
    {
        test::CRandomNumbers rng;

        maths::CCalendarCyclicTest orig(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= 12787200; time += HALF_HOUR)
        {
            rng.generateNormalSamples(0.0, 10.0, 1, error);
            orig.add(time, error[0]);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            orig.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG("XML representation:\n" << origXml);

        maths::CCalendarCyclicTest restored(HALF_HOUR);
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CCalendarCyclicTest::acceptRestoreTraverser, &restored, _1)));
        }
        CPPUNIT_ASSERT_EQUAL(orig.checksum(), restored.checksum());

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restored.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test *CTrendTestsTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CTrendTestsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testTrend",
                                   &CTrendTestsTest::testTrend) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testRandomizedPeriodicity",
                                   &CTrendTestsTest::testRandomizedPeriodicity) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testDiurnalPeriodicity",
                                   &CTrendTestsTest::testDiurnalPeriodicity) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testDiurnalPeriodicityWithMissingValues",
                                   &CTrendTestsTest::testDiurnalPeriodicityWithMissingValues) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testScanningPeriodicity",
                                   &CTrendTestsTest::testScanningPeriodicity) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testCalendarCyclic",
                                   &CTrendTestsTest::testCalendarCyclic) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testPersist",
                                   &CTrendTestsTest::testPersist) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testDiurnalInitialisation",
                                   &CTrendTestsTest::testDiurnalInitialisation) );

    return suiteOfTests;
}
