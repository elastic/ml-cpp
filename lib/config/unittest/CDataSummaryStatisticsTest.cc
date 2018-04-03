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

#include "CDataSummaryStatisticsTest.h"

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CStatisticalTests.h>

#include <config/CDataSummaryStatistics.h>

#include <test/CRandomNumbers.h>

#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cstdio>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

void CDataSummaryStatisticsTest::testRate(void)
{
    LOG_DEBUG("");
    LOG_DEBUG("+----------------------------------------+");
    LOG_DEBUG("|  CDataSummaryStatisticsTest::testRate  |");
    LOG_DEBUG("+----------------------------------------+");

    // Test we correctly estimate a range of rates.

    double rate[] = { 10.0, 100.0, 500.0 };
    double n = 100000.0;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(rate); ++i)
    {
        TDoubleVec times;
        rng.generateUniformSamples(0.0, n / rate[i], static_cast<std::size_t>(n), times);

        config::CDataSummaryStatistics summary;
        for (std::size_t j = 0u; j < times.size(); ++j)
        {
            summary.add(static_cast<core_t::TTime>(times[j]));
        }

        LOG_DEBUG("earliest = " << summary.earliest()
                  << ", latest = " << summary.latest());
        LOG_DEBUG("rate = " << summary.meanRate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(rate[i], summary.meanRate(), 2.0 * rate[i] * rate[i] / n);
    }
}

void CDataSummaryStatisticsTest::testCategoricalDistinctCount(void)
{
    LOG_DEBUG("");
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CDataSummaryStatisticsTest::testCategoricalDistinctCount  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    // Test we correctly compute the distinct count with and
    // without sketching.

    {
        LOG_DEBUG("*** Exact ***");

        test::CRandomNumbers rng;
        std::size_t n[] = { 10, 100, 1000 };
        for (std::size_t i = 0u; i < boost::size(n); ++i)
        {
            TStrVec categories;
            rng.generateWords(5, n[i], categories);

            config::CCategoricalDataSummaryStatistics summary(100);
            for (std::size_t j = 0u; j < categories.size(); ++j)
            {
                summary.add(static_cast<core_t::TTime>(j), categories[j]);
            }

            LOG_DEBUG("# categories = " << categories.size()
                      << ", distinct count = " << summary.distinctCount());
        }
    }

    {
        LOG_DEBUG("*** Sketched ***");

        config::CCategoricalDataSummaryStatistics summary(100);
        for (std::size_t i = 0u; i < 1000000; ++i)
        {
            summary.add(static_cast<core_t::TTime>(i), core::CStringUtils::typeToString(i));
        }

        LOG_DEBUG("# categories = 1000000, distinct count = " << summary.distinctCount());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1000000.0,
                                     static_cast<double>(summary.distinctCount()),
                                     0.005 * 1000000.0);
    }
}

void CDataSummaryStatisticsTest::testCategoricalTopN(void)
{
    LOG_DEBUG("");
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CDataSummaryStatisticsTest::testCategoricalTopN  |");
    LOG_DEBUG("+---------------------------------------------------+");

    // Test we are able to accurately estimate the counts of some
    // heavy hitting categories.

    test::CRandomNumbers rng;

    TStrVec categories;
    rng.generateWords(5, 100000, categories);

    std::size_t freq[]   = { 0, 23, 59, 100, 110, 174, 230, 540, 672, 810 };
    std::size_t counts[] = { 0,  0,  0,   0,   0,   0,   0,   0,   0,   0 };

    config::CCategoricalDataSummaryStatistics summary(20);

    TDoubleVec p;
    TSizeVec index;

    for (std::size_t j = 0u; j < 2000000; ++j)
    {
        rng.generateUniformSamples(0.0, 1.0, 1, p);

        if (p[0] < 0.05)
        {
            std::size_t b = std::upper_bound(boost::begin(freq),
                                             boost::end(freq),
                                             j / 2000) - boost::begin(freq);
            rng.generateUniformSamples(0, b, 1, index);
            index[0] = freq[index[0]];
        }
        else
        {
            rng.generateUniformSamples(0, categories.size(), 1, index);
        }

        const std::size_t *f = std::lower_bound(boost::begin(freq),
                                                boost::end(freq), index[0]);
        if (f != boost::end(freq) && *f == index[0])
        {
            ++counts[f - boost::begin(freq)];
        }

        summary.add(static_cast<core_t::TTime>(j), categories[index[0]]);
    }

    config::CCategoricalDataSummaryStatistics::TStrSizePrVec topn;
    summary.topN(topn);

    TMeanAccumulator meanError;
    for (std::size_t i = 0u; i < boost::size(freq); ++i)
    {
        LOG_DEBUG("");
        LOG_DEBUG("actual:    " << categories[freq[i]] << " appeared " << counts[i] << " times");
        LOG_DEBUG("estimated: " << topn[i].first << " appeared " << topn[i].second << " times");

        double exact  = static_cast<double>(counts[i]);
        double approx = static_cast<double>(topn[i].second);
        double error  = std::fabs(exact - approx) / exact;

        CPPUNIT_ASSERT_EQUAL(categories[freq[i]], topn[i].first);
        CPPUNIT_ASSERT(error < 0.05);
        meanError.add(error);
    }

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.005);
}

void CDataSummaryStatisticsTest::testNumericBasicStatistics(void)
{
    LOG_DEBUG("");
    LOG_DEBUG("+----------------------------------------------------------+");
    LOG_DEBUG("|  CDataSummaryStatisticsTest::testNumericBasicStatistics  |");
    LOG_DEBUG("+----------------------------------------------------------+");

    // Test the minimum, median and maximum of a variety of data sets.

    {
        LOG_DEBUG("*** Linear Ramp ***");

        config::CNumericDataSummaryStatistics summary(true);

        for (std::size_t i = 0u; i <= 500; ++i)
        {
            summary.add(static_cast<std::size_t>(i), core::CStringUtils::typeToString(i));
        }

        CPPUNIT_ASSERT_EQUAL(0.0, summary.minimum());
        CPPUNIT_ASSERT_EQUAL(250.0, summary.median());
        CPPUNIT_ASSERT_EQUAL(500.0, summary.maximum());
    }

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("*** Uniform ***");

        for (std::size_t i = 0u; i < 10; ++i)
        {
            TDoubleVec samples;
            rng.generateUniformSamples(-10.0, 50.0, 1000, samples);

            config::CNumericDataSummaryStatistics summary(false);
            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                summary.add(static_cast<core_t::TTime>(j), core::CStringUtils::typeToString(samples[j]));
            }

            LOG_DEBUG("minimum = " << summary.minimum());
            LOG_DEBUG("median  = " << summary.median());
            LOG_DEBUG("maximum = " << summary.maximum());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-10.0, summary.minimum(), 0.15);
            CPPUNIT_ASSERT_DOUBLES_EQUAL( 20.0, summary.median(),  1.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL( 50.0, summary.maximum(), 0.15);
        }
    }

    {
        LOG_DEBUG("*** Log-Normal ***");

        double location = 1.0;
        double scale = std::sqrt(2.0);

        boost::math::lognormal_distribution<> lognormal(1.0, std::sqrt(2.0));
        LOG_DEBUG("distribution median = " << boost::math::median(lognormal));

        TMeanAccumulator meanError;
        for (std::size_t i = 0u; i < 10; ++i)
        {
            TDoubleVec samples;
            rng.generateLogNormalSamples(location, scale * scale, 1000, samples);

            config::CNumericDataSummaryStatistics summary(false);
            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                summary.add(static_cast<core_t::TTime>(j), core::CStringUtils::typeToString(samples[j]));
            }

            LOG_DEBUG("median  = " << summary.median());
            CPPUNIT_ASSERT(std::fabs(summary.median() - boost::math::median(lognormal)) < 0.25);

            meanError.add(  std::fabs(summary.median() - boost::math::median(lognormal))
                          / boost::math::median(lognormal));
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.05);
    }
}

void CDataSummaryStatisticsTest::testNumericDistribution(void)
{
    LOG_DEBUG("");
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CDataSummaryStatisticsTest::testNumericDistribution  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec samples;

    {
        // Test heavy tail.

        rng.generateLogNormalSamples(1.0, 3.0, 10000, samples);

        boost::math::lognormal_distribution<> d(1.0, std::sqrt(3.0));

        config::CNumericDataSummaryStatistics statistics(false);
        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            statistics.add(static_cast<core_t::TTime>(i), core::CStringUtils::typeToString(samples[i]));
        }

        config::CNumericDataSummaryStatistics::TDoubleDoublePrVec chart;
        statistics.densityChart(chart);

        TMeanAccumulator meanAbsError;
        TMeanAccumulator mean;

        for (std::size_t i = 0u; i < chart.size(); ++i)
        {
            if (chart[i].first < 0.0)
            {
                continue;
            }
            double fexpected = boost::math::pdf(d, std::max(chart[i].first, 0.0));
            double f = chart[i].second;
            LOG_DEBUG("x = " << chart[i].first << ", fexpected(x) = " << fexpected << ", f(x) = " << f);
            meanAbsError.add(std::fabs(f - fexpected));
            mean.add(fexpected);
        }

        LOG_DEBUG("meanAbsError = " << maths::CBasicStatistics::mean(meanAbsError));
        LOG_DEBUG("mean = " << maths::CBasicStatistics::mean(mean));
        CPPUNIT_ASSERT(  maths::CBasicStatistics::mean(meanAbsError)
                       / maths::CBasicStatistics::mean(mean) < 0.3);
    }

    {
        // Test mixture distribution.

        samples.clear();
        TDoubleVec modeSamples;
        rng.generateNormalSamples(10.0, 10.0, 100, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.generateGammaSamples(100.0, 1.0, 200, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.generateNormalSamples(200.0, 100.0, 150, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.generateGammaSamples(100.0, 5.0, 100, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());

        double weights[] = { 1.0 / 5.5, 2.0 / 5.5, 1.5 / 5.5, 1.0 / 5.5 };
        boost::math::normal_distribution<> m0(10.0, std::sqrt(10.0));
        boost::math::gamma_distribution<>  m1(100.0, 1.0);
        boost::math::normal_distribution<> m2(200.0, 10.0);
        boost::math::gamma_distribution<>  m3(100.0, 5.0);

        rng.random_shuffle(samples.begin(), samples.end());

        config::CNumericDataSummaryStatistics statistics(false);
        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            statistics.add(static_cast<core_t::TTime>(i), core::CStringUtils::typeToString(samples[i]));
        }

        config::CNumericDataSummaryStatistics::TDoubleDoublePrVec chart;
        statistics.densityChart(chart);

        TMeanAccumulator meanAbsError;
        TMeanAccumulator meanRelError;

        for (std::size_t i = 0u; i < chart.size(); ++i)
        {
            double fexpected =  weights[0] * boost::math::pdf(m0, chart[i].first)
                              + weights[1] * boost::math::pdf(m1, chart[i].first)
                              + weights[2] * boost::math::pdf(m2, chart[i].first)
                              + weights[3] * boost::math::pdf(m3, chart[i].first);
            double f = chart[i].second;
            LOG_DEBUG("x = " << chart[i].first << ", fexpected(x) = " << fexpected << ", f(x) = " << f);
            meanAbsError.add(std::fabs(f - fexpected));
            meanRelError.add(std::fabs(std::log(f) - std::log(fexpected)) / std::fabs(std::log(fexpected)));
        }

        LOG_DEBUG("meanAbsError = " << maths::CBasicStatistics::mean(meanAbsError));
        LOG_DEBUG("meanRelError = " << maths::CBasicStatistics::mean(meanRelError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanAbsError) < 0.005);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelError) < 0.05);
    }
}

CppUnit::Test *CDataSummaryStatisticsTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CDataSummaryStatisticsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CDataSummaryStatisticsTest>(
                                   "CDataSummaryStatisticsTest::testRate",
                                   &CDataSummaryStatisticsTest::testRate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDataSummaryStatisticsTest>(
                                   "CDataSummaryStatisticsTest::testCategoricalDistinctCount",
                                   &CDataSummaryStatisticsTest::testCategoricalDistinctCount) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDataSummaryStatisticsTest>(
                                   "CDataSummaryStatisticsTest::testCategoricalTopN",
                                   &CDataSummaryStatisticsTest::testCategoricalTopN) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDataSummaryStatisticsTest>(
                                   "CDataSummaryStatisticsTest::testNumericBasicStatistics",
                                   &CDataSummaryStatisticsTest::testNumericBasicStatistics) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDataSummaryStatisticsTest>(
                                   "CDataSummaryStatisticsTest::testNumericDistribution",
                                   &CDataSummaryStatisticsTest::testNumericDistribution) );

    return suiteOfTests;
}
