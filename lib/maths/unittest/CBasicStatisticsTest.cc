/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsCovariances.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSampling.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CBasicStatisticsTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMeanVarSkewAccumulator = maths::CBasicStatistics::SSampleMeanVarSkew<double>::TAccumulator;
using TMeanAccumulator2Vec = core::CSmallVector<TMeanAccumulator, 2>;
using TMeanVarAccumulator2Vec = core::CSmallVector<TMeanVarAccumulator, 2>;
using TMeanVarSkewAccumulator2Vec = core::CSmallVector<TMeanVarSkewAccumulator, 2>;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
using TMeanVarSkewAccumulatorVec = std::vector<TMeanVarSkewAccumulator>;

const std::string TAG("a");

struct SRestore {
    using result_type = bool;

    template<typename T>
    bool operator()(std::vector<T>& restored, core::CStateRestoreTraverser& traverser) const {
        return core::CPersistUtils::restore(TAG, restored, traverser);
    }

    template<typename T>
    bool operator()(T& restored, core::CStateRestoreTraverser& traverser) const {
        return restored.fromDelimited(traverser.value());
    }
};
}

BOOST_AUTO_TEST_CASE(testMean) {
    double sample[] = {0.9, 10.0, 5.6, 1.23, -12.3, 445.2, 0.0, 1.2};

    maths::CBasicStatistics::TDoubleVec sampleVec(
        sample, sample + sizeof(sample) / sizeof(sample[0]));

    double mean = maths::CBasicStatistics::mean(sampleVec);

    // Compare with hand calculated value
    BOOST_REQUIRE_EQUAL(56.47875, mean);
}

BOOST_AUTO_TEST_CASE(testVarianceAtPercentile) {

    // Test that the variance at a percentile is correctly calibrated.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    TMeanAccumulator bias;

    for (auto percentile : {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0}) {

        for (std::size_t n : {5, 20, 50}) {

            double varianceAtPercentile{maths::CBasicStatistics::varianceAtPercentile(
                percentile, 1.0, static_cast<double>(n - 1))};

            double percentageLessThan{0.0};
            for (std::size_t i = 0; i < 1000; ++i) {
                rng.generateNormalSamples(0.0, 1.0, n, samples);
                double variance{maths::CBasicStatistics::variance(std::accumulate(
                    samples.begin(), samples.end(), TMeanVarAccumulator{},
                    [](TMeanVarAccumulator moments, double value) {
                        moments.add(value);
                        return moments;
                    }))};
                if (variance < varianceAtPercentile) {
                    percentageLessThan += 0.1;
                }
            }

            LOG_DEBUG(<< "variance(" << percentile << ") = " << varianceAtPercentile
                      << ", % less than = " << percentageLessThan);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(percentile, percentageLessThan, 4.0);
            bias.add(percentile - percentageLessThan);
        }
    }

    LOG_DEBUG(<< "bias = " << maths::CBasicStatistics::mean(bias));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(bias) < 0.1);
}

BOOST_AUTO_TEST_CASE(testCentralMoments) {
    LOG_DEBUG(<< "Test mean double");
    {
        double samples[] = {0.9, 10.0, 5.6, 1.23, -12.3, 7.2, 0.0, 1.2};
        TMeanAccumulator acc;

        size_t count = sizeof(samples) / sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        BOOST_REQUIRE_EQUAL(count, static_cast<size_t>(maths::CBasicStatistics::count(acc)));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.72875, maths::CBasicStatistics::mean(acc), 0.000005);

        double n0 = maths::CBasicStatistics::count(acc);
        maths::CBasicStatistics::scale(0.5, acc);
        double n1 = maths::CBasicStatistics::count(acc);
        BOOST_REQUIRE_EQUAL(n1, 0.5 * n0);
    }

    LOG_DEBUG(<< "Test mean float");
    {
        using TFloatMeanAccumulator = maths::CBasicStatistics::SSampleMean<float>::TAccumulator;

        float samples[] = {0.9f, 10.0f, 5.6f, 1.23f, -12.3f, 7.2f, 0.0f, 1.2f};

        TFloatMeanAccumulator acc;

        size_t count = sizeof(samples) / sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        BOOST_REQUIRE_EQUAL(count, static_cast<size_t>(maths::CBasicStatistics::count(acc)));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.72875f, maths::CBasicStatistics::mean(acc), 0.000005f);
    }

    LOG_DEBUG(<< "Test mean and variance");
    {
        double samples[] = {0.9, 10.0, 5.6, 1.23, -12.3, 7.2, 0.0, 1.2};

        TMeanVarAccumulator acc;

        size_t count = sizeof(samples) / sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        BOOST_REQUIRE_EQUAL(count, static_cast<size_t>(maths::CBasicStatistics::count(acc)));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.72875, maths::CBasicStatistics::mean(acc), 0.000005);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(44.90633, maths::CBasicStatistics::variance(acc), 0.000005);

        double n0 = maths::CBasicStatistics::count(acc);
        maths::CBasicStatistics::scale(0.5, acc);
        double n1 = maths::CBasicStatistics::count(acc);
        BOOST_REQUIRE_EQUAL(n1, 0.5 * n0);
    }

    LOG_DEBUG(<< "Test mean, variance and skew");
    {
        double samples[] = {0.9, 10.0, 5.6, 1.23, -12.3, 7.2, 0.0, 1.2};

        TMeanVarSkewAccumulator acc;

        size_t count = sizeof(samples) / sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        BOOST_REQUIRE_EQUAL(count, static_cast<size_t>(maths::CBasicStatistics::count(acc)));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.72875, maths::CBasicStatistics::mean(acc), 0.000005);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(44.90633, maths::CBasicStatistics::variance(acc), 0.000005);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(-0.82216, maths::CBasicStatistics::skewness(acc), 0.000005);

        double n0 = maths::CBasicStatistics::count(acc);
        maths::CBasicStatistics::scale(0.5, acc);
        double n1 = maths::CBasicStatistics::count(acc);
        BOOST_REQUIRE_EQUAL(n1, 0.5 * n0);
    }

    LOG_DEBUG(<< "Test weighted update");
    {
        double samples[] = {0.9, 1.0, 2.3, 1.5};
        std::size_t weights[] = {1, 4, 2, 3};

        {
            TMeanAccumulator acc1;
            TMeanAccumulator acc2;

            for (size_t i = 0; i < boost::size(samples); ++i) {
                acc1.add(samples[i], static_cast<double>(weights[i]));
                for (std::size_t j = 0u; j < weights[i]; ++j) {
                    acc2.add(samples[i]);
                }
            }

            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc1),
                                         maths::CBasicStatistics::mean(acc2), 1e-10);
        }

        {
            TMeanVarAccumulator acc1;
            TMeanVarAccumulator acc2;

            for (size_t i = 0; i < boost::size(samples); ++i) {
                acc1.add(samples[i], static_cast<double>(weights[i]));
                for (std::size_t j = 0u; j < weights[i]; ++j) {
                    acc2.add(samples[i]);
                }
            }

            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc1),
                                         maths::CBasicStatistics::mean(acc2), 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::variance(acc1),
                                         maths::CBasicStatistics::variance(acc2), 1e-10);
        }

        {
            TMeanVarSkewAccumulator acc1;
            TMeanVarSkewAccumulator acc2;

            for (size_t i = 0; i < boost::size(samples); ++i) {
                acc1.add(samples[i], static_cast<double>(weights[i]));
                for (std::size_t j = 0u; j < weights[i]; ++j) {
                    acc2.add(samples[i]);
                }
            }

            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc1),
                                         maths::CBasicStatistics::mean(acc2), 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::variance(acc1),
                                         maths::CBasicStatistics::variance(acc2), 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::skewness(acc1),
                                         maths::CBasicStatistics::skewness(acc2), 1e-10);
        }
    }

    LOG_DEBUG(<< "Test addition");
    {
        // Test addition.
        double samples1[] = {0.9, 10.0, 5.6, 1.23};
        double samples2[] = {-12.3, 7.2, 0.0, 1.2};

        size_t count1 = sizeof(samples1) / sizeof(samples1[0]);
        size_t count2 = sizeof(samples2) / sizeof(samples2[0]);

        {
            TMeanAccumulator acc1;
            TMeanAccumulator acc2;

            acc1 = std::for_each(samples1, samples1 + count1, acc1);
            acc2 = std::for_each(samples2, samples2 + count2, acc2);

            BOOST_REQUIRE_EQUAL(
                count1 + count2,
                static_cast<size_t>(maths::CBasicStatistics::count(acc1 + acc2)));

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.72875, maths::CBasicStatistics::mean(acc1 + acc2), 0.000005);
        }

        {
            TMeanVarAccumulator acc1;
            TMeanVarAccumulator acc2;

            acc1 = std::for_each(samples1, samples1 + count1, acc1);
            acc2 = std::for_each(samples2, samples2 + count2, acc2);

            BOOST_REQUIRE_EQUAL(
                count1 + count2,
                static_cast<size_t>(maths::CBasicStatistics::count(acc1 + acc2)));

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.72875, maths::CBasicStatistics::mean(acc1 + acc2), 0.000005);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                44.90633, maths::CBasicStatistics::variance(acc1 + acc2), 0.000005);
        }

        {
            TMeanVarSkewAccumulator acc1;
            TMeanVarSkewAccumulator acc2;

            acc1 = std::for_each(samples1, samples1 + count1, acc1);
            acc2 = std::for_each(samples2, samples2 + count2, acc2);

            BOOST_REQUIRE_EQUAL(
                count1 + count2,
                static_cast<size_t>(maths::CBasicStatistics::count(acc1 + acc2)));

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.72875, maths::CBasicStatistics::mean(acc1 + acc2), 0.000005);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                44.90633, maths::CBasicStatistics::variance(acc1 + acc2), 0.000005);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                -0.82216, maths::CBasicStatistics::skewness(acc1 + acc2), 0.000005);
        }
    }

    LOG_DEBUG(<< "Test subtraction");
    {
        test::CRandomNumbers rng;

        LOG_DEBUG(<< "Test mean");
        {
            TMeanAccumulator acc1;
            TMeanAccumulator acc2;

            TDoubleVec samples;
            rng.generateNormalSamples(2.0, 3.0, 40u, samples);

            for (std::size_t j = 1u; j < samples.size(); ++j) {
                LOG_DEBUG(<< "split = " << j << "/" << samples.size() - j);

                for (std::size_t i = 0u; i < j; ++i) {
                    acc1.add(samples[i]);
                }
                for (std::size_t i = j; i < samples.size(); ++i) {
                    acc2.add(samples[i]);
                }

                TMeanAccumulator sum = acc1 + acc2;

                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(acc1),
                                    maths::CBasicStatistics::count(sum - acc2));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc1),
                                             maths::CBasicStatistics::mean(sum - acc2),
                                             1e-10);
                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(acc2),
                                    maths::CBasicStatistics::count(sum - acc1));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc2),
                                             maths::CBasicStatistics::mean(sum - acc1),
                                             1e-10);
            }
        }
        LOG_DEBUG(<< "Test mean and variance");
        {
            TMeanVarAccumulator acc1;
            TMeanVarAccumulator acc2;

            TDoubleVec samples;
            rng.generateGammaSamples(3.0, 3.0, 40u, samples);

            for (std::size_t j = 1u; j < samples.size(); ++j) {
                LOG_DEBUG(<< "split = " << j << "/" << samples.size() - j);

                for (std::size_t i = 0u; i < j; ++i) {
                    acc1.add(samples[i]);
                }
                for (std::size_t i = j; i < samples.size(); ++i) {
                    acc2.add(samples[i]);
                }

                TMeanVarAccumulator sum = acc1 + acc2;

                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(acc1),
                                    maths::CBasicStatistics::count(sum - acc2));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc1),
                                             maths::CBasicStatistics::mean(sum - acc2),
                                             1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::variance(acc1),
                    maths::CBasicStatistics::variance(sum - acc2), 1e-10);
                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(acc2),
                                    maths::CBasicStatistics::count(sum - acc1));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc2),
                                             maths::CBasicStatistics::mean(sum - acc1),
                                             1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::variance(acc2),
                    maths::CBasicStatistics::variance(sum - acc1), 1e-10);
            }
        }
        LOG_DEBUG(<< "Test mean, variance and skew");
        {
            TMeanVarSkewAccumulator acc1;
            TMeanVarSkewAccumulator acc2;

            TDoubleVec samples;
            rng.generateLogNormalSamples(1.1, 1.0, 40u, samples);

            for (std::size_t j = 1u; j < samples.size(); ++j) {
                LOG_DEBUG(<< "split = " << j << "/" << samples.size() - j);

                for (std::size_t i = 0u; i < j; ++i) {
                    acc1.add(samples[i]);
                }
                for (std::size_t i = j; i < samples.size(); ++i) {
                    acc2.add(samples[i]);
                }

                TMeanVarSkewAccumulator sum = acc1 + acc2;

                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(acc1),
                                    maths::CBasicStatistics::count(sum - acc2));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc1),
                                             maths::CBasicStatistics::mean(sum - acc2),
                                             1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::variance(acc1),
                    maths::CBasicStatistics::variance(sum - acc2), 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::skewness(acc1),
                    maths::CBasicStatistics::skewness(sum - acc2), 1e-10);
                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(acc2),
                                    maths::CBasicStatistics::count(sum - acc1));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(acc2),
                                             maths::CBasicStatistics::mean(sum - acc1),
                                             1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::variance(acc2),
                    maths::CBasicStatistics::variance(sum - acc1), 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::skewness(acc2),
                    maths::CBasicStatistics::skewness(sum - acc1), 1e-10);
            }
        }
    }

    LOG_DEBUG(<< "test vector");
    {
        using TVectorMeanAccumulator =
            maths::CBasicStatistics::SSampleMean<maths::CVectorNx1<double, 4>>::TAccumulator;
        using TVectorMeanVarAccumulator =
            maths::CBasicStatistics::SSampleMeanVar<maths::CVectorNx1<double, 4>>::TAccumulator;
        using TVectorMeanVarSkewAccumulator =
            maths::CBasicStatistics::SSampleMeanVarSkew<maths::CVectorNx1<double, 4>>::TAccumulator;

        test::CRandomNumbers rng;

        {
            LOG_DEBUG(<< "Test mean");

            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 1.0, 120, samples);

            TMeanAccumulator means[4];
            TVectorMeanAccumulator vectorMean;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                maths::CVectorNx1<double, 4> v;
                for (std::size_t j = 0u; j < 4; ++i, ++j) {
                    means[j].add(samples[i]);
                    v(j) = samples[i];
                }
                LOG_DEBUG(<< "v = " << v);
                vectorMean.add(v);

                BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::count(means[0]),
                                    maths::CBasicStatistics::count(vectorMean));
                for (std::size_t j = 0u; j < 4; ++j) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::mean(means[j]),
                        (maths::CBasicStatistics::mean(vectorMean))(j), 1e-14);
                }
            }
        }
        {
            LOG_DEBUG(<< "Test mean and variance");

            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 1.0, 120, samples);

            TMeanVarAccumulator meansAndVariances[4];
            TVectorMeanVarAccumulator vectorMeanAndVariances;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                maths::CVectorNx1<double, 4> v;
                for (std::size_t j = 0u; j < 4; ++i, ++j) {
                    meansAndVariances[j].add(samples[i]);
                    v(j) = samples[i];
                }
                LOG_DEBUG(<< "v = " << v);
                vectorMeanAndVariances.add(v);

                BOOST_REQUIRE_EQUAL(
                    maths::CBasicStatistics::count(meansAndVariances[0]),
                    maths::CBasicStatistics::count(vectorMeanAndVariances));
                for (std::size_t j = 0u; j < 4; ++j) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::mean(meansAndVariances[j]),
                        (maths::CBasicStatistics::mean(vectorMeanAndVariances))(j), 1e-14);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::variance(meansAndVariances[j]),
                        (maths::CBasicStatistics::variance(vectorMeanAndVariances))(j),
                        1e-14);
                }
            }
        }
        {
            LOG_DEBUG(<< "Test mean, variance and skew");

            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 1.0, 120, samples);

            TMeanVarSkewAccumulator meansVariancesAndSkews[4];
            TVectorMeanVarSkewAccumulator vectorMeanVarianceAndSkew;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                maths::CVectorNx1<double, 4> v;
                for (std::size_t j = 0u; j < 4; ++i, ++j) {
                    meansVariancesAndSkews[j].add(samples[i]);
                    v(j) = samples[i];
                }
                LOG_DEBUG(<< "v = " << v);
                vectorMeanVarianceAndSkew.add(v);

                BOOST_REQUIRE_EQUAL(
                    maths::CBasicStatistics::count(meansVariancesAndSkews[0]),
                    maths::CBasicStatistics::count(vectorMeanVarianceAndSkew));
                for (std::size_t j = 0u; j < 4; ++j) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::mean(meansVariancesAndSkews[j]),
                        (maths::CBasicStatistics::mean(vectorMeanVarianceAndSkew))(j), 1e-14);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::variance(meansVariancesAndSkews[j]),
                        (maths::CBasicStatistics::variance(vectorMeanVarianceAndSkew))(j),
                        1e-14);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::skewness(meansVariancesAndSkews[j]),
                        (maths::CBasicStatistics::skewness(vectorMeanVarianceAndSkew))(j),
                        1e-14);
                }
            }
        }
    }

    LOG_DEBUG(<< "Test persistence of collections");
    {
        LOG_DEBUG(<< "Test means");
        {
            TMeanAccumulatorVec moments(1);
            moments[0].add(2.0);
            moments[0].add(3.0);

            {
                core::CRapidXmlStatePersistInserter inserter("root");
                core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG(<< "Moments XML representation:\n" << xml);

                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanAccumulatorVec restored;
                BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                    SRestore(), std::ref(restored), std::placeholders::_1)));
                LOG_DEBUG(<< "restored = " << core::CContainerPrinter::print(restored));
                BOOST_REQUIRE_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    BOOST_REQUIRE_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }

            moments.push_back(TMeanAccumulator());
            moments.push_back(TMeanAccumulator());
            moments[1].add(3.0);
            moments[1].add(6.0);
            moments[2].add(10.0);
            moments[2].add(11.0);
            moments[2].add(12.0);

            {
                core::CRapidXmlStatePersistInserter inserter("root");
                core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG(<< "Moments XML representation:\n" << xml);

                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanAccumulatorVec restored;
                BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                    SRestore(), std::ref(restored), std::placeholders::_1)));
                LOG_DEBUG(<< "restored = " << core::CContainerPrinter::print(restored));
                BOOST_REQUIRE_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    BOOST_REQUIRE_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }
        }
        LOG_DEBUG(<< "Test means and variances");
        {
            TMeanVarAccumulatorVec moments(1);
            moments[0].add(2.0);
            moments[0].add(3.0);
            moments[0].add(3.5);
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG(<< "Moments XML representation:\n" << xml);

                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarAccumulatorVec restored;
                BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                    SRestore(), std::ref(restored), std::placeholders::_1)));
                LOG_DEBUG(<< "restored = " << core::CContainerPrinter::print(restored));
                BOOST_REQUIRE_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    BOOST_REQUIRE_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }

            moments.push_back(TMeanVarAccumulator());
            moments.push_back(TMeanVarAccumulator());
            moments[1].add(3.0);
            moments[1].add(6.0);
            moments[1].add(6.0);
            moments[2].add(10.0);
            moments[2].add(11.0);
            moments[2].add(12.0);
            moments[2].add(12.0);
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG(<< "Moments XML representation:\n" << xml);

                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarAccumulatorVec restored;
                BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                    SRestore(), std::ref(restored), std::placeholders::_1)));
                LOG_DEBUG(<< "restored = " << core::CContainerPrinter::print(restored));
                BOOST_REQUIRE_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    BOOST_REQUIRE_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }
        }
        LOG_DEBUG(<< "Test means, variances and skews");
        {
            TMeanVarSkewAccumulatorVec moments(1);
            moments[0].add(2.0);
            moments[0].add(3.0);
            moments[0].add(3.5);
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG(<< "Moments XML representation:\n" << xml);

                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarSkewAccumulatorVec restored;
                BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                    SRestore(), std::ref(restored), std::placeholders::_1)));
                LOG_DEBUG(<< "restored = " << core::CContainerPrinter::print(restored));
                BOOST_REQUIRE_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    BOOST_REQUIRE_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }

            moments.push_back(TMeanVarSkewAccumulator());
            moments.push_back(TMeanVarSkewAccumulator());
            moments[1].add(3.0);
            moments[1].add(6.0);
            moments[1].add(6.0);
            moments[2].add(10.0);
            moments[2].add(11.0);
            moments[2].add(12.0);
            moments[2].add(12.0);
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG(<< "Moments XML representation:\n" << xml);

                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarSkewAccumulatorVec restored;
                BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                    SRestore(), std::ref(restored), std::placeholders::_1)));
                LOG_DEBUG(<< "restored = " << core::CContainerPrinter::print(restored));
                BOOST_REQUIRE_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    BOOST_REQUIRE_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }
        }
    }

    BOOST_REQUIRE_EQUAL(
        true, core::memory_detail::SDynamicSizeAlwaysZero<TMeanAccumulator>::value());
    BOOST_REQUIRE_EQUAL(
        true, core::memory_detail::SDynamicSizeAlwaysZero<TMeanVarAccumulator>::value());
    BOOST_REQUIRE_EQUAL(
        true, core::memory_detail::SDynamicSizeAlwaysZero<TMeanVarSkewAccumulator>::value());
}

BOOST_AUTO_TEST_CASE(testVectorCentralMoments) {
    using TDouble2Vec = core::CSmallVector<double, 2>;

    {
        TMeanAccumulator2Vec moments1(2);
        TMeanAccumulatorVec moments2(2);
        moments1[0].add(2.0);
        moments1[0].add(5.0);
        moments1[0].add(2.9);
        moments1[1].add(4.0);
        moments1[1].add(3.0);
        moments2[0].add(2.0);
        moments2[0].add(5.0);
        moments2[0].add(2.9);
        moments2[1].add(4.0);
        moments2[1].add(3.0);
        TDouble2Vec counts1 = maths::CBasicStatistics::count(moments1);
        TDouble2Vec means1 = maths::CBasicStatistics::mean(moments1);
        TDoubleVec counts2 = maths::CBasicStatistics::count(moments2);
        TDoubleVec means2 = maths::CBasicStatistics::mean(moments2);
        BOOST_REQUIRE_EQUAL(std::string("[3, 2]"), core::CContainerPrinter::print(counts1));
        BOOST_REQUIRE_EQUAL(std::string("[3.3, 3.5]"),
                            core::CContainerPrinter::print(means1));
        BOOST_REQUIRE_EQUAL(std::string("[3, 2]"), core::CContainerPrinter::print(counts2));
        BOOST_REQUIRE_EQUAL(std::string("[3.3, 3.5]"),
                            core::CContainerPrinter::print(means2));
    }
    {
        TMeanVarAccumulator2Vec moments1(2);
        TMeanVarAccumulatorVec moments2(2);
        moments1[0].add(2.0);
        moments1[0].add(4.0);
        moments1[1].add(3.0);
        moments1[1].add(4.0);
        moments1[1].add(5.0);
        moments2[0].add(2.0);
        moments2[0].add(4.0);
        moments2[1].add(3.0);
        moments2[1].add(4.0);
        moments2[1].add(5.0);
        TDouble2Vec counts1 = maths::CBasicStatistics::count(moments1);
        TDouble2Vec means1 = maths::CBasicStatistics::mean(moments1);
        TDouble2Vec vars1 = maths::CBasicStatistics::variance(moments1);
        TDouble2Vec mlvars1 = maths::CBasicStatistics::maximumLikelihoodVariance(moments1);
        TDoubleVec counts2 = maths::CBasicStatistics::count(moments2);
        TDoubleVec means2 = maths::CBasicStatistics::mean(moments2);
        TDoubleVec vars2 = maths::CBasicStatistics::variance(moments2);
        TDouble2Vec mlvars2 = maths::CBasicStatistics::maximumLikelihoodVariance(moments2);
        BOOST_REQUIRE_EQUAL(std::string("[2, 3]"), core::CContainerPrinter::print(counts1));
        BOOST_REQUIRE_EQUAL(std::string("[3, 4]"), core::CContainerPrinter::print(means1));
        BOOST_REQUIRE_EQUAL(std::string("[2, 1]"), core::CContainerPrinter::print(vars1));
        BOOST_REQUIRE_EQUAL(std::string("[1, 0.6666667]"),
                            core::CContainerPrinter::print(mlvars1));
        BOOST_REQUIRE_EQUAL(std::string("[2, 3]"), core::CContainerPrinter::print(counts2));
        BOOST_REQUIRE_EQUAL(std::string("[3, 4]"), core::CContainerPrinter::print(means2));
        BOOST_REQUIRE_EQUAL(std::string("[2, 1]"), core::CContainerPrinter::print(vars2));
        BOOST_REQUIRE_EQUAL(std::string("[1, 0.6666667]"),
                            core::CContainerPrinter::print(mlvars2));
    }
    {
        TMeanVarSkewAccumulator2Vec moments1(2);
        TMeanVarSkewAccumulatorVec moments2(2);
        moments1[0].add(2.0);
        moments1[0].add(4.0);
        moments1[1].add(2.0);
        moments1[1].add(5.0);
        moments1[1].add(5.0);
        moments2[0].add(2.0);
        moments2[0].add(4.0);
        moments2[1].add(2.0);
        moments2[1].add(5.0);
        moments2[1].add(5.0);
        TDouble2Vec counts1 = maths::CBasicStatistics::count(moments1);
        TDouble2Vec means1 = maths::CBasicStatistics::mean(moments1);
        TDouble2Vec vars1 = maths::CBasicStatistics::variance(moments1);
        TDouble2Vec mlvars1 = maths::CBasicStatistics::maximumLikelihoodVariance(moments1);
        TDouble2Vec skews1 = maths::CBasicStatistics::skewness(moments1);
        TDoubleVec counts2 = maths::CBasicStatistics::count(moments2);
        TDoubleVec means2 = maths::CBasicStatistics::mean(moments2);
        TDoubleVec vars2 = maths::CBasicStatistics::variance(moments2);
        TDouble2Vec mlvars2 = maths::CBasicStatistics::maximumLikelihoodVariance(moments2);
        TDouble2Vec skews2 = maths::CBasicStatistics::skewness(moments2);
        BOOST_REQUIRE_EQUAL(std::string("[2, 3]"), core::CContainerPrinter::print(counts1));
        BOOST_REQUIRE_EQUAL(std::string("[3, 4]"), core::CContainerPrinter::print(means1));
        BOOST_REQUIRE_EQUAL(std::string("[2, 3]"), core::CContainerPrinter::print(vars1));
        BOOST_REQUIRE_EQUAL(std::string("[1, 2]"), core::CContainerPrinter::print(mlvars1));
        BOOST_REQUIRE_EQUAL(std::string("[0, -0.3849002]"),
                            core::CContainerPrinter::print(skews1));
        BOOST_REQUIRE_EQUAL(std::string("[2, 3]"), core::CContainerPrinter::print(counts2));
        BOOST_REQUIRE_EQUAL(std::string("[3, 4]"), core::CContainerPrinter::print(means2));
        BOOST_REQUIRE_EQUAL(std::string("[2, 3]"), core::CContainerPrinter::print(vars2));
        BOOST_REQUIRE_EQUAL(std::string("[1, 2]"), core::CContainerPrinter::print(mlvars2));
        BOOST_REQUIRE_EQUAL(std::string("[0, -0.3849002]"),
                            core::CContainerPrinter::print(skews2));
    }
}

BOOST_AUTO_TEST_CASE(testCovariances) {
    LOG_DEBUG(<< "N(3,I)");
    {
        const double raw[][3] = {
            {2.58894, 2.87211, 1.62609}, {3.88246, 2.98577, 2.70981},
            {2.03317, 3.33715, 2.93560}, {3.30100, 4.38844, 1.65705},
            {2.12426, 2.21127, 2.57000}, {4.21041, 4.20745, 1.90752},
            {3.56139, 3.14454, 0.89316}, {4.29444, 1.58715, 3.58402},
            {3.06731, 3.91581, 2.85951}, {3.62798, 2.28786, 2.89994},
            {2.05834, 2.96137, 3.57654}, {2.72185, 3.36003, 3.09708},
            {0.94924, 2.19797, 3.30941}, {2.11159, 2.49182, 3.56793},
            {3.10364, 0.32747, 3.62487}, {2.28235, 3.83542, 3.35942},
            {3.30549, 2.95951, 2.97006}, {3.05787, 2.94188, 2.64095},
            {3.98245, 2.02892, 3.07909}, {3.81189, 2.89389, 3.81389},
            {3.32811, 3.88484, 4.17866}, {2.06964, 3.80683, 2.46835},
            {4.58989, 2.00321, 1.93029}, {2.51484, 4.46106, 3.71248},
            {3.30729, 2.44768, 3.43241}, {3.52222, 2.91724, 1.49631},
            {1.71826, 4.79752, 4.38398}, {3.14173, 3.16237, 2.49654},
            {3.26538, 2.21858, 5.05477}, {2.88352, 1.94396, 3.08744}};

        const double expectedMean[] = {3.013898, 2.952637, 2.964104};
        const double expectedCovariances[][3] = {{0.711903, -0.174535, -0.199460},
                                                 {-0.174535, 0.935285, -0.091192},
                                                 {-0.199460, -0.091192, 0.833710}};

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 3>> covariances1(3);
        maths::CBasicStatistics::SSampleCovariances<maths::CVector<double>> covariances2(3);
        maths::CBasicStatistics::SSampleCovariances<maths::CDenseVector<double>> covariances3(3);

        for (std::size_t i = 0u; i < boost::size(raw); ++i) {
            LOG_DEBUG(<< "v = " << core::CContainerPrinter::print(raw[i]));
            covariances1.add(maths::CVectorNx1<double, 3>(raw[i]));
            covariances2.add(maths::CVector<double>(std::begin(raw[i]), std::end(raw[i])));
            maths::CDenseVector<double> v(3);
            v << raw[i][0], raw[i][1], raw[i][2];
            covariances3.add(v);
        }

        LOG_DEBUG(<< "count1 = " << maths::CBasicStatistics::count(covariances1));
        LOG_DEBUG(<< "mean1 = " << maths::CBasicStatistics::mean(covariances1));
        LOG_DEBUG(<< "covariances1 = " << maths::CBasicStatistics::covariances(covariances1));
        LOG_DEBUG(<< "count2 = " << maths::CBasicStatistics::count(covariances2));
        LOG_DEBUG(<< "mean2 = " << maths::CBasicStatistics::mean(covariances2));
        LOG_DEBUG(<< "covariances2 = " << maths::CBasicStatistics::covariances(covariances2));
        LOG_DEBUG(<< "count3 = " << maths::CBasicStatistics::count(covariances3));
        LOG_DEBUG(<< "mean3 = " << maths::CBasicStatistics::mean(covariances3).transpose());
        LOG_DEBUG(<< "covariances3 =\n"
                  << maths::CBasicStatistics::covariances(covariances3));

        BOOST_REQUIRE_EQUAL(static_cast<double>(boost::size(raw)),
                            maths::CBasicStatistics::count(covariances1));
        BOOST_REQUIRE_EQUAL(static_cast<double>(boost::size(raw)),
                            maths::CBasicStatistics::count(covariances2));
        BOOST_REQUIRE_EQUAL(static_cast<double>(boost::size(raw)),
                            maths::CBasicStatistics::count(covariances3));

        for (std::size_t i = 0u; i < 3; ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expectedMean[i], (maths::CBasicStatistics::mean(covariances1))(i), 2e-6);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expectedMean[i], (maths::CBasicStatistics::mean(covariances2))(i), 2e-6);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expectedMean[i], (maths::CBasicStatistics::mean(covariances3))(i), 2e-6);
            for (std::size_t j = 0u; j < 3; ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    expectedCovariances[i][j],
                    (maths::CBasicStatistics::covariances(covariances1))(i, j), 2e-6);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    expectedCovariances[i][j],
                    (maths::CBasicStatistics::covariances(covariances2))(i, j), 2e-6);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    expectedCovariances[i][j],
                    (maths::CBasicStatistics::covariances(covariances3))(i, j), 2e-6);
            }
        }

        bool dynamicSizeAlwaysZero = core::memory_detail::SDynamicSizeAlwaysZero<
            maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 3>>>::value();
        BOOST_REQUIRE_EQUAL(true, dynamicSizeAlwaysZero);
    }

    {
        using TVectorVec = std::vector<maths::CVectorNx1<double, 4>>;

        double mean_[] = {1.0, 3.0, 2.0, 7.0};
        maths::CVectorNx1<double, 4> mean(mean_);

        double covariances1_[] = {1.0, 1.0, 1.0, 1.0};
        double covariances2_[] = {-1.0, 1.0, 0.0, 0.0};
        double covariances3_[] = {-1.0, -1.0, 2.0, 0.0};
        double covariances4_[] = {-1.0, -1.0, -1.0, 3.0};

        maths::CVectorNx1<double, 4> covariances1(covariances1_);
        maths::CVectorNx1<double, 4> covariances2(covariances2_);
        maths::CVectorNx1<double, 4> covariances3(covariances3_);
        maths::CVectorNx1<double, 4> covariances4(covariances4_);

        maths::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::CSymmetricMatrixNxN<double, 4>(
                       maths::E_OuterProduct, covariances1 / covariances1.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(
                      maths::E_OuterProduct, covariances2 / covariances2.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(
                      maths::E_OuterProduct, covariances3 / covariances3.euclidean()) +
            2.0 * maths::CSymmetricMatrixNxN<double, 4>(
                      maths::E_OuterProduct, covariances4 / covariances4.euclidean()));

        std::size_t n = 10000u;

        TVectorVec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, n, samples);

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 4>> sampleCovariance(4);
        sampleCovariance.add(samples);

        LOG_DEBUG(<< "expected mean = " << mean);
        LOG_DEBUG(<< "expected covariances = " << covariance);

        LOG_DEBUG(<< "mean = " << maths::CBasicStatistics::mean(sampleCovariance));
        LOG_DEBUG(<< "covariances = "
                  << maths::CBasicStatistics::covariances(sampleCovariance));

        for (std::size_t i = 0u; i < 4; ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                mean(i), (maths::CBasicStatistics::mean(sampleCovariance))(i), 0.05);
            for (std::size_t j = 0u; j < 4; ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    covariance(i, j),
                    (maths::CBasicStatistics::covariances(sampleCovariance))(i, j), 0.16);
            }
        }
    }

    {
        test::CRandomNumbers rng;

        std::vector<double> coordinates;
        rng.generateUniformSamples(5.0, 10.0, 400, coordinates);

        std::vector<maths::CVectorNx1<double, 4>> points;
        for (std::size_t i = 0u; i < coordinates.size(); i += 4) {
            double c[] = {coordinates[i + 0], coordinates[i + 1],
                          coordinates[i + 2], coordinates[i + 3]};
            points.push_back(maths::CVectorNx1<double, 4>(c));
        }

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 4>> expectedSampleCovariances(
            4);
        for (std::size_t i = 0u; i < points.size(); ++i) {
            expectedSampleCovariances.add(points[i]);
        }

        std::string expectedDelimited = expectedSampleCovariances.toDelimited();
        LOG_DEBUG(<< "delimited = " << expectedDelimited);

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 4>> sampleCovariances(4);
        BOOST_TEST_REQUIRE(sampleCovariances.fromDelimited(expectedDelimited));

        BOOST_REQUIRE_EQUAL(expectedSampleCovariances.checksum(),
                            sampleCovariances.checksum());

        std::string delimited = sampleCovariances.toDelimited();
        BOOST_REQUIRE_EQUAL(expectedDelimited, delimited);
    }
}

BOOST_AUTO_TEST_CASE(testCovariancesLedoitWolf) {
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TVector2 = maths::CVectorNx1<double, 2>;
    using TVector2Vec = std::vector<TVector2>;
    using TMatrix2 = maths::CSymmetricMatrixNxN<double, 2>;
    using TDenseVector = maths::CDenseVector<double>;
    using TDenseVectorVec = std::vector<TDenseVector>;
    using TDenseMatrix = maths::CDenseMatrix<double>;

    test::CRandomNumbers rng;

    double means[][2] = {
        {10.0, 10.0}, {20.0, 150.0}, {-10.0, -20.0}, {-20.0, 40.0}, {40.0, 90.0}};

    double covariances[][2][2] = {{{40.0, 0.0}, {0.0, 40.0}},
                                  {{20.0, 5.0}, {5.0, 10.0}},
                                  {{300.0, -70.0}, {-70.0, 60.0}},
                                  {{100.0, 20.0}, {20.0, 60.0}},
                                  {{50.0, -10.0}, {-10.0, 60.0}}};

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator error;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator errorLW;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        LOG_DEBUG(<< "*** test " << i << " ***");

        TDoubleVec mean(std::begin(means[i]), std::end(means[i]));
        TDoubleVecVec covariance;
        for (std::size_t j = 0u; j < boost::size(covariances[i]); ++j) {
            covariance.emplace_back(std::begin(covariances[i][j]),
                                    std::end(covariances[i][j]));
        }
        TMatrix2 covExpected(covariance);
        LOG_DEBUG(<< "cov expected = " << covExpected);

        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 50, samples);

        // Test the frobenius norm of the error in the covariance matrix.

        for (std::size_t j = 3u; j < samples.size(); ++j) {
            TVector2Vec jsamples;
            TDenseVectorVec jsamples2;
            for (std::size_t k = 0u; k < j; ++k) {
                jsamples.emplace_back(samples[k]);
                TDenseVector v(2);
                v << samples[k][0], samples[k][1];
                jsamples2.push_back(v);
            }

            maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 2>> cov(2);
            cov.add(jsamples);

            maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 2>> covLW(2);
            maths::CBasicStatistics::covariancesLedoitWolf(jsamples, covLW);

            maths::CBasicStatistics::SSampleCovariances<maths::CDenseVector<double>> covLW2(2);
            maths::CBasicStatistics::covariancesLedoitWolf(jsamples2, covLW2);

            const TMatrix2& covML =
                maths::CBasicStatistics::maximumLikelihoodCovariances(cov);
            const TMatrix2& covLWML =
                maths::CBasicStatistics::maximumLikelihoodCovariances(covLW);
            const TDenseMatrix& covLWML2 =
                maths::CBasicStatistics::maximumLikelihoodCovariances(covLW2);

            for (std::size_t r = 0u; r < 2; ++r) {
                for (std::size_t c = 0u; c < 2; ++c) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(covLWML(r, c), covLWML2(r, c), 1e-10);
                }
            }

            double errorML = (covML - covExpected).frobenius();
            double errorLWML = (covLWML - covExpected).frobenius();

            if (j % 5 == 0) {
                LOG_DEBUG(<< "cov ML   = " << covML);
                LOG_DEBUG(<< "cov LWML = " << covLWML);
                LOG_DEBUG(<< "error ML = " << errorML << ", error LWML = " << errorLWML);
            }
            BOOST_TEST_REQUIRE(errorLWML < 6.0 * errorML);
            error.add(errorML / covExpected.frobenius());
            errorLW.add(errorLWML / covExpected.frobenius());
        }
    }

    LOG_DEBUG(<< "error    = " << error);
    LOG_DEBUG(<< "error LW = " << errorLW);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(errorLW) <
                       0.9 * maths::CBasicStatistics::mean(error));
}

BOOST_AUTO_TEST_CASE(testMedian) {
    {
        maths::CBasicStatistics::TDoubleVec sampleVec;

        double median = maths::CBasicStatistics::median(sampleVec);

        BOOST_REQUIRE_EQUAL(0.0, median);
    }
    {
        double sample[] = {1.0};

        maths::CBasicStatistics::TDoubleVec sampleVec(
            sample, sample + sizeof(sample) / sizeof(sample[0]));

        double median = maths::CBasicStatistics::median(sampleVec);

        BOOST_REQUIRE_EQUAL(1.0, median);
    }
    {
        double sample[] = {2.0, 1.0};

        maths::CBasicStatistics::TDoubleVec sampleVec(
            sample, sample + sizeof(sample) / sizeof(sample[0]));

        double median = maths::CBasicStatistics::median(sampleVec);

        BOOST_REQUIRE_EQUAL(1.5, median);
    }
    {
        double sample[] = {3.0, 1.0, 2.0};

        maths::CBasicStatistics::TDoubleVec sampleVec(
            sample, sample + sizeof(sample) / sizeof(sample[0]));

        double median = maths::CBasicStatistics::median(sampleVec);

        BOOST_REQUIRE_EQUAL(2.0, median);
    }
    {
        double sample[] = {3.0, 5.0, 9.0, 1.0, 2.0, 6.0, 7.0, 4.0, 8.0};

        maths::CBasicStatistics::TDoubleVec sampleVec(
            sample, sample + sizeof(sample) / sizeof(sample[0]));

        double median = maths::CBasicStatistics::median(sampleVec);

        BOOST_REQUIRE_EQUAL(5.0, median);
    }
    {
        double sample[] = {3.0, 5.0, 10.0, 2.0, 6.0, 7.0, 1.0, 9.0, 4.0, 8.0};

        maths::CBasicStatistics::TDoubleVec sampleVec(
            sample, sample + sizeof(sample) / sizeof(sample[0]));

        double median = maths::CBasicStatistics::median(sampleVec);

        BOOST_REQUIRE_EQUAL(5.5, median);
    }
}

BOOST_AUTO_TEST_CASE(testMad) {
    using TSizeVec = std::vector<std::size_t>;

    // Edge cases 0, 1, 2 elements and > half values equal.
    TDoubleVec samples;
    samples.assign({5.0});
    BOOST_REQUIRE_EQUAL(0.0, maths::CBasicStatistics::mad(samples));
    samples.assign({5.0, 6.0});
    BOOST_REQUIRE_EQUAL(0.5, maths::CBasicStatistics::mad(samples));
    samples.assign({6.0, 6.0, 6.0, 2.0, -100.0});
    BOOST_REQUIRE_EQUAL(0.0, maths::CBasicStatistics::mad(samples));
    samples.assign({6.0, 6.0, 6.0, 6.0, -100.0, 1.0});
    BOOST_REQUIRE_EQUAL(0.0, maths::CBasicStatistics::mad(samples));

    // Odd/even number of samples.
    samples.assign({12.2, 11.8, 1.0, 30.2, 5.9, 209.0, -390.3, 37.0});
    BOOST_REQUIRE_CLOSE_ABSOLUTE(14.6, maths::CBasicStatistics::mad(samples), 1e-15);
    samples.assign({12.2, 11.8, 1.0, 30.2, 5.9, 209.0, -390.3, 37.0, 51.2});
    BOOST_REQUIRE_CLOSE_ABSOLUTE(18.0, maths::CBasicStatistics::mad(samples), 1e-15);

    // Random.
    test::CRandomNumbers rng;
    TSizeVec size;
    for (std::size_t test = 0; test < 100; ++test) {
        rng.generateUniformSamples(1, 40, 1, size);
        rng.generateUniformSamples(0.0, 100.0, size[0], samples);
        double mad{maths::CBasicStatistics::mad(samples)};
        double median{maths::CBasicStatistics::median(samples)};
        for (auto& sample : samples) {
            sample = std::fabs(sample - median);
        }
        BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::median(samples), mad);
    }
}

BOOST_AUTO_TEST_CASE(testOrderStatistics) {
    // Test that the order statistics accumulators work for finding min and max
    // elements of a collection.

    using TMinStatsStack = maths::CBasicStatistics::COrderStatisticsStack<double, 2u>;
    using TMaxStatsStack =
        maths::CBasicStatistics::COrderStatisticsStack<double, 3u, std::greater<double>>;
    using TMinStatsHeap = maths::CBasicStatistics::COrderStatisticsHeap<double>;
    using TMaxStatsHeap =
        maths::CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;

    {
        // Test on the stack min, max, combine and persist and restore.

        double data[] = {1.0, 2.3, 1.1, 1.0, 5.0, 3.0, 11.0, 0.2, 15.8, 12.3};

        TMinStatsStack minValues;
        TMaxStatsStack maxValues;
        TMinStatsStack minFirstHalf;
        TMinStatsStack minSecondHalf;

        for (size_t i = 0; i < boost::size(data); ++i) {
            minValues.add(data[i]);
            maxValues.add(data[i]);
            (2 * i < boost::size(data) ? minFirstHalf : minSecondHalf).add(data[i]);
        }

        std::sort(std::begin(data), std::end(data));
        minValues.sort();
        LOG_DEBUG(<< "x_1 = " << minValues[0] << ", x_2 = " << minValues[1]);
        BOOST_TEST_REQUIRE(std::equal(minValues.begin(), minValues.end(), data));

        std::sort(std::begin(data), std::end(data), std::greater<double>());
        maxValues.sort();
        LOG_DEBUG(<< "x_n = " << maxValues[0] << ", x_(n-1) = " << maxValues[1]
                  << ", x_(n-2) = " << maxValues[2]);
        BOOST_TEST_REQUIRE(std::equal(maxValues.begin(), maxValues.end(), data));

        BOOST_REQUIRE_EQUAL(static_cast<size_t>(2), minValues.count());
        BOOST_REQUIRE_EQUAL(static_cast<size_t>(3), maxValues.count());

        TMinStatsStack minFirstPlusSecondHalf = (minFirstHalf + minSecondHalf);
        minFirstPlusSecondHalf.sort();
        BOOST_TEST_REQUIRE(std::equal(minValues.begin(), minValues.end(),
                                      minFirstPlusSecondHalf.begin()));

        // Test persist is idempotent.

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, minValues.toDelimited());
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "Stats XML representation:\n" << origXml);

        // Restore the XML into stats object.
        TMinStatsStack restoredMinValues;
        {
            core::CRapidXmlParser parser;
            BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                SRestore(), std::ref(restoredMinValues), std::placeholders::_1)));
        }

        // The XML representation of the new stats object should be unchanged.
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, restoredMinValues.toDelimited());
            inserter.toXml(newXml);
        }
        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }

    {
        // Test on the heap min, max, combine and persist and restore.

        double data[] = {1.0, 2.3, 1.1, 1.0, 5.0, 3.0, 11.0, 0.2, 15.8, 12.3};

        TMinStatsHeap min2Values(2);
        TMaxStatsHeap max3Values(3);
        TMaxStatsHeap max20Values(20);

        for (size_t i = 0; i < boost::size(data); ++i) {
            min2Values.add(data[i]);
            max3Values.add(data[i]);
            max20Values.add(data[i]);
        }

        std::sort(std::begin(data), std::end(data));
        min2Values.sort();
        LOG_DEBUG(<< "x_1 = " << min2Values[0] << ", x_2 = " << min2Values[1]);
        BOOST_TEST_REQUIRE(std::equal(min2Values.begin(), min2Values.end(), data));

        std::sort(std::begin(data), std::end(data), std::greater<double>());
        max3Values.sort();
        LOG_DEBUG(<< "x_n = " << max3Values[0] << ", x_(n-1) = " << max3Values[1]
                  << ", x_(n-2) = " << max3Values[2]);
        BOOST_TEST_REQUIRE(std::equal(max3Values.begin(), max3Values.end(), data));

        max20Values.sort();
        BOOST_REQUIRE_EQUAL(boost::size(data), max20Values.count());
        BOOST_TEST_REQUIRE(std::equal(max20Values.begin(), max20Values.end(), data));

        // Test persist is idempotent.

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, max20Values.toDelimited());
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "Stats XML representation:\n" << origXml);

        // Restore the XML into stats object.
        TMinStatsHeap restoredMaxValues(20);
        {
            core::CRapidXmlParser parser;
            BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
                SRestore(), std::ref(restoredMaxValues), std::placeholders::_1)));
        }

        // The XML representation of the new stats object should be unchanged.
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, restoredMaxValues.toDelimited());
            inserter.toXml(newXml);
        }
        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }
    {
        // Test we correctly age the minimum value accumulator.
        TMinStatsStack test;
        test.add(15.0);
        test.age(0.5);
        BOOST_REQUIRE_EQUAL(30.0, test[0]);
    }
    {
        // Test we correctly age the maximum value accumulator.
        TMaxStatsStack test;
        test.add(15.0);
        test.age(0.5);
        BOOST_REQUIRE_EQUAL(7.5, test[0]);
    }
    {
        // Test biggest.
        TMinStatsHeap min(5);
        TMaxStatsHeap max(5);
        min.add(1.0);
        max.add(1.0);
        BOOST_REQUIRE_EQUAL(1.0, min.biggest());
        BOOST_REQUIRE_EQUAL(1.0, max.biggest());
        std::size_t i{0};
        for (auto value : {3.6, -6.1, 1.0, 3.4}) {
            min.add(value);
            max.add(value);
            if (i++ == 0) {
                BOOST_REQUIRE_EQUAL(3.6, min.biggest());
                BOOST_REQUIRE_EQUAL(1.0, max.biggest());
            } else {
                BOOST_REQUIRE_EQUAL(3.6, min.biggest());
                BOOST_REQUIRE_EQUAL(-6.1, max.biggest());
            }
        }
        min.add(0.9);
        max.add(0.9);
        BOOST_REQUIRE_EQUAL(3.4, min.biggest());
        BOOST_REQUIRE_EQUAL(0.9, max.biggest());
    }
    {
        // Test memory.
        BOOST_REQUIRE_EQUAL(
            true, core::memory_detail::SDynamicSizeAlwaysZero<TMinStatsStack>::value());
        BOOST_REQUIRE_EQUAL(
            true, core::memory_detail::SDynamicSizeAlwaysZero<TMaxStatsStack>::value());
        BOOST_REQUIRE_EQUAL(
            false, core::memory_detail::SDynamicSizeAlwaysZero<TMinStatsHeap>::value());
        BOOST_REQUIRE_EQUAL(
            false, core::memory_detail::SDynamicSizeAlwaysZero<TMaxStatsHeap>::value());
    }
    {
        // Test to from delimited with callback to persist values.

        using TDoubleDoublePr = std::pair<double, double>;
        using TDoubleDoublePrMinAccumulator =
            ml::maths::CBasicStatistics::COrderStatisticsStack<TDoubleDoublePr, 2u>;

        TDoubleDoublePrMinAccumulator orig;
        orig.add({1.0, 3.2});
        orig.add({3.1, 1.2});

        auto toDelimited = [](const TDoubleDoublePr& value) {
            return ml::core::CStringUtils::typeToStringPrecise(
                       value.first, ml::core::CIEEE754::E_DoublePrecision) +
                   ml::maths::CBasicStatistics::EXTERNAL_DELIMITER +
                   ml::core::CStringUtils::typeToStringPrecise(
                       value.second, ml::core::CIEEE754::E_DoublePrecision);
        };
        std::string delimited{orig.toDelimited(toDelimited)};
        LOG_DEBUG(<< "delimited = " << delimited);

        TDoubleDoublePrMinAccumulator restored;
        restored.fromDelimited(delimited, [](const std::string& value, TDoubleDoublePr& result) {
            std::size_t pos{value.find(ml::maths::CBasicStatistics::EXTERNAL_DELIMITER)};
            return ml::core::CStringUtils::stringToType(value.substr(0, pos),
                                                        result.first) &&
                   ml::core::CStringUtils::stringToType(value.substr(pos + 1),
                                                        result.second);
        });

        BOOST_REQUIRE_EQUAL(delimited, restored.toDelimited(toDelimited));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(orig[0].first, restored[0].first, 1e-15);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(orig[0].second, restored[0].second, 1e-15);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(orig[1].first, restored[1].first, 1e-15);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(orig[1].second, restored[1].second, 1e-15);
    }
}

BOOST_AUTO_TEST_CASE(testMinMax) {
    TDoubleVec positive{1.0, 2.7, 4.0, 0.3, 11.7};
    TDoubleVec negative{-3.7, -0.8, -18.2, -0.8};
    TDoubleVec mixed{1.3, -8.0, 2.1};

    {
        maths::CBasicStatistics::CMinMax<double> minmax;
        BOOST_TEST_REQUIRE(!minmax.initialized());
        minmax.add(positive);
        BOOST_TEST_REQUIRE(minmax.initialized());
        BOOST_REQUIRE_EQUAL(0.3, minmax.min());
        BOOST_REQUIRE_EQUAL(11.7, minmax.max());
        BOOST_REQUIRE_EQUAL(0.3, minmax.signMargin());
    }
    {
        maths::CBasicStatistics::CMinMax<double> minmax;
        BOOST_TEST_REQUIRE(!minmax.initialized());
        minmax.add(negative);
        BOOST_TEST_REQUIRE(minmax.initialized());
        BOOST_REQUIRE_EQUAL(-18.2, minmax.min());
        BOOST_REQUIRE_EQUAL(-0.8, minmax.max());
        BOOST_REQUIRE_EQUAL(-0.8, minmax.signMargin());
    }
    {
        maths::CBasicStatistics::CMinMax<double> minmax;
        BOOST_TEST_REQUIRE(!minmax.initialized());
        minmax.add(mixed);
        BOOST_TEST_REQUIRE(minmax.initialized());
        BOOST_REQUIRE_EQUAL(-8.0, minmax.min());
        BOOST_REQUIRE_EQUAL(2.1, minmax.max());
        BOOST_REQUIRE_EQUAL(0.0, minmax.signMargin());
    }
    {
        maths::CBasicStatistics::CMinMax<double> minmax1;
        maths::CBasicStatistics::CMinMax<double> minmax2;
        maths::CBasicStatistics::CMinMax<double> minmax12;
        minmax1.add(positive);
        minmax2.add(negative);
        minmax12.add(positive);
        minmax12.add(negative);
        BOOST_REQUIRE_EQUAL((minmax1 + minmax2).checksum(), minmax12.checksum());
    }
}

BOOST_AUTO_TEST_SUITE_END()
