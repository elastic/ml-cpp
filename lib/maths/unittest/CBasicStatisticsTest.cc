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

#include "CBasicStatisticsTest.h"

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSampling.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

#include <stdlib.h>

namespace {

typedef ml::maths::CBasicStatistics::SSampleMean<double>::TAccumulator        TMeanAccumulator;
typedef ml::maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator     TMeanVarAccumulator;
typedef ml::maths::CBasicStatistics::SSampleMeanVarSkew<double>::TAccumulator TMeanVarSkewAccumulator;
typedef ml::core::CSmallVector<TMeanAccumulator, 2>                           TMeanAccumulator2Vec;
typedef ml::core::CSmallVector<TMeanVarAccumulator, 2>                        TMeanVarAccumulator2Vec;
typedef ml::core::CSmallVector<TMeanVarSkewAccumulator, 2>                    TMeanVarSkewAccumulator2Vec;
typedef std::vector<TMeanAccumulator>                                         TMeanAccumulatorVec;
typedef std::vector<TMeanVarAccumulator>                                      TMeanVarAccumulatorVec;
typedef std::vector<TMeanVarSkewAccumulator>                                  TMeanVarSkewAccumulatorVec;

const std::string TAG("a");

struct SRestore {
    typedef bool result_type;

    template<typename T>
    bool operator()(std::vector<T> &restored, ml::core::CStateRestoreTraverser &traverser) const {
        return ml::core::CPersistUtils::restore(TAG, restored, traverser);
    }

    template<typename T>
    bool operator()(T &restored, ml::core::CStateRestoreTraverser &traverser) const {
        return restored.fromDelimited(traverser.value());
    }
};

}


CppUnit::Test *CBasicStatisticsTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CBasicStatisticsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testMean",
                               &CBasicStatisticsTest::testMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testCentralMoments",
                               &CBasicStatisticsTest::testCentralMoments) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testVectorCentralMoments",
                               &CBasicStatisticsTest::testVectorCentralMoments) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testCovariances",
                               &CBasicStatisticsTest::testCovariances) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testCovariancesLedoitWolf",
                               &CBasicStatisticsTest::testCovariancesLedoitWolf) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testMedian",
                               &CBasicStatisticsTest::testMedian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testOrderStatistics",
                               &CBasicStatisticsTest::testOrderStatistics) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBasicStatisticsTest>(
                               "CBasicStatisticsTest::testMinMax",
                               &CBasicStatisticsTest::testMinMax) );

    return suiteOfTests;
}

void CBasicStatisticsTest::testMean(void) {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testMean |");
    LOG_DEBUG("+---------------------------------+");

    double sample[] = { 0.9, 10.0, 5.6, 1.23, -12.3, 445.2, 0.0, 1.2 };

    ml::maths::CBasicStatistics::TDoubleVec sampleVec(sample, sample+sizeof(sample)/sizeof(sample[0]));

    double mean = ml::maths::CBasicStatistics::mean(sampleVec);

    // Compare with hand calculated value
    CPPUNIT_ASSERT_EQUAL(56.47875, mean);
}

void CBasicStatisticsTest::testCentralMoments(void) {
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testCentralMoments  |");
    LOG_DEBUG("+--------------------------------------------+");

    typedef std::vector<double> TDoubleVec;

    LOG_DEBUG("Test mean double");
    {
        double           samples[] = { 0.9, 10.0, 5.6, 1.23, -12.3, 7.2, 0.0, 1.2 };
        TMeanAccumulator acc;

        size_t count = sizeof(samples)/sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        CPPUNIT_ASSERT_EQUAL(count,
                             static_cast<size_t>(ml::maths::CBasicStatistics::count(acc)));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                     ml::maths::CBasicStatistics::mean(acc),
                                     0.000005);

        double n0 = ml::maths::CBasicStatistics::count(acc);
        ml::maths::CBasicStatistics::scale(0.5, acc);
        double n1 = ml::maths::CBasicStatistics::count(acc);
        CPPUNIT_ASSERT_EQUAL(n1, 0.5 * n0);
    }

    LOG_DEBUG("Test mean float");
    {
        typedef ml::maths::CBasicStatistics::SSampleMean<float>::TAccumulator TFloatMeanAccumulator;

        float samples[] = { 0.9f, 10.0f, 5.6f, 1.23f, -12.3f, 7.2f, 0.0f, 1.2f };

        TFloatMeanAccumulator acc;

        size_t count = sizeof(samples)/sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        CPPUNIT_ASSERT_EQUAL(count,
                             static_cast<size_t>(ml::maths::CBasicStatistics::count(acc)));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                     ml::maths::CBasicStatistics::mean(acc),
                                     0.000005);
    }

    LOG_DEBUG("Test mean and variance");
    {
        double samples[] = { 0.9, 10.0, 5.6, 1.23, -12.3, 7.2, 0.0, 1.2 };

        TMeanVarAccumulator acc;

        size_t count = sizeof(samples)/sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        CPPUNIT_ASSERT_EQUAL(count,
                             static_cast<size_t>(ml::maths::CBasicStatistics::count(acc)));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                     ml::maths::CBasicStatistics::mean(acc),
                                     0.000005);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(44.90633,
                                     ml::maths::CBasicStatistics::variance(acc),
                                     0.000005);

        double n0 = ml::maths::CBasicStatistics::count(acc);
        ml::maths::CBasicStatistics::scale(0.5, acc);
        double n1 = ml::maths::CBasicStatistics::count(acc);
        CPPUNIT_ASSERT_EQUAL(n1, 0.5 * n0);
    }

    LOG_DEBUG("Test mean, variance and skew");
    {
        double samples[] = { 0.9, 10.0, 5.6, 1.23, -12.3, 7.2, 0.0, 1.2 };

        TMeanVarSkewAccumulator acc;

        size_t count = sizeof(samples)/sizeof(samples[0]);
        acc = std::for_each(samples, samples + count, acc);

        CPPUNIT_ASSERT_EQUAL(count,
                             static_cast<size_t>(ml::maths::CBasicStatistics::count(acc)));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                     ml::maths::CBasicStatistics::mean(acc),
                                     0.000005);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(44.90633,
                                     ml::maths::CBasicStatistics::variance(acc),
                                     0.000005);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.82216,
                                     ml::maths::CBasicStatistics::skewness(acc),
                                     0.000005);

        double n0 = ml::maths::CBasicStatistics::count(acc);
        ml::maths::CBasicStatistics::scale(0.5, acc);
        double n1 = ml::maths::CBasicStatistics::count(acc);
        CPPUNIT_ASSERT_EQUAL(n1, 0.5 * n0);
    }

    LOG_DEBUG("Test weighted update");
    {
        double      samples[] = { 0.9, 1.0, 2.3, 1.5 };
        std::size_t weights[] = { 1, 4, 2, 3 };

        {
            TMeanAccumulator acc1;
            TMeanAccumulator acc2;

            for (size_t i = 0; i < boost::size(samples); ++i) {
                acc1.add(samples[i], static_cast<double>(weights[i]));
                for (std::size_t j = 0u; j < weights[i]; ++j) {
                    acc2.add(samples[i]);
                }
            }

            CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc1),
                                         ml::maths::CBasicStatistics::mean(acc2),
                                         1e-10);
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

            CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc1),
                                         ml::maths::CBasicStatistics::mean(acc2),
                                         1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::variance(acc1),
                                         ml::maths::CBasicStatistics::variance(acc2),
                                         1e-10);
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

            CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc1),
                                         ml::maths::CBasicStatistics::mean(acc2),
                                         1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::variance(acc1),
                                         ml::maths::CBasicStatistics::variance(acc2),
                                         1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::skewness(acc1),
                                         ml::maths::CBasicStatistics::skewness(acc2),
                                         1e-10);
        }
    }

    LOG_DEBUG("Test addition");
    {
        // Test addition.
        double samples1[] = { 0.9, 10.0, 5.6, 1.23 };
        double samples2[] = { -12.3, 7.2, 0.0, 1.2 };

        size_t count1 = sizeof(samples1)/sizeof(samples1[0]);
        size_t count2 = sizeof(samples2)/sizeof(samples2[0]);

        {
            TMeanAccumulator acc1;
            TMeanAccumulator acc2;

            acc1 = std::for_each(samples1, samples1 + count1, acc1);
            acc2 = std::for_each(samples2, samples2 + count2, acc2);

            CPPUNIT_ASSERT_EQUAL(count1 + count2,
                                 static_cast<size_t>(ml::maths::CBasicStatistics::count(acc1 + acc2)));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                         ml::maths::CBasicStatistics::mean(acc1 + acc2),
                                         0.000005);
        }

        {
            TMeanVarAccumulator acc1;
            TMeanVarAccumulator acc2;

            acc1 = std::for_each(samples1, samples1 + count1, acc1);
            acc2 = std::for_each(samples2, samples2 + count2, acc2);

            CPPUNIT_ASSERT_EQUAL(count1 + count2,
                                 static_cast<size_t>(ml::maths::CBasicStatistics::count(acc1 + acc2)));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                         ml::maths::CBasicStatistics::mean(acc1 + acc2),
                                         0.000005);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(44.90633,
                                         ml::maths::CBasicStatistics::variance(acc1 + acc2),
                                         0.000005);
        }

        {
            TMeanVarSkewAccumulator acc1;
            TMeanVarSkewAccumulator acc2;

            acc1 = std::for_each(samples1, samples1 + count1, acc1);
            acc2 = std::for_each(samples2, samples2 + count2, acc2);

            CPPUNIT_ASSERT_EQUAL(count1 + count2,
                                 static_cast<size_t>(ml::maths::CBasicStatistics::count(acc1 + acc2)));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.72875,
                                         ml::maths::CBasicStatistics::mean(acc1 + acc2),
                                         0.000005);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(44.90633,
                                         ml::maths::CBasicStatistics::variance(acc1 + acc2),
                                         0.000005);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.82216,
                                         ml::maths::CBasicStatistics::skewness(acc1 + acc2), \
                                         0.000005);
        }
    }

    LOG_DEBUG("Test subtraction");
    {
        ml::test::CRandomNumbers rng;

        LOG_DEBUG("Test mean");
        {
            TMeanAccumulator acc1;
            TMeanAccumulator acc2;

            TDoubleVec samples;
            rng.generateNormalSamples(2.0, 3.0, 40u, samples);

            for (std::size_t j = 1u; j < samples.size(); ++j) {
                LOG_DEBUG("split = " << j << "/" << samples.size() - j);

                for (std::size_t i = 0u; i < j; ++i) {
                    acc1.add(samples[i]);
                }
                for (std::size_t i = j; i < samples.size(); ++i) {
                    acc2.add(samples[i]);
                }

                TMeanAccumulator sum = acc1 + acc2;

                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(acc1),
                                     ml::maths::CBasicStatistics::count(sum - acc2));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc1),
                                             ml::maths::CBasicStatistics::mean(sum - acc2),
                                             1e-10);
                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(acc2),
                                     ml::maths::CBasicStatistics::count(sum - acc1));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc2),
                                             ml::maths::CBasicStatistics::mean(sum - acc1),
                                             1e-10);
            }
        }
        LOG_DEBUG("Test mean and variance");
        {
            TMeanVarAccumulator acc1;
            TMeanVarAccumulator acc2;

            TDoubleVec samples;
            rng.generateGammaSamples(3.0, 3.0, 40u, samples);

            for (std::size_t j = 1u; j < samples.size(); ++j) {
                LOG_DEBUG("split = " << j << "/" << samples.size() - j);

                for (std::size_t i = 0u; i < j; ++i) {
                    acc1.add(samples[i]);
                }
                for (std::size_t i = j; i < samples.size(); ++i) {
                    acc2.add(samples[i]);
                }

                TMeanVarAccumulator sum = acc1 + acc2;

                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(acc1),
                                     ml::maths::CBasicStatistics::count(sum - acc2));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc1),
                                             ml::maths::CBasicStatistics::mean(sum - acc2),
                                             1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::variance(acc1),
                                             ml::maths::CBasicStatistics::variance(sum - acc2),
                                             1e-10);
                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(acc2),
                                     ml::maths::CBasicStatistics::count(sum - acc1));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc2),
                                             ml::maths::CBasicStatistics::mean(sum - acc1),
                                             1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::variance(acc2),
                                             ml::maths::CBasicStatistics::variance(sum - acc1),
                                             1e-10);
            }
        }
        LOG_DEBUG("Test mean, variance and skew");
        {
            TMeanVarSkewAccumulator acc1;
            TMeanVarSkewAccumulator acc2;

            TDoubleVec samples;
            rng.generateLogNormalSamples(1.1, 1.0, 40u, samples);

            for (std::size_t j = 1u; j < samples.size(); ++j) {
                LOG_DEBUG("split = " << j << "/" << samples.size() - j);

                for (std::size_t i = 0u; i < j; ++i) {
                    acc1.add(samples[i]);
                }
                for (std::size_t i = j; i < samples.size(); ++i) {
                    acc2.add(samples[i]);
                }

                TMeanVarSkewAccumulator sum = acc1 + acc2;

                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(acc1),
                                     ml::maths::CBasicStatistics::count(sum - acc2));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc1),
                                             ml::maths::CBasicStatistics::mean(sum - acc2),
                                             1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::variance(acc1),
                                             ml::maths::CBasicStatistics::variance(sum - acc2),
                                             1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::skewness(acc1),
                                             ml::maths::CBasicStatistics::skewness(sum - acc2),
                                             1e-10);
                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(acc2),
                                     ml::maths::CBasicStatistics::count(sum - acc1));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::mean(acc2),
                                             ml::maths::CBasicStatistics::mean(sum - acc1),
                                             1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::variance(acc2),
                                             ml::maths::CBasicStatistics::variance(sum - acc1),
                                             1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(ml::maths::CBasicStatistics::skewness(acc2),
                                             ml::maths::CBasicStatistics::skewness(sum - acc1),
                                             1e-10);
            }
        }
    }

    LOG_DEBUG("test vector") {
        typedef ml::maths::CBasicStatistics::SSampleMean<ml::maths::CVectorNx1<double, 4> >::TAccumulator        TVectorMeanAccumulator;
        typedef ml::maths::CBasicStatistics::SSampleMeanVar<ml::maths::CVectorNx1<double, 4> >::TAccumulator     TVectorMeanVarAccumulator;
        typedef ml::maths::CBasicStatistics::SSampleMeanVarSkew<ml::maths::CVectorNx1<double, 4> >::TAccumulator TVectorMeanVarSkewAccumulator;

        ml::test::CRandomNumbers rng;

        {
            LOG_DEBUG("Test mean");

            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 1.0, 120, samples);

            TMeanAccumulator       means[4];
            TVectorMeanAccumulator vectorMean;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                ml::maths::CVectorNx1<double, 4> v;
                for (std::size_t j = 0u; j < 4; ++i, ++j) {
                    means[j].add(samples[i]);
                    v(j) = samples[i];
                }
                LOG_DEBUG("v = " << v);
                vectorMean.add(v);

                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(means[0]),
                                     ml::maths::CBasicStatistics::count(vectorMean));
                for (std::size_t j = 0u; j < 4; ++j) {
                    CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::mean(means[j]),
                                         (ml::maths::CBasicStatistics::mean(vectorMean))(j));
                }
            }
        }
        {
            LOG_DEBUG("Test mean and variance");

            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 1.0, 120, samples);

            TMeanVarAccumulator       meansAndVariances[4];
            TVectorMeanVarAccumulator vectorMeanAndVariances;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                ml::maths::CVectorNx1<double, 4> v;
                for (std::size_t j = 0u; j < 4; ++i, ++j) {
                    meansAndVariances[j].add(samples[i]);
                    v(j) = samples[i];
                }
                LOG_DEBUG("v = " << v);
                vectorMeanAndVariances.add(v);

                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(meansAndVariances[0]),
                                     ml::maths::CBasicStatistics::count(vectorMeanAndVariances));
                for (std::size_t j = 0u; j < 4; ++j) {
                    CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::mean(meansAndVariances[j]),
                                         (ml::maths::CBasicStatistics::mean(vectorMeanAndVariances))(j));
                    CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::variance(meansAndVariances[j]),
                                         (ml::maths::CBasicStatistics::variance(vectorMeanAndVariances))(j));
                }
            }
        }
        {
            LOG_DEBUG("Test mean, variance and skew");

            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 1.0, 120, samples);

            TMeanVarSkewAccumulator       meansVariancesAndSkews[4];
            TVectorMeanVarSkewAccumulator vectorMeanVarianceAndSkew;

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                ml::maths::CVectorNx1<double, 4> v;
                for (std::size_t j = 0u; j < 4; ++i, ++j) {
                    meansVariancesAndSkews[j].add(samples[i]);
                    v(j) = samples[i];
                }
                LOG_DEBUG("v = " << v);
                vectorMeanVarianceAndSkew.add(v);

                CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::count(meansVariancesAndSkews[0]),
                                     ml::maths::CBasicStatistics::count(vectorMeanVarianceAndSkew));
                for (std::size_t j = 0u; j < 4; ++j) {
                    CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::mean(meansVariancesAndSkews[j]),
                                         (ml::maths::CBasicStatistics::mean(vectorMeanVarianceAndSkew))(j));
                    CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::variance(meansVariancesAndSkews[j]),
                                         (ml::maths::CBasicStatistics::variance(vectorMeanVarianceAndSkew))(j));
                    CPPUNIT_ASSERT_EQUAL(ml::maths::CBasicStatistics::skewness(meansVariancesAndSkews[j]),
                                         (ml::maths::CBasicStatistics::skewness(vectorMeanVarianceAndSkew))(j));
                }
            }
        }
    }

    LOG_DEBUG("Test persistence of collections");
    {
        LOG_DEBUG("Test means");
        {
            TMeanAccumulatorVec moments(1);
            moments[0].add(2.0);
            moments[0].add(3.0);

            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                ml::core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG("Moments XML representation:\n" << xml);

                ml::core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
                ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanAccumulatorVec                      restored;
                CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(SRestore(), boost::ref(restored), _1)));
                LOG_DEBUG("restored = " << ml::core::CContainerPrinter::print(restored));
                CPPUNIT_ASSERT_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(moments[i].checksum(), restored[i].checksum());
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
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                ml::core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG("Moments XML representation:\n" << xml);

                ml::core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
                ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanAccumulatorVec                      restored;
                CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(SRestore(), boost::ref(restored), _1)));
                LOG_DEBUG("restored = " << ml::core::CContainerPrinter::print(restored));
                CPPUNIT_ASSERT_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }
        }
        LOG_DEBUG("Test means and variances");
        {
            TMeanVarAccumulatorVec moments(1);
            moments[0].add(2.0);
            moments[0].add(3.0);
            moments[0].add(3.5);
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                ml::core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG("Moments XML representation:\n" << xml);

                ml::core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
                ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarAccumulatorVec                   restored;
                CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(SRestore(), boost::ref(restored), _1)));
                LOG_DEBUG("restored = " << ml::core::CContainerPrinter::print(restored));
                CPPUNIT_ASSERT_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(moments[i].checksum(), restored[i].checksum());
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
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                ml::core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG("Moments XML representation:\n" << xml);

                ml::core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
                ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarAccumulatorVec                   restored;
                CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(SRestore(), boost::ref(restored), _1)));
                LOG_DEBUG("restored = " << ml::core::CContainerPrinter::print(restored));
                CPPUNIT_ASSERT_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }
        }
        LOG_DEBUG("Test means, variances and skews");
        {
            TMeanVarSkewAccumulatorVec moments(1);
            moments[0].add(2.0);
            moments[0].add(3.0);
            moments[0].add(3.5);
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                ml::core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG("Moments XML representation:\n" << xml);

                ml::core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
                ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarSkewAccumulatorVec               restored;
                CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(SRestore(), boost::ref(restored), _1)));
                LOG_DEBUG("restored = " << ml::core::CContainerPrinter::print(restored));
                CPPUNIT_ASSERT_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(moments[i].checksum(), restored[i].checksum());
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
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                ml::core::CPersistUtils::persist(TAG, moments, inserter);
                std::string xml;
                inserter.toXml(xml);
                LOG_DEBUG("Moments XML representation:\n" << xml);

                ml::core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
                ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
                TMeanVarSkewAccumulatorVec               restored;
                CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(SRestore(), boost::ref(restored), _1)));
                LOG_DEBUG("restored = " << ml::core::CContainerPrinter::print(restored));
                CPPUNIT_ASSERT_EQUAL(moments.size(), restored.size());
                for (std::size_t i = 0u; i < restored.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(moments[i].checksum(), restored[i].checksum());
                }
            }
        }
    }

    CPPUNIT_ASSERT_EQUAL(true, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMeanAccumulator>::value());
    CPPUNIT_ASSERT_EQUAL(true, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMeanVarAccumulator>::value());
    CPPUNIT_ASSERT_EQUAL(true, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMeanVarSkewAccumulator>::value());
}

void CBasicStatisticsTest::testVectorCentralMoments(void) {
    LOG_DEBUG("+--------------------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testVectorCentralMoments  |");
    LOG_DEBUG("+--------------------------------------------------+");

    typedef ml::core::CSmallVector<double, 2> TDouble2Vec;
    typedef std::vector<double>               TDoubleVec;

    {
        TMeanAccumulator2Vec moments1(2);
        TMeanAccumulatorVec  moments2(2);
        moments1[0].add(2.0); moments1[0].add(5.0); moments1[0].add(2.9); moments1[1].add(4.0); moments1[1].add(3.0);
        moments2[0].add(2.0); moments2[0].add(5.0); moments2[0].add(2.9); moments2[1].add(4.0); moments2[1].add(3.0);
        TDouble2Vec counts1 = ml::maths::CBasicStatistics::count(moments1);
        TDouble2Vec means1 = ml::maths::CBasicStatistics::mean(moments1);
        TDoubleVec  counts2 = ml::maths::CBasicStatistics::count(moments2);
        TDoubleVec  means2 = ml::maths::CBasicStatistics::mean(moments2);
        CPPUNIT_ASSERT_EQUAL(std::string("[3, 2]"), ml::core::CContainerPrinter::print(counts1));
        CPPUNIT_ASSERT_EQUAL(std::string("[3.3, 3.5]"), ml::core::CContainerPrinter::print(means1));
        CPPUNIT_ASSERT_EQUAL(std::string("[3, 2]"), ml::core::CContainerPrinter::print(counts2));
        CPPUNIT_ASSERT_EQUAL(std::string("[3.3, 3.5]"), ml::core::CContainerPrinter::print(means2));
    }
    {
        TMeanVarAccumulator2Vec moments1(2);
        TMeanVarAccumulatorVec  moments2(2);
        moments1[0].add(2.0); moments1[0].add(4.0); moments1[1].add(3.0); moments1[1].add(4.0); moments1[1].add(5.0);
        moments2[0].add(2.0); moments2[0].add(4.0); moments2[1].add(3.0); moments2[1].add(4.0); moments2[1].add(5.0);
        TDouble2Vec counts1 = ml::maths::CBasicStatistics::count(moments1);
        TDouble2Vec means1 = ml::maths::CBasicStatistics::mean(moments1);
        TDouble2Vec vars1 = ml::maths::CBasicStatistics::variance(moments1);
        TDouble2Vec mlvars1 = ml::maths::CBasicStatistics::maximumLikelihoodVariance(moments1);
        TDoubleVec  counts2 = ml::maths::CBasicStatistics::count(moments2);
        TDoubleVec  means2 = ml::maths::CBasicStatistics::mean(moments2);
        TDoubleVec  vars2 = ml::maths::CBasicStatistics::variance(moments2);
        TDouble2Vec mlvars2 = ml::maths::CBasicStatistics::maximumLikelihoodVariance(moments2);
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 3]"), ml::core::CContainerPrinter::print(counts1));
        CPPUNIT_ASSERT_EQUAL(std::string("[3, 4]"), ml::core::CContainerPrinter::print(means1));
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 1]"), ml::core::CContainerPrinter::print(vars1));
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.6666667]"), ml::core::CContainerPrinter::print(mlvars1));
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 3]"), ml::core::CContainerPrinter::print(counts2));
        CPPUNIT_ASSERT_EQUAL(std::string("[3, 4]"), ml::core::CContainerPrinter::print(means2));
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 1]"), ml::core::CContainerPrinter::print(vars2));
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.6666667]"), ml::core::CContainerPrinter::print(mlvars2));
    }
    {
        TMeanVarSkewAccumulator2Vec moments1(2);
        TMeanVarSkewAccumulatorVec  moments2(2);
        moments1[0].add(2.0); moments1[0].add(4.0); moments1[1].add(2.0); moments1[1].add(5.0); moments1[1].add(5.0);
        moments2[0].add(2.0); moments2[0].add(4.0); moments2[1].add(2.0); moments2[1].add(5.0); moments2[1].add(5.0);
        TDouble2Vec counts1 = ml::maths::CBasicStatistics::count(moments1);
        TDouble2Vec means1 = ml::maths::CBasicStatistics::mean(moments1);
        TDouble2Vec vars1 = ml::maths::CBasicStatistics::variance(moments1);
        TDouble2Vec mlvars1 = ml::maths::CBasicStatistics::maximumLikelihoodVariance(moments1);
        TDouble2Vec skews1 = ml::maths::CBasicStatistics::skewness(moments1);
        TDoubleVec  counts2 = ml::maths::CBasicStatistics::count(moments2);
        TDoubleVec  means2 = ml::maths::CBasicStatistics::mean(moments2);
        TDoubleVec  vars2 = ml::maths::CBasicStatistics::variance(moments2);
        TDouble2Vec mlvars2 = ml::maths::CBasicStatistics::maximumLikelihoodVariance(moments2);
        TDouble2Vec skews2 = ml::maths::CBasicStatistics::skewness(moments2);
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 3]"), ml::core::CContainerPrinter::print(counts1));
        CPPUNIT_ASSERT_EQUAL(std::string("[3, 4]"), ml::core::CContainerPrinter::print(means1));
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 3]"), ml::core::CContainerPrinter::print(vars1));
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 2]"), ml::core::CContainerPrinter::print(mlvars1));
        CPPUNIT_ASSERT_EQUAL(std::string("[0, -0.3849002]"), ml::core::CContainerPrinter::print(skews1));
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 3]"), ml::core::CContainerPrinter::print(counts2));
        CPPUNIT_ASSERT_EQUAL(std::string("[3, 4]"), ml::core::CContainerPrinter::print(means2));
        CPPUNIT_ASSERT_EQUAL(std::string("[2, 3]"), ml::core::CContainerPrinter::print(vars2));
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 2]"), ml::core::CContainerPrinter::print(mlvars2));
        CPPUNIT_ASSERT_EQUAL(std::string("[0, -0.3849002]"), ml::core::CContainerPrinter::print(skews2));
    }
}

void CBasicStatisticsTest::testCovariances(void) {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testCovariances  |");
    LOG_DEBUG("+-----------------------------------------+");

    LOG_DEBUG("N(3,I)");
    {
        const double raw[][3] =
        {
            { 2.58894, 2.87211, 1.62609 },
            { 3.88246, 2.98577, 2.70981 },
            { 2.03317, 3.33715, 2.93560 },
            { 3.30100, 4.38844, 1.65705 },
            { 2.12426, 2.21127, 2.57000 },
            { 4.21041, 4.20745, 1.90752 },
            { 3.56139, 3.14454, 0.89316 },
            { 4.29444, 1.58715, 3.58402 },
            { 3.06731, 3.91581, 2.85951 },
            { 3.62798, 2.28786, 2.89994 },
            { 2.05834, 2.96137, 3.57654 },
            { 2.72185, 3.36003, 3.09708 },
            { 0.94924, 2.19797, 3.30941 },
            { 2.11159, 2.49182, 3.56793 },
            { 3.10364, 0.32747, 3.62487 },
            { 2.28235, 3.83542, 3.35942 },
            { 3.30549, 2.95951, 2.97006 },
            { 3.05787, 2.94188, 2.64095 },
            { 3.98245, 2.02892, 3.07909 },
            { 3.81189, 2.89389, 3.81389 },
            { 3.32811, 3.88484, 4.17866 },
            { 2.06964, 3.80683, 2.46835 },
            { 4.58989, 2.00321, 1.93029 },
            { 2.51484, 4.46106, 3.71248 },
            { 3.30729, 2.44768, 3.43241 },
            { 3.52222, 2.91724, 1.49631 },
            { 1.71826, 4.79752, 4.38398 },
            { 3.14173, 3.16237, 2.49654 },
            { 3.26538, 2.21858, 5.05477 },
            { 2.88352, 1.94396, 3.08744 }
        };

        const double expectedMean[] = { 3.013898, 2.952637, 2.964104 };
        const double expectedCovariances[][3] =
        {
            {  0.711903, -0.174535, -0.199460 },
            { -0.174535,  0.935285, -0.091192 },
            { -0.199460, -0.091192,  0.833710 }
        };

        ml::maths::CBasicStatistics::SSampleCovariances<double, 3> covariances;

        for (std::size_t i = 0u; i < boost::size(raw); ++i) {
            ml::maths::CVectorNx1<double, 3> v(raw[i]);
            LOG_DEBUG("v = " << v);
            covariances.add(v);
        }

        LOG_DEBUG("count = " << ml::maths::CBasicStatistics::count(covariances));
        LOG_DEBUG("mean = " << ml::maths::CBasicStatistics::mean(covariances));
        LOG_DEBUG("covariances = " << ml::maths::CBasicStatistics::covariances(covariances));

        CPPUNIT_ASSERT_EQUAL(static_cast<double>(boost::size(raw)),
                             ml::maths::CBasicStatistics::count(covariances));
        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean[i],
                                         (ml::maths::CBasicStatistics::mean(covariances))(i),
                                         2e-6);
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedCovariances[i][j],
                                             (ml::maths::CBasicStatistics::covariances(covariances))(i, j),
                                             2e-6);
            }
        }

        bool dynamicSizeAlwaysZero = ml::core::memory_detail::SDynamicSizeAlwaysZero<
            ml::maths::CBasicStatistics::SSampleCovariances<double, 3> >::value();
        CPPUNIT_ASSERT_EQUAL(true, dynamicSizeAlwaysZero);
    }

    {
        typedef std::vector<ml::maths::CVectorNx1<double, 4> > TVectorVec;

        double                           mean_[] = { 1.0, 3.0, 2.0, 7.0 };
        ml::maths::CVectorNx1<double, 4> mean(mean_);

        double covariances1_[] = {  1.0,  1.0,  1.0, 1.0 };
        double covariances2_[] = { -1.0,  1.0,  0.0, 0.0 };
        double covariances3_[] = { -1.0, -1.0,  2.0, 0.0 };
        double covariances4_[] = { -1.0, -1.0, -1.0, 3.0 };

        ml::maths::CVectorNx1<double, 4> covariances1(covariances1_);
        ml::maths::CVectorNx1<double, 4> covariances2(covariances2_);
        ml::maths::CVectorNx1<double, 4> covariances3(covariances3_);
        ml::maths::CVectorNx1<double, 4> covariances4(covariances4_);

        ml::maths::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * ml::maths::CSymmetricMatrixNxN<double, 4>(ml::maths::E_OuterProduct,
                                                             covariances1 / covariances1.euclidean())
            +  5.0 * ml::maths::CSymmetricMatrixNxN<double, 4>(ml::maths::E_OuterProduct,
                                                               covariances2 / covariances2.euclidean())
            +  5.0 * ml::maths::CSymmetricMatrixNxN<double, 4>(ml::maths::E_OuterProduct,
                                                               covariances3 / covariances3.euclidean())
            +  2.0 * ml::maths::CSymmetricMatrixNxN<double, 4>(ml::maths::E_OuterProduct,
                                                               covariances4 / covariances4.euclidean()));

        std::size_t n = 10000u;

        TVectorVec samples;
        ml::maths::CSampling::multivariateNormalSample(mean, covariance, n, samples);

        ml::maths::CBasicStatistics::SSampleCovariances<double, 4> sampleCovariance;

        for (std::size_t i = 0u; i < n; ++i) {
            sampleCovariance.add(samples[i]);
        }

        LOG_DEBUG("expected mean = " << mean);
        LOG_DEBUG("expected covariances = " << covariance);

        LOG_DEBUG("mean = " << ml::maths::CBasicStatistics::mean(sampleCovariance));
        LOG_DEBUG("covariances = " << ml::maths::CBasicStatistics::covariances(sampleCovariance));

        for (std::size_t i = 0u; i < 4; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(mean(i),
                                         (ml::maths::CBasicStatistics::mean(sampleCovariance))(i),
                                         0.05);
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(covariance(i, j),
                                             (ml::maths::CBasicStatistics::covariances(sampleCovariance))(i, j),
                                             0.16);

            }
        }
    }

    {
        ml::test::CRandomNumbers rng;

        std::vector<double> coordinates;
        rng.generateUniformSamples(5.0, 10.0, 400, coordinates);

        std::vector<ml::maths::CVectorNx1<double, 4> > points;
        for (std::size_t i = 0u; i < coordinates.size(); i += 4) {
            double c[] =
            {
                coordinates[i+0],
                coordinates[i+1],
                coordinates[i+2],
                coordinates[i+3]
            };
            points.push_back(ml::maths::CVectorNx1<double, 4>(c));
        }

        ml::maths::CBasicStatistics::SSampleCovariances<double, 4> expectedSampleCovariances;
        for (std::size_t i = 0u; i < points.size(); ++i) {
            expectedSampleCovariances.add(points[i]);
        }

        std::string expectedDelimited = expectedSampleCovariances.toDelimited();
        LOG_DEBUG("delimited = " << expectedDelimited);

        ml::maths::CBasicStatistics::SSampleCovariances<double, 4> sampleCovariances;
        CPPUNIT_ASSERT(sampleCovariances.fromDelimited(expectedDelimited));

        CPPUNIT_ASSERT_EQUAL(expectedSampleCovariances.checksum(), sampleCovariances.checksum());

        std::string delimited = sampleCovariances.toDelimited();
        CPPUNIT_ASSERT_EQUAL(expectedDelimited, delimited);
    }
}

void CBasicStatisticsTest::testCovariancesLedoitWolf(void) {
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testCovariancesLedoitWolf  |");
    LOG_DEBUG("+---------------------------------------------------+");

    typedef std::vector<double>                       TDoubleVec;
    typedef std::vector<TDoubleVec>                   TDoubleVecVec;
    typedef ml::maths::CVectorNx1<double, 2>          TVector2;
    typedef std::vector<TVector2>                     TVector2Vec;
    typedef ml::maths::CSymmetricMatrixNxN<double, 2> TMatrix2;

    ml::test::CRandomNumbers rng;

    double means[][2] =
    {
        {  10.0,  10.0  },
        {  20.0,  150.0 },
        { -10.0, -20.0  },
        { -20.0,  40.0  },
        {  40.0,  90.0  }
    };

    double covariances[][2][2] =
    {
        { {  40.0,   0.0 }, {   0.0, 40.0 } },
        { {  20.0,   5.0 }, {   5.0, 10.0 } },
        { { 300.0, -70.0 }, { -70.0, 60.0 } },
        { { 100.0,  20.0 }, {  20.0, 60.0 } },
        { {  50.0, -10.0 }, { -10.0, 60.0 } }
    };

    ml::maths::CBasicStatistics::SSampleMean<double>::TAccumulator error;
    ml::maths::CBasicStatistics::SSampleMean<double>::TAccumulator errorLW;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        LOG_DEBUG("*** test " << i << " ***");

        TDoubleVec    mean(boost::begin(means[i]), boost::end(means[i]));
        TDoubleVecVec covariance;
        for (std::size_t j = 0u; j < boost::size(covariances[i]); ++j) {
            covariance.push_back(TDoubleVec(boost::begin(covariances[i][j]),
                                            boost::end(covariances[i][j])));
        }
        TMatrix2 covExpected(covariance);
        LOG_DEBUG("cov expected = " << covExpected);

        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 50, samples);

        // Test the frobenius norm of the error in the covariance matrix.

        for (std::size_t j = 3u; j < samples.size(); ++j) {
            TVector2Vec jsamples;
            for (std::size_t k = 0u; k < j; ++k) {
                jsamples.push_back(TVector2(samples[k]));
            }

            ml::maths::CBasicStatistics::SSampleCovariances<double, 2> cov;
            cov.add(jsamples);

            ml::maths::CBasicStatistics::SSampleCovariances<double, 2> covLW;
            ml::maths::CBasicStatistics::covariancesLedoitWolf(jsamples, covLW);

            const TMatrix2 &covML   = ml::maths::CBasicStatistics::maximumLikelihoodCovariances(cov);
            const TMatrix2 &covLWML = ml::maths::CBasicStatistics::maximumLikelihoodCovariances(covLW);

            double errorML   = (covML - covExpected).frobenius();
            double errorLWML = (covLWML - covExpected).frobenius();

            if (j % 5 == 0) {
                LOG_DEBUG("cov ML   = " << covML);
                LOG_DEBUG("cov LWML = " << covLWML);
                LOG_DEBUG("error ML = " << errorML << ", error LWML = " << errorLWML);
            }
            CPPUNIT_ASSERT(errorLWML < 6.0 * errorML);
            error.add(errorML / covExpected.frobenius());
            errorLW.add(errorLWML / covExpected.frobenius());
        }
    }

    LOG_DEBUG("error    = " << error);
    LOG_DEBUG("error LW = " << errorLW);
    CPPUNIT_ASSERT(  ml::maths::CBasicStatistics::mean(errorLW)
                     < 0.9 * ml::maths::CBasicStatistics::mean(error));
}

void CBasicStatisticsTest::testMedian(void) {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testMedian  |");
    LOG_DEBUG("+------------------------------------+");

    {
        ml::maths::CBasicStatistics::TDoubleVec sampleVec;

        double median = ml::maths::CBasicStatistics::median(sampleVec);

        CPPUNIT_ASSERT_EQUAL(0.0, median);
    }
    {
        double sample[] = { 1.0 };

        ml::maths::CBasicStatistics::TDoubleVec sampleVec(sample, sample+sizeof(sample)/sizeof(sample[0]));

        double median = ml::maths::CBasicStatistics::median(sampleVec);

        CPPUNIT_ASSERT_EQUAL(1.0, median);
    }
    {
        double sample[] = { 2.0, 1.0 };

        ml::maths::CBasicStatistics::TDoubleVec sampleVec(sample, sample+sizeof(sample)/sizeof(sample[0]));

        double median = ml::maths::CBasicStatistics::median(sampleVec);

        CPPUNIT_ASSERT_EQUAL(1.5, median);
    }
    {
        double sample[] = { 3.0, 1.0, 2.0 };

        ml::maths::CBasicStatistics::TDoubleVec sampleVec(sample, sample+sizeof(sample)/sizeof(sample[0]));

        double median = ml::maths::CBasicStatistics::median(sampleVec);

        CPPUNIT_ASSERT_EQUAL(2.0, median);
    }
    {
        double sample[] = { 3.0, 5.0, 9.0, 1.0, 2.0, 6.0, 7.0, 4.0, 8.0 };

        ml::maths::CBasicStatistics::TDoubleVec sampleVec(sample, sample+sizeof(sample)/sizeof(sample[0]));

        double median = ml::maths::CBasicStatistics::median(sampleVec);

        CPPUNIT_ASSERT_EQUAL(5.0, median);
    }
    {
        double sample[] = { 3.0, 5.0, 10.0, 2.0, 6.0, 7.0, 1.0, 9.0, 4.0, 8.0 };

        ml::maths::CBasicStatistics::TDoubleVec sampleVec(sample, sample+sizeof(sample)/sizeof(sample[0]));

        double median = ml::maths::CBasicStatistics::median(sampleVec);

        CPPUNIT_ASSERT_EQUAL(5.5, median);
    }
}

void CBasicStatisticsTest::testOrderStatistics(void) {
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testOrderStatistics  |");
    LOG_DEBUG("+---------------------------------------------+");

    // Test that the order statistics accumulators work for finding min and max
    // elements of a collection.

    typedef ml::maths::CBasicStatistics::COrderStatisticsStack<double, 2u>                        TMinStatsStack;
    typedef ml::maths::CBasicStatistics::COrderStatisticsStack<double, 3u, std::greater<double> > TMaxStatsStack;
    typedef ml::maths::CBasicStatistics::COrderStatisticsHeap<double>                             TMinStatsHeap;
    typedef ml::maths::CBasicStatistics::COrderStatisticsHeap<double, std::greater<double> >      TMaxStatsHeap;

    {
        // Test on the stack min, max, combine and persist and restore.

        double data[] = { 1.0, 2.3, 1.1, 1.0, 5.0, 3.0, 11.0, 0.2, 15.8, 12.3 };

        TMinStatsStack minValues;
        TMaxStatsStack maxValues;
        TMinStatsStack minFirstHalf;
        TMinStatsStack minSecondHalf;

        for (size_t i = 0; i < boost::size(data); ++i) {
            minValues.add(data[i]);
            maxValues.add(data[i]);
            (2 * i < boost::size(data) ? minFirstHalf : minSecondHalf).add(data[i]);
        }

        std::sort(boost::begin(data), boost::end(data));
        minValues.sort();
        LOG_DEBUG("x_1 = " << minValues[0]
                           << ", x_2 = " << minValues[1]);
        CPPUNIT_ASSERT(std::equal(minValues.begin(), minValues.end(), data));

        std::sort(boost::begin(data), boost::end(data), std::greater<double>());
        maxValues.sort();
        LOG_DEBUG("x_n = " << maxValues[0]
                           << ", x_(n-1) = " << maxValues[1]
                           << ", x_(n-2) = " << maxValues[2]);
        CPPUNIT_ASSERT(std::equal(maxValues.begin(), maxValues.end(), data));

        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), minValues.count());
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), maxValues.count());

        TMinStatsStack minFirstPlusSecondHalf = (minFirstHalf + minSecondHalf);
        minFirstPlusSecondHalf.sort();
        CPPUNIT_ASSERT(std::equal(minValues.begin(), minValues.end(),
                                  minFirstPlusSecondHalf.begin()));

        // Test persist is idempotent.

        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, minValues.toDelimited());
            inserter.toXml(origXml);
        }

        LOG_DEBUG("Stats XML representation:\n" << origXml);

        // Restore the XML into stats object.
        TMinStatsStack restoredMinValues;
        {
            ml::core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                               boost::bind(SRestore(), boost::ref(restoredMinValues), _1)));
        }

        // The XML representation of the new stats object should be unchanged.
        std::string newXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, restoredMinValues.toDelimited());
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }

    {
        // Test on the heap min, max, combine and persist and restore.

        double data[] = { 1.0, 2.3, 1.1, 1.0, 5.0, 3.0, 11.0, 0.2, 15.8, 12.3 };

        TMinStatsHeap min2Values(2);
        TMaxStatsHeap max3Values(3);
        TMaxStatsHeap max20Values(20);

        for (size_t i = 0; i < boost::size(data); ++i) {
            min2Values.add(data[i]);
            max3Values.add(data[i]);
            max20Values.add(data[i]);
        }

        std::sort(boost::begin(data), boost::end(data));
        min2Values.sort();
        LOG_DEBUG("x_1 = " << min2Values[0]
                           << ", x_2 = " << min2Values[1]);
        CPPUNIT_ASSERT(std::equal(min2Values.begin(), min2Values.end(), data));

        std::sort(boost::begin(data), boost::end(data), std::greater<double>());
        max3Values.sort();
        LOG_DEBUG("x_n = " << max3Values[0]
                           << ", x_(n-1) = " << max3Values[1]
                           << ", x_(n-2) = " << max3Values[2]);
        CPPUNIT_ASSERT(std::equal(max3Values.begin(), max3Values.end(), data));

        max20Values.sort();
        CPPUNIT_ASSERT_EQUAL(boost::size(data), max20Values.count());
        CPPUNIT_ASSERT(std::equal(max20Values.begin(), max20Values.end(), data));

        // Test persist is idempotent.

        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, max20Values.toDelimited());
            inserter.toXml(origXml);
        }

        LOG_DEBUG("Stats XML representation:\n" << origXml);

        // Restore the XML into stats object.
        TMinStatsHeap restoredMaxValues(20);
        {
            ml::core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                               boost::bind(SRestore(), boost::ref(restoredMaxValues), _1)));
        }

        // The XML representation of the new stats object should be unchanged.
        std::string newXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            inserter.insertValue(TAG, restoredMaxValues.toDelimited());
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
    {
        // Test we correctly age the minimum value accumulator.
        TMinStatsStack test;
        test.add(15.0);
        test.age(0.5);
        CPPUNIT_ASSERT_EQUAL(30.0, test[0]);
    }
    {
        // Test we correctly age the maximum value accumulator.
        TMaxStatsStack test;
        test.add(15.0);
        test.age(0.5);
        CPPUNIT_ASSERT_EQUAL(7.5, test[0]);
    }
    {
        // Test biggest.
        TMinStatsHeap min(5);
        TMaxStatsHeap max(5);
        min.add(1.0);
        max.add(1.0);
        CPPUNIT_ASSERT_EQUAL(1.0, min.biggest());
        CPPUNIT_ASSERT_EQUAL(1.0, max.biggest());
        std::size_t i{0};
        for (auto value : { 3.6, -6.1, 1.0, 3.4 }) {
            min.add(value);
            max.add(value);
            if (i++ == 0) {
                CPPUNIT_ASSERT_EQUAL(3.6, min.biggest());
                CPPUNIT_ASSERT_EQUAL(1.0, max.biggest());
            } else {
                CPPUNIT_ASSERT_EQUAL( 3.6, min.biggest());
                CPPUNIT_ASSERT_EQUAL(-6.1, max.biggest());
            }
        }
        min.add(0.9);
        max.add(0.9);
        CPPUNIT_ASSERT_EQUAL(3.4, min.biggest());
        CPPUNIT_ASSERT_EQUAL(0.9, max.biggest());
    }
    {
        // Test memory.
        CPPUNIT_ASSERT_EQUAL(true, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMinStatsStack>::value());
        CPPUNIT_ASSERT_EQUAL(true, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMaxStatsStack>::value());
        CPPUNIT_ASSERT_EQUAL(false, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMinStatsHeap>::value());
        CPPUNIT_ASSERT_EQUAL(false, ml::core::memory_detail::SDynamicSizeAlwaysZero<TMaxStatsHeap>::value());
    }
}

void CBasicStatisticsTest::testMinMax(void) {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CBasicStatisticsTest::testMinMax  |");
    LOG_DEBUG("+------------------------------------+");

    typedef std::vector<double> TDoubleVec;

    TDoubleVec positive{1.0, 2.7, 4.0, 0.3, 11.7};
    TDoubleVec negative{-3.7, -0.8, -18.2, -0.8};
    TDoubleVec mixed{1.3, -8.0, 2.1};

    {
        ml::maths::CBasicStatistics::CMinMax<double> minmax;
        CPPUNIT_ASSERT(!minmax.initialized());
        minmax.add(positive);
        CPPUNIT_ASSERT(minmax.initialized());
        CPPUNIT_ASSERT_EQUAL(0.3,  minmax.min());
        CPPUNIT_ASSERT_EQUAL(11.7, minmax.max());
        CPPUNIT_ASSERT_EQUAL(0.3,  minmax.signMargin());
    }
    {
        ml::maths::CBasicStatistics::CMinMax<double> minmax;
        CPPUNIT_ASSERT(!minmax.initialized());
        minmax.add(negative);
        CPPUNIT_ASSERT(minmax.initialized());
        CPPUNIT_ASSERT_EQUAL(-18.2, minmax.min());
        CPPUNIT_ASSERT_EQUAL(-0.8,  minmax.max());
        CPPUNIT_ASSERT_EQUAL(-0.8,  minmax.signMargin());
    }
    {
        ml::maths::CBasicStatistics::CMinMax<double> minmax;
        CPPUNIT_ASSERT(!minmax.initialized());
        minmax.add(mixed);
        CPPUNIT_ASSERT(minmax.initialized());
        CPPUNIT_ASSERT_EQUAL(-8.0, minmax.min());
        CPPUNIT_ASSERT_EQUAL( 2.1, minmax.max());
        CPPUNIT_ASSERT_EQUAL( 0.0, minmax.signMargin());
    }
    {
        ml::maths::CBasicStatistics::CMinMax<double> minmax1;
        ml::maths::CBasicStatistics::CMinMax<double> minmax2;
        ml::maths::CBasicStatistics::CMinMax<double> minmax12;
        minmax1.add(positive);
        minmax2.add(negative);
        minmax12.add(positive);
        minmax12.add(negative);
        CPPUNIT_ASSERT_EQUAL((minmax1 + minmax2).checksum(), minmax12.checksum());
    }
}
