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

#include "CKMeansOnlineTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CKMeansOnline.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CRestoreParams.h>

#include <test/CRandomNumbers.h>

using namespace ml;

namespace {
typedef std::vector<double> TDoubleVec;
typedef std::vector<std::size_t> TSizeVec;
typedef std::vector<TSizeVec> TSizeVecVec;
typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TMeanVarAccumulator;
typedef maths::CVectorNx1<double, 2> TVector2;
typedef std::vector<TVector2> TVector2Vec;
typedef std::vector<TVector2Vec> TVector2VecVec;
typedef maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator TMean2Accumulator;
typedef maths::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator TMeanVar2Accumulator;
typedef maths::CVectorNx1<double, 5> TVector5;
typedef std::vector<TVector5> TVector5Vec;
typedef maths::CBasicStatistics::SSampleMeanVar<TVector5>::TAccumulator TMeanVar5Accumulator;

template<typename POINT>
class CKMeansOnlineTestForTest : public maths::CKMeansOnline<POINT> {
public:
    typedef typename maths::CKMeansOnline<POINT>::TSphericalClusterVec TSphericalClusterVec;
    typedef typename maths::CKMeansOnline<POINT>::TDoubleMeanVarAccumulator TDoubleMeanVarAccumulator;
    typedef typename maths::CKMeansOnline<POINT>::TFloatMeanAccumulatorDoublePr TFloatMeanAccumulatorDoublePr;

public:
    CKMeansOnlineTestForTest(std::size_t k, double decayRate = 0.0) : maths::CKMeansOnline<POINT>(k, decayRate) {}

    static void add(const POINT& x, double count, TFloatMeanAccumulatorDoublePr& cluster) {
        maths::CKMeansOnline<POINT>::add(x, count, cluster);
    }

    static double variance(const TDoubleMeanVarAccumulator& moments) { return maths::CKMeansOnline<POINT>::variance(moments); }
};

template<typename POINT>
std::string print(const POINT& point) {
    std::ostringstream result;
    result << point;
    return result.str();
}
}

void CKMeansOnlineTest::testVariance(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testVariance  |");
    LOG_DEBUG("+-----------------------------------+");

    // Check that the variance calculation gives the correct
    // spherical variance.

    test::CRandomNumbers rng;

    for (std::size_t t = 1u; t <= 50; ++t) {
        LOG_DEBUG("*** test = " << t << " ***");

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 50, coordinates);
        TVector5Vec points;
        for (std::size_t i = 0u; i < coordinates.size(); i += 5) {
            double c[] = {coordinates[i + 0], coordinates[i + 1], coordinates[i + 2], coordinates[i + 3], coordinates[i + 4]};
            points.push_back(TVector5(c));
        }

        TMeanVar5Accumulator actual;
        actual.add(points);

        TMeanVarAccumulator expected;
        for (std::size_t i = 0u; i < coordinates.size(); ++i) {
            expected.add(coordinates[i] - maths::CBasicStatistics::mean(actual)(i % 5));
        }

        LOG_DEBUG("actual   = " << CKMeansOnlineTestForTest<TVector5>::variance(actual));
        LOG_DEBUG("expected = " << maths::CBasicStatistics::maximumLikelihoodVariance(expected));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::maximumLikelihoodVariance(expected),
                                     CKMeansOnlineTestForTest<TVector5>::variance(actual),
                                     1e-10 * maths::CBasicStatistics::maximumLikelihoodVariance(expected));
    }
}

void CKMeansOnlineTest::testAdd(void) {
    LOG_DEBUG("+------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testAdd  |");
    LOG_DEBUG("+------------------------------+");

    // Test that we correctly compute the mean and spherical
    // variance.

    typedef std::pair<TMean2Accumulator, double> TMean2AccumulatorDoublePr;

    test::CRandomNumbers rng;

    for (std::size_t t = 1u; t <= 50; ++t) {
        LOG_DEBUG("*** test = " << t << " ***");

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 40, coordinates);
        TDoubleVec counts;
        rng.generateUniformSamples(1.0, 2.0, 20, counts);
        TVector2Vec points;
        for (std::size_t i = 0u; i < coordinates.size(); i += 2) {
            double c[] = {coordinates[i + 0], coordinates[i + 1]};
            points.push_back(TVector2(c));
        }

        TMean2AccumulatorDoublePr actual;
        TMeanVar2Accumulator expected;
        for (std::size_t i = 0u; i < points.size(); ++i) {
            CKMeansOnlineTestForTest<TVector2>::add(points[i], counts[i], actual);
            expected.add(points[i], counts[i]);
        }

        TVector2 ones(1.0);

        LOG_DEBUG("actual   = " << maths::CBasicStatistics::mean(actual.first) << "," << actual.second);
        LOG_DEBUG("expected = "
                  << maths::CBasicStatistics::mean(expected) << ","
                  << maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones) / static_cast<double>(ones.dimension()));

        CPPUNIT_ASSERT_EQUAL(print(maths::CBasicStatistics::mean(expected)), print(maths::CBasicStatistics::mean(actual.first)));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones) / static_cast<double>(ones.dimension()),
            actual.second,
            1e-10 * maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones) / static_cast<double>(ones.dimension()));
    }
}

void CKMeansOnlineTest::testReduce(void) {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testReduce  |");
    LOG_DEBUG("+---------------------------------+");

    // Test some invariants:
    //   - Number of clusters should be no more than k after
    //     reduce.
    //   - The count of the points should be unchanged.
    //   - The centroid of the points should be unchanged.
    //   - The total spherical variance of the points should
    //     be unchanged.

    test::CRandomNumbers rng;

    for (std::size_t t = 1u; t <= 10; ++t) {
        LOG_DEBUG("*** test = " << t << " ***");

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 42, coordinates);
        TDoubleVec counts;
        rng.generateUniformSamples(1.0, 2.0, 21, counts);
        TVector2Vec points;
        for (std::size_t i = 0u; i < coordinates.size(); i += 2) {
            double c[] = {coordinates[i + 0], coordinates[i + 1]};
            points.push_back(TVector2(c));
        }

        maths::CKMeansOnline<TVector2> kmeans(10);

        TMeanVar2Accumulator expected;

        TVector2 ones(1.0);

        for (std::size_t i = 0u; i < points.size(); ++i) {
            kmeans.add(points[i], counts[i]);
            expected.add(points[i], counts[i]);

            if ((i + 1) % 7 == 0) {
                CKMeansOnlineTestForTest<TVector2>::TSphericalClusterVec clusters;
                kmeans.clusters(clusters);
                CPPUNIT_ASSERT(clusters.size() <= 10);

                TMeanVar2Accumulator actual;
                for (std::size_t j = 0u; j < clusters.size(); ++j) {
                    actual.add(clusters[j]);
                }

                LOG_DEBUG("expected = " << expected);
                LOG_DEBUG("actual   = " << actual);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::count(expected), maths::CBasicStatistics::count(actual), 1e-10);
                CPPUNIT_ASSERT_EQUAL(print(maths::CBasicStatistics::mean(expected)), print(maths::CBasicStatistics::mean(actual)));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones),
                                             maths::CBasicStatistics::maximumLikelihoodVariance(actual).inner(ones),
                                             1e-10 * maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones));
            }
        }
    }
}

void CKMeansOnlineTest::testClustering(void) {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testClustering  |");
    LOG_DEBUG("+-------------------------------------+");

    // Test we are reliably able to find as approximately as good
    // clusterings as k-means working on the full data set.

    test::CRandomNumbers rng;

    {
        TMeanVarAccumulator cost;
        TMeanVarAccumulator costOnline;

        double a[] = {0.0, 20.0};
        double b[] = {5.0, 30.0};
        TVector2Vec points;
        for (std::size_t i = 0u; i < 2; ++i) {
            TDoubleVec coordinates;
            rng.generateUniformSamples(a[i], b[i], 200, coordinates);
            for (std::size_t j = 0u; j < coordinates.size(); j += 2) {
                double c[] = {coordinates[j + 0], coordinates[j + 1]};
                points.push_back(TVector2(c));
            }
        }

        for (std::size_t t = 1u; t <= 10; ++t) {
            LOG_DEBUG("*** test = " << t << " ***");

            maths::CKMeansFast<TVector2> kmeans;
            double cost_ = std::numeric_limits<double>::max();
            kmeans.setPoints(points);
            TVector2Vec centres;
            TVector2VecVec clusters;
            maths::CPRNG::CXorOShiro128Plus rng_;
            for (std::size_t i = 0u; i < 10; ++i) {
                maths::CKMeansPlusPlusInitialization<TVector2, maths::CPRNG::CXorOShiro128Plus> seedCentres(rng_);
                seedCentres.run(points, 2, centres);
                kmeans.setCentres(centres);
                kmeans.run(10);
                maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC> criterion;
                kmeans.clusters(clusters);
                criterion.add(clusters);
                cost_ = std::min(cost_, criterion.calculate());
            }

            maths::CKMeansOnline<TVector2> kmeansOnline(24);
            double costOnline_ = std::numeric_limits<double>::max();
            {
                for (std::size_t i = 0u; i < points.size(); ++i) {
                    kmeansOnline.add(points[i]);
                }
                maths::CKMeansOnline<TVector2>::TSphericalClusterVecVec clustersOnline;
                kmeansOnline.kmeans(2, clustersOnline);
                maths::CSphericalGaussianInfoCriterion<maths::CKMeansOnline<TVector2>::TSphericalCluster, maths::E_BIC> criterion;
                criterion.add(clustersOnline);
                costOnline_ = criterion.calculate();
            }
            LOG_DEBUG("cost = " << cost_ << ", cost online = " << costOnline_);

            cost.add(cost_);
            costOnline.add(costOnline_);

            rng.random_shuffle(points.begin(), points.end());
        }

        LOG_DEBUG("cost        = " << cost);
        LOG_DEBUG("cost online = " << costOnline);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(costOnline), maths::CBasicStatistics::mean(cost), 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            ::sqrt(maths::CBasicStatistics::variance(costOnline)), ::sqrt(maths::CBasicStatistics::variance(cost)), 1e-10);
    }

    {
        TMeanVarAccumulator cost;
        TMeanVarAccumulator costOnline;

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 1000, coordinates);
        TVector2Vec points;
        for (std::size_t i = 0u; i < coordinates.size(); i += 2) {
            double v[] = {coordinates[i + 0], coordinates[i + 1]};
            points.push_back(TVector2(v));
        }

        for (std::size_t t = 1u; t <= 20; ++t) {
            LOG_DEBUG("*** test = " << t << " ***");

            maths::CKMeansFast<TVector2> kmeans;
            maths::CKMeansOnline<TVector2> kmeansOnline(24);

            double cost_ = std::numeric_limits<double>::max();
            kmeans.setPoints(points);
            TVector2Vec centres;
            TVector2VecVec clusters;
            maths::CPRNG::CXorOShiro128Plus rng_;
            for (std::size_t i = 0u; i < 10; ++i) {
                maths::CKMeansPlusPlusInitialization<TVector2, maths::CPRNG::CXorOShiro128Plus> seedCentres(rng_);
                seedCentres.run(points, 3, centres);
                kmeans.setCentres(centres);
                kmeans.run(10);
                maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC> criterion;
                kmeans.clusters(clusters);
                criterion.add(clusters);
                cost_ = std::min(cost_, criterion.calculate());
            }

            double costOnline_ = std::numeric_limits<double>::max();
            {
                for (std::size_t i = 0u; i < points.size(); ++i) {
                    kmeansOnline.add(points[i]);
                }
                maths::CKMeansOnline<TVector2>::TSphericalClusterVecVec clustersOnline;
                kmeansOnline.kmeans(3, clustersOnline);
                maths::CSphericalGaussianInfoCriterion<maths::CKMeansOnline<TVector2>::TSphericalCluster, maths::E_BIC> criterion;
                criterion.add(clustersOnline);
                costOnline_ = criterion.calculate();
            }
            LOG_DEBUG("cost = " << cost_ << ", cost online = " << costOnline_);

            cost.add(cost_);
            costOnline.add(costOnline_);

            rng.random_shuffle(points.begin(), points.end());
        }

        LOG_DEBUG("cost        = " << cost);
        LOG_DEBUG("cost online = " << costOnline);

        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(costOnline) <= 1.01 * maths::CBasicStatistics::mean(cost));
        CPPUNIT_ASSERT(::sqrt(maths::CBasicStatistics::variance(costOnline)) <= 26.0 * ::sqrt(maths::CBasicStatistics::variance(cost)));
    }
}

void CKMeansOnlineTest::testSplit(void) {
    LOG_DEBUG("+--------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testSplit  |");
    LOG_DEBUG("+--------------------------------+");

    // Test that the clusters are divided amoung the clusterers
    // in the split as expected.

    typedef std::vector<maths::CKMeansOnline<TVector2>> TKMeansOnline2Vec;

    test::CRandomNumbers rng;

    double m[] = {5.0, 15.0};
    double v[] = {5.0, 10.0};
    TVector2Vec points;
    for (std::size_t i = 0u; i < 2; ++i) {
        TDoubleVec coordinates;
        rng.generateNormalSamples(m[i], v[i], 350, coordinates);
        for (std::size_t j = 0u; j < coordinates.size(); j += 2) {
            double c[] = {coordinates[j + 0], coordinates[j + 1]};
            points.push_back(TVector2(c));
        }
    }

    maths::CKMeansOnline<TVector2> kmeansOnline(30);
    for (std::size_t i = 0u; i < points.size(); ++i) {
        kmeansOnline.add(points[i]);
    }
    CPPUNIT_ASSERT(!kmeansOnline.buffering());

    std::size_t one[] = {0, 2, 7, 18, 19, 22};
    std::size_t two[] = {3, 4, 5, 6, 10, 11, 23, 24};
    std::size_t three[] = {1, 8, 9, 12, 13, 14, 15, 16, 17};
    std::size_t four[] = {20, 21, 25, 26, 27, 28};
    std::size_t five[] = {29};
    TSizeVecVec split;
    split.push_back(TSizeVec(boost::begin(one), boost::end(one)));
    split.push_back(TSizeVec(boost::begin(two), boost::end(two)));
    split.push_back(TSizeVec(boost::begin(three), boost::end(three)));
    split.push_back(TSizeVec(boost::begin(four), boost::end(four)));
    split.push_back(TSizeVec(boost::begin(five), boost::end(five)));

    maths::CKMeansOnline<TVector2>::TSphericalClusterVec clusters;
    kmeansOnline.clusters(clusters);
    TKMeansOnline2Vec clusterers;
    kmeansOnline.split(split, clusterers);

    CPPUNIT_ASSERT_EQUAL(split.size(), clusterers.size());
    for (std::size_t i = 0u; i < split.size(); ++i) {
        maths::CKMeansOnline<TVector2>::TSphericalClusterVec actual;
        clusterers[i].clusters(actual);
        CPPUNIT_ASSERT(!clusterers[i].buffering());
        CPPUNIT_ASSERT_EQUAL(split[i].size(), actual.size());

        maths::CKMeansOnline<TVector2>::TSphericalClusterVec expected;
        for (std::size_t j = 0u; j < split[i].size(); ++j) {
            expected.push_back(clusters[split[i][j]]);
        }
        LOG_DEBUG("expected clusters = " << core::CContainerPrinter::print(expected));
        LOG_DEBUG("actual clusters   = " << core::CContainerPrinter::print(actual));

        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected), core::CContainerPrinter::print(actual));
    }
}

void CKMeansOnlineTest::testMerge(void) {
    LOG_DEBUG("+--------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testMerge  |");
    LOG_DEBUG("+--------------------------------+");

    // Test some invariants:
    //   - Number of clusters should be no more than k after merge.
    //   - The count of the points should be unchanged.
    //   - The centroid of the points should be unchanged.
    //   - The total spherical variance of the points should
    //     be unchanged.

    test::CRandomNumbers rng;

    double m[] = {5.0, 15.0};
    double v[] = {5.0, 10.0};
    TVector2Vec points[2];
    for (std::size_t i = 0u; i < 2; ++i) {
        TDoubleVec coordinates;
        rng.generateNormalSamples(m[i], v[i], 350, coordinates);
        for (std::size_t j = 0u; j < coordinates.size(); j += 2) {
            double c[] = {coordinates[j + 0], coordinates[j + 1]};
            points[i].push_back(TVector2(c));
        }
    }

    maths::CKMeansOnline<TVector2> kmeans[] = {maths::CKMeansOnline<TVector2>(20), maths::CKMeansOnline<TVector2>(25)};
    for (std::size_t i = 0u; i < 2; ++i) {
        for (std::size_t j = 0u; j < points[i].size(); ++j) {
            kmeans[i].add(points[i][j]);
        }
    }

    TMeanVar2Accumulator expected;
    for (std::size_t i = 0u; i < 2; ++i) {
        CKMeansOnlineTestForTest<TVector2>::TSphericalClusterVec clusters;
        kmeans[i].clusters(clusters);
        for (std::size_t j = 0u; j < clusters.size(); ++j) {
            expected.add(clusters[j]);
        }
    }

    kmeans[0].merge(kmeans[1]);

    TMeanVar2Accumulator actual;
    CKMeansOnlineTestForTest<TVector2>::TSphericalClusterVec clusters;
    kmeans[0].clusters(clusters);
    for (std::size_t j = 0u; j < clusters.size(); ++j) {
        actual.add(clusters[j]);
    }

    TVector2 ones(1.0);

    LOG_DEBUG("expected = " << expected);
    LOG_DEBUG("actual   = " << actual);
    CPPUNIT_ASSERT_EQUAL(maths::CBasicStatistics::count(expected), maths::CBasicStatistics::count(actual));
    CPPUNIT_ASSERT_EQUAL(print(maths::CBasicStatistics::mean(expected)), print(maths::CBasicStatistics::mean(actual)));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones),
                                 maths::CBasicStatistics::maximumLikelihoodVariance(actual).inner(ones),
                                 1e-10 * maths::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones));
}

void CKMeansOnlineTest::testPropagateForwardsByTime(void) {
    LOG_DEBUG("+--------------------------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testPropagateForwardsByTime  |");
    LOG_DEBUG("+--------------------------------------------------+");

    // Test pruning of dead clusters.

    test::CRandomNumbers rng;

    double m = 5.0;
    double v = 4.0;
    TVector2Vec points;
    TDoubleVec coordinates;
    rng.generateNormalSamples(m, v, 700, coordinates);
    double outlier_[] = {50.0, 20.0};
    TVector2 outlier(outlier_);
    for (std::size_t i = 0u; i < coordinates.size(); i += 2) {
        double c[] = {coordinates[i + 0], coordinates[i + 1]};
        points.push_back(TVector2(c));
        if (i == 200) {
            points.push_back(outlier);
        }
    }

    maths::CKMeansOnline<TVector2> kmeans(5, 0.1);
    for (std::size_t i = 0u; i < points.size(); ++i) {
        kmeans.add(points[i]);
    }

    CKMeansOnlineTestForTest<TVector2>::TSphericalClusterVec clusters;
    kmeans.clusters(clusters);
    LOG_DEBUG("clusters before = " << core::CContainerPrinter::print(clusters));

    kmeans.propagateForwardsByTime(7.0);

    kmeans.clusters(clusters);
    LOG_DEBUG("clusters after  = " << core::CContainerPrinter::print(clusters));

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), clusters.size());
    for (std::size_t i = 0u; i < clusters.size(); ++i) {
        CPPUNIT_ASSERT(clusters[i] != outlier);
    }
}

void CKMeansOnlineTest::testSample(void) {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testSample  |");
    LOG_DEBUG("+---------------------------------+");

    // We test that for a small number of samples we get back
    // exactly the points we have added and for a large number
    // of samples we sample the modes of the mixture correctly.

    typedef maths::CSymmetricMatrixNxN<double, 2> TMatrix2;

    maths::CSampling::seed();

    std::size_t n[] = {500, 500};
    double means[][2] = {{0.0, 10.0}, {20.0, 30.0}};
    double covariances[][3] = {{10.0, 2.0, 8.0}, {15.0, 5.0, 12.0}};

    maths::CBasicStatistics::SSampleCovariances<double, 2> expectedSampleCovariances[2];
    TVector2Vec samples;

    for (std::size_t i = 0u; i < 2; ++i) {
        TVector2 mean(means[i]);
        TMatrix2 covariance(covariances[i], covariances[i] + 3);
        TVector2Vec modeSamples;
        maths::CSampling::multivariateNormalSample(mean, covariance, n[i], modeSamples);
        expectedSampleCovariances[i].add(modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
    }

    test::CRandomNumbers rng;

    rng.random_shuffle(samples.begin(), samples.end());

    maths::CKMeansOnline<TVector2> kmeans(10, 0.1);

    TVector2Vec expectedSampled;
    for (std::size_t i = 0u; i < 10; ++i) {
        expectedSampled.push_back(samples[i]);
        std::sort(expectedSampled.begin(), expectedSampled.end());

        kmeans.add(samples[i]);
        TVector2Vec sampled;
        kmeans.sample(i + 1, sampled);
        std::sort(sampled.begin(), sampled.end());

        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSampled), core::CContainerPrinter::print(sampled));
    }

    for (std::size_t i = 10u; i < samples.size(); ++i) {
        kmeans.add(samples[i]);
    }

    TVector2Vec sampled;
    kmeans.sample(50u, sampled);
    std::sort(sampled.begin(), sampled.end());
    LOG_DEBUG("sampled = " << core::CContainerPrinter::print(sampled));

    maths::CBasicStatistics::SSampleCovariances<double, 2> sampleCovariances[2];
    for (std::size_t i = 0u; i < sampled.size(); ++i) {
        if ((sampled[i] - TVector2(means[0])).euclidean() < (sampled[i] - TVector2(means[1])).euclidean()) {
            sampleCovariances[0].add(sampled[i]);
        } else {
            sampleCovariances[1].add(sampled[i]);
        }
    }

    TVector2 expectedMean0 = maths::CBasicStatistics::mean(expectedSampleCovariances[0]);
    TMatrix2 expectedCovariance0 = maths::CBasicStatistics::covariances(expectedSampleCovariances[0]);
    TVector2 expectedMean1 = maths::CBasicStatistics::mean(expectedSampleCovariances[1]);
    TMatrix2 expectedCovariance1 = maths::CBasicStatistics::covariances(expectedSampleCovariances[1]);
    TVector2 mean0 = maths::CBasicStatistics::mean(sampleCovariances[0]);
    TMatrix2 covariance0 = maths::CBasicStatistics::covariances(sampleCovariances[0]);
    TVector2 mean1 = maths::CBasicStatistics::mean(sampleCovariances[1]);
    TMatrix2 covariance1 = maths::CBasicStatistics::covariances(sampleCovariances[1]);

    LOG_DEBUG("expected mean, variance 0 = " << expectedMean0 << ", " << expectedCovariance0);
    LOG_DEBUG("mean, variance 0          = " << mean0 << ", " << covariance0);
    LOG_DEBUG("expected mean, variance 1 = " << expectedMean1 << ", " << expectedCovariance1);
    LOG_DEBUG("mean, variance 1          = " << mean1 << ", " << covariance1);

    double meanError0 = (mean0 - expectedMean0).euclidean() / expectedMean0.euclidean();
    double covarianceError0 = (covariance0 - expectedCovariance0).frobenius() / expectedCovariance0.frobenius();
    LOG_DEBUG("mean error 0 = " << meanError0 << ", covariance error 0 = " << covarianceError0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meanError0, 0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, covarianceError0, 0.27);

    double meanError1 = (mean1 - expectedMean1).euclidean() / expectedMean0.euclidean();
    double covarianceError1 = (covariance1 - expectedCovariance1).frobenius() / expectedCovariance1.frobenius();
    LOG_DEBUG("mean error 1 = " << meanError1 << ", covariance error 1 = " << covarianceError1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meanError1, 0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, covarianceError1, 0.24);
}

void CKMeansOnlineTest::testPersist(void) {
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CKMeansOnlineTest::testPersist  |");
    LOG_DEBUG("+----------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec coordinates;
    rng.generateUniformSamples(0.0, 400.0, 998, coordinates);
    TVector2Vec points;
    for (std::size_t i = 0u; i < coordinates.size(); i += 2) {
        points.push_back(TVector2(&coordinates[i], &coordinates[i + 2]));
    }

    maths::CKMeansOnline<TVector2> origKmeans(25, 0.1);
    for (std::size_t i = 0u; i < points.size(); ++i) {
        origKmeans.add(points[i]);
    }

    LOG_DEBUG("k-means = " << origKmeans.print());

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origKmeans.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG("original k-means XML = " << origXml);

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CKMeansOnline<TVector2> restoredKmeans(0);
        maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                                 0.1,
                                                 maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                                 maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                                 maths::MINIMUM_CATEGORY_COUNT);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            boost::bind(&maths::CKMeansOnline<TVector2>::acceptRestoreTraverser, &restoredKmeans, boost::cref(params), _1)));

        LOG_DEBUG("orig checksum = " << origKmeans.checksum() << ", new checksum = " << restoredKmeans.checksum());
        CPPUNIT_ASSERT_EQUAL(origKmeans.checksum(), restoredKmeans.checksum());

        std::string newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredKmeans.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test* CKMeansOnlineTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CKMeansOnlineTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testVariance", &CKMeansOnlineTest::testVariance));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testAdd", &CKMeansOnlineTest::testAdd));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testReduce", &CKMeansOnlineTest::testReduce));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testClustering", &CKMeansOnlineTest::testClustering));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testSplit", &CKMeansOnlineTest::testSplit));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testMerge", &CKMeansOnlineTest::testMerge));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testPropagateForwardsByTime",
                                                                     &CKMeansOnlineTest::testPropagateForwardsByTime));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testSample", &CKMeansOnlineTest::testSample));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansOnlineTest>("CKMeansOnlineTest::testPersist", &CKMeansOnlineTest::testPersist));

    return suiteOfTests;
}
