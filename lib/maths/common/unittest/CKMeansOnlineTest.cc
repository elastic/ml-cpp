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

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CKMeansOnline.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CRestoreParams.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CKMeansOnlineTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TMeanVarAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TVector2 = maths::common::CVectorNx1<double, 2>;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecVec = std::vector<TVector2Vec>;
using TFloatVector2 = maths::common::CVectorNx1<maths::common::CFloatStorage, 2>;
using TMean2Accumulator = maths::common::CBasicStatistics::SSampleMean<TVector2>::TAccumulator;
using TMeanVar2Accumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator;
using TVector5 = maths::common::CVectorNx1<double, 5>;
using TVector5Vec = std::vector<TVector5>;
using TMeanVar5Accumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<TVector5>::TAccumulator;

template<typename POINT>
class CKMeansOnlineForTest : public maths::common::CKMeansOnline<POINT> {
public:
    using TSphericalClusterVec = typename maths::common::CKMeansOnline<POINT>::TSphericalClusterVec;
    using TStorageCoordinate = typename maths::common::CKMeansOnline<POINT>::TStorageCoordinate;
    using TStoragePoint = typename maths::common::CKMeansOnline<POINT>::TStoragePoint;
    using TStoragePointMeanAccumulatorDoublePr =
        typename maths::common::CKMeansOnline<POINT>::TStoragePointMeanAccumulatorDoublePr;
    using TStoragePointMeanAccumulatorDoublePrVec =
        typename maths::common::CKMeansOnline<POINT>::TStoragePointMeanAccumulatorDoublePrVec;
    using TDoublePointMeanVarAccumulator =
        typename maths::common::CKMeansOnline<POINT>::TDoublePointMeanVarAccumulator;
    using TDoublePoint = typename maths::common::CKMeansOnline<POINT>::TDoublePoint;

public:
    CKMeansOnlineForTest(std::size_t k, double decayRate = 0.0)
        : maths::common::CKMeansOnline<POINT>(k, decayRate) {}

    static void deduplicate(TStoragePointMeanAccumulatorDoublePrVec& clusters) {
        maths::common::CKMeansOnline<POINT>::deduplicate(clusters);
    }

    static void add(const POINT& mx, double count, TStoragePointMeanAccumulatorDoublePr& cluster) {
        double nx{count};
        TDoublePoint vx{maths::common::las::zero(mx)};
        double nc{maths::common::CBasicStatistics::count(cluster.first)};
        TDoublePoint mc{maths::common::CBasicStatistics::mean(cluster.first)};
        TDoublePoint vc{cluster.second * maths::common::las::ones(mx)};
        TDoublePointMeanVarAccumulator moments{
            maths::common::CBasicStatistics::momentsAccumulator(nc, mc, vc) +
            maths::common::CBasicStatistics::momentsAccumulator(nx, mx, vx)};
        TStorageCoordinate ncx{maths::common::CBasicStatistics::count(moments)};
        TStoragePoint mcx{maths::common::CBasicStatistics::mean(moments)};
        cluster.first = maths::common::CBasicStatistics::momentsAccumulator(ncx, mcx);
        cluster.second = variance(moments);
    }

    static double variance(const TDoublePointMeanVarAccumulator& moments) {
        return maths::common::CKMeansOnline<POINT>::variance(moments);
    }
};

template<typename POINT>
std::string print(const POINT& point) {
    std::ostringstream result;
    result << point;
    return result.str();
}
}

BOOST_AUTO_TEST_CASE(testVariance) {
    // Check that the variance calculation gives the correct
    // spherical variance.

    test::CRandomNumbers rng;

    for (std::size_t t = 1; t <= 50; ++t) {
        LOG_DEBUG(<< "*** test = " << t << " ***");

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 50, coordinates);
        TVector5Vec points;
        for (std::size_t i = 0; i < coordinates.size(); i += 5) {
            double c[]{coordinates[i + 0], coordinates[i + 1], coordinates[i + 2],
                       coordinates[i + 3], coordinates[i + 4]};
            points.push_back(TVector5(c));
        }

        TMeanVar5Accumulator actual;
        actual.add(points);

        TMeanVarAccumulator expected;
        for (std::size_t i = 0; i < coordinates.size(); ++i) {
            expected.add(coordinates[i] -
                         maths::common::CBasicStatistics::mean(actual)(i % 5));
        }

        LOG_DEBUG(<< "actual   = " << CKMeansOnlineForTest<TVector5>::variance(actual));
        LOG_DEBUG(<< "expected = "
                  << maths::common::CBasicStatistics::maximumLikelihoodVariance(expected));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            maths::common::CBasicStatistics::maximumLikelihoodVariance(expected),
            CKMeansOnlineForTest<TVector5>::variance(actual),
            1e-10 * maths::common::CBasicStatistics::maximumLikelihoodVariance(expected));
    }
}

BOOST_AUTO_TEST_CASE(testAdd) {
    // Test that we correctly compute the mean and spherical
    // variance.

    using TMean2AccumulatorDoublePr = std::pair<TMean2Accumulator, double>;

    test::CRandomNumbers rng;

    for (std::size_t t = 1; t <= 50; ++t) {
        LOG_DEBUG(<< "*** test = " << t << " ***");

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 40, coordinates);
        TDoubleVec counts;
        rng.generateUniformSamples(1.0, 2.0, 20, counts);
        TVector2Vec points;
        for (std::size_t i = 0; i < coordinates.size(); i += 2) {
            double c[]{coordinates[i + 0], coordinates[i + 1]};
            points.push_back(TVector2(c));
        }

        TMean2AccumulatorDoublePr actual;
        TMeanVar2Accumulator expected;
        for (std::size_t i = 0; i < points.size(); ++i) {
            CKMeansOnlineForTest<TVector2>::add(points[i], counts[i], actual);
            expected.add(points[i], counts[i]);
        }

        TVector2 ones(1.0);

        LOG_DEBUG(<< "actual   = "
                  << maths::common::CBasicStatistics::mean(actual.first) << ","
                  << actual.second);
        LOG_DEBUG(<< "expected = " << maths::common::CBasicStatistics::mean(expected) << ","
                  << maths::common::CBasicStatistics::maximumLikelihoodVariance(expected)
                             .inner(ones) /
                         static_cast<double>(ones.dimension()));

        BOOST_REQUIRE_EQUAL(print(maths::common::CBasicStatistics::mean(expected)),
                            print(maths::common::CBasicStatistics::mean(actual.first)));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            maths::common::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones) /
                static_cast<double>(ones.dimension()),
            actual.second,
            1e-10 *
                maths::common::CBasicStatistics::maximumLikelihoodVariance(expected)
                    .inner(ones) /
                static_cast<double>(ones.dimension()));
    }
}

BOOST_AUTO_TEST_CASE(testDeduplicate) {
    // Test we behaviour:
    //   - If all points are duplicates
    //   - If no points are duplicates
    //   - For random permutation of duplicates

    CKMeansOnlineForTest<TVector2>::TStoragePointMeanAccumulatorDoublePrVec points;

    points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                            maths::common::CFloatStorage{1.0}, TFloatVector2{0.0}),
                        0.0);
    points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                            maths::common::CFloatStorage{2.0}, TFloatVector2{0.0}),
                        0.0);
    points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                            maths::common::CFloatStorage{1.0}, TFloatVector2{0.0}),
                        0.0);
    CKMeansOnlineForTest<TVector2>::deduplicate(points);
    BOOST_REQUIRE_EQUAL(1, points.size());
    BOOST_REQUIRE_EQUAL(TFloatVector2{0.0},
                        maths::common::CBasicStatistics::mean(points[0].first));
    BOOST_REQUIRE_EQUAL(4.0, maths::common::CBasicStatistics::count(points[0].first));
    BOOST_REQUIRE_EQUAL(0.0, points[0].second);
    points.clear();

    points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                            maths::common::CFloatStorage{1.0}, TFloatVector2{0.0}),
                        0.0);
    points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                            maths::common::CFloatStorage{2.0}, TFloatVector2{1.0}),
                        0.0);
    points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                            maths::common::CFloatStorage{1.0}, TFloatVector2{2.0}),
                        0.0);
    CKMeansOnlineForTest<TVector2>::deduplicate(points);
    BOOST_REQUIRE_EQUAL(3, points.size());
    BOOST_REQUIRE_EQUAL(TFloatVector2{0.0},
                        maths::common::CBasicStatistics::mean(points[0].first));
    BOOST_REQUIRE_EQUAL(1.0, maths::common::CBasicStatistics::count(points[0].first));
    BOOST_REQUIRE_EQUAL(0.0, points[0].second);
    BOOST_REQUIRE_EQUAL(TFloatVector2{1.0},
                        maths::common::CBasicStatistics::mean(points[1].first));
    BOOST_REQUIRE_EQUAL(2.0, maths::common::CBasicStatistics::count(points[1].first));
    BOOST_REQUIRE_EQUAL(0.0, points[1].second);
    BOOST_REQUIRE_EQUAL(TFloatVector2{2.0},
                        maths::common::CBasicStatistics::mean(points[2].first));
    BOOST_REQUIRE_EQUAL(1.0, maths::common::CBasicStatistics::count(points[2].first));
    BOOST_REQUIRE_EQUAL(0.0, points[2].second);
    points.clear();

    test::CRandomNumbers rng;
    for (std::size_t t = 1; t <= 100; ++t) {
        for (std::size_t i = 0; i < 5; ++i) {
            points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                                    maths::common::CFloatStorage{static_cast<double>(i)},
                                    TFloatVector2{0.0}),
                                0.0);
        }
        for (std::size_t i = 0; i < 7; ++i) {
            points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                                    maths::common::CFloatStorage{1.0}, TFloatVector2{1.0}),
                                0.0);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            points.emplace_back(maths::common::CBasicStatistics::momentsAccumulator(
                                    maths::common::CFloatStorage{2.0}, TFloatVector2{2.0}),
                                0.0);
        }
        rng.random_shuffle(points.begin(), points.end());

        CKMeansOnlineForTest<TVector2>::deduplicate(points);

        BOOST_REQUIRE_EQUAL(3, points.size());
        BOOST_REQUIRE_EQUAL(TFloatVector2{0.0},
                            maths::common::CBasicStatistics::mean(points[0].first));
        BOOST_REQUIRE_EQUAL(10.0,
                            maths::common::CBasicStatistics::count(points[0].first));
        BOOST_REQUIRE_EQUAL(TFloatVector2{1.0},
                            maths::common::CBasicStatistics::mean(points[1].first));
        BOOST_REQUIRE_EQUAL(7.0,
                            maths::common::CBasicStatistics::count(points[1].first));
        BOOST_REQUIRE_EQUAL(TFloatVector2{2.0},
                            maths::common::CBasicStatistics::mean(points[2].first));
        BOOST_REQUIRE_EQUAL(6.0,
                            maths::common::CBasicStatistics::count(points[2].first));

        points.clear();
    }
}

BOOST_AUTO_TEST_CASE(testReduce) {
    // Test some invariants:
    //   - Number of clusters should be no more than k after
    //     reduce.
    //   - The count of the points should be unchanged.
    //   - The centroid of the points should be unchanged.
    //   - The total spherical variance of the points should
    //     be unchanged.

    test::CRandomNumbers rng;

    for (std::size_t t = 1; t <= 10; ++t) {
        LOG_DEBUG(<< "*** test = " << t << " ***");

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 42, coordinates);
        TDoubleVec counts;
        rng.generateUniformSamples(1.0, 2.0, 21, counts);
        TVector2Vec points;
        for (std::size_t i = 0; i < coordinates.size(); i += 2) {
            double c[]{coordinates[i + 0], coordinates[i + 1]};
            points.push_back(TVector2(c));
        }

        maths::common::CKMeansOnline<TVector2> kmeans(10);

        TMeanVar2Accumulator expected;

        TVector2 ones(1.0);

        for (std::size_t i = 0; i < points.size(); ++i) {
            kmeans.add(points[i], counts[i]);
            expected.add(points[i], counts[i]);

            if (((i - 10) + 1) % 7 == 0) {
                CKMeansOnlineForTest<TVector2>::TSphericalClusterVec clusters;
                kmeans.clusters(clusters);
                BOOST_TEST_REQUIRE(clusters.size() <= 10);

                TMeanVar2Accumulator actual;
                for (std::size_t j = 0; j < clusters.size(); ++j) {
                    actual.add(clusters[j]);
                }

                LOG_DEBUG(<< "expected = " << expected);
                LOG_DEBUG(<< "actual   = " << actual);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::common::CBasicStatistics::count(expected),
                    maths::common::CBasicStatistics::count(actual), 1e-10);
                BOOST_REQUIRE_EQUAL(
                    print(maths::common::CBasicStatistics::mean(expected)),
                    print(maths::common::CBasicStatistics::mean(actual)));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::common::CBasicStatistics::maximumLikelihoodVariance(expected)
                        .inner(ones),
                    maths::common::CBasicStatistics::maximumLikelihoodVariance(actual)
                        .inner(ones),
                    1e-10 * maths::common::CBasicStatistics::maximumLikelihoodVariance(expected)
                                .inner(ones));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testClustering) {
    // Test we are reliably able to find as approximately as good
    // clusterings as k-means working on the full data set.

    using TKMeansPlusPlusInitialization =
        maths::common::CKMeansPlusPlusInitialization<TVector2, maths::common::CPRNG::CXorOShiro128Plus>;
    using TSphericalClusterBic = maths::common::CSphericalGaussianInfoCriterion<
        maths::common::CKMeansOnline<TVector2>::TSphericalCluster, maths::common::E_BIC>;
    using TVectorBic =
        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC>;

    test::CRandomNumbers rng;

    {
        TMeanVarAccumulator cost;
        TMeanVarAccumulator costOnline;

        double a[]{0.0, 20.0};
        double b[]{5.0, 30.0};
        TVector2Vec points;
        for (std::size_t i = 0; i < 2; ++i) {
            TDoubleVec coordinates;
            rng.generateUniformSamples(a[i], b[i], 200, coordinates);
            for (std::size_t j = 0; j < coordinates.size(); j += 2) {
                double c[]{coordinates[j + 0], coordinates[j + 1]};
                points.push_back(TVector2(c));
            }
        }

        for (std::size_t t = 1; t <= 10; ++t) {
            LOG_DEBUG(<< "*** test = " << t << " ***");

            maths::common::CKMeans<TVector2> kmeans;
            double cost_ = std::numeric_limits<double>::max();
            kmeans.setPoints(points);
            TVector2Vec centres;
            TVector2VecVec clusters;
            maths::common::CPRNG::CXorOShiro128Plus rng_;
            for (std::size_t i = 0; i < 10; ++i) {
                TKMeansPlusPlusInitialization seedCentres(rng_);
                seedCentres.run(points, 2, centres);
                kmeans.setCentres(centres);
                kmeans.run(10);
                kmeans.clusters(clusters);
                TVectorBic bic;
                bic.add(clusters);
                cost_ = std::min(cost_, bic.calculate());
            }

            maths::common::CKMeansOnline<TVector2> kmeansOnline(24);
            double costOnline_ = std::numeric_limits<double>::max();
            {
                for (const auto& point : points) {
                    kmeansOnline.add(point);
                }
                maths::common::CKMeansOnline<TVector2>::TSphericalClusterVecVec clustersOnline;
                kmeansOnline.kmeans(2, clustersOnline);
                TSphericalClusterBic bic;
                bic.add(clustersOnline);
                costOnline_ = bic.calculate();
            }
            LOG_DEBUG(<< "cost = " << cost_ << ", cost online = " << costOnline_);

            cost.add(cost_);
            costOnline.add(costOnline_);

            rng.random_shuffle(points.begin(), points.end());
        }

        LOG_DEBUG(<< "cost        = " << cost);
        LOG_DEBUG(<< "cost online = " << costOnline);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::common::CBasicStatistics::mean(costOnline),
                                     maths::common::CBasicStatistics::mean(cost), 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            std::sqrt(maths::common::CBasicStatistics::variance(costOnline)),
            std::sqrt(maths::common::CBasicStatistics::variance(cost)), 1e-10);
    }

    {
        TMeanVarAccumulator cost;
        TMeanVarAccumulator costOnline;

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 10.0, 1000, coordinates);
        TVector2Vec points;
        for (std::size_t i = 0; i < coordinates.size(); i += 2) {
            double v[]{coordinates[i + 0], coordinates[i + 1]};
            points.push_back(TVector2(v));
        }

        for (std::size_t t = 1; t <= 20; ++t) {
            LOG_DEBUG(<< "*** test = " << t << " ***");

            maths::common::CKMeans<TVector2> kmeans;
            maths::common::CKMeansOnline<TVector2> kmeansOnline(24);

            double cost_ = std::numeric_limits<double>::max();
            kmeans.setPoints(points);
            TVector2Vec centres;
            TVector2VecVec clusters;
            maths::common::CPRNG::CXorOShiro128Plus rng_;
            for (std::size_t i = 0; i < 10; ++i) {
                TKMeansPlusPlusInitialization seedCentres(rng_);
                seedCentres.run(points, 3, centres);
                kmeans.setCentres(centres);
                kmeans.run(10);
                kmeans.clusters(clusters);
                TVectorBic bic;
                bic.add(clusters);
                cost_ = std::min(cost_, bic.calculate());
            }

            double costOnline_ = std::numeric_limits<double>::max();
            {
                for (std::size_t i = 0; i < points.size(); ++i) {
                    kmeansOnline.add(points[i]);
                }
                maths::common::CKMeansOnline<TVector2>::TSphericalClusterVecVec clustersOnline;
                kmeansOnline.kmeans(3, clustersOnline);
                TSphericalClusterBic bic;
                bic.add(clustersOnline);
                costOnline_ = bic.calculate();
            }
            LOG_DEBUG(<< "cost = " << cost_ << ", cost online = " << costOnline_);

            cost.add(cost_);
            costOnline.add(costOnline_);

            rng.random_shuffle(points.begin(), points.end());
        }

        LOG_DEBUG(<< "cost        = " << cost);
        LOG_DEBUG(<< "cost online = " << costOnline);

        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(costOnline) <=
                           1.01 * maths::common::CBasicStatistics::mean(cost));
        BOOST_TEST_REQUIRE(
            std::sqrt(maths::common::CBasicStatistics::variance(costOnline)) <=
            26.0 * std::sqrt(maths::common::CBasicStatistics::variance(cost)));
    }
}

BOOST_AUTO_TEST_CASE(testSplit) {
    // Test that the clusters are divided amoung the clusterers
    // in the split as expected.

    using TKMeansOnline2Vec = std::vector<maths::common::CKMeansOnline<TVector2>>;

    test::CRandomNumbers rng;

    double m[]{5.0, 15.0};
    double v[]{5.0, 10.0};
    TVector2Vec points;
    for (std::size_t i = 0; i < 2; ++i) {
        TDoubleVec coordinates;
        rng.generateNormalSamples(m[i], v[i], 352, coordinates);
        for (std::size_t j = 0; j < coordinates.size(); j += 2) {
            double c[]{coordinates[j + 0], coordinates[j + 1]};
            points.push_back(TVector2(c));
        }
    }

    maths::common::CKMeansOnline<TVector2> kmeansOnline(30);
    const std::size_t BUFFERING{0};
    const std::size_t NOT_BUFFERING{1};
    std::size_t counts[2]{0, 0};
    for (std::size_t i = 0; i < 30; ++i) {
        kmeansOnline.add(points[i]);
    }
    for (std::size_t i = 30; i < points.size(); ++i) {
        kmeansOnline.add(points[i]);
        ++counts[kmeansOnline.buffering() ? BUFFERING : NOT_BUFFERING];
    }
    BOOST_REQUIRE_EQUAL(counts[BUFFERING], maths::common::CKMeansOnline<TVector2>::BUFFER_SIZE *
                                               counts[NOT_BUFFERING]);

    std::size_t one[]{0, 2, 7, 18, 19, 22};
    std::size_t two[]{3, 4, 5, 6, 10, 11, 23, 24};
    std::size_t three[]{1, 8, 9, 12, 13, 14, 15, 16, 17};
    std::size_t four[]{20, 21, 25, 26, 27, 28};
    std::size_t five[]{29};
    TSizeVecVec split;
    split.push_back(TSizeVec(std::begin(one), std::end(one)));
    split.push_back(TSizeVec(std::begin(two), std::end(two)));
    split.push_back(TSizeVec(std::begin(three), std::end(three)));
    split.push_back(TSizeVec(std::begin(four), std::end(four)));
    split.push_back(TSizeVec(std::begin(five), std::end(five)));

    maths::common::CKMeansOnline<TVector2>::TSphericalClusterVec clusters;
    kmeansOnline.clusters(clusters);
    TKMeansOnline2Vec clusterers;
    kmeansOnline.split(split, clusterers);

    BOOST_REQUIRE_EQUAL(split.size(), clusterers.size());
    for (std::size_t i = 0; i < split.size(); ++i) {
        maths::common::CKMeansOnline<TVector2>::TSphericalClusterVec actual;
        clusterers[i].clusters(actual);
        BOOST_TEST_REQUIRE(!clusterers[i].buffering());
        BOOST_REQUIRE_EQUAL(split[i].size(), actual.size());

        maths::common::CKMeansOnline<TVector2>::TSphericalClusterVec expected;
        for (std::size_t j = 0; j < split[i].size(); ++j) {
            expected.push_back(clusters[split[i][j]]);
        }
        LOG_DEBUG(<< "expected clusters = " << core::CContainerPrinter::print(expected));
        LOG_DEBUG(<< "actual clusters   = " << core::CContainerPrinter::print(actual));

        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expected),
                            core::CContainerPrinter::print(actual));
    }
}

BOOST_AUTO_TEST_CASE(testMerge) {
    // Test some invariants:
    //   - Number of clusters should be no more than k after merge.
    //   - The count of the points should be unchanged.
    //   - The centroid of the points should be unchanged.
    //   - The total spherical variance of the points should
    //     be unchanged.

    test::CRandomNumbers rng;

    double m[]{5.0, 15.0};
    double v[]{5.0, 10.0};
    TVector2Vec points[2];
    for (std::size_t i = 0; i < 2; ++i) {
        TDoubleVec coordinates;
        rng.generateNormalSamples(m[i], v[i], 350, coordinates);
        for (std::size_t j = 0; j < coordinates.size(); j += 2) {
            double c[]{coordinates[j + 0], coordinates[j + 1]};
            points[i].push_back(TVector2(c));
        }
    }

    maths::common::CKMeansOnline<TVector2> kmeans[]{
        maths::common::CKMeansOnline<TVector2>(20),
        maths::common::CKMeansOnline<TVector2>(25)};
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < points[i].size(); ++j) {
            kmeans[i].add(points[i][j]);
        }
    }

    TMeanVar2Accumulator expected;
    for (std::size_t i = 0; i < 2; ++i) {
        CKMeansOnlineForTest<TVector2>::TSphericalClusterVec clusters;
        kmeans[i].clusters(clusters);
        for (std::size_t j = 0; j < clusters.size(); ++j) {
            expected.add(clusters[j]);
        }
    }

    kmeans[0].merge(kmeans[1]);

    TMeanVar2Accumulator actual;
    CKMeansOnlineForTest<TVector2>::TSphericalClusterVec clusters;
    kmeans[0].clusters(clusters);
    for (std::size_t j = 0; j < clusters.size(); ++j) {
        actual.add(clusters[j]);
    }

    TVector2 ones(1.0);

    LOG_DEBUG(<< "expected = " << expected);
    LOG_DEBUG(<< "actual   = " << actual);
    BOOST_REQUIRE_EQUAL(maths::common::CBasicStatistics::count(expected),
                        maths::common::CBasicStatistics::count(actual));
    BOOST_REQUIRE_EQUAL(print(maths::common::CBasicStatistics::mean(expected)),
                        print(maths::common::CBasicStatistics::mean(actual)));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        maths::common::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones),
        maths::common::CBasicStatistics::maximumLikelihoodVariance(actual).inner(ones),
        1e-10 *
            maths::common::CBasicStatistics::maximumLikelihoodVariance(expected).inner(ones));
}

BOOST_AUTO_TEST_CASE(testPropagateForwardsByTime) {
    // Test pruning of dead clusters.

    test::CRandomNumbers rng;

    double m = 5.0;
    double v = 4.0;
    TVector2Vec points;
    TDoubleVec coordinates;
    rng.generateNormalSamples(m, v, 700, coordinates);
    double outlier_[]{50.0, 20.0};
    TVector2 outlier(outlier_);
    for (std::size_t i = 0; i < coordinates.size(); i += 2) {
        double c[]{coordinates[i + 0], coordinates[i + 1]};
        points.push_back(TVector2(c));
        if (i == 200) {
            points.push_back(outlier);
        }
    }

    maths::common::CKMeansOnline<TVector2> kmeans(5, 0.1);
    for (std::size_t i = 0; i < points.size(); ++i) {
        kmeans.add(points[i]);
    }

    CKMeansOnlineForTest<TVector2>::TSphericalClusterVec clusters;
    kmeans.clusters(clusters);
    LOG_DEBUG(<< "clusters before = " << core::CContainerPrinter::print(clusters));

    kmeans.propagateForwardsByTime(7.0);

    kmeans.clusters(clusters);
    LOG_DEBUG(<< "clusters after  = " << core::CContainerPrinter::print(clusters));

    BOOST_REQUIRE_EQUAL(std::size_t(4), clusters.size());
    for (std::size_t i = 0; i < clusters.size(); ++i) {
        BOOST_TEST_REQUIRE(clusters[i] != outlier);
    }
}

BOOST_AUTO_TEST_CASE(testSample) {
    // We test that for a small number of samples we get back
    // exactly the points we have added and for a large number
    // of samples we sample the modes of the mixture correctly.

    using TMatrix2 = maths::common::CSymmetricMatrixNxN<double, 2>;
    using TCovariances2 =
        maths::common::CBasicStatistics::SSampleCovariances<maths::common::CVectorNx1<double, 2>>;

    maths::common::CSampling::seed();

    std::size_t n[]{500, 500};
    double means[][2]{{0.0, 10.0}, {20.0, 30.0}};
    double covariances[][3]{{10.0, 2.0, 8.0}, {15.0, 5.0, 12.0}};

    TCovariances2 expectedSampleCovariances[]{TCovariances2(2), TCovariances2(2)};
    TVector2Vec samples;

    for (std::size_t i = 0; i < 2; ++i) {
        TVector2 mean(means[i]);
        TMatrix2 covariance(covariances[i], covariances[i] + 3);
        TVector2Vec modeSamples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, n[i], modeSamples);
        expectedSampleCovariances[i].add(modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
    }

    test::CRandomNumbers rng;

    rng.random_shuffle(samples.begin(), samples.end());

    maths::common::CKMeansOnline<TVector2> kmeans(10, 0.1);

    TVector2Vec expectedSampled;
    for (std::size_t i = 0; i < 10; ++i) {
        expectedSampled.push_back(samples[i]);
        std::sort(expectedSampled.begin(), expectedSampled.end());

        kmeans.add(samples[i]);
        TVector2Vec sampled;
        kmeans.sample(i + 1, sampled);
        std::sort(sampled.begin(), sampled.end());

        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedSampled),
                            core::CContainerPrinter::print(sampled));
    }

    for (std::size_t i = 10; i < samples.size(); ++i) {
        kmeans.add(samples[i]);
    }

    TVector2Vec sampled;
    kmeans.sample(50u, sampled);
    std::sort(sampled.begin(), sampled.end());
    LOG_DEBUG(<< "sampled = " << core::CContainerPrinter::print(sampled));

    TCovariances2 sampleCovariances[]{TCovariances2(2), TCovariances2(2)};
    for (std::size_t i = 0; i < sampled.size(); ++i) {
        if ((sampled[i] - TVector2(means[0])).euclidean() <
            (sampled[i] - TVector2(means[1])).euclidean()) {
            sampleCovariances[0].add(sampled[i]);
        } else {
            sampleCovariances[1].add(sampled[i]);
        }
    }

    TVector2 expectedMean0 =
        maths::common::CBasicStatistics::mean(expectedSampleCovariances[0]);
    TMatrix2 expectedCovariance0 =
        maths::common::CBasicStatistics::covariances(expectedSampleCovariances[0]);
    TVector2 expectedMean1 =
        maths::common::CBasicStatistics::mean(expectedSampleCovariances[1]);
    TMatrix2 expectedCovariance1 =
        maths::common::CBasicStatistics::covariances(expectedSampleCovariances[1]);
    TVector2 mean0 = maths::common::CBasicStatistics::mean(sampleCovariances[0]);
    TMatrix2 covariance0 =
        maths::common::CBasicStatistics::covariances(sampleCovariances[0]);
    TVector2 mean1 = maths::common::CBasicStatistics::mean(sampleCovariances[1]);
    TMatrix2 covariance1 =
        maths::common::CBasicStatistics::covariances(sampleCovariances[1]);

    LOG_DEBUG(<< "expected mean, variance 0 = " << expectedMean0 << ", " << expectedCovariance0);
    LOG_DEBUG(<< "mean, variance 0          = " << mean0 << ", " << covariance0);
    LOG_DEBUG(<< "expected mean, variance 1 = " << expectedMean1 << ", " << expectedCovariance1);
    LOG_DEBUG(<< "mean, variance 1          = " << mean1 << ", " << covariance1);

    double meanError0 = (mean0 - expectedMean0).euclidean() / expectedMean0.euclidean();
    double covarianceError0 = (covariance0 - expectedCovariance0).frobenius() /
                              expectedCovariance0.frobenius();
    LOG_DEBUG(<< "mean error 0 = " << meanError0 << ", covariance error 0 = " << covarianceError0);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, meanError0, 0.01);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, covarianceError0, 0.27);

    double meanError1 = (mean1 - expectedMean1).euclidean() / expectedMean0.euclidean();
    double covarianceError1 = (covariance1 - expectedCovariance1).frobenius() /
                              expectedCovariance1.frobenius();
    LOG_DEBUG(<< "mean error 1 = " << meanError1 << ", covariance error 1 = " << covarianceError1);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, meanError1, 0.01);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, covarianceError1, 0.24);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    test::CRandomNumbers rng;

    TDoubleVec coordinates;
    rng.generateUniformSamples(0.0, 400.0, 998, coordinates);
    TVector2Vec points;
    for (std::size_t i = 0; i < coordinates.size(); i += 2) {
        points.push_back(TVector2(&coordinates[i], &coordinates[i + 2]));
    }

    maths::common::CKMeansOnline<TVector2> origKmeans(25, 0.1);
    for (std::size_t i = 0; i < points.size(); ++i) {
        origKmeans.add(points[i]);
    }

    LOG_DEBUG(<< "k-means = " << origKmeans.print());

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origKmeans.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG(<< "original k-means XML = " << origXml);

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::common::CKMeansOnline<TVector2> restoredKmeans(0);
        maths::common::SDistributionRestoreParams params(
            maths_t::E_ContinuousData, 0.1, maths::common::MINIMUM_CLUSTER_SPLIT_FRACTION,
            maths::common::MINIMUM_CLUSTER_SPLIT_COUNT,
            maths::common::MINIMUM_CATEGORY_COUNT);
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
            &maths::common::CKMeansOnline<TVector2>::acceptRestoreTraverser,
            &restoredKmeans, std::cref(params), std::placeholders::_1)));

        LOG_DEBUG(<< "orig checksum = " << origKmeans.checksum()
                  << ", new checksum = " << restoredKmeans.checksum());
        BOOST_REQUIRE_EQUAL(origKmeans.checksum(), restoredKmeans.checksum());

        std::string newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredKmeans.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }
}

BOOST_AUTO_TEST_SUITE_END()
