/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CKMeans.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSphericalCluster.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CKMeansTest)

using namespace ml;

namespace {

//! \brief Expose internals of k-means for testing.
template<typename POINT>
class CKMeansForTest : maths::CKMeans<POINT> {
public:
    using TBoundingBox = typename maths::CKMeans<POINT>::TBoundingBox;
    using TKdTreeNodeData = typename maths::CKMeans<POINT>::CKdTreeNodeData;
    using TDataPropagator = typename maths::CKMeans<POINT>::SDataPropagator;
    using TCentreFilter = typename maths::CKMeans<POINT>::CCentreFilter;
    using TCentroidComputer = typename maths::CKMeans<POINT>::CCentroidComputer;
    using TClosestPointsCollector = typename maths::CKMeans<POINT>::CClosestPointsCollector;
};
}

using TDoubleVec = std::vector<double>;
using TVector2 = maths::CVectorNx1<double, 2>;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecVec = std::vector<TVector2Vec>;
using TMatrix2 = maths::CSymmetricMatrixNxN<double, 2>;
using TMatrix2Vec = std::vector<TMatrix2>;
using TVector4 = maths::CVectorNx1<double, 4>;
using TVector4Vec = std::vector<TVector4>;
using TMean2Accumulator = maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator;
using TMean2AccumulatorVec = std::vector<TMean2Accumulator>;
using TMean4Accumulator = maths::CBasicStatistics::SSampleMean<TVector4>::TAccumulator;
using TMean4AccumulatorVec = std::vector<TMean4Accumulator>;

namespace {

template<typename POINT>
struct SKdTreeDataInvariantsChecker {
    using TData = typename CKMeansForTest<POINT>::TKdTreeNodeData;
    using TMeanAccumulator = typename maths::CBasicStatistics::SSampleMean<POINT>::TAccumulator;
    using TBoundingBox = typename CKMeansForTest<POINT>::TBoundingBox;

    void operator()(const typename maths::CKdTree<POINT, TData>::SNode& node) const {
        TMeanAccumulator centroid;

        TBoundingBox bb(node.s_Point);
        centroid.add(node.s_Point);

        if (node.s_LeftChild) {
            bb.add(node.s_LeftChild->boundingBox());
            centroid += *node.s_LeftChild->centroid();
        }
        if (node.s_RightChild) {
            bb.add(node.s_RightChild->boundingBox());
            centroid += *node.s_RightChild->centroid();
        }

        BOOST_CHECK_EQUAL(bb.print(), node.boundingBox().print());
        BOOST_CHECK_EQUAL(maths::CBasicStatistics::print(centroid),
                          maths::CBasicStatistics::print(*node.centroid()));
    }
};

template<typename POINT>
class CCentreFilterChecker {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TPointVec = std::vector<POINT>;
    using TData = typename CKMeansForTest<POINT>::TKdTreeNodeData;
    using TCentreFilter = typename CKMeansForTest<POINT>::TCentreFilter;

public:
    CCentreFilterChecker(const TPointVec& centres, std::size_t& numberAdmitted)
        : m_Centres(centres), m_CentreFilter(centres),
          m_NumberAdmitted(numberAdmitted) {}

    bool operator()(const typename maths::CKdTree<POINT, TData>::SNode& node) const {
        using TDoubleSizePr = std::pair<double, std::size_t>;

        m_CentreFilter.prune(node.boundingBox());
        const TSizeVec& filtered = m_CentreFilter.filter();
        maths::CBasicStatistics::COrderStatisticsStack<TDoubleSizePr, 2> closest;
        for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
            closest.add(TDoubleSizePr((m_Centres[i] - node.s_Point).euclidean(), i));
        }
        closest.sort();
        if (std::find(filtered.begin(), filtered.end(), closest[0].second) ==
            filtered.end()) {
            LOG_DEBUG(<< "filtered = " << core::CContainerPrinter::print(filtered));
            LOG_DEBUG(<< "closest  = " << closest.print());
            BOOST_TEST(false);
        }
        if (filtered.size() > 1) {
            m_NumberAdmitted += filtered.size();
        }
        return true;
    }

private:
    TPointVec m_Centres;
    mutable TCentreFilter m_CentreFilter;
    std::size_t& m_NumberAdmitted;
};

template<typename POINT>
std::pair<std::size_t, double> closest(const std::vector<POINT>& y, const POINT& x) {
    std::size_t closest = 0u;
    double dmin = (x - y[0]).euclidean();
    for (std::size_t i = 1u; i < y.size(); ++i) {
        double di = (x - y[i]).euclidean();
        if (di < dmin) {
            closest = i;
            dmin = di;
        }
    }
    return std::pair<std::size_t, double>(closest, dmin);
}

template<typename POINT>
bool kmeans(const std::vector<POINT>& points, std::size_t iterations, std::vector<POINT>& centres) {
    using TMeanAccumlator = typename maths::CBasicStatistics::SSampleMean<POINT>::TAccumulator;

    std::vector<TMeanAccumlator> centroids;
    for (std::size_t i = 0u; i < iterations; ++i) {
        centroids.clear();
        centroids.resize(centres.size());

        for (std::size_t j = 0u; j < points.size(); ++j) {
            std::size_t centre = closest(centres, points[j]).first;
            centroids[centre].add(points[j]);
        }

        bool converged = true;
        for (std::size_t j = 0u; j < centres.size(); ++j) {
            if (maths::CBasicStatistics::mean(centroids[j]) != centres[j]) {
                centres[j] = maths::CBasicStatistics::mean(centroids[j]);
                converged = false;
            }
        }

        if (converged) {
            return true;
        }
    }

    return false;
}

double square(double x) {
    return x * x;
}

double sumSquareResiduals(const TVector2VecVec& points) {
    double result = 0.0;
    for (std::size_t i = 0u; i < points.size(); ++i) {
        TMean2Accumulator m_;
        m_.add(points[i]);
        TVector2 m = maths::CBasicStatistics::mean(m_);
        for (std::size_t j = 0u; j < points[i].size(); ++j) {
            result += square((points[i][j] - m).euclidean());
        }
    }
    return result;
}
}

BOOST_AUTO_TEST_CASE(testDataPropagation) {
    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG(<< "Test " << i);
        TDoubleVec samples;
        rng.generateUniformSamples(-400.0, 400.0, 1000, samples);
        {
            maths::CKdTree<TVector2, CKMeansForTest<TVector2>::TKdTreeNodeData> tree;
            TVector2Vec points;
            for (std::size_t j = 0u; j < samples.size(); j += 2) {
                points.push_back(TVector2(&samples[j], &samples[j + 2]));
            }
            tree.build(points);
            tree.postorderDepthFirst(CKMeansForTest<TVector2>::TDataPropagator());
            tree.postorderDepthFirst(SKdTreeDataInvariantsChecker<TVector2>());
        }
        {
            maths::CKdTree<TVector4, CKMeansForTest<TVector4>::TKdTreeNodeData> tree;
            TVector4Vec points;
            for (std::size_t j = 0u; j < samples.size(); j += 4) {
                points.push_back(TVector4(&samples[j], &samples[j + 4]));
            }
            tree.build(points);
            tree.postorderDepthFirst(CKMeansForTest<TVector4>::TDataPropagator());
            tree.postorderDepthFirst(SKdTreeDataInvariantsChecker<TVector4>());
        }
    }
}

BOOST_AUTO_TEST_CASE(testFilter) {
    // Test that the closest centre to each point is never removed
    // by the centre filter and that we get good speed up in terms
    // of the number of centre point comparisons avoided.

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG(<< "Test " << i);
        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 40, samples2);

        {
            LOG_DEBUG(<< "Vector2");
            maths::CKdTree<TVector2, CKMeansForTest<TVector2>::TKdTreeNodeData> tree;

            TVector2Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 2) {
                points.push_back(TVector2(&samples1[j], &samples1[j + 2]));
            }
            tree.build(points);
            TVector2Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 2) {
                centres.push_back(TVector2(&samples2[j], &samples2[j + 2]));
            }
            LOG_DEBUG(<< "  centres = " << core::CContainerPrinter::print(centres));
            tree.postorderDepthFirst(CKMeansForTest<TVector2>::TDataPropagator());

            std::size_t numberAdmitted = 0;
            CCentreFilterChecker<TVector2> checker(centres, numberAdmitted);
            tree.preorderDepthFirst(checker);
            double speedup = static_cast<double>(points.size()) *
                             static_cast<double>(centres.size()) /
                             static_cast<double>(numberAdmitted);
            LOG_DEBUG(<< "  speedup = " << speedup);
            BOOST_TEST(speedup > 30.0);
        }

        {
            LOG_DEBUG(<< "Vector4");
            maths::CKdTree<TVector4, CKMeansForTest<TVector4>::TKdTreeNodeData> tree;

            TVector4Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 4) {
                points.push_back(TVector4(&samples1[j], &samples1[j + 4]));
            }
            tree.build(points);
            TVector4Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 4) {
                centres.push_back(TVector4(&samples2[j], &samples2[j + 4]));
            }
            LOG_DEBUG(<< "  centres = " << core::CContainerPrinter::print(centres));
            tree.postorderDepthFirst(CKMeansForTest<TVector4>::TDataPropagator());

            std::size_t numberAdmitted = 0;
            CCentreFilterChecker<TVector4> checker(centres, numberAdmitted);
            tree.preorderDepthFirst(checker);
            double speedup = static_cast<double>(points.size()) *
                             static_cast<double>(centres.size()) /
                             static_cast<double>(numberAdmitted);
            LOG_DEBUG(<< "  speedup = " << speedup);
            BOOST_TEST(speedup > 5.5);
        }
    }
}

BOOST_AUTO_TEST_CASE(testCentroids) {
    // Check that the centroids computed are the centroids for
    // each cluster, i.e. the centroid of the points closest to
    // each cluster centre.

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG(<< "Test " << i);
        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 20, samples2);

        {
            LOG_DEBUG(<< "Vector2");
            maths::CKdTree<TVector2, CKMeansForTest<TVector2>::TKdTreeNodeData> tree;

            TVector2Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 2) {
                points.push_back(TVector2(&samples1[j], &samples1[j + 2]));
            }
            tree.build(points);
            TVector2Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 2) {
                centres.push_back(TVector2(&samples2[j], &samples2[j + 2]));
            }
            tree.postorderDepthFirst(CKMeansForTest<TVector2>::TDataPropagator());

            TMean2AccumulatorVec centroids(centres.size());
            CKMeansForTest<TVector2>::TCentroidComputer computer(centres, centroids);
            tree.preorderDepthFirst(computer);

            TMean2AccumulatorVec expectedCentroids(centres.size());
            for (std::size_t j = 0u; j < points.size(); ++j) {
                expectedCentroids[closest(centres, points[j]).first].add(points[j]);
            }
            LOG_DEBUG(<< "  expected centroids = "
                      << core::CContainerPrinter::print(expectedCentroids));
            LOG_DEBUG(<< "  centroids          = "
                      << core::CContainerPrinter::print(centroids));
            BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expectedCentroids),
                              core::CContainerPrinter::print(centroids));
        }
        {
            LOG_DEBUG(<< "Vector4");
            maths::CKdTree<TVector4, CKMeansForTest<TVector4>::TKdTreeNodeData> tree;

            TVector4Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 4) {
                points.push_back(TVector4(&samples1[j], &samples1[j + 4]));
            }
            tree.build(points);
            TVector4Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 4) {
                centres.push_back(TVector4(&samples2[j], &samples2[j + 4]));
            }
            tree.postorderDepthFirst(CKMeansForTest<TVector4>::TDataPropagator());

            TMean4AccumulatorVec centroids(centres.size());
            CKMeansForTest<TVector4>::TCentroidComputer computer(centres, centroids);
            tree.preorderDepthFirst(computer);

            TMean4AccumulatorVec expectedCentroids(centres.size());
            for (std::size_t j = 0u; j < points.size(); ++j) {
                expectedCentroids[closest(centres, points[j]).first].add(points[j]);
            }
            LOG_DEBUG(<< "  expected centroids = "
                      << core::CContainerPrinter::print(expectedCentroids));
            LOG_DEBUG(<< "  centroids          = "
                      << core::CContainerPrinter::print(centroids));
            BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expectedCentroids),
                              core::CContainerPrinter::print(centroids));
        }
    }
}

BOOST_AUTO_TEST_CASE(testClosestPoints) {
    // Check the obvious invariant that the closest point to each
    // centre is closer to that centre than any other.

    using TVector4VecVec = std::vector<TVector4Vec>;

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG(<< "Test " << i);
        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 20, samples2);

        {
            maths::CKdTree<TVector2, CKMeansForTest<TVector2>::TKdTreeNodeData> tree;

            TVector2Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 2) {
                points.push_back(TVector2(&samples1[j], &samples1[j + 2]));
            }
            tree.build(points);
            TVector2Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 2) {
                centres.push_back(TVector2(&samples2[j], &samples2[j + 2]));
            }
            tree.postorderDepthFirst(CKMeansForTest<TVector2>::TDataPropagator());

            TVector2VecVec closestPoints;
            CKMeansForTest<TVector2>::TClosestPointsCollector collector(
                points.size(), centres, closestPoints);
            tree.postorderDepthFirst(collector);

            for (std::size_t j = 0u; j < closestPoints.size(); ++j) {
                for (std::size_t k = 0u; k < closestPoints[j].size(); ++k) {
                    BOOST_CHECK_EQUAL(closest(centres, closestPoints[j][k]).first, j);
                }
            }
        }
        {
            maths::CKdTree<TVector4, CKMeansForTest<TVector4>::TKdTreeNodeData> tree;

            TVector4Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 4) {
                points.push_back(TVector4(&samples1[j], &samples1[j + 4]));
            }
            tree.build(points);
            TVector4Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 4) {
                centres.push_back(TVector4(&samples2[j], &samples2[j + 4]));
            }
            tree.postorderDepthFirst(CKMeansForTest<TVector4>::TDataPropagator());

            TVector4VecVec closestPoints;
            CKMeansForTest<TVector4>::TClosestPointsCollector collector(
                points.size(), centres, closestPoints);
            tree.postorderDepthFirst(collector);

            for (std::size_t j = 0u; j < closestPoints.size(); ++j) {
                for (std::size_t k = 0u; k < closestPoints[j].size(); ++k) {
                    BOOST_CHECK_EQUAL(closest(centres, closestPoints[j][k]).first, j);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testRun) {
    // Test k-means correctly identifies two separated uniform
    // random clusters in the data.

    test::CRandomNumbers rng;

    for (std::size_t t = 1u; t <= 100; ++t) {
        LOG_DEBUG(<< "Test " << t);

        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 20, samples2);

        {
            TVector2Vec points;
            for (std::size_t i = 0u; i < samples1.size(); i += 2) {
                points.push_back(TVector2(&samples1[i], &samples1[i + 2]));
            }
            TVector2Vec centres;
            for (std::size_t i = 0u; i < samples2.size(); i += 2) {
                centres.push_back(TVector2(&samples2[i], &samples2[i + 2]));
            }

            maths::CKMeans<TVector2> kmeansFast;
            TVector2Vec pointsCopy(points);
            kmeansFast.setPoints(pointsCopy);
            TVector2Vec centresCopy(centres);
            kmeansFast.setCentres(centresCopy);

            bool fastConverged = kmeansFast.run(25);
            bool converged = kmeans(points, 25, centres);

            LOG_DEBUG(<< "converged      = " << converged);
            LOG_DEBUG(<< "fast converged = " << fastConverged);
            LOG_DEBUG(<< "centres      = " << core::CContainerPrinter::print(centres));
            LOG_DEBUG(<< "fast centres = "
                      << core::CContainerPrinter::print(kmeansFast.centres()));
            BOOST_CHECK_EQUAL(converged, fastConverged);
            BOOST_CHECK_EQUAL(core::CContainerPrinter::print(centres),
                              core::CContainerPrinter::print(kmeansFast.centres()));
        }
    }
}

BOOST_AUTO_TEST_CASE(testRunWithSphericalClusters) {
    // The idea of this test is simply to check that we get the
    // same result working with clusters of points or their
    // spherical cluster representation.

    using TSphericalCluster2 = maths::CSphericalCluster<TVector2>::Type;
    using TSphericalCluster2Vec = std::vector<TSphericalCluster2>;
    using TMeanVar2Accumulator = maths::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator;

    double means[][2] = {{1.0, 1.0},   {2.0, 1.5},   {1.5, 1.5},
                         {1.9, 1.5},   {1.0, 1.5},   {10.0, 15.0},
                         {12.0, 13.5}, {12.0, 11.5}, {14.0, 10.5}};
    std::size_t counts[] = {10, 15, 5, 8, 17, 10, 11, 8, 12};
    double lowerTriangle[] = {1.0, 0.0, 1.0};

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 50; ++t) {
        LOG_DEBUG(<< "*** trial = " << t + 1 << " ***");

        TVector2Vec points;
        TSphericalCluster2Vec clusters;

        for (std::size_t i = 0u; i < boost::size(means); ++i) {
            TVector2Vec pointsi;
            TVector2 mean(&means[i][0], &means[i][2]);
            TMatrix2 covariances(&lowerTriangle[0], &lowerTriangle[3]);
            maths::CSampling::multivariateNormalSample(mean, covariances, counts[i], pointsi);
            points.insert(points.end(), pointsi.begin(), pointsi.end());
            TMeanVar2Accumulator moments;
            moments.add(pointsi);
            double n = maths::CBasicStatistics::count(moments);
            TVector2 m = maths::CBasicStatistics::mean(moments);
            TVector2 v = maths::CBasicStatistics::variance(moments);
            TSphericalCluster2::TAnnotation countAndVariance(n, (v(0) + v(1)) / 2.0);
            TSphericalCluster2 cluster(m, countAndVariance);
            clusters.push_back(cluster);
        }

        TDoubleVec coordinates;
        rng.generateUniformSamples(0.0, 15.0, 4, coordinates);
        TVector2Vec centresPoints;
        centresPoints.push_back(TVector2(&coordinates[0], &coordinates[2]));
        centresPoints.push_back(TVector2(&coordinates[2], &coordinates[4]));
        TSphericalCluster2Vec centresClusters;
        centresClusters.push_back(TVector2(&coordinates[0], &coordinates[2]));
        centresClusters.push_back(TVector2(&coordinates[2], &coordinates[4]));
        LOG_DEBUG(<< "centres = " << core::CContainerPrinter::print(centresClusters));

        maths::CKMeans<TVector2> kmeansPoints;
        kmeansPoints.setPoints(points);
        kmeansPoints.setCentres(centresPoints);
        kmeansPoints.run(20);

        maths::CKMeans<TSphericalCluster2> kmeansClusters;
        kmeansClusters.setPoints(clusters);
        kmeansClusters.setCentres(centresClusters);
        kmeansClusters.run(20);

        TVector2Vec kmeansPointsCentres = kmeansPoints.centres();
        TSphericalCluster2Vec kmeansClustersCentres_ = kmeansClusters.centres();
        TVector2Vec kmeansClustersCentres(kmeansClustersCentres_.begin(),
                                          kmeansClustersCentres_.end());
        std::sort(kmeansPointsCentres.begin(), kmeansPointsCentres.end());
        std::sort(kmeansClustersCentres.begin(), kmeansClustersCentres.end());

        LOG_DEBUG(<< "k-means points   = "
                  << core::CContainerPrinter::print(kmeansPointsCentres));
        LOG_DEBUG(<< "k-means clusters = "
                  << core::CContainerPrinter::print(kmeansClustersCentres));
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(kmeansPointsCentres),
                          core::CContainerPrinter::print(kmeansClustersCentres));
    }
}

BOOST_AUTO_TEST_CASE(testPlusPlus) {
    // Test the k-means++ sampling scheme always samples all the
    // clusters present in the data and generally results in lower
    // square residuals of the points from the cluster centres.

    using TSizeVec = std::vector<std::size_t>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TVector2VecCItr = TVector2Vec::const_iterator;

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    std::size_t k = 5u;

    TMeanAccumulator numberClustersSampled;
    double minSSRRatio = std::numeric_limits<double>::max();
    TMeanAccumulator meanSSRRatio;
    double maxSSRRatio = 0.0;

    for (std::size_t t = 0u; t < 100; ++t) {
        TSizeVec sizes;
        sizes.push_back(400);
        sizes.push_back(300);
        sizes.push_back(500);
        sizes.push_back(800);

        TVector2Vec means;
        TMatrix2Vec covariances;
        TVector2VecVec points;
        rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

        TVector2Vec flatPoints;
        for (std::size_t i = 0u; i < points.size(); ++i) {
            flatPoints.insert(flatPoints.end(), points[i].begin(), points[i].end());
            std::sort(points[i].begin(), points[i].end());
        }
        LOG_TRACE(<< "# points = " << flatPoints.size());

        TVector2Vec randomCentres;
        TSizeVec random;
        rng.generateUniformSamples(0, flatPoints.size(), k, random);
        LOG_DEBUG(<< "random = " << core::CContainerPrinter::print(random));
        for (std::size_t i = 0u; i < k; ++i) {
            randomCentres.push_back(flatPoints[random[i]]);
        }

        TVector2Vec plusPlusCentres;
        maths::CPRNG::CXorOShiro128Plus rng_;
        maths::CKMeansPlusPlusInitialization<TVector2, maths::CPRNG::CXorOShiro128Plus> kmeansPlusPlus(
            rng_);
        kmeansPlusPlus.run(flatPoints, k, plusPlusCentres);

        TSizeVec sampledClusters;
        for (std::size_t i = 0u; i < plusPlusCentres.size(); ++i) {
            std::size_t j = 0u;
            for (/**/; j < points.size(); ++j) {
                TVector2VecCItr next = std::lower_bound(
                    points[j].begin(), points[j].end(), plusPlusCentres[i]);
                if (next != points[j].end() && *next == plusPlusCentres[i]) {
                    break;
                }
            }
            sampledClusters.push_back(j);
        }
        std::sort(sampledClusters.begin(), sampledClusters.end());
        sampledClusters.erase(
            std::unique(sampledClusters.begin(), sampledClusters.end()),
            sampledClusters.end());
        BOOST_TEST(sampledClusters.size() >= 2);
        numberClustersSampled.add(static_cast<double>(sampledClusters.size()));

        maths::CKMeans<TVector2> kmeans;
        kmeans.setPoints(flatPoints);

        double ssrRandom;
        {
            kmeans.setCentres(randomCentres);
            kmeans.run(20);
            TVector2VecVec clusters;
            kmeans.clusters(clusters);
            ssrRandom = sumSquareResiduals(clusters);
        }

        double ssrPlusPlus;
        {
            kmeans.setCentres(plusPlusCentres);
            kmeans.run(20);
            TVector2VecVec clusters;
            kmeans.clusters(clusters);
            ssrPlusPlus = sumSquareResiduals(clusters);
        }

        LOG_DEBUG(<< "S.S.R. random    = " << ssrRandom);
        LOG_DEBUG(<< "S.S.R. plus plus = " << ssrPlusPlus);

        minSSRRatio = std::min(minSSRRatio, ssrPlusPlus / ssrRandom);
        meanSSRRatio.add(ssrPlusPlus / ssrRandom);
        maxSSRRatio = std::max(maxSSRRatio, ssrPlusPlus / ssrRandom);
    }

    LOG_DEBUG(<< "# clusters sampled = "
              << maths::CBasicStatistics::mean(numberClustersSampled));
    LOG_DEBUG(<< "min ratio  = " << minSSRRatio);
    LOG_DEBUG(<< "mean ratio = " << maths::CBasicStatistics::mean(meanSSRRatio));
    LOG_DEBUG(<< "max ratio  = " << maxSSRRatio);

    BOOST_TEST(minSSRRatio < 0.14);
    BOOST_TEST(maths::CBasicStatistics::mean(meanSSRRatio) < 0.9);
    BOOST_TEST(maxSSRRatio < 9.0);
    BOOST_CHECK_CLOSE_ABSOLUTE(4.0, maths::CBasicStatistics::mean(numberClustersSampled), 0.3);
}

BOOST_AUTO_TEST_SUITE_END()
