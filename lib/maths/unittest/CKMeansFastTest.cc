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

#include "CKMeansFastTest.h"

#include <core/CLogger.h>

#include <maths/CKMeansFast.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSphericalCluster.h>

#include <test/CRandomNumbers.h>

using namespace ml;

namespace {

//! \brief Expose internals of k-means for testing.
template<typename POINT>
class CKMeansFastForTest : maths::CKMeansFast<POINT> {
public:
    typedef typename maths::CKMeansFast<POINT>::TBoundingBox TBoundingBox;
    typedef typename maths::CKMeansFast<POINT>::CKdTreeNodeData TKdTreeNodeData;
    typedef typename maths::CKMeansFast<POINT>::SDataPropagator TDataPropagator;
    typedef typename maths::CKMeansFast<POINT>::CCentreFilter TCentreFilter;
    typedef typename maths::CKMeansFast<POINT>::CCentroidComputer TCentroidComputer;
    typedef typename maths::CKMeansFast<POINT>::CClosestPointsCollector TClosestPointsCollector;
};
}

typedef std::vector<double> TDoubleVec;
typedef maths::CVectorNx1<double, 2> TVector2;
typedef std::vector<TVector2> TVector2Vec;
typedef std::vector<TVector2Vec> TVector2VecVec;
typedef maths::CSymmetricMatrixNxN<double, 2> TMatrix2;
typedef std::vector<TMatrix2> TMatrix2Vec;
typedef maths::CVectorNx1<double, 4> TVector4;
typedef std::vector<TVector4> TVector4Vec;
typedef maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator TMean2Accumulator;
typedef std::vector<TMean2Accumulator> TMean2AccumulatorVec;
typedef maths::CBasicStatistics::SSampleMean<TVector4>::TAccumulator TMean4Accumulator;
typedef std::vector<TMean4Accumulator> TMean4AccumulatorVec;

namespace {

template<typename POINT>
struct SKdTreeDataInvariantsChecker {
    typedef typename CKMeansFastForTest<POINT>::TKdTreeNodeData TData;
    typedef typename maths::CBasicStatistics::SSampleMean<POINT>::TAccumulator TMeanAccumulator;
    typedef typename CKMeansFastForTest<POINT>::TBoundingBox TBoundingBox;

    void operator()(const typename maths::CKdTree<POINT, TData>::SNode& node) const {
        TMeanAccumulator centroid;

        TBoundingBox bb(node.s_Point);
        centroid.add(node.s_Point);

        if (node.s_LeftChild) {
            bb.add(node.s_LeftChild->boundingBox());
            centroid += node.s_LeftChild->centroid();
        }
        if (node.s_RightChild) {
            bb.add(node.s_RightChild->boundingBox());
            centroid += node.s_RightChild->centroid();
        }

        CPPUNIT_ASSERT_EQUAL(bb.print(), node.boundingBox().print());
        CPPUNIT_ASSERT_EQUAL(maths::CBasicStatistics::print(centroid),
                             maths::CBasicStatistics::print(node.centroid()));
    }
};

template<typename POINT>
class CCentreFilterChecker {
public:
    typedef std::vector<std::size_t> TSizeVec;
    typedef std::vector<POINT> TPointVec;
    typedef typename CKMeansFastForTest<POINT>::TKdTreeNodeData TData;
    typedef typename CKMeansFastForTest<POINT>::TCentreFilter TCentreFilter;

public:
    CCentreFilterChecker(const TPointVec& centres, std::size_t& numberAdmitted)
        : m_Centres(centres), m_CentreFilter(centres), m_NumberAdmitted(numberAdmitted) {}

    bool operator()(const typename maths::CKdTree<POINT, TData>::SNode& node) const {
        typedef std::pair<double, std::size_t> TDoubleSizePr;

        m_CentreFilter.prune(node.boundingBox());
        const TSizeVec& filtered = m_CentreFilter.filter();
        maths::CBasicStatistics::COrderStatisticsStack<TDoubleSizePr, 2> closest;
        for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
            closest.add(TDoubleSizePr((m_Centres[i] - node.s_Point).euclidean(), i));
        }
        closest.sort();
        if (std::find(filtered.begin(), filtered.end(), closest[0].second) == filtered.end()) {
            LOG_DEBUG("filtered = " << core::CContainerPrinter::print(filtered));
            LOG_DEBUG("closest  = " << closest.print());
            CPPUNIT_ASSERT(false);
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
    typedef typename maths::CBasicStatistics::SSampleMean<POINT>::TAccumulator TMeanAccumlator;

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

void CKMeansFastTest::testDataPropagation(void) {
    LOG_DEBUG("+----------------------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testDataPropagation  |");
    LOG_DEBUG("+----------------------------------------+");

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG("Test " << i);
        TDoubleVec samples;
        rng.generateUniformSamples(-400.0, 400.0, 1000, samples);
        {
            maths::CKdTree<TVector2, CKMeansFastForTest<TVector2>::TKdTreeNodeData> tree;
            TVector2Vec points;
            for (std::size_t j = 0u; j < samples.size(); j += 2) {
                points.push_back(TVector2(&samples[j], &samples[j + 2]));
            }
            tree.build(points);
            tree.postorderDepthFirst(CKMeansFastForTest<TVector2>::TDataPropagator());
            tree.postorderDepthFirst(SKdTreeDataInvariantsChecker<TVector2>());
        }
        {
            maths::CKdTree<TVector4, CKMeansFastForTest<TVector4>::TKdTreeNodeData> tree;
            TVector4Vec points;
            for (std::size_t j = 0u; j < samples.size(); j += 4) {
                points.push_back(TVector4(&samples[j], &samples[j + 4]));
            }
            tree.build(points);
            tree.postorderDepthFirst(CKMeansFastForTest<TVector4>::TDataPropagator());
            tree.postorderDepthFirst(SKdTreeDataInvariantsChecker<TVector4>());
        }
    }
}

void CKMeansFastTest::testFilter(void) {
    LOG_DEBUG("+-------------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testFilter  |");
    LOG_DEBUG("+-------------------------------+");

    // Test that the closest centre to each point is never removed
    // by the centre filter and that we get good speed up in terms
    // of the number of centre point comparisons avoided.

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG("Test " << i);
        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 40, samples2);

        {
            LOG_DEBUG("Vector2");
            maths::CKdTree<TVector2, CKMeansFastForTest<TVector2>::TKdTreeNodeData> tree;

            TVector2Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 2) {
                points.push_back(TVector2(&samples1[j], &samples1[j + 2]));
            }
            tree.build(points);
            TVector2Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 2) {
                centres.push_back(TVector2(&samples2[j], &samples2[j + 2]));
            }
            LOG_DEBUG("  centres = " << core::CContainerPrinter::print(centres));
            tree.postorderDepthFirst(CKMeansFastForTest<TVector2>::TDataPropagator());

            std::size_t numberAdmitted = 0;
            CCentreFilterChecker<TVector2> checker(centres, numberAdmitted);
            tree.preorderDepthFirst(checker);
            double speedup = static_cast<double>(points.size()) *
                             static_cast<double>(centres.size()) /
                             static_cast<double>(numberAdmitted);
            LOG_DEBUG("  speedup = " << speedup);
            CPPUNIT_ASSERT(speedup > 30.0);
        }

        {
            LOG_DEBUG("Vector4");
            maths::CKdTree<TVector4, CKMeansFastForTest<TVector4>::TKdTreeNodeData> tree;

            TVector4Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 4) {
                points.push_back(TVector4(&samples1[j], &samples1[j + 4]));
            }
            tree.build(points);
            TVector4Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 4) {
                centres.push_back(TVector4(&samples2[j], &samples2[j + 4]));
            }
            LOG_DEBUG("  centres = " << core::CContainerPrinter::print(centres));
            tree.postorderDepthFirst(CKMeansFastForTest<TVector4>::TDataPropagator());

            std::size_t numberAdmitted = 0;
            CCentreFilterChecker<TVector4> checker(centres, numberAdmitted);
            tree.preorderDepthFirst(checker);
            double speedup = static_cast<double>(points.size()) *
                             static_cast<double>(centres.size()) /
                             static_cast<double>(numberAdmitted);
            LOG_DEBUG("  speedup = " << speedup);
            CPPUNIT_ASSERT(speedup > 5.5);
        }
    }
}

void CKMeansFastTest::testCentroids(void) {
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testCentroids  |");
    LOG_DEBUG("+----------------------------------+");

    // Check that the centroids computed are the centroids for
    // each cluster, i.e. the centroid of the points closest to
    // each cluster centre.

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG("Test " << i);
        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 20, samples2);

        {
            LOG_DEBUG("Vector2");
            maths::CKdTree<TVector2, CKMeansFastForTest<TVector2>::TKdTreeNodeData> tree;

            TVector2Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 2) {
                points.push_back(TVector2(&samples1[j], &samples1[j + 2]));
            }
            tree.build(points);
            TVector2Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 2) {
                centres.push_back(TVector2(&samples2[j], &samples2[j + 2]));
            }
            tree.postorderDepthFirst(CKMeansFastForTest<TVector2>::TDataPropagator());

            TMean2AccumulatorVec centroids(centres.size());
            CKMeansFastForTest<TVector2>::TCentroidComputer computer(centres, centroids);
            tree.preorderDepthFirst(computer);

            TMean2AccumulatorVec expectedCentroids(centres.size());
            for (std::size_t j = 0u; j < points.size(); ++j) {
                expectedCentroids[closest(centres, points[j]).first].add(points[j]);
            }
            LOG_DEBUG(
                "  expected centroids = " << core::CContainerPrinter::print(expectedCentroids));
            LOG_DEBUG("  centroids          = " << core::CContainerPrinter::print(centroids));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedCentroids),
                                 core::CContainerPrinter::print(centroids));
        }
        {
            LOG_DEBUG("Vector4");
            maths::CKdTree<TVector4, CKMeansFastForTest<TVector4>::TKdTreeNodeData> tree;

            TVector4Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 4) {
                points.push_back(TVector4(&samples1[j], &samples1[j + 4]));
            }
            tree.build(points);
            TVector4Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 4) {
                centres.push_back(TVector4(&samples2[j], &samples2[j + 4]));
            }
            tree.postorderDepthFirst(CKMeansFastForTest<TVector4>::TDataPropagator());

            TMean4AccumulatorVec centroids(centres.size());
            CKMeansFastForTest<TVector4>::TCentroidComputer computer(centres, centroids);
            tree.preorderDepthFirst(computer);

            TMean4AccumulatorVec expectedCentroids(centres.size());
            for (std::size_t j = 0u; j < points.size(); ++j) {
                expectedCentroids[closest(centres, points[j]).first].add(points[j]);
            }
            LOG_DEBUG(
                "  expected centroids = " << core::CContainerPrinter::print(expectedCentroids));
            LOG_DEBUG("  centroids          = " << core::CContainerPrinter::print(centroids));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedCentroids),
                                 core::CContainerPrinter::print(centroids));
        }
    }
}

void CKMeansFastTest::testClosestPoints(void) {
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testClosestPoints  |");
    LOG_DEBUG("+--------------------------------------+");

    // Check the obvious invariant that the closest point to each
    // centre is closer to that centre than any other.

    typedef std::vector<TVector2Vec> TVector2VecVec;
    typedef std::vector<TVector4Vec> TVector4VecVec;

    test::CRandomNumbers rng;

    for (std::size_t i = 1u; i <= 100; ++i) {
        LOG_DEBUG("Test " << i);
        TDoubleVec samples1;
        rng.generateUniformSamples(-400.0, 400.0, 4000, samples1);
        TDoubleVec samples2;
        rng.generateUniformSamples(-500.0, 500.0, 20, samples2);

        {
            maths::CKdTree<TVector2, CKMeansFastForTest<TVector2>::TKdTreeNodeData> tree;

            TVector2Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 2) {
                points.push_back(TVector2(&samples1[j], &samples1[j + 2]));
            }
            tree.build(points);
            TVector2Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 2) {
                centres.push_back(TVector2(&samples2[j], &samples2[j + 2]));
            }
            tree.postorderDepthFirst(CKMeansFastForTest<TVector2>::TDataPropagator());

            TVector2VecVec closestPoints;
            CKMeansFastForTest<TVector2>::TClosestPointsCollector collector(points.size(),
                                                                            centres,
                                                                            closestPoints);
            tree.postorderDepthFirst(collector);

            for (std::size_t j = 0u; j < closestPoints.size(); ++j) {
                for (std::size_t k = 0u; k < closestPoints[j].size(); ++k) {
                    CPPUNIT_ASSERT_EQUAL(closest(centres, closestPoints[j][k]).first, j);
                }
            }
        }
        {
            maths::CKdTree<TVector4, CKMeansFastForTest<TVector4>::TKdTreeNodeData> tree;

            TVector4Vec points;
            for (std::size_t j = 0u; j < samples1.size(); j += 4) {
                points.push_back(TVector4(&samples1[j], &samples1[j + 4]));
            }
            tree.build(points);
            TVector4Vec centres;
            for (std::size_t j = 0u; j < samples2.size(); j += 4) {
                centres.push_back(TVector4(&samples2[j], &samples2[j + 4]));
            }
            tree.postorderDepthFirst(CKMeansFastForTest<TVector4>::TDataPropagator());

            TVector4VecVec closestPoints;
            CKMeansFastForTest<TVector4>::TClosestPointsCollector collector(points.size(),
                                                                            centres,
                                                                            closestPoints);
            tree.postorderDepthFirst(collector);

            for (std::size_t j = 0u; j < closestPoints.size(); ++j) {
                for (std::size_t k = 0u; k < closestPoints[j].size(); ++k) {
                    CPPUNIT_ASSERT_EQUAL(closest(centres, closestPoints[j][k]).first, j);
                }
            }
        }
    }
}

void CKMeansFastTest::testRun(void) {
    LOG_DEBUG("+----------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testRun  |");
    LOG_DEBUG("+----------------------------+");

    // Test k-means correctly identifies two separated uniform
    // random clusters in the data.

    test::CRandomNumbers rng;

    for (std::size_t t = 1u; t <= 100; ++t) {
        LOG_DEBUG("Test " << t);

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

            maths::CKMeansFast<TVector2> kmeansFast;
            TVector2Vec pointsCopy(points);
            kmeansFast.setPoints(pointsCopy);
            TVector2Vec centresCopy(centres);
            kmeansFast.setCentres(centresCopy);

            bool fastConverged = kmeansFast.run(25);
            bool converged = kmeans(points, 25, centres);

            LOG_DEBUG("converged      = " << converged);
            LOG_DEBUG("fast converged = " << fastConverged);
            LOG_DEBUG("centres      = " << core::CContainerPrinter::print(centres));
            LOG_DEBUG("fast centres = " << core::CContainerPrinter::print(kmeansFast.centres()));
            CPPUNIT_ASSERT_EQUAL(converged, fastConverged);
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(centres),
                                 core::CContainerPrinter::print(kmeansFast.centres()));
        }
    }
}

void CKMeansFastTest::testRunWithSphericalClusters(void) {
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testRunWithSphericalClusters  |");
    LOG_DEBUG("+-------------------------------------------------+");

    // The idea of this test is simply to check that we get the
    // same result working with clusters of points or their
    // spherical cluster representation.

    typedef maths::CSphericalCluster<TVector2>::Type TSphericalCluster2;
    typedef std::vector<TSphericalCluster2> TSphericalCluster2Vec;
    typedef maths::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator TMeanVar2Accumulator;

    double means[][2] = {{1.0, 1.0},
                         {2.0, 1.5},
                         {1.5, 1.5},
                         {1.9, 1.5},
                         {1.0, 1.5},
                         {10.0, 15.0},
                         {12.0, 13.5},
                         {12.0, 11.5},
                         {14.0, 10.5}};
    std::size_t counts[] = {10, 15, 5, 8, 17, 10, 11, 8, 12};
    double lowerTriangle[] = {1.0, 0.0, 1.0};

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 50; ++t) {
        LOG_DEBUG("*** trial = " << t + 1 << " ***");

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
        LOG_DEBUG("centres = " << core::CContainerPrinter::print(centresClusters));

        maths::CKMeansFast<TVector2> kmeansPoints;
        kmeansPoints.setPoints(points);
        kmeansPoints.setCentres(centresPoints);
        kmeansPoints.run(20);

        maths::CKMeansFast<TSphericalCluster2> kmeansClusters;
        kmeansClusters.setPoints(clusters);
        kmeansClusters.setCentres(centresClusters);
        kmeansClusters.run(20);

        TVector2Vec kmeansPointsCentres = kmeansPoints.centres();
        TSphericalCluster2Vec kmeansClustersCentres_ = kmeansClusters.centres();
        TVector2Vec kmeansClustersCentres(kmeansClustersCentres_.begin(),
                                          kmeansClustersCentres_.end());
        std::sort(kmeansPointsCentres.begin(), kmeansPointsCentres.end());
        std::sort(kmeansClustersCentres.begin(), kmeansClustersCentres.end());

        LOG_DEBUG("k-means points   = " << core::CContainerPrinter::print(kmeansPointsCentres));
        LOG_DEBUG("k-means clusters = " << core::CContainerPrinter::print(kmeansClustersCentres));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(kmeansPointsCentres),
                             core::CContainerPrinter::print(kmeansClustersCentres));
    }
}

void CKMeansFastTest::testPlusPlus(void) {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CKMeansFastTest::testPlusPlus  |");
    LOG_DEBUG("+---------------------------------+");

    // Test the k-means++ sampling scheme always samples all the
    // clusters present in the data and generally results in lower
    // square residuals of the points from the cluster centres.

    typedef std::vector<std::size_t> TSizeVec;
    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
    typedef TVector2Vec::const_iterator TVector2VecCItr;

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
        LOG_TRACE("# points = " << flatPoints.size());

        TVector2Vec randomCentres;
        TSizeVec random;
        rng.generateUniformSamples(0, flatPoints.size(), k, random);
        LOG_DEBUG("random = " << core::CContainerPrinter::print(random));
        for (std::size_t i = 0u; i < k; ++i) {
            randomCentres.push_back(flatPoints[random[i]]);
        }

        TVector2Vec plusPlusCentres;
        maths::CPRNG::CXorOShiro128Plus rng_;
        maths::CKMeansPlusPlusInitialization<TVector2, maths::CPRNG::CXorOShiro128Plus>
            kmeansPlusPlus(rng_);
        kmeansPlusPlus.run(flatPoints, k, plusPlusCentres);

        TSizeVec sampledClusters;
        for (std::size_t i = 0u; i < plusPlusCentres.size(); ++i) {
            std::size_t j = 0u;
            for (/**/; j < points.size(); ++j) {
                TVector2VecCItr next =
                    std::lower_bound(points[j].begin(), points[j].end(), plusPlusCentres[i]);
                if (next != points[j].end() && *next == plusPlusCentres[i]) {
                    break;
                }
            }
            sampledClusters.push_back(j);
        }
        std::sort(sampledClusters.begin(), sampledClusters.end());
        sampledClusters.erase(std::unique(sampledClusters.begin(), sampledClusters.end()),
                              sampledClusters.end());
        CPPUNIT_ASSERT(sampledClusters.size() >= 2);
        numberClustersSampled.add(static_cast<double>(sampledClusters.size()));

        maths::CKMeansFast<TVector2> kmeans;
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

        LOG_DEBUG("S.S.R. random    = " << ssrRandom);
        LOG_DEBUG("S.S.R. plus plus = " << ssrPlusPlus);

        minSSRRatio = std::min(minSSRRatio, ssrPlusPlus / ssrRandom);
        meanSSRRatio.add(ssrPlusPlus / ssrRandom);
        maxSSRRatio = std::max(maxSSRRatio, ssrPlusPlus / ssrRandom);
    }

    LOG_DEBUG("# clusters sampled = " << maths::CBasicStatistics::mean(numberClustersSampled));
    LOG_DEBUG("min ratio  = " << minSSRRatio);
    LOG_DEBUG("mean ratio = " << maths::CBasicStatistics::mean(meanSSRRatio));
    LOG_DEBUG("max ratio  = " << maxSSRRatio);

    CPPUNIT_ASSERT(minSSRRatio < 0.14);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanSSRRatio) < 0.9);
    CPPUNIT_ASSERT(maxSSRRatio < 9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, maths::CBasicStatistics::mean(numberClustersSampled), 0.3);
}

CppUnit::Test* CKMeansFastTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CKMeansFastTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testDataPropagation",
                                                 &CKMeansFastTest::testDataPropagation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testFilter",
                                                                   &CKMeansFastTest::testFilter));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testCentroids",
                                                 &CKMeansFastTest::testCentroids));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testClosestPoints",
                                                 &CKMeansFastTest::testClosestPoints));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testRun",
                                                                   &CKMeansFastTest::testRun));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testRunWithSphericalClusters",
                                                 &CKMeansFastTest::testRunWithSphericalClusters));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKMeansFastTest>("CKMeansFastTest::testPlusPlus",
                                                                   &CKMeansFastTest::testPlusPlus));

    return suiteOfTests;
}
