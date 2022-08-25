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

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraPersist.h>
#include <maths/common/CLinearAlgebraTools.h>
#include <maths/common/CSampling.h>
#include <maths/common/CXMeans.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <cstdint>

using TVector2 = ml::maths::common::CVectorNx1<double, 2>;
using TVector4 = ml::maths::common::CVectorNx1<double, 4>;
using TVector4Vec = std::vector<TVector4>;

// TODO boost test: replace these output operators
// with a more generic way of printing clusters in
// the production code
namespace ml {
namespace maths {
namespace common {

std::ostream& operator<<(std::ostream& strm,
                         const typename CXMeans<TVector2>::CCluster& cluster) {
    return strm << "Cluster{ cost: " << cluster.cost()
                << ", centre: " << cluster.centre().toDelimited()
                << ", points checksum: " << cluster.checksum() << " }";
}

std::ostream& operator<<(std::ostream& strm,
                         const typename CXMeans<TVector4>::CCluster& cluster) {
    return strm << "Cluster{ cost: " << cluster.cost()
                << ", centre: " << cluster.centre().toDelimited()
                << ", points checksum: " << cluster.checksum() << " }";
}

std::ostream& operator<<(std::ostream& strm,
                         const typename CXMeans<TVector4Vec>::CCluster& cluster) {
    strm << "Cluster{ cost: " << cluster.cost() << ", centre: ";
    const char* delim{"("};
    for (const auto& element : cluster.centre()) {
        strm << delim << element.toDelimited() << ')';
        delim = ", (";
    }
    return strm << ", points checksum: " << cluster.checksum() << " }";
}
}
}
}

BOOST_AUTO_TEST_SUITE(CXMeansTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TUInt64Vec = std::vector<uint64_t>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecCItr = TVector2Vec::const_iterator;
using TVector2VecVec = std::vector<TVector2Vec>;
using TMeanVar2Accumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator;
using TMatrix2 = maths::common::CSymmetricMatrixNxN<double, 2>;
using TMatrix2Vec = std::vector<TMatrix2>;
using TMeanVar4Accumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<TVector4>::TAccumulator;
using TMatrix4 = maths::common::CSymmetricMatrixNxN<double, 4>;
using TMatrix4Vec = std::vector<TMatrix4>;

//! \brief Expose internals of x-means for testing.
template<typename POINT, typename COST = maths::common::CSphericalGaussianInfoCriterion<POINT, maths::common::E_BIC>>
class CXMeansForTest : public maths::common::CXMeans<POINT, COST> {
public:
    using TUInt64USet = typename maths::common::CXMeans<POINT, COST>::TUInt64USet;

public:
    CXMeansForTest(std::size_t kmax)
        : maths::common::CXMeans<POINT, COST>(kmax) {}

    void improveParams(std::size_t kmeansIterations) {
        this->maths::common::CXMeans<POINT, COST>::improveParams(kmeansIterations);
    }

    bool improveStructure(std::size_t clusterSeeds, std::size_t kmeansIterations) {
        return this->maths::common::CXMeans<POINT, COST>::improveStructure(
            clusterSeeds, kmeansIterations);
    }

    const TUInt64USet& inactive() const {
        return this->maths::common::CXMeans<POINT, COST>::inactive();
    }
};

template<typename POINT>
double logfSphericalGaussian(const POINT& mean, double variance, const POINT& x) {
    double d = static_cast<double>(x.dimension());
    double r = (x - mean).euclidean();
    return -0.5 * d * std::log(boost::math::double_constants::two_pi * variance) -
           0.5 * r * r / variance;
}

class CEmpiricalKullbackLeibler {
public:
    double value() const {
        return maths::common::CBasicStatistics::mean(m_Divergence) -
               std::log(maths::common::CBasicStatistics::count(m_Divergence));
    }

    template<typename POINT>
    void add(const std::vector<POINT>& points) {
        typename maths::common::CBasicStatistics::SSampleMeanVar<POINT>::TAccumulator moments;
        moments.add(points);
        POINT mean = maths::common::CBasicStatistics::mean(moments);
        POINT variances = maths::common::CBasicStatistics::variance(moments);

        maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator variance_;
        for (std::size_t i = 0; i < variances.dimension(); ++i) {
            variance_.add(variances(i));
        }
        double variance = maths::common::CBasicStatistics::mean(variance_);
        for (std::size_t i = 0; i < points.size(); ++i) {
            m_Divergence.add(-logfSphericalGaussian(mean, variance, points[i]));
        }
    }

private:
    maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator m_Divergence;
};

void computePurities(const TSizeVecVec& clusters, TDoubleVec& purities) {
    purities.clear();
    purities.resize(clusters.size());

    TSizeVec counts;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
        counts.clear();
        for (std::size_t j = 0; j < clusters[i].size(); ++j) {
            counts.resize(std::max(counts.size(), clusters[i][j] + 1));
            ++counts[clusters[i][j]];
        }
        purities[i] =
            static_cast<double>(*std::max_element(counts.begin(), counts.end())) /
            static_cast<double>(clusters[i].size());
    }
}
}

BOOST_AUTO_TEST_CASE(testCluster) {
    // Test basic accessors and checksum functionality of cluster.

    maths::common::CSampling::seed();

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(-100.0, 400.0, 800, samples);

    for (std::size_t t = 0; t < 100; ++t) {
        LOG_DEBUG(<< "Test " << t);
        {
            maths::common::CXMeans<TVector2>::CCluster cluster1;
            maths::common::CXMeans<TVector2>::CCluster cluster2;

            BOOST_REQUIRE_EQUAL(0, cluster1.size());
            BOOST_REQUIRE_EQUAL(0, cluster2.size());

            TVector2Vec points;
            for (std::size_t i = 0; i < samples.size(); i += 2) {
                points.push_back(TVector2(&samples[i], &samples[i + 2]));
            }
            TVector2Vec pointsCopy(points);
            cluster1.points(pointsCopy);
            rng.random_shuffle(points.begin(), points.end());
            cluster2.points(points);

            BOOST_REQUIRE_EQUAL(samples.size() / 2, cluster1.size());
            BOOST_REQUIRE_EQUAL(samples.size() / 2, cluster2.size());
            BOOST_REQUIRE_EQUAL(cluster1.checksum(), cluster2.checksum());
            BOOST_TEST_REQUIRE(cluster1 == cluster2);
            BOOST_TEST_REQUIRE(!(cluster1 < cluster2));
            BOOST_TEST_REQUIRE(!(cluster2 < cluster1));
        }
        {
            maths::common::CXMeans<TVector4>::CCluster cluster1;
            maths::common::CXMeans<TVector4>::CCluster cluster2;

            BOOST_REQUIRE_EQUAL(0, cluster1.size());
            BOOST_REQUIRE_EQUAL(0, cluster2.size());

            TVector4Vec points;
            for (std::size_t i = 0; i < samples.size(); i += 4) {
                points.push_back(TVector4(&samples[i], &samples[i + 4]));
            }
            TVector4Vec pointsCopy(points);
            cluster1.points(pointsCopy);
            rng.random_shuffle(points.begin(), points.end());
            cluster2.points(points);

            BOOST_REQUIRE_EQUAL(samples.size() / 4, cluster1.size());
            BOOST_REQUIRE_EQUAL(samples.size() / 4, cluster2.size());
            BOOST_REQUIRE_EQUAL(cluster1.checksum(), cluster2.checksum());
            BOOST_TEST_REQUIRE(cluster1 == cluster2);
            BOOST_TEST_REQUIRE(!(cluster1 < cluster2));
            BOOST_TEST_REQUIRE(!(cluster2 < cluster1));
        }
    }
}

BOOST_AUTO_TEST_CASE(testImproveStructure) {
    // Test improve structure finds an obvious split of the data.

    maths::common::CSampling::seed();

    double means[][2] = {{10.0, 20.0}, {50.0, 30.0}};
    double covariances[][3] = {{10.0, -3.0, 15.0}, {20.0, 2.0, 5.0}};

    TMeanAccumulator meanError;

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "Test " << t);

        TVector2Vec points;
        for (std::size_t i = 0; i < 2; ++i) {
            TVector2 mean(&means[i][0], &means[i][2]);
            TMatrix2 covariance(&covariances[i][0], &covariances[i][3]);
            TVector2Vec cluster;
            maths::common::CSampling::multivariateNormalSample(mean, covariance, 500, cluster);
            points.insert(points.end(), cluster.begin(), cluster.end());
        }

        CXMeansForTest<TVector2> xmeans(5);
        xmeans.setPoints(points);
        xmeans.improveStructure(2, 5);

        TVector2Vec clusters;
        TUInt64Vec oldChecksums;
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            clusters.push_back(xmeans.clusters()[i].centre());
            oldChecksums.push_back(xmeans.clusters()[i].checksum());
        }
        std::sort(clusters.begin(), clusters.end());
        std::sort(oldChecksums.begin(), oldChecksums.end());
        LOG_DEBUG(<< "centres = " << clusters);

        for (std::size_t i = 0; i < clusters.size(); ++i) {
            TVector2 mean(&means[i][0], &means[i][2]);
            double error = (clusters[i] - mean).euclidean();
            BOOST_TEST_REQUIRE(error < 0.75);
            meanError.add(error);
        }

        // Check that we've marked any clusters which haven't changed
        // as inactive.
        xmeans.improveStructure(2, 5);
        TUInt64Vec newChecksums;
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            clusters.push_back(xmeans.clusters()[i].centre());
            newChecksums.push_back(xmeans.clusters()[i].checksum());
        }
        std::sort(newChecksums.begin(), newChecksums.end());

        TUInt64Vec inactive;
        std::set_intersection(oldChecksums.begin(), oldChecksums.end(),
                              newChecksums.begin(), newChecksums.end(),
                              std::back_inserter(inactive));
        LOG_DEBUG(<< "inactive = " << inactive);
        for (std::size_t i = 0; i < inactive.size(); ++i) {
            BOOST_TEST_REQUIRE(xmeans.inactive().count(inactive[i]) > 0);
        }
    }

    LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(meanError));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < 0.25);
}

BOOST_AUTO_TEST_CASE(testImproveParams) {
    // Test that improve params is equivalent to a round of k-means
    // on current state cluster centres.

    maths::common::CSampling::seed();

    double means[][2] = {{10.0, 20.0}, {30.0, 30.0}};
    double covariances[][3] = {{10.0, -3.0, 15.0}, {20.0, 2.0, 5.0}};

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "Test " << t);

        TVector2Vec points;
        for (std::size_t i = 0; i < 2; ++i) {
            TVector2 mean(&means[i][0], &means[i][2]);
            TMatrix2 covariance(&covariances[i][0], &covariances[i][3]);
            TVector2Vec cluster;
            maths::common::CSampling::multivariateNormalSample(mean, covariance, 500, cluster);
            points.insert(points.end(), cluster.begin(), cluster.end());
        }

        maths::common::CKMeans<TVector2> kmeans;
        kmeans.setPoints(points);

        CXMeansForTest<TVector2> xmeans(5);
        xmeans.setPoints(points);
        xmeans.improveStructure(2, 1);

        TVector2Vec seedCentres;
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            seedCentres.push_back(xmeans.clusters()[i].centre());
        }
        std::sort(seedCentres.begin(), seedCentres.end());
        LOG_DEBUG(<< "seed centres = " << seedCentres);

        kmeans.setCentres(seedCentres);
        kmeans.run(5);

        xmeans.improveParams(5);

        TVector2Vec expectedCentres = kmeans.centres();
        std::sort(expectedCentres.begin(), expectedCentres.end());

        TVector2Vec centres;
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            centres.push_back(xmeans.clusters()[i].centre());
        }
        std::sort(centres.begin(), centres.end());

        LOG_DEBUG(<< "expected centres = " << expectedCentres);
        LOG_DEBUG(<< "centres          = " << centres);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedCentres),
                            core::CContainerPrinter::print(centres));
    }
}

BOOST_AUTO_TEST_CASE(testOneCluster) {
    // Test it typically chooses just one cluster and that when we
    // do choose to split it is because a spherical Gaussian is a
    // bad approximation.

    maths::common::CSampling::seed();

    const std::size_t size = 500;

    test::CRandomNumbers rng;

    TMeanAccumulator meanNumberClusters;

    TVector2Vec means;
    TMatrix2Vec covariances;
    TVector2VecVec points;

    for (std::size_t t = 0; t < 50; ++t) {
        LOG_DEBUG(<< "*** test = " << t << " ***");

        TSizeVec sizes(1, size);
        rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

        LOG_DEBUG(<< "  mean       = " << means);

        CEmpiricalKullbackLeibler kl;
        kl.add(points[0]);

        maths::common::CXMeans<TVector2> xmeans(10);
        xmeans.setPoints(points[0]);
        xmeans.run(3, 3, 5);

        CEmpiricalKullbackLeibler klc;
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            klc.add(xmeans.clusters()[i].points());
        }

        LOG_DEBUG(<< "  centres               = " << xmeans.centres());
        LOG_DEBUG(<< "  points empirical KL   = " << kl.value());
        LOG_DEBUG(<< "  clusters empirical KL = " << klc.value());

        meanNumberClusters.add(static_cast<double>(xmeans.clusters().size()));
        if (xmeans.clusters().size() > 1) {
            BOOST_TEST_REQUIRE(kl.value() - klc.value() > 0.7);
        }
    }

    LOG_DEBUG(<< "mean number clusters = "
              << maths::common::CBasicStatistics::mean(meanNumberClusters));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanNumberClusters) < 1.15);
}

BOOST_AUTO_TEST_CASE(testFiveClusters) {
    // Test x-means clustering quality on data with five clusters.

    maths::common::CSampling::seed();

    const std::size_t sizes_[] = {500, 800, 100, 400, 600};
    TSizeVec sizes(std::begin(sizes_), std::end(sizes_));

    test::CRandomNumbers rng;

    TMeanVarAccumulator meanNumberClusters;
    TMeanAccumulator klgain;
    TMeanAccumulator meanTotalPurity;

    TVector2Vec means;
    TMatrix2Vec covariances;
    TVector2VecVec points;
    TVector2Vec flatPoints;

    //std::ofstream file;
    //file.open("results.m");

    for (std::size_t t = 0; t < 50; ++t) {
        LOG_DEBUG(<< "*** test = " << t + 1 << " ***");

        rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

        LOG_DEBUG(<< "  means       = " << means);

        flatPoints.clear();
        CEmpiricalKullbackLeibler kl;

        for (std::size_t i = 0; i < points.size(); ++i) {
            kl.add(points[i]);
            flatPoints.insert(flatPoints.end(), points[i].begin(), points[i].end());
            std::sort(points[i].begin(), points[i].end());
        }

        std::size_t ne = flatPoints.size();

        maths::common::CXMeans<TVector2, maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_AICc>> xmeans(
            10);
        xmeans.setPoints(flatPoints);
        xmeans.run(3, 3, 5);

        CEmpiricalKullbackLeibler klc;
        TSizeVecVec trueClusters(xmeans.clusters().size());

        std::size_t n = 0;
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            const TVector2Vec& clusterPoints = xmeans.clusters()[i].points();

            klc.add(clusterPoints);
            n += clusterPoints.size();

            //file << "y" << t+1 << i+1 << " = [";
            for (std::size_t j = 0; j < clusterPoints.size(); ++j) {
                //file << clusterPoints[j](0) << "," << clusterPoints[j](1) << "\n";

                std::size_t k = 0;
                for (/**/; k < points.size(); ++k) {
                    for (TVector2VecCItr
                             itr = std::lower_bound(points[k].begin(), points[k].end(),
                                                    clusterPoints[j]),
                             end = std::upper_bound(points[k].begin(), points[k].end(),
                                                    clusterPoints[j]);
                         itr != end; ++itr) {
                        if (clusterPoints[j] == *itr) {
                            goto FoundPoint;
                        }
                    }
                }

                LOG_ERROR(<< "Didn't find " << clusterPoints[j]);
                BOOST_TEST_REQUIRE(false);

            FoundPoint:
                trueClusters[i].push_back(k);
            }
            //file << "];\n";
        }

        BOOST_REQUIRE_EQUAL(ne, n);

        TDoubleVec purities;
        computePurities(trueClusters, purities);

        double minPurity = 1.0;
        TMeanAccumulator totalPurity;
        for (std::size_t i = 0; i < purities.size(); ++i) {
            minPurity = std::min(minPurity, purities[i]);
            totalPurity.add(purities[i],
                            static_cast<double>(xmeans.clusters()[i].size()));
        }

        LOG_DEBUG(<< "  centres               = " << xmeans.centres());
        LOG_DEBUG(<< "  purities              = " << purities);
        LOG_DEBUG(<< "  points empirical KL   = " << kl.value());
        LOG_DEBUG(<< "  clusters empirical KL = " << klc.value());
        LOG_DEBUG(<< "  minPurity             = " << minPurity);
        LOG_DEBUG(<< "  totalPurity           = "
                  << maths::common::CBasicStatistics::mean(totalPurity));
        BOOST_TEST_REQUIRE(minPurity > 0.39);
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(totalPurity) > 0.54);

        meanNumberClusters.add(static_cast<double>(xmeans.clusters().size()));
        klgain.add((kl.value() - klc.value()));
        meanTotalPurity.add(maths::common::CBasicStatistics::mean(totalPurity));
    }

    LOG_DEBUG(<< "mean number clusters = "
              << maths::common::CBasicStatistics::mean(meanNumberClusters));
    LOG_DEBUG(<< "sd number clusters = "
              << std::sqrt(maths::common::CBasicStatistics::variance(meanNumberClusters)));
    LOG_DEBUG(<< "KL gain = " << maths::common::CBasicStatistics::mean(klgain));
    LOG_DEBUG(<< "mean total purity = "
              << maths::common::CBasicStatistics::mean(meanTotalPurity));

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        5.0, maths::common::CBasicStatistics::mean(meanNumberClusters), 0.3);
    BOOST_TEST_REQUIRE(
        std::sqrt(maths::common::CBasicStatistics::variance(meanNumberClusters)) < 0.9);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(klgain) > -0.1);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanTotalPurity) > 0.93);
}

BOOST_AUTO_TEST_CASE(testTwentyClusters) {
    // Test x-means clustering quality on data with twenty clusters.

    maths::common::CSampling::seed();

    const std::size_t sizes_[] = {1800, 800,  1100, 400, 600,  400, 600,
                                  1300, 400,  900,  500, 700,  400, 800,
                                  1500, 1200, 500,  300, 1200, 800};
    TSizeVec sizes(std::begin(sizes_), std::end(sizes_));

    test::CRandomNumbers rng;

    TVector2Vec means;
    TMatrix2Vec covariances;
    TVector2VecVec points;

    rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

    CEmpiricalKullbackLeibler kl;
    TVector2Vec flatPoints;

    for (std::size_t i = 0; i < points.size(); ++i) {
        kl.add(points[i]);
        flatPoints.insert(flatPoints.end(), points[i].begin(), points[i].end());
        std::sort(points[i].begin(), points[i].end());
    }

    std::size_t ne = flatPoints.size();

    maths::common::CXMeans<TVector2, maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_AICc>> xmeans(
        40);
    xmeans.setPoints(flatPoints);
    xmeans.run(4, 4, 5);

    LOG_DEBUG(<< "# clusters = " << xmeans.clusters().size());

    //std::ofstream file;
    //file.open("results.m");

    CEmpiricalKullbackLeibler klc;
    TSizeVecVec trueClusters(xmeans.clusters().size());

    std::size_t n = 0;
    for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
        const TVector2Vec& clusterPoints = xmeans.clusters()[i].points();

        klc.add(clusterPoints);
        n += clusterPoints.size();

        //file << "y" << i+1 << " = [";
        for (std::size_t j = 0; j < clusterPoints.size(); ++j) {
            //file << clusterPoints[j](0) << "," << clusterPoints[j](1) << "\n";

            std::size_t k = 0;
            for (/**/; k < points.size(); ++k) {
                for (TVector2VecCItr
                         itr = std::lower_bound(points[k].begin(),
                                                points[k].end(), clusterPoints[j]),
                         end = std::upper_bound(points[k].begin(),
                                                points[k].end(), clusterPoints[j]);
                     itr != end; ++itr) {
                    if (clusterPoints[j] == *itr) {
                        goto FoundPoint;
                    }
                }
            }

            LOG_ERROR(<< "Didn't find " << clusterPoints[j]);
            BOOST_TEST_REQUIRE(false);

        FoundPoint:
            trueClusters[i].push_back(k);
        }
        //file << "];\n";
    }

    BOOST_REQUIRE_EQUAL(ne, n);

    TDoubleVec purities;
    computePurities(trueClusters, purities);

    double minPurity = 1.0;
    TMeanAccumulator totalPurity;
    for (std::size_t i = 0; i < purities.size(); ++i) {
        minPurity = std::min(minPurity, purities[i]);
        totalPurity.add(purities[i], static_cast<double>(xmeans.clusters()[i].size()));
    }

    LOG_DEBUG(<< "purities              = " << purities);
    LOG_DEBUG(<< "points empirical KL   = " << kl.value());
    LOG_DEBUG(<< "clusters empirical KL = " << klc.value());
    LOG_DEBUG(<< "minPurity             = " << minPurity);
    LOG_DEBUG(<< "totalPurity           = "
              << maths::common::CBasicStatistics::mean(totalPurity));

    BOOST_REQUIRE_CLOSE_ABSOLUTE(20.0, static_cast<double>(xmeans.clusters().size()), 6.0);
    BOOST_TEST_REQUIRE(klc.value() <
                       kl.value() + 0.05 * std::max(std::fabs(klc.value()),
                                                    std::fabs(kl.value())));
    BOOST_TEST_REQUIRE(minPurity > 0.4);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(totalPurity) > 0.8);
}

BOOST_AUTO_TEST_CASE(testPoorlyConditioned) {
    // Test we can handle poorly conditioned covariance matrices.

    maths::common::CSampling::seed();

    double points_[][2] = {
        {0.0, 0.0},    {1.0, 0.5},    {2.0, 1.0},    {3.0, 1.5},
        {4.0, 2.0},    {5.0, 2.5},    {6.0, 3.0},    {7.0, 3.5},
        {8.0, 4.0},    {9.0, 4.5},    {101.0, 21.9}, {102.0, 21.2},
        {101.5, 22.0}, {104.0, 23.0}, {102.6, 21.4}, {101.3, 22.0},
        {101.2, 21.0}, {101.1, 22.1}, {101.7, 23.0}, {101.0, 24.0},
        {50.0, 50.0},  {51.0, 51.0},  {50.0, 51.0},  {54.0, 53.0},
        {52.0, 51.0},  {51.0, 52.0},  {51.0, 52.0},  {53.0, 53.0},
        {53.0, 52.0},  {52.0, 54.0},  {52.0, 52.0},  {52.0, 52.0},
        {53.0, 52.0},  {51.0, 52.0}};

    TVector2Vec cluster1;
    for (std::size_t i = 0; i < 10; ++i) {
        cluster1.push_back(TVector2(&points_[i][0], &points_[i][2]));
    }
    std::sort(cluster1.begin(), cluster1.end());
    TVector2Vec cluster2;
    for (std::size_t i = 10; i < 20; ++i) {
        cluster2.push_back(TVector2(&points_[i][0], &points_[i][2]));
    }
    std::sort(cluster2.begin(), cluster2.end());
    TVector2Vec cluster3;
    for (std::size_t i = 20; i < std::size(points_); ++i) {
        cluster3.push_back(TVector2(&points_[i][0], &points_[i][2]));
    }
    std::sort(cluster3.begin(), cluster3.end());

    maths::common::CXMeans<TVector2, maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_BIC>> xmeans(
        5);
    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "*** test = " << t << " ***");

        TVector2Vec points;
        for (std::size_t i = 0; i < std::size(points_); ++i) {
            points.push_back(TVector2(&points_[i][0], &points_[i][2]));
        }

        xmeans.setPoints(points);
        xmeans.run(4, 4, 5);

        LOG_DEBUG(<< "# clusters = " << xmeans.clusters().size());
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            TVector2Vec clusterPoints = xmeans.clusters()[i].points();
            std::sort(clusterPoints.begin(), clusterPoints.end());
            LOG_DEBUG(<< "points = " << clusterPoints);
            BOOST_REQUIRE(clusterPoints == cluster1 ||
                          clusterPoints == cluster2 || clusterPoints == cluster3);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
