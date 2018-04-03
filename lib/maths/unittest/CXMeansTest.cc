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

#include "CXMeansTest.h"

#include <core/CLogger.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSampling.h>
#include <maths/CXMeans.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/math/constants/constants.hpp>
#include <boost/range.hpp>

#include <stdint.h>

using namespace ml;

namespace
{

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TUInt64Vec = std::vector<uint64_t>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TVector2 = maths::CVectorNx1<double, 2>;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecCItr = TVector2Vec::const_iterator;
using TVector2VecVec = std::vector<TVector2Vec>;
using TMeanVar2Accumulator = maths::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator;
using TMatrix2 = maths::CSymmetricMatrixNxN<double, 2>;
using TMatrix2Vec = std::vector<TMatrix2>;
using TVector4 = maths::CVectorNx1<double, 4>;
using TVector4Vec = std::vector<TVector4>;
using TMeanVar4Accumulator = maths::CBasicStatistics::SSampleMeanVar<TVector4>::TAccumulator;
using TMatrix4 = maths::CSymmetricMatrixNxN<double, 4>;
using TMatrix4Vec = std::vector<TMatrix4>;

//! \brief Expose internals of x-means for testing.
template<typename POINT,
         typename COST = maths::CSphericalGaussianInfoCriterion<POINT, maths::E_BIC>>
class CXMeansForTest : public maths::CXMeans<POINT, COST>
{
    public:
        using TUInt64USet = typename maths::CXMeans<POINT, COST>::TUInt64USet;

    public:
        CXMeansForTest(std::size_t kmax) :
            maths::CXMeans<POINT, COST>(kmax)
        {}

        void improveParams(std::size_t kmeansIterations)
        {
            this->maths::CXMeans<POINT, COST>::improveParams(kmeansIterations);
        }

        bool improveStructure(std::size_t clusterSeeds,
                              std::size_t kmeansIterations)
        {
            return this->maths::CXMeans<POINT, COST>::improveStructure(clusterSeeds, kmeansIterations);
        }

        const TUInt64USet &inactive() const
        {
            return this->maths::CXMeans<POINT, COST>::inactive();
        }
};

template<typename POINT>
double logfSphericalGaussian(const POINT &mean,
                             double variance,
                             const POINT &x)
{
    double d = static_cast<double>(x.dimension());
    double r = (x - mean).euclidean();
    return -0.5 * d * std::log(boost::math::double_constants::two_pi * variance)
           -0.5 * r * r / variance;
}

class CEmpiricalKullbackLeibler
{
    public:
        double value() const
        {
            return   maths::CBasicStatistics::mean(m_Divergence)
                   - std::log(maths::CBasicStatistics::count(m_Divergence));
        }

        template<typename POINT>
        void add(const std::vector<POINT> &points)
        {
            typename maths::CBasicStatistics::SSampleMeanVar<POINT>::TAccumulator moments;
            moments.add(points);
            POINT mean = maths::CBasicStatistics::mean(moments);
            POINT variances = maths::CBasicStatistics::variance(moments);

            maths::CBasicStatistics::SSampleMean<double>::TAccumulator variance_;
            for (std::size_t i = 0u; i < variances.dimension(); ++i)
            {
                variance_.add(variances(i));
            }
            double variance = maths::CBasicStatistics::mean(variance_);
            for (std::size_t i = 0u; i < points.size(); ++i)
            {
                m_Divergence.add(-logfSphericalGaussian(mean, variance, points[i]));
            }
        }

    private:
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator m_Divergence;
};

void computePurities(const TSizeVecVec &clusters,
                     TDoubleVec &purities)
{
    purities.clear();
    purities.resize(clusters.size());

    TSizeVec counts;
    for (std::size_t i = 0u; i < clusters.size(); ++i)
    {
        counts.clear();
        for (std::size_t j = 0u; j < clusters[i].size(); ++j)
        {
            counts.resize(std::max(counts.size(), clusters[i][j] + 1));
            ++counts[clusters[i][j]];
        }
        purities[i] =  static_cast<double>(*std::max_element(counts.begin(), counts.end()))
                     / static_cast<double>(clusters[i].size());
    }
}

}

void CXMeansTest::testCluster()
{
    LOG_DEBUG("+----------------------------+");
    LOG_DEBUG("|  CXMeansTest::testCluster  |");
    LOG_DEBUG("+----------------------------+");

    // Test basic accessors and checksum functionality of cluster.

    using TDoubleVec = std::vector<double>;

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(-100.0, 400.0, 800, samples);

    for (std::size_t t = 0u; t < 100; ++t)
    {
        LOG_DEBUG("Test " << t);
        {
            maths::CXMeans<TVector2>::CCluster cluster1;
            maths::CXMeans<TVector2>::CCluster cluster2;

            CPPUNIT_ASSERT_EQUAL(std::size_t(0), cluster1.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(0), cluster2.size());

            TVector2Vec points;
            for (std::size_t i = 0u; i < samples.size(); i += 2)
            {
                points.push_back(TVector2(&samples[i], &samples[i + 2]));
            }
            TVector2Vec pointsCopy(points);
            cluster1.points(pointsCopy);
            rng.random_shuffle(points.begin(), points.end());
            cluster2.points(points);

            CPPUNIT_ASSERT_EQUAL(samples.size() / 2, cluster1.size());
            CPPUNIT_ASSERT_EQUAL(samples.size() / 2, cluster2.size());
            CPPUNIT_ASSERT_EQUAL(cluster1.checksum(), cluster2.checksum());
            CPPUNIT_ASSERT(cluster1 == cluster2);
            CPPUNIT_ASSERT(!(cluster1 < cluster2));
            CPPUNIT_ASSERT(!(cluster2 < cluster1));
        }
        {
            maths::CXMeans<TVector4>::CCluster cluster1;
            maths::CXMeans<TVector4>::CCluster cluster2;

            CPPUNIT_ASSERT_EQUAL(std::size_t(0), cluster1.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(0), cluster2.size());

            TVector4Vec points;
            for (std::size_t i = 0u; i < samples.size(); i += 4)
            {
                points.push_back(TVector4(&samples[i], &samples[i + 4]));
            }
            TVector4Vec pointsCopy(points);
            cluster1.points(pointsCopy);
            rng.random_shuffle(points.begin(), points.end());
            cluster2.points(points);

            CPPUNIT_ASSERT_EQUAL(samples.size() / 4, cluster1.size());
            CPPUNIT_ASSERT_EQUAL(samples.size() / 4, cluster2.size());
            CPPUNIT_ASSERT_EQUAL(cluster1.checksum(), cluster2.checksum());
            CPPUNIT_ASSERT(cluster1 == cluster2);
            CPPUNIT_ASSERT(!(cluster1 < cluster2));
            CPPUNIT_ASSERT(!(cluster2 < cluster1));
        }
    }
}

void CXMeansTest::testImproveStructure()
{
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CXMeansTest::testImproveStructure  |");
    LOG_DEBUG("+-------------------------------------+");

    // Test improve structure finds an obvious split of the data.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    maths::CSampling::seed();

    double means[][2] = { { 10.0, 20.0 }, { 50.0, 30.0 } };
    double covariances[][3] = { { 10.0, -3.0, 15.0 }, { 20.0, 2.0, 5.0 } };

    TMeanAccumulator meanError;

    for (std::size_t t = 0u; t < 10; ++t)
    {
        LOG_DEBUG("Test " << t);

        TVector2Vec points;
        for (std::size_t i = 0u; i < 2; ++i)
        {
            TVector2 mean(&means[i][0], &means[i][2]);
            TMatrix2 covariance(&covariances[i][0], &covariances[i][3]);
            TVector2Vec cluster;
            maths::CSampling::multivariateNormalSample(mean, covariance, 500, cluster);
            points.insert(points.end(), cluster.begin(), cluster.end());
        }

        CXMeansForTest<TVector2> xmeans(5);
        xmeans.setPoints(points);
        xmeans.improveStructure(2, 5);

        TVector2Vec clusters;
        TUInt64Vec oldChecksums;
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            clusters.push_back(xmeans.clusters()[i].centre());
            oldChecksums.push_back(xmeans.clusters()[i].checksum());
        }
        std::sort(clusters.begin(), clusters.end());
        std::sort(oldChecksums.begin(), oldChecksums.end());
        LOG_DEBUG("centres = " << core::CContainerPrinter::print(clusters));

        for (std::size_t i = 0u; i < clusters.size(); ++i)
        {
            TVector2 mean(&means[i][0], &means[i][2]);
            double error = (clusters[i] - mean).euclidean();
            CPPUNIT_ASSERT(error < 0.75);
            meanError.add(error);
        }

        // Check that we've marked any clusters which haven't changed
        // as inactive.
        xmeans.improveStructure(2, 5);
        TUInt64Vec newChecksums;
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            clusters.push_back(xmeans.clusters()[i].centre());
            newChecksums.push_back(xmeans.clusters()[i].checksum());
        }
        std::sort(newChecksums.begin(), newChecksums.end());

        TUInt64Vec inactive;
        std::set_intersection(oldChecksums.begin(), oldChecksums.end(),
                              newChecksums.begin(), newChecksums.end(),
                              std::back_inserter(inactive));
        LOG_DEBUG("inactive = " << core::CContainerPrinter::print(inactive));
        for (std::size_t i = 0u; i < inactive.size(); ++i)
        {
            CPPUNIT_ASSERT(xmeans.inactive().count(inactive[i]) > 0);
        }
    }

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.25);
}

void CXMeansTest::testImproveParams()
{
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CXMeansTest::testImproveParams  |");
    LOG_DEBUG("+----------------------------------+");

    // Test that improve params is equivalent to a round of k-means
    // on current state cluster centres.

    maths::CSampling::seed();

    double means[][2] = { { 10.0, 20.0 }, { 30.0, 30.0 } };
    double covariances[][3] = { { 10.0, -3.0, 15.0 }, { 20.0, 2.0, 5.0 } };

    for (std::size_t t = 0u; t < 10; ++t)
    {
        LOG_DEBUG("Test " << t);

        TVector2Vec points;
        for (std::size_t i = 0u; i < 2; ++i)
        {
            TVector2 mean(&means[i][0], &means[i][2]);
            TMatrix2 covariance(&covariances[i][0], &covariances[i][3]);
            TVector2Vec cluster;
            maths::CSampling::multivariateNormalSample(mean, covariance, 500, cluster);
            points.insert(points.end(), cluster.begin(), cluster.end());
        }

        maths::CKMeansFast<TVector2> kmeans;
        kmeans.setPoints(points);

        CXMeansForTest<TVector2> xmeans(5);
        xmeans.setPoints(points);
        xmeans.improveStructure(2, 1);

        TVector2Vec seedCentres;
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            seedCentres.push_back(xmeans.clusters()[i].centre());
        }
        std::sort(seedCentres.begin(), seedCentres.end());
        LOG_DEBUG("seed centres = " << core::CContainerPrinter::print(seedCentres));

        kmeans.setCentres(seedCentres);
        kmeans.run(5);

        xmeans.improveParams(5);

        TVector2Vec expectedCentres = kmeans.centres();
        std::sort(expectedCentres.begin(), expectedCentres.end());

        TVector2Vec centres;
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            centres.push_back(xmeans.clusters()[i].centre());
        }
        std::sort(centres.begin(), centres.end());

        LOG_DEBUG("expected centres = " << core::CContainerPrinter::print(expectedCentres));
        LOG_DEBUG("centres          = " << core::CContainerPrinter::print(centres));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedCentres),
                             core::CContainerPrinter::print(centres));
    }
}

void CXMeansTest::testOneCluster()
{
    LOG_DEBUG("+-------------------------------+");
    LOG_DEBUG("|  CXMeansTest::testOneCluster  |");
    LOG_DEBUG("+-------------------------------+");

    // Test it typically chooses just one cluster and that when we
    // do choose to split it is because a spherical Gaussian is a
    // bad approximation.

    maths::CSampling::seed();

    const std::size_t size = 500;

    test::CRandomNumbers rng;

    TMeanAccumulator meanNumberClusters;


    TVector2Vec means;
    TMatrix2Vec covariances;
    TVector2VecVec points;

    for (std::size_t t = 0; t < 50; ++t)
    {
        LOG_DEBUG("*** test = " << t << " ***");

        TSizeVec sizes(1, size);
        rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

        LOG_DEBUG("  mean       = " << core::CContainerPrinter::print(means));

        CEmpiricalKullbackLeibler kl;
        kl.add(points[0]);

        maths::CXMeans<TVector2> xmeans(10);
        xmeans.setPoints(points[0]);
        xmeans.run(3, 3, 5);

        CEmpiricalKullbackLeibler klc;
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            klc.add(xmeans.clusters()[i].points());
        }

        LOG_DEBUG("  centres               = " << core::CContainerPrinter::print(xmeans.centres()));
        LOG_DEBUG("  points empirical KL   = " << kl.value());
        LOG_DEBUG("  clusters empirical KL = " << klc.value());

        meanNumberClusters.add(static_cast<double>(xmeans.clusters().size()));
        if (xmeans.clusters().size() > 1)
        {
            CPPUNIT_ASSERT(kl.value() - klc.value() > 0.7);
        }
    }

    LOG_DEBUG("mean number clusters = "
              << maths::CBasicStatistics::mean(meanNumberClusters));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanNumberClusters) < 1.15);
}

void CXMeansTest::testFiveClusters()
{
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CXMeansTest::testFiveClusters  |");
    LOG_DEBUG("+---------------------------------+");

    // Test x-means clustering quality on data with five clusters.

    maths::CSampling::seed();

    const std::size_t sizes_[] = { 500, 800, 100, 400, 600 };
    TSizeVec sizes(boost::begin(sizes_), boost::end(sizes_));

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

    for (std::size_t t = 0; t < 50; ++t)
    {
        LOG_DEBUG("*** test = " << t+1 << " ***");

        rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

        LOG_DEBUG("  means       = " << core::CContainerPrinter::print(means));

        flatPoints.clear();
        CEmpiricalKullbackLeibler kl;

        for (std::size_t i = 0u; i < points.size(); ++i)
        {
            kl.add(points[i]);
            flatPoints.insert(flatPoints.end(),
                              points[i].begin(),
                              points[i].end());
            std::sort(points[i].begin(), points[i].end());
        }

        std::size_t ne = flatPoints.size();

        maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_AICc>> xmeans(10);
        xmeans.setPoints(flatPoints);
        xmeans.run(3, 3, 5);

        CEmpiricalKullbackLeibler klc;
        TSizeVecVec trueClusters(xmeans.clusters().size());

        std::size_t n = 0u;
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            const TVector2Vec &clusterPoints = xmeans.clusters()[i].points();

            klc.add(clusterPoints);
            n += clusterPoints.size();

            //file << "y" << t+1 << i+1 << " = [";
            for (std::size_t j = 0u; j < clusterPoints.size(); ++j)
            {
                //file << clusterPoints[j](0) << "," << clusterPoints[j](1) << "\n";

                std::size_t k = 0u;
                for (/**/; k < points.size(); ++k)
                {
                    for (TVector2VecCItr itr = std::lower_bound(points[k].begin(),
                                                                points[k].end(),
                                                                clusterPoints[j]),
                                         end = std::upper_bound(points[k].begin(),
                                                                points[k].end(),
                                                                clusterPoints[j]);
                         itr != end;
                         ++itr)
                    {
                        if (clusterPoints[j] == *itr)
                        {
                            goto FoundPoint;
                        }
                    }
                }

                LOG_ERROR("Didn't find " << clusterPoints[j]);
                CPPUNIT_ASSERT(false);

                FoundPoint:
                trueClusters[i].push_back(k);
            }
            //file << "];\n";
        }

        CPPUNIT_ASSERT_EQUAL(ne, n);

        TDoubleVec purities;
        computePurities(trueClusters, purities);

        double minPurity = 1.0;
        TMeanAccumulator totalPurity;
        for (std::size_t i = 0u; i < purities.size(); ++i)
        {
            minPurity = std::min(minPurity, purities[i]);
            totalPurity.add(purities[i],
                            static_cast<double>(xmeans.clusters()[i].size()));
        }

        LOG_DEBUG("  centres               = " << core::CContainerPrinter::print(xmeans.centres()));
        LOG_DEBUG("  purities              = " << core::CContainerPrinter::print(purities));
        LOG_DEBUG("  points empirical KL   = " << kl.value());
        LOG_DEBUG("  clusters empirical KL = " << klc.value());
        LOG_DEBUG("  minPurity             = " << minPurity);
        LOG_DEBUG("  totalPurity           = " << maths::CBasicStatistics::mean(totalPurity));
        CPPUNIT_ASSERT(minPurity                                  > 0.39);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(totalPurity) > 0.54);

        meanNumberClusters.add(static_cast<double>(xmeans.clusters().size()));
        klgain.add((kl.value() - klc.value()));
        meanTotalPurity.add(maths::CBasicStatistics::mean(totalPurity));
    }

    LOG_DEBUG("mean number clusters = "
              << maths::CBasicStatistics::mean(meanNumberClusters));
    LOG_DEBUG("sd number clusters = "
              << std::sqrt(maths::CBasicStatistics::variance(meanNumberClusters)));
    LOG_DEBUG("KL gain = " << maths::CBasicStatistics::mean(klgain));
    LOG_DEBUG("mean total purity = " << maths::CBasicStatistics::mean(meanTotalPurity));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, maths::CBasicStatistics::mean(meanNumberClusters), 0.3);
    CPPUNIT_ASSERT(std::sqrt(maths::CBasicStatistics::variance(meanNumberClusters)) < 0.9);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(klgain) > -0.1);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanTotalPurity) > 0.93);
}

void CXMeansTest::testTwentyClusters()
{
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CXMeansTest::testTwentyClusters  |");
    LOG_DEBUG("+-----------------------------------+");

    // Test x-means clustering quality on data with twenty clusters.

    maths::CSampling::seed();

    const std::size_t sizes_[] =
        {
            1800, 800, 1100,  400,  600,
             400, 600, 1300,  400,  900,
             500, 700,  400,  800, 1500,
            1200, 500,  300, 1200,  800
        };
    TSizeVec sizes(boost::begin(sizes_), boost::end(sizes_));

    test::CRandomNumbers rng;

    TVector2Vec means;
    TMatrix2Vec covariances;
    TVector2VecVec points;

    rng.generateRandomMultivariateNormals(sizes, means, covariances, points);

    CEmpiricalKullbackLeibler kl;
    TVector2Vec flatPoints;

    for (std::size_t i = 0u; i < points.size(); ++i)
    {
        kl.add(points[i]);
        flatPoints.insert(flatPoints.end(),
                          points[i].begin(),
                          points[i].end());
        std::sort(points[i].begin(), points[i].end());
    }

    std::size_t ne = flatPoints.size();

    maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_AICc>> xmeans(40);
    xmeans.setPoints(flatPoints);
    xmeans.run(4, 4, 5);

    LOG_DEBUG("# clusters = " << xmeans.clusters().size());

    //std::ofstream file;
    //file.open("results.m");

    CEmpiricalKullbackLeibler klc;
    TSizeVecVec trueClusters(xmeans.clusters().size());

    std::size_t n = 0u;
    for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
    {
        const TVector2Vec &clusterPoints = xmeans.clusters()[i].points();

        klc.add(clusterPoints);
        n += clusterPoints.size();

        //file << "y" << i+1 << " = [";
        for (std::size_t j = 0u; j < clusterPoints.size(); ++j)
        {
            //file << clusterPoints[j](0) << "," << clusterPoints[j](1) << "\n";

            std::size_t k = 0u;
            for (/**/; k < points.size(); ++k)
            {
                for (TVector2VecCItr itr = std::lower_bound(points[k].begin(),
                                                            points[k].end(),
                                                            clusterPoints[j]),
                                     end = std::upper_bound(points[k].begin(),
                                                            points[k].end(),
                                                            clusterPoints[j]);
                     itr != end;
                     ++itr)
                {
                    if (clusterPoints[j] == *itr)
                    {
                        goto FoundPoint;
                    }
                }
            }

            LOG_ERROR("Didn't find " << clusterPoints[j]);
            CPPUNIT_ASSERT(false);

            FoundPoint:
            trueClusters[i].push_back(k);
        }
        //file << "];\n";
    }

    CPPUNIT_ASSERT_EQUAL(ne, n);

    TDoubleVec purities;
    computePurities(trueClusters, purities);

    double minPurity = 1.0;
    TMeanAccumulator totalPurity;
    for (std::size_t i = 0u; i < purities.size(); ++i)
    {
        minPurity = std::min(minPurity, purities[i]);
        totalPurity.add(purities[i],
                        static_cast<double>(xmeans.clusters()[i].size()));
    }

    LOG_DEBUG("purities              = " << core::CContainerPrinter::print(purities));
    LOG_DEBUG("points empirical KL   = " << kl.value());
    LOG_DEBUG("clusters empirical KL = " << klc.value());
    LOG_DEBUG("minPurity             = " << minPurity);
    LOG_DEBUG("totalPurity           = " << maths::CBasicStatistics::mean(totalPurity));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, static_cast<double>(xmeans.clusters().size()), 6.0);
    CPPUNIT_ASSERT(klc.value() < kl.value() + 0.05 * std::max(std::fabs(klc.value()), std::fabs(kl.value())));
    CPPUNIT_ASSERT(minPurity > 0.4);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(totalPurity) > 0.8);
}

void CXMeansTest::testPoorlyConditioned()
{
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CXMeansTest::testPoorlyConditioned  |");
    LOG_DEBUG("+--------------------------------------+");

    // Test we can handle poorly conditioned covariance matrices.

    maths::CSampling::seed();

    double points_[][2] =
        {
            { 0.0, 0.0 },
            { 1.0, 0.5 },
            { 2.0, 1.0 },
            { 3.0, 1.5 },
            { 4.0, 2.0 },
            { 5.0, 2.5 },
            { 6.0, 3.0 },
            { 7.0, 3.5 },
            { 8.0, 4.0 },
            { 9.0, 4.5 },
            { 101.0, 21.9 },
            { 102.0, 21.2 },
            { 101.5, 22.0 },
            { 104.0, 23.0 },
            { 102.6, 21.4 },
            { 101.3, 22.0 },
            { 101.2, 21.0 },
            { 101.1, 22.1 },
            { 101.7, 23.0 },
            { 101.0, 24.0 },
            { 50.0, 50.0 },
            { 51.0, 51.0 },
            { 50.0, 51.0 },
            { 54.0, 53.0 },
            { 52.0, 51.0 },
            { 51.0, 52.0 },
            { 51.0, 52.0 },
            { 53.0, 53.0 },
            { 53.0, 52.0 },
            { 52.0, 54.0 },
            { 52.0, 52.0 },
            { 52.0, 52.0 },
            { 53.0, 52.0 },
            { 51.0, 52.0 }
        };

    TVector2Vec cluster1;
    for (std::size_t i = 0u; i < 10; ++i)
    {
        cluster1.push_back(TVector2(&points_[i][0], &points_[i][2]));
    }
    std::sort(cluster1.begin(), cluster1.end());
    TVector2Vec cluster2;
    for (std::size_t i = 10u; i < 20; ++i)
    {
        cluster2.push_back(TVector2(&points_[i][0], &points_[i][2]));
    }
    std::sort(cluster2.begin(), cluster2.end());
    TVector2Vec cluster3;
    for (std::size_t i = 20u; i < boost::size(points_); ++i)
    {
        cluster3.push_back(TVector2(&points_[i][0], &points_[i][2]));
    }
    std::sort(cluster3.begin(), cluster3.end());

    maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_BIC>> xmeans(5);
    for (std::size_t t = 0u; t < 10; ++t)
    {
        LOG_DEBUG("*** test = " << t << " ***");

        TVector2Vec points;
        for (std::size_t i = 0u; i < boost::size(points_); ++i)
        {
            points.push_back(TVector2(&points_[i][0], &points_[i][2]));
        }

        xmeans.setPoints(points);
        xmeans.run(4, 4, 5);

        LOG_DEBUG("# clusters = " << xmeans.clusters().size());
        for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i)
        {
            TVector2Vec clusterPoints = xmeans.clusters()[i].points();
            std::sort(clusterPoints.begin(), clusterPoints.end());
            LOG_DEBUG("points = " << core::CContainerPrinter::print(clusterPoints));
            CPPUNIT_ASSERT(   clusterPoints == cluster1
                           || clusterPoints == cluster2
                           || clusterPoints == cluster3);
        }
    }
}

CppUnit::Test *CXMeansTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CXMeansTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testCluster",
                                   &CXMeansTest::testCluster) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testImproveStructure",
                                   &CXMeansTest::testImproveStructure) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testImproveParams",
                                   &CXMeansTest::testImproveParams) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testOneCluster",
                                   &CXMeansTest::testOneCluster) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testFiveClusters",
                                   &CXMeansTest::testFiveClusters) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testTwentyClusters",
                                   &CXMeansTest::testTwentyClusters) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXMeansTest>(
                                   "CXMeansTest::testPoorlyConditioned",
                                   &CXMeansTest::testPoorlyConditioned) );

    return suiteOfTests;
}
