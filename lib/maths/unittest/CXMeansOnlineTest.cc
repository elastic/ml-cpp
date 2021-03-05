/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CRestoreParams.h>
#include <maths/CXMeans.h>
#include <maths/CXMeansOnline.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CXMeansOnlineTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TVector2 = maths::CVectorNx1<double, 2>;
using TCovariances2 = maths::CBasicStatistics::SSampleCovariances<TVector2>;
using TCovariances2Vec = std::vector<TCovariances2>;
using TXMeans2 = maths::CXMeansOnline<double, 2>;
using TPoint = TXMeans2::TPointPrecise;
using TPointVec = std::vector<TPoint>;
using TPointVecVec = std::vector<TPointVec>;
using TMatrix = TXMeans2::TMatrixPrecise;
using TMatrixVec = std::vector<TMatrix>;

template<typename T, std::size_t N>
class CXMeansOnlineForTest : public maths::CXMeansOnline<T, N> {
public:
    using TSizeDoublePr2Vec = typename maths::CXMeansOnline<T, N>::TSizeDoublePr2Vec;
    using TPoint = typename maths::CXMeansOnline<T, N>::TPointPrecise;
    using TClusterVec = typename maths::CXMeansOnline<T, N>::TClusterVec;
    using maths::CXMeansOnline<T, N>::add;

public:
    CXMeansOnlineForTest(maths_t::EDataType dataType,
                         maths_t::EClusterWeightCalc weightCalc,
                         double decayRate = 0.0,
                         double minimumClusterFraction = 0.0)
        : maths::CXMeansOnline<T, N>(dataType, weightCalc, decayRate, minimumClusterFraction) {
    }

    void add(const TPoint& x, double count = 1.0) {
        TSizeDoublePr2Vec dummy;
        this->maths::CXMeansOnline<T, N>::add(x, dummy, count);
    }

    const TClusterVec& clusters() const {
        return this->maths::CXMeansOnline<T, N>::clusters();
    }
};

using TXMeans2ForTest = CXMeansOnlineForTest<double, 2>;
using TXMeans2FloatForTest = CXMeansOnlineForTest<maths::CFloatStorage, 2>;

bool restore(const maths::SDistributionRestoreParams& params,
             core::CRapidXmlStateRestoreTraverser& traverser,
             TXMeans2::CCluster& result) {
    return traverser.traverseSubLevel(
        std::bind(&TXMeans2::CCluster::acceptRestoreTraverser, &result,
                  std::cref(params), std::placeholders::_1));
}
}

BOOST_AUTO_TEST_CASE(testCluster) {
    // Test the core functionality of cluster.

    TXMeans2 clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight, 0.1);
    TXMeans2::CCluster cluster(clusterer);

    double x1[][2]{{1.1, 2.0}, {2.3, 2.1}, {1.5, 1.4}, {0.9, 0.8}, {4.7, 3.9},
                   {3.2, 3.2}, {2.8, 2.7}, {2.3, 1.5}, {1.9, 1.6}, {2.6, 2.1},
                   {2.0, 2.2}, {1.7, 1.9}, {1.8, 1.7}, {2.1, 1.9}};
    double c1[]{1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    TCovariances2 moments(2);
    for (std::size_t i = 0; i < boost::size(x1); ++i) {
        cluster.add(TPoint(x1[i]), c1[i]);
        moments.add(TPoint(x1[i]), TPoint(c1[i]));
    }
    LOG_DEBUG(<< "count  = " << cluster.count());
    LOG_DEBUG(<< "centre = " << cluster.centre());
    LOG_DEBUG(<< "spread = " << cluster.spread());
    LOG_DEBUG(<< "weight = " << cluster.weight(maths_t::E_ClustersFractionWeight));

    double expectedCount = maths::CBasicStatistics::count(moments);
    TPoint expectedCentre = maths::CBasicStatistics::mean(moments);
    double expectedSpread = std::sqrt(
        maths::CBasicStatistics::maximumLikelihoodCovariances(moments).trace() / 2.0);
    LOG_DEBUG(<< "expected count  = " << expectedCount);
    LOG_DEBUG(<< "expected centre = " << expectedCentre);
    LOG_DEBUG(<< "expected spread = " << expectedSpread);
    BOOST_REQUIRE_EQUAL(expectedCount, cluster.count());
    BOOST_TEST_REQUIRE((cluster.centre() - expectedCentre).euclidean() < 1e-10);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedSpread, cluster.spread(), 0.05 * expectedSpread);
    BOOST_REQUIRE_EQUAL(1.0, cluster.weight(maths_t::E_ClustersEqualWeight));
    BOOST_REQUIRE_EQUAL(expectedCount, cluster.weight(maths_t::E_ClustersFractionWeight));

    cluster.propagateForwardsByTime(5.0);
    LOG_DEBUG(<< "centre = " << cluster.centre());
    LOG_DEBUG(<< "spread = " << cluster.spread());
    LOG_DEBUG(<< "count  = " << cluster.count());
    LOG_DEBUG(<< "weight = " << cluster.weight(maths_t::E_ClustersFractionWeight));
    BOOST_TEST_REQUIRE(cluster.count() < expectedCount);
    BOOST_TEST_REQUIRE((cluster.centre() - expectedCentre).euclidean() < 1e-10);
    BOOST_TEST_REQUIRE(cluster.spread() >= expectedSpread);
    BOOST_REQUIRE_EQUAL(1.0, cluster.weight(maths_t::E_ClustersEqualWeight));
    BOOST_TEST_REQUIRE(cluster.weight(maths_t::E_ClustersFractionWeight) < expectedCount);

    TPointVec samples;
    cluster.sample(10, samples);
    LOG_DEBUG(<< "samples = " << core::CContainerPrinter::print(samples));

    TCovariances2 sampleMoments(2);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        sampleMoments.add(samples[i]);
    }
    TPoint sampleCentre = maths::CBasicStatistics::mean(sampleMoments);
    double sampleSpread =
        std::sqrt(maths::CBasicStatistics::covariances(sampleMoments).trace() / 2.0);
    LOG_DEBUG(<< "sample centre = " << sampleCentre);
    LOG_DEBUG(<< "sample spread = " << sampleSpread);
    BOOST_TEST_REQUIRE((sampleCentre - cluster.centre()).euclidean() < 1e-10);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(cluster.spread(), sampleSpread, 0.1);

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        std::log(cluster.count()),
        -cluster.logLikelihoodFromCluster(maths_t::E_ClustersEqualWeight, TPoint(1.5)) +
            cluster.logLikelihoodFromCluster(maths_t::E_ClustersFractionWeight, TPoint(1.5)),
        1e-10);

    uint64_t origChecksum = cluster.checksum(0);
    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        cluster.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Cluster XML representation:\n" << origXml);

    // Restore the XML into a new cluster.
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    TXMeans2::CCluster restoredCluster(clusterer);
    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    restore(params, traverser, restoredCluster);
    uint64_t restoredChecksum = restoredCluster.checksum(0);
    BOOST_REQUIRE_EQUAL(origChecksum, restoredChecksum);

    double x2[][2]{{10.3, 10.4}, {10.6, 10.5}, {10.7, 11.0}, {9.8, 10.2},
                   {11.2, 11.4}, {11.0, 10.7}, {11.5, 11.3}};
    double c2[]{2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0};
    for (std::size_t i = 0; i < boost::size(x2); ++i) {
        cluster.add(TPoint(x2[i]), c2[i]);
    }
    maths::CPRNG::CXorOShiro128Plus rng;
    TXMeans2::TOptionalClusterClusterPr split =
        cluster.split(rng, 5.0, clusterer.indexGenerator());
    BOOST_TEST_REQUIRE(split.has_value());
    TPointVec centres;
    centres.push_back(split->first.centre());
    centres.push_back(split->second.centre());
    TDoubleVec spreads;
    spreads.push_back(split->first.spread());
    spreads.push_back(split->second.spread());
    maths::COrderings::simultaneousSort(centres, spreads);
    LOG_DEBUG(<< "centres = " << core::CContainerPrinter::print(centres));
    LOG_DEBUG(<< "spreads = " << core::CContainerPrinter::print(spreads));
    double expectedCentres[][2] = {{2.25, 2.1125}, {10.64, 10.75}};
    BOOST_TEST_REQUIRE((centres[0] - TPoint(expectedCentres[0])).euclidean() < 1e-5);
    BOOST_TEST_REQUIRE((centres[1] - TPoint(expectedCentres[1])).euclidean() < 1e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.796035, spreads[0], 1e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.513273, spreads[1], 1e-5);

    // Test that we don't want to immediately merge clusters.

    BOOST_REQUIRE_EQUAL(false, split->first.shouldMerge(split->second));
    BOOST_REQUIRE_EQUAL(false, split->second.shouldMerge(split->first));

    if (split->first.centre() < split->second.centre()) {
        for (std::size_t i = 0; i < boost::size(x1); ++i) {
            split->second.add(TPoint(x1[i]), c1[i]);
        }
        for (std::size_t i = 0; i < boost::size(x2); ++i) {
            split->first.add(TPoint(x2[i]), c2[i]);
        }
    } else {
        for (std::size_t i = 0; i < boost::size(x1); ++i) {
            split->first.add(TPoint(x1[i]), c1[i]);
        }
        for (std::size_t i = 0; i < boost::size(x2); ++i) {
            split->second.add(TPoint(x2[i]), c2[i]);
        }
    }

    BOOST_REQUIRE_EQUAL(true, split->first.shouldMerge(split->second));
    BOOST_REQUIRE_EQUAL(true, split->second.shouldMerge(split->first));
}

BOOST_AUTO_TEST_CASE(testClusteringVanilla) {
    // This tests that the chance of splitting data with a single
    // cluster is low and that we accurately find a small number
    // of significant clusters.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    double means[][2]{{10, 15}, {40, 10}, {12, 35}};
    double covariances[][2][2]{
        {{10, 2}, {2, 15}}, {{30, 8}, {8, 15}}, {{20, -11}, {-11, 25}}};

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "*** test " << t << " ***");

        TDoubleVec mean(&means[0][0], &means[0][2]);
        TDoubleVecVec covariance;
        for (std::size_t i = 0; i < 2; ++i) {
            covariance.emplace_back(&covariances[0][i][0], &covariances[0][i][2]);
        }
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 200, samples);

        TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight);
        std::size_t n = 0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            clusterer.add(TPoint(samples[i]));
            n += clusterer.numberClusters();
        }
        double s = static_cast<double>(samples.size());
        double c = static_cast<double>(n) / s;
        LOG_DEBUG(<< "# clusters = " << c);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, c, 1.0 / s);
    }

    // We use the cluster moments to indirectly measure the purity
    // of the clusters we find.

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMeanError;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanCovError;

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "*** test " << t << " ***");

        TDoubleVecVec samples;
        TPointVec centres;
        TCovariances2Vec expectedMoments(boost::size(means), TCovariances2(2));

        for (std::size_t i = 0; i < boost::size(means); ++i) {
            TDoubleVec mean(&means[i][0], &means[i][2]);
            TDoubleVecVec covariance;
            for (std::size_t j = 0; j < 2; ++j) {
                covariance.emplace_back(&covariances[i][j][0], &covariances[i][j][2]);
            }
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(mean, covariance, 200, samples_);
            samples.insert(samples.end(), samples_.begin(), samples_.end());
            for (std::size_t j = 0; j < samples_.size(); ++j) {
                expectedMoments[i].add(TPoint(samples_[j]));
            }
        }
        rng.random_shuffle(samples.begin(), samples.end());

        TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            clusterer.add(TPoint(samples[i]));
        }

        const TXMeans2ForTest::TClusterVec& clusters = clusterer.clusters();
        LOG_DEBUG(<< "# clusters = " << clusters.size());
        BOOST_REQUIRE_EQUAL(std::size_t(3), clusters.size());

        for (std::size_t i = 0; i < clusters.size(); ++i) {
            LOG_DEBUG(<< "moments = "
                      << maths::CBasicStatistics::print(clusters[i].covariances()));

            maths::CBasicStatistics::COrderStatisticsStack<double, 1> meanError;
            maths::CBasicStatistics::COrderStatisticsStack<double, 1> covError;
            for (std::size_t j = 0; j < expectedMoments.size(); ++j) {
                meanError.add(
                    (maths::CBasicStatistics::mean(clusters[i].covariances()) -
                     maths::CBasicStatistics::mean(expectedMoments[j]))
                        .euclidean() /
                    maths::CBasicStatistics::mean(expectedMoments[j]).euclidean());
                covError.add(
                    (maths::CBasicStatistics::covariances(clusters[i].covariances()) -
                     maths::CBasicStatistics::covariances(expectedMoments[j]))
                        .frobenius() /
                    maths::CBasicStatistics::covariances(expectedMoments[j]).frobenius());
            }
            LOG_DEBUG(<< "mean error = " << meanError[0]);
            LOG_DEBUG(<< "covariance error = " << covError[0]);
            BOOST_TEST_REQUIRE(meanError[0] < 0.04);
            BOOST_TEST_REQUIRE(covError[0] < 0.39);
            meanMeanError.add(meanError[0]);
            meanCovError.add(covError[0]);
        }
    }

    LOG_DEBUG(<< "mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
    LOG_DEBUG(<< "mean cov error = " << maths::CBasicStatistics::mean(meanCovError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMeanError) < 0.005);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanCovError) < 0.06);
}

BOOST_AUTO_TEST_CASE(testClusteringWithOutliers) {
    // Test that we are still able to find significant clusters
    // in the presence of a small number of significant outliers.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    double means[][2]{{10, 15}, {40, 10}};
    double covariances[][2][2]{{{10, 2}, {2, 15}}, {{30, 8}, {8, 15}}};

    double outliers_[][2] = {{600, 10}, {650, 11}, {610, 12}, {700, 16}, {690, 14}};
    TDoubleVecVec outliers;
    for (std::size_t i = 0; i < boost::size(outliers_); ++i) {
        outliers.push_back(TDoubleVec(std::begin(outliers_[i]), std::end(outliers_[i])));
    }

    // We use the cluster moments to indirectly measure the purity
    // of the clusters we find.

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMeanError;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanCovError;

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "*** test " << t << " ***");

        TDoubleVecVec samples;
        TPointVec centres;
        TCovariances2Vec expectedMoments(boost::size(means), TCovariances2(2));

        for (std::size_t i = 0; i < boost::size(means); ++i) {
            TDoubleVec mean(&means[i][0], &means[i][2]);
            TDoubleVecVec covariance;
            for (std::size_t j = 0; j < 2; ++j) {
                covariance.push_back(
                    TDoubleVec(&covariances[i][j][0], &covariances[i][j][2]));
            }
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(mean, covariance, 200, samples_);
            samples.insert(samples.end(), samples_.begin(), samples_.end());
            for (std::size_t j = 0; j < samples_.size(); ++j) {
                expectedMoments[i].add(TPoint(samples_[j]));
            }
        }
        for (std::size_t i = 0; i < outliers.size(); ++i) {
            expectedMoments[1].add(TPoint(outliers[i]));
        }
        rng.random_shuffle(samples.begin(), samples.end());

        TXMeans2ForTest clusterer(maths_t::E_ContinuousData,
                                  maths_t::E_ClustersFractionWeight, 0.0, 0.01);

        for (std::size_t i = 0; i < outliers.size(); ++i) {
            clusterer.add(TPoint(outliers[i]));
        }
        for (std::size_t i = 0; i < samples.size(); ++i) {
            clusterer.add(TPoint(samples[i]));
        }

        const TXMeans2ForTest::TClusterVec& clusters = clusterer.clusters();
        LOG_DEBUG(<< "# clusters = " << clusters.size());
        BOOST_REQUIRE_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t i = 0; i < clusters.size(); ++i) {
            LOG_DEBUG(<< "moments = "
                      << maths::CBasicStatistics::print(clusters[i].covariances()));

            maths::CBasicStatistics::COrderStatisticsStack<double, 1> meanError;
            maths::CBasicStatistics::COrderStatisticsStack<double, 1> covError;
            for (std::size_t j = 0; j < expectedMoments.size(); ++j) {
                meanError.add(
                    (maths::CBasicStatistics::mean(clusters[i].covariances()) -
                     maths::CBasicStatistics::mean(expectedMoments[j]))
                        .euclidean() /
                    maths::CBasicStatistics::mean(expectedMoments[j]).euclidean());
                covError.add(
                    (maths::CBasicStatistics::covariances(clusters[i].covariances()) -
                     maths::CBasicStatistics::covariances(expectedMoments[j]))
                        .frobenius() /
                    maths::CBasicStatistics::covariances(expectedMoments[j]).frobenius());
            }

            LOG_DEBUG(<< "meanError = " << meanError[0]);
            LOG_DEBUG(<< "covError  = " << covError[0]);
            BOOST_TEST_REQUIRE(meanError[0] < 0.06);
            BOOST_TEST_REQUIRE(covError[0] < 0.2);
            meanMeanError.add(meanError[0]);
            meanCovError.add(covError[0]);
        }
    }

    LOG_DEBUG(<< "mean meanError = " << maths::CBasicStatistics::mean(meanMeanError));
    LOG_DEBUG(<< "mean covError  = " << maths::CBasicStatistics::mean(meanCovError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMeanError) < 0.03);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanCovError) < 0.06);
}

BOOST_AUTO_TEST_CASE(testManyClusters) {
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    maths::CSampling::seed();

    // Test we are able to model a reasonably large number of clusters
    // well. In particular, we test the log likelihood of the data for
    // the estimated distribution versus the generating distribution are
    // close on the order of the data's differential entropy given the
    // generating distribution.

    TSizeVec sizes{1800, 800, 1100, 400, 600,  400,  600, 1300, 400,  900,
                   500,  700, 400,  800, 1500, 1200, 500, 300,  1200, 800};

    double Z = static_cast<double>(std::accumulate(sizes.begin(), sizes.end(), 0));

    test::CRandomNumbers rng;

    TPointVec means;
    TMatrixVec covariances;
    TPointVecVec samples_;
    rng.generateRandomMultivariateNormals(sizes, means, covariances, samples_);
    TPointVec samples;
    for (std::size_t i = 0; i < samples_.size(); ++i) {
        for (std::size_t j = 0; j < samples_[i].size(); ++j) {
            samples.push_back(samples_[i][j]);
        }
    }

    TDoubleVec lgenerating(samples.size());
    TMeanAccumulator differentialEntropy;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        lgenerating[i] = 0.0;
        for (std::size_t j = 0; j < means.size(); ++j) {
            double lj;
            maths::gaussianLogLikelihood(covariances[j], samples[i] - means[j], lj);
            lgenerating[i] += static_cast<double>(sizes[j]) * std::exp(lj);
        }
        lgenerating[i] /= Z;
        differentialEntropy.add(-std::log(lgenerating[i]));
    }
    LOG_DEBUG(<< "differentialEntropy = "
              << maths::CBasicStatistics::mean(differentialEntropy));

    for (std::size_t t = 0; t < 5; ++t) {
        LOG_DEBUG(<< "*** test " << t << " ***");

        rng.random_shuffle(samples.begin(), samples.end());

        TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            clusterer.add(samples[i]);
        }

        const TXMeans2ForTest::TClusterVec& clusters = clusterer.clusters();
        LOG_DEBUG(<< "# clusters = " << clusters.size());

        TMeanAccumulator loss;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            double l = 0.0;
            for (std::size_t j = 0; j < clusters.size(); ++j) {
                double n = maths::CBasicStatistics::count(clusters[j].covariances());
                const TPoint& mean =
                    maths::CBasicStatistics::mean(clusters[j].covariances());
                const TMatrix& covariance = maths::CBasicStatistics::maximumLikelihoodCovariances(
                    clusters[j].covariances());
                double lj;
                maths::gaussianLogLikelihood(covariance, samples[i] - mean, lj);
                l += n * std::exp(lj);
            }
            l /= Z;
            loss.add(std::log(lgenerating[i]) - std::log(l));
        }
        LOG_DEBUG(<< "loss = " << maths::CBasicStatistics::mean(loss));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(loss) <
                           0.02 * maths::CBasicStatistics::mean(differentialEntropy));
    }
}

BOOST_AUTO_TEST_CASE(testAdaption) {
    // Test a case where the cluster pattern changes over time.
    // Specifically, the data set starts with one cluster then
    // a new cluster appears and subsequently disappears.

    using TDoubleVecVecVec = std::vector<TDoubleVecVec>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    TDoubleVecVec means{{10, 15}, {30, 10}, {10, 15}, {30, 10}};
    TDoubleVecVecVec covariances{
        {{10, 2}, {2, 15}}, {{30, 8}, {8, 15}}, {{100, 2}, {2, 15}}, {{100, 2}, {2, 15}}};

    LOG_DEBUG(<< "Clusters Split and Merge");
    std::size_t n[][4]{{200, 0, 0, 0}, {100, 100, 0, 0}, {0, 0, 300, 300}};

    TCovariances2 totalCovariances(2);
    TCovariances2 modeCovariances[4]{TCovariances2(2), TCovariances2(2),
                                     TCovariances2(2), TCovariances2(2)};

    TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight);

    TMeanAccumulator meanMeanError;
    TMeanAccumulator meanCovError;

    for (std::size_t i = 0; i < boost::size(n); ++i) {
        TDoubleVecVec samples;
        for (std::size_t j = 0; j < boost::size(n[i]); ++j) {
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(means[j], covariances[j], n[i][j], samples_);
            for (std::size_t k = 0; k < samples_.size(); ++k) {
                modeCovariances[j].add(TPoint(samples_[k]));
                totalCovariances.add(TPoint(samples_[k]));
            }
            samples.insert(samples.end(), samples_.begin(), samples_.end());
        }
        rng.random_shuffle(samples.begin(), samples.end());
        LOG_DEBUG(<< "# samples = " << samples.size());

        for (std::size_t j = 0; j < samples.size(); ++j) {
            clusterer.add(TPoint(samples[j]));
        }

        const TXMeans2ForTest::TClusterVec& clusters = clusterer.clusters();
        LOG_DEBUG(<< "# clusters = " << clusters.size());

        for (std::size_t j = 0; j < clusters.size(); ++j) {
            maths::CBasicStatistics::COrderStatisticsStack<double, 1> meanError;
            maths::CBasicStatistics::COrderStatisticsStack<double, 1> covError;

            if (clusters.size() == 1) {
                meanError.add((maths::CBasicStatistics::mean(clusters[j].covariances()) -
                               maths::CBasicStatistics::mean(totalCovariances))
                                  .euclidean());
                covError.add(
                    (maths::CBasicStatistics::covariances(clusters[j].covariances()) -
                     maths::CBasicStatistics::covariances(totalCovariances))
                        .frobenius() /
                    maths::CBasicStatistics::covariances(totalCovariances).frobenius());
            } else {
                for (std::size_t k = 0; k < boost::size(modeCovariances); ++k) {
                    meanError.add(
                        (maths::CBasicStatistics::mean(clusters[j].covariances()) -
                         maths::CBasicStatistics::mean(modeCovariances[k]))
                            .euclidean() /
                        maths::CBasicStatistics::mean(modeCovariances[k]).euclidean());
                    covError.add(
                        (maths::CBasicStatistics::covariances(clusters[j].covariances()) -
                         maths::CBasicStatistics::covariances(modeCovariances[k]))
                            .frobenius() /
                        maths::CBasicStatistics::covariances(modeCovariances[k]).frobenius());
                }
            }

            LOG_DEBUG(<< "mean error = " << meanError[0]);
            LOG_DEBUG(<< "cov error  = " << covError[0]);
            BOOST_TEST_REQUIRE(meanError[0] < 0.04);
            BOOST_TEST_REQUIRE(covError[0] < 0.2);

            meanMeanError.add(meanError[0]);
            meanCovError.add(covError[0]);
        }
    }

    LOG_DEBUG(<< "mean meanError = " << maths::CBasicStatistics::mean(meanMeanError));
    LOG_DEBUG(<< "mean covError  = " << maths::CBasicStatistics::mean(meanCovError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMeanError) < 0.01);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanCovError) < 0.1);
}

BOOST_AUTO_TEST_CASE(testLargeHistory) {
    // If we get a lot of history, because we detect that the system
    // is stable and reduce the decay rate then we should also reduce
    // the fraction of points required to create a cluster.

    TXMeans2ForTest reference(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight,
                              0.001, // decay rate
                              0.05); // minimum cluster fraction
    TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight,
                              0.001, // decay rate
                              0.05); // minimum cluster fraction

    // Set the decay rate to simulate decay rate control.
    clusterer.decayRate(0.0001);

    test::CRandomNumbers rng;
    TDoubleVec samples1;
    rng.generateNormalSamples(5.0, 1.0, 20000, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(15.0, 1.0, 200, samples2);

    TPointVec samples;
    for (std::size_t i = 0; i < samples1.size(); i += 2) {
        samples.emplace_back(TDoubleVec(&samples1[i], &samples1[i + 2]));
    }
    for (std::size_t i = 0; i < samples2.size(); i += 2) {
        samples.emplace_back(TDoubleVec(&samples2[i], &samples2[i + 2]));
    }
    rng.random_shuffle(samples.begin() + 5000, samples.end());

    for (const auto& sample : samples) {
        for (std::size_t i = 0; i < 3; ++i) {
            reference.add(sample);
            clusterer.add(sample);
        }
        reference.propagateForwardsByTime(1.0);
        clusterer.propagateForwardsByTime(1.0);
    }

    BOOST_REQUIRE_EQUAL(std::size_t(1), reference.clusters().size());
    BOOST_REQUIRE_EQUAL(std::size_t(2), clusterer.clusters().size());
}

BOOST_AUTO_TEST_CASE(testRemove) {
    // Test some edge cases: removing fails when the clusterer has no data or when
    // the incorrect index is specified. Also that remove removes the correct cluster
    // and merges its state with the nearest remaining cluster.

    TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight,
                              0.0001, // decay rate
                              0.05);  // minimum cluster fraction

    test::CRandomNumbers rng;
    TDoubleVec samples1;
    rng.generateNormalSamples(5.0, 1.0, 2000, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(15.0, 1.0, 200, samples2);

    TPointVec samples;
    for (std::size_t i = 0; i < samples1.size(); i += 2) {
        samples.emplace_back(TDoubleVec(&samples1[i], &samples1[i + 2]));
    }
    for (std::size_t i = 0; i < samples2.size(); i += 2) {
        samples.emplace_back(TDoubleVec(&samples2[i], &samples2[i + 2]));
    }
    rng.random_shuffle(samples.begin() + 5000, samples.end());

    BOOST_REQUIRE_EQUAL(false, clusterer.remove(0));

    for (const auto& sample : samples) {
        clusterer.add(sample);
        clusterer.propagateForwardsByTime(1.0);
    }

    BOOST_REQUIRE_EQUAL(2, clusterer.clusters().size());
    double count{clusterer.clusters()[0].count() + clusterer.clusters()[1].count()};

    BOOST_REQUIRE_EQUAL(false, clusterer.remove(3));

    BOOST_REQUIRE_EQUAL(true, clusterer.remove(1));

    BOOST_REQUIRE_EQUAL(1, clusterer.clusters().size());
    BOOST_REQUIRE_EQUAL(0, clusterer.clusters()[0].index());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(count, clusterer.clusters()[0].count(), 1e-3);

    BOOST_REQUIRE_EQUAL(false, clusterer.remove(0));
}

BOOST_AUTO_TEST_CASE(testLatLongData) {
    // Check that the log likelihood of the data in the lat_long.csv
    // is significantly increased by clustering.

    using TTimeDoubleVecPr = std::pair<core_t::TTime, TDoubleVec>;
    using TTimeDoubleVecPrVec = std::vector<TTimeDoubleVecPr>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    TTimeDoubleVecPrVec timeseries;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/lat_lng.csv", timeseries, test::CTimeSeriesTestData::CSV_UNIX_BIVALUED_REGEX));
    BOOST_TEST_REQUIRE(!timeseries.empty());

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    std::size_t n = timeseries.size();

    TCovariances2 reference(2);
    TXMeans2FloatForTest clusterer(maths_t::E_ContinuousData,
                                   maths_t::E_ClustersFractionWeight, 0.0005);

    for (std::size_t i = 0; i < n; ++i) {
        TPoint x(timeseries[i].second);
        reference.add(x);
        clusterer.add(x);
        clusterer.propagateForwardsByTime(1.0);
    }

    TMeanAccumulator LLR;
    TMeanAccumulator LLC;

    for (std::size_t i = 0; i < n; ++i) {
        TPoint x(timeseries[i].second);

        {
            TPoint mean = maths::CBasicStatistics::mean(reference);
            TMatrix covariance = maths::CBasicStatistics::covariances(reference);
            double ll;
            maths::gaussianLogLikelihood(covariance, x - mean, ll);
            LLR.add(ll);
        }

        {
            double ll = 0.0;
            double Z = 0.0;
            const TXMeans2FloatForTest::TClusterVec& clusters = clusterer.clusters();
            for (std::size_t j = 0; j < clusters.size(); ++j) {
                double w = maths::CBasicStatistics::count(clusters[j].covariances());
                TPoint mean = maths::CBasicStatistics::mean(clusters[j].covariances());
                TMatrix covariance =
                    maths::CBasicStatistics::covariances(clusters[j].covariances());
                double llj;
                maths::gaussianLogLikelihood(covariance, x - mean, llj);
                ll += w * std::exp(llj);
                Z += w;
                LOG_TRACE(<< "cluster " << j << " count " << clusters[j].count()
                          << " centre " << clusters[j].centre());
            }
            ll /= Z;
            LLC.add(std::log(ll));
        }
    }

    LOG_DEBUG(<< "gaussian log(L)  = " << maths::CBasicStatistics::mean(LLR));
    LOG_DEBUG(<< "clustered log(L) = " << maths::CBasicStatistics::mean(LLC));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(LLC) >
                       0.5 * maths::CBasicStatistics::mean(LLR));
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that persistence is idempotent.

    test::CRandomNumbers rng;

    double means[][2]{{10, 15}, {40, 10}, {12, 35}};
    double covariances[][2][2]{
        {{10, 2}, {2, 15}}, {{30, 8}, {8, 15}}, {{20, -11}, {-11, 25}}};

    TDoubleVecVec samples;
    TPointVec centres;

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        TDoubleVec mean(&means[i][0], &means[i][2]);
        TDoubleVecVec covariance;
        for (std::size_t j = 0; j < 2; ++j) {
            covariance.push_back(TDoubleVec(&covariances[i][j][0], &covariances[i][j][2]));
        }
        TDoubleVecVec samples_;
        rng.generateMultivariateNormalSamples(mean, covariance, 200, samples_);
        samples.insert(samples.end(), samples_.begin(), samples_.end());
    }
    rng.random_shuffle(samples.begin(), samples.end());

    TXMeans2ForTest clusterer(maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        clusterer.add(TPoint(samples[i]));
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        clusterer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Clusterer XML representation:\n" << origXml);

    // Restore the XML into a new clusterer.
    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, 0.15, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    maths::CXMeansOnline<double, 2> restoredClusterer(params, traverser);

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredClusterer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_SUITE_END()
