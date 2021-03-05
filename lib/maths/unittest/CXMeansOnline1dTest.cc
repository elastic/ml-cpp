/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CRestoreParams.h>
#include <maths/CXMeansOnline1d.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>

BOOST_AUTO_TEST_SUITE(CXMeansOnline1dTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TClusterVec = maths::CXMeansOnline1d::TClusterVec;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

bool restore(const maths::SDistributionRestoreParams& params,
             core::CRapidXmlStateRestoreTraverser& traverser,
             maths::CXMeansOnline1d::CCluster& result) {
    return traverser.traverseSubLevel(
        std::bind(&maths::CXMeansOnline1d::CCluster::acceptRestoreTraverser,
                  &result, std::cref(params), std::placeholders::_1));
}

void debug(const TClusterVec& clusters) {
    std::ostringstream c;
    c << "[";
    for (std::size_t j = 0; j < clusters.size(); ++j) {
        c << " (" << clusters[j].weight(maths_t::E_ClustersFractionWeight)
          << ", " << clusters[j].centre() << ", " << clusters[j].spread() << ")";
    }
    c << " ]";
    LOG_DEBUG(<< "clusters = " << c.str());
}
}

BOOST_AUTO_TEST_CASE(testCluster) {
    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight, 0.1);
    maths::CXMeansOnline1d::CCluster cluster(clusterer);

    double x1[] = {1.1, 2.3, 1.5, 0.9, 4.7, 3.2, 2.8, 2.3, 1.9, 2.6};
    double c1[] = {1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0};
    TDoubleVec values;

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator moments;
    for (std::size_t i = 0; i < boost::size(x1); ++i) {
        cluster.add(x1[i], c1[i]);
        moments.add(x1[i], c1[i]);
        for (std::size_t j = 0; j < static_cast<std::size_t>(c1[i]); ++j) {
            values.push_back(x1[i]);
        }
    }
    LOG_DEBUG(<< "count  = " << cluster.count());
    LOG_DEBUG(<< "centre = " << cluster.centre());
    LOG_DEBUG(<< "spread = " << cluster.spread());
    LOG_DEBUG(<< "weight = " << cluster.weight(maths_t::E_ClustersFractionWeight));

    double expectedCount = maths::CBasicStatistics::count(moments);
    double expectedCentre = maths::CBasicStatistics::mean(moments);
    double expectedSpread = std::sqrt(maths::CBasicStatistics::variance(moments));
    LOG_DEBUG(<< "expected count  = " << expectedCount);
    LOG_DEBUG(<< "expected centre = " << expectedCentre);
    LOG_DEBUG(<< "expected spread = " << expectedSpread);
    BOOST_REQUIRE_EQUAL(expectedCount, cluster.count());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedCentre, cluster.centre(), 5e-7);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedSpread, cluster.spread(), 0.05 * expectedSpread);
    BOOST_REQUIRE_EQUAL(1.0, cluster.weight(maths_t::E_ClustersEqualWeight));
    BOOST_REQUIRE_EQUAL(expectedCount, cluster.weight(maths_t::E_ClustersFractionWeight));

    cluster.propagateForwardsByTime(5.0);
    LOG_DEBUG(<< "centre = " << cluster.centre());
    LOG_DEBUG(<< "spread = " << cluster.spread());
    LOG_DEBUG(<< "count  = " << cluster.count());
    LOG_DEBUG(<< "weight = " << cluster.weight(maths_t::E_ClustersFractionWeight));
    BOOST_TEST_REQUIRE(cluster.count() < expectedCount);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedCentre, cluster.centre(), 5e-7);
    BOOST_TEST_REQUIRE(cluster.spread() > expectedSpread);
    BOOST_REQUIRE_EQUAL(1.0, cluster.weight(maths_t::E_ClustersEqualWeight));
    BOOST_TEST_REQUIRE(cluster.weight(maths_t::E_ClustersFractionWeight) < expectedCount);

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator percentileError;
    std::sort(values.begin(), values.end());
    for (std::size_t i = 0; i < 10; ++i) {
        double p = static_cast<double>(10 * i) + 5.0;
        double expectedPercentile = values[static_cast<std::size_t>(
            p / 100.0 * static_cast<double>(values.size()) + 0.5)];
        LOG_DEBUG(<< p << " percentile = " << cluster.percentile(p));
        LOG_DEBUG(<< p << " expected percentile = " << expectedPercentile);
        double error = std::fabs(cluster.percentile(p) - expectedPercentile);
        BOOST_TEST_REQUIRE(error < 0.5);
        percentileError.add(error / expectedPercentile);
    }
    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(percentileError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(percentileError) < 0.1);

    TDoubleVec samples;
    cluster.sample(10, 0.0, 5.0, samples);
    LOG_DEBUG(<< "samples = " << core::CContainerPrinter::print(samples));

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator sampleMoments;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        sampleMoments.add(samples[i]);
    }
    double sampleCentre = maths::CBasicStatistics::mean(sampleMoments);
    double sampleSpread = std::sqrt(maths::CBasicStatistics::variance(sampleMoments));
    LOG_DEBUG(<< "sample centre = " << sampleCentre);
    LOG_DEBUG(<< "sample spread = " << sampleSpread);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(cluster.centre(), sampleCentre, 0.02);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(cluster.spread(), sampleSpread, 0.2);

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        std::log(cluster.count()),
        -cluster.logLikelihoodFromCluster(maths_t::E_ClustersEqualWeight, 1.5) +
            cluster.logLikelihoodFromCluster(maths_t::E_ClustersFractionWeight, 1.5),
        1e-10);

    uint64_t origChecksum = cluster.checksum(0);
    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        cluster.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Cluster XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::CXMeansOnline1d::CCluster restoredCluster(clusterer);
    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    restore(params, traverser, restoredCluster);
    uint64_t restoredChecksum = restoredCluster.checksum(0);
    BOOST_REQUIRE_EQUAL(origChecksum, restoredChecksum);

    double x2[] = {10.3, 10.6, 10.7, 9.8, 11.2, 11.0};
    double c2[] = {2.0, 1.0, 1.0, 2.0, 2.0, 1.0};
    for (std::size_t i = 0; i < boost::size(x2); ++i) {
        cluster.add(x2[i], c2[i]);
    }
    maths::CXMeansOnline1d::TOptionalClusterClusterPr split =
        cluster.split(maths::CAvailableModeDistributions::ALL, 5.0, 0.0,
                      std::make_pair(0.0, 15.0), clusterer.indexGenerator());
    BOOST_TEST_REQUIRE(split.has_value());
    LOG_DEBUG(<< "left centre  = " << split->first.centre());
    LOG_DEBUG(<< "left spread  = " << split->first.spread());
    LOG_DEBUG(<< "right centre = " << split->second.centre());
    LOG_DEBUG(<< "right spread = " << split->second.spread());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.4, split->first.centre(), 0.05);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(1.1, split->first.spread(), 0.1);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(10.5, split->second.centre(), 0.05);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6, split->second.spread(), 0.1);
}

BOOST_AUTO_TEST_CASE(testMixtureOfGaussians) {
    test::CRandomNumbers rng;

    // Test 1:
    //   * Cluster 1 = N(7, 1),     100 points
    //   * Cluster 2 = N(16, 2.25), 200 points
    //   * Cluster 3 = N(35, 2.25), 150 points
    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    {
        LOG_DEBUG(<< "*** Test 1 ***");

        TDoubleVec mode1;
        rng.generateNormalSamples(7.0, 1.0, 100u, mode1);
        TDoubleVec mode2;
        rng.generateNormalSamples(16.0, 2.25, 200u, mode2);
        TDoubleVec mode3;
        rng.generateNormalSamples(35.0, 2.25, 150u, mode3);

        TMeanVarAccumulator expectedClusters[] = {
            TMeanVarAccumulator(), TMeanVarAccumulator(), TMeanVarAccumulator()};
        expectedClusters[0].add(mode1);
        expectedClusters[1].add(mode2);
        expectedClusters[2].add(mode3);

        TDoubleVec samples;
        samples.reserve(mode1.size() + mode2.size() + mode3.size());
        std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
        std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));
        std::copy(mode3.begin(), mode3.end(), std::back_inserter(samples));

        double meanError = 0.0;
        double spreadError = 0.0;

        for (unsigned int i = 0; i < 50; ++i) {
            // Randomize the input order.
            rng.random_shuffle(samples.begin(), samples.end());

            maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                             maths::CAvailableModeDistributions::ALL,
                                             maths_t::E_ClustersFractionWeight, 0.001);

            //std::ostringstream name;
            //name << "results.m." << i;
            //std::ofstream file;
            //file.open(name.str().c_str());

            for (std::size_t j = 0; j < samples.size(); ++j) {
                if (j % 50 == 0) {
                    LOG_DEBUG(<< "time = " << j);
                }

                clusterer.add(samples[j], dummy);
                clusterer.propagateForwardsByTime(1.0);

                //if (j > 0 && j % 10 == 0)
                //{
                //    file << "x = [ ";
                //    for (std::size_t k = 0; k < j; ++k)
                //    {
                //        file << samples[k] << " ";
                //    }
                //    file << "];\nscatter(x, zeros(1, length(x)), 10, 'r', 'x');\nhold on;\n";
                //    file << clusterer.printClusters();
                //    file << "axis([0 45 0 0.21])\n";
                //    file << "input(\"time = " << j << "\");\n\n";
                //}
            }

            // Check we've got three clusters and their position
            // and spread is what we'd expect.
            const TClusterVec& clusters = clusterer.clusters();

            debug(clusters);
            LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expectedClusters));

            LOG_DEBUG(<< "# clusters = " << clusters.size());
            BOOST_REQUIRE_EQUAL(std::size_t(3), clusters.size());

            for (std::size_t j = 0; j < clusters.size(); ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::mean(expectedClusters[j]),
                    clusters[j].centre(), 0.1);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                    clusters[j].spread(), 0.4);
                meanError += std::fabs(clusters[j].centre() -
                                       maths::CBasicStatistics::mean(expectedClusters[j]));
                spreadError += std::fabs(
                    clusters[j].spread() -
                    std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
            }
        }

        meanError /= 150.0;
        spreadError /= 150.0;

        LOG_DEBUG(<< "meanError = " << meanError << ", spreadError = " << spreadError);
        BOOST_TEST_REQUIRE(meanError < 0.012);
        BOOST_TEST_REQUIRE(meanError < 0.013);
    }

    // Test 2:
    //   * Cluster 1 = N(5, 1),    10 points
    //   * Cluster 2 = N(15, 2), 1600 points
    {
        LOG_DEBUG(<< "*** Test 2 ***");

        TDoubleVec mode1;
        rng.generateNormalSamples(5.0, 1.0, 10u, mode1);
        TDoubleVec mode2;
        rng.generateNormalSamples(15.0, 2.0, 1600u, mode2);

        TMeanVarAccumulator expectedClusters;
        expectedClusters.add(mode2);

        TDoubleVec samples;
        samples.reserve(mode1.size() + mode2.size());
        std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
        std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));

        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight, 0.001);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            if (j % 50 == 0) {
                LOG_DEBUG(<< "time = " << j);
            }

            clusterer.add(samples[j], dummy);
            clusterer.propagateForwardsByTime(1.0);
        }

        // Check we've got one cluster (the small cluster should
        // have been deleted) and its mean and spread is what we'd
        // expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG(<< "expected = " << expectedClusters);

        BOOST_REQUIRE_EQUAL(std::size_t(1), clusters.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(expectedClusters),
                                     clusters[0].centre(), 0.05);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            std::sqrt(maths::CBasicStatistics::variance(expectedClusters)),
            clusters[0].spread(), 0.3);
    }

    // Test 3:
    //   * Cluster 1 = N(7, 1),  100 points
    //   * Cluster 2 = N(11, 1), 200 points
    {
        LOG_DEBUG(<< "*** Test 3 ***");

        TDoubleVec mode1;
        rng.generateNormalSamples(7.0, 1.0, 100u, mode1);
        TDoubleVec mode2;
        rng.generateNormalSamples(11.0, 1.0, 200u, mode2);

        TMeanVarAccumulator expectedClusters[] = {TMeanVarAccumulator(),
                                                  TMeanVarAccumulator()};
        expectedClusters[0].add(mode1);
        expectedClusters[1].add(mode2);

        TDoubleVec samples;
        samples.reserve(mode1.size() + mode2.size());
        std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
        std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));

        double meanError = 0.0;
        double spreadError = 0.0;

        for (unsigned int i = 0; i < 50; ++i) {
            // Randomize the input order.
            rng.random_shuffle(samples.begin(), samples.end());

            maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                             maths::CAvailableModeDistributions::ALL,
                                             maths_t::E_ClustersFractionWeight, 0.001);

            for (std::size_t j = 0; j < samples.size(); ++j) {
                if (j % 50 == 0) {
                    LOG_DEBUG(<< "time = " << j);
                }

                clusterer.add(samples[j], dummy);
                clusterer.propagateForwardsByTime(1.0);
            }

            // Check we've got one cluster and its position
            // and spread is what we'd expect.
            const TClusterVec& clusters = clusterer.clusters();

            debug(clusters);

            BOOST_REQUIRE_EQUAL(std::size_t(2), clusters.size());
            for (std::size_t j = 0; j < clusters.size(); ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::mean(expectedClusters[j]),
                    clusters[j].centre(), 0.4);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                    clusters[j].spread(), 0.3);
                meanError += std::fabs(clusters[j].centre() -
                                       maths::CBasicStatistics::mean(expectedClusters[j]));
                spreadError += std::fabs(
                    clusters[j].spread() -
                    std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
            }
        }

        meanError /= 100.0;
        spreadError /= 100.0;

        LOG_DEBUG(<< "meanError = " << meanError << ", spreadError = " << spreadError);
        BOOST_TEST_REQUIRE(meanError < 0.14);
        BOOST_TEST_REQUIRE(spreadError < 0.11);
    }
}

BOOST_AUTO_TEST_CASE(testMixtureOfUniforms) {
    test::CRandomNumbers rng;

    // * Cluster 1 = U([12, 15]), 100 points
    // * Cluster 2 = U([25, 30]), 200 points
    TDoubleVec mode1;
    rng.generateUniformSamples(12.0, 15.0, 100u, mode1);
    TDoubleVec mode2;
    rng.generateUniformSamples(25.0, 30.0, 200u, mode2);

    TMeanVarAccumulator expectedClusters[] = {TMeanVarAccumulator(), TMeanVarAccumulator()};
    expectedClusters[0].add(mode1);
    expectedClusters[1].add(mode2);

    TDoubleVec samples;
    samples.reserve(mode1.size() + mode2.size());
    std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
    std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));

    double meanError = 0.0;
    double spreadError = 0.0;

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (unsigned int i = 0; i < 50; ++i) {
        // Randomize the input order.
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight, 0.001);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            if (j % 50 == 0) {
                LOG_DEBUG(<< "time = " << j);
            }

            clusterer.add(samples[j], dummy);
        }

        // Check we've got two clusters and their position
        // and spread is what we'd expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expectedClusters));
        LOG_DEBUG(<< "# clusters = " << clusters.size());
        BOOST_REQUIRE_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t j = 0; j < clusters.size(); ++j) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(expectedClusters[j]),
                                         clusters[j].centre(), 0.01);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                clusters[j].spread(), 0.02);
            meanError += std::fabs(clusters[j].centre() -
                                   maths::CBasicStatistics::mean(expectedClusters[j]));
            spreadError += std::fabs(
                clusters[j].spread() -
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
        }
    }

    meanError /= 100.0;
    spreadError /= 100.0;

    LOG_DEBUG(<< "meanError = " << meanError << ", spreadError = " << spreadError);
    BOOST_TEST_REQUIRE(meanError < 1e-5);
    BOOST_TEST_REQUIRE(spreadError < 0.01);
}

BOOST_AUTO_TEST_CASE(testMixtureOfLogNormals) {
    test::CRandomNumbers rng;

    // * Cluster 1 = LogNormal(3, 0.01), 100 points
    // * Cluster 2 = LogNormal(4, 0.01), 200 points
    TDoubleVec mode1;
    rng.generateLogNormalSamples(3.4, 0.01, 150u, mode1);
    TDoubleVec mode2;
    rng.generateLogNormalSamples(4.0, 0.01, 100u, mode2);

    TMeanVarAccumulator expectedClusters[] = {TMeanVarAccumulator(), TMeanVarAccumulator()};
    expectedClusters[0].add(mode1);
    expectedClusters[1].add(mode2);

    TDoubleVec samples;
    samples.reserve(mode1.size() + mode2.size());
    std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
    std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));

    double meanError = 0.0;
    double spreadError = 0.0;

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (unsigned int i = 0; i < 50; ++i) {
        // Randomize the input order.
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight, 0.001);

        //std::ostringstream name;
        //name << "results.m." << i;
        //std::ofstream file;
        //file.open(name.str().c_str());

        for (std::size_t j = 0; j < samples.size(); ++j) {
            if (j % 50 == 0) {
                LOG_DEBUG(<< "time = " << j);
            }

            clusterer.add(samples[j], dummy);

            //if (j > 0 && j % 10 == 0)
            //{
            //    file << "x = [ ";
            //    for (std::size_t k = 0; k < j; ++k)
            //    {
            //        file << samples[k] << " ";
            //    }
            //    file << "];\nscatter(x, zeros(1, length(x)), 10, 'r', 'x');\nhold on;\n";
            //    file << clusterer.printClusters();
            //    file << "axis([10 80 0 0.15])\n";
            //    file << "input(\"time = " << j << "\");\n\n";
            //}
        }

        // Check we've got two clusters and their position
        // and spread is what we'd expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expectedClusters));
        LOG_DEBUG(<< "# clusters = " << clusters.size());
        BOOST_REQUIRE_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t j = 0; j < clusters.size(); ++j) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre(),
                0.03 * std::max(maths::CBasicStatistics::mean(expectedClusters[j]),
                                clusters[j].centre()));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                clusters[j].spread(),
                0.5 * std::max(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                               clusters[j].spread()));
            meanError += std::fabs(clusters[j].centre() -
                                   maths::CBasicStatistics::mean(expectedClusters[j]));
            spreadError += std::fabs(
                clusters[j].spread() -
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
        }
    }

    meanError /= 100.0;
    spreadError /= 100.0;

    LOG_DEBUG(<< "meanError = " << meanError << ", spreadError = " << spreadError);
    BOOST_TEST_REQUIRE(meanError < 0.1);
    BOOST_TEST_REQUIRE(spreadError < 0.14);
}

BOOST_AUTO_TEST_CASE(testOutliers) {
    test::CRandomNumbers rng;

    TDoubleVec mode1;
    rng.generateNormalSamples(7.0, 1.0, 50u, mode1);
    TDoubleVec mode2;
    rng.generateNormalSamples(18.0, 1.0, 50u, mode2);
    TDoubleVec outliers(7u, 2000.0);

    TMeanVarAccumulator expectedClusters[] = {TMeanVarAccumulator(), TMeanVarAccumulator()};
    expectedClusters[0].add(mode1);
    expectedClusters[1].add(mode2);
    expectedClusters[1].add(outliers);

    TDoubleVec samples;
    samples.reserve(mode1.size() + mode2.size());
    std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
    std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));

    double meanError = 0.0;
    double spreadError = 0.0;
    double n = 0.0;

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (unsigned int i = 0; i < 50; ++i) {
        // Randomize the input order.
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight,
                                         0.001, // decay rate
                                         0.01); // mode fraction

        for (std::size_t j = 0; j < outliers.size(); ++j) {
            clusterer.add(outliers[j], dummy);
        }

        for (std::size_t j = 0; j < samples.size(); ++j) {
            clusterer.add(samples[j], dummy);
        }

        // Check we've got two clusters and their position
        // and spread is what we'd expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expectedClusters));
        LOG_DEBUG(<< "# clusters = " << clusters.size());

        if (clusters.size() != 2)
            continue;

        n += 1.0;
        BOOST_REQUIRE_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t j = 0; j < clusters.size(); ++j) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre(),
                0.01 * std::max(maths::CBasicStatistics::mean(expectedClusters[j]),
                                clusters[j].centre()));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                clusters[j].spread(),
                0.03 * std::max(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                                clusters[j].spread()));
            meanError += std::fabs(clusters[j].centre() -
                                   maths::CBasicStatistics::mean(expectedClusters[j]));
            spreadError += std::fabs(
                clusters[j].spread() -
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
        }
    }

    meanError /= n;
    spreadError /= n;

    LOG_DEBUG(<< "meanError = " << meanError
              << ", spreadError = " << spreadError << ", n = " << n);

    BOOST_TEST_REQUIRE(meanError < 0.15);
    BOOST_TEST_REQUIRE(spreadError < 1.0);
}

BOOST_AUTO_TEST_CASE(testManyClusters) {
    using TTimeDoublePr = std::pair<core_t::TTime, double>;
    using TTimeDoublePrVec = std::vector<TTimeDoublePr>;

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/times.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
    BOOST_TEST_REQUIRE(!timeseries.empty());

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    maths::CXMeansOnline1d clusterer(maths_t::E_IntegerData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight,
                                     0.001, // decay rate
                                     0.01,  // mode fraction
                                     2);    // mode count

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime tow = timeseries[i].first % core::constants::WEEK;
        clusterer.add(static_cast<double>(tow), dummy);
    }

    // Check we've got ten clusters.

    const TClusterVec& clusters = clusterer.clusters();
    debug(clusters);
    BOOST_REQUIRE_EQUAL(std::size_t(10), clusters.size());
}

BOOST_AUTO_TEST_CASE(testLowVariation) {
    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight);

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (std::size_t i = 0; i < 200; ++i) {
        clusterer.add(static_cast<double>(i % 2), dummy);
    }

    const TClusterVec& clusters = clusterer.clusters();
    debug(clusters);
    BOOST_REQUIRE_EQUAL(std::size_t(2), clusters.size());
}

BOOST_AUTO_TEST_CASE(testAdaption) {
    // Test a case where the cluster pattern changes over time.
    // Specifically, the data set starts with one cluster then
    // a new cluster appears and subsequently disappears.

    test::CRandomNumbers rng;

    TDoubleVec mode1;
    rng.generateNormalSamples(3000.0, 10000.0, 100u, mode1);
    TDoubleVec mode2a;
    rng.generateNormalSamples(3000.0, 10000.0, 100u, mode2a);
    TDoubleVec mode2b;
    rng.generateNormalSamples(4000.0, 90000.0, 100u, mode2b);
    TDoubleVec mode3;
    rng.generateNormalSamples(5000.0, 10000.0, 1000u, mode3);

    TDoubleVec mode2;
    mode2.reserve(mode2a.size() + mode2b.size());
    std::copy(mode2a.begin(), mode2a.end(), std::back_inserter(mode2));
    std::copy(mode2b.begin(), mode2b.end(), std::back_inserter(mode2));
    rng.random_shuffle(mode2.begin(), mode2.end());

    TDoubleVec samples;
    samples.reserve(mode1.size() + mode2.size() + mode3.size());
    std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
    std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));
    std::copy(mode3.begin(), mode3.end(), std::back_inserter(samples));

    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight, 0.01);

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;

    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;

    std::size_t i{0};
    auto addSamplesToClusterer = [&](const std::size_t& limit,
                                     const TDoubleVecVec& expectedModes) {
        TMeanVarAccumulatorVec expectedClusters;
        for (const auto& mode : expectedModes) {
            expectedClusters.push_back(TMeanVarAccumulator());
            expectedClusters.back().add(mode);
        }
        for (; i < limit; ++i) {
            clusterer.add(static_cast<double>(samples[i]), dummy);
            clusterer.propagateForwardsByTime(1.0);
        }

        const TClusterVec& clusters = clusterer.clusters();
        debug(clusters);
        BOOST_REQUIRE_EQUAL(expectedClusters.size(), clusters.size());

        for (std::size_t j = 0; j < clusters.size(); ++j) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre(),
                0.01 * std::max(maths::CBasicStatistics::mean(expectedClusters[j]),
                                clusters[j].centre()));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                clusters[j].spread(),
                0.04 * std::max(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                                clusters[j].spread()));
        }
    };

    addSamplesToClusterer(mode1.size(), {mode1});
    addSamplesToClusterer(mode1.size() + mode2.size(), {mode2a, mode2b});
    addSamplesToClusterer(samples.size(), {mode3});
}

BOOST_AUTO_TEST_CASE(testLargeHistory) {
    // If we get a lot of history, because we detect that the system
    // is stable and reduce the decay rate then we should also reduce
    // the fraction of points required to create a cluster.

    maths::CXMeansOnline1d reference(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight,
                                     0.001, // decay rate
                                     0.05); // minimum cluster fraction
    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight,
                                     0.001, // decay rate
                                     0.05); // minimum cluster fraction

    // Set the decay rate to simulate decay rate control.
    clusterer.decayRate(0.0001);

    test::CRandomNumbers rng;
    TDoubleVec samples1;
    rng.generateNormalSamples(5.0, 1.0, 10000, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(15.0, 1.0, 100, samples2);

    TDoubleVec samples(samples1);
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    rng.random_shuffle(samples.begin() + 5000, samples.end());

    for (const auto& sample : samples) {
        for (std::size_t i = 0; i < 3; ++i) {
            reference.add(sample);
            clusterer.add(sample);
        }
        reference.propagateForwardsByTime(1.0);
        clusterer.propagateForwardsByTime(1.0);
    }

    BOOST_REQUIRE_EQUAL(1, reference.clusters().size());
    BOOST_REQUIRE_EQUAL(2, clusterer.clusters().size());
}

BOOST_AUTO_TEST_CASE(testRemove) {
    // Test some edge cases: removing fails when the clusterer has no data or when
    // the incorrect index is specified. Also that remove removes the correct cluster
    // and merges its state with the nearest remaining cluster.

    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight,
                                     0.001, // decay rate
                                     0.05); // minimum cluster fraction

    test::CRandomNumbers rng;
    TDoubleVec samples1;
    rng.generateNormalSamples(5.0, 1.0, 1000, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(15.0, 1.0, 100, samples2);

    TDoubleVec samples(samples1);
    samples.insert(samples.end(), samples2.begin(), samples2.end());
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

BOOST_AUTO_TEST_CASE(testPersist) {
    test::CRandomNumbers rng;

    TDoubleVec mode1;
    rng.generateNormalSamples(7.0, 1.0, 100, mode1);

    TDoubleVec mode2;
    rng.generateNormalSamples(15.0, 2.25, 200, mode2);

    TDoubleVec mode3;
    rng.generateNormalSamples(35.0, 2.25, 150, mode3);

    TDoubleVec samples;
    samples.reserve(mode1.size() + mode2.size() + mode3.size());
    std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
    std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));
    std::copy(mode3.begin(), mode3.end(), std::back_inserter(samples));

    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersEqualWeight, 0.05);

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (std::size_t j = 0; j < samples.size(); ++j) {
        clusterer.add(samples[j], dummy);
        clusterer.propagateForwardsByTime(1.0);
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
    maths::CXMeansOnline1d restoredClusterer(params, traverser);

    // The XML representation of the new filter should be the same
    // as the original.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredClusterer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testPruneEmptyCluster) {
    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight);

    maths::CXMeansOnline1d::CCluster cluster1(clusterer);
    cluster1.add(1.0, 12.0);
    cluster1.add(1.1, 2.0);
    cluster1.add(1.2, 3.0);
    cluster1.add(1.3, 16.0);
    cluster1.add(1.4, 6.0);
    cluster1.add(1.5, 1.0);
    cluster1.add(1.6, 3.0);
    clusterer.m_Clusters.push_back(cluster1);

    maths::CXMeansOnline1d::CCluster cluster2(clusterer);
    cluster2.add(4.4, 15.0);
    cluster2.add(4.5, 2.0);
    cluster2.add(4.6, 14.0);
    cluster2.add(4.7, 5.0);
    cluster2.add(4.8, 3.0);
    cluster2.add(4.9, 1.0);

    clusterer.m_Clusters.push_back(cluster2);

    maths::CXMeansOnline1d::CCluster cluster_empty(clusterer);
    clusterer.m_Clusters.push_back(cluster_empty);

    BOOST_REQUIRE_EQUAL(std::size_t(4), clusterer.clusters().size());
    clusterer.prune();
    BOOST_REQUIRE_EQUAL(std::size_t(2), clusterer.clusters().size());
}

BOOST_AUTO_TEST_SUITE_END()
