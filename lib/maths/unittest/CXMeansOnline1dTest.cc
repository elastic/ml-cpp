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

#include "CXMeansOnline1dTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CRestoreParams.h>
#include <maths/CXMeansOnline1d.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/range.hpp>

#include <algorithm>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TClusterVec = maths::CXMeansOnline1d::TClusterVec;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

bool restore(const maths::SDistributionRestoreParams& params,
             core::CRapidXmlStateRestoreTraverser& traverser,
             maths::CXMeansOnline1d::CCluster& result) {
    return traverser.traverseSubLevel(
        boost::bind(&maths::CXMeansOnline1d::CCluster::acceptRestoreTraverser, &result, boost::cref(params), _1));
}

void debug(const TClusterVec& clusters) {
    std::ostringstream c;
    c << "[";
    for (std::size_t j = 0u; j < clusters.size(); ++j) {
        c << " (" << clusters[j].weight(maths_t::E_ClustersFractionWeight) << ", " << clusters[j].centre() << ", " << clusters[j].spread()
          << ")";
    }
    c << " ]";
    LOG_DEBUG("clusters = " << c.str());
}
}

void CXMeansOnline1dTest::testCluster() {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testCluster  |");
    LOG_DEBUG("+------------------------------------+");

    maths::CXMeansOnline1d clusterer(
        maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.1);
    maths::CXMeansOnline1d::CCluster cluster(clusterer);

    double x1[] = {1.1, 2.3, 1.5, 0.9, 4.7, 3.2, 2.8, 2.3, 1.9, 2.6};
    double c1[] = {1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0};
    TDoubleVec values;

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator moments;
    for (std::size_t i = 0u; i < boost::size(x1); ++i) {
        cluster.add(x1[i], c1[i]);
        moments.add(x1[i], c1[i]);
        for (std::size_t j = 0u; j < static_cast<std::size_t>(c1[i]); ++j) {
            values.push_back(x1[i]);
        }
    }
    LOG_DEBUG("count  = " << cluster.count());
    LOG_DEBUG("centre = " << cluster.centre());
    LOG_DEBUG("spread = " << cluster.spread());
    LOG_DEBUG("weight = " << cluster.weight(maths_t::E_ClustersFractionWeight));

    double expectedCount = maths::CBasicStatistics::count(moments);
    double expectedCentre = maths::CBasicStatistics::mean(moments);
    double expectedSpread = std::sqrt(maths::CBasicStatistics::variance(moments));
    LOG_DEBUG("expected count  = " << expectedCount);
    LOG_DEBUG("expected centre = " << expectedCentre);
    LOG_DEBUG("expected spread = " << expectedSpread);
    CPPUNIT_ASSERT_EQUAL(expectedCount, cluster.count());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedCentre, cluster.centre(), 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedSpread, cluster.spread(), 0.05 * expectedSpread);
    CPPUNIT_ASSERT_EQUAL(1.0, cluster.weight(maths_t::E_ClustersEqualWeight));
    CPPUNIT_ASSERT_EQUAL(expectedCount, cluster.weight(maths_t::E_ClustersFractionWeight));

    cluster.propagateForwardsByTime(5.0);
    LOG_DEBUG("centre = " << cluster.centre());
    LOG_DEBUG("spread = " << cluster.spread());
    LOG_DEBUG("count  = " << cluster.count());
    LOG_DEBUG("weight = " << cluster.weight(maths_t::E_ClustersFractionWeight));
    CPPUNIT_ASSERT(cluster.count() < expectedCount);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedCentre, cluster.centre(), 1e-10);
    CPPUNIT_ASSERT(cluster.spread() > expectedSpread);
    CPPUNIT_ASSERT_EQUAL(1.0, cluster.weight(maths_t::E_ClustersEqualWeight));
    CPPUNIT_ASSERT(cluster.weight(maths_t::E_ClustersFractionWeight) < expectedCount);

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator percentileError;
    std::sort(values.begin(), values.end());
    for (std::size_t i = 0u; i < 10; ++i) {
        double p = static_cast<double>(10 * i) + 5.0;
        double expectedPercentile = values[static_cast<std::size_t>(p / 100.0 * static_cast<double>(values.size()) + 0.5)];
        LOG_DEBUG(p << " percentile = " << cluster.percentile(p));
        LOG_DEBUG(p << " expected percentile = " << expectedPercentile);
        double error = std::fabs(cluster.percentile(p) - expectedPercentile);
        CPPUNIT_ASSERT(error < 0.5);
        percentileError.add(error / expectedPercentile);
    }
    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(percentileError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(percentileError) < 0.1);

    TDoubleVec samples;
    cluster.sample(10, 0.0, 5.0, samples);
    LOG_DEBUG("samples = " << core::CContainerPrinter::print(samples));

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator sampleMoments;
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        sampleMoments.add(samples[i]);
    }
    double sampleCentre = maths::CBasicStatistics::mean(sampleMoments);
    double sampleSpread = std::sqrt(maths::CBasicStatistics::variance(sampleMoments));
    LOG_DEBUG("sample centre = " << sampleCentre);
    LOG_DEBUG("sample spread = " << sampleSpread);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cluster.centre(), sampleCentre, 0.02);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cluster.spread(), sampleSpread, 0.2);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(cluster.count()),
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

    LOG_DEBUG("Cluster XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::CXMeansOnline1d::CCluster restoredCluster(clusterer);
    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                             0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    restore(params, traverser, restoredCluster);
    uint64_t restoredChecksum = restoredCluster.checksum(0);
    CPPUNIT_ASSERT_EQUAL(origChecksum, restoredChecksum);

    double x2[] = {10.3, 10.6, 10.7, 9.8, 11.2, 11.0};
    double c2[] = {2.0, 1.0, 1.0, 2.0, 2.0, 1.0};
    for (std::size_t i = 0u; i < boost::size(x2); ++i) {
        cluster.add(x2[i], c2[i]);
    }
    maths::CXMeansOnline1d::TOptionalClusterClusterPr split =
        cluster.split(maths::CAvailableModeDistributions::ALL, 5.0, 0.0, std::make_pair(0.0, 15.0), clusterer.indexGenerator());
    CPPUNIT_ASSERT(split);
    LOG_DEBUG("left centre  = " << split->first.centre());
    LOG_DEBUG("left spread  = " << split->first.spread());
    LOG_DEBUG("right centre = " << split->second.centre());
    LOG_DEBUG("right spread = " << split->second.spread());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.4, split->first.centre(), 0.05);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.1, split->first.spread(), 0.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.5, split->second.centre(), 0.05);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6, split->second.spread(), 0.1);
}

void CXMeansOnline1dTest::testMixtureOfGaussians() {
    LOG_DEBUG("+-----------------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testMixtureOfGaussians  |");
    LOG_DEBUG("+-----------------------------------------------+");

    test::CRandomNumbers rng;

    // Test 1:
    //   * Cluster 1 = N(7, 1),     100 points
    //   * Cluster 2 = N(15, 2.25), 200 points
    //   * Cluster 3 = N(35, 2.25), 150 points
    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    {
        LOG_DEBUG("*** Test 1 ***");

        TDoubleVec mode1;
        rng.generateNormalSamples(7.0, 1.0, 100u, mode1);
        TDoubleVec mode2;
        rng.generateNormalSamples(16.0, 2.25, 200u, mode2);
        TDoubleVec mode3;
        rng.generateNormalSamples(35.0, 2.25, 150u, mode3);

        TMeanVarAccumulator expectedClusters[] = {TMeanVarAccumulator(), TMeanVarAccumulator(), TMeanVarAccumulator()};
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

        for (unsigned int i = 0u; i < 50u; ++i) {
            // Randomize the input order.
            rng.random_shuffle(samples.begin(), samples.end());

            maths::CXMeansOnline1d clusterer(
                maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.001);

            //std::ostringstream name;
            //name << "results.m." << i;
            //std::ofstream file;
            //file.open(name.str().c_str());

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                if (j % 50 == 0) {
                    LOG_DEBUG("time = " << j);
                }

                clusterer.add(samples[j], dummy);
                clusterer.propagateForwardsByTime(1.0);

                //if (j > 0 && j % 10 == 0)
                //{
                //    file << "x = [ ";
                //    for (std::size_t k = 0u; k < j; ++k)
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
            LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedClusters))

            LOG_DEBUG("# clusters = " << clusters.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(3), clusters.size());

            for (std::size_t j = 0u; j < clusters.size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre(), 0.1);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])), clusters[j].spread(), 0.4);
                meanError += std::fabs(clusters[j].centre() - maths::CBasicStatistics::mean(expectedClusters[j]));
                spreadError += std::fabs(clusters[j].spread() - std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
            }
        }

        meanError /= 150.0;
        spreadError /= 150.0;

        LOG_DEBUG("meanError = " << meanError << ", spreadError = " << spreadError);
        CPPUNIT_ASSERT(meanError < 0.012);
        CPPUNIT_ASSERT(meanError < 0.013);
    }

    // Test 2:
    //   * Cluster 1 = N(5, 1),    10 points
    //   * Cluster 2 = N(15, 2), 1600 points
    {
        LOG_DEBUG("*** Test 2 ***");

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

        maths::CXMeansOnline1d clusterer(
            maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.001);

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            if (j % 50 == 0) {
                LOG_DEBUG("time = " << j);
            }

            clusterer.add(samples[j], dummy);
            clusterer.propagateForwardsByTime(1.0);
        }

        // Check we've got one cluster (the small cluster should
        // have been deleted) and its mean and spread is what we'd
        // expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG("expected = " << expectedClusters);

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), clusters.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(expectedClusters), clusters[0].centre(), 0.05);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(std::sqrt(maths::CBasicStatistics::variance(expectedClusters)), clusters[0].spread(), 0.3);
    }

    // Test 3:
    //   * Cluster 1 = N(7, 1),  100 points
    //   * Cluster 2 = N(11, 1), 200 points
    {
        LOG_DEBUG("*** Test 3 ***");

        TDoubleVec mode1;
        rng.generateNormalSamples(7.0, 1.0, 100u, mode1);
        TDoubleVec mode2;
        rng.generateNormalSamples(11.0, 1.0, 200u, mode2);

        TMeanVarAccumulator expectedClusters[] = {TMeanVarAccumulator(), TMeanVarAccumulator()};
        expectedClusters[0].add(mode1);
        expectedClusters[1].add(mode2);

        TDoubleVec samples;
        samples.reserve(mode1.size() + mode2.size());
        std::copy(mode1.begin(), mode1.end(), std::back_inserter(samples));
        std::copy(mode2.begin(), mode2.end(), std::back_inserter(samples));

        double meanError = 0.0;
        double spreadError = 0.0;

        for (unsigned int i = 0u; i < 50u; ++i) {
            // Randomize the input order.
            rng.random_shuffle(samples.begin(), samples.end());

            maths::CXMeansOnline1d clusterer(
                maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.001);

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                if (j % 50 == 0) {
                    LOG_DEBUG("time = " << j);
                }

                clusterer.add(samples[j], dummy);
                clusterer.propagateForwardsByTime(1.0);
            }

            // Check we've got one cluster and its position
            // and spread is what we'd expect.
            const TClusterVec& clusters = clusterer.clusters();

            debug(clusters);

            CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusters.size());
            for (std::size_t j = 0u; j < clusters.size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre(), 0.4);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])), clusters[j].spread(), 0.3);
                meanError += std::fabs(clusters[j].centre() - maths::CBasicStatistics::mean(expectedClusters[j]));
                spreadError += std::fabs(clusters[j].spread() - std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
            }
        }

        meanError /= 100.0;
        spreadError /= 100.0;

        LOG_DEBUG("meanError = " << meanError << ", spreadError = " << spreadError);
        CPPUNIT_ASSERT(meanError < 0.14);
        CPPUNIT_ASSERT(spreadError < 0.11);
    }
}

void CXMeansOnline1dTest::testMixtureOfUniforms() {
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testMixtureOfUniforms  |");
    LOG_DEBUG("+----------------------------------------------+");

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
    for (unsigned int i = 0u; i < 50u; ++i) {
        // Randomize the input order.
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CXMeansOnline1d clusterer(
            maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.001);

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            if (j % 50 == 0) {
                LOG_DEBUG("time = " << j);
            }

            clusterer.add(samples[j], dummy);
        }

        // Check we've got two clusters and their position
        // and spread is what we'd expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedClusters));
        LOG_DEBUG("# clusters = " << clusters.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t j = 0u; j < clusters.size(); ++j) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])), clusters[j].spread(), 0.02);
            meanError += std::fabs(clusters[j].centre() - maths::CBasicStatistics::mean(expectedClusters[j]));
            spreadError += std::fabs(clusters[j].spread() - std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
        }
    }

    meanError /= 100.0;
    spreadError /= 100.0;

    LOG_DEBUG("meanError = " << meanError << ", spreadError = " << spreadError);
    CPPUNIT_ASSERT(meanError < 1e-5);
    CPPUNIT_ASSERT(spreadError < 0.01);
}

void CXMeansOnline1dTest::testMixtureOfLogNormals() {
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testMixtureOfLogNormals  |");
    LOG_DEBUG("+------------------------------------------------+");

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
    for (unsigned int i = 0u; i < 50u; ++i) {
        // Randomize the input order.
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CXMeansOnline1d clusterer(
            maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.001);

        //std::ostringstream name;
        //name << "results.m." << i;
        //std::ofstream file;
        //file.open(name.str().c_str());

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            if (j % 50 == 0) {
                LOG_DEBUG("time = " << j);
            }

            clusterer.add(samples[j], dummy);

            //if (j > 0 && j % 10 == 0)
            //{
            //    file << "x = [ ";
            //    for (std::size_t k = 0u; k < j; ++k)
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
        LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedClusters));
        LOG_DEBUG("# clusters = " << clusters.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t j = 0u; j < clusters.size(); ++j) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(expectedClusters[j]),
                                         clusters[j].centre(),
                                         0.03 * std::max(maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre()));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                clusters[j].spread(),
                0.5 * std::max(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])), clusters[j].spread()));
            meanError += std::fabs(clusters[j].centre() - maths::CBasicStatistics::mean(expectedClusters[j]));
            spreadError += std::fabs(clusters[j].spread() - std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
        }
    }

    meanError /= 100.0;
    spreadError /= 100.0;

    LOG_DEBUG("meanError = " << meanError << ", spreadError = " << spreadError);
    CPPUNIT_ASSERT(meanError < 0.1);
    CPPUNIT_ASSERT(spreadError < 0.14);
}

void CXMeansOnline1dTest::testOutliers() {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testOutliers  |");
    LOG_DEBUG("+-------------------------------------+");

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
    for (unsigned int i = 0u; i < 50u; ++i) {
        // Randomize the input order.
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight,
                                         0.001, // decay rate
                                         0.01); // mode fraction

        for (std::size_t j = 0u; j < outliers.size(); ++j) {
            clusterer.add(outliers[j], dummy);
        }

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            clusterer.add(samples[j], dummy);
        }

        // Check we've got two clusters and their position
        // and spread is what we'd expect.
        const TClusterVec& clusters = clusterer.clusters();

        debug(clusters);
        LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedClusters));
        LOG_DEBUG("# clusters = " << clusters.size());

        if (clusters.size() != 2)
            continue;

        n += 1.0;
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusters.size());

        for (std::size_t j = 0u; j < clusters.size(); ++j) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(expectedClusters[j]),
                                         clusters[j].centre(),
                                         0.01 * std::max(maths::CBasicStatistics::mean(expectedClusters[j]), clusters[j].centre()));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])),
                clusters[j].spread(),
                0.03 * std::max(std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])), clusters[j].spread()));
            meanError += std::fabs(clusters[j].centre() - maths::CBasicStatistics::mean(expectedClusters[j]));
            spreadError += std::fabs(clusters[j].spread() - std::sqrt(maths::CBasicStatistics::variance(expectedClusters[j])));
        }
    }

    meanError /= n;
    spreadError /= n;

    LOG_DEBUG("meanError = " << meanError << ", spreadError = " << spreadError << ", n = " << n);

    CPPUNIT_ASSERT(meanError < 0.15);
    CPPUNIT_ASSERT(spreadError < 1.0);
}

void CXMeansOnline1dTest::testManyClusters() {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testManyClusters  |");
    LOG_DEBUG("+-----------------------------------------+");

    using TTimeDoublePr = std::pair<core_t::TTime, double>;
    using TTimeDoublePrVec = std::vector<TTimeDoublePr>;

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    CPPUNIT_ASSERT(
        test::CTimeSeriesTestData::parse("testfiles/times.csv", timeseries, startTime, endTime, test::CTimeSeriesTestData::CSV_UNIX_REGEX));
    CPPUNIT_ASSERT(!timeseries.empty());

    LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10) << " ...");

    maths::CXMeansOnline1d clusterer(maths_t::E_IntegerData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight,
                                     0.001, // decay rate
                                     0.01,  // mode fraction
                                     2);    // mode count

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (std::size_t i = 0u; i < timeseries.size(); ++i) {
        core_t::TTime tow = timeseries[i].first % core::constants::WEEK;
        clusterer.add(static_cast<double>(tow), dummy);
    }

    // Check we've got ten clusters.

    const TClusterVec& clusters = clusterer.clusters();
    debug(clusters);
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), clusters.size());
}

void CXMeansOnline1dTest::testLowVariation() {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testLowVariation  |");
    LOG_DEBUG("+-----------------------------------------+");

    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight);

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (std::size_t i = 0u; i < 200; ++i) {
        clusterer.add(static_cast<double>(i % 2), dummy);
    }

    const TClusterVec& clusters = clusterer.clusters();
    debug(clusters);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusters.size());
}

void CXMeansOnline1dTest::testAdaption() {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testAdaption  |");
    LOG_DEBUG("+-------------------------------------+");

    // Test a case where the cluster pattern changes over time.
    // Specifically, the data set starts with one cluster then
    // a new cluster appears and subsequently disappears.

    // TODO
}

void CXMeansOnline1dTest::testLargeHistory() {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testLargeHistory  |");
    LOG_DEBUG("+-----------------------------------------+");

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

    clusterer.decayRate(0.0001);

    test::CRandomNumbers rng;
    TDoubleVec samples1;
    rng.generateNormalSamples(5.0, 1.0, 10000, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(15.0, 1.0, 100, samples2);

    TDoubleVec samples;
    samples.assign(samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    rng.random_shuffle(samples.begin() + 5000, samples.end());

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        reference.add(samples[i]);
        clusterer.add(samples[i]);
        reference.propagateForwardsByTime(1.0);
        clusterer.propagateForwardsByTime(1.0);
    }

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), reference.clusters().size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusterer.clusters().size());
}

void CXMeansOnline1dTest::testPersist() {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testPersist  |");
    LOG_DEBUG("+------------------------------------+");

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

    maths::CXMeansOnline1d clusterer(
        maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersEqualWeight, 0.05);

    maths::CXMeansOnline1d::TSizeDoublePr2Vec dummy;
    for (std::size_t j = 0u; j < samples.size(); ++j) {
        clusterer.add(samples[j], dummy);
        clusterer.propagateForwardsByTime(1.0);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        clusterer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("Clusterer XML representation:\n" << origXml);

    // Restore the XML into a new clusterer.
    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                             0.15,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
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
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CXMeansOnline1dTest::testPruneEmptyCluster() {
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CXMeansOnline1dTest::testPruneEmptyCluster  |");
    LOG_DEBUG("+----------------------------------------------+");

    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight);

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

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), clusterer.clusters().size());
    clusterer.prune();
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), clusterer.clusters().size());
}

CppUnit::Test* CXMeansOnline1dTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CXMeansOnline1dTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testCluster", &CXMeansOnline1dTest::testCluster));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testMixtureOfGaussians",
                                                                       &CXMeansOnline1dTest::testMixtureOfGaussians));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testMixtureOfUniforms",
                                                                       &CXMeansOnline1dTest::testMixtureOfUniforms));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testMixtureOfLogNormals",
                                                                       &CXMeansOnline1dTest::testMixtureOfLogNormals));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testOutliers", &CXMeansOnline1dTest::testOutliers));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testManyClusters", &CXMeansOnline1dTest::testManyClusters));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testLowVariation", &CXMeansOnline1dTest::testLowVariation));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testAdaption", &CXMeansOnline1dTest::testAdaption));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testLargeHistory", &CXMeansOnline1dTest::testLargeHistory));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testPersist", &CXMeansOnline1dTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXMeansOnline1dTest>("CXMeansOnline1dTest::testPruneEmptyCluster",
                                                                       &CXMeansOnline1dTest::testPruneEmptyCluster));

    return suiteOfTests;
}
