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

#include "CRandomProjectionClustererTest.h"

#include <maths/CLinearAlgebraTools.h>
#include <maths/CRandomProjectionClusterer.h>
#include <maths/CSetTools.h>

#include <test/CRandomNumbers.h>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TVector = maths::CVector<double>;
using TVector5 = maths::CVectorNx1<double, 5>;
using TCovariances = maths::CBasicStatistics::SSampleCovariances<double, 5>;

struct SFirstLess {
    bool operator()(const TSizeVec& lhs, const TSizeVec& rhs) const { return lhs[0] < rhs[0]; }
};

template<std::size_t N>
class CRandomProjectionClustererForTest : public maths::CRandomProjectionClustererBatch<N> {
public:
    using TVectorArrayVec = typename maths::CRandomProjectionClustererBatch<N>::TVectorArrayVec;
    using TDoubleVecVec = typename maths::CRandomProjectionClustererBatch<N>::TDoubleVecVec;
    using TVectorNx1VecVec = typename maths::CRandomProjectionClustererBatch<N>::TVectorNx1VecVec;
    using TSvdNxNVecVec = typename maths::CRandomProjectionClustererBatch<N>::TSvdNxNVecVec;
    using TSizeUSet = typename maths::CRandomProjectionClustererBatch<N>::TSizeUSet;
    using TMeanAccumulatorVecVec = typename maths::CRandomProjectionClustererBatch<N>::TMeanAccumulatorVecVec;

public:
    CRandomProjectionClustererForTest(double compression = 1.0) : maths::CRandomProjectionClustererBatch<N>(compression) {}

    const TVectorArrayVec& projections() const { return this->maths::CRandomProjectionClustererBatch<N>::projections(); }

    template<typename CLUSTERER>
    void clusterProjections(CLUSTERER clusterer, TDoubleVecVec& W, TVectorNx1VecVec& M, TSvdNxNVecVec& C, TSizeUSet& I) const {
        std::size_t b = this->projectedData().size();
        W.resize(b);
        M.resize(b);
        C.resize(b);
        this->maths::CRandomProjectionClustererBatch<N>::clusterProjections(clusterer, W, M, C, I);
    }

    void neighbourhoods(const TSizeUSet& I, TSizeVecVec& H) const { this->maths::CRandomProjectionClustererBatch<N>::neighbourhoods(I, H); }

    void
    similarities(const TDoubleVecVec& W, const TVectorNx1VecVec& M, const TSvdNxNVecVec& C, const TSizeVecVec& H, TDoubleVecVec& S) const {
        this->maths::CRandomProjectionClustererBatch<N>::similarities(W, M, C, H, S);
    }

    void clusterNeighbourhoods(TDoubleVecVec& S, const TSizeVecVec& H, TSizeVecVec& result) const {
        this->maths::CRandomProjectionClustererBatch<N>::clusterNeighbourhoods(S, H, result);
    }
};
}

void CRandomProjectionClustererTest::testGenerateProjections() {
    LOG_DEBUG("+-----------------------------------------------------------+");
    LOG_DEBUG("|  CRandomProjectionClustererTest::testGenerateProjections  |");
    LOG_DEBUG("+-----------------------------------------------------------+");

    using TVectorArrayVec = CRandomProjectionClustererForTest<5>::TVectorArrayVec;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    // Test corner case when projected dimension is greater
    // than the data dimension.

    {

        CRandomProjectionClustererForTest<5> clusterer;
        CPPUNIT_ASSERT(clusterer.initialise(5, 3));

        const TVectorArrayVec& projections = clusterer.projections();
        LOG_DEBUG("projections = " << core::CContainerPrinter::print(projections));

        CPPUNIT_ASSERT_EQUAL(std::string("[[[1 0 0], [0 1 0], [0 0 1], [0 0 0], [0 0 0]]]"), core::CContainerPrinter::print(projections));
    }

    // Test that the projections are mutually orthonormal and
    // approximately independent.

    TMeanAccumulator error;

    for (std::size_t t = 10; t < 50; ++t) {
        LOG_DEBUG("*** trial = " << t << " ***");

        CRandomProjectionClustererForTest<5> clusterer;

        CPPUNIT_ASSERT(clusterer.initialise(6, t));

        const TVectorArrayVec& projections = clusterer.projections();
        CPPUNIT_ASSERT_EQUAL(std::size_t(6), projections.size());

        for (std::size_t i = 0u; i < projections.size(); ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, projections[i][j].inner(projections[i][j]), 1e-10);

                for (std::size_t k = j + 1; k < 5; ++k) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, projections[i][j].inner(projections[i][k]), 1e-10);
                }
            }
        }

        TMeanVarAccumulator moments;

        for (std::size_t i = 0u; i < projections.size(); ++i) {
            for (std::size_t j = i + 1; j < projections.size(); ++j) {
                for (std::size_t k = 0u; k < 5; ++k) {
                    for (std::size_t l = 0u; l < 5; ++l) {
                        moments.add(projections[i][k].inner(projections[j][l]));
                    }
                }
            }
        }

        LOG_DEBUG("Expected variance = " << 1.0 / static_cast<double>(t));
        LOG_DEBUG("Actual variance   = " << maths::CBasicStatistics::variance(moments));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, maths::CBasicStatistics::mean(moments), 1.0 / static_cast<double>(t));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            1.0 / static_cast<double>(t), maths::CBasicStatistics::variance(moments), 0.2 / static_cast<double>(t));

        error.add(static_cast<double>(t) * std::fabs(maths::CBasicStatistics::variance(moments) - 1.0 / static_cast<double>(t)));
    }

    LOG_DEBUG("Relative error = " << 100.0 * maths::CBasicStatistics::mean(error) << "%");
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.06);
}

void CRandomProjectionClustererTest::testClusterProjections() {
    LOG_DEBUG("+----------------------------------------------------------+");
    LOG_DEBUG("|  CRandomProjectionClustererTest::testClusterProjections  |");
    LOG_DEBUG("+----------------------------------------------------------+");

    // Test that we get the cluster weights, means and covariance
    // matrices, and the sampled points we expect. Note that we
    // create a trivial to cluster data set since we don't want
    // to test clustering quality in this test.

    test::CRandomNumbers rng;
    std::size_t dimension = 30u;

    TDoubleVec mean1(dimension, 0.0);
    TDoubleVec mean2(dimension, 10.0);
    TDoubleVecVec covariance(dimension, TDoubleVec(dimension, 0.0));
    for (std::size_t i = 0u; i < 30; ++i) {
        covariance[i][i] = 1.0;
    }

    TDoubleVecVec samples1;
    TDoubleVecVec samples2;
    rng.generateMultivariateNormalSamples(mean1, covariance, 20, samples1);
    rng.generateMultivariateNormalSamples(mean2, covariance, 30, samples2);

    CRandomProjectionClustererForTest<5> clusterer;
    clusterer.initialise(4, dimension);
    for (std::size_t i = 0u; i < samples1.size(); ++i) {
        clusterer.add(TVector(samples1[i].begin(), samples1[i].end()));
    }
    for (std::size_t i = 0u; i < samples2.size(); ++i) {
        clusterer.add(TVector(samples2[i].begin(), samples2[i].end()));
    }

    TDoubleVec expectedWeights;
    CRandomProjectionClustererForTest<5>::TVectorNx1VecVec expectedMeans;
    expectedWeights.push_back(static_cast<double>(samples1.size()) / static_cast<double>(samples1.size() + samples2.size()));
    expectedWeights.push_back(static_cast<double>(samples2.size()) / static_cast<double>(samples1.size() + samples2.size()));
    std::sort(expectedWeights.begin(), expectedWeights.end());
    for (std::size_t i = 0u; i < clusterer.projections().size(); ++i) {
        CRandomProjectionClustererForTest<5>::TVectorNx1Vec means;
        {
            TCovariances covariances;
            for (std::size_t j = 0u; j < samples1.size(); ++j) {
                TVector x(samples1[j].begin(), samples1[j].end());
                TVector5 projection;
                for (std::size_t k = 0u; k < 5; ++k) {
                    projection(k) = clusterer.projections()[i][k].inner(x);
                }
                covariances.add(projection);
            }
            means.push_back(maths::CBasicStatistics::mean(covariances));
        }
        {
            TCovariances covariances;
            for (std::size_t j = 0u; j < samples2.size(); ++j) {
                TVector x(samples2[j].begin(), samples2[j].end());
                TVector5 projection;
                for (std::size_t k = 0u; k < 5; ++k) {
                    projection(k) = clusterer.projections()[i][k].inner(x);
                }
                covariances.add(projection);
            }
            means.push_back(maths::CBasicStatistics::mean(covariances));
        }
        std::sort(means.begin(), means.end());
        expectedMeans.push_back(means);
    }
    LOG_DEBUG("expected weights = " << core::CContainerPrinter::print(expectedWeights));
    LOG_DEBUG("expected means   = " << core::CContainerPrinter::print(expectedMeans));

    TDoubleVecVec weights_;
    CRandomProjectionClustererForTest<5>::TVectorNx1VecVec means;
    CRandomProjectionClustererForTest<5>::TSvdNxNVecVec covariances;
    CRandomProjectionClustererForTest<5>::TSizeUSet samples;
    clusterer.clusterProjections(
        maths::forRandomProjectionClusterer(maths::CKMeansFast<TVector5>(), 2, 5), weights_, means, covariances, samples);

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), weights_.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), means.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), covariances.size());
    CPPUNIT_ASSERT(samples.size() >= std::size_t(8));

    TDoubleVec weights(2, 0.0);
    for (std::size_t i = 0u; i < 4; ++i) {
        std::sort(weights_[i].begin(), weights_[i].end());
        weights[0] += weights_[i][0] / 4.0;
        weights[1] += weights_[i][1] / 4.0;
        std::sort(means[i].begin(), means[i].end());
    }
    LOG_DEBUG("weights = " << core::CContainerPrinter::print(weights));
    LOG_DEBUG("means   = " << core::CContainerPrinter::print(means));

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedWeights), core::CContainerPrinter::print(weights));
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMeans), core::CContainerPrinter::print(means));
}

void CRandomProjectionClustererTest::testNeighbourhoods() {
    LOG_DEBUG("+------------------------------------------------------+");
    LOG_DEBUG("|  CRandomProjectionClustererTest::testNeighbourhoods  |");
    LOG_DEBUG("+------------------------------------------------------+");

    // Test that the neighbourhoods for each point agree reasonably
    // accurately with the points nearest neighbours. The agreement
    // isn't perfect because we don't store the full points so are
    // computing distances projections.

    using TVectorVec = std::vector<TVector>;

    test::CRandomNumbers rng;

    std::size_t dimension = 30u;
    std::size_t n[] = {30, 50, 40};
    TDoubleVec means[3] = {};
    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        rng.generateUniformSamples(0.0, 10.0, dimension, means[i]);
        LOG_DEBUG("mean = " << core::CContainerPrinter::print(means[i]));
    }
    TDoubleVecVec covariances[] = {TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0)),
                                   TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0)),
                                   TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0))};
    for (std::size_t i = 0u; i < boost::size(covariances); ++i) {
        for (std::size_t j = 0u; j < 30; ++j) {
            covariances[i][j][j] = 1.0 + static_cast<double>(i);
        }
    }

    TVectorVec sampleVectors;

    CRandomProjectionClustererForTest<5> clusterer;
    clusterer.initialise(4, dimension);
    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(means[i], covariances[i], n[i], samples);
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            clusterer.add(TVector(samples[j]));
            sampleVectors.push_back(TVector(samples[j]));
        }
    }

    TDoubleVecVec weights;
    CRandomProjectionClustererForTest<5>::TVectorNx1VecVec clusterMeans;
    CRandomProjectionClustererForTest<5>::TSvdNxNVecVec clusterCovariances;
    CRandomProjectionClustererForTest<5>::TSizeUSet examples;
    clusterer.clusterProjections(
        maths::forRandomProjectionClusterer(maths::CKMeansFast<TVector5>(), 3, 5), weights, clusterMeans, clusterCovariances, examples);
    LOG_DEBUG("examples = " << core::CContainerPrinter::print(examples));

    TSizeVecVec neighbourhoods(examples.size());
    clusterer.neighbourhoods(examples, neighbourhoods);

    TSizeVecVec expectedNeighbourhoods(examples.size());

    TVectorVec exampleVectors;
    for (auto i = examples.begin(); i != examples.end(); ++i) {
        LOG_DEBUG("example = " << sampleVectors[*i]);
        exampleVectors.push_back(sampleVectors[*i]);
    }
    for (std::size_t i = 0u; i < sampleVectors.size(); ++i) {
        std::size_t closest = 0u;
        double distance = (sampleVectors[i] - exampleVectors[0]).euclidean();
        for (std::size_t j = 1u; j < exampleVectors.size(); ++j) {
            double dj = (sampleVectors[i] - exampleVectors[j]).euclidean();
            if (dj < distance) {
                closest = j;
                distance = dj;
            }
        }
        expectedNeighbourhoods[closest].push_back(i);
    }

    LOG_DEBUG("neighbours          = " << core::CContainerPrinter::print(neighbourhoods));
    LOG_DEBUG("expected neighbours = " << core::CContainerPrinter::print(expectedNeighbourhoods));

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanJaccard;
    for (std::size_t i = 0u; i < neighbourhoods.size(); ++i) {
        double jaccard = maths::CSetTools::jaccard(
            neighbourhoods[i].begin(), neighbourhoods[i].end(), expectedNeighbourhoods[i].begin(), expectedNeighbourhoods[i].end());
        LOG_DEBUG("jaccard = " << jaccard);
        meanJaccard.add(jaccard, static_cast<double>(expectedNeighbourhoods[i].size()));
        CPPUNIT_ASSERT(jaccard > 0.1);
    }

    LOG_DEBUG("mean jaccard = " << maths::CBasicStatistics::mean(meanJaccard));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanJaccard) > 0.35);
}

void CRandomProjectionClustererTest::testSimilarities() {
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CRandomProjectionClustererTest::testSimilarities  |");
    LOG_DEBUG("+----------------------------------------------------+");

    test::CRandomNumbers rng;

    std::size_t dimension = 30u;
    std::size_t n[] = {30, 50, 40};
    TDoubleVec means[3] = {};
    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        rng.generateUniformSamples(0.0, 10.0, dimension, means[i]);
        LOG_DEBUG("mean = " << core::CContainerPrinter::print(means[i]));
    }
    TDoubleVecVec covariances[] = {TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0)),
                                   TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0)),
                                   TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0))};
    for (std::size_t i = 0u; i < boost::size(covariances); ++i) {
        for (std::size_t j = 0u; j < 30; ++j) {
            covariances[i][j][j] = 1.0 + static_cast<double>(i);
        }
    }

    TSizeVec clusters;

    CRandomProjectionClustererForTest<5> clusterer(1.5);
    clusterer.initialise(4, dimension);
    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(means[i], covariances[i], n[i], samples);
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            clusterer.add(TVector(samples[j]));
            clusters.push_back(i);
        }
    }

    TDoubleVecVec weights;
    CRandomProjectionClustererForTest<5>::TVectorNx1VecVec clusterMeans;
    CRandomProjectionClustererForTest<5>::TSvdNxNVecVec clusterCovariances;
    CRandomProjectionClustererForTest<5>::TSizeUSet examples;
    clusterer.clusterProjections(
        maths::forRandomProjectionClusterer(maths::CKMeansFast<TVector5>(), 3, 5), weights, clusterMeans, clusterCovariances, examples);
    LOG_DEBUG("examples = " << core::CContainerPrinter::print(examples));

    TSizeVecVec expectedConnectivity(examples.size(), TSizeVec(examples.size()));
    TSizeVec examples_(examples.begin(), examples.end());
    for (std::size_t i = 0u; i < examples_.size(); ++i) {
        for (std::size_t j = 0u; j <= i; ++j) {
            expectedConnectivity[i][j] = expectedConnectivity[j][i] = clusters[examples_[i]] == clusters[examples_[j]] ? 1 : 0;
        }
    }
    LOG_DEBUG("expected connectivity =");
    for (std::size_t i = 0u; i < expectedConnectivity.size(); ++i) {
        LOG_DEBUG("  " << core::CContainerPrinter::print(expectedConnectivity[i]));
    }

    TSizeVecVec neighbourhoods(examples.size());
    clusterer.neighbourhoods(examples, neighbourhoods);

    TDoubleVecVec similarities(examples.size());
    clusterer.similarities(weights, clusterMeans, clusterCovariances, neighbourhoods, similarities);

    TSizeVecVec connectivity(examples.size(), TSizeVec(examples.size()));
    for (std::size_t i = 0u; i < similarities.size(); ++i) {
        TDoubleVec s;
        for (std::size_t j = 0u; j <= i; ++j) {
            s.push_back(similarities[i][j]);
            connectivity[i][j] = connectivity[j][i] = similarities[i][j] < 10.0 ? 1 : 0;
        }
        LOG_DEBUG(core::CContainerPrinter::print(s));
    }
    LOG_DEBUG("connectivity =");
    for (std::size_t i = 0u; i < connectivity.size(); ++i) {
        LOG_DEBUG("  " << core::CContainerPrinter::print(connectivity[i]));
    }

    for (std::size_t i = 0u; i < expectedConnectivity.size(); ++i) {
        for (std::size_t j = 0u; j <= i; ++j) {
            CPPUNIT_ASSERT_EQUAL(expectedConnectivity[i][j], connectivity[i][j]);
        }
    }
}

void CRandomProjectionClustererTest::testClusterNeighbourhoods() {
    LOG_DEBUG("+-------------------------------------------------------------+");
    LOG_DEBUG("|  CRandomProjectionClustererTest::testClusterNeighbourhoods  |");
    LOG_DEBUG("+-------------------------------------------------------------+");

    // Test we recover the true clusters.

    test::CRandomNumbers rng;

    std::size_t dimension = 30u;
    std::size_t n[] = {30, 50, 40};
    TDoubleVec means[3] = {};
    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        rng.generateUniformSamples(0.0, 10.0, dimension, means[i]);
        LOG_DEBUG("mean = " << core::CContainerPrinter::print(means[i]));
    }
    TDoubleVecVec covariances[] = {TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0)),
                                   TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0)),
                                   TDoubleVecVec(dimension, TDoubleVec(dimension, 0.0))};
    for (std::size_t i = 0u; i < boost::size(covariances); ++i) {
        for (std::size_t j = 0u; j < 30; ++j) {
            covariances[i][j][j] = 1.0 + static_cast<double>(i);
        }
    }

    TSizeVec clusters;

    CRandomProjectionClustererForTest<5> clusterer(1.5);
    clusterer.initialise(4, dimension);
    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(means[i], covariances[i], n[i], samples);
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            clusterer.add(TVector(samples[j]));
            clusters.push_back(i);
        }
    }

    TDoubleVecVec weights;
    CRandomProjectionClustererForTest<5>::TVectorNx1VecVec clusterMeans;
    CRandomProjectionClustererForTest<5>::TSvdNxNVecVec clusterCovariances;
    CRandomProjectionClustererForTest<5>::TSizeUSet examples;
    clusterer.clusterProjections(
        maths::forRandomProjectionClusterer(maths::CKMeansFast<TVector5>(), 3, 5), weights, clusterMeans, clusterCovariances, examples);
    LOG_DEBUG("examples = " << core::CContainerPrinter::print(examples));

    TSizeVecVec neighbourhoods(examples.size());
    clusterer.neighbourhoods(examples, neighbourhoods);

    TDoubleVecVec similarities(examples.size());
    clusterer.similarities(weights, clusterMeans, clusterCovariances, neighbourhoods, similarities);

    TSizeVecVec expectedClustering(boost::size(n));
    LOG_DEBUG("expected clustering =");
    for (std::size_t i = 0u, j = 0u; i < boost::size(n); ++i) {
        for (std::size_t ni = j + n[i]; j < ni; ++j) {
            expectedClustering[i].push_back(j);
        }
        LOG_DEBUG("  " << core::CContainerPrinter::print(expectedClustering[i]));
    }

    TSizeVecVec clustering;
    clusterer.clusterNeighbourhoods(similarities, neighbourhoods, clustering);

    for (std::size_t i = 0u; i < clustering.size(); ++i) {
        std::sort(clustering[i].begin(), clustering[i].end());
    }
    std::sort(clustering.begin(), clustering.end(), SFirstLess());

    LOG_DEBUG("clustering =");
    for (std::size_t i = 0u; i < clustering.size(); ++i) {
        LOG_DEBUG("  " << core::CContainerPrinter::print(clustering[i]));
    }

    for (std::size_t i = 0u; i < expectedClustering.size(); ++i) {
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedClustering[i]), core::CContainerPrinter::print(clustering[i]));
    }
}

void CRandomProjectionClustererTest::testAccuracy() {
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CRandomProjectionClustererTest::testAccuracy  |");
    LOG_DEBUG("+------------------------------------------------+");
}

CppUnit::Test* CRandomProjectionClustererTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CRandomProjectionClustererTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CRandomProjectionClustererTest>(
        "CRandomProjectionClustererTest::testGenerateProjections", &CRandomProjectionClustererTest::testGenerateProjections));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRandomProjectionClustererTest>("CRandomProjectionClustererTest::testClusterProjections",
                                                                                  &CRandomProjectionClustererTest::testClusterProjections));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRandomProjectionClustererTest>("CRandomProjectionClustererTest::testNeighbourhoods",
                                                                                  &CRandomProjectionClustererTest::testNeighbourhoods));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRandomProjectionClustererTest>("CRandomProjectionClustererTest::testSimilarities",
                                                                                  &CRandomProjectionClustererTest::testSimilarities));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRandomProjectionClustererTest>(
        "CRandomProjectionClustererTest::testClusterNeighbourhoods", &CRandomProjectionClustererTest::testClusterNeighbourhoods));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRandomProjectionClustererTest>("CRandomProjectionClustererTest::testAccuracy",
                                                                                  &CRandomProjectionClustererTest::testAccuracy));

    return suiteOfTests;
}
