/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include "CModelToolsTest.h"

#include <core/CLogger.h>
#include <core/CSmallVector.h>

#include <maths/CMultimodalPrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>
#include <maths/CXMeansOnline1d.h>

#include <model/CModelTools.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDouble2Vec = core::CSmallVector<double, 2>;

const double MINIMUM_SEASONAL_SCALE{0.25};
const double DECAY_RATE{0.0005};
const std::size_t TAG{0u};

maths::CModelParams params(core_t::TTime bucketLength) {
    using TTimeDoubleMap = std::map<core_t::TTime, double>;
    static TTimeDoubleMap learnRates;
    learnRates[bucketLength] = static_cast<double>(bucketLength) / 1800.0;
    double minimumSeasonalVarianceScale{MINIMUM_SEASONAL_SCALE};
    return maths::CModelParams{bucketLength, learnRates[bucketLength], DECAY_RATE, minimumSeasonalVarianceScale};
}

maths::CNormalMeanPrecConjugate normal() {
    return maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, DECAY_RATE);
}

maths::CMultimodalPrior multimodal() {
    maths::CXMeansOnline1d clusterer{
        maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, DECAY_RATE};
    return maths::CMultimodalPrior{maths_t::E_ContinuousData, clusterer, normal(), DECAY_RATE};
}
}

void CModelToolsTest::testFuzzyDeduplicate() {
    LOG_DEBUG(<< "*** CModelToolsTest::testFuzzyDeduplicate ***");

    test::CRandomNumbers rng;

    TDoubleVec values;
    TDoubleVec uniques;
    values.reserve(200000);
    uniques.reserve(30000);

    LOG_DEBUG(<< "Normal");
    for (auto variance : {1.0, 10.0, 100.0, 1000.0}) {
        rng.generateNormalSamples(0.0, variance, 200000, values);

        model::CModelTools::CFuzzyDeduplicate fuzzy;
        for (auto value : values) {
            fuzzy.add(TDouble2Vec{value});
        }
        fuzzy.computeEpsilons(600, 10000);

        boost::math::normal normal{variance, std::sqrt(variance)};
        double eps{(boost::math::quantile(normal, 0.9) - boost::math::quantile(normal, 0.1)) / 10000.0};
        LOG_DEBUG(<< "eps = " << eps);

        uniques.clear();
        for (auto value : values) {
            std::size_t duplicate{fuzzy.duplicate(300, TDouble2Vec{value})};
            if (duplicate < uniques.size()) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(uniques[duplicate], value, 1.5 * eps);
            } else {
                uniques.push_back(value);
            }
        }

        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        CPPUNIT_ASSERT(uniques.size() < 25000);
    }

    LOG_DEBUG(<< "Uniform");
    for (auto range : {1.0, 10.0, 100.0, 1000.0}) {
        rng.generateUniformSamples(0.0, range, 200000, values);

        model::CModelTools::CFuzzyDeduplicate fuzzy;
        for (auto value : values) {
            fuzzy.add(TDouble2Vec{value});
        }
        fuzzy.computeEpsilons(600, 10000);

        double eps{0.8 * range / 10000.0};
        LOG_DEBUG(<< "eps = " << eps);

        uniques.clear();
        for (auto value : values) {
            std::size_t duplicate{fuzzy.duplicate(300, TDouble2Vec{value})};
            if (duplicate < uniques.size()) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(uniques[duplicate], value, 1.5 * eps);
            } else {
                uniques.push_back(value);
            }
        }

        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        CPPUNIT_ASSERT(uniques.size() < 15000);
    }

    LOG_DEBUG(<< "Sorted Uniform");
    {
        rng.generateUniformSamples(0.0, 100.0, 200000, values);
        std::sort(values.begin(), values.end());

        model::CModelTools::CFuzzyDeduplicate fuzzy;
        for (auto value : values) {
            fuzzy.add(TDouble2Vec{value});
        }
        fuzzy.computeEpsilons(600, 10000);

        double eps{80.0 / 10000.0};
        LOG_DEBUG(<< "eps = " << eps);

        uniques.clear();
        for (auto value : values) {
            std::size_t duplicate{fuzzy.duplicate(300, TDouble2Vec{value})};
            if (duplicate < uniques.size()) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(uniques[duplicate], value, 1.5 * eps);
            } else {
                uniques.push_back(value);
            }
        }

        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        CPPUNIT_ASSERT(uniques.size() < 15000);
    }

    LOG_DEBUG(<< "Log-Normal");
    for (auto variance : {1.0, 2.0, 4.0, 8.0}) {
        rng.generateLogNormalSamples(variance, variance, 200000, values);

        model::CModelTools::CFuzzyDeduplicate fuzzy;
        for (auto value : values) {
            fuzzy.add(TDouble2Vec{value});
        }
        fuzzy.computeEpsilons(600, 10000);

        boost::math::lognormal lognormal{variance, std::sqrt(variance)};
        double eps{(boost::math::quantile(lognormal, 0.9) - boost::math::quantile(lognormal, 0.1)) / 10000.0};
        LOG_DEBUG(<< "eps = " << eps);

        uniques.clear();
        for (auto value : values) {
            std::size_t duplicate{fuzzy.duplicate(300, TDouble2Vec{value})};
            if (duplicate < uniques.size()) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(uniques[duplicate], value, 1.5 * eps);
            } else {
                uniques.push_back(value);
            }
        }

        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        CPPUNIT_ASSERT(uniques.size() < 30000);
    }
}

void CModelToolsTest::testProbabilityCache() {
    LOG_DEBUG(<< "*** CModelToolsTest::testProbabilityCache ***");

    using TBool2Vec = core::CSmallVector<bool, 2>;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
    using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
    using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
    using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
    using TDouble2Vec4VecVec = std::vector<TDouble2Vec4Vec>;
    using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    core_t::TTime bucketLength{1800};

    maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
    maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, multimodal(), nullptr, false};
    test::CRandomNumbers rng;

    core_t::TTime time_{0};
    TDouble2Vec4Vec weight{TDouble2Vec{1.0}};
    TDouble2Vec4VecVec weights{weight};

    {
        TDoubleVec samples_[3];
        rng.generateNormalSamples(0.0, 1.0, 100, samples_[0]);
        rng.generateNormalSamples(20.0, 3.0, 300, samples_[1]);
        rng.generateNormalSamples(29.0, 1.0, 100, samples_[2]);
        TDoubleVec samples;
        samples.insert(samples.end(), samples_[0].begin(), samples_[0].end());
        samples.insert(samples.end(), samples_[1].begin(), samples_[1].end());
        samples.insert(samples.end(), samples_[2].begin(), samples_[2].end());
        rng.random_shuffle(samples.begin(), samples.end());
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false)
                .propagationInterval(1.0)
                .weightStyles(maths::CConstantWeights::COUNT)
                .trendWeights(weights)
                .priorWeights(weights);
            model.addSamples(params, {core::make_triple(time_, TDouble2Vec(1, sample), TAG)});
        }
    }

    model_t::EFeature feature{model_t::E_IndividualMeanByPerson};
    std::size_t id{0u};
    TTime2Vec1Vec time{TTime2Vec{time_}};
    TDouble2Vec1Vec sample{TDouble2Vec{0.0}};

    LOG_DEBUG(<< "Test Random");

    for (std::size_t t = 0; t < 5; ++t) {
        TDoubleVec samples_[3];
        rng.generateNormalSamples(0.0, 1.0, 10000, samples_[0]);
        rng.generateNormalSamples(20.0, 3.0, 10000, samples_[1]);
        rng.generateNormalSamples(10.0, 1000.0, 100, samples_[2]);
        TDoubleVec samples;
        samples.insert(samples.end(), samples_[0].begin(), samples_[0].end());
        samples.insert(samples.end(), samples_[1].begin(), samples_[1].end());
        samples.insert(samples.end(), samples_[2].begin(), samples_[2].end());
        rng.random_shuffle(samples.begin(), samples.end());
        LOG_DEBUG(<< "# samples = " << samples.size());

        model::CModelTools::CProbabilityCache cache(0.05);
        TMeanAccumulator error;
        std::size_t hits{0u};

        for (auto sample_ : samples) {
            sample[0][0] = sample_;

            maths::CModelProbabilityParams params;
            params.addCalculation(maths_t::E_TwoSided)
                .seasonalConfidenceInterval(0.0)
                .addBucketEmpty(TBool2Vec{false})
                .weightStyles(maths::CConstantWeights::COUNT)
                .addWeights(weight);
            double expectedProbability;
            TTail2Vec expectedTail;
            bool conditional;
            TSize1Vec mostAnomalousCorrelate;
            model.probability(params, time, sample, expectedProbability, expectedTail, conditional, mostAnomalousCorrelate);

            double probability;
            TTail2Vec tail;
            if (cache.lookup(feature, id, sample, probability, tail, conditional, mostAnomalousCorrelate)) {
                ++hits;
                error.add(std::fabs(probability - expectedProbability) / expectedProbability);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProbability, probability, 0.05 * expectedProbability);
                CPPUNIT_ASSERT_EQUAL(expectedTail[0], tail[0]);
                CPPUNIT_ASSERT_EQUAL(false, conditional);
                CPPUNIT_ASSERT(mostAnomalousCorrelate.empty());
            } else {
                cache.addModes(feature, id, model);
                cache.addProbability(feature, id, sample, expectedProbability, expectedTail, false, mostAnomalousCorrelate);
            }
        }

        LOG_DEBUG(<< "hits = " << hits);
        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(hits > 19000);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.001);
    }

    LOG_DEBUG(<< "Test Adversary");
    {
        model::CModelTools::CProbabilityCache cache(0.05);
        for (auto sample_ : {0.5, 1.4, 23.0, 26.0, 20.0}) {
            sample[0][0] = sample_;

            maths::CModelProbabilityParams params;
            params.addCalculation(maths_t::E_TwoSided)
                .seasonalConfidenceInterval(0.0)
                .addBucketEmpty(TBool2Vec{false})
                .weightStyles(maths::CConstantWeights::COUNT)
                .addWeights(weight);
            double expectedProbability;
            TTail2Vec expectedTail;
            bool conditional;
            TSize1Vec mostAnomalousCorrelate;
            model.probability(params, time, sample, expectedProbability, expectedTail, conditional, mostAnomalousCorrelate);
            LOG_DEBUG(<< "probability = " << expectedProbability << ", tail = " << expectedTail);

            double probability;
            TTail2Vec tail;
            if (cache.lookup(feature, id, sample, probability, tail, conditional, mostAnomalousCorrelate)) {
                // Shouldn't have any cache hits.
                CPPUNIT_ASSERT(false);
            } else {
                cache.addModes(feature, id, model);
                cache.addProbability(feature, id, sample, expectedProbability, expectedTail, false, mostAnomalousCorrelate);
            }
        }
    }
}

CppUnit::Test* CModelToolsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CModelToolsTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CModelToolsTest>("CModelToolsTest::testFuzzyDeduplicate", &CModelToolsTest::testFuzzyDeduplicate));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CModelToolsTest>("CModelToolsTest::testProbabilityCache", &CModelToolsTest::testProbabilityCache));

    return suiteOfTests;
}
