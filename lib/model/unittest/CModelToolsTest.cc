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
#include <core/CSmallVector.h>
#include <core/Constants.h>

#include <maths/common/CMultimodalPrior.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/CXMeansOnline1d.h>

#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesModel.h>

#include <model/CModelTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/test/unit_test.hpp>

#include <map>

BOOST_AUTO_TEST_SUITE(CModelToolsTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TTimeVec = std::vector<core_t::TTime>;

const double MINIMUM_SEASONAL_SCALE{0.25};
const double DECAY_RATE{0.0005};
const std::size_t TAG{0u};

maths::common::CModelParams params(core_t::TTime bucketLength) {
    using TTimeDoubleMap = std::map<core_t::TTime, double>;
    static TTimeDoubleMap learnRates;
    learnRates[bucketLength] = static_cast<double>(bucketLength) / 1800.0;
    double minimumSeasonalVarianceScale{MINIMUM_SEASONAL_SCALE};
    return maths::common::CModelParams{bucketLength,
                                       learnRates[bucketLength],
                                       DECAY_RATE,
                                       minimumSeasonalVarianceScale,
                                       6 * core::constants::HOUR,
                                       24 * core::constants::HOUR};
}

maths::common::CNormalMeanPrecConjugate normal() {
    return maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE);
}

maths::common::CMultimodalPrior multimodal() {
    maths::common::CXMeansOnline1d clusterer{
        maths_t::E_ContinuousData, maths::common::CAvailableModeDistributions::ALL,
        maths_t::E_ClustersFractionWeight, DECAY_RATE};
    return maths::common::CMultimodalPrior{maths_t::E_ContinuousData, clusterer,
                                           normal(), DECAY_RATE};
}
}

BOOST_AUTO_TEST_CASE(testFuzzyDeduplicate) {
    // Test de-duplication and fuzzy de-duplication.
    //
    // We shouldn't have repeated values and no value should be further
    // from its fuzzy duplicate than an expected tolerance.
    //
    // Finally, we test we get good reduction in the number of values
    // when fuzzily de-duplicating over a range of data distributions.

    test::CRandomNumbers rng;

    auto fuzzyUniqueValues = [&](const TDoubleVec& values, double q80, TDoubleVec& uniques) {
        std::size_t desiredSamples{10000};
        double tolerance{1.5 * q80 / static_cast<double>(desiredSamples)};
        LOG_DEBUG(<< "tolerance = " << tolerance);

        model::CModelTools::CFuzzyDeduplicate duplicates;
        for (auto value : values) {
            duplicates.add({value});
        }
        duplicates.computeEpsilons(600 /*bucketLength*/, desiredSamples);

        uniques.clear();
        for (auto value : values) {
            std::size_t duplicate{duplicates.duplicate(300, {value})};
            if (duplicate < uniques.size()) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(uniques[duplicate], value, tolerance);
            } else {
                uniques.push_back(value);
            }
        }
    };

    LOG_DEBUG(<< "Duplicates");
    {
        model::CModelTools::CFuzzyDeduplicate duplicates;
        TTimeVec times{300, 300, 300, 900, 300, 300, 300};
        TDoubleVec values{1.0, 1.0, 4.0, 1.0, 3.0, 1.2, 25.0};
        TDoubleVec uniques;
        for (std::size_t i = 0; i < times.size(); ++i) {
            std::size_t duplicate{duplicates.duplicate(times[i], {values[i]})};
            if (duplicate == uniques.size()) {
                uniques.push_back(values[i]);
            } else {
                BOOST_REQUIRE_EQUAL(values[i], uniques[duplicate]);
            }
        }
        BOOST_REQUIRE_EQUAL(std::string("[1, 4, 1, 3, 1.2, 25]"),
                            core::CContainerPrinter::print(uniques));
    }

    TDoubleVec values;
    TDoubleVec uniques;
    values.reserve(200000);
    uniques.reserve(30000);

    LOG_DEBUG(<< "Normal");
    for (auto variance : {1.0, 10.0, 100.0, 1000.0}) {
        rng.generateNormalSamples(0.0, variance, 200000, values);
        boost::math::normal normal{variance, std::sqrt(variance)};
        double q80{(boost::math::quantile(normal, 0.9) - boost::math::quantile(normal, 0.1))};
        fuzzyUniqueValues(values, q80, uniques);
        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        BOOST_TEST_REQUIRE(uniques.size() < 25000);
    }

    LOG_DEBUG(<< "Uniform");
    for (auto range : {1.0, 10.0, 100.0, 1000.0}) {
        rng.generateUniformSamples(0.0, range, 200000, values);
        double q80{0.8 * range};
        fuzzyUniqueValues(values, q80, uniques);
        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        BOOST_TEST_REQUIRE(uniques.size() < 15000);
    }

    LOG_DEBUG(<< "Sorted Uniform");
    {
        rng.generateUniformSamples(0.0, 100.0, 200000, values);
        std::sort(values.begin(), values.end());
        double q80{80.0};
        fuzzyUniqueValues(values, q80, uniques);
        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        BOOST_TEST_REQUIRE(uniques.size() < 15000);
    }

    LOG_DEBUG(<< "Log-Normal");
    for (auto variance : {1.0, 2.0, 4.0, 8.0}) {
        rng.generateLogNormalSamples(variance, variance, 200000, values);
        boost::math::lognormal lognormal{variance, std::sqrt(variance)};
        double q80{boost::math::quantile(lognormal, 0.9) -
                   boost::math::quantile(lognormal, 0.1)};
        fuzzyUniqueValues(values, q80, uniques);
        LOG_DEBUG(<< "Got " << uniques.size() << "/" << values.size());
        BOOST_TEST_REQUIRE(uniques.size() < 30000);
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityCache) {
    // Test the error introduced by caching the probability and that we
    // don't get any errors in the tailness we calculate for the value.

    using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
    using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
    using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
    using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;
    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;

    core_t::TTime bucketLength{1800};

    maths::time_series::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
    maths::time_series::CUnivariateTimeSeriesModel model{
        params(bucketLength), 0, trend, multimodal(), nullptr, nullptr, false};
    test::CRandomNumbers rng;

    core_t::TTime time_{0};
    TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};

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
            maths::common::CModelAddSamplesParams params;
            params.isInteger(false)
                .propagationInterval(1.0)
                .trendWeights(weights)
                .priorWeights(weights);
            model.addSamples(
                params, {core::make_triple(time_, TDouble2Vec(1, sample), TAG)});
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

            maths::common::CModelProbabilityParams params;
            params.addCalculation(maths_t::E_TwoSided)
                .seasonalConfidenceInterval(0.0)
                .addWeights(weights[0]);
            maths::common::SModelProbabilityResult expectedResult;
            model.probability(params, time, sample, expectedResult);

            maths::common::SModelProbabilityResult result;
            if (cache.lookup(feature, id, sample, result)) {
                ++hits;
                error.add(std::fabs(result.s_Probability - expectedResult.s_Probability) /
                          expectedResult.s_Probability);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedResult.s_Probability,
                                             result.s_Probability,
                                             0.05 * expectedResult.s_Probability);
                BOOST_REQUIRE_EQUAL(expectedResult.s_Tail[0], result.s_Tail[0]);
                BOOST_REQUIRE_EQUAL(false, result.s_Conditional);
                BOOST_TEST_REQUIRE(result.s_MostAnomalousCorrelate.empty());
            } else {
                cache.addModes(feature, id, model);
                cache.addProbability(feature, id, sample, expectedResult);
            }
        }

        LOG_DEBUG(<< "hits = " << hits);
        LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(hits > 19000);
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.001);
    }

    LOG_DEBUG(<< "Test Adversary");
    {
        model::CModelTools::CProbabilityCache cache(0.05);
        for (auto sample_ : {0.5, 1.4, 23.0, 26.0, 20.0}) {
            sample[0][0] = sample_;

            maths::common::CModelProbabilityParams params;
            params.addCalculation(maths_t::E_TwoSided)
                .seasonalConfidenceInterval(0.0)
                .addWeights(weights[0]);
            maths::common::SModelProbabilityResult expectedResult;
            model.probability(params, time, sample, expectedResult);
            LOG_DEBUG(<< "probability = " << expectedResult.s_Probability
                      << ", tail = " << expectedResult.s_Tail);

            maths::common::SModelProbabilityResult result;
            if (cache.lookup(feature, id, sample, result)) {
                // Shouldn't have any cache hits.
                BOOST_TEST_REQUIRE(false);
            } else {
                cache.addModes(feature, id, model);
                cache.addProbability(feature, id, sample, expectedResult);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
