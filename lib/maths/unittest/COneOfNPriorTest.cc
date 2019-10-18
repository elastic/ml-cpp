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

#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPoissonMeanConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/CXMeansOnline1d.h>

#include "TestUtils.h"

#include <test/CRandomNumbers.h>
#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <memory>
#include <numeric>

BOOST_AUTO_TEST_SUITE(COneOfNPriorTest)

using namespace ml;
using namespace handy_typedefs;

namespace {

using TUIntVec = std::vector<unsigned int>;
using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TPriorPtr = std::unique_ptr<maths::CPrior>;
using TPriorPtrVec = std::vector<TPriorPtr>;
using TOptionalDouble = boost::optional<double>;
using CGammaRateConjugate = CPriorTestInterfaceMixin<maths::CGammaRateConjugate>;
using CLogNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CLogNormalMeanPrecConjugate>;
using CMultimodalPrior = CPriorTestInterfaceMixin<maths::CMultimodalPrior>;
using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;
using COneOfNPrior = CPriorTestInterfaceMixin<maths::COneOfNPrior>;
using CPoissonMeanConjugate = CPriorTestInterfaceMixin<maths::CPoissonMeanConjugate>;
using TWeightFunc = maths_t::TDoubleWeightsAry (*)(double);

COneOfNPrior::TPriorPtrVec clone(const TPriorPtrVec& models,
                                 const TOptionalDouble& decayRate = TOptionalDouble()) {
    COneOfNPrior::TPriorPtrVec result;
    result.reserve(models.size());
    for (std::size_t i = 0u; i < models.size(); ++i) {
        result.push_back(TPriorPtr(models[i]->clone()));
        if (decayRate) {
            result.back()->decayRate(*decayRate);
        }
    }
    return result;
}

void truncateUpTo(const double& value, TDoubleVec& samples) {
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        samples[i] = std::max(samples[i], value);
    }
}

double sum(const TDoubleVec& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

using maths_t::E_ContinuousData;
using maths_t::E_IntegerData;
}

BOOST_AUTO_TEST_CASE(testFilter) {
    TPriorPtrVec models;
    models.push_back(TPriorPtr(
        maths::CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(maths::CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                   .clone()));
    models.push_back(TPriorPtr(
        maths::CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(
        TPriorPtr(maths::CPoissonMeanConjugate::nonInformativePrior().clone()));

    COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(10.0, 3.0, 20, samples);

    // Make sure we don't have negative values.
    truncateUpTo(0.0, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    filter.removeModels(maths::CPrior::CModelFilter().remove(maths::CPrior::E_Constant));

    BOOST_CHECK_EQUAL(std::size_t(4), filter.models().size());

    filter.removeModels(
        maths::CPrior::CModelFilter().remove(maths::CPrior::E_Poisson).remove(maths::CPrior::E_Gamma));

    BOOST_CHECK_EQUAL(std::size_t(2), filter.models().size());
    BOOST_CHECK_EQUAL(maths::CPrior::E_LogNormal, filter.models()[0]->type());
    BOOST_CHECK_EQUAL(maths::CPrior::E_Normal, filter.models()[1]->type());
    TDoubleVec weights = filter.weights();
    BOOST_CHECK_CLOSE_ABSOLUTE(
        1.0, std::accumulate(weights.begin(), weights.end(), 0.0), 1e-6);
}

BOOST_AUTO_TEST_CASE(testMultipleUpdate) {
    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    using TEqual = maths::CEqualWithTolerance<double>;

    TPriorPtrVec models;
    models.push_back(
        TPriorPtr(maths::CPoissonMeanConjugate::nonInformativePrior().clone()));
    models.push_back(TPriorPtr(
        maths::CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    const double mean = 10.0;
    const double variance = 3.0;

    COneOfNPrior filter1(maths::COneOfNPrior(clone(models), E_ContinuousData));
    COneOfNPrior filter2(maths::COneOfNPrior(clone(models), E_ContinuousData));

    test::CRandomNumbers rng;

    // Deal with improper prior pathology.
    TDoubleVec seedSamples;
    rng.generateNormalSamples(mean, variance, 2, seedSamples);
    for (std::size_t i = 0u; i < seedSamples.size(); ++i) {
        TDouble1Vec sample(1, seedSamples[i]);
        filter1.addSamples(sample);
        filter2.addSamples(sample);
    }

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    // Make sure we don't have negative values.
    truncateUpTo(0.0, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter1.addSamples(TDouble1Vec(1, samples[i]));
    }
    filter2.addSamples(samples);

    LOG_DEBUG(<< filter1.print());
    LOG_DEBUG(<< "vs");
    LOG_DEBUG(<< filter2.print());
    TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-3);

    TDoubleVec weights1 = filter1.weights();
    TDoubleVec weights2 = filter2.weights();
    LOG_DEBUG(<< "weight1 = " << core::CContainerPrinter::print(weights1));
    LOG_DEBUG(<< "weight2 = " << core::CContainerPrinter::print(weights2));
    BOOST_TEST(weights1.size() == weights2.size());
    BOOST_TEST(std::equal(weights1.begin(), weights1.end(), weights2.begin(), equal));

    COneOfNPrior::TPriorCPtrVec models1 = filter1.models();
    COneOfNPrior::TPriorCPtrVec models2 = filter2.models();
    BOOST_TEST(models1.size() == models2.size());

    const maths::CPoissonMeanConjugate* poisson1 =
        dynamic_cast<const maths::CPoissonMeanConjugate*>(models1[0]);
    const maths::CPoissonMeanConjugate* poisson2 =
        dynamic_cast<const maths::CPoissonMeanConjugate*>(models2[0]);
    BOOST_TEST(poisson1);
    BOOST_TEST(poisson2);
    BOOST_TEST(poisson1->equalTolerance(*poisson2, equal));

    const maths::CNormalMeanPrecConjugate* normal1 =
        dynamic_cast<const maths::CNormalMeanPrecConjugate*>(models1[1]);
    const maths::CNormalMeanPrecConjugate* normal2 =
        dynamic_cast<const maths::CNormalMeanPrecConjugate*>(models2[1]);
    BOOST_TEST(normal1);
    BOOST_TEST(normal2);
    BOOST_TEST(normal1->equalTolerance(*normal2, equal));

    // Test the count weight is equivalent to adding repeated samples.

    double x = 3.0;
    std::size_t count = 10;

    for (std::size_t j = 0u; j < count; ++j) {
        filter1.addSamples(TDouble1Vec(1, x));
    }
    filter2.addSamples({x}, {maths_t::countWeight(static_cast<double>(count))});

    BOOST_CHECK_EQUAL(filter1.checksum(), filter2.checksum());
}

BOOST_AUTO_TEST_CASE(testWeights) {
    test::CRandomNumbers rng;

    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(
            CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-10);
        const double decayRates[] = {0.0, 0.001, 0.01};

        for (std::size_t rate = 0; rate < boost::size(decayRates); ++rate) {
            // Test that the filter weights stay normalized.
            COneOfNPrior filter(maths::COneOfNPrior(
                clone(models, decayRates[rate]), E_ContinuousData, decayRates[rate]));

            const double mean = 20.0;
            const double variance = 3.0;

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, 1000, samples);

            // Make sure we don't have negative values.
            truncateUpTo(0.0, samples);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                filter.addSamples(TDouble1Vec(1, samples[i]));
                BOOST_TEST(equal(sum(filter.weights()), 1.0));
                filter.propagateForwardsByTime(1.0);
                BOOST_TEST(equal(sum(filter.weights()), 1.0));
            }
        }
    }

    {
        // Test that non-zero decay rate behaves as expected.

        const double decayRates[] = {0.0002, 0.001, 0.005};

        const double rate = 5.0;

        double previousLogWeightRatio = -500;

        for (std::size_t decayRate = 0; decayRate < boost::size(decayRates); ++decayRate) {
            TPriorPtrVec models;
            models.push_back(
                TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
            models.push_back(TPriorPtr(
                CNormalMeanPrecConjugate::nonInformativePrior(E_IntegerData).clone()));
            COneOfNPrior filter(maths::COneOfNPrior(
                clone(models, decayRates[decayRate]), E_IntegerData, decayRates[decayRate]));

            TUIntVec samples;
            rng.generatePoissonSamples(rate, 10000, samples);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
                filter.propagateForwardsByTime(1.0);
            }

            TDoubleVec logWeights = filter.logWeights();

            LOG_DEBUG(<< "log weights ratio = "
                      << (logWeights[1] - logWeights[0]) / previousLogWeightRatio);

            // Should be approximately 0.2: we reduce the filter memory
            // by a factor of 5 each iteration.
            BOOST_TEST((logWeights[1] - logWeights[0]) / previousLogWeightRatio > 0.15);
            BOOST_TEST((logWeights[1] - logWeights[0]) / previousLogWeightRatio < 0.35);
            previousLogWeightRatio = logWeights[1] - logWeights[0];
        }
    }
}

BOOST_AUTO_TEST_CASE(testModels) {
    // Test the models posterior mean values.

    // Since the component model's posterior distributions are tested
    // separately the focus of this test is only to check the expected
    // posterior values.

    test::CRandomNumbers rng;

    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(
            CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        // The mean of the Poisson model and the mean and variance of the
        // Normal model are all close to the rate r.
        const double rate = 2.0;
        const double mean = rate;
        const double variance = rate;

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TUIntVec samples;
        rng.generatePoissonSamples(rate, 3000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
        }

        COneOfNPrior::TPriorCPtrVec posteriorModels = filter.models();
        const maths::CPoissonMeanConjugate* poissonModel =
            dynamic_cast<const maths::CPoissonMeanConjugate*>(posteriorModels[0]);
        const maths::CNormalMeanPrecConjugate* normalModel =
            dynamic_cast<const maths::CNormalMeanPrecConjugate*>(posteriorModels[1]);
        BOOST_TEST(poissonModel);
        BOOST_TEST(normalModel);

        LOG_DEBUG(<< "Poisson mean = " << poissonModel->priorMean()
                  << ", expectedMean = " << rate);
        LOG_DEBUG(<< "Normal mean = " << normalModel->mean() << ", expectedMean = " << mean
                  << ", precision = " << normalModel->precision()
                  << ", expectedPrecision " << (1.0 / variance));

        BOOST_TEST(std::fabs(poissonModel->priorMean() - rate) / rate < 0.01);
        BOOST_TEST(std::fabs(normalModel->mean() - mean) / mean < 0.01);
        BOOST_TEST(std::fabs(normalModel->precision() - 1.0 / variance) * variance < 0.06);
    }

    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(
            CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        const double mean = 10.0;
        const double variance = 2.0;

        // The mean of the Poisson model should be the mean of the Gaussian.
        const double rate = 10.0;

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 1000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            filter.addSamples(TDouble1Vec(1, samples[i]));
        }

        COneOfNPrior::TPriorCPtrVec posteriorModels = filter.models();
        const maths::CPoissonMeanConjugate* poissonModel =
            dynamic_cast<const maths::CPoissonMeanConjugate*>(posteriorModels[0]);
        const maths::CNormalMeanPrecConjugate* normalModel =
            dynamic_cast<const maths::CNormalMeanPrecConjugate*>(posteriorModels[1]);
        BOOST_TEST(poissonModel);
        BOOST_TEST(normalModel);

        LOG_DEBUG(<< "Poisson mean = " << poissonModel->priorMean()
                  << ", expectedMean = " << rate);
        LOG_DEBUG(<< "Normal mean = " << normalModel->mean() << ", expectedMean = " << mean
                  << ", precision = " << normalModel->precision()
                  << ", expectedPrecision " << (1.0 / variance));

        BOOST_TEST(std::fabs(poissonModel->priorMean() - rate) / rate < 0.01);
        BOOST_TEST(std::fabs(normalModel->mean() - mean) / mean < 0.01);
        BOOST_TEST(std::fabs(normalModel->precision() - 1.0 / variance) * variance < 0.15);
    }
}

BOOST_AUTO_TEST_CASE(testModelSelection) {
    test::CRandomNumbers rng;

    {
        // Generate Poisson samples and update the mixed model. The log weight
        // of the normal should diminish relative to the weight of the Poisson
        // on average per sample (uniform law of large numbers) as:
        //   log(2 * pi * e * rate) / 2 - E[ log(f(X)) ]
        //
        // where, f(x) is the Poisson density function and X ~ f(x). Note
        // because we limit the difference in log-likelihoods we expect this
        // to be somewhat larger than the expectation.

        TPriorPtrVec models;
        models.emplace_back(CPoissonMeanConjugate::nonInformativePrior().clone());
        models.emplace_back(
            CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone());

        const unsigned int numberSamples = 10000;
        const double rate = 2.0;
        const double mean = rate;
        const double variance = rate;

        boost::math::poisson poisson(rate);
        boost::math::normal normal(mean, std::sqrt(variance));

        double poissonExpectedLogWeight = -maths::CTools::differentialEntropy(poisson);
        double normalExpectedLogWeight = -maths::CTools::differentialEntropy(normal);

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TUIntVec samples;
        rng.generatePoissonSamples(rate, numberSamples, samples);

        for (auto sample : samples) {
            filter.addSamples(TDouble1Vec{static_cast<double>(sample)});
        }

        double expectedLogWeightRatio = (normalExpectedLogWeight - poissonExpectedLogWeight) *
                                        static_cast<double>(numberSamples);

        TDoubleVec logWeights = filter.logWeights();
        double logWeightRatio = logWeights[1] - logWeights[0];

        LOG_DEBUG(<< "expectedLogWeightRatio = " << expectedLogWeightRatio
                  << ", logWeightRatio = " << logWeightRatio);

        BOOST_TEST(logWeightRatio > expectedLogWeightRatio);
        BOOST_TEST(logWeightRatio < 0.95 * expectedLogWeightRatio);
    }

    {
        // Generate log normal samples and update the mixed model. The normal
        // weight should diminish relative to the log-normal weight on average
        // per sample (uniform law of large numbers) as:
        //   log(2 * pi * e * var(X)) - E[ log(f(X)) ]
        //
        // where, f(x) is the log-normal density function and X ~ f(x). Note
        // because we limit the difference in log-likelihoods we expect this
        // to be somewhat larger than the expectation.

        TPriorPtrVec models;
        models.emplace_back(
            CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone());
        models.emplace_back(
            CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone());

        const unsigned int numberSamples = 10000;
        const double location = 1.0;
        const double squareScale = 0.5;

        boost::math::lognormal logNormal(location, std::sqrt(squareScale));
        boost::math::normal normal(boost::math::mean(logNormal),
                                   boost::math::standard_deviation(logNormal));

        double logNormalExpectedLogWeight = -maths::CTools::differentialEntropy(logNormal);
        double normalExpectedLogWeight = -maths::CTools::differentialEntropy(normal);

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, numberSamples, samples);

        for (auto sample : samples) {
            filter.addSamples(TDouble1Vec{sample});
        }

        double expectedLogWeightRatio = (normalExpectedLogWeight - logNormalExpectedLogWeight) *
                                        static_cast<double>(numberSamples);

        TDoubleVec logWeights = filter.logWeights();
        double logWeightRatio = logWeights[1] - logWeights[0];

        LOG_DEBUG(<< "expectedLogWeightRatio = " << expectedLogWeightRatio
                  << ", logWeightRatio = " << logWeightRatio);

        BOOST_TEST(logWeightRatio > expectedLogWeightRatio);
        BOOST_TEST(logWeightRatio < 0.75 * expectedLogWeightRatio);
    }
    {
        // Check we correctly select the multimodal model when the data have
        // clusters.

        TDoubleVec mode1;
        rng.generateNormalSamples(10.0, 9.0, 100, mode1);
        TDoubleVec mode2;
        rng.generateNormalSamples(22.0, 4.0, 100, mode2);
        TDoubleVec samples(mode1.begin(), mode1.end());
        samples.insert(samples.end(), mode2.begin(), mode2.end());
        rng.random_shuffle(samples.begin(), samples.end());

        TPriorPtrVec models;
        models.push_back(TPriorPtr(
            CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight);
        maths::CNormalMeanPrecConjugate normal =
            maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
        maths::COneOfNPrior::TPriorPtrVec mode;
        mode.push_back(TPriorPtr(normal.clone()));
        models.push_back(TPriorPtr(new maths::CMultimodalPrior(
            maths_t::E_ContinuousData, clusterer,
            maths::COneOfNPrior(mode, maths_t::E_ContinuousData))));
        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            filter.addSamples(TDouble1Vec(1, samples[i]));
        }

        TDoubleVec logWeights = filter.logWeights();
        double logWeightRatio = logWeights[0] - logWeights[1];

        LOG_DEBUG(<< "logWeightRatio = " << logWeightRatio);
        BOOST_TEST(std::exp(logWeightRatio) < 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihood) {
    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] = {E_ContinuousData, E_IntegerData};

    for (std::size_t t = 0u; t < boost::size(dataTypes); ++t) {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(
            CNormalMeanPrecConjugate::nonInformativePrior(dataTypes[t]).clone()));
        models.push_back(TPriorPtr(
            CLogNormalMeanPrecConjugate::nonInformativePrior(dataTypes[t]).clone()));
        models.push_back(TPriorPtr(
            CGammaRateConjugate::nonInformativePrior(dataTypes[t]).clone()));
        COneOfNPrior filter(maths::COneOfNPrior(clone(models), dataTypes[t]));

        const double location = 1.0;
        const double squareScale = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, 10, samples);
        filter.addSamples(samples);

        TWeightFunc weightsFuncs[]{static_cast<TWeightFunc>(maths_t::countWeight),
                                   static_cast<TWeightFunc>(maths_t::winsorisationWeight)};
        double weights[]{0.1, 1.0, 10.0};

        for (std::size_t i = 0u; i < boost::size(weightsFuncs); ++i) {
            for (std::size_t j = 0u; j < boost::size(weights); ++j) {
                double lb, ub;
                filter.minusLogJointCdf({10000.0}, {weightsFuncs[i](weights[j])}, lb, ub);
                LOG_DEBUG(<< "-log(c.d.f) = " << (lb + ub) / 2.0);
                BOOST_TEST(lb >= 0.0);
                BOOST_TEST(ub >= 0.0);
            }
        }
    }

    // Check that the marginal likelihood and c.d.f. agree for some
    // test data, that the c.d.f. <= 1 and c.d.f. + c.d.f. complement = 1.

    static const double EPS = 1e-3;

    test::CRandomNumbers rng;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(
        CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(
        CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

    TDoubleVec samples;
    rng.generateLogNormalSamples(1.0, 1.0, 99, samples);

    for (std::size_t i = 0u; i < 2; ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }
    for (std::size_t i = 2u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(99.0);
        LOG_DEBUG(<< "interval = " << core::CContainerPrinter::print(interval));

        double x = interval.first;
        double dx = (interval.second - interval.first) / 20.0;
        for (std::size_t j = 0u; j < 20; ++j, x += dx) {
            double fx;
            BOOST_TEST(filter.jointLogMarginalLikelihood(TDouble1Vec(1, x), fx) ==
                           maths_t::E_FpNoErrors);
            fx = std::exp(fx);

            double lb;
            double ub;
            BOOST_TEST(filter.minusLogJointCdf(TDouble1Vec(1, x + EPS), lb, ub));
            double FxPlusEps = std::exp(-(lb + ub) / 2.0);

            BOOST_TEST(filter.minusLogJointCdf(TDouble1Vec(1, x - EPS), lb, ub));
            double FxMinusEps = std::exp(-(lb + ub) / 2.0);

            double dFdx = (FxPlusEps - FxMinusEps) / (2.0 * EPS);

            LOG_DEBUG(<< "x = " << x << ", f(x) = " << fx << ", dF(x)/dx = " << dFdx);
            BOOST_CHECK_CLOSE_ABSOLUTE(fx, dFdx, std::max(1e-5, 1e-3 * FxPlusEps));

            BOOST_TEST(filter.minusLogJointCdf(TDouble1Vec(1, x), lb, ub));
            double Fx = std::exp(-(lb + ub) / 2.0);

            BOOST_TEST(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lb, ub));
            double FxComplement = std::exp(-(lb + ub) / 2.0);
            LOG_DEBUG(<< "F(x) = " << Fx << " 1 - F(x) = " << FxComplement);

            BOOST_CHECK_CLOSE_ABSOLUTE(1.0, Fx + FxComplement, 1e-3);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMean) {
    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "****** Normal ******");

        const double means[] = {10.0, 50.0};
        const double variances[] = {1.0, 10.0};

        for (std::size_t i = 0u; i < boost::size(means); ++i) {
            for (std::size_t j = 0u; j < boost::size(variances); ++j) {
                LOG_DEBUG(<< "*** mean = " << means[i]
                          << ", variance = " << variances[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));

                COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateNormalSamples(means[i], variances[j], 100, samples);

                for (std::size_t k = 0u; k < samples.size(); ++k) {
                    filter.addSamples(TDouble1Vec(1, samples[k]));

                    double expectedMean;
                    BOOST_TEST(filter.marginalLikelihoodMeanForTest(expectedMean));

                    if (k % 10 == 0) {
                        LOG_DEBUG(<< "marginalLikelihoodMean = " << filter.marginalLikelihoodMean()
                                  << ", expectedMean = " << expectedMean);
                    }

                    // The error is at the precision of the numerical integration.
                    BOOST_CHECK_CLOSE_ABSOLUTE(expectedMean,
                                                 filter.marginalLikelihoodMean(),
                                                 0.01 * expectedMean);
                }
            }
        }
    }

    {
        LOG_DEBUG(<< "****** Log Normal ******");

        const double locations[] = {0.1, 1.0};
        const double squareScales[] = {0.1, 1.0};

        for (std::size_t i = 0u; i < boost::size(locations); ++i) {
            for (std::size_t j = 0u; j < boost::size(squareScales); ++j) {
                LOG_DEBUG(<< "*** location = " << locations[i]
                          << ", squareScale = " << squareScales[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));

                COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateLogNormalSamples(locations[i], squareScales[j], 10, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateLogNormalSamples(locations[i], squareScales[j], 100, samples);

                TMeanAccumulator relativeError;

                for (std::size_t k = 0u; k < samples.size(); ++k) {
                    filter.addSamples(TDouble1Vec(1, samples[k]));

                    double expectedMean;
                    BOOST_TEST(filter.marginalLikelihoodMeanForTest(expectedMean));

                    if (k % 10 == 0) {
                        LOG_DEBUG(<< "marginalLikelihoodMean = " << filter.marginalLikelihoodMean()
                                  << ", expectedMean = " << expectedMean);
                    }

                    BOOST_CHECK_CLOSE_ABSOLUTE(expectedMean,
                                                 filter.marginalLikelihoodMean(),
                                                 0.2 * expectedMean);

                    relativeError.add(std::fabs(filter.marginalLikelihoodMean() - expectedMean) /
                                      expectedMean);
                }

                LOG_DEBUG(<< "relativeError = "
                          << maths::CBasicStatistics::mean(relativeError));
                BOOST_TEST(maths::CBasicStatistics::mean(relativeError) < 0.02);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMode) {
    // Test that the marginal likelihood mode is near the maximum
    // of the marginal likelihood.

    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "****** Normal ******");

        const double means[] = {10.0, 50.0};
        const double variances[] = {1.0, 10.0};

        for (std::size_t i = 0u; i < boost::size(means); ++i) {
            for (std::size_t j = 0u; j < boost::size(variances); ++j) {
                LOG_DEBUG(<< "*** mean = " << means[i]
                          << ", variance = " << variances[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));

                COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
                filter.addSamples(seedSamples);
                TDoubleVec samples;
                rng.generateNormalSamples(means[i], variances[j], 20, samples);
                filter.addSamples(samples);

                std::size_t iterations = 12;
                double mode;
                double fmode;
                maths::CCompositeFunctions::CExp<maths::CPrior::CLogMarginalLikelihood> likelihood(
                    filter);
                double a = means[i] - 2.0 * std::sqrt(variances[j]);
                double b = means[i] + 2.0 * std::sqrt(variances[j]);
                maths::CSolvers::maximize(a, b, likelihood(a), likelihood(b),
                                          likelihood, 0.0, iterations, mode, fmode);

                LOG_DEBUG(<< "marginalLikelihoodMode = " << filter.marginalLikelihoodMode()
                          << ", expectedMode = " << mode);

                BOOST_CHECK_CLOSE_ABSOLUTE(mode, filter.marginalLikelihoodMode(),
                                             0.01 * mode);
            }
        }
    }

    {
        LOG_DEBUG(<< "****** Log Normal ******");

        const double locations[] = {0.1, 1.0};
        const double squareScales[] = {0.1, 2.0};

        for (std::size_t i = 0u; i < boost::size(locations); ++i) {
            for (std::size_t j = 0u; j < boost::size(squareScales); ++j) {
                LOG_DEBUG(<< "*** location = " << locations[i]
                          << ", squareScale = " << squareScales[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));

                COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateLogNormalSamples(locations[i], squareScales[j], 10, seedSamples);
                filter.addSamples(seedSamples);
                TDoubleVec samples;
                rng.generateLogNormalSamples(locations[i], squareScales[j], 20, samples);
                filter.addSamples(samples);

                std::size_t iterations = 12;
                double mode;
                double fmode;
                maths::CCompositeFunctions::CExp<maths::CPrior::CLogMarginalLikelihood> likelihood(
                    filter);
                boost::math::lognormal_distribution<> logNormal(
                    locations[i], std::sqrt(squareScales[j]));
                double a = 0.01;
                double b = boost::math::mode(logNormal) +
                           1.0 * boost::math::standard_deviation(logNormal);
                maths::CSolvers::maximize(a, b, likelihood(a), likelihood(b),
                                          likelihood, 0.0, iterations, mode, fmode);

                LOG_DEBUG(<< "marginalLikelihoodMode = " << filter.marginalLikelihoodMode()
                          << ", expectedMode = " << mode);

                BOOST_CHECK_CLOSE_ABSOLUTE(mode, filter.marginalLikelihoodMode(),
                                             0.05 * mode);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodVariance) {
    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "****** Normal ******");

        double means[] = {10.0, 100.0};
        double variances[] = {1.0, 10.0};

        for (std::size_t i = 0u; i < boost::size(means); ++i) {
            for (std::size_t j = 0u; j < boost::size(variances); ++j) {
                LOG_DEBUG(<< "*** mean = " << means[i]
                          << ", variance = " << variances[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));
                models.push_back(TPriorPtr(
                    CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));

                COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateNormalSamples(means[i], variances[j], 100, samples);

                TMeanAccumulator relativeError;
                for (std::size_t k = 0u; k < samples.size(); ++k) {
                    filter.addSamples(TDouble1Vec(1, samples[k]));
                    double expectedVariance;
                    BOOST_TEST(filter.marginalLikelihoodVarianceForTest(expectedVariance));
                    if (k % 10 == 0) {
                        LOG_DEBUG(<< "marginalLikelihoodVariance = "
                                  << filter.marginalLikelihoodVariance()
                                  << ", expectedVariance = " << expectedVariance);
                    }

                    // The error is at the precision of the numerical integration.
                    BOOST_CHECK_CLOSE_ABSOLUTE(expectedVariance,
                                                 filter.marginalLikelihoodVariance(),
                                                 0.02 * expectedVariance);

                    relativeError.add(std::fabs(expectedVariance -
                                                filter.marginalLikelihoodVariance()) /
                                      expectedVariance);
                }

                LOG_DEBUG(<< "relativeError = "
                          << maths::CBasicStatistics::mean(relativeError));
                BOOST_TEST(maths::CBasicStatistics::mean(relativeError) < 2e-3);
            }
        }
    }

    {
        LOG_DEBUG(<< "****** Gamma ******");

        const double shapes[] = {5.0, 20.0, 40.0};
        const double scales[] = {1.0, 10.0, 20.0};

        for (std::size_t i = 0u; i < boost::size(shapes); ++i) {
            for (std::size_t j = 0u; j < boost::size(scales); ++j) {
                LOG_DEBUG(<< "*** shape = " << shapes[i]
                          << ", scale = " << scales[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData)
                                               .clone()));
                models.push_back(TPriorPtr(
                    CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));

                COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateGammaSamples(shapes[i], scales[j], 10, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateGammaSamples(shapes[i], scales[j], 100, samples);

                TMeanAccumulator relativeError;

                for (std::size_t k = 0u; k < samples.size(); ++k) {
                    filter.addSamples(TDouble1Vec(1, samples[k]));

                    double expectedVariance;
                    BOOST_TEST(filter.marginalLikelihoodVarianceForTest(expectedVariance));

                    if (k % 10 == 0) {
                        LOG_DEBUG(<< "marginalLikelihoodVariance = "
                                  << filter.marginalLikelihoodVariance()
                                  << ", expectedVariance = " << expectedVariance);
                    }

                    // The error is mainly due to the truncation in the
                    // integration range used to compute the expected mean.
                    BOOST_CHECK_CLOSE_ABSOLUTE(expectedVariance,
                                                 filter.marginalLikelihoodVariance(),
                                                 0.01 * expectedVariance);

                    relativeError.add(std::fabs(expectedVariance -
                                                filter.marginalLikelihoodVariance()) /
                                      expectedVariance);
                }

                LOG_DEBUG(<< "relativeError = "
                          << maths::CBasicStatistics::mean(relativeError));
                BOOST_TEST(maths::CBasicStatistics::mean(relativeError) < 3e-3);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihood) {
    // Test we sample the constitute priors in proportion to their weights.

    const double mean = 5.0;
    const double variance = 2.0;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(
        CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(
        CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 20, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    TDoubleVec weights = filter.weights();
    LOG_DEBUG(<< "weights = " << core::CContainerPrinter::print(weights));

    TDouble1Vec sampled;
    filter.sampleMarginalLikelihood(10, sampled);

    // We expect 5 samples from the normal and 5 from the log normal.
    COneOfNPrior::TPriorCPtrVec posteriorModels = filter.models();
    TDouble1Vec normalSamples;
    posteriorModels[0]->sampleMarginalLikelihood(5, normalSamples);
    TDouble1Vec logNormalSamples;
    posteriorModels[1]->sampleMarginalLikelihood(5, logNormalSamples);

    TDoubleVec expectedSampled(normalSamples);
    expectedSampled.insert(expectedSampled.end(), logNormalSamples.begin(),
                           logNormalSamples.end());

    LOG_DEBUG(<< "expected samples = " << core::CContainerPrinter::print(expectedSampled)
              << ", samples = " << core::CContainerPrinter::print(sampled));

    BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expectedSampled),
                         core::CContainerPrinter::print(sampled));

    rng.generateNormalSamples(mean, variance, 80, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    weights = filter.weights();
    LOG_DEBUG(<< "weights = " << core::CContainerPrinter::print(weights));

    filter.sampleMarginalLikelihood(20, sampled);

    // We expect 20 samples from the normal and 0 from the log normal.
    posteriorModels = filter.models();
    posteriorModels[0]->sampleMarginalLikelihood(20, normalSamples);
    posteriorModels[1]->sampleMarginalLikelihood(0, logNormalSamples);

    expectedSampled = normalSamples;
    expectedSampled.insert(expectedSampled.end(), logNormalSamples.begin(),
                           logNormalSamples.end());

    LOG_DEBUG(<< "expected samples = " << core::CContainerPrinter::print(expectedSampled)
              << ", samples = " << core::CContainerPrinter::print(sampled));

    BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expectedSampled),
                         core::CContainerPrinter::print(sampled));
}

BOOST_AUTO_TEST_CASE(testCdf) {
    // Test error cases and the invariant "cdf" + "cdf complement" = 1

    const double mean = 20.0;
    const double variance = 5.0;
    const std::size_t n[] = {20u, 80u};

    test::CRandomNumbers rng;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(
        CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(
        CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(
        CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));
    COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, n[i], samples);

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, samples[j]));
        }

        double lb, ub;
        BOOST_TEST(!filter.minusLogJointCdf(TDouble1Vec(), lb, ub));
        BOOST_TEST(!filter.minusLogJointCdfComplement(TDouble1Vec(), lb, ub));

        for (std::size_t j = 1u; j < 500; ++j) {
            double x = static_cast<double>(j) / 2.0;

            BOOST_TEST(filter.minusLogJointCdf(TDouble1Vec(1, x), lb, ub));
            double f = (lb + ub) / 2.0;
            BOOST_TEST(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lb, ub));
            double fComplement = (lb + ub) / 2.0;

            LOG_DEBUG(<< "log(F(x)) = " << (f == 0.0 ? f : -f) << ", log(1 - F(x)) = "
                      << (fComplement == 0.0 ? fComplement : -fComplement));
            BOOST_CHECK_CLOSE_ABSOLUTE(1.0, std::exp(-f) + std::exp(-fComplement), 1e-10);
        }
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySamples) {
    // We simply test that the calculation yields the weighted sum
    // of component model calculations (which is its definition).

    const double location = 0.7;
    const double squareScale = 1.3;
    const double vs[] = {0.5, 1.0, 2.0};

    TPriorPtrVec initialModels;
    initialModels.push_back(TPriorPtr(
        CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    initialModels.push_back(TPriorPtr(
        CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    COneOfNPrior filter(maths::COneOfNPrior(clone(initialModels), E_ContinuousData));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(location, squareScale, 200, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        TDouble1Vec sample(1, samples[i]);
        filter.addSamples(sample);

        double lb, ub;
        maths_t::ETail tail;

        BOOST_TEST(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                             sample, lb, ub));

        BOOST_CHECK_EQUAL(lb, ub);
        double probability = (lb + ub) / 2.0;

        double expectedProbability = 0.0;

        TDoubleVec weights(filter.weights());
        COneOfNPrior::TPriorCPtrVec models(filter.models());
        for (std::size_t j = 0u; j < weights.size(); ++j) {
            double weight = weights[j];
            BOOST_TEST(models[j]->probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, {sample[0]},
                maths_t::CUnitWeights::SINGLE_UNIT, lb, ub, tail));
            BOOST_CHECK_EQUAL(lb, ub);
            double modelProbability = (lb + ub) / 2.0;
            expectedProbability += weight * modelProbability;
        }

        LOG_DEBUG(<< "weights = " << core::CContainerPrinter::print(weights)
                  << ", expectedProbability = " << expectedProbability
                  << ", probability = " << probability);
        BOOST_CHECK_CLOSE_ABSOLUTE(expectedProbability, probability,
                                     1e-3 * std::max(expectedProbability, probability));

        for (std::size_t k = 0u; ((i + 1) % 11 == 0) && k < boost::size(vs); ++k) {
            double mode = filter.marginalLikelihoodMode(
                maths_t::countVarianceScaleWeight(vs[k]));
            double ss[] = {0.9 * mode, 1.1 * mode};

            LOG_DEBUG(<< "vs = " << vs[k] << ", mode = " << mode);

            if (mode > 0.0) {
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, {ss[0]},
                    {maths_t::countVarianceScaleWeight(vs[k])}, lb, ub, tail);
                BOOST_CHECK_EQUAL(maths_t::E_LeftTail, tail);
                if (mode > 0.0) {
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, TDouble1Vec(ss, ss + 2),
                        maths_t::TDoubleWeightsAry1Vec(
                            2, maths_t::countVarianceScaleWeight(vs[k])),
                        lb, ub, tail);
                    BOOST_CHECK_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_OneSidedBelow, TDouble1Vec(ss, ss + 2),
                        maths_t::TDoubleWeightsAry1Vec(
                            2, maths_t::countVarianceScaleWeight(vs[k])),
                        lb, ub, tail);
                    BOOST_CHECK_EQUAL(maths_t::E_LeftTail, tail);
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_OneSidedAbove, TDouble1Vec(ss, ss + 2),
                        maths_t::TDoubleWeightsAry1Vec(
                            2, maths_t::countVarianceScaleWeight(vs[k])),
                        lb, ub, tail);
                    BOOST_CHECK_EQUAL(maths_t::E_RightTail, tail);
                }
            }
            if (mode > 0.0) {
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, {ss[1]},
                    {maths_t::countVarianceScaleWeight(vs[k])}, lb, ub, tail);
                BOOST_CHECK_EQUAL(maths_t::E_RightTail, tail);
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, TDouble1Vec(ss, ss + 2),
                    maths_t::TDoubleWeightsAry1Vec(
                        2, maths_t::countVarianceScaleWeight(vs[k])),
                    lb, ub, tail);
                BOOST_CHECK_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_OneSidedBelow, TDouble1Vec(ss, ss + 2),
                    maths_t::TDoubleWeightsAry1Vec(
                        2, maths_t::countVarianceScaleWeight(vs[k])),
                    lb, ub, tail);
                BOOST_CHECK_EQUAL(maths_t::E_LeftTail, tail);
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_OneSidedAbove, TDouble1Vec(ss, ss + 2),
                    maths_t::TDoubleWeightsAry1Vec(
                        2, maths_t::countVarianceScaleWeight(vs[k])),
                    lb, ub, tail);
                BOOST_CHECK_EQUAL(maths_t::E_RightTail, tail);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that persist/restore is idempotent.

    TPriorPtrVec models;
    models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
    models.push_back(TPriorPtr(
        CNormalMeanPrecConjugate::nonInformativePrior(E_IntegerData).clone()));

    const double mean = 10.0;
    const double variance = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    // Make sure we don't have negative values.
    truncateUpTo(0.0, samples);

    maths::COneOfNPrior origFilter(clone(models), E_IntegerData);
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        origFilter.addSamples({samples[i]}, maths_t::CUnitWeights::SINGLE_UNIT);
    }
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "One-of-N prior XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(
        E_IntegerData, decayRate + 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    maths::COneOfNPrior restoredFilter(params, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    BOOST_CHECK_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_CHECK_EQUAL(origXml, newXml);
}


BOOST_AUTO_TEST_SUITE_END()
