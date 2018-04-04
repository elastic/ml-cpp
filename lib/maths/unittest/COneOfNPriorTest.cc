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

#include "COneOfNPriorTest.h"

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

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <numeric>

using namespace ml;
using namespace handy_typedefs;

namespace
{

using TUIntVec = std::vector<unsigned int>;
using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TPriorPtr = boost::shared_ptr<maths::CPrior>;
using TPriorPtrVec = std::vector<TPriorPtr>;
using TOptionalDouble = boost::optional<double>;
using CGammaRateConjugate = CPriorTestInterfaceMixin<maths::CGammaRateConjugate>;
using CLogNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CLogNormalMeanPrecConjugate>;
using CMultimodalPrior = CPriorTestInterfaceMixin<maths::CMultimodalPrior>;
using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;
using COneOfNPrior = CPriorTestInterfaceMixin<maths::COneOfNPrior>;
using CPoissonMeanConjugate = CPriorTestInterfaceMixin<maths::CPoissonMeanConjugate>;

COneOfNPrior::TPriorPtrVec clone(const TPriorPtrVec &models,
                                 const TOptionalDouble &decayRate = TOptionalDouble())
{
    COneOfNPrior::TPriorPtrVec result;
    result.reserve(models.size());
    for (std::size_t i = 0u; i < models.size(); ++i)
    {
        result.push_back(COneOfNPrior::TPriorPtr(models[i]->clone()));
        if (decayRate)
        {
            result.back()->decayRate(*decayRate);
        }
    }
    return result;
}

void truncateUpTo(const double &value, TDoubleVec &samples)
{
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        samples[i] = std::max(samples[i], value);
    }
}

double sum(const TDoubleVec &values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

using maths_t::E_ContinuousData;
using maths_t::E_IntegerData;

}

void COneOfNPriorTest::testFilter()
{
    LOG_DEBUG("+--------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testFilter  |");
    LOG_DEBUG("+--------------------------------+");

    TPriorPtrVec models;
    models.push_back(TPriorPtr(maths::CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(maths::CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(maths::CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(maths::CPoissonMeanConjugate::nonInformativePrior().clone()));

    COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(10.0, 3.0, 20, samples);

    // Make sure we don't have negative values.
    truncateUpTo(0.0, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    filter.removeModels(maths::CPrior::CModelFilter().remove(maths::CPrior::E_Constant));

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), filter.models().size());

    filter.removeModels(maths::CPrior::CModelFilter().remove(maths::CPrior::E_Poisson)
                                                     .remove(maths::CPrior::E_Gamma));

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), filter.models().size());
    CPPUNIT_ASSERT_EQUAL(maths::CPrior::E_LogNormal, filter.models()[0]->type());
    CPPUNIT_ASSERT_EQUAL(maths::CPrior::E_Normal, filter.models()[1]->type());
    TDoubleVec weights = filter.weights();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, std::accumulate(weights.begin(), weights.end(), 0.0), 1e-6);
}

void COneOfNPriorTest::testMultipleUpdate()
{
    LOG_DEBUG("+----------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testMultipleUpdate  |");
    LOG_DEBUG("+----------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    using TEqual = maths::CEqualWithTolerance<double>;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(maths::CPoissonMeanConjugate::nonInformativePrior().clone()));
    models.push_back(TPriorPtr(maths::CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    const double mean = 10.0;
    const double variance = 3.0;

    COneOfNPrior filter1(maths::COneOfNPrior(clone(models), E_ContinuousData));
    COneOfNPrior filter2(maths::COneOfNPrior(clone(models), E_ContinuousData));

    test::CRandomNumbers rng;

    // Deal with improper prior pathology.
    TDoubleVec seedSamples;
    rng.generateNormalSamples(mean, variance, 2, seedSamples);
    for (std::size_t i = 0u; i < seedSamples.size(); ++i)
    {
        TDouble1Vec sample(1, seedSamples[i]);
        filter1.addSamples(sample);
        filter2.addSamples(sample);
    }

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    // Make sure we don't have negative values.
    truncateUpTo(0.0, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter1.addSamples(TDouble1Vec(1, samples[i]));
    }
    filter2.addSamples(samples);

    LOG_DEBUG(filter1.print());
    LOG_DEBUG("vs");
    LOG_DEBUG(filter2.print());
    TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-6);

    TDoubleVec weights1 = filter1.weights();
    TDoubleVec weights2 = filter2.weights();
    LOG_DEBUG("weight1 = " << core::CContainerPrinter::print(weights1));
    LOG_DEBUG("weight2 = " << core::CContainerPrinter::print(weights2));
    CPPUNIT_ASSERT(weights1.size() == weights2.size());
    CPPUNIT_ASSERT(std::equal(weights1.begin(), weights1.end(),
                              weights2.begin(),
                              equal));

    COneOfNPrior::TPriorCPtrVec models1 = filter1.models();
    COneOfNPrior::TPriorCPtrVec models2 = filter2.models();
    CPPUNIT_ASSERT(models1.size() == models2.size());

    const maths::CPoissonMeanConjugate *poisson1 =
            dynamic_cast<const maths::CPoissonMeanConjugate*>(models1[0]);
    const maths::CPoissonMeanConjugate *poisson2 =
            dynamic_cast<const maths::CPoissonMeanConjugate*>(models2[0]);
    CPPUNIT_ASSERT(poisson1 && poisson2);
    CPPUNIT_ASSERT(poisson1->equalTolerance(*poisson2, equal));

    const maths::CNormalMeanPrecConjugate *normal1 =
            dynamic_cast<const maths::CNormalMeanPrecConjugate*>(models1[1]);
    const maths::CNormalMeanPrecConjugate *normal2 =
            dynamic_cast<const maths::CNormalMeanPrecConjugate*>(models2[1]);
    CPPUNIT_ASSERT(normal1 && normal2);
    CPPUNIT_ASSERT(normal1->equalTolerance(*normal2, equal));


    // Test the count weight is equivalent to adding repeated samples.

    double x = 3.0;
    std::size_t count = 10;

    for (std::size_t j = 0u; j < count; ++j)
    {
        filter1.addSamples(TDouble1Vec(1, x));
    }
    filter2.addSamples(maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight),
                       TDouble1Vec(1, x),
                       TDouble4Vec1Vec(1, TDouble4Vec(1, static_cast<double>(count))));

    CPPUNIT_ASSERT_EQUAL(filter1.checksum(), filter2.checksum());
}

void COneOfNPriorTest::testWeights()
{
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testWeights  |");
    LOG_DEBUG("+---------------------------------+");

    test::CRandomNumbers rng;

    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-10);
        const double decayRates[] = { 0.0, 0.001, 0.01 };

        for (std::size_t rate = 0; rate < boost::size(decayRates); ++rate)
        {
            // Test that the filter weights stay normalized.
            COneOfNPrior filter(maths::COneOfNPrior(clone(models, decayRates[rate]),
                                                    E_ContinuousData,
                                                    decayRates[rate]));

            const double mean = 20.0;
            const double variance = 3.0;

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, 1000, samples);

            // Make sure we don't have negative values.
            truncateUpTo(0.0, samples);

            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                filter.addSamples(TDouble1Vec(1, samples[i]));
                CPPUNIT_ASSERT(equal(sum(filter.weights()), 1.0));
                filter.propagateForwardsByTime(1.0);
                CPPUNIT_ASSERT(equal(sum(filter.weights()), 1.0));
            }
        }
    }

    {
        // Test that non-zero decay rate behaves as expected.

        const double decayRates[] = { 0.0002, 0.001, 0.005 };

        const double rate = 5.0;

        double previousLogWeightRatio = -500;

        for (std::size_t decayRate = 0; decayRate < boost::size(decayRates); ++decayRate)
        {
            TPriorPtrVec models;
            models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
            models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_IntegerData).clone()));
            COneOfNPrior filter(maths::COneOfNPrior(clone(models, decayRates[decayRate]),
                                                    E_IntegerData,
                                                    decayRates[decayRate]));

            TUIntVec samples;
            rng.generatePoissonSamples(rate, 10000, samples);

            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
                filter.propagateForwardsByTime(1.0);
            }

            TDoubleVec logWeights = filter.logWeights();

            LOG_DEBUG("log weights ratio = " << (logWeights[1] - logWeights[0]) / previousLogWeightRatio);

            // Should be approximately 0.2: we reduce the filter memory
            // by a factor of 5 each iteration.
            CPPUNIT_ASSERT((logWeights[1] - logWeights[0]) / previousLogWeightRatio > 0.15);
            CPPUNIT_ASSERT((logWeights[1] - logWeights[0]) / previousLogWeightRatio < 0.35);
            previousLogWeightRatio = logWeights[1] - logWeights[0];
        }
    }
}

void COneOfNPriorTest::testModels()
{
    LOG_DEBUG("+--------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testModels  |");
    LOG_DEBUG("+--------------------------------+");

    // Test the models posterior mean values.

    // Since the component model's posterior distributions are tested
    // separately the focus of this test is only to check the expected
    // posterior values.

    test::CRandomNumbers rng;

    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        // The mean of the Poisson model and the mean and variance of the
        // Normal model are all close to the rate r.
        const double rate = 2.0;
        const double mean = rate;
        const double variance = rate;

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TUIntVec samples;
        rng.generatePoissonSamples(rate, 3000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
        }

        COneOfNPrior::TPriorCPtrVec posteriorModels = filter.models();
        const maths::CPoissonMeanConjugate *poissonModel =
                dynamic_cast<const maths::CPoissonMeanConjugate*>(posteriorModels[0]);
        const maths::CNormalMeanPrecConjugate *normalModel =
                dynamic_cast<const maths::CNormalMeanPrecConjugate*>(posteriorModels[1]);
        CPPUNIT_ASSERT(poissonModel && normalModel);

        LOG_DEBUG("Poisson mean = " << poissonModel->priorMean()
                  << ", expectedMean = " << rate);
        LOG_DEBUG("Normal mean = " << normalModel->mean()
                  << ", expectedMean = " << mean
                  << ", precision = " << normalModel->precision()
                  << ", expectedPrecision " << (1.0 / variance));

        CPPUNIT_ASSERT(std::fabs(poissonModel->priorMean() - rate) / rate < 0.01);
        CPPUNIT_ASSERT(std::fabs(normalModel->mean() - mean) / mean < 0.01);
        CPPUNIT_ASSERT(std::fabs(normalModel->precision() - 1.0 / variance) * variance < 0.06);
    }

    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        const double mean = 10.0;
        const double variance = 2.0;

        // The mean of the Poisson model should be the mean of the Gaussian.
        const double rate = 10.0;

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 1000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            filter.addSamples(TDouble1Vec(1, samples[i]));
        }

        COneOfNPrior::TPriorCPtrVec posteriorModels = filter.models();
        const maths::CPoissonMeanConjugate *poissonModel =
                dynamic_cast<const maths::CPoissonMeanConjugate*>(posteriorModels[0]);
        const maths::CNormalMeanPrecConjugate *normalModel =
                dynamic_cast<const maths::CNormalMeanPrecConjugate*>(posteriorModels[1]);
        CPPUNIT_ASSERT(poissonModel && normalModel);

        LOG_DEBUG("Poisson mean = " << poissonModel->priorMean()
                  << ", expectedMean = " << rate);
        LOG_DEBUG("Normal mean = " << normalModel->mean()
                  << ", expectedMean = " << mean
                  << ", precision = " << normalModel->precision()
                  << ", expectedPrecision " << (1.0 / variance));

        CPPUNIT_ASSERT(std::fabs(poissonModel->priorMean() - rate) / rate < 0.01);
        CPPUNIT_ASSERT(std::fabs(normalModel->mean() - mean) / mean < 0.01);
        CPPUNIT_ASSERT(std::fabs(normalModel->precision() - 1.0 / variance) * variance < 0.15);
    }
}

void COneOfNPriorTest::testModelSelection()
{
    LOG_DEBUG("+----------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testModelSelection  |");
    LOG_DEBUG("+----------------------------------------+");

    test::CRandomNumbers rng;

    {
        // Generate Poisson samples and update the mixed model. The log normal
        // weight should diminish relative to the Poisson weight on average
        // per sample (uniform law of large numbers) as:
        //   log(2 * pi * e * r) / 2 - E[ log(f(x)) ]
        //
        // where,
        //   f(x) is the Poisson density function with mean r,
        //   E[.] is the expectation with respect to f(x).

        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        const unsigned int nSamples = 10000u;
        const double rate = 2.0;
        const double mean = rate;
        const double variance = rate;

        boost::math::poisson_distribution<> poisson(rate);
        boost::math::normal_distribution<> normal(mean, std::sqrt(variance));

        double poissonExpectedLogWeight = -maths::CTools::differentialEntropy(poisson);
        double normalExpectedLogWeight  = -maths::CTools::differentialEntropy(normal);

        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        TUIntVec samples;
        rng.generatePoissonSamples(rate, nSamples, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
        }

        double expectedLogWeightRatio =
                 (normalExpectedLogWeight - poissonExpectedLogWeight)
                 * static_cast<double>(nSamples);

        TDoubleVec logWeights = filter.logWeights();
        double logWeightRatio = logWeights[1] - logWeights[0];

        LOG_DEBUG("expectedLogWeightRatio = " << expectedLogWeightRatio
                  << ", logWeightRatio = " << logWeightRatio);

        CPPUNIT_ASSERT(std::fabs(logWeightRatio - expectedLogWeightRatio)
                       / std::fabs(expectedLogWeightRatio) < 0.05);
    }

    {
        // Generate Normal samples and update the mixed model. The Poisson
        // weight should diminish relative to the normal weight on average
        // per sample (uniform law of large numbers) as:
        //   log(2 * pi * e * m) - log(2 * pi * e * v)
        //
        // when,
        //   m, which is the Gaussian mean, is large,
        //   v, which is the Gaussian variance, is small compared to m.
        //
        // This is when the moment matched Gaussian approximation is reasonable.
        // Note, the tails have a pretty large impact on this integral because
        // we are taking the expectation of the log. Really we want to compute
        // the expectation of the log of the appropriate negative binomial density
        // function, but reasonable correlation with the approximate value is
        // a sufficiently good test.

        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

        const unsigned int nSamples[] = { 1000u, 2000u, 3000u };
        const double mean = 100.0;
        const double variance = 5.0;

        boost::math::normal_distribution<> poissonApprox(mean, std::sqrt(mean));
        boost::math::normal_distribution<> normal(mean, std::sqrt(variance));

        double poissonExpectedLogWeight = -maths::CTools::differentialEntropy(poissonApprox);
        double normalExpectedLogWeight  = -maths::CTools::differentialEntropy(normal);

        for (size_t n = 0; n < boost::size(nSamples); ++n)
        {
            COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, nSamples[n], samples);

            // Make sure we don't have negative values.
            truncateUpTo(0.0, samples);

            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                filter.addSamples(TDouble1Vec(1, samples[i]));
            }

            double expectedLogWeightRatio =
                    (poissonExpectedLogWeight - normalExpectedLogWeight)
                    * static_cast<double>(nSamples[n]);

            TDoubleVec logWeights = filter.logWeights();
            double logWeightRatio = logWeights[0] - logWeights[1];

            LOG_DEBUG("expectedLogWeightRatio = " << expectedLogWeightRatio
                      << ", logWeightRatio = " << logWeightRatio);

            CPPUNIT_ASSERT(std::fabs(logWeightRatio - expectedLogWeightRatio)
                           / std::fabs(expectedLogWeightRatio) < 0.35);
        }
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
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
        maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight);
        maths::CNormalMeanPrecConjugate normal =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
        maths::COneOfNPrior::TPriorPtrVec mode;
        mode.push_back(COneOfNPrior::TPriorPtr(normal.clone()));
        models.push_back(TPriorPtr(new maths::CMultimodalPrior(maths_t::E_ContinuousData,
                                                               clusterer,
                                                               maths::COneOfNPrior(mode, maths_t::E_ContinuousData))));
        COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            filter.addSamples(TDouble1Vec(1, samples[i]));
        }

        TDoubleVec logWeights = filter.logWeights();
        double logWeightRatio = logWeights[0] - logWeights[1];

        LOG_DEBUG("logWeightRatio = " << logWeightRatio);
        CPPUNIT_ASSERT(std::exp(logWeightRatio) < 1e-6);
  }
}

void COneOfNPriorTest::testMarginalLikelihood()
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testMarginalLikelihood  |");
    LOG_DEBUG("+--------------------------------------------+");

    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] = { E_ContinuousData, E_IntegerData };

    for (std::size_t t = 0u; t < boost::size(dataTypes); ++t)
    {
        TPriorPtrVec models;
        models.push_back(TPriorPtr(CPoissonMeanConjugate::nonInformativePrior().clone()));
        models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(dataTypes[t]).clone()));
        models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(dataTypes[t]).clone()));
        models.push_back(TPriorPtr(CGammaRateConjugate::nonInformativePrior(dataTypes[t]).clone()));
        COneOfNPrior filter(maths::COneOfNPrior(clone(models), dataTypes[t]));

        const double location = 1.0;
        const double squareScale = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, 10, samples);
        filter.addSamples(samples);

        maths_t::ESampleWeightStyle weightStyles[] =
            {
                maths_t::E_SampleCountWeight,
                maths_t::E_SampleWinsorisationWeight,
                maths_t::E_SampleCountWeight
            };
        double weights[] = { 0.1, 1.0, 10.0 };

        for (std::size_t i = 0u; i < boost::size(weightStyles); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(weights); ++j)
            {
                double lb, ub;
                filter.minusLogJointCdf(maths_t::TWeightStyleVec(1, weightStyles[i]),
                                        TDouble1Vec(1, 10000.0),
                                        TDouble4Vec1Vec(1, TDouble4Vec(1, weights[j])),
                                        lb, ub);
                LOG_DEBUG("-log(c.d.f) = " << (lb + ub) / 2.0);
                CPPUNIT_ASSERT(lb >= 0.0);
                CPPUNIT_ASSERT(ub >= 0.0);
            }
        }
    }

    // Check that the marginal likelihood and c.d.f. agree for some
    // test data, that the c.d.f. <= 1 and c.d.f. + c.d.f. complement = 1.

    static const double EPS = 1e-3;

    test::CRandomNumbers rng;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

    TDoubleVec samples;
    rng.generateLogNormalSamples(1.0, 1.0, 99, samples);

    for (std::size_t i = 0u; i < 2; ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }
    for (std::size_t i = 2u; i < samples.size(); ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(99.0);
        LOG_DEBUG("interval = " << core::CContainerPrinter::print(interval));

        double x = interval.first;
        double dx = (interval.second - interval.first) / 20.0;
        for (std::size_t j = 0u; j < 20; ++j, x += dx)
        {
            double fx;
            CPPUNIT_ASSERT(filter.jointLogMarginalLikelihood(TDouble1Vec(1, x), fx)
                               == maths_t::E_FpNoErrors);
            fx = std::exp(fx);

            double lb;
            double ub;
            CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x + EPS), lb, ub));
            double FxPlusEps = std::exp(-(lb + ub) / 2.0);

            CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x - EPS), lb, ub));
            double FxMinusEps = std::exp(-(lb + ub) / 2.0);

            double dFdx = (FxPlusEps - FxMinusEps) / (2.0 * EPS);

            LOG_DEBUG("x = " << x << ", f(x) = " << fx << ", dF(x)/dx = " << dFdx);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(fx, dFdx, std::max(1e-5, 1e-3 * FxPlusEps));


            CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x), lb, ub));
            double Fx = std::exp(-(lb + ub) / 2.0);

            CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lb, ub));
            double FxComplement = std::exp(-(lb + ub) / 2.0);
            LOG_DEBUG("F(x) = " << Fx << " 1 - F(x) = " << FxComplement);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, Fx + FxComplement, 1e-3);
        }
    }
}

void COneOfNPriorTest::testMarginalLikelihoodMean()
{
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testMarginalLikelihoodMean  |");
    LOG_DEBUG("+------------------------------------------------+");

    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("****** Normal ******");

        const double means[] = { 10.0, 50.0 };
        const double variances[] = { 1.0, 10.0 };

        for (std::size_t i = 0u; i < boost::size(means); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(variances); ++j)
            {
                LOG_DEBUG("*** mean = " << means[i] << ", variance = " << variances[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

                COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateNormalSamples(means[i], variances[j], 100, samples);

                for (std::size_t k = 0u; k < samples.size(); ++k)
                {
                    filter.addSamples(TDouble1Vec(1, samples[k]));

                    double expectedMean;
                    CPPUNIT_ASSERT(filter.marginalLikelihoodMeanForTest(expectedMean));

                    if (k % 10 == 0)
                    {
                        LOG_DEBUG("marginalLikelihoodMean = " << filter.marginalLikelihoodMean()
                                  << ", expectedMean = " << expectedMean);
                    }

                    // The error is at the precision of the numerical integration.
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean,
                                                 filter.marginalLikelihoodMean(),
                                                 0.01 * expectedMean);
                }
            }
        }
    }

    {
        LOG_DEBUG("****** Log Normal ******");

        const double locations[] = { 0.1, 1.0 };
        const double squareScales[] = { 0.1, 1.0 };

        for (std::size_t i = 0u; i < boost::size(locations); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(squareScales); ++j)
            {
                LOG_DEBUG("*** location = " << locations[i]
                          << ", squareScale = " << squareScales[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

                COneOfNPrior filter(maths::COneOfNPrior(models, E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateLogNormalSamples(locations[i], squareScales[j], 10, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateLogNormalSamples(locations[i], squareScales[j], 100, samples);

                TMeanAccumulator relativeError;

                for (std::size_t k = 0u; k < samples.size(); ++k)
                {
                    filter.addSamples(TDouble1Vec(1, samples[k]));

                    double expectedMean;
                    CPPUNIT_ASSERT(filter.marginalLikelihoodMeanForTest(expectedMean));

                    if (k % 10 == 0)
                    {
                        LOG_DEBUG("marginalLikelihoodMean = " << filter.marginalLikelihoodMean()
                                  << ", expectedMean = " << expectedMean);
                    }

                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean,
                                                 filter.marginalLikelihoodMean(),
                                                 0.2 * expectedMean);

                    relativeError.add(std::fabs(filter.marginalLikelihoodMean() - expectedMean) / expectedMean);
                }

                LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 0.02);
            }
        }
    }
}

void COneOfNPriorTest::testMarginalLikelihoodMode()
{
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testMarginalLikelihoodMode  |");
    LOG_DEBUG("+------------------------------------------------+");

    // Test that the marginal likelihood mode is near the maximum
    // of the marginal likelihood.

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("****** Normal ******");

        const double means[] = { 10.0, 50.0 };
        const double variances[] = { 1.0, 10.0 };

        for (std::size_t i = 0u; i < boost::size(means); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(variances); ++j)
            {
                LOG_DEBUG("*** mean = " << means[i] << ", variance = " << variances[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

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
                maths::CCompositeFunctions::CExp<maths::CPrior::CLogMarginalLikelihood> likelihood(filter);
                double a = means[i] - 2.0 * std::sqrt(variances[j]);
                double b = means[i] + 2.0 * std::sqrt(variances[j]);
                maths::CSolvers::maximize(a, b, likelihood(a), likelihood(b), likelihood, 0.0, iterations, mode, fmode);

                LOG_DEBUG("marginalLikelihoodMode = " << filter.marginalLikelihoodMode()
                          << ", expectedMode = " << mode);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(mode, filter.marginalLikelihoodMode(), 0.01 * mode);
            }
        }
    }

    {
        LOG_DEBUG("****** Log Normal ******");

        const double locations[] = { 0.1, 1.0 };
        const double squareScales[] = { 0.1, 2.0 };

        for (std::size_t i = 0u; i < boost::size(locations); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(squareScales); ++j)
            {
                LOG_DEBUG("*** location = " << locations[i]
                          << ", squareScale = " << squareScales[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

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
                maths::CCompositeFunctions::CExp<maths::CPrior::CLogMarginalLikelihood> likelihood(filter);
                boost::math::lognormal_distribution<> logNormal(locations[i], std::sqrt(squareScales[j]));
                double a = 0.01;
                double b = boost::math::mode(logNormal) + 1.0 * boost::math::standard_deviation(logNormal);
                maths::CSolvers::maximize(a, b, likelihood(a), likelihood(b), likelihood, 0.0, iterations, mode, fmode);

                LOG_DEBUG("marginalLikelihoodMode = " << filter.marginalLikelihoodMode()
                          << ", expectedMode = " << mode);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(mode, filter.marginalLikelihoodMode(), 0.05 * mode);
            }
        }
    }
}

void COneOfNPriorTest::testMarginalLikelihoodVariance()
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testMarginalLikelihoodVariance  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("****** Normal ******");

        double means[] = { 10.0, 100.0 };
        double variances[] = { 1.0, 10.0 };

        for (std::size_t i = 0u; i < boost::size(means); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(variances); ++j)
            {
                LOG_DEBUG("*** mean = " << means[i]
                          << ", variance = " << variances[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
                models.push_back(TPriorPtr(CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));

                COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateNormalSamples(means[i], variances[j], 100, samples);

                TMeanAccumulator relativeError;
                for (std::size_t k = 0u; k < samples.size(); ++k)
                {
                    filter.addSamples(TDouble1Vec(1, samples[k]));
                    double expectedVariance;
                    CPPUNIT_ASSERT(filter.marginalLikelihoodVarianceForTest(expectedVariance));
                    if (k % 10 == 0)
                    {
                        LOG_DEBUG("marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                                  << ", expectedVariance = " << expectedVariance);
                    }

                    // The error is at the precision of the numerical integration.
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedVariance,
                                                 filter.marginalLikelihoodVariance(),
                                                 0.02 * expectedVariance);

                    relativeError.add(std::fabs(expectedVariance - filter.marginalLikelihoodVariance())
                                      / expectedVariance);
                }

                LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 2e-3);
            }
        }
    }

    {
        LOG_DEBUG("****** Gamma ******");

        const double shapes[] = { 5.0, 20.0, 40.0 };
        const double scales[] = { 1.0, 10.0, 20.0 };

        for (std::size_t i = 0u; i < boost::size(shapes); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(scales); ++j)
            {
                LOG_DEBUG("*** shape = " << shapes[i]
                          << ", scale = " << scales[j] << " ***");

                TPriorPtrVec models;
                models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
                models.push_back(TPriorPtr(CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));

                COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

                TDoubleVec seedSamples;
                rng.generateGammaSamples(shapes[i], scales[j], 10, seedSamples);
                filter.addSamples(seedSamples);

                TDoubleVec samples;
                rng.generateGammaSamples(shapes[i], scales[j], 100, samples);

                TMeanAccumulator relativeError;

                for (std::size_t k = 0u; k < samples.size(); ++k)
                {
                    filter.addSamples(TDouble1Vec(1, samples[k]));

                    double expectedVariance;
                    CPPUNIT_ASSERT(filter.marginalLikelihoodVarianceForTest(expectedVariance));

                    if (k % 10 == 0)
                    {
                        LOG_DEBUG("marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                                  << ", expectedVariance = " << expectedVariance);
                    }

                    // The error is mainly due to the truncation in the
                    // integration range used to compute the expected mean.
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedVariance,
                                                 filter.marginalLikelihoodVariance(),
                                                 0.01 * expectedVariance);

                    relativeError.add(std::fabs(expectedVariance - filter.marginalLikelihoodVariance())
                                      / expectedVariance);
                }

                LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 3e-3);
            }
        }
    }
}

void COneOfNPriorTest::testSampleMarginalLikelihood()
{
    LOG_DEBUG("+--------------------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG("+--------------------------------------------------+");

    // Test we sample the constitute priors in proportion to their weights.

    const double mean = 5.0;
    const double variance = 2.0;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 20, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    TDoubleVec weights = filter.weights();
    LOG_DEBUG("weights = " << core::CContainerPrinter::print(weights));

    TDouble1Vec sampled;
    filter.sampleMarginalLikelihood(10, sampled);

    // We expect 5 samples from the normal and 5 from the log normal.
    COneOfNPrior::TPriorCPtrVec posteriorModels = filter.models();
    TDouble1Vec normalSamples;
    posteriorModels[0]->sampleMarginalLikelihood(5, normalSamples);
    TDouble1Vec logNormalSamples;
    posteriorModels[1]->sampleMarginalLikelihood(5, logNormalSamples);

    TDoubleVec expectedSampled(normalSamples);
    expectedSampled.insert(expectedSampled.end(),
                           logNormalSamples.begin(),
                           logNormalSamples.end());

    LOG_DEBUG("expected samples = " << core::CContainerPrinter::print(expectedSampled)
              << ", samples = " << core::CContainerPrinter::print(sampled));

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSampled),
                         core::CContainerPrinter::print(sampled));

    rng.generateNormalSamples(mean, variance, 80, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    weights = filter.weights();
    LOG_DEBUG("weights = " << core::CContainerPrinter::print(weights));

    filter.sampleMarginalLikelihood(20, sampled);

    // We expect 20 samples from the normal and 0 from the log normal.
    posteriorModels = filter.models();
    posteriorModels[0]->sampleMarginalLikelihood(20, normalSamples);
    posteriorModels[1]->sampleMarginalLikelihood(0, logNormalSamples);

    expectedSampled = normalSamples;
    expectedSampled.insert(expectedSampled.end(),
                           logNormalSamples.begin(),
                           logNormalSamples.end());

    LOG_DEBUG("expected samples = " << core::CContainerPrinter::print(expectedSampled)
              << ", samples = " << core::CContainerPrinter::print(sampled));

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSampled),
                         core::CContainerPrinter::print(sampled));
}

void COneOfNPriorTest::testCdf()
{
    LOG_DEBUG("+-----------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testCdf  |");
    LOG_DEBUG("+-----------------------------+");

    // Test error cases and the invariant "cdf" + "cdf complement" = 1

    const double mean = 20.0;
    const double variance = 5.0;
    const std::size_t n[] = { 20u, 80u };

    test::CRandomNumbers rng;

    TPriorPtrVec models;
    models.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    models.push_back(TPriorPtr(CGammaRateConjugate::nonInformativePrior(E_ContinuousData).clone()));
    COneOfNPrior filter(maths::COneOfNPrior(clone(models), E_ContinuousData));

    for (std::size_t i = 0u; i < boost::size(n); ++i)
    {
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, n[i], samples);

        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            filter.addSamples(TDouble1Vec(1, samples[j]));
        }

        double lb, ub;
        CPPUNIT_ASSERT(!filter.minusLogJointCdf(TDouble1Vec(), lb, ub));
        CPPUNIT_ASSERT(!filter.minusLogJointCdfComplement(TDouble1Vec(), lb, ub));

        for (std::size_t j = 1u; j < 500; ++j)
        {
            double x = static_cast<double>(j) / 2.0;

            CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x), lb, ub));
            double f = (lb + ub) / 2.0;
            CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lb, ub));
            double fComplement = (lb + ub) / 2.0;

            LOG_DEBUG("log(F(x)) = " << (f == 0.0 ? f : -f)
                      << ", log(1 - F(x)) = " << (fComplement == 0.0 ? fComplement : -fComplement));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, std::exp(-f) + std::exp(-fComplement), 1e-10);
        }
    }
}

void COneOfNPriorTest::testProbabilityOfLessLikelySamples()
{
    LOG_DEBUG("+--------------------------------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+--------------------------------------------------------+");

    // We simply test that the calculation yields the weighted sum
    // of component model calculations (which is its definition).

    const double location = 0.7;
    const double squareScale = 1.3;
    const double vs[] = { 0.5, 1.0, 2.0 };

    TPriorPtrVec initialModels;
    initialModels.push_back(TPriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));
    initialModels.push_back(TPriorPtr(CLogNormalMeanPrecConjugate::nonInformativePrior(E_ContinuousData).clone()));

    COneOfNPrior filter(maths::COneOfNPrior(clone(initialModels), E_ContinuousData));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(location, squareScale, 200, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        TDouble1Vec sample(1, samples[i]);
        filter.addSamples(sample);

        double lb, ub;
        maths_t::ETail tail;

        CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lb, ub));

        CPPUNIT_ASSERT_EQUAL(lb, ub);
        double probability = (lb + ub) / 2.0;

        double expectedProbability = 0.0;

        TDoubleVec weights(filter.weights());
        COneOfNPrior::TPriorCPtrVec models(filter.models());
        for (std::size_t j = 0u; j < weights.size(); ++j)
        {
            double weight = weights[j];
            CPPUNIT_ASSERT(models[j]->probabilityOfLessLikelySamples(
                                              maths_t::E_TwoSided,
                                              maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight),
                                              TDouble1Vec(1, sample[0]),
                                              TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)),
                                              lb, ub, tail));
            CPPUNIT_ASSERT_EQUAL(lb, ub);
            double modelProbability = (lb + ub) / 2.0;
            expectedProbability += weight * modelProbability;
        }

        LOG_DEBUG("weights = " << core::CContainerPrinter::print(weights)
                  << ", expectedProbability = " << expectedProbability
                  << ", probability = " << probability);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProbability,
                                     probability,
                                     1e-3 * std::max(expectedProbability, probability));

        maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);

        for (std::size_t k = 0u; ((i+1) % 11 == 0) && k < boost::size(vs); ++k)
        {
            double mode = filter.marginalLikelihoodMode(weightStyle, TDouble4Vec(1, vs[k]));
            double ss[] = { 0.9 * mode, 1.1 * mode };

            LOG_DEBUG("vs = " << vs[k] << ", mode = " << mode);

            if (mode > 0.0)
            {
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyle,
                                                      TDouble1Vec(1, ss[0]),
                                                      TDouble4Vec1Vec(1, TDouble4Vec(1, vs[k])),
                                                      lb, ub, tail);
                CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
                if (mode > 0.0)
                {
                    filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                          weightStyle,
                                                          TDouble1Vec(ss, ss + 2),
                                                          TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                          lb, ub, tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                    filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedBelow,
                                                          weightStyle,
                                                          TDouble1Vec(ss, ss + 2),
                                                          TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                          lb, ub, tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
                    filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedAbove,
                                                          weightStyle,
                                                          TDouble1Vec(ss, ss + 2),
                                                          TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                          lb, ub, tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
                }
            }
            if (mode > 0.0)
            {
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyle,
                                                      TDouble1Vec(1, ss[1]),
                                                      TDouble4Vec1Vec(1, TDouble4Vec(1, vs[k])),
                                                      lb, ub, tail);
                CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyle,
                                                      TDouble1Vec(ss, ss + 2),
                                                      TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                      lb, ub, tail);
                CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedBelow,
                                                      weightStyle,
                                                      TDouble1Vec(ss, ss + 2),
                                                      TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                      lb, ub, tail);
                CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
                filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedAbove,
                                                      weightStyle,
                                                      TDouble1Vec(ss, ss + 2),
                                                      TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                      lb, ub, tail);
                CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
            }
        }
    }
}

void COneOfNPriorTest::testPersist()
{
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  COneOfNPriorTest::testPersist  |");
    LOG_DEBUG("+---------------------------------+");

    // Check that persist/restore is idempotent.

    TPriorPtrVec models;
    models.push_back(TPriorPtr(
            CPoissonMeanConjugate::nonInformativePrior().clone()));
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
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        origFilter.addSamples(maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight),
                              TDouble1Vec(1, samples[i]),
                              TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)));
    }
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("One-of-N prior XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
 
    maths::SDistributionRestoreParams params(E_IntegerData,
                                             decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::COneOfNPrior restoredFilter(params, traverser);

    LOG_DEBUG("orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test *COneOfNPriorTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("COneOfNPriorTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testFilter",
                                   &COneOfNPriorTest::testFilter) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testMultipleUpdate",
                                   &COneOfNPriorTest::testMultipleUpdate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testWeights",
                                   &COneOfNPriorTest::testWeights) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testModels",
                                   &COneOfNPriorTest::testModels) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testModelSelection",
                                   &COneOfNPriorTest::testModelSelection) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testMarginalLikelihood",
                                   &COneOfNPriorTest::testMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testSampleMarginalLikelihood",
                                   &COneOfNPriorTest::testSampleMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testMarginalLikelihoodMean",
                                   &COneOfNPriorTest::testMarginalLikelihoodMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testMarginalLikelihoodMode",
                                   &COneOfNPriorTest::testMarginalLikelihoodMode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testMarginalLikelihoodVariance",
                                   &COneOfNPriorTest::testMarginalLikelihoodVariance) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testCdf",
                                   &COneOfNPriorTest::testCdf) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testProbabilityOfLessLikelySamples",
                                   &COneOfNPriorTest::testProbabilityOfLessLikelySamples) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COneOfNPriorTest>(
                                   "COneOfNPriorTest::testPersist",
                                   &COneOfNPriorTest::testPersist) );

    return suiteOfTests;
}



