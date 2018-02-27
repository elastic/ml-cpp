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

#include "CLogNormalMeanPrecConjugateTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include "TestUtils.h"

#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

#include <math.h>

using namespace ml;
using namespace handy_typedefs;

namespace
{

typedef std::vector<double> TDoubleVec;
typedef std::pair<double, double> TDoubleDoublePr;
typedef std::vector<TDoubleDoublePr> TDoubleDoublePrVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TMeanVarAccumulator;
typedef CPriorTestInterfaceMixin<maths::CLogNormalMeanPrecConjugate> CLogNormalMeanPrecConjugate;

CLogNormalMeanPrecConjugate makePrior(maths_t::EDataType dataType = maths_t::E_ContinuousData,
                                      const double &offset = 0.0,
                                      const double &decayRate = 0.0)
{
    return CLogNormalMeanPrecConjugate::nonInformativePrior(dataType, offset, decayRate, 0.0);
}

}

void CLogNormalMeanPrecConjugateTest::testMultipleUpdate(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testMultipleUpdate  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    typedef maths::CEqualWithTolerance<double> TEqual;

    const maths_t::EDataType dataTypes[] =
        {
            maths_t::E_IntegerData,
            maths_t::E_ContinuousData
        };

    const double location = ::log(10.0);
    const double squareScale = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(location, squareScale, 100, samples);

    for (std::size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        CLogNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CLogNormalMeanPrecConjugate filter2(filter1);

        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            filter1.addSamples(TDouble1Vec(1, samples[j]));
        }
        filter2.addSamples(samples);

        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-2);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }

    // Test with variance scale.

    double mean = ::exp(location + squareScale / 2.0);
    double variance = mean * mean * (::exp(squareScale) - 1.0);
    LOG_DEBUG("mean = " << mean << " variance = " << variance);

    double scaledSquareScale = ::log(1.0 + 2.0 * variance / mean / mean);
    double scaledLocation = ::log(mean) - scaledSquareScale / 2.0;
    double scaledMean = ::exp(scaledLocation + scaledSquareScale / 2.0);
    double scaledVariance = scaledMean * scaledMean * (::exp(scaledSquareScale) - 1.0);
    LOG_DEBUG("scaled mean = " << scaledMean << " scaled variance = " << scaledVariance);

    TDoubleVec scaledSamples;
    rng.generateLogNormalSamples(scaledLocation, scaledSquareScale, 100, scaledSamples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        CLogNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        filter1.addSamples(samples);
        CLogNormalMeanPrecConjugate filter2(filter1);

        maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
        for (std::size_t j = 0u; j < scaledSamples.size(); ++j)
        {
            filter1.addSamples(weightStyle,
                               TDouble1Vec(1, scaledSamples[j]),
                               TDouble4Vec1Vec(1, TDouble4Vec(1, 2.0)));
        }
        filter2.addSamples(weightStyle,
                           scaledSamples,
                           TDouble4Vec1Vec(scaledSamples.size(), TDouble4Vec(1, 2.0)));

        LOG_DEBUG(filter1.print());
        LOG_DEBUG("vs");
        LOG_DEBUG(filter2.print());
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.015);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }

    // Test the count weight is equivalent to adding repeated samples.

    for (size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        CLogNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CLogNormalMeanPrecConjugate filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;

        for (std::size_t j = 0u; j < count; ++j)
        {
            filter1.addSamples(TDouble1Vec(1, x));
        }
        filter2.addSamples(maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight),
                           TDouble1Vec(1, x),
                           TDouble4Vec1Vec(1, TDouble4Vec(1, static_cast<double>(count))));

        LOG_DEBUG(filter1.print());
        LOG_DEBUG("vs");
        LOG_DEBUG(filter2.print());
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-3);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }
}

void CLogNormalMeanPrecConjugateTest::testPropagation(void)
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testPropagation  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that propagation doesn't affect the expected values
    // of likelihood mean and precision.

    const double eps = 1e-12;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(1.0, 0.3, 500, samples);

    CLogNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, 0.1));

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
    }

    double mean = filter.normalMean();
    double precision = filter.normalPrecision();

    filter.propagateForwardsByTime(5.0);

    double propagatedMean = filter.normalMean();
    double propagatedPrecision = filter.normalPrecision();

    LOG_DEBUG("mean = " << mean
              << ", precision = " << precision
              << ", propagatedMean = " << propagatedMean
              << ", propagatedPrecision = " << propagatedPrecision);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(mean, propagatedMean, eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(precision, propagatedPrecision, eps);
}

void CLogNormalMeanPrecConjugateTest::testMeanEstimation(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testMeanEstimation  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    // We are going to test that we correctly estimate the distribution
    // for the mean of the exponentiated Gaussian of a log-normal process
    // by checking that the true mean lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = { 0.0, 0.001, 0.01 };

    const unsigned int nTests = 500u;
    const double testIntervals[] = { 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0 };

    for (size_t i = 0; i < boost::size(decayRates); ++i)
    {
        test::CRandomNumbers rng;

        unsigned int errors[] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };

        for (unsigned int test = 0; test < nTests; ++test)
        {
            double location = ::log(0.5 * (test + 1));
            double squareScale = 4.0;

            TDoubleVec samples;
            rng.generateLogNormalSamples(location, squareScale, 500, samples);

            CLogNormalMeanPrecConjugate filter(
                    makePrior(maths_t::E_ContinuousData, 0.0, decayRates[i]));

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (size_t j = 0u; j < boost::size(testIntervals); ++j)
            {
                TDoubleDoublePr confidenceInterval =
                        filter.confidenceIntervalNormalMean(testIntervals[j]);
                if (location < confidenceInterval.first ||
                    location > confidenceInterval.second)
                {
                    ++errors[j];
                }
            }
        }

        for (size_t j = 0; j < boost::size(testIntervals); ++j)
        {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_DEBUG("interval = " << interval
                      << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            if (decayRates[i] == 0.0)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[j]), 4.0);
            }
            else
            {
                CPPUNIT_ASSERT(interval <= (100.0 - testIntervals[j]));
            }
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testPrecisionEstimation(void)
{
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testPrecisionEstimation  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    // We are going to test that we correctly estimate a distribution for
    // the precision of the exponentiated Gaussian of a log-normal process by
    // checking that the true precision lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = { 0.0, 0.001, 0.01 };

    const unsigned int nTests = 500u;
    const double testIntervals[] = { 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0 };

    for (size_t i = 0; i < boost::size(decayRates); ++i)
    {
        test::CRandomNumbers rng;

        unsigned int errors[] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };

        for (unsigned int test = 0; test < nTests; ++test)
        {
            double location = 1.0;
            double squareScale = 0.002 * static_cast<double>(test + 1);
            double precision = 1 / squareScale;

            TDoubleVec samples;
            rng.generateLogNormalSamples(location, squareScale, 500, samples);

            CLogNormalMeanPrecConjugate filter(
                    makePrior(maths_t::E_ContinuousData, 0.0, decayRates[i]));

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (size_t j = 0; j < boost::size(testIntervals); ++j)
            {
                TDoubleDoublePr confidenceInterval =
                        filter.confidenceIntervalNormalPrecision(testIntervals[j]);

                if (precision < confidenceInterval.first ||
                    precision > confidenceInterval.second)
                {
                    ++errors[j];
                }
            }
        }

        for (size_t j = 0; j < boost::size(testIntervals); ++j)
        {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_DEBUG("interval = " << interval
                      << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            if (decayRates[i] == 0.0)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[j]), 4.0);
            }
            else
            {
                CPPUNIT_ASSERT(interval <= (100.0 - testIntervals[j]));
            }
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testMarginalLikelihood(void)
{
    LOG_DEBUG("+-----------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testMarginalLikelihood  |");
    LOG_DEBUG("+-----------------------------------------------------------+");

    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] =
        {
            maths_t::E_ContinuousData,
            maths_t::E_IntegerData
        };
    for (std::size_t t = 0u; t < boost::size(dataTypes); ++t)
    {
        CLogNormalMeanPrecConjugate filter(makePrior(dataTypes[t]));

        const double location = 1.0;
        const double squareScale = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, 200, samples);
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
    // test data and that the c.d.f. <= 1 and that the expected value
    // of the log likelihood tends to the differential entropy.

    const double decayRates[] = { 0.0, 0.001, 0.01 };

    const double location = 3.0;
    const double squareScale = 1.0;

    unsigned int numberSamples[] = { 2u, 10u, 500u };
    const double tolerance = 1e-3;

    test::CRandomNumbers rng;

    for (size_t i = 0; i < boost::size(numberSamples); ++i)
    {
        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, numberSamples[i], samples);

        for (size_t j = 0; j < boost::size(decayRates); ++j)
        {
            CLogNormalMeanPrecConjugate filter(
                    makePrior(maths_t::E_ContinuousData, 0.0, decayRates[j]));

            for (std::size_t k = 0u; k < samples.size(); ++k)
            {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                filter.propagateForwardsByTime(1.0);
            }

            // We'll check that the p.d.f. is close to the derivative of the
            // c.d.f. at a range of deltas from the true location.

            const double eps = 1e-4;
            double deltas[] =
                {
                    -5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0
                };

            for (size_t k = 0; k < boost::size(deltas); ++k)
            {
                double x = ::exp(location + deltas[k] * ::sqrt(squareScale));
                TDouble1Vec sample(1, x);

                LOG_DEBUG("number = " << numberSamples[i]
                          << ", sample = " << sample[0]);

                double logLikelihood = 0.0;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                     filter.jointLogMarginalLikelihood(sample, logLikelihood));
                double pdf = ::exp(logLikelihood);

                double lowerBound = 0.0, upperBound = 0.0;
                sample[0] -= eps;
                CPPUNIT_ASSERT(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                double minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtMinusEps = ::exp(-minusLogCdf);
                CPPUNIT_ASSERT(minusLogCdf >= 0.0);

                sample[0] += 2.0 * eps;
                CPPUNIT_ASSERT(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtPlusEps = ::exp(-minusLogCdf);
                CPPUNIT_ASSERT(minusLogCdf >= 0.0);

                double dcdfdx = (cdfAtPlusEps - cdfAtMinusEps) / 2.0 / eps;

                LOG_DEBUG("pdf(x) = " << pdf << ", d(cdf)/dx = " << dcdfdx);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(pdf, dcdfdx, tolerance);
            }
        }
    }

    {
        // Test that the sample mean of the log likelihood tends to the
        // expected log likelihood for a log-normal distribution (uniform
        // law of large numbers), which is just the differential entropy
        // of a log-normal R.V.

        boost::math::lognormal_distribution<> logNormal(location, ::sqrt(squareScale));
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(logNormal);

        CLogNormalMeanPrecConjugate filter(makePrior());

        double differentialEntropy = 0.0;

        TDoubleVec seedSamples;
        rng.generateLogNormalSamples(location, squareScale, 100, seedSamples);
        filter.addSamples(seedSamples);

        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, 100000, samples);
        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            TDouble1Vec sample(1, samples[i]);
            filter.addSamples(sample);
            double logLikelihood = 0.0;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                 filter.jointLogMarginalLikelihood(sample, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(samples.size());

        LOG_DEBUG("differentialEntropy = " << differentialEntropy
                  << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 5e-3);
    }

    {
        const double varianceScales[] =
            {
                0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0,
                1.2, 1.5, 2.0, 2.5, 3.0,
                4.0, 5.0
            };
        boost::math::lognormal_distribution<> logNormal(location, ::sqrt(squareScale));

        CLogNormalMeanPrecConjugate filter(makePrior());
        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, 1000, samples);
        filter.addSamples(samples);

        const double percentages[] =
            {
                5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 95.0
            };

        {
            // Test that marginal likelihood confidence intervals are
            // what we'd expect for various variance scales.

            TMeanAccumulator error;
            for (std::size_t i = 0u; i < boost::size(percentages); ++i)
            {
                double q1, q2;
                filter.marginalLikelihoodQuantileForTest(50.0 - percentages[i] / 2.0, 1e-3, q1);
                filter.marginalLikelihoodQuantileForTest(50.0 + percentages[i] / 2.0, 1e-3, q2);
                TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(percentages[i]);
                LOG_DEBUG("[q1, q2] = [" << q1 << ", " << q2 << "]"
                          << ", interval = " << core::CContainerPrinter::print(interval));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, 1e-3);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 1e-3);
                error.add(::fabs(interval.first - q1));
                error.add(::fabs(interval.second - q2));
            }
            LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 1e-3);
        }
        {
            TMeanAccumulator totalError;
            for (std::size_t i = 0u; i < boost::size(varianceScales); ++i)
            {
                TMeanAccumulator error;
                double vs = varianceScales[i];
                double shift = ::log(1.0 + vs * (::exp(squareScale) - 1.0)) - squareScale;
                double shiftedLocation = location - 0.5 * shift;
                double shiftedSquareScale = squareScale + shift;
                boost::math::lognormal_distribution<> scaledLogNormal(shiftedLocation,
                                                                      ::sqrt(shiftedSquareScale));
                LOG_DEBUG("*** vs = " <<   boost::math::variance(scaledLogNormal)
                                         / boost::math::variance(logNormal) << " ***");
                for (std::size_t j = 0u; j < boost::size(percentages); ++j)
                {
                    double q1 = boost::math::quantile(scaledLogNormal, (50.0 - percentages[j] / 2.0) / 100.0);
                    double q2 = boost::math::quantile(scaledLogNormal, (50.0 + percentages[j] / 2.0) / 100.0);
                    TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(
                                                              percentages[j],
                                                              maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight),
                                                              TDouble4Vec(1, vs));
                    LOG_DEBUG("[q1, q2] = [" << q1 << ", " << q2 << "]"
                              << ", interval = " << core::CContainerPrinter::print(interval));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, std::max(0.5, 0.2 * q1));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.1 * q2);
                    error.add(::fabs(interval.first - q1) / q1);
                    error.add(::fabs(interval.second - q2) / q2);
                }
                LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.07);
                totalError += error;
            }
            LOG_DEBUG("totalError = " << maths::CBasicStatistics::mean(totalError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(totalError) < 0.06);
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMean(void)
{
    LOG_DEBUG("+---------------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMean  |");
    LOG_DEBUG("+---------------------------------------------------------------+");

    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    const double locations[] = { 0.1, 1.0, 3.0 };
    const double squareScales[] = { 0.1, 1.0, 3.0 };

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(locations); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(squareScales); ++j)
        {
            LOG_DEBUG("*** location = " << locations[i]
                      << ", squareScale = " << squareScales[j] << " ***");

            CLogNormalMeanPrecConjugate filter(makePrior());

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
                                             0.35 * expectedMean);

                relativeError.add(::fabs(filter.marginalLikelihoodMean() - expectedMean)
                                  / expectedMean);
            }

            LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 0.07);
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMode(void)
{
    LOG_DEBUG("+---------------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMode  |");
    LOG_DEBUG("+---------------------------------------------------------------+");

    // Test that the marginal likelihood mode is what we'd expect
    // with variances variance scales.

    const double locations[] = { 0.1, 1.0, 3.0 };
    const double squareScales[] = { 0.1, 1.0, 3.0 };
    const double varianceScales[] =
        {
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
        };

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(locations); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(squareScales); ++j)
        {
            LOG_DEBUG("*** location = " << locations[i]
                      << ", squareScale = " << squareScales[j] << " ***");

            boost::math::lognormal_distribution<> logNormal(locations[i],
                                                            ::sqrt(squareScales[j]));

            CLogNormalMeanPrecConjugate filter(makePrior());
            TDoubleVec samples;
            rng.generateLogNormalSamples(locations[i], squareScales[j], 1000, samples);
            filter.addSamples(samples);

            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
            TDouble4Vec weight(1, 1.0);
            TMeanAccumulator error;
            for (std::size_t k = 0u; k < boost::size(varianceScales); ++k)
            {
                double vs = varianceScales[k];
                weight[0] = vs;
                double shift = ::log(1.0 + vs * (::exp(squareScales[j]) - 1.0)) - squareScales[j];
                double shiftedLocation = locations[i] - 0.5 * shift;
                double shiftedSquareScale = squareScales[j] + shift;
                boost::math::lognormal_distribution<> scaledLogNormal(shiftedLocation,
                                                                      ::sqrt(shiftedSquareScale));
                double expectedMode = boost::math::mode(scaledLogNormal);
                LOG_DEBUG("dm = " <<   boost::math::mean(scaledLogNormal)
                                     - boost::math::mean(logNormal)
                          << ", vs = " <<   boost::math::variance(scaledLogNormal)
                                          / boost::math::variance(logNormal)
                          << ", marginalLikelihoodMode = " << filter.marginalLikelihoodMode(weightStyle, weight)
                          << ", expectedMode = " << expectedMode);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode,
                                             filter.marginalLikelihoodMode(weightStyle, weight),
                                             1.0);
                error.add(::fabs(filter.marginalLikelihoodMode(weightStyle, weight) - expectedMode));
            }
            LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.26);
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance(void)
{
    LOG_DEBUG("+-------------------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance  |");
    LOG_DEBUG("+-------------------------------------------------------------------+");

    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    const double locations[] = { 0.1, 1.0, 3.0 };
    const double squareScales[] = { 0.1, 1.0, 3.0 };

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(locations); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(squareScales); ++j)
        {
            LOG_DEBUG("*** location = " << locations[i]
                      << ", squareScale = " << squareScales[j] << " ***");

            CLogNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateLogNormalSamples(locations[i], squareScales[j], 10, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateLogNormalSamples(locations[i], squareScales[j], 200, samples);

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

                relativeError.add(::fabs(filter.marginalLikelihoodVariance() - expectedVariance)
                                  / expectedVariance);
            }

            LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 0.23);
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testSampleMarginalLikelihood(void)
{
    LOG_DEBUG("+-----------------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG("+-----------------------------------------------------------------+");

    // We're going to test two properties of the sampling:
    //   1) That the sample mean is equal to the marginal
    //      likelihood mean.
    //   2) That the sample percentiles match the distribution
    //      percentiles.
    // I want to cross check these with the implementations of the
    // jointLogMarginalLikelihood and minusLogJointCdf so use these
    // to compute the mean and percentiles.

    const double mean = 0.9;
    const double squareScale = 1.2;

    const double eps = 1e-3;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(mean, squareScale, 50, samples);

    CLogNormalMeanPrecConjugate filter(makePrior());

    TDouble1Vec sampled;

    for (std::size_t i = 0u; i < 1u; ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));
        sampled.clear();
        filter.sampleMarginalLikelihood(10, sampled);
        CPPUNIT_ASSERT_EQUAL(i + 1, sampled.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(samples[i], sampled[i], eps);
    }

    TMeanAccumulator meanMeanError;

    std::size_t numberSampled = 20u;
    for (std::size_t i = 1u; i < samples.size(); ++i)
    {
        filter.addSamples(TDouble1Vec(1, samples[i]));
        sampled.clear();
        filter.sampleMarginalLikelihood(numberSampled, sampled);

        // The error is due to the approximation of the likelihood
        // function by a moment matched log-normal. This becomes
        // increasingly accurate as the number of updates increases.
        if (i >= 10u)
        {
            TMeanVarAccumulator sampledMoments;
            sampledMoments = std::for_each(sampled.begin(), sampled.end(), sampledMoments);
            CPPUNIT_ASSERT_EQUAL(numberSampled, sampled.size());

            LOG_DEBUG("expectedMean = " << filter.marginalLikelihoodMean()
                      << ", sampledMean = " << maths::CBasicStatistics::mean(sampledMoments));
            LOG_DEBUG("expectedVar = " << filter.marginalLikelihoodVariance()
                      << ", sampledVar = " << maths::CBasicStatistics::variance(sampledMoments));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodMean(),
                                         maths::CBasicStatistics::mean(sampledMoments),
                                         0.8);
            meanMeanError.add(::fabs(  filter.marginalLikelihoodMean()
                                     - maths::CBasicStatistics::mean(sampledMoments)));
        }

        std::sort(sampled.begin(), sampled.end());
        for (std::size_t j = 1u; j < sampled.size(); ++j)
        {
            double q = 100.0 * static_cast<double>(j) / static_cast<double>(numberSampled);

            double expectedQuantile;
            CPPUNIT_ASSERT(filter.marginalLikelihoodQuantileForTest(q, eps, expectedQuantile));

            LOG_DEBUG("quantile = " << q
                      << ", x_quantile = " << expectedQuantile
                      << ", quantile range = [" << sampled[j - 1] << "," << sampled[j] << "]");
            CPPUNIT_ASSERT(expectedQuantile >= sampled[j - 1] - 0.2 * std::max(6.0 - static_cast<double>(i), 0.0));
            CPPUNIT_ASSERT(expectedQuantile <= sampled[j] + 1.2 * std::max(6.0 - static_cast<double>(i), 0.0));
        }
    }

    LOG_DEBUG("mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.25);
}

void CLogNormalMeanPrecConjugateTest::testCdf(void)
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testCdf  |");
    LOG_DEBUG("+--------------------------------------------+");

    // Test error cases.
    //
    // Test some invariants:
    //   "cdf" + "cdf complement" = 1
    //    cdf x for x < 0 = 1
    //    cdf complement x for x < 0 = 0

    const double location = 2.0;
    const double squareScale = 0.8;
    const std::size_t n[] = { 20u, 80u };

    test::CRandomNumbers rng;

    CLogNormalMeanPrecConjugate filter(makePrior());

    for (std::size_t i = 0u; i < boost::size(n); ++i)
    {
        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, n[i], samples);

        filter.addSamples(samples);

        double lowerBound;
        double upperBound;
        CPPUNIT_ASSERT(!filter.minusLogJointCdf(TDouble1Vec(), lowerBound, upperBound));
        CPPUNIT_ASSERT(!filter.minusLogJointCdfComplement(TDouble1Vec(), lowerBound, upperBound));

        CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, -1.0), lowerBound, upperBound));
        double f = (lowerBound + upperBound) / 2.0;
        CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, -1.0), lowerBound, upperBound));
        double fComplement = (lowerBound + upperBound) / 2.0;
        LOG_DEBUG("log(F(x)) = " << -f
                  << ", log(1 - F(x)) = " << fComplement);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(::log(std::numeric_limits<double>::min()), -f, 1e-10);
        CPPUNIT_ASSERT_EQUAL(1.0, ::exp(-fComplement));

        for (std::size_t j = 1u; j < 500; ++j)
        {
            double x = static_cast<double>(j) / 2.0;

            CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x), lowerBound, upperBound));
            f = (lowerBound + upperBound) / 2.0;
            CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lowerBound, upperBound));
            fComplement = (lowerBound + upperBound) / 2.0;
            LOG_DEBUG("log(F(x)) = " << (f == 0.0 ? f : -f)
                      << ", log(1 - F(x)) = " << (fComplement == 0.0 ? fComplement : -fComplement));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ::exp(-f) + ::exp(-fComplement), 1e-10);
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples(void)
{
    LOG_DEBUG("+-----------------------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+-----------------------------------------------------------------------+");

    // We test that the probability of less likely samples calculation
    // agrees with the chance of seeing a sample with lower marginal
    // likelihood, up to the sampling error.
    //
    // We also check that the tail calculation attributes samples to
    // the appropriate tail of the distribution.

    const double means[] = { 0.1, 1.5, 3.0 };
    const double squareScales[] = { 0.2, 0.4, 1.5 };
    const double vs[] = { 0.5, 1.0, 2.0 };

    test::CRandomNumbers rng;

    TMeanAccumulator meanError;

    for (size_t i = 0; i < boost::size(means); ++i)
    {
        for (size_t j = 0; j < boost::size(squareScales); ++j)
        {
            LOG_DEBUG("means = " << means[i]
                      << ", scale = " << ::sqrt(squareScales[j]));

            TDoubleVec samples;
            rng.generateLogNormalSamples(means[i], squareScales[j], 1000, samples);

            CLogNormalMeanPrecConjugate filter(makePrior());
            filter.addSamples(samples);

            double location = filter.normalMean();
            double scale    = ::sqrt(1.0 / filter.normalPrecision());

            TDoubleVec likelihoods;
            for (std::size_t k = 0u; k < samples.size(); ++k)
            {
                double likelihood;
                filter.jointLogMarginalLikelihood(TDouble1Vec(1, samples[k]), likelihood);
                likelihoods.push_back(likelihood);
            }
            std::sort(likelihoods.begin(), likelihoods.end());

            boost::math::lognormal_distribution<> lognormal(location, scale);
            for (std::size_t k = 1u; k < 10; ++k)
            {
                double x = boost::math::quantile(lognormal, static_cast<double>(k) / 10.0);

                TDouble1Vec sample(1, x);
                double fx;
                filter.jointLogMarginalLikelihood(sample, fx);

                double px =   static_cast<double>(std::lower_bound(likelihoods.begin(),
                                                                   likelihoods.end(), fx)
                                                  - likelihoods.begin())
                            / static_cast<double>(likelihoods.size());

                double lb, ub;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lb, ub);

                double ssd = ::sqrt(px * (1.0 - px) / static_cast<double>(samples.size()));

                LOG_DEBUG("expected P(x) = " << px
                          << ", actual P(x) = " << (lb + ub) / 2.0
                          << " sample sd = " << ssd);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(px, (lb + ub) / 2.0, 3.0 * ssd);

                meanError.add(::fabs(px - (lb + ub) / 2.0));
            }

            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);

            for (std::size_t k = 0u; k < boost::size(vs); ++k)
            {
                double mode = filter.marginalLikelihoodMode(weightStyle,
                                                            TDouble4Vec(1, vs[k]));
                double ss[] = { 0.9 * mode, 1.1 * mode };

                LOG_DEBUG("vs = " << vs[k] << ", mode = " << mode);

                double lb, ub;
                maths_t::ETail tail;

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

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.01);
}

void CLogNormalMeanPrecConjugateTest::testAnomalyScore(void)
{
    LOG_DEBUG("+-----------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testAnomalyScore  |");
    LOG_DEBUG("+-----------------------------------------------------+");

    // This test pushes 500 samples through the filter and adds in
    // anomalous signals in the bins at 30, 120, 300 and 420 with
    // magnitude 4, 5, 10 and 15 standard deviations, respectively,
    // and checks the anomaly score has:
    //   1) high probability of detecting the anomalies, and
    //   2) a very low rate of false positives.

    typedef std::vector<unsigned int> TUIntVec;

    const double decayRates[] = { 0.0, 0.001, 0.01 };

    const double means[] = { 0.1, 1.5, 3.0 };
    const double squareScales[] = { 0.2, 0.4, 1.5 };

    const double threshold = 0.02;

    const unsigned int anomalyTimes[] = { 30u, 120u, 300u, 420u };
    const double anomalies[] = { 4.0, 5.0, 10.0, 15.0, 0.0 };

    test::CRandomNumbers rng;

    unsigned int test = 0;

    std::ofstream file;
    file.open("results.m");

    double totalFalsePositiveRate = 0.0;
    std::size_t totalPositives[] = { 0u, 0u, 0u };

    for (size_t i = 0; i < boost::size(means); ++i)
    {
        for (size_t j = 0; j < boost::size(squareScales); ++j)
        {
            LOG_DEBUG("mean = " << means[i] << ", scale = " << ::sqrt(squareScales[j]));

            boost::math::lognormal_distribution<> logNormal(means[i], ::sqrt(squareScales[j]));

            TDoubleVec samples;
            rng.generateLogNormalSamples(means[i], squareScales[j], 500, samples);

            for (size_t k = 0; k < boost::size(decayRates); ++k)
            {
                CLogNormalMeanPrecConjugate filter(
                        makePrior(maths_t::E_ContinuousData, 0.0, decayRates[k]));

                ++test;

                std::ostringstream x;
                std::ostringstream scores;
                x << "x" << test << " = [";
                scores << "score" << test << " = [";

                TUIntVec candidateAnomalies;
                for (unsigned int time = 0; time < samples.size(); ++time)
                {
                    double anomaly = anomalies[std::find(boost::begin(anomalyTimes),
                                                         boost::end(anomalyTimes), time)
                                               - boost::begin(anomalyTimes)]
                                     * boost::math::standard_deviation(logNormal);
                    double sample = samples[time] + anomaly;

                    TDouble1Vec sampleVec(1, sample);
                    filter.addSamples(sampleVec);

                    double score;
                    filter.anomalyScore(maths_t::E_TwoSided, sampleVec, score);
                    if (score > threshold)
                    {
                        candidateAnomalies.push_back(time);
                    }

                    filter.propagateForwardsByTime(1.0);

                    x << time << " ";
                    scores << score << " ";
                }

                x << "];\n";
                scores << "];\n";
                file << x.str() << scores.str()
                     << "plot(x" << test << ", score" << test << ");\n"
                     << "input(\"Hit any key for next test\");\n\n";

                TUIntVec falsePositives;
                std::set_difference(candidateAnomalies.begin(),
                                    candidateAnomalies.end(),
                                    boost::begin(anomalyTimes),
                                    boost::end(anomalyTimes),
                                    std::back_inserter(falsePositives));

                double falsePositiveRate =  static_cast<double>(falsePositives.size())
                                          / static_cast<double>(samples.size());

                totalFalsePositiveRate += falsePositiveRate;

                TUIntVec positives;
                std::set_intersection(candidateAnomalies.begin(),
                                      candidateAnomalies.end(),
                                      boost::begin(anomalyTimes),
                                      boost::end(anomalyTimes),
                                      std::back_inserter(positives));

                LOG_DEBUG("falsePositiveRate = " << falsePositiveRate
                          << ", positives = " << positives.size());

                // False alarm rate should be less than 1%.
                CPPUNIT_ASSERT(falsePositiveRate <= 0.01);

                // Should detect at least the two big anomalies.
                CPPUNIT_ASSERT(positives.size() >= 2u);

                totalPositives[k] += positives.size();
            }
        }
    }

    totalFalsePositiveRate /= static_cast<double>(test);

    LOG_DEBUG("totalFalsePositiveRate = " << totalFalsePositiveRate);

    for (size_t i = 0; i < boost::size(totalPositives); ++i)
    {
        LOG_DEBUG("positives = " << totalPositives[i]);
        CPPUNIT_ASSERT(totalPositives[i] >= 20u);
    }

    // Total false alarm rate should be less than 0.4%.
    CPPUNIT_ASSERT(totalFalsePositiveRate < 0.004);
}

void CLogNormalMeanPrecConjugateTest::testOffset(void)
{
    LOG_DEBUG("+-----------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testOffset  |");
    LOG_DEBUG("+-----------------------------------------------+");

    // The idea of this test is to check that the offset correctly cancels
    // out a translation applied to a log-normally distributed data set.

    const maths_t::EDataType dataTypes[] =
        {
            maths_t::E_IntegerData,
            maths_t::E_ContinuousData
        };
    const double offsets[] = { -0.5, 0.5 };
    const double decayRates[] = { 0.0, 0.001, 0.01 };

    const double location = 3.0;
    const double squareScale = 1.0;

    const double eps = 1e-8;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(location, squareScale, 100, samples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        for (size_t j = 0; j < boost::size(offsets); ++j)
        {
            for (size_t k = 0; k < boost::size(decayRates); ++k)
            {
                CLogNormalMeanPrecConjugate filter1(makePrior(dataTypes[i], offsets[j], decayRates[k]));
                CLogNormalMeanPrecConjugate filter2(makePrior(dataTypes[i], 0.0, decayRates[k]));

                for (std::size_t l = 0u; l < samples.size(); ++l)
                {
                    double offsetSample = samples[l] - offsets[j];
                    TDouble1Vec offsetSampleVec(1, offsetSample);
                    filter1.addSamples(offsetSampleVec);
                    filter1.propagateForwardsByTime(1.0);

                    double x = samples[l];
                    TDouble1Vec sample(1, x);
                    filter2.addSamples(sample);
                    filter2.propagateForwardsByTime(1.0);

                    double likelihood1;
                    filter1.jointLogMarginalLikelihood(offsetSampleVec, likelihood1);
                    double lowerBound1, upperBound1;
                    filter1.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                           offsetSampleVec,
                                                           lowerBound1,
                                                           upperBound1);
                    CPPUNIT_ASSERT_EQUAL(lowerBound1, upperBound1);
                    double probability1 = (lowerBound1 + upperBound1) / 2.0;

                    double likelihood2;
                    filter2.jointLogMarginalLikelihood(sample, likelihood2);
                    double lowerBound2, upperBound2;
                    filter2.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                           sample,
                                                           lowerBound2,
                                                           upperBound2);
                    CPPUNIT_ASSERT_EQUAL(lowerBound2, upperBound2);
                    double probability2 = (lowerBound2 + upperBound2) / 2.0;

                    CPPUNIT_ASSERT_DOUBLES_EQUAL(likelihood1, likelihood2, eps);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(probability1, probability2, eps);
                }

                typedef maths::CEqualWithTolerance<double> TEqual;
                TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, eps);
                CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
            }
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testIntegerData(void)
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testIntegerData  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // The idea of this test is to check that the inferred model agrees in the
    // limit (large n) with the model inferred from such data.

    const double locations[] = { 0.2, 1.0, 1.5 };
    const double squareScales[] = { 0.5, 2.0 };
    const std::size_t nSamples = 100000u;

    for (std::size_t i = 0; i < boost::size(locations); ++i)
    {
        for (std::size_t j = 0; j < boost::size(squareScales); ++j)
        {
        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateLogNormalSamples(locations[i], squareScales[j], nSamples, samples);

        TDoubleVec uniform;
        rng.generateUniformSamples(0.0, 1.0, nSamples, uniform);

        CLogNormalMeanPrecConjugate filter1(makePrior(maths_t::E_IntegerData, 0.1));
        CLogNormalMeanPrecConjugate filter2(makePrior(maths_t::E_ContinuousData, 0.1));

        for (std::size_t k = 0; k < nSamples; ++k)
        {
            double x = ::floor(samples[k]);

            TDouble1Vec sample(1, x);
            filter1.addSamples(sample);

            sample[0] += uniform[k];
            filter2.addSamples(sample);
        }

        typedef maths::CEqualWithTolerance<double> TEqual;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.01);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));

        TMeanAccumulator meanLogLikelihood1;
        TMeanAccumulator meanLogLikelihood2;
        for (std::size_t k = 0u; k < nSamples; ++k)
        {
            double x = ::floor(samples[k]);

            TDouble1Vec sample(1, x);
            double logLikelihood1;
            filter1.jointLogMarginalLikelihood(sample, logLikelihood1);
            meanLogLikelihood1.add(-logLikelihood1);

            sample[0] += uniform[k];
            double logLikelihood2;
            filter2.jointLogMarginalLikelihood(sample, logLikelihood2);
            meanLogLikelihood2.add(-logLikelihood2);
        }

        LOG_DEBUG("meanLogLikelihood1 = " << maths::CBasicStatistics::mean(meanLogLikelihood1)
                  << ", meanLogLikelihood2 = " << maths::CBasicStatistics::mean(meanLogLikelihood2));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(meanLogLikelihood1),
                                     maths::CBasicStatistics::mean(meanLogLikelihood2),
                                     0.05);
        }
    }

    TMeanAccumulator meanError;

    for (size_t i = 0; i < boost::size(locations); ++i)
    {
        for (std::size_t j = 0; j < boost::size(squareScales); ++j)
        {
            test::CRandomNumbers rng;

            TDoubleVec seedSamples;
            rng.generateLogNormalSamples(locations[i], squareScales[j], 200, seedSamples);
            CLogNormalMeanPrecConjugate filter1(makePrior(maths_t::E_IntegerData, 0.1));
            filter1.addSamples(seedSamples);

            CLogNormalMeanPrecConjugate filter2 = filter1;
            filter2.dataType(maths_t::E_ContinuousData);

            TDoubleVec samples;
            rng.generateLogNormalSamples(locations[i], squareScales[j], nSamples, samples);

            TDoubleVec uniform;
            rng.generateUniformSamples(0.0, 1.0, nSamples, uniform);

            TMeanAccumulator meanProbability1;
            TMeanAccumulator meanProbability2;

            for (std::size_t k = 0; k < nSamples; ++k)
            {
                double x = ::floor(samples[k]);

                TDouble1Vec sample(1, x);

                double l1, u1;
                CPPUNIT_ASSERT(filter1.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, l1, u1));
                CPPUNIT_ASSERT_EQUAL(l1, u1);
                double p1 = (l1 + u1) / 2.0;
                meanProbability1.add(p1);

                sample[0] += uniform[k];
                double l2, u2;
                CPPUNIT_ASSERT(filter2.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, l2, u2));
                CPPUNIT_ASSERT_EQUAL(l2, u2);
                double p2 = (l2 + u2) / 2.0;
                meanProbability2.add(p2);
            }

            double p1 = maths::CBasicStatistics::mean(meanProbability1);
            double p2 = maths::CBasicStatistics::mean(meanProbability2);
            LOG_DEBUG("location = " << locations[i]
                      << ", p1 = " << p1
                      << ", p2 = " << p2);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 0.05 * p1);

            meanError.add(fabs(p1 - p2));
        }
    }

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.005);
}

void CLogNormalMeanPrecConjugateTest::testLowVariationData(void)
{
    LOG_DEBUG("+---------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testLowVariationData  |");
    LOG_DEBUG("+---------------------------------------------------------+");

    {
        CLogNormalMeanPrecConjugate filter(makePrior(maths_t::E_IntegerData));
        for (std::size_t i = 0u; i < 100; ++i)
        {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG("68% confidence interval "
                  << core::CContainerPrinter::print(interval)
                  << ", approximate variance = " << sigma * sigma);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, 1.0 / (sigma * sigma), 0.15);
    }
    {
        CLogNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData));
        for (std::size_t i = 0u; i < 100; ++i)
        {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG("68% confidence interval "
                  << core::CContainerPrinter::print(interval)
                  << ", approximate s.t.d. = " << sigma);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1e-4, sigma / 430.5, 5e-5);
    }
}

void CLogNormalMeanPrecConjugateTest::testPersist(void)
{
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testPersist  |");
    LOG_DEBUG("+------------------------------------------------+");

    const double location = ::log(10.0);
    const double squareScale = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(location, squareScale, 100, samples);

    maths::CLogNormalMeanPrecConjugate origFilter(makePrior());
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

    LOG_DEBUG("Log normal mean conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                             decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::CLogNormalMeanPrecConjugate restoredFilter(params, traverser);

    LOG_DEBUG("orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CLogNormalMeanPrecConjugateTest::testVarianceScale(void)
{
    LOG_DEBUG("+------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testVarianceScale  |");
    LOG_DEBUG("+------------------------------------------------------+");

    // The strategy for this test is to check we correctly account
    // for variance scaling by scaling the variance of a collection
    // of samples and then checking that the percentiles for those
    // samples "probability of less likely sample" are correct. In
    // particular, we expect that the calculation correctly predicts
    // the fraction of samples which have lower probability for all
    // percentiles (if the variance scaling was off we'd either get
    // too few or too many).
    //
    // We use the same idea for testing the variance scaling for
    // the marginal likelihood. The sample mean of the scaled log
    // likelihood tends to the expected log likelihood for a scaled
    // log-normal distribution (uniform law of large numbers), which
    // is just the differential entropy of a scaled log-normal R.V.
    //
    // Finally, we test update with scaled samples produces the
    // correct posterior.

    maths_t::ESampleWeightStyle scales[] =
        {
            maths_t::E_SampleSeasonalVarianceScaleWeight,
            maths_t::E_SampleCountVarianceScaleWeight
        };

    for (std::size_t s = 0u; s < boost::size(scales); ++s)
    {
        const double location = 2.0;
        const double squareScale = 1.5;
        {
            boost::math::lognormal_distribution<> logNormal(location, ::sqrt(squareScale));
            LOG_DEBUG("mean = " << boost::math::mean(logNormal)
                      << ", variance = " << boost::math::variance(logNormal));
        }

        const double varianceScales[] =
            {
                0.20, 0.50, 0.75, 1.50, 2.00, 5.00
            };

        LOG_DEBUG("");
        LOG_DEBUG("****** probabilityOfLessLikelySamples ******");

        const double percentiles[] =
            {
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0
            };

        const std::size_t nSamples[] = { 10u, 20u, 40u, 80u, 1000u };
        const std::size_t nScaledSamples = 50000u;

        double percentileErrorTolerance = 0.08;
        double meanPercentileErrorTolerance = 0.055;
        double totalMeanPercentileErrorTolerance = 0.005;

        double totalUnscaledMeanPercentileError = 0.0;
        double totalMeanPercentileError = 0.0;
        double trials = 0.0;
        for (size_t i = 0; i < boost::size(nSamples); ++i)
        {
            LOG_DEBUG("**** nSamples = " << nSamples[i] << " ****");

            test::CRandomNumbers rng;

            TDoubleVec samples;
            rng.generateLogNormalSamples(location, squareScale, nSamples[i], samples);

            CLogNormalMeanPrecConjugate filter(makePrior());
            filter.addSamples(samples);
            filter.checksum();

            double unscaledMeanPercentileError = 0.0;
            TDoubleVec unscaledPercentileErrors;
            {
                TDoubleVec unscaledSamples;
                rng.generateLogNormalSamples(location, squareScale, nScaledSamples, unscaledSamples);

                TDoubleVec probabilities;
                probabilities.reserve(nScaledSamples);
                for (std::size_t j = 0; j < unscaledSamples.size(); ++j)
                {
                    TDouble1Vec sample(1, unscaledSamples[j]);

                    double lowerBound, upperBound;
                    CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                                         sample,
                                                                         lowerBound,
                                                                         upperBound));
                    CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                    double probability = (lowerBound + upperBound) / 2.0;
                    probabilities.push_back(probability);
                }
                std::sort(probabilities.begin(), probabilities.end());

                for (size_t j = 0; j < boost::size(percentiles); ++j)
                {
                    std::size_t index = static_cast<std::size_t>(
                            static_cast<double>(nScaledSamples) * percentiles[j]/100.0);
                    double error = ::fabs(probabilities[index] - percentiles[j]/100.0);
                    unscaledPercentileErrors.push_back(error);
                    unscaledMeanPercentileError += error;
                }
                unscaledMeanPercentileError /= static_cast<double>(boost::size(percentiles));
            }

            for (size_t j = 0; j < boost::size(varianceScales); ++j)
            {
                LOG_DEBUG("**** variance scale = " << varianceScales[j] << " ****");

                double ss = ::log(1.0 + varianceScales[j] * (::exp(squareScale) - 1.0));
                double shiftedLocation = location + (squareScale - ss) / 2.0;
                {
                    boost::math::lognormal_distribution<> logNormal(shiftedLocation, ::sqrt(ss));
                    LOG_DEBUG("mean = " << boost::math::mean(logNormal)
                              << ", variance = " << boost::math::variance(logNormal));
                }

                TDoubleVec scaledSamples;
                rng.generateLogNormalSamples(shiftedLocation, ss, nScaledSamples, scaledSamples);

                TDoubleVec probabilities;
                probabilities.reserve(nScaledSamples);
                for (std::size_t k = 0; k < scaledSamples.size(); ++k)
                {
                    double lowerBound, upperBound;
                    maths_t::ETail tail;
                    CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(
                                                  maths_t::E_TwoSided,
                                                  maths_t::TWeightStyleVec(1, scales[s]),
                                                  TDouble1Vec(1, scaledSamples[k]),
                                                  TDouble4Vec1Vec(1, TDouble4Vec(1, varianceScales[j])),
                                                  lowerBound,
                                                  upperBound,
                                                  tail));
                    CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                    double probability = (lowerBound + upperBound) / 2.0;
                    probabilities.push_back(probability);
                }
                std::sort(probabilities.begin(), probabilities.end());

                double meanPercentileError = 0.0;
                for (size_t k = 0; k < boost::size(percentiles); ++k)
                {
                    std::size_t index = static_cast<std::size_t>(
                            static_cast<double>(nScaledSamples) * percentiles[k]/100.0);
                    double error = ::fabs(probabilities[index] - percentiles[k]/100.0);
                    meanPercentileError += error;
                    double threshold = percentileErrorTolerance + unscaledPercentileErrors[k];

                    LOG_DEBUG("percentile = " << percentiles[k]
                              << ", probability = " << probabilities[index]
                              << ", error = " << error
                              << ", error threshold = " << threshold);

                    CPPUNIT_ASSERT(error < threshold);
                }
                meanPercentileError /= static_cast<double>(boost::size(percentiles));

                double threshold = meanPercentileErrorTolerance + unscaledMeanPercentileError;

                LOG_DEBUG("mean error = " << meanPercentileError
                          << ", mean error threshold = " << threshold);

                CPPUNIT_ASSERT(meanPercentileError < threshold);

                totalMeanPercentileError += meanPercentileError;
                totalUnscaledMeanPercentileError += unscaledMeanPercentileError;
                trials += 1.0;
            }
        }
        totalMeanPercentileError /= trials;
        totalUnscaledMeanPercentileError /= trials;

        {
            double threshold = totalMeanPercentileErrorTolerance + totalUnscaledMeanPercentileError;
            LOG_DEBUG("total unscaled mean error = " << totalUnscaledMeanPercentileError);
            LOG_DEBUG("total mean error = " << totalMeanPercentileError
                    << ", total mean error threshold = " << threshold);
            CPPUNIT_ASSERT(totalMeanPercentileError < threshold);
        }

        LOG_DEBUG("");
        LOG_DEBUG("****** jointLogMarginalLikelihood ******");

        test::CRandomNumbers rng;

        for (size_t i = 0; i < boost::size(varianceScales); ++i)
        {
            LOG_DEBUG("**** variance scale = " << varianceScales[i] << " ****");

            double ss = ::log(1.0 + varianceScales[i] * (::exp(squareScale) - 1.0));
            double shiftedLocation = location + (squareScale - ss) / 2.0;
            boost::math::lognormal_distribution<> logNormal(shiftedLocation, ::sqrt(ss));
            {
                LOG_DEBUG("mean = " << boost::math::mean(logNormal)
                          << ", variance = " << boost::math::variance(logNormal));
            }
            double expectedDifferentialEntropy = maths::CTools::differentialEntropy(logNormal);

            CLogNormalMeanPrecConjugate filter(makePrior());

            double differentialEntropy = 0.0;

            TDoubleVec seedSamples;
            rng.generateLogNormalSamples(location, squareScale, 100, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec scaledSamples;
            rng.generateLogNormalSamples(shiftedLocation, ss, 100000, scaledSamples);

            for (std::size_t j = 0u; j < scaledSamples.size(); ++j)
            {
                double logLikelihood = 0.0;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                     filter.jointLogMarginalLikelihood(
                                                    maths_t::TWeightStyleVec(1, scales[s]),
                                                    TDouble1Vec(1, scaledSamples[j]),
                                                    TDouble4Vec1Vec(1, TDouble4Vec(1, varianceScales[i])),
                                                    logLikelihood));
                differentialEntropy -= logLikelihood;
            }

            differentialEntropy /= static_cast<double>(scaledSamples.size());

            LOG_DEBUG("differentialEntropy = " << differentialEntropy
                      << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 0.5);
        }
    }

    const maths_t::EDataType dataTypes[] =
        {
            maths_t::E_IntegerData,
            maths_t::E_ContinuousData
        };

    const double maximumMeanError[] = { 0.5, 0.5 };
    const double maximumVarianceError[] = { 1.4, 1.0 };
    const double maximumMeanMeanError[] = { 0.02, 0.01 };
    const double maximumMeanVarianceError[] = { 0.18, 0.1 };

    for (std::size_t s = 0u; s < boost::size(scales); ++s)
    {
        for (std::size_t t = 0u; t < boost::size(dataTypes); ++t)
        {
            const double means[] = { 0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0, 1000000.0 };
            const double variances[] = { 0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0, 1000000.0 };
            const double varianceScales[] = { 0.1, 0.5, 1.0, 2.0, 10.0, 100.0 };

            maths_t::TWeightStyleVec weightStyle(1, scales[s]);
            TDoubleVec samples;
            TDouble4Vec1Vec weights;

            test::CRandomNumbers rng;

            TMeanAccumulator meanMeanError;
            TMeanAccumulator meanVarianceError;

            for (std::size_t i = 0u; i < boost::size(means); ++i)
            {
                for (std::size_t j = 0u; j < boost::size(variances); ++j)
                {
                    double mean = means[i];
                    double variance = variances[j];

                    // We don't include very skewed distributions because they
                    // are hard estimate accurately even without scaling due to
                    // relatively frequent large outliers.
                    if (mean <= 0.1 * variance)
                    {
                        continue;
                    }

                    // We purposely don't estimate true variance in this case.
                    if (::sqrt(variance) < mean * maths::MINIMUM_COEFFICIENT_OF_VARIATION)
                    {
                        continue;
                    }

                    double squareScale = ::log(1.0 + variance / (mean * mean));
                    double location = ::log(mean) - squareScale / 2.0;
                    double precision = 1.0 / squareScale;
                    {
                        boost::math::lognormal_distribution<> logNormal(location, ::sqrt(squareScale));
                        LOG_DEBUG("");
                        LOG_DEBUG("****** mean = " << boost::math::mean(logNormal)
                                  << ", variance = " << boost::math::variance(logNormal) << " ******");
                        LOG_DEBUG("location = " << location
                                  << ", precision = " << precision);
                    }

                    for (std::size_t k = 0u; k < boost::size(varianceScales); ++k)
                    {
                        double scale = varianceScales[k];
                        if (scale * variance >= 100.0 * mean)
                        {
                            continue;
                        }
                        LOG_DEBUG("*** scale = " << scale << " ***");

                        double scaledSquareScale = ::log(1.0 + variance * scale / (mean * mean));
                        double scaledLocation = ::log(mean) - scaledSquareScale / 2.0;
                        double scaledPrecision = 1.0 / scaledSquareScale;
                        {
                            boost::math::lognormal_distribution<> logNormal(scaledLocation, ::sqrt(scaledSquareScale));
                            LOG_DEBUG("scaled mean = " << boost::math::mean(logNormal)
                                      << ", scaled variance = " << boost::math::variance(logNormal));
                            LOG_DEBUG("scaled location = " << scaledLocation
                                      << ", scaled precision = " << scaledPrecision);
                        }

                        TMeanAccumulator meanError;
                        TMeanAccumulator varianceError;
                        for (unsigned int test = 0u; test < 5; ++test)
                        {
                            CLogNormalMeanPrecConjugate filter(makePrior(dataTypes[t]));

                            rng.generateLogNormalSamples(location, squareScale, 200, samples);
                            weights.clear();
                            weights.resize(samples.size(), TDouble4Vec(1, 1.0));
                            filter.addSamples(weightStyle, samples, weights);
                            rng.generateLogNormalSamples(scaledLocation, scaledSquareScale, 200, samples);
                            weights.clear();
                            weights.resize(samples.size(), TDouble4Vec(1, scale));
                            filter.addSamples(weightStyle, samples, weights);

                            boost::math::lognormal_distribution<> logNormal(filter.normalMean(),
                                                                            ::sqrt(1.0 / filter.normalPrecision()));
                            double dm = (dataTypes[t] == maths_t::E_IntegerData ? 0.5 : 0.0);
                            double dv = (dataTypes[t] == maths_t::E_IntegerData ? 1.0 / 12.0 : 0.0);
                            double trialMeanError = ::fabs(boost::math::mean(logNormal) - (mean + dm))
                                                    / std::max(1.0, mean);
                            double trialVarianceError = ::fabs(boost::math::variance(logNormal) - (variance + dv))
                                                        / std::max(1.0, variance);

                            LOG_DEBUG("trial mean error = " << trialMeanError);
                            LOG_DEBUG("trial variance error = " << trialVarianceError);

                            meanError.add(trialMeanError);
                            varianceError.add(trialVarianceError);
                        }

                        LOG_DEBUG("mean error = "
                                  << maths::CBasicStatistics::mean(meanError));
                        LOG_DEBUG("variance error = "
                                  << maths::CBasicStatistics::mean(varianceError));

                        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < maximumMeanError[t]);
                        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceError) < maximumVarianceError[t]);

                        meanMeanError += meanError;
                        meanVarianceError += varianceError;
                    }
                }
            }

            LOG_DEBUG("mean mean error = "
                      << maths::CBasicStatistics::mean(meanMeanError));
            LOG_DEBUG("mean variance error = "
                      << maths::CBasicStatistics::mean(meanVarianceError));

            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < maximumMeanMeanError[t]);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanVarianceError) < maximumMeanVarianceError[t]);
        }
    }
}

void CLogNormalMeanPrecConjugateTest::testNegativeSample(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CLogNormalMeanPrecConjugateTest::testNegativeSample  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    // Test that we recover roughly the same distribution after adjusting
    // the offset. The idea of this test is to run two priors side by side,
    // one with a large enough offset that it never needs to adjust the
    // offset and the other which will adjust and check that we get broadly
    // similar distributions at the end.

    const double location = ::log(2.0);
    const double squareScale = 1.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateLogNormalSamples(location, squareScale, 100, samples);

    CLogNormalMeanPrecConjugate filter1 =
            CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.0, 0.0, 0.2);
    CLogNormalMeanPrecConjugate filter2 =
            CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 1.74524, 0.0, 0.2);

    filter1.addSamples(samples);
    filter2.addSamples(samples);

    TDouble1Vec negative(1, -0.29);
    filter1.addSamples(negative);
    filter2.addSamples(negative);

    CPPUNIT_ASSERT_EQUAL(filter1.numberSamples(), filter2.numberSamples());

    typedef maths::CEqualWithTolerance<double> TEqual;
    TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.1);
    CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
}

CppUnit::Test* CLogNormalMeanPrecConjugateTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CLogNormalMeanPrecConjugateTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testMultipleUpdate",
                                   &CLogNormalMeanPrecConjugateTest::testMultipleUpdate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testPropagation",
                                   &CLogNormalMeanPrecConjugateTest::testPropagation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testMeanEstimation",
                                   &CLogNormalMeanPrecConjugateTest::testMeanEstimation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testPrecisionEstimation",
                                   &CLogNormalMeanPrecConjugateTest::testPrecisionEstimation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testMarginalLikelihood",
                                   &CLogNormalMeanPrecConjugateTest::testMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMean",
                                   &CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMode",
                                   &CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodMode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance",
                                   &CLogNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testSampleMarginalLikelihood",
                                   &CLogNormalMeanPrecConjugateTest::testSampleMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testCdf",
                                   &CLogNormalMeanPrecConjugateTest::testCdf) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples",
                                   &CLogNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testAnomalyScore",
                                   &CLogNormalMeanPrecConjugateTest::testAnomalyScore) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testOffset",
                                   &CLogNormalMeanPrecConjugateTest::testOffset) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testIntegerData",
                                   &CLogNormalMeanPrecConjugateTest::testIntegerData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testLowVariationData",
                                   &CLogNormalMeanPrecConjugateTest::testLowVariationData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testPersist",
                                   &CLogNormalMeanPrecConjugateTest::testPersist) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testVarianceScale",
                                   &CLogNormalMeanPrecConjugateTest::testVarianceScale) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogNormalMeanPrecConjugateTest>(
                                   "CLogNormalMeanPrecConjugateTest::testNegativeSample",
                                   &CLogNormalMeanPrecConjugateTest::testNegativeSample) );

    return suiteOfTests;
}

