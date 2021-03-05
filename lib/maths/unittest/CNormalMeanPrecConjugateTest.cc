/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPriorDetail.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>

BOOST_AUTO_TEST_SUITE(CNormalMeanPrecConjugateTest)

using namespace ml;
using namespace handy_typedefs;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;
using TWeightFunc = maths_t::TDoubleWeightsAry (*)(double);

CNormalMeanPrecConjugate makePrior(maths_t::EDataType dataType = maths_t::E_ContinuousData,
                                   const double& decayRate = 0.0) {
    return CNormalMeanPrecConjugate::nonInformativePrior(dataType, decayRate);
}
}

BOOST_AUTO_TEST_CASE(testMultipleUpdate) {
    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    using TEqual = maths::CEqualWithTolerance<double>;

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double mean = 10.0;
    const double variance = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    for (std::size_t i = 0; i < boost::size(dataTypes); ++i) {
        CNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CNormalMeanPrecConjugate filter2(filter1);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter1.addSamples(TDouble1Vec(1, samples[j]));
        }
        filter2.addSamples(samples);

        LOG_DEBUG(<< filter1.print() << "\nvs" << filter2.print());
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-4);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }

    // Test with variance scale.

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CNormalMeanPrecConjugate filter2(filter1);

        maths_t::TDoubleWeightsAry1Vec weights;
        weights.resize(samples.size(), maths_t::countVarianceScaleWeight(2.0));
        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter1.addSamples({samples[j]}, {weights[j]});
        }
        filter2.addSamples(samples, weights);

        LOG_DEBUG(<< filter1.print() << "\nvs" << filter2.print());
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }

    // Test the count weight is equivalent to adding repeated samples.

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CNormalMeanPrecConjugate filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;
        for (std::size_t j = 0; j < count; ++j) {
            filter1.addSamples(TDouble1Vec(1, x));
        }
        filter2.addSamples({x}, {maths_t::countWeight(static_cast<double>(count))});

        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }
}

BOOST_AUTO_TEST_CASE(testPropagation) {
    // Test that propagation doesn't affect the expected values
    // of likelihood mean and precision.

    const double eps = 1e-7;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(1.0, 3.0, 500, samples);

    CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, 0.1));

    for (std::size_t i = 0; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
    }

    double mean = filter.mean();
    double precision = filter.precision();

    filter.propagateForwardsByTime(5.0);

    double propagatedMean = filter.mean();
    double propagatedPrecision = filter.precision();

    LOG_DEBUG(<< "mean = " << mean << ", precision = " << precision
              << ", propagatedMean = " << propagatedMean
              << ", propagatedPrecision = " << propagatedPrecision);

    BOOST_REQUIRE_CLOSE_ABSOLUTE(mean, propagatedMean, eps);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(precision, propagatedPrecision, eps);
}

BOOST_AUTO_TEST_CASE(testMeanEstimation) {
    // We are going to test that we correctly estimate a distribution
    // for the mean of the Gaussian process by checking that the true
    // mean of a Gaussian process lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 500;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        for (unsigned int test = 0; test < nTests; ++test) {
            double mean = 0.5 * (test + 1);
            double variance = 4.0;

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, 500, samples);

            CNormalMeanPrecConjugate filter(
                makePrior(maths_t::E_ContinuousData, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval =
                    filter.confidenceIntervalMean(testIntervals[j]);

                if (mean < confidenceInterval.first || mean > confidenceInterval.second) {
                    ++errors[j];
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_TRACE(<< "interval = " << interval
                      << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            if (decayRates[i] == 0.0) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(interval, (100.0 - testIntervals[j]), 4.0);
            } else {
                BOOST_TEST_REQUIRE(interval <= (100.0 - testIntervals[j]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testPrecisionEstimation) {
    // We are going to test that we correctly estimate a distribution
    // for the precision of the Gaussian process by checking that the
    // true precision of a Gaussian process lies in various confidence
    // intervals the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 1000;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        for (unsigned int test = 0; test < nTests; ++test) {
            double mean = 0.5 * (test + 1);
            double variance = 2.0 + 0.01 * test;
            double precision = 1 / variance;

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, 1000, samples);

            CNormalMeanPrecConjugate filter(
                makePrior(maths_t::E_ContinuousData, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval =
                    filter.confidenceIntervalPrecision(testIntervals[j]);

                if (precision < confidenceInterval.first ||
                    precision > confidenceInterval.second) {
                    ++errors[j];
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_TRACE(<< "interval = " << interval
                      << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            if (decayRates[i] == 0.0) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(interval, (100.0 - testIntervals[j]), 3.0);
            } else {
                BOOST_TEST_REQUIRE(interval <= (100.0 - testIntervals[j]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihood) {
    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] = {maths_t::E_ContinuousData, maths_t::E_IntegerData};
    for (std::size_t t = 0; t < boost::size(dataTypes); ++t) {
        CNormalMeanPrecConjugate filter(makePrior(dataTypes[t]));

        const double mean = 1.0;
        const double variance = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 200, samples);
        filter.addSamples(samples);

        TWeightFunc weightsFuncs[]{static_cast<TWeightFunc>(maths_t::countWeight),
                                   static_cast<TWeightFunc>(maths_t::winsorisationWeight)};
        double weights[]{0.1, 1.0, 10.0};

        for (std::size_t i = 0; i < boost::size(weightsFuncs); ++i) {
            for (std::size_t j = 0; j < boost::size(weights); ++j) {
                double lb, ub;
                filter.minusLogJointCdf({1000.0}, {weightsFuncs[i](weights[j])}, lb, ub);
                LOG_DEBUG(<< "-log(c.d.f) = " << (lb + ub) / 2.0);
                BOOST_TEST_REQUIRE(lb >= 0.0);
                BOOST_TEST_REQUIRE(ub >= 0.0);
            }
        }
    }

    // Check that the marginal likelihood and c.d.f. agree for some
    // test data and that the c.d.f. <= 1.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const double mean = 5.0;
    const double variance = 1.0;
    test::CRandomNumbers rng;

    unsigned int numberSamples[] = {2u, 10u, 500u};
    const double tolerance = 1e-3;

    for (std::size_t i = 0; i < boost::size(numberSamples); ++i) {
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, numberSamples[i], samples);

        for (std::size_t j = 0; j < boost::size(decayRates); ++j) {
            CNormalMeanPrecConjugate filter(
                makePrior(maths_t::E_ContinuousData, decayRates[j]));

            for (std::size_t k = 0; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                filter.propagateForwardsByTime(1.0);
            }

            // We'll check that the p.d.f. is close to the derivative of the
            // c.d.f. at a range of deltas from the true mean.

            const double eps = 1e-4;
            double deltas[] = {-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0,
                               0.5,  1.0,  2.0,  3.0,  4.0,  5.0};

            for (std::size_t k = 0; k < boost::size(deltas); ++k) {
                double x = mean + deltas[k] * std::sqrt(variance);
                TDouble1Vec sample(1, x);

                LOG_TRACE(<< "number = " << numberSamples[i] << ", sample = " << sample[0]);

                double logLikelihood = 0.0;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    filter.jointLogMarginalLikelihood(sample, logLikelihood));
                double pdf = std::exp(logLikelihood);

                double lowerBound = 0.0, upperBound = 0.0;
                sample[0] -= eps;
                BOOST_TEST_REQUIRE(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                double minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtMinusEps = std::exp(-minusLogCdf);
                BOOST_TEST_REQUIRE(minusLogCdf >= 0.0);

                sample[0] += 2.0 * eps;
                BOOST_TEST_REQUIRE(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtPlusEps = std::exp(-minusLogCdf);
                BOOST_TEST_REQUIRE(minusLogCdf >= 0.0);

                double dcdfdx = (cdfAtPlusEps - cdfAtMinusEps) / 2.0 / eps;

                LOG_TRACE(<< "pdf(x) = " << pdf << ", d(cdf)/dx = " << dcdfdx);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, dcdfdx, tolerance);
            }
        }
    }

    {
        // Test that the sample mean of the log likelihood tends to the
        // expected log likelihood for a normal distribution (uniform law
        // of large numbers), which is just the differential entropy of
        // a normal R.V.

        boost::math::normal_distribution<> normal(mean, std::sqrt(variance));
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(normal);

        CNormalMeanPrecConjugate filter(makePrior());

        double differentialEntropy = 0.0;

        TDoubleVec seedSamples;
        rng.generateNormalSamples(mean, variance, 100, seedSamples);
        filter.addSamples(seedSamples);

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 100000, samples);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            TDouble1Vec sample(1, samples[i]);
            filter.addSamples(sample);
            double logLikelihood = 0.0;
            BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                filter.jointLogMarginalLikelihood(sample, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(samples.size());

        LOG_DEBUG(<< "differentialEntropy = " << differentialEntropy
                  << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedDifferentialEntropy, differentialEntropy, 2e-3);
    }

    {
        boost::math::normal_distribution<> normal(mean, std::sqrt(variance));
        const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                         0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
                                         2.0, 2.5, 3.0, 4.0, 5.0};

        CNormalMeanPrecConjugate filter(makePrior());
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 1000, samples);
        filter.addSamples(samples);

        const double percentages[] = {5.0,  10.0, 20.0, 30.0, 40.0,
                                      50.0, 60.0, 70.0, 80.0, 95.0};

        {
            // Test that marginal likelihood confidence intervals are
            // what we'd expect for various variance scales.

            TMeanAccumulator error;
            for (std::size_t i = 0; i < boost::size(percentages); ++i) {
                double q1, q2;
                filter.marginalLikelihoodQuantileForTest(50.0 - percentages[i] / 2.0,
                                                         1e-3, q1);
                filter.marginalLikelihoodQuantileForTest(50.0 + percentages[i] / 2.0,
                                                         1e-3, q2);
                TDoubleDoublePr interval =
                    filter.marginalLikelihoodConfidenceInterval(percentages[i]);
                LOG_TRACE(<< "[q1, q2] = [" << q1 << ", " << q2 << "]"
                          << ", interval = " << core::CContainerPrinter::print(interval));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(q1, interval.first, 0.005);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(q2, interval.second, 0.005);
                error.add(std::fabs(interval.first - q1));
                error.add(std::fabs(interval.second - q2));
            }
            LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 1e-3);
        }
        {
            TMeanAccumulator totalError;
            for (std::size_t i = 0; i < boost::size(varianceScales); ++i) {
                TMeanAccumulator error;
                double vs = varianceScales[i];
                boost::math::normal_distribution<> scaledNormal(mean, std::sqrt(vs * variance));
                LOG_DEBUG(<< "*** vs = " << vs << " ***");
                for (std::size_t j = 0; j < boost::size(percentages); ++j) {
                    double q1 = boost::math::quantile(
                        scaledNormal, (50.0 - percentages[j] / 2.0) / 100.0);
                    double q2 = boost::math::quantile(
                        scaledNormal, (50.0 + percentages[j] / 2.0) / 100.0);
                    TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(
                        percentages[j], maths_t::countVarianceScaleWeight(vs));
                    LOG_TRACE(<< "[q1, q2] = [" << q1 << ", " << q2 << "]"
                              << ", interval = " << core::CContainerPrinter::print(interval));
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(q1, interval.first, 0.3);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(q2, interval.second, 0.3);
                    error.add(std::fabs(interval.first - q1));
                    error.add(std::fabs(interval.second - q2));
                }
                LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
                BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.1);
                totalError += error;
            }
            LOG_DEBUG(<< "totalError = " << maths::CBasicStatistics::mean(totalError));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(totalError) < 0.06);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMean) {
    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    const double means[] = {1.0, 5.0, 100.0};
    const double variances[] = {2.0, 5.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        for (std::size_t j = 0; j < boost::size(variances); ++j) {
            LOG_DEBUG(<< "*** mean = " << means[i]
                      << ", variance = " << variances[j] << " ***");

            CNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateNormalSamples(means[i], variances[j], 1, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 100, samples);

            TMeanAccumulator relativeError;
            for (std::size_t k = 0; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));

                double expectedMean;
                BOOST_TEST_REQUIRE(filter.marginalLikelihoodMeanForTest(expectedMean));

                LOG_TRACE(<< "marginalLikelihoodMean = " << filter.marginalLikelihoodMean()
                          << ", expectedMean = " << expectedMean);

                // The error is at the precision of the numerical integration.
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMean,
                                             filter.marginalLikelihoodMean(), 0.01);

                relativeError.add(
                    std::fabs(expectedMean - filter.marginalLikelihoodMean()) / expectedMean);
            }

            LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(relativeError) < 1e-4);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMode) {
    // Test that the marginal likelihood mode is what we'd expect
    // with variances variance scales.

    const double means[] = {1.0, 5.0, 100.0};
    const double variances[] = {2.0, 5.0, 20.0};
    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
                                     2.0, 2.5, 3.0, 4.0, 5.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        for (std::size_t j = 0; j < boost::size(variances); ++j) {
            LOG_DEBUG(<< "*** mean = " << means[i]
                      << ", variance = " << variances[j] << " ***");

            CNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 1000, samples);
            filter.addSamples(samples);

            maths_t::TDoubleWeightsAry weight(maths_t::CUnitWeights::UNIT);

            for (std::size_t k = 0; k < boost::size(varianceScales); ++k) {
                double vs = varianceScales[i];
                maths_t::setCountVarianceScale(vs, weight);
                boost::math::normal_distribution<> scaledNormal(
                    means[i], std::sqrt(vs * variances[j]));
                double expectedMode = boost::math::mode(scaledNormal);
                LOG_DEBUG(<< "marginalLikelihoodMode = " << filter.marginalLikelihoodMode(weight)
                          << ", expectedMode = " << expectedMode);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode,
                                             filter.marginalLikelihoodMode(weight),
                                             0.12 * std::sqrt(variances[j]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodVariance) {
    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    const double means[] = {1.0, 5.0, 100.0};
    const double variances[] = {2.0, 5.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        for (std::size_t j = 0; j < boost::size(variances); ++j) {
            LOG_DEBUG(<< "*** mean = " << means[i]
                      << ", variance = " << variances[j] << " ***");

            CNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 100, samples);

            TMeanAccumulator relativeError;
            for (std::size_t k = 0; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                double expectedVariance;
                BOOST_TEST_REQUIRE(filter.marginalLikelihoodVarianceForTest(expectedVariance));
                LOG_TRACE(<< "marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                          << ", expectedVariance = " << expectedVariance);

                // The error is at the precision of the numerical integration.
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    expectedVariance, filter.marginalLikelihoodVariance(), 0.2);

                relativeError.add(std::fabs(expectedVariance -
                                            filter.marginalLikelihoodVariance()) /
                                  expectedVariance);
            }

            LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(relativeError) < 1e-3);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihood) {
    // We're going to test two properties of the sampling:
    //   1) That the sample mean is equal to the marginal
    //      likelihood mean.
    //   2) That the sample percentiles match the distribution
    //      percentiles.
    // We want to cross check these with the implementations of the
    // jointLogMarginalLikelihood and minusLogJointCdf so use these
    // to compute the mean and percentiles.

    const double mean = 5.0;
    const double variance = 3.0;

    const double eps = 1e-3;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 150, samples);

    CNormalMeanPrecConjugate filter(makePrior());

    TDouble1Vec sampled;

    for (std::size_t i = 0; i < 1; ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(10, sampled);

        BOOST_REQUIRE_EQUAL(i + 1, sampled.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(samples[i], sampled[i], eps);
    }

    TMeanAccumulator meanVarError;

    std::size_t numberSampled = 20;
    for (std::size_t i = 1; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(numberSampled, sampled);
        BOOST_REQUIRE_EQUAL(numberSampled, sampled.size());

        TMeanVarAccumulator sampledMoments;
        sampledMoments = std::for_each(sampled.begin(), sampled.end(), sampledMoments);

        LOG_DEBUG(<< "expectedMean = " << filter.marginalLikelihoodMean()
                  << ", sampledMean = " << maths::CBasicStatistics::mean(sampledMoments));
        LOG_DEBUG(<< "expectedVariance = " << filter.marginalLikelihoodVariance() << ", sampledVariance = "
                  << maths::CBasicStatistics::variance(sampledMoments));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(filter.marginalLikelihoodMean(),
                                     maths::CBasicStatistics::mean(sampledMoments), 1e-8);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(filter.marginalLikelihoodVariance(),
                                     maths::CBasicStatistics::variance(sampledMoments),
                                     0.2 * filter.marginalLikelihoodVariance());
        meanVarError.add(std::fabs(filter.marginalLikelihoodVariance() -
                                   maths::CBasicStatistics::variance(sampledMoments)) /
                         filter.marginalLikelihoodVariance());

        std::sort(sampled.begin(), sampled.end());
        for (std::size_t j = 1; j < sampled.size(); ++j) {
            double q = 100.0 * static_cast<double>(j) / static_cast<double>(numberSampled);

            double expectedQuantile;
            BOOST_TEST_REQUIRE(filter.marginalLikelihoodQuantileForTest(q, eps, expectedQuantile));

            LOG_TRACE(<< "quantile = " << q << ", x_quantile = " << expectedQuantile << ", quantile range = ["
                      << sampled[j - 1] << "," << sampled[j] << "]");

            BOOST_TEST_REQUIRE(expectedQuantile >= sampled[j - 1]);
            BOOST_TEST_REQUIRE(expectedQuantile <= sampled[j]);
        }
    }

    LOG_DEBUG(<< "mean variance error = " << maths::CBasicStatistics::mean(meanVarError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanVarError) < 0.04);
}

BOOST_AUTO_TEST_CASE(testCdf) {
    // Test error cases.
    //
    // Test some invariants:
    //   "cdf" + "cdf complement" = 1

    const double mean = 20.0;
    const double variance = 5.0;
    const std::size_t n[] = {20u, 80u};

    test::CRandomNumbers rng;

    CNormalMeanPrecConjugate filter(makePrior());

    for (std::size_t i = 0; i < boost::size(n); ++i) {
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, n[i], samples);

        filter.addSamples(samples);

        double lowerBound;
        double upperBound;
        BOOST_TEST_REQUIRE(!filter.minusLogJointCdf(TDouble1Vec(), lowerBound, upperBound));
        BOOST_TEST_REQUIRE(!filter.minusLogJointCdfComplement(
            TDouble1Vec(), lowerBound, upperBound));

        for (std::size_t j = 1; j < 500; ++j) {
            double x = static_cast<double>(j) / 2.0;

            BOOST_TEST_REQUIRE(filter.minusLogJointCdf(TDouble1Vec(1, x),
                                                       lowerBound, upperBound));
            double f = (lowerBound + upperBound) / 2.0;
            BOOST_TEST_REQUIRE(filter.minusLogJointCdfComplement(
                TDouble1Vec(1, x), lowerBound, upperBound));
            double fComplement = (lowerBound + upperBound) / 2.0;

            LOG_TRACE(<< "log(F(x)) = " << (f == 0.0 ? f : -f) << ", log(1 - F(x)) = "
                      << (fComplement == 0.0 ? fComplement : -fComplement));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, std::exp(-f) + std::exp(-fComplement), 1e-10);
        }
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySamples) {
    // We test that the probability of less likely samples calculation
    // agrees with the chance of seeing a sample with lower marginal
    // likelihood, up to the sampling error.
    //
    // We also check that the tail calculation attributes samples to
    // the appropriate tail of the distribution.

    const double means[] = {0.1, 1.5, 3.0};
    const double variances[] = {0.2, 0.4, 1.5};
    const double vs[] = {0.5, 1.0, 2.0};

    test::CRandomNumbers rng;

    TMeanAccumulator meanError;

    for (size_t i = 0; i < boost::size(means); ++i) {
        for (size_t j = 0; j < boost::size(variances); ++j) {
            LOG_DEBUG(<< "means = " << means[i] << ", variance = " << variances[j]);

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 1000, samples);

            CNormalMeanPrecConjugate filter(makePrior());
            filter.addSamples(samples);

            double mean = filter.mean();
            double sd = std::sqrt(1.0 / filter.precision());

            TDoubleVec likelihoods;
            for (std::size_t k = 0; k < samples.size(); ++k) {
                double likelihood;
                filter.jointLogMarginalLikelihood(TDouble1Vec(1, samples[k]), likelihood);
                likelihoods.push_back(likelihood);
            }
            std::sort(likelihoods.begin(), likelihoods.end());

            boost::math::normal_distribution<> normal(mean, sd);
            for (std::size_t k = 1; k < 10; ++k) {
                double x = boost::math::quantile(normal, static_cast<double>(k) / 10.0);

                TDouble1Vec sample(1, x);
                double fx;
                filter.jointLogMarginalLikelihood(sample, fx);

                double px = static_cast<double>(std::lower_bound(likelihoods.begin(),
                                                                 likelihoods.end(), fx) -
                                                likelihoods.begin()) /
                            static_cast<double>(likelihoods.size());

                double lb, ub;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lb, ub);

                double ssd = std::sqrt(px * (1.0 - px) /
                                       static_cast<double>(samples.size()));

                LOG_TRACE(<< "expected P(x) = " << px << ", actual P(x) = "
                          << (lb + ub) / 2.0 << " sample sd = " << ssd);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(px, (lb + ub) / 2.0, 3.0 * ssd);

                meanError.add(std::fabs(px - (lb + ub) / 2.0));
            }

            for (std::size_t k = 0; k < boost::size(vs); ++k) {
                double mode = filter.marginalLikelihoodMode(
                    maths_t::countVarianceScaleWeight(vs[k]));
                double ss[] = {0.9 * mode, 1.1 * mode};

                LOG_DEBUG(<< "vs = " << vs[k] << ", mode = " << mode);

                double lb, ub;
                maths_t::ETail tail;

                {
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, {ss[0]},
                        {maths_t::countVarianceScaleWeight(vs[k])}, lb, ub, tail);
                    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                    if (mode > 0.0) {
                        filter.probabilityOfLessLikelySamples(
                            maths_t::E_TwoSided, TDouble1Vec(ss, ss + 2),
                            maths_t::TDoubleWeightsAry1Vec(
                                2, maths_t::countVarianceScaleWeight(vs[k])),
                            lb, ub, tail);
                        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                        filter.probabilityOfLessLikelySamples(
                            maths_t::E_OneSidedBelow, TDouble1Vec(ss, ss + 2),
                            maths_t::TDoubleWeightsAry1Vec(
                                2, maths_t::countVarianceScaleWeight(vs[k])),
                            lb, ub, tail);
                        BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                        filter.probabilityOfLessLikelySamples(
                            maths_t::E_OneSidedAbove, TDouble1Vec(ss, ss + 2),
                            maths_t::TDoubleWeightsAry1Vec(
                                2, maths_t::countVarianceScaleWeight(vs[k])),
                            lb, ub, tail);
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    }
                }
                if (mode > 0.0) {
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, {ss[1]},
                        {maths_t::countVarianceScaleWeight(vs[k])}, lb, ub, tail);
                    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, TDouble1Vec(ss, ss + 2),
                        maths_t::TDoubleWeightsAry1Vec(
                            2, maths_t::countVarianceScaleWeight(vs[k])),
                        lb, ub, tail);
                    BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_OneSidedBelow, TDouble1Vec(ss, ss + 2),
                        maths_t::TDoubleWeightsAry1Vec(
                            2, maths_t::countVarianceScaleWeight(vs[k])),
                        lb, ub, tail);
                    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_OneSidedAbove, TDouble1Vec(ss, ss + 2),
                        maths_t::TDoubleWeightsAry1Vec(
                            2, maths_t::countVarianceScaleWeight(vs[k])),
                        lb, ub, tail);
                    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                }
            }
        }
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.01);
}

BOOST_AUTO_TEST_CASE(testAnomalyScore) {
    // This test pushes 500 samples through the filter and adds in
    // anomalous signals in the bins at 30, 120, 300 and 420 with
    // magnitude 4, 5, 10 and 15 standard deviations, respectively,
    // and checks the anomaly score has:
    //   1) high probability of detecting the anomalies, and
    //   2) a very low rate of false positives.

    using TUIntVec = std::vector<unsigned int>;

    const double decayRates[] = {0.0, 0.001, 0.01};

    const double means[] = {3.0, 15.0, 200.0};
    const double variances[] = {2.0, 5.0, 50.0};

    const double threshold = 0.01;

    const unsigned int anomalyTimes[] = {30u, 120u, 300u, 420u};
    const double anomalies[] = {4.0, 5.0, 10.0, 15.0, 0.0};

    test::CRandomNumbers rng;

    unsigned int test = 0;

    //std::ofstream file;
    //file.open("results.m");

    double totalFalsePositiveRate = 0.0;
    std::size_t totalPositives[] = {0u, 0u, 0u};

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        for (std::size_t j = 0; j < boost::size(variances); ++j) {
            LOG_DEBUG(<< "mean = " << means[i] << ", variance = " << variances[j]);

            boost::math::normal_distribution<> normal(means[i], std::sqrt(variances[j]));

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 500, samples);

            for (std::size_t k = 0; k < boost::size(decayRates); ++k) {
                CNormalMeanPrecConjugate filter(
                    makePrior(maths_t::E_ContinuousData, decayRates[k]));

                ++test;

                //std::ostringstream x;
                //std::ostringstream scores;
                //x << "x" << test << " = [";
                //scores << "score" << test << " = [";

                TUIntVec candidateAnomalies;
                for (unsigned int time = 0; time < samples.size(); ++time) {
                    double sample =
                        samples[time] +
                        (anomalies[std::find(std::begin(anomalyTimes), std::end(anomalyTimes), time) -
                                   std::begin(anomalyTimes)] *
                         boost::math::standard_deviation(normal));

                    TDouble1Vec sampleVec(1, sample);
                    filter.addSamples(sampleVec);

                    double score;
                    filter.anomalyScore(maths_t::E_TwoSided, sampleVec, score);
                    if (score > threshold) {
                        candidateAnomalies.push_back(time);
                    }

                    filter.propagateForwardsByTime(1.0);

                    //x << time << " ";
                    //scores << score << " ";
                }

                //x << "];\n";
                //scores << "];\n";
                //file << x.str() << scores.str() << "plot(x" << test << ", score"
                //     << test << ");\n"
                //     << "input(\"Hit any key for next test\");\n\n";

                TUIntVec falsePositives;
                std::set_difference(candidateAnomalies.begin(),
                                    candidateAnomalies.end(),
                                    std::begin(anomalyTimes), std::end(anomalyTimes),
                                    std::back_inserter(falsePositives));

                double falsePositiveRate = static_cast<double>(falsePositives.size()) /
                                           static_cast<double>(samples.size());

                totalFalsePositiveRate += falsePositiveRate;

                TUIntVec positives;
                std::set_intersection(candidateAnomalies.begin(),
                                      candidateAnomalies.end(),
                                      std::begin(anomalyTimes), std::end(anomalyTimes),
                                      std::back_inserter(positives));

                LOG_DEBUG(<< "falsePositiveRate = " << falsePositiveRate
                          << ", positives = " << positives.size());

                // False alarm rate should be less than 0.6%.
                BOOST_TEST_REQUIRE(falsePositiveRate <= 0.006);

                // Should detect at least the three biggest anomalies.
                BOOST_TEST_REQUIRE(positives.size() >= 3u);

                totalPositives[k] += positives.size();
            }
        }
    }

    totalFalsePositiveRate /= static_cast<double>(test);

    LOG_DEBUG(<< "totalFalsePositiveRate = " << totalFalsePositiveRate);

    for (std::size_t i = 0; i < boost::size(totalPositives); ++i) {
        LOG_DEBUG(<< "positives = " << totalPositives[i]);

        // Should detect all but one anomaly.
        BOOST_TEST_REQUIRE(totalPositives[i] >= 32u);
    }

    // Total false alarm rate should be less than 0.3%.
    BOOST_TEST_REQUIRE(totalFalsePositiveRate < 0.003);
}

BOOST_AUTO_TEST_CASE(testIntegerData) {
    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // The idea of this test is to check that the inferred model agrees in the
    // limit (large n) with the model inferred from such data.

    const double mean = 12.0;
    const double variance = 3.0;
    const std::size_t nSamples = 100000;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, nSamples, samples);

    TDoubleVec uniform;
    rng.generateUniformSamples(0.0, 1.0, nSamples, uniform);

    {
        CNormalMeanPrecConjugate filter1(makePrior(maths_t::E_IntegerData));
        CNormalMeanPrecConjugate filter2(makePrior(maths_t::E_ContinuousData));

        for (std::size_t i = 0; i < nSamples; ++i) {
            double x = floor(samples[i]);

            TDouble1Vec sample(1, x);
            filter1.addSamples(sample);

            sample[0] += uniform[i];
            filter2.addSamples(sample);
        }

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.001);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));

        TMeanAccumulator meanLogLikelihood1;
        TMeanAccumulator meanLogLikelihood2;
        for (std::size_t j = 0; j < nSamples; ++j) {
            double x = std::floor(samples[j]);

            TDouble1Vec sample(1, x);
            double logLikelihood1;
            filter1.jointLogMarginalLikelihood(sample, logLikelihood1);
            meanLogLikelihood1.add(-logLikelihood1);

            sample[0] += uniform[j];
            double logLikelihood2;
            filter2.jointLogMarginalLikelihood(sample, logLikelihood2);
            meanLogLikelihood2.add(-logLikelihood2);
        }

        LOG_DEBUG(<< "meanLogLikelihood1 = " << maths::CBasicStatistics::mean(meanLogLikelihood1)
                  << ", meanLogLikelihood2 = "
                  << maths::CBasicStatistics::mean(meanLogLikelihood2));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            maths::CBasicStatistics::mean(meanLogLikelihood1),
            maths::CBasicStatistics::mean(meanLogLikelihood2), 0.02);
    }

    {
        TDoubleVec seedSamples;
        rng.generateNormalSamples(mean, variance, 100, seedSamples);
        CNormalMeanPrecConjugate filter1(makePrior(maths_t::E_IntegerData, 0.1));
        filter1.addSamples(seedSamples);

        CNormalMeanPrecConjugate filter2 = filter1;
        filter2.dataType(maths_t::E_ContinuousData);

        TMeanAccumulator meanProbability1;
        TMeanAccumulator meanProbability2;

        for (std::size_t i = 0; i < nSamples; ++i) {
            double x = std::floor(samples[i]);

            TDouble1Vec sample(1, x);

            double l1, u1;
            BOOST_TEST_REQUIRE(filter1.probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, sample, l1, u1));
            BOOST_REQUIRE_EQUAL(l1, u1);
            double p1 = (l1 + u1) / 2.0;
            meanProbability1.add(p1);

            sample[0] += uniform[i];
            double l2, u2;
            BOOST_TEST_REQUIRE(filter2.probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, sample, l2, u2));
            BOOST_REQUIRE_EQUAL(l2, u2);
            double p2 = (l2 + u2) / 2.0;
            meanProbability2.add(p2);
        }

        double p1 = maths::CBasicStatistics::mean(meanProbability1);
        double p2 = maths::CBasicStatistics::mean(meanProbability2);
        LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.01 * p1);
    }
}

BOOST_AUTO_TEST_CASE(testLowVariationData) {
    {
        CNormalMeanPrecConjugate filter(makePrior(maths_t::E_IntegerData));
        for (std::size_t i = 0; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG(<< "68% confidence interval " << core::CContainerPrinter::print(interval)
                  << ", approximate variance = " << sigma * sigma);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(12.0, 1.0 / (sigma * sigma), 0.15);
    }
    {
        CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData));
        for (std::size_t i = 0; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG(<< "68% confidence interval " << core::CContainerPrinter::print(interval)
                  << ", approximate s.t.d. = " << sigma);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            1.0 / maths::MINIMUM_COEFFICIENT_OF_VARIATION / 430.5, 1.0 / sigma, 7.0);
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that persist/restore is idempotent.

    const double mean = 10.0;
    const double variance = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    maths::CNormalMeanPrecConjugate origFilter(makePrior());
    for (std::size_t i = 0; i < samples.size(); ++i) {
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

    LOG_DEBUG(<< "Normal mean conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, decayRate + 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    maths::CNormalMeanPrecConjugate restoredFilter(params, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    BOOST_REQUIRE_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testSeasonalVarianceScale) {
    // We are test:
    //   1) The marginal likelihood is normalized.
    //   2) E[(X - m)^2] w.r.t. the log-likelihood is scaled.
    //   3) E[(X - m)^2] is close to marginalLikelihoodVariance.
    //   4) The mode is at the maximum of the marginal likelihood.
    //   5) dF/dx = exp(log-likelihood) with different scales.
    //   6) The probability of less likely sample transforms as
    //      expected.
    //   7) Updating with scaled samples behaves as expected.

    const double means[] = {0.2, 1.0, 20.0};
    const double variances[] = {0.2, 1.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        for (std::size_t j = 0; j < boost::size(variances); ++j) {
            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 100, samples);

            double varianceScales[] = {0.2, 0.5, 1.0, 2.0, 5.0};
            maths_t::TDoubleWeightsAry weight(maths_t::CUnitWeights::UNIT);

            double m;
            double v;

            {
                CNormalMeanPrecConjugate filter(makePrior());
                filter.addSamples(samples);

                m = filter.marginalLikelihoodMean();
                v = filter.marginalLikelihoodVariance();
                double s = std::sqrt(v);
                LOG_DEBUG(<< "m = " << m << ", v = " << v);

                double points[] = {m - 3.0 * s, m - s, m, m + s, m + 3.0 * s};

                double unscaledExpectationVariance;
                filter.expectation(CVarianceKernel(filter.marginalLikelihoodMean()),
                                   100, unscaledExpectationVariance);
                LOG_DEBUG(<< "unscaledExpectationVariance = " << unscaledExpectationVariance);

                for (std::size_t k = 0; k < boost::size(varianceScales); ++k) {
                    double vs = varianceScales[k];
                    maths_t::setSeasonalVarianceScale(vs, weight);
                    LOG_DEBUG(<< "*** variance scale = " << vs << " ***");

                    double Z;
                    filter.expectation(C1dUnitKernel(), 50, Z, weight);
                    LOG_DEBUG(<< "Z = " << Z);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, Z, 1e-3);

                    LOG_DEBUG(<< "sv = " << filter.marginalLikelihoodVariance(weight));
                    double expectationVariance;
                    filter.expectation(CVarianceKernel(filter.marginalLikelihoodMean()),
                                       100, expectationVariance, weight);
                    LOG_DEBUG(<< "expectationVariance = " << expectationVariance);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        vs * unscaledExpectationVariance, expectationVariance,
                        0.01 * vs * unscaledExpectationVariance);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        filter.marginalLikelihoodVariance(weight), expectationVariance,
                        0.01 * filter.marginalLikelihoodVariance(weight));

                    double mode = filter.marginalLikelihoodMode(weight);
                    double fm;
                    double fmMinusEps, fmPlusEps;
                    filter.jointLogMarginalLikelihood({mode - 1e-3}, {weight}, fmMinusEps);
                    filter.jointLogMarginalLikelihood({mode}, {weight}, fm);
                    filter.jointLogMarginalLikelihood({mode + 1e-3}, {weight}, fmPlusEps);
                    LOG_DEBUG(<< "log(f(mode)) = " << fm << ", log(f(mode - eps)) = " << fmMinusEps
                              << ", log(f(mode + eps)) = " << fmPlusEps);
                    BOOST_TEST_REQUIRE(fm > fmMinusEps);
                    BOOST_TEST_REQUIRE(fm > fmPlusEps);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        0.0, (std::exp(fmPlusEps) - std::exp(fmMinusEps)) / 2e-3, 1e-6);
                    TDouble1Vec sample(1, 0.0);
                    for (std::size_t l = 0; l < boost::size(points); ++l) {
                        TDouble1Vec x(1, points[l]);
                        double fx;
                        filter.jointLogMarginalLikelihood(x, {weight}, fx);
                        TDouble1Vec xMinusEps(1, points[l] - 1e-3);
                        TDouble1Vec xPlusEps(1, points[l] + 1e-3);
                        double lb, ub;
                        filter.minusLogJointCdf(xPlusEps, {weight}, lb, ub);
                        double FxPlusEps = std::exp(-(lb + ub) / 2.0);
                        filter.minusLogJointCdf(xMinusEps, {weight}, lb, ub);
                        double FxMinusEps = std::exp(-(lb + ub) / 2.0);
                        LOG_TRACE(<< "x = " << points[l] << ", log(f(x)) = " << fx
                                  << ", F(x - eps) = " << FxMinusEps
                                  << ", F(x + eps) = " << FxPlusEps << ", log(dF/dx)) = "
                                  << std::log((FxPlusEps - FxMinusEps) / 2e-3));
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(
                            fx, std::log((FxPlusEps - FxMinusEps) / 2e-3),
                            0.05 * std::fabs(fx));

                        sample[0] = m + (points[l] - m) / std::sqrt(vs);
                        maths_t::setSeasonalVarianceScale(1.0, weight);
                        double expectedLowerBound;
                        double expectedUpperBound;
                        maths_t::ETail expectedTail;
                        filter.probabilityOfLessLikelySamples(
                            maths_t::E_TwoSided, sample, {weight},
                            expectedLowerBound, expectedUpperBound, expectedTail);

                        sample[0] = points[l];
                        maths_t::setSeasonalVarianceScale(vs, weight);
                        double lowerBound;
                        double upperBound;
                        maths_t::ETail tail;
                        filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                              sample, {weight}, lowerBound,
                                                              upperBound, tail);

                        LOG_TRACE(<< "expectedLowerBound = " << expectedLowerBound);
                        LOG_TRACE(<< "lowerBound         = " << lowerBound);
                        LOG_TRACE(<< "expectedUpperBound = " << expectedUpperBound);
                        LOG_TRACE(<< "upperBound         = " << upperBound);
                        LOG_TRACE(<< "expectedTail       = " << expectedTail);
                        LOG_TRACE(<< "tail               = " << tail);

                        if ((expectedLowerBound + expectedUpperBound) < 0.02) {
                            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                                std::log(expectedLowerBound), std::log(lowerBound),
                                0.1 * std::fabs(std::log(expectedLowerBound)));
                            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                                std::log(expectedUpperBound), std::log(upperBound),
                                0.1 * std::fabs(std::log(expectedUpperBound)));
                        } else {
                            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedLowerBound, lowerBound,
                                                         0.01 * expectedLowerBound);
                            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedUpperBound, upperBound,
                                                         0.01 * expectedUpperBound);
                        }
                        BOOST_REQUIRE_EQUAL(expectedTail, tail);
                    }
                }
            }
            for (std::size_t k = 0; k < boost::size(varianceScales); ++k) {
                double vs = varianceScales[k];

                rng.random_shuffle(samples.begin(), samples.end());

                CNormalMeanPrecConjugate filter(makePrior());
                maths_t::setSeasonalVarianceScale(vs, weight);
                for (std::size_t l = 0; l < samples.size(); ++l) {
                    filter.addSamples({samples[l]}, {weight});
                }

                double sm = filter.marginalLikelihoodMean();
                double sv = filter.marginalLikelihoodVariance();
                LOG_DEBUG(<< "m  = " << m << ", v  = " << v);
                LOG_DEBUG(<< "sm = " << sm << ", sv = " << sv);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(m, sm, std::fabs(0.25 * m));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(v / vs, sv, 0.05 * v / vs);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testCountVarianceScale) {
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
    // normal distribution (uniform law of large numbers), which
    // is just the differential entropy of a scaled normal R.V.
    //
    // Finally, we test update with scaled samples produces the
    // correct posterior.

    const double mean = 12.0;
    const double variance = 3.0;

    const double varianceScales[] = {0.20, 0.50, 0.75, 1.50, 2.00, 5.00};

    LOG_DEBUG(<< "****** probabilityOfLessLikelySamples ******");

    const double percentiles[] = {10.0, 20.0, 30.0, 40.0, 50.0,
                                  60.0, 70.0, 80.0, 90.0};
    const std::size_t nSamples[] = {30u, 1000u};
    const std::size_t nScaledSamples = 10000;

    double percentileErrorTolerances[] = {0.15, 0.03};
    double totalErrorTolerances[] = {0.25, 0.13};
    double totalTotalError = 0.0;

    for (std::size_t i = 0; i < boost::size(nSamples); ++i) {
        LOG_DEBUG(<< "**** nSamples = " << nSamples[i] << " ****");

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, nSamples[i], samples);

        CNormalMeanPrecConjugate filter(makePrior());
        filter.addSamples(samples);

        double expectedTotalError = 0.0;
        TDoubleVec expectedPercentileErrors;
        {
            TDoubleVec unscaledSamples;
            rng.generateNormalSamples(mean, variance, nScaledSamples, unscaledSamples);

            TDoubleVec probabilities;
            probabilities.reserve(nScaledSamples);
            for (std::size_t j = 0; j < unscaledSamples.size(); ++j) {
                TDouble1Vec sample(1, unscaledSamples[j]);

                double lowerBound, upperBound;
                BOOST_TEST_REQUIRE(filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, sample, lowerBound, upperBound));
                BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                double probability = (lowerBound + upperBound) / 2.0;
                probabilities.push_back(probability);
            }
            std::sort(probabilities.begin(), probabilities.end());

            for (std::size_t j = 0; j < boost::size(percentiles); ++j) {
                std::size_t index = static_cast<std::size_t>(
                    static_cast<double>(nScaledSamples) * percentiles[j] / 100.0);
                double error = std::fabs(probabilities[index] - percentiles[j] / 100.0);
                expectedPercentileErrors.push_back(error);
                expectedTotalError += error;
            }
        }

        for (std::size_t j = 0; j < boost::size(varianceScales); ++j) {
            LOG_DEBUG(<< "**** variance scale = " << varianceScales[j] << " ****");

            TDoubleVec scaledSamples;
            rng.generateNormalSamples(mean, varianceScales[j] * variance,
                                      nScaledSamples, scaledSamples);

            TDoubleVec probabilities;
            probabilities.reserve(nScaledSamples);
            for (std::size_t k = 0; k < scaledSamples.size(); ++k) {

                double lowerBound, upperBound;
                maths_t::ETail tail;
                BOOST_TEST_REQUIRE(filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, {scaledSamples[k]},
                    {maths_t::countVarianceScaleWeight(varianceScales[j])},
                    lowerBound, upperBound, tail));
                BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                double probability = (lowerBound + upperBound) / 2.0;
                probabilities.push_back(probability);
            }
            std::sort(probabilities.begin(), probabilities.end());

            double totalError = 0.0;
            for (std::size_t k = 0; k < boost::size(percentiles); ++k) {
                std::size_t index = static_cast<std::size_t>(
                    static_cast<double>(nScaledSamples) * percentiles[k] / 100.0);
                double error = fabs(probabilities[index] - percentiles[k] / 100.0);
                totalError += error;
                double errorThreshold = percentileErrorTolerances[i] +
                                        expectedPercentileErrors[k];

                LOG_TRACE(<< "percentile = " << percentiles[k] << ", probability = "
                          << probabilities[index] << ", error = " << error
                          << ", error threshold = " << errorThreshold);

                BOOST_TEST_REQUIRE(error < errorThreshold);
            }

            double totalErrorThreshold = totalErrorTolerances[i] + expectedTotalError;

            LOG_DEBUG(<< "totalError = " << totalError
                      << ", totalError threshold = " << totalErrorThreshold);

            BOOST_TEST_REQUIRE(totalError < totalErrorThreshold);
            totalTotalError += totalError;
        }
    }

    LOG_DEBUG(<< "total totalError = " << totalTotalError);
    BOOST_TEST_REQUIRE(totalTotalError < 3.5);

    LOG_DEBUG(<< "****** jointLogMarginalLikelihood ******");

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(varianceScales); ++i) {
        LOG_DEBUG(<< "**** variance scale = " << varianceScales[i] << " ****");

        boost::math::normal_distribution<> normal(
            mean, std::sqrt(varianceScales[i] * variance));
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(normal);

        CNormalMeanPrecConjugate filter(makePrior());

        double differentialEntropy = 0.0;

        TDoubleVec seedSamples;
        rng.generateNormalSamples(mean, variance, 1000, seedSamples);
        filter.addSamples(seedSamples);

        TDoubleVec scaledSamples;
        rng.generateNormalSamples(mean, varianceScales[i] * variance, 10000, scaledSamples);
        for (std::size_t j = 0; j < scaledSamples.size(); ++j) {
            double logLikelihood = 0.0;
            BOOST_REQUIRE_EQUAL(
                maths_t::E_FpNoErrors,
                filter.jointLogMarginalLikelihood(
                    {scaledSamples[j]},
                    {maths_t::countVarianceScaleWeight(varianceScales[i])}, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(scaledSamples.size());

        LOG_DEBUG(<< "differentialEntropy = " << differentialEntropy
                  << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedDifferentialEntropy, differentialEntropy, 0.03);
    }

    LOG_DEBUG(<< "****** addSamples ******");

    // This tests update with variable variance scale. In particular,
    // we update with samples from N(0,1) and N(0,5) and test that
    // the variance is correctly estimated if we compensate using a
    // variance scale.

    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};
    unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

    double variances[] = {1.0, 5.0};
    double precision = 1 / variances[0];

    for (std::size_t t = 0; t < 1000; ++t) {
        CNormalMeanPrecConjugate filter(makePrior());

        for (std::size_t i = 0; i < boost::size(variances); ++i) {
            TDoubleVec samples;
            rng.generateNormalSamples(0.0, variances[i], 1000, samples);
            filter.addSamples(samples, maths_t::TDoubleWeightsAry1Vec(
                                           samples.size(), maths_t::countVarianceScaleWeight(
                                                               variances[i])));
        }

        for (std::size_t i = 0; i < boost::size(testIntervals); ++i) {
            TDoubleDoublePr confidenceInterval =
                filter.confidenceIntervalPrecision(testIntervals[i]);
            if (precision < confidenceInterval.first ||
                precision > confidenceInterval.second) {
                ++errors[i];
            }
        }
    }

    for (std::size_t i = 0; i < boost::size(testIntervals); ++i) {
        double interval = 100.0 * errors[i] / 1000.0;
        LOG_DEBUG(<< "interval = " << interval
                  << ", expectedInterval = " << (100.0 - testIntervals[i]));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(interval, (100.0 - testIntervals[i]), 4.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
