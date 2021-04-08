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
#include <maths/CGammaRateConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/distributions/gamma.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

BOOST_AUTO_TEST_SUITE(CGammaRateConjugateTest)

using namespace ml;
using namespace handy_typedefs;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using CGammaRateConjugate = CPriorTestInterfaceMixin<maths::CGammaRateConjugate>;
using TWeightFunc = maths_t::TDoubleWeightsAry (*)(double);

CGammaRateConjugate makePrior(maths_t::EDataType dataType = maths_t::E_ContinuousData,
                              const double& offset = 0.0,
                              const double& decayRate = 0.0) {
    return CGammaRateConjugate::nonInformativePrior(dataType, offset, decayRate, 0.0);
}
}

BOOST_AUTO_TEST_CASE(testMultipleUpdate) {
    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    const maths_t::EDataType dataTypes[]{maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double shape = 2.0;
    const double scale = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 100, samples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CGammaRateConjugate filter1(makePrior(dataTypes[i]));
        CGammaRateConjugate filter2(filter1);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter1.addSamples(TDouble1Vec{samples[j]});
        }
        filter2.addSamples(samples);

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 1e-3);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }

    // Test with variance scale.

    TDoubleVec scaledSamples;
    rng.generateGammaSamples(shape / 2.0, 2.0 * scale, 100, scaledSamples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CGammaRateConjugate filter1(makePrior(dataTypes[i]));
        filter1.addSamples(samples);
        CGammaRateConjugate filter2(filter1);

        for (std::size_t j = 0; j < scaledSamples.size(); ++j) {
            filter1.addSamples({scaledSamples[j]},
                               {ml::maths_t::countVarianceScaleWeight(2.0)});
        }
        filter2.addSamples(scaledSamples, maths_t::TDoubleWeightsAry1Vec(
                                              scaledSamples.size(),
                                              maths_t::countVarianceScaleWeight(2.0)));

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.03);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }

    // Test the count weight is equivalent to adding repeated samples.

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CGammaRateConjugate filter1(makePrior(dataTypes[i]));
        CGammaRateConjugate filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;

        for (std::size_t j = 0; j < count; ++j) {
            filter1.addSamples(TDouble1Vec{x});
        }
        filter2.addSamples({x}, {maths_t::countWeight(static_cast<double>(count))});

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.01);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }
}

BOOST_AUTO_TEST_CASE(testPropagation) {
    // The quantities are preserved up to the solving tolerance given that
    // the updated count is still relatively large so the digamma function
    // is very nearly equal to the log function.
    const double eps = 1e-3;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 3.0, 500, samples);

    CGammaRateConjugate filter(makePrior(maths_t::E_ContinuousData, 0.1));

    for (std::size_t i = 0; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));
    }

    double shape = filter.likelihoodShape();
    double rate = filter.likelihoodRate();

    filter.propagateForwardsByTime(5.0);

    double propagatedShape = filter.likelihoodShape();
    double propagatedRate = filter.likelihoodRate();

    LOG_DEBUG(<< "shape = " << shape << ", rate = " << rate << ", propagatedShape = "
              << propagatedShape << ", propagatedRate = " << propagatedRate);

    BOOST_REQUIRE_CLOSE_ABSOLUTE(shape, propagatedShape, eps);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(rate, propagatedRate, eps);
}

BOOST_AUTO_TEST_CASE(testShapeEstimation) {
    // The idea here is to check that the likelihood shape estimate converges
    // to the correct value for a range of distribution parameters. We do not
    // use any explicit bounds on the convergence rates so simply check that
    // we do get closer as the number of samples increases.

    const double decayRates[] = {0.0, 0.001, 0.01};

    for (size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        double tests = 0.0;
        double errorIncreased = 0.0;

        for (unsigned int test = 0; test < 100; ++test) {
            double shape = 0.5 * (test + 1.0);
            double scale = 2.0;

            TDoubleVec samples;
            rng.generateGammaSamples(shape, scale, 5050, samples);

            using TGammaRateConjugateVec = std::vector<CGammaRateConjugate>;

            unsigned int nAggregate = 50;
            TGammaRateConjugateVec filters(
                nAggregate, makePrior(maths_t::E_ContinuousData, 0.0, decayRates[i]));

            double previousError = std::numeric_limits<double>::max();
            double averageShape = 0.0;

            for (std::size_t j = 0; j < samples.size() / nAggregate; ++j) {
                double error = 0.0;
                averageShape = 0.0;
                for (std::size_t k = 0; k < nAggregate; ++k) {
                    filters[k].addSamples(TDouble1Vec(1, samples[nAggregate * j + k]));
                    filters[k].propagateForwardsByTime(1.0);

                    error += fabs(shape - filters[k].likelihoodShape());
                    averageShape += filters[k].likelihoodShape();
                }
                error /= static_cast<double>(nAggregate);
                averageShape /= static_cast<double>(nAggregate);

                if (j > 0u && j % 20u == 0u) {
                    if (error > previousError) {
                        errorIncreased += 1.0;
                    }
                    tests += 1.0;
                    previousError = error;
                }
            }

            LOG_TRACE(<< "shape = " << shape << ", averageShape = " << averageShape);

            // Average error after 100 updates should be less than 8%.
            BOOST_REQUIRE_CLOSE_ABSOLUTE(shape, averageShape, 0.08 * shape);
        }

        // Error should only increase in at most 7% of measurements.
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, errorIncreased, 0.07 * tests);
    }
}

BOOST_AUTO_TEST_CASE(testRateEstimation) {
    // We are going to test that we correctly estimate a distribution
    // for the rate of the gamma process by checking that the true
    // rate of a gamma process lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 100;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    for (size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        for (unsigned int test = 0; test < nTests; ++test) {
            double shape = 2.0;
            double scale = 0.2 * (test + 1.0);
            double rate = 1.0 / scale;

            TDoubleVec samples;
            rng.generateGammaSamples(shape, scale, 100, samples);

            CGammaRateConjugate filter(makePrior(maths_t::E_ContinuousData, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval =
                    filter.confidenceIntervalRate(testIntervals[j]);

                if (rate < confidenceInterval.first || rate > confidenceInterval.second) {
                    ++errors[j];
                }
            }
        }

        for (size_t j = 0; j < boost::size(testIntervals); ++j) {
            // The number of errors should be inside the percentile bounds.
            unsigned int maximumErrors = static_cast<unsigned int>(
                std::ceil((1.0 - testIntervals[j] / 100.0) * nTests));

            LOG_TRACE(<< "errors = " << errors[j] << ", maximumErrors = " << maximumErrors);

            BOOST_TEST_REQUIRE(errors[j] <= maximumErrors + 2);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihood) {
    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] = {maths_t::E_ContinuousData, maths_t::E_IntegerData};
    for (std::size_t t = 0; t < boost::size(dataTypes); ++t) {
        CGammaRateConjugate filter(makePrior());

        const double shape = 1.0;
        const double scale = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateGammaSamples(shape, scale, 200, samples);
        filter.addSamples(samples);

        TWeightFunc weightsFuncs[]{static_cast<TWeightFunc>(maths_t::countWeight),
                                   static_cast<TWeightFunc>(maths_t::winsorisationWeight)};
        double weights[]{0.1, 0.9, 10.0};

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

    const double shape = 5.0;
    const double scale = 1.0;

    boost::math::gamma_distribution<> gamma(shape, scale);
    double mean = boost::math::mean(gamma);
    double variance = boost::math::variance(gamma);

    test::CRandomNumbers rng;

    unsigned int numberSamples[] = {4u, 10u, 300u, 500u};
    const double tolerances[] = {1e-8, 1e-8, 0.01, 0.001};

    for (size_t i = 0; i < boost::size(numberSamples); ++i) {
        TDoubleVec samples;
        rng.generateGammaSamples(shape, scale, numberSamples[i], samples);

        for (size_t j = 0; j < boost::size(decayRates); ++j) {
            CGammaRateConjugate filter(
                makePrior(maths_t::E_ContinuousData, 0.0, decayRates[j]));

            for (std::size_t k = 0; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                filter.propagateForwardsByTime(1.0);
            }

            // We'll check that the p.d.f. is close to the derivative
            // of the c.d.f. at a range of deltas from the true mean.

            const double eps = 1e-4;
            double deltas[] = {-2.0, -1.6, -1.2, -0.8, -0.4, -0.2, 0.0,
                               0.5,  1.0,  2.0,  3.0,  4.0,  5.0};

            for (size_t k = 0; k < boost::size(deltas); ++k) {
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

                BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, dcdfdx, tolerances[i]);
            }
        }
    }

    {
        // Test that the sample expectation of the log likelihood tends to
        // the expected log likelihood for a gamma distribution (uniform law
        // of large numbers), which is just the differential entropy of a
        // gamma R.V.

        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(gamma);

        CGammaRateConjugate filter(makePrior());

        double differentialEntropy = 0.0;

        TDoubleVec seedSamples;
        rng.generateGammaSamples(shape, scale, 100, seedSamples);
        filter.addSamples(seedSamples);

        TDoubleVec samples;
        rng.generateGammaSamples(shape, scale, 100000, samples);
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

        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedDifferentialEntropy, differentialEntropy, 0.0025);
    }

    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
                                     2.0, 2.5, 3.0, 4.0, 5.0};

    CGammaRateConjugate filter(makePrior());
    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 1000, samples);
    filter.addSamples(samples);

    const double percentages[] = {5.0,  10.0, 20.0, 30.0, 40.0,
                                  50.0, 60.0, 70.0, 80.0, 95.0};

    {
        // Test that marginal likelihood confidence intervals are
        // what we'd expect for various variance scales.

        TMeanAccumulator error;
        for (std::size_t i = 0; i < boost::size(percentages); ++i) {
            double q1, q2;
            filter.marginalLikelihoodQuantileForTest(50.0 - percentages[i] / 2.0, 1e-3, q1);
            filter.marginalLikelihoodQuantileForTest(50.0 + percentages[i] / 2.0, 1e-3, q2);
            TDoubleDoublePr interval =
                filter.marginalLikelihoodConfidenceInterval(percentages[i]);
            LOG_TRACE(<< "[q1, q2] = [" << q1 << ", " << q2 << "]"
                      << ", interval = " << core::CContainerPrinter::print(interval));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(q1, interval.first, 0.02);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(q2, interval.second, 0.02);
            error.add(std::fabs(interval.first - q1));
            error.add(std::fabs(interval.second - q2));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 4e-3);
    }
    {
        maths_t::TDoubleWeightsAry weight(maths_t::CUnitWeights::UNIT);
        TMeanAccumulator totalError;
        for (std::size_t i = 0; i < boost::size(varianceScales); ++i) {
            TMeanAccumulator error;
            double vs = varianceScales[i];
            maths_t::setCountVarianceScale(vs, weight);
            LOG_DEBUG(<< "*** vs = " << vs << " ***");
            for (std::size_t j = 0; j < boost::size(percentages); ++j) {
                boost::math::gamma_distribution<> scaledGamma(shape / vs, vs * scale);
                double q1 = boost::math::quantile(
                    scaledGamma, (50.0 - percentages[j] / 2.0) / 100.0);
                double q2 = boost::math::quantile(
                    scaledGamma, (50.0 + percentages[j] / 2.0) / 100.0);
                TDoubleDoublePr interval =
                    filter.marginalLikelihoodConfidenceInterval(percentages[j], weight);
                LOG_TRACE(<< "[q1, q2] = [" << q1 << ", " << q2 << "]"
                          << ", interval = " << core::CContainerPrinter::print(interval));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(q1, interval.first, 0.4);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(q2, interval.second, 0.4);
                error.add(std::fabs(interval.first - q1));
                error.add(std::fabs(interval.second - q2));
            }
            LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.09);
            totalError += error;
        }
        LOG_DEBUG(<< "totalError = " << maths::CBasicStatistics::mean(totalError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(totalError) < 0.042);
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMean) {
    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    const double shapes[] = {5.0, 20.0, 40.0};
    const double scales[] = {1.0, 10.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(shapes); ++i) {
        for (std::size_t j = 0; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "*** shape = " << shapes[i] << ", scale = " << scales[j] << " ***");

            CGammaRateConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shapes[i], scales[j], 3, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 100, samples);

            for (std::size_t k = 0; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));

                double expectedMean;
                BOOST_TEST_REQUIRE(filter.marginalLikelihoodMeanForTest(expectedMean));

                LOG_TRACE(<< "marginalLikelihoodMean = " << filter.marginalLikelihoodMean()
                          << ", expectedMean = " << expectedMean);

                // The error is mainly due to the truncation in the
                // integration range used to compute the expected mean.
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    expectedMean, filter.marginalLikelihoodMean(), 1e-3 * expectedMean);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMode) {
    // Test that the marginal likelihood mode is what we'd expect
    // with variances variance scales.

    const double shapes[] = {5.0, 20.0, 40.0};
    const double scales[] = {1.0, 10.0, 20.0};
    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                     0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(shapes); ++i) {
        for (std::size_t j = 0; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "*** shape = " << shapes[i] << ", scale = " << scales[j] << " ***");

            CGammaRateConjugate filter(makePrior());

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 1000, samples);
            filter.addSamples(samples);

            TMeanAccumulator relativeError;
            maths_t::TDoubleWeightsAry weight(maths_t::CUnitWeights::UNIT);
            for (std::size_t k = 0; k < boost::size(varianceScales); ++k) {
                double vs = varianceScales[k];
                maths_t::setCountVarianceScale(vs, weight);
                boost::math::gamma_distribution<> scaledGamma(shapes[i] / vs,
                                                              vs * scales[j]);
                double expectedMode = boost::math::mode(scaledGamma);
                LOG_TRACE(<< "marginalLikelihoodMode = " << filter.marginalLikelihoodMode(weight)
                          << ", expectedMode = " << expectedMode);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode,
                                             filter.marginalLikelihoodMode(weight),
                                             0.28 * expectedMode + 0.3);
                double error = std::fabs(filter.marginalLikelihoodMode(weight) - expectedMode);
                relativeError.add(error == 0.0 ? 0.0 : error / expectedMode);
            }
            LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(relativeError) < 0.08);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodVariance) {
    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    const double shapes[] = {5.0, 20.0, 40.0};
    const double scales[] = {1.0, 10.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(shapes); ++i) {
        for (std::size_t j = 0; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "*** shape = " << shapes[i] << ", scale = " << scales[j] << " ***");

            CGammaRateConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shapes[i], scales[j], 10, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 100, samples);

            TMeanAccumulator relativeError;

            for (std::size_t k = 0; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));

                double expectedVariance;
                BOOST_TEST_REQUIRE(filter.marginalLikelihoodVarianceForTest(expectedVariance));

                LOG_TRACE(<< "marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                          << ", expectedVariance = " << expectedVariance);

                // The error is mainly due to the truncation in the
                // integration range used to compute the expected mean.
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedVariance,
                                             filter.marginalLikelihoodVariance(),
                                             0.01 * expectedVariance);

                relativeError.add(std::fabs(expectedVariance -
                                            filter.marginalLikelihoodVariance()) /
                                  expectedVariance);
            }

            LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(relativeError) < 0.0012);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihood) {
    // We're going to test two properties of the sampling:
    //   1) That the sample mean is equal to the marginal
    //      likelihood mean.
    //   2) That the sample percentiles match the distribution
    //      percentiles.
    // I want to cross check these with the implementations of the
    // jointLogMarginalLikelihood and minusLogJointCdf so use these
    // to compute the mean and percentiles.

    const double shape = 5.0;
    const double scale = 1.0;

    const double eps = 1e-3;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 50, samples);

    CGammaRateConjugate filter(makePrior());

    TDouble1Vec sampled;

    TMeanVarAccumulator sampleMeanVar;

    for (std::size_t i = 0; i < 3; ++i) {
        sampleMeanVar.add(samples[i]);

        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(10, sampled);

        TMeanVarAccumulator sampledMeanVar;
        sampledMeanVar = std::for_each(sampled.begin(), sampled.end(), sampledMeanVar);

        BOOST_REQUIRE_EQUAL(i + 1, sampled.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(sampleMeanVar),
                                     maths::CBasicStatistics::mean(sampledMeanVar), eps);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::variance(sampleMeanVar),
                                     maths::CBasicStatistics::variance(sampledMeanVar), eps);
    }

    TMeanAccumulator meanVarError;

    std::size_t numberSampled = 20;
    for (std::size_t i = 3; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(numberSampled, sampled);
        BOOST_REQUIRE_EQUAL(numberSampled, sampled.size());

        TMeanVarAccumulator sampledMoments;
        sampledMoments = std::for_each(sampled.begin(), sampled.end(), sampledMoments);

        LOG_TRACE(<< "expectedMean = " << filter.marginalLikelihoodMean()
                  << ", sampledMean = " << maths::CBasicStatistics::mean(sampledMoments));
        LOG_TRACE(<< "expectedVar = " << filter.marginalLikelihoodVariance() << ", sampledVar = "
                  << maths::CBasicStatistics::variance(sampledMoments));

        BOOST_REQUIRE_CLOSE_ABSOLUTE(filter.marginalLikelihoodMean(),
                                     maths::CBasicStatistics::mean(sampledMoments), 1e-8);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(filter.marginalLikelihoodVariance(),
                                     maths::CBasicStatistics::variance(sampledMoments),
                                     0.25 * filter.marginalLikelihoodVariance());
        meanVarError.add(std::fabs(filter.marginalLikelihoodVariance() -
                                   maths::CBasicStatistics::variance(sampledMoments)) /
                         filter.marginalLikelihoodVariance());

        std::sort(sampled.begin(), sampled.end());
        for (std::size_t j = 1; j < sampled.size(); ++j) {
            double q = 100.0 * static_cast<double>(j) / static_cast<double>(numberSampled);

            double expectedQuantile;
            BOOST_TEST_REQUIRE(filter.marginalLikelihoodQuantileForTest(q, eps, expectedQuantile));

            LOG_TRACE(<< "quantile = " << q << ", x_quantile = " << expectedQuantile << ", quantile range = ["
                      << sampled[j - 1u] << "," << sampled[j] << "]");

            BOOST_TEST_REQUIRE(expectedQuantile >= sampled[j - 1u]);
            BOOST_TEST_REQUIRE(expectedQuantile <= sampled[j]);
        }
    }

    LOG_DEBUG(<< "mean variance error = " << maths::CBasicStatistics::mean(meanVarError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanVarError) < 0.025);
}

BOOST_AUTO_TEST_CASE(testCdf) {
    // Test error cases.
    //
    // Test some invariants:
    //   "cdf" + "cdf complement" = 1
    //    cdf x for x < 0 = 1
    //    cdf complement x for x < 0 = 0

    double shape = 5.0;
    double scale = 0.5;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 100, samples);

    CGammaRateConjugate filter(makePrior());

    filter.addSamples(samples);

    double lowerBound;
    double upperBound;
    BOOST_TEST_REQUIRE(!filter.minusLogJointCdf(TDouble1Vec(), lowerBound, upperBound));
    BOOST_TEST_REQUIRE(!filter.minusLogJointCdfComplement(TDouble1Vec(), lowerBound, upperBound));

    BOOST_TEST_REQUIRE(filter.minusLogJointCdf(TDouble1Vec(1, -1.0), lowerBound, upperBound));
    double f = (lowerBound + upperBound) / 2.0;
    BOOST_TEST_REQUIRE(filter.minusLogJointCdfComplement(TDouble1Vec(1, -1.0),
                                                         lowerBound, upperBound));
    double fComplement = (lowerBound + upperBound) / 2.0;
    LOG_DEBUG(<< "log(F(x)) = " << -f << ", log(1 - F(x)) = " << fComplement);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(std::log(std::numeric_limits<double>::min()), -f, 1e-10);
    BOOST_REQUIRE_EQUAL(1.0, std::exp(-fComplement));

    for (std::size_t i = 1; i < 500; ++i) {
        double x = static_cast<double>(i) / 5.0;

        BOOST_TEST_REQUIRE(filter.minusLogJointCdf(TDouble1Vec(1, x), lowerBound, upperBound));
        f = (lowerBound + upperBound) / 2.0;
        BOOST_TEST_REQUIRE(filter.minusLogJointCdfComplement(
            TDouble1Vec(1, x), lowerBound, upperBound));
        fComplement = (lowerBound + upperBound) / 2.0;

        LOG_TRACE(<< "log(F(x)) = " << (f == 0.0 ? f : -f) << ", log(1 - F(x)) = "
                  << (fComplement == 0.0 ? fComplement : -fComplement));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, std::exp(-f) + std::exp(-fComplement), 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySamples) {
    // We test that the probability of less likely samples calculation
    // agrees with the chance of seeing a sample with lower marginal
    // likelihood, up to the sampling error.
    //
    // We also check that the tail calculation attributes samples to
    // the appropriate tail of the distribution.

    const double shapes[] = {0.4, 10.0, 200.0};
    const double scales[] = {0.1, 5.0, 50.0};
    const double vs[] = {0.5, 1.0, 2.0};

    test::CRandomNumbers rng;

    TMeanAccumulator meanError;

    for (size_t i = 0; i < boost::size(shapes); ++i) {
        for (size_t j = 0; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "shape = " << shapes[i] << ", scale = " << scales[j]);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 1000, samples);

            CGammaRateConjugate filter(makePrior());
            filter.addSamples(samples);

            double shape_ = filter.likelihoodShape();
            double rate_ = filter.likelihoodRate();

            TDoubleVec likelihoods;
            for (std::size_t k = 0; k < samples.size(); ++k) {
                double likelihood;
                filter.jointLogMarginalLikelihood(TDouble1Vec(1, samples[k]), likelihood);
                likelihoods.push_back(likelihood);
            }
            std::sort(likelihoods.begin(), likelihoods.end());

            boost::math::gamma_distribution<> gamma(shape_, 1.0 / rate_);
            for (std::size_t k = 1; k < 10; ++k) {
                double x = boost::math::quantile(gamma, static_cast<double>(k) / 10.0);

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

    const double shapes[] = {0.4, 10.0, 200.0};
    const double scales[] = {0.1, 5.0, 50.0};

    const double threshold = 0.02;

    const unsigned int anomalyTimes[] = {30u, 120u, 300u, 420u};
    const double anomalies[] = {4.0, 5.0, 10.0, 15.0, 0.0};

    test::CRandomNumbers rng;

    unsigned int test = 0;

    //std::ofstream file;
    //file.open("results.m");

    double totalFalsePositiveRate = 0.0;
    std::size_t totalPositives[] = {0u, 0u, 0u};

    for (size_t i = 0; i < boost::size(shapes); ++i) {
        for (size_t j = 0; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "shape = " << shapes[i] << ", scale = " << scales[j]);

            boost::math::gamma_distribution<> gamma(shapes[i], scales[j]);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 500, samples);

            for (size_t k = 0; k < boost::size(decayRates); ++k) {
                CGammaRateConjugate filter(
                    makePrior(maths_t::E_ContinuousData, 0.0, decayRates[k]));

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
                         boost::math::standard_deviation(gamma));

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

                // Should detect at least the two big anomalies.
                BOOST_TEST_REQUIRE(positives.size() >= 2);

                totalPositives[k] += positives.size();
            }
        }
    }

    totalFalsePositiveRate /= static_cast<double>(test);

    LOG_DEBUG(<< "totalFalsePositiveRate = " << totalFalsePositiveRate);
    for (size_t i = 0; i < boost::size(totalPositives); ++i) {
        LOG_DEBUG(<< "positives = " << totalPositives[i]);

        BOOST_TEST_REQUIRE(totalPositives[i] >= 24);
    }

    // Total false alarm rate should be less than 0.11%.
    BOOST_TEST_REQUIRE(totalFalsePositiveRate < 0.0011);
}

BOOST_AUTO_TEST_CASE(testOffset) {
    // The idea of this test is to check that the offset correctly cancels
    // out a translation applied to a gamma distributed data set.

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};
    const double offsets[] = {-0.5, 0.5};
    const double decayRates[] = {0.0, 0.001, 0.01};

    const double shape = 5.0;
    const double scale = 1.0;

    const double eps = 1e-8;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 100, samples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        for (size_t j = 0; j < boost::size(offsets); ++j) {
            for (size_t k = 0; k < boost::size(decayRates); ++k) {
                CGammaRateConjugate filter1(
                    makePrior(dataTypes[i], offsets[j], decayRates[k]));
                CGammaRateConjugate filter2(makePrior(dataTypes[i], 0.0, decayRates[k]));

                for (std::size_t l = 0; l < samples.size(); ++l) {
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
                    filter1.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, offsetSampleVec, lowerBound1, upperBound1);
                    BOOST_REQUIRE_EQUAL(lowerBound1, upperBound1);
                    double probability1 = (lowerBound1 + upperBound1) / 2.0;

                    double likelihood2;
                    filter2.jointLogMarginalLikelihood(sample, likelihood2);
                    double lowerBound2, upperBound2;
                    filter2.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, sample, lowerBound2, upperBound2);
                    BOOST_REQUIRE_EQUAL(lowerBound2, upperBound2);
                    double probability2 = (lowerBound2 + upperBound2) / 2.0;

                    BOOST_REQUIRE_CLOSE_ABSOLUTE(likelihood1, likelihood2, eps);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(probability1, probability2, eps);
                }

                using TEqual = maths::CEqualWithTolerance<double>;
                TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, eps);
                BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testIntegerData) {
    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // The idea of this test is to check that the inferred model agrees in the
    // limit (large n) with the model inferred from such data.

    const double shapes[] = {0.2, 1.0, 4.5};
    const double scales[] = {0.2, 1.0, 4.5};
    const std::size_t nSamples = 25000;

    for (size_t i = 0; i < boost::size(shapes); ++i) {
        for (size_t j = 0; j < boost::size(scales); ++j) {
            test::CRandomNumbers rng;

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], nSamples, samples);

            TDoubleVec uniform;
            rng.generateUniformSamples(0.0, 1.0, nSamples, uniform);

            CGammaRateConjugate filter1(makePrior(maths_t::E_IntegerData, 0.1));
            CGammaRateConjugate filter2(makePrior(maths_t::E_ContinuousData, 0.1));

            for (std::size_t k = 0; k < nSamples; ++k) {
                double x = std::floor(samples[k]);

                TDouble1Vec sample(1, x);
                filter1.addSamples(sample);

                sample[0] += uniform[k];
                filter2.addSamples(sample);
            }

            using TEqual = maths::CEqualWithTolerance<double>;
            TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.02);
            BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
        }
    }

    TMeanAccumulator meanError;

    for (size_t i = 0; i < boost::size(shapes); ++i) {
        for (size_t j = 0; j < boost::size(scales); ++j) {
            test::CRandomNumbers rng;

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shapes[i], scales[j], 100, seedSamples);
            CGammaRateConjugate filter1(makePrior(maths_t::E_IntegerData, 0.1));
            filter1.addSamples(seedSamples);

            CGammaRateConjugate filter2 = filter1;
            filter2.dataType(maths_t::E_ContinuousData);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], nSamples, samples);

            TDoubleVec uniform;
            rng.generateUniformSamples(0.0, 1.0, nSamples, uniform);

            TMeanAccumulator meanProbability1;
            TMeanAccumulator meanProbability2;

            for (std::size_t k = 0; k < nSamples; ++k) {
                double x = std::floor(samples[k]);

                TDouble1Vec sample(1, x);

                double l1, u1;
                BOOST_TEST_REQUIRE(filter1.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, sample, l1, u1));
                BOOST_REQUIRE_EQUAL(l1, u1);
                double p1 = (l1 + u1) / 2.0;
                meanProbability1.add(p1);

                sample[0] += uniform[k];
                double l2, u2;
                BOOST_TEST_REQUIRE(filter2.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, sample, l2, u2));
                BOOST_REQUIRE_EQUAL(l2, u2);
                double p2 = (l2 + u2) / 2.0;
                meanProbability2.add(p2);
            }

            double p1 = maths::CBasicStatistics::mean(meanProbability1);
            double p2 = maths::CBasicStatistics::mean(meanProbability2);
            LOG_DEBUG(<< "shape = " << shapes[i] << ", rate = " << scales[j]
                      << ", p1 = " << p1 << ", p2 = " << p2);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.15 * p1);
            meanError.add(fabs(p1 - p2));
        }
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.016);
}

BOOST_AUTO_TEST_CASE(testLowVariationData) {
    {
        CGammaRateConjugate filter(makePrior(maths_t::E_IntegerData));
        for (std::size_t i = 0; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG(<< "68% confidence interval " << core::CContainerPrinter::print(interval)
                  << ", approximate variance = " << sigma * sigma);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(12.0, 1.0 / (sigma * sigma), 0.5);
    }
    {
        CGammaRateConjugate filter(makePrior(maths_t::E_ContinuousData));
        for (std::size_t i = 0; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG(<< "68% confidence interval " << core::CContainerPrinter::print(interval)
                  << ", approximate s.t.d. = " << sigma);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(1e-4, sigma / 430.5, 5e-6);
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 3.0, 500, samples);

    maths::CGammaRateConjugate origFilter(makePrior(maths_t::E_ContinuousData, 0.1));
    for (std::size_t i = 0; i < samples.size(); ++i) {
        origFilter.addSamples({samples[i]}, maths_t::CUnitWeights::SINGLE_UNIT);
    }
    double decayRate = origFilter.decayRate();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Gamma rate conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, decayRate + 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    maths::CGammaRateConjugate restoredFilter(params, traverser);

    uint64_t checksum = origFilter.checksum();
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

BOOST_AUTO_TEST_CASE(testVarianceScale) {
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
    // gamma distribution (uniform law of large numbers), which is
    // just the differential entropy of a scaled gamma R.V.
    //
    // Finally, we test update with scaled samples produces the
    // correct posterior.

    TWeightFunc weightsFuncs[]{
        static_cast<TWeightFunc>(maths_t::seasonalVarianceScaleWeight),
        static_cast<TWeightFunc>(maths_t::countVarianceScaleWeight)};

    for (std::size_t s = 0; s < boost::size(weightsFuncs); ++s) {
        const double shape = 3.0;
        const double scale = 3.0;

        const double varianceScales[] = {0.20, 0.50, 0.75, 1.50, 2.00, 5.00};

        test::CRandomNumbers rng;

        LOG_DEBUG(<< "");
        LOG_DEBUG(<< "****** probabilityOfLessLikelySamples ******");

        {
            const double percentiles[] = {10.0, 20.0, 30.0, 40.0, 50.0,
                                          60.0, 70.0, 80.0, 90.0};

            const std::size_t nSamples = 1000;
            const std::size_t nScaledSamples = 10000;

            TDoubleVec samples;
            rng.generateGammaSamples(shape, scale, nSamples, samples);

            CGammaRateConjugate filter(makePrior());
            filter.addSamples(samples);

            TDoubleVec expectedPercentileErrors;
            double expectedTotalError = 0.0;

            {
                TDoubleVec unscaledSamples;
                rng.generateGammaSamples(shape, scale, nScaledSamples, unscaledSamples);

                TDoubleVec probabilities;
                probabilities.reserve(nScaledSamples);
                for (std::size_t i = 0; i < unscaledSamples.size(); ++i) {
                    TDouble1Vec sample(1, unscaledSamples[i]);

                    double lowerBound, upperBound;
                    BOOST_TEST_REQUIRE(filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, sample, lowerBound, upperBound));
                    BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                    double probability = (lowerBound + upperBound) / 2.0;
                    probabilities.push_back(probability);
                }
                std::sort(probabilities.begin(), probabilities.end());

                for (size_t i = 0; i < boost::size(percentiles); ++i) {
                    std::size_t index = static_cast<std::size_t>(
                        static_cast<double>(nScaledSamples) * percentiles[i] / 100.0);
                    double error = fabs(probabilities[index] - percentiles[i] / 100.0);
                    expectedPercentileErrors.push_back(error);
                    expectedTotalError += error;
                }
            }

            for (size_t i = 0; i < boost::size(varianceScales); ++i) {
                LOG_DEBUG(<< "**** variance scale = " << varianceScales[i] << " ****");

                double scaledShape = shape / varianceScales[i];
                double ss = varianceScales[i] * scale;
                boost::math::gamma_distribution<> gamma(scaledShape, ss);
                LOG_DEBUG(<< "mean = " << boost::math::mean(gamma)
                          << ", variance = " << boost::math::variance(gamma));

                TDoubleVec scaledSamples;
                rng.generateGammaSamples(scaledShape, ss, nScaledSamples, scaledSamples);

                TDoubleVec probabilities;
                probabilities.reserve(nScaledSamples);
                for (std::size_t j = 0; j < scaledSamples.size(); ++j) {
                    double lowerBound, upperBound;
                    maths_t::ETail tail;
                    BOOST_TEST_REQUIRE(filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, {scaledSamples[j]},
                        {weightsFuncs[s](varianceScales[i])}, lowerBound, upperBound, tail));
                    BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                    double probability = (lowerBound + upperBound) / 2.0;
                    probabilities.push_back(probability);
                }
                std::sort(probabilities.begin(), probabilities.end());

                double totalError = 0.0;
                for (size_t j = 0; j < boost::size(percentiles); ++j) {
                    std::size_t index = static_cast<std::size_t>(
                        static_cast<double>(nScaledSamples) * percentiles[j] / 100.0);
                    double error = fabs(probabilities[index] - percentiles[j] / 100.0);
                    totalError += error;
                    double errorThreshold = 0.017 + expectedPercentileErrors[j];

                    LOG_TRACE(<< "percentile = " << percentiles[j] << ", probability = "
                              << probabilities[index] << ", error = " << error
                              << ", error threshold = " << errorThreshold);

                    BOOST_TEST_REQUIRE(error < errorThreshold);
                }

                double totalErrorThreshold = 0.1 + expectedTotalError;

                LOG_DEBUG(<< "total error = " << totalError
                          << ", totalError threshold = " << totalErrorThreshold);

                BOOST_TEST_REQUIRE(totalError < totalErrorThreshold);
            }
        }

        LOG_DEBUG(<< "");
        LOG_DEBUG(<< "****** jointLogMarginalLikelihood ******");

        for (size_t i = 0; i < boost::size(varianceScales); ++i) {
            LOG_DEBUG(<< "**** variance scale = " << varianceScales[i] << " ****");

            double scaledShape = shape / varianceScales[i];
            double scaledScale = varianceScales[i] * scale;
            boost::math::gamma_distribution<> gamma(scaledShape, scaledScale);
            LOG_DEBUG(<< "mean = " << boost::math::mean(gamma)
                      << ", variance = " << boost::math::variance(gamma));
            double expectedDifferentialEntropy = maths::CTools::differentialEntropy(gamma);

            CGammaRateConjugate filter(makePrior());

            double differentialEntropy = 0.0;

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shape, scale, 150, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec scaledSamples;
            rng.generateGammaSamples(scaledShape, scaledScale, 50000, scaledSamples);

            for (std::size_t j = 0; j < scaledSamples.size(); ++j) {
                double logLikelihood = 0.0;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    filter.jointLogMarginalLikelihood(
                                        {scaledSamples[j]},
                                        {weightsFuncs[s](varianceScales[i])}, logLikelihood));
                differentialEntropy -= logLikelihood;
            }

            differentialEntropy /= static_cast<double>(scaledSamples.size());

            LOG_DEBUG(<< "differentialEntropy = " << differentialEntropy
                      << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedDifferentialEntropy,
                                         differentialEntropy, 0.05);
        }
    }

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double maximumMeanError[] = {0.08, 0.11};
    const double maximumVarianceError[] = {1.0, 0.2};
    const double maximumMeanMeanError[] = {0.01, 0.01};
    const double maximumMeanVarianceError[] = {0.08, 0.05};

    for (std::size_t s = 0; s < boost::size(weightsFuncs); ++s) {
        for (std::size_t t = 0; t < boost::size(dataTypes); ++t) {
            const double shapes[] = {1.0,    10.0,     100.0,
                                     1000.0, 100000.0, 1000000.0};
            const double rates[] = {1.0,    10.0,     100.0,
                                    1000.0, 100000.0, 1000000.0};
            const double varianceScales[] = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};

            TDoubleVec samples;
            maths_t::TDoubleWeightsAry1Vec weights;

            test::CRandomNumbers rng;

            TMeanAccumulator meanMeanError;
            TMeanAccumulator meanVarianceError;

            for (std::size_t i = 0; i < boost::size(shapes); ++i) {
                for (std::size_t j = 0; j < boost::size(rates); ++j) {
                    double shape = shapes[i];
                    double rate = rates[j];

                    // We purposely don't estimate true variance in this case.
                    if (shape < rate * rate * maths::MINIMUM_COEFFICIENT_OF_VARIATION) {
                        continue;
                    }

                    LOG_TRACE(<< "****** shape = " << shape << ", rate = " << rate << " ******");

                    double mean = shape / rate;
                    double variance = mean / rate;

                    for (std::size_t k = 0; k < boost::size(varianceScales); ++k) {
                        double scale = varianceScales[k];
                        LOG_TRACE(<< "*** scale = " << scale << " ***");

                        double scaledShape = shape / scale;
                        double scaledRate = rate / scale;
                        LOG_TRACE(<< "scaled shape = " << scaledShape
                                  << ", scaled rate = " << scaledRate);

                        TMeanAccumulator meanError;
                        TMeanAccumulator varianceError;
                        for (unsigned int test = 0; test < 5; ++test) {
                            CGammaRateConjugate filter(makePrior(dataTypes[t]));

                            rng.generateGammaSamples(shape, 1.0 / rate, 200, samples);
                            weights.clear();
                            weights.resize(samples.size(), maths_t::CUnitWeights::UNIT);
                            filter.addSamples(samples, weights);
                            rng.generateGammaSamples(scaledShape, 1.0 / scaledRate,
                                                     200, samples);
                            weights.clear();
                            weights.resize(samples.size(), weightsFuncs[s](scale));
                            filter.addSamples(samples, weights);

                            double estimatedMean = filter.likelihoodShape() /
                                                   filter.likelihoodRate();
                            double estimatedVariance = estimatedMean /
                                                       filter.likelihoodRate();
                            double dm = (dataTypes[t] == maths_t::E_IntegerData ? 0.5 : 0.0);
                            double dv = (dataTypes[t] == maths_t::E_IntegerData ? 1.0 / 12.0 : 0.0);
                            double trialMeanError = std::fabs(estimatedMean - (mean + dm)) /
                                                    std::max(1.0, mean + dm);
                            double trialVarianceError =
                                std::fabs(estimatedVariance - (variance + dv)) /
                                std::max(1.0, variance + dv);

                            LOG_TRACE(<< "trial mean error = " << trialMeanError);
                            LOG_TRACE(<< "trial variance error = " << trialVarianceError);

                            meanError.add(trialMeanError);
                            varianceError.add(trialVarianceError);
                        }

                        LOG_TRACE(<< "mean error = "
                                  << maths::CBasicStatistics::mean(meanError));
                        LOG_TRACE(<< "variance error = "
                                  << maths::CBasicStatistics::mean(varianceError));

                        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) <
                                           maximumMeanError[t]);
                        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceError) <
                                           maximumVarianceError[t]);

                        meanMeanError += meanError;
                        meanVarianceError += varianceError;
                    }
                }
            }

            LOG_DEBUG(<< "mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
            LOG_DEBUG(<< "mean variance error = "
                      << maths::CBasicStatistics::mean(meanVarianceError));

            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMeanError) <
                               maximumMeanMeanError[t]);
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanVarianceError) <
                               maximumMeanVarianceError[t]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testNegativeSample) {
    // Test that we recover roughly the same distribution after adjusting
    // the offset. The idea of this test is to run two priors side by side,
    // one with a large enough offset that it never needs to adjust the
    // offset and the other which will adjust and check that we get broadly
    // similar distributions at the end.

    const double shape = 4.0;
    const double scale = 1.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 100, samples);

    CGammaRateConjugate filter1(CGammaRateConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 0.0, 0.0, 0.2));
    CGammaRateConjugate filter2(CGammaRateConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 1.2586, 0.0, 0.2));

    filter1.addSamples(samples);
    filter2.addSamples(samples);

    TDouble1Vec negative(1, -0.29);
    filter1.addSamples(negative);
    filter2.addSamples(negative);

    BOOST_REQUIRE_EQUAL(filter1.numberSamples(), filter2.numberSamples());

    using TEqual = maths::CEqualWithTolerance<double>;
    TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.1);
    BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
}

BOOST_AUTO_TEST_SUITE_END()
