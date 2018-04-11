/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CGammaRateConjugateTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/distributions/gamma.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

using namespace ml;
using namespace handy_typedefs;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using CGammaRateConjugate = CPriorTestInterfaceMixin<maths::CGammaRateConjugate>;

CGammaRateConjugate
makePrior(maths_t::EDataType dataType = maths_t::E_ContinuousData, const double& offset = 0.0, const double& decayRate = 0.0) {
    return CGammaRateConjugate::nonInformativePrior(dataType, offset, decayRate, 0.0);
}
}

void CGammaRateConjugateTest::testMultipleUpdate() {
    LOG_DEBUG(<< "+-----------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testMultipleUpdate  |");
    LOG_DEBUG(<< "+-----------------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double shape = 2.0;
    const double scale = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 100, samples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CGammaRateConjugate filter1(makePrior(dataTypes[i]));
        CGammaRateConjugate filter2(filter1);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter1.addSamples(TDouble1Vec(1, samples[j]));
        }
        filter2.addSamples(samples);

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 1e-3);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }

    // Test with variance scale.

    TDoubleVec scaledSamples;
    rng.generateGammaSamples(shape / 2.0, 2.0 * scale, 100, scaledSamples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CGammaRateConjugate filter1(makePrior(dataTypes[i]));
        filter1.addSamples(samples);
        CGammaRateConjugate filter2(filter1);

        maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
        for (std::size_t j = 0u; j < scaledSamples.size(); ++j) {
            filter1.addSamples(weightStyle, TDouble1Vec(1, scaledSamples[j]), TDouble4Vec1Vec(1, TDouble4Vec(1, 2.0)));
        }
        filter2.addSamples(weightStyle, scaledSamples, TDouble4Vec1Vec(scaledSamples.size(), TDouble4Vec(1, 2.0)));

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.03);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }

    // Test the count weight is equivalent to adding repeated samples.

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CGammaRateConjugate filter1(makePrior(dataTypes[i]));
        CGammaRateConjugate filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;

        for (std::size_t j = 0u; j < count; ++j) {
            filter1.addSamples(TDouble1Vec(1, x));
        }
        filter2.addSamples(maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight),
                           TDouble1Vec(1, x),
                           TDouble4Vec1Vec(1, TDouble4Vec(1, static_cast<double>(count))));

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.01);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }
}

void CGammaRateConjugateTest::testPropagation() {
    LOG_DEBUG(<< "+--------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testPropagation  |");
    LOG_DEBUG(<< "+--------------------------------------------+");

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

    LOG_DEBUG(<< "shape = " << shape << ", rate = " << rate << ", propagatedShape = " << propagatedShape
              << ", propagatedRate = " << propagatedRate);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(shape, propagatedShape, eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(rate, propagatedRate, eps);
}

void CGammaRateConjugateTest::testShapeEstimation() {
    LOG_DEBUG(<< "+------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testShapeEstimation  |");
    LOG_DEBUG(<< "+------------------------------------------------+");

    // The idea here is to check that the likelihood shape estimate converges
    // to the correct value for a range of distribution parameters. We do not
    // use any explicit bounds on the convergence rates so simply check that
    // we do get closer as the number of samples increases.

    const double decayRates[] = {0.0, 0.001, 0.01};

    for (size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        double tests = 0.0;
        double errorIncreased = 0.0;

        for (unsigned int test = 0u; test < 100u; ++test) {
            double shape = 0.5 * (test + 1.0);
            double scale = 2.0;

            TDoubleVec samples;
            rng.generateGammaSamples(shape, scale, 5050, samples);

            using TGammaRateConjugateVec = std::vector<CGammaRateConjugate>;

            unsigned int nAggregate = 50u;
            TGammaRateConjugateVec filters(nAggregate, makePrior(maths_t::E_ContinuousData, 0.0, decayRates[i]));

            double previousError = std::numeric_limits<double>::max();
            double averageShape = 0.0;

            for (std::size_t j = 0u; j < samples.size() / nAggregate; ++j) {
                double error = 0.0;
                averageShape = 0.0;
                for (std::size_t k = 0u; k < nAggregate; ++k) {
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

            LOG_DEBUG(<< "shape = " << shape << ", averageShape = " << averageShape);

            // Average error after 100 updates should be less than 8%.
            CPPUNIT_ASSERT_DOUBLES_EQUAL(shape, averageShape, 0.08 * shape);
        }

        // Error should only increase in at most 7% of measurements.
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, errorIncreased, 0.07 * tests);
    }
}

void CGammaRateConjugateTest::testRateEstimation() {
    LOG_DEBUG(<< "+-----------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testRateEstimation  |");
    LOG_DEBUG(<< "+-----------------------------------------------+");

    // We are going to test that we correctly estimate a distribution
    // for the rate of the gamma process by checking that the true
    // rate of a gamma process lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 100u;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0};

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

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval = filter.confidenceIntervalRate(testIntervals[j]);

                if (rate < confidenceInterval.first || rate > confidenceInterval.second) {
                    ++errors[j];
                }
            }
        }

        for (size_t j = 0; j < boost::size(testIntervals); ++j) {
            // The number of errors should be inside the percentile bounds.
            unsigned int maximumErrors = static_cast<unsigned int>(std::ceil((1.0 - testIntervals[j] / 100.0) * nTests));

            LOG_DEBUG(<< "errors = " << errors[j] << ", maximumErrors = " << maximumErrors);

            CPPUNIT_ASSERT(errors[j] <= maximumErrors + 2);
        }
    }
}

void CGammaRateConjugateTest::testMarginalLikelihood() {
    LOG_DEBUG(<< "+---------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testMarginalLikelihood  |");
    LOG_DEBUG(<< "+---------------------------------------------------+");

    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] = {maths_t::E_ContinuousData, maths_t::E_IntegerData};
    for (std::size_t t = 0u; t < boost::size(dataTypes); ++t) {
        CGammaRateConjugate filter(makePrior());

        const double shape = 1.0;
        const double scale = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateGammaSamples(shape, scale, 200, samples);
        filter.addSamples(samples);

        maths_t::ESampleWeightStyle weightStyles[] = {
            maths_t::E_SampleCountWeight, maths_t::E_SampleWinsorisationWeight, maths_t::E_SampleCountWeight};
        double weights[] = {0.1, 1.0, 10.0};

        for (std::size_t i = 0u; i < boost::size(weightStyles); ++i) {
            for (std::size_t j = 0u; j < boost::size(weights); ++j) {
                double lb, ub;
                filter.minusLogJointCdf(maths_t::TWeightStyleVec(1, weightStyles[i]),
                                        TDouble1Vec(1, 1000.0),
                                        TDouble4Vec1Vec(1, TDouble4Vec(1, weights[j])),
                                        lb,
                                        ub);
                LOG_DEBUG(<< "-log(c.d.f) = " << (lb + ub) / 2.0);
                CPPUNIT_ASSERT(lb >= 0.0);
                CPPUNIT_ASSERT(ub >= 0.0);
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
            CGammaRateConjugate filter(makePrior(maths_t::E_ContinuousData, 0.0, decayRates[j]));

            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                filter.propagateForwardsByTime(1.0);
            }

            // We'll check that the p.d.f. is close to the derivative
            // of the c.d.f. at a range of deltas from the true mean.

            const double eps = 1e-4;
            double deltas[] = {-2.0, -1.6, -1.2, -0.8, -0.4, -0.2, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};

            for (size_t k = 0; k < boost::size(deltas); ++k) {
                double x = mean + deltas[k] * std::sqrt(variance);
                TDouble1Vec sample(1, x);

                LOG_DEBUG(<< "number = " << numberSamples[i] << ", sample = " << sample[0]);

                double logLikelihood = 0.0;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter.jointLogMarginalLikelihood(sample, logLikelihood));
                double pdf = std::exp(logLikelihood);

                double lowerBound = 0.0, upperBound = 0.0;
                sample[0] -= eps;
                CPPUNIT_ASSERT(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                double minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtMinusEps = std::exp(-minusLogCdf);
                CPPUNIT_ASSERT(minusLogCdf >= 0.0);

                sample[0] += 2.0 * eps;
                CPPUNIT_ASSERT(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtPlusEps = std::exp(-minusLogCdf);
                CPPUNIT_ASSERT(minusLogCdf >= 0.0);

                double dcdfdx = (cdfAtPlusEps - cdfAtMinusEps) / 2.0 / eps;

                LOG_DEBUG(<< "pdf(x) = " << pdf << ", d(cdf)/dx = " << dcdfdx);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(pdf, dcdfdx, tolerances[i]);
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
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            TDouble1Vec sample(1, samples[i]);
            filter.addSamples(sample);
            double logLikelihood = 0.0;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter.jointLogMarginalLikelihood(sample, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(samples.size());

        LOG_DEBUG(<< "differentialEntropy = " << differentialEntropy << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 0.0025);
    }

    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};

    CGammaRateConjugate filter(makePrior());
    TDoubleVec samples;
    rng.generateGammaSamples(shape, scale, 1000, samples);
    filter.addSamples(samples);

    const double percentages[] = {5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 95.0};

    {
        // Test that marginal likelihood confidence intervals are
        // what we'd expect for various variance scales.

        TMeanAccumulator error;
        for (std::size_t i = 0u; i < boost::size(percentages); ++i) {
            double q1, q2;
            filter.marginalLikelihoodQuantileForTest(50.0 - percentages[i] / 2.0, 1e-3, q1);
            filter.marginalLikelihoodQuantileForTest(50.0 + percentages[i] / 2.0, 1e-3, q2);
            TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(percentages[i]);
            LOG_DEBUG(<< "[q1, q2] = [" << q1 << ", " << q2 << "]"
                      << ", interval = " << core::CContainerPrinter::print(interval));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, 0.02);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.02);
            error.add(std::fabs(interval.first - q1));
            error.add(std::fabs(interval.second - q2));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 4e-3);
    }
    {
        maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
        TDouble4Vec weight(1, 1.0);
        TMeanAccumulator totalError;
        for (std::size_t i = 0u; i < boost::size(varianceScales); ++i) {
            TMeanAccumulator error;
            double vs = varianceScales[i];
            weight[0] = vs;
            LOG_DEBUG(<< "*** vs = " << vs << " ***");
            for (std::size_t j = 0u; j < boost::size(percentages); ++j) {
                boost::math::gamma_distribution<> scaledGamma(shape / vs, vs * scale);
                double q1 = boost::math::quantile(scaledGamma, (50.0 - percentages[j] / 2.0) / 100.0);
                double q2 = boost::math::quantile(scaledGamma, (50.0 + percentages[j] / 2.0) / 100.0);
                TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(percentages[j], weightStyle, weight);
                LOG_DEBUG(<< "[q1, q2] = [" << q1 << ", " << q2 << "]"
                          << ", interval = " << core::CContainerPrinter::print(interval));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, 0.4);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.4);
                error.add(std::fabs(interval.first - q1));
                error.add(std::fabs(interval.second - q2));
            }
            LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.09);
            totalError += error;
        }
        LOG_DEBUG(<< "totalError = " << maths::CBasicStatistics::mean(totalError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(totalError) < 0.042);
    }
}

void CGammaRateConjugateTest::testMarginalLikelihoodMean() {
    LOG_DEBUG(<< "+-------------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testMarginalLikelihoodMean  |");
    LOG_DEBUG(<< "+-------------------------------------------------------+");

    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    const double shapes[] = {5.0, 20.0, 40.0};
    const double scales[] = {1.0, 10.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(shapes); ++i) {
        for (std::size_t j = 0u; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "*** shape = " << shapes[i] << ", scale = " << scales[j] << " ***");

            CGammaRateConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shapes[i], scales[j], 3, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 100, samples);

            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));

                double expectedMean;
                CPPUNIT_ASSERT(filter.marginalLikelihoodMeanForTest(expectedMean));

                if (k % 10 == 0) {
                    LOG_DEBUG(<< "marginalLikelihoodMean = " << filter.marginalLikelihoodMean() << ", expectedMean = " << expectedMean);
                }

                // The error is mainly due to the truncation in the
                // integration range used to compute the expected mean.
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, filter.marginalLikelihoodMean(), 1e-3 * expectedMean);
            }
        }
    }
}

void CGammaRateConjugateTest::testMarginalLikelihoodMode() {
    LOG_DEBUG(<< "+-------------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testMarginalLikelihoodMode  |");
    LOG_DEBUG(<< "+-------------------------------------------------------+");

    // Test that the marginal likelihood mode is what we'd expect
    // with variances variance scales.

    const double shapes[] = {5.0, 20.0, 40.0};
    const double scales[] = {1.0, 10.0, 20.0};
    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(shapes); ++i) {
        for (std::size_t j = 0u; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "*** shape = " << shapes[i] << ", scale = " << scales[j] << " ***");

            CGammaRateConjugate filter(makePrior());

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 1000, samples);
            filter.addSamples(samples);

            TMeanAccumulator relativeError;
            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
            TDouble4Vec weight(1, 1.0);
            for (std::size_t k = 0u; k < boost::size(varianceScales); ++k) {
                double vs = varianceScales[k];
                weight[0] = vs;
                boost::math::gamma_distribution<> scaledGamma(shapes[i] / vs, vs * scales[j]);
                double expectedMode = boost::math::mode(scaledGamma);
                LOG_DEBUG(<< "marginalLikelihoodMode = " << filter.marginalLikelihoodMode(weightStyle, weight)
                          << ", expectedMode = " << expectedMode);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode, filter.marginalLikelihoodMode(weightStyle, weight), 0.28 * expectedMode + 0.3);
                double error = std::fabs(filter.marginalLikelihoodMode(weightStyle, weight) - expectedMode);
                relativeError.add(error == 0.0 ? 0.0 : error / expectedMode);
            }
            LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 0.08);
        }
    }
}

void CGammaRateConjugateTest::testMarginalLikelihoodVariance() {
    LOG_DEBUG(<< "+-----------------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testMarginalLikelihoodVariance  |");
    LOG_DEBUG(<< "+-----------------------------------------------------------+");

    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    const double shapes[] = {5.0, 20.0, 40.0};
    const double scales[] = {1.0, 10.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(shapes); ++i) {
        for (std::size_t j = 0u; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "*** shape = " << shapes[i] << ", scale = " << scales[j] << " ***");

            CGammaRateConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shapes[i], scales[j], 10, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 100, samples);

            TMeanAccumulator relativeError;

            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));

                double expectedVariance;
                CPPUNIT_ASSERT(filter.marginalLikelihoodVarianceForTest(expectedVariance));

                if (k % 10 == 0) {
                    LOG_DEBUG(<< "marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                              << ", expectedVariance = " << expectedVariance);
                }

                // The error is mainly due to the truncation in the
                // integration range used to compute the expected mean.
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedVariance, filter.marginalLikelihoodVariance(), 0.01 * expectedVariance);

                relativeError.add(std::fabs(expectedVariance - filter.marginalLikelihoodVariance()) / expectedVariance);
            }

            LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 0.0012);
        }
    }
}

void CGammaRateConjugateTest::testSampleMarginalLikelihood() {
    LOG_DEBUG(<< "+---------------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG(<< "+---------------------------------------------------------+");

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

    for (std::size_t i = 0u; i < 3; ++i) {
        sampleMeanVar.add(samples[i]);

        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(10, sampled);

        TMeanVarAccumulator sampledMeanVar;
        sampledMeanVar = std::for_each(sampled.begin(), sampled.end(), sampledMeanVar);

        CPPUNIT_ASSERT_EQUAL(i + 1, sampled.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(sampleMeanVar), maths::CBasicStatistics::mean(sampledMeanVar), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            maths::CBasicStatistics::variance(sampleMeanVar), maths::CBasicStatistics::variance(sampledMeanVar), eps);
    }

    TMeanAccumulator meanVarError;

    std::size_t numberSampled = 20u;
    for (std::size_t i = 3u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(numberSampled, sampled);
        CPPUNIT_ASSERT_EQUAL(numberSampled, sampled.size());

        TMeanVarAccumulator sampledMoments;
        sampledMoments = std::for_each(sampled.begin(), sampled.end(), sampledMoments);

        LOG_DEBUG(<< "expectedMean = " << filter.marginalLikelihoodMean()
                  << ", sampledMean = " << maths::CBasicStatistics::mean(sampledMoments));
        LOG_DEBUG(<< "expectedVar = " << filter.marginalLikelihoodVariance()
                  << ", sampledVar = " << maths::CBasicStatistics::variance(sampledMoments));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodMean(), maths::CBasicStatistics::mean(sampledMoments), 1e-8);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodVariance(),
                                     maths::CBasicStatistics::variance(sampledMoments),
                                     0.25 * filter.marginalLikelihoodVariance());
        meanVarError.add(std::fabs(filter.marginalLikelihoodVariance() - maths::CBasicStatistics::variance(sampledMoments)) /
                         filter.marginalLikelihoodVariance());

        std::sort(sampled.begin(), sampled.end());
        for (std::size_t j = 1u; j < sampled.size(); ++j) {
            double q = 100.0 * static_cast<double>(j) / static_cast<double>(numberSampled);

            double expectedQuantile;
            CPPUNIT_ASSERT(filter.marginalLikelihoodQuantileForTest(q, eps, expectedQuantile));

            LOG_DEBUG(<< "quantile = " << q << ", x_quantile = " << expectedQuantile << ", quantile range = [" << sampled[j - 1u] << ","
                      << sampled[j] << "]");

            CPPUNIT_ASSERT(expectedQuantile >= sampled[j - 1u]);
            CPPUNIT_ASSERT(expectedQuantile <= sampled[j]);
        }
    }

    LOG_DEBUG(<< "mean variance error = " << maths::CBasicStatistics::mean(meanVarError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanVarError) < 0.025);
}

void CGammaRateConjugateTest::testCdf() {
    LOG_DEBUG(<< "+------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testCdf  |");
    LOG_DEBUG(<< "+------------------------------------+");

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

    maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountWeight);
    double lowerBound;
    double upperBound;
    CPPUNIT_ASSERT(!filter.minusLogJointCdf(TDouble1Vec(), lowerBound, upperBound));
    CPPUNIT_ASSERT(!filter.minusLogJointCdfComplement(TDouble1Vec(), lowerBound, upperBound));

    CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, -1.0), lowerBound, upperBound));
    double f = (lowerBound + upperBound) / 2.0;
    CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, -1.0), lowerBound, upperBound));
    double fComplement = (lowerBound + upperBound) / 2.0;
    LOG_DEBUG(<< "log(F(x)) = " << -f << ", log(1 - F(x)) = " << fComplement);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(std::numeric_limits<double>::min()), -f, 1e-10);
    CPPUNIT_ASSERT_EQUAL(1.0, std::exp(-fComplement));

    for (std::size_t i = 1u; i < 500; ++i) {
        double x = static_cast<double>(i) / 5.0;

        CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x), lowerBound, upperBound));
        f = (lowerBound + upperBound) / 2.0;
        CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lowerBound, upperBound));
        fComplement = (lowerBound + upperBound) / 2.0;

        LOG_DEBUG(<< "log(F(x)) = " << (f == 0.0 ? f : -f) << ", log(1 - F(x)) = " << (fComplement == 0.0 ? fComplement : -fComplement));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, std::exp(-f) + std::exp(-fComplement), 1e-10);
    }
}

void CGammaRateConjugateTest::testProbabilityOfLessLikelySamples() {
    LOG_DEBUG(<< "+---------------------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG(<< "+---------------------------------------------------------------+");

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
            for (std::size_t k = 0u; k < samples.size(); ++k) {
                double likelihood;
                filter.jointLogMarginalLikelihood(TDouble1Vec(1, samples[k]), likelihood);
                likelihoods.push_back(likelihood);
            }
            std::sort(likelihoods.begin(), likelihoods.end());

            boost::math::gamma_distribution<> gamma(shape_, 1.0 / rate_);
            for (std::size_t k = 1u; k < 10; ++k) {
                double x = boost::math::quantile(gamma, static_cast<double>(k) / 10.0);

                TDouble1Vec sample(1, x);
                double fx;
                filter.jointLogMarginalLikelihood(sample, fx);

                double px = static_cast<double>(std::lower_bound(likelihoods.begin(), likelihoods.end(), fx) - likelihoods.begin()) /
                            static_cast<double>(likelihoods.size());

                double lb, ub;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lb, ub);

                double ssd = std::sqrt(px * (1.0 - px) / static_cast<double>(samples.size()));

                LOG_DEBUG(<< "expected P(x) = " << px << ", actual P(x) = " << (lb + ub) / 2.0 << " sample sd = " << ssd);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(px, (lb + ub) / 2.0, 3.0 * ssd);

                meanError.add(std::fabs(px - (lb + ub) / 2.0));
            }

            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);

            for (std::size_t k = 0u; k < boost::size(vs); ++k) {
                double mode = filter.marginalLikelihoodMode(weightStyle, TDouble1Vec(1, vs[k]));
                double ss[] = {0.9 * mode, 1.1 * mode};

                LOG_DEBUG(<< "vs = " << vs[k] << ", mode = " << mode);

                double lb, ub;
                maths_t::ETail tail;

                {
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, weightStyle, TDouble1Vec(1, ss[0]), TDouble4Vec1Vec(1, TDouble4Vec(1, vs[k])), lb, ub, tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
                    if (mode > 0.0) {
                        filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                              weightStyle,
                                                              TDouble1Vec(ss, ss + 2),
                                                              TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                              lb,
                                                              ub,
                                                              tail);
                        CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                        filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedBelow,
                                                              weightStyle,
                                                              TDouble1Vec(ss, ss + 2),
                                                              TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                              lb,
                                                              ub,
                                                              tail);
                        CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
                        filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedAbove,
                                                              weightStyle,
                                                              TDouble1Vec(ss, ss + 2),
                                                              TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                              lb,
                                                              ub,
                                                              tail);
                        CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
                    }
                }
                if (mode > 0.0) {
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, weightStyle, TDouble1Vec(1, ss[1]), TDouble4Vec1Vec(1, TDouble4Vec(1, vs[k])), lb, ub, tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
                    filter.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, weightStyle, TDouble1Vec(ss, ss + 2), TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])), lb, ub, tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                    filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedBelow,
                                                          weightStyle,
                                                          TDouble1Vec(ss, ss + 2),
                                                          TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                          lb,
                                                          ub,
                                                          tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
                    filter.probabilityOfLessLikelySamples(maths_t::E_OneSidedAbove,
                                                          weightStyle,
                                                          TDouble1Vec(ss, ss + 2),
                                                          TDouble4Vec1Vec(2, TDouble4Vec(1, vs[k])),
                                                          lb,
                                                          ub,
                                                          tail);
                    CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
                }
            }
        }
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.01);
}

void CGammaRateConjugateTest::testAnomalyScore() {
    LOG_DEBUG(<< "+---------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testAnomalyScore  |");
    LOG_DEBUG(<< "+---------------------------------------------+");

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

    std::ofstream file;
    file.open("results.m");

    double totalFalsePositiveRate = 0.0;
    std::size_t totalPositives[] = {0u, 0u, 0u};

    for (size_t i = 0; i < boost::size(shapes); ++i) {
        for (size_t j = 0; j < boost::size(scales); ++j) {
            LOG_DEBUG(<< "shape = " << shapes[i] << ", scale = " << scales[j]);

            boost::math::gamma_distribution<> gamma(shapes[i], scales[j]);

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], 500, samples);

            for (size_t k = 0; k < boost::size(decayRates); ++k) {
                CGammaRateConjugate filter(makePrior(maths_t::E_ContinuousData, 0.0, decayRates[k]));

                ++test;

                std::ostringstream x;
                std::ostringstream scores;
                x << "x" << test << " = [";
                scores << "score" << test << " = [";

                TUIntVec candidateAnomalies;
                for (unsigned int time = 0; time < samples.size(); ++time) {
                    double sample =
                        samples[time] +
                        (anomalies[std::find(boost::begin(anomalyTimes), boost::end(anomalyTimes), time) - boost::begin(anomalyTimes)] *
                         boost::math::standard_deviation(gamma));

                    TDouble1Vec sampleVec(1, sample);
                    filter.addSamples(sampleVec);

                    double score;
                    filter.anomalyScore(maths_t::E_TwoSided, sampleVec, score);
                    if (score > threshold) {
                        candidateAnomalies.push_back(time);
                    }

                    filter.propagateForwardsByTime(1.0);

                    x << time << " ";
                    scores << score << " ";
                }

                x << "];\n";
                scores << "];\n";
                file << x.str() << scores.str() << "plot(x" << test << ", score" << test << ");\n"
                     << "input(\"Hit any key for next test\");\n\n";

                TUIntVec falsePositives;
                std::set_difference(candidateAnomalies.begin(),
                                    candidateAnomalies.end(),
                                    boost::begin(anomalyTimes),
                                    boost::end(anomalyTimes),
                                    std::back_inserter(falsePositives));

                double falsePositiveRate = static_cast<double>(falsePositives.size()) / static_cast<double>(samples.size());

                totalFalsePositiveRate += falsePositiveRate;

                TUIntVec positives;
                std::set_intersection(candidateAnomalies.begin(),
                                      candidateAnomalies.end(),
                                      boost::begin(anomalyTimes),
                                      boost::end(anomalyTimes),
                                      std::back_inserter(positives));

                LOG_DEBUG(<< "falsePositiveRate = " << falsePositiveRate << ", positives = " << positives.size());

                // False alarm rate should be less than 0.6%.
                CPPUNIT_ASSERT(falsePositiveRate <= 0.006);

                // Should detect at least the two big anomalies.
                CPPUNIT_ASSERT(positives.size() >= 2);

                totalPositives[k] += positives.size();
            }
        }
    }

    totalFalsePositiveRate /= static_cast<double>(test);

    LOG_DEBUG(<< "totalFalsePositiveRate = " << totalFalsePositiveRate);
    for (size_t i = 0; i < boost::size(totalPositives); ++i) {
        LOG_DEBUG(<< "positives = " << totalPositives[i]);

        CPPUNIT_ASSERT(totalPositives[i] >= 24);
    }

    // Total false alarm rate should be less than 0.11%.
    CPPUNIT_ASSERT(totalFalsePositiveRate < 0.0011);
}

void CGammaRateConjugateTest::testOffset() {
    LOG_DEBUG(<< "+---------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testOffset  |");
    LOG_DEBUG(<< "+---------------------------------------+");

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
                CGammaRateConjugate filter1(makePrior(dataTypes[i], offsets[j], decayRates[k]));
                CGammaRateConjugate filter2(makePrior(dataTypes[i], 0.0, decayRates[k]));

                for (std::size_t l = 0u; l < samples.size(); ++l) {
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
                    filter1.probabilityOfLessLikelySamples(maths_t::E_TwoSided, offsetSampleVec, lowerBound1, upperBound1);
                    CPPUNIT_ASSERT_EQUAL(lowerBound1, upperBound1);
                    double probability1 = (lowerBound1 + upperBound1) / 2.0;

                    double likelihood2;
                    filter2.jointLogMarginalLikelihood(sample, likelihood2);
                    double lowerBound2, upperBound2;
                    filter2.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound2, upperBound2);
                    CPPUNIT_ASSERT_EQUAL(lowerBound2, upperBound2);
                    double probability2 = (lowerBound2 + upperBound2) / 2.0;

                    CPPUNIT_ASSERT_DOUBLES_EQUAL(likelihood1, likelihood2, eps);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(probability1, probability2, eps);
                }

                using TEqual = maths::CEqualWithTolerance<double>;
                TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, eps);
                CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
            }
        }
    }
}

void CGammaRateConjugateTest::testIntegerData() {
    LOG_DEBUG(<< "+--------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testIntegerData  |");
    LOG_DEBUG(<< "+--------------------------------------------+");

    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // The idea of this test is to check that the inferred model agrees in the
    // limit (large n) with the model inferred from such data.

    const double shapes[] = {0.2, 1.0, 4.5};
    const double scales[] = {0.2, 1.0, 4.5};
    const std::size_t nSamples = 25000u;

    for (size_t i = 0; i < boost::size(shapes); ++i) {
        for (size_t j = 0; j < boost::size(scales); ++j) {
            test::CRandomNumbers rng;

            TDoubleVec samples;
            rng.generateGammaSamples(shapes[i], scales[j], nSamples, samples);

            TDoubleVec uniform;
            rng.generateUniformSamples(0.0, 1.0, nSamples, uniform);

            CGammaRateConjugate filter1(makePrior(maths_t::E_IntegerData, 0.1));
            CGammaRateConjugate filter2(makePrior(maths_t::E_ContinuousData, 0.1));

            for (std::size_t k = 0u; k < nSamples; ++k) {
                double x = std::floor(samples[k]);

                TDouble1Vec sample(1, x);
                filter1.addSamples(sample);

                sample[0] += uniform[k];
                filter2.addSamples(sample);
            }

            using TEqual = maths::CEqualWithTolerance<double>;
            TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.02);
            CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
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

            for (std::size_t k = 0u; k < nSamples; ++k) {
                double x = std::floor(samples[k]);

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
            LOG_DEBUG(<< "shape = " << shapes[i] << ", rate = " << scales[j] << ", p1 = " << p1 << ", p2 = " << p2);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 0.15 * p1);
            meanError.add(fabs(p1 - p2));
        }
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.016);
}

void CGammaRateConjugateTest::testLowVariationData() {
    LOG_DEBUG(<< "+-------------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testLowVariationData  |");
    LOG_DEBUG(<< "+-------------------------------------------------+");

    {
        CGammaRateConjugate filter(makePrior(maths_t::E_IntegerData));
        for (std::size_t i = 0u; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG(<< "68% confidence interval " << core::CContainerPrinter::print(interval)
                  << ", approximate variance = " << sigma * sigma);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, 1.0 / (sigma * sigma), 0.5);
    }
    {
        CGammaRateConjugate filter(makePrior(maths_t::E_ContinuousData));
        for (std::size_t i = 0u; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG(<< "68% confidence interval " << core::CContainerPrinter::print(interval) << ", approximate s.t.d. = " << sigma);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1e-4, sigma / 430.5, 5e-6);
    }
}

void CGammaRateConjugateTest::testPersist() {
    LOG_DEBUG(<< "+----------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testPersist  |");
    LOG_DEBUG(<< "+----------------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 3.0, 500, samples);

    maths::CGammaRateConjugate origFilter(makePrior(maths_t::E_ContinuousData, 0.1));
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        origFilter.addSamples(
            maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight), TDouble1Vec(1, samples[i]), TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)));
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
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                             decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::CGammaRateConjugate restoredFilter(params, traverser);

    uint64_t checksum = origFilter.checksum();
    LOG_DEBUG(<< "orig checksum = " << checksum << " restored checksum = " << restoredFilter.checksum());
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

void CGammaRateConjugateTest::testVarianceScale() {
    LOG_DEBUG(<< "+----------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testVarianceScale  |");
    LOG_DEBUG(<< "+----------------------------------------------+");

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

    maths_t::ESampleWeightStyle scales[] = {maths_t::E_SampleSeasonalVarianceScaleWeight, maths_t::E_SampleCountVarianceScaleWeight};

    for (std::size_t s = 0u; s < boost::size(scales); ++s) {
        const double shape = 3.0;
        const double scale = 3.0;

        const double varianceScales[] = {0.20, 0.50, 0.75, 1.50, 2.00, 5.00};

        test::CRandomNumbers rng;

        LOG_DEBUG(<< "");
        LOG_DEBUG(<< "****** probabilityOfLessLikelySamples ******");

        {
            const double percentiles[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};

            const std::size_t nSamples = 1000u;
            const std::size_t nScaledSamples = 10000u;

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
                    CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound, upperBound));
                    CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                    double probability = (lowerBound + upperBound) / 2.0;
                    probabilities.push_back(probability);
                }
                std::sort(probabilities.begin(), probabilities.end());

                for (size_t i = 0; i < boost::size(percentiles); ++i) {
                    std::size_t index = static_cast<std::size_t>(static_cast<double>(nScaledSamples) * percentiles[i] / 100.0);
                    double error = fabs(probabilities[index] - percentiles[i] / 100.0);
                    expectedPercentileErrors.push_back(error);
                    expectedTotalError += error;
                }
            }

            for (size_t i = 0; i < boost::size(varianceScales); ++i) {
                LOG_DEBUG(<< "**** variance scale = " << varianceScales[i] << " ****");

                double scaledShape = shape / varianceScales[i];
                double ss = varianceScales[i] * scale;
                {
                    boost::math::gamma_distribution<> gamma(scaledShape, ss);
                    LOG_DEBUG(<< "mean = " << boost::math::mean(gamma) << ", variance = " << boost::math::variance(gamma));
                }

                TDoubleVec scaledSamples;
                rng.generateGammaSamples(scaledShape, ss, nScaledSamples, scaledSamples);

                TDoubleVec probabilities;
                probabilities.reserve(nScaledSamples);
                for (std::size_t j = 0; j < scaledSamples.size(); ++j) {
                    double lowerBound, upperBound;
                    maths_t::ETail tail;
                    CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                                         maths_t::TWeightStyleVec(1, scales[s]),
                                                                         TDouble1Vec(1, scaledSamples[j]),
                                                                         TDouble4Vec1Vec(1, TDouble4Vec(1, varianceScales[i])),
                                                                         lowerBound,
                                                                         upperBound,
                                                                         tail));
                    CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                    double probability = (lowerBound + upperBound) / 2.0;
                    probabilities.push_back(probability);
                }
                std::sort(probabilities.begin(), probabilities.end());

                double totalError = 0.0;
                for (size_t j = 0; j < boost::size(percentiles); ++j) {
                    std::size_t index = static_cast<std::size_t>(static_cast<double>(nScaledSamples) * percentiles[j] / 100.0);
                    double error = fabs(probabilities[index] - percentiles[j] / 100.0);
                    totalError += error;
                    double errorThreshold = 0.017 + expectedPercentileErrors[j];

                    LOG_DEBUG(<< "percentile = " << percentiles[j] << ", probability = " << probabilities[index] << ", error = " << error
                              << ", error threshold = " << errorThreshold);

                    CPPUNIT_ASSERT(error < errorThreshold);
                }

                double totalErrorThreshold = 0.1 + expectedTotalError;

                LOG_DEBUG(<< "total error = " << totalError << ", totalError threshold = " << totalErrorThreshold);

                CPPUNIT_ASSERT(totalError < totalErrorThreshold);
            }
        }

        LOG_DEBUG(<< "");
        LOG_DEBUG(<< "****** jointLogMarginalLikelihood ******");

        for (size_t i = 0; i < boost::size(varianceScales); ++i) {
            LOG_DEBUG(<< "**** variance scale = " << varianceScales[i] << " ****");

            double scaledShape = shape / varianceScales[i];
            double scaledScale = varianceScales[i] * scale;
            boost::math::gamma_distribution<> gamma(scaledShape, scaledScale);
            { LOG_DEBUG(<< "mean = " << boost::math::mean(gamma) << ", variance = " << boost::math::variance(gamma)); }
            double expectedDifferentialEntropy = maths::CTools::differentialEntropy(gamma);

            CGammaRateConjugate filter(makePrior());

            double differentialEntropy = 0.0;

            TDoubleVec seedSamples;
            rng.generateGammaSamples(shape, scale, 150, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec scaledSamples;
            rng.generateGammaSamples(scaledShape, scaledScale, 50000, scaledSamples);

            for (std::size_t j = 0u; j < scaledSamples.size(); ++j) {
                double logLikelihood = 0.0;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                     filter.jointLogMarginalLikelihood(maths_t::TWeightStyleVec(1, scales[s]),
                                                                       TDouble1Vec(1, scaledSamples[j]),
                                                                       TDouble4Vec1Vec(1, TDouble4Vec(1, varianceScales[i])),
                                                                       logLikelihood));
                differentialEntropy -= logLikelihood;
            }

            differentialEntropy /= static_cast<double>(scaledSamples.size());

            LOG_DEBUG(<< "differentialEntropy = " << differentialEntropy
                      << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 0.05);
        }
    }

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double maximumMeanError[] = {0.08, 0.11};
    const double maximumVarianceError[] = {1.0, 0.2};
    const double maximumMeanMeanError[] = {0.01, 0.01};
    const double maximumMeanVarianceError[] = {0.08, 0.05};

    for (std::size_t s = 0u; s < boost::size(scales); ++s) {
        for (std::size_t t = 0u; t < boost::size(dataTypes); ++t) {
            const double shapes[] = {1.0, 10.0, 100.0, 1000.0, 100000.0, 1000000.0};
            const double rates[] = {1.0, 10.0, 100.0, 1000.0, 100000.0, 1000000.0};
            const double varianceScales[] = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};

            maths_t::TWeightStyleVec weightStyle(1, scales[s]);
            TDoubleVec samples;
            TDouble4Vec1Vec weights;

            test::CRandomNumbers rng;

            TMeanAccumulator meanMeanError;
            TMeanAccumulator meanVarianceError;

            for (std::size_t i = 0u; i < boost::size(shapes); ++i) {
                for (std::size_t j = 0u; j < boost::size(rates); ++j) {
                    double shape = shapes[i];
                    double rate = rates[j];

                    // We purposely don't estimate true variance in this case.
                    if (shape < rate * rate * maths::MINIMUM_COEFFICIENT_OF_VARIATION) {
                        continue;
                    }

                    LOG_DEBUG(<< "");
                    LOG_DEBUG(<< "****** shape = " << shape << ", rate = " << rate << " ******");

                    double mean = shape / rate;
                    double variance = mean / rate;

                    for (std::size_t k = 0u; k < boost::size(varianceScales); ++k) {
                        double scale = varianceScales[k];
                        LOG_DEBUG(<< "*** scale = " << scale << " ***");

                        double scaledShape = shape / scale;
                        double scaledRate = rate / scale;
                        LOG_DEBUG(<< "scaled shape = " << scaledShape << ", scaled rate = " << scaledRate);

                        TMeanAccumulator meanError;
                        TMeanAccumulator varianceError;
                        for (unsigned int test = 0u; test < 5; ++test) {
                            CGammaRateConjugate filter(makePrior(dataTypes[t]));

                            rng.generateGammaSamples(shape, 1.0 / rate, 200, samples);
                            weights.clear();
                            weights.resize(samples.size(), TDouble4Vec(1, 1.0));
                            filter.addSamples(weightStyle, samples, weights);
                            rng.generateGammaSamples(scaledShape, 1.0 / scaledRate, 200, samples);
                            weights.clear();
                            weights.resize(samples.size(), TDouble4Vec(1, scale));
                            filter.addSamples(weightStyle, samples, weights);

                            double estimatedMean = filter.likelihoodShape() / filter.likelihoodRate();
                            double estimatedVariance = estimatedMean / filter.likelihoodRate();
                            double dm = (dataTypes[t] == maths_t::E_IntegerData ? 0.5 : 0.0);
                            double dv = (dataTypes[t] == maths_t::E_IntegerData ? 1.0 / 12.0 : 0.0);
                            double trialMeanError = std::fabs(estimatedMean - (mean + dm)) / std::max(1.0, mean + dm);
                            double trialVarianceError = std::fabs(estimatedVariance - (variance + dv)) / std::max(1.0, variance + dv);

                            LOG_DEBUG(<< "trial mean error = " << trialMeanError);
                            LOG_DEBUG(<< "trial variance error = " << trialVarianceError);

                            meanError.add(trialMeanError);
                            varianceError.add(trialVarianceError);
                        }

                        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
                        LOG_DEBUG(<< "variance error = " << maths::CBasicStatistics::mean(varianceError));

                        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < maximumMeanError[t]);
                        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceError) < maximumVarianceError[t]);

                        meanMeanError += meanError;
                        meanVarianceError += varianceError;
                    }
                }
            }

            LOG_DEBUG(<< "mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
            LOG_DEBUG(<< "mean variance error = " << maths::CBasicStatistics::mean(meanVarianceError));

            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < maximumMeanMeanError[t]);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanVarianceError) < maximumMeanVarianceError[t]);
        }
    }
}

void CGammaRateConjugateTest::testNegativeSample() {
    LOG_DEBUG(<< "+-----------------------------------------------+");
    LOG_DEBUG(<< "|  CGammaRateConjugateTest::testNegativeSample  |");
    LOG_DEBUG(<< "+-----------------------------------------------+");

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

    CGammaRateConjugate filter1(CGammaRateConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.0, 0.0, 0.2));
    CGammaRateConjugate filter2(CGammaRateConjugate::nonInformativePrior(maths_t::E_ContinuousData, 1.2586, 0.0, 0.2));

    filter1.addSamples(samples);
    filter2.addSamples(samples);

    TDouble1Vec negative(1, -0.29);
    filter1.addSamples(negative);
    filter2.addSamples(negative);

    CPPUNIT_ASSERT_EQUAL(filter1.numberSamples(), filter2.numberSamples());

    using TEqual = maths::CEqualWithTolerance<double>;
    TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.1);
    CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
}

CppUnit::Test* CGammaRateConjugateTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CGammaRateConjugateTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testMultipleUpdate",
                                                                           &CGammaRateConjugateTest::testMultipleUpdate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testPropagation",
                                                                           &CGammaRateConjugateTest::testPropagation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testShapeEstimation",
                                                                           &CGammaRateConjugateTest::testShapeEstimation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testRateEstimation",
                                                                           &CGammaRateConjugateTest::testRateEstimation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testMarginalLikelihood",
                                                                           &CGammaRateConjugateTest::testMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testMarginalLikelihoodMean",
                                                                           &CGammaRateConjugateTest::testMarginalLikelihoodMean));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testMarginalLikelihoodMode",
                                                                           &CGammaRateConjugateTest::testMarginalLikelihoodMode));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testMarginalLikelihoodVariance",
                                                                           &CGammaRateConjugateTest::testMarginalLikelihoodVariance));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testSampleMarginalLikelihood",
                                                                           &CGammaRateConjugateTest::testSampleMarginalLikelihood));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testCdf", &CGammaRateConjugateTest::testCdf));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testProbabilityOfLessLikelySamples",
                                                                           &CGammaRateConjugateTest::testProbabilityOfLessLikelySamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testAnomalyScore",
                                                                           &CGammaRateConjugateTest::testAnomalyScore));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testOffset", &CGammaRateConjugateTest::testOffset));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testIntegerData",
                                                                           &CGammaRateConjugateTest::testIntegerData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testLowVariationData",
                                                                           &CGammaRateConjugateTest::testLowVariationData));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testPersist", &CGammaRateConjugateTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testVarianceScale",
                                                                           &CGammaRateConjugateTest::testVarianceScale));
    suiteOfTests->addTest(new CppUnit::TestCaller<CGammaRateConjugateTest>("CGammaRateConjugateTest::testNegativeSample",
                                                                           &CGammaRateConjugateTest::testNegativeSample));

    return suiteOfTests;
}
