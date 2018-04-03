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

#include "CNormalMeanPrecConjugateTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPriorDetail.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

#include <math.h>

using namespace ml;
using namespace handy_typedefs;

namespace {

typedef std::vector<double> TDoubleVec;
typedef std::pair<double, double> TDoubleDoublePr;
typedef std::vector<TDoubleDoublePr> TDoubleDoublePrVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TMeanVarAccumulator;
typedef CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate> CNormalMeanPrecConjugate;

CNormalMeanPrecConjugate makePrior(maths_t::EDataType dataType = maths_t::E_ContinuousData, const double& decayRate = 0.0) {
    return CNormalMeanPrecConjugate::nonInformativePrior(dataType, decayRate);
}
}

void CNormalMeanPrecConjugateTest::testMultipleUpdate(void) {
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testMultipleUpdate  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    typedef maths::CEqualWithTolerance<double> TEqual;

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double mean = 10.0;
    const double variance = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    for (std::size_t i = 0; i < boost::size(dataTypes); ++i) {
        CNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CNormalMeanPrecConjugate filter2(filter1);

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            filter1.addSamples(TDouble1Vec(1, samples[j]));
        }
        filter2.addSamples(samples);

        LOG_DEBUG(filter1.print());
        LOG_DEBUG("vs");
        LOG_DEBUG(filter2.print());
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }

    // Test with variance scale.

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CNormalMeanPrecConjugate filter2(filter1);

        maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            filter1.addSamples(weightStyle, TDouble1Vec(1, samples[j]), TDouble4Vec1Vec(1, TDouble4Vec(1, 2.0)));
        }
        filter2.addSamples(weightStyle, samples, TDouble4Vec1Vec(samples.size(), TDouble4Vec(1, 2.0)));

        LOG_DEBUG(filter1.print());
        LOG_DEBUG("vs");
        LOG_DEBUG(filter2.print());
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }

    // Test the count weight is equivalent to adding repeated samples.

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        CNormalMeanPrecConjugate filter1(makePrior(dataTypes[i]));
        CNormalMeanPrecConjugate filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;
        for (std::size_t j = 0u; j < count; ++j) {
            filter1.addSamples(TDouble1Vec(1, x));
        }
        filter2.addSamples(maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight),
                           TDouble1Vec(1, x),
                           TDouble4Vec1Vec(1, TDouble4Vec(1, static_cast<double>(count))));

        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));
    }
}

void CNormalMeanPrecConjugateTest::testPropagation(void) {
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testPropagation  |");
    LOG_DEBUG("+-------------------------------------------------+");

    // Test that propagation doesn't affect the expected values
    // of likelihood mean and precision.

    const double eps = 1e-12;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(1.0, 3.0, 500, samples);

    CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, 0.1));

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
    }

    double mean = filter.mean();
    double precision = filter.precision();

    filter.propagateForwardsByTime(5.0);

    double propagatedMean = filter.mean();
    double propagatedPrecision = filter.precision();

    LOG_DEBUG("mean = " << mean << ", precision = " << precision << ", propagatedMean = " << propagatedMean
                        << ", propagatedPrecision = " << propagatedPrecision);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(mean, propagatedMean, eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(precision, propagatedPrecision, eps);
}

void CNormalMeanPrecConjugateTest::testMeanEstimation(void) {
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testMeanEstimation  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // We are going to test that we correctly estimate a distribution
    // for the mean of the Gaussian process by checking that the true
    // mean of a Gaussian process lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 500u;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0};

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        for (unsigned int test = 0; test < nTests; ++test) {
            double mean = 0.5 * (test + 1);
            double variance = 4.0;

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, 500, samples);

            CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, decayRates[i]));

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval = filter.confidenceIntervalMean(testIntervals[j]);

                if (mean < confidenceInterval.first || mean > confidenceInterval.second) {
                    ++errors[j];
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_DEBUG("interval = " << interval << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            if (decayRates[i] == 0.0) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[j]), 4.0);
            } else {
                CPPUNIT_ASSERT(interval <= (100.0 - testIntervals[j]));
            }
        }
    }
}

void CNormalMeanPrecConjugateTest::testPrecisionEstimation(void) {
    LOG_DEBUG("+---------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testPrecisionEstimation  |");
    LOG_DEBUG("+---------------------------------------------------------+");

    // We are going to test that we correctly estimate a distribution
    // for the precision of the Gaussian process by checking that the
    // true precision of a Gaussian process lies in various confidence
    // intervals the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 1000u;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0};

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        for (unsigned int test = 0; test < nTests; ++test) {
            double mean = 0.5 * (test + 1);
            double variance = 2.0 + 0.01 * test;
            double precision = 1 / variance;

            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, 1000, samples);

            CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, decayRates[i]));

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval = filter.confidenceIntervalPrecision(testIntervals[j]);

                if (precision < confidenceInterval.first || precision > confidenceInterval.second) {
                    ++errors[j];
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_DEBUG("interval = " << interval << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            if (decayRates[i] == 0.0) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[j]), 3.0);
            } else {
                CPPUNIT_ASSERT(interval <= (100.0 - testIntervals[j]));
            }
        }
    }
}

void CNormalMeanPrecConjugateTest::testMarginalLikelihood(void) {
    LOG_DEBUG("+--------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testMarginalLikelihood  |");
    LOG_DEBUG("+--------------------------------------------------------+");

    // Check that the c.d.f. <= 1 at extreme.
    maths_t::EDataType dataTypes[] = {maths_t::E_ContinuousData, maths_t::E_IntegerData};
    for (std::size_t t = 0u; t < boost::size(dataTypes); ++t) {
        CNormalMeanPrecConjugate filter(makePrior(dataTypes[t]));

        const double mean = 1.0;
        const double variance = 1.0;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 200, samples);
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
                LOG_DEBUG("-log(c.d.f) = " << (lb + ub) / 2.0);
                CPPUNIT_ASSERT(lb >= 0.0);
                CPPUNIT_ASSERT(ub >= 0.0);
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
            CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, decayRates[j]));

            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                filter.propagateForwardsByTime(1.0);
            }

            // We'll check that the p.d.f. is close to the derivative of the
            // c.d.f. at a range of deltas from the true mean.

            const double eps = 1e-4;
            double deltas[] = {-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};

            for (std::size_t k = 0; k < boost::size(deltas); ++k) {
                double x = mean + deltas[k] * ::sqrt(variance);
                TDouble1Vec sample(1, x);

                LOG_DEBUG("number = " << numberSamples[i] << ", sample = " << sample[0]);

                double logLikelihood = 0.0;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter.jointLogMarginalLikelihood(sample, logLikelihood));
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
        // expected log likelihood for a normal distribution (uniform law
        // of large numbers), which is just the differential entropy of
        // a normal R.V.

        boost::math::normal_distribution<> normal(mean, ::sqrt(variance));
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(normal);

        CNormalMeanPrecConjugate filter(makePrior());

        double differentialEntropy = 0.0;

        TDoubleVec seedSamples;
        rng.generateNormalSamples(mean, variance, 100, seedSamples);
        filter.addSamples(seedSamples);

        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 100000, samples);
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            TDouble1Vec sample(1, samples[i]);
            filter.addSamples(sample);
            double logLikelihood = 0.0;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter.jointLogMarginalLikelihood(sample, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(samples.size());

        LOG_DEBUG("differentialEntropy = " << differentialEntropy << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 2e-3);
    }

    {
        boost::math::normal_distribution<> normal(mean, ::sqrt(variance));
        const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};

        CNormalMeanPrecConjugate filter(makePrior());
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, 1000, samples);
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
                LOG_DEBUG("[q1, q2] = [" << q1 << ", " << q2 << "]"
                                         << ", interval = " << core::CContainerPrinter::print(interval));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, 0.005);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.005);
                error.add(::fabs(interval.first - q1));
                error.add(::fabs(interval.second - q2));
            }
            LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 1e-3);
        }
        {
            TMeanAccumulator totalError;
            for (std::size_t i = 0u; i < boost::size(varianceScales); ++i) {
                TMeanAccumulator error;
                double vs = varianceScales[i];
                boost::math::normal_distribution<> scaledNormal(mean, ::sqrt(vs * variance));
                LOG_DEBUG("*** vs = " << vs << " ***");
                for (std::size_t j = 0u; j < boost::size(percentages); ++j) {
                    double q1 = boost::math::quantile(scaledNormal, (50.0 - percentages[j] / 2.0) / 100.0);
                    double q2 = boost::math::quantile(scaledNormal, (50.0 + percentages[j] / 2.0) / 100.0);
                    TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(
                        percentages[j], maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight), TDouble4Vec(1, vs));
                    LOG_DEBUG("[q1, q2] = [" << q1 << ", " << q2 << "]"
                                             << ", interval = " << core::CContainerPrinter::print(interval));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, 0.3);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.3);
                    error.add(::fabs(interval.first - q1));
                    error.add(::fabs(interval.second - q2));
                }
                LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.1);
                totalError += error;
            }
            LOG_DEBUG("totalError = " << maths::CBasicStatistics::mean(totalError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(totalError) < 0.06);
        }
    }
}

void CNormalMeanPrecConjugateTest::testMarginalLikelihoodMean(void) {
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testMarginalLikelihoodMean  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    // Test that the expectation of the marginal likelihood matches
    // the expected mean of the marginal likelihood.

    const double means[] = {1.0, 5.0, 100.0};
    const double variances[] = {2.0, 5.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        for (std::size_t j = 0u; j < boost::size(variances); ++j) {
            LOG_DEBUG("*** mean = " << means[i] << ", variance = " << variances[j] << " ***");

            CNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateNormalSamples(means[i], variances[j], 1, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 100, samples);

            TMeanAccumulator relativeError;
            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));

                double expectedMean;
                CPPUNIT_ASSERT(filter.marginalLikelihoodMeanForTest(expectedMean));

                if (k % 10 == 0) {
                    LOG_DEBUG("marginalLikelihoodMean = " << filter.marginalLikelihoodMean() << ", expectedMean = " << expectedMean);
                }

                // The error is at the precision of the numerical integration.
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, filter.marginalLikelihoodMean(), 0.01);

                relativeError.add(::fabs(expectedMean - filter.marginalLikelihoodMean()) / expectedMean);
            }

            LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 1e-4);
        }
    }
}

void CNormalMeanPrecConjugateTest::testMarginalLikelihoodMode(void) {
    LOG_DEBUG("+-----------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testMarginalLikelihoodMode |");
    LOG_DEBUG("+-----------------------------------------------------------+");

    // Test that the marginal likelihood mode is what we'd expect
    // with variances variance scales.

    const double means[] = {1.0, 5.0, 100.0};
    const double variances[] = {2.0, 5.0, 20.0};
    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        for (std::size_t j = 0u; j < boost::size(variances); ++j) {
            LOG_DEBUG("*** mean = " << means[i] << ", variance = " << variances[j] << " ***");

            CNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 1000, samples);
            filter.addSamples(samples);

            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
            TDouble4Vec weight(1, 1.0);

            for (std::size_t k = 0u; k < boost::size(varianceScales); ++k) {
                double vs = varianceScales[i];
                weight[0] = vs;
                boost::math::normal_distribution<> scaledNormal(means[i], ::sqrt(vs * variances[j]));
                double expectedMode = boost::math::mode(scaledNormal);
                LOG_DEBUG("marginalLikelihoodMode = " << filter.marginalLikelihoodMode(weightStyle, weight)
                                                      << ", expectedMode = " << expectedMode);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode, filter.marginalLikelihoodMode(weightStyle, weight), 0.12 * ::sqrt(variances[j]));
            }
        }
    }
}

void CNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance(void) {
    LOG_DEBUG("+----------------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance  |");
    LOG_DEBUG("+----------------------------------------------------------------+");

    // Test that the expectation of the residual from the mean for
    // the marginal likelihood matches the expected variance of the
    // marginal likelihood.

    const double means[] = {1.0, 5.0, 100.0};
    const double variances[] = {2.0, 5.0, 20.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        for (std::size_t j = 0u; j < boost::size(variances); ++j) {
            LOG_DEBUG("*** mean = " << means[i] << ", variance = " << variances[j] << " ***");

            CNormalMeanPrecConjugate filter(makePrior());

            TDoubleVec seedSamples;
            rng.generateNormalSamples(means[i], variances[j], 5, seedSamples);
            filter.addSamples(seedSamples);

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 100, samples);

            TMeanAccumulator relativeError;
            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                double expectedVariance;
                CPPUNIT_ASSERT(filter.marginalLikelihoodVarianceForTest(expectedVariance));
                if (k % 10 == 0) {
                    LOG_DEBUG("marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                                                              << ", expectedVariance = " << expectedVariance);
                }

                // The error is at the precision of the numerical integration.
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedVariance, filter.marginalLikelihoodVariance(), 0.2);

                relativeError.add(::fabs(expectedVariance - filter.marginalLikelihoodVariance()) / expectedVariance);
            }

            LOG_DEBUG("relativeError = " << maths::CBasicStatistics::mean(relativeError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(relativeError) < 1e-3);
        }
    }
}

void CNormalMeanPrecConjugateTest::testSampleMarginalLikelihood(void) {
    LOG_DEBUG("+--------------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG("+--------------------------------------------------------------+");

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

    for (std::size_t i = 0u; i < 1u; ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(10, sampled);

        CPPUNIT_ASSERT_EQUAL(i + 1, sampled.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(samples[i], sampled[i], eps);
    }

    TMeanAccumulator meanVarError;

    std::size_t numberSampled = 20u;
    for (std::size_t i = 1u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(numberSampled, sampled);
        CPPUNIT_ASSERT_EQUAL(numberSampled, sampled.size());

        TMeanVarAccumulator sampledMoments;
        sampledMoments = std::for_each(sampled.begin(), sampled.end(), sampledMoments);

        LOG_DEBUG("expectedMean = " << filter.marginalLikelihoodMean()
                                    << ", sampledMean = " << maths::CBasicStatistics::mean(sampledMoments));
        LOG_DEBUG("expectedVariance = " << filter.marginalLikelihoodVariance()
                                        << ", sampledVariance = " << maths::CBasicStatistics::variance(sampledMoments));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodMean(), maths::CBasicStatistics::mean(sampledMoments), 1e-8);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodVariance(),
                                     maths::CBasicStatistics::variance(sampledMoments),
                                     0.2 * filter.marginalLikelihoodVariance());
        meanVarError.add(::fabs(filter.marginalLikelihoodVariance() - maths::CBasicStatistics::variance(sampledMoments)) /
                         filter.marginalLikelihoodVariance());

        std::sort(sampled.begin(), sampled.end());
        for (std::size_t j = 1u; j < sampled.size(); ++j) {
            double q = 100.0 * static_cast<double>(j) / static_cast<double>(numberSampled);

            double expectedQuantile;
            CPPUNIT_ASSERT(filter.marginalLikelihoodQuantileForTest(q, eps, expectedQuantile));

            LOG_DEBUG("quantile = " << q << ", x_quantile = " << expectedQuantile << ", quantile range = [" << sampled[j - 1] << ","
                                    << sampled[j] << "]");

            CPPUNIT_ASSERT(expectedQuantile >= sampled[j - 1]);
            CPPUNIT_ASSERT(expectedQuantile <= sampled[j]);
        }
    }

    LOG_DEBUG("mean variance error = " << maths::CBasicStatistics::mean(meanVarError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanVarError) < 0.04);
}

void CNormalMeanPrecConjugateTest::testCdf(void) {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testCdf  |");
    LOG_DEBUG("+-----------------------------------------+");

    // Test error cases.
    //
    // Test some invariants:
    //   "cdf" + "cdf complement" = 1

    const double mean = 20.0;
    const double variance = 5.0;
    const std::size_t n[] = {20u, 80u};

    test::CRandomNumbers rng;

    CNormalMeanPrecConjugate filter(makePrior());

    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        TDoubleVec samples;
        rng.generateNormalSamples(mean, variance, n[i], samples);

        filter.addSamples(samples);

        double lowerBound;
        double upperBound;
        CPPUNIT_ASSERT(!filter.minusLogJointCdf(TDouble1Vec(), lowerBound, upperBound));
        CPPUNIT_ASSERT(!filter.minusLogJointCdfComplement(TDouble1Vec(), lowerBound, upperBound));

        for (std::size_t j = 1u; j < 500; ++j) {
            double x = static_cast<double>(j) / 2.0;

            CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x), lowerBound, upperBound));
            double f = (lowerBound + upperBound) / 2.0;
            CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lowerBound, upperBound));
            double fComplement = (lowerBound + upperBound) / 2.0;

            LOG_DEBUG("log(F(x)) = " << (f == 0.0 ? f : -f) << ", log(1 - F(x)) = " << (fComplement == 0.0 ? fComplement : -fComplement));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ::exp(-f) + ::exp(-fComplement), 1e-10);
        }
    }
}

void CNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples(void) {
    LOG_DEBUG("+--------------------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+--------------------------------------------------------------------+");

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
            LOG_DEBUG("means = " << means[i] << ", variance = " << variances[j]);

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 1000, samples);

            CNormalMeanPrecConjugate filter(makePrior());
            filter.addSamples(samples);

            double mean = filter.mean();
            double sd = ::sqrt(1.0 / filter.precision());

            TDoubleVec likelihoods;
            for (std::size_t k = 0u; k < samples.size(); ++k) {
                double likelihood;
                filter.jointLogMarginalLikelihood(TDouble1Vec(1, samples[k]), likelihood);
                likelihoods.push_back(likelihood);
            }
            std::sort(likelihoods.begin(), likelihoods.end());

            boost::math::normal_distribution<> normal(mean, sd);
            for (std::size_t k = 1u; k < 10; ++k) {
                double x = boost::math::quantile(normal, static_cast<double>(k) / 10.0);

                TDouble1Vec sample(1, x);
                double fx;
                filter.jointLogMarginalLikelihood(sample, fx);

                double px = static_cast<double>(std::lower_bound(likelihoods.begin(), likelihoods.end(), fx) - likelihoods.begin()) /
                            static_cast<double>(likelihoods.size());

                double lb, ub;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lb, ub);

                double ssd = ::sqrt(px * (1.0 - px) / static_cast<double>(samples.size()));

                LOG_DEBUG("expected P(x) = " << px << ", actual P(x) = " << (lb + ub) / 2.0 << " sample sd = " << ssd);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(px, (lb + ub) / 2.0, 3.0 * ssd);

                meanError.add(::fabs(px - (lb + ub) / 2.0));
            }

            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);

            for (std::size_t k = 0u; k < boost::size(vs); ++k) {
                double mode = filter.marginalLikelihoodMode(weightStyle, TDouble4Vec(1, vs[k]));
                double ss[] = {0.9 * mode, 1.1 * mode};

                LOG_DEBUG("vs = " << vs[k] << ", mode = " << mode);

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

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.01);
}

void CNormalMeanPrecConjugateTest::testAnomalyScore(void) {
    LOG_DEBUG("+--------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testAnomalyScore  |");
    LOG_DEBUG("+--------------------------------------------------+");

    // This test pushes 500 samples through the filter and adds in
    // anomalous signals in the bins at 30, 120, 300 and 420 with
    // magnitude 4, 5, 10 and 15 standard deviations, respectively,
    // and checks the anomaly score has:
    //   1) high probability of detecting the anomalies, and
    //   2) a very low rate of false positives.

    typedef std::vector<unsigned int> TUIntVec;

    const double decayRates[] = {0.0, 0.001, 0.01};

    const double means[] = {3.0, 15.0, 200.0};
    const double variances[] = {2.0, 5.0, 50.0};

    const double threshold = 0.01;

    const unsigned int anomalyTimes[] = {30u, 120u, 300u, 420u};
    const double anomalies[] = {4.0, 5.0, 10.0, 15.0, 0.0};

    test::CRandomNumbers rng;

    unsigned int test = 0;

    std::ofstream file;
    file.open("results.m");

    double totalFalsePositiveRate = 0.0;
    std::size_t totalPositives[] = {0u, 0u, 0u};

    for (std::size_t i = 0; i < boost::size(means); ++i) {
        for (std::size_t j = 0; j < boost::size(variances); ++j) {
            LOG_DEBUG("mean = " << means[i] << ", variance = " << variances[j]);

            boost::math::normal_distribution<> normal(means[i], ::sqrt(variances[j]));

            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 500, samples);

            for (std::size_t k = 0; k < boost::size(decayRates); ++k) {
                CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData, decayRates[k]));

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
                         boost::math::standard_deviation(normal));

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

                LOG_DEBUG("falsePositiveRate = " << falsePositiveRate << ", positives = " << positives.size());

                // False alarm rate should be less than 0.6%.
                CPPUNIT_ASSERT(falsePositiveRate <= 0.006);

                // Should detect at least the three biggest anomalies.
                CPPUNIT_ASSERT(positives.size() >= 3u);

                totalPositives[k] += positives.size();
            }
        }
    }

    totalFalsePositiveRate /= static_cast<double>(test);

    LOG_DEBUG("totalFalsePositiveRate = " << totalFalsePositiveRate);

    for (std::size_t i = 0; i < boost::size(totalPositives); ++i) {
        LOG_DEBUG("positives = " << totalPositives[i]);

        // Should detect all but one anomaly.
        CPPUNIT_ASSERT(totalPositives[i] >= 32u);
    }

    // Total false alarm rate should be less than 0.3%.
    CPPUNIT_ASSERT(totalFalsePositiveRate < 0.003);
}

void CNormalMeanPrecConjugateTest::testIntegerData(void) {
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testIntegerData  |");
    LOG_DEBUG("+-------------------------------------------------+");

    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // The idea of this test is to check that the inferred model agrees in the
    // limit (large n) with the model inferred from such data.

    const double mean = 12.0;
    const double variance = 3.0;
    const std::size_t nSamples = 100000u;

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

        typedef maths::CEqualWithTolerance<double> TEqual;
        TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.001);
        CPPUNIT_ASSERT(filter1.equalTolerance(filter2, equal));

        TMeanAccumulator meanLogLikelihood1;
        TMeanAccumulator meanLogLikelihood2;
        for (std::size_t j = 0u; j < nSamples; ++j) {
            double x = ::floor(samples[j]);

            TDouble1Vec sample(1, x);
            double logLikelihood1;
            filter1.jointLogMarginalLikelihood(sample, logLikelihood1);
            meanLogLikelihood1.add(-logLikelihood1);

            sample[0] += uniform[j];
            double logLikelihood2;
            filter2.jointLogMarginalLikelihood(sample, logLikelihood2);
            meanLogLikelihood2.add(-logLikelihood2);
        }

        LOG_DEBUG("meanLogLikelihood1 = " << maths::CBasicStatistics::mean(meanLogLikelihood1)
                                          << ", meanLogLikelihood2 = " << maths::CBasicStatistics::mean(meanLogLikelihood2));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            maths::CBasicStatistics::mean(meanLogLikelihood1), maths::CBasicStatistics::mean(meanLogLikelihood2), 0.02);
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
            double x = ::floor(samples[i]);

            TDouble1Vec sample(1, x);

            double l1, u1;
            CPPUNIT_ASSERT(filter1.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, l1, u1));
            CPPUNIT_ASSERT_EQUAL(l1, u1);
            double p1 = (l1 + u1) / 2.0;
            meanProbability1.add(p1);

            sample[0] += uniform[i];
            double l2, u2;
            CPPUNIT_ASSERT(filter2.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, l2, u2));
            CPPUNIT_ASSERT_EQUAL(l2, u2);
            double p2 = (l2 + u2) / 2.0;
            meanProbability2.add(p2);
        }

        double p1 = maths::CBasicStatistics::mean(meanProbability1);
        double p2 = maths::CBasicStatistics::mean(meanProbability2);
        LOG_DEBUG("p1 = " << p1 << ", p2 = " << p2);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 0.01 * p1);
    }
}

void CNormalMeanPrecConjugateTest::testLowVariationData(void) {
    LOG_DEBUG("+------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testLowVariationData  |");
    LOG_DEBUG("+------------------------------------------------------+");

    {
        CNormalMeanPrecConjugate filter(makePrior(maths_t::E_IntegerData));
        for (std::size_t i = 0u; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG("68% confidence interval " << core::CContainerPrinter::print(interval) << ", approximate variance = " << sigma * sigma);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, 1.0 / (sigma * sigma), 0.15);
    }
    {
        CNormalMeanPrecConjugate filter(makePrior(maths_t::E_ContinuousData));
        for (std::size_t i = 0u; i < 100; ++i) {
            filter.addSamples(TDouble1Vec(1, 430.0));
        }

        TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(68.0);
        double sigma = (interval.second - interval.first) / 2.0;
        LOG_DEBUG("68% confidence interval " << core::CContainerPrinter::print(interval) << ", approximate s.t.d. = " << sigma);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / maths::MINIMUM_COEFFICIENT_OF_VARIATION / 430.5, 1.0 / sigma, 7.0);
    }
}

void CNormalMeanPrecConjugateTest::testPersist(void) {
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testPersist  |");
    LOG_DEBUG("+---------------------------------------------+");

    // Check that persist/restore is idempotent.

    const double mean = 10.0;
    const double variance = 3.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(mean, variance, 100, samples);

    maths::CNormalMeanPrecConjugate origFilter(makePrior());
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        origFilter.addSamples(
            maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight), TDouble1Vec(1, samples[i]), TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)));
    }
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("Normal mean conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                             decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::CNormalMeanPrecConjugate restoredFilter(params, traverser);

    LOG_DEBUG("orig checksum = " << checksum << " restored checksum = " << restoredFilter.checksum());
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

void CNormalMeanPrecConjugateTest::testSeasonalVarianceScale(void) {
    LOG_DEBUG("+-----------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testSeasonalVarianceScale  |");
    LOG_DEBUG("+-----------------------------------------------------------+");

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

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        for (std::size_t j = 0u; j < boost::size(variances); ++j) {
            TDoubleVec samples;
            rng.generateNormalSamples(means[i], variances[j], 100, samples);

            double varianceScales[] = {0.2, 0.5, 1.0, 2.0, 5.0};
            maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleSeasonalVarianceScaleWeight);
            TDouble4Vec weight(1, 1.0);
            TDouble4Vec1Vec weights(1, weight);

            double m;
            double v;

            {
                CNormalMeanPrecConjugate filter(makePrior());
                filter.addSamples(samples);

                m = filter.marginalLikelihoodMean();
                v = filter.marginalLikelihoodVariance();
                double s = ::sqrt(v);
                LOG_DEBUG("m = " << m << ", v = " << v);

                double points[] = {m - 3.0 * s, m - s, m, m + s, m + 3.0 * s};

                double unscaledExpectationVariance;
                filter.expectation(CVarianceKernel(filter.marginalLikelihoodMean()), 100, unscaledExpectationVariance);
                LOG_DEBUG("unscaledExpectationVariance = " << unscaledExpectationVariance);

                for (std::size_t k = 0u; k < boost::size(varianceScales); ++k) {
                    double vs = varianceScales[k];
                    weight[0] = vs;
                    weights[0][0] = vs;
                    LOG_DEBUG("*** variance scale = " << vs << " ***");

                    double Z;
                    filter.expectation(C1dUnitKernel(), 50, Z, weightStyle, weight);
                    LOG_DEBUG("Z = " << Z);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, Z, 1e-3);

                    LOG_DEBUG("sv = " << filter.marginalLikelihoodVariance(weightStyle, weight));
                    double expectationVariance;
                    filter.expectation(CVarianceKernel(filter.marginalLikelihoodMean()), 100, expectationVariance, weightStyle, weight);
                    LOG_DEBUG("expectationVariance = " << expectationVariance);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        vs * unscaledExpectationVariance, expectationVariance, 0.01 * vs * unscaledExpectationVariance);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodVariance(weightStyle, weight),
                                                 expectationVariance,
                                                 0.01 * filter.marginalLikelihoodVariance(weightStyle, weight));

                    double mode = filter.marginalLikelihoodMode(weightStyle, weight);
                    double fm;
                    double fmMinusEps, fmPlusEps;
                    filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, mode - 1e-3), weights, fmMinusEps);
                    filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, mode), weights, fm);
                    filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, mode + 1e-3), weights, fmPlusEps);
                    LOG_DEBUG("log(f(mode)) = " << fm << ", log(f(mode - eps)) = " << fmMinusEps << ", log(f(mode + eps)) = " << fmPlusEps);
                    CPPUNIT_ASSERT(fm > fmMinusEps);
                    CPPUNIT_ASSERT(fm > fmPlusEps);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, (::exp(fmPlusEps) - ::exp(fmMinusEps)) / 2e-3, 1e-6);
                    TDouble1Vec sample(1, 0.0);
                    for (std::size_t l = 0u; l < boost::size(points); ++l) {
                        TDouble1Vec x(1, points[l]);
                        double fx;
                        filter.jointLogMarginalLikelihood(weightStyle, x, weights, fx);
                        TDouble1Vec xMinusEps(1, points[l] - 1e-3);
                        TDouble1Vec xPlusEps(1, points[l] + 1e-3);
                        double lb, ub;
                        filter.minusLogJointCdf(weightStyle, xPlusEps, weights, lb, ub);
                        double FxPlusEps = ::exp(-(lb + ub) / 2.0);
                        filter.minusLogJointCdf(weightStyle, xMinusEps, weights, lb, ub);
                        double FxMinusEps = ::exp(-(lb + ub) / 2.0);
                        LOG_DEBUG("x = " << points[l] << ", log(f(x)) = " << fx << ", F(x - eps) = " << FxMinusEps
                                         << ", F(x + eps) = " << FxPlusEps << ", log(dF/dx)) = " << ::log((FxPlusEps - FxMinusEps) / 2e-3));
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(fx, ::log((FxPlusEps - FxMinusEps) / 2e-3), 0.05 * ::fabs(fx));

                        sample[0] = m + (points[l] - m) / ::sqrt(vs);
                        weights[0][0] = 1.0;
                        double expectedLowerBound;
                        double expectedUpperBound;
                        maths_t::ETail expectedTail;
                        filter.probabilityOfLessLikelySamples(
                            maths_t::E_TwoSided, weightStyle, sample, weights, expectedLowerBound, expectedUpperBound, expectedTail);

                        sample[0] = points[l];
                        weights[0][0] = vs;
                        double lowerBound;
                        double upperBound;
                        maths_t::ETail tail;
                        filter.probabilityOfLessLikelySamples(
                            maths_t::E_TwoSided, weightStyle, sample, weights, lowerBound, upperBound, tail);

                        LOG_DEBUG("expectedLowerBound = " << expectedLowerBound);
                        LOG_DEBUG("lowerBound         = " << lowerBound);
                        LOG_DEBUG("expectedUpperBound = " << expectedUpperBound);
                        LOG_DEBUG("upperBound         = " << upperBound);
                        LOG_DEBUG("expectedTail       = " << expectedTail);
                        LOG_DEBUG("tail               = " << tail);

                        if ((expectedLowerBound + expectedUpperBound) < 0.02) {
                            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                                ::log(expectedLowerBound), ::log(lowerBound), 0.1 * ::fabs(::log(expectedLowerBound)));
                            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                                ::log(expectedUpperBound), ::log(upperBound), 0.1 * ::fabs(::log(expectedUpperBound)));
                        } else {
                            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedLowerBound, lowerBound, 0.01 * expectedLowerBound);
                            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedUpperBound, upperBound, 0.01 * expectedUpperBound);
                        }
                        CPPUNIT_ASSERT_EQUAL(expectedTail, tail);
                    }
                }
            }
            for (std::size_t k = 0u; k < boost::size(varianceScales); ++k) {
                double vs = varianceScales[k];

                rng.random_shuffle(samples.begin(), samples.end());

                CNormalMeanPrecConjugate filter(makePrior());
                weights[0][0] = vs;
                for (std::size_t l = 0u; l < samples.size(); ++l) {
                    filter.addSamples(weightStyle, TDouble1Vec(1, samples[l]), weights);
                }

                double sm = filter.marginalLikelihoodMean();
                double sv = filter.marginalLikelihoodVariance();
                LOG_DEBUG("m  = " << m << ", v  = " << v);
                LOG_DEBUG("sm = " << sm << ", sv = " << sv);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(m, sm, ::fabs(0.25 * m));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(v / vs, sv, 0.05 * v / vs);
            }
        }
    }
}

void CNormalMeanPrecConjugateTest::testCountVarianceScale(void) {
    LOG_DEBUG("+--------------------------------------------------------+");
    LOG_DEBUG("|  CNormalMeanPrecConjugateTest::testCountVarianceScale  |");
    LOG_DEBUG("+--------------------------------------------------------+");

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

    LOG_DEBUG("");
    LOG_DEBUG("****** probabilityOfLessLikelySamples ******");

    const double percentiles[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    const std::size_t nSamples[] = {30u, 1000u};
    const std::size_t nScaledSamples = 10000u;

    double percentileErrorTolerances[] = {0.15, 0.03};
    double totalErrorTolerances[] = {0.25, 0.13};
    double totalTotalError = 0.0;

    for (std::size_t i = 0; i < boost::size(nSamples); ++i) {
        LOG_DEBUG("**** nSamples = " << nSamples[i] << " ****");

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
                CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
                double probability = (lowerBound + upperBound) / 2.0;
                probabilities.push_back(probability);
            }
            std::sort(probabilities.begin(), probabilities.end());

            for (std::size_t j = 0; j < boost::size(percentiles); ++j) {
                std::size_t index = static_cast<std::size_t>(static_cast<double>(nScaledSamples) * percentiles[j] / 100.0);
                double error = ::fabs(probabilities[index] - percentiles[j] / 100.0);
                expectedPercentileErrors.push_back(error);
                expectedTotalError += error;
            }
        }

        for (std::size_t j = 0; j < boost::size(varianceScales); ++j) {
            LOG_DEBUG("**** variance scale = " << varianceScales[j] << " ****");

            TDoubleVec scaledSamples;
            rng.generateNormalSamples(mean, varianceScales[j] * variance, nScaledSamples, scaledSamples);

            TDoubleVec probabilities;
            probabilities.reserve(nScaledSamples);
            for (std::size_t k = 0; k < scaledSamples.size(); ++k) {

                double lowerBound, upperBound;
                maths_t::ETail tail;
                CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                                     maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight),
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

            double totalError = 0.0;
            for (std::size_t k = 0; k < boost::size(percentiles); ++k) {
                std::size_t index = static_cast<std::size_t>(static_cast<double>(nScaledSamples) * percentiles[k] / 100.0);
                double error = fabs(probabilities[index] - percentiles[k] / 100.0);
                totalError += error;
                double errorThreshold = percentileErrorTolerances[i] + expectedPercentileErrors[k];

                LOG_DEBUG("percentile = " << percentiles[k] << ", probability = " << probabilities[index] << ", error = " << error
                                          << ", error threshold = " << errorThreshold);

                CPPUNIT_ASSERT(error < errorThreshold);
            }

            double totalErrorThreshold = totalErrorTolerances[i] + expectedTotalError;

            LOG_DEBUG("totalError = " << totalError << ", totalError threshold = " << totalErrorThreshold);

            CPPUNIT_ASSERT(totalError < totalErrorThreshold);
            totalTotalError += totalError;
        }
    }

    LOG_DEBUG("total totalError = " << totalTotalError);
    CPPUNIT_ASSERT(totalTotalError < 3.5);

    LOG_DEBUG("");
    LOG_DEBUG("****** jointLogMarginalLikelihood ******");

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(varianceScales); ++i) {
        LOG_DEBUG("**** variance scale = " << varianceScales[i] << " ****");

        boost::math::normal_distribution<> normal(mean, ::sqrt(varianceScales[i] * variance));
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(normal);

        CNormalMeanPrecConjugate filter(makePrior());

        double differentialEntropy = 0.0;

        TDoubleVec seedSamples;
        rng.generateNormalSamples(mean, variance, 1000, seedSamples);
        filter.addSamples(seedSamples);

        TDoubleVec scaledSamples;
        rng.generateNormalSamples(mean, varianceScales[i] * variance, 10000, scaledSamples);
        for (std::size_t j = 0u; j < scaledSamples.size(); ++j) {
            double logLikelihood = 0.0;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                 filter.jointLogMarginalLikelihood(maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight),
                                                                   TDouble1Vec(1, scaledSamples[j]),
                                                                   TDouble4Vec1Vec(1, TDouble4Vec(1, varianceScales[i])),
                                                                   logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(scaledSamples.size());

        LOG_DEBUG("differentialEntropy = " << differentialEntropy << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 0.03);
    }

    LOG_DEBUG("");
    LOG_DEBUG("****** addSamples ******");

    // This tests update with variable variance scale. In particular,
    // we update with samples from N(0,1) and N(0,5) and test that
    // the variance is correctly estimated if we compensate using a
    // variance scale.

    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0};
    unsigned int errors[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

    maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
    double variances[] = {1.0, 5.0};
    double precision = 1 / variances[0];

    for (std::size_t t = 0; t < 1000; ++t) {
        CNormalMeanPrecConjugate filter(makePrior());

        for (std::size_t i = 0u; i < boost::size(variances); ++i) {
            TDoubleVec samples;
            rng.generateNormalSamples(0.0, variances[i], 1000, samples);
            TDouble4Vec1Vec weights(samples.size(), TDouble4Vec(1, variances[i]));
            filter.addSamples(weightStyle, samples, weights);
        }

        for (std::size_t i = 0; i < boost::size(testIntervals); ++i) {
            TDoubleDoublePr confidenceInterval = filter.confidenceIntervalPrecision(testIntervals[i]);
            if (precision < confidenceInterval.first || precision > confidenceInterval.second) {
                ++errors[i];
            }
        }
    }

    for (std::size_t i = 0; i < boost::size(testIntervals); ++i) {
        double interval = 100.0 * errors[i] / 1000.0;
        LOG_DEBUG("interval = " << interval << ", expectedInterval = " << (100.0 - testIntervals[i]));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[i]), 4.0);
    }
}

CppUnit::Test* CNormalMeanPrecConjugateTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CNormalMeanPrecConjugateTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testMultipleUpdate",
                                                                                &CNormalMeanPrecConjugateTest::testMultipleUpdate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testPropagation",
                                                                                &CNormalMeanPrecConjugateTest::testPropagation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testMeanEstimation",
                                                                                &CNormalMeanPrecConjugateTest::testMeanEstimation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testPrecisionEstimation",
                                                                                &CNormalMeanPrecConjugateTest::testPrecisionEstimation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testMarginalLikelihood",
                                                                                &CNormalMeanPrecConjugateTest::testMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testMarginalLikelihoodMean",
                                                                                &CNormalMeanPrecConjugateTest::testMarginalLikelihoodMean));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testMarginalLikelihoodMode",
                                                                                &CNormalMeanPrecConjugateTest::testMarginalLikelihoodMode));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>(
        "CNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance", &CNormalMeanPrecConjugateTest::testMarginalLikelihoodVariance));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>(
        "CNormalMeanPrecConjugateTest::testSampleMarginalLikelihood", &CNormalMeanPrecConjugateTest::testSampleMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testCdf",
                                                                                &CNormalMeanPrecConjugateTest::testCdf));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples",
                                                              &CNormalMeanPrecConjugateTest::testProbabilityOfLessLikelySamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testAnomalyScore",
                                                                                &CNormalMeanPrecConjugateTest::testAnomalyScore));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testIntegerData",
                                                                                &CNormalMeanPrecConjugateTest::testIntegerData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testLowVariationData",
                                                                                &CNormalMeanPrecConjugateTest::testLowVariationData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testPersist",
                                                                                &CNormalMeanPrecConjugateTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testSeasonalVarianceScale",
                                                                                &CNormalMeanPrecConjugateTest::testSeasonalVarianceScale));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNormalMeanPrecConjugateTest>("CNormalMeanPrecConjugateTest::testCountVarianceScale",
                                                                                &CNormalMeanPrecConjugateTest::testCountVarianceScale));

    return suiteOfTests;
}
