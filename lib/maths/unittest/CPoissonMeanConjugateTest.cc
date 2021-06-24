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
#include <maths/CPoissonMeanConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/distributions/poisson.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

BOOST_AUTO_TEST_SUITE(CPoissonMeanConjugateTest)

using namespace ml;
using namespace handy_typedefs;

namespace {

using TUIntVec = std::vector<unsigned int>;
using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using CPoissonMeanConjugate = CPriorTestInterfaceMixin<maths::CPoissonMeanConjugate>;
}

BOOST_AUTO_TEST_CASE(testMultipleUpdate) {
    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    using TEqual = maths::CEqualWithTolerance<double>;

    const double rate = 5.0;

    test::CRandomNumbers rng;
    TUIntVec samples_;
    rng.generatePoissonSamples(rate, 100, samples_);
    TDoubleVec samples;
    for (std::size_t i = 0; i < samples_.size(); ++i) {
        samples.push_back(static_cast<double>(samples_[i]));
    }

    {
        CPoissonMeanConjugate filter1(CPoissonMeanConjugate::nonInformativePrior());
        CPoissonMeanConjugate filter2(filter1);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            filter1.addSamples(TDouble1Vec(1, samples[i]));
        }
        filter2.addSamples(samples);

        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }

    // Test with variance scale.

    {
        CPoissonMeanConjugate filter1(CPoissonMeanConjugate::nonInformativePrior());
        CPoissonMeanConjugate filter2(filter1);

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

    {
        CPoissonMeanConjugate filter1(CPoissonMeanConjugate::nonInformativePrior());
        CPoissonMeanConjugate filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;
        for (std::size_t j = 0; j < count; ++j) {
            filter1.addSamples(TDouble1Vec(1, x));
        }
        filter2.addSamples({x}, {maths_t::countWeight(10.0)});

        LOG_DEBUG(<< filter1.print() << "\nvs" << filter2.print());
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
        BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
    }
}

BOOST_AUTO_TEST_CASE(testPropagation) {
    // Test that propagation doesn't affect the expected values
    // of likelihood mean.

    const double eps = 1e-7;

    test::CRandomNumbers rng;

    TUIntVec samples;
    rng.generatePoissonSamples(1.0, 500, samples);

    CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior(0.0, 0.1));

    for (std::size_t i = 0; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
    }

    double mean = filter.marginalLikelihoodMean();

    filter.propagateForwardsByTime(5.0);

    double propagatedMean = filter.marginalLikelihoodMean();

    LOG_DEBUG(<< "mean = " << mean << ", propagatedMean = " << propagatedMean);

    BOOST_REQUIRE_CLOSE_ABSOLUTE(mean, propagatedMean, eps);
}

BOOST_AUTO_TEST_CASE(testMeanEstimation) {
    // We are going to test that we correctly estimate a distribution
    // for the mean of the Poisson process by checking that the true
    // mean of a Poisson process lies in various confidence intervals
    // the correct percentage of the times.

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 500;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        double errors[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        for (unsigned int test = 0; test < nTests; ++test) {
            double rate = test + 1;
            TUIntVec samples;
            rng.generatePoissonSamples(rate, 500, samples);

            CPoissonMeanConjugate filter(
                CPoissonMeanConjugate::nonInformativePrior(0.0, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[j])));
                filter.propagateForwardsByTime(1.0);
            }

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePr confidenceInterval =
                    filter.meanConfidenceInterval(testIntervals[j]);

                if (rate < confidenceInterval.first || rate > confidenceInterval.second) {
                    errors[j] += 1.0;
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            double interval = 100.0 * errors[j] / static_cast<double>(nTests);

            LOG_DEBUG(<< "interval = " << interval
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

BOOST_AUTO_TEST_CASE(testMarginalLikelihood) {
    {
        // Check that the marginal likelihood and c.d.f. agree for some
        // test data and that the c.d.f. <= 1.

        const double rate = 2.0;
        test::CRandomNumbers rng;
        TUIntVec samples;
        rng.generatePoissonSamples(rate, 500, samples);

        const double epsilon = 1e-9;

        const double decayRates[] = {0.0, 0.001, 0.01};

        for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
            CPoissonMeanConjugate filter(
                CPoissonMeanConjugate::nonInformativePrior(0.0, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[j])));
                filter.propagateForwardsByTime(1.0);
            }

            double cdf = 0.0;
            for (unsigned int x = 0; x < 20; ++x) {
                double logLikelihood = 0.0;
                TDouble1Vec sample(1, static_cast<double>(x));
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    filter.jointLogMarginalLikelihood(sample, logLikelihood));
                cdf += std::exp(logLikelihood);

                double lb, ub;
                BOOST_TEST_REQUIRE(filter.minusLogJointCdf(sample, lb, ub));
                BOOST_REQUIRE_EQUAL(lb, ub);
                double minusLogCdf = (lb + ub) / 2.0;

                LOG_TRACE(<< "sample = " << x << ", -log(cdf) = " << (-std::log(cdf))
                          << ", minusLogCdf = " << minusLogCdf);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(minusLogCdf, -std::log(cdf), epsilon);
                BOOST_TEST_REQUIRE(minusLogCdf >= 0.0);
            }
        }
    }

    {
        // Now test a range of priors.

        const double shapes[] = {25.0, 80.0, 600.0, 1200.0};
        const double rates[] = {5.0, 4.0, 10.0, 3.0};
        BOOST_TEST_REQUIRE(boost::size(shapes) == boost::size(rates));

        // We'll sample the c.d.f. at mean -2, -1, 0, 1 and 2 s.t.d.
        const double sampleStds[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

        for (std::size_t i = 0; i < boost::size(shapes); ++i) {
            CPoissonMeanConjugate filter(
                maths::CPoissonMeanConjugate(0.0, shapes[i], rates[i]));

            for (std::size_t j = 0; j < boost::size(sampleStds); ++j) {
                double mean = filter.marginalLikelihoodMean();
                unsigned int sample =
                    static_cast<unsigned int>(mean + sampleStds[j] * std::sqrt(mean));

                double lb = 0.0, ub = 0.0;
                BOOST_TEST_REQUIRE(filter.minusLogJointCdf(
                    TDouble1Vec(1, static_cast<double>(sample)), lb, ub));
                BOOST_REQUIRE_EQUAL(lb, ub);
                double minusLogCdf = (lb + ub) / 2.0;
                BOOST_TEST_REQUIRE(minusLogCdf >= 0.0);

                double cdf = 0.0;
                for (unsigned int x = 0; x <= sample; ++x) {
                    double logLikelihood = 0.0;
                    BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                        filter.jointLogMarginalLikelihood(
                                            TDouble1Vec(1, static_cast<double>(x)), logLikelihood));
                    cdf += std::exp(logLikelihood);
                    cdf = std::min(cdf, 1.0);
                }

                LOG_TRACE(<< "-log(cdf) = " << -std::log(cdf)
                          << ", minusLogCdf = " << minusLogCdf);

                // We'll tolerate a 5% error in the -log(c.d.f.) since
                // we're approximating for large mean.
                BOOST_REQUIRE_CLOSE_ABSOLUTE(minusLogCdf, -std::log(cdf),
                                             -0.05 * std::log(cdf));
            }
        }
    }

    {
        // Test that the sample mean of the log likelihood tends to the
        // expected log likelihood for a poisson distribution (uniform
        // law of large numbers), which is just the differential entropy
        // of a poisson R.V.

        const double rate = 3.0;

        boost::math::poisson_distribution<> poisson(rate);
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(poisson);

        test::CRandomNumbers rng;

        CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());

        double differentialEntropy = 0.0;

        TUIntVec seedSamples;
        rng.generatePoissonSamples(rate, 100, seedSamples);
        for (std::size_t i = 0; i < seedSamples.size(); ++i) {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(seedSamples[i])));
        }

        TUIntVec samples;
        rng.generatePoissonSamples(rate, 5000, samples);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            TDouble1Vec sample(1, static_cast<double>(samples[i]));
            filter.addSamples(sample);
            double logLikelihood = 0.0;
            BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                filter.jointLogMarginalLikelihood(sample, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(samples.size());

        LOG_DEBUG(<< "differentialEntropy = " << differentialEntropy
                  << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        BOOST_TEST_REQUIRE(std::fabs(differentialEntropy - expectedDifferentialEntropy) < 0.01);
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMode) {
    // Test that the marginal likelihood mode is what we'd expect
    // with variances variance scales.

    const double rates[] = {0.1, 5.0, 100.0};
    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
                                     2.0, 2.5, 3.0, 4.0, 5.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(rates); ++i) {
        LOG_DEBUG(<< "*** rate = " << rates[i] << " ***");

        boost::math::poisson_distribution<> poisson(rates[i]);

        CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());
        TUIntVec samples;
        rng.generatePoissonSamples(rates[i], 1000, samples);
        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[j])));
        }

        maths_t::TDoubleWeightsAry weight(maths_t::CUnitWeights::UNIT);

        for (std::size_t j = 0; j < boost::size(varianceScales); ++j) {
            double vs = varianceScales[j];
            maths_t::setCountVarianceScale(vs, weight);
            double expectedMode = boost::math::mode(poisson);
            LOG_DEBUG(<< "marginalLikelihoodMode = " << filter.marginalLikelihoodMode(weight)
                      << ", expectedMode = " << expectedMode);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode,
                                         filter.marginalLikelihoodMode(weight), 1.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodVariance) {
    const double rates[] = {0.1, 5.0, 100.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(rates); ++i) {
        LOG_DEBUG(<< "*** rate = " << rates[i] << " ***");
        CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());

        TUIntVec seedSamples;
        rng.generatePoissonSamples(rates[i], 5, seedSamples);
        for (std::size_t j = 0; j < seedSamples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(seedSamples[j])));
        }

        TUIntVec samples;
        rng.generatePoissonSamples(rates[i], 100, samples);

        TMeanAccumulator relativeError;
        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[j])));
            double expectedVariance;
            BOOST_TEST_REQUIRE(filter.marginalLikelihoodVarianceForTest(expectedVariance));
            LOG_TRACE(<< "marginalLikelihoodVariance = " << filter.marginalLikelihoodVariance()
                      << ", expectedVariance = " << expectedVariance);

            // The error is at the precision of the numerical integration.
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedVariance,
                                         filter.marginalLikelihoodVariance(),
                                         0.3 * expectedVariance);

            relativeError.add(std::fabs(expectedVariance - filter.marginalLikelihoodVariance()) /
                              expectedVariance);
        }

        LOG_DEBUG(<< "relativeError = " << maths::CBasicStatistics::mean(relativeError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(relativeError) < 0.16);
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

    const double rates[] = {5.0, 200.0};

    const double eps = 1e-3;

    for (std::size_t i = 0; i < boost::size(rates); ++i) {
        test::CRandomNumbers rng;

        TUIntVec samples;
        rng.generatePoissonSamples(rates[i], 50, samples);

        CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());

        TDouble1Vec sampled;

        TMeanAccumulator meanVarError;

        std::size_t numberSampled = 20;
        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, samples[j]));

            sampled.clear();
            filter.sampleMarginalLikelihood(numberSampled, sampled);
            BOOST_REQUIRE_EQUAL(numberSampled, sampled.size());

            TMeanVarAccumulator sampledMomemts;
            sampledMomemts = std::for_each(sampled.begin(), sampled.end(), sampledMomemts);

            LOG_DEBUG(<< "expectedMean = " << filter.marginalLikelihoodMean() << ", sampledMean = "
                      << maths::CBasicStatistics::mean(sampledMomemts));
            LOG_DEBUG(<< "expectedMean = " << filter.marginalLikelihoodVariance() << ", sampledVariance = "
                      << maths::CBasicStatistics::variance(sampledMomemts));

            BOOST_REQUIRE_CLOSE_ABSOLUTE(filter.marginalLikelihoodMean(),
                                         maths::CBasicStatistics::mean(sampledMomemts), 1e-8);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(filter.marginalLikelihoodVariance(),
                                         maths::CBasicStatistics::variance(sampledMomemts),
                                         0.15 * filter.marginalLikelihoodVariance());
            meanVarError.add(std::fabs(filter.marginalLikelihoodVariance() -
                                       maths::CBasicStatistics::variance(sampledMomemts)) /
                             filter.marginalLikelihoodVariance());

            std::sort(sampled.begin(), sampled.end());
            for (std::size_t k = 3; k < sampled.size(); ++k) {
                double q = 100.0 * static_cast<double>(k) / static_cast<double>(numberSampled);

                double expectedQuantile;
                BOOST_TEST_REQUIRE(filter.marginalLikelihoodQuantileForTest(
                    q, eps, expectedQuantile));

                LOG_TRACE(<< "quantile = " << q << ", x_quantile = " << expectedQuantile
                          << ", quantile range = [" << sampled[k - 3] << ","
                          << sampled[k] << "]");

                // Because the c.d.f. function for discrete R.V.s includes
                // the value of the p.d.f. the interval that contains the
                // k'th quantile extends below the k'th sample. Depending
                // on the value of the p.d.f. this can include multiple
                // samples. For example, if the first non-zero p.d.f. value
                // equals 0.4 then that single point contains the first to
                // fortieth percentile points of the distribution.

                BOOST_TEST_REQUIRE(expectedQuantile >= sampled[k - 3] - 0.5);
                BOOST_TEST_REQUIRE(expectedQuantile <= sampled[k] + 0.5);
            }
        }

        LOG_DEBUG(<< "mean variance error = " << maths::CBasicStatistics::mean(meanVarError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanVarError) < 0.05);
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihoodInSupportBounds) {
    // Here we test that the retrieved samples are
    // a) positive
    // b) monotonically increasing
    // c) No ERROR log messages are generated
    // The input sample and weight are chosen to be ones that generate
    // an initial sample that is less than an offset value of 0.0

    const char* logFile = "test.log";

    std::remove(logFile);
    // log at level ERROR only
    BOOST_TEST_REQUIRE(ml::core::CLogger::instance().reconfigureFromFile(
        "testfiles/testLogErrors.boost.log.ini"));

    CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());

    TDouble1Vec sampleVec(1, 283.3232);
    maths_t::TDoubleWeightsAry1Vec weightVec(1, maths_t::countWeight(0.0206505));

    filter.addSamples(sampleVec, weightVec);

    std::size_t numberSampled = 50;
    TDouble1Vec sampled;

    filter.sampleMarginalLikelihood(numberSampled, sampled);

    double prevSample{0.0};
    for (double sample : sampled) {
        BOOST_TEST_REQUIRE(sample > prevSample);
        BOOST_TEST_REQUIRE(sample >= 0.0);
        prevSample = sample;
    }

    // Revert to the default logger settings
    ml::core::CLogger::instance().reset();

    std::ifstream log(logFile);
    // Boost.Log only creates files when the first message is logged,
    // and here we're asserting no messages logged
    if (log.is_open()) {
        char line[256];
        while (log.getline(line, 256)) {
            LOG_DEBUG(<< "Got '" << line << "'");
            BOOST_TEST_REQUIRE(false);
        }
        log.close();
        std::remove(logFile);
    }
}

BOOST_AUTO_TEST_CASE(testCdf) {
    // Test error cases.
    //
    // Test some invariants:
    //   "cdf" + "cdf complement" = 1
    //    cdf x for x < 0 = 1
    //    cdf complement x for x < 0 = 0

    const double rate = 5.0;
    const std::size_t n[] = {20u, 80u};

    test::CRandomNumbers rng;

    CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());

    for (std::size_t i = 0; i < boost::size(n); ++i) {
        TUIntVec samples;
        rng.generatePoissonSamples(rate, n[i], samples);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, samples[j]));
        }

        double lb, ub;
        BOOST_TEST_REQUIRE(!filter.minusLogJointCdf(TDouble1Vec(), lb, ub));
        BOOST_TEST_REQUIRE(!filter.minusLogJointCdfComplement(TDouble1Vec(), lb, ub));

        BOOST_TEST_REQUIRE(filter.minusLogJointCdf(TDouble1Vec(1, -1.0), lb, ub));
        double f = (lb + ub) / 2.0;
        BOOST_TEST_REQUIRE(filter.minusLogJointCdfComplement(TDouble1Vec(1, -1.0), lb, ub));
        double fComplement = (lb + ub) / 2.0;
        LOG_DEBUG(<< "log(F(x)) = " << -f << ", log(1 - F(x)) = " << fComplement);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(std::log(std::numeric_limits<double>::min()), -f, 1e-10);
        BOOST_REQUIRE_EQUAL(1.0, std::exp(-fComplement));

        for (std::size_t j = 1; j < 500; ++j) {
            double x = static_cast<double>(j) / 2.0;

            BOOST_TEST_REQUIRE(filter.minusLogJointCdf(TDouble1Vec(1, x), lb, ub));
            f = (lb + ub) / 2.0;
            BOOST_TEST_REQUIRE(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lb, ub));
            fComplement = (lb + ub) / 2.0;

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

    const double rates[] = {0.1, 10.0, 50.0};
    const double vs[] = {0.5, 1.0, 2.0};

    test::CRandomNumbers rng;

    TMeanAccumulator meanError;

    for (size_t i = 0; i < boost::size(rates); ++i) {
        LOG_DEBUG(<< "rate = " << rates[i]);

        TUIntVec samples;
        rng.generatePoissonSamples(rates[i], 1000, samples);

        CPoissonMeanConjugate filter(CPoissonMeanConjugate::nonInformativePrior());
        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[j])));
        }

        double mean = filter.priorMean();

        TDoubleVec likelihoods;
        for (std::size_t j = 0; j < samples.size(); ++j) {
            double likelihood;
            filter.jointLogMarginalLikelihood(TDouble1Vec(1, samples[j]), likelihood);
            likelihoods.push_back(likelihood);
        }
        std::sort(likelihoods.begin(), likelihoods.end());

        boost::math::poisson_distribution<> poisson(mean);
        for (std::size_t k = 1; k < 10; ++k) {
            double x = boost::math::quantile(poisson, static_cast<double>(k) / 10.0);

            TDouble1Vec sample(1, x);
            double fx;
            filter.jointLogMarginalLikelihood(sample, fx);

            double px = static_cast<double>(std::upper_bound(likelihoods.begin(),
                                                             likelihoods.end(), fx) -
                                            likelihoods.begin()) /
                        static_cast<double>(likelihoods.size());

            double lb, ub;
            filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lb, ub);

            double ssd = std::sqrt(px * (1.0 - px) / static_cast<double>(samples.size()));

            LOG_TRACE(<< "x = " << x << ", expected P(x) = " << px << ", actual P(x) = "
                      << (lb + ub) / 2.0 << " sample sd = " << ssd);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(px, (lb + ub) / 2.0, 8.0 * ssd);

            meanError.add(std::fabs(px - (lb + ub) / 2.0));
        }

        for (std::size_t k = 0; k < boost::size(vs); ++k) {
            double mode = filter.marginalLikelihoodMode(
                maths_t::countVarianceScaleWeight(vs[k]));
            double ss[] = {0.9 * mode, 1.1 * mode};

            LOG_DEBUG(<< "vs = " << vs[k] << ", mode = " << mode);

            double lb, ub;
            maths_t::ETail tail;

            if (mode > 0.0) {
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

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.02);
}

BOOST_AUTO_TEST_CASE(testAnomalyScore) {
    // This test pushes 500 samples through the filter and adds in
    // anomalous signals in the bins at 30, 120, 300 and 420 with
    // magnitude 4, 5, 10 and 15 standard deviations, respectively,
    // and checks the anomaly score has:
    //   1) high probability of detecting the anomalies, and
    //   2) a very low rate of false positives.

    const double decayRates[] = {0.0, 0.001, 0.1};

    const double processRates[] = {3.0, 15.0, 200.0};

    const double threshold = 0.02;

    const unsigned int anomalyTimes[] = {30u, 120u, 300u, 420u};
    const double anomalies[] = {4.0, 5.0, 10.0, 15.0, 0.0};

    test::CRandomNumbers rng;

    unsigned int test = 0;

    //std::ofstream file;
    //file.open("results.m");

    double totalFalsePositiveRate = 0.0;
    std::size_t totalPositives[] = {0u, 0u, 0u};

    for (std::size_t i = 0; i < boost::size(processRates); ++i) {
        LOG_DEBUG(<< "processRate = " << processRates[i]);

        boost::math::poisson_distribution<> poisson(processRates[i]);

        TUIntVec samples;
        rng.generatePoissonSamples(processRates[i], 500, samples);

        for (std::size_t j = 0; j < boost::size(decayRates); ++j) {
            CPoissonMeanConjugate filter(
                CPoissonMeanConjugate::nonInformativePrior(0.0, decayRates[j]));

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
                     boost::math::standard_deviation(poisson));

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
            std::set_difference(candidateAnomalies.begin(), candidateAnomalies.end(),
                                std::begin(anomalyTimes), std::end(anomalyTimes),
                                std::back_inserter(falsePositives));

            double falsePositiveRate = static_cast<double>(falsePositives.size()) /
                                       static_cast<double>(samples.size());

            totalFalsePositiveRate += falsePositiveRate;

            TUIntVec positives;
            std::set_intersection(candidateAnomalies.begin(), candidateAnomalies.end(),
                                  std::begin(anomalyTimes), std::end(anomalyTimes),
                                  std::back_inserter(positives));

            LOG_DEBUG(<< "falsePositiveRate = " << falsePositiveRate
                      << ", positives = " << positives.size());

            // False alarm rate should be less than 0.4%.
            BOOST_TEST_REQUIRE(falsePositiveRate <= 0.02);

            // Should detect at least the three biggest anomalies.
            BOOST_TEST_REQUIRE(positives.size() >= 3u);

            totalPositives[j] += positives.size();
        }
    }

    totalFalsePositiveRate /= static_cast<double>(test);
    LOG_DEBUG(<< "totalFalsePositiveRate = " << totalFalsePositiveRate);

    for (std::size_t i = 0; i < boost::size(totalPositives); ++i) {
        LOG_DEBUG(<< "positives = " << totalPositives[i]);

        // Should detect all but one anomaly.
        BOOST_TEST_REQUIRE(totalPositives[i] >= 11u);
    }

    // Total false alarm rate should be less than 0.2%.
    BOOST_TEST_REQUIRE(totalFalsePositiveRate <= 0.004);
}

BOOST_AUTO_TEST_CASE(testOffset) {
    // The idea of this test is to check that the offset correctly cancels
    // out a translation applied to a log-normally distributed data set.

    const double offsets[] = {-0.5, 0.5};
    const double decayRates[] = {0.0, 0.001, 0.01};

    const double rate = 4.0;

    const double eps = 1e-8;

    test::CRandomNumbers rng;

    TUIntVec samples;
    rng.generatePoissonSamples(rate, 100, samples);

    for (std::size_t i = 0; i < boost::size(offsets); ++i) {
        for (std::size_t j = 0; j < boost::size(decayRates); ++j) {
            CPoissonMeanConjugate filter1(CPoissonMeanConjugate::nonInformativePrior(
                offsets[i], decayRates[j]));
            CPoissonMeanConjugate filter2(
                CPoissonMeanConjugate::nonInformativePrior(0.0, decayRates[j]));

            for (std::size_t k = 0; k < samples.size(); ++k) {
                TDouble1Vec offsetSample(1, samples[k] - offsets[i]);
                filter1.addSamples(offsetSample);
                filter1.propagateForwardsByTime(1.0);

                double x = samples[k];
                TDouble1Vec sample(1, x);
                filter2.addSamples(sample);
                filter2.propagateForwardsByTime(1.0);

                double likelihood1;
                filter1.jointLogMarginalLikelihood(offsetSample, likelihood1);
                double lb1, ub1;
                filter1.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                       offsetSample, lb1, ub1);
                BOOST_REQUIRE_EQUAL(lb1, ub1);
                double probability1 = (lb1 + ub1) / 2.0;

                double likelihood2;
                filter2.jointLogMarginalLikelihood(sample, likelihood2);
                double lb2, ub2;
                filter2.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                       sample, lb2, ub2);
                BOOST_REQUIRE_EQUAL(lb2, ub2);
                double probability2 = (lb2 + ub2) / 2.0;

                BOOST_REQUIRE_CLOSE_ABSOLUTE(likelihood1, likelihood2, eps);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(probability1, probability2, eps);
            }

            using TEqual = maths::CEqualWithTolerance<double>;
            TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, eps);
            BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    const double rate = 5.0;

    test::CRandomNumbers rng;

    TUIntVec samples;
    rng.generatePoissonSamples(rate, 100, samples);

    maths::CPoissonMeanConjugate origFilter(CPoissonMeanConjugate::nonInformativePrior());
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

    LOG_DEBUG(<< "Poisson mean conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter.
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, decayRate + 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    maths::CPoissonMeanConjugate restoredFilter(params, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    BOOST_REQUIRE_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same
    // as the original.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testNegativeSample) {
    // Test that we recover roughly the same distribution after adjusting
    // the offset. The idea of this test is to run two priors side by side,
    // one with a large enough offset that it never needs to adjust the
    // offset and the other which will adjust and check that we get broadly
    // similar distributions at the end.

    const double rate = 4.0;

    test::CRandomNumbers rng;

    TUIntVec samples_;
    rng.generatePoissonSamples(rate, 100, samples_);
    TDoubleVec samples;
    samples.reserve(samples_.size());
    for (std::size_t i = 0; i < samples_.size(); ++i) {
        samples.push_back(static_cast<double>(samples_[i]));
    }

    CPoissonMeanConjugate filter1 = CPoissonMeanConjugate::nonInformativePrior();
    CPoissonMeanConjugate filter2 = CPoissonMeanConjugate::nonInformativePrior(0.5);

    filter1.addSamples(samples);
    filter2.addSamples(samples);

    TDouble1Vec negative(1, -0.49);
    filter1.addSamples(negative);
    filter2.addSamples(negative);

    BOOST_REQUIRE_EQUAL(filter1.numberSamples(), filter2.numberSamples());

    using TEqual = maths::CEqualWithTolerance<double>;
    TEqual equal(maths::CToleranceTypes::E_RelativeTolerance, 0.002);
    BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
}

BOOST_AUTO_TEST_SUITE_END()
