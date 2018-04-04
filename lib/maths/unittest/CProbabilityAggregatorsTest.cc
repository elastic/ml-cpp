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

#include "CProbabilityAggregatorsTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CIntegration.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <test/CRandomNumbers.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>

using namespace ml;
using namespace maths;
using namespace test;

namespace {

using TDoubleVec = std::vector<double>;

class CGammaKernel {
public:
    CGammaKernel(const double& s, const double& x) : m_S(s), m_X(x) {}

    bool operator()(const double& u, double& result) const {
        result = std::pow(m_X - std::log(1.0 - u / m_S), m_S - 1.0);
        return true;
    }

private:
    double m_S;
    double m_X;
};

double logUpperIncompleteGamma(double s, double x) {
    if (s <= 1.0) {
        // We want to evaluate:
        //   Int_u=x,inf{ (u^(s-1) * exp(-u) }du
        //
        // Change variables to:
        //   t = s * (1 - exp(x-u))
        //
        // to get:
        //   exp(-x)/s Int_t=0,s{ (x - log(1 - t/s)) ^ (s-1) }dt
        //
        // which we integrate numerically.

        double remainder = 0.0;

        CGammaKernel kernel(s, x);
        int n = 40;
        for (int i = 0; i < n; ++i) {
            double a = s * static_cast<double>(i) / static_cast<double>(n);
            double b = s * (static_cast<double>(i) + 1.0) / static_cast<double>(n);
            double partialRemainder;
            CIntegration::gaussLegendre<CIntegration::OrderFive>(kernel, a, b, partialRemainder);
            remainder += partialRemainder;
        }

        return -x - std::log(s) + std::log(remainder);
    }

    // This uses the standard recurrence relation for the upper incomplete
    // gamma function,
    //   g(s,x) = (s - 1) * g(s-1,x) + x^(s-1) * exp(x)

    double t1 = logUpperIncompleteGamma(s - 1.0, x) + std::log(s - 1.0);
    double t2 = (s - 1.0) * std::log(x) - x;
    double normalizer = std::max(t1, t2);
    return normalizer + std::log(std::exp(t1 - normalizer) + std::exp(t2 - normalizer));
}

class CExpectedLogProbabilityOfMFromNExtremeSamples {
public:
    using TMinValueAccumulator = CBasicStatistics::COrderStatisticsHeap<double>;

    class CLogIntegrand {
    public:
        CLogIntegrand(const TDoubleVec& limits, std::size_t n, std::size_t m, std::size_t i) : m_Limits(limits), m_N(n), m_M(m), m_I(i) {}

        bool operator()(double x, double& result) const {
            result = this->evaluate(x);
            return true;
        }

    private:
        double evaluate(double x) const {
            if (m_I == m_M) {
                return static_cast<double>(m_N - m_M) * std::log(1.0 - x);
            }
            double result;
            CLogIntegrand f(m_Limits, m_N, m_M, m_I + 1u);
            CIntegration::logGaussLegendre<CIntegration::OrderTen>(f, x, m_Limits[m_I], result);
            return result;
        }

        TDoubleVec m_Limits;
        std::size_t m_N;
        std::size_t m_M;
        std::size_t m_I;
    };

public:
    CExpectedLogProbabilityOfMFromNExtremeSamples(std::size_t m) : m_P(m), m_N(0u) {}

    void add(const double& probability) {
        m_P.add(probability);
        ++m_N;
    }

    double calculate() {
        double result;
        m_P.sort();
        TDoubleVec p(m_P.begin(), m_P.end());
        CLogIntegrand f(p, m_N, p.size(), 1u);
        CIntegration::logGaussLegendre<CIntegration::OrderTen>(f, 0, p[0], result);
        result += boost::math::lgamma(static_cast<double>(m_N) + 1.0) - boost::math::lgamma(static_cast<double>(m_N - p.size()) + 1.0);
        return result;
    }

private:
    TMinValueAccumulator m_P;
    std::size_t m_N;
};
}

void CProbabilityAggregatorsTest::testJointProbabilityOfLessLikelySamples() {
    LOG_DEBUG("+------------------------------------------------------------------------+");
    LOG_DEBUG("|  CProbabilityAggregatorsTest::testJointProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+------------------------------------------------------------------------+");

    // Test case that overflows boost incomplete gamma function.

    {
        CJointProbabilityOfLessLikelySamples jointProbability;
        jointProbability.add(0.999999, 0.001);
        jointProbability.add(1.0, 1900.0);
        double probability;
        jointProbability.calculate(probability);
        LOG_DEBUG("probability = " << probability);
        CPPUNIT_ASSERT_EQUAL(1.0, probability);
    }

    // The idea of this test is to check that the probabilities
    // of seeing lower likelihood products for multiple independent
    // normal samples are correctly predicted. We also test the
    // invariant of average probability.

    CRandomNumbers rng;

    {
        const unsigned int numberSamples = 20000u;

        const double percentiles[] = {0.02, 0.1, 0.3, 0.5};

        TDoubleVec samples1;
        rng.generateNormalSamples(1.0, 3.0, numberSamples, samples1);
        boost::math::normal_distribution<> normal1(1.0, std::sqrt(3.0));

        TDoubleVec samples2;
        rng.generateNormalSamples(10.0, 15.0, numberSamples, samples2);
        boost::math::normal_distribution<> normal2(10.0, std::sqrt(15.0));

        TDoubleVec samples3;
        rng.generateNormalSamples(0.0, 1.0, numberSamples, samples3);
        boost::math::normal_distribution<> normal3(0.0, std::sqrt(1.0));

        double totalExpectedCount = 0.0;
        double totalCount = 0.0;

        for (size_t i = 0; i < boost::size(percentiles); ++i) {
            for (size_t j = 0; j < boost::size(percentiles); ++j) {
                for (size_t k = 0; k < boost::size(percentiles); ++k) {
                    LOG_DEBUG("percentile1 = " << percentiles[i] << ", percentile2 = " << percentiles[j]
                                               << ", percentile3 = " << percentiles[k]);

                    double probabilities[] = {2.0 * percentiles[i], 2.0 * percentiles[j], 2.0 * percentiles[k]};

                    CJointProbabilityOfLessLikelySamples jointProbability;
                    for (size_t l = 0; l < boost::size(probabilities); ++l) {
                        LOG_DEBUG("probability = " << probabilities[l]);
                        jointProbability.add(probabilities[l]);
                    }

                    double expectedCount;
                    CPPUNIT_ASSERT(jointProbability.calculate(expectedCount));
                    expectedCount *= static_cast<double>(numberSamples);

                    double count = 0.0;

                    double quantile1 = boost::math::quantile(normal1, percentiles[i]);
                    double quantile2 = boost::math::quantile(normal2, percentiles[j]);
                    double quantile3 = boost::math::quantile(normal3, percentiles[k]);
                    double likelihood =
                        CTools::safePdf(normal1, quantile1) * CTools::safePdf(normal2, quantile2) * CTools::safePdf(normal3, quantile3);

                    for (unsigned int sample = 0; sample < numberSamples; ++sample) {
                        double sampleLikelihood = CTools::safePdf(normal1, samples1[sample]) * CTools::safePdf(normal2, samples2[sample]) *
                                                  CTools::safePdf(normal3, samples3[sample]);
                        if (sampleLikelihood < likelihood) {
                            count += 1.0;
                        }
                    }

                    LOG_DEBUG("count = " << count << ", expectedCount = " << expectedCount);

                    double error = std::fabs(count - expectedCount) / std::max(count, expectedCount);
                    CPPUNIT_ASSERT(error < 0.2);

                    totalExpectedCount += expectedCount;
                    totalCount += count;
                }
            }
        }

        double totalError = std::fabs(totalCount - totalExpectedCount) / std::max(totalCount, totalExpectedCount);
        LOG_DEBUG("totalError = " << totalError);
        CPPUNIT_ASSERT(totalError < 0.01);
    }

    {
        TDoubleVec probabilities;
        rng.generateUniformSamples(0.0, 1.0, 100u, probabilities);
        std::fill_n(std::back_inserter(probabilities), 5u, 1e-4);
        CJointProbabilityOfLessLikelySamples expectedJointProbability;
        for (std::size_t i = 0u; i < probabilities.size(); ++i) {
            expectedJointProbability.add(probabilities[i]);

            double p;
            CPPUNIT_ASSERT(expectedJointProbability.averageProbability(p));

            CJointProbabilityOfLessLikelySamples jointProbability;
            jointProbability.add(p, static_cast<double>(i + 1));

            double expectedProbability;
            CPPUNIT_ASSERT(expectedJointProbability.calculate(expectedProbability));

            double probability;
            CPPUNIT_ASSERT(jointProbability.calculate(probability));

            LOG_DEBUG("probability = " << probability << ", expectedProbability = " << expectedProbability);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProbability, probability, 1e-5 * expectedProbability);
        }
    }
}

void CProbabilityAggregatorsTest::testLogJointProbabilityOfLessLikelySamples() {
    LOG_DEBUG("+---------------------------------------------------------------------------+");
    LOG_DEBUG("|  CProbabilityAggregatorsTest::testLogJointProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+---------------------------------------------------------------------------+");

    {
        std::ifstream ifs("testfiles/probabilities");

        CPPUNIT_ASSERT(ifs.is_open());

        CJointProbabilityOfLessLikelySamples jointProbability;
        CLogJointProbabilityOfLessLikelySamples logJointProbability;

        std::string line;
        while (std::getline(ifs, line)) {
            double probability;
            CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(line, probability));
            logJointProbability.add(probability);
            jointProbability.add(probability);
        }

        double s = jointProbability.numberSamples() / 2.0;
        double x = jointProbability.distance() / 2.0;

        double logP = logUpperIncompleteGamma(s, x) - boost::math::lgamma(s);
        LOG_DEBUG("log(p) = " << logP);

        double lowerBound, upperBound;
        CPPUNIT_ASSERT(logJointProbability.calculateLowerBound(lowerBound));
        CPPUNIT_ASSERT(logJointProbability.calculateUpperBound(upperBound));
        LOG_DEBUG("log(pu) - log(p) = " << upperBound - logP << ", log(p) - log(pl) " << logP - lowerBound);

        CPPUNIT_ASSERT(logP < upperBound);
        CPPUNIT_ASSERT(logP > lowerBound);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(upperBound, lowerBound, std::fabs(5e-6 * upperBound));
    }

    // Now test the quality of bounds near underflow.
    {

        const double p[] = {1e-1, 1e-2, 1e-3, 1e-4};
        const double expectedErrors[] = {7.7e-4, 2.6e-4, 2e-4, 1.7e-4};

        for (size_t i = 0; i < boost::size(p); ++i) {
            LOG_DEBUG("p = " << p[i]);

            CJointProbabilityOfLessLikelySamples jointProbability;
            CLogJointProbabilityOfLessLikelySamples logJointProbability;

            double error = 0.0;

            int count = 0;
            for (;;) {
                if (count >= 20) {
                    break;
                }
                jointProbability.add(p[i]);
                logJointProbability.add(p[i]);

                double probability;
                CPPUNIT_ASSERT(jointProbability.calculate(probability));
                if (probability < 10.0 * std::numeric_limits<double>::min()) {
                    ++count;

                    double s = jointProbability.numberSamples() / 2.0;
                    double x = jointProbability.distance() / 2.0;
                    LOG_DEBUG("s = " << s << ", x = " << x);

                    double logP = logUpperIncompleteGamma(s, x) - boost::math::lgamma(s);
                    LOG_DEBUG("log(p) = " << logP);

                    double lowerBound, upperBound;
                    CPPUNIT_ASSERT(logJointProbability.calculateLowerBound(lowerBound));
                    CPPUNIT_ASSERT(logJointProbability.calculateUpperBound(upperBound));
                    LOG_DEBUG("log(pu) - log(p) = " << upperBound - logP << ", log(p) - log(pl) " << logP - lowerBound);

                    CPPUNIT_ASSERT(logP < upperBound);
                    CPPUNIT_ASSERT(logP > lowerBound);

                    CPPUNIT_ASSERT_DOUBLES_EQUAL(upperBound, lowerBound, std::fabs(8e-4 * upperBound));

                    error += (upperBound - lowerBound) / std::fabs(upperBound);
                } else if (jointProbability.numberSamples() > 1.0) {
                    double s = jointProbability.numberSamples() / 2.0;
                    double x = jointProbability.distance() / 2.0;

                    double logP = logUpperIncompleteGamma(s, x) - boost::math::lgamma(s);

                    double lowerBound, upperBound;
                    CPPUNIT_ASSERT(logJointProbability.calculateLowerBound(lowerBound));
                    CPPUNIT_ASSERT(logJointProbability.calculateUpperBound(upperBound));

                    // Test the test function.
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(probability), logP, 2e-5);

                    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(probability), upperBound, 1e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(probability), lowerBound, 1e-6);
                }
            }

            error /= static_cast<double>(count);
            LOG_DEBUG("mean relative interval = " << error);
            CPPUNIT_ASSERT(error < expectedErrors[i]);
        }
    }
}

void CProbabilityAggregatorsTest::testProbabilityOfExtremeSample() {
    LOG_DEBUG("+-------------------------------------------------------------+");
    LOG_DEBUG("|  CProbabilityAggregatorsTest::testProbabilityExtremeSample  |");
    LOG_DEBUG("+-------------------------------------------------------------+");

    // The idea of this test is to check that the extreme sample
    // probability is correctly predicted.

    std::size_t sampleSizes[] = {2u, 20u, 1500u};

    double probabilities[] = {0.1, 0.05, 0.01, 0.001, 0.000001};

    CRandomNumbers rng;

    double totalError = 0.0;
    double totalProbability = 0.0;

    for (size_t i = 0; i < boost::size(sampleSizes); ++i) {
        for (size_t j = 0; j < boost::size(probabilities); ++j) {
            CProbabilityOfExtremeSample probabilityCalculator;
            for (std::size_t k = 0u; k < sampleSizes[i]; ++k) {
                // Add on a small positive number to make sure we are
                // sampling the minimum probability.
                double noise = static_cast<double>(k % 20) / 50.0;
                probabilityCalculator.add(probabilities[j] + noise);
            }

            double probability;
            CPPUNIT_ASSERT(probabilityCalculator.calculate(probability));

            LOG_DEBUG("sample size = " << sampleSizes[i] << ", extreme sample probability = " << probabilities[j]
                                       << ", probability = " << probability);

            unsigned int nTrials = 10000u;
            unsigned int count = 0;

            for (unsigned int k = 0; k < nTrials; ++k) {
                TDoubleVec samples;
                rng.generateNormalSamples(0.0, 1.0, sampleSizes[i], samples);
                boost::math::normal_distribution<> normal(0.0, std::sqrt(1.0));

                using TMinValue = CBasicStatistics::COrderStatisticsStack<double, 1u>;

                TMinValue minValue;
                for (std::size_t l = 0u; l < samples.size(); ++l) {
                    double p = 2.0 * boost::math::cdf(normal, -std::fabs(samples[l]));
                    minValue.add(p);
                }

                if (minValue[0] < probabilities[j]) {
                    ++count;
                }
            }

            double expectedProbability = static_cast<double>(count) / static_cast<double>(nTrials);
            LOG_DEBUG("count = " << count << ", expectedProbability = " << expectedProbability
                                 << ", error = " << std::fabs(probability - expectedProbability));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(probability, expectedProbability, 0.012);

            totalError += std::fabs(probability - expectedProbability);
            totalProbability += std::max(probability, expectedProbability);
        }
    }

    LOG_DEBUG("totalError = " << totalError << ", totalProbability = " << totalProbability);
    CPPUNIT_ASSERT(totalError / totalProbability < 0.01);
}

void CProbabilityAggregatorsTest::testProbabilityOfMFromNExtremeSamples() {
    LOG_DEBUG("+----------------------------------------------------------------------+");
    LOG_DEBUG("|  CProbabilityAggregatorsTest::testProbabilityOfMFromNExtremeSamples  |");
    LOG_DEBUG("+----------------------------------------------------------------------+");

    // We perform four tests:
    //   1) A test that the numerical integral is close to the
    //      closed form integral.
    //   2) That we correctly predict the probability of the event
    //      event {P(X(i)) < pi} for a range of pi.
    //   3) A test of numerical robustness.
    //   4) Problem case that was causing nan due to overflow.
    //   5) Problem case that was causing log of negative number.
    //   6) Case that underflows double.
    //   7) A problem case causing nan due to overflow of 1 / coefficient.
    //   8) Problem causing inf probability.
    //   9) Another problem case causing nan due to overflow of 1 / coefficient.
    //  10) Underflow of numerical integration.

    {
        double probabilities[] = {0.5, 0.5, 0.4, 0.02, 0.7, 0.9, 0.4, 0.2, 0.03, 0.5, 0.6};

        for (std::size_t i = 1u; i < 6u; ++i) {
            CExpectedLogProbabilityOfMFromNExtremeSamples expectedProbabilityCalculator(i);
            CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(i);

            for (std::size_t j = 0; j < boost::size(probabilities); ++j) {
                expectedProbabilityCalculator.add(probabilities[j]);
                probabilityCalculator.add(probabilities[j]);
            }

            double p1 = expectedProbabilityCalculator.calculate();
            double p2;
            CPPUNIT_ASSERT(probabilityCalculator.calculate(p2));

            LOG_DEBUG("log(probability) = " << p2 << ", expected log(probability) = " << p1);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-8 * std::fabs(std::max(p1, p2)));
        }
    }

    {
        double probabilities[] = {0.0001, 0.005, 0.01, 0.1, 0.2};

        std::size_t numberProbabilities = boost::size(probabilities);

        const std::size_t numberSamples = 50u;

        CRandomNumbers rng;

        for (std::size_t i = 2; i < 4; ++i) {
            CPPUNIT_ASSERT(i <= numberProbabilities);

            using TSizeVec = std::vector<size_t>;

            TSizeVec index(i, 0);
            for (std::size_t j = 1; j < i; ++j) {
                index[j] = j;
            }

            TSizeVec lastIndex(i, 0);
            for (std::size_t j = 0; j < i; ++j) {
                lastIndex[j] = numberProbabilities - i + j;
            }

            double totalError = 0.0;
            double totalProbability = 0.0;
            for (;;) {
                TDoubleVec extremeSampleProbabilities;
                for (std::size_t j = 0u; j < index.size(); ++j) {
                    extremeSampleProbabilities.push_back(probabilities[index[j]]);
                }
                LOG_DEBUG("extreme samples probabilities = " << core::CContainerPrinter::print(extremeSampleProbabilities));

                CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(i);

                for (std::size_t j = 0u; j < index.size(); ++j) {
                    probabilityCalculator.add(probabilities[index[j]]);
                }
                for (std::size_t j = 0u; j < numberSamples - index.size(); ++j) {
                    probabilityCalculator.add(0.3);
                }

                double p;
                CPPUNIT_ASSERT(probabilityCalculator.calculate(p));
                p = std::exp(p);

                unsigned int nTrials = 50000u;
                unsigned int count = 0;

                for (unsigned int j = 0; j < nTrials; ++j) {
                    TDoubleVec samples;
                    rng.generateNormalSamples(0.0, 1.0, numberSamples, samples);
                    boost::math::normal_distribution<> normal(0.0, std::sqrt(1.0));

                    using TMinValues = CBasicStatistics::COrderStatisticsHeap<double>;

                    TMinValues minValues(i);
                    for (std::size_t k = 0u; k < samples.size(); ++k) {
                        double p1 = 2.0 * boost::math::cdf(normal, -std::fabs(samples[k]));
                        minValues.add(p1);
                    }

                    ++count;

                    minValues.sort();
                    for (size_t k = 0; k < i; ++k) {
                        if (minValues[k] > probabilities[index[k]]) {
                            --count;
                            break;
                        }
                    }
                }

                double expectedProbability = static_cast<double>(count) / static_cast<double>(nTrials);

                double error = std::fabs(p - expectedProbability);
                double relativeError = error / std::max(p, expectedProbability);

                LOG_DEBUG("probability = " << p << ", expectedProbability = " << expectedProbability << ", error = " << error
                                           << ", relative error = " << relativeError);

                CPPUNIT_ASSERT(relativeError < 0.33);

                totalError += error;
                totalProbability += std::max(p, expectedProbability);

                if (index >= lastIndex) {
                    break;
                }

                for (std::size_t j = i; j > 0; --j) {
                    if (index[j - 1] < numberProbabilities + j - i - 1) {
                        std::size_t next = ++index[j - 1];

                        for (++j, ++next; j < i + 1; ++j, ++next) {
                            index[j - 1] = std::min(next, numberProbabilities - 1);
                        }
                        break;
                    }
                }
            }

            LOG_DEBUG("totalError = " << totalError << ", totalRelativeError = " << (totalError / totalProbability));

            CPPUNIT_ASSERT(totalError < 0.01 * totalProbability);
        }
    }

    {
        double probabilities[] = {1.90005e-6,
                                  2.09343e-5,
                                  2.36102e-5,
                                  2.36102e-4,
                                  3.21197e-4,
                                  0.104481,
                                  0.311476,
                                  0.46037,
                                  0.958691,
                                  0.144973,
                                  0.345924,
                                  0.111316,
                                  0.346185,
                                  0.993074,
                                  0.0902145,
                                  0.0902145,
                                  0.673371,
                                  0.346075,
                                  0.346025};
        std::size_t n = boost::size(probabilities);
        std::size_t numberSamples[] = {n, 10 * n, 1000 * n};

        for (std::size_t i = 1u; i < 6; ++i) {
            LOG_DEBUG("M = " << i);

            for (std::size_t j = 0; j < boost::size(numberSamples); ++j) {
                LOG_DEBUG("N = " << numberSamples[j]);

                CExpectedLogProbabilityOfMFromNExtremeSamples expectedProbabilityCalculator(i);
                CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(i);

                for (std::size_t k = 0; k < numberSamples[j]; ++k) {
                    expectedProbabilityCalculator.add(probabilities[k % n]);
                    probabilityCalculator.add(probabilities[k % n]);
                }

                double p1 = expectedProbabilityCalculator.calculate();
                double p2;
                CPPUNIT_ASSERT(probabilityCalculator.calculate(p2));

                LOG_DEBUG("log(probability) = " << p2 << ", expected log(probability) = " << p1);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-4 * std::fabs(std::max(p1, p2)));
            }
        }
    }

    {
        double probabilities[] = {
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012, 0.9917012,
            0.9917012};

        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(5);
        for (std::size_t i = 0u; i < boost::size(probabilities); ++i) {
            probabilityCalculator.add(probabilities[i]);
        }

        double p;
        CPPUNIT_ASSERT(probabilityCalculator.calculate(p));
        LOG_DEBUG("log(probability) = " << p << ", expected log(probability) = 0");
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, p, 1e-6);
    }

    {
        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(5);
        for (std::size_t i = 0u; i < 10; ++i) {
            probabilityCalculator.add(1.0 - 1e-10);
        }

        double p;
        CPPUNIT_ASSERT(probabilityCalculator.calculate(p));
        LOG_DEBUG("log(probability) = " << p << ", expected log(probability) = 0");
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, p, 1e-6);
    }

    {
        CExpectedLogProbabilityOfMFromNExtremeSamples expectedProbabilityCalculator(3);
        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(3);
        for (std::size_t i = 0u; i < 10; ++i) {
            expectedProbabilityCalculator.add(maths::CTools::smallestProbability());
            probabilityCalculator.add(maths::CTools::smallestProbability());
        }

        double p1 = expectedProbabilityCalculator.calculate();
        double p2;
        CPPUNIT_ASSERT(probabilityCalculator.calculate(p2));
        LOG_DEBUG("probability = " << p2 << ", expectedProbability = " << p1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-8 * std::fabs(std::max(p1, p2)));
    }

    {
        // Note the high tolerance for this test is due to error in the
        // numerical integration.

        CExpectedLogProbabilityOfMFromNExtremeSamples expectedProbabilityCalculator(5);
        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(5);

        double pmin[] = {0.004703117, 0.05059556, 1.0 - std::numeric_limits<double>::epsilon(), 1.0, 1.0};
        for (std::size_t i = 0; i < boost::size(pmin); ++i) {
            probabilityCalculator.add(pmin[i]);
            expectedProbabilityCalculator.add(pmin[i]);
        }
        for (std::size_t i = boost::size(pmin); i < 22; ++i) {
            probabilityCalculator.add(1.0);
            expectedProbabilityCalculator.add(1.0);
        }

        double p1 = expectedProbabilityCalculator.calculate();
        double p2;
        CPPUNIT_ASSERT(probabilityCalculator.calculate(p2));
        LOG_DEBUG("probability = " << p2 << ", expectedProbability = " << p1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-8 * std::fabs(std::max(p1, p2)));
    }

    {
        double p[] = {0.000234811, 1 - 2e-16, 1 - 1.5e-16};

        CExpectedLogProbabilityOfMFromNExtremeSamples expectedProbabilityCalculator(3);
        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(3);

        expectedProbabilityCalculator.add(p[0]);
        expectedProbabilityCalculator.add(p[1]);
        expectedProbabilityCalculator.add(p[2]);
        probabilityCalculator.add(p[0]);
        probabilityCalculator.add(p[1]);
        probabilityCalculator.add(p[2]);

        for (std::size_t i = 0u; i < 19; ++i) {
            expectedProbabilityCalculator.add(1.0);
            probabilityCalculator.add(1.0);
        }

        double p1 = expectedProbabilityCalculator.calculate();
        double p2;
        probabilityCalculator.calculate(p2);
        LOG_DEBUG("probability = " << p2 << ", expectedProbability = " << p1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-8 * std::fabs(std::max(p1, p2)));
    }

    {
        double probabilities[] = {0.08528782661735056, 0.3246988524001009, 0.5428693993904167, 0.9999999999999999, 0.9999999999999999};

        CExpectedLogProbabilityOfMFromNExtremeSamples expectedProbabilityCalculator(5);
        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(5);

        for (std::size_t i = 0u; i < boost::size(probabilities); ++i) {
            expectedProbabilityCalculator.add(probabilities[i]);
            probabilityCalculator.add(probabilities[i]);
        }
        for (std::size_t i = 0u; i < 19; ++i) {
            expectedProbabilityCalculator.add(1.0);
            probabilityCalculator.add(1.0);
        }

        double p1 = expectedProbabilityCalculator.calculate();
        double p2;
        probabilityCalculator.calculate(p2);
        LOG_DEBUG("probability = " << p2 << ", expectedProbability = " << p1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-8 * std::fabs(std::max(p1, p2)));
    }

    {
        double probabilities[] = {3.622684004911715e-76, 3.622684004911715e-76, 0.1534837115755979, 0.1608058997234747, 0.5143979767475618};

        CLogProbabilityOfMFromNExtremeSamples probabilityCalculator(5);
        for (std::size_t i = 0; i < 21402; ++i) {
            probabilityCalculator.add(1.0);
        }
        for (std::size_t i = 0; i < 5; ++i) {
            probabilityCalculator.add(probabilities[i]);
        }
        double p1 = -306.072;
        double p2;
        probabilityCalculator.calculate(p2);
        LOG_DEBUG("probability = " << p2 << ", expectedProbability = " << p1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(p1, p2, 1e-3);
    }
}

CppUnit::Test* CProbabilityAggregatorsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CProbabilityAggregatorsTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CProbabilityAggregatorsTest>("CProbabilityAggregatorsTest::testJointProbabilityOfLessLikelySamples",
                                                             &CProbabilityAggregatorsTest::testJointProbabilityOfLessLikelySamples));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CProbabilityAggregatorsTest>("CProbabilityAggregatorsTest::testLogJointProbabilityOfLessLikelySamples",
                                                             &CProbabilityAggregatorsTest::testLogJointProbabilityOfLessLikelySamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProbabilityAggregatorsTest>(
        "CProbabilityAggregatorsTest::testProbabilityOfExtremeSample", &CProbabilityAggregatorsTest::testProbabilityOfExtremeSample));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CProbabilityAggregatorsTest>("CProbabilityAggregatorsTest::testProbabilityOfMFromNExtremeSamples",
                                                             &CProbabilityAggregatorsTest::testProbabilityOfMFromNExtremeSamples));

    return suiteOfTests;
}
