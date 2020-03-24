/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>
#include <maths/CStatisticalTests.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <numeric>

BOOST_AUTO_TEST_SUITE(CSamplingTest)

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

using namespace ml;

namespace {

using TDoubleVecVec = std::vector<TDoubleVec>;

double multinomialProbability(const TDoubleVec& probabilities, const TSizeVec& counts) {
    std::size_t n = std::accumulate(counts.begin(), counts.end(), std::size_t(0));
    double logP = std::lgamma(static_cast<double>(n + 1));
    for (std::size_t i = 0u; i < counts.size(); ++i) {
        double ni = static_cast<double>(counts[i]);
        if (ni > 0.0) {
            logP += ni * std::log(probabilities[i]) - std::lgamma(ni + 1.0);
        }
    }
    return std::exp(logP);
}

namespace test_detail {

//! Subtract two vectors.
TDoubleVec minus(const TDoubleVec& v1, const TDoubleVec& v2) {
    TDoubleVec result;
    for (std::size_t i = 0u; i < v1.size(); ++i) {
        result.push_back(v1[i] - v2[i]);
    }
    return result;
}

//! Subtract two matrices.
TDoubleVecVec minus(const TDoubleVecVec& m1, const TDoubleVecVec& m2) {
    TDoubleVecVec result;
    for (std::size_t i = 0u; i < m1.size(); ++i) {
        result.push_back(TDoubleVec());
        for (std::size_t j = 0u; j < m1[i].size(); ++j) {
            result.back().push_back(m1[i][j] - m2[i][j]);
        }
    }
    return result;
}

//! Compute the outer product of two vectors.
TDoubleVecVec outer(const TDoubleVec& v1, const TDoubleVec& v2) {
    TDoubleVecVec result;
    for (std::size_t i = 0u; i < v1.size(); ++i) {
        result.push_back(TDoubleVec());
        for (std::size_t j = 0u; j < v2.size(); ++j) {
            result.back().push_back(v1[i] * v2[j]);
        }
    }
    return result;
}

//! Add two matrices.
void add(const TDoubleVecVec& m1, TDoubleVecVec& m2) {
    for (std::size_t i = 0u; i < m1.size(); ++i) {
        for (std::size_t j = 0u; j < m1[i].size(); ++j) {
            m2[i][j] += m1[i][j];
        }
    }
}

//! Divide a matrix by a constant.
void divide(TDoubleVecVec& m, double c) {
    for (std::size_t i = 0u; i < m.size(); ++i) {
        for (std::size_t j = 0u; j < m[i].size(); ++j) {
            m[i][j] /= c;
        }
    }
}

//! Euclidean norm of a vector.
double euclidean(const TDoubleVec& v) {
    double result = 0.0;
    for (std::size_t i = 0u; i < v.size(); ++i) {
        result += v[i] * v[i];
    }
    return std::sqrt(result);
}

//! Frobenius norm of a matrix.
double frobenius(const TDoubleVecVec& m) {
    double result = 0.0;
    for (std::size_t i = 0u; i < m.size(); ++i) {
        for (std::size_t j = 0u; j < m.size(); ++j) {
            result += m[i][j] * m[i][j];
        }
    }
    return std::sqrt(result);
}
}
}

BOOST_AUTO_TEST_CASE(testMultinomialSample) {
    using TSizeVecDoubleMap = std::map<TSizeVec, double>;
    using TSizeVecDoubleMapCItr = TSizeVecDoubleMap::const_iterator;

    maths::CSampling::seed();

    double probabilities_[] = {0.4, 0.25, 0.2, 0.15};

    TDoubleVec probabilities(std::begin(probabilities_), std::end(probabilities_));

    TSizeVecDoubleMap empiricalProbabilities;

    std::size_t n = 1000000u;

    TSizeVec sample;
    for (std::size_t i = 0u; i < n; ++i) {
        maths::CSampling::multinomialSampleFast(probabilities, 20, sample);
        empiricalProbabilities[sample] += 1.0 / static_cast<double>(n);
    }

    double error = 0.0;
    double pTotal = 0.0;
    for (TSizeVecDoubleMapCItr pItr = empiricalProbabilities.begin();
         pItr != empiricalProbabilities.end(); ++pItr) {
        LOG_DEBUG(<< "counts = " << core::CContainerPrinter::print(pItr->first));
        BOOST_REQUIRE_EQUAL(size_t(20), std::accumulate(pItr->first.begin(),
                                                        pItr->first.end(), size_t(0)));

        double p = multinomialProbability(probabilities, pItr->first);
        double pe = pItr->second;
        LOG_DEBUG(<< "p  = " << p << ", pe = " << pe);
        BOOST_TEST_REQUIRE(std::fabs(pe - p) < std::max(0.27 * p, 3e-5));
        error += std::fabs(pe - p);
        pTotal += p;
    }

    LOG_DEBUG(<< "pTotal = " << pTotal << ", error = " << error);
    BOOST_TEST_REQUIRE(error < 0.02 * pTotal);
}

BOOST_AUTO_TEST_CASE(testMultivariateNormalSample) {
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    maths::CSampling::seed();

    {
        double m[] = {1.0, 3.0, 5.0};
        TDoubleVec m_(std::begin(m), std::end(m));
        double C[][3] = {{3.0, 1.0, 0.1}, {1.0, 2.0, -0.3}, {0.1, -0.3, 1.0}};
        TDoubleVecVec C_;
        C_.push_back(TDoubleVec(std::begin(C[0]), std::end(C[0])));
        C_.push_back(TDoubleVec(std::begin(C[1]), std::end(C[1])));
        C_.push_back(TDoubleVec(std::begin(C[2]), std::end(C[2])));

        TDoubleVecVec samples;
        maths::CSampling::multivariateNormalSample(m_, C_, 1000, samples);

        TMeanAccumulator mean[3];
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            mean[0].add(samples[i][0]);
            mean[1].add(samples[i][1]);
            mean[2].add(samples[i][2]);
        }

        TDoubleVec mean_;
        for (std::size_t i = 0u; i < 3; ++i) {
            mean_.push_back(maths::CBasicStatistics::mean(mean[i]));
        }
        LOG_DEBUG(<< "actual mean = " << core::CContainerPrinter::print(m_));
        LOG_DEBUG(<< "sample mean = " << core::CContainerPrinter::print(mean_));
        {
            TDoubleVec error = test_detail::minus(mean_, m_);
            LOG_DEBUG(<< "||error|| = " << test_detail::euclidean(error));
            LOG_DEBUG(<< "||m|| = " << test_detail::euclidean(m_));
            BOOST_TEST_REQUIRE(test_detail::euclidean(error) <
                               0.02 * test_detail::euclidean(m_));
        }

        // Get the sample covariance matrix.
        TDoubleVecVec covariance(3, TDoubleVec(3, 0.0));
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            test_detail::add(test_detail::outer(test_detail::minus(samples[i], mean_),
                                                test_detail::minus(samples[i], mean_)),
                             covariance);
        }
        test_detail::divide(covariance, static_cast<double>(samples.size() - 1));
        LOG_DEBUG(<< "actual covariance = " << core::CContainerPrinter::print(covariance));
        LOG_DEBUG(<< "sample covariance = " << core::CContainerPrinter::print(covariance));
        {
            // The cast of the minus() function is necessary to avoid overload
            // ambiguity with std::minus (found via ADL if <functional> is
            // indirectly included)
            TDoubleVecVec error = test_detail::minus(covariance, C_);
            LOG_DEBUG(<< "||error|| = " << test_detail::frobenius(error));
            LOG_DEBUG(<< "||C|| = " << test_detail::frobenius(C_));
            BOOST_TEST_REQUIRE(test_detail::frobenius(error) <
                               0.1 * test_detail::frobenius(C_));
        }
    }
}

BOOST_AUTO_TEST_CASE(testReservoirSampling) {

    // Check we uniformly sample from a stream.

    using TSampler = maths::CSampling::CReservoirSampler<double>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    TDoubleVec samples(200);
    TSampler sampler{200, [&](std::size_t slot, const double& value) {
                         samples[slot] = value;
                     }};

    TMeanAccumulator pValue;
    for (std::size_t t = 0; t < 100; ++t) {
        samples.assign(200, 0.0);
        sampler.reset();
        for (double x = 0.0; x < 1000.0; x += 1.0) {
            sampler.sample(x);
        }

        maths::CStatisticalTests::CCramerVonMises cvm{20};
        for (const auto& sample : samples) {
            cvm.addF(sample / 1000.0);
        }

        // The p-value is small if the samples *aren't* distributed as expected.
        BOOST_TEST_REQUIRE(cvm.pValue() > 0.05);
        pValue.add(cvm.pValue());
    }

    LOG_DEBUG(<< "mean p-value = " << maths::CBasicStatistics::mean(pValue));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(pValue) > 0.3);
}

BOOST_AUTO_TEST_CASE(testVectorDissimilaritySampler) {

    // Test the average distance between points is significantly larger than
    // for uniform random sampling.

    using TVector = maths::CDenseVector<double>;
    using TVectorVec = std::vector<TVector>;
    using TReservoirSampler = maths::CSampling::CReservoirSampler<TVector>;
    using TDissimilaritySampler = maths::CSampling::CVectorDissimilaritySampler<TVector>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    std::size_t numberSamples{100};

    TVectorVec samples(numberSamples);
    TReservoirSampler randomSampler{
        numberSamples,
        [&](std::size_t slot, const TVector& value) { samples[slot] = value; }};
    TDissimilaritySampler dissimilaritySampler{numberSamples};

    test::CRandomNumbers rng;

    TMeanAccumulator percentageSeparationIncrease;

    TVector x{4};
    TDoubleVec p;
    TDoubleVec components;
    TDoubleVec shift{-1.0, 0.0, 10.0, -20.0};
    for (std::size_t t = 0; t < 50; ++t) {
        samples.assign(numberSamples, TVector::Zero(3));
        randomSampler.reset();
        dissimilaritySampler.reset();
        for (std::size_t i = 0; i < 1000; ++i) {
            rng.generateLogNormalSamples(1.0, 2.0, 4, components);
            for (std::size_t j = 0; j < components.size(); ++j) {
                x(j) = components[j];
            }
            randomSampler.sample(x);
            dissimilaritySampler.sample(x);
        }

        TMeanAccumulator randomSeparation;
        TMeanAccumulator dissimilaritySeparation;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            for (std::size_t j = 0; j < samples.size(); ++j) {
                double distance{(samples[i] - samples[j]).norm()};
                randomSeparation.add(distance);
                distance = (dissimilaritySampler.samples()[i] -
                            dissimilaritySampler.samples()[j])
                               .norm();
                dissimilaritySeparation.add(distance);
            }
        }
        LOG_TRACE(<< "random mean separation = " << maths::CBasicStatistics::mean(randomSeparation)
                  << ", dissimilar mean separation = "
                  << maths::CBasicStatistics::mean(dissimilaritySeparation));
        percentageSeparationIncrease.add(
            100.0 *
            (maths::CBasicStatistics::mean(dissimilaritySeparation) -
             maths::CBasicStatistics::mean(randomSeparation)) /
            maths::CBasicStatistics::mean(randomSeparation));
    }
    LOG_DEBUG(<< "% separation increase = "
              << maths::CBasicStatistics::mean(percentageSeparationIncrease));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(percentageSeparationIncrease) > 50.0);
}

BOOST_AUTO_TEST_SUITE_END()
