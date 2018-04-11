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

#include "CSamplingTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CSampling.h>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/range.hpp>

#include <numeric>

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

using namespace ml;

namespace {

using TDoubleVecVec = std::vector<TDoubleVec>;

double multinomialProbability(const TDoubleVec& probabilities, const TSizeVec& counts) {
    std::size_t n = std::accumulate(counts.begin(), counts.end(), std::size_t(0));
    double logP = boost::math::lgamma(static_cast<double>(n + 1));
    for (std::size_t i = 0u; i < counts.size(); ++i) {
        double ni = static_cast<double>(counts[i]);
        if (ni > 0.0) {
            logP += ni * std::log(probabilities[i]) - boost::math::lgamma(ni + 1.0);
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

void CSamplingTest::testMultinomialSample() {
    LOG_DEBUG(<< "+----------------------------------------+");
    LOG_DEBUG(<< "|  CSamplingTest::testMultinomialSample  |");
    LOG_DEBUG(<< "+----------------------------------------+");

    using TSizeVecDoubleMap = std::map<TSizeVec, double>;
    using TSizeVecDoubleMapCItr = TSizeVecDoubleMap::const_iterator;

    maths::CSampling::seed();

    double probabilities_[] = {0.4, 0.25, 0.2, 0.15};

    TDoubleVec probabilities(boost::begin(probabilities_), boost::end(probabilities_));

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
        CPPUNIT_ASSERT_EQUAL(size_t(20), std::accumulate(pItr->first.begin(),
                                                         pItr->first.end(), size_t(0)));

        double p = multinomialProbability(probabilities, pItr->first);
        double pe = pItr->second;
        LOG_DEBUG(<< "p  = " << p << ", pe = " << pe);
        CPPUNIT_ASSERT(std::fabs(pe - p) < std::max(0.27 * p, 3e-5));
        error += std::fabs(pe - p);
        pTotal += p;
    }

    LOG_DEBUG(<< "pTotal = " << pTotal << ", error = " << error);
    CPPUNIT_ASSERT(error < 0.02 * pTotal);
}

void CSamplingTest::testMultivariateNormalSample() {
    LOG_DEBUG(<< "+-----------------------------------------------+");
    LOG_DEBUG(<< "|  CSamplingTest::testMultivariateNormalSample  |");
    LOG_DEBUG(<< "+-----------------------------------------------+");

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    maths::CSampling::seed();

    {
        double m[] = {1.0, 3.0, 5.0};
        TDoubleVec m_(boost::begin(m), boost::end(m));
        double C[][3] = {{3.0, 1.0, 0.1}, {1.0, 2.0, -0.3}, {0.1, -0.3, 1.0}};
        TDoubleVecVec C_;
        C_.push_back(TDoubleVec(boost::begin(C[0]), boost::end(C[0])));
        C_.push_back(TDoubleVec(boost::begin(C[1]), boost::end(C[1])));
        C_.push_back(TDoubleVec(boost::begin(C[2]), boost::end(C[2])));

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
            CPPUNIT_ASSERT(test_detail::euclidean(error) <
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
            CPPUNIT_ASSERT(test_detail::frobenius(error) < 0.1 * test_detail::frobenius(C_));
        }
    }
}

CppUnit::Test* CSamplingTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSamplingTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSamplingTest>(
        "CSamplingTest::testMultinomialSample", &CSamplingTest::testMultinomialSample));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSamplingTest>(
        "CSamplingTest::testMultivariateNormalSample",
        &CSamplingTest::testMultivariateNormalSample));

    return suiteOfTests;
}
