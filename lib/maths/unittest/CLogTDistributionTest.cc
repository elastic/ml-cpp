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

#include "CLogTDistributionTest.h"

#include <core/CLogger.h>

#include <maths/CLogTDistribution.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

#include <math.h>

using namespace ml;
using namespace maths;
using namespace test;

typedef std::vector<double>        TDoubleVec;
typedef TDoubleVec::iterator       TDoubleVecItr;
typedef TDoubleVec::const_iterator TDoubleVecCItr;

void CLogTDistributionTest::testMode(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CLogTDistributionTest::testMode  |");
    LOG_DEBUG("+-----------------------------------+");

    // The mode of the distribution should be at the maximum
    // of the distribution, i.e. p.d.f. derivative should be
    // zero and curvature should be positive.

    const double eps = 1e-5;

    const double degreesFreedoms[] = { 2.0, 10.0, 40.0 };
    const double locations[] = { 1.0, 2.0, 3.0, 4.0, 6.0 };
    const double squareScales[] = { 0.5, 1, 1.5, 2.0, 3.0 };

    for (size_t i = 0; i < boost::size(degreesFreedoms); ++i) {
        for (size_t j = 0; j < boost::size(locations); ++j) {
            for (size_t k = 0; k < boost::size(squareScales); ++k) {
                LOG_DEBUG("degrees freedom = " << degreesFreedoms[i]
                          << ", location = " << locations[j]
                          << ", scale = " << ::sqrt(squareScales[k]));

                CLogTDistribution logt(degreesFreedoms[i],
                                       locations[j],
                                       ::sqrt(squareScales[k]));

                double x = mode(logt);

                if (x != 0.0) {
                    double pMinusEps = pdf(logt, x - eps);
                    double p = pdf(logt, x);
                    double pPlusEps = pdf(logt, x + eps);

                    double derivative = (pPlusEps - pMinusEps) / 2.0 / eps;
                    double curvature = (pPlusEps - 2.0 * p + pMinusEps) / eps / eps;

                    // Gradient zero + curvature negative => maximum.
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, derivative, 1e-6);
                    CPPUNIT_ASSERT(curvature < 0.0);
                }
            }
        }
    }
}

void CLogTDistributionTest::testPdf(void) {
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CLogTDistributionTest::testPdf  |");
    LOG_DEBUG("+----------------------------------+");

    // Check that the p.d.f. is the derivative of the c.d.f.

    const double tolerance = 1e-6;
    const double eps = 1e-6;

    const double degreesFreedom[] = { 2.0, 10.0, 40.0 };
    const double locations[] = { 1.0, 2.0, 3.0 };
    const double squareScales[] = { 0.5, 1, 1.5 };
    size_t nTests = boost::size(degreesFreedom);
    nTests = std::min(nTests, boost::size(locations));
    nTests = std::min(nTests, boost::size(squareScales));

    for (size_t test = 0; test < nTests; ++test) {
        CLogTDistribution logt(degreesFreedom[test],
                               locations[test],
                               ::sqrt(squareScales[test]));

        for (unsigned int p = 1; p < 100; ++p) {
            double q = static_cast<double>(p) / 100.0;
            double x = quantile(logt, q);

            double pdf = maths::pdf(logt, x);
            double dcdfdx = (cdf(logt, x + eps) - cdf(logt, x - eps)) / 2.0 / eps;

            LOG_DEBUG("percentile = " << p << "%"
                      << ", pdf = " << pdf
                      << ", dcdfdx = " << dcdfdx);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(pdf, dcdfdx, tolerance);
        }
    }
}

void CLogTDistributionTest::testCdf(void) {
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CLogTDistributionTest::testCdf  |");
    LOG_DEBUG("+----------------------------------+");

    // The idea here is that the distribution should describe data
    // generated by exp(X / s + m)) where X is student's t.

    const size_t nSamples = 100000u;

    const double degreesFreedom[] = { 2.0, 10.0, 40.0 };
    const double locations[] = { 1.0, 2.0, 3.0 };
    const double squareScales[] = { 0.5, 1, 1.5 };
    size_t nTests = boost::size(degreesFreedom);
    nTests = std::min(nTests, boost::size(locations));
    nTests = std::min(nTests, boost::size(squareScales));

    CRandomNumbers rng;

    for (size_t test = 0; test < nTests; ++test) {
        TDoubleVec samples;
        rng.generateStudentsSamples(degreesFreedom[test], nSamples, samples);

        for (TDoubleVecItr sampleItr = samples.begin();
             sampleItr != samples.end();
             ++sampleItr) {
            *sampleItr = ::exp(*sampleItr * ::sqrt(squareScales[test]) + locations[test]);
        }

        // Check the data percentiles.
        CLogTDistribution logt(degreesFreedom[test],
                               locations[test],
                               ::sqrt(squareScales[test]));

        std::sort(samples.begin(), samples.end());
        for (unsigned int p = 1; p < 100; ++p) {
            double x = samples[nSamples * p / 100];
            double actualCdf = cdf(logt, x);
            double expectedCdf = static_cast<double>(p) / 100;

            LOG_DEBUG("percentile = " << p << "%"
                      << ", actual cdf = " << actualCdf
                      << ", expected cdf = " << expectedCdf);

            // No more than a 10% error in the sample percentile.
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedCdf, actualCdf, 0.1 * expectedCdf);
        }
    }
}

void CLogTDistributionTest::testQuantile(void) {
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CLogTDistributionTest::testQuantile  |");
    LOG_DEBUG("+---------------------------------------+");

    // Check that the quantile is the inverse of the c.d.f.

    const double degreesFreedom[] = { 2.0, 10.0, 40.0 };
    const double locations[] = { 1.0, 2.0, 3.0 };
    const double squareScales[] = { 0.5, 1, 1.5 };
    size_t nTests = boost::size(degreesFreedom);
    nTests = std::min(nTests, boost::size(locations));
    nTests = std::min(nTests, boost::size(squareScales));

    for (size_t test = 0; test < nTests; ++test) {
        CLogTDistribution logt(degreesFreedom[test],
                               locations[test],
                               ::sqrt(squareScales[test]));

        for (unsigned int p = 1; p < 100; ++p) {
            double q = static_cast<double>(p) / 100.0;

            // Check that the quantile function is the inverse
            // of the c.d.f.
            CPPUNIT_ASSERT_DOUBLES_EQUAL(q, cdf(logt, quantile(logt, q)), 1e-10);
        }
    }
}

CppUnit::Test *CLogTDistributionTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CLogTDistributionTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CLogTDistributionTest>(
                               "CLogTDistributionTest::testMode",
                               &CLogTDistributionTest::testMode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogTDistributionTest>(
                               "CLogTDistributionTest::testPdf",
                               &CLogTDistributionTest::testPdf) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogTDistributionTest>(
                               "CLogTDistributionTest::testCdf",
                               &CLogTDistributionTest::testCdf) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLogTDistributionTest>(
                               "CLogTDistributionTest::testQuantile",
                               &CLogTDistributionTest::testQuantile) );

    return suiteOfTests;
}
