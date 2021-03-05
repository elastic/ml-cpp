/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CMixtureDistribution.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CMixtureDistributionTest)

using namespace ml;
using namespace maths;
using namespace test;

using TDoubleVec = std::vector<double>;
using TNormalVec = std::vector<boost::math::normal_distribution<>>;
using TLogNormalVec = std::vector<boost::math::lognormal_distribution<>>;
using TGammaVec = std::vector<boost::math::gamma_distribution<>>;

BOOST_AUTO_TEST_CASE(testSupport) {
    {
        boost::math::normal_distribution<> n1(0.0, 1.0);
        boost::math::normal_distribution<> n2(5.0, 1.0);
        TDoubleVec weights;
        weights.push_back(0.5);
        weights.push_back(0.5);
        TNormalVec modes;
        modes.push_back(n1);
        modes.push_back(n2);
        CMixtureDistribution<boost::math::normal_distribution<>> mixture(weights, modes);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(boost::math::support(n1)),
                            core::CContainerPrinter::print(support(mixture)));
    }
    {
        boost::math::lognormal_distribution<> l1(1.0, 0.5);
        boost::math::lognormal_distribution<> l2(2.0, 0.02);
        TDoubleVec weights;
        weights.push_back(0.6);
        weights.push_back(0.4);
        TLogNormalVec modes;
        modes.push_back(l1);
        modes.push_back(l2);
        CMixtureDistribution<boost::math::lognormal_distribution<>> mixture(weights, modes);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(boost::math::support(l1)),
                            core::CContainerPrinter::print(support(mixture)));
    }
}

BOOST_AUTO_TEST_CASE(testMode) {
    // The mode of the distribution should be at the maximum
    // of the distribution, i.e. p.d.f. derivative should be
    // zero and curvature should be positive.

    const double eps = 1e-5;

    {
        LOG_DEBUG(<< "Mixture Two Normals");

        double means[][2] = {{0.0, 10.0}, {0.0, 9.0}, {0.0, 8.0}, {0.0, 7.0},
                             {0.0, 6.0},  {0.0, 5.0}, {0.0, 4.0}, {0.0, 3.0},
                             {0.0, 2.0},  {0.0, 1.0}};

        for (std::size_t i = 0; i < boost::size(means); ++i) {
            LOG_DEBUG(<< "means = " << core::CContainerPrinter::print(means[i]));
            TDoubleVec weights;
            weights.push_back(0.6);
            weights.push_back(0.4);
            boost::math::normal_distribution<> n1(means[i][0], 1.0);
            boost::math::normal_distribution<> n2(means[i][1], 1.0);
            TNormalVec modes;
            modes.push_back(n1);
            modes.push_back(n2);
            CMixtureDistribution<boost::math::normal_distribution<>> mixture(weights, modes);

            double x = mode(mixture);

            double pMinusEps = pdf(mixture, x - eps);
            double p = pdf(mixture, x);
            double pPlusEps = pdf(mixture, x + eps);

            double derivative = (pPlusEps - pMinusEps) / 2.0 / eps;
            double curvature = (pPlusEps - 2.0 * p + pMinusEps) / eps / eps;

            LOG_DEBUG(<< "x = " << x << ", df/dx = " << derivative
                      << ", d^2f/dx^2 = " << curvature);

            // Gradient zero + curvature negative => maximum.
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, derivative, 1e-6);
            BOOST_TEST_REQUIRE(curvature < 0.0);
        }
    }

    {
        LOG_DEBUG(<< "Mixture Three Normals");
        TDoubleVec weights;
        weights.push_back(0.4);
        weights.push_back(0.5);
        weights.push_back(0.1);
        boost::math::normal_distribution<> n1(1.5, 1.0);
        boost::math::normal_distribution<> n2(5.0, 3.0);
        boost::math::normal_distribution<> n3(6.0, 2.0);
        TNormalVec modes;
        modes.push_back(n1);
        modes.push_back(n2);
        modes.push_back(n3);
        CMixtureDistribution<boost::math::normal_distribution<>> mixture(weights, modes);

        double x = mode(mixture);

        double pMinusEps = pdf(mixture, x - eps);
        double p = pdf(mixture, x);
        double pPlusEps = pdf(mixture, x + eps);

        double derivative = (pPlusEps - pMinusEps) / 2.0 / eps;
        double curvature = (pPlusEps - 2.0 * p + pMinusEps) / eps / eps;

        LOG_DEBUG(<< "x = " << x << ", df/dx = " << derivative << ", d^2f/dx^2 = " << curvature);

        // Gradient zero + curvature negative => maximum.
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, derivative, 1e-6);
        BOOST_TEST_REQUIRE(curvature < 0.0);
    }
    {
        LOG_DEBUG(<< "Mixture Two Log-Normals");
        TDoubleVec weights;
        weights.push_back(0.6);
        weights.push_back(0.4);
        boost::math::lognormal_distribution<> l1(1.0, 0.5);
        boost::math::lognormal_distribution<> l2(2.0, 0.1);
        TLogNormalVec modes;
        modes.push_back(l1);
        modes.push_back(l2);
        CMixtureDistribution<boost::math::lognormal_distribution<>> mixture(weights, modes);

        double x = mode(mixture);

        double pMinusEps = pdf(mixture, x - eps);
        double p = pdf(mixture, x);
        double pPlusEps = pdf(mixture, x + eps);

        double derivative = (pPlusEps - pMinusEps) / 2.0 / eps;
        double curvature = (pPlusEps - 2.0 * p + pMinusEps) / eps / eps;

        LOG_DEBUG(<< "x = " << x << ", df/dx = " << derivative << ", d^2f/dx^2 = " << curvature);

        // Gradient zero + curvature negative => maximum.
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, derivative, 1e-6);
        BOOST_TEST_REQUIRE(curvature < 0.0);
    }
}

BOOST_AUTO_TEST_CASE(testPdf) {
    // Check that the p.d.f. is the derivative of the c.d.f.

    const double tolerance = 1e-6;
    const double eps = 1e-6;

    {
        double weights[][2] = {
            {0.5, 0.5},   {0.3, 0.7}, {0.6, 0.4}, {0.5, 0.5},   {0.1, 0.9},
            {0.61, 0.39}, {0.7, 0.3}, {0.8, 0.2}, {0.15, 0.85}, {0.3, 0.7}};
        double means[][2] = {{0.0, 10.0}, {1.0, 9.0}, {1.4, 6.0}, {0.0, 7.0},
                             {3.0, 7.5},  {0.0, 5.0}, {2.0, 4.0}, {1.0, 3.0},
                             {1.1, 2.0},  {3.0, 3.2}};
        double variances[][2] = {
            {0.3, 10.0}, {1.0, 0.4}, {1.4, 6.0}, {3.0, 1.1}, {3.0, 3.5},
            {1.0, 5.0},  {2.3, 4.0}, {3.0, 1.0}, {1.1, 1.0}, {3.0, 3.2}};

        BOOST_REQUIRE_EQUAL(boost::size(weights), boost::size(means));
        BOOST_REQUIRE_EQUAL(boost::size(means), boost::size(variances));

        for (size_t i = 0; i < boost::size(weights); ++i) {
            LOG_TRACE(<< "*** Test Case " << i << " ***");

            TDoubleVec w;
            w.push_back(weights[i][0]);
            w.push_back(weights[i][1]);
            boost::math::normal_distribution<> n1(means[i][0], variances[i][0]);
            boost::math::normal_distribution<> n2(means[i][1], variances[i][1]);
            TNormalVec modes;
            modes.push_back(n1);
            modes.push_back(n2);
            CMixtureDistribution<boost::math::normal_distribution<>> mixture(w, modes);

            for (unsigned int p = 1; p < 100; ++p) {
                double q = static_cast<double>(p) / 100.0;
                double x = quantile(mixture, q);

                double f = pdf(mixture, x);
                double dFdx = (cdf(mixture, x + eps) - cdf(mixture, x - eps)) / 2.0 / eps;

                LOG_TRACE(<< "percentile = " << p << "%"
                          << ", f = " << f << ", dF/dx = " << dFdx);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(f, dFdx, tolerance);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testCdf) {
    // The idea here is that the distribution should describe data
    // generated by a mixture of distributions.

    const std::size_t nSamples = 100000;

    const double weights[][2] = {
        {0.3, 0.7}, {0.5, 0.5}, {0.6, 0.4}, {0.35, 0.65}, {0.55, 0.45}};
    const double shapes[][2] = {
        {10.0, 30.0}, {5.0, 25.0}, {20.0, 25.0}, {4.0, 50.0}, {11.0, 33.0}};
    const double scales[][2] = {{0.3, 0.2}, {1.0, 1.1}, {0.9, 0.95}, {0.4, 1.2}, {2.3, 2.1}};

    BOOST_REQUIRE_EQUAL(boost::size(weights), boost::size(shapes));
    BOOST_REQUIRE_EQUAL(boost::size(shapes), boost::size(scales));

    CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(weights); ++i) {
        LOG_TRACE(<< "*** Test Case " << i << " ***");

        TDoubleVec samples1;
        rng.generateGammaSamples(
            shapes[i][0], scales[i][0],
            static_cast<std::size_t>(weights[i][0] * static_cast<double>(nSamples)),
            samples1);
        TDoubleVec samples2;
        rng.generateGammaSamples(
            shapes[i][1], scales[i][1],
            static_cast<std::size_t>(weights[i][1] * static_cast<double>(nSamples)),
            samples2);

        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        std::sort(samples.begin(), samples.end());

        TDoubleVec w;
        w.push_back(weights[i][0]);
        w.push_back(weights[i][1]);
        boost::math::gamma_distribution<> g1(shapes[i][0], scales[i][0]);
        boost::math::gamma_distribution<> g2(shapes[i][1], scales[i][1]);
        TGammaVec modes;
        modes.push_back(g1);
        modes.push_back(g2);
        CMixtureDistribution<boost::math::gamma_distribution<>> mixture(w, modes);

        // Check the data percentiles.
        for (unsigned int p = 1; p < 100; ++p) {
            double x = samples[nSamples * p / 100];
            double actualCdf = cdf(mixture, x);
            double expectedCdf = static_cast<double>(p) / 100;

            LOG_TRACE(<< "percentile = " << p << "%"
                      << ", actual cdf = " << actualCdf
                      << ", expected cdf = " << expectedCdf);

            // No more than a 10% error in the sample percentile.
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedCdf, actualCdf, 0.1 * expectedCdf);
        }
    }
}

BOOST_AUTO_TEST_CASE(testQuantile) {
    // Check that the quantile is the inverse of the c.d.f.

    const double weights[][3] = {
        {0.4, 0.3, 0.3}, {0.1, 0.4, 0.5}, {0.6, 0.2, 0.2}, {0.1, 0.8, 0.1}, {0.25, 0.3, 0.45}};
    const double locations[][3] = {
        {1.0, 1.9, 2.2}, {0.9, 1.8, 3.0}, {2.0, 4.0, 4.5}, {0.1, 0.3, 0.4}, {0.2, 1.3, 4.8}};
    const double scales[][3] = {
        {0.1, 0.04, 0.5}, {0.8, 0.3, 0.6}, {0.5, 0.3, 0.4}, {0.3, 0.08, 0.9}, {0.1, 0.2, 1.0}};

    BOOST_REQUIRE_EQUAL(boost::size(weights), boost::size(locations));
    BOOST_REQUIRE_EQUAL(boost::size(locations), boost::size(scales));

    for (std::size_t i = 0; i < boost::size(weights); ++i) {
        LOG_TRACE(<< "*** Test " << i << " ***");

        TDoubleVec w;
        w.push_back(weights[i][0]);
        w.push_back(weights[i][1]);
        w.push_back(weights[i][2]);
        boost::math::lognormal_distribution<> l1(locations[i][0], scales[i][0]);
        boost::math::lognormal_distribution<> l2(locations[i][1], scales[i][1]);
        boost::math::lognormal_distribution<> l3(locations[i][2], scales[i][2]);
        TLogNormalVec modes;
        modes.push_back(l1);
        modes.push_back(l2);
        modes.push_back(l3);
        CMixtureDistribution<boost::math::lognormal_distribution<>> mixture(w, modes);

        for (unsigned int p = 1; p < 100; ++p) {
            double q = static_cast<double>(p) / 100.0;
            double f = cdf(mixture, quantile(mixture, q));
            LOG_TRACE(<< "Error = " << std::fabs(q - f));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(q, f, 1e-10);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
