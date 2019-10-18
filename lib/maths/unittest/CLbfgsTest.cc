/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CLbfgs.h>
#include <maths/CLinearAlgebraEigen.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CLbfgsTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TVector = maths::CDenseVector<double>;
using TVectorVec = std::vector<TVector>;

BOOST_AUTO_TEST_CASE(testQuadtratic) {

    test::CRandomNumbers rng;

    TVector diagonal(10);
    for (std::size_t i = 0; i < 10; ++i) {
        diagonal(i) = static_cast<double>(i);
    }

    auto f = [&](const TVector& x) {
        return x.transpose() * diagonal.asDiagonal() * x;
    };
    auto g = [&](const TVector& x) { return 2.0 * diagonal.asDiagonal() * x; };

    maths::CLbfgs<TVector> lbfgs{10};

    // Check convergence rate is super-linear (compare to gradient descent).

    TVector x0{100.0 * diagonal};

    for (std::size_t i = 10; i < 15; ++i) {
        double fx;
        std::tie(std::ignore, fx) = lbfgs.minimize(f, g, x0, 1e-12, i);

        // Gradient descent with back tracking.
        TVector x{x0};
        for (std::size_t j = 0; j < i; ++j) {
            double fl(f(x));
            double s{1.0};
            for (std::size_t k = 0;
                 k < 10 && fl - f(x - s * g(x)) < 1e-4 * g(x).norm(); ++k, s *= 0.5) {
            }
            x -= s * g(x);
        }

        BOOST_TEST(fx < 0.02 * static_cast<double>(f(x)));
    }

    // Check convergence to the minimum of a quadtratic form for a variety of
    // matrix conditions and start positions.

    TDoubleVec samples;

    double fmean{0.0};

    for (std::size_t test = 0; test < 1000; ++test) {

        rng.generateLogNormalSamples(0.0, 3.0, 10, samples);

        for (std::size_t i = 0; i < 10; ++i) {
            diagonal(i) = samples[i];
        }
        LOG_TRACE(<< "diagonal = " << diagonal.transpose());

        rng.generateUniformSamples(-10.0, 10.0, 10, samples);
        for (std::size_t i = 0; i < 10; ++i) {
            x0(i) = samples[i];
        }
        LOG_TRACE(<< "x0 = " << x0.transpose());

        TVector x;
        double fx;
        for (double scale : {0.01, 1.0, 100.0}) {
            std::tie(x, fx) = lbfgs.minimize(f, g, scale * x0);
            BOOST_CHECK_EQUAL(fx, static_cast<double>(f(x)));
            BOOST_CHECK_CLOSE_ABSOLUTE(0.0, fx, 0.5 * scale * scale);
        }

        std::tie(std::ignore, fx) = lbfgs.minimize(f, g, x0);
        fmean += fx / 1000.0;
    }

    BOOST_CHECK_CLOSE_ABSOLUTE(0.0, fmean, 5e-3);
}

BOOST_AUTO_TEST_CASE(testSingularHessian) {

    // Test we converge to the global minimum of a convex function when the Hessian
    // is mostly singular.

    // We use f(x) = max_{p(i)}{||x - p(i)||}. The Hessian is zero except at the
    // boundary between regions with different furthest points.

    test::CRandomNumbers rng;

    TVectorVec points(10, TVector{10});

    auto f = [&](const TVector& x) {
        double result{0.0};
        for (const auto& p : points) {
            result = std::max(result, (p - x).norm());
        }
        return result;
    };
    auto g = [&](const TVector& x) {
        TVector furthest{x};
        for (const auto& p : points) {
            if ((p - x).norm() > (furthest - x).norm()) {
                furthest = p;
            }
        }
        return TVector{(x - furthest) / (x - furthest).norm()};
    };

    maths::CLbfgs<TVector> lbfgs{5};

    for (std::size_t test = 0; test < 10; ++test) {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 100, samples);

        for (std::size_t i = 0; i < samples.size(); /**/) {
            for (std::size_t j = 0; j < 10; ++i, ++j) {
                points[i / 10](j) = samples[i];
            }
        }

        TSizeVec x0;
        rng.generateUniformSamples(0, 10, 1, x0);

        TVector x;
        double fx;
        std::tie(x, fx) = lbfgs.minimize(f, g, points[x0[0]], 1e-8, 75);

        // Test we're near the minimum.
        TVector eps{10};
        eps.setZero();
        for (std::size_t j = 0; j < 10; ++j) {
            eps(j) = 0.2;
            BOOST_TEST(f(x - eps) > fx);
            BOOST_TEST(f(x + eps) > fx);
            eps(j) = 0.0;
        }
    }
}

BOOST_AUTO_TEST_CASE(testConstrainedMinimize) {

    // Check convergence to the minimum of a quadtratic form for a variety of
    // matrix conditions and start positions.

    test::CRandomNumbers rng;

    TVector diagonal{10};
    for (std::size_t i = 0; i < 10; ++i) {
        diagonal(i) = static_cast<double>(i);
    }

    auto f = [&](const TVector& x) {
        return x.transpose() * diagonal.asDiagonal() * x;
    };
    auto g = [&](const TVector& x) { return 2.0 * diagonal.asDiagonal() * x; };

    maths::CLbfgs<TVector> lbfgs{10};

    TDoubleVec samples;
    TDoubleVec bb;

    double ferror{0.0};

    TVector x0{10};
    TVector a{10};
    TVector b{10};
    TVector xmin{10};

    for (std::size_t test = 0; test < 100; ++test) {

        rng.generateLogNormalSamples(0.0, 3.0, 10, samples);

        for (std::size_t i = 0; i < 10; ++i) {
            diagonal(i) = samples[i];
        }
        LOG_TRACE(<< "diagonal = " << diagonal.transpose());

        rng.generateUniformSamples(-10.0, 10.0, 10, samples);
        for (std::size_t i = 0; i < 10; ++i) {
            x0(i) = samples[i];
        }
        LOG_TRACE(<< "x0 = " << x0.transpose());

        rng.generateUniformSamples(-10.0, 10.0, 20, samples);
        for (std::size_t i = 0; i < 10; ++i) {
            a(i) = std::min(samples[i], samples[10 + i]);
            b(i) = std::max(samples[i], samples[10 + i]);
            xmin(i) = a(i) * b(i) < 0.0 ? 0.0 : (b(i) < 0.0 ? b(i) : a(i));
        }
        LOG_TRACE(<< "a  = " << a.transpose());
        LOG_TRACE(<< "b  = " << b.transpose());
        LOG_TRACE(<< "x* = " << xmin.transpose());

        TVector x;
        double fx;
        std::tie(x, fx) = lbfgs.constrainedMinimize(f, g, a, b, x0, 0.2);

        BOOST_CHECK_EQUAL(fx, static_cast<double>(f(x)));
        BOOST_CHECK_CLOSE_ABSOLUTE(f(xmin), fx, 1e-3);

        ferror += std::fabs(fx - f(xmin)) / 100.0;
    }

    BOOST_CHECK_CLOSE_ABSOLUTE(0.0, ferror, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
