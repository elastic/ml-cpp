/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CSolvers.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <utility>

BOOST_AUTO_TEST_SUITE(CSolversTest)

using namespace ml;
using namespace maths;

namespace {

//! Root at 5.
double f1(const double& x) {
    return x - 5.0;
}

//! Roots at 1 and 2.
double f2(const double& x) {
    return x * x - 3.0 * x + 2.0;
}

//! Note that this is a contractive map (i.e. df/dx < 1)
//! so we can find the root by an iterative scheme.
//! There is a root is in the interval:\n
//!   [0.739085133215160, 0.739085133215161]
double f3(const double& x) {
    return std::cos(x) - x;
}

//! Root at x = 2/3.
double f4(const double& x) {
    return x <= 2.0 / 3.0 ? std::pow(std::fabs(x - 2.0 / 3.0), 0.2)
                          : -std::pow(std::fabs(x - 2.0 / 3.0), 0.2);
}

//! This has local maxima at 4 and 10.
double f5(const double& x) {
    return 1.1 * std::exp(-(x - 4.0) * (x - 4.0)) +
           0.4 * std::exp(-(x - 10.0) * (x - 10.0) / 4.0);
}

//! This has local maxima at 4, 6 and 10.
double f6(const double& x) {
    return 1.1 * std::exp(-2.0 * (x - 4.0) * (x - 4.0)) +
           0.1 * std::exp(-(x - 6.0) * (x - 6.0)) +
           0.4 * std::exp(-(x - 10.0) * (x - 10.0) / 2.0);
}

class CLog {
public:
    using result_type = double;

public:
    double operator()(const double& x) const {
        if (x <= 0.0) {
            throw std::range_error("Bad value to log " +
                                   core::CStringUtils::typeToString(x));
        }
        return std::log(x);
    }
};
}

BOOST_AUTO_TEST_CASE(testBracket) {
    {
        CCompositeFunctions::CMinusConstant<CLog> f(CLog(), 0.0);
        std::size_t maxIterations = 10;
        double a = 0.5, b = 0.5;
        double fa = f(a), fb = f(b);
        BOOST_TEST_REQUIRE(CSolvers::rightBracket(a, b, fa, fb, f, maxIterations));
        LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << fa
                  << ", f(b) = " << fb << ", maxIterations = " << maxIterations);
        BOOST_REQUIRE_EQUAL(f(a), fa);
        BOOST_REQUIRE_EQUAL(f(b), fb);
        BOOST_TEST_REQUIRE(fa * fb <= 0.0);
    }

    {
        CCompositeFunctions::CMinusConstant<CLog> f(CLog(), 5.0);
        std::size_t maxIterations = 10;
        double a = 0.5, b = 0.6;
        double fa = f(a), fb = f(b);
        BOOST_TEST_REQUIRE(CSolvers::rightBracket(a, b, fa, fb, f, maxIterations));
        LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << fa
                  << ", f(b) = " << fb << ", maxIterations = " << maxIterations);
        BOOST_REQUIRE_EQUAL(f(a), fa);
        BOOST_REQUIRE_EQUAL(f(b), fb);
        BOOST_TEST_REQUIRE(fa * fb <= 0.0);
    }

    {
        CCompositeFunctions::CMinusConstant<CLog> f(CLog(), 10.0);
        std::size_t maxIterations = 10;
        double a = 0.5, b = 5.0;
        double fa = f(a), fb = f(b);
        BOOST_TEST_REQUIRE(CSolvers::rightBracket(a, b, fa, fb, f, maxIterations));
        LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << fa
                  << ", f(b) = " << fb << ", maxIterations = " << maxIterations);
        BOOST_REQUIRE_EQUAL(f(a), fa);
        BOOST_REQUIRE_EQUAL(f(b), fb);
        BOOST_TEST_REQUIRE(fa * fb <= 0.0);
    }

    {
        CCompositeFunctions::CMinusConstant<CLog> f(CLog(), 0.0);
        std::size_t maxIterations = 10;
        double a = 100.0, b = 100.0;
        double fa = f(a), fb = f(b);
        BOOST_TEST_REQUIRE(CSolvers::leftBracket(
            a, b, fa, fb, f, maxIterations, std::numeric_limits<double>::min()));
        LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << fa
                  << ", f(b) = " << fb << ", maxIterations = " << maxIterations);
        BOOST_REQUIRE_EQUAL(f(a), fa);
        BOOST_REQUIRE_EQUAL(f(b), fb);
        BOOST_TEST_REQUIRE(fa * fb <= 0.0);
    }
}

BOOST_AUTO_TEST_CASE(testBisection) {
    double a, b;
    double bestGuess;
    std::size_t iterations;

    // Test that the method fails if the root isn't bracketed.
    {
        a = 6.0;
        b = 10.0;
        iterations = 5;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.1);
        BOOST_TEST_REQUIRE(!CSolvers::bisection(a, b, &f1, iterations, equal, bestGuess));
    }

    // Test that the method terminates if hits the root.
    {
        a = 0.0;
        b = 10.0;
        iterations = 10;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.1);
        BOOST_TEST_REQUIRE(CSolvers::bisection(a, b, -5.0, 5.0, &f1, iterations,
                                               equal, bestGuess));
        BOOST_REQUIRE_EQUAL(static_cast<std::size_t>(1), iterations);
        BOOST_REQUIRE_EQUAL(5.0, bestGuess);
        BOOST_REQUIRE_EQUAL(5.0, a);
        BOOST_REQUIRE_EQUAL(5.0, b);
    }

    // Test that the best guess correctly linearly interpolates
    // the function.
    {
        a = 2.0;
        b = 7.0;
        iterations = 10;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.1);
        BOOST_TEST_REQUIRE(CSolvers::bisection(a, b, -5.0, 5.0, &f1, iterations,
                                               equal, bestGuess));
        LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << f1(a)
                  << ", f(b) = " << f1(b) << ", iterations = " << iterations
                  << ", bestGuess = " << bestGuess);
        BOOST_REQUIRE_EQUAL(5.0, bestGuess);
    }

    // Test convergence on f(x) = cos(x) - x.
    {
        LOG_DEBUG(<< "-");
        LOG_DEBUG(<< "*** f(x) = cos(x) - x ***");
        double lastError = 0.7390851332151607;
        for (std::size_t i = 3; i < 20; ++i) {
            a = -10.0;
            b = 10.0;
            iterations = i;
            CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.0);
            CSolvers::bisection(a, b, &f3, iterations, equal, bestGuess);

            LOG_DEBUG(<< "iterations = " << iterations);
            BOOST_REQUIRE_EQUAL(i, iterations);

            LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << f3(a)
                      << ", f(b) = " << f3(b));
            BOOST_TEST_REQUIRE(f3(a) * f3(b) <= 0.0);

            double error = std::fabs(bestGuess - 0.7390851332151607);
            LOG_DEBUG(<< "bestGuess = " << bestGuess
                      << ", f(bestGuess) = " << f3(bestGuess) << ", error = " << error);
            BOOST_TEST_REQUIRE(error < std::fabs((a + b) / 2.0 - 0.7390851332151607));
            double convergenceFactor = error / lastError;
            lastError = error;
            if (i != 3) {
                LOG_DEBUG(<< "convergenceFactor = " << convergenceFactor);
            }
            LOG_DEBUG(<< "-");
        }

        double meanConvergenceFactor = std::pow(lastError / 0.7390851332151607, 1.0 / 20.0);
        LOG_DEBUG(<< "mean convergence factor = " << meanConvergenceFactor);
        BOOST_TEST_REQUIRE(meanConvergenceFactor < 0.4);
    }

    // Test convergence of f(x) = {  |x - 2.0/3.0|^0.2  x <= 2.0/3.0
    //                            { -|x - 2.0/3.0|^0.2  otherwise
    //
    // Note that we get uneven convergence rates spread across
    // each set of 4 iterations, but on average converge at a
    // rate of 0.5 per iteration.
    {
        LOG_DEBUG(<< "-");
        LOG_DEBUG(<< "*** f(x) = {  |x - 2.0/3.0|^0.2  x <= 2.0/3.0 ***");
        LOG_DEBUG(<< "           { -|x - 2.0/3.0|^0.2  otherwise");
        double lastInterval = 20.0;
        double lastError = 2.0 / 3.0;
        double convergenceFactor = 1.0;
        for (std::size_t i = 3; i < 40; ++i) {
            a = -10.0;
            b = 10.0;
            iterations = i;
            CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.0);
            CSolvers::bisection(a, b, &f4, iterations, equal, bestGuess);

            LOG_DEBUG(<< "iterations = " << iterations);
            BOOST_REQUIRE_EQUAL(i, iterations);

            LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << f4(a)
                      << ", f(b) = " << f4(b));
            BOOST_TEST_REQUIRE(f4(a) * f4(b) <= 0.0);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.5 * lastInterval, b - a, 1e-5);
            lastInterval = b - a;

            double error = std::fabs(bestGuess - 2.0 / 3.0);
            LOG_DEBUG(<< "bestGuess = " << bestGuess
                      << ", f(bestGuess) = " << f4(bestGuess) << ", error = " << error);
            BOOST_TEST_REQUIRE(error < std::fabs((a + b) / 2.0 - 2.0 / 3.0));
            convergenceFactor *= (error / lastError);
            lastError = error;

            if ((i - 2) % 4 == 0) {
                convergenceFactor = std::pow(convergenceFactor, 0.25);
                LOG_DEBUG(<< "convergence factor = " << convergenceFactor);
                if (i - 2 != 4) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.5, convergenceFactor, 1e-5);
                }
                convergenceFactor = 1.0;
            }
            LOG_DEBUG(<< "-");
        }

        double meanConvergenceFactor = std::pow(lastError / (2.0 / 3.0), 1.0 / 40.0);
        LOG_DEBUG(<< "mean convergence factor = " << meanConvergenceFactor);
        BOOST_TEST_REQUIRE(meanConvergenceFactor < 0.56);
    }
}

BOOST_AUTO_TEST_CASE(testBrent) {
    double a, b;
    double bestGuess;
    std::size_t iterations;

    // Test that the method fails if the root isn't bracketed.
    {
        a = 6.0;
        b = 10.0;
        iterations = 5;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.1);
        BOOST_TEST_REQUIRE(!CSolvers::brent(a, b, &f1, iterations, equal, bestGuess));
    }

    // Test that the method terminates if hits the root.
    {
        a = 0.0;
        b = 16.0;
        iterations = 10;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.1);
        BOOST_TEST_REQUIRE(CSolvers::brent(a, b, -5.0, 11.0, &f1, iterations, equal, bestGuess));
        BOOST_REQUIRE_EQUAL(static_cast<std::size_t>(1), iterations);
        BOOST_REQUIRE_EQUAL(5.0, bestGuess);
        BOOST_REQUIRE_EQUAL(5.0, a);
        BOOST_REQUIRE_EQUAL(5.0, b);
    }

    // Test the quadratic interpolation will solve for quadratic exactly.
    {
        a = 1.5;
        b = 5.0;
        iterations = 10;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.1);
        BOOST_TEST_REQUIRE(CSolvers::brent(a, b, &f2, iterations, equal, bestGuess));
        BOOST_REQUIRE_EQUAL(static_cast<std::size_t>(6), iterations);
        BOOST_REQUIRE_EQUAL(2.0, bestGuess);
        BOOST_REQUIRE_EQUAL(2.0, a);
        BOOST_REQUIRE_EQUAL(2.0, b);
    }

    // Test convergence on f(x) = cos(x) - x.
    {
        LOG_DEBUG(<< "*** f(x) = cos(x) - x ***");
        double lastError = 0.7390851332151607;
        for (std::size_t i = 3; i < 8; ++i) {
            a = -10.0;
            b = 10.0;
            iterations = i;
            CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.0);
            CSolvers::brent(a, b, &f3, iterations, equal, bestGuess);

            LOG_DEBUG(<< "-");
            LOG_DEBUG(<< "iterations = " << iterations);
            BOOST_REQUIRE_EQUAL(i, iterations);

            LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << f3(a)
                      << ", f(b) = " << f3(b));
            BOOST_TEST_REQUIRE(f3(a) * f3(b) <= 0.0);

            double error = std::fabs(bestGuess - 0.7390851332151607);
            LOG_DEBUG(<< "bestGuess = " << bestGuess
                      << ", f(bestGuess) = " << f3(bestGuess) << ", error = " << error);
            BOOST_TEST_REQUIRE(error < std::fabs((a + b) / 2.0 - 0.7390851332151607));
            double convergenceFactor = error / lastError;
            lastError = error;
            if (i != 3) {
                LOG_DEBUG(<< "convergenceFactor = " << convergenceFactor);
                BOOST_TEST_REQUIRE(convergenceFactor < 0.75);
            }
        }

        BOOST_TEST_REQUIRE(lastError < 5e-16);
    }

    // Test convergence on f(x) = {  |x - 2.0/3.0|^0.2  x <= 2.0/3.0
    //                            { -|x - 2.0/3.0|^0.2  otherwise
    {
        LOG_DEBUG(<< "-");
        LOG_DEBUG(<< "*** f(x) = {  |x - 2.0/3.0|^0.2  x <= 2.0/3.0 ***");
        LOG_DEBUG(<< "           { -|x - 2.0/3.0|^0.2  otherwise");
        double lastError = 2.0 / 3.0;
        for (std::size_t i = 3; i < 40; ++i) {
            a = -10.0;
            b = 10.0;
            iterations = i;
            CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 0.0);
            CSolvers::brent(a, b, &f4, iterations, equal, bestGuess);

            LOG_DEBUG(<< "-");
            LOG_DEBUG(<< "iterations = " << iterations);
            BOOST_REQUIRE_EQUAL(i, iterations);

            LOG_DEBUG(<< "a = " << a << ", b = " << b << ", f(a) = " << f4(a)
                      << ", f(b) = " << f4(b));
            BOOST_TEST_REQUIRE(f4(a) * f4(b) <= 0.0);

            double error = std::fabs(bestGuess - 2.0 / 3.0);
            LOG_DEBUG(<< "bestGuess = " << bestGuess
                      << ", f(bestGuess) = " << f4(bestGuess) << ", error = " << error);
            BOOST_TEST_REQUIRE(error < std::fabs((a + b) / 2.0 - 2.0 / 3.0));
            double convergenceFactor = error / lastError;
            lastError = error;
            LOG_DEBUG(<< "convergence factor = " << convergenceFactor);
        }

        double meanConvergenceFactor = std::pow(lastError / (2.0 / 3.0), 1.0 / 40.0);
        LOG_DEBUG(<< "mean convergence factor = " << meanConvergenceFactor);
        BOOST_TEST_REQUIRE(meanConvergenceFactor < 0.505);
    }
}

BOOST_AUTO_TEST_CASE(testSublevelSet) {
    using TDoubleDoublePr = std::pair<double, double>;

    // Should converge immediately to minimum of quadratic.
    TDoubleDoublePr sublevelSet;
    CSolvers::sublevelSet(0.0, 8.0, 2.0, 42.0, &f2, 0.0, 10, sublevelSet);
    LOG_DEBUG(<< "sublevelSet = " << core::CContainerPrinter::print(sublevelSet));

    LOG_DEBUG(<< "*** f(x) = 1.1 * exp(-(x-4)^2) + 0.4 * exp(-(x-10)^2/4) ***");

    double fmax = 0.9 * f5(10.0);
    for (std::size_t i = 0; i < 30; ++i, fmax *= 0.9) {
        LOG_DEBUG(<< "fmax = " << fmax);

        if (CSolvers::sublevelSet(4.0, 10.0, f5(4.0), f5(10.0), &f5, fmax, 10, sublevelSet)) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(fmax, f5(sublevelSet.first), 1e-5);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(fmax, f5(sublevelSet.second), 1e-5);
        } else {
            BOOST_TEST_REQUIRE(sublevelSet.second - sublevelSet.first < 1e-4);
        }
        LOG_DEBUG(<< "sublevelSet = " << core::CContainerPrinter::print(sublevelSet));
        LOG_DEBUG(<< "f(a) = " << f5(sublevelSet.first)
                  << ", f(b) = " << f5(sublevelSet.second));
    }

    LOG_DEBUG(<< "*** f(x) = 1.1 * exp(-2.0*(x-4)^2) + 0.1 * exp(-(x-6)^2) + 0.4 * exp(-(x-10)^2/2) ***");

    fmax = 0.9 * f6(10.0);
    for (std::size_t i = 0; i < 15; ++i, fmax *= 0.9) {
        LOG_DEBUG(<< "fmax = " << fmax);

        bool found = CSolvers::sublevelSet(4.0, 10.0, f6(4.0), f6(10.0), &f6,
                                           fmax, 15, sublevelSet);

        LOG_DEBUG(<< "sublevelSet = " << core::CContainerPrinter::print(sublevelSet));
        LOG_DEBUG(<< "f(a) = " << f6(sublevelSet.first)
                  << ", f(b) = " << f6(sublevelSet.second));

        if (found) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(fmax, f6(sublevelSet.first), 1e-4);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(fmax, f6(sublevelSet.second), 1e-4);
        } else {
            BOOST_TEST_REQUIRE(sublevelSet.second - sublevelSet.first < 1e-4);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
