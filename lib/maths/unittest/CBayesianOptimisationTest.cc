/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBayesianOptimisationTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CLinearAlgebraEigen.h>

#include <test/CRandomNumbers.h>

#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TVector = maths::CDenseVector<double>;

TVector vector(std::vector<double> components) {
    TVector result(components.size());
    int i = 0;
    for (auto component : components) {
        result(i++) = component;
    }
    return result;
}
}

void CBayesianOptimisationTest::testLikelihoodGradient() {

    // Test that the likelihood gradient matches the numerical gradient.

    test::CRandomNumbers rng;
    TDoubleVec coordinates;

    for (std::size_t test = 0; test < 10; ++test) {

        maths::CBayesianOptimisation bopt{
            {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

        for (std::size_t i = 0; i < 4; ++i) {
            rng.generateUniformSamples(-10.0, 10.0, 4, coordinates);
            TVector x{vector(coordinates)};
            bopt.add(x, x.squaredNorm(), 1.0);
        }
        bopt.maximumLikelihoodKernel();

        maths::CBayesianOptimisation::TLikelihoodFunc l;
        maths::CBayesianOptimisation::TLikelihoodGradientFunc g;
        std::tie(l, g) = bopt.minusLikelihoodAndGradient();

        TDoubleVec parameters;
        for (std::size_t probe = 0; probe < 10; ++probe) {

            rng.generateUniformSamples(0.1, 1.0, 5, parameters);

            TVector a{5};
            for (std::size_t i = 0; i < 5; ++i) {
                a(i) = parameters[i];
            }

            TVector expectedGradient{5};
            TVector eps{5};
            eps.setZero();
            for (std::size_t i = 0; i < 5; ++i) {
                eps(i) = 1e-3;
                expectedGradient(i) = (l(a + eps) - l(a - eps)) / 2e-3;
                eps(i) = 0.0;
            }

            TVector gradient{g(a)};

            CPPUNIT_ASSERT((expectedGradient - gradient).norm() <
                           1e-3 * expectedGradient.norm());
        }
    }
}

void CBayesianOptimisationTest::testMaximumLikelihoodKernel() {

    // Check that the kernel parameters we choose are at a minimum of the likelihood
    // as a function of those parameters.

    test::CRandomNumbers rng;
    TDoubleVec coordinates;
    TDoubleVec noise;

    for (std::size_t test = 0; test < 50; ++test) {

        maths::CBayesianOptimisation bopt{
            {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

        double scale{5.0 / std::sqrt(static_cast<double>(test + 1))};

        for (std::size_t i = 0; i < 10; ++i) {
            rng.generateUniformSamples(-10.0, 10.0, 4, coordinates);
            rng.generateNormalSamples(0.0, 2.0, 1, noise);
            TVector x{vector(coordinates)};
            double fx{scale * x.squaredNorm() + noise[0]};
            bopt.add(x, fx, 1.0);
        }

        TVector parameters{bopt.maximumLikelihoodKernel()};

        maths::CBayesianOptimisation::TLikelihoodFunc l;
        maths::CBayesianOptimisation::TLikelihoodGradientFunc g;
        std::tie(l, g) = bopt.minusLikelihoodAndGradient();

        double minusML{l(parameters)};
        LOG_TRACE(<< "maximum likelihood = " << -minusML);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, g(parameters).norm(), 0.05);

        TVector eps{parameters.size()};
        eps.setZero();
        for (std::size_t i = 0; i < 4; ++i) {
            eps(i) = 0.1;
            double minusLPlusEps{l(parameters + eps)};
            eps(i) = -0.1;
            double minusLMinusEps{l(parameters + eps)};
            eps(i) = 0.0;
            CPPUNIT_ASSERT(minusML < minusLPlusEps);
            CPPUNIT_ASSERT(minusML < minusLMinusEps);
        }
    }
}

void CBayesianOptimisationTest::testExpectedImprovementGradient() {

    // Test that the expected improvement gradient matches the numerical gradient.

    test::CRandomNumbers rng;
    TDoubleVec coordinates;

    for (std::size_t test = 0; test < 1; ++test) {

        maths::CBayesianOptimisation bopt{
            {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

        for (std::size_t i = 0; i < 8; ++i) {
            rng.generateUniformSamples(-10.0, 10.0, 4, coordinates);
            TVector x{vector(coordinates)};
            bopt.add(x, x.squaredNorm(), 1.0);
        }
        bopt.maximumLikelihoodKernel();

        maths::CBayesianOptimisation::TEIFunc ei;
        maths::CBayesianOptimisation::TEIGradientFunc eig;
        std::tie(ei, eig) = bopt.minusExpectedImprovementAndGradient();

        TDoubleVec parameters;
        for (std::size_t probe = 0; probe < 10; ++probe) {
            rng.generateUniformSamples(-0.5, 0.5, 4, coordinates);

            TVector x{4};
            for (std::size_t i = 0; i < 4; ++i) {
                x(i) = coordinates[i];
            }

            TVector expectedGradient{4};
            TVector eps{4};
            eps.setZero();
            for (std::size_t i = 0; i < 4; ++i) {
                eps(i) = 1e-3;
                expectedGradient(i) = (ei(x + eps) - ei(x - eps)) / 2e-3;
                eps(i) = 0.0;
            }

            TVector gradient{eig(x)};

            CPPUNIT_ASSERT((expectedGradient - gradient).norm() <
                           1e-2 * expectedGradient.norm());
        }
    }
}

void CBayesianOptimisationTest::testMaximumExpectedImprovement() {

    // This tests the efficiency of the search on a variety of non-convex functions.
    // We check the value of the function we find after fixed number of iterations
    // vs a random search baseline.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;
    TDoubleVec centreCoordinates;
    TDoubleVec coordinateScales;
    TDoubleVec evaluationCoordinates;
    TDoubleVec randomSearch;

    TVector a(vector({-10.0, -10.0, -10.0, -10.0}));
    TVector b(vector({10.0, 10.0, 10.0, 10.0}));

    TMeanAccumulator gain;

    for (std::size_t test = 0; test < 20; ++test) {

        rng.generateUniformSamples(-10.0, 10.0, 12, centreCoordinates);
        rng.generateUniformSamples(0.3, 4.0, 12, coordinateScales);

        // Use sum of some different quadratric functions.
        TVector centres[]{TVector{4}, TVector{4}, TVector{4}};
        TVector scales[]{TVector{4}, TVector{4}, TVector{4}};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                centres[i](j) = centreCoordinates[4 * i + j];
                scales[i](j) = coordinateScales[4 * i + j];
            }
        }
        auto f = [&](const TVector& x) {
            double f1{(x - centres[0]).transpose() * scales[0].asDiagonal() *
                      (x - centres[0])};
            double f2{(x - centres[1]).transpose() * scales[1].asDiagonal() *
                      (x - centres[1])};
            double f3{(x - centres[2]).transpose() * scales[2].asDiagonal() *
                      (x - centres[2])};
            return f1 + f2 + f3;
        };

        maths::CBayesianOptimisation bopt{
            {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

        double fminBopt{std::numeric_limits<double>::max()};
        double fminRs{std::numeric_limits<double>::max()};

        for (std::size_t i = 0; i < 5; ++i) {
            rng.generateUniformSamples(-10.0, 10.0, 4, evaluationCoordinates);
            TVector x{vector(evaluationCoordinates)};
            LOG_TRACE(<< "initial " << x.transpose() << ", f(initial) = " << f(x));
            bopt.add(x, f(x), 10.0);
            fminBopt = std::min(fminBopt, f(x));
            fminRs = std::min(fminRs, f(x));
        }

        LOG_TRACE(<< "Bayesian optimisation...");
        for (std::size_t i = 0; i < 10; ++i) {
            TVector x{bopt.maximumExpectedImprovement()};
            LOG_TRACE(<< "x = " << x.transpose() << ", f(x) = " << f(x));
            bopt.add(x, f(x), 10.0);
            fminBopt = std::min(fminBopt, f(x));
        }

        LOG_TRACE(<< "random search...");
        for (std::size_t i = 0; i < 10; ++i) {
            rng.generateUniformSamples(0.0, 1.0, 4, randomSearch);
            TVector x{a + vector(randomSearch).asDiagonal() * (b - a)};
            LOG_TRACE(<< "x = " << x.transpose() << ", f(x) = " << f(x));
            fminRs = std::min(fminRs, f(x));
        }

        LOG_TRACE(<< "gain = " << fminRs / fminBopt);
        gain.add(fminRs / fminBopt);
    }

    LOG_DEBUG(<< "mean gain = " << maths::CBasicStatistics::mean(gain));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(gain) > 1.29);
}

void CBayesianOptimisationTest::testPersistRestore() {
    // 1d
    {
        std::vector<double> minBoundary{0.};
        std::vector<double> maxBoundary{10.};
        // empty
        {
            std::vector<std::vector<double>> parameterFunctionValues{};
            testPersistRestoreSubroutine(minBoundary, maxBoundary, parameterFunctionValues);
        }
        // with data
        {
            std::vector<std::vector<double>> parameterFunctionValues{
                {5., 1., 0.2},
                {7., 1., 0.2},
            };
            testPersistRestoreSubroutine(minBoundary, maxBoundary, parameterFunctionValues);
        }
    }

    // 2d
    {
        std::vector<double> minBoundary{0., -1.};
        std::vector<double> maxBoundary{10., 1.};
        // empty
        {
            std::vector<std::vector<double>> parameterFunctionValues{};
            testPersistRestoreSubroutine(minBoundary, maxBoundary, parameterFunctionValues);
        }
        // with data
        {
            std::vector<std::vector<double>> parameterFunctionValues{
                {5., 0., 1., 0.2},
                {7., 0., 1., 0.2},
            };
            testPersistRestoreSubroutine(minBoundary, maxBoundary, parameterFunctionValues);
        }
    }
}

void CBayesianOptimisationTest::testPersistRestoreSubroutine(
    const std::vector<double>& minBoundary,
    const std::vector<double>& maxBoundary,
    const std::vector<std::vector<double>>& parameterFunctionValues) const {
    std::stringstream persistOnceSStream;
    std::stringstream persistTwiceSStream;
    std::size_t dimensions = minBoundary.size();
    // persist
    {
        maths::CBayesianOptimisation::TDoubleDoublePrVec parameterBoundaries;
        for (std::size_t i = 0u; i < dimensions; ++i) {
            parameterBoundaries.emplace_back(minBoundary[i], maxBoundary[i]);
        }
        maths::CBayesianOptimisation bayesianOptimisation(parameterBoundaries);
        if (parameterFunctionValues.empty() == false) {
            for (auto parameterFunctionValue : parameterFunctionValues) {
                maths::CBayesianOptimisation::TVector parameter(dimensions);
                for (std::size_t i = 0u; i < dimensions; ++i) {
                    parameter(i) = parameterFunctionValue[i];
                }
                bayesianOptimisation.add(parameter, parameterFunctionValue[dimensions],
                                         parameterFunctionValue[dimensions + 1]);
            }
        }

        core::CJsonStatePersistInserter inserter(persistOnceSStream);
        bayesianOptimisation.acceptPersistInserter(inserter);
        persistOnceSStream.flush();
    }
    // and restore
    {
        maths::CBayesianOptimisation bayesianOptimisation({});
        core::CJsonStateRestoreTraverser traverser(persistOnceSStream);
        bayesianOptimisation.acceptRestoreTraverser(traverser);

        core::CJsonStatePersistInserter inserter(persistTwiceSStream);
        bayesianOptimisation.acceptPersistInserter(inserter);
        persistTwiceSStream.flush();
    }
    CPPUNIT_ASSERT_EQUAL(persistOnceSStream.str(), persistTwiceSStream.str());
}

CppUnit::Test* CBayesianOptimisationTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBayesianOptimisationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBayesianOptimisationTest>(
        "CBayesianOptimisationTest::testLikelihoodGradient",
        &CBayesianOptimisationTest::testLikelihoodGradient));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBayesianOptimisationTest>(
        "CBayesianOptimisationTest::testMaximumLikelihoodKernel",
        &CBayesianOptimisationTest::testMaximumLikelihoodKernel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBayesianOptimisationTest>(
        "CBayesianOptimisationTest::testExpectedImprovementGradient",
        &CBayesianOptimisationTest::testExpectedImprovementGradient));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBayesianOptimisationTest>(
        "CBayesianOptimisationTest::testMaximumExpectedImprovement",
        &CBayesianOptimisationTest::testMaximumExpectedImprovement));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBayesianOptimisationTest>(
        "CBayesianOptimisationTest::testPersistRestore",
        &CBayesianOptimisationTest::testPersistRestore));

    return suiteOfTests;
}
