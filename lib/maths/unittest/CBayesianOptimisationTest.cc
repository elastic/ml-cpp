/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBayesianOptimisationTest.h"

#include <maths/CBayesianOptimisation.h>
#include <maths/CLinearAlgebraEigen.h>

#include <test/CRandomNumbers.h>

#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TVector = maths::CDenseVector<double>;

TVector vector(std::initializer_list<double> components) {
    TVector result(components.size());
    int i = 0;
    for (auto component : components) {
        result(i++) = component;
    }
    return result;
}

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

    maths::CBayesianOptimisation bopt{
        {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

    bopt.add(vector({1.0, 1.0, 1.0, 1.0}), 4.0, 1.0);
    bopt.add(vector({2.0, 1.0, -2.0, 1.0}), 10.0, 1.0);
    bopt.add(vector({0.5, -1.0, 3.0, -1.0}), 11.25, 1.0);
    bopt.add(vector({-1.0, -1.0, -2.0, -2.0}), 10.0, 1.0);

    maths::CBayesianOptimisation::TLikelihoodFunc l;
    maths::CBayesianOptimisation::TLikelihoodGradientFunc g;
    std::tie(l, g) = bopt.minusLikelihoodAndGradient();

    test::CRandomNumbers rng;
    TDoubleVec parameters;

    for (std::size_t test = 0; test < 100; ++test) {

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

        LOG_DEBUG(<< expectedGradient.transpose());
        LOG_DEBUG(<< gradient.transpose());

        CPPUNIT_ASSERT((expectedGradient - gradient).norm() <
                       2e-5 * expectedGradient.norm());
    }
}

void CBayesianOptimisationTest::testMaximumLikelihoodKernel() {

    // Check that the parameters we choose are at a minimum of the likelihood as
    // a function of the kernel parameters.

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

    maths::CBayesianOptimisation bopt{
        {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

    bopt.add(vector({1.0, 1.0, 1.0, 1.0}), 4.0, 1.0);
    bopt.add(vector({2.0, 1.0, -2.0, 1.0}), 10.0, 1.0);
    bopt.add(vector({0.5, -1.0, 3.0, -1.0}), 11.25, 1.0);
    bopt.add(vector({-1.0, -1.0, -2.0, -2.0}), 10.0, 1.0);

    maths::CBayesianOptimisation::TEIFunc ei;
    maths::CBayesianOptimisation::TEIGradientFunc eig;
    std::tie(ei, eig) = bopt.minusExpectedImprovementAndGradient();

    test::CRandomNumbers rng;
    TDoubleVec coordinates;

    for (std::size_t test = 0; test < 100; ++test) {

        rng.generateUniformSamples(0.1, 1.0, 4, coordinates);

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
                       2e-5 * expectedGradient.norm());
    }
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

    return suiteOfTests;
}
