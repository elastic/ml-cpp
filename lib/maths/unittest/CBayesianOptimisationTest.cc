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

void CBayesianOptimisationTest::testLikelihood() {

    // Test that the likelihood gradient matches the numerical likelihood gradient.

    maths::CBayesianOptimisation bopt{
        {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

    bopt.add(vector({1.0, 1.0, 1.0, 1.0}), 4.0, 1.0);
    bopt.add(vector({2.0, 1.0, -2.0, 1.0}), 10.0, 1.0);
    bopt.add(vector({0.5, -1.0, 3.0, -1.0}), 11.25, 1.0);
    bopt.add(vector({-1.0, -1.0, -2.0, -2.0}), 10.0, 1.0);

    auto l = bopt.minusLikelihood();
    auto g = bopt.minusLikelihoodGradient();

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

        CPPUNIT_ASSERT((expectedGradient - gradient).norm() <
                       2e-5 * expectedGradient.norm());
    }
}

CppUnit::Test* CBayesianOptimisationTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBayesianOptimisationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBayesianOptimisationTest>(
        "CBayesianOptimisationTest::testLikelihood", &CBayesianOptimisationTest::testLikelihood));

    return suiteOfTests;
}
