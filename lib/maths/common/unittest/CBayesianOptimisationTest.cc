/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBayesianOptimisation.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CSampling.h>
#include <maths/common/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CBayesianOptimisationTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TVector = maths::common::CDenseVector<double>;
using TVectorVec = std::vector<TVector>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
struct SFunctionParams {
    double s_Xl;
    double s_Xu;
    double s_F0;
    double s_Scale;
};
using TFunctionParamsVec = std::vector<SFunctionParams>;

TVector vector(TDoubleVec components) {
    TVector result(components.size());
    int i = 0;
    for (auto component : components) {
        result(i++) = component;
    }
    return result;
}

void testPersistRestoreIsIdempotent(const TDoubleVec& minBoundary,
                                    const TDoubleVec& maxBoundary,
                                    const std::vector<TDoubleVec>& parameterFunctionValues) {
    std::stringstream persistOnceSStream;
    std::stringstream persistTwiceSStream;
    std::size_t dimensions = minBoundary.size();

    std::string topLevelTag{"bayesian_optimisation"};

    // persist
    {
        maths::common::CBayesianOptimisation::TDoubleDoublePrVec parameterBoundaries;
        for (std::size_t i = 0; i < dimensions; ++i) {
            parameterBoundaries.emplace_back(minBoundary[i], maxBoundary[i]);
        }
        maths::common::CBayesianOptimisation bayesianOptimisation{parameterBoundaries};
        if (parameterFunctionValues.size() > 0) {
            for (auto parameterFunctionValue : parameterFunctionValues) {
                maths::common::CBayesianOptimisation::TVector parameter(dimensions);
                for (std::size_t i = 0; i < dimensions; ++i) {
                    parameter(i) = parameterFunctionValue[i];
                }
                bayesianOptimisation.add(parameter, parameterFunctionValue[dimensions],
                                         parameterFunctionValue[dimensions + 1]);
            }
        }

        core::CJsonStatePersistInserter inserter(persistOnceSStream);
        inserter.insertLevel(
            topLevelTag, std::bind(&maths::common::CBayesianOptimisation::acceptPersistInserter,
                                   &bayesianOptimisation, std::placeholders::_1));
        persistOnceSStream.flush();
    }
    // and restore
    {
        core::CJsonStateRestoreTraverser traverser{persistOnceSStream};
        maths::common::CBayesianOptimisation bayesianOptimisation{traverser};

        core::CJsonStatePersistInserter inserter(persistTwiceSStream);
        inserter.insertLevel(
            topLevelTag, std::bind(&maths::common::CBayesianOptimisation::acceptPersistInserter,
                                   &bayesianOptimisation, std::placeholders::_1));
        persistTwiceSStream.flush();
    }
    LOG_DEBUG(<< "First string " << persistOnceSStream.str());
    LOG_DEBUG(<< "Second string " << persistTwiceSStream.str());
    BOOST_REQUIRE_EQUAL(persistOnceSStream.str(), persistTwiceSStream.str());
}

maths::common::CBayesianOptimisation
initBayesianOptimization(std::size_t dim, std::size_t numSamples, double min, double max) {
    test::CRandomNumbers rng;
    TDoubleVec trainSamples(numSamples * dim);
    rng.generateUniformSamples(min, max, trainSamples.size(), trainSamples);
    maths::common::CBayesianOptimisation::TDoubleDoublePrVec boundaries;
    boundaries.reserve(dim);
    for (std::size_t d = 0; d < dim; ++d) {
        boundaries.emplace_back(min, max);
    }
    maths::common::CBayesianOptimisation bopt{boundaries};
    for (std::size_t i = 0; i < numSamples; i += 2) {
        TVector x{vector({trainSamples[i], trainSamples[i + 1]})};
        bopt.add(x, x.squaredNorm(), 1.0);
    }
    TDoubleVec kernelParameters(dim + 1, 0.5);
    kernelParameters[0] = 0.7;
    bopt.kernelParameters(vector(kernelParameters));
    return bopt;
}
}

BOOST_AUTO_TEST_CASE(testLikelihoodGradient) {

    // Test that the likelihood gradient matches the numerical gradient.

    test::CRandomNumbers rng;
    TDoubleVec coordinates;

    for (std::size_t test = 0; test < 10; ++test) {

        maths::common::CBayesianOptimisation bopt{
            {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

        for (std::size_t i = 0; i < 4; ++i) {
            rng.generateUniformSamples(-10.0, 10.0, 4, coordinates);
            TVector x{vector(coordinates)};
            bopt.add(x, x.squaredNorm(), 1.0);
        }
        bopt.maximumLikelihoodKernel();

        maths::common::CBayesianOptimisation::TLikelihoodFunc l;
        maths::common::CBayesianOptimisation::TLikelihoodGradientFunc g;
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

            BOOST_TEST_REQUIRE((expectedGradient - gradient).norm() <
                               1e-3 * expectedGradient.norm());
        }
    }
}

BOOST_AUTO_TEST_CASE(testMaximumLikelihoodKernel) {

    // Check that the kernel parameters we choose are at a minimum of the likelihood
    // as a function of those parameters.

    test::CRandomNumbers rng;
    TDoubleVec coordinates;
    TDoubleVec noise;

    for (std::size_t test = 0; test < 50; ++test) {

        maths::common::CBayesianOptimisation bopt{
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

        maths::common::CBayesianOptimisation::TLikelihoodFunc l;
        maths::common::CBayesianOptimisation::TLikelihoodGradientFunc g;
        std::tie(l, g) = bopt.minusLikelihoodAndGradient();

        double minusML{l(parameters)};
        LOG_TRACE(<< "maximum likelihood = " << -minusML);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, g(parameters).norm(), 0.05);

        TVector eps{parameters.size()};
        eps.setZero();
        for (std::size_t i = 0; i < 4; ++i) {
            eps(i) = 0.1;
            double minusLPlusEps{l(parameters + eps)};
            eps(i) = -0.1;
            double minusLMinusEps{l(parameters + eps)};
            eps(i) = 0.0;
            BOOST_TEST_REQUIRE(minusML < minusLPlusEps);
            BOOST_TEST_REQUIRE(minusML < minusLMinusEps);
        }
    }
}

BOOST_AUTO_TEST_CASE(testExpectedImprovementGradient) {

    // Test that the expected improvement gradient matches the numerical gradient.

    test::CRandomNumbers rng;
    TDoubleVec coordinates;

    for (std::size_t test = 0; test < 1; ++test) {

        maths::common::CBayesianOptimisation bopt{
            {{-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}}};

        for (std::size_t i = 0; i < 8; ++i) {
            rng.generateUniformSamples(-10.0, 10.0, 4, coordinates);
            TVector x{vector(coordinates)};
            bopt.add(x, x.squaredNorm(), 1.0);
        }
        bopt.maximumLikelihoodKernel();

        maths::common::CBayesianOptimisation::TEIFunc ei;
        maths::common::CBayesianOptimisation::TEIGradientFunc eig;
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

            BOOST_TEST_REQUIRE((expectedGradient - gradient).norm() <
                               1e-2 * expectedGradient.norm());
        }
    }
}

BOOST_AUTO_TEST_CASE(testMaximumExpectedImprovement) {

    // This tests the efficiency of the search on a variety of non-convex functions.
    // We check the value of the function we find after fixed number of iterations
    // vs a random search baseline.

    test::CRandomNumbers rng;
    TDoubleVec centreCoordinates;
    TDoubleVec coordinateScales;
    TDoubleVec evaluationCoordinates;
    TDoubleVec randomSearch;

    TVector a(vector({-10.0, -10.0, -10.0, -10.0}));
    TVector b(vector({10.0, 10.0, 10.0, 10.0}));

    std::size_t wins{0};
    std::size_t losses{0};
    TMeanAccumulator meanImprovementBopt;
    TMeanAccumulator meanImprovementRs;

    for (std::size_t test = 0; test < 50; ++test) {

        rng.generateUniformSamples(-10.0, 10.0, 12, centreCoordinates);
        rng.generateUniformSamples(0.3, 4.0, 12, coordinateScales);

        // Use sum of some different quadratic functions.
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
            return 100.0 + f1 - 0.2 * f2 + f3;
        };

        maths::common::CBayesianOptimisation bopt{
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
        double f0Bopt{fminBopt};
        for (std::size_t i = 0; i < 30; ++i) {
            TVector x;
            std::tie(x, std::ignore) = bopt.maximumExpectedImprovement();
            LOG_TRACE(<< "x = " << x.transpose() << ", f(x) = " << f(x));
            bopt.add(x, f(x), 10.0);
            fminBopt = std::min(fminBopt, f(x));
        }
        double improvementBopt{(f0Bopt - fminBopt) / f0Bopt};

        LOG_TRACE(<< "random search...");
        double f0Rs{fminRs};
        for (std::size_t i = 0; i < 30; ++i) {
            rng.generateUniformSamples(0.0, 1.0, 4, randomSearch);
            TVector x{a + vector(randomSearch).asDiagonal() * (b - a)};
            LOG_TRACE(<< "x = " << x.transpose() << ", f(x) = " << f(x));
            fminRs = std::min(fminRs, f(x));
        }
        double improvementRs{(f0Rs - fminRs) / f0Rs};

        LOG_DEBUG(<< "% improvement BO = " << 100.0 * improvementBopt
                  << ", % improvement RS = " << 100.0 * improvementRs);
        wins += improvementBopt > improvementRs ? 1 : 0;
        losses += improvementBopt > improvementRs ? 0 : 1;
        meanImprovementBopt.add(improvementBopt);
        meanImprovementRs.add(improvementRs);
    }

    LOG_DEBUG(<< "wins = " << wins << ", losses = " << losses);
    LOG_DEBUG(<< "mean % improvement BO = "
              << 100.0 * maths::common::CBasicStatistics::mean(meanImprovementBopt));
    LOG_DEBUG(<< "mean % improvement RS = "
              << 100.0 * maths::common::CBasicStatistics::mean(meanImprovementRs));
    BOOST_TEST_REQUIRE(wins > static_cast<std::size_t>(0.95 * 50)); // 95% better
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanImprovementBopt) >
                       1.6 * maths::common::CBasicStatistics::mean(meanImprovementRs)); // 60% mean improvement
}

BOOST_AUTO_TEST_CASE(testKernelInvariants) {

    // Test that the kernel parameters we estimate do not change when:
    //   1. Changing the function domain,
    //   2. Changing the function level,
    //   3. Linearly scaling the function.

    TFunctionParamsVec tests{{0.0, 100.0, 0.0, 1.0},
                             {0.0, 1000.0, 0.0, 1.0},
                             {0.0, 100.0, 10.0, 1.0},
                             {0.0, 100.0, 0.0, 2.0}};

    TVectorVec kernelParameters;

    for (const auto& test : tests) {

        test::CRandomNumbers rng;

        std::size_t dimension{2};
        double xl{test.s_Xl};
        double xu{test.s_Xu};
        double f0{test.s_F0};
        double scale{test.s_Scale};

        TDoubleVec coords;
        rng.generateUniformSamples(xl, xu, dimension * 20, coords);

        maths::common::CBayesianOptimisation::TDoubleDoublePrVec bb;
        for (std::size_t i = 0; i < dimension; ++i) {
            bb.emplace_back(xl, xu);
        }

        maths::common::CBayesianOptimisation bopt{bb};
        for (std::size_t i = 0; i < 10; ++i) {
            TVector x{dimension};
            for (std::size_t j = 0; j < dimension; ++j) {
                x(j) = coords[i * dimension + j];
            }
            bopt.maximumLikelihoodKernel();
            bopt.add(x, scale * x.norm() + f0, scale * scale * (xu - xl) * (xu - xl) * 0.0001);
        }

        kernelParameters.push_back(bopt.maximumLikelihoodKernel());
    }

    for (std::size_t i = 1; i < kernelParameters.size(); ++i) {
        BOOST_TEST_REQUIRE((kernelParameters[i] - kernelParameters[0]).norm() < 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(testPersistRestore) {
    // 1d
    {
        TDoubleVec minBoundary{0.0};
        TDoubleVec maxBoundary{10.0};
        // empty
        {
            std::vector<TDoubleVec> parameterFunctionValues{};
            testPersistRestoreIsIdempotent(minBoundary, maxBoundary, parameterFunctionValues);
        }
        // with data
        {
            std::vector<TDoubleVec> parameterFunctionValues{
                {5.0, 1.0, 0.2},
                {7.0, 1.0, 0.2},
            };
            testPersistRestoreIsIdempotent(minBoundary, maxBoundary, parameterFunctionValues);
        }
    }

    // 2d
    {
        TDoubleVec minBoundary{0.0, -1.0};
        TDoubleVec maxBoundary{10.0, 1.0};
        // empty
        {
            std::vector<TDoubleVec> parameterFunctionValues{};
            testPersistRestoreIsIdempotent(minBoundary, maxBoundary, parameterFunctionValues);
        }
        // with data
        {
            std::vector<TDoubleVec> parameterFunctionValues{
                {5.0, 0.0, 1.0, 0.2},
                {7.0, 0.0, 1.0, 0.2},
            };
            testPersistRestoreIsIdempotent(minBoundary, maxBoundary, parameterFunctionValues);
        }
    }
}

BOOST_AUTO_TEST_CASE(testEvaluate) {
    TDoubleVec coordinates{0.25, 0.5, 0.75};
    for (auto scale : {1.0, 0.5, 2.0}) {
        maths::common::CBayesianOptimisation bopt{{{0.0, scale}, {0.0, scale}}};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                TVector x{vector({scale * coordinates[i], scale * coordinates[j]})};
                bopt.add(x, x.squaredNorm() / maths::common::CTools::pow2(scale), 0.0);
            }
        }

        // Because we scale the values in add if we fix the kernel parameters
        // then the GP value is the same at the same relative positions within
        // the bounding box.
        TVector kernelParameters(vector({1.0, 0.5, 0.5}));
        bopt.kernelParameters(kernelParameters);

        TDoubleVecVec testPoints{{0.3 * scale, 0.3 * scale},
                                 {0.3 * scale, 0.6 * scale},
                                 {0.6 * scale, 0.3 * scale}};
        TDoubleVec testTargets{0.17823499, 0.45056931, 0.45056931};

        for (std::size_t i = 0; i < testPoints.size(); ++i) {
            TVector x{vector(testPoints[i])};
            double actualTarget{bopt.evaluate(x)};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(actualTarget, testTargets[i], 1e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE(testEvaluate1D) {
    test::CRandomNumbers rng;
    std::size_t dim{2};
    std::size_t mcSamples{100};
    maths::common::CBayesianOptimisation bopt{initBayesianOptimization(dim, 20, 0.0, 1.0)};
    double f0{bopt.anovaConstantFactor()};

    TDoubleVecVec testSamples;
    maths::common::CSampling::sobolSequenceSample(dim, mcSamples, testSamples);

    TDoubleVec testInput(1);
    rng.generateUniformSamples(0, 1, 1, testInput);

    for (int d = 0; d < static_cast<int>(dim); ++d) {
        TMeanAccumulator meanAccumulator;
        double ftActual{bopt.evaluate1D(testInput[0], d)};
        for (std::size_t i = 0; i < mcSamples; ++i) {
            TVector input{vector(testSamples[i])};
            input(d) = testInput[0];
            meanAccumulator.add(bopt.evaluate(input) - f0);
        }
        double ftExpected{maths::common::CBasicStatistics::mean(meanAccumulator)};
        BOOST_REQUIRE_CLOSE_ABSOLUTE(ftActual, ftExpected, 5e-4);
    }
}

BOOST_AUTO_TEST_CASE(testAnovaConstantFactor) {
    std::size_t dim{2};
    std::size_t mcSamples{1000};
    TDoubleVecVec testSamples;
    maths::common::CSampling::sobolSequenceSample(dim, mcSamples, testSamples);

    auto verify = [&](double min, double max) {
        TMeanAccumulator meanAccumulator;
        maths::common::CBayesianOptimisation bopt{initBayesianOptimization(dim, 20, min, max)};
        double f0Actual{bopt.anovaConstantFactor()};
        for (std::size_t i = 0; i < mcSamples; ++i) {
            TVector input{(vector(testSamples[i]) * (max - min)).array() + min};
            meanAccumulator.add(bopt.evaluate(input));
        }
        double f0Expected{maths::common::CBasicStatistics::mean(meanAccumulator)};
        BOOST_REQUIRE_CLOSE_ABSOLUTE(f0Actual, f0Expected, 5e-3);
    };
    verify(0.0, 1.0);
    verify(-3.0, 3.0);
    verify(0.2, 0.8);
}

BOOST_AUTO_TEST_CASE(testAnovaTotalVariance) {
    std::size_t dim{2};
    std::size_t mcSamples{1000};
    TDoubleVecVec testSamples;
    maths::common::CSampling::sobolSequenceSample(dim, mcSamples, testSamples);

    auto verify = [&](double min, double max) {
        TMeanAccumulator meanAccumulator;
        maths::common::CBayesianOptimisation bopt{initBayesianOptimization(dim, 20, min, max)};
        double f0{bopt.anovaConstantFactor()};
        double totalVarianceActual{bopt.anovaTotalVariance()};
        for (std::size_t i = 0; i < mcSamples; ++i) {
            TVector input{(vector(testSamples[i]) * (max - min)).array() + min};
            meanAccumulator.add(maths::common::CTools::pow2(bopt.evaluate(input) - f0));
        }
        double totalVarianceExpected{maths::common::CBasicStatistics::mean(meanAccumulator)};
        BOOST_REQUIRE_CLOSE_ABSOLUTE(totalVarianceActual, totalVarianceExpected, 5e-3);
    };
    verify(0.0, 1.0);
    verify(-3.0, 3.0);
    verify(0.2, 0.8);
}

BOOST_AUTO_TEST_CASE(testAnovaMainEffect) {
    std::size_t dim{2};
    std::size_t mcSamples{1000};
    TDoubleVecVec testSamples;
    maths::common::CSampling::sobolSequenceSample(1, mcSamples, testSamples);

    auto verify = [&](double min, double max) {
        maths::common::CBayesianOptimisation bopt{initBayesianOptimization(dim, 20, min, max)};
        for (std::size_t d = 0; d < dim; ++d) {
            TMeanAccumulator meanAccumulator;
            for (std::size_t i = 0; i < mcSamples; ++i) {
                TVector input{(vector(testSamples[i]) * (max - min)).array() + min};
                meanAccumulator.add(maths::common::CTools::pow2(
                    bopt.evaluate1D(input[0], static_cast<int>(d))));
            }
            double mainEffectExpected(maths::common::CBasicStatistics::mean(meanAccumulator));
            double mainEffectActual{bopt.anovaMainEffect(static_cast<int>(d))};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(mainEffectActual, mainEffectExpected, 5e-3);
        }
    };
    verify(0.0, 1.0);
    verify(-3.0, 3.0);
    verify(0.2, 0.8);
}

BOOST_AUTO_TEST_CASE(testAnovaInvariants) {

    // Test that the various parts of ANOVA change as we expect when:
    //   1. Changing the function level,
    //   2. Linearly scaling the function.

    TFunctionParamsVec tests{
        {0.0, 100.0, 0.0, 1.0}, {0.0, 100.0, 10.0, 1.0}, {0.0, 100.0, 0.0, 2.0}};

    TDoubleVec evaluateResults;
    TDoubleVecVec evaluate1DResults;
    TDoubleVec totalVarianceResults;
    TDoubleVec totalCoefficientOfVariationResults;

    for (const auto& test : tests) {

        test::CRandomNumbers rng;

        std::size_t dimension{2};
        double xl{test.s_Xl};
        double xu{test.s_Xu};
        double f0{test.s_F0};
        double scale{test.s_Scale};

        TDoubleVec coords;
        rng.generateUniformSamples(xl, xu, dimension * 20, coords);

        maths::common::CBayesianOptimisation::TDoubleDoublePrVec bb;
        for (std::size_t i = 0; i < dimension; ++i) {
            bb.emplace_back(xl, xu);
        }

        maths::common::CBayesianOptimisation bopt{bb};
        for (std::size_t i = 0; i < 10; ++i) {
            TVector x{dimension};
            for (std::size_t j = 0; j < dimension; ++j) {
                x(j) = coords[i * dimension + j];
            }
            bopt.maximumLikelihoodKernel();
            bopt.add(x, scale * x.norm() + f0, scale * scale * (xu - xl) * (xu - xl) * 0.001);
        }

        TVector probe{dimension};
        rng.generateUniformSamples(xl, xu, dimension, coords);
        for (std::size_t i = 0; i < dimension; ++i) {
            probe(i) = coords[i];
        }
        evaluateResults.push_back(bopt.evaluate(probe));
        evaluate1DResults.emplace_back();
        for (std::size_t i = 0; i < dimension; ++i) {
            evaluate1DResults.back().push_back(
                bopt.evaluate1D(probe[i], static_cast<int>(i)));
        }
        totalVarianceResults.push_back(bopt.anovaTotalVariance());
        totalCoefficientOfVariationResults.push_back(bopt.anovaTotalCoefficientOfVariation());
    }

    LOG_DEBUG(<< "evaluate      = " << core::CContainerPrinter::print(evaluateResults));
    LOG_DEBUG(<< "evaluate1D    = " << core::CContainerPrinter::print(evaluate1DResults));
    LOG_DEBUG(<< "totalVariance = " << core::CContainerPrinter::print(totalVarianceResults));
    LOG_DEBUG(<< "totalCoefficientOfVariationResults = "
              << core::CContainerPrinter::print(totalCoefficientOfVariationResults));

    for (std::size_t i = 1; i < tests.size(); ++i) {
        double f0{tests[i].s_F0};
        double scale{tests[i].s_Scale};
        BOOST_REQUIRE_CLOSE(evaluateResults[i], scale * evaluateResults[0] + f0, 1e-3);
        for (std::size_t j = 0; j < evaluate1DResults[i].size(); ++j) {
            BOOST_REQUIRE_CLOSE(evaluate1DResults[i][j],
                                scale * evaluate1DResults[0][j] + f0, 1e-3);
        }
        BOOST_REQUIRE_CLOSE(totalVarianceResults[i],
                            scale * scale * totalVarianceResults[0], 1e-3);
    }
    BOOST_TEST_REQUIRE(totalCoefficientOfVariationResults[1] <
                       totalCoefficientOfVariationResults[0]);
    BOOST_REQUIRE_CLOSE(totalCoefficientOfVariationResults[2],
                        totalCoefficientOfVariationResults[0], 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()
