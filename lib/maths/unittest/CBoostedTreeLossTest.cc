/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <boost/test/tools/old/interface.hpp>
#include <cmath>
#include <core/CContainerPrinter.h>

#include <maths/CLbfgs.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CPRNG.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>
#include <maths/CBasicStatistics.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CBoostedTreeLossTest)

using namespace ml;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleVector = maths::boosted_tree::CLoss::TDoubleVector;
using TDoubleVectorVec = std::vector<TDoubleVector>;
using TMemoryMappedFloatVector = maths::boosted_tree::CLoss::TMemoryMappedFloatVector;
using maths::boosted_tree::CBinomialLogisticLoss;
using maths::boosted_tree::CMultinomialLogisticLoss;
using maths::boosted_tree_detail::CArgMinBinomialLogisticLossImpl;
using maths::boosted_tree_detail::CArgMinMultinomialLogisticLossImpl;
using maths::boosted_tree_detail::CArgMinMsleImpl;

namespace {
void minimizeGridSearch(std::function<double(const TDoubleVector&)> objective,
                        double scale,
                        int d,
                        TDoubleVector& x,
                        double& min,
                        TDoubleVector& argmin) {
    if (d == x.size()) {
        double value{objective(x)};
        if (value < min) {
            min = value;
            argmin = x;
        }
        return;
    }
    for (double xd : {-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5}) {
        x(d) = scale * xd;
        minimizeGridSearch(objective, scale, d + 1, x, min, argmin);
    }
}
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticMinimizerEdgeCases) {

    // All predictions equal and zero.
    {
        CArgMinBinomialLogisticLossImpl argmin{0.0};
        maths::CFloatStorage storage[]{0.0};
        TMemoryMappedFloatVector prediction{storage, 1};
        argmin.add(prediction, 0.0);
        argmin.add(prediction, 1.0);
        argmin.add(prediction, 1.0);
        argmin.add(prediction, 0.0);
        BOOST_REQUIRE_EQUAL(false, argmin.nextPass());
        BOOST_REQUIRE_EQUAL(0.0, argmin.value()[0]);
    }

    // All predictions are equal and 0.5.
    {
        test::CRandomNumbers rng;

        TDoubleVec labels;
        rng.generateUniformSamples(0.0, 1.0, 1000, labels);
        for (auto& label : labels) {
            label = std::floor(label + 0.3);
        }

        CArgMinBinomialLogisticLossImpl argmin{0.0};
        std::size_t numberPasses{0};
        std::size_t counts[]{0, 0};

        do {
            ++numberPasses;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                maths::CFloatStorage storage[]{0.5};
                TMemoryMappedFloatVector prediction{storage, 1};
                argmin.add(prediction, labels[i]);
                ++counts[static_cast<std::size_t>(labels[i])];
            }
        } while (argmin.nextPass());

        double p{static_cast<double>(counts[1]) / 1000.0};
        double expected{std::log(p / (1.0 - p)) - 0.5};
        double actual{argmin.value()[0]};

        BOOST_REQUIRE_EQUAL(std::size_t{1}, numberPasses);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 0.01 * std::fabs(expected));
    }

    // Test underflow of probabilities.
    {
        CArgMinBinomialLogisticLossImpl argmin{0.0};

        TDoubleVec predictions{-500.0, -30.0, -15.0, -400.0};
        TDoubleVec labels{1.0, 1.0, 0.0, 1.0};
        do {
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                maths::CFloatStorage storage[]{predictions[i]};
                TMemoryMappedFloatVector prediction{storage, 1};
                argmin.add(prediction, labels[i]);
            }
        } while (argmin.nextPass());

        double minimizer{argmin.value()[0]};

        // Check we're at the minimum.
        CBinomialLogisticLoss loss;
        TDoubleVec losses;
        for (double eps : {-10.0, 0.0, 10.0}) {
            double lossAtEps{0.0};
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                maths::CFloatStorage storage[]{predictions[i] + minimizer + eps};
                TMemoryMappedFloatVector probe{storage, 1};
                lossAtEps += loss.value(probe, labels[i]);
            }
            losses.push_back(lossAtEps);
        }
        BOOST_TEST_REQUIRE(losses[0] >= losses[1]);
        BOOST_TEST_REQUIRE(losses[2] >= losses[1]);
    }
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticMinimizerRandom) {

    // Test that we a good approximation of the additive term for the log-odds
    // which minimises the exact cross entropy objective.

    test::CRandomNumbers rng;

    TDoubleVec labels;
    TDoubleVec predictions;

    for (auto lambda : {0.0, 10.0}) {

        LOG_DEBUG(<< "lambda = " << lambda);

        // The exact objective.
        auto exactObjective = [&](double weight) {
            double loss{0.0};
            for (std::size_t i = 0; i < labels.size(); ++i) {
                double p{maths::CTools::logisticFunction(predictions[i] + weight)};
                loss -= (1.0 - labels[i]) * maths::CTools::fastLog(1.0 - p) +
                        labels[i] * maths::CTools::fastLog(p);
            }
            return loss + lambda * maths::CTools::pow2(weight);
        };

        // This loop is fuzzing the predicted log-odds and testing we get consistently
        // good estimates of the true minimizer.
        for (std::size_t t = 0; t < 10; ++t) {

            double min{std::numeric_limits<double>::max()};
            double max{-min};

            rng.generateUniformSamples(0.0, 1.0, 1000, labels);
            for (auto& label : labels) {
                label = std::floor(label + 0.5);
            }
            predictions.clear();
            for (const auto& label : labels) {
                TDoubleVec weight;
                rng.generateNormalSamples(label, 2.0, 1, weight);
                predictions.push_back(weight[0]);
                min = std::min(min, weight[0]);
                max = std::max(max, weight[0]);
            }

            double expected;
            double objectiveAtExpected;
            std::size_t maxIterations{20};
            maths::CSolvers::minimize(-max, -min, exactObjective(-max),
                                      exactObjective(-min), exactObjective, 1e-3,
                                      maxIterations, expected, objectiveAtExpected);
            LOG_DEBUG(<< "expected = " << expected
                      << " objective(expected) = " << objectiveAtExpected);

            CArgMinBinomialLogisticLossImpl argmin{lambda};
            CArgMinBinomialLogisticLossImpl argminPartition[2]{{lambda}, {lambda}};
            auto nextPass = [&] {
                bool done{argmin.nextPass() == false};
                done &= (argminPartition[0].nextPass() == false);
                done &= (argminPartition[1].nextPass() == false);
                return done == false;
            };

            do {
                for (std::size_t i = 0; i < labels.size() / 2; ++i) {
                    maths::CFloatStorage storage[]{predictions[i]};
                    TMemoryMappedFloatVector prediction{storage, 1};
                    argmin.add(prediction, labels[i]);
                    argminPartition[0].add(prediction, labels[i]);
                }
                for (std::size_t i = labels.size() / 2; i < labels.size(); ++i) {
                    maths::CFloatStorage storage[]{predictions[i]};
                    TMemoryMappedFloatVector prediction{storage, 1};
                    argmin.add(prediction, labels[i]);
                    argminPartition[1].add(prediction, labels[i]);
                }
                argminPartition[0].merge(argminPartition[1]);
                argminPartition[1] = argminPartition[0];
            } while (nextPass());

            double actual{argmin.value()(0)};
            double actualPartition{argminPartition[0].value()(0)};
            double objectiveAtActual{exactObjective(actual)};
            LOG_DEBUG(<< "actual = " << actual << " objective(actual) = " << objectiveAtActual);

            // We should be within 1% for the value and 0.001% for the objective
            // at the value.
            BOOST_REQUIRE_EQUAL(actual, actualPartition);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 0.01 * std::fabs(expected));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(objectiveAtExpected, objectiveAtActual,
                                         1e-5 * objectiveAtExpected);
        }
    }
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticLossGradientAndCurvature) {

    // Test that the loss gradient and curvature functions are close to the
    // numerical derivatives of the objective.

    test::CRandomNumbers rng;

    double eps{1e-3};
    auto derivative = [eps](const std::function<double(TDoubleVector)>& f,
                            TDoubleVector prediction, std::size_t i) {
        prediction(i) += eps;
        double result{f(prediction)};
        prediction(i) -= 2.0 * eps;
        result -= f(prediction);
        result /= 2.0 * eps;
        return result;
    };

    for (std::size_t t = 0; t < 100; ++t) {

        CBinomialLogisticLoss loss;

        TDoubleVec predictions{0.0};
        if (t > 0) {
            rng.generateUniformSamples(-1.0, 1.0, 1, predictions);
        }

        TDoubleVec actual;
        rng.generateUniformSamples(0.0, 2.0, 1, actual);
        actual[0] = std::floor(actual[0]);

        double expectedGradient;
        double expectedCurvature;
        {
            auto gradient = [&](TDoubleVector prediction) {
                return derivative(
                    [&](TDoubleVector prediction_) {
                        maths::CFloatStorage storage[]{prediction_(0)};
                        return loss.value(TMemoryMappedFloatVector{storage, 1}, actual[0]);
                    },
                    prediction, 0);
            };
            TDoubleVector prediction{1};
            prediction(0) = predictions[0];
            expectedGradient = gradient(prediction);
            expectedCurvature = derivative(gradient, prediction, 0);
        }

        double actualGradient;
        double actualCurvature;
        {
            maths::CFloatStorage storage[]{predictions[0]};
            TMemoryMappedFloatVector prediction{storage, 1};
            loss.gradient(prediction, actual[0], [&](std::size_t, double gradient) {
                actualGradient = gradient;
            });
            loss.curvature(prediction, actual[0], [&](std::size_t, double curvature) {
                actualCurvature = curvature;
            });
        }
        if (t % 10 == 0) {
            LOG_DEBUG(<< "actual gradient    = " << actualGradient);
            LOG_DEBUG(<< "expected gradient  = " << expectedGradient);
            LOG_DEBUG(<< "actual curvature   = " << actualCurvature);
            LOG_DEBUG(<< "expected curvature = " << expectedCurvature);
        }

        BOOST_REQUIRE_SMALL(std::fabs(expectedGradient - actualGradient), 0.001);
        BOOST_REQUIRE_SMALL(std::fabs(expectedCurvature - actualCurvature), 0.02);
    }
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticLossForUnderflow) {

    // Test the behaviour of value, gradient and curvature of the logistic loss in
    // the vicinity the point at which we switch to using Taylor expansion of the
    // logistic function is as expected.

    double eps{100.0 * std::numeric_limits<double>::epsilon()};

    CBinomialLogisticLoss loss;

    // Losses should be very nearly linear function of log-odds when they're large.
    {
        maths::CFloatStorage storage[]{1.0 - std::log(eps), 1.0 + std::log(eps)};
        TMemoryMappedFloatVector predictions[]{{&storage[0], 1}, {&storage[1], 1}};
        TDoubleVec previousLoss{loss.value(predictions[0], 0.0),
                                loss.value(predictions[1], 1.0)};
        LOG_DEBUG(<< core::CContainerPrinter::print(previousLoss));
        for (double scale : {0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0}) {
            storage[0] = scale - std::log(eps);
            storage[1] = scale + std::log(eps);
            TDoubleVec currentLoss{loss.value(predictions[0], 0.0),
                                   loss.value(predictions[1], 1.0)};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.25, previousLoss[0] - currentLoss[0], 0.005);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(-0.25, previousLoss[1] - currentLoss[1], 0.005);
            previousLoss = currentLoss;
        }
    }

    // The gradient and curvature should be proportional to the exponential of the
    // log-odds when they're small.
    {
        auto readDerivatives = [&](double prediction, TDoubleVec& gradients,
                                   TDoubleVec& curvatures) {
            maths::CFloatStorage storage[]{prediction + std::log(eps),
                                           prediction - std::log(eps)};
            TMemoryMappedFloatVector predictions[]{{&storage[0], 1}, {&storage[1], 1}};
            loss.gradient(predictions[0], 0.0, [&](std::size_t, double value) {
                gradients[0] = value;
            });
            loss.gradient(predictions[1], 1.0, [&](std::size_t, double value) {
                gradients[1] = value;
            });
            loss.curvature(predictions[0], 0.0, [&](std::size_t, double value) {
                curvatures[0] = value;
            });
            loss.curvature(predictions[1], 1.0, [&](std::size_t, double value) {
                curvatures[1] = value;
            });
        };

        TDoubleVec previousGradient(2);
        TDoubleVec previousCurvature(2);
        readDerivatives(1.0, previousGradient, previousCurvature);

        for (double scale : {0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0}) {
            TDoubleVec currentGradient(2);
            TDoubleVec currentCurvature(2);
            readDerivatives(scale, currentGradient, currentCurvature);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(0.25), previousGradient[0] / currentGradient[0], 0.01);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(-0.25), previousGradient[1] / currentGradient[1], 0.01);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(0.25), previousCurvature[0] / currentCurvature[0], 0.01);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(-0.25), previousCurvature[1] / currentCurvature[1], 0.01);
            previousGradient = currentGradient;
            previousCurvature = currentCurvature;
        }
    }
}

BOOST_AUTO_TEST_CASE(testMultinomialLogisticArgminObjective) {

    // Test that the gradient function is close to the numerical derivative
    // of the objective.

    maths::CPRNG::CXorOShiro128Plus rng;
    test::CRandomNumbers testRng;

    for (std::size_t t = 0; t < 10; ++t) {

        CArgMinMultinomialLogisticLossImpl argmin{3, 0.1 * static_cast<double>(t + 1), rng};

        TDoubleVec labels;
        testRng.generateUniformSamples(0.0, 3.0, 20, labels);

        TDoubleVec predictions;
        if (t == 0) {
            predictions.resize(3 * labels.size(), 0.0);
        } else {
            testRng.generateUniformSamples(-1.0, 1.0, 3 * labels.size(), predictions);
        }

        do {
            for (std::size_t i = 0; i < labels.size(); ++i) {
                maths::CFloatStorage storage[]{predictions[3 * i + 0],
                                               predictions[3 * i + 1],
                                               predictions[3 * i + 2]};
                TMemoryMappedFloatVector prediction{storage, 3};
                argmin.add(prediction, std::floor(labels[i]));
            }
        } while (argmin.nextPass());

        auto objective = argmin.objective();
        auto objectiveGradient = argmin.objectiveGradient();

        double eps{1e-3};
        TDoubleVec probes;
        testRng.generateUniformSamples(-1.0, 1.0, 30, probes);
        for (std::size_t i = 0; i < probes.size(); i += 3) {
            TDoubleVector probe{3};
            probe(0) = probes[i];
            probe(1) = probes[i + 1];
            probe(2) = probes[i + 2];

            TDoubleVector expectedGradient{3};
            for (std::size_t j = 0; j < 3; ++j) {
                TDoubleVector shift{TDoubleVector::Zero(3)};
                shift(j) = eps;
                expectedGradient(j) =
                    (objective(probe + shift) - objective(probe - shift)) / (2.0 * eps);
            }
            TDoubleVector actualGradient{objectiveGradient(probe)};

            BOOST_REQUIRE_SMALL((expectedGradient - actualGradient).norm(), eps);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMultinomialLogisticMinimizerEdgeCases) {

    maths::CPRNG::CXorOShiro128Plus rng;
    test::CRandomNumbers testRng;

    // All predictions equal and zero.
    {
        CArgMinMultinomialLogisticLossImpl argmin{3, 0.0, rng};

        maths::CFloatStorage storage[]{0.0, 0.0, 0.0};
        TMemoryMappedFloatVector prediction{storage, 3};
        argmin.add(prediction, 0.0);
        argmin.add(prediction, 1.0);
        argmin.add(prediction, 1.0);
        argmin.add(prediction, 0.0);
        argmin.add(prediction, 2.0);
        argmin.add(prediction, 1.0);
        BOOST_REQUIRE_EQUAL(false, argmin.nextPass());

        TDoubleVector expectedProbabilities{3};
        expectedProbabilities(0) = 2.0 / 6.0;
        expectedProbabilities(1) = 3.0 / 6.0;
        expectedProbabilities(2) = 1.0 / 6.0;
        TDoubleVector actualProbabilities{maths::CTools::softmax(argmin.value())};

        BOOST_REQUIRE_SMALL((actualProbabilities - expectedProbabilities).norm(), 1e-3);
    }

    // All predictions are equal and 0.5.
    for (std::size_t t = 0; t < 10; ++t) {

        TDoubleVec labels;
        testRng.generateUniformSamples(0.0, 2.0, 20, labels);
        for (auto& label : labels) {
            label = std::floor(label + 0.3);
        }

        CArgMinMultinomialLogisticLossImpl argmin{3, 0.0, rng};

        std::size_t numberPasses{0};
        std::size_t counts[]{0, 0, 0};
        maths::CFloatStorage storage[]{0.5, 0.5, 0.5};
        TMemoryMappedFloatVector prediction{storage, 3};

        do {
            ++numberPasses;
            for (const auto& label : labels) {
                argmin.add(prediction, label);
                ++counts[static_cast<std::size_t>(label)];
            }
        } while (argmin.nextPass());

        BOOST_REQUIRE_EQUAL(std::size_t{1}, numberPasses);

        TDoubleVector expectedProbabilities{3};
        expectedProbabilities(0) = static_cast<double>(counts[0]) / 20.0;
        expectedProbabilities(1) = static_cast<double>(counts[1]) / 20.0;
        expectedProbabilities(2) = static_cast<double>(counts[2]) / 20.0;
        TDoubleVector actualLogit{prediction + argmin.value()};
        TDoubleVector actualProbabilities{maths::CTools::softmax(actualLogit)};

        BOOST_REQUIRE_SMALL((actualProbabilities - expectedProbabilities).norm(), 0.001);
    }

    // Test underflow of probabilities.
    LOG_DEBUG(<< "Test underflow");
    {
        CArgMinMultinomialLogisticLossImpl argmin{3, 0.0, rng};

        TDoubleVecVec predictions{{-230.0, -200.0, -200.0},
                                  {-30.0, -10.0, -20.0},
                                  {-15.0, -50.0, -30.0},
                                  {-400.0, -350.0, -300.0}};
        TDoubleVec labels{1.0, 1.0, 0.0, 2.0};
        do {
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                maths::CFloatStorage storage[]{predictions[i][0], predictions[i][1],
                                               predictions[i][2]};
                TMemoryMappedFloatVector prediction{storage, 3};
                argmin.add(prediction, labels[i]);
            }
        } while (argmin.nextPass());

        TDoubleVector minimizer{argmin.value()};

        // Check we're at a minimum.
        CMultinomialLogisticLoss loss{3};
        for (std::size_t i = 0; i < 3; ++i) {
            TDoubleVec losses;
            for (double eps : {-30.0, 0.0, 30.0}) {
                double lossAtEps{0.0};
                for (std::size_t j = 0; j < predictions.size(); ++j) {
                    maths::CFloatStorage storage[]{predictions[j][0] + minimizer(0),
                                                   predictions[j][1] + minimizer(1),
                                                   predictions[j][2] + minimizer(2)};
                    storage[i] += eps;
                    TMemoryMappedFloatVector probe{storage, 3};
                    lossAtEps += loss.value(probe, labels[j]);
                }
                losses.push_back(lossAtEps);
            }
            LOG_DEBUG(<< core::CContainerPrinter::print(losses));
            BOOST_TEST_REQUIRE(losses[0] >= losses[1]);
            BOOST_TEST_REQUIRE(losses[2] >= losses[1]);
        }
    }

    // All labels equal.
    {
        CArgMinMultinomialLogisticLossImpl argmin{3, 0.0, rng};

        maths::CFloatStorage storage[]{0.0, 0.0, 0.0};
        TMemoryMappedFloatVector prediction{storage, 3};
        TDoubleVec labels{1.0, 1.0, 1.0, 1.0};

        do {
            for (const auto& label : labels) {
                argmin.add(prediction, label);
            }
        } while (argmin.nextPass());

        TDoubleVector minimizer{argmin.value()};

        double totalLoss{0.0};
        CMultinomialLogisticLoss loss{3};
        for (const auto& label : labels) {
            maths::CFloatStorage probeStorage[]{prediction(0) + minimizer(0),
                                                prediction(1) + minimizer(1),
                                                prediction(2) + minimizer(2)};
            TMemoryMappedFloatVector probe{probeStorage, 3};
            totalLoss += loss.value(probe, label);
        }

        BOOST_REQUIRE_SMALL(totalLoss, 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(testMultinomialLogisticMinimizerRandom) {

    // Test that we have a good approximation of the additive term for the log-odds
    // which minimises the exact cross entropy objective.

    maths::CPRNG::CXorOShiro128Plus rng;
    test::CRandomNumbers testRng;

    TDoubleVec labels;
    TDoubleVectorVec predictions;

    double scales[]{6.0, 1.0};
    double lambdas[]{0.0, 10.0};

    for (auto i : {0, 1}) {

        double lambda{lambdas[i]};

        LOG_DEBUG(<< "lambda = " << lambda);

        // The exact objective.
        auto exactObjective = [&](const TDoubleVector& weight) {
            double loss{0.0};
            for (std::size_t j = 0; j < labels.size(); ++j) {
                TDoubleVector probabilities{predictions[j] + weight};
                probabilities = maths::CTools::softmax(std::move(probabilities));
                loss -= maths::CTools::fastLog(probabilities(static_cast<int>(labels[j])));
            }
            return loss + lambda * weight.squaredNorm();
        };

        // This loop is fuzzing the predicted log-odds and testing we get consistently
        // good estimates of the true minimizer.

        double sumObjectiveGridSearch{0.0};
        double sumObjectiveAtActual{0.0};

        for (std::size_t t = 0; t < 10; ++t) {

            testRng.generateUniformSamples(0.0, 2.0, 500, labels);
            for (auto& label : labels) {
                label = std::floor(label + 0.5);
            }

            predictions.clear();
            for (const auto& label : labels) {
                TDoubleVec prediction;
                testRng.generateNormalSamples(0.0, 2.0, 3, prediction);
                prediction[static_cast<std::size_t>(label)] += 1.0;
                predictions.push_back(TDoubleVector::fromStdVector(prediction));
            }

            TDoubleVector weight{3};
            double objectiveGridSearch{std::numeric_limits<double>::max()};
            TDoubleVector argminGridSearch;
            minimizeGridSearch(exactObjective, scales[i], 0, weight,
                               objectiveGridSearch, argminGridSearch);
            LOG_DEBUG(<< "argmin grid search = " << argminGridSearch.transpose()
                      << ", min objective grid search = " << objectiveGridSearch);

            std::size_t numberPasses{0};
            maths::CFloatStorage storage[]{0.0, 0.0, 0.0};
            TMemoryMappedFloatVector prediction{storage, 3};

            CArgMinMultinomialLogisticLossImpl argmin{3, lambda, rng};

            do {
                ++numberPasses;
                for (std::size_t j = 0; j < labels.size(); ++j) {
                    storage[0] = predictions[j](0);
                    storage[1] = predictions[j](1);
                    storage[2] = predictions[j](2);
                    argmin.add(prediction, labels[j]);
                }
            } while (argmin.nextPass());

            BOOST_REQUIRE_EQUAL(std::size_t{2}, numberPasses);

            TDoubleVector actual{argmin.value()};
            double objectiveAtActual{exactObjective(actual)};
            LOG_DEBUG(<< "actual = " << actual.transpose()
                      << ", objective(actual) = " << objectiveAtActual);

            BOOST_TEST_REQUIRE(objectiveAtActual < 1.01 * objectiveGridSearch);

            sumObjectiveGridSearch += objectiveGridSearch;
            sumObjectiveAtActual += objectiveAtActual;
        }

        LOG_DEBUG(<< "sum min objective grid search = " << sumObjectiveGridSearch);
        LOG_DEBUG(<< "sum objective(actual) = " << sumObjectiveAtActual);
        BOOST_TEST_REQUIRE(sumObjectiveAtActual < sumObjectiveGridSearch);
    }
}

BOOST_AUTO_TEST_CASE(testMultinomialLogisticLossGradientAndCurvature) {

    // Test that the loss gradient and curvature functions are close to the
    // numerical derivatives of the objective.

    test::CRandomNumbers rng;

    double eps{1e-3};
    auto derivative = [eps](const std::function<double(TDoubleVector)>& f,
                            TDoubleVector prediction, std::size_t i) {
        prediction(i) += eps;
        double result{f(prediction)};
        prediction(i) -= 2.0 * eps;
        result -= f(prediction);
        result /= 2.0 * eps;
        return result;
    };

    for (std::size_t t = 0; t < 100; ++t) {

        CMultinomialLogisticLoss loss{3};

        TDoubleVec predictions(3, 0.0);
        if (t > 0) {
            rng.generateUniformSamples(-1.0, 1.0, 3, predictions);
        }

        TDoubleVec actual;
        rng.generateUniformSamples(0.0, 3.0, 1, actual);
        actual[0] = std::floor(actual[0]);

        TDoubleVector expectedGradient{3};
        TDoubleVector expectedCurvature{6};

        for (std::size_t i = 0, k = 0; i < 3; ++i) {
            auto gi = [&](TDoubleVector prediction) {
                return derivative(
                    [&](TDoubleVector prediction_) {
                        maths::CFloatStorage storage[]{
                            prediction_(0), prediction_(1), prediction_(2)};
                        return loss.value(TMemoryMappedFloatVector{storage, 3}, actual[0]);
                    },
                    prediction, i);
            };

            TDoubleVector prediction{3};
            prediction(0) = predictions[0];
            prediction(1) = predictions[1];
            prediction(2) = predictions[2];

            expectedGradient(i) = gi(prediction);
            for (std::size_t j = i; j < 3; ++j, ++k) {
                expectedCurvature(k) = derivative(gi, prediction, j);
            }
        }

        maths::CFloatStorage storage[]{predictions[0], predictions[1], predictions[2]};
        TMemoryMappedFloatVector prediction{storage, 3};
        TDoubleVector actualGradient{3};
        loss.gradient(prediction, actual[0], [&](std::size_t k, double gradient) {
            actualGradient(k) = gradient;
        });
        TDoubleVector actualCurvature{6};
        loss.curvature(prediction, actual[0], [&](std::size_t k, double curvature) {
            actualCurvature(k) = curvature;
        });
        if (t % 10 == 0) {
            LOG_DEBUG(<< "actual gradient    = " << actualGradient.transpose());
            LOG_DEBUG(<< "expected gradient  = " << expectedGradient.transpose());
            LOG_DEBUG(<< "actual curvature   = " << actualCurvature.transpose());
            LOG_DEBUG(<< "expected curvature = " << expectedCurvature.transpose());
        }

        BOOST_REQUIRE_SMALL((expectedGradient - actualGradient).norm(), 0.001);
        BOOST_REQUIRE_SMALL((expectedCurvature - actualCurvature).norm(), 0.02);
    }
}

BOOST_AUTO_TEST_CASE(testMultinomialLogisticLossForUnderflow) {

    // Test the behaviour of value, gradient and Hessian of the logistic loss in
    // the regime where the probabilities underflow.

    using TFloatVec = std::vector<maths::CFloatStorage>;

    double eps{100.0 * std::numeric_limits<double>::epsilon()};

    auto logits = [](double x, TFloatVec& result) { result.assign({0.0, x}); };

    CMultinomialLogisticLoss loss{2};

    // Losses should be very nearly linear function of log-odds when they're large.
    {
        TFloatVec storage[2];
        logits(1.0 - std::log(eps), storage[0]);
        logits(1.0 + std::log(eps), storage[1]);

        TMemoryMappedFloatVector predictions[]{{&storage[0][0], 2}, {&storage[1][0], 2}};
        TDoubleVec previousLoss{loss.value(predictions[0], 0.0),
                                loss.value(predictions[1], 1.0)};

        for (double scale : {0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0}) {
            logits(scale - std::log(eps), storage[0]);
            logits(scale + std::log(eps), storage[1]);
            TDoubleVec currentLoss{loss.value(predictions[0], 0.0),
                                   loss.value(predictions[1], 1.0)};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.25, previousLoss[0] - currentLoss[0], 0.005);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(-0.25, previousLoss[1] - currentLoss[1], 0.005);
            previousLoss = currentLoss;
        }
    }

    // The gradient and curvature should be proportional to the exponential of the
    // log-odds when they're small.
    {
        auto readDerivatives = [&](double prediction, TDoubleVecVec& gradients,
                                   TDoubleVecVec& curvatures) {
            TFloatVec storage[2];
            logits(prediction + std::log(eps), storage[0]);
            logits(prediction - std::log(eps), storage[1]);
            TMemoryMappedFloatVector predictions[]{{&storage[0][0], 2},
                                                   {&storage[1][0], 2}};
            loss.gradient(predictions[0], 0.0, [&](std::size_t i, double value) {
                gradients[0][i] = value;
            });
            loss.gradient(predictions[1], 1.0, [&](std::size_t i, double value) {
                gradients[1][i] = value;
            });
            loss.curvature(predictions[0], 0.0, [&](std::size_t i, double value) {
                curvatures[0][i] = value;
            });
            loss.curvature(predictions[1], 1.0, [&](std::size_t i, double value) {
                curvatures[1][i] = value;
            });
        };

        TDoubleVecVec previousGradient(2, TDoubleVec(2));
        TDoubleVecVec previousCurvature(2, TDoubleVec(3));
        readDerivatives(1.0, previousGradient, previousCurvature);

        for (double scale : {0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0}) {
            TDoubleVecVec currentGradient(2, TDoubleVec(2));
            TDoubleVecVec currentCurvature(2, TDoubleVec(3));
            readDerivatives(scale, currentGradient, currentCurvature);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(0.25), previousGradient[0][1] / currentGradient[0][1], 0.04);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(-0.25), previousGradient[1][1] / currentGradient[1][1], 0.04);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(0.25), previousCurvature[0][2] / currentCurvature[0][2], 0.04);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                std::exp(-0.25), previousCurvature[1][2] / currentCurvature[1][2], 0.04);
            previousGradient = currentGradient;
            previousCurvature = currentCurvature;
        }
    }
}

BOOST_AUTO_TEST_CASE(testMsleArgminObjective){
    // Test that the gradient function is close to the numerical derivative
    // of the objective.
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    maths::CPRNG::CXorOShiro128Plus rng;
    test::CRandomNumbers testRng;

    for (std::size_t t = 0; t < 10; ++t) {
        TMeanAccumulator expectedErrorAccumulator;
        double lambda{0.1 * static_cast<double>(t + 1)};
        CArgMinMsleImpl argmin{lambda};

        TDoubleVec targets;
        testRng.generateUniformSamples(0.0, 3.0, 20, targets);

        TDoubleVec predictions;
        predictions.resize(targets.size(), 0.0);
        testRng.generateUniformSamples(0.0, 3.0, targets.size(), predictions);

        do {
            for (std::size_t i = 0; i < targets.size(); ++i) {
                maths::CFloatStorage storage[]{std::log(predictions[i])};
                TMemoryMappedFloatVector prediction{storage, 1};
                argmin.add(prediction, targets[i]);
            }
        } while (argmin.nextPass());

        auto objective = argmin.objective();

        for (std::size_t i = 0; i < targets.size(); ++i) {
            double error{std::log(targets[i]+1) - std::log(predictions[i]+1)};
            expectedErrorAccumulator.add(error*error);
        }
        double expectedObjectiveValue{maths::CBasicStatistics::mean(expectedErrorAccumulator) + lambda};
        // LOG_DEBUG(<< "Objective " << objective(0.0) << " expected " << expectedObjectiveValue);
        BOOST_REQUIRE_SMALL(std::abs(objective(0.0)-expectedObjectiveValue), 1e-5);
        // TDoubleVec probes;
        // testRng.generateUniformSamples(-1.0, 1.0, 30, probes);
        // for (std::size_t i = 0; i < probes.size(); i += 3) {
        //     TDoubleVector probe{3};
        //     probe(0) = probes[i];
        //     probe(1) = probes[i + 1];
        //     probe(2) = probes[i + 2];

        //     TDoubleVector expectedGradient{3};
        //     for (std::size_t j = 0; j < 3; ++j) {
        //         TDoubleVector shift{TDoubleVector::Zero(3)};
        //         shift(j) = eps;
        //         expectedGradient(j) =
        //             (objective(probe + shift) - objective(probe - shift)) / (2.0 * eps);
        //     }
        //     TDoubleVector actualGradient{objectiveGradient(probe)};

        //     BOOST_REQUIRE_SMALL((expectedGradient - actualGradient).norm(), eps);
        // }
    }
}

BOOST_AUTO_TEST_SUITE_END()
