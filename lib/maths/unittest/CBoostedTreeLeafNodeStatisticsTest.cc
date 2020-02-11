/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeLeafNodeStatistics.h>

#include <core/CLogger.h>

#include <maths/CBoostedTreeUtils.h>
#include <maths/CLinearAlgebraEigen.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CBoostedTreeLeafNodeStatisticsTest)

using namespace ml;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TFloatVec = std::vector<maths::CFloatStorage>;
using TVector = maths::CDenseVector<double>;
using TVectorVec = std::vector<TVector>;
using TVectorVecVec = std::vector<TVectorVec>;
using TMatrix = maths::CDenseMatrix<double>;
using TMatrixVec = std::vector<TMatrix>;
using TMatrixVecVec = std::vector<TMatrixVec>;
using TImmutableRadixSet = maths::CBoostedTreeLeafNodeStatistics::TImmutableRadixSet;
using TImmutableRadixSetVec = maths::CBoostedTreeLeafNodeStatistics::TImmutableRadixSetVec;
using TDerivatives = maths::CBoostedTreeLeafNodeStatistics::CDerivatives;
using TPerSplitDerivatives = maths::CBoostedTreeLeafNodeStatistics::CPerSplitDerivatives;

namespace {

template<typename T>
maths::CMemoryMappedDenseVector<T> makeGradient(T* storage, std::size_t n) {
    return maths::CMemoryMappedDenseVector<T>{storage, static_cast<int>(n)};
}

template<typename T>
maths::CMemoryMappedDenseVector<T> makeCurvature(T* storage, std::size_t n) {
    return maths::CMemoryMappedDenseVector<T>(storage, static_cast<int>(n));
}

template<typename T>
TMatrix rowMajorHessian(std::size_t n, const maths::CMemoryMappedDenseVector<T>& curvatures) {
    TMatrix result{n, n};
    for (std::size_t i = 0, k = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j, ++k) {
            result(i, j) = result(j, i) = curvatures(k);
        }
    }
    return result;
}

void testDerivativesFor(std::size_t numberParameters) {

    LOG_DEBUG(<< "Testing " << numberParameters << " parameters");

    test::CRandomNumbers rng;

    std::size_t numberGradients{numberParameters};
    std::size_t numberCurvatures{numberParameters * (numberParameters + 1) / 2};

    TDoubleVecVec gradients(numberGradients);
    TDoubleVecVec curvatures(numberCurvatures);
    for (std::size_t i = 0; i < numberGradients; ++i) {
        rng.generateUniformSamples(-1.0, 1.5, 20, gradients[i]);
    }
    for (std::size_t i = 0; i < numberCurvatures; ++i) {
        rng.generateUniformSamples(0.1, 0.5, 20, curvatures[i]);
    }

    LOG_DEBUG(<< "Accumulate");

    TDoubleVec storage1(numberGradients * (numberGradients + 1), 0.0);
    TDerivatives derivatives1{numberParameters, &storage1[0]};

    for (std::size_t j = 0; j < 10; ++j) {
        TFloatVec storage;
        for (std::size_t i = 0; i < numberGradients; ++i) {
            storage.push_back(gradients[i][j]);
        }
        for (std::size_t i = 0; i < numberCurvatures; ++i) {
            storage.push_back(curvatures[i][j]);
        }
        auto gradient = makeGradient(&storage[0], numberGradients);
        auto curvature = makeCurvature(&storage[numberGradients], numberCurvatures);
        derivatives1.add(1, gradient, curvature);
    }
    derivatives1.remapCurvature();

    BOOST_REQUIRE_EQUAL(10, derivatives1.count());
    for (std::size_t i = 0; i < numberGradients; ++i) {
        BOOST_REQUIRE_CLOSE(
            std::accumulate(gradients[i].begin(), gradients[i].begin() + 10, 0.0),
            derivatives1.gradient()(i), 1e-4);
    }
    for (std::size_t j = 0, k = 0; j < numberGradients; ++j) {
        for (std::size_t i = j; i < numberGradients; ++i, ++k) {
            BOOST_REQUIRE_CLOSE(std::accumulate(curvatures[k].begin(),
                                                curvatures[k].begin() + 10, 0.0),
                                derivatives1.curvature()(i, j), 1e-4);
        }
    }

    LOG_DEBUG(<< "Merge");

    TDoubleVec storage2(numberGradients * (numberGradients + 1), 0.0);
    TDerivatives derivatives2{numberParameters, &storage2[0]};

    for (std::size_t j = 10; j < 20; ++j) {
        TFloatVec storage;
        for (std::size_t i = 0; i < numberGradients; ++i) {
            storage.push_back(gradients[i][j]);
        }
        for (std::size_t i = 0; i < numberCurvatures; ++i) {
            storage.push_back(curvatures[i][j]);
        }
        auto gradient = makeGradient(&storage[0], numberGradients);
        auto curvature = makeCurvature(&storage[numberGradients], numberCurvatures);
        derivatives2.add(1, gradient, curvature);
    }
    derivatives2.remapCurvature();

    derivatives1.merge(derivatives2);

    BOOST_REQUIRE_EQUAL(20, derivatives1.count());
    for (std::size_t i = 0; i < numberGradients; ++i) {
        BOOST_REQUIRE_CLOSE(std::accumulate(gradients[i].begin(), gradients[i].end(), 0.0),
                            derivatives1.gradient()(i), 1e-4);
    }
    for (std::size_t j = 0, k = 0; j < numberGradients; ++j) {
        for (std::size_t i = j; i < numberGradients; ++i, ++k) {
            BOOST_REQUIRE_CLOSE(
                std::accumulate(curvatures[k].begin(), curvatures[k].end(), 0.0),
                derivatives1.curvature()(i, j), 1e-4);
        }
    }

    LOG_DEBUG(<< "Difference");

    TDoubleVec storage3(numberGradients * (numberGradients + 1), 0.0);
    TDerivatives derivatives3{numberParameters, &storage2[0]};

    derivatives3.assignDifference(derivatives1, derivatives2);

    BOOST_REQUIRE_EQUAL(10, derivatives3.count());
    for (std::size_t i = 0; i < numberGradients; ++i) {
        BOOST_REQUIRE_CLOSE(
            std::accumulate(gradients[i].begin(), gradients[i].begin() + 10, 0.0),
            derivatives3.gradient()(i), 1e-4);
    }
    for (std::size_t j = 0, k = 0; j < numberGradients; ++j) {
        for (std::size_t i = j; i < numberGradients; ++i, ++k) {
            BOOST_REQUIRE_CLOSE(std::accumulate(curvatures[k].begin(),
                                                curvatures[k].begin() + 10, 0.0),
                                derivatives3.curvature()(i, j), 1e-4);
        }
    }
}

void testPerSplitDerivativesFor(std::size_t numberParameters) {

    LOG_DEBUG(<< "Testing " << numberParameters << " parameters");

    TImmutableRadixSetVec featureSplits;
    featureSplits.push_back(TImmutableRadixSet{1.0, 2.0, 3.0});
    featureSplits.push_back(TImmutableRadixSet{0.1, 0.7, 1.1, 1.4});

    test::CRandomNumbers rng;

    std::size_t numberSamples{20};
    std::size_t numberGradients{numberParameters};
    std::size_t numberCurvatures{numberParameters * (numberParameters + 1) / 2};

    for (std::size_t t = 0; t < 100; ++t) {
        TSizeVec features;
        TSizeVec splits[2];
        TDoubleVec uniform01;
        TDoubleVec gradients;
        TDoubleVec curvatures;
        rng.generateUniformSamples(0, 2, numberSamples, features);
        rng.generateUniformSamples(0, featureSplits[0].size() + 1,
                                   numberSamples, splits[0]);
        rng.generateUniformSamples(0, featureSplits[1].size() + 1,
                                   numberSamples, splits[1]);
        rng.generateUniformSamples(0.0, 1.0, numberSamples, uniform01);
        rng.generateUniformSamples(-1.5, 1.0, numberSamples * numberGradients, gradients);
        rng.generateUniformSamples(0.1, 0.5, numberSamples * numberCurvatures, curvatures);

        TSizeVecVec expectedCounts(2);
        TVectorVecVec expectedGradients(2);
        TMatrixVecVec expectedCurvatures(2);
        TSizeVec expectedMissingCounts(2, 0);
        TVectorVec expectedMissingGradients(2, TVector::Zero(numberParameters));
        TMatrixVec expectedMissingCurvatures(2, TMatrix::Zero(numberParameters, numberParameters));
        for (std::size_t i = 0; i < 2; ++i) {
            expectedCounts[i].resize(featureSplits[i].size() + 1, 0);
            expectedGradients[i].resize(featureSplits[i].size() + 1,
                                        TVector::Zero(numberParameters));
            expectedCurvatures[i].resize(featureSplits[i].size() + 1,
                                         TMatrix::Zero(numberParameters, numberParameters));
        }

        auto addDerivatives = [&](TPerSplitDerivatives& derivatives) {
            for (std::size_t i = 0, j = 0, k = 0; i < numberSamples;
                 ++i, j += numberGradients, k += numberCurvatures) {

                TFloatVec storage;
                storage.insert(storage.end(), &gradients[j], &gradients[j + numberGradients]);
                storage.insert(storage.end(), &curvatures[j],
                               &curvatures[k + numberCurvatures]);
                auto gradient = makeGradient(&storage[0], numberGradients);
                auto curvature = makeCurvature(&storage[numberGradients], numberCurvatures);

                if (uniform01[i] < 0.1) {
                    derivatives.addMissingDerivatives(features[i], gradient, curvature);
                    ++expectedMissingCounts[features[i]];
                    expectedMissingGradients[features[i]] += gradient;
                    expectedMissingCurvatures[features[i]] +=
                        rowMajorHessian(numberParameters, curvature);
                } else {
                    derivatives.addDerivatives(features[i], splits[features[i]][i],
                                               gradient, curvature);
                    ++expectedCounts[features[i]][splits[features[i]][i]];
                    expectedGradients[features[i]][splits[features[i]][i]] += gradient;
                    expectedCurvatures[features[i]][splits[features[i]][i]] +=
                        rowMajorHessian(numberParameters, curvature);
                }
            }
            derivatives.remapCurvature();
        };

        auto validate = [&](const TPerSplitDerivatives& derivatives) {
            for (std::size_t i = 0; i < expectedCounts.size(); ++i) {
                for (std::size_t j = 0; j < expectedGradients[i].size(); ++j) {
                    TMatrix curvature{
                        derivatives.curvature(i, j).selfadjointView<Eigen::Lower>()};
                    BOOST_REQUIRE_EQUAL(expectedCounts[i][j], derivatives.count(i, j));
                    BOOST_REQUIRE_EQUAL(expectedGradients[i][j],
                                        derivatives.gradient(i, j));
                    BOOST_REQUIRE_EQUAL(expectedCurvatures[i][j], curvature);
                }
            }
            for (std::size_t i = 0; i < expectedMissingCounts.size(); ++i) {
                TMatrix curvature{
                    derivatives.missingCurvature(i).selfadjointView<Eigen::Lower>()};
                BOOST_REQUIRE_EQUAL(expectedMissingCounts[i], derivatives.missingCount(i));
                BOOST_REQUIRE_EQUAL(expectedMissingGradients[i],
                                    derivatives.missingGradient(i));
                BOOST_REQUIRE_EQUAL(expectedMissingCurvatures[i], curvature);
            }
        };

        LOG_TRACE(<< "Test accumulation");

        TPerSplitDerivatives derivatives1{featureSplits, numberParameters};

        addDerivatives(derivatives1);
        validate(derivatives1);

        LOG_TRACE(<< "Test merge");

        rng.generateUniformSamples(0.0, 1.0, numberSamples, uniform01);
        rng.generateUniformSamples(-1.5, 1.0, numberSamples * numberGradients, gradients);
        rng.generateUniformSamples(0.1, 0.5, numberSamples * numberCurvatures, curvatures);

        TPerSplitDerivatives derivatives2{featureSplits, numberParameters};

        addDerivatives(derivatives2);
        derivatives1.merge(derivatives2);
        validate(derivatives1);

        LOG_TRACE(<< "Test copy");

        TPerSplitDerivatives derivatives3{derivatives1};
        BOOST_REQUIRE_EQUAL(derivatives1.checksum(), derivatives3.checksum());
    }
}
}

BOOST_AUTO_TEST_CASE(testDerivatives) {

    // Test individual derivatives accumulation for single and multi parameter
    // loss functions.

    testDerivativesFor(1 /*loss function parameter*/);
    testDerivativesFor(3 /*loss function parameters*/);
}

BOOST_AUTO_TEST_CASE(testPerSplitDerivatives) {

    // Test per split derivatives accumulation for single and multi parameter
    // loss functions.

    testPerSplitDerivativesFor(1 /*loss function parameter*/);
    testPerSplitDerivativesFor(3 /*loss function parameters*/);
}

BOOST_AUTO_TEST_SUITE_END()
