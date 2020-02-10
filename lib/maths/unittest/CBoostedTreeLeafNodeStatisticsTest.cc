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
using TDerivativesVec = maths::CBoostedTreeLeafNodeStatistics::TDerivativesVec;
using TDerivativesVecVec = maths::CBoostedTreeLeafNodeStatistics::TDerivativesVecVec;
using TDerivativesAccumulator = maths::CBoostedTreeLeafNodeStatistics::CDerivativesAccumulator;
using TSplitDerivativesAccumulator = maths::CBoostedTreeLeafNodeStatistics::CSplitDerivativesAccumulator;

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

void testDerivativesAccumulatorFor(std::size_t numberParameters) {

    LOG_DEBUG(<< "Testing " << numberParameters << " parameters");

    test::CRandomNumbers rng;

    std::size_t numberGradients{numberParameters};
    std::size_t numberCurvatures{
        maths::boosted_tree_detail::lossHessianStoredSize(numberParameters)};

    TDoubleVecVec gradients(numberGradients);
    TDoubleVecVec curvatures(numberCurvatures);
    for (std::size_t i = 0; i < numberGradients; ++i) {
        rng.generateUniformSamples(-1.0, 1.5, 20, gradients[i]);
    }
    for (std::size_t i = 0; i < numberCurvatures; ++i) {
        rng.generateUniformSamples(0.1, 0.5, 20, curvatures[i]);
    }

    TDoubleVec storage1(numberGradients + numberCurvatures, 0.0);
    auto totalGradient1 = makeGradient(&storage1[0], numberGradients);
    auto totalCurvature1 = makeCurvature(&storage1[numberGradients], numberCurvatures);
    TDerivativesAccumulator accumulator1{totalGradient1, totalCurvature1};

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
        accumulator1.add(1, gradient, curvature);
    }

    BOOST_REQUIRE_EQUAL(10, accumulator1.count());
    for (std::size_t i = 0; i < numberGradients; ++i) {
        BOOST_REQUIRE_CLOSE(
            std::accumulate(gradients[i].begin(), gradients[i].begin() + 10, 0.0),
            accumulator1.gradient()(i), 1e-4);
    }
    for (std::size_t i = 0; i < numberCurvatures; ++i) {
        BOOST_REQUIRE_CLOSE(std::accumulate(curvatures[i].begin(),
                                            curvatures[i].begin() + 10, 0.0),
                            accumulator1.curvature()(i), 1e-4);
    }

    TDoubleVec storage2(numberGradients + numberCurvatures, 0.0);
    auto totalGradient2 = makeGradient(&storage2[0], numberGradients);
    auto totalCurvature2 = makeCurvature(&storage2[numberGradients], numberCurvatures);
    TDerivativesAccumulator accumulator2{totalGradient2, totalCurvature2};

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
        accumulator2.add(1, gradient, curvature);
    }

    BOOST_REQUIRE_EQUAL(10, accumulator2.count());
    for (std::size_t i = 0; i < numberGradients; ++i) {
        BOOST_REQUIRE_CLOSE(
            std::accumulate(gradients[i].begin() + 10, gradients[i].end(), 0.0),
            accumulator2.gradient()(i), 1e-4);
    }
    for (std::size_t i = 0; i < numberCurvatures; ++i) {
        BOOST_REQUIRE_CLOSE(
            std::accumulate(curvatures[i].begin() + 10, curvatures[i].end(), 0.0),
            accumulator2.curvature()(i), 1e-4);
    }

    accumulator1.merge(accumulator2);
    BOOST_REQUIRE_EQUAL(20, accumulator1.count());
    for (std::size_t i = 0; i < numberGradients; ++i) {
        BOOST_REQUIRE_CLOSE(std::accumulate(gradients[i].begin(), gradients[i].end(), 0.0),
                            accumulator1.gradient()(i), 1e-4);
    }
    for (std::size_t i = 0; i < numberCurvatures; ++i) {
        BOOST_REQUIRE_CLOSE(std::accumulate(curvatures[i].begin(), curvatures[i].end(), 0.0),
                            accumulator1.curvature()(i), 1e-4);
    }
}

void testSplitDerivativesAccumulatorFor(std::size_t numberParameters) {

    LOG_DEBUG(<< "Testing " << numberParameters << " parameters");

    TImmutableRadixSetVec featureSplits;
    featureSplits.push_back(TImmutableRadixSet{1.0, 2.0, 3.0});
    featureSplits.push_back(TImmutableRadixSet{0.1, 0.7, 1.1, 1.4});

    test::CRandomNumbers rng;

    std::size_t numberSamples{20};
    std::size_t numberGradients{numberParameters};
    std::size_t numberCurvatures{
        maths::boosted_tree_detail::lossHessianStoredSize(numberParameters)};

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

        auto addDerivatives = [&](TSplitDerivativesAccumulator& accumulator) {
            for (std::size_t i = 0, j = 0, k = 0; i < numberSamples;
                ++i, j += numberGradients, k += numberCurvatures) {

                TFloatVec storage;
                storage.insert(storage.end(), &gradients[j], &gradients[j + numberGradients]);
                storage.insert(storage.end(), &curvatures[j], &curvatures[k + numberCurvatures]);
                auto gradient = makeGradient(&storage[0], numberGradients);
                auto curvature = makeCurvature(&storage[numberGradients], numberCurvatures);

                if (uniform01[i] < 0.1) {
                    accumulator.addMissingDerivatives(features[i], gradient, curvature);
                    ++expectedMissingCounts[features[i]];
                    expectedMissingGradients[features[i]] += gradient;
                    expectedMissingCurvatures[features[i]] += rowMajorHessian(numberParameters, curvature);
                } else {
                    accumulator.addDerivatives(features[i], splits[features[i]][i],
                                               gradient, curvature);
                    ++expectedCounts[features[i]][splits[features[i]][i]];
                    expectedGradients[features[i]][splits[features[i]][i]] += gradient;
                    expectedCurvatures[features[i]][splits[features[i]][i]] +=
                        rowMajorHessian(numberParameters, curvature);
                }
            }
        };

        auto validate = [&](const TDerivativesVecVec& derivatives, 
                            const TDerivativesVec& missingDerivatives) {
            BOOST_REQUIRE_EQUAL(expectedCounts.size(), derivatives.size());
            for (std::size_t i = 0; i < expectedCounts.size(); ++i) {
                BOOST_REQUIRE_EQUAL(expectedCounts[i].size(), derivatives[i].size());
                for (std::size_t j = 0; j < expectedGradients[i].size(); ++j) {
                    TMatrix curvature{
                        derivatives[i][j].s_Curvature.selfadjointView<Eigen::Upper>()};
                    BOOST_REQUIRE_EQUAL(expectedCounts[i][j], derivatives[i][j].s_Count);
                    BOOST_REQUIRE_EQUAL(expectedGradients[i][j], derivatives[i][j].s_Gradient);
                    BOOST_REQUIRE_EQUAL(expectedCurvatures[i][j], curvature);
                }
            }
            BOOST_REQUIRE_EQUAL(expectedMissingCounts.size(), missingDerivatives.size());
            for (std::size_t i = 0; i < expectedMissingCounts.size(); ++i) {
                TMatrix curvature{
                    missingDerivatives[i].s_Curvature.selfadjointView<Eigen::Upper>()};
                BOOST_REQUIRE_EQUAL(expectedMissingCounts[i], missingDerivatives[i].s_Count);
                BOOST_REQUIRE_EQUAL(expectedMissingGradients[i], missingDerivatives[i].s_Gradient);
                BOOST_REQUIRE_EQUAL(expectedMissingCurvatures[i], curvature);
            }
        };

        LOG_TRACE(<< "Test accumulation");

        TSplitDerivativesAccumulator accumulator1{featureSplits, numberParameters};
        addDerivatives(accumulator1);

        TDerivativesVecVec derivatives;
        TDerivativesVec missingDerivatives;
        std::tie(derivatives, missingDerivatives) = accumulator1.read();

        validate(derivatives, missingDerivatives);

        LOG_TRACE(<< "Test merge");

        rng.generateUniformSamples(0.0, 1.0, numberSamples, uniform01);
        rng.generateUniformSamples(-1.5, 1.0, numberSamples * numberGradients, gradients);
        rng.generateUniformSamples(0.1, 0.5, numberSamples * numberCurvatures, curvatures);

        TSplitDerivativesAccumulator accumulator2{featureSplits, numberParameters};
        addDerivatives(accumulator2);
        accumulator1.merge(accumulator2);

        std::tie(derivatives, missingDerivatives) = accumulator1.read();

        validate(derivatives, missingDerivatives);

        LOG_TRACE(<< "Test copy");

        TSplitDerivativesAccumulator accumulator3{accumulator1};
        BOOST_REQUIRE_EQUAL(accumulator1.checksum(), accumulator3.checksum());
    }
}
}

BOOST_AUTO_TEST_CASE(testDerivativesAccumulator) {

    // Test individual derivatives accumulation for single and multi parameter
    // loss functions.

    testDerivativesAccumulatorFor(1 /*loss function parameter*/);
    testDerivativesAccumulatorFor(3 /*loss function parameters*/);
}

BOOST_AUTO_TEST_CASE(testSplitDerivativesAccumulator) {

    // Test per split derivatives accumulation for single and multi parameter
    // loss functions.

    testSplitDerivativesAccumulatorFor(1 /*loss function parameter*/);
    testSplitDerivativesAccumulatorFor(3 /*loss function parameters*/);
}

BOOST_AUTO_TEST_SUITE_END()
