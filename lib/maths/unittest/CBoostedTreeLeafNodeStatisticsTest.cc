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

#include <core/CLogger.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeLeafNodeStatistics.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CQuantileSketch.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>

BOOST_AUTO_TEST_SUITE(CBoostedTreeLeafNodeStatisticsTest)

using namespace ml;
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TFloatVec = maths::CBoostedTreeLeafNodeStatistics::TFloatVec;
using TFloatVecVec = maths::CBoostedTreeLeafNodeStatistics::TFloatVecVec;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TAlignedFloatVec =
    std::vector<maths::CFloatStorage, core::CAlignedAllocator<maths::CFloatStorage>>;
using TAlignedDoubleVec = std::vector<double, core::CAlignedAllocator<double>>;
using TVector = maths::CDenseVector<double>;
using TVectorVec = std::vector<TVector>;
using TVectorVecVec = std::vector<TVectorVec>;
using TMatrix = maths::CDenseMatrix<double>;
using TMatrixVec = std::vector<TMatrix>;
using TMatrixVecVec = std::vector<TMatrixVec>;
using TDerivatives = maths::CBoostedTreeLeafNodeStatistics::CDerivatives;
using TSplitsDerivatives = maths::CBoostedTreeLeafNodeStatistics::CSplitsDerivatives;

namespace {

template<typename T>
maths::CMemoryMappedDenseVector<T> makeVector(T* storage, std::size_t n) {
    return maths::CMemoryMappedDenseVector<T>{storage, static_cast<int>(n)};
}

template<Eigen::AlignmentType ALIGNMENT, typename T>
maths::CMemoryMappedDenseVector<T, ALIGNMENT> makeAlignedVector(T* storage, std::size_t n) {
    return maths::CMemoryMappedDenseVector<T, ALIGNMENT>{storage, static_cast<int>(n)};
}

template<typename T>
TMatrix columnMajorHessian(std::size_t n, const maths::CMemoryMappedDenseVector<T>& curvatures) {
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

    std::size_t paddedNumberGradients{core::CAlignment::roundup<double>(
        core::CAlignment::E_Aligned16, numberGradients)};

    TAlignedDoubleVec storage1(paddedNumberGradients + numberGradients * numberGradients, 0.0);
    TDerivatives derivatives1{numberParameters, &storage1[0],
                              &storage1[paddedNumberGradients]};

    for (std::size_t j = 0; j < 10; ++j) {
        TAlignedFloatVec rowStorage;
        for (std::size_t i = 0; i < numberGradients; ++i) {
            rowStorage.push_back(gradients[i][j]);
        }
        for (std::size_t i = 0; i < numberCurvatures; ++i) {
            rowStorage.push_back(curvatures[i][j]);
        }
        auto derivatives_ = makeAlignedVector<Eigen::Aligned16>(
            &rowStorage[0], numberGradients + numberCurvatures);
        derivatives1.add(1, derivatives_);
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

    TAlignedDoubleVec storage2(paddedNumberGradients + numberGradients * numberGradients, 0.0);
    TDerivatives derivatives2{numberParameters, &storage2[0],
                              &storage2[paddedNumberGradients]};

    for (std::size_t j = 10; j < 20; ++j) {
        TAlignedFloatVec storage;
        for (std::size_t i = 0; i < numberGradients; ++i) {
            storage.push_back(gradients[i][j]);
        }
        for (std::size_t i = 0; i < numberCurvatures; ++i) {
            storage.push_back(curvatures[i][j]);
        }
        auto derivatives = makeAlignedVector<Eigen::Aligned16>(
            &storage[0], numberGradients + numberCurvatures);
        derivatives2.add(1, derivatives);
    }
    derivatives2.remapCurvature();

    derivatives1.add(derivatives2);

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

    derivatives1.subtract(derivatives2);

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
}

void testPerSplitDerivativesFor(std::size_t numberParameters) {

    LOG_DEBUG(<< "Testing " << numberParameters << " parameters");

    TFloatVecVec featureSplits;
    featureSplits.push_back(TFloatVec{1.0, 2.0, 3.0});
    featureSplits.push_back(TFloatVec{0.1, 0.7, 1.1, 1.4});

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

        auto addDerivatives = [&](TSplitsDerivatives& derivatives) {
            for (std::size_t i = 0, j = 0, k = 0; i < numberSamples;
                 ++i, j += numberGradients, k += numberCurvatures) {

                TAlignedFloatVec storage;
                storage.insert(storage.end(), &gradients[j], &gradients[j + numberGradients]);
                storage.insert(storage.end(), &curvatures[j],
                               &curvatures[k + numberCurvatures]);
                auto derivatives_ = makeAlignedVector<Eigen::Aligned16>(
                    &storage[0], numberGradients + numberCurvatures);
                auto gradient = makeVector(&storage[0], numberGradients);
                auto curvature = makeVector(&storage[numberGradients], numberCurvatures);

                if (uniform01[i] < 0.1) {
                    derivatives.addMissingDerivatives(features[i], derivatives_);
                    ++expectedMissingCounts[features[i]];
                    expectedMissingGradients[features[i]] += gradient;
                    expectedMissingCurvatures[features[i]] +=
                        columnMajorHessian(numberParameters, curvature);
                } else {
                    derivatives.addDerivatives(features[i], splits[features[i]][i], derivatives_);
                    ++expectedCounts[features[i]][splits[features[i]][i]];
                    expectedGradients[features[i]][splits[features[i]][i]] += gradient;
                    expectedCurvatures[features[i]][splits[features[i]][i]] +=
                        columnMajorHessian(numberParameters, curvature);
                }
            }
            derivatives.remapCurvature();
        };

        auto validate = [&](const TSplitsDerivatives& derivatives) {
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

        TSplitsDerivatives derivatives1{featureSplits, numberParameters};

        addDerivatives(derivatives1);
        validate(derivatives1);

        LOG_TRACE(<< "Test merge");

        rng.generateUniformSamples(0.0, 1.0, numberSamples, uniform01);
        rng.generateUniformSamples(-1.5, 1.0, numberSamples * numberGradients, gradients);
        rng.generateUniformSamples(0.1, 0.5, numberSamples * numberCurvatures, curvatures);

        TSplitsDerivatives derivatives2{featureSplits, numberParameters};

        addDerivatives(derivatives2);
        derivatives1.add(derivatives2);
        validate(derivatives1);

        LOG_TRACE(<< "Test copy");

        TSplitsDerivatives derivatives3{derivatives1};
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

BOOST_AUTO_TEST_CASE(testGainBoundComputation) {

    // Check the node gain upper bounds are always larger than the actual node gains.

    using TRegularization = maths::CBoostedTreeRegularization<double>;
    using TLeafNodeStatisticsPtr = maths::CBoostedTreeLeafNodeStatistics::TPtr;
    using TNodeVec = maths::CBoostedTree::TNodeVec;

    std::size_t cols{2};
    TSizeVec extraColumns{2, 3, 4, 5, 6};
    std::size_t rows{50};
    std::size_t numberThreads{1};

    for (std::size_t seed = 0; seed < 1000; ++seed) {
        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, rows);
        test::CRandomNumbers rng;
        rng.seed(seed);
        auto frame = core::makeMainStorageDataFrame(cols, rows).first;
        frame->categoricalColumns(TBoolVec{false, false});
        frame->resizeColumns(numberThreads, cols + extraColumns.size());
        TDoubleVec features;
        features.reserve(rows);
        while (true) {
            rng.generateUniformSamples(0.0, 1.0, rows, features);
            const auto[min, max] =
                std::minmax_element(features.begin(), features.end());
            if (*min < 0.25 && *max > 0.75) {
                break;
            }
        }

        TDoubleVec targets;
        targets.reserve(features.size());
        rng.generateUniformSamples(-10.0, 10.0, features.size(), targets);
        TDoubleVec predictions(rows, 0.0);
        TDoubleVec curvature(rows, 2.0);
        TDoubleVec weights(rows, 1.0);

        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                *(column) = features[i];
                *(++column) = targets[i];
                *(++column) = predictions[i];
                *(++column) = targets[i] - predictions[i];
                *(++column) = curvature[i];
                *(++column) = weights[i];
            });
            sketch.add(features[i]);
        }
        frame->finishWritingRows();

        TFloatVecVec featureSplits;
        TDoubleVec splitValues(3);
        sketch.quantile(25.0, splitValues[0]);
        sketch.quantile(50.0, splitValues[1]);
        sketch.quantile(75.0, splitValues[2]);
        featureSplits.emplace_back(splitValues.begin(), splitValues.end());

        maths::CDataFrameCategoryEncoder encoder{{numberThreads, *frame, 1}};

        frame->writeColumns(1, [&](core::CDataFrame::TRowItr beginRows,
                                   core::CDataFrame::TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                maths::CPackedUInt8Decorator::TUInt8Ary splits;
                splits.fill(0);
                splits[0] = static_cast<std::uint8_t>(
                    std::upper_bound(featureSplits[0].begin(),
                                     featureSplits[0].end(), (*row)[0]) -
                    featureSplits[0].begin());
                *maths::boosted_tree_detail::beginSplits(*row, extraColumns) =
                    maths::CPackedUInt8Decorator{splits};
            }
        });

        maths::CBoostedTreeLeafNodeStatistics::CWorkspace workspace;
        workspace.reinitialize(numberThreads, featureSplits, 1);

        core::CPackedBitVector trainingRowMask(rows, true);

        TSizeVec treeFeatureBag{0};
        TSizeVec nodeFeatureBag{0};

        TRegularization regularization;
        regularization.softTreeDepthLimit(1.0).softTreeDepthTolerance(1.0);

        TNodeVec tree(1);

        auto rootSplit = std::make_shared<maths::CBoostedTreeLeafNodeStatistics>(
            0 /*root*/, extraColumns, 1, numberThreads, *frame, regularization, featureSplits,
            treeFeatureBag, nodeFeatureBag, 0 /*depth*/, trainingRowMask, workspace);

        std::size_t splitFeature;
        double splitValue;
        std::tie(splitFeature, splitValue) = rootSplit->bestSplit();
        bool assignMissingToLeft{rootSplit->assignMissingToLeft()};

        std::size_t leftChildId, rightChildId;
        std::tie(leftChildId, rightChildId) = tree[rootSplit->id()].split(
            splitFeature, splitValue, assignMissingToLeft, rootSplit->gain(),
            rootSplit->curvature(), tree);

        TLeafNodeStatisticsPtr leftChild;
        TLeafNodeStatisticsPtr rightChild;
        std::tie(leftChild, rightChild) = rootSplit->split(
            leftChildId, rightChildId, numberThreads, 0.0, *frame, encoder, regularization,
            treeFeatureBag, nodeFeatureBag, tree[rootSplit->id()], workspace);
        if (leftChild != nullptr) {
            BOOST_TEST_REQUIRE(rootSplit->leftChildMaxGain() >= leftChild->gain());
        }
        if (rightChild != nullptr) {
            BOOST_TEST_REQUIRE(rootSplit->rightChildMaxGain() >= rightChild->gain());
        }
        BOOST_REQUIRE(rightChild != nullptr || leftChild != nullptr);
    }
}

BOOST_AUTO_TEST_SUITE_END()
