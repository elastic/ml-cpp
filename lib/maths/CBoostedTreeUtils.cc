/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeUtils.h>

#include <core/Concurrency.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/MathsTypes.h>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
using namespace boosted_tree;

namespace {
using TDoubleVector = CDenseVector<double>;
using TDoubleVectorVec = std::vector<TDoubleVector>;
using TDoubleVectorVecVec = std::vector<TDoubleVectorVec>;

void propagateLossGradients(std::size_t node,
                            const std::vector<CBoostedTreeNode>& tree,
                            TDoubleVectorVec& lossGradients) {

    if (tree[node].isLeaf()) {
        return;
    }

    propagateLossGradients(tree[node].leftChildIndex(), tree, lossGradients);
    propagateLossGradients(tree[node].rightChildIndex(), tree, lossGradients);

    // A post order depth first traversal means that loss gradients are written
    // to child nodes before they are read here.
    lossGradients[node] += lossGradients[tree[node].leftChildIndex()];
    lossGradients[node] += lossGradients[tree[node].rightChildIndex()];
}
}

std::size_t rootIndex() {
    return 0;
}

const CBoostedTreeNode& root(const std::vector<CBoostedTreeNode>& tree) {
    return tree[rootIndex()];
}

CBoostedTreeNode& root(std::vector<CBoostedTreeNode>& tree) {
    return tree[rootIndex()];
}

void zeroPrediction(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, 0.0);
    }
}

void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Gradient] + i, 0.0);
    }
}

void writeLossGradient(const TRowRef& row,
                       bool newExample,
                       const TSizeVec& extraColumns,
                       const CDataFrameCategoryEncoder& encoder,
                       const CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Gradient] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.gradient(encoder.encode(row), newExample, prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0, size = lossHessianUpperTriangleSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(extraColumns[E_Curvature] + i, 0.0);
    }
}

void writeLossCurvature(const TRowRef& row,
                        bool newExample,
                        const TSizeVec& extraColumns,
                        const CDataFrameCategoryEncoder& encoder,
                        const CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Curvature] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.curvature(encoder.encode(row), newExample, prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

TDoubleVec
retrainTreeSelectionProbabilities(std::size_t numberThreads,
                                  const core::CDataFrame& frame,
                                  const TSizeVec& extraColumns,
                                  std::size_t dependentVariable,
                                  const CDataFrameCategoryEncoder& encoder,
                                  const core::CPackedBitVector& oldTrainingDataRowMask,
                                  const core::CPackedBitVector& newTrainingDataRowMask,
                                  const CLoss& loss,
                                  const TRegularization& regularization,
                                  double eta,
                                  double etaGrowthRate,
                                  const std::vector<std::vector<CBoostedTreeNode>>& forest) {

    using TFloatVec = std::vector<CFloatStorage>;
    using TRowItr = core::CDataFrame::TRowItr;
    using TLossUPtrVec = std::vector<CLoss::TLossUPtr>;

    std::size_t numberLossParameters{loss.numberParameters()};

    TLossUPtrVec losses;
    losses.reserve(forest.size());
    for (const auto& tree : forest) {
        losses.push_back(
            loss.incremental(eta, regularization.treeTopologyChangePenalty(), tree));
        eta *= etaGrowthRate;
    }

    auto makeComputeTotalLossGradient = [&](const bool& isOldTrainingData) {
        return [&](TDoubleVectorVecVec& leafLossGradients, TRowItr beginRows, TRowItr endRows) {
            TFloatVec storage(numberLossParameters);
            for (auto row = beginRows; row != endRows; ++row) {
                auto prediction = readPrediction(*row, extraColumns, numberLossParameters);
                double actual{readActual(*row, dependentVariable)};
                double weight{readExampleWeight(*row, extraColumns)};
                for (std::size_t i = 0; i < forest.size(); ++i) {
                    auto encodedRow = encoder.encode(*row);
                    std::size_t leaf{root(forest[i]).leafIndex(encodedRow, forest[i])};
                    for (int j = 0; j < prediction.size(); ++j) {
                        storage[j] = prediction(j) - forest[i][leaf].value()(i);
                    }
                    // The prediction after removing the tree from the ensemble.
                    auto predictionMinusTree = TMemoryMappedFloatVector(
                        storage.data(), static_cast<int>(numberLossParameters));
                    auto writer = [&](std::size_t j, double gradient) {
                        leafLossGradients[i][leaf](j) = gradient;
                    };
                    losses[i]->gradient(encodedRow, isOldTrainingData,
                                        predictionMinusTree, actual, writer, weight);
                }
            }
        };
    };

    auto makeZeroLeafLossGradients = [&] {
        TDoubleVectorVecVec leafLossGradients;
        leafLossGradients.reserve(forest.size());
        for (const auto& tree : forest) {
            leafLossGradients.emplace_back(tree.size(), TDoubleVector::Zero(numberLossParameters));
        }
        return leafLossGradients;
    };

    auto computeLeafLossGradients = [&](bool oldTrainingData,
                                        const core::CPackedBitVector& rowMask) {
        auto results = frame.readRows(
            numberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(makeComputeTotalLossGradient(oldTrainingData),
                                       makeZeroLeafLossGradients()),
            &rowMask);

        TDoubleVectorVecVec leafLossGradients{std::move(results.first[0].s_FunctionState)};
        for (std::size_t i = 1; i < results.first.size(); ++i) {
            auto& resultsForWorker = results.first[i].s_FunctionState;
            for (std::size_t j = 0; j < resultsForWorker.size(); ++j) {
                for (std::size_t k = 0; k < resultsForWorker[j].size(); ++k) {
                    leafLossGradients[j][k] += resultsForWorker[j][k];
                }
            }
        }

        return leafLossGradients;
    };

    // Compute the sum of loss gradients for each leaf for the old and
    // new data.

    auto oldTrainingDataLossGradients = computeLeafLossGradients(true, oldTrainingDataRowMask);
    auto newTrainingDataLossGradients = computeLeafLossGradients(false, newTrainingDataRowMask);

    for (std::size_t i = 0; i < forest.size(); ++i) {
        propagateLossGradients(rootIndex(), forest[i], oldTrainingDataLossGradients[i]);
        propagateLossGradients(rootIndex(), forest[i], newTrainingDataLossGradients[i]);
    }

    // We interested in choosing trees for which the total gradient at all
    // nodes is the largest. These at least locally would give the largest
    // gain in loss by adjusting.

    TDoubleVec result(forest.size(), 0.0);
    double Z{0.0};
    for (std::size_t i = 0; i < forest.size(); ++i) {
        for (std::size_t j = 0; j < forest[i].size(); ++j) {
            result[i] += (newTrainingDataLossGradients[i][j] +
                          oldTrainingDataLossGradients[i][j])
                             .lpNorm<1>();
        }
        Z += result[i];
    }
    for (auto& p : result) {
        p /= Z;
    }

    return result;
}
}
}
}
