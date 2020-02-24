/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTreeShapFeatureImportance.h>

#include <maths/CLinearAlgebraShims.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {

using TRowItr = core::CDataFrame::TRowItr;

void CTreeShapFeatureImportance::shap(core::CDataFrame& frame,
                                      const CDataFrameCategoryEncoder& encoder,
                                      std::size_t offset) {
    std::size_t maxDepth{0};
    for (auto& tree : m_Trees) {
        maxDepth = std::max(maxDepth, updateNodeValues(tree, 0, 0));
    }

    frame.writeColumns(m_NumberThreads, [&](TRowItr beginRows, TRowItr endRows) {
        // When traversing a tree, we successively copy the parent path and add one
        // new element to it. This means that if a tree has maxDepth depth, we store
        // 1, 2, ... (maxDepth + 1) elements. The "+1" here comes from the fact that
        // the initial element in the path has split feature -1. Altogether it results
        // in ((maxDepth + 1) * (maxDepth + 2)) / 2 elements to be stored.
        TElementVec pathVector(((maxDepth + 1) * (maxDepth + 2)) / 2);
        TDoubleVec scaleVector(((maxDepth + 1) * (maxDepth + 2)) / 2);
        TVectorVec shap;
        for (auto row = beginRows; row != endRows; ++row) {
            auto encodedRow{encoder.encode(*row)};
            shap.assign(encoder.numberInputColumns(), las::zero(m_Trees[0][0].value()));
            for (const auto& tree : m_Trees) {
                this->shapRecursive(tree, encoder, encodedRow, 0, 1.0, 1.0, -1,
                                    CSplitPath{pathVector.begin(), scaleVector.begin()},
                                    0, shap);
            }
            for (std::size_t i = 0; i < shap.size(); ++i) {
                // TODO fixme
                row->writeColumn(offset + i, shap[i](0));
            }
        }
    });
}

CTreeShapFeatureImportance::CTreeShapFeatureImportance(TTreeVec trees, std::size_t threads)
    : m_Trees{std::move(trees)}, m_NumberThreads{threads} {
}

std::size_t CTreeShapFeatureImportance::updateNodeValues(TTree& tree,
                                                         std::size_t nodeIndex,
                                                         std::size_t depth) {
    auto& node = tree[nodeIndex];
    if (node.isLeaf()) {
        return 0;
    }

    std::size_t depthLeft{CTreeShapFeatureImportance::updateNodeValues(
        tree, node.leftChildIndex(), depth + 1)};
    std::size_t depthRight{CTreeShapFeatureImportance::updateNodeValues(
        tree, node.rightChildIndex(), depth + 1)};

    auto& leftChild = tree[node.leftChildIndex()];
    auto& rightChild = tree[node.rightChildIndex()];
    double leftWeight{static_cast<double>(leftChild.numberSamples())};
    double rightWeight{static_cast<double>(rightChild.numberSamples())};
    node.value((leftWeight * leftChild.value() + rightWeight * rightChild.value()) /
               (leftWeight + rightWeight));

    return std::max(depthLeft, depthRight) + 1;
}

void CTreeShapFeatureImportance::shapRecursive(const TTree& tree,
                                               const CDataFrameCategoryEncoder& encoder,
                                               const CEncodedDataFrameRowRef& encodedRow,
                                               std::size_t nodeIndex,
                                               double parentFractionZero,
                                               double parentFractionOne,
                                               int parentFeatureIndex,
                                               const CSplitPath& parentSplitPath,
                                               int nextIndex,
                                               TVectorVec& shap) const {
    CSplitPath splitPath{parentSplitPath, nextIndex};

    CTreeShapFeatureImportance::extendPath(splitPath, parentFractionZero, parentFractionOne,
                                           parentFeatureIndex, nextIndex);
    if (tree[nodeIndex].isLeaf()) {
        const TVector& leafValue{tree[nodeIndex].value()};
        for (int i = 1; i < nextIndex; ++i) {
            double scale{CTreeShapFeatureImportance::sumUnwoundPath(splitPath, i, nextIndex)};
            std::size_t inputColumnIndex{
                encoder.encoding(splitPath.featureIndex(i)).inputColumnIndex()};
            // inputColumnIndex is read by seeing what the feature at position i is on the path to this leaf.
            // fractionOnes(i) is an indicator variable which tells us if we condition on this variable
            // do we visit this path from that node or not, fractionZeros(i) tells us what proportion of
            // all training data which reaches that node visits this path, i.e. the case we're averaging
            // over that feature. So this is telling us exactly about the difference in E[f | S U {i}] -
            // E[f | S] given we're examining only that part of the sample space concerned with this leaf.
            //
            // The key observation is that the leaves form a disjoint partition of the
            // sample space (set of all training data) so we can compute the full
            // expectation as a simple sum over the contributions of the individual leaves.
            shap[inputColumnIndex] +=
                scale * (splitPath.fractionOnes(i) - splitPath.fractionZeros(i)) * leafValue;
        }
    } else {
        std::size_t hotIndex, coldIndex;
        if (tree[nodeIndex].assignToLeft(encodedRow)) {
            hotIndex = tree[nodeIndex].leftChildIndex();
            coldIndex = tree[nodeIndex].rightChildIndex();
        } else {
            hotIndex = tree[nodeIndex].rightChildIndex();
            coldIndex = tree[nodeIndex].leftChildIndex();
        }
        double incomingFractionZero{1.0};
        double incomingFractionOne{1.0};
        int splitFeature{static_cast<int>(tree[nodeIndex].splitFeature())};

        int pathIndex{splitPath.find(splitFeature, nextIndex)};
        if (pathIndex >= 0) {
            // Since we pass splitPath by reference, we need to backup the object before unwinding it.
            incomingFractionZero = splitPath.fractionZeros(pathIndex);
            incomingFractionOne = splitPath.fractionOnes(pathIndex);
            CTreeShapFeatureImportance::unwindPath(splitPath, pathIndex, nextIndex);
        }

        double hotFractionZero{static_cast<double>(tree[hotIndex].numberSamples()) /
                               tree[nodeIndex].numberSamples()};
        double coldFractionZero{static_cast<double>(tree[coldIndex].numberSamples()) /
                                tree[nodeIndex].numberSamples()};
        this->shapRecursive(tree, encoder, encodedRow, hotIndex,
                            incomingFractionZero * hotFractionZero, incomingFractionOne,
                            splitFeature, splitPath, nextIndex, shap);
        this->shapRecursive(tree, encoder, encodedRow, coldIndex,
                            incomingFractionZero * coldFractionZero, 0.0,
                            splitFeature, splitPath, nextIndex, shap);
    }
}

void CTreeShapFeatureImportance::extendPath(CSplitPath& splitPath,
                                            double fractionZero,
                                            double fractionOne,
                                            int featureIndex,
                                            int& nextIndex) {
    // Update binomial coefficients to be able to compute Equation (2) from the paper.  In particular,
    // we have in the line path.s_Scale[i + 1] += fractionOne * path.s_Scale[i] * (i + 1.0) / (pathDepth +
    // 1.0) that if we're on the "one" path, i.e. if the last feature selects this path if we include that
    // feature in S (then fractionOne is 1), and we need to consider all the additional ways we now have of
    // constructing each S of each given cardinality i + 1. Each of these come by adding the last feature
    // to sets of size i and we **also** need to scale by the difference in binomial coefficients as both M
    // increases by one and i increases by one. So we get additive term 1{last feature selects path if in S}
    // * scale(i) * (i+1)! (M+1-(i+1)-1)!/(M+1)! / (i! (M-i-1)!/ M!), whence += scale(i) * (i+1) / (M+1).
    splitPath.setValues(nextIndex, fractionOne, fractionZero, featureIndex);
    auto scalePath{splitPath.scale()};
    if (nextIndex == 0) {
        scalePath[nextIndex] = 1.0;
    } else {
        scalePath[nextIndex] = 0.0;
    }
    double stepDown{fractionOne / static_cast<double>(nextIndex + 1)};
    double stepUp{fractionZero / static_cast<double>(nextIndex + 1)};
    double countDown{static_cast<double>(nextIndex) * stepDown};
    double countUp{stepUp};
    for (int i = nextIndex - 1; i >= 0; --i, countDown -= stepDown, countUp += stepUp) {
        scalePath[i + 1] += scalePath[i] * countDown;
        scalePath[i] *= countUp;
    }

    ++nextIndex;
}

double CTreeShapFeatureImportance::sumUnwoundPath(const CSplitPath& path, int pathIndex, int nextIndex) {
    const auto& scalePath{path.scale()};
    double total{0.0};
    int pathDepth{nextIndex - 1};
    double nextFractionOne{scalePath[pathDepth]};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};
    if (fractionOne != 0) {
        double pD{static_cast<double>(pathDepth + 1)};
        double stepUp{fractionZero / pD};
        double stepDown{fractionOne / pD};
        double countUp{stepUp};
        double countDown{(pD - 1.0) * stepDown};
        for (int i = pathDepth - 1; i >= 0; --i, countUp += stepUp, countDown -= stepDown) {
            double tmp{nextFractionOne / countDown};
            nextFractionOne = scalePath[i] - tmp * countUp;
            total += tmp;
        }
    } else {
        double pD{static_cast<double>(pathDepth)};
        total =
            std::accumulate(scalePath, scalePath + pathDepth, 0.0,
                            [&pD](double a, double b) { return a + b / pD--; });
        total *= static_cast<double>(pathDepth + 1) /
                 (fractionZero + std::numeric_limits<double>::epsilon());
    }

    return total;
}

void CTreeShapFeatureImportance::unwindPath(CSplitPath& path, int pathIndex, int& nextIndex) {
    auto& scalePath{path.scale()};
    int pathDepth{nextIndex - 1};
    double nextFractionOne{scalePath[pathDepth]};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        double stepUp{fractionZero / static_cast<double>(pathDepth + 1)};
        double stepDown{fractionOne / static_cast<double>(nextIndex)};
        double countUp{0.0};
        double countDown{static_cast<double>(nextIndex) * stepDown};
        for (int i = pathDepth; i >= 0; --i, countUp += stepUp, countDown -= stepDown) {
            double tmp{nextFractionOne / countDown};
            nextFractionOne = scalePath[i] - tmp * countUp;
            scalePath[i] = tmp;
        }
    } else {
        double stepDown{(fractionZero + std::numeric_limits<double>::epsilon()) /
                        static_cast<double>(pathDepth + 1)};
        double countDown{static_cast<double>(pathDepth) * stepDown};
        for (int i = 0; i <= pathDepth; ++i, countDown -= stepDown) {
            scalePath[i] = scalePath[i] / countDown;
        }
    }
    for (int i = pathIndex; i < pathDepth; ++i) {
        path.setValues(i, path.fractionOnes(i + 1), path.fractionZeros(i + 1),
                       path.featureIndex(i + 1));
    }
    --nextIndex;
}
}
}
