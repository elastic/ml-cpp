/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTreeShapFeatureImportance.h>

#include <boost/optional.hpp>

#include <algorithm>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {

using TRowItr = core::CDataFrame::TRowItr;

void CTreeShapFeatureImportance::shap(core::CDataFrame& frame,
                                      const CDataFrameCategoryEncoder& encoder,
                                      std::size_t offset) {
    TSizeVec maxDepthVec;
    maxDepthVec.reserve(m_Trees.size());
    for (auto& tree : m_Trees) {
        auto samplesPerNode = CTreeShapFeatureImportance::samplesPerNode(
            tree, frame, encoder, m_NumberThreads);
        std::size_t maxDepth =
            CTreeShapFeatureImportance::updateNodeValues(tree, 0, samplesPerNode, 0);
        maxDepthVec.push_back(maxDepth);
        m_SamplesPerNode.emplace_back(std::move(samplesPerNode));
    }
    std::size_t maxDepthOverall =
        *std::max_element(maxDepthVec.begin(), maxDepthVec.end());

    auto result = frame.writeColumns(m_NumberThreads, [&](const TRowItr& beginRows,
                                                          const TRowItr& endRows) {
        // need a bit more memory than max depth
        TElementVec pathVector(((maxDepthOverall + 2) * (maxDepthOverall + 3)) / 2);
        TDoubleVec scaleVector(((maxDepthOverall + 2) * (maxDepthOverall + 3)) / 2);
        for (auto row = beginRows; row != endRows; ++row) {
            auto encodedRow{encoder.encode(*row)};
            for (std::size_t i = 0; i < m_Trees.size(); ++i) {
                CTreeShapFeatureImportance::shapRecursive(
                    m_Trees[i], m_SamplesPerNode[i], encoder, encodedRow, 0,
                    1.0, 1.0, -1, offset, row, 0,
                    CPathElementAccessor(pathVector.begin(), scaleVector.begin()));
            }
        }
    });
}

CTreeShapFeatureImportance::TDoubleVec
CTreeShapFeatureImportance::samplesPerNode(const TTree& tree,
                                           const core::CDataFrame& frame,
                                           const CDataFrameCategoryEncoder& encoder,
                                           std::size_t numThreads) {
    auto result = frame.readRows(
        numThreads, core::bindRetrievableState(
                        [&](TDoubleVec& samplesPerNode,
                            const TRowItr& beginRows, const TRowItr& endRows) {
                            for (auto row = beginRows; row != endRows; ++row) {
                                auto encodedRow{encoder.encode(*row)};
                                const CBoostedTreeNode* node{&tree[0]};
                                samplesPerNode[0] += 1.0;
                                std::size_t nextIndex;
                                while (node->isLeaf() == false) {
                                    if (node->assignToLeft(encodedRow)) {
                                        nextIndex = node->leftChildIndex();
                                    } else {
                                        nextIndex = node->rightChildIndex();
                                    }
                                    samplesPerNode[nextIndex] += 1.0;
                                    node = &(tree[nextIndex]);
                                }
                            }
                        },
                        TDoubleVec(tree.size())));

    auto& state = result.first;
    TDoubleVec totalSamplesPerNode{std::move(state[0].s_FunctionState)};
    for (std::size_t i = 1; i < state.size(); ++i) {
        for (std::size_t nodeIndex = 0; nodeIndex < totalSamplesPerNode.size(); ++nodeIndex) {
            totalSamplesPerNode[nodeIndex] += state[i].s_FunctionState[nodeIndex];
        }
    }

    return totalSamplesPerNode;
}

CTreeShapFeatureImportance::CTreeShapFeatureImportance(TTreeVec trees, std::size_t threads)
    : m_Trees{std::move(trees)}, m_NumberThreads{threads}, m_SamplesPerNode() {
    m_SamplesPerNode.reserve(m_Trees.size());
}

std::size_t CTreeShapFeatureImportance::updateNodeValues(TTree& tree,
                                                         std::size_t nodeIndex,
                                                         const TDoubleVec& samplesPerNode,
                                                         std::size_t depth) {
    auto& node{tree[nodeIndex]};
    if (node.isLeaf()) {
        return 0;
    }

    std::size_t depthLeft{CTreeShapFeatureImportance::updateNodeValues(
        tree, node.leftChildIndex(), samplesPerNode, depth + 1)};
    std::size_t depthRight{CTreeShapFeatureImportance::updateNodeValues(
        tree, node.rightChildIndex(), samplesPerNode, depth + 1)};

    double leftWeight{samplesPerNode[node.leftChildIndex()]};
    double rightWeight{samplesPerNode[node.rightChildIndex()]};
    double averageValue{(leftWeight * tree[node.leftChildIndex()].value() +
                         rightWeight * tree[node.rightChildIndex()].value()) /
                        (leftWeight + rightWeight)};
    node.value(averageValue);
    return std::max(depthLeft, depthRight) + 1;
}

void CTreeShapFeatureImportance::shapRecursive(const TTree& tree,
                                               const TDoubleVec& samplesPerNode,
                                               const CDataFrameCategoryEncoder& encoder,
                                               const CEncodedDataFrameRowRef& encodedRow,
                                               std::size_t nodeIndex,
                                               double parentFractionZero,
                                               double parentFractionOne,
                                               int parentFeatureIndex,
                                               std::size_t offset,
                                               core::CDataFrame::TRowItr& row,
                                               int nextIndex,
                                               const CPathElementAccessor& parentSplitPath) const {
    CPathElementAccessor splitPath{parentSplitPath, nextIndex};

    CTreeShapFeatureImportance::extendPath(splitPath, parentFractionZero, parentFractionOne,
                                           parentFeatureIndex, nextIndex);
    if (tree[nodeIndex].isLeaf()) {
        double leafValue = tree[nodeIndex].value();
        for (int i = 1; i < nextIndex; ++i) {
            double scale = CTreeShapFeatureImportance::sumUnwoundPath(splitPath, i, nextIndex);
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
            row->writeColumn(
                offset + inputColumnIndex,
                (*row)[offset + inputColumnIndex] +
                    scale * (splitPath.fractionOnes(i) - splitPath.fractionZeros(i)) * leafValue);
        }

    } else {
        maths::CBoostedTreeNode::TNodeIndex hotIndex, coldIndex;
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

        double hotFractionZero = samplesPerNode[hotIndex] / samplesPerNode[nodeIndex];
        double coldFractionZero = samplesPerNode[coldIndex] / samplesPerNode[nodeIndex];
        this->shapRecursive(tree, samplesPerNode, encoder, encodedRow, hotIndex,
                            incomingFractionZero * hotFractionZero, incomingFractionOne,
                            splitFeature, offset, row, nextIndex, splitPath);
        this->shapRecursive(tree, samplesPerNode, encoder, encodedRow, coldIndex,
                            incomingFractionZero * coldFractionZero, 0.0,
                            splitFeature, offset, row, nextIndex, splitPath);
    }
}

void CTreeShapFeatureImportance::extendPath(CPathElementAccessor& path,
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
    path.setValues(nextIndex, fractionOne, fractionZero, featureIndex);
    auto scalePath{path.scale()};
    if (nextIndex == 0) {
        scalePath[nextIndex] = 1.0;
    } else {
        scalePath[nextIndex] = 0.0;
    }
    double stepDown{fractionOne / static_cast<double>(nextIndex + 1)};
    double stepUp {fractionZero / static_cast<double>(nextIndex + 1)};
    double countDown{static_cast<double>(nextIndex)*stepDown};
    double countUp{stepUp};
    for (int i = (nextIndex - 1); i >= 0; --i, countDown -= stepDown, countUp += stepUp) {
        scalePath[i + 1] += scalePath[i] * countDown;
        scalePath[i] *= countUp;
    }

    ++nextIndex;
}

double CTreeShapFeatureImportance::sumUnwoundPath(const CPathElementAccessor& path,
                                                  int pathIndex,
                                                  int nextIndex) {
    const auto& scalePath{path.scale()};
    double total{0.0};
    int pathDepth = nextIndex - 1;
    double nextFractionOne{scalePath[pathDepth]};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};
    if (fractionOne != 0) {
        double pD = static_cast<double>(pathDepth + 1);
        double countUp{fractionZero / pD};
        double countDown{(pD - 1.0) * (fractionOne / pD)};
        for (int i = pathDepth - 1; i >= 0; --i) {
            double tmp = nextFractionOne / countDown;
            nextFractionOne = scalePath[i] - tmp * countUp;
            total += tmp;
            countUp += fractionZero / pD;
            countDown -= (fractionOne / pD);
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

void CTreeShapFeatureImportance::unwindPath(CPathElementAccessor& path,
                                            int pathIndex,
                                            int& nextIndex) {
    auto& scalePath{path.scale()};
    int pathDepth{nextIndex - 1};
    double nextFractionOne{scalePath[pathDepth]};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        double countUp{0.0};
        double countDown{static_cast<double>(nextIndex)};
        for (int i = pathDepth; i >= 0; --i, ++countUp, --countDown) {
            double tmp = nextFractionOne * static_cast<double>(nextIndex) /
                         (countDown * fractionOne);
            nextFractionOne =
                scalePath[i] -
                tmp * countUp * (fractionZero / static_cast<double>(pathDepth + 1));
            scalePath[i] = tmp;
        }
    } else {
        double pD{static_cast<double>(pathDepth)};
        for (int i = 0; i <= pathDepth; ++i, --pD) {
            scalePath[i] = scalePath[i] *
                           (static_cast<double>(pathDepth + 1) /
                            (fractionZero + std::numeric_limits<double>::epsilon())) /
                           pD;
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
