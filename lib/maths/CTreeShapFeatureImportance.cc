/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTreeShapFeatureImportance.h>

#include <boost/optional.hpp>

#include <algorithm>
#include <limits>

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
        TElementVec pathVector;
        TDoubleVec scaleVector;
        // need a bit more memory than max depth
        pathVector.reserve(((maxDepthOverall + 2) * (maxDepthOverall + 3)) / 2);
        scaleVector.reserve(((maxDepthOverall + 2) * (maxDepthOverall + 3)) / 2);
        for (auto row = beginRows; row != endRows; ++row) {
            auto encodedRow{encoder.encode(*row)};
            for (std::size_t i = 0; i < m_Trees.size(); ++i) {
                CTreeShapFeatureImportance::shapRecursive(
                    m_Trees[i], m_SamplesPerNode[i], encoder, encodedRow, 0, 1.0, 1.0,
                    -1, offset, row, pathVector.begin(), 0, scaleVector.begin());
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
                                               TElementIt parentSplitPath,
                                               int nextIndex,
                                               TDoubleVecIt parentScalePath) const {
    ElementAccessor splitPath{parentSplitPath + nextIndex};
    std::copy(parentSplitPath, parentSplitPath + nextIndex, splitPath.begin());

    TDoubleVecIt scalePath{parentScalePath + nextIndex};
    std::copy(parentScalePath, parentScalePath + nextIndex, scalePath);

    CTreeShapFeatureImportance::extendPath(splitPath, scalePath,
                                           parentFractionZero, parentFractionOne,
                                           parentFeatureIndex, nextIndex);
    if (tree[nodeIndex].isLeaf()) {
        double leafValue = tree[nodeIndex].value();
        for (int i = 1; i < nextIndex; ++i) {
            double scale = CTreeShapFeatureImportance::sumUnwoundPath(
                splitPath, i, nextIndex, scalePath);
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
        auto featureIndexEnd{(splitPath.begin() + nextIndex)};
        auto it = std::find_if(splitPath.begin(), featureIndexEnd,
                               [splitFeature](const SPathElement& el) {
                                   return el.s_FeatureIndex == splitFeature;
                               });
        if (it != featureIndexEnd) {
            // Since we pass splitPath by reference, we need to backup the object before unwinding it.
            int pathIndex{std::distance(splitPath.begin(), it)};
            incomingFractionZero = splitPath.fractionZeros(pathIndex);
            incomingFractionOne = splitPath.fractionOnes(pathIndex);
            CTreeShapFeatureImportance::unwindPath(splitPath, pathIndex, nextIndex, scalePath);
        }

        double hotFractionZero = samplesPerNode[hotIndex] / samplesPerNode[nodeIndex];
        double coldFractionZero = samplesPerNode[coldIndex] / samplesPerNode[nodeIndex];
        this->shapRecursive(tree, samplesPerNode, encoder, encodedRow, hotIndex,
                            incomingFractionZero * hotFractionZero,
                            incomingFractionOne, splitFeature, offset, row,
                            splitPath.begin(), nextIndex, scalePath);
        this->shapRecursive(tree, samplesPerNode, encoder, encodedRow, coldIndex,
                            incomingFractionZero * coldFractionZero, 0.0, splitFeature,
                            offset, row, splitPath.begin(), nextIndex, scalePath);
    }
}

void CTreeShapFeatureImportance::extendPath(ElementAccessor& path,
                                            TDoubleVecIt& scalePath,
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

    path.featureIndex(nextIndex) = featureIndex;
    path.fractionZeros(nextIndex) = fractionZero;
    path.fractionOnes(nextIndex) = fractionOne;
    if (nextIndex == 0) {
        scalePath[nextIndex] = 1.0;
    } else {
        scalePath[nextIndex] = 0.0;
    }

    for (int i = (nextIndex - 1); i >= 0; --i) {
        scalePath[i + 1] += fractionOne * scalePath[ i] *
                                static_cast<double>(i + 1) /
                                static_cast<double>(nextIndex + 1);
        scalePath[i] = fractionZero * scalePath[i] *
                           static_cast<double>(nextIndex - i) /
                           static_cast<double>(nextIndex + 1);
    }

    ++nextIndex;
}

double CTreeShapFeatureImportance::sumUnwoundPath(const ElementAccessor& path,
                                                  int pathIndex,
                                                  int nextIndex,
                                                  const TDoubleVecIt& scalePath) {
    double total{0.0};
    int pathDepth = nextIndex - 1;
    double nextFractionOne{scalePath[pathDepth]};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};
    if (fractionOne != 0) {
        for (int i = pathDepth - 1; i >= 0; --i) {
            double tmp = nextFractionOne * static_cast<double>(pathDepth + 1) /
                         (static_cast<double>(i + 1) * fractionOne +
                          std::numeric_limits<double>::epsilon());
            nextFractionOne = scalePath[i] - tmp * fractionZero *
                                                 static_cast<double>(pathDepth - i) /
                                                 static_cast<double>(pathDepth + 1);
            total += tmp;
        }
    } else {
        for (int i = pathDepth - 1; i >= 0; --i) {
            total += scalePath[i] * static_cast<double>(pathDepth + 1) /
                     (fractionZero * static_cast<double>(pathDepth - i) +
                      std::numeric_limits<double>::epsilon());
        }
    }

    return total;
}

void CTreeShapFeatureImportance::unwindPath(ElementAccessor& path,
                                            int pathIndex,
                                            int& nextIndex,
                                            TDoubleVecIt& scalePath) {
    int pathDepth = nextIndex - 1;
    double nextFractionOne{scalePath[pathDepth]};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        for (int i = pathDepth; i >= 0; --i) {
            double tmp = nextFractionOne * (pathDepth + 1) /
                         (static_cast<double>(i + 1) * fractionOne);
            nextFractionOne = scalePath[i] - tmp * fractionZero * (pathDepth - i) /
                                                 (pathDepth + 1);
            scalePath[i] = tmp;
        }
    } else {
        for (int i = pathDepth; i >= 0; --i) {
            scalePath[i] = scalePath[i] * static_cast<double>(pathDepth + 1) /
                           (fractionZero * static_cast<double>(pathDepth - i));
        }
    }
    for (int i = pathIndex; i < nextIndex - 1; ++i) {
        path.featureIndex(i) = path.featureIndex(i + 1);
        path.fractionZeros(i) = path.fractionZeros(i + 1);
        path.fractionOnes(i) = path.fractionOnes(i + 1);
    }
    --nextIndex;
}
}
}
