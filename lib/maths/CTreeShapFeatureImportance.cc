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
        std::size_t maxDepth = CTreeShapFeatureImportance::updateNodeValues(tree, 0, 0);
        maxDepthVec.push_back(maxDepth);
    }

    auto result = frame.writeColumns(m_NumberThreads, [&](const TRowItr& beginRows,
                                                          const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            auto encodedRow{encoder.encode(*row)};
            for (std::size_t i = 0; i < m_Trees.size(); ++i) {
                SPath path(maxDepthVec[i] + 1);
                CTreeShapFeatureImportance::shapRecursive(
                    m_Trees[i], encoder, encodedRow, path, 0, 1.0, 1.0, -1, offset, row);
            }
        }
    });
}

CTreeShapFeatureImportance::CTreeShapFeatureImportance(TTreeVec trees, std::size_t threads)
    : m_Trees{std::move(trees)}, m_NumberThreads{threads} {
}

size_t CTreeShapFeatureImportance::updateNodeValues(TTree& tree,
                                                    std::size_t nodeIndex,
                                                    std::size_t depth) {
    auto& node{tree[nodeIndex]};
    if (node.isLeaf()) {
        return 0;
    }

    std::size_t depthLeft{CTreeShapFeatureImportance::updateNodeValues(
        tree, node.leftChildIndex(), depth + 1)};
    std::size_t depthRight{CTreeShapFeatureImportance::updateNodeValues(
        tree, node.rightChildIndex(), depth + 1)};

    double leftWeight{tree[node.leftChildIndex()].numberSamples()};
    double rightWeight{tree[node.rightChildIndex()].numberSamples()};
    double averageValue{(leftWeight * tree[node.leftChildIndex()].value() +
                         rightWeight * tree[node.rightChildIndex()].value()) /
                        (leftWeight + rightWeight)};
    node.value(averageValue);
    return std::max(depthLeft, depthRight) + 1;
}

void CTreeShapFeatureImportance::shapRecursive(const TTree& tree,
                                               const CDataFrameCategoryEncoder& encoder,
                                               const CEncodedDataFrameRowRef& encodedRow,
                                               SPath& splitPath,
                                               std::size_t nodeIndex,
                                               double parentFractionZero,
                                               double parentFractionOne,
                                               int parentFeatureIndex,
                                               std::size_t offset,
                                               core::CDataFrame::TRowItr& row) const {
    boost::optional<SPath> backupPath;
    CTreeShapFeatureImportance::extendPath(splitPath, parentFractionZero,
                                           parentFractionOne, parentFeatureIndex);
    if (tree[nodeIndex].isLeaf()) {
        double leafValue = tree[nodeIndex].value();
        for (int i = 1; i <= splitPath.depth(); ++i) {
            double scale = CTreeShapFeatureImportance::sumUnwoundPath(splitPath, i);
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
        auto featureIndexEnd{(splitPath.s_FeatureIndex.begin() + splitPath.nextIndex())};
        auto it = std::find(splitPath.s_FeatureIndex.begin(), featureIndexEnd, splitFeature);
        if (it != featureIndexEnd) {
            // Since we pass splitPath by reference, we need to backup the object before unwinding it.
            backupPath = splitPath;
            auto pathIndex = static_cast<std::size_t>(
                std::distance(splitPath.s_FeatureIndex.begin(), it));
            incomingFractionZero = splitPath.fractionZeros(pathIndex);
            incomingFractionOne = splitPath.fractionOnes(pathIndex);
            CTreeShapFeatureImportance::unwindPath(splitPath, pathIndex);
        }

        double hotFractionZero = static_cast<double>(tree[hotIndex].numberSamples()) /
                                 tree[nodeIndex].numberSamples();
        double coldFractionZero = static_cast<double>(tree[coldIndex].numberSamples()) /
                                  tree[nodeIndex].numberSamples();
        std::size_t nextIndex = splitPath.nextIndex();
        this->shapRecursive(tree, encoder, encodedRow, splitPath, hotIndex,
                            incomingFractionZero * hotFractionZero,
                            incomingFractionOne, splitFeature, offset, row);
        this->unwindPath(splitPath, nextIndex);
        this->shapRecursive(tree, encoder, encodedRow, splitPath, coldIndex,
                            incomingFractionZero * coldFractionZero, 0.0,
                            splitFeature, offset, row);
        this->unwindPath(splitPath, nextIndex);
        if (backupPath) {
            // now we swap to restore the data before unwinding
            splitPath = std::move(backupPath.get());
        }
    }
}

void CTreeShapFeatureImportance::extendPath(SPath& path,
                                            double fractionZero,
                                            double fractionOne,
                                            int featureIndex) {
    // Update binomial coefficients to be able to compute Equation (2) from the paper.  In particular,
    // we have in the line path.s_Scale[i + 1] += fractionOne * path.s_Scale[i] * (i + 1.0) / (pathDepth +
    // 1.0) that if we're on the "one" path, i.e. if the last feature selects this path if we include that
    // feature in S (then fractionOne is 1), and we need to consider all the additional ways we now have of
    // constructing each S of each given cardinality i + 1. Each of these come by adding the last feature
    // to sets of size i and we **also** need to scale by the difference in binomial coefficients as both M
    // increases by one and i increases by one. So we get additive term 1{last feature selects path if in S}
    // * scale(i) * (i+1)! (M+1-(i+1)-1)!/(M+1)! / (i! (M-i-1)!/ M!), whence += scale(i) * (i+1) / (M+1).
    path.extend(featureIndex, fractionZero, fractionOne);
    int pathDepth = static_cast<int>(path.depth());
    for (int i = (pathDepth - 1); i >= 0; --i) {
        path.s_Scale[i + 1] += fractionOne * path.s_Scale[i] *
                               static_cast<double>(i + 1.0) /
                               static_cast<double>(pathDepth + 1.0);
        path.s_Scale[i] = fractionZero * path.s_Scale[i] *
                          static_cast<double>(pathDepth - i) /
                          static_cast<double>(pathDepth + 1.0);
    }
}

double CTreeShapFeatureImportance::sumUnwoundPath(const SPath& path, std::size_t pathIndex) {
    double total{0.0};
    int pathDepth = static_cast<int>(path.depth());
    double nextFractionOne{path.scale(pathDepth)};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        for (int i = pathDepth - 1; i >= 0; --i) {
            double tmp = nextFractionOne * static_cast<double>(pathDepth + 1) /
                         (static_cast<double>(i + 1) * fractionOne +
                          std::numeric_limits<double>::epsilon());
            nextFractionOne = path.scale(i) - tmp * fractionZero *
                                                  static_cast<double>(pathDepth - i) /
                                                  static_cast<double>(pathDepth + 1);
            total += tmp;
        }
    } else {
        for (int i = pathDepth - 1; i >= 0; --i) {
            total += path.scale(i) * static_cast<double>(pathDepth + 1) /
                     (fractionZero * static_cast<double>(pathDepth - i) +
                      std::numeric_limits<double>::epsilon());
        }
    }

    return total;
}

void CTreeShapFeatureImportance::unwindPath(SPath& path, std::size_t pathIndex) {
    int pathDepth = static_cast<int>(path.depth());
    double nextFractionOne{path.scale(pathDepth)};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        for (int i = pathDepth - 1; i >= 0; --i) {
            double tmp = nextFractionOne * (pathDepth + 1) /
                         (static_cast<double>(i + 1) * fractionOne);
            nextFractionOne = path.scale(i) - tmp * fractionZero * (pathDepth - i) /
                                                  (pathDepth + 1);
            path.s_Scale[i] = tmp;
        }
    } else {
        for (int i = pathDepth - 1; i >= 0; --i) {
            path.s_Scale[i] = path.scale(i) * static_cast<double>(pathDepth + 1) /
                              (fractionZero * static_cast<double>(pathDepth - i));
        }
    }
    path.reduce(pathIndex);
}
}
}
