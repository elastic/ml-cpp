/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTreeShapFeatureImportance.h>

#include <algorithm>
#include <tuple>

namespace ml {
namespace maths {


using TRowItr = core::CDataFrame::TRowItr;


CTreeShapFeatureImportance::TDoubleVecVec
CTreeShapFeatureImportance::shap(const core::CDataFrame &frame, const CDataFrameCategoryEncoder &encoder) {


    std::vector<std::size_t> maxDepthVec;
    maxDepthVec.reserve(m_Trees.size());
    for (auto &tree : m_Trees) {
        auto samplesPerNode = this->samplesPerNode(tree, frame, encoder);
        std::size_t maxDepth = this->updateNodeValues(tree, 0, samplesPerNode, 0);
        maxDepthVec.push_back(maxDepth);
        m_SamplesPerNode.emplace_back(std::move(samplesPerNode));
    }

    auto result = frame.readRows(1, 0, frame.numberRows(), core::bindRetrievableState(
            [&](auto &phiVec, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    auto encodedRow{encoder.encode(*row)};
                    TDoubleVec phi(frame.numberColumns() + 1, 0);
                    for (int i = 0; i < m_Trees.size(); ++i) {
                        phi[frame.numberColumns()] += m_Trees[i][0].value();
                        this->shapRecursive(m_Trees[i], m_SamplesPerNode[i], encodedRow, phi, SPath(maxDepthVec[i] + 1), 0,
                                            1.0, 1.0,
                                            -1, 0);
                    }
                    for (int j = 0; j < phi.size(); ++j) {
                        phi[j] /= m_Trees.size();
                    }
                    phiVec.emplace_back(std::move(phi));
                }
            }, TDoubleVecVec())
    );

    auto &state = result.first;
    TDoubleVecVec phiVec{std::move(state[0].s_FunctionState)};
    for (int i = 1; i < state.size(); ++i) {
        auto &otherPhiVec = state[i].s_FunctionState;
        phiVec.insert(phiVec.end(), otherPhiVec.begin(), otherPhiVec.end());
    }

    return phiVec;
}

CTreeShapFeatureImportance::TDoubleVec
CTreeShapFeatureImportance::samplesPerNode(const TTree &tree, const core::CDataFrame &frame,
                                           const CDataFrameCategoryEncoder &encoder) const {
    auto result = frame.readRows(1, 0, frame.numberRows(), core::bindRetrievableState(
            [&](std::vector<double> &state, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    auto encodedRow{encoder.encode(*row)};
                    auto node{tree[0]};
                    state[0] += 1;
                    std::size_t nextIndex;
                    while (node.isLeaf() == false) {
                        if (node.assignToLeft(encodedRow)) {
                            nextIndex = node.leftChildIndex();
                        } else {
                            nextIndex = node.rightChildIndex();
                        }
                        state[nextIndex] += 1.0;
                        node = tree[nextIndex];
                    }
                }
            }, TDoubleVec(tree.size())));

    auto &state = result.first;
    TDoubleVec totalSamplesPerNode{std::move(state[0].s_FunctionState)};
    for (std::size_t i = 1; i < state.size(); ++i) {
        for (std::size_t nodeIndex = 0; nodeIndex < totalSamplesPerNode.size(); ++nodeIndex) {
            totalSamplesPerNode[nodeIndex] += state[i].s_FunctionState[nodeIndex];
        }
    }

    return totalSamplesPerNode;
}

ml::maths::CTreeShapFeatureImportance::CTreeShapFeatureImportance(TTreeVec trees) : m_Trees{std::move(trees)}, m_SamplesPerNode() {
m_SamplesPerNode.reserve(m_Trees.size());
}

std::size_t
CTreeShapFeatureImportance::updateNodeValues(TTree &tree, std::size_t nodeIndex, const TDoubleVec &samplesPerNode,
                                             std::size_t depth) {
    auto &node{tree[nodeIndex]};
    if (node.isLeaf()) {
        return 0;
    }

    std::size_t depthLeft = this->updateNodeValues(tree, node.leftChildIndex(), samplesPerNode, depth + 1);
    std::size_t depthRight = this->updateNodeValues(tree, node.rightChildIndex(), samplesPerNode, depth + 1);

    double leftWeight = samplesPerNode[node.leftChildIndex()];
    double rightWeight = samplesPerNode[node.rightChildIndex()];
    double averageValue =
            (leftWeight * tree[node.leftChildIndex()].value() +
             rightWeight * tree[node.rightChildIndex()].value()) /
            (leftWeight + rightWeight);
    node.value(averageValue);
    return std::max(depthLeft, depthRight) + 1;
}

void CTreeShapFeatureImportance::shapRecursive(const TTree &tree, const TDoubleVec &samplesPerNode,
                                               const CEncodedDataFrameRowRef &encodedRow,
                                               CTreeShapFeatureImportance::TDoubleVec &phi, SPath splitPath,
                                               std::size_t nodeIndex, double parentFractionZero,
                                               double parentFractionOne, int parentFeatureIndex, int uniqueDepth) {
    this->extendPath(splitPath, parentFractionZero, parentFractionOne, parentFeatureIndex);
    if (tree[nodeIndex].isLeaf()) {
        double leafValue = tree[nodeIndex].value();
        for (int i = 1; i <= uniqueDepth; ++i) {
            double scale = this->sumUnwoundPath(splitPath, i);
            phi[splitPath.featureIndex(i)] +=
                    scale * (splitPath.fractionOnes(i) - splitPath.fractionZeros(i)) * leafValue;
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
        size_t splitFeature = tree[nodeIndex].splitFeature();
        auto it = std::find(splitPath.s_FeatureIndex.begin(), splitPath.s_FeatureIndex.end(),
                            splitFeature);
        if (it != splitPath.s_FeatureIndex.end()) {
            auto pathIndex = std::distance(splitPath.s_FeatureIndex.begin(), it);
            incomingFractionZero = splitPath.fractionZeros(pathIndex);
            incomingFractionOne = splitPath.fractionOnes(pathIndex);
            this->unwindPath(splitPath, pathIndex, uniqueDepth);
        }

        double hotFractionZero = samplesPerNode[hotIndex] / samplesPerNode[nodeIndex];
        double coldFractionZero = samplesPerNode[coldIndex] / samplesPerNode[nodeIndex];

        this->shapRecursive(tree, samplesPerNode, encodedRow, phi, splitPath, hotIndex,
                            incomingFractionZero * hotFractionZero,
                            incomingFractionOne, splitFeature, uniqueDepth + 1);
        this->shapRecursive(tree, samplesPerNode, encodedRow, phi, splitPath, coldIndex,
                            incomingFractionZero * coldFractionZero, 0.0,
                            splitFeature, uniqueDepth + 1);
    }

}

void
CTreeShapFeatureImportance::extendPath(CTreeShapFeatureImportance::SPath &path, double fractionZero, double fractionOne,
                                       int featureIndex) {
    path.extend(featureIndex, fractionZero, fractionOne);
    std::size_t pathDepth{path.depth()};
    for (int i = pathDepth - 1; i >= 0; --i) {
        path.s_Scale[i + 1] += fractionOne * path.s_Scale[i] * (i + 1.0) / (pathDepth + 1.0);
        path.s_Scale[i] = fractionZero * path.s_Scale[i] * (pathDepth - i) / (pathDepth + 1.0);
    }
}

double CTreeShapFeatureImportance::sumUnwoundPath(const CTreeShapFeatureImportance::SPath &path, int pathIndex) const {
    double total{0.0};
    std::size_t pathDepth{path.depth()};
    double nextFractionOne{path.scale(pathDepth)};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        for (int i = pathDepth - 1; i >= 0; --i) {
            double tmp = nextFractionOne * (pathDepth + 1) / ((i + 1) * fractionOne);
            nextFractionOne = path.scale(i) - tmp * fractionZero * (pathDepth - i) / (pathDepth + 1);
            total += tmp;
        }
    } else {
        for (int i = pathDepth - 1; i >= 0; --i) {
            total += path.scale(i) * (pathDepth + 1) / (fractionZero * (pathDepth - i));
        }
    }

    return total;
}

void CTreeShapFeatureImportance::unwindPath(CTreeShapFeatureImportance::SPath &path, int pathIndex, int uniqueDepth) {
    double pathDepth = path.depth();
    double nextFractionOne{path.scale(pathDepth)};
    double fractionOne{path.fractionOnes(pathIndex)};
    double fractionZero{path.fractionZeros(pathIndex)};

    if (fractionOne != 0) {
        for (int i = pathDepth - 1; i >= 0; --i) {
            double tmp = nextFractionOne * (pathDepth + 1) / ((i + 1) * fractionOne);
            nextFractionOne = path.scale(i) - tmp * fractionZero * (pathDepth - i) / (pathDepth + 1);
            path.s_Scale[i] = tmp;
        }
    } else {
        for (int i = pathDepth - 1; i >= 0; --i) {
            path.s_Scale[i] = path.scale(i) * (pathDepth + 1) / (fractionZero * (pathDepth - i));
        }
    }

    for (int i = pathIndex; i < pathDepth; ++i) {
        path.s_FeatureIndex[i] = path.s_FeatureIndex[i + 1];
        path.s_FractionOnes[i] = path.s_FractionOnes[i + 1];
        path.s_FractionZeros[i] = path.s_FractionZeros[i + 1];
    }

}

}
}
