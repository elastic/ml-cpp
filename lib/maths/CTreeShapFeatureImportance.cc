/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "core/Concurrency.h"
#include <maths/CTreeShapFeatureImportance.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>

#include <maths/CLinearAlgebraShims.h>
#include <maths/COrderings.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {

CTreeShapFeatureImportance::CTreeShapFeatureImportance(const core::CDataFrame& frame,
                                                       const CDataFrameCategoryEncoder& encoder,
                                                       TTreeVec& forest,
                                                       std::size_t numberTopShapValues)
    : m_NumberTopShapValues{numberTopShapValues}, m_Encoder{&encoder}, m_Forest{&forest} {

    m_ColumnNames.reserve(frame.columnNames().size());
    for (const auto& name : frame.columnNames()) {
        m_ColumnNames.push_back(SHAP_PREFIX + name);
    }

    // When traversing a tree, we successively copy the parent path and add one
    // new element to it. This means that if a tree has maxDepth depth, we store
    // 1, 2, ... (maxDepth + 1) elements. The "+1" here comes from the fact that
    // the initial element in the path has split feature -1. Altogether it results
    // in ((maxDepth + 1) * (maxDepth + 2)) / 2 elements to be stored.
    std::size_t maxDepth{depth(forest)};
    m_PathStorage.resize((maxDepth + 1) * (maxDepth + 2) / 2);
    m_ScaleStorage.resize((maxDepth + 1) * (maxDepth + 2) / 2);

    computeInternalNodeValues(forest);
}

void CTreeShapFeatureImportance::shap(const TRowRef& row, TShapWriter writer) {

    if (m_NumberTopShapValues == 0) {
        return;
    }

    auto encodedRow{m_Encoder->encode(row)};
    m_ShapValues.assign(m_Encoder->numberInputColumns(),
                        las::zero((*m_Forest)[0][0].value()));
    for (const auto& tree : *m_Forest) {
        this->shapRecursive(tree, encodedRow, 0, 1.0, 1.0, -1,
                            CSplitPath{m_PathStorage.begin(), m_ScaleStorage.begin()},
                            0, m_ShapValues);
    }

    m_TopShapValues.resize(m_ShapValues.size());
    std::iota(m_TopShapValues.begin(), m_TopShapValues.end(), 0);
    if (m_NumberTopShapValues < m_TopShapValues.size()) {
        std::nth_element(
            m_TopShapValues.begin(), m_TopShapValues.begin() + m_NumberTopShapValues,
            m_TopShapValues.end(), [this](std::size_t lhs, std::size_t rhs) {
                return COrderings::lexicographical_compare(
                    -las::L1(m_ShapValues[lhs]), lhs, -las::L1(m_ShapValues[rhs]), rhs);
            });
        m_TopShapValues.resize(m_NumberTopShapValues);
        std::sort(m_TopShapValues.begin(), m_TopShapValues.end());
    }

    writer(m_TopShapValues, m_ColumnNames, m_ShapValues);
}

void CTreeShapFeatureImportance::computeNumberSamples(std::size_t numberThreads,
                                                      const core::CDataFrame& frame,
                                                      const CDataFrameCategoryEncoder& encoder,
                                                      TTreeVec& forest) {
    using TRowItr = core::CDataFrame::TRowItr;
    for (auto& tree : forest) {
        if (tree.size() == 1) {
            tree[0].numberSamples(frame.numberRows());
        } else {
            auto result = frame.readRows(
                numberThreads,
                core::bindRetrievableState(
                    [&](TSizeVec& samplesPerNode, TRowItr beginRows, TRowItr endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            auto encodedRow{encoder.encode(*row)};
                            const CBoostedTreeNode* node{&tree[0]};
                            samplesPerNode[0] += 1;
                            std::size_t nextIndex;
                            while (node->isLeaf() == false) {
                                nextIndex = node->assignToLeft(encodedRow)
                                                ? node->leftChildIndex()
                                                : node->rightChildIndex();
                                samplesPerNode[nextIndex] += 1;
                                node = &(tree[nextIndex]);
                            }
                        }
                    },
                    TSizeVec(tree.size())));
            auto& state = result.first;
            TSizeVec totalSamplesPerNode{std::move(state[0].s_FunctionState)};
            for (std::size_t i = 1; i < state.size(); ++i) {
                for (std::size_t nodeIndex = 0;
                     nodeIndex < totalSamplesPerNode.size(); ++nodeIndex) {
                    totalSamplesPerNode[nodeIndex] += state[i].s_FunctionState[nodeIndex];
                }
            }
            for (std::size_t i = 0; i < tree.size(); ++i) {
                tree[i].numberSamples(totalSamplesPerNode[i]);
            }
        }
    }
}

void CTreeShapFeatureImportance::computeInternalNodeValues(TTreeVec& forest) {
    for (auto& tree : forest) {
        computeInternalNodeValues(tree, 0);
    }
}

void CTreeShapFeatureImportance::computeInternalNodeValues(TTree& tree, std::size_t nodeIndex) {
    auto& node = tree[nodeIndex];
    if (node.isLeaf() == false) {
        computeInternalNodeValues(tree, node.leftChildIndex());
        computeInternalNodeValues(tree, node.rightChildIndex());
        auto& leftChild = tree[node.leftChildIndex()];
        auto& rightChild = tree[node.rightChildIndex()];
        double leftWeight{static_cast<double>(leftChild.numberSamples())};
        double rightWeight{static_cast<double>(rightChild.numberSamples())};
        node.value((leftWeight * leftChild.value() + rightWeight * rightChild.value()) /
                   (leftWeight + rightWeight));
    }
}

std::size_t CTreeShapFeatureImportance::depth(const TTreeVec& forest) {
    std::size_t maxDepth{0};
    for (const auto& tree : forest) {
        maxDepth = std::max(maxDepth, depth(tree, 0));
    }
    return maxDepth;
}

std::size_t CTreeShapFeatureImportance::depth(const TTree& tree, std::size_t nodeIndex) {
    auto& node = tree[nodeIndex];
    return node.isLeaf() ? 0
                         : std::max(depth(tree, node.leftChildIndex()),
                                    depth(tree, node.rightChildIndex())) +
                               1;
}

void CTreeShapFeatureImportance::shapRecursive(const TTree& tree,
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
                m_Encoder->encoding(splitPath.featureIndex(i)).inputColumnIndex()};
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
            incomingFractionZero = splitPath.fractionZeros(pathIndex);
            incomingFractionOne = splitPath.fractionOnes(pathIndex);
            CTreeShapFeatureImportance::unwindPath(splitPath, pathIndex, nextIndex);
        }

        double hotFractionZero{incomingFractionZero *
                               static_cast<double>(tree[hotIndex].numberSamples()) /
                               static_cast<double>(tree[nodeIndex].numberSamples())};
        double coldFractionZero{incomingFractionZero *
                                static_cast<double>(tree[coldIndex].numberSamples()) /
                                static_cast<double>(tree[nodeIndex].numberSamples())};
        this->shapRecursive(tree, encodedRow, hotIndex, hotFractionZero, incomingFractionOne,
                            splitFeature, splitPath, nextIndex, shap);
        this->shapRecursive(tree, encodedRow, coldIndex, coldFractionZero, 0.0,
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

const std::string CTreeShapFeatureImportance::SHAP_PREFIX{"feature_importance."};
}
}
