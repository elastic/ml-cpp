/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBoostedTree.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/test/unit_test.hpp>

#include <numeric>
#include <set>

BOOST_AUTO_TEST_SUITE(CTreeShapFeatureImportanceTest)

using namespace ml;
namespace tt = boost::test_tools;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TTree = std::vector<maths::CBoostedTreeNode>;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TTreeShapFeatureImportanceUPtr = std::unique_ptr<maths::CTreeShapFeatureImportance>;
using TEncoderUPtr = std::unique_ptr<maths::CDataFrameCategoryEncoder>;
using TRowItr = core::CDataFrame::TRowItr;
using TSizeSet = std::set<std::size_t>;
using TSizePowerset = std::set<TSizeSet>;
using TSizeVec = std::vector<std::size_t>;
using TVector = maths::CDenseVector<double>;

TVector toVector(double value) {
    TVector result{1};
    result(0) = value;
    return result;
}

class CStubMakeDataFrameCategoryEncoder final : public maths::CMakeDataFrameCategoryEncoder {
public:
    CStubMakeDataFrameCategoryEncoder(std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      std::size_t targetColumn,
                                      std::size_t numberColumns = 2)
        : CMakeDataFrameCategoryEncoder(numberThreads, frame, targetColumn), m_NumberColumns{numberColumns} {
    }

    CMakeDataFrameCategoryEncoder::TEncodingUPtrVec makeEncodings() override {
        TEncodingUPtrVec result;
        for (std::size_t i = 0; i < m_NumberColumns; ++i) {
            result.push_back(std::make_unique<maths::CDataFrameCategoryEncoder::CIdentityEncoding>(
                i, 1.0));
        }
        return result;
    }

private:
    std::size_t m_NumberColumns;
};

struct SFixtureSingleTree {
    SFixtureSingleTree() : s_Frame{}, s_TreeFeatureImportance{}, s_Encoder{} {
        TDoubleVecVec data{{0.25, 0.25}, {0.25, 0.75}, {0.75, 0.25}, {0.75, 0.75}};

        s_Frame = core::makeMainStorageDataFrame(s_NumberFeatures, s_NumberRows).first;
        for (std::size_t i = 0; i < s_NumberRows; ++i) {
            s_Frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j < s_NumberFeatures; ++j, ++column) {
                    *column = data[i][j];
                }
            });
        }
        s_Frame->finishWritingRows();

        TTree tree(1);
        tree[0].split(0, 0.5, true, 0.0, 0.0, tree);
        tree[1].split(1, 0.5, true, 0.0, 0.0, tree);
        tree[2].split(1, 0.5, true, 0.0, 0.0, tree);
        tree[3].value(3);
        tree[4].value(8);
        tree[5].value(13);
        tree[6].value(18);

        tree[0].numberSamples(4);
        tree[1].numberSamples(2);
        tree[2].numberSamples(2);
        tree[3].numberSamples(1);
        tree[4].numberSamples(1);
        tree[5].numberSamples(1);
        tree[6].numberSamples(1);

        s_TreeFeatureImportance =
            std::make_unique<maths::CTreeShapFeatureImportance, std::initializer_list<TTree>>(
                {tree});
        CStubMakeDataFrameCategoryEncoder stubParameters{1, *s_Frame, 0};
        s_Encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(stubParameters);
    }

    TDataFrameUPtr s_Frame;
    std::size_t s_NumberFeatures{2};
    std::size_t s_NumberRows{4};
    TTreeShapFeatureImportanceUPtr s_TreeFeatureImportance;
    TEncoderUPtr s_Encoder;
};

struct SFixtureSingleTreeRandom {
    SFixtureSingleTreeRandom() : s_TreeFeatureImportance{}, s_Encoder{} {
        test::CRandomNumbers rng;
        this->initFrame(rng);

        CStubMakeDataFrameCategoryEncoder stubParameters{1, *s_Frame, 0, s_NumberFeatures};
        s_Encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(stubParameters);

        this->initTree(rng);

        s_TreeFeatureImportance =
            std::make_unique<maths::CTreeShapFeatureImportance, std::initializer_list<TTree>>(
                {s_Tree});
    }

    void initFrame(test::CRandomNumbers& rng) {
        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, s_NumberRows * s_NumberFeatures, values);

        s_Frame = core::makeMainStorageDataFrame(s_NumberFeatures, s_NumberRows).first;
        for (std::size_t i = 0; i < s_NumberRows; ++i) {
            s_Frame->writeRow([&](core::CDataFrame::TFloatVecItr column, int32_t&) {
                for (std::size_t j = 0; j < s_NumberFeatures; ++j, ++column) {
                    *column = values[i * s_NumberFeatures + j];
                }
            });
        }
        s_Frame->finishWritingRows();
    }

    void initTree(test::CRandomNumbers& rng) {
        s_Tree.reserve(s_NumberInnerNodes * 2 + 1);
        TDoubleVecVec bottom;
        bottom.reserve(s_NumberInnerNodes);
        TDoubleVecVec top;
        top.reserve(s_NumberInnerNodes);
        TSizeVec splitFeature(1);
        TDoubleVec splitThreshold(1);

        s_Tree.emplace_back();
        bottom.emplace_back(s_NumberFeatures, -10);
        top.emplace_back(s_NumberFeatures, 10);
        for (std::size_t nodeIndex = 0; nodeIndex < s_NumberInnerNodes; ++nodeIndex) {
            rng.generateUniformSamples(0, s_NumberFeatures, 1, splitFeature);
            rng.generateUniformSamples(bottom[nodeIndex][splitFeature[0]],
                                       top[nodeIndex][splitFeature[0]], 1, splitThreshold);
            s_Tree[nodeIndex].split(splitFeature[0], splitThreshold[0], true,
                                    0.0, 0.0, s_Tree);
            // keep the management of the boundaries, to make sure the generated thresholds are realistic
            TDoubleVec leftChildBottom{bottom[nodeIndex]};
            TDoubleVec rightChildBottom{bottom[nodeIndex]};
            TDoubleVec leftChildTop{top[nodeIndex]};
            TDoubleVec rightChildTop{top[nodeIndex]};
            leftChildTop[splitFeature[0]] = splitThreshold[0];
            rightChildBottom[splitFeature[0]] = splitThreshold[0];
            bottom.push_back(std::move(leftChildBottom));
            bottom.push_back(std::move(rightChildBottom));
            top.push_back(std::move(leftChildTop));
            top.push_back(std::move(rightChildTop));
        }

        std::size_t numberLeafs{s_NumberInnerNodes + 1};
        TDoubleVec leafValues(numberLeafs);
        rng.generateUniformSamples(-5, 5, numberLeafs, leafValues);
        for (std::size_t i = 0; i < numberLeafs; ++i) {
            s_Tree[s_NumberInnerNodes + i].value(toVector(leafValues[i]));
        }

        // set correct number samples
        auto result = s_Frame->readRows(
            1, core::bindRetrievableState(
                   [&](TSizeVec& numberSamples, const TRowItr& beginRows, const TRowItr& endRows) {
                       for (auto row = beginRows; row != endRows; ++row) {
                           auto node{&(s_Tree[0])};
                           auto encodedRow{s_Encoder->encode(*row)};
                           numberSamples[0] += 1;
                           std::size_t nextIndex;
                           while (node->isLeaf() == false) {
                               if (node->assignToLeft(encodedRow)) {
                                   nextIndex = node->leftChildIndex();
                               } else {
                                   nextIndex = node->rightChildIndex();
                               }
                               numberSamples[nextIndex] += 1;
                               node = &(s_Tree[nextIndex]);
                           }
                       }
                   },
                   TSizeVec(s_Tree.size())));
        TSizeVec numberSamples{std::move(result.first[0].s_FunctionState)};
        for (std::size_t i = 0; i < numberSamples.size(); ++i) {
            s_Tree[i].numberSamples(numberSamples[i]);
        }
    }

    TDataFrameUPtr s_Frame;
    std::size_t s_NumberFeatures{5};
    std::size_t s_NumberRows{1000};
    std::size_t s_NumberInnerNodes{15};
    TTreeShapFeatureImportanceUPtr s_TreeFeatureImportance;
    TEncoderUPtr s_Encoder;
    TTree s_Tree;
};

struct SFixtureMultipleTrees {
    SFixtureMultipleTrees()
        : s_Frame{}, s_TreeFeatureImportance{}, s_Encoder{} {
        TDoubleVecVec data{
            {0.0, 0.9}, {0.1, 0.8}, {0.2, 0.7}, {0.3, 0.6}, {0.4, 0.5},
            {0.5, 0.4}, {0.6, 0.3}, {0.7, 0.2}, {0.8, 0.1}, {0.9, 0.0},
        };

        s_Frame = core::makeMainStorageDataFrame(s_NumberFeatures, s_NumberRows).first;
        for (std::size_t i = 0; i < s_NumberRows; ++i) {
            s_Frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j < s_NumberFeatures; ++j, ++column) {
                    *column = data[i][j];
                }
            });
        }
        s_Frame->finishWritingRows();

        TTree tree1(1);
        tree1[0].split(0, 0.55, true, 0.0, 0.0, tree1);
        tree1[0].numberSamples(10);
        tree1[1].split(0, 0.41, true, 0.0, 0.0, tree1);
        tree1[1].numberSamples(6);
        tree1[2].split(1, 0.25, true, 0.0, 0.0, tree1);
        tree1[2].numberSamples(4);
        tree1[3].value(toVector(1.18230136));
        tree1[3].numberSamples(5);
        tree1[4].value(toVector(1.98006658));
        tree1[4].numberSamples(1);
        tree1[5].value(toVector(3.25350885));
        tree1[5].numberSamples(3);
        tree1[6].value(toVector(2.42384369));
        tree1[6].numberSamples(1);

        TTree tree2(1);
        tree2[0].split(0, 0.45, true, 0.0, 0.0, tree2);
        tree2[0].numberSamples(10);
        tree2[1].split(0, 0.25, true, 0.0, 0.0, tree2);
        tree2[1].numberSamples(5);
        tree2[2].split(0, 0.59, true, 0.0, 0.0, tree2);
        tree2[2].numberSamples(5);
        tree2[3].value(toVector(1.04476388));
        tree2[3].numberSamples(3);
        tree2[4].value(toVector(1.52799228));
        tree2[4].numberSamples(2);
        tree2[5].value(toVector(1.98006658));
        tree2[5].numberSamples(1);
        tree2[6].value(toVector(2.950216));
        tree2[6].numberSamples(4);

        s_TreeFeatureImportance =
            std::make_unique<maths::CTreeShapFeatureImportance, std::initializer_list<TTree>>(
                {tree1, tree2});
        CStubMakeDataFrameCategoryEncoder stubParameters{1, *s_Frame, 0};
        s_Encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(stubParameters);
    }

    TDataFrameUPtr s_Frame;
    std::size_t s_NumberFeatures{2};
    std::size_t s_NumberRows{10};
    TTreeShapFeatureImportanceUPtr s_TreeFeatureImportance;
    TEncoderUPtr s_Encoder;
};

class BruteForceTreeShap {
public:
    BruteForceTreeShap(const TTree& tree, std::size_t numberFeatures)
        : m_Tree{tree}, m_Powerset{}, m_NumberFeatures{numberFeatures} {
        this->initPowerset({}, numberFeatures);
    }

    TDoubleVecVec shap(const core::CDataFrame& frame,
                       const maths::CDataFrameCategoryEncoder& encoder,
                       std::size_t numThreads) {
        auto result = frame.readRows(
            numThreads,
            core::bindRetrievableState(
                [&](TDoubleVecVec& phiVec, const TRowItr& beginRows, const TRowItr& endRows) {
                    phiVec.reserve(frame.numberRows());
                    for (auto row = beginRows; row != endRows; ++row) {
                        TDoubleVec phi(row->numberColumns(), 0.0);
                        auto encodedRow{encoder.encode(*row)};
                        for (std::size_t encodedColumnIndex = 0;
                             encodedColumnIndex < encodedRow.numberColumns();
                             ++encodedColumnIndex) {
                            std::size_t inputColumnIndex{
                                encoder.encoding(encodedColumnIndex).inputColumnIndex()};
                            for (TSizeSet S : m_Powerset) { // iterate over all sets without inputColumnIndex
                                if (S.find(inputColumnIndex) != S.end()) {
                                    continue;
                                }
                                double scalingFactor{
                                    1.0 /
                                    (boost::math::binomial_coefficient<double>(
                                         static_cast<unsigned>(m_NumberFeatures),
                                         static_cast<unsigned>(S.size())) *
                                     (static_cast<double>(m_NumberFeatures - S.size())))};
                                double fWithoutIndex =
                                    this->conditionalExpectation(encodedRow, S);
                                S.insert(inputColumnIndex);
                                double fWithIndex =
                                    this->conditionalExpectation(encodedRow, S);
                                phi[inputColumnIndex] +=
                                    scalingFactor * (fWithIndex - fWithoutIndex);
                            }
                        }
                        phiVec.push_back(std::move(phi));
                    }
                },
                TDoubleVecVec()));
        return result.first[0].s_FunctionState;
    }

    double conditionalExpectation(const maths::CEncodedDataFrameRowRef& x,
                                  const TSizeSet& S) {
        return this->conditionalExpectation(x, S, 0ul, 1.0);
    }

private:
    void initPowerset(TSizeSet workset, std::size_t n) {
        if (n == 0) { // all elements has been considered
            m_Powerset.insert(std::move(workset));
            return;
        }

        //branch without n-th element
        initPowerset(workset, n - 1);

        //branch with n-th element
        workset.emplace(n - 1);
        initPowerset(workset, n - 1);
    }

    double conditionalExpectation(const maths::CEncodedDataFrameRowRef& x,
                                  const TSizeSet& S,
                                  std::size_t nodeIndex,
                                  double weight) {
        if (m_Tree[nodeIndex].isLeaf()) {
            // TODO fixme
            return weight * m_Tree[nodeIndex].value()(0);
        } else {
            auto leftChildIndex{m_Tree[nodeIndex].leftChildIndex()};
            auto rightChildIndex{m_Tree[nodeIndex].rightChildIndex()};
            if (S.find(m_Tree[nodeIndex].splitFeature()) != S.end()) {
                if (m_Tree[nodeIndex].assignToLeft(x)) {
                    return this->conditionalExpectation(x, S, leftChildIndex, weight);
                } else {
                    return this->conditionalExpectation(x, S, rightChildIndex, weight);
                }

            } else {
                return this->conditionalExpectation(
                           x, S, leftChildIndex,
                           weight * m_Tree[leftChildIndex].numberSamples() /
                               m_Tree[nodeIndex].numberSamples()) +
                       this->conditionalExpectation(
                           x, S, rightChildIndex,
                           weight * m_Tree[rightChildIndex].numberSamples() /
                               m_Tree[nodeIndex].numberSamples());
            }
        }
    }

private:
    const TTree& m_Tree;
    TSizePowerset m_Powerset{};
    std::size_t m_NumberFeatures;
};

BOOST_FIXTURE_TEST_CASE(testSingleTreeExpectedNodeValues, SFixtureSingleTree) {

    std::size_t depth = maths::CTreeShapFeatureImportance::updateNodeValues(
        s_TreeFeatureImportance->trees()[0], 0, 0);
    BOOST_TEST_REQUIRE(depth == 2);
    TDoubleVec expectedValues{10.5, 5.5, 15.5, 3.0, 8.0, 13.0, 18.0};
    auto& tree{s_TreeFeatureImportance->trees()[0]};
    for (std::size_t i = 0; i < tree.size(); ++i) {
        // TODO fixme
        BOOST_TEST_REQUIRE(tree[i].value()(0) == expectedValues[i]);
    }
}

BOOST_FIXTURE_TEST_CASE(testSingleTreeShap, SFixtureSingleTree) {
    std::size_t offset{s_Frame->numberColumns()};
    s_Frame->resizeColumns(1, offset * 2);
    s_TreeFeatureImportance->shap(*s_Frame, *s_Encoder, offset);
    TDoubleVecVec expectedPhi{{-5., -2.5}, {-5., 2.5}, {5., -2.5}, {5., 2.5}};
    s_Frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t col = 0; col < 2; ++col) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedPhi[row->index()][col],
                                             static_cast<double>((*row)[offset + col]), 1e-7);
            }
        }
    });
}

BOOST_FIXTURE_TEST_CASE(testMultipleTreesShap, SFixtureMultipleTrees) {
    TDoubleVecVec expectedPhi{
        {-1.65320002, -0.12444978}, {-1.65320002, -0.12444978},
        {-1.65320002, -0.12444978}, {-1.16997162, -0.12444978},
        {-1.16997162, -0.12444978}, {0.0798679, -0.12444978},
        {1.80491886, -0.4355742},   {2.0538184, 0.1451914},
        {2.0538184, 0.1451914},     {2.0538184, 0.1451914}};
    std::size_t offset{s_Frame->numberColumns()};
    s_Frame->resizeColumns(1, offset * 2);
    s_TreeFeatureImportance->shap(*s_Frame, *s_Encoder, offset);
    s_Frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t col = 0; col < 2; ++col) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedPhi[row->index()][col],
                                             static_cast<double>((*row)[offset + col]), 1e-7);
            }
        }
    });
}

BOOST_FIXTURE_TEST_CASE(testSingleTreeBruteForceShap, SFixtureSingleTree) {
    BruteForceTreeShap bfShap(s_TreeFeatureImportance->trees()[0], s_NumberFeatures);
    auto actualPhi = bfShap.shap(*s_Frame, *s_Encoder, 1);
    TDoubleVecVec expectedPhi{{-5., -2.5}, {-5., 2.5}, {5., -2.5}, {5., 2.5}};
    for (std::size_t i = 0; i < s_NumberRows; ++i) {
        for (std::size_t j = 0; j < s_NumberFeatures; ++j) {
            BOOST_TEST_REQUIRE(expectedPhi[i][j], actualPhi[i][j]);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testSingleTreeShapRandomDataFrame, SFixtureSingleTreeRandom) {
    // Compare tree shap algorithm with the brute force approach (Algorithm
    // 1 in paper by Lundberg et al.) on a random data set with a random tree.
    BruteForceTreeShap bfShap(this->s_Tree, s_NumberFeatures);
    auto expectedPhi = bfShap.shap(*s_Frame, *s_Encoder, 1);
    std::size_t offset{s_Frame->numberColumns()};
    s_Frame->resizeColumns(1, offset * 2);
    s_TreeFeatureImportance->shap(*s_Frame, *s_Encoder, offset);
    s_Frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t col = 0; col < s_NumberFeatures; ++col) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedPhi[row->index()][col],
                                             static_cast<double>((*row)[offset + col]), 1e-5);
            }
        }
    });
}

BOOST_AUTO_TEST_SUITE_END()
