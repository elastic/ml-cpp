/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBoostedTree.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTreeShapFeatureImportanceTest)

using namespace ml;
namespace tt = boost::test_tools;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TTree = std::vector<maths::CBoostedTreeNode>;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TTreeShapFeatureImportanceUPtr= std::unique_ptr<maths::CTreeShapFeatureImportance>;
using TEncoderUPtr = std::unique_ptr<maths::CDataFrameCategoryEncoder>;

class CStubMakeDataFrameCategoryEncoder final : public maths::CMakeDataFrameCategoryEncoder {
public:
    CStubMakeDataFrameCategoryEncoder(size_t numberThreads, const core::CDataFrame &frame, size_t targetColumn)
            : CMakeDataFrameCategoryEncoder(numberThreads, frame, targetColumn) {}

    TEncodingUPtrVec makeEncodings() {
        TEncodingUPtrVec result;
        result.push_back(std::make_unique<maths::CDataFrameCategoryEncoder::CIdentityEncoding>(0, 1.0));
        result.push_back(std::make_unique<maths::CDataFrameCategoryEncoder::CIdentityEncoding>(1, 1.0));
        return result;
    }

};

struct SFixtureSingleTree {
    SFixtureSingleTree() {
        TDoubleVecVec data{
                {0.25, 0.25},
                {0.25, 0.75},
                {0.75, 0.25},
                {0.75, 0.75}
        };

        frame = core::makeMainStorageDataFrame(numberFeatures, numberRows).first;
        for (std::size_t i = 0; i < numberRows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t &) {
                                for (std::size_t j = 0; j < numberFeatures; ++j, ++column) {
                                    *column = data[i][j];
                                }
                            }
            );
        }
        frame->finishWritingRows();

        TTree tree;
        tree.emplace_back();
        tree[0].split(0, 0.5, true, 0.0, 0.0, tree);
        tree[1].split(1, 0.5, true, 0.0, 0.0, tree);
        tree[2].split(1, 0.5, true, 0.0, 0.0, tree);
        tree[3].value(3);
        tree[4].value(8);
        tree[5].value(13);
        tree[6].value(18);

        treeFeatureImportance = std::make_unique<maths::CTreeShapFeatureImportance, std::initializer_list<TTree>>(
                {tree});
        CStubMakeDataFrameCategoryEncoder stubParameters{1, *frame, 0};
        encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(&stubParameters);
    }

    ~SFixtureSingleTree() {}

    TDataFrameUPtr frame;
    std::size_t numberFeatures{2};
    std::size_t numberRows{4};
    TTreeShapFeatureImportanceUPtr treeFeatureImportance;
    TEncoderUPtr encoder;

};

struct SFixtureMultipleTrees {
    SFixtureMultipleTrees() {
        TDoubleVecVec data{
                {0.0, 0.9},
                {0.1, 0.8},
                {0.2, 0.7},
                {0.3, 0.6},
                {0.4, 0.5},
                {0.5, 0.4},
                {0.6, 0.3},
                {0.7, 0.2},
                {0.8, 0.1},
                {0.9, 0.0},
        };

        frame = core::makeMainStorageDataFrame(numberFeatures, numberRows).first;
        for (std::size_t i = 0; i < numberRows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t &) {
                                for (std::size_t j = 0; j < numberFeatures; ++j, ++column) {
                                    *column = data[i][j];
                                }
                            }
            );
        }
        frame->finishWritingRows();

        TTree tree1;
        tree1.emplace_back();
        tree1[0].split(0, 0.55, true, 0.0, 0.0, tree1);
        tree1[1].split(0, 0.41, true, 0.0, 0.0, tree1);
        tree1[2].split(1, 0.25, true, 0.0, 0.0, tree1);
        tree1[3].value(1.18230136);
        tree1[4].value(1.98006658);
        tree1[5].value(3.25350885);
        tree1[6].value(2.42384369);

        TTree tree2;
        tree2.emplace_back();
        tree2[0].split(0, 0.45, true, 0.0, 0.0, tree2);
        tree2[1].split(0, 0.25, true, 0.0, 0.0, tree2);
        tree2[2].split(0, 0.59, true, 0.0, 0.0, tree2);
        tree2[3].value(1.04476388);
        tree2[4].value(1.52799228);
        tree2[5].value(1.98006658);
        tree2[6].value(2.950216);

        treeFeatureImportance = std::make_unique<maths::CTreeShapFeatureImportance,
                std::initializer_list<TTree>>({tree1, tree2});
        CStubMakeDataFrameCategoryEncoder stubParameters{1, *frame, 0};
        encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(&stubParameters);
    }

    ~SFixtureMultipleTrees() {}

    TDataFrameUPtr frame;
    std::size_t numberFeatures{2};
    std::size_t numberRows{10};
    TTreeShapFeatureImportanceUPtr treeFeatureImportance;
    TEncoderUPtr encoder;

};

BOOST_FIXTURE_TEST_CASE(testSingleTreeSamplesPerNode, SFixtureSingleTree) {

    auto samplesPerNode = treeFeatureImportance->samplesPerNode(treeFeatureImportance->trees()[0], *frame, *encoder);
    TDoubleVec expectedSamplesPerNode{4, 2, 2, 1, 1, 1, 1};
    BOOST_TEST_REQUIRE(samplesPerNode == expectedSamplesPerNode);
}

BOOST_FIXTURE_TEST_CASE(testSingleTreeExpectedNodeValues, SFixtureSingleTree) {

    TDoubleVec samplesPerNode{4, 2, 2, 1, 1, 1, 1};
    std::size_t depth = treeFeatureImportance->updateNodeValues(treeFeatureImportance->trees()[0], 0, samplesPerNode,
                                                                0);
    BOOST_TEST_REQUIRE(depth == 2);
    TDoubleVec expectedValues{10.5, 5.5, 15.5, 3.0, 8.0, 13.0, 18.0};
    auto &tree{treeFeatureImportance->trees()[0]};
    for (int i = 0; i < tree.size(); ++i) {
        BOOST_TEST_REQUIRE(tree[i].value() == expectedValues[i]);
    }
}

BOOST_FIXTURE_TEST_CASE(testSingleTreeShapNotNormalized, SFixtureSingleTree) {
    TDoubleVecVec actualPhi = treeFeatureImportance->shap(*frame, *encoder);
    TDoubleVecVec expectedPhi{
            {-5., -2.5, 10.5},
            {-5., 2.5,  10.5},
            {5.,  -2.5, 10.5},
            {5.,  2.5,  10.5}};
    BOOST_TEST_REQUIRE(actualPhi == expectedPhi);
}

BOOST_FIXTURE_TEST_CASE(testMultipleTreesShapNotNormalized, SFixtureMultipleTrees) {
    TDoubleVecVec actualPhi = treeFeatureImportance->shap(*frame, *encoder);
    TDoubleVecVec expectedPhi{
            {-0.82660001, -0.06222489, 2.00235752},
            {-0.82660001, -0.06222489, 2.00235752},
            {-0.82660001, -0.06222489, 2.00235752},
            {-0.58498581, -0.06222489, 2.00235752},
            {-0.58498581, -0.06222489, 2.00235752},
            {0.03993395,  -0.06222489, 2.00235752},
            {0.90245943,  -0.2177871,  2.00235752},
            {1.0269092,   0.0725957,   2.00235752},
            {1.0269092,   0.0725957,   2.00235752},
            {1.0269092,   0.0725957,   2.00235752}};
    BOOST_TEST_REQUIRE(actualPhi.size() == expectedPhi.size());
    for (int i = 0; i < actualPhi.size(); ++i) {
        BOOST_TEST_REQUIRE(actualPhi[i].size() == expectedPhi[i].size());
        for (int j = 0; j < actualPhi[i].size(); ++j) {
            BOOST_TEST_REQUIRE(actualPhi[i][j] == expectedPhi[i][j], tt::tolerance(0.000001));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
