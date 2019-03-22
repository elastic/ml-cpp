/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBoostedTreeTest.h"

#include <core/CDataFrame.h>

#include <maths/CBoostedTree.h>

#include <boost/filesystem.hpp>

#include <test/CRandomNumbers.h>

#include <functional>
#include <memory>
#include <utility>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;
using TRowRef = core::CDataFrame::TRowRef;
using TRowItr = core::CDataFrame::TRowItr;

void CBoostedTreeTest::testPiecewiseConstant() {

    test::CRandomNumbers rng;

    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{500};

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(
                   boost::filesystem::current_path().string(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    for (std::size_t test = 0; test < 1; ++test) {
        TDoubleVec p;
        TDoubleVec v;
        rng.generateUniformSamples(0.0, 10.0, 2 * cols - 2, p);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, v);
        for (std::size_t i = 0; i < p.size(); i += 2) {
            std::sort(p.begin() + i, p.begin() + i + 2);
        }

        auto f = [&p, &v, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                if (row[i] >= p[2 * i] && row[i] < p[2 * i + 1]) {
                    result += v[i];
                }
            }
            return result;
        };

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
        }

        TDoubleVec noise;
        rng.generateUniformSamples(-0.2, 0.2, rows, noise);

        for (const auto& factory : {makeOnDisk, makeMainMemory}) {

            auto frame = factory();

            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                    for (std::size_t j = 0; j < cols - 1; ++j, ++column) {
                        *column = x[j][i];
                    }
                });
            }
            frame->finishWritingRows();
            frame->writeColumns(1, [&](TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    row->writeColumn(cols - 1, f(*row) + noise[row->index()]);
                }
            });

            maths::CBoostedTree regression{
                1, cols - 1, std::make_unique<maths::boosted_tree::CMse>()};

            regression.train(*frame);
        }
    }
}

void CBoostedTreeTest::testLinear() {
}

void CBoostedTreeTest::testNonLinear() {
}

void CBoostedTreeTest::testConstantFeatures() {
}

void CBoostedTreeTest::testConstantObjective() {
}

void CBoostedTreeTest::testMissingData() {
}

void CBoostedTreeTest::testErrors() {
}

CppUnit::Test* CBoostedTreeTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBoostedTreeTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testPiecewiseConstant", &CBoostedTreeTest::testPiecewiseConstant));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testLinear", &CBoostedTreeTest::testLinear));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testNonLinear", &CBoostedTreeTest::testNonLinear));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testConstantFeatures", &CBoostedTreeTest::testConstantFeatures));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testMissingData", &CBoostedTreeTest::testMissingData));

    return suiteOfTests;
}
