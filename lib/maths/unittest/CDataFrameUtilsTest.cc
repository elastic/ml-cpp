/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameUtilsTest.h"

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameUtils.h>

#include <test/CRandomNumbers.h>

#include <boost/filesystem.hpp>

#include <vector>

using namespace ml;

void CDataFrameUtilsTest::testStandardizeColumns() {

    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TMeanVarAccumulatorVec =
        std::vector<maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator>;
    using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;

    test::CRandomNumbers rng;

    std::size_t rows{2000};
    std::size_t cols{4};
    std::size_t capacity{500};

    TDoubleVecVec values(4);
    TMeanVarAccumulatorVec moments(4);
    {
        std::size_t i = 0;
        for (auto a : {-10.0, 0.0}) {
            for (auto b : {5.0, 30.0}) {
                rng.generateUniformSamples(a, b, rows, values[i++]);
            }
        }
        for (i = 0; i < cols; ++i) {
            moments[i].add(values[i]);
        }
    }

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(
                   boost::filesystem::current_path().string(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    core::stopDefaultAsyncExecutor();

    for (auto threads : {1, 4}) {

        for (const auto& factory : {makeOnDisk, makeMainMemory}) {

            auto frame = factory();

            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&values, i, cols](core::CDataFrame::TFloatVecItr column,
                                                   std::int32_t&) {
                    for (std::size_t j = 0; j < cols; ++j, ++column) {
                        *column = values[j][i];
                    }
                });
            }
            frame->finishWritingRows();

            CPPUNIT_ASSERT(maths::CDataFrameUtils::standardizeColumns(threads, *frame));

            // Check the column values are what we expect given the data we generated.

            bool passed{true};
            frame->readRows(1, [&](core::CDataFrame::TRowItr beginRows,
                                   core::CDataFrame::TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t j = 0; j < row->numberColumns(); ++j) {
                        double mean{maths::CBasicStatistics::mean(moments[j])};
                        double sd{std::sqrt(maths::CBasicStatistics::variance(moments[j]))};
                        double expected{(values[j][row->index()] - mean) / sd};
                        if (std::fabs((*row)[j] - expected) > 1e-6) {
                            LOG_ERROR(<< "Expected " << expected << " got " << (*row)[j]);
                            passed = false;
                        }
                    }
                }
            });

            CPPUNIT_ASSERT(passed);

            // Check that the mean and variance of the columns are zero and one,
            // respectively.

            TMeanVarAccumulatorVec columnsMoments(cols);
            frame->readRows(1, [&](core::CDataFrame::TRowItr beginRows,
                                   core::CDataFrame::TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t j = 0; j < row->numberColumns(); ++j) {
                        columnsMoments[j].add((*row)[j]);
                    }
                }
            });

            for (const auto& columnMoments : columnsMoments) {
                double mean{maths::CBasicStatistics::mean(columnMoments)};
                double variance{maths::CBasicStatistics::variance(columnMoments)};
                LOG_DEBUG(<< "mean = " << mean << ", variance = " << variance);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mean, 1e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, variance, 1e-6);
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

CppUnit::Test* CDataFrameUtilsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameUtilsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testStandardizeColumns",
        &CDataFrameUtilsTest::testStandardizeColumns));

    return suiteOfTests;
}
