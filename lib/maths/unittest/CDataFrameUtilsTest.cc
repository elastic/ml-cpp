/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameUtilsTest.h"

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CQuantileSketch.h>

#include <test/CRandomNumbers.h>

#include <boost/filesystem.hpp>

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;

void CDataFrameUtilsTest::testStandardizeColumns() {

    using TMeanVarAccumulatorVec =
        std::vector<maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator>;

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

void CDataFrameUtilsTest::testColumnQuantiles() {

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TQuantileSketchVec = std::vector<maths::CQuantileSketch>;

    test::CRandomNumbers rng;

    std::size_t rows{2000};
    std::size_t cols{4};
    std::size_t capacity{500};

    TDoubleVecVec values(4);
    TQuantileSketchVec expectedQuantiles(4, {maths::CQuantileSketch::E_Linear, 100});
    {
        std::size_t i = 0;
        for (auto a : {-10.0, 0.0}) {
            for (auto b : {5.0, 30.0}) {
                rng.generateUniformSamples(a, b, rows, values[i++]);
            }
        }
        for (i = 0; i < cols; ++i) {
            for (auto x : values[i]) {
                expectedQuantiles[i].add(x);
            }
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

            TQuantileSketchVec actualQuantiles(4, {maths::CQuantileSketch::E_Linear, 100});
            CPPUNIT_ASSERT(maths::CDataFrameUtils::columnQuantiles(threads, *frame, actualQuantiles));

            // Check the quantile sketches match.

            TMeanAccumulatorVec columnsMae(4);

            for (std::size_t i = 5; i < 100; i += 5) {
                for (std::size_t j = 0; j < cols; ++j) {
                    double x{static_cast<double>(i) / 100.0};
                    double qa, qe;
                    CPPUNIT_ASSERT(expectedQuantiles[j].quantile(x, qe));
                    CPPUNIT_ASSERT(actualQuantiles[j].quantile(x, qa));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(qe, qa, 0.01 * std::max(std::fabs(qa), 1.5));
                    columnsMae[j].add(std::fabs(qa - qe));
                }
            }

            TMeanAccumulator mae;
            for (std::size_t i = 0; i < columnsMae.size(); ++i) {
                LOG_DEBUG(<< "Column MAE = " << maths::CBasicStatistics::mean(columnsMae[i]));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(columnsMae[i]) < 0.01);
                mae += columnsMae[i];
            }
            LOG_DEBUG(<< "MAE = " << maths::CBasicStatistics::mean(mae));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(mae) < 0.005);
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
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testColumnQuantiles",
        &CDataFrameUtilsTest::testColumnQuantiles));

    return suiteOfTests;
}
