/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameUtilsTest.h"

#include <core/CPackedBitVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CMic.h>
#include <maths/COrderings.h>
#include <maths/CQuantileSketch.h>

#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <functional>
#include <limits>
#include <numeric>
#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;
using TQuantileSketchVec = std::vector<maths::CQuantileSketch>;

auto generateCategoricalData(test::CRandomNumbers& rng,
                             std::size_t rows,
                             std::size_t cols,
                             TDoubleVec expectedFrequencies) {

    TDoubleVecVec frequencies;
    rng.generateDirichletSamples(expectedFrequencies, cols, frequencies);

    TDoubleVecVec values(cols);
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        for (std::size_t j = 0; j < frequencies[i].size(); ++j) {
            std::size_t target{static_cast<std::size_t>(
                static_cast<double>(rows) * frequencies[i][j] + 0.5)};
            values[i].resize(values[i].size() + target, static_cast<double>(j));
        }
        values[i].resize(rows, values[i].back());
        rng.random_shuffle(values[i].begin(), values[i].end());
        rng.discard(1000000); // Make sure the categories are not correlated
    }

    return std::make_pair(frequencies, values);
}

core::CPackedBitVector maskAll(std::size_t rows) {
    return {rows, true};
}
}

void CDataFrameUtilsTest::testColumnDataTypes() {

    test::CRandomNumbers rng;

    std::size_t rows{2000};
    std::size_t cols{4};

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols).first; }};

    TSizeVec columnMask(cols);
    std::iota(columnMask.begin(), columnMask.end(), 0);

    core::stopDefaultAsyncExecutor();

    for (auto threads : {1, 2}) {
        for (const auto& factory : {makeOnDisk, makeMainMemory}) {

            auto frame = factory();

            double min{0.0};
            double max{10.0};
            maths::CDataFrameUtils::TDataTypeVec expectedTypes{
                {true, max, min}, {false, max, min}, {false, max, min}, {false, max, min}};

            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                    TDoubleVec values;
                    rng.generateUniformSamples(min, max, cols, values);
                    *(column++) = std::floor(values[0]);
                    expectedTypes[0].s_Min =
                        std::min(expectedTypes[0].s_Min, std::floor(values[0]));
                    expectedTypes[0].s_Max =
                        std::max(expectedTypes[0].s_Max, std::floor(values[0]));
                    for (std::size_t j = 1; j < cols; ++j, ++column) {
                        *column = values[j];
                        expectedTypes[j].s_Min =
                            std::min(maths::CFloatStorage{expectedTypes[j].s_Min},
                                     maths::CFloatStorage{values[j]});
                        expectedTypes[j].s_Max =
                            std::max(maths::CFloatStorage{expectedTypes[j].s_Max},
                                     maths::CFloatStorage{values[j]});
                    }
                });
            }
            frame->finishWritingRows();

            maths::CDataFrameUtils::TDataTypeVec actualTypes(maths::CDataFrameUtils::columnDataTypes(
                threads, *frame, maskAll(rows), columnMask));

            // Round trip the expected types to a string to check persistence.

            maths::CDataFrameUtils::TDataTypeVec restoredTypes;
            std::string delimitedCollection{core::CPersistUtils::toString(
                expectedTypes,
                [](const auto& type) { return type.toDelimited(); },
                maths::CDataFrameUtils::SDataType::EXTERNAL_DELIMITER)};
            LOG_DEBUG(<< "delimited = " << delimitedCollection);
            CPPUNIT_ASSERT(core::CPersistUtils::fromString(
                delimitedCollection,
                [](const std::string& delimited, auto& type) {
                    return type.fromDelimited(delimited);
                },
                restoredTypes, maths::CDataFrameUtils::SDataType::EXTERNAL_DELIMITER));

            CPPUNIT_ASSERT_EQUAL(expectedTypes.size(), actualTypes.size());
            for (std::size_t i = 0; i < expectedTypes.size(); ++i) {
                double eps{100.0 * std::numeric_limits<double>::epsilon()};
                CPPUNIT_ASSERT_EQUAL(expectedTypes[i].s_IsInteger,
                                     actualTypes[i].s_IsInteger);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedTypes[i].s_Min,
                                             actualTypes[i].s_Min,
                                             eps * expectedTypes[i].s_Min);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedTypes[i].s_Max,
                                             actualTypes[i].s_Max,
                                             eps * expectedTypes[i].s_Max);
                CPPUNIT_ASSERT_EQUAL(expectedTypes[i].s_IsInteger,
                                     restoredTypes[i].s_IsInteger);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedTypes[i].s_Min,
                                             restoredTypes[i].s_Min,
                                             eps * expectedTypes[i].s_Min);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedTypes[i].s_Max,
                                             restoredTypes[i].s_Max,
                                             eps * expectedTypes[i].s_Max);
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

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
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
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
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    TSizeVec columnMask(cols);
    std::iota(columnMask.begin(), columnMask.end(), 0);

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

            maths::CQuantileSketch sketch{maths::CQuantileSketch::E_Linear, 100};
            TQuantileSketchVec actualQuantiles;
            CPPUNIT_ASSERT(maths::CDataFrameUtils::columnQuantiles(
                threads, *frame, maskAll(rows), columnMask, sketch, actualQuantiles));

            // Check the quantile sketches match.

            TMeanAccumulatorVec columnsMae(4);

            for (std::size_t i = 5; i < 100; i += 5) {
                for (std::size_t feature = 0; feature < columnMask.size(); ++feature) {
                    double x{static_cast<double>(i)};
                    double qa, qe;
                    CPPUNIT_ASSERT(expectedQuantiles[feature].quantile(x, qe));
                    CPPUNIT_ASSERT(actualQuantiles[feature].quantile(x, qa));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        qe, qa, 0.02 * std::max(std::fabs(qa), 1.5));
                    columnsMae[feature].add(std::fabs(qa - qe));
                }
            }

            TMeanAccumulator mae;
            for (std::size_t i = 0; i < columnsMae.size(); ++i) {
                LOG_DEBUG(<< "Column MAE = "
                          << maths::CBasicStatistics::mean(columnsMae[i]));
                CPPUNIT_ASSERT(maths::CBasicStatistics::mean(columnsMae[i]) < 0.03);
                mae += columnsMae[i];
            }
            LOG_DEBUG(<< "MAE = " << maths::CBasicStatistics::mean(mae));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(mae) < 0.015);
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CDataFrameUtilsTest::testColumnQuantilesWithEncoding() {

    test::CRandomNumbers rng;

    std::size_t rows{5000};
    std::size_t cols{6};
    std::size_t capacity{500};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.96, 5.01, rows, features[0]);
    std::for_each(features[0].begin(), features[0].end(),
                  [](double& category) { category = std::floor(category); });
    for (std::size_t i = 1; i + 1 < features.size(); ++i) {
        rng.generateNormalSamples(0.0, 9.0, rows, features[i]);
    }
    rng.generateUniformSamples(0.97, 5.03, rows, features[cols - 2]);
    std::for_each(features[cols - 2].begin(), features[cols - 2].end(),
                  [](double& category) { category = std::floor(category); });

    TDoubleVec weights;
    rng.generateUniformSamples(1.0, 10.0, cols - 1, weights);
    auto target = [&weights](const TDoubleVec& rowFeatures) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * rowFeatures[i];
        }
        return result;
    };

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    frame->categoricalColumns(TBoolVec{false, true, false, false, false, true});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&features, target, i, rowFeatures = TDoubleVec{} ](
            core::CDataFrame::TFloatVecItr column, std::int32_t&) mutable {
            rowFeatures.resize(features.size());
            for (std::size_t j = 0; j < features.size(); ++j) {
                rowFeatures[j] = features[j][i];
            }
            *column++ = target(rowFeatures);
            for (std::size_t j = 0; j < rowFeatures.size(); ++j, ++column) {
                *column = rowFeatures[j];
            }
        });
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{{1, *frame, 0}};

    TSizeVec columnMask(encoder.numberEncodedColumns());
    std::iota(columnMask.begin(), columnMask.end(), 0);

    TQuantileSketchVec expectedQuantiles{columnMask.size(),
                                         {maths::CQuantileSketch::E_Linear, 100}};
    frame->readRows(1, [&](core::CDataFrame::TRowItr beginRows, core::CDataFrame::TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            maths::CEncodedDataFrameRowRef encodedRow{encoder.encode(*row)};
            for (std::size_t i = 0; i < columnMask.size(); ++i) {
                expectedQuantiles[i].add(encodedRow[columnMask[i]]);
            }
        }
    });

    TQuantileSketchVec actualQuantiles;
    maths::CQuantileSketch sketch{maths::CQuantileSketch::E_Linear, 100};
    CPPUNIT_ASSERT(maths::CDataFrameUtils::columnQuantiles(
        1, *frame, maskAll(rows), columnMask, sketch, actualQuantiles, &encoder));

    for (std::size_t i = 5; i < 100; i += 5) {
        for (std::size_t feature = 0; feature < columnMask.size(); ++feature) {
            double x{static_cast<double>(i)};
            double qa, qe;
            CPPUNIT_ASSERT(expectedQuantiles[feature].quantile(x, qe));
            CPPUNIT_ASSERT(actualQuantiles[feature].quantile(x, qa));
            CPPUNIT_ASSERT_EQUAL(qe, qa);
        }
    }
}

void CDataFrameUtilsTest::testMicWithColumn() {

    test::CRandomNumbers rng;

    std::size_t capacity{500};
    std::size_t numberRows{2000};
    std::size_t numberCols{4};

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(),
                                              numberCols, numberRows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{[=] {
        return core::makeMainStorageDataFrame(numberCols, capacity).first;
    }};

    // Test we get exactly the value we expect when the number of rows is less
    // than the target sample size.

    for (const auto& factory : {makeOnDisk, makeMainMemory}) {

        auto frame = factory();

        TDoubleVecVec rows;

        for (std::size_t i = 0; i < numberRows; ++i) {

            TDoubleVec row;
            rng.generateUniformSamples(-5.0, 5.0, 4, row);
            row[3] = 2.0 * row[0] - 1.5 * row[1] + 4.0 * row[2];
            rows.push_back(row);

            frame->writeRow([&row, numberCols](core::CDataFrame::TFloatVecItr column,
                                               std::int32_t&) {
                for (std::size_t j = 0; j < numberCols; ++j, ++column) {
                    *column = row[j];
                }
            });
        }
        frame->finishWritingRows();

        TDoubleVec expected(4, 0.0);
        for (std::size_t j : {0, 1, 2}) {
            maths::CMic mic;
            for (const auto& row : rows) {
                mic.add(row[j], row[3]);
            }
            expected[j] = mic.compute();
        }

        TDoubleVec actual(maths::CDataFrameUtils::metricMicWithColumn(
            maths::CDataFrameUtils::CMetricColumnValue{3}, *frame,
            maskAll(numberRows), {0, 1, 2}));

        LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expected));
        LOG_DEBUG(<< "actual   = " << core::CContainerPrinter::print(actual));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected),
                             core::CContainerPrinter::print(actual));
    }

    // Test missing values.

    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        auto frame = factory();

        TDoubleVecVec rows;
        TSizeVec missing(4, 0);

        for (std::size_t i = 0; i < numberRows; ++i) {

            TDoubleVec row;
            rng.generateUniformSamples(-5.0, 5.0, 4, row);
            row[3] = 2.0 * row[0] - 1.5 * row[1] + 4.0 * row[2];
            for (std::size_t j = 0; j < row.size(); ++j) {
                TDoubleVec p;
                rng.generateUniformSamples(0.0, 1.0, 1, p);
                if (p[0] < 0.01) {
                    row[j] = core::CDataFrame::valueOfMissing();
                    ++missing[j];
                }
            }
            rows.push_back(row);

            frame->writeRow([&row, numberCols](core::CDataFrame::TFloatVecItr column,
                                               std::int32_t&) {
                for (std::size_t j = 0; j < numberCols; ++j, ++column) {
                    *column = row[j];
                }
            });
        }
        frame->finishWritingRows();

        TDoubleVec expected(4, 0.0);
        for (std::size_t j : {0, 1, 2}) {
            maths::CMic mic;
            for (const auto& row : rows) {
                if (maths::CDataFrameUtils::isMissing(row[j]) == false &&
                    maths::CDataFrameUtils::isMissing(row[3]) == false) {
                    mic.add(row[j], row[3]);
                }
            }
            expected[j] = (1.0 - static_cast<double>(missing[j]) /
                                     static_cast<double>(rows.size())) *
                          mic.compute();
        }

        TDoubleVec actual(maths::CDataFrameUtils::metricMicWithColumn(
            maths::CDataFrameUtils::CMetricColumnValue{3}, *frame,
            maskAll(numberRows), {0, 1, 2}));

        LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expected));
        LOG_DEBUG(<< "actual   = " << core::CContainerPrinter::print(actual));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected),
                             core::CContainerPrinter::print(actual));
    }
}

void CDataFrameUtilsTest::testCategoryFrequencies() {

    std::size_t rows{5000};
    std::size_t cols{4};
    std::size_t capacity{500};

    test::CRandomNumbers rng;

    TDoubleVecVec expectedFrequencies;
    TDoubleVecVec values;
    std::tie(expectedFrequencies, values) = generateCategoricalData(
        rng, rows, cols, {10.0, 30.0, 1.0, 5.0, 15.0, 9.0, 20.0, 10.0});

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    core::stopDefaultAsyncExecutor();

    for (auto threads : {1, 4}) {

        for (const auto& factory : {makeOnDisk, makeMainMemory}) {

            auto frame = factory();

            frame->categoricalColumns(TBoolVec{true, false, true, false});
            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&values, i, cols](core::CDataFrame::TFloatVecItr column,
                                                   std::int32_t&) {
                    for (std::size_t j = 0; j < cols; ++j, ++column) {
                        *column = values[j][i];
                    }
                });
            }
            frame->finishWritingRows();

            TDoubleVecVec actualFrequencies{maths::CDataFrameUtils::categoryFrequencies(
                threads, *frame, maskAll(rows), {0, 1, 2, 3})};

            CPPUNIT_ASSERT_EQUAL(std::size_t{4}, actualFrequencies.size());
            for (std::size_t i : {0, 2}) {
                CPPUNIT_ASSERT_EQUAL(actualFrequencies.size(),
                                     expectedFrequencies.size());
                for (std::size_t j = 0; j < actualFrequencies[i].size(); ++j) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFrequencies[i][j],
                                                 actualFrequencies[i][j],
                                                 1.0 / static_cast<double>(rows));
                }
            }
            for (std::size_t i : {1, 3}) {
                CPPUNIT_ASSERT(actualFrequencies[i].empty());
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CDataFrameUtilsTest::testMeanValueOfTargetForCategories() {

    std::size_t rows{2000};
    std::size_t cols{4};
    std::size_t capacity{500};

    test::CRandomNumbers rng;

    TDoubleVecVec frequencies;
    TDoubleVecVec values;
    std::tie(frequencies, values) = generateCategoricalData(
        rng, rows, cols - 1, {10.0, 30.0, 1.0, 5.0, 15.0, 9.0, 20.0, 10.0});

    values.resize(cols);
    values[cols - 1].resize(rows, 0.0);
    TMeanAccumulatorVecVec expectedMeans(cols, TMeanAccumulatorVec(8));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j + 1 < cols; ++j) {
            values[cols - 1][i] += values[j][i];
        }
        for (std::size_t j = 0; j + 1 < cols; ++j) {
            expectedMeans[j][static_cast<std::size_t>(values[j][i])].add(
                values[cols - 1][i]);
        }
    }

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    core::stopDefaultAsyncExecutor();

    for (auto threads : {1, 4}) {

        for (const auto& factory : {makeOnDisk, makeMainMemory}) {

            auto frame = factory();

            frame->categoricalColumns(TBoolVec{true, false, true, false});
            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&values, i, cols](core::CDataFrame::TFloatVecItr column,
                                                   std::int32_t&) {
                    for (std::size_t j = 0; j < cols; ++j, ++column) {
                        *column = values[j][i];
                    }
                });
            }
            frame->finishWritingRows();

            TDoubleVecVec actualMeans(maths::CDataFrameUtils::meanValueOfTargetForCategories(
                maths::CDataFrameUtils::CMetricColumnValue{3}, threads, *frame,
                maskAll(rows), {0, 1, 2}));

            CPPUNIT_ASSERT_EQUAL(std::size_t{4}, actualMeans.size());
            for (std::size_t i : {0, 2}) {
                CPPUNIT_ASSERT_EQUAL(actualMeans.size(), expectedMeans.size());
                for (std::size_t j = 0; j < actualMeans[i].size(); ++j) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        maths::CBasicStatistics::mean(expectedMeans[i][j]),
                        actualMeans[i][j],
                        static_cast<double>(std::numeric_limits<float>::epsilon()) *
                            maths::CBasicStatistics::mean(expectedMeans[i][j]));
                }
            }
            for (std::size_t i : {1, 3}) {
                CPPUNIT_ASSERT(actualMeans[i].empty());
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CDataFrameUtilsTest::testMeanValueOfTargetForCategoriesWithMissing() {

    // Test that rows missing the target variable are ignored.

    std::size_t rows{2000};
    std::size_t cols{4};
    std::size_t capacity{500};

    test::CRandomNumbers rng;

    TDoubleVecVec frequencies;
    TDoubleVecVec values;
    std::tie(frequencies, values) = generateCategoricalData(
        rng, rows, cols - 1, {10.0, 30.0, 1.0, 5.0, 15.0, 9.0, 20.0, 10.0});

    TDoubleVec uniform01;
    rng.generateUniformSamples(0.0, 1.0, rows, uniform01);

    values.resize(cols);
    values[cols - 1].resize(rows, 0.0);
    TMeanAccumulatorVecVec expectedMeans(cols, TMeanAccumulatorVec(8));
    for (std::size_t i = 0; i < rows; ++i) {
        if (uniform01[i] < 0.9) {
            for (std::size_t j = 0; j + 1 < cols; ++j) {
                values[cols - 1][i] += values[j][i];
            }
            for (std::size_t j = 0; j + 1 < cols; ++j) {
                expectedMeans[j][static_cast<std::size_t>(values[j][i])].add(
                    values[cols - 1][i]);
            }
        } else {
            values[cols - 1][i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    frame->categoricalColumns(TBoolVec{true, false, true, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&values, i, cols](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j < cols; ++j, ++column) {
                *column = values[j][i];
            }
        });
    }
    frame->finishWritingRows();

    TDoubleVecVec actualMeans(maths::CDataFrameUtils::meanValueOfTargetForCategories(
        maths::CDataFrameUtils::CMetricColumnValue{3}, 1, *frame,
        core::CPackedBitVector{rows, true}, {0, 1, 2}));

    CPPUNIT_ASSERT_EQUAL(std::size_t{4}, actualMeans.size());
    for (std::size_t i : {0, 2}) {
        CPPUNIT_ASSERT_EQUAL(actualMeans.size(), expectedMeans.size());
        for (std::size_t j = 0; j < actualMeans[i].size(); ++j) {
            CPPUNIT_ASSERT_EQUAL(maths::CBasicStatistics::mean(expectedMeans[i][j]),
                                 actualMeans[i][j]);
        }
    }
}

void CDataFrameUtilsTest::testCategoryMicWithColumn() {

    std::size_t rows{5000};
    std::size_t cols{4};
    std::size_t capacity{2000};

    test::CRandomNumbers rng;

    TDoubleVecVec frequencies;
    TDoubleVecVec values;
    std::tie(frequencies, values) =
        generateCategoricalData(rng, rows, cols - 1, {20.0, 60.0, 5.0, 15.0, 1.0});

    values.resize(cols);
    rng.generateNormalSamples(0.0, 1.0, rows, values[cols - 1]);
    for (std::size_t i = 0; i < rows; ++i) {
        values[cols - 1][i] += 2.0 * values[2][i];
    }

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    core::stopDefaultAsyncExecutor();

    for (auto threads : {1, 4}) {

        for (const auto& factory : {makeOnDisk, makeMainMemory}) {

            auto frame = factory();

            frame->categoricalColumns(TBoolVec{true, false, true, false});
            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&values, i, cols](core::CDataFrame::TFloatVecItr column,
                                                   std::int32_t&) {
                    for (std::size_t j = 0; j < cols; ++j, ++column) {
                        *column = values[j][i];
                    }
                });
            }
            frame->finishWritingRows();

            auto mics = maths::CDataFrameUtils::categoricalMicWithColumn(
                maths::CDataFrameUtils::CMetricColumnValue{3}, threads, *frame,
                maskAll(rows), {0, 1, 2},
                {{[](std::size_t, std::size_t sampleColumn, std::size_t category) {
                      return std::make_unique<maths::CDataFrameUtils::COneHotCategoricalColumnValue>(
                          sampleColumn, category);
                  },
                  0.01}})[0];

            LOG_DEBUG(<< "mics[0] = " << core::CContainerPrinter::print(mics[0]));
            LOG_DEBUG(<< "mics[2] = " << core::CContainerPrinter::print(mics[2]));

            CPPUNIT_ASSERT_EQUAL(std::size_t{4}, mics.size());
            for (const auto& mic : mics) {
                CPPUNIT_ASSERT(std::is_sorted(
                    mic.begin(), mic.end(), [](const auto& lhs, const auto& rhs) {
                        return maths::COrderings::lexicographical_compare(
                            -lhs.second, lhs.first, -rhs.second, rhs.first);
                    }));
            }
            for (std::size_t i : {0, 2}) {
                CPPUNIT_ASSERT_EQUAL(std::size_t{5}, mics[i].size());
            }
            for (std::size_t i : {1, 3}) {
                CPPUNIT_ASSERT(mics[i].empty());
            }

            CPPUNIT_ASSERT(mics[0][0].second < 0.05);
            CPPUNIT_ASSERT(mics[2][0].second > 0.50);

            TSizeVec categoryOrder;
            for (const auto& category : mics[2]) {
                categoryOrder.push_back(category.first);
            }
            CPPUNIT_ASSERT_EQUAL(std::string{"[1, 3, 0, 4, 2]"},
                                 core::CContainerPrinter::print(categoryOrder));
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

CppUnit::Test* CDataFrameUtilsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameUtilsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testColumnDataTypes", &CDataFrameUtilsTest::testColumnDataTypes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testStandardizeColumns",
        &CDataFrameUtilsTest::testStandardizeColumns));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testColumnQuantiles", &CDataFrameUtilsTest::testColumnQuantiles));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testColumnQuantilesWithEncoding",
        &CDataFrameUtilsTest::testColumnQuantilesWithEncoding));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testMicWithColumn", &CDataFrameUtilsTest::testMicWithColumn));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testCategoryFrequencies",
        &CDataFrameUtilsTest::testCategoryFrequencies));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testMeanValueOfTargetForCategories",
        &CDataFrameUtilsTest::testMeanValueOfTargetForCategories));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testMeanValueOfTargetForCategoriesWithMissing",
        &CDataFrameUtilsTest::testMeanValueOfTargetForCategoriesWithMissing));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameUtilsTest>(
        "CDataFrameUtilsTest::testCategoryMicWithColumn",
        &CDataFrameUtilsTest::testCategoryMicWithColumn));

    return suiteOfTests;
}
