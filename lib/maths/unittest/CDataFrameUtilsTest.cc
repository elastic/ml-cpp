/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CPackedBitVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CMic.h>
#include <maths/COrderings.h>
#include <maths/CQuantileSketch.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <limits>
#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameUtilsTest)

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
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

core::CPackedBitVector generateRandomRowMask(test::CRandomNumbers& rng, std::size_t numberRows) {
    TSizeVec sampleCount;
    rng.generateUniformSamples(numberRows / 2, 3 * numberRows / 2, 1, sampleCount);

    TSizeVec sampledRows;
    rng.generateUniformSamples(0, numberRows, sampleCount[0], sampledRows);
    std::sort(sampledRows.begin(), sampledRows.end());
    sampledRows.erase(std::unique(sampledRows.begin(), sampledRows.end()),
                      sampledRows.end());

    core::CPackedBitVector rowMask;
    for (auto i : sampledRows) {
        rowMask.extend(false, i - rowMask.size());
        rowMask.extend(true);
    }
    rowMask.extend(false, numberRows - rowMask.size());
    return rowMask;
}
}

BOOST_AUTO_TEST_CASE(testColumnDataTypes) {

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
            BOOST_TEST_REQUIRE(core::CPersistUtils::fromString(
                delimitedCollection,
                [](const std::string& delimited, auto& type) {
                    return type.fromDelimited(delimited);
                },
                restoredTypes, maths::CDataFrameUtils::SDataType::EXTERNAL_DELIMITER));

            BOOST_REQUIRE_EQUAL(expectedTypes.size(), actualTypes.size());
            for (std::size_t i = 0; i < expectedTypes.size(); ++i) {
                double eps{100.0 * std::numeric_limits<double>::epsilon()};
                BOOST_REQUIRE_EQUAL(expectedTypes[i].s_IsInteger, actualTypes[i].s_IsInteger);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedTypes[i].s_Min,
                                             actualTypes[i].s_Min,
                                             eps * expectedTypes[i].s_Min);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedTypes[i].s_Max,
                                             actualTypes[i].s_Max,
                                             eps * expectedTypes[i].s_Max);
                BOOST_REQUIRE_EQUAL(expectedTypes[i].s_IsInteger,
                                    restoredTypes[i].s_IsInteger);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedTypes[i].s_Min,
                                             restoredTypes[i].s_Min,
                                             eps * expectedTypes[i].s_Min);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedTypes[i].s_Max,
                                             restoredTypes[i].s_Max,
                                             eps * expectedTypes[i].s_Max);
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testStandardizeColumns) {

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

            BOOST_TEST_REQUIRE(maths::CDataFrameUtils::standardizeColumns(threads, *frame));

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

            BOOST_TEST_REQUIRE(passed);

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
                BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, mean, 1e-6);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, variance, 1e-6);
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testColumnQuantiles) {

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

            TQuantileSketchVec actualQuantiles;
            bool successful;
            std::tie(actualQuantiles, successful) = maths::CDataFrameUtils::columnQuantiles(
                threads, *frame, maskAll(rows), columnMask,
                maths::CQuantileSketch{maths::CQuantileSketch::E_Linear, 100});
            BOOST_TEST_REQUIRE(successful);

            // Check the quantile sketches match.

            TMeanAccumulatorVec columnsMae(4);

            for (std::size_t i = 5; i < 100; i += 5) {
                for (std::size_t feature = 0; feature < columnMask.size(); ++feature) {
                    double x{static_cast<double>(i)};
                    double qa, qe;
                    BOOST_TEST_REQUIRE(expectedQuantiles[feature].quantile(x, qe));
                    BOOST_TEST_REQUIRE(actualQuantiles[feature].quantile(x, qa));
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        qe, qa, 0.02 * std::max(std::fabs(qa), 1.5));
                    columnsMae[feature].add(std::fabs(qa - qe));
                }
            }

            TMeanAccumulator mae;
            for (std::size_t i = 0; i < columnsMae.size(); ++i) {
                LOG_DEBUG(<< "Column MAE = "
                          << maths::CBasicStatistics::mean(columnsMae[i]));
                BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(columnsMae[i]) < 0.03);
                mae += columnsMae[i];
            }
            LOG_DEBUG(<< "MAE = " << maths::CBasicStatistics::mean(mae));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(mae) < 0.015);
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testColumnQuantilesWithEncoding) {

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
    bool successful;
    std::tie(actualQuantiles, successful) = maths::CDataFrameUtils::columnQuantiles(
        1, *frame, maskAll(rows), columnMask,
        maths::CQuantileSketch{maths::CQuantileSketch::E_Linear, 100}, &encoder);
    BOOST_TEST_REQUIRE(successful);

    for (std::size_t i = 5; i < 100; i += 5) {
        for (std::size_t feature = 0; feature < columnMask.size(); ++feature) {
            double x{static_cast<double>(i)};
            double qa, qe;
            BOOST_TEST_REQUIRE(expectedQuantiles[feature].quantile(x, qe));
            BOOST_TEST_REQUIRE(actualQuantiles[feature].quantile(x, qa));
            BOOST_REQUIRE_EQUAL(qe, qa);
        }
    }
}

BOOST_AUTO_TEST_CASE(testStratifiedCrossValidationRowMasks) {

    // Check some invariants of the test and train masks:
    //   1) The folds are approximately the same size,
    //   2) The test masks are disjoint for each fold,
    //   3) The train and test masks are disjoint for a given fold,
    //   4) They're all subsets of the initial mask supplied,
    //   5) The number of examples in each category per fold is proportional to
    //      their overall frequency.

    using TDoubleDoubleUMap = boost::unordered_map<double, double>;

    test::CRandomNumbers testRng;
    maths::CPRNG::CXorOShiro128Plus rng;

    std::size_t numberRows{2000};
    std::size_t numberCols{1};
    std::size_t numberBins{10};

    for (std::size_t trial = 0; trial < 10; ++trial) {

        TDoubleVec categories;
        testRng.generateNormalSamples(0.0, 3.0, numberRows, categories);
        TSizeVec numberFolds;
        testRng.generateUniformSamples(2, 6, 1, numberFolds);

        auto frame = core::makeMainStorageDataFrame(numberCols).first;
        frame->categoricalColumns(TBoolVec{true});
        for (std::size_t i = 0; i < numberRows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                *column = std::floor(std::fabs(categories[i]));
            });
        }
        frame->finishWritingRows();

        core::CPackedBitVector allTrainingRowsMask{generateRandomRowMask(testRng, numberRows)};

        TDoubleDoubleUMap categoryCounts;
        for (auto i = allTrainingRowsMask.beginOneBits();
             i != allTrainingRowsMask.endOneBits(); ++i) {
            categoryCounts[std::floor(std::fabs(categories[*i]))] += 1.0;
        }

        maths::CDataFrameUtils::TPackedBitVectorVec trainingRowMasks;
        maths::CDataFrameUtils::TPackedBitVectorVec testingRowMasks;
        std::tie(trainingRowMasks, testingRowMasks, std::ignore) =
            maths::CDataFrameUtils::stratifiedCrossValidationRowMasks(
                1, *frame, 0, rng, numberFolds[0], numberBins, allTrainingRowsMask);

        BOOST_REQUIRE_EQUAL(numberFolds[0], trainingRowMasks.size());
        BOOST_REQUIRE_EQUAL(numberFolds[0], testingRowMasks.size());

        core::CPackedBitVector allTestingRowsMask{numberRows, false};
        for (std::size_t fold = 0; fold < numberFolds[0]; ++fold) {
            // Count should be very nearly the expected value.
            double expectedTestRowCount{allTrainingRowsMask.manhattan() /
                                        static_cast<double>(numberFolds[0])};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedTestRowCount,
                                         testingRowMasks[fold].manhattan(), 10.0);
            BOOST_REQUIRE_EQUAL(0.0, testingRowMasks[fold].inner(allTestingRowsMask));
            BOOST_REQUIRE_EQUAL(0.0, trainingRowMasks[fold].inner(testingRowMasks[fold]));
            BOOST_REQUIRE_EQUAL(trainingRowMasks[fold].manhattan(),
                                trainingRowMasks[fold].inner(allTrainingRowsMask));
            BOOST_REQUIRE_EQUAL(testingRowMasks[fold].manhattan(),
                                testingRowMasks[fold].inner(allTrainingRowsMask));
            allTestingRowsMask |= testingRowMasks[fold];

            TDoubleDoubleUMap testingCategoryCounts;
            frame->readRows(1, 0, frame->numberRows(),
                            [&](core::CDataFrame::TRowItr beginRows,
                                core::CDataFrame::TRowItr endRows) {
                                for (auto row = beginRows; row != endRows; ++row) {
                                    testingCategoryCounts[(*row)[0]] += 1.0;
                                }
                            },
                            &testingRowMasks[fold]);
            for (const auto& count : categoryCounts) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    count.second / static_cast<double>(numberFolds[0]),
                    testingCategoryCounts[count.first], 5.0);
            }
        }
    }

    for (std::size_t trial = 0; trial < 10; ++trial) {

        TDoubleVec value;
        testRng.generateNormalSamples(0.0, 3.0, numberRows, value);
        TSizeVec numberFolds;
        testRng.generateUniformSamples(2, 6, 1, numberFolds);

        auto frame = core::makeMainStorageDataFrame(numberCols).first;
        frame->categoricalColumns(TBoolVec{false});
        for (std::size_t i = 0; i < numberRows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column,
                                std::int32_t&) { *column = value[i]; });
        }
        frame->finishWritingRows();

        core::CPackedBitVector allTrainingRowsMask{generateRandomRowMask(testRng, numberRows)};

        maths::CDataFrameUtils::TPackedBitVectorVec testingRowMasks;
        std::tie(std::ignore, testingRowMasks, std::ignore) =
            maths::CDataFrameUtils::stratifiedCrossValidationRowMasks(
                1, *frame, 0, rng, numberFolds[0], numberBins, allTrainingRowsMask);

        TDoubleVecVec targetDecile(numberFolds[0], TDoubleVec(numberBins));

        core::CPackedBitVector allTestingRowsMask{numberRows, false};
        for (std::size_t fold = 0; fold < numberFolds[0]; ++fold) {
            // Count should be very nearly the expected value.
            double expectedTestRowCount{allTrainingRowsMask.manhattan() /
                                        static_cast<double>(numberFolds[0])};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedTestRowCount,
                                         testingRowMasks[fold].manhattan(), 10.0);
            BOOST_REQUIRE_EQUAL(0.0, testingRowMasks[fold].inner(allTestingRowsMask));
            BOOST_REQUIRE_EQUAL(testingRowMasks[fold].manhattan(),
                                testingRowMasks[fold].inner(allTrainingRowsMask));
            allTestingRowsMask |= testingRowMasks[fold];

            TDoubleVec values;
            frame->readRows(1, 0, frame->numberRows(),
                            [&](core::CDataFrame::TRowItr beginRows,
                                core::CDataFrame::TRowItr endRows) {
                                for (auto row = beginRows; row != endRows; ++row) {
                                    values.push_back((*row)[0]);
                                }
                            },
                            &testingRowMasks[fold]);
            std::sort(values.begin(), values.end());
            for (std::size_t i = 1; i < numberBins; ++i) {
                targetDecile[fold][i] = values[(i * values.size()) / numberBins];
            }
        }

        for (std::size_t i = 1; i < numberBins; ++i) {
            TMeanVarAccumulator testTargetDecileMoments;
            for (std::size_t fold = 0; fold < numberFolds[0]; ++fold) {
                testTargetDecileMoments.add(targetDecile[fold][i]);
            }
            LOG_DEBUG(<< "variance in test set target percentile = "
                      << maths::CBasicStatistics::variance(testTargetDecileMoments));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(testTargetDecileMoments) < 0.02);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMicWithColumn) {

    // Test we get the exact MICe value when the number of rows is less than
    // the target sample size.

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
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expected),
                            core::CContainerPrinter::print(actual));
    }
}

BOOST_AUTO_TEST_CASE(testMicWithColumnWithMissing) {

    // Test we get the exact MICe value with missing values when the number
    // of rows is less than the target sample size.

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

    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        auto frame = factory();

        TDoubleVecVec rows;
        TSizeVec missing(4, 0);

        for (std::size_t i = 0; i < numberRows; ++i) {

            TDoubleVec row;
            rng.generateUniformSamples(-5.0, 5.0, 4, row);
            row[3] = 2.0 * row[0] - 1.5 * row[1] + 4.0 * row[2];
            for (std::size_t j = 0; j < row.size(); ++j) {
                TDoubleVec u01;
                rng.generateUniformSamples(0.0, 1.0, 1, u01);
                if (u01[0] < 0.01) {
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
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expected),
                            core::CContainerPrinter::print(actual));
    }
}

BOOST_AUTO_TEST_CASE(testCategoryFrequencies) {

    // Test we get the correct frequencies for each category.

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

            BOOST_REQUIRE_EQUAL(std::size_t{4}, actualFrequencies.size());
            for (std::size_t i : {0, 2}) {
                BOOST_REQUIRE_EQUAL(actualFrequencies.size(),
                                    expectedFrequencies.size());
                for (std::size_t j = 0; j < actualFrequencies[i].size(); ++j) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedFrequencies[i][j],
                                                 actualFrequencies[i][j],
                                                 1.0 / static_cast<double>(rows));
                }
            }
            for (std::size_t i : {1, 3}) {
                BOOST_TEST_REQUIRE(actualFrequencies[i].empty());
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testCategoryFrequenciesWithMissing) {

    // Test we get the correct frequencies for each category with missing values.

    std::size_t rows{5000};
    std::size_t cols{4};
    std::size_t capacity{500};
    double probabilityMissing{0.01};
    double missingStandardDeviation{
        std::sqrt(probabilityMissing * static_cast<double>(rows))};

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

    for (const auto& factory : {makeOnDisk, makeMainMemory}) {

        auto frame = factory();
        frame->categoricalColumns(TBoolVec{true, false, true, false});

        TDoubleVec u01;
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) mutable {
                for (std::size_t j = 0; j < cols; ++j, ++column) {
                    rng.generateUniformSamples(0.0, 1.0, 1, u01);
                    if (u01[0] < probabilityMissing) {
                        *column = core::CDataFrame::valueOfMissing();
                    } else {
                        *column = values[j][i];
                    }
                }
            });
        }
        frame->finishWritingRows();

        TDoubleVecVec actualFrequencies{maths::CDataFrameUtils::categoryFrequencies(
            1, *frame, maskAll(rows), {0, 1, 2, 3})};

        BOOST_REQUIRE_EQUAL(std::size_t{4}, actualFrequencies.size());
        for (std::size_t i : {0, 2}) {
            BOOST_REQUIRE_EQUAL(actualFrequencies.size(), expectedFrequencies.size());
            for (std::size_t j = 0; j < actualFrequencies[i].size(); ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    expectedFrequencies[i][j], actualFrequencies[i][j],
                    3.0 * missingStandardDeviation / static_cast<double>(rows));
            }
        }
        for (std::size_t i : {1, 3}) {
            BOOST_TEST_REQUIRE(actualFrequencies[i].empty());
        }
    }
}

BOOST_AUTO_TEST_CASE(testMeanValueOfTargetForCategories) {

    // Test we get the correct mean values for each category.

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

            BOOST_REQUIRE_EQUAL(std::size_t{4}, actualMeans.size());
            for (std::size_t i : {0, 2}) {
                BOOST_REQUIRE_EQUAL(actualMeans.size(), expectedMeans.size());
                for (std::size_t j = 0; j < actualMeans[i].size(); ++j) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::mean(expectedMeans[i][j]),
                        actualMeans[i][j],
                        static_cast<double>(std::numeric_limits<float>::epsilon()) *
                            maths::CBasicStatistics::mean(expectedMeans[i][j]));
                }
            }
            for (std::size_t i : {1, 3}) {
                BOOST_TEST_REQUIRE(actualMeans[i].empty());
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testMeanValueOfTargetForCategoriesWithMissing) {

    // Test that rows missing the target variable are ignored.

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
    TDoubleVec u01;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j + 1 < cols; ++j) {
            rng.generateUniformSamples(0.0, 1.0, 1, u01);
            if (u01[0] < 0.01) {
                values[j][i] = core::CDataFrame::valueOfMissing();
            }
        }
        rng.generateUniformSamples(0.0, 1.0, 1, u01);
        if (u01[0] < 0.9) {
            for (std::size_t j = 0; j + 1 < cols; ++j) {
                if (maths::CDataFrameUtils::isMissing(values[j][i]) == false) {
                    values[cols - 1][i] += values[j][i];
                }
            }
            for (std::size_t j = 0; j + 1 < cols; ++j) {
                if (maths::CDataFrameUtils::isMissing(values[j][i]) == false) {
                    expectedMeans[j][static_cast<std::size_t>(values[j][i])].add(
                        values[cols - 1][i]);
                }
            }
        } else {
            values[cols - 1][i] = core::CDataFrame::valueOfMissing();
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

    BOOST_REQUIRE_EQUAL(std::size_t{4}, actualMeans.size());
    for (std::size_t i : {0, 2}) {
        BOOST_REQUIRE_EQUAL(actualMeans.size(), expectedMeans.size());
        for (std::size_t j = 0; j < actualMeans[i].size(); ++j) {
            BOOST_REQUIRE_EQUAL(maths::CBasicStatistics::mean(expectedMeans[i][j]),
                                actualMeans[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testCategoryMicWithColumn) {

    // Test one uncorrelated and one uncorrelated categorical field MICe.

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

            BOOST_REQUIRE_EQUAL(std::size_t{4}, mics.size());
            for (const auto& mic : mics) {
                BOOST_TEST_REQUIRE(std::is_sorted(
                    mic.begin(), mic.end(), [](const auto& lhs, const auto& rhs) {
                        return maths::COrderings::lexicographical_compare(
                            -lhs.second, lhs.first, -rhs.second, rhs.first);
                    }));
            }
            for (std::size_t i : {0, 2}) {
                BOOST_REQUIRE_EQUAL(std::size_t{5}, mics[i].size());
            }
            for (std::size_t i : {1, 3}) {
                BOOST_TEST_REQUIRE(mics[i].empty());
            }

            BOOST_TEST_REQUIRE(mics[0][0].second < 0.05);
            BOOST_TEST_REQUIRE(mics[2][0].second > 0.50);

            // The expected order is a function of both the category frequency
            // and its order since the target value is order + noise so the
            // larger the order the smaller the noise, relatively.
            TSizeVec categoryOrder;
            for (const auto& category : mics[2]) {
                categoryOrder.push_back(category.first);
            }
            BOOST_REQUIRE_EQUAL(std::string{"[1, 3, 0, 4, 2]"},
                                core::CContainerPrinter::print(categoryOrder));
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testCategoryMicWithColumnWithMissing) {

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
    TDoubleVec u01;
    for (std::size_t i = 0; i < rows; ++i) {
        values[cols - 1][i] += 2.0 * values[2][i];
        for (std::size_t j = 0; j < cols - 1; ++j) {
            rng.generateUniformSamples(0.0, 1.0, 1, u01);
            if (u01[0] < 0.01) {
                values[j][i] = core::CDataFrame::valueOfMissing();
            }
        }
    }

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

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
            maths::CDataFrameUtils::CMetricColumnValue{3}, 1, *frame,
            maskAll(rows), {0, 1, 2},
            {{[](std::size_t, std::size_t sampleColumn, std::size_t category) {
                  return std::make_unique<maths::CDataFrameUtils::COneHotCategoricalColumnValue>(
                      sampleColumn, category);
              },
              0.01}})[0];

        LOG_DEBUG(<< "mics[0] = " << core::CContainerPrinter::print(mics[0]));
        LOG_DEBUG(<< "mics[2] = " << core::CContainerPrinter::print(mics[2]));

        BOOST_REQUIRE_EQUAL(std::size_t{4}, mics.size());
        for (const auto& mic : mics) {
            BOOST_TEST_REQUIRE(std::is_sorted(
                mic.begin(), mic.end(), [](const auto& lhs, const auto& rhs) {
                    return maths::COrderings::lexicographical_compare(
                        -lhs.second, lhs.first, -rhs.second, rhs.first);
                }));
        }
        for (std::size_t i : {0, 2}) {
            BOOST_REQUIRE_EQUAL(std::size_t{5}, mics[i].size());
        }
        for (std::size_t i : {1, 3}) {
            BOOST_TEST_REQUIRE(mics[i].empty());
        }

        BOOST_TEST_REQUIRE(mics[0][0].second < 0.04);
        BOOST_TEST_REQUIRE(mics[2][0].second > 0.49);

        // The expected order is a function of both the category frequency
        // and its order since the target value is order + noise so the
        // larger the order the smaller the noise, relatively.
        TSizeVec categoryOrder;
        for (const auto& category : mics[2]) {
            categoryOrder.push_back(category.first);
        }
        BOOST_REQUIRE_EQUAL(std::string{"[1, 3, 0, 4, 2]"},
                            core::CContainerPrinter::print(categoryOrder));
    }
}

BOOST_AUTO_TEST_CASE(testMaximumMinimumRecallClassWeights) {

    // Test we reliably increase the minimum class recall for predictions with uneven accuracy.

    using TDoubleVector = maths::CDenseVector<double>;
    using TMemoryMappedFloatVector = maths::CMemoryMappedDenseVector<maths::CFloatStorage>;

    std::size_t rows{5000};
    std::size_t capacity{2000};

    test::CRandomNumbers rng;

    for (std::size_t numberClasses : {2, 3}) {

        std::size_t cols{numberClasses + 1};

        auto readPrediction = [&](const core::CDataFrame::TRowRef& row) {
            return TMemoryMappedFloatVector{row.data(), static_cast<int>(numberClasses)};
        };

        TBoolVec categoricalColumns(cols, false);
        categoricalColumns[numberClasses] = true;

        for (std::size_t t = 0; t < 5; ++t) {
            core::stopDefaultAsyncExecutor();

            TDoubleVec predictions;
            TSizeVec category;
            auto frame = core::makeMainStorageDataFrame(cols, capacity).first;
            frame->categoricalColumns(categoricalColumns);
            for (std::size_t i = 0; i < rows; ++i) {
                frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                    rng.generateUniformSamples(0, numberClasses, 1, category);
                    rng.generateNormalSamples(0.0, 1.0, numberClasses, predictions);
                    for (std::size_t j = 0; j < numberClasses; ++j) {
                        column[j] += predictions[j];
                    }
                    column[category[0]] += static_cast<double>(category[0] + 1);
                    column[numberClasses] = static_cast<double>(category[0]);
                });
            }
            frame->finishWritingRows();

            TDoubleVecVec minRecalls(2, TDoubleVec(2));
            TDoubleVecVec maxRecalls(2, TDoubleVec(2));

            std::size_t i{0};
            for (auto numberThreads : {1, 4}) {

                auto weights = maths::CDataFrameUtils::maximumMinimumRecallClassWeights(
                    numberThreads, *frame, maskAll(rows), numberClasses,
                    numberClasses, readPrediction);

                TDoubleVector prediction;
                TDoubleVector correct[2]{TDoubleVector::Zero(numberClasses),
                                         TDoubleVector::Zero(numberClasses)};
                TDoubleVector counts{TDoubleVector::Zero(numberClasses)};

                frame->readRows(1, [&](core::CDataFrame::TRowItr beginRows,
                                       core::CDataFrame::TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        prediction = readPrediction(*row);
                        maths::CTools::inplaceSoftmax(prediction);
                        std::size_t weightedPredictedClass;
                        weights.cwiseProduct(prediction).maxCoeff(&weightedPredictedClass);
                        std::size_t actualClass{
                            static_cast<std::size_t>((*row)[numberClasses])};
                        if (weightedPredictedClass == actualClass) {
                            correct[0](actualClass) += 1.0;
                        }
                        std::size_t unweightedPredictedClass;
                        prediction.maxCoeff(&unweightedPredictedClass);
                        if (unweightedPredictedClass == actualClass) {
                            correct[1](actualClass) += 1.0;
                        }
                        counts(actualClass) += 1.0;
                    }
                });

                LOG_TRACE(<< "weighted class recalls = "
                          << correct[0].cwiseQuotient(counts).transpose());
                LOG_TRACE(<< "unweighted class recalls = "
                          << correct[1].cwiseQuotient(counts).transpose());

                minRecalls[i][0] = correct[0].cwiseQuotient(counts).minCoeff();
                maxRecalls[i][0] = correct[0].cwiseQuotient(counts).maxCoeff();
                minRecalls[i][1] = correct[1].cwiseQuotient(counts).minCoeff();
                maxRecalls[i][1] = correct[1].cwiseQuotient(counts).maxCoeff();

                ++i;
                core::startDefaultAsyncExecutor();
            }

            LOG_DEBUG(<< "min recalls = " << core::CContainerPrinter::print(minRecalls));
            LOG_DEBUG(<< "max recalls = " << core::CContainerPrinter::print(maxRecalls));

            // Threaded and non-threaded results are close.
            BOOST_REQUIRE_CLOSE(minRecalls[0][0], minRecalls[1][0], 1.0); // 1 %

            // We improved the minimum class recall by at least 10%.
            BOOST_TEST_REQUIRE(minRecalls[0][0] > 1.1 * minRecalls[0][1]);

            // The minimum and maximum class recalls are close: we're at the global maximum.
            BOOST_TEST_REQUIRE(1.06 * minRecalls[0][0] > maxRecalls[0][0]);
            BOOST_TEST_REQUIRE(1.06 * minRecalls[1][0] > maxRecalls[1][0]);
        }
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_SUITE_END()
