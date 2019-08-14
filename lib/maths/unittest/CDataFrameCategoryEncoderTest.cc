/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameCategoryEncoderTest.h"

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameCategoryEncoder.h>

#include <test/CRandomNumbers.h>

#include <numeric>
#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

void CDataFrameCategoryEncoderTest::testOneHotEncoding() {

    // Test one-hot encoding of two categories carrying a lot of information
    // about the target.

    TDoubleVec categoryValue{-15.0, 20.0, 0.0};

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        return categoryValue[static_cast<std::size_t>(std::min(features[0][row], 2.0))] +
               2.8 * features[1][row] - 5.3 * features[2][row];
    };

    test::CRandomNumbers rng;

    core::stopDefaultAsyncExecutor();

    for (std::size_t threads : {1, 2}) {

        std::size_t rows{300};
        std::size_t cols{4};
        double numberCategories{5.0};

        TDoubleVecVec features(cols - 1);
        rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
        rng.generateNormalSamples(0.0, 4.0, rows, features[1]);
        rng.generateNormalSamples(2.0, 2.0, rows, features[2]);

        auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

        frame->categoricalColumns({true, false, false, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                *(column++) = std::floor(features[0][i]);
                for (std::size_t j = 1; j + 1 < cols; ++j, ++column) {
                    *column = features[j][i];
                }
                *column = target(features, i);
            });
        }
        frame->finishWritingRows();

        maths::CDataFrameCategoryEncoder encoder{threads, *frame, {0, 1, 2}, 3, 50};

        for (std::size_t i = 0; i < cols; ++i) {
            CPPUNIT_ASSERT_EQUAL(bool{frame->columnIsCategorical()[i]},
                                 encoder.columnIsCategorical(i));
        }

        CPPUNIT_ASSERT_EQUAL(
            std::string{"[1, 2]"},
            core::CContainerPrinter::print(encoder.selectedMetricFeatures()));

        TSizeVec expectedColumns{0, 0, 0, 0, 1, 2, 3};
        CPPUNIT_ASSERT_EQUAL(expectedColumns.size(), encoder.numberFeatures());
        for (std::size_t i = 0; i < expectedColumns.size(); ++i) {
            CPPUNIT_ASSERT_EQUAL(expectedColumns[i], encoder.column(i));
        }

        TSizeVecVec expectedOneHotEncodedCategories{{0, 1}, {}, {}, {}};
        for (std::size_t i = 0; i < cols; ++i) {
            CPPUNIT_ASSERT_EQUAL(expectedOneHotEncodedCategories[i].size(),
                                 encoder.numberOneHotEncodedCategories(i));
            if (encoder.columnIsCategorical(i)) {
                for (auto j : expectedOneHotEncodedCategories[i]) {
                    CPPUNIT_ASSERT_EQUAL(true, encoder.isOne(j, i, j));
                }
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CDataFrameCategoryEncoderTest::testMeanValueEncoding() {

    // Test common features are mean value encoded and that we get the right
    // mean target values.

    test::CRandomNumbers rng;

    TDoubleVec categoryValue{-15.0, 20.0, 0.0};
    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        std::size_t category{static_cast<std::size_t>(std::min(features[0][row], 2.0))};
        return categoryValue[category] + 1.5 * features[1][row] - 5.3 * features[2][row];
    };

    std::size_t rows{500};
    std::size_t cols{4};
    double numberCategories{10.0};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
    rng.generateNormalSamples(0.0, 4.0, rows, features[1]);
    rng.generateNormalSamples(2.0, 2.0, rows, features[2]);

    core::stopDefaultAsyncExecutor();

    for (std::size_t threads : {1, 2}) {

        auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

        TMeanAccumulatorVec expectedTargetMeanValues(static_cast<std::size_t>(numberCategories));

        frame->categoricalColumns({true, false, false, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                *(column++) = std::floor(features[0][i]);
                for (std::size_t j = 1; j + 1 < cols; ++j, ++column) {
                    *column = features[j][i];
                }
                *column = target(features, i);
                std::size_t category{static_cast<std::size_t>(features[0][i])};
                expectedTargetMeanValues[category].add(*column);
            });
        }
        frame->finishWritingRows();

        maths::CDataFrameCategoryEncoder encoder{threads, *frame, {0, 1, 2}, 3, 50};

        for (std::size_t i = 0; i < expectedTargetMeanValues.size(); ++i) {
            CPPUNIT_ASSERT_EQUAL(maths::CBasicStatistics::mean(expectedTargetMeanValues[i]),
                                 encoder.targetMeanValue(0, i));
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CDataFrameCategoryEncoderTest::testEncodingRare() {

    // Test we get the rare features we expect given the frequency threshold.

    test::CRandomNumbers rng;

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        return features[0][row] + 1.5 * features[1][row] - 5.3 * features[2][row];
    };

    std::size_t rows{1000};
    std::size_t cols{4};
    double numberCategories{25.0};

    TDoubleVecVec features(cols - 1);
    rng.generateNormalSamples(0.0, 4.0, rows, features[0]);
    rng.generateNormalSamples(2.0, 2.0, rows, features[1]);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[2]);

    auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

    TSizeVec categoryCounts(static_cast<std::size_t>(numberCategories), 0);

    frame->categoricalColumns({false, false, true, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j + 2 < cols; ++j, ++column) {
                *column = features[j][i];
            }
            *(column++) = std::floor(features[2][i]);
            *column = target(features, i);
            ++categoryCounts[static_cast<std::size_t>(features[2][i])];
        });
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{1, *frame, {0, 1, 2}, 3, 50, 0.1};

    CPPUNIT_ASSERT(encoder.hasRareCategories(2));
    for (std::size_t i = 0; i < categoryCounts.size(); ++i) {
        CPPUNIT_ASSERT_EQUAL(categoryCounts[i] < 50, encoder.isRareCategory(2, i));
    }
}

void CDataFrameCategoryEncoderTest::testCorrelatedFeatures() {

    // Test the case that if two fields are strongly correlated we will
    // tend to just select one or the other.

    test::CRandomNumbers rng;

    // Two correlated metrics + 4 independent metrics.
    {
        auto target = [&](const TDoubleVecVec& features, std::size_t row) {
            return 5.3 * features[0][row] + 4.8 * features[1][row] +
                   1.6 * features[2][row] - 1.6 * features[3][row] +
                   1.3 * features[4][row] - 5.5 * features[5][row];
        };

        std::size_t rows{100};
        std::size_t cols{7};

        TDoubleVecVec features(cols - 1);
        rng.generateNormalSamples(0.0, 4.0, rows, features[0]);
        rng.generateNormalSamples(0.0, 0.4, rows, features[1]);
        for (std::size_t i = 0; i < rows; ++i) {
            features[1][i] += features[0][i];
        }
        rng.discard(1000000);
        rng.generateNormalSamples(0.0, 4.0, rows, features[2]);
        rng.discard(1000000);
        rng.generateNormalSamples(0.0, 4.0, rows, features[3]);
        rng.discard(1000000);
        rng.generateNormalSamples(0.0, 4.0, rows, features[4]);
        rng.discard(1000000);
        rng.generateNormalSamples(0.0, 4.0, rows, features[5]);

        auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

        frame->categoricalColumns({false, false, false, false, false, false, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                    *column = features[j][i];
                }
                *column = target(features, i);
            });
        }
        frame->finishWritingRows();

        maths::CDataFrameCategoryEncoder encoder{1, *frame, {0, 1, 2, 3, 4, 5}, 6, 50};

        // Dispite both carrying a lot of information about the target nearly
        // the same information is carried by columns 0 and 1 so we should
        // choose columns 0 or 1 and column 5.

        CPPUNIT_ASSERT_EQUAL(
            std::string{"[1, 5]"},
            core::CContainerPrinter::print(encoder.selectedMetricFeatures()));
    }

    // Two correlated categorical fields.
    {
        auto target = [&](const TDoubleVecVec& features, std::size_t row) {
            return std::floor(features[0][row]) + std::floor(features[1][row]);
        };

        std::size_t rows{200};
        std::size_t cols{3};
        double numberCategories{3.0};

        TDoubleVecVec features(cols - 1);
        rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
        features[1] = features[0];

        auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

        frame->categoricalColumns({true, true, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                    *column = features[j][i];
                }
                *column = target(features, i);
            });
        }
        frame->finishWritingRows();

        maths::CDataFrameCategoryEncoder encoder{1, *frame, {0, 1}, 6, 50};

        CPPUNIT_ASSERT_EQUAL(
            std::string{"[0]"},
            core::CContainerPrinter::print(encoder.selectedCategoricalFeatures()));
        CPPUNIT_ASSERT_EQUAL(static_cast<std::size_t>(numberCategories),
                             encoder.numberOneHotEncodedCategories(0));
    }
}

void CDataFrameCategoryEncoderTest::testEncodedDataFrameRowRef() {

    // Test we get the feature vectors we expect after encoding.

    // The feature vector layout for each encoded category is as follows:
    // | one-hot | is rare | mean target value |

    TDoubleVec categoryValue[2]{{-15.0, 20.0, 0.0}, {10.0, -10.0, 0.0}};

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        std::size_t categories[]{
            static_cast<std::size_t>(std::min(features[0][row], 2.0)),
            static_cast<std::size_t>(std::min(features[3][row], 2.0))};
        return categoryValue[0][categories[0]] + categoryValue[1][categories[1]] +
               2.6 * features[1][row] - 5.3 * features[2][row];
    };

    test::CRandomNumbers rng;

    core::stopDefaultAsyncExecutor();

    std::size_t rows{500};
    std::size_t cols{5};
    double numberCategories{4.1};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
    rng.generateNormalSamples(0.0, 4.0, rows, features[1]);
    rng.generateNormalSamples(2.0, 2.0, rows, features[2]);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[3]);

    auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

    TMeanAccumulatorVec expectedTargetMeanValues[2]{
        TMeanAccumulatorVec(static_cast<std::size_t>(std::ceil(numberCategories))),
        TMeanAccumulatorVec(static_cast<std::size_t>(std::ceil(numberCategories)))};

    frame->categoricalColumns({true, false, false, true, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            *(column++) = std::floor(features[0][i]);
            for (std::size_t j = 1; j + 2 < cols; ++j, ++column) {
                *column = features[j][i];
            }
            *(column++) = std::floor(features[3][i]);
            *column = target(features, i);
            std::size_t categories[]{static_cast<std::size_t>(features[0][i]),
                                     static_cast<std::size_t>(features[3][i])};
            expectedTargetMeanValues[0][categories[0]].add(*column);
            expectedTargetMeanValues[1][categories[1]].add(*column);
        });
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{1, *frame, {0, 1, 2, 3}, 4, 50};

    auto expectedEncoded = [&](const core::CDataFrame::TRowRef& row, std::size_t i) {

        // We should have one-hot encoded categories 0 and 1 for each categorical
        // feature, category 4 should be rare and 2 and 3 should be mean target
        // encoded.

        std::size_t categories[]{static_cast<std::size_t>(row[0]),
                                 static_cast<std::size_t>(row[3])};

        if (i < 2) {
            return categories[0] == i ? 1.0 : 0.0; // one-hot
        }
        if (i == 2) {
            return categories[0] == 4 ? 1.0 : 0.0; // rare
        }
        if (i == 3) {
            return maths::CBasicStatistics::mean(
                expectedTargetMeanValues[0][categories[0]]); // mean target
        }
        if (i < 6) {
            return static_cast<double>(row[i - 3]); // metrics
        }
        if (i < 8) {
            return categories[1] == i - 6 ? 1.0 : 0.0; // one-hot
        }
        if (i == 8) {
            return categories[1] == 4 ? 1.0 : 0.0; // rare
        }
        if (i == 9) {
            return maths::CBasicStatistics::mean(
                expectedTargetMeanValues[1][categories[1]]); // mean target
        }
        return static_cast<double>(row[4]); // target
    };

    bool passed{true};

    frame->readRows(1, [&](core::CDataFrame::TRowItr beginRows, core::CDataFrame::TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            if (passed) {
                auto encoded = encoder.encode(*row);
                passed = passed && encoded.index() == row->index();
                passed = passed && encoded.numberColumns() == 11;
                for (std::size_t i = 0; i < encoded.numberColumns(); ++i) {
                    passed = passed &&
                             std::fabs(encoded[i] - expectedEncoded(*row, i)) < 1e-6;
                    if (passed == false) {
                        LOG_DEBUG(<< i << " " << encoded[i] << " "
                                  << expectedEncoded(*row, i));
                    }
                }
            }
        }
    });

    CPPUNIT_ASSERT(passed);
}

CppUnit::Test* CDataFrameCategoryEncoderTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameCategoryEncoderTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameCategoryEncoderTest>(
        "CDataFrameCategoryEncoderTest::testOneHotEncoding",
        &CDataFrameCategoryEncoderTest::testOneHotEncoding));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameCategoryEncoderTest>(
        "CDataFrameCategoryEncoderTest::testMeanValueEncoding",
        &CDataFrameCategoryEncoderTest::testMeanValueEncoding));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameCategoryEncoderTest>(
        "CDataFrameCategoryEncoderTest::testEncodingRare",
        &CDataFrameCategoryEncoderTest::testEncodingRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameCategoryEncoderTest>(
        "CDataFrameCategoryEncoderTest::testCorrelatedFeatures",
        &CDataFrameCategoryEncoderTest::testCorrelatedFeatures));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameCategoryEncoderTest>(
        "CDataFrameCategoryEncoderTest::testEncodedDataFrameRowRef",
        &CDataFrameCategoryEncoderTest::testEncodedDataFrameRowRef));

    return suiteOfTests;
}
