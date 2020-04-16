/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CPackedBitVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameCategoryEncoder.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <numeric>
#include <sstream>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameCategoryEncoderTest)

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TFloatVec =
    std::vector<maths::CFloatStorage, core::CAlignedAllocator<maths::CFloatStorage>>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;
}

BOOST_AUTO_TEST_CASE(testOneHotEncoding) {

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

        frame->categoricalColumns(TBoolVec{true, false, false, false});
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

        maths::CDataFrameCategoryEncoder encoder{{threads, *frame, 3}};

        for (std::size_t i = 0; i < encoder.numberEncodedColumns(); ++i) {
            switch (i) {
            case 0:
                BOOST_TEST_REQUIRE(maths::E_OneHot == encoder.encoding(i).type());
                BOOST_REQUIRE_EQUAL(1.0, encoder.encoding(i).encode(0.0));
                break;
            case 1:
                BOOST_TEST_REQUIRE(maths::E_OneHot == encoder.encoding(i).type());
                BOOST_REQUIRE_EQUAL(1.0, encoder.encoding(i).encode(1.0));
                break;
            default:
                BOOST_TEST_REQUIRE(maths::E_OneHot != encoder.encoding(i).type());
                break;
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testMeanValueEncoding) {

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

        frame->categoricalColumns(TBoolVec{true, false, false, false});
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

        maths::CMakeDataFrameCategoryEncoder factory{threads, *frame, 3};
        maths::CDataFrameCategoryEncoder encoder{factory};
        factory.makeEncodings();

        TMeanAccumulator oneHotTargetMean;
        TMeanAccumulator rareTargetMean;
        for (std::size_t i = 0; i < expectedTargetMeanValues.size(); ++i) {
            if (factory.usesOneHotEncoding(0, i)) {
                oneHotTargetMean += expectedTargetMeanValues[i];
            } else if (factory.isRareCategory(0, i)) {
                rareTargetMean += expectedTargetMeanValues[i];
            }
        }
        for (std::size_t i = 0; i < expectedTargetMeanValues.size(); ++i) {
            if (factory.usesOneHotEncoding(0, i)) {
                expectedTargetMeanValues[i] = oneHotTargetMean;
            } else if (factory.isRareCategory(0, i)) {
                expectedTargetMeanValues[i] = rareTargetMean;
            }
        }

        for (std::size_t i = 0; i < encoder.numberEncodedColumns(); ++i) {
            if (encoder.encoding(i).type() == maths::E_TargetMean) {
                for (std::size_t j = 0; j < expectedTargetMeanValues.size(); ++j) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        maths::CBasicStatistics::mean(expectedTargetMeanValues[j]),
                        encoder.encoding(i).encode(static_cast<double>(j)),
                        static_cast<double>(std::numeric_limits<float>::epsilon()) *
                            std::fabs(maths::CBasicStatistics::mean(
                                expectedTargetMeanValues[i])));
                }
                break;
            }
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testRareCategories) {

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

    frame->categoricalColumns(TBoolVec{false, false, true, false});
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

    auto factory = maths::CMakeDataFrameCategoryEncoder{1, *frame, 3}.minimumFrequencyToOneHotEncode(
        0.1);
    factory.makeEncodings();

    for (std::size_t i = 0; i < categoryCounts.size(); ++i) {
        BOOST_REQUIRE_EQUAL(categoryCounts[i] < 50, factory.isRareCategory(2, i));
    }
}

BOOST_AUTO_TEST_CASE(testCorrelatedFeatures) {

    // Test the case that if two fields are strongly correlated we will
    // tend to just select one or the other.

    test::CRandomNumbers rng;

    // Two correlated + 4 independent metrics.
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

        frame->categoricalColumns(TBoolVec{false, false, false, false, false, false, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                    *column = features[j][i];
                }
                *column = target(features, i);
            });
        }
        frame->finishWritingRows();

        maths::CDataFrameCategoryEncoder encoder{{1, *frame, 6}};

        // Dispite both carrying a lot of information about the target nearly
        // the same information is carried by columns 0 and 1 so we should
        // choose feature 0 or 1 and feature 5.

        TSizeVec expectedColumns{1, 5, 6};
        BOOST_REQUIRE_EQUAL(expectedColumns.size(), encoder.numberEncodedColumns());
        for (std::size_t i = 0; i < encoder.numberEncodedColumns(); ++i) {
            BOOST_REQUIRE_EQUAL(expectedColumns[i], encoder.encoding(i).inputColumnIndex());
        }
    }

    // Two correlated + two independent categorical fields.
    {
        auto target = [&](const TDoubleVecVec& features, std::size_t row) {
            return std::floor(features[0][row]) + std::floor(features[1][row]) +
                   2.0 * (std::floor(features[2][row]) + std::floor(features[3][row]));
        };

        std::size_t rows{200};
        std::size_t cols{5};
        double numberCategories{4.0};

        TDoubleVecVec features(cols - 1);
        rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
        features[1] = features[0];
        rng.discard(100000);
        rng.generateUniformSamples(0.0, numberCategories, rows, features[2]);
        rng.discard(100000);
        rng.generateUniformSamples(0.0, numberCategories, rows, features[3]);

        auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

        frame->categoricalColumns(TBoolVec{true, true, true, true, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                    *column = features[j][i];
                }
                *column = target(features, i);
            });
        }
        frame->finishWritingRows();

        maths::CDataFrameCategoryEncoder encoder{{1, *frame, 4}};

        // Dispite both carrying a lot of information about the target nearly
        // the same information is carried by columns 0 and 1 so we should
        // choose feature 0 or 1 and features 2 and 3.

        TSizeVec expectedColumns{0, 0, 2, 3, 4};
        BOOST_REQUIRE_EQUAL(expectedColumns.size(), encoder.numberEncodedColumns());
        for (std::size_t i = 0; i < encoder.numberEncodedColumns(); ++i) {
            BOOST_REQUIRE_EQUAL(expectedColumns[i], encoder.encoding(i).inputColumnIndex());
        }
    }
}

BOOST_AUTO_TEST_CASE(testWithRowMask) {

    // Test the invariant that the encoding for the row mask equals the
    // encoding on a reduced frame containing only the masked rows.

    test::CRandomNumbers rng;

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        return features[0][row] + features[1][row] + features[2][row];
    };

    std::size_t rows{500};
    std::size_t cols{4};

    core::CPackedBitVector rowMask;
    TDoubleVec uniform01;
    rng.generateUniformSamples(0.0, 1.0, rows, uniform01);
    for (auto u : uniform01) {
        rowMask.extend(u < 0.5);
    }

    TDoubleVecVec features(cols - 1);
    rng.generateNormalSamples(-1.0, 2.0, rows, features[0]);
    rng.generateNormalSamples(0.0, 4.0, rows, features[1]);
    rng.generateNormalSamples(2.0, 2.0, rows, features[2]);

    auto frame = core::makeMainStorageDataFrame(cols).first;
    auto maskedFrame = core::makeMainStorageDataFrame(cols).first;

    for (std::size_t i = 0; i < rows; ++i) {
        auto writeOneRow = [&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                *column = features[j][i];
            }
            *column = target(features, i);
        };

        frame->writeRow(writeOneRow);
        if (rowMask[i]) {
            maskedFrame->writeRow(writeOneRow);
        }
    }
    frame->finishWritingRows();
    maskedFrame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{
        maths::CMakeDataFrameCategoryEncoder{1, *frame, 3}.rowMask(rowMask)};
    maths::CDataFrameCategoryEncoder maskedEncoder{{1, *maskedFrame, 3}};

    BOOST_REQUIRE_EQUAL(encoder.checksum(), maskedEncoder.checksum());
}

BOOST_AUTO_TEST_CASE(testEncodingOfCategoricalTarget) {

    // Test the target uses identity encoding.

    test::CRandomNumbers rng;

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        return std::floor(features[0][row] + features[1][row] + features[2][row]);
    };

    std::size_t rows{500};
    std::size_t cols{4};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, 2.0, rows, features[0]);
    rng.generateUniformSamples(0.0, 2.0, rows, features[1]);
    rng.generateUniformSamples(0.0, 2.0, rows, features[2]);

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{false, false, false, true});
    for (std::size_t i = 0; i < rows; ++i) {
        auto writeOneRow = [&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                *column = features[j][i];
            }
            *column = target(features, i);
        };
        frame->writeRow(writeOneRow);
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{{1, *frame, 3}};

    for (std::size_t i = 0; i < encoder.numberEncodedColumns(); ++i) {
        BOOST_REQUIRE_EQUAL(maths::E_IdentityEncoding, encoder.encoding(i).type());
    }
}

BOOST_AUTO_TEST_CASE(testEncodedDataFrameRowRef) {

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

    std::size_t rows{500};
    std::size_t cols{5};
    double numberCategories{4.1};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
    rng.generateNormalSamples(0.0, 4.0, rows, features[1]);
    rng.generateNormalSamples(2.0, 2.0, rows, features[2]);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[3]);

    auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

    TSizeVec categorical{0, 3};
    TMeanAccumulatorVecVec expectedTargetMeanValues(cols);
    TDoubleVecVec expectedFrequencies(cols);
    for (auto feature : categorical) {
        expectedTargetMeanValues[feature] = TMeanAccumulatorVec(
            static_cast<std::size_t>(std::ceil(numberCategories)));
        expectedFrequencies[feature] =
            TDoubleVec(static_cast<std::size_t>(std::ceil(numberCategories)));
    }

    TSizeVecVec expectedOneHot{{0, 1, 2}, {}, {}, {0, 1}, {}};
    TSizeVecVec expectedRare{{4}, {}, {}, {4}, {}};

    auto expandOneHot = [&](std::size_t feature, std::size_t category) {
        if (std::binary_search(expectedOneHot[feature].begin(),
                               expectedOneHot[feature].end(), category)) {
            return expectedOneHot[feature];
        }
        return TSizeVec{category};
    };

    frame->categoricalColumns(TBoolVec{true, false, false, true, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            *(column++) = std::floor(features[0][i]);
            for (std::size_t j = 1; j + 2 < cols; ++j, ++column) {
                *column = features[j][i];
            }
            *(column++) = std::floor(features[3][i]);
            *column = target(features, i);

            for (auto feature : categorical) {
                std::size_t category{static_cast<std::size_t>(features[feature][i])};
                for (auto category_ : expandOneHot(feature, category)) {
                    expectedTargetMeanValues[feature][category_].add(*column);
                }
                expectedFrequencies[feature][category] += 1.0 / static_cast<double>(rows);
            }
        });
    }
    frame->finishWritingRows();

    for (auto feature : categorical) {
        TMeanAccumulator meanFrequency;
        for (auto category : expectedOneHot[feature]) {
            double frequency{expectedFrequencies[feature][category]};
            meanFrequency.add(frequency, frequency);
        }
        for (auto category : expectedOneHot[feature]) {
            expectedFrequencies[feature][category] =
                maths::CBasicStatistics::mean(meanFrequency);
        }
    }

    maths::CMakeDataFrameCategoryEncoder factory{1, *frame, 4};
    maths::CDataFrameCategoryEncoder encoder{factory};
    factory.makeEncodings();
    LOG_DEBUG(<< "# features = " << encoder.numberEncodedColumns());

    auto expectedEncoded = [&](const core::CDataFrame::TRowRef& row, std::size_t i) {

        TSizeVec categories(cols);
        for (auto feature : categorical) {
            categories[feature] = static_cast<std::size_t>(row[feature]);
        }

        if (i < expectedOneHot[0].size()) {
            return categories[0] == expectedOneHot[0][factory.encoding(i)] ? 1.0 : 0.0; // one-hot
        }
        if (i < expectedOneHot[0].size() + 1) {
            return expectedFrequencies[0][categories[0]]; // frequency
        }
        if (i < expectedOneHot[0].size() + 2) {
            return maths::CBasicStatistics::mean(
                expectedTargetMeanValues[0][categories[0]]); // target mean
        }
        if (i < expectedOneHot[0].size() + 4) {
            return static_cast<double>(row[encoder.encoding(i).inputColumnIndex()]); // metrics
        }
        if (i < expectedOneHot[0].size() + 4 + expectedOneHot[3].size()) {
            return categories[3] == expectedOneHot[3][factory.encoding(i)] ? 1.0 : 0.0; // one-hot
        }
        if (i < expectedOneHot[0].size() + 4 + expectedOneHot[3].size() + 1) {
            return maths::CBasicStatistics::mean(
                expectedTargetMeanValues[3][categories[3]]); // target mean
        }
        return static_cast<double>(row[encoder.encoding(i).inputColumnIndex()]); // target
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
                        LOG_DEBUG(<< i << " got " << encoded[i] << " expected "
                                  << expectedEncoded(*row, i));
                    }
                }
            }
        }
    });

    BOOST_TEST_REQUIRE(passed);
}

BOOST_AUTO_TEST_CASE(testUnseenCategoryEncoding) {

    // Test categories we didn't supply when computing the encoding.

    test::CRandomNumbers rng;

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        return features[0][row] + features[1][row] + features[2][row];
    };

    std::size_t rows{500};
    std::size_t cols{4};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, 2.0, rows, features[0]);
    rng.generateUniformSamples(0.0, 4.0, rows, features[1]);
    rng.generateUniformSamples(0.0, 3.0, rows, features[2]);

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{true, true, true, false});
    for (std::size_t i = 0; i < rows; ++i) {
        auto writeOneRow = [&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j + 1 < cols; ++j, ++column) {
                *column = std::floor(features[j][i]);
            }
            *column = target(features, i);
        };

        frame->writeRow(writeOneRow);
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{{1, *frame, 3}};

    TAlignedFloatVec unseen{3.0, 5.0, 4.0, 1.5};
    core::CDataFrame::TRowRef row{rows, unseen.begin(), unseen.end(), 0};

    auto encodedRow = encoder.encode(row);

    // Check some properties we know must hold.

    std::ostringstream rep;
    for (std::size_t i = 0; i < encodedRow.numberColumns() - 1; ++i) {
        if (encoder.isBinary(i)) {
            BOOST_REQUIRE_EQUAL(maths::CFloatStorage{0.0}, encodedRow[i]);
        } else {
            BOOST_TEST_REQUIRE(encodedRow[i] > 0.0);
        }
        rep << " " << encodedRow[i];
    }
    BOOST_REQUIRE_EQUAL(maths::CFloatStorage{1.5},
                        encodedRow[encodedRow.numberColumns() - 1]);
    LOG_DEBUG(<< "encoded = [" << rep.str() << "]");
}

BOOST_AUTO_TEST_CASE(testDiscardNuisanceFeatures) {

    // Test we discard features altogether which don't carry any information.

    test::CRandomNumbers rng;

    std::size_t rows{10000};
    std::size_t cols{7};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, 6.0, rows, features[0]);
    rng.generateUniformSamples(0.0, 5.0, rows, features[1]);
    rng.generateUniformSamples(0.0, 7.0, rows, features[2]);
    rng.generateUniformSamples(0.0, 5.0, rows, features[3]);
    rng.generateUniformSamples(0.0, 6.0, rows, features[4]);
    rng.generateUniformSamples(0.0, 5.0, rows, features[5]);

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{false, false, false, false, false, false, false});
    for (std::size_t i = 0; i < rows; ++i) {
        auto writeOneRow = [&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            double target{0.0};
            std::size_t j = 0;
            for (/**/; j + 2 < cols; ++j, ++column) {
                target += * column = features[j][i];
            }
            *(column++) = features[j][i];
            *column = target;
        };

        frame->writeRow(writeOneRow);
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{
        maths::CMakeDataFrameCategoryEncoder{1, *frame, 6}.minimumRelativeMicToSelectFeature(0.02)};

    LOG_DEBUG(<< "number selected features = " << encoder.numberEncodedColumns()
              << " / " << cols);
    BOOST_REQUIRE_EQUAL(cols - 1, encoder.numberEncodedColumns());
    for (std::size_t i = 0; i < encoder.numberEncodedColumns(); ++i) {
        BOOST_TEST_REQUIRE(encoder.encoding(i).inputColumnIndex() != 5);
    }
}

BOOST_AUTO_TEST_CASE(testPersistRestore) {

    // Test checksum of restored encoder matches persisted one.

    TDoubleVec categoryValue[2]{{-15.0, 20.0, 0.0}, {10.0, -10.0, 0.0}};

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        std::size_t categories[]{
            static_cast<std::size_t>(std::min(features[0][row], 2.0)),
            static_cast<std::size_t>(std::min(features[3][row], 2.0))};
        return categoryValue[0][categories[0]] + categoryValue[1][categories[1]] +
               2.6 * features[1][row] - 5.3 * features[2][row];
    };

    test::CRandomNumbers rng;

    std::size_t rows{500};
    std::size_t cols{5};
    double numberCategories{4.1};

    TDoubleVecVec features(cols - 1);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[0]);
    rng.generateNormalSamples(0.0, 4.0, rows, features[1]);
    rng.generateNormalSamples(2.0, 2.0, rows, features[2]);
    rng.generateUniformSamples(0.0, numberCategories, rows, features[3]);

    auto frame = core::makeMainStorageDataFrame(cols, 2 * rows).first;

    frame->categoricalColumns(TBoolVec{true, false, false, true, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            *(column++) = std::floor(features[0][i]);
            for (std::size_t j = 1; j + 2 < cols; ++j, ++column) {
                *column = features[j][i];
            }
            *(column++) = std::floor(features[3][i]);
            *column = target(features, i);
        });
    }
    frame->finishWritingRows();

    maths::CDataFrameCategoryEncoder encoder{{1, *frame, 4}};

    std::stringstream persistTo;
    core::CJsonStatePersistInserter inserter{persistTo};
    inserter.insertLevel("top-level", std::bind(&maths::CDataFrameCategoryEncoder::acceptPersistInserter,
                                                &encoder, std::placeholders::_1));
    encoder.acceptPersistInserter(inserter);
    persistTo.flush();

    LOG_DEBUG(<< "persisted " << persistTo.str());

    try {
        core::CJsonStateRestoreTraverser traverser{persistTo};
        maths::CDataFrameCategoryEncoder restoredEncoder{traverser};
        BOOST_REQUIRE_EQUAL(encoder.checksum(), restoredEncoder.checksum());

    } catch (const std::exception& e) { BOOST_FAIL(e.what()); }
}

BOOST_AUTO_TEST_SUITE_END()
