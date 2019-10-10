/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBoostedTreeTest.h"

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CRegex.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <functional>
#include <memory>
#include <utility>

using namespace ml;

using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;
using TFactoryFuncVec = std::vector<TFactoryFunc>;
using TFactoryFuncVecVec = std::vector<TFactoryFuncVec>;
using TRowRef = core::CDataFrame::TRowRef;
using TRowItr = core::CDataFrame::TRowItr;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

namespace {

template<typename F>
auto computeEvaluationMetrics(const core::CDataFrame& frame,
                              std::size_t beginTestRows,
                              std::size_t endTestRows,
                              std::size_t columnHoldingPrediction,
                              const F& target,
                              double noiseVariance) {

    TMeanVarAccumulator functionMoments;
    TMeanVarAccumulator modelPredictionErrorMoments;

    frame.readRows(1, beginTestRows, endTestRows, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            functionMoments.add(target(*row));
            modelPredictionErrorMoments.add(target(*row) - (*row)[columnHoldingPrediction]);
        }
    });

    LOG_TRACE(<< "function moments = " << functionMoments);
    LOG_TRACE(<< "model prediction error moments = " << modelPredictionErrorMoments);

    double functionVariance{maths::CBasicStatistics::variance(functionMoments)};
    double predictionErrorMean{maths::CBasicStatistics::mean(modelPredictionErrorMoments)};
    double predictionErrorVariance{maths::CBasicStatistics::variance(modelPredictionErrorMoments)};
    double rSquared{1.0 - (predictionErrorVariance - noiseVariance) /
                              (functionVariance - noiseVariance)};

    return std::make_pair(predictionErrorMean, rSquared);
}

template<typename F>
void fillDataFrame(std::size_t trainRows,
                   std::size_t testRows,
                   std::size_t cols,
                   const TBoolVec& categoricalColumns,
                   const TDoubleVecVec& regressors,
                   const TDoubleVec& noise,
                   const F& target,
                   core::CDataFrame& frame) {

    std::size_t rows{trainRows + testRows};
    frame.categoricalColumns(categoricalColumns);
    for (std::size_t i = 0; i < rows; ++i) {
        frame.writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j < cols - 1; ++j, ++column) {
                *column = regressors[j][i];
            }
        });
    }
    frame.finishWritingRows();
    frame.writeColumns(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double targetValue{row->index() < trainRows
                                   ? target(*row) + noise[row->index()]
                                   : core::CDataFrame::valueOfMissing()};
            row->writeColumn(cols - 1, targetValue);
        }
    });
}

template<typename F>
void fillDataFrame(std::size_t trainRows,
                   std::size_t testRows,
                   std::size_t cols,
                   const TDoubleVecVec& regressors,
                   const TDoubleVec& noise,
                   const F& target,
                   core::CDataFrame& frame) {
    fillDataFrame(trainRows, testRows, cols, TBoolVec(cols, false), regressors,
                  noise, target, frame);
}

template<typename F>
auto predictAndComputeEvaluationMetrics(const F& generateFunction,
                                        test::CRandomNumbers& rng,
                                        std::size_t trainRows,
                                        std::size_t testRows,
                                        std::size_t cols,
                                        std::size_t capacity,
                                        double noiseVariance) {

    std::size_t rows{trainRows + testRows};

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    TFactoryFuncVecVec factories{
        {makeOnDisk, makeMainMemory}, {makeMainMemory}, {makeMainMemory}};

    TDoubleVecVec modelBias(factories.size());
    TDoubleVecVec modelRSquared(factories.size());

    for (std::size_t test = 0; test < factories.size(); ++test) {

        auto target = generateFunction(rng, cols);

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

        for (const auto& factory : factories[test]) {

            auto frame = factory();

            fillDataFrame(trainRows, testRows, cols, x, noise, target, *frame);

            auto regression =
                maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
                    *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

            regression->train();
            regression->predict();

            double bias;
            double rSquared;
            std::tie(bias, rSquared) = computeEvaluationMetrics(
                *frame, trainRows, rows,
                regression->columnHoldingPrediction(frame->numberColumns()),
                target, noiseVariance / static_cast<double>(rows));
            modelBias[test].push_back(bias);
            modelRSquared[test].push_back(rSquared);
        }
    }
    LOG_DEBUG(<< "bias = " << core::CContainerPrinter::print(modelBias));
    LOG_DEBUG(<< " R^2 = " << core::CContainerPrinter::print(modelRSquared));

    return std::make_pair(std::move(modelBias), std::move(modelRSquared));
}
}

void CBoostedTreeTest::testPiecewiseConstant() {

    // Test regression quality on piecewise constant function.

    auto generatePiecewiseConstant = [](test::CRandomNumbers& rng, std::size_t cols) {
        TDoubleVec p;
        TDoubleVec v;
        rng.generateUniformSamples(0.0, 10.0, 2 * cols - 2, p);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, v);
        for (std::size_t i = 0; i < p.size(); i += 2) {
            std::sort(p.begin() + i, p.begin() + i + 2);
        }

        return [p, v, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                if (row[i] >= p[2 * i] && row[i] < p[2 * i + 1]) {
                    result += v[i];
                }
            }
            return result;
        };
    };

    TDoubleVecVec modelBias;
    TDoubleVecVec modelRSquared;

    test::CRandomNumbers rng;
    double noiseVariance{0.2};
    std::size_t trainRows{1000};
    std::size_t testRows{200};
    std::size_t cols{6};
    std::size_t capacity{250};

    std::tie(modelBias, modelRSquared) = predictAndComputeEvaluationMetrics(
        generatePiecewiseConstant, rng, trainRows, testRows, cols, capacity, noiseVariance);

    TMeanAccumulator meanModelRSquared;

    for (std::size_t i = 0; i < modelBias.size(); ++i) {

        // In and out-of-core agree.
        for (std::size_t j = 1; j < modelBias[i].size(); ++j) {
            CPPUNIT_ASSERT_EQUAL(modelBias[i][0], modelBias[i][j]);
            CPPUNIT_ASSERT_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Unbiased...
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0.0, modelBias[i][0],
            4.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.96);

        meanModelRSquared.add(modelRSquared[i][0]);
    }

    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.97);
}

void CBoostedTreeTest::testLinear() {

    // Test regression quality on linear function.

    auto generateLinear = [](test::CRandomNumbers& rng, std::size_t cols) {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);

        return [m, s, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    };

    TDoubleVecVec modelBias;
    TDoubleVecVec modelRSquared;

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t trainRows{1000};
    std::size_t testRows{200};
    std::size_t cols{6};
    std::size_t capacity{500};

    std::tie(modelBias, modelRSquared) = predictAndComputeEvaluationMetrics(
        generateLinear, rng, trainRows, testRows, cols, capacity, noiseVariance);

    TMeanAccumulator meanModelRSquared;

    for (std::size_t i = 0; i < modelBias.size(); ++i) {

        // In and out-of-core agree.
        for (std::size_t j = 1; j < modelBias[i].size(); ++j) {
            CPPUNIT_ASSERT_EQUAL(modelBias[i][0], modelBias[i][j]);
            CPPUNIT_ASSERT_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Unbiased...
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0.0, modelBias[i][0],
            4.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.97);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.97);
}

void CBoostedTreeTest::testNonLinear() {

    // Test regression quality on non-linear function.

    auto generateNonLinear = [](test::CRandomNumbers& rng, std::size_t cols) {

        cols = cols - 1;

        TDoubleVec mean;
        TDoubleVec slope;
        TDoubleVec curve;
        TDoubleVec cross;
        rng.generateUniformSamples(0.0, 10.0, cols, mean);
        rng.generateUniformSamples(-10.0, 10.0, cols, slope);
        rng.generateUniformSamples(-1.0, 1.0, cols, curve);
        rng.generateUniformSamples(-0.5, 0.5, cols * cols, cross);

        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols; ++i) {
                result += mean[i] + (slope[i] + curve[i] * row[i]) * row[i];
            }
            for (std::size_t i = 0; i < cols; ++i) {
                for (std::size_t j = 0; j < i; ++j) {
                    result += cross[i * cols + j] * row[i] * row[j];
                }
            }
            return result;
        };
    };

    TDoubleVecVec modelBias;
    TDoubleVecVec modelRSquared;

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t trainRows{500};
    std::size_t testRows{100};
    std::size_t cols{6};
    std::size_t capacity{500};

    std::tie(modelBias, modelRSquared) = predictAndComputeEvaluationMetrics(
        generateNonLinear, rng, trainRows, testRows, cols, capacity, noiseVariance);

    TMeanAccumulator meanModelRSquared;

    for (std::size_t i = 0; i < modelBias.size(); ++i) {

        // In and out-of-core agree.
        for (std::size_t j = 1; j < modelBias[i].size(); ++j) {
            CPPUNIT_ASSERT_EQUAL(modelBias[i][0], modelBias[i][j]);
            CPPUNIT_ASSERT_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Unbiased...
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0.0, modelBias[i][0],
            4.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.96);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.97);
}

void CBoostedTreeTest::testThreading() {

    // Test we get the same results whether we run with multiple threads or not.
    // Note because we compute approximate quantiles for a partition of the data
    // (one subset for each thread) and merge we get slightly different results
    // if running multiple vs single threaded. However, we should get the same
    // results whether we actually execute in parallel or not provided we perform
    // the same partitioning. Therefore, we test with two logical threads but
    // with and without starting the thread pool.

    test::CRandomNumbers rng;
    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{100};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [m, s, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 0.1, rows, noise);

    core::stopDefaultAsyncExecutor();

    TDoubleVec modelBias;
    TDoubleVec modelMse;

    std::string tests[]{"serial", "parallel"};

    for (std::size_t test = 0; test < 2; ++test) {

        LOG_DEBUG(<< tests[test]);

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(rows, 0, cols, x, noise, target, *frame);

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(2).buildFor(
            *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

        regression->train();
        regression->predict();

        TMeanVarAccumulator modelPredictionErrorMoments;

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                std::size_t index{regression->columnHoldingPrediction(row->numberColumns())};
                modelPredictionErrorMoments.add(target(*row) - (*row)[index]);
            }
        });

        LOG_DEBUG(<< "model prediction error moments = " << modelPredictionErrorMoments);

        modelBias.push_back(maths::CBasicStatistics::mean(modelPredictionErrorMoments));
        modelMse.push_back(maths::CBasicStatistics::variance(modelPredictionErrorMoments));

        core::startDefaultAsyncExecutor();
    }

    CPPUNIT_ASSERT_EQUAL(modelBias[0], modelBias[1]);
    CPPUNIT_ASSERT_EQUAL(modelMse[0], modelMse[1]);

    core::stopDefaultAsyncExecutor();
}

void CBoostedTreeTest::testConstantFeatures() {

    // Test constant features are excluded from the model.

    test::CRandomNumbers rng;
    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{500};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [m, s, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    TDoubleVecVec x(cols - 1, TDoubleVec(rows, 1.0));
    for (std::size_t i = 0; i < cols - 2; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 0.1, rows, noise);

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    fillDataFrame(rows, 0, cols, x, noise, target, *frame);

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
        *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

    regression->train();

    TDoubleVec featureWeights(regression->featureWeights());

    LOG_DEBUG(<< "feature weights = " << core::CContainerPrinter::print(featureWeights));
    CPPUNIT_ASSERT(featureWeights[cols - 2] < 1e-4);
}

void CBoostedTreeTest::testConstantTarget() {

    // Test we correctly deal with a constant dependent variable.

    test::CRandomNumbers rng;
    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{500};

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < x.size(); ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    fillDataFrame(rows, 0, cols, x, TDoubleVec(rows, 0.0),
                  [](const TRowRef&) { return 1.0; }, *frame);

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
        *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

    regression->train();

    TMeanAccumulator modelPredictionError;

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            std::size_t index{regression->columnHoldingPrediction(row->numberColumns())};
            modelPredictionError.add(1.0 - (*row)[index]);
        }
    });

    LOG_DEBUG(<< "mean prediction error = "
              << maths::CBasicStatistics::mean(modelPredictionError));
    CPPUNIT_ASSERT_EQUAL(0.0, maths::CBasicStatistics::mean(modelPredictionError));
}

void CBoostedTreeTest::testCategoricalRegressors() {

    // Test automatic handling of categorical regressors.

    test::CRandomNumbers rng;

    std::size_t trainRows{1000};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};
    std::size_t cols{6};
    std::size_t capacity{500};

    TDoubleVecVec offsets{{0.0, 0.0, 12.0, -3.0, 0.0},
                          {12.0, 1.0, -3.0, 0.0, 0.0, 2.0, 16.0, 0.0, 0.0, -6.0}};
    TDoubleVec weights{0.7, 0.2, -0.4};
    auto target = [&](const TRowRef& x) {
        double result{offsets[0][static_cast<std::size_t>(x[0])] +
                      offsets[1][static_cast<std::size_t>(x[1])]};
        for (std::size_t i = 2; i < cols - 1; ++i) {
            result += weights[i - 2] * x[i];
        }
        return result;
    };

    TDoubleVecVec regressors(cols - 1);
    rng.generateMultinomialSamples({0.0, 1.0, 2.0, 3.0, 4.0},
                                   {0.03, 0.17, 0.3, 0.1, 0.4}, rows, regressors[0]);
    rng.generateMultinomialSamples(
        {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
        {0.03, 0.07, 0.08, 0.02, 0.2, 0.15, 0.1, 0.05, 0.26, 0.04}, rows, regressors[1]);
    for (std::size_t i = 2; i < regressors.size(); ++i) {
        rng.generateUniformSamples(-10.0, 10.0, rows, regressors[i]);
    }

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    frame->categoricalColumns(TBoolVec{true, true, false, false, false, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j < cols - 1; ++j, ++column) {
                *column = regressors[j][i];
            }
        });
    }
    frame->finishWritingRows();
    frame->writeColumns(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double targetValue{row->index() < trainRows
                                   ? target(*row)
                                   : core::CDataFrame::valueOfMissing()};
            row->writeColumn(cols - 1, targetValue);
        }
    });

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
        *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        regression->columnHoldingPrediction(frame->numberColumns()), target, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, modelBias, 0.1);
    CPPUNIT_ASSERT(modelRSquared > 0.91);
}

void CBoostedTreeTest::testIntegerRegressor() {

    // Test a simple integer regressor.

    test::CRandomNumbers rng;

    std::size_t trainRows{1000};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};

    auto frame = core::makeMainStorageDataFrame(2).first;

    frame->categoricalColumns(TBoolVec{false, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            TDoubleVec regressor;
            rng.generateUniformSamples(1.0, 4.0, 1, regressor);
            *(column++) = std::floor(regressor[0]);
            *column = i < trainRows ? 10.0 * std::floor(regressor[0])
                                    : core::CDataFrame::valueOfMissing();
        });
    }
    frame->finishWritingRows();

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
        *frame, std::make_unique<maths::boosted_tree::CMse>(), 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        regression->columnHoldingPrediction(frame->numberColumns()),
        [&](const TRowRef& x) { return 10.0 * x[0]; }, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, modelBias, 0.05);
    CPPUNIT_ASSERT(modelRSquared > 0.99);
}

void CBoostedTreeTest::testSingleSplit() {

    // We were getting an out-of-bound read in initialization when running on
    // data with only two distinct values for the target variable. This test
    // fails intermittently without the fix.

    test::CRandomNumbers rng;

    std::size_t rows{100};
    std::size_t cols{2};

    TDoubleVec x;
    rng.generateUniformSamples(0.0, 1.0, rows, x);
    for (auto& xi : x) {
        xi = std::floor(xi + 0.5);
    }

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{false, false});
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            *(column++) = x[i];
            *column = 10.0 * x[i];
        });
    }
    frame->finishWritingRows();

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
        *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

    regression->train();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, 0, rows, regression->columnHoldingPrediction(frame->numberColumns()),
        [](const TRowRef& row) { return 10.0 * row[0]; }, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    CPPUNIT_ASSERT(modelRSquared > 0.99);
}

void CBoostedTreeTest::testTranslationInvariance() {

    // We should get similar performance independent of fixed shifts for the target.

    using TTargetFunc = std::function<double(const TRowRef& row)>;

    test::CRandomNumbers rng;

    std::size_t trainRows{1000};
    std::size_t rows{1200};
    std::size_t cols{4};
    std::size_t capacity{1200};

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TTargetFunc target{[&](const TRowRef& row) {
        double result{0.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += row[i];
        }
        return result;
    }};
    TTargetFunc shiftedTarget{[&](const TRowRef& row) {
        double result{10000.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += row[i];
        }
        return result;
    }};

    TDoubleVec rsquared;

    for (const auto& target_ : {target, shiftedTarget}) {

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(trainRows, rows - trainRows, cols, x,
                      TDoubleVec(rows, 0.0), target_, *frame);

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
            *frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

        regression->train();
        regression->predict();

        double modelBias;
        double modelRSquared;
        std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
            *frame, trainRows, rows,
            regression->columnHoldingPrediction(frame->numberColumns()), target_, 0.0);

        LOG_DEBUG(<< "bias = " << modelBias);
        LOG_DEBUG(<< " R^2 = " << modelRSquared);
        rsquared.push_back(modelRSquared);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(rsquared[0], rsquared[1], 5e-3);
}

std::size_t maxDepth(const std::vector<maths::CBoostedTreeNode>& tree,
                     const maths::CBoostedTreeNode& node,
                     std::size_t depth) {
    std::size_t result{depth};
    if (node.leftChildIndex() >= 0) {
        result = std::max(result, maxDepth(tree, tree[node.leftChildIndex()], depth + 1));
    }
    if (node.rightChildIndex() >= 0) {
        result = std::max(result, maxDepth(tree, tree[node.rightChildIndex()], depth + 1));
    }
    return result;
}

void CBoostedTreeTest::testDepthBasedRegularization() {

    // Test that the trained tree depth is correctly limited based on a target.

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t rows{1000};
    std::size_t cols{4};
    std::size_t capacity{1200};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);

        return [m, s, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

    for (auto targetDepth : {3.0, 5.0}) {
        LOG_DEBUG(<< "target depth = " << targetDepth);

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(rows, 0, cols, x, noise, target, *frame);

        auto regression =
            maths::CBoostedTreeFactory::constructFromParameters(1)
                .treeSizePenaltyMultiplier(0.0)
                .leafWeightPenaltyMultiplier(0.0)
                .softTreeDepthLimit(targetDepth)
                .softTreeDepthTolerance(0.05)
                .buildFor(*frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

        regression->train();

        TMeanAccumulator meanDepth;
        for (const auto& tree : regression->trainedModel()) {
            CPPUNIT_ASSERT(maxDepth(tree, tree[0], 0) <= static_cast<std::size_t>(targetDepth));
            meanDepth.add(static_cast<double>(maxDepth(tree, tree[0], 0)));
        }
        LOG_DEBUG(<< "mean depth = " << maths::CBasicStatistics::mean(meanDepth));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanDepth) > targetDepth - 1.0);
    }
}

void CBoostedTreeTest::testLogisticMinimizer() {

    // Test that we a good approximation of the additive term for the log-odds
    // which minimises the cross entropy objective.

    using maths::boosted_tree_detail::CArgMinLogisticImpl;

    test::CRandomNumbers rng;

    TDoubleVec labels;
    TDoubleVec weights;

    // All predictions equal and zero.
    {
        CArgMinLogisticImpl argmin{0.0};
        argmin.add(0.0, 0.0);
        argmin.add(0.0, 1.0);
        argmin.add(0.0, 1.0);
        argmin.add(0.0, 0.0);
        argmin.nextPass();
        CPPUNIT_ASSERT_EQUAL(0.0, argmin.value());
    }
    // All predictions are equal.
    {
        rng.generateUniformSamples(0.0, 1.0, 1000, labels);
        for (auto& label : labels) {
            label = std::floor(label + 0.3);
        }
        weights.resize(labels.size(), 0.0);

        CArgMinLogisticImpl argmin{0.0};
        std::size_t numberPasses{0};
        std::size_t counts[2]{0, 0};

        do {
            ++numberPasses;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                argmin.add(weights[i], labels[i]);
                ++counts[static_cast<std::size_t>(labels[i])];
            }
        } while (argmin.nextPass());

        double p{static_cast<double>(counts[1]) / 1000.0};
        double expected{std::log(p / (1.0 - p))};
        double actual{argmin.value()};

        CPPUNIT_ASSERT_EQUAL(std::size_t{1}, numberPasses);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 0.01 * std::fabs(expected));
    }

    for (auto lambda : {0.0, 10.0}) {

        LOG_DEBUG(<< "lambda = " << lambda);

        // The true objective.
        auto objective = [&](double weight) {
            double loss{0.0};
            for (std::size_t i = 0; i < labels.size(); ++i) {
                double p{maths::CTools::logisticFunction(weights[i] + weight)};
                loss -= (1.0 - labels[i]) * maths::CTools::fastLog(1.0 - p) +
                        labels[i] * maths::CTools::fastLog(p);
            }
            return loss + lambda * maths::CTools::pow2(weight);
        };

        // This loop is fuzzing the predicted log-odds and testing we get consistently
        // good estimates of the true minimizer.
        for (std::size_t t = 0; t < 10; ++t) {

            double min{std::numeric_limits<double>::max()};
            double max{-min};

            rng.generateUniformSamples(0.0, 1.0, 1000, labels);
            for (auto& label : labels) {
                label = std::floor(label + 0.5);
            }
            weights.clear();
            for (const auto& label : labels) {
                TDoubleVec weight;
                rng.generateNormalSamples(label, 2.0, 1, weight);
                weights.push_back(weight[0]);
                min = std::min(min, weight[0]);
                max = std::max(max, weight[0]);
            }

            double expected;
            double objectiveAtExpected;
            std::size_t maxIterations{20};
            maths::CSolvers::minimize(-max, -min, objective(-max), objective(-min),
                                      objective, 1e-3, maxIterations, expected,
                                      objectiveAtExpected);
            LOG_DEBUG(<< "expected = " << expected
                      << " objective at expected = " << objectiveAtExpected);

            CArgMinLogisticImpl argmin{lambda};
            CArgMinLogisticImpl argminPartition[2]{{lambda}, {lambda}};
            auto nextPass = [&] {
                bool done{argmin.nextPass() == false};
                done &= (argminPartition[0].nextPass() == false);
                done &= (argminPartition[1].nextPass() == false);
                return done == false;
            };

            do {
                for (std::size_t i = 0; i < labels.size() / 2; ++i) {
                    argmin.add(weights[i], labels[i]);
                    argminPartition[0].add(weights[i], labels[i]);
                }
                for (std::size_t i = labels.size() / 2; i < labels.size(); ++i) {
                    argmin.add(weights[i], labels[i]);
                    argminPartition[1].add(weights[i], labels[i]);
                }
                argminPartition[0].merge(argminPartition[1]);
                argminPartition[1] = argminPartition[0];
            } while (nextPass());

            double actual{argmin.value()};
            double actualPartition{argminPartition[0].value()};
            LOG_DEBUG(<< "actual = " << actual
                      << " objective at actual = " << objective(actual));

            // We should be within 1% for the value and 0.001% for the objective
            // at the value.
            CPPUNIT_ASSERT_EQUAL(actual, actualPartition);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 0.01 * std::fabs(expected));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(objectiveAtExpected, objective(actual),
                                         1e-5 * objectiveAtExpected);
        }
    }
}

void CBoostedTreeTest::testLogisticRegression() {

    // The idea of this test is to create a random linear relationship between
    // the feature values and the log-odds of class 1, i.e.
    //
    //   log-odds(class_1) = sum_i{ w * x_i }
    //
    // where, w is some fixed weight vector and x_i denoted the i'th feature vector.
    // We try to recover this relationship in logistic regression by observing
    // the actual labels. We want to test that we've roughly correctly estimated the
    // log-odds function. However, we target the cross-entropy so the error in our
    // estimates p_i^ should be measured in terms of cross entropy: sum_i{ p_i log(p_i^) }
    // where p_i = logistic(sum_i{ w_i * x_i}).

    test::CRandomNumbers rng;

    std::size_t trainRows{1000};
    std::size_t rows{1200};
    std::size_t cols{4};
    std::size_t capacity{600};

    TMeanAccumulator meanExcessCrossEntropy;
    for (std::size_t test = 0; test < 3; ++test) {
        TDoubleVec weights;
        rng.generateUniformSamples(-2.0, 2.0, cols - 1, weights);
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 1.0, rows, noise);
        TDoubleVec uniform01;
        rng.generateUniformSamples(0.0, 1.0, rows, uniform01);

        auto probability = [&](const TRowRef& row) {
            double x{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                x += weights[i] * row[i];
            }
            return maths::CTools::logisticFunction(x + noise[row.index()]);
        };

        auto target = [&](const TRowRef& row) {
            return uniform01[row.index()] < probability(row) ? 1.0 : 0.0;
        };

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            rng.generateUniformSamples(0.0, 4.0, rows, x[i]);
        }

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(trainRows, rows - trainRows, cols, {false, false, false, true},
                      x, TDoubleVec(rows, 0.0), target, *frame);

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(1).buildFor(
            *frame, std::make_unique<maths::boosted_tree::CLogistic>(), cols - 1);

        regression->train();
        regression->predict();

        double actualCrossEntropy{0.0};
        double minimumCrossEntropy{0.0};
        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (row->index() >= trainRows) {
                    std::size_t index{
                        regression->columnHoldingPrediction(row->numberColumns())};
                    actualCrossEntropy -=
                        probability(*row) *
                        std::log(maths::CTools::logisticFunction((*row)[index]));
                    minimumCrossEntropy -= probability(*row) *
                                           std::log(probability(*row));
                }
            }
        });
        LOG_DEBUG(<< "actual cross entropy = " << actualCrossEntropy
                  << ", minimum cross entropy = " << minimumCrossEntropy);

        // We should be with 40% of the minimum possible cross entropy.
        CPPUNIT_ASSERT(actualCrossEntropy < 1.4 * minimumCrossEntropy);
        meanExcessCrossEntropy.add(actualCrossEntropy / minimumCrossEntropy);
    }

    LOG_DEBUG(<< "mean excess cross entropy = "
              << maths::CBasicStatistics::mean(meanExcessCrossEntropy));

    // We should be within 25% of the minimum possible cross entropy on average.
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanExcessCrossEntropy) < 1.25);
}

void CBoostedTreeTest::testEstimateMemoryUsedByTrain() {

    // Test estimation of the memory used training a model.

    test::CRandomNumbers rng;

    std::size_t rows{1000};
    std::size_t cols{6};
    std::size_t capacity{600};

    for (std::size_t test = 0; test < 3; ++test) {
        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
        }

        auto target = [&](std::size_t i) {
            double result{0.0};
            for (std::size_t j = 0; j < cols - 1; ++j) {
                result += x[j][i];
            }
            return result;
        };

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;
        frame->categoricalColumns(TBoolVec{true, false, false, false, false, false});
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                *(column++) = std::floor(x[0][i]);
                for (std::size_t j = 1; j < cols - 1; ++j, ++column) {
                    *column = x[j][i];
                }
                *column = target(i);
            });
        }
        frame->finishWritingRows();

        std::int64_t estimatedMemory(
            maths::CBoostedTreeFactory::constructFromParameters(1).estimateMemoryUsage(
                rows, cols));

        std::int64_t memoryUsage{0};
        std::int64_t maxMemoryUsage{0};
        auto regression =
            maths::CBoostedTreeFactory::constructFromParameters(1)
                .memoryUsageCallback([&](std::int64_t delta) {
                    memoryUsage += delta;
                    maxMemoryUsage = std::max(maxMemoryUsage, memoryUsage);
                    LOG_TRACE(<< "current memory = " << memoryUsage
                              << ", high water mark = " << maxMemoryUsage);
                })
                .buildFor(*frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);

        regression->train();

        LOG_DEBUG(<< "estimated memory usage = " << estimatedMemory);
        LOG_DEBUG(<< "high water mark = " << maxMemoryUsage);

        CPPUNIT_ASSERT(maxMemoryUsage < estimatedMemory);
    }
}

void CBoostedTreeTest::testProgressMonitoring() {

    // Test progress monitoring invariants.

    test::CRandomNumbers rng;

    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{600};

    TDoubleVecVec x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    auto target = [](const TRowRef& row) { return row[0] + 3.0 * row[3]; };

    core::stopDefaultAsyncExecutor();

    std::string tests[]{"serial", "parallel"};

    for (std::size_t threads : {1, 2}) {

        LOG_DEBUG(<< tests[threads == 1 ? 0 : 1]);

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(rows, 0, cols, x, TDoubleVec(rows, 0.0), target, *frame);

        std::atomic_int totalFractionalProgress{0};

        auto reportProgress = [&totalFractionalProgress](double fractionalProgress) {
            totalFractionalProgress.fetch_add(
                static_cast<int>(65536.0 * fractionalProgress + 0.5));
        };

        std::atomic_bool finished{false};

        std::thread worker{[&]() {
            auto regression =
                maths::CBoostedTreeFactory::constructFromParameters(threads)
                    .progressCallback(reportProgress)
                    .buildFor(*frame, std::make_unique<maths::boosted_tree::CMse>(),
                              cols - 1);

            regression->train();
            finished.store(true);
        }};

        int lastTotalFractionalProgress{0};
        int lastProgressReport{0};

        bool monotonic{true};
        std::size_t percentage{0};
        while (finished.load() == false) {
            if (totalFractionalProgress.load() > lastProgressReport) {
                LOG_DEBUG(<< percentage << "% complete");
                percentage += 10;
                lastProgressReport += 6554;
            }
            monotonic &= (totalFractionalProgress.load() >= lastTotalFractionalProgress);
            lastTotalFractionalProgress = totalFractionalProgress.load();
        }
        worker.join();

        CPPUNIT_ASSERT(monotonic);

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void CBoostedTreeTest::testMissingData() {
}

void CBoostedTreeTest::testPersistRestore() {

    std::size_t rows{50};
    std::size_t cols{3};
    std::size_t capacity{50};
    test::CRandomNumbers rng;

    std::stringstream persistOnceSStream;
    std::stringstream persistTwiceSStream;

    // Generate completely random data.
    TDoubleVecVec x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j < cols; ++j, ++column) {
                *column = x[j][i];
            }
        });
    }
    frame->finishWritingRows();

    // persist
    {
        auto boostedTree =
            maths::CBoostedTreeFactory::constructFromParameters(1)
                .numberFolds(2)
                .maximumNumberTrees(2)
                .maximumOptimisationRoundsPerHyperparameter(3)
                .buildFor(*frame, std::make_unique<maths::boosted_tree::CMse>(), cols - 1);
        core::CJsonStatePersistInserter inserter(persistOnceSStream);
        boostedTree->acceptPersistInserter(inserter);
        persistOnceSStream.flush();
    }
    // restore
    auto boostedTree = maths::CBoostedTreeFactory::constructFromString(persistOnceSStream)
                           .restoreFor(*frame, cols - 1);
    {
        core::CJsonStatePersistInserter inserter(persistTwiceSStream);
        boostedTree->acceptPersistInserter(inserter);
        persistTwiceSStream.flush();
    }
    CPPUNIT_ASSERT_EQUAL(persistOnceSStream.str(), persistTwiceSStream.str());
    LOG_DEBUG(<< "First string " << persistOnceSStream.str());
    LOG_DEBUG(<< "Second string " << persistTwiceSStream.str());

    // and even run
    CPPUNIT_ASSERT_NO_THROW(boostedTree->train());
    CPPUNIT_ASSERT_NO_THROW(boostedTree->predict());

    // TODO test persist and restore produces same train result.
}

void CBoostedTreeTest::testRestoreErrorHandling() {

    auto errorHandler = [](std::string error) {
        throw std::runtime_error{error};
    };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    std::size_t cols{3};
    std::size_t capacity{50};

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    std::stringstream errorInBayesianOptimisationState;
    errorInBayesianOptimisationState
        << "{\"bayesian_optimization\":"
           "{\"min_boundary\":{\"dense_vector\":\"-9.191737e-1:-2.041179:-3.506558:1.025:2e-1\"},"
           "\"max_boundary\":{\"dense_vector\":\"3.685997:2.563991:-1.203973:a:8e-1\"},"
           "\"error_variances\":\"\",\"kernel_parameters\":{\"dense_vector\":\"1:1:1:1:1:1\"},"
           "\"min_kernel_coordinate_distance_scales\":{\"dense_vector\":\"1e-3:1e-3:1e-3:1e-3:1e-3\"},"
           "\"function_mean_values\":{\"d\":\"0\"}},\"best_forest_test_loss\":\"1.797693e308\","
           "\"current_round\":\"0\",\"dependent_variable\":\"2\",\"eta_growth_rate_per_tree\":\"1.05\","
           "\"eta\":\"1e-1\",\"feature_bag_fraction\":\"5e-1\",\"feature_sample_probabilities\":\"1:0:0\","
           "\"gamma\":\"1.298755\",\"lambda\":\"3.988485\",\"maximum_attempts_to_add_tree\":\"3\","
           "\"maximum_optimisation_rounds_per_hyperparameter\":\"3\",\"maximum_tree_size_fraction\":\"10\","
           "\"missing_feature_row_masks\":{\"d\":\"3\",\"a\":\"50:0:1:50\",\"a\":\"50:0:1:50\",\"a\":\"50:0:1:50\"},"
           "\"number_folds\":\"2\",\"number_rounds\":\"15\",\"number_splits_per_feature\":\"40\","
           "\"number_threads\":\"1\",\"rows_per_feature\":\"50\","
           "\"testing_row_masks\":{\"d\":\"2\",\"a\":\"50:1:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\","
           "\"a\":\"50:0:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\"},\"maximum_number_trees\":\"2\","
           "\"training_row_masks\":{\"d\":\"2\",\"a\":\"50:0:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\","
           "\"a\":\"50:1:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\"},\"best_forest\":{\"d\":\"0\"},"
           "\"best_hyperparameters\":{\"hyperparam_lambda\":\"0\",\"hyperparam_gamma\":\"0\","
           "\"hyperparam_eta\":\"0\",\"hyperparam_eta_growth_rate_per_tree\":\"0\","
           "\"hyperparam_feature_bag_fraction\":\"0\",\"hyperparam_feature_sample_probabilities\":\"\"},"
           "\"eta_override\":\"false;0\",\"feature_bag_fraction_override\":\"false;0\",\"gamma_override\":\"false;0\","
           "\"lambda_override\":\"false;0\",\"maximum_number_trees_override\":\"true;2\",\"loss\":\"mse\"}";
    errorInBayesianOptimisationState.flush();

    bool throwsExceptions{false};
    try {
        auto boostedTree = maths::CBoostedTreeFactory::constructFromString(errorInBayesianOptimisationState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        LOG_DEBUG(<< "got = " << e.what());
        throwsExceptions = true;
        core::CRegex re;
        re.init("Input error:.*");
        CPPUNIT_ASSERT(re.matches(e.what()));
    }
    CPPUNIT_ASSERT(throwsExceptions);

    std::stringstream errorInBoostedTreeImplState;
    errorInBoostedTreeImplState
        << "{\"bayesian_optimization\":"
           "{\"min_boundary\":{\"dense_vector\":\"-9.191737e-1:-2.041179:-3.506558:1.025:2e-1\"},"
           "\"max_boundary\":{\"dense_vector\":\"3.685997:2.563991:-1.203973:0.1:8e-1\"},"
           "\"error_variances\":\"\",\"kernel_parameters\":{\"dense_vector\":\"1:1:1:1:1:1\"},"
           "\"min_kernel_coordinate_distance_scales\":{\"dense_vector\":\"1e-3:1e-3:1e-3:1e-3:1e-3\"},"
           "\"function_mean_values\":{\"d\":\"0\"}},\"best_forest_test_loss\":\"1.797693e308\","
           "\"current_round\":\"0\",\"dependent_variable\":\"2\",\"eta_growth_rate_per_tree\":\"1.05\","
           "\"eta\":\"1e-1\",\"feature_bag_fraction\":\"5e-1\",\"feature_sample_probabilities\":\"1:0:0\","
           "\"gamma\":\"1.298755\",\"lambda\":\"3.988485\",\"maximum_attempts_to_add_tree\":\"3\","
           "\"maximum_optimisation_rounds_per_hyperparameter\":\"3\",\"maximum_tree_size_fraction\":\"10\","
           "\"missing_feature_row_masks\":{\"d\":\"3\",\"a\":\"50:0:1:50\",\"a\":\"50:0:1:50\",\"a\":\"50:0:1:50\"},"
           "\"number_folds\":\"\",\"number_rounds\":\"15\",\"number_splits_per_feature\":\"40\","
           "\"number_threads\":\"1\",\"rows_per_feature\":\"50\","
           "\"testing_row_masks\":{\"d\":\"2\",\"a\":\"50:1:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\","
           "\"a\":\"50:0:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\"},\"maximum_number_trees\":\"2\","
           "\"training_row_masks\":{\"d\":\"2\",\"a\":\"50:0:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\","
           "\"a\":\"50:1:1:5:1:1:5:3:3:3:1:1:1:1:4:1:4:3:6:1:1:2:1:2\"},\"best_forest\":{\"d\":\"0\"},"
           "\"best_hyperparameters\":{\"hyperparam_lambda\":\"0\",\"hyperparam_gamma\":\"0\","
           "\"hyperparam_eta\":\"0\",\"hyperparam_eta_growth_rate_per_tree\":\"0\","
           "\"hyperparam_feature_bag_fraction\":\"0\",\"hyperparam_feature_sample_probabilities\":\"\"},"
           "\"eta_override\":\"false;0\",\"feature_bag_fraction_override\":\"false;0\",\"gamma_override\":\"false;0\","
           "\"lambda_override\":\"false;0\",\"maximum_number_trees_override\":\"true;2\",\"loss\":\"mse\"}";
    errorInBoostedTreeImplState.flush();

    throwsExceptions = false;
    try {
        auto boostedTree = maths::CBoostedTreeFactory::constructFromString(errorInBoostedTreeImplState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        LOG_DEBUG(<< "got = " << e.what());
        throwsExceptions = true;
        core::CRegex re;
        re.init("Input error:.*");
        CPPUNIT_ASSERT(re.matches(e.what()));
    }
    CPPUNIT_ASSERT(throwsExceptions);
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
        "CBoostedTreeTest::testThreading", &CBoostedTreeTest::testThreading));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testConstantFeatures", &CBoostedTreeTest::testConstantFeatures));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testConstantTarget", &CBoostedTreeTest::testConstantTarget));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testCategoricalRegressors",
        &CBoostedTreeTest::testCategoricalRegressors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testIntegerRegressor", &CBoostedTreeTest::testIntegerRegressor));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testSingleSplit", &CBoostedTreeTest::testSingleSplit));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testTranslationInvariance",
        &CBoostedTreeTest::testTranslationInvariance));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testDepthBasedRegularization",
        &CBoostedTreeTest::testDepthBasedRegularization));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testLogisticMinimizer", &CBoostedTreeTest::testLogisticMinimizer));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testLogisticRegression", &CBoostedTreeTest::testLogisticRegression));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testEstimateMemoryUsedByTrain",
        &CBoostedTreeTest::testEstimateMemoryUsedByTrain));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testProgressMonitoring", &CBoostedTreeTest::testProgressMonitoring));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testMissingData", &CBoostedTreeTest::testMissingData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testPersistRestore", &CBoostedTreeTest::testPersistRestore));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testRestoreErrorHandling", &CBoostedTreeTest::testRestoreErrorHandling));

    return suiteOfTests;
}
