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

#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <functional>
#include <memory>
#include <utility>

using namespace ml;

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
                    double targetValue{row->index() < trainRows
                                           ? target(*row) + noise[row->index()]
                                           : core::CDataFrame::valueOfMissing()};
                    row->writeColumn(cols - 1, targetValue);
                }
            });

            auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                                  1, std::make_unique<maths::boosted_tree::CMse>())
                                  .buildFor(*frame, cols - 1);

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
            7.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.9);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.93);
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
            7.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.95);

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
            6.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.92);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.95);
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

    TDoubleVec m;
    TDoubleVec s;
    rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
    rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);

    auto f = [m, s, cols](const TRowRef& row) {
        double result{0.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += m[i] + s[i] * row[i];
        }
        return result;
    };

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

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                              2, std::make_unique<maths::boosted_tree::CMse>())
                              .buildFor(*frame, cols - 1);

        regression->train();
        regression->predict();

        TMeanVarAccumulator modelPredictionErrorMoments;

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                std::size_t index{regression->columnHoldingPrediction(row->numberColumns())};
                modelPredictionErrorMoments.add(f(*row) - (*row)[index]);
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

    TDoubleVec m;
    TDoubleVec s;
    rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
    rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);

    auto f = [m, s, cols](const TRowRef& row) {
        double result{0.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += m[i] + s[i] * row[i];
        }
        return result;
    };

    TDoubleVecVec x(cols - 1, TDoubleVec(rows, 1.0));
    for (std::size_t i = 0; i < cols - 2; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 0.1, rows, noise);

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

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

    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j < x.size(); ++j, ++column) {
                *column = x[j][i];
            }
        });
    }
    frame->finishWritingRows();
    frame->writeColumns(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            row->writeColumn(cols - 1, 1.0);
        }
    });

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

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

    frame->categoricalColumns({true, true, false, false, false, false});
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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        regression->columnHoldingPrediction(frame->numberColumns()), target, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, modelBias, 0.15);
    CPPUNIT_ASSERT(modelRSquared > 0.9);
}

void CBoostedTreeTest::testIntegerRegressor() {

    // Test a simple integer regressor.

    test::CRandomNumbers rng;

    std::size_t trainRows{1000};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};

    auto frame = core::makeMainStorageDataFrame(2).first;

    frame->categoricalColumns({false, false});
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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, 1);

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

void CBoostedTreeTest::testEstimateMemoryUsedByTrain() {

    // Test estimation of the memory used training a model.

    test::CRandomNumbers rng;

    std::size_t rows{1000};
    std::size_t cols{6};
    std::size_t capacity{600};

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
    frame->categoricalColumns({true, false, false, false, false, false});
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

    std::int64_t estimatedMemory(maths::CBoostedTreeFactory::constructFromParameters(
                                     1, std::make_unique<maths::boosted_tree::CMse>())
                                     .estimateMemoryUsage(rows, cols));

    std::int64_t memoryUsage{0};
    std::int64_t maxMemoryUsage{0};
    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .memoryUsageCallback([&](std::int64_t delta) {
                              memoryUsage += delta;
                              maxMemoryUsage = std::max(maxMemoryUsage, memoryUsage);
                              LOG_TRACE(<< "current memory = " << memoryUsage
                                        << ", high water mark = " << maxMemoryUsage);
                          })
                          .buildFor(*frame, cols - 1);

    regression->train();

    LOG_DEBUG(<< "estimated memory usage = " << estimatedMemory);
    LOG_DEBUG(<< "high water mark = " << maxMemoryUsage);

    // Currently, the estimated memory is a little over 3 times the high water
    // mark for the test data.

    CPPUNIT_ASSERT(maxMemoryUsage < estimatedMemory);
    CPPUNIT_ASSERT(4 * maxMemoryUsage > estimatedMemory);
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

    core::stopDefaultAsyncExecutor();

    std::string tests[]{"serial", "parallel"};

    for (std::size_t threads : {1, 2}) {

        LOG_DEBUG(<< tests[threads == 1 ? 0 : 1]);

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j < cols; ++j, ++column) {
                    *column = x[j][i];
                }
            });
        }
        frame->finishWritingRows();

        std::atomic_int totalFractionalProgress{0};

        auto reportProgress = [&totalFractionalProgress](double fractionalProgress) {
            totalFractionalProgress.fetch_add(
                static_cast<int>(65536.0 * fractionalProgress + 0.5));
        };

        std::atomic_bool finished{false};

        std::thread worker{[&]() {
            auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                                  threads, std::make_unique<maths::boosted_tree::CMse>())
                                  .progressCallback(reportProgress)
                                  .buildFor(*frame, cols - 1);

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
        auto boostedTree = maths::CBoostedTreeFactory::constructFromParameters(
                               1, std::make_unique<maths::boosted_tree::CMse>())
                               .numberFolds(2)
                               .maximumNumberTrees(2)
                               .maximumOptimisationRoundsPerHyperparameter(3)
                               .buildFor(*frame, cols - 1);
        core::CJsonStatePersistInserter inserter(persistOnceSStream);
        boostedTree->acceptPersistInserter(inserter);
        persistOnceSStream.flush();
    }
    // restore
    auto boostedTree =
        maths::CBoostedTreeFactory::constructFromString(persistOnceSStream, *frame);
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
        << "{\"boosted_tree_impl\":{\"bayesian_optimization\":"
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
           "\"lambda_override\":\"false;0\",\"maximum_number_trees_override\":\"true;2\",\"loss\":\"mse\"}}";
    errorInBayesianOptimisationState.flush();

    bool throwsExceptions{false};
    try {
        auto boostedTree = maths::CBoostedTreeFactory::constructFromString(
            errorInBayesianOptimisationState, *frame);
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
        << "{\"boosted_tree_impl\":{\"bayesian_optimization\":"
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
           "\"lambda_override\":\"false;0\",\"maximum_number_trees_override\":\"true;2\",\"loss\":\"mse\"}}";
    errorInBoostedTreeImplState.flush();

    throwsExceptions = false;
    try {
        auto boostedTree = maths::CBoostedTreeFactory::constructFromString(
            errorInBoostedTreeImplState, *frame);
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
