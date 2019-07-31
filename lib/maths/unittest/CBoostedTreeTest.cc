/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBoostedTreeTest.h"

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>

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
auto predictionStatistics(test::CRandomNumbers& rng,
                          std::size_t trainRows,
                          std::size_t testRows,
                          std::size_t cols,
                          std::size_t capacity,
                          const F& generateFunction,
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

        auto f = generateFunction(rng, cols);

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
                                           ? f(*row) + noise[row->index()]
                                           : std::numeric_limits<double>::quiet_NaN()};
                    row->writeColumn(cols - 1, targetValue);
                }
            });

            auto regression =
                maths::CBoostedTreeFactory::constructFromParameters(
                    1, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
                    .buildFor(*frame);

            regression->train();
            regression->predict();

            TMeanVarAccumulator functionMoments;
            TMeanVarAccumulator modelPredictionErrorMoments;

            frame->readRows(1, trainRows, rows, [&](TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    std::size_t index{
                        regression->columnHoldingPrediction(row->numberColumns())};
                    functionMoments.add(f(*row));
                    modelPredictionErrorMoments.add(f(*row) - (*row)[index]);
                }
            });

            LOG_DEBUG(<< "function moments = " << functionMoments);
            LOG_DEBUG(<< "model prediction error moments = " << modelPredictionErrorMoments);

            double functionVariance{maths::CBasicStatistics::variance(functionMoments)};
            double predictionErrorMean{maths::CBasicStatistics::mean(modelPredictionErrorMoments)};
            double predictionErrorVariance{
                maths::CBasicStatistics::variance(modelPredictionErrorMoments)};

            modelBias[test].push_back(predictionErrorMean);
            modelRSquared[test].push_back(
                1.0 -
                (predictionErrorVariance - noiseVariance / static_cast<double>(rows)) /
                    (functionVariance - noiseVariance / static_cast<double>(rows)));
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

    std::tie(modelBias, modelRSquared) = predictionStatistics(
        rng, trainRows, testRows, cols, capacity, generatePiecewiseConstant, noiseVariance);

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
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.92);
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

    std::tie(modelBias, modelRSquared) = predictionStatistics(
        rng, trainRows, testRows, cols, capacity, generateLinear, noiseVariance);

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

    std::tie(modelBias, modelRSquared) = predictionStatistics(
        rng, trainRows, testRows, cols, capacity, generateNonLinear, noiseVariance);

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
        CPPUNIT_ASSERT(modelRSquared[i][0] > 0.92);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanModelRSquared) > 0.95);
}

void CBoostedTreeTest::testThreading() {

    // Test we get the same results whether we thread the code or not. Note
    // because we compute approximate quantiles for each thread and merge we
    // get slightly different results if threaded vs single threaded. However,
    // we should get the same results whether executed asynchronously or not
    // so test with and without starting the thread pool.

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
                              2, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
                              .buildFor(*frame);

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

    std::unique_ptr<maths::CBoostedTree> regression =
        maths::CBoostedTreeFactory::constructFromParameters(
            1, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
            .buildFor(*frame);

    regression->train();

    TDoubleVec featureWeights(regression->featureWeights());

    LOG_DEBUG(<< "feature weights = " << core::CContainerPrinter::print(featureWeights));
    CPPUNIT_ASSERT(featureWeights[cols - 2] < 1e-4);
}

void CBoostedTreeTest::testConstantObjective() {

    // Test we correctly deal with a constant dependent variable.

    test::CRandomNumbers rng;

    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{500};

    TDoubleVecVec x(cols - 1, TDoubleVec(rows, 1.0));
    for (std::size_t i = 0; i < cols - 2; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

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
            row->writeColumn(cols - 1, 1.0);
        }
    });

    std::unique_ptr<maths::CBoostedTree> regression =
        maths::CBoostedTreeFactory::constructFromParameters(
            1, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
            .buildFor(*frame);

    regression->train();

    TMeanVarAccumulator modelPredictionErrorMoments;

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            std::size_t index{regression->columnHoldingPrediction(row->numberColumns())};
            modelPredictionErrorMoments.add(1.0 - (*row)[index]);
        }
    });

    LOG_DEBUG(<< "mean prediction error = "
              << maths::CBasicStatistics::mean(modelPredictionErrorMoments));
    CPPUNIT_ASSERT_EQUAL(0.0, maths::CBasicStatistics::mean(modelPredictionErrorMoments));
}

void CBoostedTreeTest::testMissingData() {
}

void CBoostedTreeTest::testErrors() {
}

void CBoostedTreeTest::testPersistRestore() {
    std::size_t rows{50};
    std::size_t cols{3};
    std::size_t capacity{50};
    test::CRandomNumbers rng;

    std::stringstream persistOnceSStream;
    std::stringstream persistTwiceSStream;

    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    TDoubleVecVec x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }
    auto frame = makeMainMemory();

    // generate completely random data
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
                               1, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
                               .numberFolds(2)
                               .maximumNumberTrees(2)
                               .maximumOptimisationRoundsPerHyperparameter(3)
                               .buildFor(*frame);
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
        "CBoostedTreeTest::testConstantObjective", &CBoostedTreeTest::testConstantObjective));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testMissingData", &CBoostedTreeTest::testMissingData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testErrors", &CBoostedTreeTest::testErrors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testPersistRestore", &CBoostedTreeTest::testPersistRestore));

    return suiteOfTests;
}
