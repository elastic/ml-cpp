/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBoostedTreeTest.h"

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>

#include <boost/filesystem.hpp>

#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <functional>
#include <memory>
#include <utility>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;
using TRowRef = core::CDataFrame::TRowRef;
using TRowItr = core::CDataFrame::TRowItr;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

namespace {

template<typename F>
auto predictionStatistics(test::CRandomNumbers& rng,
                          std::size_t rows,
                          std::size_t cols,
                          std::size_t capacity,
                          const F& generateFunction,
                          double noiseVariance) {

    TFactoryFunc makeOnDisk{[=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows, capacity)
            .first;
    }};
    TFactoryFunc makeMainMemory{
        [=] { return core::makeMainStorageDataFrame(cols, capacity).first; }};

    TDoubleVec modelPredictionBias;
    TDoubleVec modelPredictionMseImprovement;

    for (std::size_t test = 0; test < 3; ++test) {

        auto f = generateFunction(rng, cols);

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

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

            std::unique_ptr<maths::CBoostedTree> regression =
                maths::CBoostedTreeFactory::constructFromParameters(
                    1, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
                    .frame(*frame);

            regression->train(*frame);
            regression->predict(*frame);

            TMeanVarAccumulator functionMoments;
            TMeanVarAccumulator modelPredictionErrorMoments;

            frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
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

            modelPredictionBias.push_back(predictionErrorMean);
            modelPredictionMseImprovement.push_back(
                (functionVariance - noiseVariance / static_cast<double>(rows)) /
                (predictionErrorVariance - noiseVariance / static_cast<double>(rows)));
        }
    }

    return std::make_pair(modelPredictionBias, modelPredictionMseImprovement);
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

    TDoubleVec modelPredictionBias;
    TDoubleVec modelPredictionMseImprovement;

    test::CRandomNumbers rng;
    double noiseVariance{0.2};
    std::size_t rows{1000};
    std::size_t cols{6};
    std::size_t capacity{250};

    std::tie(modelPredictionBias, modelPredictionMseImprovement) = predictionStatistics(
        rng, rows, cols, capacity, generatePiecewiseConstant, noiseVariance);

    TMeanAccumulator meanMseImprovement;

    for (std::size_t i = 1; i < modelPredictionBias.size(); i += 2) {

        // In and out-of-core agree.
        CPPUNIT_ASSERT_EQUAL(modelPredictionBias[i], modelPredictionBias[i - 1]);
        CPPUNIT_ASSERT_EQUAL(modelPredictionMseImprovement[i],
                             modelPredictionMseImprovement[i - 1]);

        // Unbiased...
        CPPUNIT_ASSERT(modelPredictionBias[i] <
                       2.5 * std::sqrt(noiseVariance / static_cast<double>(rows)));
        // Good reduction in MSE...
        CPPUNIT_ASSERT(modelPredictionMseImprovement[i] > 13.0);

        meanMseImprovement.add(modelPredictionMseImprovement[i]);
    }
    LOG_DEBUG(<< "mean MSE improvement = "
              << maths::CBasicStatistics::mean(meanMseImprovement));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMseImprovement) > 20.0);
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

    TDoubleVec modelPredictionBias;
    TDoubleVec modelPredictionMseImprovement;

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t rows{1000};
    std::size_t cols{6};
    std::size_t capacity{500};

    std::tie(modelPredictionBias, modelPredictionMseImprovement) =
        predictionStatistics(rng, rows, cols, capacity, generateLinear, noiseVariance);

    TMeanAccumulator meanMseImprovement;

    for (std::size_t i = 1; i < modelPredictionBias.size(); i += 2) {

        // In and out-of-core agree.
        CPPUNIT_ASSERT_EQUAL(modelPredictionBias[i], modelPredictionBias[i - 1]);
        CPPUNIT_ASSERT_EQUAL(modelPredictionMseImprovement[i],
                             modelPredictionMseImprovement[i - 1]);

        // Unbiased...
        CPPUNIT_ASSERT(std::fabs(modelPredictionBias[i]) <
                       2.5 * std::sqrt(noiseVariance / static_cast<double>(rows)));
        // Good reduction in MSE...
        CPPUNIT_ASSERT(modelPredictionMseImprovement[i] > 20.0);

        meanMseImprovement.add(modelPredictionMseImprovement[i]);
    }
    LOG_DEBUG(<< "mean MSE improvement = "
              << maths::CBasicStatistics::mean(meanMseImprovement));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMseImprovement) > 35.0);
}

void CBoostedTreeTest::testNonLinear() {

    // Test regression quality on non-linear function.

    auto generateNonLinear = [](test::CRandomNumbers& rng, std::size_t cols) {
        TDoubleVec m;
        TDoubleVec s;
        TDoubleVec c;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        rng.generateUniformSamples(-1.0, 1.0, cols - 1, c);

        return [m, s, c, cols](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + (s[i] + c[i] * row[i]) * row[i];
            }
            return result;
        };
    };

    TDoubleVec modelPredictionBias;
    TDoubleVec modelPredictionMseImprovement;

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t rows{500};
    std::size_t cols{6};
    std::size_t capacity{500};

    std::tie(modelPredictionBias, modelPredictionMseImprovement) =
        predictionStatistics(rng, rows, cols, capacity, generateNonLinear, noiseVariance);

    TMeanAccumulator meanMseImprovement;

    for (std::size_t i = 1; i < modelPredictionBias.size(); i += 2) {

        // In and out-of-core agree.
        CPPUNIT_ASSERT_EQUAL(modelPredictionBias[i], modelPredictionBias[i - 1]);
        CPPUNIT_ASSERT_EQUAL(modelPredictionMseImprovement[i],
                             modelPredictionMseImprovement[i - 1]);

        // Unbiased...
        CPPUNIT_ASSERT(std::fabs(modelPredictionBias[i]) <
                       2.5 * std::sqrt(noiseVariance / static_cast<double>(rows)));
        // Good reduction in MSE...
        CPPUNIT_ASSERT(modelPredictionMseImprovement[i] > 30.0);

        meanMseImprovement.add(modelPredictionMseImprovement[i]);
    }
    LOG_DEBUG(<< "mean MSE improvement = "
              << maths::CBasicStatistics::mean(meanMseImprovement));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMseImprovement) > 50.0);
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

    TDoubleVec modelPredictionBias;
    TDoubleVec modelPredictionMse;

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

        std::unique_ptr<maths::CBoostedTree> regression =
            maths::CBoostedTreeFactory::constructFromParameters(
                2, cols - 1, std::make_unique<maths::boosted_tree::CMse>())
                .frame(*frame);

        regression->train(*frame);
        regression->predict(*frame);

        TMeanVarAccumulator modelPredictionErrorMoments;

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                std::size_t index{regression->columnHoldingPrediction(row->numberColumns())};
                modelPredictionErrorMoments.add(f(*row) - (*row)[index]);
            }
        });

        LOG_DEBUG(<< "model prediction error moments = " << modelPredictionErrorMoments);

        modelPredictionBias.push_back(maths::CBasicStatistics::mean(modelPredictionErrorMoments));
        modelPredictionMse.push_back(
            maths::CBasicStatistics::variance(modelPredictionErrorMoments));

        core::startDefaultAsyncExecutor();
    }

    CPPUNIT_ASSERT_EQUAL(modelPredictionBias[0], modelPredictionBias[1]);
    CPPUNIT_ASSERT_EQUAL(modelPredictionMse[0], modelPredictionMse[1]);

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
            .frame(*frame);

    regression->train(*frame);

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
            .frame(*frame);

    regression->train(*frame);

    TMeanVarAccumulator modelPredictionErrorMoments;

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            std::size_t index{regression->columnHoldingPrediction(row->numberColumns())};
            modelPredictionErrorMoments.add(1.0 - (*row)[index]);
        }
    });

    LOG_DEBUG(<< maths::CBasicStatistics::mean(modelPredictionErrorMoments));
    // TODO using eta < 1 in this case causes bias. Trap earlier?
    //CPPUNIT_ASSERT_EQUAL(0.0, maths::CBasicStatistics::mean(modelPredictionErrorMoments));
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
        "CBoostedTreeTest::testThreading", &CBoostedTreeTest::testThreading));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testConstantFeatures", &CBoostedTreeTest::testConstantFeatures));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testConstantObjective", &CBoostedTreeTest::testConstantObjective));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testMissingData", &CBoostedTreeTest::testMissingData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeTest>(
        "CBoostedTreeTest::testErrors", &CBoostedTreeTest::testErrors));

    return suiteOfTests;
}
