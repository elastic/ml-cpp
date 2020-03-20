/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CRegex.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <boost/make_shared.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <streambuf>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CBoostedTreeTest)

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

class CTestInstrumentation : public maths::CDataFrameTrainBoostedTreeInstrumentationStub {
public:
    using TIntVec = std::vector<int>;

public:
    CTestInstrumentation()
        : m_TotalFractionalProgress{0}, m_MemoryUsage{0}, m_MaxMemoryUsage{0} {}

    int progress() const {
        return (100 * m_TotalFractionalProgress.load()) / 65536;
    }
    TIntVec tenPercentProgressPoints() const {
        return m_TenPercentProgressPoints;
    }
    std::int64_t maxMemoryUsage() const { return m_MaxMemoryUsage.load(); }

    void updateProgress(double fractionalProgress) override {
        int progress{m_TotalFractionalProgress.fetch_add(
            static_cast<int>(65536.0 * fractionalProgress + 0.5))};
        // This needn't be protected because progress is only written from one thread and
        // the tests arrange that it is never read at the same time it is being written.
        if (m_TenPercentProgressPoints.empty() ||
            100 * progress > 65536 * (m_TenPercentProgressPoints.back() + 10)) {
            m_TenPercentProgressPoints.push_back(10 * ((10 * progress) / 65536));
        }
    }

    void updateMemoryUsage(std::int64_t delta) override {
        std::int64_t memory{m_MemoryUsage.fetch_add(delta)};
        std::int64_t previousMaxMemoryUsage{m_MaxMemoryUsage.load(std::memory_order_relaxed)};
        while (previousMaxMemoryUsage < memory &&
               m_MaxMemoryUsage.compare_exchange_weak(previousMaxMemoryUsage, memory) == false) {
        }
        LOG_TRACE(<< "current memory = " << m_MemoryUsage.load()
                  << ", high water mark = " << m_MaxMemoryUsage.load());
    }

private:
    std::atomic_int m_TotalFractionalProgress;
    TIntVec m_TenPercentProgressPoints;
    std::atomic<std::int64_t> m_MemoryUsage;
    std::atomic<std::int64_t> m_MaxMemoryUsage;
};

template<typename F, typename G>
auto computeEvaluationMetrics(const core::CDataFrame& frame,
                              std::size_t beginTestRows,
                              std::size_t endTestRows,
                              const F& actual,
                              const G& target,
                              double noiseVariance) {

    TMeanVarAccumulator functionMoments;
    TMeanVarAccumulator modelPredictionErrorMoments;

    frame.readRows(1, beginTestRows, endTestRows, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            functionMoments.add(target(*row));
            modelPredictionErrorMoments.add(target(*row) - actual(*row));
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

            auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                                  1, std::make_unique<maths::boosted_tree::CMse>())
                                  .buildFor(*frame, cols - 1);

            regression->train();
            regression->predict();

            double bias;
            double rSquared;
            std::tie(bias, rSquared) = computeEvaluationMetrics(
                *frame, trainRows, rows,
                [&](const TRowRef& row) {
                    return regression->readPrediction(row)[0];
                },
                target, noiseVariance / static_cast<double>(rows));
            modelBias[test].push_back(bias);
            modelRSquared[test].push_back(rSquared);
        }
    }
    LOG_DEBUG(<< "bias = " << core::CContainerPrinter::print(modelBias));
    LOG_DEBUG(<< " R^2 = " << core::CContainerPrinter::print(modelRSquared));

    return std::make_pair(std::move(modelBias), std::move(modelRSquared));
}

void readFileToStream(const std::string& filename, std::stringstream& stream) {
    std::ifstream file(filename);
    BOOST_TEST_REQUIRE(file.is_open());
    std::string str((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    stream << str;
    stream.flush();
}
}

BOOST_AUTO_TEST_CASE(testPiecewiseConstant) {

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
            BOOST_REQUIRE_EQUAL(modelBias[i][0], modelBias[i][j]);
            BOOST_REQUIRE_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Unbiased...
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, modelBias[i][0],
            9.1 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i][0] > 0.97);

        meanModelRSquared.add(modelRSquared[i][0]);
    }

    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanModelRSquared) > 0.98);
}

BOOST_AUTO_TEST_CASE(testLinear) {

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
            BOOST_REQUIRE_EQUAL(modelBias[i][0], modelBias[i][j]);
            BOOST_REQUIRE_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Unbiased...
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, modelBias[i][0],
            4.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i][0] > 0.97);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanModelRSquared) > 0.98);
}

BOOST_AUTO_TEST_CASE(testNonLinear) {

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
            BOOST_REQUIRE_EQUAL(modelBias[i][0], modelBias[i][j]);
            BOOST_REQUIRE_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Unbiased...
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, modelBias[i][0],
            5.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i][0] > 0.97);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanModelRSquared) > 0.98);
}

BOOST_AUTO_TEST_CASE(testThreading) {

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

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                              2, std::make_unique<maths::boosted_tree::CMse>())
                              .buildFor(*frame, cols - 1);

        regression->train();
        regression->predict();

        TMeanVarAccumulator modelPredictionErrorMoments;

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                modelPredictionErrorMoments.add(
                    target(*row) - regression->readPrediction(*row)[0]);
            }
        });

        LOG_DEBUG(<< "model prediction error moments = " << modelPredictionErrorMoments);

        modelBias.push_back(maths::CBasicStatistics::mean(modelPredictionErrorMoments));
        modelMse.push_back(maths::CBasicStatistics::variance(modelPredictionErrorMoments));

        core::startDefaultAsyncExecutor();
    }

    BOOST_REQUIRE_EQUAL(modelBias[0], modelBias[1]);
    BOOST_REQUIRE_EQUAL(modelMse[0], modelMse[1]);

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testConstantFeatures) {

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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

    regression->train();

    TDoubleVec featureWeightsForTraining(regression->featureWeightsForTraining());

    LOG_DEBUG(<< "feature weights = "
              << core::CContainerPrinter::print(featureWeightsForTraining));
    BOOST_TEST_REQUIRE(featureWeightsForTraining[cols - 2] < 1e-4);
}

BOOST_AUTO_TEST_CASE(testConstantTarget) {

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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

    regression->train();

    TMeanAccumulator modelPredictionError;

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            modelPredictionError.add(1.0 - regression->readPrediction(*row)[0]);
        }
    });

    LOG_DEBUG(<< "mean prediction error = "
              << maths::CBasicStatistics::mean(modelPredictionError));
    BOOST_REQUIRE_EQUAL(0.0, maths::CBasicStatistics::mean(modelPredictionError));
}

BOOST_AUTO_TEST_CASE(testCategoricalRegressors) {

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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        [&](const TRowRef& row) { return regression->readPrediction(row)[0]; },
        target, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, modelBias, 0.08);
    BOOST_TEST_REQUIRE(modelRSquared > 0.98);
}

BOOST_AUTO_TEST_CASE(testIntegerRegressor) {

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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        [&](const TRowRef& row) { return regression->readPrediction(row)[0]; },
        [&](const TRowRef& x) { return 10.0 * x[0]; }, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, modelBias, 0.08);
    BOOST_TEST_REQUIRE(modelRSquared > 0.98);
}

BOOST_AUTO_TEST_CASE(testSingleSplit) {

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

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

    regression->train();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, 0, rows,
        [&](const TRowRef& row) { return regression->readPrediction(row)[0]; },
        [](const TRowRef& row) { return 10.0 * row[0]; }, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, modelBias, 0.21);
    BOOST_TEST_REQUIRE(modelRSquared > 0.97);
}

BOOST_AUTO_TEST_CASE(testTranslationInvariance) {

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

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                              1, std::make_unique<maths::boosted_tree::CMse>())
                              .buildFor(*frame, cols - 1);

        regression->train();
        regression->predict();

        double modelBias;
        double modelRSquared;
        std::tie(modelBias, modelRSquared) =
            computeEvaluationMetrics(*frame, trainRows, rows,
                                     [&](const TRowRef& row) {
                                         return regression->readPrediction(row)[0];
                                     },
                                     target_, 0.0);

        LOG_DEBUG(<< "bias = " << modelBias);
        LOG_DEBUG(<< " R^2 = " << modelRSquared);
        rsquared.push_back(modelRSquared);
    }

    BOOST_REQUIRE_CLOSE_ABSOLUTE(rsquared[0], rsquared[1], 0.01);
}

std::size_t maxDepth(const std::vector<maths::CBoostedTreeNode>& tree,
                     const maths::CBoostedTreeNode& node,
                     std::size_t depth) {
    std::size_t result{depth};
    if (node.isLeaf() == false) {
        result = std::max(result, maxDepth(tree, tree[node.leftChildIndex()], depth + 1));
        result = std::max(result, maxDepth(tree, tree[node.rightChildIndex()], depth + 1));
    }
    return result;
}

BOOST_AUTO_TEST_CASE(testDepthBasedRegularization) {

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

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                              1, std::make_unique<maths::boosted_tree::CMse>())
                              .treeSizePenaltyMultiplier(0.0)
                              .leafWeightPenaltyMultiplier(0.0)
                              .softTreeDepthLimit(targetDepth)
                              .softTreeDepthTolerance(0.01)
                              .buildFor(*frame, cols - 1);

        regression->train();

        TMeanAccumulator meanDepth;
        for (const auto& tree : regression->trainedModel()) {
            BOOST_TEST_REQUIRE(maxDepth(tree, tree[0], 0) <=
                               static_cast<std::size_t>(targetDepth));
            meanDepth.add(static_cast<double>(maxDepth(tree, tree[0], 0)));
        }
        LOG_DEBUG(<< "mean depth = " << maths::CBasicStatistics::mean(meanDepth));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanDepth) > targetDepth - 1.2);
    }
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticRegression) {

    // The idea of this test is to create a random linear relationship between
    // the feature values and the log-odds of class 1, i.e.
    //
    //   log-odds(class_1) = sum_i{ w * x_i + noise }
    //
    // where, w is some fixed weight vector and x_i denoted the i'th feature vector.
    //
    // We try to recover this relationship in logistic regression by observing
    // the actual labels and want to test that we've roughly correctly estimated
    // the linear function. Because we target the cross-entropy we're effectively
    // targeting relative error in the estimated probabilities. Therefore, we bound
    // the log of the ratio between the actual and predicted class probabilities.

    test::CRandomNumbers rng;

    std::size_t trainRows{1000};
    std::size_t rows{1200};
    std::size_t cols{4};
    std::size_t capacity{600};

    TMeanAccumulator meanLogRelativeError;

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

        auto classifier =
            maths::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::boosted_tree::CBinomialLogisticLoss>())
                .buildFor(*frame, cols - 1);

        classifier->train();
        classifier->predict();

        TMeanAccumulator logRelativeError;
        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (row->index() >= trainRows) {
                    double expectedProbability{probability(*row)};
                    double actualProbability{classifier->readPrediction(*row)[1]};
                    logRelativeError.add(
                        std::log(std::max(actualProbability, expectedProbability) /
                                 std::min(actualProbability, expectedProbability)));
                }
            }
        });
        LOG_DEBUG(<< "log relative error = "
                  << maths::CBasicStatistics::mean(logRelativeError));

        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(logRelativeError) < 0.71);
        meanLogRelativeError.add(maths::CBasicStatistics::mean(logRelativeError));
    }

    LOG_DEBUG(<< "mean log relative error = "
              << maths::CBasicStatistics::mean(meanLogRelativeError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanLogRelativeError) < 0.5);
}

BOOST_AUTO_TEST_CASE(testImbalancedClasses) {

    // Test we get similar per class precision and recall with unbalanced training
    // data when using the calculated decision threshold to assign to class one.

    test::CRandomNumbers rng;

    std::size_t trainRows{2000};
    std::size_t classes[]{1, 0, 1, 0};
    std::size_t classesRowCounts[]{1600, 400, 100, 100};
    std::size_t cols{3};

    TDoubleVecVec x;
    TDoubleVec means{0.0, 3.0};
    TDoubleVec variances{6.0, 6.0};
    for (std::size_t i = 0; i < 4; ++i) {
        TDoubleVecVec xi;
        double mean{means[classes[i]]};
        double variance{variances[classes[i]]};
        rng.generateMultivariateNormalSamples(
            {mean, mean}, {{variance, 0.0}, {0.0, variance}}, classesRowCounts[i], xi);
        x.insert(x.end(), xi.begin(), xi.end());
    }

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{false, false, true});
    for (std::size_t i = 0, index = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < classesRowCounts[i]; ++j, ++index) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t k = 0; k < cols - 1; ++k, ++column) {
                    *column = x[index][k];
                }
                *column = index < trainRows ? static_cast<double>(i)
                                            : core::CDataFrame::valueOfMissing();
            });
        }
    }
    frame->finishWritingRows();

    auto classification =
        maths::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::boosted_tree::CBinomialLogisticLoss>())
            .buildFor(*frame, cols - 1);

    classification->train();
    classification->predict();

    TDoubleVec precisions;
    TDoubleVec recalls;
    {
        TDoubleVec truePositives(2, 0.0);
        TDoubleVec trueNegatives(2, 0.0);
        TDoubleVec falsePositives(2, 0.0);
        TDoubleVec falseNegatives(2, 0.0);
        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                double prediction{
                    classification->readAndAdjustPrediction(*row)[1] < 0.5 ? 0.0 : 1.0};
                if (row->index() >= trainRows &&
                    row->index() < trainRows + classesRowCounts[2]) {
                    // Actual is zero.
                    (prediction == 0.0 ? truePositives[0] : falseNegatives[0]) += 1.0;
                    (prediction == 0.0 ? trueNegatives[1] : falsePositives[1]) += 1.0;
                } else if (row->index() >= trainRows + classesRowCounts[2]) {
                    // Actual is one.
                    (prediction == 1.0 ? truePositives[1] : falseNegatives[1]) += 1.0;
                    (prediction == 1.0 ? trueNegatives[0] : falsePositives[0]) += 1.0;
                }
            }
        });
        precisions.push_back(truePositives[0] / (truePositives[0] + falsePositives[0]));
        precisions.push_back(truePositives[1] / (truePositives[1] + falsePositives[1]));
        recalls.push_back(truePositives[0] / (truePositives[0] + falseNegatives[0]));
        recalls.push_back(truePositives[1] / (truePositives[1] + falseNegatives[1]));
    }

    LOG_DEBUG(<< "precisions = " << core::CContainerPrinter::print(precisions));
    LOG_DEBUG(<< "recalls    = " << core::CContainerPrinter::print(recalls));

    BOOST_TEST_REQUIRE(std::fabs(precisions[0] - precisions[1]) < 0.1);
    BOOST_TEST_REQUIRE(std::fabs(recalls[0] - recalls[1]) < 0.14);
}

BOOST_AUTO_TEST_CASE(testMultinomialLogisticRegression) {

    // The idea of this test is to create a random linear relationship between
    // the feature values and the logit, i.e. logit_i = W * x_i for matrix W is
    // some fixed weight matrix and x_i denoted the i'th feature vector.
    //
    // We try to recover this relationship in logistic regression by observing
    // the actual labels and want to test that we've roughly correctly estimated
    // the linear function. Because we target the cross-entropy we're effectively
    // targeting relative error in the estimated probabilities. Therefore, we bound
    // the log of the ratio between the actual and predicted class probabilities.

    using TVector = maths::CDenseVector<double>;
    using TMemoryMappedMatrix = maths::CMemoryMappedDenseMatrix<double>;

    maths::CPRNG::CXorOShiro128Plus rng;
    test::CRandomNumbers testRng;

    std::size_t trainRows{1000};
    std::size_t rows{1200};
    std::size_t cols{4};
    std::size_t capacity{600};
    int numberClasses{3};
    int numberFeatures{static_cast<int>(cols - 1)};

    TMeanAccumulator meanLogRelativeError;

    TDoubleVec weights;
    TDoubleVec noise;
    TDoubleVec uniform01;

    for (std::size_t test = 0; test < 3; ++test) {
        testRng.generateUniformSamples(-2.0, 2.0, numberClasses * numberFeatures, weights);
        testRng.generateNormalSamples(0.0, 1.0, numberFeatures * rows, noise);
        testRng.generateUniformSamples(0.0, 1.0, rows, uniform01);

        auto probability = [&](const TRowRef& row) {
            TMemoryMappedMatrix W(&weights[0], numberClasses, numberFeatures);
            TVector x(numberFeatures);
            TVector n{numberFeatures};
            for (int i = 0; i < numberFeatures; ++i) {
                x(i) = row[i];
                n(i) = noise[numberFeatures * row.index() + i];
            }
            TVector logit{W * x + n};
            return maths::CTools::softmax(std::move(logit));
        };

        auto target = [&](const TRowRef& row) {
            TDoubleVec probabilities{probability(row).to<TDoubleVec>()};
            return static_cast<double>(maths::CSampling::categoricalSample(rng, probabilities));
        };

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            testRng.generateUniformSamples(0.0, 4.0, rows, x[i]);
        }

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(trainRows, rows - trainRows, cols, {false, false, false, true},
                      x, TDoubleVec(rows, 0.0), target, *frame);

        auto classifier =
            maths::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::boosted_tree::CMultinomialLogisticLoss>(numberClasses))
                .buildFor(*frame, cols - 1);

        classifier->train();
        classifier->predict();

        TMeanAccumulator logRelativeError;
        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (row->index() >= trainRows) {
                    TVector expectedProbability{probability(*row)};
                    TVector actualProbability{
                        TVector::fromSmallVector(classifier->readPrediction(*row))};
                    logRelativeError.add(
                        (expectedProbability.cwiseMax(actualProbability).array() /
                         expectedProbability.cwiseMin(actualProbability).array())
                            .log()
                            .sum() /
                        3.0);
                }
            }
        });
        LOG_DEBUG(<< "log relative error = "
                  << maths::CBasicStatistics::mean(logRelativeError));

        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(logRelativeError) < 2.1);
        meanLogRelativeError.add(maths::CBasicStatistics::mean(logRelativeError));
    }

    LOG_DEBUG(<< "mean log relative error = "
              << maths::CBasicStatistics::mean(meanLogRelativeError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanLogRelativeError) < 1.3);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsedByTrain) {

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

        std::int64_t estimatedMemory(maths::CBoostedTreeFactory::constructFromParameters(
                                         1, std::make_unique<maths::boosted_tree::CMse>())
                                         .estimateMemoryUsage(rows, cols));

        CTestInstrumentation instrumentation;
        auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                              1, std::make_unique<maths::boosted_tree::CMse>())
                              .analysisInstrumentation(instrumentation)
                              .buildFor(*frame, cols - 1);

        regression->train();

        LOG_DEBUG(<< "estimated memory usage = " << estimatedMemory);
        LOG_DEBUG(<< "high water mark = " << instrumentation.maxMemoryUsage());

        BOOST_TEST_REQUIRE(instrumentation.maxMemoryUsage() < estimatedMemory);
    }
}

BOOST_AUTO_TEST_CASE(testProgressMonitoring) {

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

        CTestInstrumentation instrumentation;
        std::atomic_bool finished{false};

        std::thread worker{[&]() {
            auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                                  threads, std::make_unique<maths::boosted_tree::CMse>())
                                  .analysisInstrumentation(instrumentation)
                                  .buildFor(*frame, cols - 1);

            regression->train();
            finished.store(true);
        }};

        int lastProgressReport{0};

        bool monotonic{true};
        int percentage{0};
        while (finished.load() == false) {
            if (instrumentation.progress() > percentage) {
                LOG_DEBUG(<< percentage << "% complete");
                percentage += 10;
            }
            monotonic &= (instrumentation.progress() >= lastProgressReport);
            lastProgressReport = instrumentation.progress();
        }
        worker.join();

        BOOST_TEST_REQUIRE(monotonic);
        LOG_DEBUG(<< "progress points = "
                  << core::CContainerPrinter::print(instrumentation.tenPercentProgressPoints()));
        BOOST_REQUIRE_EQUAL("[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]",
                            core::CContainerPrinter::print(
                                instrumentation.tenPercentProgressPoints()));

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testMissingFeatures) {

    // Test censoring, i.e. data missing is correlated with the target variable.

    std::size_t rows{1000};
    std::size_t cols{4};
    test::CRandomNumbers rng;

    auto frame = core::makeMainStorageDataFrame(cols).first;

    frame->categoricalColumns(TBoolVec{false, false, false, false});
    for (std::size_t i = 0; i < rows - 4; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            TDoubleVec regressors;
            rng.generateUniformSamples(0.0, 10.0, cols - 1, regressors);
            double target{0.0};
            for (auto regressor : regressors) {
                *(column++) = regressor > 9.0 ? core::CDataFrame::valueOfMissing() : regressor;
                target += regressor;
            }
            *column = target;
        });
    }
    frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
        *(column++) = core::CDataFrame::valueOfMissing();
        *(column++) = 2.0;
        *(column++) = 6.0;
        *column = core::CDataFrame::valueOfMissing();
    });
    frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
        *(column++) = 2.0;
        *(column++) = core::CDataFrame::valueOfMissing();
        *(column++) = 6.0;
        *column = core::CDataFrame::valueOfMissing();
    });
    frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
        *(column++) = 2.0;
        *(column++) = 6.0;
        *(column++) = core::CDataFrame::valueOfMissing();
        *column = core::CDataFrame::valueOfMissing();
    });
    frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
        *(column++) = core::CDataFrame::valueOfMissing();
        *(column++) = 3.0;
        *(column++) = core::CDataFrame::valueOfMissing();
        *column = core::CDataFrame::valueOfMissing();
    });
    frame->finishWritingRows();

    auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::boosted_tree::CMse>())
                          .buildFor(*frame, cols - 1);

    regression->train();
    regression->predict();

    // For each missing value given we censor for regression variables greater
    // than 9.0, target = sum_i{R_i} for regressors R_i and R_i ~ U([0,10]) so
    // we expect a n * E[U([0, 10]) | U([0, 10]) > 9.0] = 9.5 * n contribution
    // to the target for n equal to the number of missing variables.
    TDoubleVec expectedPredictions{17.5, 17.5, 17.5, 22.0};
    TDoubleVec actualPredictions;

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            if (maths::CDataFrameUtils::isMissing((*row)[cols - 1])) {
                actualPredictions.push_back(regression->readPrediction(*row)[0]);
            }
        }
    });

    BOOST_REQUIRE_EQUAL(expectedPredictions.size(), actualPredictions.size());
    for (std::size_t i = 0; i < expectedPredictions.size(); ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedPredictions[i], actualPredictions[i], 0.8);
    }
}

BOOST_AUTO_TEST_CASE(testPersistRestore) {

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
    auto boostedTree = maths::CBoostedTreeFactory::constructFromString(persistOnceSStream)
                           .restoreFor(*frame, cols - 1);
    {
        core::CJsonStatePersistInserter inserter(persistTwiceSStream);
        boostedTree->acceptPersistInserter(inserter);
        persistTwiceSStream.flush();
    }
    BOOST_REQUIRE_EQUAL(persistOnceSStream.str(), persistTwiceSStream.str());
    LOG_DEBUG(<< "First string " << persistOnceSStream.str());
    LOG_DEBUG(<< "Second string " << persistTwiceSStream.str());

    // and even run
    BOOST_REQUIRE_NO_THROW(boostedTree->train());
    BOOST_REQUIRE_NO_THROW(boostedTree->predict());

    // TODO test persist and restore produces same train result.
}

BOOST_AUTO_TEST_CASE(testRestoreErrorHandling) {

    auto errorHandler = [](std::string error) {
        throw std::runtime_error{error};
    };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    auto stream = boost::make_shared<std::ostringstream>();

    // log at level ERROR only
    BOOST_TEST_REQUIRE(core::CLogger::instance().reconfigure(stream));

    std::size_t cols{3};
    std::size_t capacity{50};

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    std::stringstream errorInBayesianOptimisationState;
    readFileToStream("testfiles/error_bayesian_optimisation_state.json",
                     errorInBayesianOptimisationState);
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
        BOOST_TEST_REQUIRE(re.matches(e.what()));
        BOOST_TEST_REQUIRE(stream->str().find("Failed to restore MAX_BOUNDARY_TAG") !=
                           std::string::npos);
    }
    BOOST_TEST_REQUIRE(throwsExceptions);

    std::stringstream errorInBoostedTreeImplState;
    readFileToStream("testfiles/error_boosted_tree_impl_state.json", errorInBoostedTreeImplState);
    errorInBoostedTreeImplState.flush();

    throwsExceptions = false;
    stream->clear();
    try {
        auto boostedTree = maths::CBoostedTreeFactory::constructFromString(errorInBoostedTreeImplState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        LOG_DEBUG(<< "got = " << e.what());
        throwsExceptions = true;
        core::CRegex re;
        re.init("Input error:.*");
        BOOST_TEST_REQUIRE(re.matches(e.what()));
        BOOST_TEST_REQUIRE(stream->str().find("Failed to restore NUMBER_FOLDS_TAG") !=
                           std::string::npos);
    }
    BOOST_TEST_REQUIRE(throwsExceptions);

    std::stringstream errorInStateVersion;
    readFileToStream("testfiles/error_no_version_state.json", errorInStateVersion);
    errorInStateVersion.flush();

    throwsExceptions = false;
    stream->clear();
    try {
        auto boostedTree = maths::CBoostedTreeFactory::constructFromString(errorInBoostedTreeImplState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        LOG_DEBUG(<< "got = " << e.what());
        throwsExceptions = true;
        core::CRegex re;
        re.init("Input error:.*");
        BOOST_TEST_REQUIRE(re.matches(e.what()));
        BOOST_TEST_REQUIRE(stream->str().find("unsupported state serialization version.") !=
                           std::string::npos);
    }
    BOOST_TEST_REQUIRE(throwsExceptions);
    ml::core::CLogger::instance().reset();
}

BOOST_AUTO_TEST_SUITE_END()
