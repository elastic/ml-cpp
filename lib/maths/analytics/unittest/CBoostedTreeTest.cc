/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <boost/test/tools/old/interface.hpp>
#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CPackedBitVector.h>
#include <core/CRegex.h>
#include <core/CStopWatch.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeFactory.h>
#include <maths/analytics/CBoostedTreeImpl.h>
#include <maths/analytics/CBoostedTreeLoss.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/COrderings.h>
#include <maths/common/CPRNG.h>
#include <maths/common/CSampling.h>
#include <maths/common/CSpline.h>
#include <maths/common/CTools.h>
#include <maths/common/CToolsDetail.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <boost/make_shared.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace analytics {
class CBoostedTreeImplForTest {
public:
    using TAnalysisInstrumentationPtr = CBoostedTreeImpl::TAnalysisInstrumentationPtr;
    using TLossFunctionUPtr = CBoostedTreeImpl::TLossFunctionUPtr;
    using TSizeVec = CBoostedTreeImpl::TSizeVec;
    using TDoubleVec = CBoostedTreeImpl::TDoubleVec;
    using TBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;

public:
    explicit CBoostedTreeImplForTest(CBoostedTreeImpl& treeImpl)
        : m_TreeImpl{treeImpl} {}
    CBoostedTreeImplForTest(const CBoostedTreeImplForTest&) = delete;
    CBoostedTreeImplForTest& operator=(const CBoostedTreeImplForTest&) = delete;

    const TDoubleVec& featureSampleProbabilities() const {
        return m_TreeImpl.featureSampleProbabilities();
    }

    void treeFeatureBag(TDoubleVec& probabilities, TSizeVec& treeFeatureBag) const {
        m_TreeImpl.treeFeatureBag(probabilities, treeFeatureBag);
    }

    void nodeFeatureBag(const TSizeVec& treeFeatureBag,
                        TDoubleVec& probabilities,
                        TSizeVec& nodeFeatureBag) const {
        m_TreeImpl.nodeFeatureBag(treeFeatureBag, probabilities, nodeFeatureBag);
    }

    TBoostedTreeUPtr cloneFor(core::CDataFrame& frame, std::size_t dependentVariable) const {
        std::stringstream persistState;
        {
            core::CJsonStatePersistInserter inserter(persistState);
            m_TreeImpl.acceptPersistInserter(inserter);
            persistState.flush();
        }
        std::stringstream stateStr0(persistState.str());
        return maths::analytics::CBoostedTreeFactory::constructFromString(stateStr0)
            .restoreFor(frame, dependentVariable);
    }

private:
    CBoostedTreeImpl& m_TreeImpl;
};
}
}
}

using namespace ml;

BOOST_TEST_DONT_PRINT_LOG_VALUE(maths::analytics::CBoostedTreeImplForTest::TSizeVec::iterator)

BOOST_AUTO_TEST_SUITE(CBoostedTreeTest)

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;
using TFactoryFuncVec = std::vector<TFactoryFunc>;
using TFactoryFuncVecVec = std::vector<TFactoryFuncVec>;
using TRowRef = core::CDataFrame::TRowRef;
using TRowItr = core::CDataFrame::TRowItr;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TLossFunctionType = maths::analytics::boosted_tree::ELossType;
using TLossFunctionUPtr = maths::analytics::CBoostedTreeFactory::TLossFunctionUPtr;
using TTargetFunc = std::function<double(const TRowRef&)>;

const double BYTES_IN_MB{static_cast<double>(core::constants::BYTES_IN_MEGABYTES)};

class CTestInstrumentation
    : public maths::analytics::CDataFrameTrainBoostedTreeInstrumentationStub {
public:
    using TIntVec = std::vector<int>;
    struct STaskProgess {
        STaskProgess(std::string name, bool monotonic, TIntVec tenPercentProgressPoints)
            : s_Name{std::move(name)}, s_Monotonic{monotonic},
              s_TenPercentProgressPoints{std::move(tenPercentProgressPoints)} {}

        std::string s_Name;
        bool s_Monotonic;
        TIntVec s_TenPercentProgressPoints;
    };
    using TTaskProgressVec = std::vector<STaskProgess>;

public:
    CTestInstrumentation()
        : m_TotalFractionalProgress{0}, m_Monotonic{true}, m_MemoryUsage{0}, m_MaxMemoryUsage{0} {
    }

    int progress() const {
        return (100 * m_TotalFractionalProgress.load()) / 65536;
    }
    std::int64_t maxMemoryUsage() const { return m_MaxMemoryUsage.load(); }

    TTaskProgressVec taskProgress() const { return m_TaskProgress; }

    void startNewProgressMonitoredTask(const std::string& task) override {
        if (m_Task.empty() == false) {
            m_TaskProgress.emplace_back(m_Task, m_Monotonic.load(), m_TenPercentProgressPoints);
        }
        m_Task = task;
        m_TotalFractionalProgress.store(0);
        m_TenPercentProgressPoints.clear();
        m_Monotonic.store(true);
    }

    void updateProgress(double fractionalProgress) override {
        int progress{m_TotalFractionalProgress.fetch_add(
            static_cast<int>(65536.0 * fractionalProgress + 0.5))};
        m_Monotonic.store(m_Monotonic.load() && fractionalProgress >= 0.0);
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
    std::string m_Task;
    std::atomic_int m_TotalFractionalProgress;
    TIntVec m_TenPercentProgressPoints;
    std::atomic_bool m_Monotonic;
    TTaskProgressVec m_TaskProgress;
    std::atomic<std::int64_t> m_MemoryUsage;
    std::atomic<std::int64_t> m_MaxMemoryUsage;
};

template<typename F, typename G>
auto computeEvaluationMetrics(const core::CDataFrame& frame,
                              std::size_t beginTestRows,
                              std::size_t endTestRows,
                              const F& actual,
                              const G& target,
                              double noiseVariance,
                              const TSizeVec& outliers = {}) {

    TMeanVarAccumulator functionMoments;
    TMeanVarAccumulator modelPredictionErrorMoments;

    frame.readRows(1, beginTestRows, endTestRows, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            if (std::binary_search(outliers.begin(), outliers.end(), row->index()) == false) {
                functionMoments.add(target(*row));
                modelPredictionErrorMoments.add(target(*row) - actual(*row));
            }
        }
    });

    LOG_TRACE(<< "function moments = " << functionMoments);
    LOG_TRACE(<< "model prediction error moments = " << modelPredictionErrorMoments);

    double functionVariance{maths::common::CBasicStatistics::variance(functionMoments)};
    double predictionErrorMean{maths::common::CBasicStatistics::mean(modelPredictionErrorMoments)};
    double predictionErrorVariance{
        maths::common::CBasicStatistics::variance(modelPredictionErrorMoments)};
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

    std::size_t offset{frame.numberRows()};

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

    frame.writeColumns(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            if (row->index() >= offset) {
                std::size_t index{row->index() - offset};
                double targetValue{index < trainRows
                                       ? target(*row) + noise[index]
                                       : core::CDataFrame::valueOfMissing()};
                row->writeColumn(cols - 1, targetValue);
            }
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

template<typename W>
void fillDataFrameWeights(std::size_t col, const W& weight, core::CDataFrame& frame) {
    frame.writeColumns(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            row->writeColumn(col, weight(*row));
        }
    });
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
                maths::analytics::CBoostedTreeFactory::constructFromParameters(
                    1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                    .buildForTrain(*frame, cols - 1);

            regression->train();
            regression->predict();

            double bias;
            double rSquared;
            std::tie(bias, rSquared) = computeEvaluationMetrics(
                *frame, trainRows, rows,
                [&](const TRowRef& row) {
                    return regression->prediction(row)[0];
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

template<typename F>
auto predictAndComputeEvaluationMetrics(const F& generateFunction,
                                        test::CRandomNumbers& rng,
                                        std::size_t trainRows,
                                        std::size_t testRows,
                                        std::size_t cols,
                                        double noiseVariance,
                                        const TSizeVec& outliers) {

    std::size_t rows{trainRows + testRows};

    TDoubleVec modelBias;
    TDoubleVec modelRSquared;

    for (std::size_t i = 0; i < 2; ++i) {
        auto target = generateFunction(rng, cols);

        TDoubleVecVec x(cols - 1);
        for (std::size_t j = 0; j < cols - 1; ++j) {
            rng.generateUniformSamples(0.0, 10.0, rows, x[j]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

        auto frame = core::makeMainStorageDataFrame(cols, rows).first;

        fillDataFrame(trainRows, testRows, cols, x, noise, target, *frame);

        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CPseudoHuber>(1.0))
                .buildForTrain(*frame, cols - 1);

        regression->train();
        regression->predict();

        double bias;
        double rSquared;
        std::tie(bias, rSquared) = computeEvaluationMetrics(
            *frame, trainRows, rows,
            [&](const TRowRef& row) { return regression->prediction(row)[0]; },
            target, noiseVariance / static_cast<double>(rows), outliers);
        modelBias.push_back(bias);
        modelRSquared.push_back(rSquared);
    }
    LOG_DEBUG(<< "bias = " << core::CContainerPrinter::print(modelBias));
    LOG_DEBUG(<< " R^2 = " << core::CContainerPrinter::print(modelRSquared));

    return std::make_pair(std::move(modelBias), std::move(modelRSquared));
}

std::size_t maxDepth(const std::vector<maths::analytics::CBoostedTreeNode>& tree,
                     const maths::analytics::CBoostedTreeNode& node,
                     std::size_t depth) {
    std::size_t result{depth};
    if (node.isLeaf() == false) {
        result = std::max(result, maxDepth(tree, tree[node.leftChildIndex()], depth + 1));
        result = std::max(result, maxDepth(tree, tree[node.rightChildIndex()], depth + 1));
    }
    return result;
}

void readFileToStream(const std::string& filename, std::stringstream& stream) {
    std::ifstream file(filename);
    BOOST_TEST_REQUIRE(file.is_open());
    std::string str((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    stream << str;
    stream.flush();
}

const std::size_t OLD_TRAINING_DATA{0};
const std::size_t NEW_TRAINING_DATA{1};
}

BOOST_AUTO_TEST_CASE(testEdgeCases) {

    // Test some edge case inputs fail gracefully.

    auto errorHandler = [](const std::string& error) {
        throw std::runtime_error{error};
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    std::size_t cols{2};

    // No data.
    try {
        auto frame = core::makeMainStorageDataFrame(cols).first;
        fillDataFrame(0, 0, 2, {}, {}, [](const TRowRef&) { return 1.0; }, *frame);
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .buildForTrain(*frame, cols - 1);
    } catch (const std::exception& e) { LOG_DEBUG(<< e.what()); }

    // No training data.
    try {
        auto frame = core::makeMainStorageDataFrame(cols).first;
        fillDataFrame(0, 1, 2, {{1.0}}, {0.0}, [](const TRowRef&) { return 1.0; }, *frame);
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .buildForTrain(*frame, cols - 1);
    } catch (const std::exception& e) { LOG_DEBUG(<< e.what()); }

    // Insufficient training data.
    try {
        auto frame = core::makeMainStorageDataFrame(cols).first;
        fillDataFrame(1, 0, 2, {{1.0}}, {0.0}, [](const TRowRef&) { return 1.0; }, *frame);
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .buildForTrain(*frame, cols - 1);
    } catch (const std::exception& e) { LOG_DEBUG(<< e.what()); }

    auto frame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(5, 0, 2, {{1.0}, {1.0}, {1.0}, {1.0}, {1.0}},
                  {0.0, 0.0, 0.0, 0.0, 0.0}, [](const TRowRef&) { return 1.0; }, *frame);

    try {
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .buildForTrain(*frame, cols - 1);
        regression->train();
    } catch (...) { BOOST_FAIL("Shouldn't throw"); }
}

BOOST_AUTO_TEST_CASE(testMsePiecewiseConstant) {

    // Test regression quality on piecewise constant function.

    auto generatePiecewiseConstant = [](test::CRandomNumbers& rng, std::size_t cols) {
        TDoubleVec p;
        TDoubleVec v;
        rng.generateUniformSamples(0.0, 10.0, 2 * cols - 2, p);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, v);
        for (std::size_t i = 0; i < p.size(); i += 2) {
            std::sort(p.begin() + i, p.begin() + i + 2);
        }

        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                if (row[i] >= p[2 * i] && row[i] < p[2 * i + 1]) {
                    result += v[i];
                }
            }
            return result;
        };
    };

    test::CRandomNumbers rng;
    double noiseVariance{0.2};
    std::size_t trainRows{500};
    std::size_t testRows{200};
    std::size_t cols{3};
    std::size_t capacity{250};

    TDoubleVecVec modelBias;
    TDoubleVecVec modelRSquared;
    std::tie(modelBias, modelRSquared) = predictAndComputeEvaluationMetrics(
        generatePiecewiseConstant, rng, trainRows, testRows, cols, capacity, noiseVariance);

    TMeanAccumulator meanModelRSquared;

    for (std::size_t i = 0; i < modelBias.size(); ++i) {

        // In and out-of-core agree.
        for (std::size_t j = 1; j < modelBias[i].size(); ++j) {
            BOOST_REQUIRE_EQUAL(modelBias[i][0], modelBias[i][j]);
            BOOST_REQUIRE_EQUAL(modelRSquared[i][0], modelRSquared[i][j]);
        }

        // Roughly unbiased...
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, modelBias[i][0],
            10.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i][0] > 0.91);

        meanModelRSquared.add(modelRSquared[i][0]);
    }

    LOG_DEBUG(<< "mean R^2 = " << maths::common::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanModelRSquared) > 0.95);
}

BOOST_AUTO_TEST_CASE(testMseLinear) {

    // Test regression quality on linear function.

    auto generateLinear = [](test::CRandomNumbers& rng, std::size_t cols) {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);

        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    };

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t trainRows{500};
    std::size_t testRows{200};
    std::size_t cols{6};
    std::size_t capacity{250};

    TDoubleVecVec modelBias;
    TDoubleVecVec modelRSquared;
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
            6.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i][0] > 0.97);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::common::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanModelRSquared) > 0.97);
}

BOOST_AUTO_TEST_CASE(testMseNonLinear) {

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

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t trainRows{500};
    std::size_t testRows{200};
    std::size_t cols{6};
    std::size_t capacity{500};

    TDoubleVecVec modelBias;
    TDoubleVecVec modelRSquared;
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
            4.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i][0] > 0.97);

        meanModelRSquared.add(modelRSquared[i][0]);
    }
    LOG_DEBUG(<< "mean R^2 = " << maths::common::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanModelRSquared) > 0.98);
}

BOOST_AUTO_TEST_CASE(testHuber) {

    // Test the quality of Huber on linear plus outliers.

    std::size_t trainRows{500};
    std::size_t testRows{200};
    double noiseVariance{100.0};
    TSizeVec outliers;
    {
        test::CRandomNumbers rng;
        for (std::size_t i = 0; i < trainRows + testRows; ++i) {
            TDoubleVec u01;
            rng.generateUniformSamples(0.0, 1.0, 1, u01);
            if (u01[0] < 0.02) {
                outliers.push_back(i);
            }
        }
    }
    LOG_DEBUG(<< "outliers = " << core::CContainerPrinter::print(outliers));

    auto generateLinearPlusOutliers = [&](test::CRandomNumbers& rng, std::size_t cols) {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);

        return [=](const TRowRef& row) {
            double result{std::binary_search(outliers.begin(), outliers.end(), row.index())
                              ? 5.0 * std::sqrt(noiseVariance)
                              : 0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    };

    test::CRandomNumbers rng;
    std::size_t cols{6};

    TDoubleVec modelBias;
    TDoubleVec modelRSquared;
    std::tie(modelBias, modelRSquared) = predictAndComputeEvaluationMetrics(
        generateLinearPlusOutliers, rng, trainRows, testRows, cols, noiseVariance, outliers);

    TMeanAccumulator meanModelRSquared;

    for (std::size_t i = 0; i < modelBias.size(); ++i) {

        // Unbiased...
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, modelBias[i],
            6.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
        // Good R^2...
        BOOST_TEST_REQUIRE(modelRSquared[i] > 0.94);

        meanModelRSquared.add(modelRSquared[i]);
    }

    LOG_DEBUG(<< "mean R^2 = " << maths::common::CBasicStatistics::mean(meanModelRSquared));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanModelRSquared) > 0.96);
}

BOOST_AUTO_TEST_CASE(testMsle) {
    // TODO #1744 test quality of MSLE on data with log-normal errors.
}

BOOST_AUTO_TEST_CASE(testNonUnitWeights) {

    // Check that downweighting outlier values improves prediction errors and bias
    // on inlier values.

    std::size_t trainRows{500};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};
    double noiseVariance{25.0};
    std::size_t cols{6};
    std::size_t weightCol{cols};

    TSizeVec outliers;
    test::CRandomNumbers rng;
    for (std::size_t i = 0; i < trainRows + testRows; ++i) {
        TDoubleVec u01;
        rng.generateUniformSamples(0.0, 1.0, 1, u01);
        if (u01[0] < 0.05) {
            outliers.push_back(i);
        }
    }
    LOG_DEBUG(<< "outliers = " << core::CContainerPrinter::print(outliers));

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    auto targetWithOutliers = [&](const TRowRef& row) {
        return target(row) +
               (std::binary_search(outliers.begin(), outliers.end(), row.index())
                    ? 8.0 * std::sqrt(noiseVariance)
                    : 0.0);
    };

    auto weight = [&](const TRowRef& row) {
        return std::binary_search(outliers.begin(), outliers.end(), row.index()) ? 0.1 : 1.0;
    };

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

    auto frameWithoutWeights = core::makeMainStorageDataFrame(cols, rows).first;
    frameWithoutWeights->columnNames({"x1", "x2", "x3", "x4", "x5", "x6"});
    fillDataFrame(trainRows, testRows, cols, x, noise, targetWithOutliers, *frameWithoutWeights);

    auto regressionWithoutWeights =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CMse>())
            .buildForTrain(*frameWithoutWeights, cols - 1);

    regressionWithoutWeights->train();
    regressionWithoutWeights->predict();

    auto frameWithWeights = core::makeMainStorageDataFrame(cols + 1, rows).first;
    frameWithWeights->columnNames({"x1", "x2", "x3", "x4", "x5", "x6", "w"});
    fillDataFrame(trainRows, testRows, cols, TBoolVec(cols + 1, false), x,
                  noise, targetWithOutliers, *frameWithWeights);
    fillDataFrameWeights(weightCol, weight, *frameWithWeights);

    auto regressionWithWeights =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CMse>())
            .rowWeightColumnName("w")
            .buildForTrain(*frameWithWeights, cols - 1);

    regressionWithWeights->train();
    regressionWithWeights->predict();

    double biasWithoutWeights;
    double rSquaredWithoutWeights;
    std::tie(biasWithoutWeights, rSquaredWithoutWeights) = computeEvaluationMetrics(
        *frameWithoutWeights, trainRows, rows,
        [&](const TRowRef& row) {
            return regressionWithoutWeights->prediction(row)[0];
        },
        target, noiseVariance / static_cast<double>(rows));
    LOG_DEBUG(<< "biasWithoutWeights = " << biasWithoutWeights
              << ", rSquaredWithoutWeights = " << rSquaredWithoutWeights);

    double biasWithWeights;
    double rSquaredWithWeights;
    std::tie(biasWithWeights, rSquaredWithWeights) = computeEvaluationMetrics(
        *frameWithWeights, trainRows, rows,
        [&](const TRowRef& row) {
            return regressionWithWeights->prediction(row)[0];
        },
        target, noiseVariance / static_cast<double>(rows));
    LOG_DEBUG(<< "biasWithWeights    = " << biasWithWeights
              << ", rSquaredWithWeights    = " << rSquaredWithWeights);

    BOOST_TEST_REQUIRE(std::fabs(biasWithWeights) < 0.2 * std::fabs(biasWithoutWeights));
    BOOST_TEST_REQUIRE(1.0 - rSquaredWithWeights < 0.8 * (1.0 - rSquaredWithoutWeights));
}

BOOST_AUTO_TEST_CASE(testLowCardinalityFeatures) {

    // Test a linear model with low cardinality features.

    std::size_t trainRows{500};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};
    double noiseVariance{9.0};
    std::size_t cols{6};

    test::CRandomNumbers rng;
    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-2.0, 2.0, cols - 1, s);
        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, noiseVariance, rows, noise);
    for (auto& ni : noise) {
        ni = std::floor(ni + 0.5);
    }

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }
    for (std::size_t i = 0; i < (cols - 1) / 2 + 1; ++i) {
        for (auto& xj : x[i]) {
            xj = std::floor(xj + 0.5);
        }
    }

    auto frame = core::makeMainStorageDataFrame(cols, rows).first;
    fillDataFrame(trainRows, testRows, cols, x, noise, target, *frame);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

    regression->train();
    regression->predict();

    double bias;
    double rSquared;
    std::tie(bias, rSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        [&](const TRowRef& row) { return regression->prediction(row)[0]; },
        target, noiseVariance / static_cast<double>(rows));
    LOG_DEBUG(<< "bias = " << bias << ", rSquared = " << rSquared);

    BOOST_TEST_REQUIRE(rSquared > 0.96);
}

BOOST_AUTO_TEST_CASE(testLowTrainFractionPerFold) {

    // Test regression using a very low train fraction per fold. This should
    // run in seconds, but we don't assert on the runtime because we don't
    // run CI on bare metal, and produce a good quality solution because the
    // final train is still on the full training set.

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t trainRows{10000};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};
    std::size_t cols{8};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
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

    auto frame = core::makeMainStorageDataFrame(cols, rows).first;
    fillDataFrame(trainRows, testRows, cols, x, noise, target, *frame);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .trainFractionPerFold(0.05)
                          .buildForTrain(*frame, cols - 1);

    core::CStopWatch timer{true};
    regression->train();
    regression->predict();
    LOG_DEBUG(<< "train duration " << timer.stop() << "ms");

    double bias;
    double rSquared;
    std::tie(bias, rSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        [&](const TRowRef& row_) { return regression->prediction(row_)[0]; },
        target, noiseVariance / static_cast<double>(rows));

    // Unbiased...
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        0.0, bias, 7.0 * std::sqrt(noiseVariance / static_cast<double>(trainRows)));
    // Good R^2...
    BOOST_TEST_REQUIRE(rSquared > 0.98);
}

BOOST_AUTO_TEST_CASE(testHoldoutRowMask) {

    // Check that we can effectively train using loss on a specified holdout set.

    test::CRandomNumbers rng;
    double noiseVariance{10.0};
    std::size_t rows{1000};
    std::size_t cols{6};

    std::size_t numberHoldoutRows{200};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
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

    auto frame = core::makeMainStorageDataFrame(cols, rows).first;
    fillDataFrame(rows, 0, cols, x, noise, target, *frame);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .numberHoldoutRows(numberHoldoutRows)
                          .buildForTrain(*frame, cols - 1);

    regression->train();
    regression->predict();

    // Test the minimum cross-validation error matches the error we compute for
    // holdout set.

    core::CPackedBitVector holdoutRowMask(numberHoldoutRows, true);
    holdoutRowMask.extend(false, rows - numberHoldoutRows);

    TMeanVarAccumulator expectedMse;
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            double actual{(*row)[cols - 1]};
                            double prediction{regression->prediction(*row)[0]};
                            expectedMse.add(maths::common::CTools::pow2(actual - prediction));
                        }
                    },
                    &holdoutRowMask);

    auto roundLosses = regression->impl().foldRoundTestLosses()[0];

    auto actualMse = *std::min_element(
        roundLosses.begin(), roundLosses.end(), [](const auto& lhs, const auto& rhs) {
            return maths::common::COrderings::lexicographical_compare(
                lhs != boost::none ? 0 : 1, *lhs, rhs != boost::none ? 0 : 1, *rhs);
        });

    BOOST_REQUIRE_CLOSE(maths::common::CBasicStatistics::mean(expectedMse), *actualMse, 1e-3);
}

BOOST_AUTO_TEST_CASE(testIncrementalHoldoutRowMask) {

    // Check that we can effectively incrementally train using loss on a specified
    // holdout set.

    test::CRandomNumbers rng;
    double noiseVariance{10.0};
    std::size_t rows{1000};
    std::size_t cols{6};

    std::size_t extraTrainingRows{200};
    std::size_t numberHoldoutRows{200};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
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

    auto frame = core::makeMainStorageDataFrame(cols).first;
    fillDataFrame(rows, 0, cols, x, noise, target, *frame);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .numberHoldoutRows(numberHoldoutRows)
                          .eta({0.02})
                          .dataSummarizationFraction(1.0)
                          .buildForTrain(*frame, cols - 1);

    regression->train();
    regression->predict();

    auto newFrame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(rows, 0, cols, x, noise, target, *newFrame);

    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, extraTrainingRows, x[i]);
    }
    rng.generateNormalSamples(0.0, noiseVariance, extraTrainingRows, noise);

    fillDataFrame(extraTrainingRows, 0, cols, x, noise, target, *newFrame);

    core::CPackedBitVector oldTrainingRowMask{rows, true};
    oldTrainingRowMask.extend(false, extraTrainingRows);
    core::CPackedBitVector newTrainingRowMask{~oldTrainingRowMask};

    double alpha{regression->hyperparameters().depthPenaltyMultiplier().value()};
    double gamma{regression->hyperparameters().treeSizePenaltyMultiplier().value()};
    double lambda{regression->hyperparameters().leafWeightPenaltyMultiplier().value()};
    regression = maths::analytics::CBoostedTreeFactory::constructFromModel(std::move(regression))
                     .numberHoldoutRows(numberHoldoutRows)
                     .newTrainingRowMask(newTrainingRowMask)
                     .depthPenaltyMultiplier({0.5 * alpha, 2.0 * alpha})
                     .treeSizePenaltyMultiplier({0.5 * gamma, 2.0 * gamma})
                     .leafWeightPenaltyMultiplier({0.5 * lambda, 2.0 * lambda})
                     .forceAcceptIncrementalTraining(true)
                     .buildForTrainIncremental(*newFrame, cols - 1);

    regression->trainIncremental();
    regression->predict();

    // Test the minimum cross-validation error matches the error we compute for
    // holdout set.

    core::CPackedBitVector holdoutRowMask(numberHoldoutRows, true);
    holdoutRowMask.extend(false, rows - numberHoldoutRows);
    TMeanAccumulator expectedMse;
    TMeanAccumulator expectedMsd;
    newFrame->readRows(
        1, 0, frame->numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                double actual{(*row)[cols - 1]};
                double prediction{regression->prediction(*row)[0]};
                double previousPrediction{regression->previousPrediction(*row)[0]};
                expectedMse.add(maths::common::CTools::pow2(actual - prediction));
                expectedMsd.add(maths::common::CTools::pow2(prediction - previousPrediction));
            }
        },
        &holdoutRowMask);

    auto roundLosses = regression->impl().foldRoundTestLosses()[0];
    auto actualMse = *std::min_element(
        roundLosses.begin(), roundLosses.end(), [](const auto& lhs, const auto& rhs) {
            return maths::common::COrderings::lexicographical_compare(
                lhs == boost::none ? 1 : 0, *lhs, rhs == boost::none ? 1 : 0, *rhs);
        });

    BOOST_REQUIRE_CLOSE(maths::common::CBasicStatistics::mean(expectedMse) +
                            0.01 * maths::common::CBasicStatistics::mean(expectedMsd),
                        *actualMse, 1e-3);
}

BOOST_AUTO_TEST_CASE(testMseIncrementalForTargetDrift) {

    // Test incremental training for the MSE objective for drift in the target value.

    test::CRandomNumbers rng;
    double noiseVariance{9.0};
    std::size_t rows{750};
    std::size_t cols{6};
    std::size_t extraTrainingRows{250};

    TTargetFunc target;
    TTargetFunc perturbedTarget;
    std::tie(target, perturbedTarget) = [&] {
        TDoubleVec p;
        TDoubleVec v;
        TDoubleVec dv;
        rng.generateUniformSamples(0.0, 10.0, 2 * cols - 2, p);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, v);
        rng.generateUniformSamples(-5.0, 5.0, cols - 1, dv);
        for (std::size_t i = 0; i < p.size(); i += 2) {
            std::sort(p.begin() + i, p.begin() + i + 2);
        }

        auto target_ = [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                if (row[i] >= p[2 * i] && row[i] < p[2 * i + 1]) {
                    result += v[i];
                }
            }
            return result;
        };
        auto perturbedTarget_ = [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                if (row[i] >= p[2 * i] && row[i] < p[2 * i + 1]) {
                    result += v[i] + dv[i];
                }
            }
            return result;
        };

        return std::make_pair(target_, perturbedTarget_);
    }();

    TDoubleVecVec x(cols - 1);
    TDoubleVec noise;
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 4.0, rows, x[i]);
    }
    rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

    auto frame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(rows, 0, cols, x, noise, target, *frame);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .eta({0.02}) // Ensure there are enough trees.
                          .buildForTrain(*frame, cols - 1);

    regression->train();

    frame->resizeColumns(1, cols);
    auto newFrame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(rows, 0, cols, x, noise, target, *newFrame);

    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, extraTrainingRows, x[i]);
    }
    rng.generateNormalSamples(0.0, noiseVariance, extraTrainingRows, noise);

    fillDataFrame(extraTrainingRows, 0, cols, x, noise, perturbedTarget, *frame);
    fillDataFrame(extraTrainingRows, 0, cols, x, noise, perturbedTarget, *newFrame);

    core::CPackedBitVector oldTrainingRowMask{rows, true};
    oldTrainingRowMask.extend(false, extraTrainingRows);
    core::CPackedBitVector newTrainingRowMask{~oldTrainingRowMask};

    TMeanAccumulator mseBeforeIncrementalTraining[2];
    TMeanAccumulator mseAfterIncrementalTraining[2];

    frame->resizeColumns(1, cols + 1);
    regression->predict();

    // Get the average error on the old and new training data from the old model.
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            mseBeforeIncrementalTraining[OLD_TRAINING_DATA].add(
                                maths::common::CTools::pow2(
                                    (*row)[cols - 1] - regression->prediction(*row)[0]));
                        }
                    },
                    &oldTrainingRowMask);
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            mseBeforeIncrementalTraining[NEW_TRAINING_DATA].add(
                                maths::common::CTools::pow2(
                                    (*row)[cols - 1] - regression->prediction(*row)[0]));
                        }
                    },
                    &newTrainingRowMask);

    double alpha{regression->hyperparameters().depthPenaltyMultiplier().value()};
    double gamma{regression->hyperparameters().treeSizePenaltyMultiplier().value()};
    double lambda{regression->hyperparameters().leafWeightPenaltyMultiplier().value()};
    regression = maths::analytics::CBoostedTreeFactory::constructFromModel(std::move(regression))
                     .newTrainingRowMask(newTrainingRowMask)
                     .depthPenaltyMultiplier({0.5 * alpha, 2.0 * alpha})
                     .treeSizePenaltyMultiplier({0.5 * gamma, 2.0 * gamma})
                     .leafWeightPenaltyMultiplier({0.5 * lambda, 2.0 * lambda})
                     .buildForTrainIncremental(*newFrame, cols - 1);

    regression->trainIncremental();
    regression->predict();

    // Get the average error on the old and new training data from the new model.
    newFrame->readRows(
        1, 0, frame->numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                mseAfterIncrementalTraining[OLD_TRAINING_DATA].add(maths::common::CTools::pow2(
                    (*row)[cols - 1] - regression->prediction(*row)[0]));
            }
        },
        &oldTrainingRowMask);
    newFrame->readRows(
        1, 0, frame->numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                mseAfterIncrementalTraining[NEW_TRAINING_DATA].add(maths::common::CTools::pow2(
                    (*row)[cols - 1] - regression->prediction(*row)[0]));
            }
        },
        &newTrainingRowMask);

    // By construction, the prediction error for the old training data must
    // increase because the old and new data distributions overlap and their
    // target values disagree. However, we should see proportionally a much
    // larger reduction in the new training data predcition error.

    double errorIncreaseOnOld{
        maths::common::CBasicStatistics::mean(mseAfterIncrementalTraining[OLD_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(mseBeforeIncrementalTraining[OLD_TRAINING_DATA]) -
        1.0};
    double errorDecreaseOnNew{
        maths::common::CBasicStatistics::mean(mseBeforeIncrementalTraining[NEW_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(mseAfterIncrementalTraining[NEW_TRAINING_DATA]) -
        1.0};

    LOG_DEBUG(<< "increase on old = " << errorIncreaseOnOld);
    LOG_DEBUG(<< "decrease on new = " << errorDecreaseOnNew);
    BOOST_TEST_REQUIRE(errorDecreaseOnNew > 25.0 * errorIncreaseOnOld);
}

BOOST_AUTO_TEST_CASE(testMseIncrementalForOutOfDomain) {

    // Test incremental training for the MSE objective for values out of the training
    // data domain domain.

    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t rows{750};
    std::size_t cols{6};
    std::size_t extraTrainingRows{250};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    auto frame = core::makeMainStorageDataFrame(cols).first;

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 4.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

    fillDataFrame(rows, 0, cols, x, noise, target, *frame);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .eta({0.02}) // Ensure there are enough trees.
                          .buildForTrain(*frame, cols - 1);
    regression->train();

    frame->resizeColumns(1, cols);
    auto newFrame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(rows, 0, cols, x, noise, target, *newFrame);

    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(10.0, 15.0, extraTrainingRows, x[i]);
    }
    rng.generateNormalSamples(0.0, noiseVariance, extraTrainingRows, noise);

    fillDataFrame(extraTrainingRows, 0, cols, x, noise, target, *frame);
    fillDataFrame(extraTrainingRows, 0, cols, x, noise, target, *newFrame);

    regression->predict();

    core::CPackedBitVector oldTrainingRowMask{rows, true};
    oldTrainingRowMask.extend(false, extraTrainingRows);
    core::CPackedBitVector newTrainingRowMask{~oldTrainingRowMask};

    TMeanAccumulator mseBeforeIncrementalTraining[2];
    TMeanAccumulator mseAfterIncrementalTraining[2];

    frame->resizeColumns(1, cols + 1);
    regression->predict();

    // Get the average error on the old and new training data from the old model.
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            mseBeforeIncrementalTraining[OLD_TRAINING_DATA].add(
                                maths::common::CTools::pow2(
                                    (*row)[cols - 1] - regression->prediction(*row)[0]));
                        }
                    },
                    &oldTrainingRowMask);
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            mseBeforeIncrementalTraining[NEW_TRAINING_DATA].add(
                                maths::common::CTools::pow2(
                                    (*row)[cols - 1] - regression->prediction(*row)[0]));
                        }
                    },
                    &newTrainingRowMask);

    double alpha{regression->hyperparameters().depthPenaltyMultiplier().value()};
    double gamma{regression->hyperparameters().treeSizePenaltyMultiplier().value()};
    double lambda{regression->hyperparameters().leafWeightPenaltyMultiplier().value()};
    regression = maths::analytics::CBoostedTreeFactory::constructFromModel(std::move(regression))
                     .newTrainingRowMask(newTrainingRowMask)
                     .depthPenaltyMultiplier({0.5 * alpha, 2.0 * alpha})
                     .treeSizePenaltyMultiplier({0.5 * gamma, 2.0 * gamma})
                     .leafWeightPenaltyMultiplier({0.5 * lambda, 2.0 * lambda})
                     .buildForTrainIncremental(*newFrame, cols - 1);

    regression->trainIncremental();
    regression->predict();

    // Get the average error on the old and new training data from the new model.
    newFrame->readRows(
        1, 0, frame->numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                mseAfterIncrementalTraining[OLD_TRAINING_DATA].add(maths::common::CTools::pow2(
                    (*row)[cols - 1] - regression->prediction(*row)[0]));
            }
        },
        &oldTrainingRowMask);
    newFrame->readRows(
        1, 0, frame->numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                mseAfterIncrementalTraining[NEW_TRAINING_DATA].add(maths::common::CTools::pow2(
                    (*row)[cols - 1] - regression->prediction(*row)[0]));
            }
        },
        &newTrainingRowMask);

    // We should see little change in the prediction errors on the old data
    // set which doesn't overlap the new data, but a large improvement on
    // the new data for constant extrapolation performs poorly.

    double errorIncreaseOnOld{
        maths::common::CBasicStatistics::mean(mseAfterIncrementalTraining[OLD_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(mseBeforeIncrementalTraining[OLD_TRAINING_DATA]) -
        1.0};
    double errorDecreaseOnNew{
        maths::common::CBasicStatistics::mean(mseBeforeIncrementalTraining[NEW_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(mseAfterIncrementalTraining[NEW_TRAINING_DATA]) -
        1.0};

    LOG_DEBUG(<< "increase on old = " << errorIncreaseOnOld);
    LOG_DEBUG(<< "decrease on new = " << errorDecreaseOnNew);
    BOOST_TEST_REQUIRE(errorDecreaseOnNew > 100.0 * errorIncreaseOnOld);
}

BOOST_AUTO_TEST_CASE(testMseIncrementalAddNewTrees) {
    // Update the base model by allowing 0, 5, and 10 new trees. Verify that the holdout error is
    // note getting worse when allowing for more model capacity.
    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    std::size_t batch1Size{500};
    std::size_t batch2Size{150};
    std::size_t numberHoldoutRows{200};
    std::size_t cols{6};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        };
    }();

    TDoubleVec noise;
    TDoubleVecVec x(cols - 1);

    auto batch1 = core::makeMainStorageDataFrame(cols).first;
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 4.0, batch1Size, x[i]);
    }
    rng.generateNormalSamples(0.0, noiseVariance, batch1Size, noise);
    fillDataFrame(batch1Size, 0, cols, x, noise, target, *batch1);

    // second batch of data extends the domain
    auto batch2 = core::makeMainStorageDataFrame(cols).first;
    fillDataFrame(batch1Size, 0, cols, x, noise, target, *batch2);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(2.0, 6.0, batch2Size, x[i]);
    }
    rng.generateNormalSamples(0.0, noiseVariance, batch2Size, noise);
    fillDataFrame(batch2Size, 0, cols, x, noise, target, *batch2);

    auto baseModel = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                         1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                         .eta({0.02}) // Ensure there are enough trees.
                         .dataSummarizationFraction(1.0)
                         .maximumNumberTrees(3)
                         .numberHoldoutRows(numberHoldoutRows)
                         .buildForTrain(*batch1, cols - 1);
    baseModel->train();

    core::CPackedBitVector batch2RowMask{batch1Size, false};
    batch2RowMask.extend(true, batch2Size);

    double alpha{baseModel->hyperparameters().depthPenaltyMultiplier().value()};
    double gamma{baseModel->hyperparameters().treeSizePenaltyMultiplier().value()};
    double lambda{baseModel->hyperparameters().leafWeightPenaltyMultiplier().value()};

    maths::analytics::CBoostedTreeImplForTest baseImpl{baseModel->impl()};

    auto updateBaseModel = [&](std::size_t maxNumNewTrees) {
        auto updatedModel =
            maths::analytics::CBoostedTreeFactory::constructFromModel(
                baseImpl.cloneFor(*batch1, cols - 1))
                .newTrainingRowMask(batch2RowMask)
                .depthPenaltyMultiplier({0.5 * alpha, 2.0 * alpha})
                .treeSizePenaltyMultiplier({0.5 * gamma, 2.0 * gamma})
                .leafWeightPenaltyMultiplier({0.5 * lambda, 2.0 * lambda})
                .maximumNumberNewTrees(maxNumNewTrees)
                .numberHoldoutRows(numberHoldoutRows)
                .forceAcceptIncrementalTraining(true)
                .buildForTrainIncremental(*batch2, cols - 1);
        updatedModel->trainIncremental();
        updatedModel->predict();
        return updatedModel;
    };

    core::CPackedBitVector holdoutRowMask(numberHoldoutRows, true);
    holdoutRowMask.extend(false, batch1Size + batch2Size - numberHoldoutRows);
    BOOST_REQUIRE_EQUAL(batch2->numberRows(), holdoutRowMask.size());

    auto computeTestError = [&](maths::analytics::CBoostedTreeFactory::TBoostedTreeUPtr&& model) {
        TMeanAccumulator squaredError;
        batch2->readRows(1, 0, batch2->numberRows(),
                         [&](const TRowItr& beginRows, const TRowItr& endRows) {
                             for (auto row = beginRows; row != endRows; ++row) {
                                 squaredError.add(maths::common::CTools::pow2(
                                     (*row)[cols - 1] - model->prediction(*row)[0]));
                             }
                         },
                         &holdoutRowMask);
        return maths::common::CBasicStatistics::mean(squaredError);
    };

    double testError0{computeTestError(updateBaseModel(0))};
    double testError5{computeTestError(updateBaseModel(5))};
    double testError10{computeTestError(updateBaseModel(10))};
    double testErrorBase{computeTestError(std::move(baseModel))};

    LOG_DEBUG(<< "Holdout errors: base = " << testErrorBase << ", 0 new trees = " << testError0
              << ", 5 new trees = " << testError5 << ", 10 new trees = " << testError10);
    BOOST_TEST_REQUIRE(testErrorBase >= testError0);
    BOOST_TEST_REQUIRE(testError0 >= testError5);
    BOOST_TEST_REQUIRE(testError5 >= testError10);
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

        core::CStopWatch watch{true};

        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                2, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .buildForTrain(*frame, cols - 1);

        regression->train();
        regression->predict();

        LOG_DEBUG(<< "took " << watch.lap() << "ms");

        TMeanVarAccumulator modelPredictionErrorMoments;

        frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                modelPredictionErrorMoments.add(target(*row) -
                                                regression->prediction(*row)[0]);
            }
        });

        LOG_DEBUG(<< "model prediction error moments = " << modelPredictionErrorMoments);

        modelBias.push_back(maths::common::CBasicStatistics::mean(modelPredictionErrorMoments));
        modelMse.push_back(maths::common::CBasicStatistics::variance(modelPredictionErrorMoments));

        core::startDefaultAsyncExecutor(2);
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

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

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

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

    regression->train();

    TMeanAccumulator modelPredictionError;

    frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            modelPredictionError.add(1.0 - regression->prediction(*row)[0]);
        }
    });

    LOG_DEBUG(<< "mean prediction error = "
              << maths::common::CBasicStatistics::mean(modelPredictionError));
    BOOST_REQUIRE_EQUAL(0.0, maths::common::CBasicStatistics::mean(modelPredictionError));
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
    frame->writeColumns(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double targetValue{row->index() < trainRows
                                   ? target(*row)
                                   : core::CDataFrame::valueOfMissing()};
            row->writeColumn(cols - 1, targetValue);
        }
    });

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        [&](const TRowRef& row) { return regression->prediction(row)[0]; }, target, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, modelBias, 0.1);
    BOOST_TEST_REQUIRE(modelRSquared > 0.98);
}

BOOST_AUTO_TEST_CASE(testFeatureBags) {

    // Test sampling of feature bags:
    //   1. Simple invariants such as tree bag containts node bag.
    //   2. The selection frequency is almost monotonic increasing with selection
    //      probability.

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
    frame->writeColumns(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double targetValue{row->index() < trainRows
                                   ? target(*row)
                                   : core::CDataFrame::valueOfMissing()};
            row->writeColumn(cols - 1, targetValue);
        }
    });

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

    maths::analytics::CBoostedTreeImplForTest impl{regression->impl()};

    TSizeVec selectedForTree(impl.featureSampleProbabilities().size(), 0);
    TSizeVec selectedForNode(impl.featureSampleProbabilities().size(), 0);

    TDoubleVec probabilities;
    TSizeVec treeFeatureBag;
    TSizeVec nodeFeatureBag;

    for (std::size_t i = 0; i < 5000; ++i) {

        probabilities = impl.featureSampleProbabilities();
        impl.treeFeatureBag(probabilities, treeFeatureBag);
        for (auto j : treeFeatureBag) {
            ++selectedForTree[j];
        }

        BOOST_TEST_REQUIRE(treeFeatureBag.empty() == false);
        BOOST_TEST_REQUIRE(
            *std::max_element(treeFeatureBag.begin(), treeFeatureBag.end()) <
            probabilities.size());
        BOOST_TEST_REQUIRE(std::is_sorted(treeFeatureBag.begin(), treeFeatureBag.end()));

        probabilities = impl.featureSampleProbabilities();
        impl.nodeFeatureBag(treeFeatureBag, probabilities, nodeFeatureBag);
        for (auto j : nodeFeatureBag) {
            ++selectedForNode[j];
        }

        BOOST_TEST_REQUIRE(nodeFeatureBag.empty() == false);
        BOOST_TEST_REQUIRE(std::is_sorted(nodeFeatureBag.begin(), nodeFeatureBag.end()));

        TSizeVec difference;
        std::set_difference(nodeFeatureBag.begin(), nodeFeatureBag.end(),
                            treeFeatureBag.begin(), treeFeatureBag.end(),
                            std::back_inserter(difference));
        BOOST_TEST_REQUIRE(difference.empty());
        BOOST_REQUIRE_EQUAL(nodeFeatureBag.end(),
                            std::unique(nodeFeatureBag.begin(), nodeFeatureBag.end()));
    }

    probabilities = impl.featureSampleProbabilities();
    maths::common::COrderings::simultaneousSort(probabilities, selectedForTree, selectedForNode);

    auto distanceToSorted = [](const TSizeVec& selected) {
        TSizeVec selectedSorted{selected};
        std::sort(selectedSorted.begin(), selectedSorted.end());

        std::size_t distance{0};
        for (std::size_t i = 0; i < selected.size(); ++i) {
            distance += std::max(selected[i], selectedSorted[i]) -
                        std::min(selected[i], selectedSorted[i]);
        }
        return static_cast<double>(distance) /
               static_cast<double>(std::accumulate(selected.begin(),
                                                   selected.end(), std::size_t{0}));
    };

    LOG_DEBUG(<< "distanceToSorted(selectedForTree) = " << distanceToSorted(selectedForTree)
              << ", distanceToSorted(selectedForNode) = " << distanceToSorted(selectedForNode));
    BOOST_TEST_REQUIRE(distanceToSorted(selectedForTree) < 0.008);
    BOOST_TEST_REQUIRE(distanceToSorted(selectedForNode) < 0.01);
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

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, 1);

    regression->train();
    regression->predict();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, trainRows, rows,
        [&](const TRowRef& row) { return regression->prediction(row)[0]; },
        [&](const TRowRef& x) { return 10.0 * x[0]; }, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, modelBias, 0.082);
    BOOST_TEST_REQUIRE(modelRSquared > 0.96);
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

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

    regression->train();

    double modelBias;
    double modelRSquared;
    std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
        *frame, 0, rows,
        [&](const TRowRef& row) { return regression->prediction(row)[0]; },
        [](const TRowRef& row) { return 10.0 * row[0]; }, 0.0);

    LOG_DEBUG(<< "bias = " << modelBias);
    LOG_DEBUG(<< " R^2 = " << modelRSquared);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, modelBias, 0.21);
    BOOST_TEST_REQUIRE(modelRSquared > 0.95);
}

BOOST_AUTO_TEST_CASE(testTranslationInvariance) {

    // We should get similar performance independent of fixed shifts for the target.

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

        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .buildForTrain(*frame, cols - 1);

        regression->train();
        regression->predict();

        double modelBias;
        double modelRSquared;
        std::tie(modelBias, modelRSquared) = computeEvaluationMetrics(
            *frame, trainRows, rows,
            [&](const TRowRef& row) { return regression->prediction(row)[0]; },
            target_, 0.0);

        LOG_DEBUG(<< "bias = " << modelBias);
        LOG_DEBUG(<< " R^2 = " << modelRSquared);
        rsquared.push_back(modelRSquared);
    }

    BOOST_REQUIRE_CLOSE_ABSOLUTE(rsquared[0], rsquared[1], 0.01);
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

        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .treeSizePenaltyMultiplier({0.0})
                .leafWeightPenaltyMultiplier({0.0})
                .softTreeDepthLimit({targetDepth})
                .softTreeDepthTolerance({0.05})
                .buildForTrain(*frame, cols - 1);

        regression->train();

        TMeanAccumulator meanDepth;
        for (const auto& tree : regression->trainedModel()) {
            BOOST_TEST_REQUIRE(maxDepth(tree, tree[0], 0) <=
                               static_cast<std::size_t>(targetDepth + 1));
            meanDepth.add(static_cast<double>(maxDepth(tree, tree[0], 0)));
        }
        LOG_DEBUG(<< "mean depth = " << maths::common::CBasicStatistics::mean(meanDepth));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanDepth) >
                           targetDepth - 1.2);
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
            return maths::common::CTools::logisticFunction(x + noise[row.index()]);
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
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
                .buildForTrain(*frame, cols - 1);

        classifier->train();
        classifier->predict();

        TMeanAccumulator logRelativeError;
        frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (row->index() >= trainRows) {
                    double expectedProbability{probability(*row)};
                    double actualProbability{classifier->prediction(*row)[1]};
                    logRelativeError.add(
                        std::log(std::max(actualProbability, expectedProbability) /
                                 std::min(actualProbability, expectedProbability)));
                }
            }
        });
        LOG_DEBUG(<< "log relative error = "
                  << maths::common::CBasicStatistics::mean(logRelativeError));

        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(logRelativeError) < 0.7);
        meanLogRelativeError.add(maths::common::CBasicStatistics::mean(logRelativeError));
    }

    LOG_DEBUG(<< "mean log relative error = "
              << maths::common::CBasicStatistics::mean(meanLogRelativeError));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanLogRelativeError) < 0.53);
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticIncrementalForTargetDrift) {

    // Test incremental training for the binomial logistic objective for drift in the
    // target value.

    test::CRandomNumbers rng;
    std::size_t rows{750};
    std::size_t cols{5};
    std::size_t extraTrainingRows{500};

    auto probability = [&] {
        TDoubleVec weights{-0.6, 1.0, 0.8, -1.2};
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 0.5, rows + extraTrainingRows, noise);
        return [=](const TRowRef& row) {
            double x{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                x += weights[i] * row[i];
            }
            return maths::common::CTools::logisticFunction(x + noise[row.index()]);
        };
    }();
    auto perturbedProbability = [&] {
        TDoubleVec weights{-0.4, 1.3, 0.9, -1.8};
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 0.5, rows + extraTrainingRows, noise);
        return [=](const TRowRef& row) {
            double x{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                x += weights[i] * row[i];
            }
            return maths::common::CTools::logisticFunction(x + noise[row.index()]);
        };
    }();

    auto target = [&] {
        TDoubleVec uniform01;
        rng.generateUniformSamples(0.0, 1.0, rows + extraTrainingRows, uniform01);
        return [=](const TRowRef& row) {
            return uniform01[row.index()] < probability(row) ? 1.0 : 0.0;
        };
    }();
    auto perturbedTarget = [&] {
        TDoubleVec uniform01;
        rng.generateUniformSamples(0.0, 1.0, rows + extraTrainingRows, uniform01);
        return [=](const TRowRef& row) {
            return uniform01[row.index()] < perturbedProbability(row) ? 1.0 : 0.0;
        };
    }();

    auto frame = core::makeMainStorageDataFrame(cols).first;

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(-3.0, 3.0, rows, x[i]);
    }

    fillDataFrame(rows, 0, cols, {false, false, false, false, true}, x,
                  TDoubleVec(rows, 0.0), target, *frame);

    auto classifier =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
            .buildForTrain(*frame, cols - 1);
    classifier->train();

    frame->resizeColumns(1, cols);
    auto newFrame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(rows, 0, cols, {false, false, false, false, true}, x,
                  TDoubleVec(rows, 0.0), target, *newFrame);

    // If the target drift isn't on a subset of feature space it is an even trade increasing
    // errors for old and decreasing for new.
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 3.0, extraTrainingRows, x[i]);
    }

    fillDataFrame(extraTrainingRows, 0, cols, {false, false, false, false, true},
                  x, TDoubleVec(extraTrainingRows, 0.0), perturbedTarget, *frame);
    fillDataFrame(extraTrainingRows, 0, cols, {false, false, false, false, true}, x,
                  TDoubleVec(extraTrainingRows, 0.0), perturbedTarget, *newFrame);

    classifier->predict();

    core::CPackedBitVector oldTrainingRowMask{rows, true};
    oldTrainingRowMask.extend(false, extraTrainingRows);
    core::CPackedBitVector newTrainingRowMask{~oldTrainingRowMask};

    TMeanAccumulator logRelativeErrorBeforeIncrementalTraining[2];
    TMeanAccumulator logRelativeErrorAfterIncrementalTraining[2];

    auto addToRelativeError = [&](const TRowRef& row, TMeanAccumulator& logRelativeError) {
        double expectedProbability{probability(row)};
        double actualProbability{classifier->prediction(row)[1]};
        logRelativeError.add(std::log(std::max(actualProbability, expectedProbability) /
                                      std::min(actualProbability, expectedProbability)));
    };
    auto addToPerturbedRelativeError = [&](const TRowRef& row, TMeanAccumulator& logRelativeError) {
        double expectedProbability{perturbedProbability(row)};
        double actualProbability{classifier->prediction(row)[1]};
        logRelativeError.add(std::log(std::max(actualProbability, expectedProbability) /
                                      std::min(actualProbability, expectedProbability)));
    };

    frame->resizeColumns(1, cols + 1);
    classifier->predict();

    // Get the average error on the old and new training data from the old model.
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            addToRelativeError(
                                *row, logRelativeErrorBeforeIncrementalTraining[OLD_TRAINING_DATA]);
                        }
                    },
                    &oldTrainingRowMask);
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            addToPerturbedRelativeError(
                                *row, logRelativeErrorBeforeIncrementalTraining[NEW_TRAINING_DATA]);
                        }
                    },
                    &newTrainingRowMask);

    double alpha{classifier->hyperparameters().depthPenaltyMultiplier().value()};
    double gamma{classifier->hyperparameters().treeSizePenaltyMultiplier().value()};
    double lambda{classifier->hyperparameters().leafWeightPenaltyMultiplier().value()};
    classifier = maths::analytics::CBoostedTreeFactory::constructFromModel(std::move(classifier))
                     .newTrainingRowMask(newTrainingRowMask)
                     .depthPenaltyMultiplier({0.5 * alpha, 2.0 * alpha})
                     .treeSizePenaltyMultiplier({0.5 * gamma, 2.0 * gamma})
                     .leafWeightPenaltyMultiplier({0.5 * lambda, 2.0 * lambda})
                     .buildForTrainIncremental(*newFrame, cols - 1);

    classifier->trainIncremental();
    classifier->predict();

    // Get the average error on the old and new training data from the new model.
    newFrame->readRows(1, 0, frame->numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               addToRelativeError(
                                   *row, logRelativeErrorAfterIncrementalTraining[OLD_TRAINING_DATA]);
                           }
                       },
                       &oldTrainingRowMask);
    newFrame->readRows(1, 0, frame->numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               addToPerturbedRelativeError(
                                   *row, logRelativeErrorAfterIncrementalTraining[NEW_TRAINING_DATA]);
                           }
                       },
                       &newTrainingRowMask);

    // We should see little change in the prediction errors on the old data
    // set which doesn't overlap the new data, but a large improvement on
    // the new data for constant extrapolation performs poorly.
    double errorIncreaseOnOld{
        maths::common::CBasicStatistics::mean(
            logRelativeErrorAfterIncrementalTraining[OLD_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(
                logRelativeErrorBeforeIncrementalTraining[OLD_TRAINING_DATA]) -
        1.0};
    double errorDecreaseOnNew{
        maths::common::CBasicStatistics::mean(
            logRelativeErrorBeforeIncrementalTraining[NEW_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(
                logRelativeErrorAfterIncrementalTraining[NEW_TRAINING_DATA]) -
        1.0};

    LOG_DEBUG(<< "increase on old = " << errorIncreaseOnOld);
    LOG_DEBUG(<< "decrease on new = " << errorDecreaseOnNew);
    BOOST_TEST_REQUIRE(errorDecreaseOnNew > 100.0 * errorIncreaseOnOld);
}

BOOST_AUTO_TEST_CASE(testBinomialLogisticIncrementalForOutOfDomain) {

    // Test incremental training for binomial logistic objective for values out of the
    // training data domain.

    test::CRandomNumbers rng;
    std::size_t rows{750};
    std::size_t cols{5};
    std::size_t extraTrainingRows{250};

    auto probability = [&] {
        TDoubleVec weights{-0.6, 1.0, -0.8, 1.2};
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 0.2, rows + extraTrainingRows, noise);
        return [=](const TRowRef& row) {
            double x{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                x += weights[i] * row[i];
            }
            return maths::common::CTools::logisticFunction(x + noise[row.index()]);
        };
    }();

    auto target = [&] {
        TDoubleVec uniform01;
        rng.generateUniformSamples(0.0, 1.0, rows + extraTrainingRows, uniform01);
        return [=](const TRowRef& row) {
            return uniform01[row.index()] < probability(row) ? 1.0 : 0.0;
        };
    }();

    auto frame = core::makeMainStorageDataFrame(cols).first;

    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(-3.0, 0.0, rows, x[i]);
    }

    fillDataFrame(rows, 0, cols, {false, false, false, false, true}, x,
                  TDoubleVec(rows, 0.0), target, *frame);

    auto classifier =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
            .buildForTrain(*frame, cols - 1);
    classifier->train();

    frame->resizeColumns(1, cols);
    auto newFrame = core::makeMainStorageDataFrame(cols).first;

    fillDataFrame(rows, 0, cols, {false, false, false, false, true}, x,
                  TDoubleVec(rows, 0.0), target, *newFrame);

    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 3.0, extraTrainingRows, x[i]);
    }

    fillDataFrame(extraTrainingRows, 0, cols, {false, false, false, false, true},
                  x, TDoubleVec(extraTrainingRows, 0.0), target, *frame);
    fillDataFrame(extraTrainingRows, 0, cols, {false, false, false, false, true},
                  x, TDoubleVec(extraTrainingRows, 0.0), target, *newFrame);

    classifier->predict();

    core::CPackedBitVector oldTrainingRowMask{rows, true};
    oldTrainingRowMask.extend(false, extraTrainingRows);
    core::CPackedBitVector newTrainingRowMask{~oldTrainingRowMask};

    TMeanAccumulator logRelativeErrorBeforeIncrementalTraining[2];
    TMeanAccumulator logRelativeErrorAfterIncrementalTraining[2];

    auto addToRelativeError = [&](const TRowRef& row, TMeanAccumulator& logRelativeError) {
        double expectedProbability{probability(row)};
        double actualProbability{classifier->prediction(row)[1]};
        logRelativeError.add(std::log(std::max(actualProbability, expectedProbability) /
                                      std::min(actualProbability, expectedProbability)));
    };

    frame->resizeColumns(1, cols + 1);
    classifier->predict();

    // Get the average error on the old and new training data from the old model.
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            addToRelativeError(
                                *row, logRelativeErrorBeforeIncrementalTraining[OLD_TRAINING_DATA]);
                        }
                    },
                    &oldTrainingRowMask);
    frame->readRows(1, 0, frame->numberRows(),
                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            addToRelativeError(
                                *row, logRelativeErrorBeforeIncrementalTraining[NEW_TRAINING_DATA]);
                        }
                    },
                    &newTrainingRowMask);

    double alpha{classifier->hyperparameters().depthPenaltyMultiplier().value()};
    double gamma{classifier->hyperparameters().treeSizePenaltyMultiplier().value()};
    double lambda{classifier->hyperparameters().leafWeightPenaltyMultiplier().value()};
    classifier = maths::analytics::CBoostedTreeFactory::constructFromModel(std::move(classifier))
                     .newTrainingRowMask(newTrainingRowMask)
                     .depthPenaltyMultiplier({0.5 * alpha, 2.0 * alpha})
                     .treeSizePenaltyMultiplier({0.5 * gamma, 2.0 * gamma})
                     .leafWeightPenaltyMultiplier({0.5 * lambda, 2.0 * lambda})
                     .buildForTrainIncremental(*newFrame, cols - 1);

    classifier->trainIncremental();
    classifier->predict();

    // Get the average error on the old and new training data from the new model.
    newFrame->readRows(1, 0, frame->numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               addToRelativeError(
                                   *row, logRelativeErrorAfterIncrementalTraining[OLD_TRAINING_DATA]);
                           }
                       },
                       &oldTrainingRowMask);
    newFrame->readRows(1, 0, frame->numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               addToRelativeError(
                                   *row, logRelativeErrorAfterIncrementalTraining[NEW_TRAINING_DATA]);
                           }
                       },
                       &newTrainingRowMask);

    // We should see little change in the prediction errors on the old data
    // set which doesn't overlap the new data, but a large improvement on
    // the new data for constant extrapolation performs poorly.
    double errorIncreaseOnOld{
        maths::common::CBasicStatistics::mean(
            logRelativeErrorAfterIncrementalTraining[OLD_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(
                logRelativeErrorBeforeIncrementalTraining[OLD_TRAINING_DATA]) -
        1.0};
    double errorDecreaseOnNew{
        maths::common::CBasicStatistics::mean(
            logRelativeErrorBeforeIncrementalTraining[NEW_TRAINING_DATA]) /
            maths::common::CBasicStatistics::mean(
                logRelativeErrorAfterIncrementalTraining[NEW_TRAINING_DATA]) -
        1.0};

    LOG_DEBUG(<< "increase on old = " << errorIncreaseOnOld);
    LOG_DEBUG(<< "decrease on new = " << errorDecreaseOnNew);
    BOOST_TEST_REQUIRE(errorDecreaseOnNew > 40.0 * errorIncreaseOnOld);
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
    for (std::size_t i = 0; i < std::size(classes); ++i) {
        TDoubleVecVec xi;
        double mean{means[classes[i]]};
        double variance{variances[classes[i]]};
        rng.generateMultivariateNormalSamples(
            {mean, mean}, {{variance, 0.0}, {0.0, variance}}, classesRowCounts[i], xi);
        x.insert(x.end(), xi.begin(), xi.end());
    }

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{false, false, true});
    for (std::size_t i = 0, index = 0; i < std::size(classes); ++i) {
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

    auto classifier =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
            .buildForTrain(*frame, cols - 1);

    classifier->train();
    classifier->predict();

    TDoubleVec precisions;
    TDoubleVec recalls;
    {
        TDoubleVec truePositives(2, 0.0);
        TDoubleVec trueNegatives(2, 0.0);
        TDoubleVec falsePositives(2, 0.0);
        TDoubleVec falseNegatives(2, 0.0);
        frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                auto scores = classifier->adjustedPrediction(*row);
                std::size_t prediction(scores[1] < scores[0] ? 0 : 1);
                if (row->index() >= trainRows &&
                    row->index() < trainRows + classesRowCounts[2]) {
                    // Actual is zero.
                    (prediction == 0 ? truePositives[0] : falseNegatives[0]) += 1.0;
                    (prediction == 0 ? trueNegatives[1] : falsePositives[1]) += 1.0;
                } else if (row->index() >= trainRows + classesRowCounts[2]) {
                    // Actual is one.
                    (prediction == 1 ? truePositives[1] : falseNegatives[1]) += 1.0;
                    (prediction == 1 ? trueNegatives[0] : falsePositives[0]) += 1.0;
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
    BOOST_TEST_REQUIRE(std::fabs(recalls[0] - recalls[1]) < 0.16);
}

BOOST_AUTO_TEST_CASE(testClassificationWeightsOverride) {

    // Test that the overrides are reflected in the classification weights used.

    using TStrVec = std::vector<std::string>;

    test::CRandomNumbers rng;

    std::size_t classes[]{1, 0};
    std::size_t classesRowCounts[]{50, 50};
    std::size_t cols{3};

    TDoubleVecVec x;
    TDoubleVec means{0.0, 3.0};
    TDoubleVec variances{6.0, 6.0};
    for (std::size_t i = 0; i < std::size(classes); ++i) {
        TDoubleVecVec xi;
        double mean{means[classes[i]]};
        double variance{variances[classes[i]]};
        rng.generateMultivariateNormalSamples(
            {mean, mean}, {{variance, 0.0}, {0.0, variance}}, classesRowCounts[i], xi);
        x.insert(x.end(), xi.begin(), xi.end());
    }

    auto frame = core::makeMainStorageDataFrame(cols).first;
    frame->categoricalColumns(TBoolVec{false, false, true});

    std::string classValues[]{"foo", "bar"};
    TStrVec row(cols);
    for (std::size_t i = 0, index = 0; i < std::size(classes); ++i) {
        for (std::size_t j = 0; j < classesRowCounts[i]; ++j, ++index) {
            row[0] = std::to_string(x[index][0]);
            row[1] = std::to_string(x[index][1]);
            row[2] = classValues[i];
            frame->parseAndWriteRow({row, 0, cols});
        }
    }
    frame->finishWritingRows();

    auto classifier =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
            .classAssignmentObjective(maths::analytics::CBoostedTree::E_Custom)
            .classificationWeights({{"foo", 0.8}, {"bar", 0.2}})
            .buildForTrain(*frame, cols - 1);

    classifier->train();

    BOOST_TEST_REQUIRE("[0.8, 0.2]", core::CContainerPrinter::print(
                                         classifier->classificationWeights()));
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

    using TVector = maths::common::CDenseVector<double>;
    using TMemoryMappedMatrix = maths::common::CMemoryMappedDenseMatrix<double>;

    maths::common::CPRNG::CXorOShiro128Plus rng;
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
            TVector result{W * x + n};
            maths::common::CTools::inplaceSoftmax(result);
            return result;
        };

        auto target = [&](const TRowRef& row) {
            TDoubleVec probabilities{probability(row).to<TDoubleVec>()};
            return static_cast<double>(
                maths::common::CSampling::categoricalSample(rng, probabilities));
        };

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            testRng.generateUniformSamples(0.0, 4.0, rows, x[i]);
        }

        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

        fillDataFrame(trainRows, rows - trainRows, cols, {false, false, false, true},
                      x, TDoubleVec(rows, 0.0), target, *frame);

        auto classifier =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMultinomialLogisticLoss>(numberClasses))
                .buildForTrain(*frame, cols - 1);

        classifier->train();
        classifier->predict();

        TMeanAccumulator logRelativeError;
        frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (row->index() >= trainRows) {
                    TVector expectedProbability{probability(*row)};
                    TVector actualProbability{
                        TVector::fromSmallVector(classifier->prediction(*row))};
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
                  << maths::common::CBasicStatistics::mean(logRelativeError));

        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(logRelativeError) < 1.9);
        meanLogRelativeError.add(maths::common::CBasicStatistics::mean(logRelativeError));
    }

    LOG_DEBUG(<< "mean log relative error = "
              << maths::common::CBasicStatistics::mean(meanLogRelativeError));
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanLogRelativeError) < 1.4);
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

        std::int64_t estimatedMemory(
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .estimateMemoryUsageForTrain(rows, cols));

        CTestInstrumentation instrumentation;
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .analysisInstrumentation(instrumentation)
                .buildForTrain(*frame, cols - 1);

        regression->train();

        LOG_DEBUG(<< "estimated memory usage = " << estimatedMemory);
        LOG_DEBUG(<< "high water mark = " << instrumentation.maxMemoryUsage());

        BOOST_TEST_REQUIRE(instrumentation.maxMemoryUsage() < estimatedMemory);
    }
}

// TODO: test for handle fatal (memory-limit-circuit-breaker)
BOOST_AUTO_TEST_CASE(testEstimateMemoryUsedByTrainWithTestRows) {

    // Test estimation of the memory used training a model.

    test::CRandomNumbers rng;

    std::size_t rows{1000};
    std::size_t cols{6};
    std::size_t capacity{600};
    std::int64_t previousEstimatedMemory{std::numeric_limits<std::int64_t>::max()};

    for (std::size_t test = 0; test < 3; ++test) {
        TDoubleVecVec x(cols - 1);
        std::size_t numTestRows{((test + 1) * 100)};
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
                if (i < numTestRows) {
                    *column = core::CDataFrame::valueOfMissing();
                } else {
                    *column = target(i);
                }
            });
        }
        frame->finishWritingRows();

        double percentTrainingRows{1.0 - static_cast<double>(numTestRows) /
                                             static_cast<double>(rows)};
        std::size_t trainRows{static_cast<std::size_t>(static_cast<double>(rows) *
                                                       percentTrainingRows)};

        std::int64_t estimatedMemory(
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .estimateMemoryUsageForTrain(trainRows, cols));

        CTestInstrumentation instrumentation;
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .analysisInstrumentation(instrumentation)
                .buildForTrain(*frame, cols - 1);

        regression->train();

        LOG_DEBUG(<< "percent training rows = " << percentTrainingRows);
        LOG_DEBUG(<< "estimated memory usage = " << estimatedMemory);
        LOG_DEBUG(<< "high water mark = " << instrumentation.maxMemoryUsage());

        BOOST_TEST_REQUIRE(instrumentation.maxMemoryUsage() < estimatedMemory);
        BOOST_TEST_REQUIRE(previousEstimatedMemory > estimatedMemory);
        previousEstimatedMemory = estimatedMemory;
    }
}

BOOST_AUTO_TEST_CASE(testDeployedMemoryLimiting) {

    // Test we always produce a model which meets our memory constraint.

    test::CRandomNumbers rng;
    double noiseVariance{10.0};
    std::size_t trainRows{800};
    std::size_t testRows{200};
    std::size_t rows{trainRows + testRows};
    std::size_t cols{8};
    std::size_t capacity{250};
    std::size_t maximumDeployedSize{20000};

    auto target = [&] {
        TDoubleVec m;
        TDoubleVec s;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
        rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
        return [=](const TRowRef& row) {
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

    auto frameConstrained = core::makeMainStorageDataFrame(cols, capacity).first;
    fillDataFrame(trainRows, testRows, cols, x, noise, target, *frameConstrained);
    auto regressionConstrained =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CMse>())
            .maximumDeployedSize(maximumDeployedSize)
            .numberFolds(2)
            .buildForTrain(*frameConstrained, cols - 1);
    regressionConstrained->train();
    regressionConstrained->predict();

    auto frameUnconstrained = core::makeMainStorageDataFrame(cols, capacity).first;
    fillDataFrame(trainRows, testRows, cols, x, noise, target, *frameUnconstrained);
    auto regressionUnconstrained =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CMse>())
            .numberFolds(2)
            .buildForTrain(*frameUnconstrained, cols - 1);
    regressionUnconstrained->train();
    regressionUnconstrained->predict();

    std::size_t deployedSizeConstrained{
        std::accumulate(regressionConstrained->trainedModel().begin(),
                        regressionConstrained->trainedModel().end(),
                        std::size_t{0}, [](std::size_t sum, const auto& tree) {
                            for (const auto& node : tree) {
                                sum += node.deployedSize();
                            }
                            return sum;
                        })};
    std::size_t deployedSizeUnconstrained{
        std::accumulate(regressionUnconstrained->trainedModel().begin(),
                        regressionUnconstrained->trainedModel().end(),
                        std::size_t{0}, [](std::size_t sum, const auto& tree) {
                            for (const auto& node : tree) {
                                sum += node.deployedSize();
                            }
                            return sum;
                        })};

    auto[modelBiasConstrained, modelRSquaredConstrained] =
        computeEvaluationMetrics(*frameConstrained, trainRows, rows,
                                 [&](const TRowRef& row) {
                                     return regressionConstrained->prediction(row)[0];
                                 },
                                 target, 0.0);

    auto[modelBiasUnconstrained, modelRSquaredUnconstrained] = computeEvaluationMetrics(
        *frameUnconstrained, trainRows, rows,
        [&](const TRowRef& row) {
            return regressionUnconstrained->prediction(row)[0];
        },
        target, 0.0);

    LOG_DEBUG(<< "deployedSizeConstrained = " << deployedSizeConstrained
              << ", modelBiasConstrained = " << modelBiasConstrained
              << ", modelRSquaredConstrained = " << modelRSquaredConstrained);
    LOG_DEBUG(<< "deployedSizeUnconstrained = " << deployedSizeUnconstrained
              << ", modelBiasUnconstrained = " << modelBiasUnconstrained
              << ", modelRSquaredUnconstrained = " << modelRSquaredUnconstrained);

    // We don't hurt model accuracy by more than 3%.
    BOOST_TEST_REQUIRE(modelRSquaredConstrained > 0.97 * modelRSquaredUnconstrained);

    // We need to constrain vs the optimum for this data and we are within
    // the memory budget.
    BOOST_TEST_REQUIRE(deployedSizeConstrained < maximumDeployedSize);
    BOOST_TEST_REQUIRE(deployedSizeConstrained < deployedSizeUnconstrained);
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
            auto regression =
                maths::analytics::CBoostedTreeFactory::constructFromParameters(
                    threads, std::make_unique<maths::analytics::boosted_tree::CMse>())
                    .analysisInstrumentation(instrumentation)
                    .earlyStoppingEnabled(false)
                    .buildForTrain(*frame, cols - 1);

            regression->train();
            finished.store(true);
        }};
        worker.join();

        // Flush the last task...
        instrumentation.startNewProgressMonitoredTask("");

        for (const auto& task : instrumentation.taskProgress()) {
            LOG_DEBUG(<< "task = " << task.s_Name);
            LOG_DEBUG(<< "monotonic = " << task.s_Monotonic);
            LOG_DEBUG(<< "progress points = "
                      << core::CContainerPrinter::print(task.s_TenPercentProgressPoints));
            if (task.s_Name == maths::analytics::CBoostedTreeFactory::FEATURE_SELECTION) {
                // We don't do feature selection (we have enough data to use all of them).
            } else if (task.s_Name == maths::analytics::CBoostedTreeFactory::COARSE_PARAMETER_SEARCH) {
                // We don't have accurate upfront estimate of the number of steps so
                // we only get progress up to 80% or 90% depending on the compiler and
                // platform. In non-test code we always pass 100% when the task is
                // complete.
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.size() >= 9);
                BOOST_TEST_REQUIRE(
                    std::is_sorted(task.s_TenPercentProgressPoints.begin(),
                                   task.s_TenPercentProgressPoints.end()));
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.front() == 0);
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.back() >= 80);
            } else if (task.s_Name == maths::analytics::CBoostedTreeFactory::FINE_TUNING_PARAMETERS) {
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.size() >= 10);
                BOOST_TEST_REQUIRE(
                    std::is_sorted(task.s_TenPercentProgressPoints.begin(),
                                   task.s_TenPercentProgressPoints.end()));
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.front() == 0);
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.back() >= 90);
            } else if (task.s_Name == maths::analytics::CBoostedTreeFactory::FINAL_TRAINING) {
                // Progress might be 90% or 100% depending on whether the final
                // progress update registered.
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.size() >= 10);
                BOOST_TEST_REQUIRE(
                    std::is_sorted(task.s_TenPercentProgressPoints.begin(),
                                   task.s_TenPercentProgressPoints.end()));
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.front() == 0);
                BOOST_TEST_REQUIRE(task.s_TenPercentProgressPoints.back() >= 90);
            }
            BOOST_TEST_REQUIRE(task.s_Monotonic);
        }

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

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);

    regression->train();
    regression->predict();

    // For each missing value given we censor for regression variables greater
    // than 9.0, target = sum_i{R_i} for regressors R_i and R_i ~ U([0,10]) so
    // we expect a n * E[U([0, 10]) | U([0, 10]) > 9.0] = 9.5 * n contribution
    // to the target for n equal to the number of missing variables.
    TDoubleVec expectedPredictions{17.5, 17.5, 17.5, 22.0};
    TDoubleVec actualPredictions;

    frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            if (maths::analytics::CDataFrameUtils::isMissing((*row)[cols - 1])) {
                actualPredictions.push_back(regression->prediction(*row)[0]);
            }
        }
    });

    BOOST_REQUIRE_EQUAL(expectedPredictions.size(), actualPredictions.size());
    for (std::size_t i = 0; i < expectedPredictions.size(); ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedPredictions[i], actualPredictions[i], 0.9);
    }
}

BOOST_AUTO_TEST_CASE(testHyperparameterOverrides) {

    // Test hyperparameter overrides are respected.

    test::CRandomNumbers rng;

    std::size_t rows{300};
    std::size_t cols{6};
    std::size_t capacity{600};

    TDoubleVecVec x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    auto target = [](const TRowRef& row) { return row[0] + 3.0 * row[3]; };

    auto frame = core::makeMainStorageDataFrame(cols, capacity).first;

    fillDataFrame(rows, 0, cols, x, TDoubleVec(rows, 0.0), target, *frame);

    CTestInstrumentation instrumentation;
    {
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .analysisInstrumentation(instrumentation)
                .maximumNumberTrees(10)
                .treeSizePenaltyMultiplier({0.1})
                .leafWeightPenaltyMultiplier({0.01})
                .buildForTrain(*frame, cols - 1);

        regression->train();

        // We use a single leaf to centre the data so end up with *at most* limit + 1 trees.
        BOOST_TEST_REQUIRE(regression->hyperparameters().maximumNumberTrees().value() <= 11);
        BOOST_REQUIRE_EQUAL(
            10, regression->hyperparameters().maximumNumberTrees().value());
        BOOST_REQUIRE_EQUAL(
            0.1, regression->hyperparameters().treeSizePenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(
            0.01, regression->hyperparameters().leafWeightPenaltyMultiplier().value());
    }
    {
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .analysisInstrumentation(instrumentation)
                .eta({0.2})
                .softTreeDepthLimit({2.0})
                .softTreeDepthTolerance({0.1})
                .buildForTrain(*frame, cols - 1);

        regression->train();

        BOOST_REQUIRE_EQUAL(0.2, regression->hyperparameters().eta().value());
        BOOST_REQUIRE_EQUAL(
            2.0, regression->hyperparameters().softTreeDepthLimit().value());
        BOOST_REQUIRE_EQUAL(
            0.1, regression->hyperparameters().softTreeDepthTolerance().value());
    }
    {
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .analysisInstrumentation(instrumentation)
                .depthPenaltyMultiplier({1.0})
                .featureBagFraction({0.4})
                .downsampleFactor({0.6})
                .buildForTrain(*frame, cols - 1);

        regression->train();

        BOOST_REQUIRE_EQUAL(
            1.0, regression->hyperparameters().depthPenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(
            0.4, regression->hyperparameters().featureBagFraction().value());
        BOOST_REQUIRE_EQUAL(
            0.6, regression->hyperparameters().downsampleFactor().value());
    }
    {
        auto regression =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .analysisInstrumentation(instrumentation)
                .depthPenaltyMultiplier({1.0})
                .softTreeDepthLimit({3.0})
                .softTreeDepthTolerance({0.1})
                .featureBagFraction({0.4})
                .downsampleFactor({0.6})
                .etaGrowthRatePerTree({1.1})
                .buildForTrain(*frame, cols - 1);

        regression->train();
        BOOST_REQUIRE_EQUAL(
            1.0, regression->hyperparameters().depthPenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(
            3.0, regression->hyperparameters().softTreeDepthLimit().value());
        BOOST_REQUIRE_EQUAL(
            0.1, regression->hyperparameters().softTreeDepthTolerance().value());
        BOOST_REQUIRE_EQUAL(
            0.4, regression->hyperparameters().featureBagFraction().value());
        BOOST_REQUIRE_EQUAL(
            1.1, regression->hyperparameters().etaGrowthRatePerTree().value());
    }
}

BOOST_AUTO_TEST_CASE(testPersistRestore) {

    // Check persist + restore is idempotent.

    std::size_t rows{50};
    std::size_t cols{3};
    std::size_t capacity{50};
    test::CRandomNumbers rng;

    std::stringstream persistOnceState;
    std::stringstream persistTwiceState;

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

    // Persist.
    {
        auto boostedTree =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .numberFolds(2)
                .maximumNumberTrees(2)
                .maximumOptimisationRoundsPerHyperparameter(3)
                .buildForTrain(*frame, cols - 1);
        boostedTree->train();
        core::CJsonStatePersistInserter inserter(persistOnceState);
        boostedTree->acceptPersistInserter(inserter);
        persistOnceState.flush();
    }
    // Restore.
    auto boostedTree = maths::analytics::CBoostedTreeFactory::constructFromString(persistOnceState)
                           .restoreFor(*frame, cols - 1);
    {
        // We need to call the inserter destructor to finish writing the state.
        core::CJsonStatePersistInserter inserter(persistTwiceState);
        boostedTree->acceptPersistInserter(inserter);
        persistTwiceState.flush();
    }
    LOG_DEBUG(<< "State " << persistOnceState.str());
    LOG_DEBUG(<< "State after restore " << persistTwiceState.str());
    BOOST_REQUIRE_EQUAL(persistOnceState.str(), persistTwiceState.str());
}

BOOST_AUTO_TEST_CASE(testPersistRestoreDuringInitialization) {

    // Grab checkpoints during initialization and check they all produce the
    // same initialized tree after restoring.

    using TSStreamVec = std::vector<std::stringstream>;

    std::size_t rows{50};
    std::size_t cols{3};
    std::size_t capacity{50};
    test::CRandomNumbers rng;

    // Generate completely random data.
    TDoubleVecVec x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    auto makeFrame = [&] {
        auto frame = core::makeMainStorageDataFrame(cols, capacity).first;
        for (std::size_t i = 0; i < rows; ++i) {
            frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
                for (std::size_t j = 0; j < cols; ++j, ++column) {
                    *column = x[j][i];
                }
            });
        }
        frame->finishWritingRows();
        return frame;
    };

    TSStreamVec checkpoints;

    auto writeCheckpoint = [&checkpoints](maths::analytics::CBoostedTree::TPersistFunc persist) {
        std::stringstream state;
        {
            // We need to call the inserter destructor to finish writing the state.
            core::CJsonStatePersistInserter inserter(state);
            persist(inserter);
            state.flush();
        }
        checkpoints.push_back(std::move(state));
    };

    std::ostringstream expectedState;
    {
        auto frame = makeFrame();
        auto boostedTree =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .numberFolds(2)
                .maximumNumberTrees(2)
                .maximumOptimisationRoundsPerHyperparameter(3)
                .trainingStateCallback(writeCheckpoint)
                .buildForTrain(*frame, cols - 1);
        core::CJsonStatePersistInserter inserter(expectedState);
        boostedTree->acceptPersistInserter(inserter);
        expectedState.flush();
    }

    // Make sure this test fails if there aren't any checkpoints.
    BOOST_TEST_REQUIRE(checkpoints.size() > 0);

    for (auto& checkpoint : checkpoints) {
        auto frame = makeFrame();
        auto boostedTree = maths::analytics::CBoostedTreeFactory::constructFromString(checkpoint)
                               .restoreFor(*frame, cols - 1);
        std::ostringstream actualState;
        {
            // We need to call the inserter destructor to finish writing the state.
            core::CJsonStatePersistInserter inserter(actualState);
            boostedTree->acceptPersistInserter(inserter);
            actualState.flush();
        }
        BOOST_REQUIRE_EQUAL(expectedState.str(), actualState.str());
    }
}

BOOST_AUTO_TEST_CASE(testRestoreErrorHandling) {

    auto errorHandler = [](std::string error) {
        throw std::runtime_error{error};
    };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    auto stream = boost::make_shared<std::ostringstream>();

    // Redirect logging to a string stream so we can check errors generated.
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
        auto boostedTree = maths::analytics::CBoostedTreeFactory::constructFromString(
                               errorInBayesianOptimisationState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        std::cout << "got = " << e.what() << std::endl;
        std::cout << "Logging = " << stream->str() << std::endl;
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
        auto boostedTree = maths::analytics::CBoostedTreeFactory::constructFromString(errorInBoostedTreeImplState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        std::cout << "got = " << e.what() << std::endl;
        std::cout << "Logging = " << stream->str() << std::endl;
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
        auto boostedTree = maths::analytics::CBoostedTreeFactory::constructFromString(errorInBoostedTreeImplState)
                               .restoreFor(*frame, 2);
    } catch (const std::exception& e) {
        std::cout << "got = " << e.what() << std::endl;
        std::cout << "Logging = " << stream->str() << std::endl;
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

BOOST_AUTO_TEST_CASE(testWorstCaseMemoryCorrection) {
    // test for 15mb
    BOOST_REQUIRE_CLOSE(static_cast<double>(maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(
                            15.0 * BYTES_IN_MB)) /
                            BYTES_IN_MB,
                        15.0, 1.0);
    // test for 50mb
    BOOST_REQUIRE_CLOSE(static_cast<double>(maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(
                            50.0 * BYTES_IN_MB)) /
                            BYTES_IN_MB,
                        24.76, 1.0);
    // test for 300mb
    BOOST_REQUIRE_CLOSE(static_cast<double>(maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(
                            300.0 * BYTES_IN_MB)) /
                            BYTES_IN_MB,
                        64.4, 1.0);
    // test for 1024mb
    BOOST_REQUIRE_CLOSE(static_cast<double>(maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(
                            1024.0 * BYTES_IN_MB)) /
                            BYTES_IN_MB,
                        179.2, 1.0);
    // test for 18000mb
    BOOST_REQUIRE_CLOSE(static_cast<double>(maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(
                            18000.0 * BYTES_IN_MB)) /
                            BYTES_IN_MB,
                        1355.75, 1.0);

    // test for monotonicity
    std::size_t numberSamples{1000};
    TDoubleVec lhs;
    lhs.reserve(numberSamples);
    TDoubleVec rhs;
    rhs.reserve(numberSamples);
    test::CRandomNumbers rng;
    rng.generateUniformSamples(0.0, 20000.0, numberSamples, lhs);
    rng.generateUniformSamples(0.0, 20000.0, numberSamples, rhs);
    for (std::size_t i = 0; i < numberSamples; ++i) {
        BOOST_TEST_REQUIRE(
            (lhs[i] <= rhs[i]) ==
            (maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(lhs[i]) <=
             maths::analytics::CBoostedTreeImpl::correctedMemoryUsage(rhs[i])));
    }
}

BOOST_AUTO_TEST_CASE(testEncodingSeparately) {
    // Test that we can do encoding separately from training.

    TDoubleVec categoryValue{-15.0, 20.0, 0.0};

    auto target = [&](const TDoubleVecVec& features, std::size_t row) {
        return categoryValue[static_cast<std::size_t>(std::min(features[0][row], 2.0))] +
               2.8 * features[1][row] - 5.3 * features[2][row];
    };

    test::CRandomNumbers rng;
    std::size_t rows{300};
    std::size_t cols{4};
    double numberCategories{5.0};
    std::uint64_t seed{1000};

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

    // Encode and persist
    std::stringstream persistState;
    {
        auto boostedTree =
            maths::analytics::CBoostedTreeFactory::constructFromParameters(
                1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                .seed(seed)
                .buildForEncode(*frame, cols - 1);
        core::CJsonStatePersistInserter inserter(persistState);
        boostedTree->acceptPersistInserter(inserter);
        persistState.flush();
    }
    // Restore from encoded state.
    auto encodedFirstModel =
        maths::analytics::CBoostedTreeFactory::constructFromString(persistState)
            .restoreFor(*frame, cols - 1);
    encodedFirstModel->train();
    encodedFirstModel->predict();

    TDoubleVec encodedFirstPredictions;
    frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            encodedFirstPredictions.push_back(encodedFirstModel->prediction(*row)[0]);
        }
    });

    // Resize data frame back to the original column number
    frame->resizeColumns(1, cols);
    auto trainedFromScratchModel =
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CMse>())
            .seed(seed)
            .buildForTrain(*frame, cols - 1);
    trainedFromScratchModel->train();
    trainedFromScratchModel->predict();

    TDoubleVec trainedFromScratchPredictions;
    frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            trainedFromScratchPredictions.push_back(
                trainedFromScratchModel->prediction(*row)[0]);
        }
    });

    BOOST_REQUIRE_EQUAL(encodedFirstPredictions.size(),
                        trainedFromScratchPredictions.size());
    for (std::size_t i = 0; i < encodedFirstPredictions.size(); ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(encodedFirstPredictions[i],
                                     trainedFromScratchPredictions[i], 0.01);
    }
}

BOOST_AUTO_TEST_CASE(testCoarseParameterTuningEarlyStopping) {

    // Check that we can effectively train using loss on a specified holdout set.

    test::CRandomNumbers rng;
    std::size_t rows{500};
    std::size_t cols{3};

    std::size_t numberHoldoutRows{200};

    auto verify = [&](double noiseVariance) {
        auto target = [&] {
            TDoubleVec m;
            TDoubleVec s;
            rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
            rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
            return [=](const TRowRef& row) {
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

        auto frame = core::makeMainStorageDataFrame(cols, rows).first;
        fillDataFrame(rows, 0, cols, x, noise, target, *frame);

        auto factory = maths::analytics::CBoostedTreeFactory::constructFromParameters(
            1, std::make_unique<maths::analytics::boosted_tree::CMse>());
        factory.numberHoldoutRows(numberHoldoutRows);
        auto regression = factory.buildForTrain(*frame, cols - 1);
        BOOST_REQUIRE(factory.hyperparametersLosses()->empty() == false);
        bool stopEarly{regression->hyperparameters().stopEarly()};
        return stopEarly;
    };
    BOOST_REQUIRE_EQUAL(verify(0.1), false);
    BOOST_REQUIRE_EQUAL(verify(1000.0), true);
}

BOOST_AUTO_TEST_SUITE_END()
