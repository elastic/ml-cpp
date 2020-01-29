/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CProgramCounters.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CTools.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/ElasticsearchStateIndex.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <memory>

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVec::iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TStrVec::iterator)

BOOST_AUTO_TEST_SUITE(CDataFrameAnalyzerTrainingTest)

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TSizeVec = std::vector<std::size_t>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TDataAdderUPtr = std::unique_ptr<core::CDataAdder>;
using TPersisterSupplier = std::function<TDataAdderUPtr()>;
using TDataSearcherUPtr = std::unique_ptr<core::CDataSearcher>;
using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;

class CTestDataSearcher : public core::CDataSearcher {
public:
    CTestDataSearcher(const std::string& data)
        : m_Stream(new std::istringstream(data)) {}

    virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
        std::istringstream* intermediateStateStream{
            static_cast<std::istringstream*>(m_Stream.get())};
        // Discard first line, which contains the state id.
        intermediateStateStream->ignore(256, '\n');
        std::string intermediateState;
        std::getline(*intermediateStateStream, intermediateState);
        return std::make_shared<std::istringstream>(intermediateState);
    }

private:
    TIStreamP m_Stream;
};

TStrVec splitOnNull(std::stringstream&& tokenStream) {
    TStrVec results;
    std::string token;
    while (std::getline(tokenStream, token, '\0')) {
        results.push_back(token);
    }
    return results;
}

rapidjson::Document treeToJsonDocument(const maths::CBoostedTree& tree) {
    std::stringstream persistStream;
    {
        core::CJsonStatePersistInserter inserter(persistStream);
        tree.acceptPersistInserter(inserter);
        persistStream.flush();
    }
    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(persistStream.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    return results;
}

auto restoreTree(std::string persistedState, TDataFrameUPtr& frame, std::size_t dependentVariable) {
    CTestDataSearcher dataSearcher(persistedState);
    auto decompressor = std::make_unique<core::CStateDecompressor>(dataSearcher);
    decompressor->setStateRestoreSearch(api::ML_STATE_INDEX,
                                        api::getStateId("testJob", "regression"));
    auto stream = decompressor->search(1, 1);
    return maths::CBoostedTreeFactory::constructFromString(*stream).restoreFor(
        *frame, dependentVariable);
}

TDataFrameUPtr setupLinearRegressionData(const TStrVec& fieldNames,
                                         TStrVec& fieldValues,
                                         api::CDataFrameAnalyzer& analyzer,
                                         const TDoubleVec& weights,
                                         const TDoubleVec& regressors,
                                         TStrVec& targets) {

    auto target = [&weights](const TDoubleVec& regressors_) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors_[i];
        }
        return core::CStringUtils::typeToStringPrecise(result, core::CIEEE754::E_DoublePrecision);
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;

    for (std::size_t i = 0; i < regressors.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = regressors[i + j];
        }

        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);
        targets.push_back(fieldValues[weights.size()]);

        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }

    frame->finishWritingRows();

    return frame;
}

TDataFrameUPtr setupBinaryClassificationData(const TStrVec& fieldNames,
                                             TStrVec& fieldValues,
                                             api::CDataFrameAnalyzer& analyzer,
                                             const TDoubleVec& weights,
                                             const TDoubleVec& regressors,
                                             TStrVec& targets) {
    TStrVec classes{"foo", "bar"};
    auto target = [&weights, &classes](const TDoubleVec& regressors_) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors_[i];
        }
        return classes[result < 0.0 ? 0 : 1];
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;
    TBoolVec categoricalFields(weights.size(), false);
    categoricalFields.push_back(true);
    frame->categoricalColumns(std::move(categoricalFields));

    for (std::size_t i = 0; i < regressors.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = regressors[i + j];
        }

        for (std::size_t j = 0; j < row.size() - 1; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);
        targets.push_back(fieldValues[weights.size()]);

        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }

    frame->finishWritingRows();

    return frame;
}

enum EPredictionType { E_Regression, E_BinaryClassification };

void appendPrediction(core::CDataFrame&, std::size_t, double prediction, double, TDoubleVec& predictions) {
    predictions.push_back(prediction);
}

void appendPrediction(core::CDataFrame& frame,
                      std::size_t columnHoldingPrediction,
                      double logOddsClass1,
                      double threshold,
                      TStrVec& predictions) {
    predictions.push_back(
        maths::CTools::logisticFunction(logOddsClass1) < threshold
            ? frame.categoricalColumnValues()[columnHoldingPrediction][0]
            : frame.categoricalColumnValues()[columnHoldingPrediction][1]);
}

template<typename T>
void addPredictionTestData(EPredictionType type,
                           const TStrVec& fieldNames,
                           TStrVec fieldValues,
                           api::CDataFrameAnalyzer& analyzer,
                           std::vector<T>& expectedPredictions,
                           std::size_t numberExamples = 100,
                           double alpha = -1.0,
                           double lambda = -1.0,
                           double gamma = -1.0,
                           double softTreeDepthLimit = -1.0,
                           double softTreeDepthTolerance = -1.0,
                           double eta = 0.0,
                           std::size_t maximumNumberTrees = 0,
                           double featureBagFraction = 0.0) {

    test::CRandomNumbers rng;

    TDoubleVec weights;
    rng.generateUniformSamples(-1.0, 1.0, fieldNames.size() - 3, weights);
    TDoubleVec regressors;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);

    TStrVec targets;
    auto frame = type == E_Regression
                     ? setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                                 weights, regressors, targets)
                     : setupBinaryClassificationData(fieldNames, fieldValues, analyzer,
                                                     weights, regressors, targets);

    maths::CBoostedTreeFactory treeFactory{
        maths::CBoostedTreeFactory::constructFromParameters(1)};
    if (alpha >= 0.0) {
        treeFactory.depthPenaltyMultiplier(alpha);
    }
    if (lambda >= 0.0) {
        treeFactory.leafWeightPenaltyMultiplier(lambda);
    }
    if (gamma >= 0.0) {
        treeFactory.treeSizePenaltyMultiplier(gamma);
    }
    if (softTreeDepthLimit >= 0.0) {
        treeFactory.softTreeDepthLimit(softTreeDepthLimit);
    }
    if (softTreeDepthTolerance >= 0.0) {
        treeFactory.softTreeDepthTolerance(softTreeDepthTolerance);
    }
    if (eta > 0.0) {
        treeFactory.eta(eta);
    }
    if (maximumNumberTrees > 0) {
        treeFactory.maximumNumberTrees(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0) {
        treeFactory.featureBagFraction(featureBagFraction);
    }

    ml::api::CDataFrameTrainBoostedTreeInstrumentation instrumentation;
    treeFactory.analysisInstrumentation(&instrumentation);

    std::unique_ptr<maths::boosted_tree::CLoss> loss;
    if (type == E_Regression) {
        loss = std::make_unique<maths::boosted_tree::CMse>();
    } else {
        loss = std::make_unique<maths::boosted_tree::CLogistic>();
    }
    auto tree = treeFactory.buildFor(*frame, std::move(loss), weights.size());

    tree->train();
    tree->predict();

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double prediction{(*row)[tree->columnHoldingPrediction()]};
            appendPrediction(*frame, weights.size(), prediction,
                             tree->probabilityAtWhichToAssignClassOne(), expectedPredictions);
        }
    });
}

template<typename F>
void testOneRunOfBoostedTreeTrainingWithStateRecovery(F makeSpec, std::size_t iterationToRestartFrom) {

    std::stringstream outputStream;
    auto outputWriterFactory = [&outputStream]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
    };

    std::size_t numberExamples{200};
    TStrVec fieldNames{"c1", "c2", "c3", "c4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};
    TDoubleVec regressors;
    test::CRandomNumbers rng;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);

    auto persistenceStream = std::make_shared<std::ostringstream>();
    TPersisterSupplier persisterSupplier = [&persistenceStream]() -> TDataAdderUPtr {
        return std::make_unique<api::CSingleStreamDataAdder>(persistenceStream);
    };

    // Compute expected tree.

    api::CDataFrameAnalyzer analyzer{
        makeSpec("target", numberExamples, persisterSupplier), outputWriterFactory};
    std::size_t dependentVariable(std::find(fieldNames.begin(), fieldNames.end(), "target") -
                                  fieldNames.begin());

    TStrVec targets;
    auto frame = setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                           weights, regressors, targets);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    TStrVec persistedStates{
        splitOnNull(std::stringstream{std::move(persistenceStream->str())})};
    auto expectedTree = restoreTree(std::move(persistedStates.back()), frame, dependentVariable);

    // Compute actual tree.

    persistenceStream->str("");

    std::istringstream intermediateStateStream{persistedStates[iterationToRestartFrom]};
    TRestoreSearcherSupplier restoreSearcherSupplier = [&intermediateStateStream]() -> TDataSearcherUPtr {
        return std::make_unique<CTestDataSearcher>(intermediateStateStream.str());
    };

    api::CDataFrameAnalyzer restoredAnalyzer{
        makeSpec("target", numberExamples, persisterSupplier), outputWriterFactory};

    targets.clear();
    setupLinearRegressionData(fieldNames, fieldValues, restoredAnalyzer,
                              weights, regressors, targets);
    restoredAnalyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    persistedStates = splitOnNull(std::stringstream{std::move(persistenceStream->str())});
    auto actualTree = restoreTree(std::move(persistedStates.back()), frame, dependentVariable);

    // Compare hyperparameters.

    rapidjson::Document expectedResults{treeToJsonDocument(*expectedTree)};
    const auto& expectedHyperparameters =
        expectedResults[maths::CBoostedTree::bestHyperparametersName()];
    const auto& expectedRegularizationHyperparameters =
        expectedHyperparameters[maths::CBoostedTree::bestRegularizationHyperparametersName()];

    rapidjson::Document actualResults{treeToJsonDocument(*actualTree)};
    const auto& actualHyperparameters =
        actualResults[maths::CBoostedTree::bestHyperparametersName()];
    const auto& actualRegularizationHyperparameters =
        actualHyperparameters[maths::CBoostedTree::bestRegularizationHyperparametersName()];

    for (const auto& key : maths::CBoostedTree::bestHyperparameterNames()) {
        if (expectedHyperparameters.HasMember(key)) {
            double expected{std::stod(expectedHyperparameters[key].GetString())};
            double actual{std::stod(actualHyperparameters[key].GetString())};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-4 * expected);
        } else if (expectedRegularizationHyperparameters.HasMember(key)) {
            double expected{std::stod(expectedRegularizationHyperparameters[key].GetString())};
            double actual{std::stod(actualRegularizationHyperparameters[key].GetString())};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-4 * expected);
        } else {
            BOOST_FAIL("Missing " + key);
        }
    }
}
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionTraining) {

    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
        outputWriterFactory};
    addPredictionTestData(E_Regression, fieldNames, fieldValues, analyzer, expectedPredictions);

    core::CStopWatch watch{true};
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
    std::uint64_t duration{watch.stop()};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedPrediction,
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble(),
                1e-4 * std::fabs(*expectedPrediction));
            ++expectedPrediction;
            BOOST_TEST_REQUIRE(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() >= 0);
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() <= 100);
            BOOST_TEST_REQUIRE(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    BOOST_TEST_REQUIRE(expectedPrediction == expectedPredictions.end());
    BOOST_TEST_REQUIRE(progressCompleted);

    LOG_DEBUG(<< "estimated memory usage = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
    LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
              << "ms");

    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(
                           counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 6000000);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1400000);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionTrainingStateReport) {

    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec("regression", "c5"),
        outputWriterFactory};
    addPredictionTestData(E_Regression, fieldNames, fieldValues, analyzer, expectedPredictions);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionTrainingWithParams) {

    // Test the regression hyperparameter settings are correctly propagated to the
    // analysis runner.

    double alpha{2.0};
    double lambda{1.0};
    double gamma{10.0};
    double softTreeDepthLimit{3.0};
    double softTreeDepthTolerance{0.1};
    double eta{0.9};
    std::size_t maximumNumberTrees{1};
    double featureBagFraction{0.3};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target",
            100, 5, 4000000, 0, 0, {}, alpha, lambda, gamma, softTreeDepthLimit,
            softTreeDepthTolerance, eta, maximumNumberTrees, featureBagFraction),
        outputWriterFactory};

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addPredictionTestData(E_Regression, fieldNames, fieldValues, analyzer,
                          expectedPredictions, 100, alpha, lambda, gamma,
                          softTreeDepthLimit, softTreeDepthTolerance, eta,
                          maximumNumberTrees, featureBagFraction);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    // Check best hyperparameters
    const auto* runner{dynamic_cast<const api::CDataFrameTrainBoostedTreeRegressionRunner*>(
        analyzer.runner())};
    const auto& boostedTree{runner->boostedTree()};
    const auto& bestHyperparameters{boostedTree.bestHyperparameters()};
    BOOST_TEST_REQUIRE(bestHyperparameters.eta() == eta);
    BOOST_TEST_REQUIRE(bestHyperparameters.featureBagFraction() == featureBagFraction);
    // TODO extend the test to add the checks for downsampleFactor and etaGrowthRatePerTree
    //    BOOST_TEST_REQUIRE(bestHyperparameters.downsampleFactor() == downsampleFactor);
    //    BOOST_TEST_REQUIRE(bestHyperparameters.etaGrowthRatePerTree() == etaGrowthRatePerTree);
    BOOST_TEST_REQUIRE(bestHyperparameters.regularization().depthPenaltyMultiplier() == alpha);
    BOOST_TEST_REQUIRE(
        bestHyperparameters.regularization().leafWeightPenaltyMultiplier() == lambda);
    BOOST_TEST_REQUIRE(bestHyperparameters.regularization().treeSizePenaltyMultiplier() == gamma);
    BOOST_TEST_REQUIRE(bestHyperparameters.regularization().softTreeDepthLimit() ==
                       softTreeDepthLimit);
    BOOST_TEST_REQUIRE(bestHyperparameters.regularization().softTreeDepthTolerance() ==
                       softTreeDepthTolerance);

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedPrediction,
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble(),
                1e-4 * std::fabs(*expectedPrediction));
            ++expectedPrediction;
            BOOST_TEST_REQUIRE(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() >= 0);
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() <= 100);
            BOOST_TEST_REQUIRE(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    BOOST_TEST_REQUIRE(expectedPrediction == expectedPredictions.end());
    BOOST_TEST_REQUIRE(progressCompleted);
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue) {

    // Test we are able to predict value rows for which the dependent variable
    // is missing.

    test::CRandomNumbers rng;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    auto target = [](double feature) { return 10.0 * feature; };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target", 50, 2, 4000000),
        outputWriterFactory};

    TDoubleVec feature;
    rng.generateUniformSamples(1.0, 3.0, 50, feature);

    TStrVec fieldNames{"feature", "target", ".", "."};
    TStrVec fieldValues(4);

    for (std::size_t i = 0; i < 40; ++i) {
        fieldValues[0] = std::to_string(feature[i]);
        fieldValues[1] = std::to_string(target(feature[i]));
        fieldValues[2] = std::to_string(i);
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    for (std::size_t i = 40; i < 50; ++i) {
        fieldValues[0] = std::to_string(feature[i]);
        fieldValues[1] = "";
        fieldValues[2] = std::to_string(i);
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    analyzer.handleRecord(fieldNames, {"", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::size_t numberResults{0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            std::size_t index(result["row_results"]["checksum"].GetUint64());
            double expected{target(feature[index])};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expected,
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble(),
                0.15 * expected);
            BOOST_REQUIRE_EQUAL(
                index < 40,
                result["row_results"]["results"]["ml"]["is_training"].GetBool());
            ++numberResults;
        }
    }
    BOOST_REQUIRE_EQUAL(std::size_t{50}, numberResults);
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionTrainingWithStateRecovery) {

    struct SHyperparameters {
        SHyperparameters(double alpha = 2.0, double lambda = 1.0, double gamma = 10.0)
            : s_Alpha{alpha}, s_Lambda{lambda}, s_Gamma{gamma} {}

        std::size_t numberUnset() const {
            return (s_Alpha < 0.0 ? 1 : 0) + (s_Lambda < 0.0 ? 1 : 0) +
                   (s_Gamma < 0.0 ? 1 : 0);
        }

        double s_Alpha;
        double s_Lambda;
        double s_Gamma;
        double s_SoftTreeDepthLimit = 3.0;
        double s_SoftTreeDepthTolerance = 0.15;
        double s_Eta = 0.9;
        std::size_t s_MaximumNumberTrees = 2;
        double s_FeatureBagFraction = 0.3;
    };

    std::size_t numberRoundsPerHyperparameter{3};

    TSizeVec intermediateIterations{0, 0, 0};
    std::size_t finalIteration{0};

    test::CRandomNumbers rng;

    for (const auto& params :
         {SHyperparameters{}, SHyperparameters{-1.0},
          SHyperparameters{-1.0, -1.0}, SHyperparameters{-1.0, -1.0, -1.0}}) {

        LOG_DEBUG(<< "Number parameters to search = " << params.numberUnset());

        auto makeSpec = [&](const std::string& dependentVariable, std::size_t numberExamples,
                            TPersisterSupplier persisterSupplier) {
            return test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                test::CDataFrameAnalysisSpecificationFactory::regression(), dependentVariable,
                numberExamples, 5, 15000000, numberRoundsPerHyperparameter, 12,
                {}, params.s_Alpha, params.s_Lambda, params.s_Gamma,
                params.s_SoftTreeDepthLimit, params.s_SoftTreeDepthTolerance,
                params.s_Eta, params.s_MaximumNumberTrees, params.s_FeatureBagFraction,
                0 /*numTopFeatureImportanceValues*/, &persisterSupplier);
        };

        finalIteration = params.numberUnset() * numberRoundsPerHyperparameter;
        if (finalIteration > 2) {
            rng.generateUniformSamples(0, finalIteration - 2, 3, intermediateIterations);
        }

        for (auto intermediateIteration : intermediateIterations) {
            LOG_DEBUG(<< "restart from " << intermediateIteration);
            testOneRunOfBoostedTreeTrainingWithStateRecovery(makeSpec, intermediateIteration);
        }
    }
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeClassifierTraining) {

    // Test the results the analyzer produces match running classification directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::classification(),
            "target", 100, 5, 6000000, 0, 0, {"target"}),
        outputWriterFactory};
    addPredictionTestData(E_BinaryClassification, fieldNames, fieldValues,
                          analyzer, expectedPredictions);

    core::CStopWatch watch{true};
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
    std::uint64_t duration{watch.stop()};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
            std::string actualPrediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetString()};
            BOOST_REQUIRE_EQUAL(*expectedPrediction, actualPrediction);
            // Check the prediction values match the first entry in the top-classes.
            BOOST_REQUIRE_EQUAL(
                result["row_results"]["results"]["ml"]["target_prediction"].GetString(),
                result["row_results"]["results"]["ml"]["top_classes"][0]["class_name"]
                    .GetString());
            BOOST_REQUIRE_EQUAL(
                result["row_results"]["results"]["ml"]["prediction_probability"].GetDouble(),
                result["row_results"]["results"]["ml"]["top_classes"][0]["class_probability"]
                    .GetDouble());
            BOOST_REQUIRE_EQUAL(
                result["row_results"]["results"]["ml"]["prediction_score"].GetDouble(),
                result["row_results"]["results"]["ml"]["top_classes"][0]["class_score"]
                    .GetDouble());
            ++expectedPrediction;
            BOOST_TEST_REQUIRE(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() >= 0);
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() <= 100);
            BOOST_TEST_REQUIRE(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    BOOST_TEST_REQUIRE(expectedPrediction == expectedPredictions.end());
    BOOST_TEST_REQUIRE(progressCompleted);

    LOG_DEBUG(<< "estimated memory usage = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
    LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
              << "ms");
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(
                           counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 6000000);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1400000);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeClassifierImbalanced) {

    // Test we get high average recall for each class when the training data are
    // imbalanced.

    using TStrSizeUMap = boost::unordered_map<std::string, std::size_t>;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    std::size_t numberExamples{500};
    TStrVec fieldNames{"f1", "f2", "f3", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "0", ""};
    test::CRandomNumbers rng;
    TDoubleVec weights{1.0, 1.0, 1.0};
    TDoubleVec regressors;
    rng.generateUniformSamples(-5.0, 10.0, numberExamples * weights.size(), regressors);

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::classification(),
            "target", numberExamples, 4, 14000000, 0, 0, {"target"}),
        outputWriterFactory};

    TStrVec actuals;
    setupBinaryClassificationData(fieldNames, fieldValues, analyzer, weights,
                                  regressors, actuals);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    TStrSizeUMap correct;
    TStrSizeUMap counts;

    auto actual = actuals.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(actual != actuals.end());
            std::string prediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetString()};

            if (*actual == prediction) {
                ++correct[*actual];
            }
            ++counts[*actual];
            ++actual;
        }
    }

    for (const auto& label : {"foo", "bar"}) {
        double recall{static_cast<double>(correct[label]) /
                      static_cast<double>(counts[label])};
        BOOST_TEST_REQUIRE(recall > 0.84);
    }
}

BOOST_AUTO_TEST_CASE(testCategoricalFields) {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    {
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                test::CDataFrameAnalysisSpecificationFactory::regression(),
                "x5", 1000, 5, 19000000, 0, 0, {"x1", "x2"}),
            outputWriterFactory};

        TStrVec x[]{{"x11", "x12", "x13", "x14", "x15"},
                    {"x21", "x22", "x23", "x24", "x25", "x26", "x27"}};

        for (std::size_t i = 0; i < 10; ++i) {
            analyzer.handleRecord({"x1", "x2", "x3", "x4", "x5", ".", "."},
                                  {x[0][i % x[0].size()], x[1][i % x[1].size()],
                                   std::to_string(i), std::to_string(i),
                                   std::to_string(i), std::to_string(i), ""});
        }
        analyzer.receivedAllRows();

        bool passed{true};

        const core::CDataFrame& frame{analyzer.dataFrame()};
        frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            std::size_t i{0};
            for (auto row = beginRows; row != endRows; ++row, ++i) {
                core::CFloatStorage expected[]{static_cast<double>(i % x[0].size()),
                                               static_cast<double>(i % x[1].size())};
                bool wasPassed{passed};
                passed &= (expected[0] == (*row)[0]);
                passed &= (expected[1] == (*row)[1]);
                if (wasPassed && passed == false) {
                    LOG_ERROR(<< "expected " << core::CContainerPrinter::print(expected)
                              << ", got [" << (*row)[0] << ", " << (*row)[1] << "]");
                }
            }
        });

        BOOST_TEST_REQUIRE(passed);
    }

    LOG_DEBUG(<< "Test overflow");
    {
        std::size_t rows{core::CDataFrame::MAX_CATEGORICAL_CARDINALITY + 3};

        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                test::CDataFrameAnalysisSpecificationFactory::regression(),
                "x5", rows, 5, 8000000000, 0, 0, {"x1"}),
            outputWriterFactory};

        TStrVec fieldNames{"x1", "x2", "x3", "x4", "x5", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "", ""};
        for (std::size_t i = 0; i < rows; ++i) {
            std::fill_n(fieldValues.begin(), 6, std::to_string(i));
            analyzer.handleRecord(fieldNames, fieldValues);
        }
        analyzer.receivedAllRows();

        bool passed{true};
        std::size_t i{0};

        const core::CDataFrame& frame{analyzer.dataFrame()};
        frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row, ++i) {
                core::CFloatStorage expected{
                    i < core::CDataFrame::MAX_CATEGORICAL_CARDINALITY
                        ? static_cast<double>(i)
                        : static_cast<double>(core::CDataFrame::MAX_CATEGORICAL_CARDINALITY)};
                bool wasPassed{passed};
                passed &= (expected == (*row)[0]);
                if (wasPassed && passed == false) {
                    LOG_ERROR(<< "expected " << expected << ", got " << (*row)[0]);
                }
            }
        });

        BOOST_TEST_REQUIRE(passed);
    }
}

BOOST_AUTO_TEST_CASE(testCategoricalFieldsEmptyAsMissing) {

    auto eq = [](double expected) {
        return [expected](double actual) { return expected == actual; };
    };

    auto missing = []() {
        return [](double actual) {
            return maths::CDataFrameUtils::isMissing(actual);
        };
    };

    auto assertRow = [&](const std::size_t row_i,
                         const std::vector<std::function<bool(double)>>& matchers,
                         const TRowRef& row) {
        BOOST_REQUIRE_MESSAGE(matchers.size() == row.numberColumns(),
                              "row " + std::to_string(row_i));
        for (std::size_t i = 0; i < row.numberColumns(); ++i) {
            BOOST_REQUIRE_MESSAGE(matchers[i](row[i]), "row " + std::to_string(row_i) + ", column " +
                                                           std::to_string(i));
        }
    };

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::classification(),
            "x5", 1000, 5, 19000000, 0, 0, {"x1", "x2", "x5"}),
        outputWriterFactory};

    TStrVec fieldNames{"x1", "x2", "x3", "x4", "x5", ".", "."};
    analyzer.handleRecord(fieldNames, {"x11", "x21", "0", "0", "x51", "0", ""});
    analyzer.handleRecord(fieldNames, {"x12", "x22", "1", "1", "x52", "1", ""});
    analyzer.handleRecord(fieldNames, {"", "x23", "2", "2", "x51", "2", ""});
    analyzer.handleRecord(fieldNames, {"x14", "x24", "3", "3", "", "3", ""});
    analyzer.handleRecord(fieldNames, {"x15", "x25", "4", "4", "x51", "4", ""});
    analyzer.handleRecord(fieldNames, {"x11", "x26", "5", "5", "x52", "5", ""});
    analyzer.handleRecord(fieldNames, {"x12", "", "6", "6", "", "6", ""});
    analyzer.handleRecord(fieldNames, {"x13", "x21", "7", "7", "", "7", ""});
    analyzer.handleRecord(fieldNames, {"x14", "x22", "8", "8", "x51", "8", ""});
    analyzer.handleRecord(fieldNames, {"", "x23", "9", "9", "x52", "9", ""});
    analyzer.receivedAllRows();

    const core::CDataFrame& frame{analyzer.dataFrame()};
    frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        std::vector<TRowRef> rows;
        std::copy(beginRows, endRows, std::back_inserter(rows));
        BOOST_REQUIRE_EQUAL(std::size_t{10}, rows.size());
        assertRow(0, {eq(0.0), eq(0.0), eq(0.0), eq(0.0), eq(0.0)}, rows[0]);
        assertRow(1, {eq(1.0), eq(1.0), eq(1.0), eq(1.0), eq(1.0)}, rows[1]);
        assertRow(2, {eq(2.0), eq(2.0), eq(2.0), eq(2.0), eq(0.0)}, rows[2]);
        assertRow(3, {eq(3.0), eq(3.0), eq(3.0), eq(3.0), missing()}, rows[3]);
        assertRow(4, {eq(4.0), eq(4.0), eq(4.0), eq(4.0), eq(0.0)}, rows[4]);
        assertRow(5, {eq(0.0), eq(5.0), eq(5.0), eq(5.0), eq(1.0)}, rows[5]);
        assertRow(6, {eq(1.0), eq(6.0), eq(6.0), eq(6.0), missing()}, rows[6]);
        assertRow(7, {eq(5.0), eq(0.0), eq(7.0), eq(7.0), missing()}, rows[7]);
        assertRow(8, {eq(3.0), eq(1.0), eq(8.0), eq(8.0), eq(0.0)}, rows[8]);
        assertRow(9, {eq(2.0), eq(2.0), eq(9.0), eq(9.0), eq(1.0)}, rows[9]);
    });
}

BOOST_AUTO_TEST_SUITE_END()
