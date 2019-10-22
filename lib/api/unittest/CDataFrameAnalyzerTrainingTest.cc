/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalyzerTrainingTest.h"

#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CProgramCounters.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CTools.h>

#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/ElasticsearchStateIndex.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <memory>

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TPoint = maths::CDenseVector<maths::CFloatStorage>;
using TPointVec = std::vector<TPoint>;
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
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);
    return results;
}

auto restoreTree(std::string persistedState, TDataFrameUPtr& frame, std::size_t dependentVariable) {
    CTestDataSearcher dataSearcher(persistedState);
    auto decompressor = std::make_unique<core::CStateDecompressor>(dataSearcher);
    decompressor->setStateRestoreSearch(api::ML_STATE_INDEX,
                                        api::getRegressionStateId("testJob"));
    auto stream = decompressor->search(1, 1);
    return maths::CBoostedTreeFactory::constructFromString(*stream).restoreFor(
        *frame, dependentVariable);
}

TDataFrameUPtr setupLinearRegressionData(const TStrVec& fieldNames,
                                         TStrVec& fieldValues,
                                         api::CDataFrameAnalyzer& analyzer,
                                         const TDoubleVec& weights,
                                         const TDoubleVec& values) {

    auto target = [&weights](const TDoubleVec& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors[i];
        }
        return core::CStringUtils::typeToStringPrecise(result, core::CIEEE754::E_DoublePrecision);
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);

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
                                             const TDoubleVec& values) {
    TStrVec classes{"foo", "bar"};
    auto target = [&weights, &classes](const TDoubleVec& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors[i];
        }
        return classes[result < 0.0 ? 0 : 1];
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;
    TBoolVec categoricalFields(weights.size(), false);
    categoricalFields.push_back(true);
    frame->categoricalColumns(std::move(categoricalFields));

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        for (std::size_t j = 0; j < row.size() - 1; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);

        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }

    frame->finishWritingRows();

    return frame;
}

enum EPredictionType { E_Regression, E_BinaryClassification };

void appendPrediction(core::CDataFrame&, std::size_t, double prediction, TDoubleVec& predictions) {
    predictions.push_back(prediction);
}

void appendPrediction(core::CDataFrame& frame,
                      std::size_t columnHoldingPrediction,
                      double logOddsClass1,
                      TStrVec& predictions) {
    predictions.push_back(
        maths::CTools::logisticFunction(logOddsClass1) < 0.5
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
    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};

    TDoubleVec values;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, values);

    auto frame = type == E_Regression
                     ? setupLinearRegressionData(fieldNames, fieldValues,
                                                 analyzer, weights, values)
                     : setupBinaryClassificationData(fieldNames, fieldValues,
                                                     analyzer, weights, values);

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

    std::unique_ptr<maths::boosted_tree::CLoss> loss;
    if (type == E_Regression) {
        loss = std::make_unique<maths::boosted_tree::CMse>();
    } else {
        loss = std::make_unique<maths::boosted_tree::CLogistic>();
    }
    auto tree = treeFactory.buildFor(*frame, std::move(loss), weights.size());

    tree->train();

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double prediction{(*row)[tree->columnHoldingPrediction(row->numberColumns())]};
            appendPrediction(*frame, weights.size(), prediction, expectedPredictions);
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
    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};
    TDoubleVec values;
    test::CRandomNumbers rng;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, values);

    auto persistenceStream = std::make_shared<std::ostringstream>();
    TPersisterSupplier persisterSupplier = [&persistenceStream]() -> TDataAdderUPtr {
        return std::make_unique<api::CSingleStreamDataAdder>(persistenceStream);
    };

    // Compute expected tree.

    api::CDataFrameAnalyzer analyzer{
        makeSpec("c5", numberExamples, persisterSupplier), outputWriterFactory};
    std::size_t dependentVariable(
        std::find(fieldNames.begin(), fieldNames.end(), "c5") - fieldNames.begin());

    auto frame = setupLinearRegressionData(fieldNames, fieldValues, analyzer, weights, values);
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
        makeSpec("c5", numberExamples, persisterSupplier), outputWriterFactory};

    setupLinearRegressionData(fieldNames, fieldValues, restoredAnalyzer, weights, values);
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
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-4 * expected);
        } else if (expectedRegularizationHyperparameters.HasMember(key)) {
            double expected{std::stod(expectedRegularizationHyperparameters[key].GetString())};
            double actual{std::stod(actualRegularizationHyperparameters[key].GetString())};
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-4 * expected);
        } else {
            CPPUNIT_FAIL("Missing " + key);
        }
    }
}
}

void CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTraining() {

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

    core::CStopWatch watch{true};
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
    std::uint64_t duration{watch.stop()};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedPrediction != expectedPredictions.end());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                *expectedPrediction,
                result["row_results"]["results"]["ml"]["c5_prediction"].GetDouble(),
                1e-4 * std::fabs(*expectedPrediction));
            ++expectedPrediction;
            CPPUNIT_ASSERT(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() >= 0);
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() <= 100);
            CPPUNIT_ASSERT(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    CPPUNIT_ASSERT(expectedPrediction == expectedPredictions.end());
    CPPUNIT_ASSERT(progressCompleted);

    LOG_DEBUG(<< "estimated memory usage = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
    LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
              << "ms");

    CPPUNIT_ASSERT(core::CProgramCounters::counter(
                       counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 2900000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1050000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

void CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithParams() {

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
            "regression", "c5", 100, 5, 3000000, 0, 0, {}, alpha, lambda, gamma, softTreeDepthLimit,
            softTreeDepthTolerance, eta, maximumNumberTrees, featureBagFraction),
        outputWriterFactory};

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addPredictionTestData(E_Regression, fieldNames, fieldValues, analyzer,
                          expectedPredictions, 100, alpha, lambda, gamma,
                          softTreeDepthLimit, softTreeDepthTolerance, eta,
                          maximumNumberTrees, featureBagFraction);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedPrediction != expectedPredictions.end());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                *expectedPrediction,
                result["row_results"]["results"]["ml"]["c5_prediction"].GetDouble(),
                1e-4 * std::fabs(*expectedPrediction));
            ++expectedPrediction;
            CPPUNIT_ASSERT(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() >= 0);
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() <= 100);
            CPPUNIT_ASSERT(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    CPPUNIT_ASSERT(expectedPrediction == expectedPredictions.end());
    CPPUNIT_ASSERT(progressCompleted);
}

void CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue() {

    // Test we are able to predict value rows for which the dependent variable
    // is missing.

    test::CRandomNumbers rng;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    auto target = [](double feature) { return 10.0 * feature; };

    api::CDataFrameAnalyzer analyzer{test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                                         "regression", "target", 50, 2, 2000000),
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
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    std::size_t numberResults{0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            std::size_t index(result["row_results"]["checksum"].GetUint64());
            double expected{target(feature[index])};
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                expected,
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble(),
                0.15 * expected);
            CPPUNIT_ASSERT_EQUAL(
                index < 40,
                result["row_results"]["results"]["ml"]["is_training"].GetBool());
            ++numberResults;
        }
    }
    CPPUNIT_ASSERT_EQUAL(std::size_t{50}, numberResults);
}

void CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithStateRecovery() {

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
                "regression", dependentVariable, numberExamples, 5, 15000000,
                numberRoundsPerHyperparameter, 12, {}, params.s_Alpha,
                params.s_Lambda, params.s_Gamma, params.s_SoftTreeDepthLimit,
                params.s_SoftTreeDepthTolerance, params.s_Eta, params.s_MaximumNumberTrees,
                params.s_FeatureBagFraction, &persisterSupplier);
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

void CDataFrameAnalyzerTrainingTest::testRunBoostedTreeClassifierTraining() {

    // Test the results the analyzer produces match running classification directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            "classification", "c5", 100, 5, 3000000, 0, 0, {"c5"}),
        outputWriterFactory};
    addPredictionTestData(E_BinaryClassification, fieldNames, fieldValues,
                          analyzer, expectedPredictions);

    core::CStopWatch watch{true};
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
    std::uint64_t duration{watch.stop()};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedPrediction != expectedPredictions.end());
            std::string actualPrediction{
                result["row_results"]["results"]["ml"]["c5_prediction"].GetString()};
            CPPUNIT_ASSERT_EQUAL(*expectedPrediction, actualPrediction);
            ++expectedPrediction;
            CPPUNIT_ASSERT(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() >= 0);
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() <= 100);
            CPPUNIT_ASSERT(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    CPPUNIT_ASSERT(expectedPrediction == expectedPredictions.end());
    CPPUNIT_ASSERT(progressCompleted);

    LOG_DEBUG(<< "estimated memory usage = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
    LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
              << "ms");
    CPPUNIT_ASSERT(core::CProgramCounters::counter(
                       counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 2900000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1050000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

CppUnit::Test* CDataFrameAnalyzerTrainingTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameAnalyzerTrainingTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTrainingTest>(
        "CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTraining",
        &CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTraining));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTrainingTest>(
        "CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithStateRecovery",
        &CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithStateRecovery));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTrainingTest>(
        "CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithParams",
        &CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithParams));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTrainingTest>(
        "CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue",
        &CDataFrameAnalyzerTrainingTest::testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTrainingTest>(
        "CDataFrameAnalyzerTrainingTest::testRunBoostedTreeClassifierTraining",
        &CDataFrameAnalyzerTrainingTest::testRunBoostedTreeClassifierTraining));

    return suiteOfTests;
}
