/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalyzerTest.h"

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CProgramCounters.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COutliers.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/ElasticsearchStateIndex.h>

#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <rapidjson/error/en.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
using TDataAdderUPtr = std::unique_ptr<ml::core::CDataAdder>;
using TPersisterSupplier = std::function<TDataAdderUPtr()>;
using TDataSearcherUPtr = std::unique_ptr<ml::core::CDataSearcher>;
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

auto outlierSpec(std::size_t rows = 110,
                 std::size_t memoryLimit = 100000,
                 std::string method = "",
                 std::size_t numberNeighbours = 0,
                 bool computeFeatureInfluence = false) {

    std::string parameters = "{\n";
    bool hasTrailingParameter{false};
    if (method != "") {
        parameters += "\"method\": \"" + method + "\"";
        hasTrailingParameter = true;
    }
    if (numberNeighbours > 0) {
        parameters += (hasTrailingParameter ? ",\n" : "");
        parameters += "\"n_neighbors\": " + core::CStringUtils::typeToString(numberNeighbours);
        hasTrailingParameter = true;
    }
    if (computeFeatureInfluence == false) {
        parameters += (hasTrailingParameter ? ",\n" : "");
        parameters += "\"compute_feature_influence\": false";
        hasTrailingParameter = true;
    } else {
        parameters += (hasTrailingParameter ? ",\n" : "");
        parameters += "\"feature_influence_threshold\": 0.0";
        hasTrailingParameter = true;
    }
    parameters += (hasTrailingParameter ? "\n" : "");
    parameters += "}\n";

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, 5, memoryLimit, 1, {}, true,
        test::CTestTmpDir::tmpDir(), "ml", "outlier_detection", parameters)};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

auto predictionSpec(std::string analysis,
                    std::string dependentVariable,
                    std::size_t rows = 100,
                    std::size_t cols = 5,
                    std::size_t memoryLimit = 3000000,
                    std::size_t numberRoundsPerHyperparameter = 0,
                    std::size_t bayesianOptimisationRestarts = 0,
                    const TStrVec& categoricalFieldNames = TStrVec{},
                    double alpha = -1.0,
                    double lambda = -1.0,
                    double gamma = -1.0,
                    double softTreeDepthLimit = -1.0,
                    double softTreeDepthTolerance = -1.0,
                    double eta = -1.0,
                    std::size_t maximumNumberTrees = 0,
                    double featureBagFraction = -1.0,
                    TPersisterSupplier* persisterSupplier = nullptr,
                    TRestoreSearcherSupplier* restoreSearcherSupplier = nullptr) {

    std::string parameters = "{\n\"dependent_variable\": \"" + dependentVariable + "\"";
    if (alpha >= 0.0) {
        parameters += ",\n\"alpha\": " + core::CStringUtils::typeToString(alpha);
    }
    if (lambda >= 0.0) {
        parameters += ",\n\"lambda\": " + core::CStringUtils::typeToString(lambda);
    }
    if (gamma >= 0.0) {
        parameters += ",\n\"gamma\": " + core::CStringUtils::typeToString(gamma);
    }
    if (softTreeDepthLimit >= 0.0) {
        parameters += ",\n\"soft_tree_depth_limit\": " +
                      core::CStringUtils::typeToString(softTreeDepthLimit);
    }
    if (softTreeDepthTolerance >= 0.0) {
        parameters += ",\n\"soft_tree_depth_tolerance\": " +
                      core::CStringUtils::typeToString(softTreeDepthTolerance);
    }
    if (eta > 0.0) {
        parameters += ",\n\"eta\": " + core::CStringUtils::typeToString(eta);
    }
    if (maximumNumberTrees > 0) {
        parameters += ",\n\"maximum_number_trees\": " +
                      core::CStringUtils::typeToString(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0) {
        parameters += ",\n\"feature_bag_fraction\": " +
                      core::CStringUtils::typeToString(featureBagFraction);
    }
    if (numberRoundsPerHyperparameter > 0) {
        parameters += ",\n\"number_rounds_per_hyperparameter\": " +
                      core::CStringUtils::typeToString(numberRoundsPerHyperparameter);
    }
    if (bayesianOptimisationRestarts > 0) {
        parameters += ",\n\"bayesian_optimisation_restarts\": " +
                      core::CStringUtils::typeToString(bayesianOptimisationRestarts);
    }
    parameters += "\n}";

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, memoryLimit, 1, categoricalFieldNames, true,
        test::CTestTmpDir::tmpDir(), "ml", analysis, parameters)};

    LOG_TRACE(<< "spec =\n" << spec);

    if (restoreSearcherSupplier != nullptr && persisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(
            spec, *persisterSupplier, *restoreSearcherSupplier);
    } else if (restoreSearcherSupplier == nullptr && persisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(spec, *persisterSupplier);
    } else {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
    }
}

void addOutlierTestData(TStrVec fieldNames,
                        TStrVec fieldValues,
                        api::CDataFrameAnalyzer& analyzer,
                        TDoubleVec& expectedScores,
                        TDoubleVecVec& expectedFeatureInfluences,
                        std::size_t numberInliers = 100,
                        std::size_t numberOutliers = 10,
                        maths::COutliers::EMethod method = maths::COutliers::E_Ensemble,
                        std::size_t numberNeighbours = 0,
                        bool computeFeatureInfluence = false) {

    test::CRandomNumbers rng;

    TDoubleVec mean{1.0, 10.0, 4.0, 8.0, 3.0};
    TDoubleVecVec covariance{{1.0, 0.1, -0.1, 0.3, 0.2},
                             {0.1, 1.3, -0.3, 0.1, 0.1},
                             {-0.1, -0.3, 2.1, 0.1, 0.2},
                             {0.3, 0.1, 0.1, 0.8, 0.2},
                             {0.2, 0.1, 0.2, 0.2, 2.2}};

    TDoubleVecVec inliers;
    rng.generateMultivariateNormalSamples(mean, covariance, numberInliers, inliers);

    TDoubleVec outliers;
    rng.generateUniformSamples(0.0, 10.0, numberOutliers * 5, outliers);

    auto frame = core::makeMainStorageDataFrame(5).first;

    for (std::size_t i = 0; i < inliers.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                inliers[i][j], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(core::CVectorRange<const TStrVec>(fieldValues, 0, 5));
    }
    for (std::size_t i = 0; i < outliers.size(); i += 5) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                outliers[i + j], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(core::CVectorRange<const TStrVec>(fieldValues, 0, 5));
    }

    frame->finishWritingRows();

    maths::COutliers::compute(
        {1, 1, true, method, numberNeighbours, computeFeatureInfluence, 0.05}, *frame);

    expectedScores.resize(numberInliers + numberOutliers);
    expectedFeatureInfluences.resize(numberInliers + numberOutliers, TDoubleVec(5));

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            expectedScores[row->index()] = (*row)[5];
            if (computeFeatureInfluence) {
                for (std::size_t i = 6; i < 11; ++i) {
                    expectedFeatureInfluences[row->index()][i - 6] = (*row)[i];
                }
            }
        }
    });
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

void CDataFrameAnalyzerTest::testWithoutControlMessages() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    std::stringstream persistStream;

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5"};
    TStrVec fieldValues{"", "", "", "", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences);

    analyzer.receivedAllRows();
    analyzer.run();

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedScore != expectedScores.end());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                *expectedScore,
                result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                1e-4 * *expectedScore);
            CPPUNIT_ASSERT(result.HasMember("progress_percent") == false);
            ++expectedScore;
        } else if (result.HasMember("progress_percent")) {
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() >= 0);
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() <= 100);
            CPPUNIT_ASSERT(result.HasMember("row_results") == false);
        }
    }
    CPPUNIT_ASSERT(expectedScore == expectedScores.end());
}

void CDataFrameAnalyzerTest::testRunOutlierDetection() {

    // Test the results the analyzer produces match running outlier detection
    // directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedScore != expectedScores.end());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                *expectedScore,
                result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                1e-4 * *expectedScore);
            ++expectedScore;
            CPPUNIT_ASSERT(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() >= 0);
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() <= 100);
            CPPUNIT_ASSERT(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    CPPUNIT_ASSERT(expectedScore == expectedScores.end());
    CPPUNIT_ASSERT(progressCompleted);

    LOG_DEBUG(<< "number partitions = "
              << core::CProgramCounters::counter(counter_t::E_DFONumberPartitions));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage));
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFONumberPartitions) == 1);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage) < 100000);
}

void CDataFrameAnalyzerTest::testRunOutlierDetectionPartitioned() {

    // Test the case we have to overflow to disk to compute outliers subject
    // to the memory constraints.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(1000), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences, 990, 10);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedScore != expectedScores.end());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                *expectedScore,
                result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                1e-4 * *expectedScore);
            ++expectedScore;
        }
    }
    CPPUNIT_ASSERT(expectedScore == expectedScores.end());

    LOG_DEBUG(<< "number partitions = "
              << core::CProgramCounters::counter(counter_t::E_DFONumberPartitions));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage));
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFONumberPartitions) > 1);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage) < 116000); // + 16%
}

void CDataFrameAnalyzerTest::testRunOutlierFeatureInfluences() {

    // Test we compute and write out the feature influences when requested.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(110, 100000, "", 0, true), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;
    TStrVec expectedNames{"feature_influence.c1", "feature_influence.c2", "feature_influence.c3",
                          "feature_influence.c4", "feature_influence.c5"};

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores, expectedFeatureInfluences,
                       100, 10, maths::COutliers::E_Ensemble, 0, true);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedFeatureInfluence = expectedFeatureInfluences.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            CPPUNIT_ASSERT(expectedFeatureInfluence != expectedFeatureInfluences.end());
            for (std::size_t i = 0; i < 5; ++i) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                    (*expectedFeatureInfluence)[i],
                    result["row_results"]["results"]["ml"][expectedNames[i]].GetDouble(),
                    1e-4 * (*expectedFeatureInfluence)[i]);
            }
            ++expectedFeatureInfluence;
        }
    }
    CPPUNIT_ASSERT(expectedFeatureInfluence == expectedFeatureInfluences.end());
}

void CDataFrameAnalyzerTest::testRunOutlierDetectionWithParams() {

    // Test the method and number of neighbours parameters are correctly
    // propagated to the analysis runner.

    TStrVec methods{"lof", "ldof", "distance_kth_nn", "distance_knn"};

    for (const auto& method :
         {maths::COutliers::E_Lof, maths::COutliers::E_Ldof,
          maths::COutliers::E_DistancekNN, maths::COutliers::E_TotalDistancekNN}) {
        for (const auto k : {5, 10}) {

            LOG_DEBUG(<< "Testing '" << methods[method] << "' and '" << k << "'");

            std::stringstream output;
            auto outputWriterFactory = [&output]() {
                return std::make_unique<core::CJsonOutputStreamWrapper>(output);
            };

            api::CDataFrameAnalyzer analyzer{
                outlierSpec(110, 1000000, methods[method], k), outputWriterFactory};

            TDoubleVec expectedScores;
            TDoubleVecVec expectedFeatureInfluences;

            TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
            TStrVec fieldValues{"", "", "", "", "", "0", ""};
            addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                               expectedFeatureInfluences, 100, 10, method, k);
            analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

            rapidjson::Document results;
            rapidjson::ParseResult ok(results.Parse(output.str()));
            CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

            auto expectedScore = expectedScores.begin();
            for (const auto& result : results.GetArray()) {
                if (result.HasMember("row_results")) {
                    CPPUNIT_ASSERT(expectedScore != expectedScores.end());
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        *expectedScore,
                        result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                        1e-6 * *expectedScore);
                    ++expectedScore;
                }
            }
            CPPUNIT_ASSERT(expectedScore == expectedScores.end());
        }
    }
}

void CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTraining() {

    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{predictionSpec("regression", "c5"), outputWriterFactory};
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
                       counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 2700000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1050000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

void CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithParams() {

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
        predictionSpec("regression", "c5", 100, 5, 3000000, 0, 0, {}, alpha,
                       lambda, gamma, softTreeDepthLimit, softTreeDepthTolerance,
                       eta, maximumNumberTrees, featureBagFraction),
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

void CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue() {

    // Test we are able to predict value rows for which the dependent variable
    // is missing.

    test::CRandomNumbers rng;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    auto target = [](double feature) { return 10.0 * feature; };

    api::CDataFrameAnalyzer analyzer{
        predictionSpec("regression", "target", 50, 2, 2000000), outputWriterFactory};

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

void CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithStateRecovery() {

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

    TSizeVec intermediateIterations;
    std::size_t finalIteration{0};

    test::CRandomNumbers rng;

    // TODO re-enable case that all parameters are set.
    for (const auto& params :
         {/*SHyperparameters{},*/ SHyperparameters{-1.0},
          SHyperparameters{-1.0, -1.0}, SHyperparameters{-1.0, -1.0, -1.0}}) {

        LOG_DEBUG(<< "Number parameters to search = " << params.numberUnset());

        auto makeSpec = [&](const std::string& dependentVariable, std::size_t numberExamples,
                            TPersisterSupplier persisterSupplier) {
            return predictionSpec("regression", dependentVariable, numberExamples,
                                  5, 15000000, numberRoundsPerHyperparameter,
                                  12, {}, params.s_Alpha, params.s_Lambda,
                                  params.s_Gamma, params.s_SoftTreeDepthLimit,
                                  params.s_SoftTreeDepthTolerance, params.s_Eta,
                                  params.s_MaximumNumberTrees,
                                  params.s_FeatureBagFraction, &persisterSupplier);
        };

        finalIteration = params.numberUnset() * numberRoundsPerHyperparameter - 1;
        rng.generateUniformSamples(0, finalIteration - 1, 3, intermediateIterations);

        for (auto intermediateIteration : intermediateIterations) {
            LOG_DEBUG(<< "restart from " << intermediateIteration);
            testOneRunOfBoostedTreeTrainingWithStateRecovery(makeSpec, intermediateIteration);
        }
    }
}

void CDataFrameAnalyzerTest::testRunBoostedTreeClassifierTraining() {

    // Test the results the analyzer produces match running classification directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        predictionSpec("classification", "c5", 100, 5, 3000000, 0, 0, {"c5"}),
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
                       counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 2600000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1050000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

void CDataFrameAnalyzerTest::testFlushMessage() {

    // Test that white space is just ignored.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
    CPPUNIT_ASSERT_EQUAL(
        true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                    {"", "", "", "", "", "", "           "}));
}

void CDataFrameAnalyzerTest::testErrors() {

    std::vector<std::string> errors;
    std::mutex errorsMutex;
    auto errorHandler = [&errors, &errorsMutex](std::string error) {
        std::lock_guard<std::mutex> lock{errorsMutex};
        errors.push_back(error);
    };

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    // Test with bad analysis specification.
    {
        errors.clear();
        api::CDataFrameAnalyzer analyzer{
            std::make_unique<api::CDataFrameAnalysisSpecification>(std::string{"junk"}),
            outputWriterFactory};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
        CPPUNIT_ASSERT_EQUAL(false,
                             analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5"},
                                                   {"10", "10", "10", "10", "10"}));
    }

    // Test special field in the wrong position
    {
        errors.clear();
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", ".", "c4", "c5", "."},
                                         {"10", "10", "10", "", "10", "10", ""}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    // Test missing special field
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", "."},
                                         {"10", "10", "10", "10", "10", ""}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    // Test bad control message
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                         {"10", "10", "10", "10", "10", "", "foo"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    // Test bad input
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                         {"10", "10", "10", "10", "10"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    // Test inconsistent number of rows
    {
        // Fewer rows than expected is ignored.
        api::CDataFrameAnalyzer analyzer{outlierSpec(2), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.empty());
    }
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(2), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    // No data.
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(2), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
        CPPUNIT_ASSERT_EQUAL(std::string{"Input error: no data sent."}, errors[0]);
    }
}

void CDataFrameAnalyzerTest::testRoundTripDocHashes() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(9), outputWriterFactory};
    for (auto i : {"1", "2", "3", "4", "5", "6", "7", "8", "9"}) {
        analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                              {i, i, i, i, i, i, ""});
    }

    analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                          {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    int expectedHash{0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            LOG_DEBUG(<< "checksum = " << result["row_results"]["checksum"].GetInt());
            CPPUNIT_ASSERT_EQUAL(++expectedHash,
                                 result["row_results"]["checksum"].GetInt());
        }
    }
}

void CDataFrameAnalyzerTest::testCategoricalFields() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    {
        api::CDataFrameAnalyzer analyzer{
            predictionSpec("regression", "x5", 1000, 5, 8500000, 0, 0, {"x1", "x2"}),
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

        CPPUNIT_ASSERT(passed);
    }

    LOG_DEBUG(<< "Test overflow");
    {
        std::size_t rows{core::CDataFrame::MAX_CATEGORICAL_CARDINALITY + 3};

        api::CDataFrameAnalyzer analyzer{
            predictionSpec("regression", "x5", rows, 5, 8000000000, 0, 0, {"x1"}),
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

        CPPUNIT_ASSERT(passed);
    }
}

void CDataFrameAnalyzerTest::testCategoricalFieldsEmptyAsMissing() {

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
        CPPUNIT_ASSERT_EQUAL_MESSAGE("row " + std::to_string(row_i),
                                     matchers.size(), row.numberColumns());
        for (std::size_t i = 0; i < row.numberColumns(); ++i) {
            CPPUNIT_ASSERT_MESSAGE("row " + std::to_string(row_i) +
                                       ", column " + std::to_string(i),
                                   matchers[i](row[i]));
        }
    };

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{predictionSpec("classification", "x5", 1000, 5,
                                                    8500000, 0, 0, {"x1", "x2", "x5"}),
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
        CPPUNIT_ASSERT_EQUAL(std::size_t{10}, rows.size());
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

CppUnit::Test* CDataFrameAnalyzerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameAnalyzerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testWithoutControlMessages",
        &CDataFrameAnalyzerTest::testWithoutControlMessages));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunOutlierDetection",
        &CDataFrameAnalyzerTest::testRunOutlierDetection));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunOutlierDetectionPartitioned",
        &CDataFrameAnalyzerTest::testRunOutlierDetectionPartitioned));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunOutlierFeatureInfluences",
        &CDataFrameAnalyzerTest::testRunOutlierFeatureInfluences));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunOutlierDetectionWithParams",
        &CDataFrameAnalyzerTest::testRunOutlierDetectionWithParams));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTraining",
        &CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTraining));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithStateRecovery",
        &CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithStateRecovery));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithParams",
        &CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithParams));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue",
        &CDataFrameAnalyzerTest::testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeClassifierTraining",
        &CDataFrameAnalyzerTest::testRunBoostedTreeClassifierTraining));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testFlushMessage", &CDataFrameAnalyzerTest::testFlushMessage));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testErrors", &CDataFrameAnalyzerTest::testErrors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRoundTripDocHashes",
        &CDataFrameAnalyzerTest::testRoundTripDocHashes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testCategoricalFields",
        &CDataFrameAnalyzerTest::testCategoricalFields));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testCategoricalFieldsEmptyAsMissing",
        &CDataFrameAnalyzerTest::testCategoricalFieldsEmptyAsMissing));

    return suiteOfTests;
}
