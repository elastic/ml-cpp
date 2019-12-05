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

#include <memory>

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVec::iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TStrVec::iterator)

BOOST_AUTO_TEST_SUITE(CDataFrameAnalyzerFeatureImportanceTest)

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
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceAllShap) {

    // Test that feature importance correctly recognize the impact of regressors in a linear model.

    double alpha{2.0};
    double lambda{1.0};
    double gamma{10.0};
    double softTreeDepthLimit{5.0};
    double softTreeDepthTolerance{0.1};
    double eta{0.9};
    std::size_t maximumNumberTrees{1};
    double featureBagFraction{1.0};

    int rows = 200;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            "regression", "c5", rows, 5, 4000000, 0, 0, {"c1"}, alpha, lambda,
            gamma, softTreeDepthLimit, softTreeDepthTolerance, eta,
            maximumNumberTrees, featureBagFraction, 4),
        outputWriterFactory};

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    test::CRandomNumbers rng;
    TDoubleVec weights{50, 150, 50, -50};

    TDoubleVec values;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * rows, values);

    // make the first column categorical
    for (auto it = values.begin(); it < values.end(); it += 4) {
        *it = (*it < 0) ? -10 : 10;
    }

    auto frame = setupLinearRegressionData(fieldNames, fieldValues, analyzer, weights, values);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    double c1Sum, c2Sum, c3Sum, c4Sum;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            c1Sum += std::fabs(result["row_results"]["results"]["ml"]["shap.c1"].GetDouble());
            c2Sum += std::fabs(result["row_results"]["results"]["ml"]["shap.c2"].GetDouble());
            c3Sum += std::fabs(result["row_results"]["results"]["ml"]["shap.c3"].GetDouble());
            c4Sum += std::fabs(result["row_results"]["results"]["ml"]["shap.c4"].GetDouble());
        }
    }
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 80); // c3 and c4 within 80% of each other
}

BOOST_AUTO_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceNoShap) {

    // Test that feature importance correctly recognize the impact of regressors in a linear model.

    double alpha{2.0};
    double lambda{1.0};
    double gamma{10.0};
    double softTreeDepthLimit{5.0};
    double softTreeDepthTolerance{0.1};
    double eta{0.9};
    std::size_t maximumNumberTrees{1};
    double featureBagFraction{1.0};

    int rows = 200;
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            "regression", "c5", rows, 5, 4000000, 0, 0, {"c1"}, alpha, lambda,
            gamma, softTreeDepthLimit, softTreeDepthTolerance, eta,
            maximumNumberTrees, featureBagFraction, 0),
        outputWriterFactory};

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    test::CRandomNumbers rng;
    TDoubleVec weights{50, 150, 50, -50};

    TDoubleVec values;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * rows, values);

    // make last column categorical
    for (auto it = values.begin(); it < values.end(); it += 4) {
        *it = (*it < 0) ? -10 : 10;
    }

    auto frame = setupLinearRegressionData(fieldNames, fieldValues, analyzer, weights, values);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c1") == false);
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c2") == false);
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c3") == false);
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c4") == false);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
