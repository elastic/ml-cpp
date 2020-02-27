/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <boost/test/tools/interface.hpp>
#include <core/CDataFrame.h>
#include <core/CStopWatch.h>
#include <core/CTimeUtils.h>

#include <maths/CBoostedTreeFactory.h>
#include <maths/CTools.h>

#include <api/CDataFrameAnalysisInstrumentation.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <rapidjson/schema.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <memory>
#include <string>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisInstrumentationTest)

using namespace ml;

namespace {

enum EPredictionType { E_Regression, E_BinaryClassification };
using TStrVec = std::vector<std::string>;
using TDoubleVec = std::vector<double>;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TBoolVec = std::vector<bool>;
using TRowItr = core::CDataFrame::TRowItr;

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

    std::unique_ptr<maths::boosted_tree::CLoss> loss;
    if (type == E_Regression) {
        loss = std::make_unique<maths::boosted_tree::CMse>();
    } else {
        loss = std::make_unique<maths::boosted_tree::CBinomialLogistic>();
    }

    maths::CBoostedTreeFactory treeFactory{
        maths::CBoostedTreeFactory::constructFromParameters(1, std::move(loss))};
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

    ml::api::CDataFrameTrainBoostedTreeInstrumentation instrumentation("testJob");
    treeFactory.analysisInstrumentation(instrumentation);

    auto tree = treeFactory.buildFor(*frame, weights.size());

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
}

BOOST_AUTO_TEST_CASE(testMemoryState) {
    std::string jobId{"testJob"};
    std::int64_t memoryUsage{1000};
    std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    std::stringstream outpustStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper(outpustStream);
        core::CRapidJsonConcurrentLineWriter writer(streamWrapper);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
        instrumentation.updateMemoryUsage(memoryUsage);
        instrumentation.writer(&writer);
        instrumentation.nextStep(0);
        outpustStream.flush();
    }
    std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(outpustStream.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);

    const auto& result{results[0]};
    BOOST_TEST_REQUIRE(result["job_id"].GetString() == jobId);
    BOOST_TEST_REQUIRE(result["type"].GetString() == "analytics_memory_usage");
    BOOST_TEST_REQUIRE(result["peak_usage_bytes"].GetInt64() == memoryUsage);
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() >= timeBefore);
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() <= timeAfter);
}

BOOST_AUTO_TEST_CASE(testAnalysisTrainState) {
    std::string jobId{"testJob"};
    std::int64_t memoryUsage{1000};
    std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    std::stringstream outputStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper(outputStream);
        core::CRapidJsonConcurrentLineWriter writer(streamWrapper);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
        instrumentation.writer(&writer);
        instrumentation.nextStep(0);
        outputStream.flush();
    }
    std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    LOG_DEBUG(<< outputStream.str());

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(outputStream.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);

    const auto& result{results[0]};
    BOOST_TEST_REQUIRE(result["job_id"].GetString() == jobId);
    BOOST_TEST_REQUIRE(result["type"].GetString() == "analytics_memory_usage");
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() >= timeBefore);
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() <= timeAfter);
}

BOOST_AUTO_TEST_CASE(testTrainingRegression) {
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

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::ifstream schemaFileStream("testfiles/instrumentation/supervised_learning_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);
    rapidjson::SchemaValidator validator(schema);

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("analysis_stats")) {
            BOOST_TEST_REQUIRE(result["analysis_stats"].HasMember("classification_stats"));
            if (result["analysis_stats"]["classification_stats"].Accept(validator) == false) {
                rapidjson::StringBuffer sb;
                validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
                sb.Clear();
                validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed");
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testTrainingClassification) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::classification(),
            "target", 100, 5, 6000000, 0, 0, {"target"}),
        outputWriterFactory};
    addPredictionTestData(E_BinaryClassification, fieldNames, fieldValues,
                          analyzer, expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    LOG_DEBUG(<< output.str());

    std::ifstream schemaFileStream("testfiles/instrumentation/supervised_learning_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);
    rapidjson::SchemaValidator validator(schema);

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("analysis_stats")) {
            BOOST_TEST_REQUIRE(result["analysis_stats"].HasMember("regression_stats"));
            if (result["analysis_stats"]["regression_stats"].Accept(validator) == false) {
                rapidjson::StringBuffer sb;
                validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
                sb.Clear();
                validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed");
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
