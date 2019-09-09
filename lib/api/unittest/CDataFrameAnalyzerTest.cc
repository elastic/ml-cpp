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
#include <core/CStopWatch.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/COutliers.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/reader.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TStrVec = std::vector<std::string>;
using TRowItr = core::CDataFrame::TRowItr;
using TPoint = maths::CDenseVector<maths::CFloatStorage>;
using TPointVec = std::vector<TPoint>;

std::vector<rapidjson::Document> stringToJsonDocumentVector(std::string input) {
    rapidjson::StringStream stringStream{input.c_str()};
    std::vector<rapidjson::Document> results;
    while (stringStream.Tell() < input.size() - 1) {
        results.emplace_back();
        rapidjson::ParseResult ok(
            results.back().ParseStream<rapidjson::ParseFlag::kParseStopWhenDoneFlag>(stringStream));
        CPPUNIT_ASSERT(static_cast<bool>(ok) == true);
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

std::string jsonObjectToString(const rapidjson::GenericValue<rapidjson::UTF8<>>& jsonObject) {
    rapidjson::StringBuffer stringBuffer;
    api::CDataFrameAnalysisSpecificationJsonWriter::TRapidJsonLineWriter writer;
    {
        writer.Reset(stringBuffer);
        writer.write(jsonObject);
        writer.Flush();
    }
    return stringBuffer.GetString();
}

std::unique_ptr<maths::CBoostedTree>
createTreeFromJsonObject(std::unique_ptr<ml::core::CDataFrame>& frame,
                         const rapidjson::GenericValue<rapidjson::UTF8<>>& jsonObject) {
    std::stringstream jsonStateStream;
    jsonStateStream << jsonObjectToString(jsonObject);
    return maths::CBoostedTreeFactory::constructFromString(jsonStateStream, *frame);
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
        rows, 5, memoryLimit, 1, {}, true, test::CTestTmpDir::tmpDir(), "ml",
        "outlier_detection", parameters)};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

auto regressionSpec(std::string dependentVariable,
                    std::size_t rows = 100,
                    std::size_t cols = 5,
                    std::size_t memoryLimit = 3000000,
                    std::size_t numberRoundsPerHyperparameter = 0,
                    const TStrVec& categoricalFieldNames = TStrVec{},
                    double lambda = -1.0,
                    double gamma = -1.0,
                    double eta = -1.0,
                    std::size_t maximumNumberTrees = 0,
                    double featureBagFraction = -1.0,
                    std::function<std::shared_ptr<std::ostream>(void)> persistStreamSupplier =
                        []() -> std::shared_ptr<std::ostream> { return nullptr; }) {

    std::string parameters = "{\n\"dependent_variable\": \"" + dependentVariable + "\"";
    if (lambda >= 0.0) {
        parameters += ",\n\"lambda\": " + core::CStringUtils::typeToString(lambda);
    }
    if (gamma >= 0.0) {
        parameters += ",\n\"gamma\": " + core::CStringUtils::typeToString(gamma);
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
    parameters += "\n}";

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        rows, cols, memoryLimit, 1, categoricalFieldNames, true,
        test::CTestTmpDir::tmpDir(), "ml", "regression", parameters)};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec, persistStreamSupplier);
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

    using TMeanVarAccumulatorVec =
        std::vector<maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator>;

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

    TPointVec points(numberInliers + numberOutliers, TPoint(5));
    TMeanVarAccumulatorVec columnMoments(5);

    for (std::size_t i = 0; i < inliers.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                inliers[i][j], core::CIEEE754::E_DoublePrecision);
            points[i](j) = inliers[i][j];
            columnMoments[j].add(inliers[i][j]);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    for (std::size_t i = 0, j = numberInliers; i < outliers.size(); ++j) {
        for (std::size_t k = 0; k < 5; ++i, ++k) {
            fieldValues[k] = core::CStringUtils::typeToStringPrecise(
                outliers[i], core::CIEEE754::E_DoublePrecision);
            points[j](k) = outliers[i];
            columnMoments[k].add(outliers[i]);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }

    auto frame = test::CDataFrameTestUtils::toMainMemoryDataFrame(points);

    maths::COutliers::compute(
        {1, 1, true, method, numberNeighbours, computeFeatureInfluence, 0.05}, *frame);

    expectedScores.resize(points.size());
    expectedFeatureInfluences.resize(points.size(), TDoubleVec(5));

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

void addRegressionTestData(TStrVec fieldNames,
                           TStrVec fieldValues,
                           api::CDataFrameAnalyzer& analyzer,
                           TDoubleVec& expectedPredictions,
                           std::size_t numberExamples = 100,
                           double lambda = -1.0,
                           double gamma = -1.0,
                           double eta = 0.0,
                           std::size_t maximumNumberTrees = 0,
                           double featureBagFraction = 0.0) {

    auto f = [](const TDoubleVec& weights, const TPoint& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors(i);
        }
        return result;
    };

    test::CRandomNumbers rng;

    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};

    TDoubleVec values;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, values);

    TPointVec rows;
    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TPoint row{weights.size() + 1};
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row(j) = values[i + j];
        }
        row(weights.size()) = f(weights, row);

        for (int j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row(j), core::CIEEE754::E_DoublePrecision);
            double xj;
            core::CStringUtils::stringToType(fieldValues[j], xj);
            row(j) = xj;
        }
        analyzer.handleRecord(fieldNames, fieldValues);

        rows.push_back(std::move(row));
    }

    auto frame = test::CDataFrameTestUtils::toMainMemoryDataFrame(rows);

    maths::CBoostedTreeFactory treeFactory{maths::CBoostedTreeFactory::constructFromParameters(
        1, std::make_unique<maths::boosted_tree::CMse>())};
    if (lambda >= 0.0) {
        treeFactory.lambda(lambda);
    }
    if (gamma >= 0.0) {
        treeFactory.gamma(gamma);
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

    std::unique_ptr<maths::CBoostedTree> tree =
        treeFactory.buildFor(*frame, weights.size());

    tree->train();

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            expectedPredictions.push_back(
                (*row)[tree->columnHoldingPrediction(row->numberColumns())]);
        }
    });
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
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage) < 110000); // + 10%
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

void CDataFrameAnalyzerTest::testRunBoostedTreeTraining() {

    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{regressionSpec("c5"), outputWriterFactory};
    addRegressionTestData(fieldNames, fieldValues, analyzer, expectedPredictions);

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
                       counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 2300000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 1050000);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    CPPUNIT_ASSERT(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

void CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithParams() {

    // Test the regression hyperparameter settings are correctly propagated to the
    // analysis runner.

    double lambda{1.0};
    double gamma{10.0};
    double eta{0.9};
    std::size_t maximumNumberTrees{1};
    double featureBagFraction{0.3};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        regressionSpec("c5", 100, 5, 3000000, 0, {}, lambda, gamma, eta,
                       maximumNumberTrees, featureBagFraction),
        outputWriterFactory};

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addRegressionTestData(fieldNames, fieldValues, analyzer, expectedPredictions, 100,
                          lambda, gamma, eta, maximumNumberTrees, featureBagFraction);
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

void CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithRowsMissingTargetValue() {

    // Test we are able to predict value rows for which the dependent variable
    // is missing.

    test::CRandomNumbers rng;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    auto target = [](double feature) { return 10.0 * feature; };

    api::CDataFrameAnalyzer analyzer{regressionSpec("target", 50, 2, 2000000),
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
    auto noPersistStreamSupplier = []() -> std::shared_ptr<std::ostream> {
        return nullptr;
    };

    // Test with bad analysis specification.
    {
        errors.clear();
        api::CDataFrameAnalyzer analyzer{
            std::make_unique<api::CDataFrameAnalysisSpecification>(
                std::string{"junk"}, noPersistStreamSupplier),
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
            regressionSpec("x5", 1000, 5, 8000000, 0, {"x1", "x2"}), outputWriterFactory};

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
                              << "got [" << (*row)[0] << ", " << (*row)[1] << "]");
                }
            }
        });

        CPPUNIT_ASSERT(passed);
    }

    LOG_DEBUG(<< "Test overflow");
    {
        std::size_t rows{api::CDataFrameAnalyzer::MAX_CATEGORICAL_CARDINALITY + 3};

        api::CDataFrameAnalyzer analyzer{
            regressionSpec("x5", rows, 5, 8000000000, 0, {"x1"}, 0, 0, 0, 0, 0),
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
                    i < api::CDataFrameAnalyzer::MAX_CATEGORICAL_CARDINALITY
                        ? static_cast<double>(i)
                        : static_cast<double>(api::CDataFrameAnalyzer::MAX_CATEGORICAL_CARDINALITY)};
                bool wasPassed{passed};
                passed &= (expected == (*row)[0]);
                if (wasPassed && passed == false) {
                    LOG_ERROR(<< "expected " << expected << " got " << (*row)[0]);
                }
            }
        });

        CPPUNIT_ASSERT(passed);
    }
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
        "CDataFrameAnalyzerTest::testRunBoostedTreeTraining",
        &CDataFrameAnalyzerTest::testRunBoostedTreeTraining));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithStateRecovery",
        &CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithStateRecovery));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithParams",
        &CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithParams));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithRowsMissingTargetValue",
        &CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithRowsMissingTargetValue));
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

    return suiteOfTests;
}

void CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithStateRecovery() {

    // no hyperparameter search
    double lambda{1.0};
    double gamma{10.0};
    double eta{0.9};
    std::size_t maximumNumberTrees{2};
    double featureBagFraction{0.3};
    size_t numberRoundsPerHyperparameter{5};

    size_t intermediateIteration{0};
    size_t finalIteration{0};

    // zero hyperparameters to search
    LOG_DEBUG(<< "No hyperparameters to search")
    testRunBoostedTreeTrainingWithStateRecoverySubroutine(
        lambda, gamma, eta, maximumNumberTrees, featureBagFraction,
        numberRoundsPerHyperparameter, intermediateIteration, finalIteration);

    // one hyperparameter to search
    LOG_DEBUG(<< "One hyperparameter to search")
    lambda = -1.0;
    gamma = 10.0;
    finalIteration = 1 * numberRoundsPerHyperparameter - 1;
    testRunBoostedTreeTrainingWithStateRecoverySubroutine(
        lambda, gamma, eta, maximumNumberTrees, featureBagFraction,
        numberRoundsPerHyperparameter, intermediateIteration, finalIteration);

    // two hyperparameters to search
    LOG_DEBUG(<< "Two hyperparameters to search")
    lambda = -1.0;
    gamma = -1.0;
    finalIteration = 2 * numberRoundsPerHyperparameter - 1;
    testRunBoostedTreeTrainingWithStateRecoverySubroutine(
        lambda, gamma, eta, maximumNumberTrees, featureBagFraction,
        numberRoundsPerHyperparameter, intermediateIteration, finalIteration);
}

void CDataFrameAnalyzerTest::testRunBoostedTreeTrainingWithStateRecoverySubroutine(
    double lambda,
    double gamma,
    double eta,
    size_t maximumNumberTrees,
    double featureBagFraction,
    size_t numberRoundsPerHyperparameter,
    size_t intermediateIteration,
    size_t finalIteration) const {
    std::stringstream outputStream;
    std::shared_ptr<std::ostringstream> persistenceStream =
        std::make_shared<std::ostringstream>();
    auto outputWriterFactory = [&outputStream]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
    };
    auto persistStreamSupplier = [&persistenceStream]() -> std::shared_ptr<std::ostream> {
        return persistenceStream;
    };

    size_t numberExamples{100};

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};

    auto f = [](const TDoubleVec& weights, const TPoint& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors(i);
        }
        return result;
    };

    api::CDataFrameAnalyzer analyzer{
        regressionSpec("c5", numberExamples, 5, 15000000,
                       numberRoundsPerHyperparameter, {}, lambda, gamma, eta,
                       maximumNumberTrees, featureBagFraction, persistStreamSupplier),
        outputWriterFactory};

    test::CRandomNumbers rng;

    TDoubleVec values;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, values);

    TPointVec rows;
    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TPoint row{weights.size() + 1};
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row(j) = values[i + j];
        }
        row(weights.size()) = f(weights, row);

        for (int j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row(j), core::CIEEE754::E_DoublePrecision);
            double xj;
            core::CStringUtils::stringToType(fieldValues[j], xj);
            row(j) = xj;
        }
        analyzer.handleRecord(fieldNames, fieldValues);

        rows.push_back(std::move(row));
    }
    auto frame = test::CDataFrameTestUtils::toMainMemoryDataFrame(rows);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    std::vector<rapidjson::Document> persistedStates =
        stringToJsonDocumentVector(persistenceStream->str());

    std::unique_ptr<maths::CBoostedTree> intermediateTree;
    std::unique_ptr<maths::CBoostedTree> finalTree;
    intermediateTree = createTreeFromJsonObject(frame, persistedStates[intermediateIteration]);
    finalTree = createTreeFromJsonObject(frame, persistedStates[finalIteration]);

    CPPUNIT_ASSERT(intermediateTree.get() != nullptr);
    intermediateTree->train();

    rapidjson::Document expectedResults{treeToJsonDocument(*finalTree)};
    const auto& expectedHyperparameters = expectedResults["best_hyperparameters"];

    rapidjson::Document actualResults{treeToJsonDocument(*intermediateTree)};
    const auto& actualHyperparameters = actualResults["best_hyperparameters"];

    auto assertDoublesEqual = [&expectedHyperparameters,
                               &actualHyperparameters](std::string key) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            std::stod(expectedHyperparameters[key].GetString()),
            std::stod(actualHyperparameters[key].GetString()), 1e-4);
    };
    auto assertDoublesArrayEqual = [&expectedHyperparameters,
                                    &actualHyperparameters](std::string key) {
        TDoubleVec expectedVector;
        core::CPersistUtils::fromString(expectedHyperparameters[key].GetString(), expectedVector);
        TDoubleVec actualVector;
        core::CPersistUtils::fromString(actualHyperparameters[key].GetString(), actualVector);
        CPPUNIT_ASSERT_EQUAL(expectedVector.size(), actualVector.size());
        for (size_t i = 0; i < expectedVector.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedVector[i], actualVector[i], 1e-4);
        }
    };
    assertDoublesEqual("hyperparam_lambda");
    assertDoublesEqual("hyperparam_gamma");
    assertDoublesEqual("hyperparam_eta");
    assertDoublesEqual("hyperparam_eta_growth_rate_per_tree");
    assertDoublesEqual("hyperparam_feature_bag_fraction");
    assertDoublesArrayEqual("hyperparam_feature_sample_probabilities");
}
