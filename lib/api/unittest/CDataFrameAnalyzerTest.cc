/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalyzerTest.h"

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/COutliers.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TStrVec = std::vector<std::string>;
using TPoint = maths::CDenseVector<maths::CFloatStorage>;
using TPointVec = std::vector<TPoint>;

std::unique_ptr<api::CDataFrameAnalysisSpecification>
outlierSpec(std::size_t rows = 110,
            std::size_t memoryLimit = 100000,
            std::string method = "",
            std::size_t numberNeighbours = 0,
            bool computeFeatureInfluence = false) {

    std::string spec{"{\n"
                     "  \"rows\": " +
                     std::to_string(rows) +
                     ",\n"
                     "  \"cols\": 5,\n"
                     "  \"memory_limit\": " +
                     std::to_string(memoryLimit) +
                     ",\n"
                     "  \"threads\": 1,\n"
                     "  \"disk_usage_allowed\": true,\n"
                     "  \"analysis\": {\n"
                     "    \"name\": \"outlier_detection\""};
    spec += ",\n    \"parameters\": {\n";
    bool hasTrailingParameter{false};
    if (method != "") {
        spec += "      \"method\": \"" + method + "\"";
        hasTrailingParameter = true;
    }
    if (numberNeighbours > 0) {
        spec += (hasTrailingParameter ? ",\n" : "");
        spec += "      \"n_neighbors\": " + core::CStringUtils::typeToString(numberNeighbours);
        hasTrailingParameter = true;
    }
    if (computeFeatureInfluence == false) {
        spec += (hasTrailingParameter ? ",\n" : "");
        spec += "      \"compute_feature_influence\": false";
        hasTrailingParameter = true;
    } else {
        spec += (hasTrailingParameter ? ",\n" : "");
        spec += "      \"feature_influence_threshold\": 0.0";
        hasTrailingParameter = true;
    }
    spec += (hasTrailingParameter ? "\n" : "");
    spec += "    }\n";
    spec += "  }\n"
            "}";

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

void addTestData(TStrVec fieldNames,
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

    frame->readRows(1, [&](core::CDataFrame::TRowItr beginRows, core::CDataFrame::TRowItr endRows) {
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
}

void CDataFrameAnalyzerTest::testWithoutControlMessages() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5"};
    TStrVec fieldValues{"", "", "", "", ""};
    addTestData(fieldNames, fieldValues, analyzer, expectedScores, expectedFeatureInfluences);

    analyzer.receivedAllRows();
    analyzer.run();

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
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
            CPPUNIT_ASSERT(result.HasMember("progress_percent") == false);
            ++expectedScore;
        } else if (result.HasMember("progress_percent")) {
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() >= 0);
            CPPUNIT_ASSERT(result["progress_percent"].GetInt() <= 100);
            CPPUNIT_ASSERT(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    CPPUNIT_ASSERT(expectedScore == expectedScores.end());
}

void CDataFrameAnalyzerTest::testRunOutlierDetection() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addTestData(fieldNames, fieldValues, analyzer, expectedScores, expectedFeatureInfluences);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
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
    addTestData(fieldNames, fieldValues, analyzer, expectedScores,
                expectedFeatureInfluences, 990, 10);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
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
    addTestData(fieldNames, fieldValues, analyzer, expectedScores, expectedFeatureInfluences,
                100, 10, maths::COutliers::E_Ensemble, 0, true);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
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

    TStrVec methods{"lof", "ldof", "knn", "tnn"};

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
            addTestData(fieldNames, fieldValues, analyzer, expectedScores,
                        expectedFeatureInfluences, 100, 10, method, k);
            analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

            rapidjson::Document results;
            rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
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
        api::CDataFrameAnalyzer analyzer{outlierSpec(2), outputWriterFactory};
        errors.clear();
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        CPPUNIT_ASSERT_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
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
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
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
        "CDataFrameAnalyzerTest::testFlushMessage", &CDataFrameAnalyzerTest::testFlushMessage));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testErrors", &CDataFrameAnalyzerTest::testErrors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRoundTripDocHashes",
        &CDataFrameAnalyzerTest::testRoundTripDocHashes));

    return suiteOfTests;
}
