/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>

#include <maths/CDataFrameUtils.h>
#include <maths/COutliers.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

#include <memory>
#include <string>
#include <vector>

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVec::iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVecVec::iterator)

BOOST_AUTO_TEST_SUITE(CDataFrameAnalyzerOutlierTest)

using namespace ml;

namespace {
using TStrVec = std::vector<std::string>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;

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
}

BOOST_AUTO_TEST_CASE(testWithoutControlMessages) {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    std::stringstream persistStream;

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};

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
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedScore != expectedScores.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedScore,
                result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                1e-4 * *expectedScore);
            BOOST_TEST_REQUIRE(result.HasMember("progress_percent") == false);
            ++expectedScore;
        } else if (result.HasMember("progress_percent")) {
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() >= 0);
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() <= 100);
            BOOST_TEST_REQUIRE(result.HasMember("row_results") == false);
        }
    }
    BOOST_TEST_REQUIRE(expectedScore == expectedScores.end());
}

BOOST_AUTO_TEST_CASE(testRunOutlierDetection) {

    // Test the results the analyzer produces match running outlier detection
    // directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    bool progressCompleted{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedScore != expectedScores.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedScore,
                result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                1e-4 * *expectedScore);
            ++expectedScore;
            BOOST_TEST_REQUIRE(result.HasMember("progress_percent") == false);
        } else if (result.HasMember("progress_percent")) {
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() >= 0);
            BOOST_TEST_REQUIRE(result["progress_percent"].GetInt() <= 100);
            BOOST_TEST_REQUIRE(result.HasMember("row_results") == false);
            progressCompleted = result["progress_percent"].GetInt() == 100;
        }
    }
    BOOST_TEST_REQUIRE(expectedScore == expectedScores.end());
    BOOST_TEST_REQUIRE(progressCompleted);

    LOG_DEBUG(<< "number partitions = "
              << core::CProgramCounters::counter(counter_t::E_DFONumberPartitions));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage));
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFONumberPartitions) == 1);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage) < 100000);
}

BOOST_AUTO_TEST_CASE(testRunOutlierDetectionPartitioned) {

    // Test the case we have to overflow to disk to compute outliers subject
    // to the memory constraints.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::outlierSpec(1000), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences, 990, 10);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedScore != expectedScores.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedScore,
                result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                1e-4 * *expectedScore);
            ++expectedScore;
        }
    }
    BOOST_TEST_REQUIRE(expectedScore == expectedScores.end());

    LOG_DEBUG(<< "number partitions = "
              << core::CProgramCounters::counter(counter_t::E_DFONumberPartitions));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage));
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFONumberPartitions) > 1);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage) <
                       116000); // + 16%
}

BOOST_AUTO_TEST_CASE(testRunOutlierFeatureInfluences) {

    // Test we compute and write out the feature influences when requested.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
                                         110, 5, 100000, "", 0, true),
                                     outputWriterFactory};

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
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    auto expectedFeatureInfluence = expectedFeatureInfluences.begin();
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(expectedFeatureInfluence !=
                               expectedFeatureInfluences.end());
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    (*expectedFeatureInfluence)[i],
                    result["row_results"]["results"]["ml"][expectedNames[i]].GetDouble(),
                    1e-4 * (*expectedFeatureInfluence)[i]);
            }
            ++expectedFeatureInfluence;
        }
    }
    BOOST_TEST_REQUIRE(expectedFeatureInfluence == expectedFeatureInfluences.end());
}

BOOST_AUTO_TEST_CASE(testRunOutlierDetectionWithParams) {

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
                test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
                    110, 5, 1000000, methods[method], k, false),
                outputWriterFactory};

            TDoubleVec expectedScores;
            TDoubleVecVec expectedFeatureInfluences;

            TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
            TStrVec fieldValues{"", "", "", "", "", "0", ""};
            addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                               expectedFeatureInfluences, 100, 10, method, k);
            analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

            rapidjson::Document results;
            rapidjson::ParseResult ok(results.Parse(output.str()));
            BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

            auto expectedScore = expectedScores.begin();
            for (const auto& result : results.GetArray()) {
                if (result.HasMember("row_results")) {
                    BOOST_TEST_REQUIRE(expectedScore != expectedScores.end());
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        *expectedScore,
                        result["row_results"]["results"]["ml"]["outlier_score"].GetDouble(),
                        1e-6 * *expectedScore);
                    ++expectedScore;
                }
            }
            BOOST_TEST_REQUIRE(expectedScore == expectedScores.end());
        }
    }
}

BOOST_AUTO_TEST_CASE(testFlushMessage) {

    // Test that white space is just ignored.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};
    BOOST_REQUIRE_EQUAL(
        true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                    {"", "", "", "", "", "", "           "}));
}

BOOST_AUTO_TEST_CASE(testErrors) {

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
        BOOST_TEST_REQUIRE(errors.size() > 0);
        BOOST_REQUIRE_EQUAL(false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5"},
                                                         {"10", "10", "10", "10", "10"}));
    }

    // Test special field in the wrong position
    {
        errors.clear();
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};
        BOOST_REQUIRE_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", ".", "c4", "c5", "."},
                                         {"10", "10", "10", "", "10", "10", ""}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.size() > 0);
    }

    // Test missing special field
    {
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};
        errors.clear();
        BOOST_REQUIRE_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", "."},
                                         {"10", "10", "10", "10", "10", ""}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.size() > 0);
    }

    // Test bad control message
    {
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};
        errors.clear();
        BOOST_REQUIRE_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                         {"10", "10", "10", "10", "10", "", "foo"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.size() > 0);
    }

    // Test bad input
    {
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(), outputWriterFactory};
        errors.clear();
        BOOST_REQUIRE_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                         {"10", "10", "10", "10", "10"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.size() > 0);
    }

    // Test inconsistent number of rows
    {
        // Fewer rows than expected is ignored.
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(2), outputWriterFactory};
        errors.clear();
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.empty());
    }
    {
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(2), outputWriterFactory};
        errors.clear();
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"10", "10", "10", "10", "10", "0", ""}));
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.size() > 0);
    }

    // No data.
    {
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::outlierSpec(2), outputWriterFactory};
        errors.clear();
        BOOST_REQUIRE_EQUAL(
            true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                                        {"", "", "", "", "", "", "$"}));
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST_REQUIRE(errors.size() > 0);
        BOOST_REQUIRE_EQUAL(std::string{"Input error: no data sent."}, errors[0]);
    }
}

BOOST_AUTO_TEST_CASE(testRoundTripDocHashes) {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::outlierSpec(9), outputWriterFactory};
    for (auto i : {"1", "2", "3", "4", "5", "6", "7", "8", "9"}) {
        analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                              {i, i, i, i, i, i, ""});
    }

    analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", ".", "."},
                          {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    int expectedHash{0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            LOG_DEBUG(<< "checksum = " << result["row_results"]["checksum"].GetInt());
            BOOST_REQUIRE_EQUAL(++expectedHash,
                                result["row_results"]["checksum"].GetInt());
        }
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
                "regression", "x5", 1000, 5, 10000000, 0, 0, {"x1", "x2"}),
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
                "regression", "x5", rows, 5, 8000000000, 0, 0, {"x1"}),
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
            "classification", "x5", 1000, 5, 10000000, 0, 0, {"x1", "x2", "x5"}),
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
