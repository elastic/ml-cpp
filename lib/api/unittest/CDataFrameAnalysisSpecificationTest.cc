/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>

#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameClassificationRunner.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CDataFrameRegressionRunner.h>

#include <test/CTestTmpDir.h>

#include "CDataFrameMockAnalysisRunner.h"

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisSpecificationTest)

using namespace ml;

namespace {
using TStrVec = std::vector<std::string>;
using TRunnerFactoryUPtr = std::unique_ptr<api::CDataFrameAnalysisRunnerFactory>;
using TRunnerFactoryUPtrVec = std::vector<TRunnerFactoryUPtr>;
}

BOOST_AUTO_TEST_CASE(testCreate) {

    // This test focuses on checking the validation code we apply to the object
    // rather than the JSON parsing so we don't bother with random fuzzing of the
    // input string and simply check validation for each field.

    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    auto runnerFactories = []() {
        TRunnerFactoryUPtr outliers{std::make_unique<api::CDataFrameOutliersRunnerFactory>()};
        TRunnerFactoryUPtr regression{
            std::make_unique<api::CDataFrameRegressionRunnerFactory>()};
        TRunnerFactoryUPtr classification{
            std::make_unique<api::CDataFrameClassificationRunnerFactory>()};
        TRunnerFactoryUPtrVec factories;
        factories.push_back(std::move(outliers));
        factories.push_back(std::move(regression));
        factories.push_back(std::move(classification));
        return factories;
    };
    auto jsonSpec = [](const std::string& jobId, const std::string& rows,
                       const std::string& cols, const std::string& memory,
                       const std::string& threads, const std::string& resultsField,
                       const TStrVec& categoricalFields, const std::string& name,
                       const std::string& parameters = "", const std::string& junk = "") {
        std::ostringstream result;
        result << "{\n";
        if (jobId.size() > 0) {
            result << "   \"job_id\": \"" << jobId << "\",\n";
        }
        if (rows.size() > 0) {
            result << "   \"rows\": " << rows << ",\n";
        }
        if (cols.size() > 0) {
            result << "   \"cols\": " << cols << ",\n";
        }
        if (memory.size() > 0) {
            result << "   \"memory_limit\": " << memory << ",\n";
        }
        if (threads.size() > 0) {
            result << "   \"threads\": " << threads << ",\n";
        }
        if (resultsField.size() > 0) {
            result << "   \"results_field\": \"" << resultsField << "\",\n";
        }
        if (categoricalFields.size() > 0) {
            result << "   \"categorical_fields\": [";
            result << " \"" << categoricalFields[0] << "\"";
            for (std::size_t i = 1; i < categoricalFields.size(); ++i) {
                result << ", \"" << categoricalFields[i] << "\"";
            }
            result << " ],\n";
        }
        if (junk.size() > 0) {
            result << "   \"" << junk << "\": 4,\n";
        }
        result << "   \"analysis\": {\n";
        if (name.size() > 0) {
            result << "     \"name\": \"" << name << "\"";
        }
        if (parameters.size() > 0) {
            result << ",\n     \"parameters\": " << parameters << ",\n";
        } else {
            result << ",\n";
        }
        result << "     \"disk_usage_allowed\": true \n}\n}";
        return result.str();
    };

    LOG_DEBUG(<< "Valid input");
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "20", "100000", "2", "custom_ml",
                              {}, "outlier_detection"));
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "20", "100000", "2",
                                        "custom_ml", {}, "outlier_detection")};
        BOOST_CHECK_EQUAL(std::string{"foo"}, spec.jobId());
        BOOST_CHECK_EQUAL(std::size_t{1000}, spec.numberRows());
        BOOST_CHECK_EQUAL(std::size_t{20}, spec.numberColumns());
        BOOST_CHECK_EQUAL(std::size_t{100000}, spec.memoryLimit());
        BOOST_CHECK_EQUAL(std::size_t{2}, spec.numberThreads());
        BOOST_CHECK_EQUAL(std::string("custom_ml"), spec.resultsField());
        BOOST_TEST(spec.categoricalFieldNames().empty());
    }
    {
        LOG_TRACE(<< jsonSpec("bar", "1000", "20", "100000", "2", "custom_ml",
                              {"x", "y"}, "outlier_detection"));
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("bar", "1000", "20", "100000", "2", "custom_ml",
                                        {"x", "y"}, "outlier_detection")};
        BOOST_CHECK_EQUAL(std::string{"bar"}, spec.jobId());
        BOOST_CHECK_EQUAL(std::size_t{1000}, spec.numberRows());
        BOOST_CHECK_EQUAL(std::size_t{20}, spec.numberColumns());
        BOOST_CHECK_EQUAL(std::size_t{100000}, spec.memoryLimit());
        BOOST_CHECK_EQUAL(std::size_t{2}, spec.numberThreads());
        BOOST_CHECK_EQUAL(std::string("custom_ml"), spec.resultsField());
        BOOST_CHECK_EQUAL(std::string("[x, y]"),
                          core::CContainerPrinter::print(spec.categoricalFieldNames()));
    }

    LOG_DEBUG(<< "Bad input");
    {
        LOG_TRACE(<< jsonSpec("foo", "", "20", "100000", "2", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(),
            jsonSpec("foo", "", "20", "100000", "2", "ml", {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "", "100000", "2", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "", "100000", "2", "ml",
                                        {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "20", "", "2", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(),
            jsonSpec("foo", "1000", "20", "", "2", "ml", {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "20", "100000", "", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "20", "100000", "", "ml",
                                        {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "20", "100000", "2", "ml", {}, ""));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "20", "100000", "2", "ml", {}, "")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "-3", "20", "100000", "2", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "-3", "20", "100000", "2", "ml",
                                        {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "0", "100000", "2", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "0", "100000", "2", "ml",
                                        {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "20", "ZZ", "2", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "20", "\"ZZ\"", "2",
                                        "ml", {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "20", "100000", "-1", "ml", {}, "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "20", "100000", "-1",
                                        "ml", {}, "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("foo", "100", "20", "100000", "2", "ml", {}, "outl1ers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(),
            jsonSpec("foo", "100", "20", "100000", "2", "ml", {}, "outl1ers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        std::string jsonSpecStr{"{\n"
                                "  \"job_id\": \"foo\",\n"
                                "  \"rows\": 1000,\n"
                                "  \"cols\": 20,\n"
                                "  \"memory_limit\": 100000,\n"
                                "  \"threads\": 2,\n"
                                "  \"results_field\": \"ml\",\n"
                                "  \"categorical_fields\": [ 2, 1 ],\n"
                                "  \"analysis\": {\n"
                                "    \"name\": \"outlier_detection\"\n"
                                "  }\n"
                                "}"};
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{runnerFactories(), jsonSpecStr};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
    {
        std::string jsonSpecStr{"{\n"
                                "  \"job_id\": \"foo\",\n"
                                "  \"rows\": 1000,\n"
                                "  \"cols\": 20,\n"
                                "  \"memory_limit\": 100000,\n"
                                "  \"threads\": 2,\n"
                                "  \"results_field\": \"ml\",\n"
                                "  \"categorical_fields\": \"x\",\n"
                                "  \"analysis\": {\n"
                                "    \"name\": \"outlier_detection\"\n"
                                "  }\n"
                                "}"};
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{runnerFactories(), jsonSpecStr};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid number neighbours");
    {
        LOG_TRACE(<< jsonSpec("foo", "100", "20", "100000", "2", "ml", {},
                              "outlier_detection", "{\"n_neighbors\": -1}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "100", "20", "100000", "2", "ml", {},
                                        "outlier_detection", "{\"n_neighbors\": -1}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid method");
    {
        LOG_TRACE(<< jsonSpec("foo", "100", "20", "100000", "2", "ml", {},
                              "outlier_detection", "{\"method\": \"lofe\"}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "100", "20", "100000", "2", "ml", {},
                                        "outlier_detection", "{\"method\": \"lofe\"}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid feature influence");
    {
        LOG_TRACE(<< jsonSpec("foo", "100", "20", "100000", "2", "ml", {}, "outlier_detection",
                              "{\"compute_feature_influence\": 1}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(),
            jsonSpec("foo", "100", "20", "100000", "2", "ml", {},
                     "outlier_detection", "{\"compute_feature_influence\": 1}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid feature influence");
    {
        LOG_TRACE(<< jsonSpec("foo", "100", "20", "100000", "2", "ml", {}, "outlier_detection",
                              "{\"compute_feature_influences\": true}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "100", "20", "100000", "2", "ml", {}, "outlier_detection",
                                        "{\"compute_feature_influences\": true}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }

    LOG_DEBUG(<< "Extra junk");
    {
        LOG_TRACE(<< jsonSpec("foo", "1000", "2", "100000", "2", "ml", {},
                              "outlier_detection", "", "threeds"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            runnerFactories(), jsonSpec("foo", "1000", "2", "100000", "2", "ml",
                                        {}, "outlier_detection", "", "threeds")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        BOOST_TEST(errors.size() > 0);
    }
}

BOOST_AUTO_TEST_CASE(testRunAnalysis) {

    // Check progress is monotonic and that it remains less than one until the end
    // of the analysis.

    auto testFactory = []() {
        TRunnerFactoryUPtr factory{std::make_unique<CDataFrameMockAnalysisRunnerFactory>()};
        TRunnerFactoryUPtrVec factories;
        factories.push_back(std::move(factory));
        return factories;
    };

    std::string jsonSpec = api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", 100, 10, 1000, 1, {}, true, test::CTestTmpDir::tmpDir(), "", "test", "");

    for (std::size_t i = 0; i < 10; ++i) {
        api::CDataFrameAnalysisSpecification spec{testFactory(), jsonSpec};

        auto frameAndDirectory = core::makeMainStorageDataFrame(10);
        auto frame = std::move(frameAndDirectory.first);

        api::CDataFrameAnalysisRunner* runner{spec.run(*frame)};
        BOOST_TEST(runner != nullptr);

        double lastProgress{runner->progress()};
        while (runner->finished() == false) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            LOG_TRACE(<< "progress = " << lastProgress);
            BOOST_TEST(runner->progress() >= lastProgress);
            lastProgress = runner->progress();
            BOOST_TEST(runner->progress() <= 1.0);
        }

        LOG_DEBUG(<< "progress = " << lastProgress);
        BOOST_CHECK_EQUAL(1.0, runner->progress());
    }
}

std::string
CDataFrameAnalysisSpecificationTest::createSpecJsonForTempDirDiskUsageTest(bool tempDirPathSet,
                                                                           bool diskUsageAllowed) {
    std::string tempDir = tempDirPathSet ? test::CTestTmpDir::tmpDir() : "";
    return api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", 100, 3, 500000, 1, {}, diskUsageAllowed, tempDir, "",
        "outlier_detection", "");
}

BOOST_AUTO_TEST_CASE(testTempDirDiskUsage) {

    std::vector<std::string> errors;
    std::mutex errorsMutex;
    auto errorHandler = [&errors, &errorsMutex](std::string error) {
        std::lock_guard<std::mutex> lock{errorsMutex};
        errors.push_back(error);
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    // No temp dir given, disk usage allowed
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForTempDirDiskUsageTest(false, true)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // single error is registered that temp dir is empty
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(1), errors.size());
        BOOST_TEST(errors[0].find("Input error: temporary directory path should"
                                  " be explicitly set if disk usage is allowed!") !=
                   std::string::npos);
    }

    // No temp dir given, no disk usage allowed
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForTempDirDiskUsageTest(false, false)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(0), errors.size());
    }

    // Temp dir given and disk usage allowed
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForTempDirDiskUsageTest(true, true)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(0), errors.size());
    }
}

BOOST_AUTO_TEST_SUITE_END()
