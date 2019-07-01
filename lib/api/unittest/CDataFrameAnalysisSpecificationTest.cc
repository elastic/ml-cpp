/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalysisSpecificationTest.h"

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CStopWatch.h>

#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameOutliersRunner.h>

#include <test/CRandomNumbers.h>

#include "CDataFrameMockAnalysisRunner.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace ml;

namespace {
using TStrVec = std::vector<std::string>;
using TRunnerFactoryUPtr = std::unique_ptr<api::CDataFrameAnalysisRunnerFactory>;
using TRunnerFactoryUPtrVec = std::vector<TRunnerFactoryUPtr>;
}

void CDataFrameAnalysisSpecificationTest::testCreate() {
    // This test focuses on checking the validation code we apply to the object
    // rather than the JSON parsing so we don't bother with random fuzzing of the
    // input string and simply check validation for each field.

    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    auto outliersFactory = []() {
        TRunnerFactoryUPtr factory{std::make_unique<api::CDataFrameOutliersRunnerFactory>()};
        TRunnerFactoryUPtrVec factories;
        factories.push_back(std::move(factory));
        return factories;
    };
    auto jsonSpec = [](const std::string& rows, const std::string& cols,
                       const std::string& memory, const std::string& threads,
                       const std::string& resultsField, const std::string& name,
                       const std::string& parameters = "", const std::string& junk = "") {
        std::ostringstream result;
        result << "{\n";
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
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "2", "custom_ml", "outlier_detection"));
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(),
            jsonSpec("1000", "20", "100000", "2", "custom_ml", "outlier_detection")};
        CPPUNIT_ASSERT_EQUAL(std::size_t{1000}, spec.numberRows());
        CPPUNIT_ASSERT_EQUAL(std::size_t{20}, spec.numberColumns());
        CPPUNIT_ASSERT_EQUAL(std::size_t{100000}, spec.memoryLimit());
        CPPUNIT_ASSERT_EQUAL(std::size_t{2}, spec.numberThreads());
        CPPUNIT_ASSERT_EQUAL(std::string("custom_ml"), spec.resultsField());
    }
    LOG_DEBUG(<< "Bad input");
    {
        LOG_TRACE(<< jsonSpec("", "20", "100000", "2", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("", "20", "100000", "2", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "", "100000", "2", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "", "100000", "2", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "", "2", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "", "2", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(),
            jsonSpec("1000", "20", "100000", "", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "2", "ml", ""));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "100000", "2", "ml", "")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("-3", "20", "100000", "2", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("-3", "20", "100000", "2", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "0", "100000", "2", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(),
            jsonSpec("1000", "0", "100000", "2", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "ZZ", "2", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(),
            jsonSpec("1000", "20", "\"ZZ\"", "2", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "-1", "ml", "outlier_detection"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(),
            jsonSpec("1000", "20", "100000", "-1", "ml", "outlier_detection")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "ml", "outl1ers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "ml", "outl1ers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid number neighbours");
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "ml",
                              "outlier_detection", "{\"n_neighbors\": -1}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "ml",
                                        "outlier_detection", "{\"n_neighbors\": -1}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid method");
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "ml",
                              "outlier_detection", "{\"method\": \"lofe\"}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "ml",
                                        "outlier_detection", "{\"method\": \"lofe\"}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid feature influence");
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "ml", "outlier_detection",
                              "{\"compute_feature_influence\": 1}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "ml", "outlier_detection",
                                        "{\"compute_feature_influence\": 1}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid feature influence");
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "ml", "outlier_detection",
                              "{\"compute_feature_influences\": true}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "ml", "outlier_detection",
                                        "{\"compute_feature_influences\": true}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Extra junk");
    {
        LOG_TRACE(<< jsonSpec("1000", "2", "100000", "2", "ml",
                              "outlier_detection", "", "threeds"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "2", "100000", "2", "ml",
                                        "outlier_detection", "", "threeds")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
}

void CDataFrameAnalysisSpecificationTest::testRunAnalysis() {
    // Check progress is monotonic and that it remains less than one until the end
    // of the analysis.

    auto testFactory = []() {
        TRunnerFactoryUPtr factory{std::make_unique<CDataFrameMockAnalysisRunnerFactory>()};
        TRunnerFactoryUPtrVec factories;
        factories.push_back(std::move(factory));
        return factories;
    };

    std::string jsonSpec{"{\n"
                         "  \"rows\": 100,\n"
                         "  \"cols\": 10,\n"
                         "  \"memory_limit\": 1000,\n"
                         "  \"threads\": 1,\n"
                         "  \"disk_usage_allowed\": true,\n"
                         "  \"analysis\": {\n"
                         "    \"name\": \"test\""
                         "  }"
                         "}"};

    for (std::size_t i = 0; i < 10; ++i) {
        api::CDataFrameAnalysisSpecification spec{testFactory(), jsonSpec};

        auto frameAndDirectory = core::makeMainStorageDataFrame(10);
        auto frame = std::move(frameAndDirectory.first);

        api::CDataFrameAnalysisRunner* runner{spec.run(*frame)};
        CPPUNIT_ASSERT(runner != nullptr);

        double lastProgress{runner->progress()};
        while (runner->finished() == false) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            LOG_TRACE(<< "progress = " << lastProgress);
            CPPUNIT_ASSERT(runner->progress() >= lastProgress);
            lastProgress = runner->progress();
            CPPUNIT_ASSERT(runner->progress() <= 1.0);
        }

        LOG_DEBUG(<< "progress = " << lastProgress);
        CPPUNIT_ASSERT_EQUAL(1.0, runner->progress());
    }
}

std::string
CDataFrameAnalysisSpecificationTest::createSpecJsonForTempDirDiskUsageTest(bool tempDirPathSet,
                                                                           bool diskUsageAllowed) {

    std::string tempDirParameter = tempDirPathSet ? "  \"temp_dir\": \"/tmp\",\n" : "";
    std::string diskUsageParameter = diskUsageAllowed ? "true" : "false";
    std::string jsonSpec{"{\n"
                         "  \"rows\": 100,\n"
                         "  \"cols\": 3,\n"
                         "  \"memory_limit\": 500000,\n" +
                         tempDirParameter + "  \"disk_usage_allowed\": " + diskUsageParameter +
                         ",\n"
                         "  \"threads\": 1,\n"
                         "  \"analysis\": {\n"
                         "    \"name\": \"outlier_detection\""
                         "  }"
                         "}"};
    return jsonSpec;
}

void CDataFrameAnalysisSpecificationTest::testTempDirDiskUsage() {

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
        CPPUNIT_ASSERT_EQUAL(1, static_cast<int>(errors.size()));
        CPPUNIT_ASSERT(errors[0].find("Temporary directory path should be explicitly set") !=
                       std::string::npos);
    }

    // No temp dir given, no disk usage allowed
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForTempDirDiskUsageTest(false, false)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }

    // Temp dir given and disk usage allowed
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForTempDirDiskUsageTest(true, true)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }
}

CppUnit::Test* CDataFrameAnalysisSpecificationTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CDataFrameAnalysisSpecificationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisSpecificationTest>(
        "CDataFrameAnalysisSpecificationTest::testCreate",
        &CDataFrameAnalysisSpecificationTest::testCreate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisSpecificationTest>(
        "CDataFrameAnalysisSpecificationTest::testRunAnalysis",
        &CDataFrameAnalysisSpecificationTest::testRunAnalysis));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisSpecificationTest>(
        "CDataFrameAnalysisSpecificationTest::testTempDirDiskUsage",
        &CDataFrameAnalysisSpecificationTest::testTempDirDiskUsage));

    return suiteOfTests;
}
