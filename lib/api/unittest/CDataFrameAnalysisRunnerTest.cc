/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalysisRunnerTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRegex.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CMemoryUsageEstimationResultJsonWriter.h>

#include <test/CTestTmpDir.h>

#include <mutex>
#include <string>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <vector>

using namespace ml;

using TBoolVec = std::vector<bool>;
using TStrVec = std::vector<std::string>;

void CDataFrameAnalysisRunnerTest::testComputeExecutionStrategyForOutliers() {

    using TSizeVec = std::vector<std::size_t>;

    TSizeVec numbersRows{100, 100000, 1000000};
    TSizeVec numbersCols{3, 10, 50};

    for (auto numberRows : numbersRows) {
        for (auto numberCols : numbersCols) {
            LOG_DEBUG(<< "# rows = " << numberRows << ", # cols = " << numberCols);

            auto spec{test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
                numberRows, numberCols, 100000000, "", 0, true)};
            api::CDataFrameOutliersRunnerFactory factory;
            auto runner = factory.make(*spec);

            LOG_DEBUG(<< "  Use main memory = " << runner->storeDataFrameInMainMemory());
            LOG_DEBUG(<< "  # partitions = " << runner->numberPartitions());
            LOG_DEBUG(<< "  # rows per partition = "
                      << runner->maximumNumberRowsPerPartition());

            // Check some invariants:
            //   1. strategy is in main memory iff the number of partitions is one,
            //   2. number partitions x maximum number rows >= number rows,
            //   3. (number partitions - 1) x maximum number rows <= number rows.

            bool inMainMemory{runner->storeDataFrameInMainMemory()};
            std::size_t numberPartitions{runner->numberPartitions()};
            std::size_t maxRowsPerPartition{runner->maximumNumberRowsPerPartition()};

            CPPUNIT_ASSERT_EQUAL(numberPartitions == 1, inMainMemory);
            CPPUNIT_ASSERT(numberPartitions * maxRowsPerPartition >= numberRows);
            CPPUNIT_ASSERT((numberPartitions - 1) * maxRowsPerPartition <= numberRows);
        }
    }

    // TODO test running memory is in acceptable range.
}

void CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag() {

    TStrVec errors;
    std::mutex errorsMutex;
    auto errorHandler = [&errors, &errorsMutex](std::string error) {
        std::lock_guard<std::mutex> lock{errorsMutex};
        errors.push_back(error);
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};
    api::CDataFrameOutliersRunnerFactory factory;

    // Test large memory requirement without disk usage
    {
        errors.clear();
        auto spec = test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
            1000, 100, 500000, "", 0, true, false);

        // single error is registered that the memory limit is to low
        LOG_DEBUG(<< "errors = " << core::CContainerPrinter::print(errors));
        core::CRegex re;
        re.init("Input error: memory limit.*");
        CPPUNIT_ASSERT_EQUAL(1, static_cast<int>(errors.size()));
        CPPUNIT_ASSERT(re.matches(errors[0]));
    }

    // Test large memory requirement with disk usage
    {
        errors.clear();
        auto spec = test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
            1000, 100, 500000, "", 0, true, true);

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }

    // Test low memory requirement without disk usage
    {
        errors.clear();
        auto spec = test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
            10, 10, 500000, "", 0, true, false);

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }
}

void testEstimateMemoryUsage(std::int64_t numberRows,
                             const std::string& expectedExpectedMemoryWithoutDisk,
                             const std::string& expectedExpectedMemoryWithDisk,
                             int expectedNumberErrors) {

    std::ostringstream sstream;
    TStrVec errors;
    std::mutex errorsMutex;
    auto errorHandler = [&errors, &errorsMutex](std::string error) {
        std::lock_guard<std::mutex> lock{errorsMutex};
        errors.push_back(error);
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    // The output writer won't close the JSON structures until is is destroyed
    {
        auto spec{test::CDataFrameAnalysisSpecificationFactory::outlierSpec(
            numberRows, 5, 100000000, "", 0, true)};

        core::CJsonOutputStreamWrapper wrappedOutStream(sstream);
        api::CMemoryUsageEstimationResultJsonWriter writer(wrappedOutStream);

        spec->estimateMemoryUsage(writer);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& result{arrayDoc[rapidjson::SizeType(0)]};
    CPPUNIT_ASSERT(result.IsObject());

    CPPUNIT_ASSERT(result.HasMember("expected_memory_without_disk"));
    CPPUNIT_ASSERT_EQUAL(expectedExpectedMemoryWithoutDisk,
                         std::string(result["expected_memory_without_disk"].GetString()));
    CPPUNIT_ASSERT(result.HasMember("expected_memory_with_disk"));
    CPPUNIT_ASSERT_EQUAL(expectedExpectedMemoryWithDisk,
                         std::string(result["expected_memory_with_disk"].GetString()));

    CPPUNIT_ASSERT_EQUAL(expectedNumberErrors, static_cast<int>(errors.size()));
}

void CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor0Rows() {
    testEstimateMemoryUsage(0, "0", "0", 1);
}

void CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor1Row() {
    testEstimateMemoryUsage(1, "6kB", "6kB", 0);
}

void CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor10Rows() {
    testEstimateMemoryUsage(10, "15kB", "13kB", 0);
}

void CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor100Rows() {
    testEstimateMemoryUsage(100, "62kB", "35kB", 0);
}

void CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor1000Rows() {
    testEstimateMemoryUsage(1000, "450kB", "143kB", 0);
}

void testColumnsForWhichEmptyIsMissing(const std::string& analysis,
                                       const std::string& dependentVariableName,
                                       const TStrVec& fieldNames,
                                       const TStrVec& categoricalFields,
                                       const TBoolVec& expectedEmptyIsMissing) {
    std::string parameters{"{\"dependent_variable\": \"" + dependentVariableName + "\"}"};
    std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", 10000, 5, 100000000, 1, categoricalFields, true,
        test::CTestTmpDir::tmpDir(), "", analysis, parameters)};
    api::CDataFrameAnalysisSpecification spec{jsonSpec};
    auto emptyIsMissing = spec.columnsForWhichEmptyIsMissing(fieldNames);
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedEmptyIsMissing),
                         core::CContainerPrinter::print(emptyIsMissing));
}

void CDataFrameAnalysisRunnerTest::testColumnsForWhichEmptyIsMissingClassification() {
    testColumnsForWhichEmptyIsMissing("classification", "class",
                                      {"feature_1", "feature_2", "feature_3", "class"},
                                      {"class"}, {false, false, false, true});
}

void CDataFrameAnalysisRunnerTest::testColumnsForWhichEmptyIsMissingRegression() {
    testColumnsForWhichEmptyIsMissing("regression", "value",
                                      {"feature_1", "feature_2", "feature_3", "value"},
                                      {}, {false, false, false, false});
}

CppUnit::Test* CDataFrameAnalysisRunnerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameAnalysisRunnerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testComputeExecutionStrategyForOutliers",
        &CDataFrameAnalysisRunnerTest::testComputeExecutionStrategyForOutliers));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag",
        &CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor0Rows",
        &CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor0Rows));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor1Row",
        &CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor1Row));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor10Rows",
        &CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor10Rows));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor100Rows",
        &CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor100Rows));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor1000Rows",
        &CDataFrameAnalysisRunnerTest::testEstimateMemoryUsageFor1000Rows));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testColumnsForWhichEmptyIsMissingClassification",
        &CDataFrameAnalysisRunnerTest::testColumnsForWhichEmptyIsMissingClassification));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testColumnsForWhichEmptyIsMissingRegression",
        &CDataFrameAnalysisRunnerTest::testColumnsForWhichEmptyIsMissingRegression));

    return suiteOfTests;
}
