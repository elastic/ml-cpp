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
#include <vector>

using namespace ml;

void CDataFrameAnalysisRunnerTest::testComputeExecutionStrategyForOutliers() {
    using TSizeVec = std::vector<std::size_t>;

    TSizeVec numbersRows{100, 100000, 1000000};
    TSizeVec numbersCols{3, 10, 50};

    for (auto numberRows : numbersRows) {
        for (auto numberCols : numbersCols) {
            LOG_DEBUG(<< "# rows = " << numberRows << ", # cols = " << numberCols);

            // Give the process approximately 100MB.
            std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
                numberRows, numberCols, 100000000, 1, {}, true,
                test::CTestTmpDir::tmpDir(), "", "outlier_detection", "")};

            api::CDataFrameAnalysisSpecification spec{jsonSpec};

            api::CDataFrameOutliersRunnerFactory factory;
            auto runner = factory.make(spec);

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

std::string
CDataFrameAnalysisRunnerTest::createSpecJsonForDiskUsageTest(std::size_t numberRows,
                                                             std::size_t numberCols,
                                                             bool diskUsageAllowed) {
    return api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        numberRows, numberCols, 500000, 1, {}, diskUsageAllowed,
        test::CTestTmpDir::tmpDir(), "", "outlier_detection", "");
}

void CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag() {

    std::vector<std::string> errors;
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
        std::string jsonSpec{createSpecJsonForDiskUsageTest(1000, 100, false)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

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
        std::string jsonSpec{createSpecJsonForDiskUsageTest(1000, 100, true)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }

    // Test low memory requirement without disk usage
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForDiskUsageTest(10, 10, false)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }
}

void CDataFrameAnalysisRunnerTest::testEstimateMemoryUsage() {

    std::vector<std::string> errors;
    std::mutex errorsMutex;
    auto errorHandler = [&errors, &errorsMutex](std::string error) {
        std::lock_guard<std::mutex> lock{errorsMutex};
        errors.push_back(error);
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};
    api::CDataFrameOutliersRunnerFactory factory;

    // Test estimation for empty data frame
    {
        errors.clear();
        std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
            0, 5, 100000000, 1, {}, true, test::CTestTmpDir::tmpDir(), "", "outlier_detection", "")};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // Check memory estimation result
        api::SMemoryUsageEstimationResult result = spec.estimateMemoryUsage();
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(result.s_MemoryUsageWithOnePartition));
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(result.s_MemoryUsageWithMaxPartitions));

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(1, static_cast<int>(errors.size()));
    }

    // Test estimation for data frame with 1 row
    {
        errors.clear();
        std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
            1, 5, 100000000, 1, {}, true, test::CTestTmpDir::tmpDir(), "", "outlier_detection", "")};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // Check memory estimation result
        api::SMemoryUsageEstimationResult result = spec.estimateMemoryUsage();
        CPPUNIT_ASSERT_EQUAL(6050, static_cast<int>(result.s_MemoryUsageWithOnePartition));
        CPPUNIT_ASSERT_EQUAL(6050, static_cast<int>(result.s_MemoryUsageWithMaxPartitions));

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }

    // Test estimation for data frame with 4 rows
    {
        errors.clear();
        std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
            4, 5, 100000000, 1, {}, true, test::CTestTmpDir::tmpDir(), "", "outlier_detection", "")};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // Check memory estimation result
        api::SMemoryUsageEstimationResult result = spec.estimateMemoryUsage();
        CPPUNIT_ASSERT_EQUAL(9104, static_cast<int>(result.s_MemoryUsageWithOnePartition));
        CPPUNIT_ASSERT_EQUAL(8528, static_cast<int>(result.s_MemoryUsageWithMaxPartitions));

        // no error should be registered
        CPPUNIT_ASSERT_EQUAL(0, static_cast<int>(errors.size()));
    }
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
        "CDataFrameAnalysisRunnerTest::testEstimateMemoryUsage",
        &CDataFrameAnalysisRunnerTest::testEstimateMemoryUsage));

    return suiteOfTests;
}
