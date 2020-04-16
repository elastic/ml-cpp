/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRegex.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CMemoryUsageEstimationResultJsonWriter.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>

#include <mutex>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisRunnerTest)

using namespace ml;

using TBoolVec = std::vector<bool>;
using TStrVec = std::vector<std::string>;

BOOST_AUTO_TEST_CASE(testComputeExecutionStrategyForOutliers) {

    using TSizeVec = std::vector<std::size_t>;

    TSizeVec numbersRows{100, 100000, 1000000};
    TSizeVec numbersCols{3, 10, 50};

    for (auto numberRows : numbersRows) {
        for (auto numberCols : numbersCols) {
            LOG_DEBUG(<< "# rows = " << numberRows << ", # cols = " << numberCols);

            test::CDataFrameAnalysisSpecificationFactory specFactory;
            auto spec = specFactory.rows(numberRows)
                            .columns(numberCols)
                            .memoryLimit(100000000)
                            .outlierComputeInfluence(true)
                            .outlierSpec();
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

            BOOST_REQUIRE_EQUAL(numberPartitions == 1, inMainMemory);
            BOOST_TEST_REQUIRE(numberPartitions * maxRowsPerPartition >= numberRows);
            BOOST_TEST_REQUIRE((numberPartitions - 1) * maxRowsPerPartition <= numberRows);
        }
    }

    // TODO test running memory is in acceptable range.
}

BOOST_AUTO_TEST_CASE(testComputeAndSaveExecutionStrategyDiskUsageFlag) {

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
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        auto spec = specFactory.rows(1000)
                        .columns(100)
                        .memoryLimit(500000)
                        .outlierComputeInfluence(true)
                        .diskUsageAllowed(false)
                        .outlierSpec();

        // single error is registered that the memory limit is to low
        LOG_DEBUG(<< "errors = " << core::CContainerPrinter::print(errors));
        core::CRegex re;
        re.init("Input error: memory limit.*");
        BOOST_REQUIRE_EQUAL(1, static_cast<int>(errors.size()));
        BOOST_TEST_REQUIRE(re.matches(errors[0]));
    }

    // Test large memory requirement with disk usage
    {
        errors.clear();
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        auto spec = specFactory.rows(1000)
                        .columns(100)
                        .memoryLimit(500000)
                        .outlierComputeInfluence(true)
                        .diskUsageAllowed(true)
                        .outlierSpec();

        // no error should be registered
        BOOST_REQUIRE_EQUAL(0, static_cast<int>(errors.size()));
    }

    // Test low memory requirement without disk usage
    {
        errors.clear();
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        auto spec = specFactory.rows(10)
                        .columns(10)
                        .memoryLimit(500000)
                        .outlierComputeInfluence(true)
                        .diskUsageAllowed(false)
                        .outlierSpec();

        // no error should be registered
        BOOST_REQUIRE_EQUAL(0, static_cast<int>(errors.size()));
    }
}

namespace {
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

    // The output writer won't close the JSON structures until is is destroyed.
    {
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        auto spec = specFactory.rows(numberRows)
                        .memoryLimit(100000000)
                        .outlierComputeInfluence(true)
                        .outlierSpec();

        core::CJsonOutputStreamWrapper wrappedOutStream(sstream);
        api::CMemoryUsageEstimationResultJsonWriter writer(wrappedOutStream);

        spec->estimateMemoryUsage(writer);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    BOOST_TEST_REQUIRE(arrayDoc.IsArray());
    BOOST_REQUIRE_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& result{arrayDoc[rapidjson::SizeType(0)]};
    BOOST_TEST_REQUIRE(result.IsObject());

    BOOST_TEST_REQUIRE(result.HasMember("expected_memory_without_disk"));
    BOOST_REQUIRE_EQUAL(expectedExpectedMemoryWithoutDisk,
                        result["expected_memory_without_disk"].GetString());
    BOOST_TEST_REQUIRE(result.HasMember("expected_memory_with_disk"));
    BOOST_REQUIRE_EQUAL(expectedExpectedMemoryWithDisk,
                        result["expected_memory_with_disk"].GetString());

    BOOST_REQUIRE_EQUAL(expectedNumberErrors, static_cast<int>(errors.size()));
}
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsageFor0Rows) {
    testEstimateMemoryUsage(0, "0", "0", 1);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsageFor1Row) {
    testEstimateMemoryUsage(1, "1mb", "1mb", 0);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsageFor10000Rows) {
    testEstimateMemoryUsage(10000, "5mb", "2mb", 0);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsageFor100000Rows) {
    testEstimateMemoryUsage(100000, "41mb", "9mb", 0);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsageFor10000000Rows) {
    testEstimateMemoryUsage(10000000, "4511mb", "88mb", 0);
}

BOOST_AUTO_TEST_SUITE_END()
