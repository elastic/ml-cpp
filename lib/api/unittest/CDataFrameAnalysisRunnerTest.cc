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

#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>

#include <mutex>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisRunnerTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testComputeExecutionStrategyForOutliers) {
    using TSizeVec = std::vector<std::size_t>;

    TSizeVec numbersRows{100, 100000, 1000000};
    TSizeVec numbersCols{3, 10, 50};

    for (auto numberRows : numbersRows) {
        for (auto numberCols : numbersCols) {
            LOG_DEBUG(<< "# rows = " << numberRows << ", # cols = " << numberCols);

            // Give the process approximately 100MB.
            std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
                "testJob", numberRows, numberCols, 100000000, 1, {}, true,
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

            BOOST_CHECK_EQUAL(numberPartitions == 1, inMainMemory);
            BOOST_TEST(numberPartitions * maxRowsPerPartition >= numberRows);
            BOOST_TEST((numberPartitions - 1) * maxRowsPerPartition <= numberRows);
        }
    }

    // TODO test running memory is in acceptable range.
}

std::string
CDataFrameAnalysisRunnerTest::createSpecJsonForDiskUsageTest(std::size_t numberRows,
                                                             std::size_t numberCols,
                                                             bool diskUsageAllowed) {
    return api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", numberRows, numberCols, 500000, 1, {}, diskUsageAllowed,
        test::CTestTmpDir::tmpDir(), "", "outlier_detection", "");
}

BOOST_AUTO_TEST_CASE(testComputeAndSaveExecutionStrategyDiskUsageFlag) {

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
        BOOST_CHECK_EQUAL(1, static_cast<int>(errors.size()));
        BOOST_TEST(re.matches(errors[0]));
    }

    // Test large memory requirement with disk usage
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForDiskUsageTest(1000, 100, true)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        BOOST_CHECK_EQUAL(0, static_cast<int>(errors.size()));
    }

    // Test low memory requirement without disk usage
    {
        errors.clear();
        std::string jsonSpec{createSpecJsonForDiskUsageTest(10, 10, false)};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        // no error should be registered
        BOOST_CHECK_EQUAL(0, static_cast<int>(errors.size()));
    }
}

void testEstimateMemoryUsage(int64_t numberRows,
                             const std::string& expected_expected_memory_without_disk,
                             const std::string& expected_expected_memory_with_disk,
                             int expected_number_errors) {

    std::ostringstream sstream;
    std::vector<std::string> errors;
    std::mutex errorsMutex;
    auto errorHandler = [&errors, &errorsMutex](std::string error) {
        std::lock_guard<std::mutex> lock{errorsMutex};
        errors.push_back(error);
    };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    // The output writer won't close the JSON structures until is is destroyed
    {
        std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
            "testJob", numberRows, 5, 100000000, 1, {}, true,
            test::CTestTmpDir::tmpDir(), "", "outlier_detection", "")};
        api::CDataFrameAnalysisSpecification spec{jsonSpec};

        core::CJsonOutputStreamWrapper wrappedOutStream(sstream);
        api::CMemoryUsageEstimationResultJsonWriter writer(wrappedOutStream);

        spec.estimateMemoryUsage(writer);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    BOOST_TEST(arrayDoc.IsArray());
    BOOST_CHECK_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& result = arrayDoc[rapidjson::SizeType(0)];
    BOOST_TEST(result.IsObject());

    BOOST_TEST(result.HasMember("expected_memory_without_disk"));
    BOOST_CHECK_EQUAL(expected_expected_memory_without_disk,
                      std::string(result["expected_memory_without_disk"].GetString()));
    BOOST_TEST(result.HasMember("expected_memory_with_disk"));
    BOOST_CHECK_EQUAL(expected_expected_memory_with_disk,
                      std::string(result["expected_memory_with_disk"].GetString()));

    BOOST_CHECK_EQUAL(expected_number_errors, static_cast<int>(errors.size()));
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsage_0) {
    testEstimateMemoryUsage(0, "0", "0", 1);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsage_1) {
    testEstimateMemoryUsage(1, "6kB", "6kB", 0);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsage_10) {
    testEstimateMemoryUsage(10, "15kB", "13kB", 0);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsage_100) {
    testEstimateMemoryUsage(100, "62kB", "35kB", 0);
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsage_1000) {
    testEstimateMemoryUsage(1000, "450kB", "143kB", 0);
}

void testColumnsForWhichEmptyIsMissing(const std::string& analysis,
                                       bool expected_dependentVariableEmptyAsMissing) {
    using TBoolVec = std::vector<bool>;
    using TStrVec = std::vector<std::string>;

    std::string parameters{"{\"dependent_variable\": \"label\"}"};
    std::string jsonSpec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", 10000, 5, 100000000, 1, {}, true,
        test::CTestTmpDir::tmpDir(), "", analysis, parameters)};
    api::CDataFrameAnalysisSpecification spec{jsonSpec};

    TStrVec fieldNames{"feature_1", "feature_2", "feature_3", "label"};
    TBoolVec emptyAsMissing{spec.columnsForWhichEmptyIsMissing(fieldNames)};

    BOOST_CHECK_EQUAL(fieldNames.size(), emptyAsMissing.size());
    BOOST_CHECK_EQUAL(false, bool(emptyAsMissing[0]));
    BOOST_CHECK_EQUAL(false, bool(emptyAsMissing[1]));
    BOOST_CHECK_EQUAL(false, bool(emptyAsMissing[2]));
    BOOST_CHECK_EQUAL(expected_dependentVariableEmptyAsMissing, bool(emptyAsMissing[3]));
}

BOOST_AUTO_TEST_CASE(testColumnsForWhichEmptyIsMissingClassification) {
    testColumnsForWhichEmptyIsMissing("classification", true);
}

BOOST_AUTO_TEST_CASE(testColumnsForWhichEmptyIsMissingRegression) {
    testColumnsForWhichEmptyIsMissing("regression", false);
}

BOOST_AUTO_TEST_SUITE_END()
