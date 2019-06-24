/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalysisRunnerTest.h"

#include <core/CLogger.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameOutliersRunner.h>

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

            std::string jsonSpec{"{\n"
                                 "  \"rows\": " +
                                 std::to_string(numberRows) +
                                 ",\n"
                                 "  \"cols\": " +
                                 std::to_string(numberCols) +
                                 ",\n"
                                 "  \"memory_limit\": 100000000,\n"
								 "  \"disk_usage_allowed\": true,\n"
                                 "  \"threads\": 1,\n"
                                 "  \"analysis\": {\n"
                                 "    \"name\": \"outlier_detection\""
                                 "  }"
                                 "}"};

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

std::string CDataFrameAnalysisRunnerTest::createSpecJsonForDiskUsageTest(
		std::size_t numberRows, std::size_t numberCols, bool diskUsageAllowed) {
	std::string jsonSpec{"{\n"
							 "  \"rows\": " +
							 std::to_string(numberRows) +
							 ",\n"
							 "  \"cols\": " +
							 std::to_string(numberCols) +
							 ",\n"
							 "  \"memory_limit\": 100,\n"
							 "  \"disk_usage_allowed\": " +
							 (diskUsageAllowed ? "true" : "false") +
							 ",\n"
							 "  \"threads\": 1,\n"
							 "}"};
	return jsonSpec;
}

void CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag() {

	std::vector<std::string> errors;
	std::mutex errorsMutex;
	auto errorHandler = [&errors, &errorsMutex](std::string error) {
		std::lock_guard<std::mutex> lock {errorsMutex};
		errors.push_back(error);
	};

	core::CLogger::CScopeSetFatalErrorHandler scope { errorHandler };

	// Test with bad analysis specification.
	{
		errors.clear();
		api::CDataFrameAnalyzer analyzer { std::make_unique<
				api::CDataFrameAnalysisSpecification>(std::string { "junk" }),
				outputWriterFactory };
		LOG_DEBUG(<< core::CContainerPrinter::print(errors));
		CPPUNIT_ASSERT(errors.size() > 0);
		CPPUNIT_ASSERT_EQUAL(false, analyzer.handleRecord( { "c1", "c2", "c3",
				"c4", "c5" }, { "10", "10", "10", "10", "10" }));
	}

	std::string jsonSpec { createSpecJsonForDiskUsageTest(1000000, 100, false) };
	try {
		new api::CDataFrameAnalysisSpecification(jsonSpec);
	} catch (const std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
//	CPPUNIT_ASSERT_THROW(new api::CDataFrameAnalysisSpecification(jsonSpec), CppUnit::Exception);

}

CppUnit::Test* CDataFrameAnalysisRunnerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameAnalysisRunnerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
        "CDataFrameAnalysisRunnerTest::testComputeExecutionStrategyForOutliers",
        &CDataFrameAnalysisRunnerTest::testComputeExecutionStrategyForOutliers));

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisRunnerTest>(
		"CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag",
		&CDataFrameAnalysisRunnerTest::testComputeAndSaveExecutionStrategyDiskUsageFlag));

    return suiteOfTests;
}


