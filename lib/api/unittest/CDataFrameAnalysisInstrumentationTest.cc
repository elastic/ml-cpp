/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CTimeUtils.h>

#include <api/CDataFrameAnalysisInstrumentation.h>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisInstrumentationTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testMemoryState) {
    std::string jobId("JOB123");
    std::int64_t memoryUsage = 1000;
    std::int64_t timestamp = core::CTimeUtils::toEpochMs(core::CTimeUtils::now());
    std::stringstream s_Output;
    {
        core::CJsonOutputStreamWrapper streamWrapper(s_Output);
        core::CRapidJsonConcurrentLineWriter writer(streamWrapper);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
        instrumentation.updateMemoryUsage(memoryUsage);
        instrumentation.writer(&writer);
        instrumentation.nextStep(0);
        s_Output.flush();
    }

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(s_Output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);

    const auto& result{results[0]};
    BOOST_TEST_REQUIRE(result["job_id"].GetString() == jobId);
    BOOST_TEST_REQUIRE(result["type"].GetString() == "analytics_memory_usage");
    BOOST_TEST_REQUIRE(result["peak_usage_bytes"].GetInt64() == memoryUsage);
    BOOST_REQUIRE_SMALL(result["timestamp"].GetInt64() - timestamp, 10l);
}

BOOST_AUTO_TEST_SUITE_END()
