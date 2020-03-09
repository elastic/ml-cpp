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
    std::string jobId{"JOB123"};
    std::int64_t memoryUsage{1000};
    std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    std::stringstream outputStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper(outputStream);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
        api::CDataFrameTrainBoostedTreeInstrumentation::CScopeSetOutputStream setStream{
            instrumentation, streamWrapper};
        instrumentation.updateMemoryUsage(memoryUsage);
        instrumentation.nextStep(0);
        outputStream.flush();
    }
    std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(outputStream.str()));

    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);
    bool hasMemoryUsage{false};
    for (auto i = results.Begin(); i != results.End(); ++i) {
        if (i->HasMember("analytics_memory_usage")) {
            BOOST_TEST_REQUIRE((*i)["analytics_memory_usage"].IsObject() == true);
            BOOST_TEST_REQUIRE((*i)["analytics_memory_usage"]["job_id"].GetString() == jobId);
            BOOST_TEST_REQUIRE(
                (*i)["analytics_memory_usage"]["peak_usage_bytes"].GetInt64() == memoryUsage);
            BOOST_TEST_REQUIRE((*i)["analytics_memory_usage"]["timestamp"].GetInt64() >= timeBefore);
            BOOST_TEST_REQUIRE((*i)["analytics_memory_usage"]["timestamp"].GetInt64() <= timeAfter);
            hasMemoryUsage = true;
        }
    }
    BOOST_TEST_REQUIRE(hasMemoryUsage);
}

BOOST_AUTO_TEST_SUITE_END()
