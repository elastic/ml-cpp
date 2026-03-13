/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include "../CResultWriter.h"

#include "../CThreadSettings.h"

#include <boost/test/unit_test.hpp>
#include <torch/csrc/api/include/torch/types.h>

#include <cstdint>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CResultWriterTest)

BOOST_AUTO_TEST_CASE(testWriteThreadSettings) {
    std::ostringstream output;
    {
        ml::torch::CResultWriter resultWriter{output};
        ml::torch::CThreadSettings threadSettings{10, 2, 3};
        resultWriter.writeThreadSettings("req1", threadSettings);
    }
    std::string expected = "[{\"request_id\":\"req1\",\"thread_settings\":"
                           "{\"num_threads_per_allocation\":2,\"num_allocations\":3}}\n]";
    std::string actual = output.str();
    LOG_INFO(<< "expected: " << expected);
    LOG_INFO(<< "actual: " << actual);
    BOOST_REQUIRE_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(testWriteAck) {
    std::ostringstream output;
    {
        ml::torch::CResultWriter resultWriter{output};
        resultWriter.writeSimpleAck("req2");
    }
    BOOST_REQUIRE_EQUAL("[{\"request_id\":\"req2\",\"ack\":{\"acknowledged\":true}}\n]",
                        output.str());
}

BOOST_AUTO_TEST_CASE(testWriteProcessStats) {
    std::ostringstream output;
    {
        ml::torch::CResultWriter resultWriter{output};
        resultWriter.writeProcessStats("req3", 42, 54);
    }
    BOOST_REQUIRE_EQUAL("[{\"request_id\":\"req3\",\"process_stats\":"
                        "{\"memory_rss\":42,\"memory_max_rss\":54}}\n]",
                        output.str());
}

BOOST_AUTO_TEST_CASE(testWriteError) {
    std::ostringstream output;
    {
        ml::torch::CResultWriter resultWriter{output};
        resultWriter.writeError("req3", "bad stuff");
    }
    BOOST_REQUIRE_EQUAL("[{\"request_id\":\"req3\",\"error\":{\"error\":\"bad stuff\"}}\n]",
                        output.str());
}

BOOST_AUTO_TEST_CASE(testCreateInnerInferenceResult) {
    std::ostringstream output;
    ml::torch::CResultWriter resultWriter{output};
    ::torch::Tensor tensor{::torch::ones({5, 3})};
    std::string innerPortion{resultWriter.createInnerResult(tensor)};
    std::string expected = "\"result\":{\"inference\":"
                           "[[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]]}";
    LOG_INFO(<< "expected: " << expected);
    LOG_INFO(<< "actual: " << innerPortion);
    BOOST_REQUIRE_EQUAL(expected, innerPortion);
}

BOOST_AUTO_TEST_CASE(testCreateInnerInferenceResultFor1DimensionalResult) {
    std::ostringstream output;
    ml::torch::CResultWriter resultWriter{output};
    ::torch::Tensor tensor{::torch::ones({1})};
    std::string innerPortion{resultWriter.createInnerResult(tensor)};
    std::string expected = "\"result\":{\"inference\":"
                           "[[[1]]]}";
    LOG_INFO(<< "expected: " << expected);
    LOG_INFO(<< "actual: " << innerPortion);
    BOOST_REQUIRE_EQUAL(expected, innerPortion);
}

BOOST_AUTO_TEST_CASE(testWrapAndWriteInferenceResult) {
    std::string innerPortion{
        "\"result\":{\"inference\":"
        "[[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]]}"};
    std::ostringstream output;
    {
        ml::torch::CResultWriter resultWriter{output};
        resultWriter.wrapAndWriteInnerResponse(innerPortion, "req4", true, 123);
    }
    std::string expected = "[{\"request_id\":\"req4\",\"cache_hit\":true,"
                           "\"time_ms\":123,\"result\":{\"inference\":"
                           "[[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]]}}\n]";
    std::string actual = output.str();

    LOG_INFO(<< "expected: " << expected);
    LOG_INFO(<< "actual  : " << actual);

    BOOST_REQUIRE_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_SUITE_END()
