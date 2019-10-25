/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/CStopWatch.h>
#include <core/CStringUtils.h>

// beware: testing internal methods of rapidjson, might break after update
#include <rapidjson/internal/dtoa.h>
#include <rapidjson/ostreamwrapper.h>

#include <boost/test/unit_test.hpp>

#include <limits>
#include <sstream>

#include <stdio.h>

BOOST_AUTO_TEST_SUITE(CRapidJsonLineWriterTest)

namespace {
const std::string STR_NAME("str");
const std::string EMPTY1_NAME("empty1");
const std::string EMPTY2_NAME("empty2");
const std::string DOUBLE_NAME("double");
const std::string NAN_NAME("nan");
const std::string INFINITY_NAME("infinity");
const std::string BOOL_NAME("bool");
const std::string INT_NAME("int");
const std::string UINT_NAME("uint");
const std::string STR_ARRAY_NAME("str[]");
const std::string DOUBLE_ARRAY_NAME("double[]");
const std::string NAN_ARRAY_NAME("nan[]");
const std::string TTIME_ARRAY_NAME("TTime[]");
}

BOOST_AUTO_TEST_CASE(testDoublePrecission) {
    std::ostringstream strm;
    {
        using TGenericLineWriter =
            ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper, rapidjson::UTF8<>,
                                           rapidjson::UTF8<>, rapidjson::CrtAllocator>;
        rapidjson::OStreamWrapper writeStream(strm);
        TGenericLineWriter writer(writeStream);

        writer.StartObject();
        writer.Key("a");
        writer.Double(3e-5);
        writer.Key("b");
        writer.Double(5e-300);
        writer.Key("c");
        writer.Double(0.0);
        writer.EndObject();
    }

    BOOST_REQUIRE_EQUAL(std::string("{\"a\":0.00003,\"b\":5e-300,\"c\":0.0}\n"),
                        strm.str());
}

BOOST_AUTO_TEST_CASE(testDoublePrecissionDtoa) {
    char buffer[100];

    char* end = rapidjson::internal::dtoa(3e-5, buffer);
    BOOST_REQUIRE_EQUAL(std::string("0.00003"),
                        std::string(buffer, static_cast<size_t>(end - buffer)));

    end = rapidjson::internal::dtoa(2e-20, buffer, 20);
    BOOST_REQUIRE_EQUAL(std::string("2e-20"),
                        std::string(buffer, static_cast<size_t>(end - buffer)));

    end = rapidjson::internal::dtoa(1e-308, buffer);
    BOOST_REQUIRE_EQUAL(std::string("1e-308"),
                        std::string(buffer, static_cast<size_t>(end - buffer)));

    end = rapidjson::internal::dtoa(1e-300, buffer, 20);
    BOOST_REQUIRE_EQUAL(std::string("0.0"),
                        std::string(buffer, static_cast<size_t>(end - buffer)));

    // test the limit, to not hardcode the string we check that it is not 0.0
    end = rapidjson::internal::dtoa(std::numeric_limits<double>::denorm_min(), buffer);
    BOOST_TEST_REQUIRE(std::string("0.0") !=
                       std::string(buffer, static_cast<size_t>(end - buffer)));

    int ret = ::snprintf(buffer, sizeof(buffer), "%g", 1e-300);

    BOOST_REQUIRE_EQUAL(std::string("1e-300"), std::string(buffer, ret));
}

BOOST_AUTO_TEST_CASE(testMicroBenchmark, *boost::unit_test::disabled()) {
    char buffer[100];
    ml::core::CStopWatch stopWatch;

    stopWatch.start();
    size_t runs = 100000000;

    for (size_t i = 0; i < runs; ++i) {
        rapidjson::internal::dtoa(3e-5, buffer);
        rapidjson::internal::dtoa(0.0, buffer);
        rapidjson::internal::dtoa(0.12345, buffer);
        rapidjson::internal::dtoa(1.43e-35, buffer);
        rapidjson::internal::dtoa(42.0, buffer);
    }
    uint64_t elapsed = stopWatch.stop();
    LOG_INFO(<< "Rapidjson dtoa " << runs << " runs took " << elapsed);
    stopWatch.reset();
    stopWatch.start();
    for (size_t i = 0; i < runs; ++i) {
        ::snprintf(buffer, sizeof(buffer), "%g", 3e-5);
        ::snprintf(buffer, sizeof(buffer), "%g", 0.0);
        ::snprintf(buffer, sizeof(buffer), "%g", 0.12345);
        ::snprintf(buffer, sizeof(buffer), "%g", 1.43e-35);
        ::snprintf(buffer, sizeof(buffer), "%g", 42.0);
    }

    elapsed = stopWatch.stop();
    LOG_INFO(<< "snprintf " << runs << " runs took " << elapsed);
}

BOOST_AUTO_TEST_SUITE_END()
