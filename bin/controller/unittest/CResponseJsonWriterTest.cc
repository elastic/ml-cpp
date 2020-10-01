/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "../CResponseJsonWriter.h"

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CResponseJsonWriterTest)

BOOST_AUTO_TEST_CASE(testResponseWriter) {
    std::ostringstream responseStream;
    ml::controller::CResponseJsonWriter responseWriter{responseStream};
    responseWriter.writeResponse(1, true, "reason a");
    responseWriter.writeResponse(3, false, "reason b");
    responseWriter.writeResponse(2, true, "reason c");

    BOOST_REQUIRE_EQUAL("{\"id\":1,\"success\":true,\"reason\":\"reason a\"}\n"
                        "{\"id\":3,\"success\":false,\"reason\":\"reason b\"}\n"
                        "{\"id\":2,\"success\":true,\"reason\":\"reason c\"}\n",
                        responseStream.str());
}

BOOST_AUTO_TEST_SUITE_END()
