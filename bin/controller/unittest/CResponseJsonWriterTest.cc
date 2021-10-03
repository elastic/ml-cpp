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

#include "../CResponseJsonWriter.h"

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CResponseJsonWriterTest)

BOOST_AUTO_TEST_CASE(testResponseWriter) {
    std::ostringstream responseStream;
    {
        ml::controller::CResponseJsonWriter responseWriter{responseStream};
        responseWriter.writeResponse(1, true, "reason a");
        responseWriter.writeResponse(3, false, "reason b");
        responseWriter.writeResponse(2, true, "reason c");
    }

    BOOST_REQUIRE_EQUAL("[{\"id\":1,\"success\":true,\"reason\":\"reason a\"}\n"
                        ",{\"id\":3,\"success\":false,\"reason\":\"reason b\"}\n"
                        ",{\"id\":2,\"success\":true,\"reason\":\"reason c\"}\n"
                        "]",
                        responseStream.str());
}

BOOST_AUTO_TEST_SUITE_END()
