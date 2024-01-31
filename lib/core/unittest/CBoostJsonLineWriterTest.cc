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

#include <core/CBoostJsonLineWriter.h>
#include <core/CLogger.h>
#include <core/CStopWatch.h>
#include <core/CStringUtils.h>

#include <boost/json.hpp>
#include <boost/test/unit_test.hpp>

#include <limits>
#include <sstream>

#include "core/CStreamWriter.h"
#include <stdio.h>

namespace json = boost::json;

BOOST_AUTO_TEST_SUITE(CBoostJsonLineWriterTest)

BOOST_AUTO_TEST_CASE(testDoublePrecision) {
    std::ostringstream strm;
    {
        using TGenericLineWriter = ml::core::CStreamWriter;
        TGenericLineWriter writer(strm);

        writer.StartObject();
        writer.Key("a");
        writer.Double(1.78e-156);
        writer.Key("b");
        writer.Double(5e-300);
        writer.Key("c");
        writer.Double(0.0);
        writer.EndObject();
    }

    BOOST_REQUIRE_EQUAL(std::string("{\"a\":1.78e-156,\"b\":5e-300,\"c\":0}\n"), strm.str());
}

BOOST_AUTO_TEST_SUITE_END()
