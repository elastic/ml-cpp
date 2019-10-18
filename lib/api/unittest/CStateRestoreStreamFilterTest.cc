/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CStateRestoreStreamFilter.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CStateRestoreStreamFilterTest)

BOOST_AUTO_TEST_CASE(testBulkIndexHeaderRemoval) {
    std::istringstream input("{\"index\":{\"_id\":\"some_id\"}}\n"
                             "{\"compressed\" : [ \"a\",\"b\"]}");

    boost::iostreams::filtering_istream in;
    in.push(ml::api::CStateRestoreStreamFilter());
    in.push(input);
    std::string output(std::istreambuf_iterator<char>{in},
                       std::istreambuf_iterator<char>{});

    std::string expected("{\"_id\":\"some_id\",\"_version\":1,\"found\":true,\"_source\":"
                         "{\"compressed\" : [ \"a\",\"b\"]}}");
    expected += '\0';
    expected += '\n';

    // replace zerobytes to avoid printing problems of cppunit
    std::replace(output.begin(), output.end(), '\0', ',');
    std::replace(expected.begin(), expected.end(), '\0', ',');

    BOOST_CHECK_EQUAL(expected, output);
}

BOOST_AUTO_TEST_CASE(testBulkIndexHeaderRemovalZerobyte) {
    std::stringstream input;

    input << "{\"index\":{\"_id\":\"some_id\"}}\n";
    input << "{\"compressed\" : [ \"a\",\"b\"]}\n";
    input << '\0';
    input << "{\"index\":{\"_id\":\"some_other_id\"}}\n";
    input << "{\"compressed\" : [ \"c\",\"d\"]}\n";

    boost::iostreams::filtering_istream in;
    in.push(ml::api::CStateRestoreStreamFilter());
    in.push(input);
    std::string output(std::istreambuf_iterator<char>{in},
                       std::istreambuf_iterator<char>{});

    std::string expected("{\"_id\":\"some_id\",\"_version\":1,\"found\":true,\"_source\":"
                         "{\"compressed\" : [ \"a\",\"b\"]}}");
    expected += '\0';
    expected += '\n';
    expected += "{\"_id\":\"some_other_id\",\"_version\":1,\"found\":true,\"_source\":"
                "{\"compressed\" : [ \"c\",\"d\"]}}";
    expected += '\0';
    expected += '\n';

    // replace zerobytes to avoid printing problems of cppunit
    std::replace(output.begin(), output.end(), '\0', ',');
    std::replace(expected.begin(), expected.end(), '\0', ',');

    BOOST_CHECK_EQUAL(expected, output);
}

BOOST_AUTO_TEST_SUITE_END()
