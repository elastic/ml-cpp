/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CStateRestoreStreamFilterTest.h"

#include <api/CStateRestoreStreamFilter.h>

#include <boost/iostreams/filtering_stream.hpp>

#include <algorithm>
#include <sstream>
#include <string>

CppUnit::Test* CStateRestoreStreamFilterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CRestoreStreamFilterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CStateRestoreStreamFilterTest>(
        "CRestoreStreamFilterTest::testBulkIndexHeaderRemoval", &CStateRestoreStreamFilterTest::testBulkIndexHeaderRemoval));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CStateRestoreStreamFilterTest>("CRestoreStreamFilterTest::testBulkIndexHeaderRemovalZerobyte",
                                                               &CStateRestoreStreamFilterTest::testBulkIndexHeaderRemovalZerobyte));

    return suiteOfTests;
}

void CStateRestoreStreamFilterTest::testBulkIndexHeaderRemoval() {
    std::istringstream input("{\"index\":{\"_id\":\"some_id\"}}\n"
                             "{\"compressed\" : [ \"a\",\"b\"]}");

    boost::iostreams::filtering_istream in;
    in.push(ml::api::CStateRestoreStreamFilter());
    in.push(input);
    std::string output(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});

    std::string expected("{\"_id\":\"some_id\",\"_version\":1,\"found\":true,\"_source\":"
                         "{\"compressed\" : [ \"a\",\"b\"]}}");
    expected += '\0';
    expected += '\n';

    // replace zerobytes to avoid printing problems of cppunit
    std::replace(output.begin(), output.end(), '\0', ',');
    std::replace(expected.begin(), expected.end(), '\0', ',');

    CPPUNIT_ASSERT_EQUAL(expected, output);
}

void CStateRestoreStreamFilterTest::testBulkIndexHeaderRemovalZerobyte() {
    std::stringstream input;

    input << "{\"index\":{\"_id\":\"some_id\"}}\n";
    input << "{\"compressed\" : [ \"a\",\"b\"]}\n";
    input << '\0';
    input << "{\"index\":{\"_id\":\"some_other_id\"}}\n";
    input << "{\"compressed\" : [ \"c\",\"d\"]}\n";

    boost::iostreams::filtering_istream in;
    in.push(ml::api::CStateRestoreStreamFilter());
    in.push(input);
    std::string output(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});

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

    CPPUNIT_ASSERT_EQUAL(expected, output);
}
