/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
        "CRestoreStreamFilterTest::testBulkIndexHeaderRemoval",
        &CStateRestoreStreamFilterTest::testBulkIndexHeaderRemoval));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStateRestoreStreamFilterTest>(
        "CRestoreStreamFilterTest::testBulkIndexHeaderRemovalZerobyte",
        &CStateRestoreStreamFilterTest::testBulkIndexHeaderRemovalZerobyte));

    return suiteOfTests;
}

void CStateRestoreStreamFilterTest::testBulkIndexHeaderRemoval(void) {
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

void CStateRestoreStreamFilterTest::testBulkIndexHeaderRemovalZerobyte(void) {
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
