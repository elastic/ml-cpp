/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#include "CJsonStateRestoreTraverserTest.h"

#include <core/CJsonStateRestoreTraverser.h>

#include <sstream>

CppUnit::Test* CJsonStateRestoreTraverserTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CJsonStateRestoreTraverserTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>("CJsonStateRestoreTraverserTest::testRestore1",
                                                                                  &CJsonStateRestoreTraverserTest::testRestore1));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>("CJsonStateRestoreTraverserTest::testRestore2",
                                                                                  &CJsonStateRestoreTraverserTest::testRestore2));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>("CJsonStateRestoreTraverserTest::testRestore3",
                                                                                  &CJsonStateRestoreTraverserTest::testRestore3));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>("CJsonStateRestoreTraverserTest::testRestore4",
                                                                                  &CJsonStateRestoreTraverserTest::testRestore4));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>(
        "CJsonStateRestoreTraverserTest::testParsingBooleanFields", &CJsonStateRestoreTraverserTest::testParsingBooleanFields));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>(
        "CJsonStateRestoreTraverserTest::testRestore1IgnoreArrays", &CJsonStateRestoreTraverserTest::testRestore1IgnoreArrays));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStateRestoreTraverserTest>(
        "CJsonStateRestoreTraverserTest::testRestore1IgnoreArraysNested", &CJsonStateRestoreTraverserTest::testRestore1IgnoreArraysNested));

    return suiteOfTests;
}

namespace {

bool traverse2ndLevel(ml::core::CStateRestoreTraverser& traverser) {
    CPPUNIT_ASSERT_EQUAL(std::string("level2A"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("3.14"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level2B"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("z"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(!traverser.next());

    return true;
}

bool traverse1stLevel1(ml::core::CStateRestoreTraverser& traverser) {
    CPPUNIT_ASSERT_EQUAL(std::string("level1A"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1B"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("25"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1C"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse2ndLevel));
    CPPUNIT_ASSERT(!traverser.next());

    return true;
}

bool traverse1stLevel2(ml::core::CStateRestoreTraverser& traverser) {
    CPPUNIT_ASSERT_EQUAL(std::string("level1A"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1B"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("25"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1C"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse2ndLevel));
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1D"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("afterAscending"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(!traverser.next());

    return true;
}

bool traverse2ndLevelEmpty(ml::core::CStateRestoreTraverser& traverser) {
    CPPUNIT_ASSERT(traverser.name().empty());
    CPPUNIT_ASSERT(traverser.value().empty());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(!traverser.next());

    return true;
}

bool traverse1stLevel3(ml::core::CStateRestoreTraverser& traverser) {
    CPPUNIT_ASSERT_EQUAL(std::string("level1A"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1B"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("25"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1C"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse2ndLevelEmpty));
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1D"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("afterAscending"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(!traverser.next());

    return true;
}

bool traverse1stLevel4(ml::core::CStateRestoreTraverser& traverser) {
    CPPUNIT_ASSERT_EQUAL(std::string("level1A"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1B"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("25"), traverser.value());
    CPPUNIT_ASSERT(!traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("level1C"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    // For this test we ignore the contents of the sub-level
    CPPUNIT_ASSERT(!traverser.next());

    return true;
}
}

void CJsonStateRestoreTraverserTest::testRestore1() {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel1));
    CPPUNIT_ASSERT(!traverser.next());
}

void CJsonStateRestoreTraverserTest::testRestore2() {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"},\"level1D\":"
                     "\"afterAscending\"}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel2));
    CPPUNIT_ASSERT(!traverser.next());
}

void CJsonStateRestoreTraverserTest::testRestore3() {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{},\"level1D\":\"afterAscending\"}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel3));
    CPPUNIT_ASSERT(!traverser.next());
}

void CJsonStateRestoreTraverserTest::testRestore4() {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel4));
    CPPUNIT_ASSERT(!traverser.next());
}

void CJsonStateRestoreTraverserTest::testParsingBooleanFields() {
    // Even though the parser doesn't handle boolean fields it should not hiccup over them
    std::string json = std::string("{\"_index\" : \"categorization-test\", \"_type\" : \"categorizerState\",") +
                       std::string("\"_id\" : \"1\",  \"_version\" : 2, \"found\" : true, ") + std::string("\"_source\":{\"a\" :\"1\"}");

    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_index"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("categorization-test"), traverser.value());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("_type"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("categorizerState"), traverser.value());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("_id"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("1"), traverser.value());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("_version"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("2"), traverser.value());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("found"), traverser.name());
    CPPUNIT_ASSERT_EQUAL(std::string("true"), traverser.value());
    CPPUNIT_ASSERT(traverser.next());
    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
}

void CJsonStateRestoreTraverserTest::testRestore1IgnoreArrays() {
    std::string json(
        "{\"_source\":{\"level1A\":\"a\",\"someArray\":[42],\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel1));
    CPPUNIT_ASSERT(!traverser.next());
}

void CJsonStateRestoreTraverserTest::testRestore1IgnoreArraysNested() {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"someArray\":[{\"nestedArray\":[42]}],\"level1B\":\"25\",\"level1C\":{\"level2A\":"
                     "\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    CPPUNIT_ASSERT_EQUAL(std::string("_source"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel1));
    CPPUNIT_ASSERT(!traverser.next());
}
