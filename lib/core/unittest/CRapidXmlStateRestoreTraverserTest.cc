/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CRapidXmlStateRestoreTraverserTest.h"

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

CppUnit::Test* CRapidXmlStateRestoreTraverserTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CRapidXmlStateRestoreTraverserTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CRapidXmlStateRestoreTraverserTest>(
        "CRapidXmlStateRestoreTraverserTest::testRestore",
        &CRapidXmlStateRestoreTraverserTest::testRestore));

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

bool traverse1stLevel(ml::core::CStateRestoreTraverser& traverser) {
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
}

void CRapidXmlStateRestoreTraverserTest::testRestore() {
    std::string xml("<root attr1=\"attrVal1\" "
                    "attr2=\"attrVal2\"><level1A>a</level1A><level1B>25</level1B><level1C><level2A>3.14</level2A><level2B>z</level2B></"
                    "level1C></root>");

    ml::core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));

    ml::core::CRapidXmlStateRestoreTraverser traverser(parser);

    CPPUNIT_ASSERT_EQUAL(std::string("root"), traverser.name());
    CPPUNIT_ASSERT(traverser.hasSubLevel());
    CPPUNIT_ASSERT(traverser.traverseSubLevel(&traverse1stLevel));
    CPPUNIT_ASSERT(!traverser.next());
}
