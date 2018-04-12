/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CXmlNodeWithChildrenTest.h"

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CXmlNodeWithChildren.h>
#include <core/CXmlNodeWithChildrenPool.h>
#include <core/CXmlParser.h>

CppUnit::Test* CXmlNodeWithChildrenTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CXmlNodeWithChildrenTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CXmlNodeWithChildrenTest>(
        "CXmlNodeWithChildrenTest::testNodeHierarchyToXml",
        &CXmlNodeWithChildrenTest::testNodeHierarchyToXml));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXmlNodeWithChildrenTest>(
        "CXmlNodeWithChildrenTest::testParserToNodeHierarchy",
        &CXmlNodeWithChildrenTest::testParserToNodeHierarchy));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXmlNodeWithChildrenTest>(
        "CXmlNodeWithChildrenTest::testPerformanceNoPool",
        &CXmlNodeWithChildrenTest::testPerformanceNoPool));
    suiteOfTests->addTest(new CppUnit::TestCaller<CXmlNodeWithChildrenTest>(
        "CXmlNodeWithChildrenTest::testPerformanceWithPool",
        &CXmlNodeWithChildrenTest::testPerformanceWithPool));

    return suiteOfTests;
}

void CXmlNodeWithChildrenTest::testNodeHierarchyToXml() {
    ml::core::CXmlParser parser;

    ml::core::CXmlNodeWithChildren twoDeepA("twoDeepA", "Element A");
    ml::core::CXmlNodeWithChildren twoDeepB("twoDeepB", "Element B");
    ml::core::CXmlNodeWithChildren twoDeepC("twoDeepC", "Element C");
    twoDeepC.attribute("type", "letter", true);
    twoDeepC.attribute("case", "upper", true);
    LOG_DEBUG(<< twoDeepC.dump());

    ml::core::CXmlNodeWithChildren oneDeep1("oneDeep1", "");
    oneDeep1.addChild(twoDeepA);
    oneDeep1.addChild(twoDeepC);

    ml::core::CXmlNodeWithChildren oneDeep2("oneDeep2", "");
    oneDeep2.attribute("type", "number", true);
    oneDeep2.attribute("value", 2, true);
    oneDeep2.addChild(twoDeepB);

    ml::core::CXmlNodeWithChildren root("root", "The root element");
    root.addChild(oneDeep1);
    root.addChild(oneDeep2);

    std::string strRep(root.dump());
    LOG_DEBUG(<< "Indented representation of XML node hierarchy is:\n"
              << strRep);

    CPPUNIT_ASSERT(strRep.find("root") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("oneDeep1") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("oneDeep2") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("twoDeepA") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("twoDeepB") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("twoDeepC") != std::string::npos);

    CPPUNIT_ASSERT(strRep.find("type") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("letter") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("case") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("upper") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("number") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("value") != std::string::npos);
    CPPUNIT_ASSERT(strRep.find("2") != std::string::npos);

    CPPUNIT_ASSERT(strRep.find("oneDeep1") < strRep.find("oneDeep2"));
    CPPUNIT_ASSERT(strRep.find("twoDeepA") < strRep.find("twoDeepB"));
    CPPUNIT_ASSERT(strRep.find("twoDeepA") < strRep.find("twoDeepC"));
    // C is a child of 1, but B is a child of 2, so C should have
    // been printed out first
    CPPUNIT_ASSERT(strRep.find("twoDeepC") < strRep.find("twoDeepB"));

    std::string xml;
    ml::core::CXmlParser::convert(root, xml);
    LOG_DEBUG(<< "XML representation of XML node hierarchy is:\n" << xml);

    CPPUNIT_ASSERT(xml.find("root") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("oneDeep1") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("oneDeep2") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("twoDeepA") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("twoDeepB") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("twoDeepC") != std::string::npos);

    CPPUNIT_ASSERT(xml.find("type") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("letter") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("case") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("upper") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("number") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("value") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("2") != std::string::npos);

    CPPUNIT_ASSERT(xml.find("oneDeep1") < xml.find("oneDeep2"));
    CPPUNIT_ASSERT(xml.find("twoDeepA") < xml.find("twoDeepB"));
    CPPUNIT_ASSERT(xml.find("twoDeepA") < xml.find("twoDeepC"));
    // C is a child of 1, but B is a child of 2, so C should have
    // been printed out first
    CPPUNIT_ASSERT(xml.find("twoDeepC") < xml.find("twoDeepB"));
}

void CXmlNodeWithChildrenTest::testParserToNodeHierarchy() {
    ml::core::CXmlParser parser;

    std::string xml = "\
<root> \
    <name1 a='sdacsdac'>value1</name1> \
    <name2>value2</name2> \
    <name3>value3</name3> \
    <complex> \
        <name4>value4</name4> \
        <name5 b='qwerty'>value5</name5> \
    </complex> \
</root>";

    CPPUNIT_ASSERT(parser.parseString(xml));

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;

    CPPUNIT_ASSERT(parser.toNodeHierarchy(rootNodePtr));

    CPPUNIT_ASSERT(rootNodePtr != nullptr);

    std::string strRep(rootNodePtr->dump());
    LOG_DEBUG(<< "Indented representation of XML node hierarchy is:\n"
              << strRep);

    CPPUNIT_ASSERT(xml.find("root") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("name1") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("a") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("sdacsdac") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("value1") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("name2") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("value2") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("name3") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("value3") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("complex") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("name4") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("value4") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("name5") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("b") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("qwerty") != std::string::npos);
    CPPUNIT_ASSERT(xml.find("value5") != std::string::npos);
}

void CXmlNodeWithChildrenTest::testPerformanceNoPool() {
    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseFile("testfiles/p2psmon.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting node hierarchy performance test with no pool at "
             << ml::core::CTimeUtils::toTimeString(start));

    static const size_t TEST_SIZE(20000);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        CPPUNIT_ASSERT(parser.toNodeHierarchy(rootNodePtr));
        CPPUNIT_ASSERT(rootNodePtr != nullptr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished node hierarchy performance test with no pool at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Node hierarchy performance test of size " << TEST_SIZE
             << " with no pool took " << (end - start) << " seconds");
}

void CXmlNodeWithChildrenTest::testPerformanceWithPool() {
    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseFile("testfiles/p2psmon.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting node hierarchy performance test with pool at "
             << ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlNodeWithChildrenPool pool;

    static const size_t TEST_SIZE(20000);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        CPPUNIT_ASSERT(parser.toNodeHierarchy(pool, rootNodePtr));
        CPPUNIT_ASSERT(rootNodePtr != nullptr);
        pool.recycle(rootNodePtr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished node hierarchy performance test with pool at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Node hierarchy performance test of size " << TEST_SIZE
             << " with pool took " << (end - start) << " seconds");
}
