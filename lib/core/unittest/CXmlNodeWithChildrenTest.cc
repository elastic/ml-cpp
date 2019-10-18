/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CXmlNodeWithChildren.h>
#include <core/CXmlNodeWithChildrenPool.h>
#include <core/CXmlParser.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CXmlNodeWithChildrenTest)

BOOST_AUTO_TEST_CASE(testNodeHierarchyToXml) {
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

    BOOST_TEST(strRep.find("root") != std::string::npos);
    BOOST_TEST(strRep.find("oneDeep1") != std::string::npos);
    BOOST_TEST(strRep.find("oneDeep2") != std::string::npos);
    BOOST_TEST(strRep.find("twoDeepA") != std::string::npos);
    BOOST_TEST(strRep.find("twoDeepB") != std::string::npos);
    BOOST_TEST(strRep.find("twoDeepC") != std::string::npos);

    BOOST_TEST(strRep.find("type") != std::string::npos);
    BOOST_TEST(strRep.find("letter") != std::string::npos);
    BOOST_TEST(strRep.find("case") != std::string::npos);
    BOOST_TEST(strRep.find("upper") != std::string::npos);
    BOOST_TEST(strRep.find("number") != std::string::npos);
    BOOST_TEST(strRep.find("value") != std::string::npos);
    BOOST_TEST(strRep.find("2") != std::string::npos);

    BOOST_TEST(strRep.find("oneDeep1") < strRep.find("oneDeep2"));
    BOOST_TEST(strRep.find("twoDeepA") < strRep.find("twoDeepB"));
    BOOST_TEST(strRep.find("twoDeepA") < strRep.find("twoDeepC"));
    // C is a child of 1, but B is a child of 2, so C should have
    // been printed out first
    BOOST_TEST(strRep.find("twoDeepC") < strRep.find("twoDeepB"));

    std::string xml;
    ml::core::CXmlParser::convert(root, xml);
    LOG_DEBUG(<< "XML representation of XML node hierarchy is:\n" << xml);

    BOOST_TEST(xml.find("root") != std::string::npos);
    BOOST_TEST(xml.find("oneDeep1") != std::string::npos);
    BOOST_TEST(xml.find("oneDeep2") != std::string::npos);
    BOOST_TEST(xml.find("twoDeepA") != std::string::npos);
    BOOST_TEST(xml.find("twoDeepB") != std::string::npos);
    BOOST_TEST(xml.find("twoDeepC") != std::string::npos);

    BOOST_TEST(xml.find("type") != std::string::npos);
    BOOST_TEST(xml.find("letter") != std::string::npos);
    BOOST_TEST(xml.find("case") != std::string::npos);
    BOOST_TEST(xml.find("upper") != std::string::npos);
    BOOST_TEST(xml.find("number") != std::string::npos);
    BOOST_TEST(xml.find("value") != std::string::npos);
    BOOST_TEST(xml.find("2") != std::string::npos);

    BOOST_TEST(xml.find("oneDeep1") < xml.find("oneDeep2"));
    BOOST_TEST(xml.find("twoDeepA") < xml.find("twoDeepB"));
    BOOST_TEST(xml.find("twoDeepA") < xml.find("twoDeepC"));
    // C is a child of 1, but B is a child of 2, so C should have
    // been printed out first
    BOOST_TEST(xml.find("twoDeepC") < xml.find("twoDeepB"));
}

BOOST_AUTO_TEST_CASE(testParserToNodeHierarchy) {
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

    BOOST_TEST(parser.parseString(xml));

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;

    BOOST_TEST(parser.toNodeHierarchy(rootNodePtr));

    BOOST_TEST(rootNodePtr != nullptr);

    std::string strRep(rootNodePtr->dump());
    LOG_DEBUG(<< "Indented representation of XML node hierarchy is:\n"
              << strRep);

    BOOST_TEST(xml.find("root") != std::string::npos);
    BOOST_TEST(xml.find("name1") != std::string::npos);
    BOOST_TEST(xml.find("a") != std::string::npos);
    BOOST_TEST(xml.find("sdacsdac") != std::string::npos);
    BOOST_TEST(xml.find("value1") != std::string::npos);
    BOOST_TEST(xml.find("name2") != std::string::npos);
    BOOST_TEST(xml.find("value2") != std::string::npos);
    BOOST_TEST(xml.find("name3") != std::string::npos);
    BOOST_TEST(xml.find("value3") != std::string::npos);
    BOOST_TEST(xml.find("complex") != std::string::npos);
    BOOST_TEST(xml.find("name4") != std::string::npos);
    BOOST_TEST(xml.find("value4") != std::string::npos);
    BOOST_TEST(xml.find("name5") != std::string::npos);
    BOOST_TEST(xml.find("b") != std::string::npos);
    BOOST_TEST(xml.find("qwerty") != std::string::npos);
    BOOST_TEST(xml.find("value5") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testPerformanceNoPool) {
    ml::core::CXmlParser parser;

    BOOST_TEST(parser.parseFile("testfiles/p2psmon.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting node hierarchy performance test with no pool at "
             << ml::core::CTimeUtils::toTimeString(start));

    static const size_t TEST_SIZE(20000);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        BOOST_TEST(parser.toNodeHierarchy(rootNodePtr));
        BOOST_TEST(rootNodePtr != nullptr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished node hierarchy performance test with no pool at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Node hierarchy performance test of size " << TEST_SIZE
             << " with no pool took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testPerformanceWithPool) {
    ml::core::CXmlParser parser;

    BOOST_TEST(parser.parseFile("testfiles/p2psmon.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting node hierarchy performance test with pool at "
             << ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlNodeWithChildrenPool pool;

    static const size_t TEST_SIZE(20000);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        BOOST_TEST(parser.toNodeHierarchy(pool, rootNodePtr));
        BOOST_TEST(rootNodePtr != nullptr);
        pool.recycle(rootNodePtr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished node hierarchy performance test with pool at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Node hierarchy performance test of size " << TEST_SIZE
             << " with pool took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_SUITE_END()
