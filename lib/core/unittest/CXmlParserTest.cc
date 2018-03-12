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
#include "CXmlParserTest.h"

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CXmlParser.h>
#include <core/CXmlNode.h>
#include <core/CXmlNodeWithChildrenPool.h>

#include <test/CTestTmpDir.h>

#include "CRapidXmlParserTest.h"

#include <fstream>

#include <stdio.h>


CppUnit::Test *CXmlParserTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CXmlParserTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse1File",
                               &CXmlParserTest::testParse1File) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse1String",
                               &CXmlParserTest::testParse1String) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse2",
                               &CXmlParserTest::testParse2) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testNavigate",
                               &CXmlParserTest::testNavigate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParseXInclude",
                               &CXmlParserTest::testParseXInclude) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse3",
                               &CXmlParserTest::testParse3) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse4",
                               &CXmlParserTest::testParse4) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse5",
                               &CXmlParserTest::testParse5) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParse6",
                               &CXmlParserTest::testParse6) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testConvert1",
                               &CXmlParserTest::testConvert1) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testConvert2",
                               &CXmlParserTest::testConvert2) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testConvert3",
                               &CXmlParserTest::testConvert3) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testConvert4",
                               &CXmlParserTest::testConvert4) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testAddNewChildNode",
                               &CXmlParserTest::testAddNewChildNode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testSetRootNode",
                               &CXmlParserTest::testSetRootNode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testDump",
                               &CXmlParserTest::testDump) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testMakeValidName",
                               &CXmlParserTest::testMakeValidName) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testChangeChild",
                               &CXmlParserTest::testChangeChild) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testHugeDoc",
                               &CXmlParserTest::testHugeDoc) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testParseSpeed",
                               &CXmlParserTest::testParseSpeed) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testConvertSpeed",
                               &CXmlParserTest::testConvertSpeed) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CXmlParserTest>(
                               "CXmlParserTest::testComplexXPath",
                               &CXmlParserTest::testComplexXPath) );

    return suiteOfTests;
}

void CXmlParserTest::testParse1File(void) {
    std::string badFileName = "./testfiles/CXmlParser_bad.xml";
    std::string goodFileName = "./testfiles/CXmlParser1.xml";

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(!parser.parseFile(badFileName));
    CPPUNIT_ASSERT(parser.parseFile(goodFileName));

    this->testParse1(parser);
}

void CXmlParserTest::testParse1String(void) {
    std::string goodString = CXmlParserTest::fileToString("./testfiles/CXmlParser1.xml");

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseString(goodString));

    this->testParse1(parser);
}

void CXmlParserTest::testParse2(void) {
    std::string goodFileName = "./testfiles/CXmlParser2.xml";

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseFile(goodFileName));

    ml::core::CXmlParser::TXmlNodeVec nodes;

    CPPUNIT_ASSERT(parser.evalXPathExpression("//badpath", nodes));
    CPPUNIT_ASSERT(nodes.empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/syslog_parser/parsetree/expression/description", nodes));
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes.size());

    CPPUNIT_ASSERT_EQUAL(std::string("description"), nodes[0].name());
    CPPUNIT_ASSERT_EQUAL(std::string("Transport node error"), nodes[0].value());
    CPPUNIT_ASSERT(nodes[0].attributes().empty());

    CPPUNIT_ASSERT_EQUAL(std::string("description"), nodes[1].name());
    CPPUNIT_ASSERT_EQUAL(std::string("Transport read error"), nodes[1].value());
    CPPUNIT_ASSERT(nodes[1].attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/syslog_parser/parsetree/expression[1]/regexes/varbind/token", nodes));
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes.size());

    CPPUNIT_ASSERT_EQUAL(std::string("token"), nodes[0].name());
    CPPUNIT_ASSERT_EQUAL(std::string(""), nodes[0].value());
    CPPUNIT_ASSERT(nodes[0].attributes().empty());

    CPPUNIT_ASSERT_EQUAL(std::string("token"), nodes[1].name());
    CPPUNIT_ASSERT_EQUAL(std::string("source"), nodes[1].value());
    CPPUNIT_ASSERT(nodes[1].attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/syslog_parser/parsetree/expression[1]/regexes/varbind/regex", nodes));
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes.size());

    CPPUNIT_ASSERT_EQUAL(std::string("regex"), nodes[0].name());
    CPPUNIT_ASSERT_EQUAL(std::string("^[[:space:]]*"), nodes[0].value());
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes[0].attributes().size());
    CPPUNIT_ASSERT(this->testAttribute(nodes[0], "function", "default"));
    CPPUNIT_ASSERT(this->testAttribute(nodes[0], "local", "BZ"));

    CPPUNIT_ASSERT_EQUAL(std::string("regex"), nodes[1].name());
    CPPUNIT_ASSERT_EQUAL(std::string("(template[[:space:]]*<[^;:{]+>[[:space:]]*)?"), nodes[1].value());
    CPPUNIT_ASSERT(nodes[1].attributes().empty());
}

void CXmlParserTest::testNavigate(void) {
    std::string goodFileName = "./testfiles/CXmlParser2.xml";

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseFile(goodFileName));

    std::string str;
    CPPUNIT_ASSERT(parser.navigateRoot());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("syslog_parser"), str);
    CPPUNIT_ASSERT(parser.navigateFirstChild());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("parsetree"), str);
    CPPUNIT_ASSERT(parser.navigateFirstChild());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("expression"), str);
    CPPUNIT_ASSERT(parser.navigateFirstChild());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("description"), str);
    CPPUNIT_ASSERT(parser.currentNodeValue(str));
    CPPUNIT_ASSERT_EQUAL(std::string("Transport node error"), str);
    CPPUNIT_ASSERT(parser.navigateNext());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("regexes"), str);
    CPPUNIT_ASSERT(parser.navigateParent());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("expression"), str);
    CPPUNIT_ASSERT(parser.navigateParent());
    CPPUNIT_ASSERT(parser.currentNodeName(str));
    CPPUNIT_ASSERT_EQUAL(std::string("parsetree"), str);
    CPPUNIT_ASSERT(!parser.navigateNext());
}

void CXmlParserTest::testParseXInclude(void) {
    std::string goodFileName = "./testfiles/CXmlParser3.xml";
    std::string badFileName = "./testfiles/CXmlParser4.xml";

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(!parser.parseFile(badFileName));
    CPPUNIT_ASSERT(parser.parseFile(goodFileName));

    ml::core::CXmlParser::TXmlNodeVec nodes;

    CPPUNIT_ASSERT(parser.evalXPathExpression("//badpath", nodes));
    CPPUNIT_ASSERT(nodes.empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/syslog_parser/parsetree/expression/description", nodes));
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes.size());

    CPPUNIT_ASSERT_EQUAL(std::string("description"), nodes[0].name());
    CPPUNIT_ASSERT_EQUAL(std::string("Transport node error"), nodes[0].value());
    CPPUNIT_ASSERT(nodes[0].attributes().empty());

    CPPUNIT_ASSERT_EQUAL(std::string("description"), nodes[1].name());
    CPPUNIT_ASSERT_EQUAL(std::string("Transport read error"), nodes[1].value());
    CPPUNIT_ASSERT(nodes[1].attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/syslog_parser/parsetree/expression[1]/regexes/varbind/token", nodes));
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes.size());

    CPPUNIT_ASSERT_EQUAL(std::string("token"), nodes[0].name());
    CPPUNIT_ASSERT_EQUAL(std::string(""), nodes[0].value());
    CPPUNIT_ASSERT(nodes[0].attributes().empty());

    CPPUNIT_ASSERT_EQUAL(std::string("token"), nodes[1].name());
    CPPUNIT_ASSERT_EQUAL(std::string("source"), nodes[1].value());
    CPPUNIT_ASSERT(nodes[1].attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/syslog_parser/parsetree/expression[1]/regexes/varbind/regex", nodes));
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes.size());

    CPPUNIT_ASSERT_EQUAL(std::string("regex"), nodes[0].name());
    CPPUNIT_ASSERT_EQUAL(std::string("^[[:space:]]*"), nodes[0].value());
    CPPUNIT_ASSERT_EQUAL(size_t(2), nodes[0].attributes().size());
    CPPUNIT_ASSERT(this->testAttribute(nodes[0], "function", "default"));
    CPPUNIT_ASSERT(this->testAttribute(nodes[0], "local", "BZ"));

    CPPUNIT_ASSERT_EQUAL(std::string("regex"), nodes[1].name());
    CPPUNIT_ASSERT_EQUAL(std::string("(template[[:space:]]*<[^;:{]+>[[:space:]]*)?"), nodes[1].value());
    CPPUNIT_ASSERT(nodes[1].attributes().empty());
}

void CXmlParserTest::testParse3(void) {
    std::string fileName = "./testfiles/CXmlParser5.xml";

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseFile(fileName));

    ml::core::CXmlParser::TXmlNodeVec arguments;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/OperationRequest/Arguments/Argument", arguments));
    CPPUNIT_ASSERT_EQUAL(size_t(7), arguments.size());

    for (ml::core::CXmlParser::TXmlNodeVecItr itr = arguments.begin();
         itr != arguments.end();
         ++itr) {
        if (itr->value() == "Service") {
            CPPUNIT_ASSERT(this->testAttribute(*itr, "Value", "AWSECommerceService"));
        } else if (itr->value() == "AssociateTag") {
            CPPUNIT_ASSERT(!this->testAttribute(*itr, "Value", ""));
        } else if (itr->value() == "SearchIndex") {
            CPPUNIT_ASSERT(this->testAttribute(*itr, "Value", "Books"));
        } else if (itr->value() == "Author") {
            CPPUNIT_ASSERT(!this->testAttribute(*itr, "Value", ""));
        } else if (itr->value() == "Hacasdasdcv") {
            CPPUNIT_ASSERT(this->testAttribute(*itr, "Value", "1A7XKHR5BYD0WPJVQEG2"));
        } else if (itr->value() == "Version") {
            CPPUNIT_ASSERT(this->testAttribute(*itr, "Value", "2006-06-28"));
        } else if (itr->value() == "Operation") {
            CPPUNIT_ASSERT(!this->testAttribute(*itr, "Value", ""));
        } else {
            CPPUNIT_ASSERT_MESSAGE(itr->dump(), false);
        }
    }
}

void CXmlParserTest::testParse4(void) {
    std::string fileName = "./testfiles/CXmlParser1.xml";

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseFile(fileName));

    bool valid(false);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/Items/Request/IsValid", valid));
    CPPUNIT_ASSERT(valid);

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/Items/TotalPages", valid));
    CPPUNIT_ASSERT(valid);

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/Items/Request/IsNotValid", valid));
    CPPUNIT_ASSERT(!valid);

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/Items/Request/IsNotValidNo", valid));
    CPPUNIT_ASSERT(!valid);

    int i;
    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/Items/TotalPages", i));
    CPPUNIT_ASSERT_EQUAL(21, i);

    // Invalid conversions
    CPPUNIT_ASSERT(!parser.evalXPathExpression("/ItemSearchResponse/Items/Request/IsValid", i));
    CPPUNIT_ASSERT(!parser.evalXPathExpression("/ItemSearchResponse/Items/Request/ItemSearchRequest", i));
    CPPUNIT_ASSERT(!parser.evalXPathExpression("/ItemSearchResponse/Items/Request/ItemSearchRequest/Author", i));
}

void CXmlParserTest::testParse5(void) {
    ml::core::CXmlParser parser;

    std::string xml = "\
<root> \
    <name1 a='sdacsdac'>value1</name1> \
    <name2>value2</name2> \
    <name3>value3</name3> \
</root>";

    CPPUNIT_ASSERT(parser.parseString(xml));

    ml::core::CXmlParser::TStrStrMap values;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/*", values));

    CPPUNIT_ASSERT_EQUAL(values.size(), size_t(3));

    ml::core::CXmlParser::TStrStrMapCItr itr = values.find("name1");
    CPPUNIT_ASSERT(itr != values.end());
    CPPUNIT_ASSERT_EQUAL(itr->second, std::string("value1"));
    itr = values.find("name2");
    CPPUNIT_ASSERT(itr != values.end());
    CPPUNIT_ASSERT_EQUAL(itr->second, std::string("value2"));
    itr = values.find("name3");
    CPPUNIT_ASSERT(itr != values.end());
    CPPUNIT_ASSERT_EQUAL(itr->second, std::string("value3"));
}

void CXmlParserTest::testParse6(void) {

    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <name a='sdacsdac'>value1</name> \
    <name>value2</name> \
    <name>value3</name> \
</root>";

        CPPUNIT_ASSERT(parser.parseString(xml));

        ml::core::CXmlParser::TStrVec values;

        CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name", values));

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(3));

        CPPUNIT_ASSERT_EQUAL(values[0], std::string("value1"));
        CPPUNIT_ASSERT_EQUAL(values[1], std::string("value2"));
        CPPUNIT_ASSERT_EQUAL(values[2], std::string("value3"));
    }
    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <names> \
    </names> \
</root>";

        CPPUNIT_ASSERT(parser.parseString(xml));

        ml::core::CXmlParser::TStrVec values;

        CPPUNIT_ASSERT(parser.evalXPathExpression("/root/names/*", values));

        CPPUNIT_ASSERT(values.empty());
    }
    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <name a='sdacsdac'>value1</name> \
    <name>value2</name> \
    <name>value3</name> \
</root>";

        CPPUNIT_ASSERT(parser.parseString(xml));

        ml::core::CXmlParser::TStrSet values;

        CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name", values));

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(3));
    }
    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <name a='sdacsdac'>value1</name> \
    <name>value2</name> \
    <name>value2</name> \
</root>";

        CPPUNIT_ASSERT(parser.parseString(xml));

        ml::core::CXmlParser::TStrSet values;

        CPPUNIT_ASSERT(!parser.evalXPathExpression("/root/name", values));
    }

}

void CXmlParserTest::testConvert1(void) {
    ml::core::CXmlParser::TStrStrMap values;

    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("key", "<&sdacasdc"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("count", "12"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("desc", "sdac asdc asdc\nadsc\nasdc\n"));

    std::string xml;
    ml::core::CXmlParser::convert("test_convert", values, xml);

    LOG_DEBUG(xml);

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseString(xml));

    ml::core::CXmlNode node;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/key", node));
    CPPUNIT_ASSERT_EQUAL(std::string("<&sdacasdc"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/count", node));
    CPPUNIT_ASSERT_EQUAL(std::string("12"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/desc", node));
    CPPUNIT_ASSERT_EQUAL(std::string("sdac asdc asdc\nadsc\nasdc\n"), node.value());
}

void CXmlParserTest::testConvert2(void) {
    ml::core::CXmlParser::TStrStrMap values;

    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("key", "<&sdacasdc"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("count", "12"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("desc", "sdac asdc asdc\nadsc\nasdc\n"));

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.convert("test_convert", values));

    ml::core::CXmlNode node;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/key", node));
    CPPUNIT_ASSERT_EQUAL(std::string("<&sdacasdc"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/count", node));
    CPPUNIT_ASSERT_EQUAL(std::string("12"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/desc", node));
    CPPUNIT_ASSERT_EQUAL(std::string("sdac asdc asdc\nadsc\nasdc\n"), node.value());
}

void CXmlParserTest::testConvert3(void) {
    ml::core::CXmlParser::TStrStrMap values;

    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("key", "<&sdacasdc"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("count", "1"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("field@name=idle cpu %", "96"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("field@name=user cpu %", "3"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("field@name=system cpu %", "1"));

    std::string xml;
    ml::core::CXmlParser::convert("test_convert", values, xml);

    LOG_DEBUG(xml);

    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.parseString(xml));

    ml::core::CXmlNode node;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/key", node));
    CPPUNIT_ASSERT_EQUAL(std::string("<&sdacasdc"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/count", node));
    CPPUNIT_ASSERT_EQUAL(std::string("1"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/field[@name='idle cpu %']", node));
    CPPUNIT_ASSERT_EQUAL(std::string("96"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/field[@name='user cpu %']", node));
    CPPUNIT_ASSERT_EQUAL(std::string("3"), node.value());
    CPPUNIT_ASSERT(parser.evalXPathExpression("/test_convert/field[@name='system cpu %']", node));
    CPPUNIT_ASSERT_EQUAL(std::string("1"), node.value());
}

void CXmlParserTest::testConvert4(void) {
    // Use a standard node hierarchy to allow for comparison with the
    // standards-compliant XML parser
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(CRapidXmlParserTest::makeTestNodeHierarchy());

    std::string converted;
    ml::core::CXmlParser::convert(*root, converted);

    LOG_DEBUG("Converted node hierarchy is:\n" << converted);

    CPPUNIT_ASSERT(converted.find("<root>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("</root>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<id>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("123") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("</id>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<parent>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("</parent>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<child>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("boo!") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("2nd") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("</child>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<child ") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("&amp; ") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<empty/>") != std::string::npos || converted.find("<empty></empty>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<dual ") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("first") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("second") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("attribute") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("got") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("</dual>") != std::string::npos);
}

void CXmlParserTest::testAddNewChildNode(void) {
    ml::core::CXmlParser parser;

    std::string xml = "\
<root> \
    <name1 a='sdacsdac'>value1</name1> \
    <name2>value2</name2> \
    <name3>value3</name3> \
</root>";

    CPPUNIT_ASSERT(parser.parseString(xml));

    std::string value;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name1", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value1"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name2", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value2"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name3", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value3"), value);

    CPPUNIT_ASSERT(parser.addNewChildNode("name4", "value4"));

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name4", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value4"), value);
}

void CXmlParserTest::testSetRootNode(void) {

    {
        ml::core::CXmlParser parser;

        CPPUNIT_ASSERT(parser.setRootNode("root"));

        CPPUNIT_ASSERT(parser.addNewChildNode("name1", "value1"));
        CPPUNIT_ASSERT(parser.addNewChildNode("name2", "value2"));

        std::string value;

        CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name1", value));
        CPPUNIT_ASSERT_EQUAL(std::string("value1"), value);
        CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name2", value));
        CPPUNIT_ASSERT_EQUAL(std::string("value2"), value);
    }
    {
        ml::core::CXmlParser parser;

        CPPUNIT_ASSERT(parser.setRootNode("root"));

        CPPUNIT_ASSERT(parser.addNewChildNode("name", "value1"));
        CPPUNIT_ASSERT(parser.addNewChildNode("name", "value2"));

        parser.dumpToStdout();
    }

}

void CXmlParserTest::testDump(void) {
    std::string fileName = "./testfiles/CXmlParser1.xml";

    ml::core::CXmlParser parser1;
    CPPUNIT_ASSERT(parser1.parseFile(fileName));
    this->testParse1(parser1);

    std::string expected = parser1.dumpToString();

    ml::core::CXmlParser parser2;
    CPPUNIT_ASSERT(parser2.parseString(expected));
    this->testParse1(parser2);
}

std::string CXmlParserTest::fileToString(const std::string &fileName) {
    std::string ret;

    std::ifstream ifs(fileName.c_str());
    CPPUNIT_ASSERT_MESSAGE(fileName, ifs.is_open());

    std::string line;
    while (std::getline(ifs, line)) {
        ret += line;
        ret += '\n';
    }

    return ret;
}

void CXmlParserTest::testParse1(const ml::core::CXmlParser &parser) {
    ml::core::CXmlNode node;
    std::string             value;

    CPPUNIT_ASSERT(!parser.evalXPathExpression("//badpath", node));

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/OperationRequest/HTTPHeaders/Header/@Value", node));
    CPPUNIT_ASSERT_EQUAL(std::string("Value"), node.name());
    CPPUNIT_ASSERT_EQUAL(std::string("Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser; Avant Browser; .NET CLR 1.0.3705; .NET CLR 2.0.50727; .NET CLR 1.1.4322; Media Center PC 4.0; InfoPath.2)"), node.value());
    CPPUNIT_ASSERT(node.attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/OperationRequest/RequestId", node));
    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/OperationRequest/RequestId", value));
    CPPUNIT_ASSERT_EQUAL(std::string("RequestId"), node.name());
    CPPUNIT_ASSERT_EQUAL(std::string("18CZWZFXKSV8F601AGMF"), node.value());
    CPPUNIT_ASSERT_EQUAL(std::string("18CZWZFXKSV8F601AGMF"), value);
    CPPUNIT_ASSERT(node.attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/ItemSearchResponse/OperationRequest/RequestProcessingTime", node));
    CPPUNIT_ASSERT_EQUAL(std::string("RequestProcessingTime"), node.name());
    CPPUNIT_ASSERT_EQUAL(std::string("1.05041599273682"), node.value());
    CPPUNIT_ASSERT(node.attributes().empty());

    CPPUNIT_ASSERT(parser.evalXPathExpression("//msg", node));
    CPPUNIT_ASSERT_EQUAL(std::string("msg"), node.name());
    CPPUNIT_ASSERT_EQUAL(std::string("\n\
            Invalid Date of Birth. <br /><i>This is a test validation message from the server </i>\n\
             "), node.value());
    CPPUNIT_ASSERT(node.attributes().empty());

    CPPUNIT_ASSERT_EQUAL(std::string("ItemSearchResponse"), parser.rootElementName());
}

bool CXmlParserTest::testAttribute(const ml::core::CXmlNode &node,
                                   const std::string &key,
                                   const std::string &expected) {
    std::string actual;
    if (node.attribute(key, actual) == false) {
        return false;
    }

    if (actual != expected) {
        LOG_ERROR(actual << ' ' << expected);
        return false;
    }

    return true;
}

void CXmlParserTest::testMakeValidName(void) {
    CPPUNIT_ASSERT_EQUAL(std::string("name"), ml::core::CXmlParser::makeValidName("name"));
    CPPUNIT_ASSERT_EQUAL(std::string("name1"), ml::core::CXmlParser::makeValidName("name1"));
    CPPUNIT_ASSERT_EQUAL(std::string("_name"), ml::core::CXmlParser::makeValidName("1name"));
    CPPUNIT_ASSERT_EQUAL(std::string("name_2"), ml::core::CXmlParser::makeValidName("name/2"));
    CPPUNIT_ASSERT_EQUAL(std::string("_name_"), ml::core::CXmlParser::makeValidName("_name_"));
    CPPUNIT_ASSERT_EQUAL(std::string("__cencl01b_System_System_Calls_sec"), ml::core::CXmlParser::makeValidName("\\\\cencl01b\\System\\System Calls/sec"));
}

void CXmlParserTest::testChangeChild(void) {
    ml::core::CXmlParser parser;

    CPPUNIT_ASSERT(parser.setRootNode("root"));
    CPPUNIT_ASSERT(parser.addNewChildNode("name1", "value1"));
    CPPUNIT_ASSERT(parser.addNewChildNode("name2", "value2"));
    CPPUNIT_ASSERT(parser.addNewChildNode("name3", "value3"));

    LOG_DEBUG(parser.dumpToString());

    std::string value;

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name1", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value1"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name2", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value2"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name3", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value3"), value);

    // Change each of the values in turn, checking state after each change
    CPPUNIT_ASSERT(parser.changeChildNodeValue("name2", "changed2"));

    LOG_DEBUG(parser.dumpToString());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name1", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value1"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name2", value));
    CPPUNIT_ASSERT_EQUAL(std::string("changed2"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name3", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value3"), value);

    CPPUNIT_ASSERT(parser.changeChildNodeValue("name1", "changed1"));

    LOG_DEBUG(parser.dumpToString());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name1", value));
    CPPUNIT_ASSERT_EQUAL(std::string("changed1"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name2", value));
    CPPUNIT_ASSERT_EQUAL(std::string("changed2"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name3", value));
    CPPUNIT_ASSERT_EQUAL(std::string("value3"), value);

    CPPUNIT_ASSERT(parser.changeChildNodeValue("name3", "changed3"));

    LOG_DEBUG(parser.dumpToString());

    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name1", value));
    CPPUNIT_ASSERT_EQUAL(std::string("changed1"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name2", value));
    CPPUNIT_ASSERT_EQUAL(std::string("changed2"), value);
    CPPUNIT_ASSERT(parser.evalXPathExpression("/root/name3", value));
    CPPUNIT_ASSERT_EQUAL(std::string("changed3"), value);
}

void CXmlParserTest::testHugeDoc(void) {
    // libxml2 can exhibit O(n^2.42) behaviour if the xmlXPathOrderDocElems()
    // function hasn't been called on the document.  Obviously this only shows
    // up as a problem in huge XML documents.

    // First, create an enormous XML document
    std::string fileName(ml::test::CTestTmpDir::tmpDir() + "/huge.xml");
    std::ofstream ofs(fileName.c_str());
    CPPUNIT_ASSERT_MESSAGE(fileName, ofs.is_open());

    ofs << "<nodes>" << std::endl;

    static const size_t NUM_NODES(300000);
    for (size_t count = 1; count <= NUM_NODES; ++count) {
        ofs << "    <node>" << count << "</node>" << std::endl;
    }

    ofs << "</nodes>" << std::endl;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting huge XPath test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlParser parser;
    ml::core::CXmlParser::TStrSet valueSet;

    CPPUNIT_ASSERT(parser.parseFile(fileName));

    // NB: If xmlXPathOrderDocElems() hasn't been called, this will take an
    // astronomical amount of time - don't wait more than a minute for it!
    CPPUNIT_ASSERT(parser.evalXPathExpression("/nodes/node", valueSet));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished huge XPath test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(NUM_NODES, valueSet.size());

    LOG_INFO("Applying an XPath to a node set with " << NUM_NODES <<
             " nodes took " << (end - start) << " seconds");

    ::remove(fileName.c_str());
}

void CXmlParserTest::testParseSpeed(void) {
    static const size_t TEST_SIZE(25000);

    std::string testString(CXmlParserTest::fileToString("./testfiles/CXmlParser2.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting parse speed test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlNodeWithChildrenPool nodePool;

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CXmlParser parser;
        CPPUNIT_ASSERT(parser.parseString(testString));

        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        CPPUNIT_ASSERT(parser.toNodeHierarchy(nodePool, rootNodePtr));

        CPPUNIT_ASSERT(rootNodePtr != 0);

        nodePool.recycle(rootNodePtr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished parse speed test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO("Parsing " << TEST_SIZE << " documents took " <<
             (end - start) << " seconds");
}

void CXmlParserTest::testConvertSpeed(void) {
    static const size_t TEST_SIZE(100000);

    // Use a standard node hierarchy to allow for comparison with the
    // standards-compliant XML parser
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(CRapidXmlParserTest::makeTestNodeHierarchy());

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting convert speed test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::string converted;
        ml::core::CXmlParser::convert(*root, converted);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished convert speed test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO("Converting " << TEST_SIZE << " documents took " <<
             (end - start) << " seconds");
}

void CXmlParserTest::testComplexXPath(void) {
    ml::core::CXmlParser parser;
    CPPUNIT_ASSERT(parser.parseFile("testfiles/withNs.xml"));

    bool disabled(false);

    // This convoluted query is for XML schemas that
    // have a default namespace but don't give it a name!
    CPPUNIT_ASSERT(parser.evalXPathExpression("//*[local-name()='title' and .='ml']/..//*[local-name()='key' and @name='disabled']",
                                              disabled));
    CPPUNIT_ASSERT_EQUAL(true, disabled);
}

