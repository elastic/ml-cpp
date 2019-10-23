/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CXmlNode.h>
#include <core/CXmlNodeWithChildrenPool.h>
#include <core/CXmlParser.h>

#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>

#include <fstream>

#include <stdio.h>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::core::CXmlParser::TStrStrMap::iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::core::CXmlParser::TStrStrMapCItr)

BOOST_AUTO_TEST_SUITE(CXmlParserTest)

std::string fileToString(const std::string& fileName) {
    std::string ret;

    std::ifstream ifs(fileName.c_str());
    BOOST_REQUIRE_MESSAGE(ifs.is_open(), fileName + " is not open");

    std::string line;
    while (std::getline(ifs, line)) {
        ret += line;
        ret += '\n';
    }

    return ret;
}

void testParseHelper(const ml::core::CXmlParser& parser) {
    ml::core::CXmlNode node;
    std::string value;

    BOOST_TEST_REQUIRE(!parser.evalXPathExpression("//badpath", node));

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/OperationRequest/HTTPHeaders/Header/@Value", node));
    BOOST_REQUIRE_EQUAL(std::string("Value"), node.name());
    BOOST_REQUIRE_EQUAL(std::string("Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser; Avant Browser; .NET CLR 1.0.3705; "
                                    ".NET CLR 2.0.50727; .NET CLR 1.1.4322; Media Center PC 4.0; InfoPath.2)"),
                        node.value());
    BOOST_TEST_REQUIRE(node.attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/OperationRequest/RequestId", node));
    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/OperationRequest/RequestId", value));
    BOOST_REQUIRE_EQUAL(std::string("RequestId"), node.name());
    BOOST_REQUIRE_EQUAL(std::string("18CZWZFXKSV8F601AGMF"), node.value());
    BOOST_REQUIRE_EQUAL(std::string("18CZWZFXKSV8F601AGMF"), value);
    BOOST_TEST_REQUIRE(node.attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/OperationRequest/RequestProcessingTime", node));
    BOOST_REQUIRE_EQUAL(std::string("RequestProcessingTime"), node.name());
    BOOST_REQUIRE_EQUAL(std::string("1.05041599273682"), node.value());
    BOOST_TEST_REQUIRE(node.attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("//msg", node));
    BOOST_REQUIRE_EQUAL(std::string("msg"), node.name());
    BOOST_REQUIRE_EQUAL(std::string("\n\
            Invalid Date of Birth. <br /><i>This is a test validation message from the server </i>\n\
             "),
                        node.value());
    BOOST_TEST_REQUIRE(node.attributes().empty());

    BOOST_REQUIRE_EQUAL(std::string("ItemSearchResponse"), parser.rootElementName());
}

bool testAttribute(const ml::core::CXmlNode& node,
                   const std::string& key,
                   const std::string& expected) {
    std::string actual;
    if (node.attribute(key, actual) == false) {
        return false;
    }

    if (actual != expected) {
        LOG_ERROR(<< actual << ' ' << expected);
        return false;
    }

    return true;
}

BOOST_AUTO_TEST_CASE(testParse1File) {
    std::string badFileName = "./testfiles/CXmlParser_bad.xml";
    std::string goodFileName = "./testfiles/CXmlParser1.xml";

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(!parser.parseFile(badFileName));
    BOOST_TEST_REQUIRE(parser.parseFile(goodFileName));

    testParseHelper(parser);
}

BOOST_AUTO_TEST_CASE(testParse1String) {
    std::string goodString = fileToString("./testfiles/CXmlParser1.xml");

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseString(goodString));

    testParseHelper(parser);
}

BOOST_AUTO_TEST_CASE(testParse2) {
    std::string goodFileName = "./testfiles/CXmlParser2.xml";

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseFile(goodFileName));

    ml::core::CXmlParser::TXmlNodeVec nodes;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("//badpath", nodes));
    BOOST_TEST_REQUIRE(nodes.empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/syslog_parser/parsetree/expression/description", nodes));
    BOOST_REQUIRE_EQUAL(size_t(2), nodes.size());

    BOOST_REQUIRE_EQUAL(std::string("description"), nodes[0].name());
    BOOST_REQUIRE_EQUAL(std::string("Transport node error"), nodes[0].value());
    BOOST_TEST_REQUIRE(nodes[0].attributes().empty());

    BOOST_REQUIRE_EQUAL(std::string("description"), nodes[1].name());
    BOOST_REQUIRE_EQUAL(std::string("Transport read error"), nodes[1].value());
    BOOST_TEST_REQUIRE(nodes[1].attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/syslog_parser/parsetree/expression[1]/regexes/varbind/token", nodes));
    BOOST_REQUIRE_EQUAL(size_t(2), nodes.size());

    BOOST_REQUIRE_EQUAL(std::string("token"), nodes[0].name());
    BOOST_REQUIRE_EQUAL(std::string(""), nodes[0].value());
    BOOST_TEST_REQUIRE(nodes[0].attributes().empty());

    BOOST_REQUIRE_EQUAL(std::string("token"), nodes[1].name());
    BOOST_REQUIRE_EQUAL(std::string("source"), nodes[1].value());
    BOOST_TEST_REQUIRE(nodes[1].attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/syslog_parser/parsetree/expression[1]/regexes/varbind/regex", nodes));
    BOOST_REQUIRE_EQUAL(size_t(2), nodes.size());

    BOOST_REQUIRE_EQUAL(std::string("regex"), nodes[0].name());
    BOOST_REQUIRE_EQUAL(std::string("^[[:space:]]*"), nodes[0].value());
    BOOST_REQUIRE_EQUAL(size_t(2), nodes[0].attributes().size());
    BOOST_TEST_REQUIRE(testAttribute(nodes[0], "function", "default"));
    BOOST_TEST_REQUIRE(testAttribute(nodes[0], "local", "BZ"));

    BOOST_REQUIRE_EQUAL(std::string("regex"), nodes[1].name());
    BOOST_REQUIRE_EQUAL(std::string("(template[[:space:]]*<[^;:{]+>[[:space:]]*)?"),
                        nodes[1].value());
    BOOST_TEST_REQUIRE(nodes[1].attributes().empty());
}

BOOST_AUTO_TEST_CASE(testNavigate) {
    std::string goodFileName = "./testfiles/CXmlParser2.xml";

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseFile(goodFileName));

    std::string str;
    BOOST_TEST_REQUIRE(parser.navigateRoot());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("syslog_parser"), str);
    BOOST_TEST_REQUIRE(parser.navigateFirstChild());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("parsetree"), str);
    BOOST_TEST_REQUIRE(parser.navigateFirstChild());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("expression"), str);
    BOOST_TEST_REQUIRE(parser.navigateFirstChild());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("description"), str);
    BOOST_TEST_REQUIRE(parser.currentNodeValue(str));
    BOOST_REQUIRE_EQUAL(std::string("Transport node error"), str);
    BOOST_TEST_REQUIRE(parser.navigateNext());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("regexes"), str);
    BOOST_TEST_REQUIRE(parser.navigateParent());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("expression"), str);
    BOOST_TEST_REQUIRE(parser.navigateParent());
    BOOST_TEST_REQUIRE(parser.currentNodeName(str));
    BOOST_REQUIRE_EQUAL(std::string("parsetree"), str);
    BOOST_TEST_REQUIRE(!parser.navigateNext());
}

BOOST_AUTO_TEST_CASE(testParseXInclude) {
    std::string goodFileName = "./testfiles/CXmlParser3.xml";
    std::string badFileName = "./testfiles/CXmlParser4.xml";

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(!parser.parseFile(badFileName));
    BOOST_TEST_REQUIRE(parser.parseFile(goodFileName));

    ml::core::CXmlParser::TXmlNodeVec nodes;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("//badpath", nodes));
    BOOST_TEST_REQUIRE(nodes.empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/syslog_parser/parsetree/expression/description", nodes));
    BOOST_REQUIRE_EQUAL(size_t(2), nodes.size());

    BOOST_REQUIRE_EQUAL(std::string("description"), nodes[0].name());
    BOOST_REQUIRE_EQUAL(std::string("Transport node error"), nodes[0].value());
    BOOST_TEST_REQUIRE(nodes[0].attributes().empty());

    BOOST_REQUIRE_EQUAL(std::string("description"), nodes[1].name());
    BOOST_REQUIRE_EQUAL(std::string("Transport read error"), nodes[1].value());
    BOOST_TEST_REQUIRE(nodes[1].attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/syslog_parser/parsetree/expression[1]/regexes/varbind/token", nodes));
    BOOST_REQUIRE_EQUAL(size_t(2), nodes.size());

    BOOST_REQUIRE_EQUAL(std::string("token"), nodes[0].name());
    BOOST_REQUIRE_EQUAL(std::string(""), nodes[0].value());
    BOOST_TEST_REQUIRE(nodes[0].attributes().empty());

    BOOST_REQUIRE_EQUAL(std::string("token"), nodes[1].name());
    BOOST_REQUIRE_EQUAL(std::string("source"), nodes[1].value());
    BOOST_TEST_REQUIRE(nodes[1].attributes().empty());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/syslog_parser/parsetree/expression[1]/regexes/varbind/regex", nodes));
    BOOST_REQUIRE_EQUAL(size_t(2), nodes.size());

    BOOST_REQUIRE_EQUAL(std::string("regex"), nodes[0].name());
    BOOST_REQUIRE_EQUAL(std::string("^[[:space:]]*"), nodes[0].value());
    BOOST_REQUIRE_EQUAL(size_t(2), nodes[0].attributes().size());
    BOOST_TEST_REQUIRE(testAttribute(nodes[0], "function", "default"));
    BOOST_TEST_REQUIRE(testAttribute(nodes[0], "local", "BZ"));

    BOOST_REQUIRE_EQUAL(std::string("regex"), nodes[1].name());
    BOOST_REQUIRE_EQUAL(std::string("(template[[:space:]]*<[^;:{]+>[[:space:]]*)?"),
                        nodes[1].value());
    BOOST_TEST_REQUIRE(nodes[1].attributes().empty());
}

BOOST_AUTO_TEST_CASE(testParse3) {
    std::string fileName = "./testfiles/CXmlParser5.xml";

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseFile(fileName));

    ml::core::CXmlParser::TXmlNodeVec arguments;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/OperationRequest/Arguments/Argument", arguments));
    BOOST_REQUIRE_EQUAL(size_t(7), arguments.size());

    for (ml::core::CXmlParser::TXmlNodeVecItr itr = arguments.begin();
         itr != arguments.end(); ++itr) {
        if (itr->value() == "Service") {
            BOOST_TEST_REQUIRE(testAttribute(*itr, "Value", "AWSECommerceService"));
        } else if (itr->value() == "AssociateTag") {
            BOOST_TEST_REQUIRE(!testAttribute(*itr, "Value", ""));
        } else if (itr->value() == "SearchIndex") {
            BOOST_TEST_REQUIRE(testAttribute(*itr, "Value", "Books"));
        } else if (itr->value() == "Author") {
            BOOST_TEST_REQUIRE(!testAttribute(*itr, "Value", ""));
        } else if (itr->value() == "Hacasdasdcv") {
            BOOST_TEST_REQUIRE(testAttribute(*itr, "Value", "1A7XKHR5BYD0WPJVQEG2"));
        } else if (itr->value() == "Version") {
            BOOST_TEST_REQUIRE(testAttribute(*itr, "Value", "2006-06-28"));
        } else if (itr->value() == "Operation") {
            BOOST_TEST_REQUIRE(!testAttribute(*itr, "Value", ""));
        } else {
            BOOST_FAIL(itr->dump());
        }
    }
}

BOOST_AUTO_TEST_CASE(testParse4) {
    std::string fileName = "./testfiles/CXmlParser1.xml";

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseFile(fileName));

    bool valid(false);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/Items/Request/IsValid", valid));
    BOOST_TEST_REQUIRE(valid);

    BOOST_TEST_REQUIRE(
        parser.evalXPathExpression("/ItemSearchResponse/Items/TotalPages", valid));
    BOOST_TEST_REQUIRE(valid);

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/Items/Request/IsNotValid", valid));
    BOOST_TEST_REQUIRE(!valid);

    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/ItemSearchResponse/Items/Request/IsNotValidNo", valid));
    BOOST_TEST_REQUIRE(!valid);

    int i;
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/ItemSearchResponse/Items/TotalPages", i));
    BOOST_REQUIRE_EQUAL(21, i);

    // Invalid conversions
    BOOST_TEST_REQUIRE(
        !parser.evalXPathExpression("/ItemSearchResponse/Items/Request/IsValid", i));
    BOOST_TEST_REQUIRE(!parser.evalXPathExpression(
        "/ItemSearchResponse/Items/Request/ItemSearchRequest", i));
    BOOST_TEST_REQUIRE(!parser.evalXPathExpression(
        "/ItemSearchResponse/Items/Request/ItemSearchRequest/Author", i));
}

BOOST_AUTO_TEST_CASE(testParse5) {
    ml::core::CXmlParser parser;

    std::string xml = "\
<root> \
    <name1 a='sdacsdac'>value1</name1> \
    <name2>value2</name2> \
    <name3>value3</name3> \
</root>";

    BOOST_TEST_REQUIRE(parser.parseString(xml));

    ml::core::CXmlParser::TStrStrMap values;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/*", values));

    BOOST_REQUIRE_EQUAL(values.size(), size_t(3));

    ml::core::CXmlParser::TStrStrMapCItr itr = values.find("name1");
    BOOST_TEST_REQUIRE(itr != values.end());
    BOOST_REQUIRE_EQUAL(itr->second, std::string("value1"));
    itr = values.find("name2");
    BOOST_TEST_REQUIRE(itr != values.end());
    BOOST_REQUIRE_EQUAL(itr->second, std::string("value2"));
    itr = values.find("name3");
    BOOST_TEST_REQUIRE(itr != values.end());
    BOOST_REQUIRE_EQUAL(itr->second, std::string("value3"));
}

BOOST_AUTO_TEST_CASE(testParse6) {

    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <name a='sdacsdac'>value1</name> \
    <name>value2</name> \
    <name>value3</name> \
</root>";

        BOOST_TEST_REQUIRE(parser.parseString(xml));

        ml::core::CXmlParser::TStrVec values;

        BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name", values));

        BOOST_REQUIRE_EQUAL(values.size(), size_t(3));

        BOOST_REQUIRE_EQUAL(values[0], std::string("value1"));
        BOOST_REQUIRE_EQUAL(values[1], std::string("value2"));
        BOOST_REQUIRE_EQUAL(values[2], std::string("value3"));
    }
    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <names> \
    </names> \
</root>";

        BOOST_TEST_REQUIRE(parser.parseString(xml));

        ml::core::CXmlParser::TStrVec values;

        BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/names/*", values));

        BOOST_TEST_REQUIRE(values.empty());
    }
    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <name a='sdacsdac'>value1</name> \
    <name>value2</name> \
    <name>value3</name> \
</root>";

        BOOST_TEST_REQUIRE(parser.parseString(xml));

        ml::core::CXmlParser::TStrSet values;

        BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name", values));

        BOOST_REQUIRE_EQUAL(values.size(), size_t(3));
    }
    {
        ml::core::CXmlParser parser;

        std::string xml = "\
<root> \
    <name a='sdacsdac'>value1</name> \
    <name>value2</name> \
    <name>value2</name> \
</root>";

        BOOST_TEST_REQUIRE(parser.parseString(xml));

        ml::core::CXmlParser::TStrSet values;

        BOOST_TEST_REQUIRE(!parser.evalXPathExpression("/root/name", values));
    }
}

BOOST_AUTO_TEST_CASE(testConvert1) {
    ml::core::CXmlParser::TStrStrMap values;

    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("key", "<&sdacasdc"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("count", "12"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("desc", "sdac asdc asdc\nadsc\nasdc\n"));

    std::string xml;
    ml::core::CXmlParser::convert("test_convert", values, xml);

    LOG_DEBUG(<< xml);

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseString(xml));

    ml::core::CXmlNode node;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/key", node));
    BOOST_REQUIRE_EQUAL(std::string("<&sdacasdc"), node.value());
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/count", node));
    BOOST_REQUIRE_EQUAL(std::string("12"), node.value());
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/desc", node));
    BOOST_REQUIRE_EQUAL(std::string("sdac asdc asdc\nadsc\nasdc\n"), node.value());
}

BOOST_AUTO_TEST_CASE(testConvert2) {
    ml::core::CXmlParser::TStrStrMap values;

    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("key", "<&sdacasdc"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("count", "12"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("desc", "sdac asdc asdc\nadsc\nasdc\n"));

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.convert("test_convert", values));

    ml::core::CXmlNode node;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/key", node));
    BOOST_REQUIRE_EQUAL(std::string("<&sdacasdc"), node.value());
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/count", node));
    BOOST_REQUIRE_EQUAL(std::string("12"), node.value());
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/desc", node));
    BOOST_REQUIRE_EQUAL(std::string("sdac asdc asdc\nadsc\nasdc\n"), node.value());
}

BOOST_AUTO_TEST_CASE(testConvert3) {
    ml::core::CXmlParser::TStrStrMap values;

    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("key", "<&sdacasdc"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("count", "1"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("field@name=idle cpu %", "96"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("field@name=user cpu %", "3"));
    values.insert(ml::core::CXmlParser::TStrStrMap::value_type("field@name=system cpu %", "1"));

    std::string xml;
    ml::core::CXmlParser::convert("test_convert", values, xml);

    LOG_DEBUG(<< xml);

    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseString(xml));

    ml::core::CXmlNode node;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/key", node));
    BOOST_REQUIRE_EQUAL(std::string("<&sdacasdc"), node.value());
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/test_convert/count", node));
    BOOST_REQUIRE_EQUAL(std::string("1"), node.value());
    BOOST_TEST_REQUIRE(
        parser.evalXPathExpression("/test_convert/field[@name='idle cpu %']", node));
    BOOST_REQUIRE_EQUAL(std::string("96"), node.value());
    BOOST_TEST_REQUIRE(
        parser.evalXPathExpression("/test_convert/field[@name='user cpu %']", node));
    BOOST_REQUIRE_EQUAL(std::string("3"), node.value());
    BOOST_TEST_REQUIRE(parser.evalXPathExpression(
        "/test_convert/field[@name='system cpu %']", node));
    BOOST_REQUIRE_EQUAL(std::string("1"), node.value());
}

BOOST_AUTO_TEST_CASE(testAddNewChildNode) {
    ml::core::CXmlParser parser;

    std::string xml = "\
<root> \
    <name1 a='sdacsdac'>value1</name1> \
    <name2>value2</name2> \
    <name3>value3</name3> \
</root>";

    BOOST_TEST_REQUIRE(parser.parseString(xml));

    std::string value;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name1", value));
    BOOST_REQUIRE_EQUAL(std::string("value1"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name2", value));
    BOOST_REQUIRE_EQUAL(std::string("value2"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name3", value));
    BOOST_REQUIRE_EQUAL(std::string("value3"), value);

    BOOST_TEST_REQUIRE(parser.addNewChildNode("name4", "value4"));

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name4", value));
    BOOST_REQUIRE_EQUAL(std::string("value4"), value);
}

BOOST_AUTO_TEST_CASE(testSetRootNode) {

    {
        ml::core::CXmlParser parser;

        BOOST_TEST_REQUIRE(parser.setRootNode("root"));

        BOOST_TEST_REQUIRE(parser.addNewChildNode("name1", "value1"));
        BOOST_TEST_REQUIRE(parser.addNewChildNode("name2", "value2"));

        std::string value;

        BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name1", value));
        BOOST_REQUIRE_EQUAL(std::string("value1"), value);
        BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name2", value));
        BOOST_REQUIRE_EQUAL(std::string("value2"), value);
    }
    {
        ml::core::CXmlParser parser;

        BOOST_TEST_REQUIRE(parser.setRootNode("root"));

        BOOST_TEST_REQUIRE(parser.addNewChildNode("name", "value1"));
        BOOST_TEST_REQUIRE(parser.addNewChildNode("name", "value2"));

        parser.dumpToStdout();
    }
}

BOOST_AUTO_TEST_CASE(testDump) {
    std::string fileName = "./testfiles/CXmlParser1.xml";

    ml::core::CXmlParser parser1;
    BOOST_TEST_REQUIRE(parser1.parseFile(fileName));
    testParseHelper(parser1);

    std::string expected = parser1.dumpToString();

    ml::core::CXmlParser parser2;
    BOOST_TEST_REQUIRE(parser2.parseString(expected));
    testParseHelper(parser2);
}

BOOST_AUTO_TEST_CASE(testMakeValidName) {
    BOOST_REQUIRE_EQUAL(std::string("name"), ml::core::CXmlParser::makeValidName("name"));
    BOOST_REQUIRE_EQUAL(std::string("name1"), ml::core::CXmlParser::makeValidName("name1"));
    BOOST_REQUIRE_EQUAL(std::string("_name"), ml::core::CXmlParser::makeValidName("1name"));
    BOOST_REQUIRE_EQUAL(std::string("name_2"), ml::core::CXmlParser::makeValidName("name/2"));
    BOOST_REQUIRE_EQUAL(std::string("_name_"), ml::core::CXmlParser::makeValidName("_name_"));
    BOOST_REQUIRE_EQUAL(std::string("__cencl01b_System_System_Calls_sec"),
                        ml::core::CXmlParser::makeValidName("\\\\cencl01b\\System\\System Calls/sec"));
}

BOOST_AUTO_TEST_CASE(testChangeChild) {
    ml::core::CXmlParser parser;

    BOOST_TEST_REQUIRE(parser.setRootNode("root"));
    BOOST_TEST_REQUIRE(parser.addNewChildNode("name1", "value1"));
    BOOST_TEST_REQUIRE(parser.addNewChildNode("name2", "value2"));
    BOOST_TEST_REQUIRE(parser.addNewChildNode("name3", "value3"));

    LOG_DEBUG(<< parser.dumpToString());

    std::string value;

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name1", value));
    BOOST_REQUIRE_EQUAL(std::string("value1"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name2", value));
    BOOST_REQUIRE_EQUAL(std::string("value2"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name3", value));
    BOOST_REQUIRE_EQUAL(std::string("value3"), value);

    // Change each of the values in turn, checking state after each change
    BOOST_TEST_REQUIRE(parser.changeChildNodeValue("name2", "changed2"));

    LOG_DEBUG(<< parser.dumpToString());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name1", value));
    BOOST_REQUIRE_EQUAL(std::string("value1"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name2", value));
    BOOST_REQUIRE_EQUAL(std::string("changed2"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name3", value));
    BOOST_REQUIRE_EQUAL(std::string("value3"), value);

    BOOST_TEST_REQUIRE(parser.changeChildNodeValue("name1", "changed1"));

    LOG_DEBUG(<< parser.dumpToString());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name1", value));
    BOOST_REQUIRE_EQUAL(std::string("changed1"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name2", value));
    BOOST_REQUIRE_EQUAL(std::string("changed2"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name3", value));
    BOOST_REQUIRE_EQUAL(std::string("value3"), value);

    BOOST_TEST_REQUIRE(parser.changeChildNodeValue("name3", "changed3"));

    LOG_DEBUG(<< parser.dumpToString());

    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name1", value));
    BOOST_REQUIRE_EQUAL(std::string("changed1"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name2", value));
    BOOST_REQUIRE_EQUAL(std::string("changed2"), value);
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/root/name3", value));
    BOOST_REQUIRE_EQUAL(std::string("changed3"), value);
}

BOOST_AUTO_TEST_CASE(testHugeDoc) {
    // libxml2 can exhibit O(n^2.42) behaviour if the xmlXPathOrderDocElems()
    // function hasn't been called on the document.  Obviously this only shows
    // up as a problem in huge XML documents.

    // First, create an enormous XML document
    std::string fileName(ml::test::CTestTmpDir::tmpDir() + "/huge.xml");
    std::ofstream ofs(fileName.c_str());
    BOOST_REQUIRE_MESSAGE(ofs.is_open(), fileName + " is not open");

    ofs << "<nodes>" << std::endl;

    static const size_t NUM_NODES(300000);
    for (size_t count = 1; count <= NUM_NODES; ++count) {
        ofs << "    <node>" << count << "</node>" << std::endl;
    }

    ofs << "</nodes>" << std::endl;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting huge XPath test at " << ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlParser parser;
    ml::core::CXmlParser::TStrSet valueSet;

    BOOST_TEST_REQUIRE(parser.parseFile(fileName));

    // NB: If xmlXPathOrderDocElems() hasn't been called, this will take an
    // astronomical amount of time - don't wait more than a minute for it!
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("/nodes/node", valueSet));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished huge XPath test at " << ml::core::CTimeUtils::toTimeString(end));

    BOOST_REQUIRE_EQUAL(NUM_NODES, valueSet.size());

    LOG_INFO(<< "Applying an XPath to a node set with " << NUM_NODES
             << " nodes took " << (end - start) << " seconds");

    ::remove(fileName.c_str());
}

BOOST_AUTO_TEST_CASE(testParseSpeed) {
    static const size_t TEST_SIZE(25000);

    std::string testString(fileToString("./testfiles/CXmlParser2.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting parse speed test at "
             << ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlNodeWithChildrenPool nodePool;

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseString(testString));

        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        BOOST_TEST_REQUIRE(parser.toNodeHierarchy(nodePool, rootNodePtr));

        BOOST_TEST_REQUIRE(rootNodePtr != nullptr);

        nodePool.recycle(rootNodePtr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished parse speed test at " << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Parsing " << TEST_SIZE << " documents took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testComplexXPath) {
    ml::core::CXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseFile("testfiles/withNs.xml"));

    bool disabled(false);

    // This convoluted query is for XML schemas that
    // have a default namespace but don't give it a name!
    BOOST_TEST_REQUIRE(parser.evalXPathExpression("//*[local-name()='title' and .='ml']/..//*[local-name()='key' and @name='disabled']",
                                                  disabled));
    BOOST_REQUIRE_EQUAL(true, disabled);
}

BOOST_AUTO_TEST_SUITE_END()
