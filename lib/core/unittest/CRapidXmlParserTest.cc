/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CTimeUtils.h>
#include <core/CXmlNodeWithChildren.h>
#include <core/CXmlNodeWithChildrenPool.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CRapidXmlParserTest)

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

ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP makeTestNodeHierarchy() {
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(
        new ml::core::CXmlNodeWithChildren("root"));

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP id(
        new ml::core::CXmlNodeWithChildren("id", "123"));
    root->addChildP(id);

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP parent(
        new ml::core::CXmlNodeWithChildren("parent"));
    root->addChildP(parent);

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP child(
        new ml::core::CXmlNodeWithChildren("child", "boo!"));
    parent->addChildP(child);

    ml::core::CXmlNode::TStrStrMap attrMap;
    attrMap["attr1"] = "you & me";

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP child2(
        new ml::core::CXmlNodeWithChildren("child", "2nd", attrMap));
    parent->addChildP(child2);

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP empty(
        new ml::core::CXmlNodeWithChildren("empty"));
    root->addChildP(empty);

    attrMap["attr1"] = "first 'attribute'";
    attrMap["attr2"] = "second \"attribute\"";
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP dualAttr(
        new ml::core::CXmlNodeWithChildren("dual", "I've got <2> attributes", attrMap));
    root->addChildP(dualAttr);

    return root;
}

void testParseHelper(const ml::core::CRapidXmlParser& parser) {
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;

    BOOST_TEST_REQUIRE(parser.toNodeHierarchy(rootNodePtr));
    BOOST_TEST_REQUIRE(rootNodePtr != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("ItemSearchResponse"), rootNodePtr->name());
    BOOST_REQUIRE_EQUAL(rootNodePtr->name(), parser.rootElementName());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& firstLevelChildren =
        rootNodePtr->children();
    BOOST_REQUIRE_EQUAL(size_t(2), firstLevelChildren.size());
    BOOST_TEST_REQUIRE(firstLevelChildren[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("OperationRequest"), firstLevelChildren[0]->name());
    BOOST_TEST_REQUIRE(firstLevelChildren[1] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("Items"), firstLevelChildren[1]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& opReqChildren =
        firstLevelChildren[0]->children();
    BOOST_REQUIRE_EQUAL(size_t(4), opReqChildren.size());
    BOOST_TEST_REQUIRE(opReqChildren[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("HTTPHeaders"), opReqChildren[0]->name());
    BOOST_TEST_REQUIRE(opReqChildren[1] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("RequestId"), opReqChildren[1]->name());
    BOOST_REQUIRE_EQUAL(std::string("18CZWZFXKSV8F601AGMF"), opReqChildren[1]->value());
    BOOST_TEST_REQUIRE(opReqChildren[2] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("Arguments"), opReqChildren[2]->name());
    BOOST_TEST_REQUIRE(opReqChildren[3] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("RequestProcessingTime"), opReqChildren[3]->name());
    BOOST_REQUIRE_EQUAL(std::string("1.05041599273682"), opReqChildren[3]->value());

    // Test CDATA
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& itemsChildren =
        firstLevelChildren[1]->children();
    BOOST_REQUIRE_EQUAL(size_t(13), itemsChildren.size());

    BOOST_TEST_REQUIRE(itemsChildren[3] != nullptr);
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& item3Children =
        itemsChildren[3]->children();
    BOOST_REQUIRE_EQUAL(size_t(4), item3Children.size());
    BOOST_TEST_REQUIRE(item3Children[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("msg"), item3Children[0]->name());
    BOOST_REQUIRE_EQUAL(std::string("\n\
            Invalid Date of Birth. <br /><i>This is a test validation message from the server </i>\n\
             "),
                        item3Children[0]->value());

    // Test escaped ampersand
    BOOST_TEST_REQUIRE(itemsChildren[10] != nullptr);
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& item10Children =
        itemsChildren[10]->children();
    BOOST_REQUIRE_EQUAL(size_t(3), item10Children.size());
    BOOST_TEST_REQUIRE(item10Children[2] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("ItemAttributes"), item10Children[2]->name());
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& itemAttributesChildren =
        item10Children[2]->children();
    BOOST_REQUIRE_EQUAL(size_t(4), itemAttributesChildren.size());
    BOOST_TEST_REQUIRE(itemAttributesChildren[1] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("Manufacturer"), itemAttributesChildren[1]->name());
    BOOST_REQUIRE_EQUAL(std::string("William Morrow & Company"),
                        itemAttributesChildren[1]->value());
}

BOOST_AUTO_TEST_CASE(testParse1) {
    std::string goodString = fileToString("./testfiles/CXmlParser1.xml");

    ml::core::CRapidXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseString(goodString));

    testParseHelper(parser);
}

BOOST_AUTO_TEST_CASE(testParse2) {
    std::string goodString = fileToString("./testfiles/CXmlParser2.xml");

    ml::core::CRapidXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseString(goodString));

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;

    BOOST_TEST_REQUIRE(parser.toNodeHierarchy(rootNodePtr));
    BOOST_TEST_REQUIRE(rootNodePtr != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("syslog_parser"), rootNodePtr->name());
    BOOST_REQUIRE_EQUAL(rootNodePtr->name(), parser.rootElementName());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& parseTree =
        rootNodePtr->children();
    BOOST_REQUIRE_EQUAL(size_t(1), parseTree.size());
    BOOST_TEST_REQUIRE(parseTree[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("parsetree"), parseTree[0]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& expression =
        parseTree[0]->children();
    BOOST_REQUIRE_EQUAL(size_t(2), expression.size());
    BOOST_TEST_REQUIRE(expression[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("expression"), expression[0]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& descriptionAndRegexes =
        expression[0]->children();
    BOOST_REQUIRE_EQUAL(size_t(2), descriptionAndRegexes.size());
    BOOST_TEST_REQUIRE(descriptionAndRegexes[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("description"), descriptionAndRegexes[0]->name());
    BOOST_REQUIRE_EQUAL(std::string("Transport node error"),
                        descriptionAndRegexes[0]->value());
    BOOST_TEST_REQUIRE(descriptionAndRegexes[1] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("regexes"), descriptionAndRegexes[1]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& varbind =
        descriptionAndRegexes[1]->children();
    BOOST_REQUIRE_EQUAL(size_t(2), varbind.size());
    BOOST_TEST_REQUIRE(varbind[0] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("varbind"), varbind[0]->name());
    BOOST_TEST_REQUIRE(varbind[1] != nullptr);
    BOOST_REQUIRE_EQUAL(std::string("varbind"), varbind[1]->name());

    // Test attributes
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& tokenAndRegex0 =
        varbind[0]->children();
    BOOST_REQUIRE_EQUAL(std::string("token"), tokenAndRegex0[0]->name());
    BOOST_REQUIRE_EQUAL(std::string(""), tokenAndRegex0[0]->value());
    BOOST_REQUIRE_EQUAL(std::string("regex"), tokenAndRegex0[1]->name());
    BOOST_REQUIRE_EQUAL(std::string("^[[:space:]]*"), tokenAndRegex0[1]->value());
    BOOST_TEST_REQUIRE(testAttribute(*(tokenAndRegex0[1]), "function", "default"));
    BOOST_TEST_REQUIRE(testAttribute(*(tokenAndRegex0[1]), "local", "BZ"));

    // Test CDATA
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& tokenAndRegex1 =
        varbind[1]->children();
    BOOST_REQUIRE_EQUAL(std::string("token"), tokenAndRegex1[0]->name());
    BOOST_REQUIRE_EQUAL(std::string("source"), tokenAndRegex1[0]->value());
    BOOST_REQUIRE_EQUAL(std::string("regex"), tokenAndRegex1[1]->name());
    BOOST_REQUIRE_EQUAL(std::string("(template[[:space:]]*<[^;:{]+>[[:space:]]*)?"),
                        tokenAndRegex1[1]->value());
}

BOOST_AUTO_TEST_CASE(testNavigate) {
    std::string goodString = fileToString("./testfiles/CXmlParser2.xml");

    ml::core::CRapidXmlParser parser;

    BOOST_TEST_REQUIRE(parser.parseString(goodString));

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

BOOST_AUTO_TEST_CASE(testConvert) {
    // Use a standard node hierarchy to allow for comparison with the
    // standards-compliant XML parser
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(makeTestNodeHierarchy());

    std::string converted;
    ml::core::CRapidXmlParser::convert(*root, converted);

    LOG_DEBUG(<< "Converted node hierarchy is:\n" << converted);

    BOOST_TEST_REQUIRE(converted.find("<root>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("</root>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("<id>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("123") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("</id>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("<parent>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("</parent>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("<child>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("boo!") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("2nd") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("</child>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("<child ") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("&amp; ") != std::string::npos);
    BOOST_REQUIRE(converted.find("<empty/>") != std::string::npos ||
                  converted.find("<empty></empty>") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("<dual ") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("first") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("second") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("attribute") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("got") != std::string::npos);
    BOOST_TEST_REQUIRE(converted.find("</dual>") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testDump) {
    std::string goodString = fileToString("./testfiles/CXmlParser1.xml");

    ml::core::CRapidXmlParser parser1;

    BOOST_TEST_REQUIRE(parser1.parseString(goodString));

    testParseHelper(parser1);

    std::string expected = parser1.dumpToString();

    ml::core::CRapidXmlParser parser2;
    BOOST_TEST_REQUIRE(parser2.parseString(expected));
    testParseHelper(parser2);
}

BOOST_AUTO_TEST_CASE(testParseSpeed) {
    static const size_t TEST_SIZE(25000);

    std::string testString(fileToString("./testfiles/CXmlParser2.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting parse speed test at "
             << ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlNodeWithChildrenPool nodePool;

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CRapidXmlParser parser;
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

BOOST_AUTO_TEST_CASE(testConvertSpeed) {
    static const size_t TEST_SIZE(100000);

    // Use a standard node hierarchy to allow for comparison with the
    // standards-compliant XML parser
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(makeTestNodeHierarchy());

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting convert speed test at "
             << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::string converted;
        ml::core::CRapidXmlParser::convert(*root, converted);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished convert speed test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Converting " << TEST_SIZE << " documents took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_SUITE_END()
