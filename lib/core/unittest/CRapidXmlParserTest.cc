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
#include "CRapidXmlParserTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CTimeUtils.h>
#include <core/CXmlNodeWithChildrenPool.h>

#include <fstream>

CppUnit::Test* CRapidXmlParserTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CRapidXmlParserTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testParse1",
                                                     &CRapidXmlParserTest::testParse1));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testParse2",
                                                     &CRapidXmlParserTest::testParse2));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testNavigate",
                                                     &CRapidXmlParserTest::testNavigate));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testConvert",
                                                     &CRapidXmlParserTest::testConvert));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testDump",
                                                     &CRapidXmlParserTest::testDump));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testParseSpeed",
                                                     &CRapidXmlParserTest::testParseSpeed));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CRapidXmlParserTest>("CRapidXmlParserTest::testConvertSpeed",
                                                     &CRapidXmlParserTest::testConvertSpeed));

    return suiteOfTests;
}

void CRapidXmlParserTest::testParse1(void) {
    std::string goodString = CRapidXmlParserTest::fileToString("./testfiles/CXmlParser1.xml");

    ml::core::CRapidXmlParser parser;

    CPPUNIT_ASSERT(parser.parseString(goodString));

    this->testParse1(parser);
}

void CRapidXmlParserTest::testParse2(void) {
    std::string goodString = CRapidXmlParserTest::fileToString("./testfiles/CXmlParser2.xml");

    ml::core::CRapidXmlParser parser;

    CPPUNIT_ASSERT(parser.parseString(goodString));

    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;

    CPPUNIT_ASSERT(parser.toNodeHierarchy(rootNodePtr));
    CPPUNIT_ASSERT(rootNodePtr != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("syslog_parser"), rootNodePtr->name());
    CPPUNIT_ASSERT_EQUAL(rootNodePtr->name(), parser.rootElementName());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& parseTree = rootNodePtr->children();
    CPPUNIT_ASSERT_EQUAL(size_t(1), parseTree.size());
    CPPUNIT_ASSERT(parseTree[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("parsetree"), parseTree[0]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& expression = parseTree[0]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(2), expression.size());
    CPPUNIT_ASSERT(expression[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("expression"), expression[0]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& descriptionAndRegexes =
        expression[0]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(2), descriptionAndRegexes.size());
    CPPUNIT_ASSERT(descriptionAndRegexes[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("description"), descriptionAndRegexes[0]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("Transport node error"), descriptionAndRegexes[0]->value());
    CPPUNIT_ASSERT(descriptionAndRegexes[1] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("regexes"), descriptionAndRegexes[1]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& varbind =
        descriptionAndRegexes[1]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(2), varbind.size());
    CPPUNIT_ASSERT(varbind[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("varbind"), varbind[0]->name());
    CPPUNIT_ASSERT(varbind[1] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("varbind"), varbind[1]->name());

    // Test attributes
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& tokenAndRegex0 = varbind[0]->children();
    CPPUNIT_ASSERT_EQUAL(std::string("token"), tokenAndRegex0[0]->name());
    CPPUNIT_ASSERT_EQUAL(std::string(""), tokenAndRegex0[0]->value());
    CPPUNIT_ASSERT_EQUAL(std::string("regex"), tokenAndRegex0[1]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("^[[:space:]]*"), tokenAndRegex0[1]->value());
    CPPUNIT_ASSERT(this->testAttribute(*(tokenAndRegex0[1]), "function", "default"));
    CPPUNIT_ASSERT(this->testAttribute(*(tokenAndRegex0[1]), "local", "BZ"));

    // Test CDATA
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& tokenAndRegex1 = varbind[1]->children();
    CPPUNIT_ASSERT_EQUAL(std::string("token"), tokenAndRegex1[0]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("source"), tokenAndRegex1[0]->value());
    CPPUNIT_ASSERT_EQUAL(std::string("regex"), tokenAndRegex1[1]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("(template[[:space:]]*<[^;:{]+>[[:space:]]*)?"),
                         tokenAndRegex1[1]->value());
}

void CRapidXmlParserTest::testNavigate(void) {
    std::string goodString = CRapidXmlParserTest::fileToString("./testfiles/CXmlParser2.xml");

    ml::core::CRapidXmlParser parser;

    CPPUNIT_ASSERT(parser.parseString(goodString));

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

ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP
CRapidXmlParserTest::makeTestNodeHierarchy(void) {
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

void CRapidXmlParserTest::testConvert(void) {
    // Use a standard node hierarchy to allow for comparison with the
    // standards-compliant XML parser
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(
        CRapidXmlParserTest::makeTestNodeHierarchy());

    std::string converted;
    ml::core::CRapidXmlParser::convert(*root, converted);

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
    CPPUNIT_ASSERT(converted.find("<empty/>") != std::string::npos ||
                   converted.find("<empty></empty>") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("<dual ") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("first") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("second") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("attribute") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("got") != std::string::npos);
    CPPUNIT_ASSERT(converted.find("</dual>") != std::string::npos);
}

void CRapidXmlParserTest::testDump(void) {
    std::string goodString = CRapidXmlParserTest::fileToString("./testfiles/CXmlParser1.xml");

    ml::core::CRapidXmlParser parser1;

    CPPUNIT_ASSERT(parser1.parseString(goodString));

    this->testParse1(parser1);

    std::string expected = parser1.dumpToString();

    ml::core::CRapidXmlParser parser2;
    CPPUNIT_ASSERT(parser2.parseString(expected));
    this->testParse1(parser2);
}

void CRapidXmlParserTest::testParse1(const ml::core::CRapidXmlParser& parser) {
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;

    CPPUNIT_ASSERT(parser.toNodeHierarchy(rootNodePtr));
    CPPUNIT_ASSERT(rootNodePtr != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("ItemSearchResponse"), rootNodePtr->name());
    CPPUNIT_ASSERT_EQUAL(rootNodePtr->name(), parser.rootElementName());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& firstLevelChildren =
        rootNodePtr->children();
    CPPUNIT_ASSERT_EQUAL(size_t(2), firstLevelChildren.size());
    CPPUNIT_ASSERT(firstLevelChildren[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("OperationRequest"), firstLevelChildren[0]->name());
    CPPUNIT_ASSERT(firstLevelChildren[1] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("Items"), firstLevelChildren[1]->name());

    const ml::core::CXmlNodeWithChildren::TChildNodePVec& opReqChildren =
        firstLevelChildren[0]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(4), opReqChildren.size());
    CPPUNIT_ASSERT(opReqChildren[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("HTTPHeaders"), opReqChildren[0]->name());
    CPPUNIT_ASSERT(opReqChildren[1] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("RequestId"), opReqChildren[1]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("18CZWZFXKSV8F601AGMF"), opReqChildren[1]->value());
    CPPUNIT_ASSERT(opReqChildren[2] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("Arguments"), opReqChildren[2]->name());
    CPPUNIT_ASSERT(opReqChildren[3] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("RequestProcessingTime"), opReqChildren[3]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("1.05041599273682"), opReqChildren[3]->value());

    // Test CDATA
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& itemsChildren =
        firstLevelChildren[1]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(13), itemsChildren.size());
    CPPUNIT_ASSERT(itemsChildren[3] != 0);
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& item3Children =
        itemsChildren[3]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(4), item3Children.size());
    CPPUNIT_ASSERT(item3Children[0] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("msg"), item3Children[0]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("\n\
            Invalid Date of Birth. <br /><i>This is a test validation message from the server </i>\n\
             "),
                         item3Children[0]->value());

    // Test escaped ampersand
    CPPUNIT_ASSERT(itemsChildren[10] != 0);
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& item10Children =
        itemsChildren[10]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(3), item10Children.size());
    CPPUNIT_ASSERT(item10Children[2] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("ItemAttributes"), item10Children[2]->name());
    const ml::core::CXmlNodeWithChildren::TChildNodePVec& itemAttributesChildren =
        item10Children[2]->children();
    CPPUNIT_ASSERT_EQUAL(size_t(4), itemAttributesChildren.size());
    CPPUNIT_ASSERT(itemAttributesChildren[1] != 0);
    CPPUNIT_ASSERT_EQUAL(std::string("Manufacturer"), itemAttributesChildren[1]->name());
    CPPUNIT_ASSERT_EQUAL(std::string("William Morrow & Company"),
                         itemAttributesChildren[1]->value());
}

std::string CRapidXmlParserTest::fileToString(const std::string& fileName) {
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

bool CRapidXmlParserTest::testAttribute(const ml::core::CXmlNode& node,
                                        const std::string& key,
                                        const std::string& expected) {
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

void CRapidXmlParserTest::testParseSpeed(void) {
    static const size_t TEST_SIZE(25000);

    std::string testString(CRapidXmlParserTest::fileToString("./testfiles/CXmlParser2.xml"));

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting parse speed test at " << ml::core::CTimeUtils::toTimeString(start));

    ml::core::CXmlNodeWithChildrenPool nodePool;

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseString(testString));

        ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP rootNodePtr;
        CPPUNIT_ASSERT(parser.toNodeHierarchy(nodePool, rootNodePtr));

        CPPUNIT_ASSERT(rootNodePtr != 0);

        nodePool.recycle(rootNodePtr);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished parse speed test at " << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO("Parsing " << TEST_SIZE << " documents took " << (end - start) << " seconds");
}

void CRapidXmlParserTest::testConvertSpeed(void) {
    static const size_t TEST_SIZE(100000);

    // Use a standard node hierarchy to allow for comparison with the
    // standards-compliant XML parser
    ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(
        CRapidXmlParserTest::makeTestNodeHierarchy());

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting convert speed test at " << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::string converted;
        ml::core::CRapidXmlParser::convert(*root, converted);
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished convert speed test at " << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO("Converting " << TEST_SIZE << " documents took " << (end - start) << " seconds");
}
