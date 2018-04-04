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
#ifndef INCLUDED_CXmlParserTest_h
#define INCLUDED_CXmlParserTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <string>

namespace ml
{
namespace core
{
class CXmlNode;
class CXmlParser;
}
}

class CXmlParserTest : public CppUnit::TestFixture
{
    public:
        void testParse1File();
        void testParse1String();
        void testParse2();
        void testNavigate();
        void testParseXInclude();
        void testParse3();
        void testParse4();
        void testParse5();
        void testParse6();
        void testConvert1();
        void testConvert2();
        void testConvert3();
        void testConvert4();
        void testAddNewChildNode();
        void testSetRootNode();
        void testDump();
        void testMakeValidName();
        void testChangeChild();
        void testHugeDoc();
        void testParseSpeed();
        void testConvertSpeed();
        void testComplexXPath();

        static CppUnit::Test *suite();

    private:
        static void testParse1(const ml::core::CXmlParser &parser);

        static std::string fileToString(const std::string &fileName);

        static bool testAttribute(const ml::core::CXmlNode &node,
                                  const std::string &key,
                                  const std::string &expected);
};

#endif // INCLUDED_CXmlParserTest_h

