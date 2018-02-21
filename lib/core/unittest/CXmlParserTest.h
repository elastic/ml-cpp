/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
        void testParse1File(void);
        void testParse1String(void);
        void testParse2(void);
        void testNavigate(void);
        void testParseXInclude(void);
        void testParse3(void);
        void testParse4(void);
        void testParse5(void);
        void testParse6(void);
        void testConvert1(void);
        void testConvert2(void);
        void testConvert3(void);
        void testConvert4(void);
        void testAddNewChildNode(void);
        void testSetRootNode(void);
        void testDump(void);
        void testMakeValidName(void);
        void testChangeChild(void);
        void testHugeDoc(void);
        void testParseSpeed(void);
        void testConvertSpeed(void);
        void testComplexXPath(void);

        static CppUnit::Test *suite();

    private:
        static void testParse1(const ml::core::CXmlParser &parser);

        static std::string fileToString(const std::string &fileName);

        static bool testAttribute(const ml::core::CXmlNode &node,
                                  const std::string &key,
                                  const std::string &expected);
};

#endif // INCLUDED_CXmlParserTest_h

