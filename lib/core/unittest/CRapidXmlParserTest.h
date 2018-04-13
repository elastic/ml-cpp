/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CRapidXmlParserTest_h
#define INCLUDED_CRapidXmlParserTest_h

#include <core/CXmlNodeWithChildren.h>

#include <cppunit/extensions/HelperMacros.h>

#include <string>

namespace ml {
namespace core {
class CRapidXmlParser;
class CXmlNode;
}
}

class CRapidXmlParserTest : public CppUnit::TestFixture {
public:
    void testParse1();
    void testParse2();
    void testNavigate();
    void testConvert();
    void testDump();
    void testParseSpeed();
    void testConvertSpeed();

    static CppUnit::Test* suite();

    static ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP makeTestNodeHierarchy();

private:
    static void testParse1(const ml::core::CRapidXmlParser& parser);

    static std::string fileToString(const std::string& fileName);

    static bool testAttribute(const ml::core::CXmlNode& node,
                              const std::string& key,
                              const std::string& expected);
};

#endif // INCLUDED_CRapidXmlParserTest_h
