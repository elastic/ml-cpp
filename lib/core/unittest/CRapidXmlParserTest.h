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
    void testParse1(void);
    void testParse2(void);
    void testNavigate(void);
    void testConvert(void);
    void testDump(void);
    void testParseSpeed(void);
    void testConvertSpeed(void);

    static CppUnit::Test* suite();

    static ml::core::CXmlNodeWithChildren::TXmlNodeWithChildrenP makeTestNodeHierarchy(void);

private:
    static void testParse1(const ml::core::CRapidXmlParser& parser);

    static std::string fileToString(const std::string& fileName);

    static bool testAttribute(const ml::core::CXmlNode& node, const std::string& key, const std::string& expected);
};

#endif // INCLUDED_CRapidXmlParserTest_h
