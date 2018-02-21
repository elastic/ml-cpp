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
#include "CPatternSetTest.h"

#include <core/CLogger.h>
#include <core/CPatternSet.h>


using namespace ml;
using namespace core;

CppUnit::Test *CPatternSetTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CPatternSetTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testInitFromJson_GivenInvalidJson",
                              &CPatternSetTest::testInitFromJson_GivenInvalidJson) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testInitFromJson_GivenNonArray",
                              &CPatternSetTest::testInitFromJson_GivenNonArray) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testInitFromJson_GivenArrayWithNonStringItem",
                              &CPatternSetTest::testInitFromJson_GivenArrayWithNonStringItem) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testInitFromJson_GivenArrayWithDuplicates",
                              &CPatternSetTest::testInitFromJson_GivenArrayWithDuplicates) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testContains_GivenFullMatchKeys",
                              &CPatternSetTest::testContains_GivenFullMatchKeys) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testContains_GivenPrefixKeys",
                              &CPatternSetTest::testContains_GivenPrefixKeys) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testContains_GivenSuffixKeys",
                              &CPatternSetTest::testContains_GivenSuffixKeys) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testContains_GivenContainsKeys",
                              &CPatternSetTest::testContains_GivenContainsKeys) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testContains_GivenMixedKeys",
                              &CPatternSetTest::testContains_GivenMixedKeys) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CPatternSetTest>(
                              "CPatternSetTest::testClear",
                              &CPatternSetTest::testClear) );

    return suiteOfTests;
}

void CPatternSetTest::testInitFromJson_GivenInvalidJson(void)
{
    std::string json("[\"foo\"");
    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json) == false);
}

void CPatternSetTest::testInitFromJson_GivenNonArray(void)
{
    std::string json("{}");
    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json) == false);
}

void CPatternSetTest::testInitFromJson_GivenArrayWithNonStringItem(void)
{
    std::string json("[\"foo\", 3]");
    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json) == false);
}

void CPatternSetTest::testInitFromJson_GivenArrayWithDuplicates(void)
{
    std::string json("[\"foo\",\"foo\", \"bar\", \"bar\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));

    CPPUNIT_ASSERT(set.contains("foo"));
    CPPUNIT_ASSERT(set.contains("bar"));
}

void CPatternSetTest::testContains_GivenFullMatchKeys(void)
{
    std::string json("[\"foo\",\"bar\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));

    CPPUNIT_ASSERT(set.contains("foo"));
    CPPUNIT_ASSERT(set.contains("bar"));
    CPPUNIT_ASSERT(set.contains("nonItem") == false);
}

void CPatternSetTest::testContains_GivenPrefixKeys(void)
{
    std::string json("[\"abc*\", \"foo*\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));

    CPPUNIT_ASSERT(set.contains("abc"));
    CPPUNIT_ASSERT(set.contains("abcd"));
    CPPUNIT_ASSERT(set.contains("zabc") == false);
    CPPUNIT_ASSERT(set.contains("foo"));
    CPPUNIT_ASSERT(set.contains("foo_"));
    CPPUNIT_ASSERT(set.contains("_foo") == false);
}

void CPatternSetTest::testContains_GivenSuffixKeys(void)
{
    std::string json("[\"*xyz\", \"*foo\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));

    CPPUNIT_ASSERT(set.contains("xyz"));
    CPPUNIT_ASSERT(set.contains("aaaaxyz"));
    CPPUNIT_ASSERT(set.contains("xyza") == false);
    CPPUNIT_ASSERT(set.contains("foo"));
    CPPUNIT_ASSERT(set.contains("_foo"));
    CPPUNIT_ASSERT(set.contains("foo_") == false);
}

void CPatternSetTest::testContains_GivenContainsKeys(void)
{
    std::string json("[\"*foo*\", \"*456*\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));

    CPPUNIT_ASSERT(set.contains("foo"));
    CPPUNIT_ASSERT(set.contains("_foo_"));
    CPPUNIT_ASSERT(set.contains("_foo"));
    CPPUNIT_ASSERT(set.contains("foo_"));
    CPPUNIT_ASSERT(set.contains("_fo_") == false);
    CPPUNIT_ASSERT(set.contains("456"));
    CPPUNIT_ASSERT(set.contains("123456"));
    CPPUNIT_ASSERT(set.contains("456789"));
    CPPUNIT_ASSERT(set.contains("123456789"));
    CPPUNIT_ASSERT(set.contains("12346789") == false);
}

void CPatternSetTest::testContains_GivenMixedKeys(void)
{
    std::string json("[\"foo\", \"foo*\", \"*foo\", \"*foo*\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));

    CPPUNIT_ASSERT(set.contains("foo"));
    CPPUNIT_ASSERT(set.contains("_foo_"));
    CPPUNIT_ASSERT(set.contains("_foo"));
    CPPUNIT_ASSERT(set.contains("foo_"));
    CPPUNIT_ASSERT(set.contains("fo") == false);
}

void CPatternSetTest::testClear(void)
{
    std::string json("[\"foo\"]");

    CPatternSet set;
    CPPUNIT_ASSERT(set.initFromJson(json));
    CPPUNIT_ASSERT(set.contains("foo"));

    set.clear();

    CPPUNIT_ASSERT(set.contains("foo") == false);
}
