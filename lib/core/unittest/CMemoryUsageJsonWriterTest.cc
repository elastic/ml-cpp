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

#include "CMemoryUsageJsonWriterTest.h"

#include <core/CMemoryUsage.h>
#include <core/CMemoryUsageJsonWriter.h>

#include <sstream>

using namespace ml;

CppUnit::Test* CMemoryUsageJsonWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMemoryUsageJsonWriterTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMemoryUsageJsonWriterTest>("CMemoryUsageJsonWriterTest::test", &CMemoryUsageJsonWriterTest::test));

    return suiteOfTests;
}

void CMemoryUsageJsonWriterTest::test(void) {
    {
        // Check that adding nothing produces nothing
        std::ostringstream ss;
        CPPUNIT_ASSERT_EQUAL(std::string(""), ss.str());

        core::CMemoryUsageJsonWriter writer(ss);
        CPPUNIT_ASSERT_EQUAL(std::string(""), ss.str());

        writer.finalise();
        CPPUNIT_ASSERT_EQUAL(std::string(""), ss.str());
    }
    {
        // Check one object
        std::ostringstream ss;
        core::CMemoryUsageJsonWriter writer(ss);
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description("Hello", 223);
        writer.addItem(description);
        writer.endObject();
        writer.finalise();
        CPPUNIT_ASSERT_EQUAL(std::string("{\"Hello\":{\"memory\":223}}\n"), ss.str());
    }
    {
        // Check one object with unused space
        std::ostringstream ss;
        core::CMemoryUsageJsonWriter writer(ss);
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description("Hello", 223, 45678);
        writer.addItem(description);
        writer.endObject();
        writer.finalise();
        CPPUNIT_ASSERT_EQUAL(std::string("{\"Hello\":{\"memory\":223,\"unused\":45678}}\n"), ss.str());
    }
    {
        // Check one empty array
        std::ostringstream ss;
        core::CMemoryUsageJsonWriter writer(ss);
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description("Hello", 223);
        writer.addItem(description);
        writer.startArray("Sheeple");
        writer.endArray();
        writer.endObject();
        writer.finalise();
        CPPUNIT_ASSERT_EQUAL(std::string("{\"Hello\":{\"memory\":223},\"Sheeple\":[]}\n"), ss.str());
    }
    {
        // Check one full array
        std::ostringstream ss;
        core::CMemoryUsageJsonWriter writer(ss);
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description("Hello", 223);
        writer.addItem(description);
        writer.startArray("Sheeple");
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description2("Womple", 44);
        writer.addItem(description2);
        writer.endObject();
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description3("Whimple", 66);
        writer.addItem(description3);
        core::CMemoryUsage::SMemoryUsage description4("magic", 7777);
        writer.addItem(description4);
        writer.endObject();
        writer.endArray();
        writer.endObject();
        writer.finalise();
        CPPUNIT_ASSERT_EQUAL(std::string("{\"Hello\":{\"memory\":223},\"Sheeple\":[{\"Womple\":{\"memory\":44}},{"
                                         "\"Whimple\":{\"memory\":66},\"magic\":{\"memory\":7777}}]}\n"),
                             ss.str());
    }
    {
        // Check sub-object
        std::ostringstream ss;
        core::CMemoryUsageJsonWriter writer(ss);
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description("Hello", 223);
        writer.addItem(description);
        writer.startArray("Sheeple");
        writer.startObject();
        core::CMemoryUsage::SMemoryUsage description1("Dumplings", 345);
        writer.addItem(description1);
        core::CMemoryUsage::SMemoryUsage description2("Gravy", 12341234);
        writer.addItem(description2);
        writer.endObject();
        writer.endArray();
        writer.endObject();
        writer.finalise();
        CPPUNIT_ASSERT_EQUAL(std::string("{\"Hello\":{\"memory\":223},\"Sheeple\":[{\"Dumplings\":{\"memory\":345},"
                                         "\"Gravy\":{\"memory\":12341234}}]}\n"),
                             ss.str());
    }
}
