/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMemoryUsage.h>
#include <core/CMemoryUsageJsonWriter.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CMemoryUsageJsonWriterTest)

using namespace ml;


BOOST_AUTO_TEST_CASE(test) {
    {
        // Check that adding nothing produces nothing
        std::ostringstream ss;
        BOOST_CHECK_EQUAL(std::string(""), ss.str());

        core::CMemoryUsageJsonWriter writer(ss);
        BOOST_CHECK_EQUAL(std::string(""), ss.str());

        writer.finalise();
        BOOST_CHECK_EQUAL(std::string(""), ss.str());
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
        BOOST_CHECK_EQUAL(std::string("{\"Hello\":{\"memory\":223}}\n"), ss.str());
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
        BOOST_CHECK_EQUAL(std::string("{\"Hello\":{\"memory\":223,\"unused\":45678}}\n"),
                             ss.str());
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
        BOOST_CHECK_EQUAL(
            std::string("{\"Hello\":{\"memory\":223},\"Sheeple\":[]}\n"), ss.str());
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
        BOOST_CHECK_EQUAL(std::string("{\"Hello\":{\"memory\":223},\"Sheeple\":[{\"Womple\":{\"memory\":44}},{\"Whimple\":{\"memory\":"
                                         "66},\"magic\":{\"memory\":7777}}]}\n"),
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
        BOOST_CHECK_EQUAL(std::string("{\"Hello\":{\"memory\":223},\"Sheeple\":[{\"Dumplings\":{\"memory\":345},\"Gravy\":{\"memory\":12341234}}]}\n"),
                             ss.str());
    }
}

BOOST_AUTO_TEST_SUITE_END()
