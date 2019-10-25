/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CStaticThreadPool.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <string>
#include <thread>

BOOST_AUTO_TEST_SUITE(CJsonOutputStreamWrapperTest)

namespace {

void task(ml::core::CJsonOutputStreamWrapper& wrapper, int id, int documents) {
    ml::core::CRapidJsonConcurrentLineWriter writer(wrapper);
    for (int i = 0; i < documents; ++i) {
        writer.StartObject();
        writer.Key("id");
        writer.Int(id);
        writer.Key("message");
        writer.Int(i);

        // this automatically causes a flush in CRapidJsonConcurrentLineWriter
        // A flush internally moves the buffer into the queue, passing it to the writer thread
        // A new buffer gets acquired for the next loop execution
        writer.EndObject();
    }
}
}

BOOST_AUTO_TEST_CASE(testConcurrentWrites) {
    std::ostringstream stringStream;

    static const int WRITERS(1500);
    static const size_t DOCUMENTS_PER_WRITER(10);
    {
        ml::core::CJsonOutputStreamWrapper wrapper(stringStream);

        ml::core::CStaticThreadPool tp(100);
        for (int i = 0; i < WRITERS; ++i) {
            tp.schedule([&wrapper, i] { task(wrapper, i, DOCUMENTS_PER_WRITER); });
        }
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(stringStream.str());

    // check that the document isn't malformed (like wrongly interleaved buffers)
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    const rapidjson::Value& allRecords = doc.GetArray();

    // check number of documents
    BOOST_REQUIRE_EQUAL(rapidjson::SizeType(WRITERS * DOCUMENTS_PER_WRITER),
                        allRecords.Size());
}

BOOST_AUTO_TEST_CASE(testShrink) {
    std::ostringstream stringStream;
    ml::core::CJsonOutputStreamWrapper wrapper(stringStream);

    size_t memoryUsageBase = wrapper.memoryUsage();
    ml::core::CJsonOutputStreamWrapper::TGenericLineWriter writer;
    rapidjson::StringBuffer* stringBuffer;

    wrapper.acquireBuffer(writer, stringBuffer);

    // this should not change anything regarding memory usage
    BOOST_REQUIRE_EQUAL(memoryUsageBase, wrapper.memoryUsage());

    size_t stringBufferSizeBase = stringBuffer->stack_.GetCapacity();
    BOOST_TEST_REQUIRE(memoryUsageBase > stringBufferSizeBase);

    // fill the buffer, expand it
    for (size_t i = 0; i < 100000; ++i) {
        stringBuffer->Put('{');
        stringBuffer->Put('}');
        stringBuffer->Put(',');
    }

    BOOST_TEST_REQUIRE(stringBufferSizeBase < stringBuffer->stack_.GetCapacity());
    BOOST_TEST_REQUIRE(wrapper.memoryUsage() > memoryUsageBase);
    BOOST_TEST_REQUIRE(wrapper.memoryUsage() > stringBuffer->stack_.GetCapacity());

    // save the original pointer as flushBuffer returns a new buffer
    rapidjson::StringBuffer* stringBufferOriginal = stringBuffer;
    wrapper.flushBuffer(writer, stringBuffer);
    wrapper.syncFlush();

    BOOST_TEST_REQUIRE(stringBuffer != stringBufferOriginal);
    BOOST_REQUIRE_EQUAL(stringBufferSizeBase, stringBuffer->stack_.GetCapacity());
    BOOST_REQUIRE_EQUAL(stringBufferSizeBase, stringBufferOriginal->stack_.GetCapacity());

    BOOST_REQUIRE_EQUAL(memoryUsageBase, wrapper.memoryUsage());
}

BOOST_AUTO_TEST_SUITE_END()
