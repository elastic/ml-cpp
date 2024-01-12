/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CBoostJsonConcurrentLineWriter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CStaticThreadPool.h>

#include <boost/json.hpp>
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
    ml::core::CBoostJsonConcurrentLineWriter writer(wrapper);
    for (int i = 0; i < documents; ++i) {
        writer.StartObject();
        writer.Key("id");
        writer.Int(id);
        writer.Key("message");
        writer.Int(i);

        // this automatically causes a flush in CBoostJsonConcurrentLineWriter
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

    json::error_code ec;
    json::value doc = json::parse(stringStream.str(), ec);

    // check that the document isn't malformed (like wrongly interleaved buffers)
    BOOST_TEST_REQUIRE(ec.failed() == false);
    const json::array& allRecords = doc.as_array();

    // check number of documents
    BOOST_REQUIRE_EQUAL(std::size_t(WRITERS * DOCUMENTS_PER_WRITER),
                        allRecords.size());
}

BOOST_AUTO_TEST_CASE(testShrink) {
    std::ostringstream stringStream;
    ml::core::CJsonOutputStreamWrapper wrapper(stringStream);

    size_t memoryUsageBase = wrapper.memoryUsage();
    ml::core::CStringBufWriter writer;
    std::string* stringBuffer;

    wrapper.acquireBuffer(writer, stringBuffer);

    // this should not change anything regarding memory usage
    BOOST_REQUIRE_EQUAL(memoryUsageBase, wrapper.memoryUsage());

    size_t stringBufferSizeBase = stringBuffer->capacity();
    BOOST_TEST_REQUIRE(memoryUsageBase > stringBufferSizeBase);

    // fill the buffer, expand it
    for (size_t i = 0; i < 100000; ++i) {
        stringBuffer->push_back('{');
        stringBuffer->push_back('}');
        stringBuffer->push_back(',');
    }

    BOOST_TEST_REQUIRE(stringBufferSizeBase < stringBuffer->capacity());
    BOOST_TEST_REQUIRE(wrapper.memoryUsage() > memoryUsageBase);
    BOOST_TEST_REQUIRE(wrapper.memoryUsage() > stringBuffer->capacity());

    // save the original pointer as flushBuffer returns a new buffer
    std::string* stringBufferOriginal = stringBuffer;
    wrapper.flushBuffer(writer, stringBuffer);
    wrapper.syncFlush();

    BOOST_TEST_REQUIRE(stringBuffer != stringBufferOriginal);
    BOOST_REQUIRE_EQUAL(stringBufferSizeBase, stringBuffer->capacity());
    BOOST_REQUIRE_EQUAL(stringBufferSizeBase, stringBufferOriginal->capacity());

    BOOST_REQUIRE_EQUAL(memoryUsageBase, wrapper.memoryUsage());
}

BOOST_AUTO_TEST_SUITE_END()
