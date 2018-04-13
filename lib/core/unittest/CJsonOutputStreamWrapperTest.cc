/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CJsonOutputStreamWrapperTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>

#include <boost/threadpool.hpp>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>

CppUnit::Test *CJsonOutputStreamWrapperTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CJsonOutputStreamWrapperTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CJsonOutputStreamWrapperTest>(
                                   "CJsonOutputStreamWrapperTest::testConcurrentWrites",
                                   &CJsonOutputStreamWrapperTest::testConcurrentWrites) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CJsonOutputStreamWrapperTest>(
                                       "CJsonOutputStreamWrapperTest::testShrink",
                                       &CJsonOutputStreamWrapperTest::testShrink) );

    return suiteOfTests;
}

namespace
{

void task(ml::core::CJsonOutputStreamWrapper &wrapper, int id, int documents)
{
    ml::core::CRapidJsonConcurrentLineWriter writer(wrapper);
    for (int i = 0; i < documents; ++i)
    {
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

void CJsonOutputStreamWrapperTest::testConcurrentWrites()
{
    std::ostringstream stringStream;

    static const size_t WRITERS(1500);
    static const size_t DOCUMENTS_PER_WRITER(10);
    {
        ml::core::CJsonOutputStreamWrapper wrapper(stringStream);

        boost::threadpool::pool tp(100);
        for (size_t i = 0; i < WRITERS; ++i)
        {
            tp.schedule(boost::bind(task, boost::ref(wrapper), i, DOCUMENTS_PER_WRITER));
        }
        tp.wait();
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(stringStream.str());

    // check that the document isn't malformed (like wrongly interleaved buffers)
    CPPUNIT_ASSERT(!doc.HasParseError());
    const rapidjson::Value &allRecords = doc.GetArray();

    // check number of documents
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(WRITERS * DOCUMENTS_PER_WRITER), allRecords.Size());
}

void CJsonOutputStreamWrapperTest::testShrink()
{
    std::ostringstream stringStream;
    ml::core::CJsonOutputStreamWrapper wrapper(stringStream);

    size_t memoryUsageBase = wrapper.memoryUsage();
    ml::core::CJsonOutputStreamWrapper::TGenericLineWriter writer;
    rapidjson::StringBuffer *stringBuffer;

    wrapper.acquireBuffer(writer, stringBuffer);

    // this should not change anything regarding memory usage
    CPPUNIT_ASSERT_EQUAL(memoryUsageBase, wrapper.memoryUsage());

    size_t stringBufferSizeBase = stringBuffer->stack_.GetCapacity();
    CPPUNIT_ASSERT(memoryUsageBase > stringBufferSizeBase);

    // fill the buffer, expand it
    for (size_t i=0; i < 100000; ++i)
    {
        stringBuffer->Put('{');
        stringBuffer->Put('}');
        stringBuffer->Put(',');
    }

    CPPUNIT_ASSERT(stringBufferSizeBase < stringBuffer->stack_.GetCapacity());
    CPPUNIT_ASSERT(wrapper.memoryUsage() > memoryUsageBase);
    CPPUNIT_ASSERT(wrapper.memoryUsage() > stringBuffer->stack_.GetCapacity());

    // save the original pointer as flushBuffer returns a new buffer
    rapidjson::StringBuffer *stringBufferOriginal = stringBuffer;
    wrapper.flushBuffer(writer, stringBuffer);
    wrapper.syncFlush();

    CPPUNIT_ASSERT(stringBuffer != stringBufferOriginal);
    CPPUNIT_ASSERT_EQUAL(stringBufferSizeBase, stringBuffer->stack_.GetCapacity());
    CPPUNIT_ASSERT_EQUAL(stringBufferSizeBase, stringBufferOriginal->stack_.GetCapacity());

    CPPUNIT_ASSERT_EQUAL(memoryUsageBase, wrapper.memoryUsage());
}

