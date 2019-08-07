/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMemoryUsageEstimationResultJsonWriterTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <model/CResourceMonitor.h>

#include <api/CMemoryUsageEstimationResultJsonWriter.h>

#include <rapidjson/document.h>

#include <string>

using namespace ml;
using namespace api;

CppUnit::Test* CMemoryUsageEstimationResultJsonWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMemoryUsageEstimationResultJsonWriterTest");
    suiteOfTests->addTest(new CppUnit::TestCaller<CMemoryUsageEstimationResultJsonWriterTest>(
        "CMemoryUsageEstimationResultJsonWriterTest::testWrite",
        &CMemoryUsageEstimationResultJsonWriterTest::testWrite));
    return suiteOfTests;
}

void CMemoryUsageEstimationResultJsonWriterTest::testWrite() {
    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
    {
        core::CJsonOutputStreamWrapper wrappedOutStream(sstream);
        CMemoryUsageEstimationResultJsonWriter writer(wrappedOutStream);
        writer.write(static_cast<std::size_t>(2000), static_cast<std::size_t>(1000));
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(object.IsObject());

    CPPUNIT_ASSERT(object.HasMember("expected_memory_usage_with_one_partition"));
    CPPUNIT_ASSERT_EQUAL(int64_t(2000), object["expected_memory_usage_with_one_partition"].GetInt64());
    CPPUNIT_ASSERT(object.HasMember("expected_memory_usage_with_max_partitions"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1000), object["expected_memory_usage_with_max_partitions"].GetInt64());
}
