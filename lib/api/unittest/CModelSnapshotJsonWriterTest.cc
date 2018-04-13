/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CModelSnapshotJsonWriterTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <model/CResourceMonitor.h>

#include <api/CModelSnapshotJsonWriter.h>

#include <rapidjson/document.h>

#include <string>

using namespace ml;
using namespace api;

CppUnit::Test* CModelSnapshotJsonWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CModelSnapshotJsonWriterTest");
    suiteOfTests->addTest(new CppUnit::TestCaller<CModelSnapshotJsonWriterTest>("CModelSnapshotJsonWriterTest::testWrite",
                                                                                &CModelSnapshotJsonWriterTest::testWrite));
    return suiteOfTests;
}

void CModelSnapshotJsonWriterTest::testWrite() {
    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
    {
        model::CResourceMonitor::SResults modelSizeStats{
            10000,                     // bytes used
            3,                         // # by fields
            1,                         // # partition fields
            150,                       // # over fields
            4,                         // # allocation failures
            model_t::E_MemoryStatusOk, // memory status
            core_t::TTime(1521046309)  // bucket start time
        };

        CModelSnapshotJsonWriter::SModelSnapshotReport report{
            "6.3.0",
            core_t::TTime(1521046309),
            "the snapshot description",
            "test_snapshot_id",
            size_t(15), // # docs
            modelSizeStats,
            "some normalizer state",
            core_t::TTime(1521046409), // last record time
            core_t::TTime(1521040000)  // last result time
        };

        core::CJsonOutputStreamWrapper wrappedOutStream(sstream);
        CModelSnapshotJsonWriter writer("job", wrappedOutStream);
        writer.write(report);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(object.IsObject());

    CPPUNIT_ASSERT(object.HasMember("model_snapshot"));
    const rapidjson::Value& snapshot = object["model_snapshot"];
    CPPUNIT_ASSERT(snapshot.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(snapshot["job_id"].GetString()));
    CPPUNIT_ASSERT(snapshot.HasMember("min_version"));
    CPPUNIT_ASSERT_EQUAL(std::string("6.3.0"), std::string(snapshot["min_version"].GetString()));
    CPPUNIT_ASSERT(snapshot.HasMember("snapshot_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("test_snapshot_id"), std::string(snapshot["snapshot_id"].GetString()));
    CPPUNIT_ASSERT(snapshot.HasMember("snapshot_doc_count"));
    CPPUNIT_ASSERT_EQUAL(int64_t(15), snapshot["snapshot_doc_count"].GetInt64());
    CPPUNIT_ASSERT(snapshot.HasMember("timestamp"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1521046309000), snapshot["timestamp"].GetInt64());
    CPPUNIT_ASSERT(snapshot.HasMember("description"));
    CPPUNIT_ASSERT_EQUAL(std::string("the snapshot description"), std::string(snapshot["description"].GetString()));
    CPPUNIT_ASSERT(snapshot.HasMember("latest_record_time_stamp"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1521046409000), snapshot["latest_record_time_stamp"].GetInt64());
    CPPUNIT_ASSERT(snapshot.HasMember("latest_result_time_stamp"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1521040000000), snapshot["latest_result_time_stamp"].GetInt64());

    CPPUNIT_ASSERT(snapshot.HasMember("model_size_stats"));
    const rapidjson::Value& modelSizeStats = snapshot["model_size_stats"];
    CPPUNIT_ASSERT(modelSizeStats.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(modelSizeStats["job_id"].GetString()));
    CPPUNIT_ASSERT(modelSizeStats.HasMember("model_bytes"));
    CPPUNIT_ASSERT_EQUAL(int64_t(20000), modelSizeStats["model_bytes"].GetInt64());
    CPPUNIT_ASSERT(modelSizeStats.HasMember("total_by_field_count"));
    CPPUNIT_ASSERT_EQUAL(int64_t(3), modelSizeStats["total_by_field_count"].GetInt64());
    CPPUNIT_ASSERT(modelSizeStats.HasMember("total_partition_field_count"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1), modelSizeStats["total_partition_field_count"].GetInt64());
    CPPUNIT_ASSERT(modelSizeStats.HasMember("total_over_field_count"));
    CPPUNIT_ASSERT_EQUAL(int64_t(150), modelSizeStats["total_over_field_count"].GetInt64());
    CPPUNIT_ASSERT(modelSizeStats.HasMember("bucket_allocation_failures_count"));
    CPPUNIT_ASSERT_EQUAL(int64_t(4), modelSizeStats["bucket_allocation_failures_count"].GetInt64());
    CPPUNIT_ASSERT(modelSizeStats.HasMember("memory_status"));
    CPPUNIT_ASSERT_EQUAL(std::string("ok"), std::string(modelSizeStats["memory_status"].GetString()));
    CPPUNIT_ASSERT(modelSizeStats.HasMember("timestamp"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1521046309000), modelSizeStats["timestamp"].GetInt64());
    CPPUNIT_ASSERT(modelSizeStats.HasMember("log_time"));

    CPPUNIT_ASSERT(snapshot.HasMember("quantiles"));
    const rapidjson::Value& quantiles = snapshot["quantiles"];
    CPPUNIT_ASSERT(quantiles.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(quantiles["job_id"].GetString()));
    CPPUNIT_ASSERT(quantiles.HasMember("quantile_state"));
    CPPUNIT_ASSERT_EQUAL(std::string("some normalizer state"), std::string(quantiles["quantile_state"].GetString()));
    CPPUNIT_ASSERT(quantiles.HasMember("timestamp"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1521040000000), quantiles["timestamp"].GetInt64());
}
