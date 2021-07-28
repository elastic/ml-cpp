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

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <model/CResourceMonitor.h>

#include <api/CModelSnapshotJsonWriter.h>

#include <rapidjson/document.h>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(CModelSnapshotJsonWriterTest)

using namespace ml;
using namespace api;

BOOST_AUTO_TEST_CASE(testWrite) {
    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
    {
        model::CResourceMonitor::SModelSizeStats modelSizeStats{
            10000,                             // bytes used
            20000,                             // bytes used (adjusted)
            30000,                             // peak bytes used
            60000,                             // peak bytes used (adjusted)
            3,                                 // # by fields
            1,                                 // # partition fields
            150,                               // # over fields
            4,                                 // # allocation failures
            model_t::E_MemoryStatusOk,         // memory status
            model_t::E_AssignmentBasisUnknown, // assignment memory basis
            core_t::TTime(1521046309),         // bucket start time
            0,                                 // model bytes exceeded
            50000,                             // model bytes memory limit
            {1000,                             // categorized messages
             100,                              // total categories
             7,                                // frequent categories
             13,                               // rare categories
             2,                                // dead categories
             8,                                // failed categories
             model_t::E_CategorizationStatusWarn}};

        CModelSnapshotJsonWriter::SModelSnapshotReport report{
            "6.3.0",
            core_t::TTime(1521046309),
            "the snapshot description",
            "test_snapshot_id",
            15, // # docs
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

    BOOST_TEST_REQUIRE(arrayDoc.IsArray());
    BOOST_REQUIRE_EQUAL(1, arrayDoc.Size());

    const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(0)];
    BOOST_TEST_REQUIRE(object.IsObject());

    BOOST_TEST_REQUIRE(object.HasMember("model_snapshot"));
    const rapidjson::Value& snapshot = object["model_snapshot"];
    BOOST_TEST_REQUIRE(snapshot.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL("job", snapshot["job_id"].GetString());
    BOOST_TEST_REQUIRE(snapshot.HasMember("min_version"));
    BOOST_REQUIRE_EQUAL("6.3.0", snapshot["min_version"].GetString());
    BOOST_TEST_REQUIRE(snapshot.HasMember("snapshot_id"));
    BOOST_REQUIRE_EQUAL("test_snapshot_id", snapshot["snapshot_id"].GetString());
    BOOST_TEST_REQUIRE(snapshot.HasMember("snapshot_doc_count"));
    BOOST_REQUIRE_EQUAL(15, snapshot["snapshot_doc_count"].GetUint64());
    BOOST_TEST_REQUIRE(snapshot.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(1521046309000, snapshot["timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(snapshot.HasMember("description"));
    BOOST_REQUIRE_EQUAL("the snapshot description", snapshot["description"].GetString());
    BOOST_TEST_REQUIRE(snapshot.HasMember("latest_record_time_stamp"));
    BOOST_REQUIRE_EQUAL(1521046409000, snapshot["latest_record_time_stamp"].GetInt64());
    BOOST_TEST_REQUIRE(snapshot.HasMember("latest_result_time_stamp"));
    BOOST_REQUIRE_EQUAL(1521040000000, snapshot["latest_result_time_stamp"].GetInt64());

    BOOST_TEST_REQUIRE(snapshot.HasMember("model_size_stats"));
    const rapidjson::Value& modelSizeStats = snapshot["model_size_stats"];
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL("job", modelSizeStats["job_id"].GetString());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("model_bytes"));
    BOOST_REQUIRE_EQUAL(20000, modelSizeStats["model_bytes"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("peak_model_bytes"));
    BOOST_REQUIRE_EQUAL(60000, modelSizeStats["peak_model_bytes"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_by_field_count"));
    BOOST_REQUIRE_EQUAL(3, modelSizeStats["total_by_field_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_partition_field_count"));
    BOOST_REQUIRE_EQUAL(1, modelSizeStats["total_partition_field_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_over_field_count"));
    BOOST_REQUIRE_EQUAL(150, modelSizeStats["total_over_field_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("bucket_allocation_failures_count"));
    BOOST_REQUIRE_EQUAL(4, modelSizeStats["bucket_allocation_failures_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("memory_status"));
    BOOST_REQUIRE_EQUAL("ok", modelSizeStats["memory_status"].GetString());
    BOOST_REQUIRE_EQUAL(false, modelSizeStats.HasMember("assignment_memory_basis"));
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("model_bytes_exceeded"));
    BOOST_REQUIRE_EQUAL(0, modelSizeStats["model_bytes_exceeded"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("model_bytes_memory_limit"));
    BOOST_REQUIRE_EQUAL(50000, modelSizeStats["model_bytes_memory_limit"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("categorized_doc_count"));
    BOOST_REQUIRE_EQUAL(1000, modelSizeStats["categorized_doc_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_category_count"));
    BOOST_REQUIRE_EQUAL(100, modelSizeStats["total_category_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("frequent_category_count"));
    BOOST_REQUIRE_EQUAL(7, modelSizeStats["frequent_category_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("rare_category_count"));
    BOOST_REQUIRE_EQUAL(13, modelSizeStats["rare_category_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("dead_category_count"));
    BOOST_REQUIRE_EQUAL(2, modelSizeStats["dead_category_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("failed_category_count"));
    BOOST_REQUIRE_EQUAL(8, modelSizeStats["failed_category_count"].GetUint64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("memory_status"));
    BOOST_REQUIRE_EQUAL("warn", modelSizeStats["categorization_status"].GetString());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(1521046309000, modelSizeStats["timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("log_time"));

    BOOST_TEST_REQUIRE(snapshot.HasMember("quantiles"));
    const rapidjson::Value& quantiles = snapshot["quantiles"];
    BOOST_TEST_REQUIRE(quantiles.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL("job", quantiles["job_id"].GetString());
    BOOST_TEST_REQUIRE(quantiles.HasMember("quantile_state"));
    BOOST_REQUIRE_EQUAL("some normalizer state", quantiles["quantile_state"].GetString());
    BOOST_TEST_REQUIRE(quantiles.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(1521040000000, quantiles["timestamp"].GetInt64());
}

BOOST_AUTO_TEST_SUITE_END()
