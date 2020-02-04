/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
        model::CResourceMonitor::SResults modelSizeStats{
            10000,                     // bytes used
            20000,                     // bytes used (adjusted)
            3,                         // # by fields
            1,                         // # partition fields
            150,                       // # over fields
            4,                         // # allocation failures
            model_t::E_MemoryStatusOk, // memory status
            core_t::TTime(1521046309), // bucket start time
            0,                         // model bytes exceeded
            50000,                     // model bytes memory limit
            1000,                      // categorized messages
            100,                       // total categories
            7,                         // frequent categories
            13,                        // rare categories
            2,                         // dead categories
            model_t::E_CategorizationStatusPoor};

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

    BOOST_TEST_REQUIRE(arrayDoc.IsArray());
    BOOST_REQUIRE_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(0)];
    BOOST_TEST_REQUIRE(object.IsObject());

    BOOST_TEST_REQUIRE(object.HasMember("model_snapshot"));
    const rapidjson::Value& snapshot = object["model_snapshot"];
    BOOST_TEST_REQUIRE(snapshot.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL(std::string("job"), std::string(snapshot["job_id"].GetString()));
    BOOST_TEST_REQUIRE(snapshot.HasMember("min_version"));
    BOOST_REQUIRE_EQUAL(std::string("6.3.0"),
                        std::string(snapshot["min_version"].GetString()));
    BOOST_TEST_REQUIRE(snapshot.HasMember("snapshot_id"));
    BOOST_REQUIRE_EQUAL(std::string("test_snapshot_id"),
                        std::string(snapshot["snapshot_id"].GetString()));
    BOOST_TEST_REQUIRE(snapshot.HasMember("snapshot_doc_count"));
    BOOST_REQUIRE_EQUAL(int64_t(15), snapshot["snapshot_doc_count"].GetInt64());
    BOOST_TEST_REQUIRE(snapshot.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(int64_t(1521046309000), snapshot["timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(snapshot.HasMember("description"));
    BOOST_REQUIRE_EQUAL(std::string("the snapshot description"),
                        std::string(snapshot["description"].GetString()));
    BOOST_TEST_REQUIRE(snapshot.HasMember("latest_record_time_stamp"));
    BOOST_REQUIRE_EQUAL(int64_t(1521046409000),
                        snapshot["latest_record_time_stamp"].GetInt64());
    BOOST_TEST_REQUIRE(snapshot.HasMember("latest_result_time_stamp"));
    BOOST_REQUIRE_EQUAL(int64_t(1521040000000),
                        snapshot["latest_result_time_stamp"].GetInt64());

    BOOST_TEST_REQUIRE(snapshot.HasMember("model_size_stats"));
    const rapidjson::Value& modelSizeStats = snapshot["model_size_stats"];
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL(std::string("job"),
                        std::string(modelSizeStats["job_id"].GetString()));
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("model_bytes"));
    BOOST_REQUIRE_EQUAL(int64_t(20000), modelSizeStats["model_bytes"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_by_field_count"));
    BOOST_REQUIRE_EQUAL(int64_t(3), modelSizeStats["total_by_field_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_partition_field_count"));
    BOOST_REQUIRE_EQUAL(int64_t(1),
                        modelSizeStats["total_partition_field_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_over_field_count"));
    BOOST_REQUIRE_EQUAL(int64_t(150), modelSizeStats["total_over_field_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("bucket_allocation_failures_count"));
    BOOST_REQUIRE_EQUAL(
        int64_t(4), modelSizeStats["bucket_allocation_failures_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("memory_status"));
    BOOST_REQUIRE_EQUAL(std::string("ok"),
                        std::string(modelSizeStats["memory_status"].GetString()));
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("model_bytes_exceeded"));
    BOOST_REQUIRE_EQUAL(int64_t(0), modelSizeStats["model_bytes_exceeded"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("model_bytes_memory_limit"));
    BOOST_REQUIRE_EQUAL(int64_t(50000),
                        modelSizeStats["model_bytes_memory_limit"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("categorized_doc_count"));
    BOOST_REQUIRE_EQUAL(int64_t(1000), modelSizeStats["categorized_doc_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("total_category_count"));
    BOOST_REQUIRE_EQUAL(int64_t(100), modelSizeStats["total_category_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("frequent_category_count"));
    BOOST_REQUIRE_EQUAL(int64_t(7), modelSizeStats["frequent_category_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("rare_category_count"));
    BOOST_REQUIRE_EQUAL(int64_t(13), modelSizeStats["rare_category_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("dead_category_count"));
    BOOST_REQUIRE_EQUAL(int64_t(2), modelSizeStats["dead_category_count"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("memory_status"));
    BOOST_REQUIRE_EQUAL(std::string("poor"),
                        std::string(modelSizeStats["categorization_status"].GetString()));
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(int64_t(1521046309000), modelSizeStats["timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(modelSizeStats.HasMember("log_time"));

    BOOST_TEST_REQUIRE(snapshot.HasMember("quantiles"));
    const rapidjson::Value& quantiles = snapshot["quantiles"];
    BOOST_TEST_REQUIRE(quantiles.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL(std::string("job"), std::string(quantiles["job_id"].GetString()));
    BOOST_TEST_REQUIRE(quantiles.HasMember("quantile_state"));
    BOOST_REQUIRE_EQUAL(std::string("some normalizer state"),
                        std::string(quantiles["quantile_state"].GetString()));
    BOOST_TEST_REQUIRE(quantiles.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(int64_t(1521040000000), quantiles["timestamp"].GetInt64());
}

BOOST_AUTO_TEST_SUITE_END()
