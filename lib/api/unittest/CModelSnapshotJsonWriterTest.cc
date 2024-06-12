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
            60000,                             // JSON memory allocator bytes used
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

    json::error_code ec;
    json::value arrayDoc = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(arrayDoc.is_array());
    BOOST_REQUIRE_EQUAL(1, arrayDoc.as_array().size());

    const json::value& object_ = arrayDoc.as_array()[0];
    BOOST_TEST_REQUIRE(object_.is_object());
    const json::object& object = object_.as_object();

    BOOST_TEST_REQUIRE(object.contains("model_snapshot"));
    const json::value& snapshot_ = object.at("model_snapshot");
    const json::object& snapshot = snapshot_.as_object();

    BOOST_TEST_REQUIRE(snapshot.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", snapshot.at("job_id").as_string());
    BOOST_TEST_REQUIRE(snapshot.contains("min_version"));
    BOOST_REQUIRE_EQUAL("6.3.0", snapshot.at("min_version").as_string());
    BOOST_TEST_REQUIRE(snapshot.contains("snapshot_id"));
    BOOST_REQUIRE_EQUAL("test_snapshot_id", snapshot.at("snapshot_id").as_string());
    BOOST_TEST_REQUIRE(snapshot.contains("snapshot_doc_count"));
    BOOST_REQUIRE_EQUAL(15, snapshot.at("snapshot_doc_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(snapshot.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(1521046309000, snapshot.at("timestamp").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(snapshot.contains("description"));
    BOOST_REQUIRE_EQUAL("the snapshot description", snapshot.at("description").as_string());
    BOOST_TEST_REQUIRE(snapshot.contains("latest_record_time_stamp"));
    BOOST_REQUIRE_EQUAL(1521046409000,
                        snapshot.at("latest_record_time_stamp").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(snapshot.contains("latest_result_time_stamp"));
    BOOST_REQUIRE_EQUAL(1521040000000,
                        snapshot.at("latest_result_time_stamp").to_number<std::int64_t>());

    BOOST_TEST_REQUIRE(snapshot.contains("model_size_stats"));
    const json::value& modelSizeStats_ = snapshot.at("model_size_stats");
    const json::object& modelSizeStats = modelSizeStats_.as_object();

    BOOST_TEST_REQUIRE(modelSizeStats.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", modelSizeStats.at("job_id").as_string());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("model_bytes"));
    BOOST_REQUIRE_EQUAL(20000, modelSizeStats.at("model_bytes").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("peak_model_bytes"));
    BOOST_REQUIRE_EQUAL(
        60000, modelSizeStats.at("peak_model_bytes").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("total_by_field_count"));
    BOOST_REQUIRE_EQUAL(
        3, modelSizeStats.at("total_by_field_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("total_partition_field_count"));
    BOOST_REQUIRE_EQUAL(
        1, modelSizeStats.at("total_partition_field_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("total_over_field_count"));
    BOOST_REQUIRE_EQUAL(
        150, modelSizeStats.at("total_over_field_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("bucket_allocation_failures_count"));
    BOOST_REQUIRE_EQUAL(
        4, modelSizeStats.at("bucket_allocation_failures_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("memory_status"));
    BOOST_REQUIRE_EQUAL("ok", modelSizeStats.at("memory_status").as_string());
    BOOST_REQUIRE_EQUAL(false, modelSizeStats.contains("assignment_memory_basis"));
    BOOST_TEST_REQUIRE(modelSizeStats.contains("model_bytes_exceeded"));
    BOOST_REQUIRE_EQUAL(
        0, modelSizeStats.at("model_bytes_exceeded").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("model_bytes_memory_limit"));
    BOOST_REQUIRE_EQUAL(
        50000, modelSizeStats.at("model_bytes_memory_limit").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("categorized_doc_count"));
    BOOST_REQUIRE_EQUAL(
        1000, modelSizeStats.at("categorized_doc_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("total_category_count"));
    BOOST_REQUIRE_EQUAL(
        100, modelSizeStats.at("total_category_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("frequent_category_count"));
    BOOST_REQUIRE_EQUAL(
        7, modelSizeStats.at("frequent_category_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("rare_category_count"));
    BOOST_REQUIRE_EQUAL(
        13, modelSizeStats.at("rare_category_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("dead_category_count"));
    BOOST_REQUIRE_EQUAL(
        2, modelSizeStats.at("dead_category_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("failed_category_count"));
    BOOST_REQUIRE_EQUAL(
        8, modelSizeStats.at("failed_category_count").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("memory_status"));
    BOOST_REQUIRE_EQUAL("warn", modelSizeStats.at("categorization_status").as_string());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(1521046309000,
                        modelSizeStats.at("timestamp").to_number<std::int64_t>());
    BOOST_TEST_REQUIRE(modelSizeStats.contains("log_time"));

    BOOST_TEST_REQUIRE(snapshot.contains("quantiles"));
    const json::value& quantiles_ = snapshot.at("quantiles");
    const json::object& quantiles = quantiles_.as_object();

    BOOST_TEST_REQUIRE(quantiles.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", quantiles.at("job_id").as_string());
    BOOST_TEST_REQUIRE(quantiles.contains("quantile_state"));
    BOOST_REQUIRE_EQUAL("some normalizer state", quantiles.at("quantile_state").as_string());
    BOOST_TEST_REQUIRE(quantiles.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(1521040000000,
                        quantiles.at("timestamp").to_number<std::int64_t>());
}

BOOST_AUTO_TEST_SUITE_END()
