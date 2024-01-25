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

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CScopedBoostJsonPoolAllocator.h>
#include <core/CSmallVector.h>
#include <core/CTimeUtils.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CStringStore.h>
#include <model/ModelTypes.h>

#include <api/CGlobalCategoryId.h>
#include <api/CJsonOutputWriter.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CJsonOutputWriterTest)

namespace {
using TDouble1Vec = ml::core::CSmallVector<double, 1>;
using TStr1Vec = ml::core::CSmallVector<std::string, 1>;
const TStr1Vec EMPTY_STRING_LIST;

void testBucketWriteHelper(bool isInterim) {
    // groups output by bucket/detector

    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        std::string partitionFieldName("tfn");
        std::string partitionFieldValue("");
        std::string overFieldName("pfn");
        std::string overFieldValue("pfv");
        std::string byFieldName("airline");
        std::string byFieldValue("GAL");
        std::string correlatedByFieldValue("BAW");
        std::string fieldName("responsetime");
        std::string function("mean");
        std::string functionDescription("mean(responsetime)");
        std::string emptyString;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;

        {
            ml::api::CHierarchicalResultsWriter::SResults result11(
                false, false, partitionFieldName, partitionFieldValue,
                overFieldName, overFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 1, function, functionDescription,
                TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0), 2.24, 0.5, 0.0,
                79, fieldName, influences, false, false, 1, 100);

            ml::api::CHierarchicalResultsWriter::SResults result112(
                false, true, partitionFieldName, partitionFieldValue,
                overFieldName, overFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 1, function, functionDescription,
                TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0), 2.24, 0.5, 0.0,
                79, fieldName, influences, false, false, 1, 100);

            ml::api::CHierarchicalResultsWriter::SResults result12(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                1, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.8, 0.0, -5.0, fieldName,
                influences, false, true, 2, 100, EMPTY_STRING_LIST, {});

            ml::api::CHierarchicalResultsWriter::SResults result13(
                ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                partitionFieldName, partitionFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 1, function, functionDescription, 42.0, 79,
                TDouble1Vec(1, 6953.0), TDouble1Vec(1, 10090.0), 2.24, 0.5, 0.0, -5.0,
                fieldName, influences, false, false, 3, 100, EMPTY_STRING_LIST, {});

            ml::api::CHierarchicalResultsWriter::SResults result14(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                1, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0, fieldName,
                influences, false, false, 4, 100, EMPTY_STRING_LIST, {});

            // 1st bucket
            BOOST_TEST_REQUIRE(writer.acceptResult(result11));
            BOOST_TEST_REQUIRE(writer.acceptResult(result11));
            BOOST_TEST_REQUIRE(writer.acceptResult(result112));
            BOOST_TEST_REQUIRE(writer.acceptResult(result12));
            BOOST_TEST_REQUIRE(writer.acceptResult(result12));
            BOOST_TEST_REQUIRE(writer.acceptResult(result13));
            BOOST_TEST_REQUIRE(writer.acceptResult(result13));
            BOOST_TEST_REQUIRE(writer.acceptResult(result14));
            BOOST_TEST_REQUIRE(writer.acceptResult(result14));
            writer.acceptBucketTimeInfluencer(1, 0.01, 13.44, 70.0);
        }

        {
            ml::api::CHierarchicalResultsWriter::SResults result21(
                false, false, partitionFieldName, partitionFieldValue,
                overFieldName, overFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 2, function, functionDescription,
                TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0), 2.24, 0.6, 0.0,
                79, fieldName, influences, false, false, 1, 100);

            ml::api::CHierarchicalResultsWriter::SResults result212(
                false, true, partitionFieldName, partitionFieldValue,
                overFieldName, overFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 2, function, functionDescription,
                TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0), 2.24, 0.6, 0.0,
                79, fieldName, influences, false, false, 1, 100);

            ml::api::CHierarchicalResultsWriter::SResults result22(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                2, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.8, 0.0, -5.0, fieldName,
                influences, false, true, 2, 100, EMPTY_STRING_LIST, {});

            ml::api::CHierarchicalResultsWriter::SResults result23(
                ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                partitionFieldName, partitionFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 2, function, functionDescription, 42.0, 79,
                TDouble1Vec(1, 6953.0), TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0,
                fieldName, influences, false, false, 3, 100, EMPTY_STRING_LIST, {});

            ml::api::CHierarchicalResultsWriter::SResults result24(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                2, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0, fieldName,
                influences, false, false, 4, 100, EMPTY_STRING_LIST, {});

            // 2nd bucket
            BOOST_TEST_REQUIRE(writer.acceptResult(result21));
            BOOST_TEST_REQUIRE(writer.acceptResult(result21));
            BOOST_TEST_REQUIRE(writer.acceptResult(result212));
            BOOST_TEST_REQUIRE(writer.acceptResult(result22));
            BOOST_TEST_REQUIRE(writer.acceptResult(result22));
            BOOST_TEST_REQUIRE(writer.acceptResult(result23));
            BOOST_TEST_REQUIRE(writer.acceptResult(result23));
            BOOST_TEST_REQUIRE(writer.acceptResult(result24));
            BOOST_TEST_REQUIRE(writer.acceptResult(result24));
            writer.acceptBucketTimeInfluencer(2, 0.01, 13.44, 70.0);
        }

        {
            ml::api::CHierarchicalResultsWriter::SResults result31(
                false, false, partitionFieldName, partitionFieldValue,
                overFieldName, overFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 3, function, functionDescription,
                TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0), 2.24, 0.8, 0.0,
                79, fieldName, influences, false, false, 1, 100);

            ml::api::CHierarchicalResultsWriter::SResults result312(
                false, true, partitionFieldName, partitionFieldValue,
                overFieldName, overFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 3, function, functionDescription,
                TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0), 2.24, 0.8, 0.0,
                79, fieldName, influences, false, false, 1, 100);

            ml::api::CHierarchicalResultsWriter::SResults result32(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                3, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0, fieldName,
                influences, false, true, 2, 100, EMPTY_STRING_LIST, {});

            ml::api::CHierarchicalResultsWriter::SResults result33(
                ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                partitionFieldName, partitionFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 3, function, functionDescription, 42.0, 79,
                TDouble1Vec(1, 6953.0), TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0,
                fieldName, influences, false, false, 3, 100, EMPTY_STRING_LIST, {});

            ml::api::CHierarchicalResultsWriter::SResults result34(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                3, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0, fieldName,
                influences, false, false, 4, 100, EMPTY_STRING_LIST, {});

            // 3rd bucket
            BOOST_TEST_REQUIRE(writer.acceptResult(result31));
            BOOST_TEST_REQUIRE(writer.acceptResult(result31));
            BOOST_TEST_REQUIRE(writer.acceptResult(result312));
            BOOST_TEST_REQUIRE(writer.acceptResult(result32));
            BOOST_TEST_REQUIRE(writer.acceptResult(result32));
            BOOST_TEST_REQUIRE(writer.acceptResult(result33));
            BOOST_TEST_REQUIRE(writer.acceptResult(result33));
            BOOST_TEST_REQUIRE(writer.acceptResult(result34));
            BOOST_TEST_REQUIRE(writer.acceptResult(result34));
            writer.acceptBucketTimeInfluencer(3, 0.01, 13.44, 70.0);
        }

        // Finished adding results
        BOOST_TEST_REQUIRE(writer.endOutputBatch(isInterim, 10U));
    }

    json::error_code ec;
    json::value arrayDoc = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(arrayDoc.is_array());

    LOG_DEBUG(<< "Results:\n" << arrayDoc);

    // There are 3 buckets and 3 record arrays in the order: r1, b1, r2, b2, r3, b3
    BOOST_REQUIRE_EQUAL(6, arrayDoc.as_array().size());

    int bucketTimes[] = {1000, 1000, 2000, 2000, 3000, 3000};

    // Assert buckets
    for (std::size_t i = 1; i < arrayDoc.as_array().size(); i = i + 2) {
        int buckettime = bucketTimes[i];
        const json::value& bucketWrapper_ = arrayDoc.as_array().at(i);
        const json::object& bucketWrapper = bucketWrapper_.as_object();
        BOOST_TEST_REQUIRE(bucketWrapper.contains("bucket"));

        const json::value& bucket_ = bucketWrapper.at("bucket");
        BOOST_TEST_REQUIRE(bucket_.is_object());
        const json::object& bucket = bucket_.as_object();
        BOOST_TEST_REQUIRE(bucket.contains("job_id"));
        BOOST_REQUIRE_EQUAL("job", bucket.at("job_id").as_string());

        // 3 detectors each have 2 records (simple count detector isn't added)
        // except the population detector which has a single record and clauses
        BOOST_REQUIRE_EQUAL(buckettime, bucket.at("timestamp").as_int64());
        BOOST_TEST_REQUIRE(bucket.contains("bucket_influencers"));
        const json::value& bucketInfluencers_ = bucket.at("bucket_influencers");
        BOOST_TEST_REQUIRE(bucketInfluencers_.is_array());
        const json::array& bucketInfluencers = bucketInfluencers_.as_array();
        BOOST_REQUIRE_EQUAL(std::size_t(1), bucketInfluencers.size());
        const json::value& bucketInfluencer_ = bucketInfluencers[std::size_t(0)];
        const json::object& bucketInfluencer = bucketInfluencer_.as_object();
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            13.44, bucketInfluencer.at("raw_anomaly_score").to_number<double>(), 0.00001);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.01, bucketInfluencer.at("probability").to_number<double>(), 0.00001);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            70.0, bucketInfluencer.at("initial_anomaly_score").to_number<double>(), 0.00001);
        BOOST_TEST_REQUIRE(bucketInfluencer.contains("anomaly_score"));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            70.0, bucketInfluencer.at("anomaly_score").to_number<double>(), 0.00001);
        BOOST_REQUIRE_EQUAL("bucket_time",
                            bucketInfluencer.at("influencer_field_name").as_string());

        BOOST_REQUIRE_EQUAL(79, bucket.at("event_count").as_int64());
        BOOST_TEST_REQUIRE(bucket.contains("anomaly_score"));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            70.0, bucket.at("anomaly_score").to_number<double>(), 0.00001);
        BOOST_TEST_REQUIRE(bucket.contains("initial_anomaly_score"));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            70.0, bucket.at("initial_anomaly_score").to_number<double>(), 0.00001);
        if (isInterim) {
            BOOST_TEST_REQUIRE(bucket.contains("is_interim"));
            BOOST_REQUIRE_EQUAL(isInterim, bucket.at("is_interim").as_bool());
        } else {
            BOOST_TEST_REQUIRE(!bucket.contains("is_interim"));
        }

        BOOST_REQUIRE_EQUAL(std::uint64_t(10ll),
                            bucket.at("processing_time_ms").to_number<std::uint64_t>());
    }

    for (std::size_t i = 0; i < arrayDoc.as_array().size(); i = i + 2) {
        int buckettime = bucketTimes[i];

        const json::value& recordsWrapper_ = arrayDoc.as_array().at(i);
        const json::object& recordsWrapper = recordsWrapper_.as_object();

        BOOST_TEST_REQUIRE(recordsWrapper.contains("records"));

        const json::value& records_ = recordsWrapper.at("records");
        BOOST_TEST_REQUIRE(records_.is_array());
        const json::array& records = records_.as_array();

        BOOST_REQUIRE_EQUAL(std::size_t(5), records.size());

        // 1st record is for population detector
        {
            const json::value& record_ = records[std::size_t(0)];
            const json::object& record = record_.as_object();
            BOOST_TEST_REQUIRE(record.contains("job_id"));

            BOOST_REQUIRE_EQUAL("job", record.at("job_id").as_string());
            BOOST_TEST_REQUIRE(record.contains("detector_index"));
            BOOST_REQUIRE_EQUAL(1, record.at("detector_index").as_int64());
            BOOST_TEST_REQUIRE(record.contains("timestamp"));
            BOOST_REQUIRE_EQUAL(buckettime, record.at("timestamp").as_int64());
            BOOST_TEST_REQUIRE(record.contains("probability"));
            BOOST_REQUIRE_EQUAL(0.0, record.at("probability").to_number<double>());
            BOOST_TEST_REQUIRE(record.contains("by_field_name"));
            BOOST_REQUIRE_EQUAL("airline", record.at("by_field_name").as_string());
            BOOST_TEST_REQUIRE(!record.contains("by_field_value"));
            BOOST_TEST_REQUIRE(!record.contains("correlated_by_field_value"));
            BOOST_TEST_REQUIRE(record.contains("function"));
            BOOST_REQUIRE_EQUAL("mean", record.at("function").as_string());
            BOOST_TEST_REQUIRE(record.contains("function_description"));
            BOOST_REQUIRE_EQUAL("mean(responsetime)",
                                record.at("function_description").as_string());
            BOOST_TEST_REQUIRE(record.contains("over_field_name"));
            BOOST_REQUIRE_EQUAL("pfn", record.at("over_field_name").as_string());
            BOOST_TEST_REQUIRE(record.contains("over_field_value"));
            BOOST_REQUIRE_EQUAL("pfv", record.at("over_field_value").as_string());
            BOOST_TEST_REQUIRE(record.contains("bucket_span"));
            BOOST_REQUIRE_EQUAL(100, record.at("bucket_span").as_int64());
            // It's hard to predict what these will be, so just assert their
            // presence
            BOOST_TEST_REQUIRE(record.contains("initial_record_score"));
            BOOST_TEST_REQUIRE(record.contains("record_score"));
            if (isInterim) {
                BOOST_TEST_REQUIRE(record.contains("is_interim"));
                BOOST_REQUIRE_EQUAL(isInterim, record.at("is_interim").as_bool());
            } else {
                BOOST_TEST_REQUIRE(!record.contains("is_interim"));
            }

            BOOST_TEST_REQUIRE(record.contains("causes"));
            const json::value& causes_ = record.at("causes");
            BOOST_TEST_REQUIRE(causes_.is_array());
            const json::array& causes = causes_.as_array();

            BOOST_REQUIRE_EQUAL(std::size_t(2), causes.size());
            for (std::size_t k = 0; k < causes.size(); k++) {
                const json::value& cause_ = causes[k];
                const json::object& cause = cause_.as_object();
                BOOST_TEST_REQUIRE(cause.contains("probability"));
                BOOST_REQUIRE_EQUAL(0.0, cause.at("probability").to_number<double>());
                BOOST_TEST_REQUIRE(cause.contains("field_name"));
                BOOST_REQUIRE_EQUAL("responsetime", cause.at("field_name").as_string());
                BOOST_TEST_REQUIRE(cause.contains("by_field_name"));
                BOOST_REQUIRE_EQUAL("airline", cause.at("by_field_name").as_string());
                BOOST_TEST_REQUIRE(cause.contains("by_field_value"));
                BOOST_REQUIRE_EQUAL("GAL", cause.at("by_field_value").as_string());
                BOOST_TEST_REQUIRE(cause.contains("correlated_by_field_value"));
                BOOST_REQUIRE_EQUAL("BAW", cause.at("correlated_by_field_value").as_string());
                BOOST_TEST_REQUIRE(cause.contains("partition_field_name"));
                BOOST_REQUIRE_EQUAL("tfn", cause.at("partition_field_name").as_string());
                BOOST_TEST_REQUIRE(cause.contains("partition_field_value"));
                BOOST_REQUIRE_EQUAL("", cause.at("partition_field_value").as_string());
                BOOST_TEST_REQUIRE(cause.contains("function"));
                BOOST_REQUIRE_EQUAL("mean", cause.at("function").as_string());
                BOOST_TEST_REQUIRE(cause.contains("function_description"));
                BOOST_REQUIRE_EQUAL("mean(responsetime)",
                                    cause.at("function_description").as_string());
                BOOST_TEST_REQUIRE(cause.contains("typical"));
                BOOST_TEST_REQUIRE(cause.at("typical").is_array());
                BOOST_REQUIRE_EQUAL(std::size_t(1),
                                    cause.at("typical").as_array().size());
                BOOST_REQUIRE_EQUAL(
                    6953.0,
                    cause.at("typical").as_array().at(std::size_t(0)).to_number<double>());
                BOOST_TEST_REQUIRE(cause.contains("actual"));
                BOOST_TEST_REQUIRE(cause.at("actual").is_array());
                BOOST_REQUIRE_EQUAL(std::size_t(1),
                                    cause.at("actual").as_array().size());
                BOOST_REQUIRE_EQUAL(
                    10090.0,
                    cause.at("actual").as_array().at(std::size_t(0)).to_number<double>());
                BOOST_TEST_REQUIRE(cause.contains("function"));
            }
        }

        // Next 2 records are for metric detector
        {
            for (std::size_t k = 1; k < 3; k++) {
                const json::value& record_ = records[k];
                const json::object& record = record_.as_object();

                BOOST_TEST_REQUIRE(record.contains("job_id"));
                BOOST_REQUIRE_EQUAL("job", record.at("job_id").as_string());
                BOOST_TEST_REQUIRE(record.contains("detector_index"));
                BOOST_REQUIRE_EQUAL(2, record.at("detector_index").as_int64());
                BOOST_TEST_REQUIRE(record.contains("timestamp"));
                BOOST_REQUIRE_EQUAL(buckettime, record.at("timestamp").as_int64());
                BOOST_TEST_REQUIRE(record.contains("probability"));
                BOOST_REQUIRE_EQUAL(0.0, record.at("probability").to_number<double>());
                BOOST_TEST_REQUIRE(record.contains("by_field_name"));
                BOOST_REQUIRE_EQUAL("airline", record.at("by_field_name").as_string());
                BOOST_TEST_REQUIRE(record.contains("by_field_value"));
                BOOST_REQUIRE_EQUAL("GAL", record.at("by_field_value").as_string());
                BOOST_TEST_REQUIRE(record.contains("correlated_by_field_value"));
                BOOST_REQUIRE_EQUAL(
                    "BAW", record.at("correlated_by_field_value").as_string());
                BOOST_TEST_REQUIRE(record.contains("typical"));
                BOOST_TEST_REQUIRE(record.at("typical").is_array());
                BOOST_REQUIRE_EQUAL(std::size_t(1),
                                    record.at("typical").as_array().size());
                BOOST_REQUIRE_EQUAL(
                    6953.0,
                    record.at("typical").as_array().at(std::size_t(0)).to_number<double>());
                BOOST_TEST_REQUIRE(record.contains("actual"));
                BOOST_TEST_REQUIRE(record.at("actual").is_array());
                BOOST_REQUIRE_EQUAL(std::size_t(1),
                                    record.at("actual").as_array().size());
                BOOST_REQUIRE_EQUAL(
                    10090.0,
                    record.at("actual").as_array().at(std::size_t(0)).to_number<double>());
                BOOST_TEST_REQUIRE(record.contains("field_name"));
                BOOST_REQUIRE_EQUAL("responsetime", record.at("field_name").as_string());
                BOOST_TEST_REQUIRE(record.contains("function"));
                BOOST_REQUIRE_EQUAL("mean", record.at("function").as_string());
                BOOST_TEST_REQUIRE(record.contains("function_description"));
                BOOST_REQUIRE_EQUAL("mean(responsetime)",
                                    record.at("function_description").as_string());
                BOOST_TEST_REQUIRE(record.contains("partition_field_name"));
                BOOST_REQUIRE_EQUAL("tfn", record.at("partition_field_name").as_string());
                BOOST_TEST_REQUIRE(record.contains("partition_field_value"));
                BOOST_REQUIRE_EQUAL("", record.at("partition_field_value").as_string());
                BOOST_TEST_REQUIRE(record.contains("bucket_span"));
                BOOST_REQUIRE_EQUAL(100, record.at("bucket_span").as_int64());
                // It's hard to predict what these will be, so just assert their
                // presence
                BOOST_TEST_REQUIRE(record.contains("initial_record_score"));
                BOOST_TEST_REQUIRE(record.contains("record_score"));
                if (isInterim) {
                    BOOST_TEST_REQUIRE(record.contains("is_interim"));
                    BOOST_REQUIRE_EQUAL(isInterim, record.at("is_interim").as_bool());
                } else {
                    BOOST_TEST_REQUIRE(!record.contains("is_interim"));
                }
            }
        }

        // Last 2 records are for event rate detector
        {
            for (std::size_t k = 3; k < 5; k++) {
                const json::value& record_ = records[k];
                const json::object& record = record_.as_object();

                BOOST_TEST_REQUIRE(record.contains("job_id"));
                BOOST_REQUIRE_EQUAL("job", record.at("job_id").as_string());
                BOOST_TEST_REQUIRE(record.contains("detector_index"));
                BOOST_REQUIRE_EQUAL(4, record.at("detector_index").as_int64());
                BOOST_TEST_REQUIRE(record.contains("timestamp"));
                BOOST_REQUIRE_EQUAL(buckettime, record.at("timestamp").as_int64());
                BOOST_TEST_REQUIRE(record.contains("probability"));
                BOOST_REQUIRE_EQUAL(0.0, record.at("probability").to_number<double>());
                BOOST_TEST_REQUIRE(record.contains("by_field_name"));
                BOOST_REQUIRE_EQUAL("airline", record.at("by_field_name").as_string());
                BOOST_TEST_REQUIRE(record.contains("by_field_value"));
                BOOST_REQUIRE_EQUAL("GAL", record.at("by_field_value").as_string());
                BOOST_TEST_REQUIRE(record.contains("correlated_by_field_value"));
                BOOST_REQUIRE_EQUAL(
                    "BAW", record.at("correlated_by_field_value").as_string());
                BOOST_TEST_REQUIRE(record.contains("typical"));
                BOOST_TEST_REQUIRE(record.at("typical").is_array());
                BOOST_REQUIRE_EQUAL(std::size_t(1),
                                    record.at("typical").as_array().size());
                BOOST_REQUIRE_EQUAL(
                    6953.0,
                    record.at("typical").as_array().at(std::size_t(0)).to_number<double>());
                BOOST_TEST_REQUIRE(record.contains("actual"));
                BOOST_TEST_REQUIRE(record.at("actual").is_array());
                BOOST_REQUIRE_EQUAL(std::size_t(1),
                                    record.at("actual").as_array().size());
                BOOST_REQUIRE_EQUAL(
                    10090.0,
                    record.at("actual").as_array().at(std::size_t(0)).to_number<double>());
                BOOST_TEST_REQUIRE(record.contains("function"));
                // This would be count in the real case with properly generated input data
                BOOST_REQUIRE_EQUAL("mean", record.at("function").as_string());
                BOOST_TEST_REQUIRE(record.contains("function_description"));
                BOOST_REQUIRE_EQUAL("mean(responsetime)",
                                    record.at("function_description").as_string());
                BOOST_TEST_REQUIRE(record.contains("partition_field_name"));
                BOOST_REQUIRE_EQUAL("tfn", record.at("partition_field_name").as_string());
                BOOST_TEST_REQUIRE(record.contains("partition_field_value"));
                BOOST_REQUIRE_EQUAL("", record.at("partition_field_value").as_string());
                BOOST_TEST_REQUIRE(record.contains("bucket_span"));
                BOOST_REQUIRE_EQUAL(100, record.at("bucket_span").as_int64());
                // It's hard to predict what these will be, so just assert their
                // presence
                BOOST_TEST_REQUIRE(record.contains("initial_record_score"));
                BOOST_TEST_REQUIRE(record.contains("record_score"));
                if (isInterim) {
                    BOOST_TEST_REQUIRE(record.contains("is_interim"));
                    BOOST_REQUIRE_EQUAL(isInterim, record.at("is_interim").as_bool());
                } else {
                    BOOST_TEST_REQUIRE(!record.contains("is_interim"));
                }
            }
        }
    }
}

void testLimitedRecordsWriteHelper(bool isInterim) {
    // Tests CJsonOutputWriter::limitNumberRecords(size_t)
    // set the record limit for each detector to 2

    std::ostringstream sstream;

    // The output writer won't close the JSON structures until it is destroyed
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        writer.limitNumberRecords(2);

        std::string partitionFieldName("tfn");
        std::string partitionFieldValue("tfv");
        std::string overFieldName("pfn");
        std::string overFieldValue("pfv");
        std::string byFieldName("airline");
        std::string byFieldValue("GAL");
        std::string fieldName("responsetime");
        std::string function("mean");
        std::string functionDescription("mean(responsetime)");
        std::string emptyString;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;

        {
            // 1st bucket
            ml::api::CHierarchicalResultsWriter::SResults result111(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 1,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 0.0, 0.1, 0.1, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result111));

            ml::api::CHierarchicalResultsWriter::SResults result112(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 1,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 0.0, 0.1, 0.2, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result112));

            ml::api::CHierarchicalResultsWriter::SResults result113(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 1,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.0, 0.0, 0.4, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result113));

            ml::api::CHierarchicalResultsWriter::SResults result114(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 1,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 12.0, 0.0, 0.4, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result114));
            BOOST_TEST_REQUIRE(writer.acceptResult(result114));

            overFieldName = "ofn";
            overFieldValue = "ofv";

            ml::api::CHierarchicalResultsWriter::SResults result121(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, emptyString, emptyString, emptyString, 1, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                12.0, 0.0, 0.01, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result121));

            ml::api::CHierarchicalResultsWriter::SResults result122(
                false, true, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, byFieldName, byFieldValue, emptyString, 1, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                12.0, 0.0, 0.01, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result122));

            ml::api::CHierarchicalResultsWriter::SResults result123(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, byFieldName, byFieldValue, emptyString, 1, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                0.5, 0.0, 0.5, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result123));

            ml::api::CHierarchicalResultsWriter::SResults result124(
                false, true, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, emptyString, emptyString, emptyString, 1, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                0.5, 0.0, 0.5, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result124));

            ml::api::CHierarchicalResultsWriter::SResults result125(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, byFieldName, byFieldValue, emptyString, 1, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                6.0, 0.0, 0.5, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result125));

            ml::api::CHierarchicalResultsWriter::SResults result126(
                false, true, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, emptyString, emptyString, emptyString, 1, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                6.0, 0.0, 0.05, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result126));
        }

        {
            // 2nd bucket
            overFieldName.clear();
            overFieldValue.clear();

            ml::api::CHierarchicalResultsWriter::SResults result211(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 2,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 1.0, 0.0, 0.05, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result211));

            ml::api::CHierarchicalResultsWriter::SResults result212(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 2,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 7.0, 0.0, 0.001, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result212));

            ml::api::CHierarchicalResultsWriter::SResults result213(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 2,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 0.6, 0.0, 0.1, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result213));
            BOOST_TEST_REQUIRE(writer.acceptResult(result213));

            overFieldName = "ofn";
            overFieldValue = "ofv";

            ml::api::CHierarchicalResultsWriter::SResults result221(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, byFieldName, byFieldValue, emptyString, 2, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                0.6, 0.0, 0.1, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result221));
            BOOST_TEST_REQUIRE(writer.acceptResult(result221));

            ml::api::CHierarchicalResultsWriter::SResults result222(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, emptyString, emptyString, emptyString, 2, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                0.6, 0.0, 0.1, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result222));

            ml::api::CHierarchicalResultsWriter::SResults result223(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, byFieldName, byFieldValue, emptyString, 2, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                3.0, 0.0, 0.02, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result223));

            ml::api::CHierarchicalResultsWriter::SResults result224(
                false, true, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, emptyString, emptyString, emptyString, 2, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                20.0, 0.0, 0.02, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result224));
        }

        {
            // 3rd bucket
            overFieldName.clear();
            overFieldValue.clear();

            ml::api::CHierarchicalResultsWriter::SResults result311(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, emptyString, 3,
                function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 30.0, 0.0, 0.02, -5.0, fieldName,
                influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result311));

            overFieldName = "ofn";
            overFieldValue = "ofv";

            ml::api::CHierarchicalResultsWriter::SResults result321(
                false, false, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, byFieldName, byFieldValue, emptyString, 3, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                31.0, 0.0, 0.0002, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result321));

            ml::api::CHierarchicalResultsWriter::SResults result322(
                false, true, partitionFieldName, partitionFieldValue, overFieldName,
                overFieldValue, emptyString, emptyString, emptyString, 3, function,
                functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
                31.0, 0.0, 0.0002, 79, fieldName, influences, false, true, 2, 100);
            BOOST_TEST_REQUIRE(writer.acceptResult(result322));
        }

        // Finished adding results
        BOOST_TEST_REQUIRE(writer.endOutputBatch(isInterim, 10U));
    }

    json::error_code ec;
    json::value arrayDoc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    LOG_DEBUG(<< "Results:\n" << arrayDoc_);

    BOOST_TEST_REQUIRE(arrayDoc_.is_array());
    const json::array& arrayDoc = arrayDoc_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(6), arrayDoc.size());

    // buckets and records are the top level objects
    // records corresponding to a bucket appear first. The bucket follows.
    // each bucket has max 2 records from either both or
    // one or the other of the 2 detectors used.
    // records are sorted by probability.
    // bucket total anomaly score is the sum of all anomalies not just those printed.
    {
        const json::value& bucketWrapper_ = arrayDoc.at(std::size_t(1));
        BOOST_TEST_REQUIRE(bucketWrapper_.is_object());
        const json::object& bucketWrapper = bucketWrapper_.as_object();
        BOOST_TEST_REQUIRE(bucketWrapper.contains("bucket"));

        const json::value& bucket_ = bucketWrapper.at("bucket");
        BOOST_TEST_REQUIRE(bucket_.is_object());
        const json::object& bucket = bucket_.as_object();
        // It's hard to predict what these will be, so just assert their presence
        BOOST_TEST_REQUIRE(bucket.contains("anomaly_score"));
        if (isInterim) {
            BOOST_TEST_REQUIRE(bucket.contains("is_interim"));
            BOOST_REQUIRE_EQUAL(isInterim, bucket.at("is_interim").as_bool());
        } else {
            BOOST_TEST_REQUIRE(!bucket.contains("is_interim"));
        }

        const json::value& recordsWrapper_ = arrayDoc.at(std::size_t(0));
        BOOST_TEST_REQUIRE(recordsWrapper_.is_object());
        const json::object& recordsWrapper = recordsWrapper_.as_object();
        BOOST_TEST_REQUIRE(recordsWrapper.contains("records"));
        const json::value& records_ = recordsWrapper.at("records");
        BOOST_TEST_REQUIRE(records_.is_array());
        const json::array& records = records_.as_array();

        double EXPECTED_PROBABILITIES[] = {0.01, 0.05, 0.001, 0.02, 0.0002};

        int probIndex = 0;
        for (std::size_t i = 0; i < records.size(); i++) {
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("detector_index"));
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("initial_record_score"));
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("record_score"));
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("probability"));
            BOOST_REQUIRE_EQUAL(
                EXPECTED_PROBABILITIES[probIndex],
                records.at(i).as_object().at("probability").to_number<double>());
            ++probIndex;

            if (isInterim) {
                BOOST_TEST_REQUIRE(records.at(i).as_object().contains("is_interim"));
                BOOST_REQUIRE_EQUAL(
                    isInterim, records.at(i).as_object().at("is_interim").as_bool());
            } else {
                BOOST_TEST_REQUIRE(!records.at(i).as_object().contains("is_interim"));
            }
        }

        BOOST_REQUIRE_EQUAL(std::size_t(2), records.size());
    }
    {
        const json::value& bucketWrapper_ = arrayDoc.at(std::size_t(3));
        BOOST_TEST_REQUIRE(bucketWrapper_.is_object());
        const json::object& bucketWrapper = bucketWrapper_.as_object();
        BOOST_TEST_REQUIRE(bucketWrapper.contains("bucket"));

        const json::value& bucket_ = bucketWrapper.at("bucket");
        BOOST_TEST_REQUIRE(bucket_.is_object());
        const json::object& bucket = bucket_.as_object();
        // It's hard to predict what these will be, so just assert their presence
        BOOST_TEST_REQUIRE(bucket.contains("anomaly_score"));
        if (isInterim) {
            BOOST_TEST_REQUIRE(bucket.contains("is_interim"));
            BOOST_REQUIRE_EQUAL(isInterim, bucket.at("is_interim").as_bool());
        } else {
            BOOST_TEST_REQUIRE(!bucket.contains("is_interim"));
        }

        const json::value& recordsWrapper_ = arrayDoc.at(std::size_t(2));
        BOOST_TEST_REQUIRE(recordsWrapper_.is_object());
        const json::object& recordsWrapper = recordsWrapper_.as_object();
        BOOST_TEST_REQUIRE(recordsWrapper.contains("records"));
        const json::value& records_ = recordsWrapper.at("records");
        BOOST_TEST_REQUIRE(records_.is_array());
        const json::array& records = records_.as_array();
        for (std::size_t i = 0; i < records.size(); i++) {
            //BOOST_REQUIRE_EQUAL(0.1, records1[std::size_t(0)]["probability").to_number<double>());
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("detector_index"));
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("initial_record_score"));
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("record_score"));
            if (isInterim) {
                BOOST_TEST_REQUIRE(records.at(i).as_object().contains("is_interim"));
                BOOST_REQUIRE_EQUAL(
                    isInterim, records.at(i).as_object().at("is_interim").as_bool());
            } else {
                BOOST_TEST_REQUIRE(!records.at(i).as_object().contains("is_interim"));
            }
        }

        BOOST_REQUIRE_EQUAL(std::size_t(2), records.size());
    }
    {
        const json::value& bucketWrapper_ = arrayDoc.at(std::size_t(5));
        BOOST_TEST_REQUIRE(bucketWrapper_.is_object());
        const json::object& bucketWrapper = bucketWrapper_.as_object();
        BOOST_TEST_REQUIRE(bucketWrapper.contains("bucket"));

        const json::value& bucket_ = bucketWrapper.at("bucket");
        BOOST_TEST_REQUIRE(bucket_.is_object());
        const json::object& bucket = bucket_.as_object();
        // It's hard to predict what these will be, so just assert their presence
        BOOST_TEST_REQUIRE(bucket.contains("anomaly_score"));
        if (isInterim) {
            BOOST_TEST_REQUIRE(bucket.contains("is_interim"));
            BOOST_REQUIRE_EQUAL(isInterim, bucket.at("is_interim").as_bool());
        } else {
            BOOST_TEST_REQUIRE(!bucket.contains("is_interim"));
        }

        const json::value& recordsWrapper_ = arrayDoc.at(std::size_t(4));
        BOOST_TEST_REQUIRE(recordsWrapper_.is_object());
        const json::object& recordsWrapper = recordsWrapper_.as_object();
        BOOST_TEST_REQUIRE(recordsWrapper.contains("records"));
        const json::value& records_ = recordsWrapper.at("records");
        BOOST_TEST_REQUIRE(records_.is_array());
        const json::array& records = records_.as_array();

        for (std::size_t i = 0; i < records.size(); i++) {
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("detector_index"));
            //BOOST_REQUIRE_EQUAL(0.1, records1[std::size_t(0)]["probability").to_number<double>());
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("initial_record_score"));
            BOOST_TEST_REQUIRE(records.at(i).as_object().contains("record_score"));
            if (isInterim) {
                BOOST_TEST_REQUIRE(records.at(i).as_object().contains("is_interim"));
                BOOST_REQUIRE_EQUAL(
                    isInterim, records.at(i).as_object().at("is_interim").as_bool());
            } else {
                BOOST_TEST_REQUIRE(!records.at(i).as_object().contains("is_interim"));
            }
        }

        BOOST_REQUIRE_EQUAL(std::size_t(2), records.size());
    }
}

ml::model::CHierarchicalResults::TNode
createInfluencerNode(const std::string& personName,
                     const std::string& personValue,
                     double probability,
                     double normalisedAnomalyScore) {
    ml::model::CHierarchicalResults::TResultSpec spec;
    spec.s_PersonFieldName = ml::model::CStringStore::names().get(personName);
    spec.s_PersonFieldValue = ml::model::CStringStore::names().get(personValue);

    ml::model::CHierarchicalResults::TNode node;
    node.s_AnnotatedProbability.s_Probability = probability;
    node.s_NormalizedAnomalyScore = normalisedAnomalyScore;
    node.s_Spec = spec;

    return node;
}

ml::model::CHierarchicalResults::TNode
createBucketInfluencerNode(const std::string& personName,
                           double probability,
                           double normalisedAnomalyScore,
                           double rawAnomalyScore) {
    ml::model::CHierarchicalResults::TResultSpec spec;
    spec.s_PersonFieldName = ml::model::CStringStore::names().get(personName);

    ml::model::CHierarchicalResults::TNode node;
    node.s_AnnotatedProbability.s_Probability = probability;
    node.s_NormalizedAnomalyScore = normalisedAnomalyScore;
    node.s_RawAnomalyScore = rawAnomalyScore;
    node.s_Spec = spec;

    return node;
}

void testThroughputHelper(bool useScopedAllocator) {
    // Write to /dev/null (Unix) or nul (Windows)
    std::ofstream ofs(ml::core::COsFileFuncs::NULL_FILENAME);
    BOOST_TEST_REQUIRE(ofs.is_open());

    ml::core::CJsonOutputStreamWrapper outputStream(ofs);
    ml::api::CJsonOutputWriter writer("job", outputStream);

    std::string partitionFieldName("tfn");
    std::string partitionFieldValue("");
    std::string overFieldName("pfn");
    std::string overFieldValue("pfv");
    std::string byFieldName("airline");
    std::string byFieldValue("GAL");
    std::string correlatedByFieldValue("BAW");
    std::string fieldName("responsetime");
    std::string function("mean");
    std::string functionDescription("mean(responsetime)");
    std::string emptyString;
    ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;

    ml::api::CHierarchicalResultsWriter::SResults result11(
        false, false, partitionFieldName, partitionFieldValue, overFieldName,
        overFieldValue, byFieldName, byFieldValue, correlatedByFieldValue, 1, function,
        functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
        2.24, 0.5, 0.0, 79, fieldName, influences, false, false, 1, 100);

    ml::api::CHierarchicalResultsWriter::SResults result112(
        false, true, partitionFieldName, partitionFieldValue, overFieldName,
        overFieldValue, byFieldName, byFieldValue, correlatedByFieldValue, 1, function,
        functionDescription, TDouble1Vec(1, 10090.0), TDouble1Vec(1, 6953.0),
        2.24, 0.5, 0.0, 79, fieldName, influences, false, false, 1, 100);

    ml::api::CHierarchicalResultsWriter::SResults result12(
        ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
        partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
        1, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
        TDouble1Vec(1, 10090.0), 2.24, 0.8, 0.0, -5.0, fieldName, influences,
        false, true, 2, 100, EMPTY_STRING_LIST, {});

    ml::api::CHierarchicalResultsWriter::SResults result13(
        ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
        partitionFieldName, partitionFieldValue, byFieldName, byFieldValue,
        correlatedByFieldValue, 1, function, functionDescription, 42.0, 79,
        TDouble1Vec(1, 6953.0), TDouble1Vec(1, 10090.0), 2.24, 0.5, 0.0, -5.0,
        fieldName, influences, false, false, 3, 100, EMPTY_STRING_LIST, {});

    ml::api::CHierarchicalResultsWriter::SResults result14(
        ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
        partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
        1, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
        TDouble1Vec(1, 10090.0), 2.24, 0.0, 0.0, -5.0, fieldName, influences,
        false, false, 4, 100, EMPTY_STRING_LIST, {});

    // 1st bucket
    writer.acceptBucketTimeInfluencer(1, 0.01, 13.44, 70.0);

    // Write the record this many times
    static const size_t TEST_SIZE(1);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        if (useScopedAllocator) {
            using TScopedAllocator =
                ml::core::CScopedBoostJsonPoolAllocator<ml::api::CJsonOutputWriter>;
            static const std::string ALLOCATOR_ID("CAnomalyJob::writeOutResults");
            TScopedAllocator scopedAllocator(ALLOCATOR_ID, writer);

            BOOST_TEST_REQUIRE(writer.acceptResult(result11));
            BOOST_TEST_REQUIRE(writer.acceptResult(result11));
            BOOST_TEST_REQUIRE(writer.acceptResult(result112));
            BOOST_TEST_REQUIRE(writer.acceptResult(result12));
            BOOST_TEST_REQUIRE(writer.acceptResult(result12));
            BOOST_TEST_REQUIRE(writer.acceptResult(result13));
            BOOST_TEST_REQUIRE(writer.acceptResult(result13));
            BOOST_TEST_REQUIRE(writer.acceptResult(result14));
            BOOST_TEST_REQUIRE(writer.acceptResult(result14));

            // Finished adding results
            BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
        } else {
            BOOST_TEST_REQUIRE(writer.acceptResult(result11));
            BOOST_TEST_REQUIRE(writer.acceptResult(result11));
            BOOST_TEST_REQUIRE(writer.acceptResult(result112));
            BOOST_TEST_REQUIRE(writer.acceptResult(result12));
            BOOST_TEST_REQUIRE(writer.acceptResult(result12));
            BOOST_TEST_REQUIRE(writer.acceptResult(result13));
            BOOST_TEST_REQUIRE(writer.acceptResult(result13));
            BOOST_TEST_REQUIRE(writer.acceptResult(result14));
            BOOST_TEST_REQUIRE(writer.acceptResult(result14));

            // Finished adding results
            BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
        }
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Writing " << TEST_SIZE << " records took " << (end - start) << " seconds");
}
}

BOOST_AUTO_TEST_CASE(testGeoResultsWrite) {

    std::string partitionFieldName("tfn");
    std::string partitionFieldValue("");
    std::string overFieldName("ofn");
    std::string overFieldValue("ofv");
    std::string byFieldName("airline");
    std::string byFieldValue("GAL");
    std::string correlatedByFieldValue("BAW");
    std::string fieldName("location");
    std::string function("lat_long");
    std::string functionDescription("lat_long(location)");
    ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
    std::string emptyString;
    // The output writer won't close the JSON structures until is is destroyed
    {
        std::ostringstream sstream;
        {
            ml::core::CJsonOutputStreamWrapper outputStream(sstream);
            ml::api::CJsonOutputWriter writer("job", outputStream);
            TDouble1Vec actual(2, 0.0);
            actual[0] = 40.0;
            actual[1] = -40.0;
            TDouble1Vec typical(2, 0.0);
            typical[0] = 90.0;
            typical[1] = -90.0;
            ml::api::CHierarchicalResultsWriter::SResults result(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 1, function, functionDescription, 2.24,
                79, typical, actual, 10.0, 10.0, 0.5, 0.0, fieldName,
                influences, false, true, 1, 1, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result));
            BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
        }
        json::error_code ec;
        json::value arrayDoc_ = json::parse(sstream.str(), ec);
        // Debug print record
        { LOG_DEBUG(<< "Results:\n" << arrayDoc_); }
        BOOST_TEST_REQUIRE(arrayDoc_.is_array());
        const json::array& arrayDoc = arrayDoc_.as_array();
        BOOST_REQUIRE_EQUAL(std::size_t(2), arrayDoc.size());
        BOOST_TEST_REQUIRE(arrayDoc.at(std::size_t(0)).as_object().contains("records"));
        const json::value& record_ =
            arrayDoc.at(std::size_t(0)).as_object().at("records").as_array().at(std::size_t(0));
        const json::object& record = record_.as_object();
        BOOST_TEST_REQUIRE(record.contains("typical"));
        BOOST_TEST_REQUIRE(record.contains("actual"));
        BOOST_TEST_REQUIRE(record.contains("geo_results"));
        auto geoResultsObject = record.at("geo_results").as_object();
        BOOST_TEST_REQUIRE(geoResultsObject.contains("actual_point"));
        BOOST_REQUIRE_EQUAL("40.000000000000,-40.000000000000",
                            geoResultsObject.at("actual_point").as_string());
        BOOST_TEST_REQUIRE(geoResultsObject.contains("typical_point"));
        BOOST_REQUIRE_EQUAL("90.000000000000,-90.000000000000",
                            geoResultsObject.at("typical_point").as_string());
    }

    {
        std::ostringstream sstream;
        {
            ml::core::CJsonOutputStreamWrapper outputStream(sstream);
            ml::api::CJsonOutputWriter writer("job", outputStream);
            TDouble1Vec actual(1, 500);
            TDouble1Vec typical(1, 64);
            ml::api::CHierarchicalResultsWriter::SResults result(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 1, function, functionDescription, 2.24,
                79, typical, actual, 10.0, 10.0, 0.5, 0.0, fieldName,
                influences, false, true, 1, 1, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result));
            BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
        }
        json::error_code ec;
        json::value arrayDoc_ = json::parse(sstream.str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        // Debug print record
        LOG_DEBUG(<< "Results:\n" << arrayDoc_);
        BOOST_TEST_REQUIRE(arrayDoc_.is_array());
        const json::array& arrayDoc = arrayDoc_.as_array();
        BOOST_REQUIRE_EQUAL(std::size_t(2), arrayDoc.size());
        BOOST_TEST_REQUIRE(arrayDoc.at(std::size_t(0)).as_object().contains("records"));
        const json::value& record_ =
            arrayDoc.at(std::size_t(0)).at("records").as_array().at(std::size_t(0));
        BOOST_TEST_REQUIRE(record_.is_object());
        json::object record = record_.as_object();
        BOOST_TEST_REQUIRE(record.contains("geo_results"));
        auto geoResultsObject = record.at("geo_results").as_object();
        BOOST_TEST_REQUIRE(!geoResultsObject.contains("actual_point"));
        BOOST_TEST_REQUIRE(!geoResultsObject.contains("typical_point"));
    }

    {
        std::ostringstream sstream;
        {
            ml::core::CJsonOutputStreamWrapper outputStream(sstream);
            ml::api::CJsonOutputWriter writer("job", outputStream);
            TDouble1Vec actual(1, 500);
            TDouble1Vec typical(1, 64);
            ml::api::CHierarchicalResultsWriter::SResults result(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue,
                correlatedByFieldValue, 1, "mean", functionDescription, 2.24,
                79, typical, actual, 10.0, 10.0, 0.5, 0.0, fieldName,
                influences, false, true, 1, 1, EMPTY_STRING_LIST, {});
            BOOST_TEST_REQUIRE(writer.acceptResult(result));
            BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
        }
        json::error_code ec;
        json::value arrayDoc_ = json::parse(sstream.str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        // Debug print record
        LOG_DEBUG(<< "Results:\n" << arrayDoc_);
        BOOST_TEST_REQUIRE(arrayDoc_.is_array());
        const json::array& arrayDoc = arrayDoc_.as_array();
        BOOST_REQUIRE_EQUAL(std::size_t(2), arrayDoc.size());
        BOOST_TEST_REQUIRE(arrayDoc.at(std::size_t(0)).as_object().contains("records"));
        const json::value& record =
            arrayDoc.at(std::size_t(0)).at("records").as_array().at(std::size_t(0));

        BOOST_TEST_REQUIRE(record.is_object());
        BOOST_REQUIRE_EQUAL(false, record.as_object().contains("geo_results"));
    }
}

BOOST_AUTO_TEST_CASE(testWriteNonAnomalousBucket) {
    std::ostringstream sstream;

    std::string function("mean");
    std::string functionDescription("mean(responsetime)");
    std::string emptyString;
    ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::api::CHierarchicalResultsWriter::SResults result(
            false, false, emptyString, emptyString, emptyString, emptyString,
            emptyString, emptyString, emptyString, 1, function,
            functionDescription, TDouble1Vec(1, 42.0), TDouble1Vec(1, 42.0),
            0.0, 0.0, 1.0, 30, emptyString, influences, false, false, 1, 100);

        BOOST_TEST_REQUIRE(writer.acceptResult(result));
        writer.acceptBucketTimeInfluencer(1, 1.0, 0.0, 0.0);
        BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 10U));
        writer.finalise();
    }

    json::error_code ec;
    json::value arrayDoc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    // Debug print record
    LOG_DEBUG(<< "Results:\n" << arrayDoc_);
    BOOST_TEST_REQUIRE(arrayDoc_.is_array());
    const json::array& arrayDoc = arrayDoc_.as_array();

    BOOST_REQUIRE_EQUAL(std::size_t(1), arrayDoc.size());

    const json::value& bucketWrapper_ = arrayDoc.at(std::size_t(0));
    BOOST_TEST_REQUIRE(bucketWrapper_.is_object());
    const json::object& bucketWrapper = bucketWrapper_.as_object();
    BOOST_TEST_REQUIRE(bucketWrapper.contains("bucket"));

    const json::value& bucket_ = bucketWrapper_.at("bucket");
    const json::object& bucket = bucket_.as_object();
    BOOST_TEST_REQUIRE(bucket.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", bucket.at("job_id").as_string());
    BOOST_REQUIRE_EQUAL(1000, bucket.at("timestamp").as_int64());
    BOOST_TEST_REQUIRE(bucket.contains("bucket_influencers") == false);
    BOOST_REQUIRE_EQUAL(0, bucket.at("event_count").as_int64());
    BOOST_TEST_REQUIRE(bucket.contains("anomaly_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, bucket.at("anomaly_score").to_number<double>(), 0.00001);
}

BOOST_AUTO_TEST_CASE(testFlush) {
    std::string testId("testflush");
    ml::core_t::TTime lastFinalizedBucketEnd(123456789);
    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        writer.acknowledgeFlush(testId, lastFinalizedBucketEnd);
    }

    json::error_code ec;
    json::value arrayDoc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(arrayDoc_.is_array());
    const json::array& arrayDoc = arrayDoc_.as_array();
    LOG_DEBUG(<< "Flush:\n" << arrayDoc);

    const json::value& flushWrapper_ = arrayDoc.at(std::size_t(0));
    BOOST_TEST_REQUIRE(flushWrapper_.is_object());
    const json::object& flushWrapper = flushWrapper_.as_object();
    BOOST_TEST_REQUIRE(flushWrapper.contains("flush"));

    const json::value& flush_ = flushWrapper.at("flush");
    BOOST_TEST_REQUIRE(flush_.is_object());
    const json::object& flush = flush_.as_object();
    BOOST_TEST_REQUIRE(flush.contains("id"));
    BOOST_REQUIRE_EQUAL(testId, flush.at("id").as_string());
    BOOST_TEST_REQUIRE(flush.contains("last_finalized_bucket_end"));
    BOOST_REQUIRE_EQUAL(lastFinalizedBucketEnd * 1000,
                        static_cast<ml::core_t::TTime>(
                            flush.at("last_finalized_bucket_end").as_int64()));
}

BOOST_AUTO_TEST_CASE(testWriteCategoryDefinition) {
    ml::api::CGlobalCategoryId categoryId{42};
    std::string terms("foo bar");
    std::string regex(".*?foo.+?bar.*");
    std::size_t maxMatchingLength(132);
    ml::api::CJsonOutputWriter::TStrFSet examples;
    examples.insert("User foo failed to log in");
    examples.insert("User bar failed to log in");

    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        writer.writeCategoryDefinition("", "", categoryId, terms, regex,
                                       maxMatchingLength, examples, 0, {});
    }

    json::error_code ec;
    json::value arrayDoc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(arrayDoc_.is_array());
    const json::array& arrayDoc = arrayDoc_.as_array();
    LOG_DEBUG(<< "CategoryDefinition:\n" << arrayDoc);

    const json::value& categoryWrapper_ = arrayDoc.at(std::size_t(0));
    BOOST_TEST_REQUIRE(categoryWrapper_.is_object());
    const json::object& categoryWrapper = categoryWrapper_.as_object();
    BOOST_TEST_REQUIRE(categoryWrapper.contains("category_definition"));

    const json::value& category_ = categoryWrapper.at("category_definition");
    BOOST_TEST_REQUIRE(category_.is_object());
    json::object category = category_.as_object();
    BOOST_TEST_REQUIRE(category.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", category.at("job_id").as_string());
    BOOST_TEST_REQUIRE(category.contains("partition_field_value") == false);
    BOOST_TEST_REQUIRE(category.contains("category_id"));
    BOOST_REQUIRE_EQUAL(categoryId.globalId(), category.at("category_id").as_int64());
    BOOST_TEST_REQUIRE(category.contains("terms"));
    BOOST_REQUIRE_EQUAL(terms, category.at("terms").as_string());
    BOOST_TEST_REQUIRE(category.contains("regex"));
    BOOST_REQUIRE_EQUAL(regex, category.at("regex").as_string());
    BOOST_TEST_REQUIRE(category.contains("max_matching_length"));
    BOOST_REQUIRE_EQUAL(
        maxMatchingLength,
        static_cast<std::size_t>(category.at("max_matching_length").as_int64()));
    BOOST_TEST_REQUIRE(category.contains("examples"));

    ml::api::CJsonOutputWriter::TStrFSet writtenExamplesSet;
    const json::value& writtenExamples_ = category.at("examples");
    const json::array& writtenExamples = writtenExamples_.as_array();
    for (std::size_t i = 0; i < writtenExamples.size(); i++) {
        writtenExamplesSet.insert(std::string(writtenExamples.at(i).as_string()));
    }
    BOOST_TEST_REQUIRE(writtenExamplesSet == examples);
}

BOOST_AUTO_TEST_CASE(testWritePerPartitionCategoryDefinition) {
    ml::api::CGlobalCategoryId categoryId{42};
    std::string terms("foo bar");
    std::string regex(".*?foo.+?bar.*");
    std::size_t maxMatchingLength(132);
    ml::api::CJsonOutputWriter::TStrFSet examples;
    examples.insert("User foo failed to log in");
    examples.insert("User bar failed to log in");

    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        writer.writeCategoryDefinition("event.dataset", "elasticsearch", categoryId, terms,
                                       regex, maxMatchingLength, examples, 0, {});
    }

    json::error_code ec;
    json::value arrayDoc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(arrayDoc_.is_array());
    const json::array& arrayDoc = arrayDoc_.as_array();
    LOG_DEBUG(<< "CategoryDefinition:\n" << arrayDoc);

    const json::value& categoryWrapper_ = arrayDoc.at(std::size_t(0));
    BOOST_TEST_REQUIRE(categoryWrapper_.is_object());
    const json::object& categoryWrapper = categoryWrapper_.as_object();
    BOOST_TEST_REQUIRE(categoryWrapper.contains("category_definition"));

    const json::value& category_ = categoryWrapper.at("category_definition");
    BOOST_TEST_REQUIRE(category_.is_object());
    const json::object& category = category_.as_object();
    BOOST_TEST_REQUIRE(category.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", category.at("job_id").as_string());
    BOOST_TEST_REQUIRE(category.contains("partition_field_name"));
    BOOST_REQUIRE_EQUAL("event.dataset", category.at("partition_field_name").as_string());
    BOOST_TEST_REQUIRE(category.contains("partition_field_value"));
    BOOST_REQUIRE_EQUAL("elasticsearch", category.at("partition_field_value").as_string());
    BOOST_TEST_REQUIRE(category.contains("category_id"));
    BOOST_REQUIRE_EQUAL(categoryId.globalId(), category.at("category_id").as_int64());
    BOOST_TEST_REQUIRE(category.contains("terms"));
    BOOST_REQUIRE_EQUAL(terms, category.at("terms").as_string());
    BOOST_TEST_REQUIRE(category.contains("regex"));
    BOOST_REQUIRE_EQUAL(regex, category.at("regex").as_string());
    BOOST_TEST_REQUIRE(category.contains("max_matching_length"));
    BOOST_REQUIRE_EQUAL(
        maxMatchingLength,
        static_cast<std::size_t>(category.at("max_matching_length").as_int64()));
    BOOST_TEST_REQUIRE(category.contains("examples"));

    ml::api::CJsonOutputWriter::TStrFSet writtenExamplesSet;
    const json::value& writtenExamples_ = category.at("examples");
    const json::array& writtenExamples = writtenExamples_.as_array();
    for (std::size_t i = 0; i < writtenExamples.size(); i++) {
        writtenExamplesSet.insert(std::string(writtenExamples.at(i).as_string()));
    }
    BOOST_TEST_REQUIRE(writtenExamplesSet == examples);
}

BOOST_AUTO_TEST_CASE(testBucketWrite) {
    testBucketWriteHelper(false);
}

BOOST_AUTO_TEST_CASE(testBucketWriteInterim) {
    testBucketWriteHelper(true);
}

BOOST_AUTO_TEST_CASE(testLimitedRecordsWrite) {
    testLimitedRecordsWriteHelper(false);
}

BOOST_AUTO_TEST_CASE(testLimitedRecordsWriteInterim) {
    testLimitedRecordsWriteHelper(true);
}

BOOST_AUTO_TEST_CASE(testWriteInfluencers) {
    std::ostringstream sstream;

    {
        std::string user("user");
        std::string daisy("daisy");
        std::string jim("jim");

        ml::model::CHierarchicalResults::TNode node1 =
            createInfluencerNode(user, daisy, 0.5, 10.0);
        ml::model::CHierarchicalResults::TNode node2 =
            createInfluencerNode(user, jim, 0.9, 100.0);

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(42), node1, false));
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(42), node2, false));

        // Finished adding results
        BOOST_TEST_REQUIRE(writer.endOutputBatch(true, 1U));
    }

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();
    LOG_DEBUG(<< "influencers:\n" << doc);

    BOOST_REQUIRE_EQUAL(std::size_t(2), doc.size());

    const json::value& influencers_ = doc.at(std::size_t(0)).as_object().at("influencers");
    BOOST_TEST_REQUIRE(influencers_.is_array());
    const json::array& influencers = influencers_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(2), influencers.size());

    const json::value& influencer_ = influencers.at(std::size_t(0));
    const json::object& influencer = influencer_.as_object();
    BOOST_TEST_REQUIRE(influencer.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", influencer.at("job_id").as_string());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.5, influencer.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        10.0, influencer.at("initial_influencer_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(influencer.contains("influencer_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        10.0, influencer.at("influencer_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("user", influencer.at("influencer_field_name").as_string());
    BOOST_REQUIRE_EQUAL("daisy", influencer.at("influencer_field_value").as_string());
    BOOST_REQUIRE_EQUAL(42000, influencer.at("timestamp").as_int64());
    BOOST_TEST_REQUIRE(influencer.at("is_interim").as_bool());
    BOOST_TEST_REQUIRE(influencer.contains("bucket_span"));

    const json::value& influencer2_ = influencers.at(std::size_t(1));
    const json::object& influencer2 = influencer2_.as_object();
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        0.9, influencer2.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        100.0, influencer2.at("initial_influencer_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(influencer2.contains("influencer_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        100.0, influencer2.at("influencer_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("user", influencer2.at("influencer_field_name").as_string());
    BOOST_REQUIRE_EQUAL("jim", influencer2.at("influencer_field_value").as_string());
    BOOST_REQUIRE_EQUAL(42000, influencer2.at("timestamp").as_int64());
    BOOST_TEST_REQUIRE(influencer2.at("is_interim").as_bool());
    BOOST_TEST_REQUIRE(influencer2.contains("bucket_span"));

    const json::value& bucket = doc.at(std::size_t(1)).as_object().at("bucket");
    BOOST_TEST_REQUIRE(bucket.as_object().contains("influencers") == false);
}

BOOST_AUTO_TEST_CASE(testWriteInfluencersWithLimit) {
    std::ostringstream sstream;

    {
        std::string user("user");
        std::string computer("computer");
        std::string monitor("monitor");
        std::string daisy("daisy");
        std::string jim("jim");
        std::string bob("bob");
        std::string laptop("laptop");

        ml::model::CHierarchicalResults::TNode node1 =
            createInfluencerNode(user, daisy, 0.5, 10.0);
        ml::model::CHierarchicalResults::TNode node2 =
            createInfluencerNode(user, jim, 0.9, 100.0);
        ml::model::CHierarchicalResults::TNode node3 =
            createInfluencerNode(user, bob, 0.3, 9.0);
        ml::model::CHierarchicalResults::TNode node4 =
            createInfluencerNode(computer, laptop, 0.3, 12.0);

        ml::model::CHierarchicalResults::TNode bnode1 =
            createBucketInfluencerNode(user, 0.5, 10.0, 1.0);
        ml::model::CHierarchicalResults::TNode bnode2 =
            createBucketInfluencerNode(computer, 0.9, 100.0, 10.0);
        ml::model::CHierarchicalResults::TNode bnode3 =
            createBucketInfluencerNode(monitor, 0.3, 9.0, 0.9);

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        writer.limitNumberRecords(2);

        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), node1, false));
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), node2, false));
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), node3, false));
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), node4, false));

        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), bnode1, true));
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), bnode2, true));
        BOOST_TEST_REQUIRE(writer.acceptInfluencer(ml::core_t::TTime(0), bnode3, true));

        // can't add a bucket influencer unless a result has been added
        std::string pfn("partition_field_name");
        std::string pfv("partition_field_value");
        std::string bfn("by_field_name");
        std::string bfv("by_field_value");
        std::string fun("function");
        std::string fund("function_description");
        std::string fn("field_name");
        std::string emptyStr;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        ml::api::CHierarchicalResultsWriter::SResults result(
            ml::api::CHierarchicalResultsWriter::E_Result, pfn, pfv, bfn, bfv,
            emptyStr, 0, fun, fund, 42.0, 79, TDouble1Vec(1, 6953.0),
            TDouble1Vec(1, 10090.0), 0.0, 0.1, 0.1, -5.0, fn, influences, false,
            true, 1, 100, EMPTY_STRING_LIST, {});

        BOOST_TEST_REQUIRE(writer.acceptResult(result));

        writer.acceptBucketTimeInfluencer(ml::core_t::TTime(0), 0.6, 1.0, 10.0);

        // Finished adding results
        BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
    }

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();

    LOG_DEBUG(<< "limited write influencers:\n" << doc);

    const json::value& influencers_ = doc.at(std::size_t(1)).as_object().at("influencers");
    BOOST_TEST_REQUIRE(influencers_.is_array());
    const json::array& influencers = influencers_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(2), influencers.size());

    const json::value& influencer_ = influencers.at(std::size_t(0));
    const json::object& influencer = influencer_.as_object();
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.9, influencer.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        100.0, influencer.at("initial_influencer_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(influencer.contains("influencer_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        100.0, influencer.at("influencer_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("user", influencer.at("influencer_field_name").as_string());
    BOOST_REQUIRE_EQUAL("jim", influencer.at("influencer_field_value").as_string());
    BOOST_TEST_REQUIRE(influencer.contains("bucket_span"));

    const json::value& influencer2_ = influencers.at(std::size_t(1));
    const json::object& influencer2 = influencer2_.as_object();
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        0.3, influencer2.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        12.0, influencer2.at("initial_influencer_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(influencer2.contains("influencer_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        12.0, influencer2.at("influencer_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("computer", influencer2.at("influencer_field_name").as_string());
    BOOST_REQUIRE_EQUAL("laptop", influencer2.at("influencer_field_value").as_string());
    BOOST_TEST_REQUIRE(influencer2.contains("bucket_span"));

    // bucket influencers
    const json::value& bucketResult_ = doc.at(std::size_t(2)).as_object().at("bucket");
    const json::object& bucketResult = bucketResult_.as_object();
    BOOST_TEST_REQUIRE(bucketResult.contains("bucket_influencers"));
    const json::value& bucketInfluencers_ = bucketResult.at("bucket_influencers");
    BOOST_TEST_REQUIRE(bucketInfluencers_.is_array());
    const json::array& bucketInfluencers = bucketInfluencers_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(3), bucketInfluencers.size());

    const json::value& binf_ = bucketInfluencers.at(std::size_t(0));
    BOOST_TEST_REQUIRE(binf_.is_object());
    const json::object& binf = binf_.as_object();
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.9, binf.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        100.0, binf.at("initial_anomaly_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(binf.contains("anomaly_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(100.0, binf.at("anomaly_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("computer", binf.at("influencer_field_name").as_string());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        10.0, binf.at("raw_anomaly_score").to_number<double>(), 0.001);

    const json::value& binf2_ = bucketInfluencers.at(std::size_t(1));
    BOOST_TEST_REQUIRE(binf2_.is_object());
    const json::object& binf2 = binf2_.as_object();
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.5, binf2.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        10.0, binf2.at("initial_anomaly_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(binf2.contains("anomaly_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(10.0, binf2.at("anomaly_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("user", binf2.at("influencer_field_name").as_string());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        1.0, binf2.at("raw_anomaly_score").to_number<double>(), 0.001);

    const json::value& binf3_ = bucketInfluencers.at(std::size_t(2));
    BOOST_TEST_REQUIRE(binf3_.is_object());
    const json::object& binf3 = binf3_.as_object();
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6, binf3.at("probability").to_number<double>(), 0.001);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        10.0, binf3.at("initial_anomaly_score").to_number<double>(), 0.001);
    BOOST_TEST_REQUIRE(binf3.contains("anomaly_score"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(10.0, binf3.at("anomaly_score").to_number<double>(), 0.001);
    BOOST_REQUIRE_EQUAL("bucket_time", binf3.at("influencer_field_name").as_string());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        1.0, binf3.at("raw_anomaly_score").to_number<double>(), 0.001);
}

BOOST_AUTO_TEST_CASE(testWriteWithInfluences) {
    std::ostringstream sstream;

    {
        std::string partitionFieldName("tfn");
        std::string partitionFieldValue("tfv");
        std::string overFieldName("pfn");
        std::string overFieldValue("pfv");
        std::string byFieldName("airline");
        std::string byFieldValue("GAL");
        std::string fieldName("responsetime");
        std::string function("mean");
        std::string functionDescription("mean(responsetime)");
        std::string emptyString;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;

        std::string user("user");
        std::string dave("dave");
        std::string jo("jo");
        std::string cat("cat");
        std::string host("host");
        std::string localhost("localhost");
        std::string webserver("web-server");

        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr field1 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(
                ml::model::CStringStore::names().get(user),
                ml::model::CStringStore::names().get(dave));
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr field2 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(
                ml::model::CStringStore::names().get(user),
                ml::model::CStringStore::names().get(cat));
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr field3 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(
                ml::model::CStringStore::names().get(user),
                ml::model::CStringStore::names().get(jo));

        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr hostField1 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(
                ml::model::CStringStore::names().get(host),
                ml::model::CStringStore::names().get(localhost));
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr hostField2 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(
                ml::model::CStringStore::names().get(host),
                ml::model::CStringStore::names().get(webserver));

        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(
            field1, 0.4));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(
            field2, 1.0));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(
            hostField1, 0.7));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(
            field3, 0.1));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(
            hostField2, 0.8));

        // The output writer won't close the JSON structures until is is destroyed

        ml::api::CHierarchicalResultsWriter::SResults result(
            ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
            partitionFieldValue, byFieldName, byFieldValue, emptyString, 1,
            function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
            TDouble1Vec(1, 10090.0), 0.0, 0.1, 0.1, -5.0, fieldName, influences,
            false, true, 1, 100, EMPTY_STRING_LIST, {});

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        BOOST_TEST_REQUIRE(writer.acceptResult(result));

        // Finished adding results
        BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
    }

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    // Debug print record
    LOG_DEBUG(<< "Results:\n" << doc_);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();

    BOOST_TEST_REQUIRE(doc.at(std::size_t(1)).as_object().contains("bucket"));
    const json::value& bucket_ = doc.at(std::size_t(1)).as_object().at("bucket");
    const json::object& bucket = bucket_.as_object();
    BOOST_TEST_REQUIRE(bucket.contains("records") == false);

    BOOST_TEST_REQUIRE(doc.at(std::size_t(0)).as_object().contains("records"));
    const json::value& records_ = doc.at(std::size_t(0)).as_object().at("records");
    const json::array& records = records_.as_array();

    BOOST_TEST_REQUIRE(records.at(std::size_t(0)).as_object().contains("influencers"));
    const json::value& influences_ = records.at(std::size_t(0)).as_object().at("influencers");
    BOOST_TEST_REQUIRE(influences_.is_array());
    const json::array& influences = influences_.as_array();

    BOOST_REQUIRE_EQUAL(std::size_t(2), influences.size());

    {
        const json::value& influence_ = influences.at(std::size_t(0));
        const json::object& influence = influence_.as_object();
        BOOST_TEST_REQUIRE(influence.contains("influencer_field_name"));
        BOOST_REQUIRE_EQUAL("host", influence.at("influencer_field_name").as_string());
        BOOST_TEST_REQUIRE(influence.contains("influencer_field_values"));
        const json::value& influencerFieldValues_ = influence.at("influencer_field_values");
        BOOST_TEST_REQUIRE(influencerFieldValues_.is_array());
        const json::array& influencerFieldValues = influencerFieldValues_.as_array();
        BOOST_REQUIRE_EQUAL(std::size_t(2), influencerFieldValues.size());

        // Check influencers are ordered
        BOOST_REQUIRE_EQUAL("web-server",
                            influencerFieldValues.at(std::size_t(0)).as_string());
        BOOST_REQUIRE_EQUAL("localhost",
                            influencerFieldValues.at(std::size_t(1)).as_string());
    }
    {
        const json::value& influence_ = influences.at(std::size_t(1));
        const json::object& influence = influence_.as_object();
        BOOST_TEST_REQUIRE(influence.contains("influencer_field_name"));
        BOOST_REQUIRE_EQUAL("user", influence.at("influencer_field_name").as_string());
        BOOST_TEST_REQUIRE(influence.contains("influencer_field_values"));
        const json::value& influencerFieldValues_ = influence.at("influencer_field_values");
        BOOST_TEST_REQUIRE(influencerFieldValues_.is_array());
        const json::array& influencerFieldValues = influencerFieldValues_.as_array();
        BOOST_REQUIRE_EQUAL(std::size_t(3), influencerFieldValues.size());

        // Check influencers are ordered
        BOOST_REQUIRE_EQUAL("cat", influencerFieldValues.at(std::size_t(0)).as_string());
        BOOST_REQUIRE_EQUAL("dave", influencerFieldValues.at(std::size_t(1)).as_string());
        BOOST_REQUIRE_EQUAL("jo", influencerFieldValues.at(std::size_t(2)).as_string());
    }
}

BOOST_AUTO_TEST_CASE(testPersistNormalizer) {
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig();

    std::ostringstream sstream;
    ml::core_t::TTime persistTime(1);
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::model::CHierarchicalResultsNormalizer normalizer(modelConfig);
        writer.persistNormalizer(normalizer, persistTime);
        writer.finalise();
    }

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    // Debug print record
    LOG_DEBUG(<< "Results:\n" << doc_);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();

    BOOST_TEST_REQUIRE(persistTime <= ml::core::CTimeUtils::now());
    BOOST_TEST_REQUIRE(persistTime > ml::core::CTimeUtils::now() - 10);

    const json::value& quantileWrapper_ = doc.at(std::size_t(0));
    const json::object& quantileWrapper = quantileWrapper_.as_object();
    BOOST_TEST_REQUIRE(quantileWrapper.contains("quantiles"));
    const json::value& quantileState_ = quantileWrapper.at("quantiles");
    const json::object& quantileState = quantileState_.as_object();
    BOOST_TEST_REQUIRE(quantileState.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", quantileState.at("job_id").as_string());
    BOOST_TEST_REQUIRE(quantileState.contains("quantile_state"));
    BOOST_TEST_REQUIRE(quantileState.contains("timestamp"));
}

BOOST_AUTO_TEST_CASE(testReportMemoryUsage) {
    std::ostringstream sstream;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::model::CResourceMonitor::SModelSizeStats resourceUsage;
        resourceUsage.s_Usage = 1;
        resourceUsage.s_AdjustedUsage = 2;
        resourceUsage.s_PeakUsage = 3;
        resourceUsage.s_AdjustedPeakUsage = 4;
        resourceUsage.s_ByFields = 5;
        resourceUsage.s_PartitionFields = 6;
        resourceUsage.s_OverFields = 7;
        resourceUsage.s_AllocationFailures = 8;
        resourceUsage.s_MemoryStatus = ml::model_t::E_MemoryStatusHardLimit;
        resourceUsage.s_AssignmentMemoryBasis = ml::model_t::E_AssignmentBasisCurrentModelBytes;
        resourceUsage.s_BucketStartTime = 9;
        resourceUsage.s_BytesExceeded = 10;
        resourceUsage.s_BytesMemoryLimit = 11;
        resourceUsage.s_OverallCategorizerStats.s_CategorizedMessages = 12;
        resourceUsage.s_OverallCategorizerStats.s_TotalCategories = 13;
        resourceUsage.s_OverallCategorizerStats.s_FrequentCategories = 14;
        resourceUsage.s_OverallCategorizerStats.s_RareCategories = 15;
        resourceUsage.s_OverallCategorizerStats.s_DeadCategories = 16;
        resourceUsage.s_OverallCategorizerStats.s_MemoryCategorizationFailures = 17;
        resourceUsage.s_OverallCategorizerStats.s_CategorizationStatus =
            ml::model_t::E_CategorizationStatusWarn;

        writer.reportMemoryUsage(resourceUsage);
        writer.endOutputBatch(false, 1ul);
    }

    LOG_DEBUG(<< sstream.str());

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();

    const json::value& resourceWrapper_ = doc.at(std::size_t(0));
    const json::object& resourceWrapper = resourceWrapper_.as_object();
    BOOST_TEST_REQUIRE(resourceWrapper.contains("model_size_stats"));
    const json::value& sizeStats_ = resourceWrapper.at("model_size_stats");
    const json::object& sizeStats = sizeStats_.as_object();

    BOOST_TEST_REQUIRE(sizeStats.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", sizeStats.at("job_id").as_string());
    BOOST_TEST_REQUIRE(sizeStats.contains("model_bytes"));
    BOOST_REQUIRE_EQUAL(2, sizeStats.at("model_bytes").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("peak_model_bytes"));
    BOOST_REQUIRE_EQUAL(4, sizeStats.at("peak_model_bytes").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("total_by_field_count"));
    BOOST_REQUIRE_EQUAL(5, sizeStats.at("total_by_field_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("total_partition_field_count"));
    BOOST_REQUIRE_EQUAL(6, sizeStats.at("total_partition_field_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("total_over_field_count"));
    BOOST_REQUIRE_EQUAL(7, sizeStats.at("total_over_field_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("bucket_allocation_failures_count"));
    BOOST_REQUIRE_EQUAL(8, sizeStats.at("bucket_allocation_failures_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(9000, sizeStats.at("timestamp").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("memory_status"));
    BOOST_REQUIRE_EQUAL("hard_limit", sizeStats.at("memory_status").as_string());
    BOOST_TEST_REQUIRE(sizeStats.contains("assignment_memory_basis"));
    BOOST_REQUIRE_EQUAL("current_model_bytes",
                        sizeStats.at("assignment_memory_basis").as_string());
    BOOST_TEST_REQUIRE(sizeStats.contains("log_time"));
    std::int64_t nowMs{ml::core::CTimeUtils::nowMs()};
    BOOST_TEST_REQUIRE(nowMs >= sizeStats.at("log_time").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("model_bytes_exceeded"));
    BOOST_REQUIRE_EQUAL(10, sizeStats.at("model_bytes_exceeded").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("model_bytes_memory_limit"));
    BOOST_REQUIRE_EQUAL(11, sizeStats.at("model_bytes_memory_limit").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("categorized_doc_count"));
    BOOST_REQUIRE_EQUAL(12, sizeStats.at("categorized_doc_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("total_category_count"));
    BOOST_REQUIRE_EQUAL(13, sizeStats.at("total_category_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("frequent_category_count"));
    BOOST_REQUIRE_EQUAL(14, sizeStats.at("frequent_category_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("rare_category_count"));
    BOOST_REQUIRE_EQUAL(15, sizeStats.at("rare_category_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("dead_category_count"));
    BOOST_REQUIRE_EQUAL(16, sizeStats.at("dead_category_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("failed_category_count"));
    BOOST_REQUIRE_EQUAL(17, sizeStats.at("failed_category_count").as_int64());
    BOOST_TEST_REQUIRE(sizeStats.contains("categorization_status"));
    BOOST_REQUIRE_EQUAL("warn", sizeStats.at("categorization_status").as_string());
}

BOOST_AUTO_TEST_CASE(testWriteCategorizerStats) {
    std::ostringstream sstream;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::model::SCategorizerStats categorizerStats;
        categorizerStats.s_CategorizedMessages = 1;
        categorizerStats.s_TotalCategories = 2;
        categorizerStats.s_FrequentCategories = 3;
        categorizerStats.s_RareCategories = 4;
        categorizerStats.s_DeadCategories = 5;
        categorizerStats.s_MemoryCategorizationFailures = 6;
        categorizerStats.s_CategorizationStatus = ml::model_t::E_CategorizationStatusOk;

        writer.writeCategorizerStats("foo", "bar", categorizerStats, 7);
        writer.endOutputBatch(false, 1ul);
    }

    LOG_DEBUG(<< sstream.str());

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();

    const json::value& resourceWrapper_ = doc.at(std::size_t(0));
    const json::object& resourceWrapper = resourceWrapper_.as_object();
    BOOST_TEST_REQUIRE(resourceWrapper.contains("categorizer_stats"));
    const json::value& categorizerStats_ = resourceWrapper.at("categorizer_stats");
    const json::object& categorizerStats = categorizerStats_.as_object();

    BOOST_TEST_REQUIRE(categorizerStats.contains("job_id"));
    BOOST_REQUIRE_EQUAL("job", categorizerStats.at("job_id").as_string());
    BOOST_TEST_REQUIRE(categorizerStats.contains("partition_field_name"));
    BOOST_REQUIRE_EQUAL("foo", categorizerStats.at("partition_field_name").as_string());
    BOOST_TEST_REQUIRE(categorizerStats.contains("partition_field_value"));
    BOOST_REQUIRE_EQUAL("bar", categorizerStats.at("partition_field_value").as_string());
    BOOST_TEST_REQUIRE(categorizerStats.contains("categorized_doc_count"));
    BOOST_REQUIRE_EQUAL(1, categorizerStats.at("categorized_doc_count").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("total_category_count"));
    BOOST_REQUIRE_EQUAL(2, categorizerStats.at("total_category_count").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("frequent_category_count"));
    BOOST_REQUIRE_EQUAL(3, categorizerStats.at("frequent_category_count").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("rare_category_count"));
    BOOST_REQUIRE_EQUAL(4, categorizerStats.at("rare_category_count").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("dead_category_count"));
    BOOST_REQUIRE_EQUAL(5, categorizerStats.at("dead_category_count").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("failed_category_count"));
    BOOST_REQUIRE_EQUAL(6, categorizerStats.at("failed_category_count").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("categorization_status"));
    BOOST_REQUIRE_EQUAL("ok", categorizerStats.at("categorization_status").as_string());
    BOOST_TEST_REQUIRE(categorizerStats.contains("categorization_status"));
    BOOST_REQUIRE_EQUAL("ok", categorizerStats.at("categorization_status").as_string());
    BOOST_TEST_REQUIRE(categorizerStats.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(7000, categorizerStats.at("timestamp").as_int64());
    BOOST_TEST_REQUIRE(categorizerStats.contains("log_time"));
    std::int64_t nowMs{ml::core::CTimeUtils::nowMs()};
    BOOST_TEST_REQUIRE(nowMs >= categorizerStats.at("log_time").as_int64());
}

BOOST_AUTO_TEST_CASE(testWriteScheduledEvent) {
    std::ostringstream sstream;

    {
        std::string partitionFieldName("tfn");
        std::string partitionFieldValue("tfv");
        std::string byFieldName("airline");
        std::string byFieldValue("GAL");
        std::string fieldName("responsetime");
        std::string function("mean");
        std::string functionDescription("mean(responsetime)");
        std::string emptyString;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        // This result has no scheduled events
        ml::api::CHierarchicalResultsWriter::SResults result(
            ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
            partitionFieldName, partitionFieldValue, byFieldName, byFieldValue,
            emptyString, 100, function, functionDescription, 42.0, 79,
            TDouble1Vec(1, 6953.0), TDouble1Vec(1, 10090.0), 0.0, 0.1, 0.1, -5.0,
            fieldName, influences, false, true, 1, 100, EMPTY_STRING_LIST, {});
        BOOST_TEST_REQUIRE(writer.acceptResult(result));

        // This result has 2 scheduled events
        std::vector<std::string> eventDescriptions{"event-foo", "event-bar"};
        ml::api::CHierarchicalResultsWriter::SResults result2(
            ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
            partitionFieldName, partitionFieldValue, byFieldName, byFieldValue,
            emptyString, 200, function, functionDescription, 42.0, 79,
            TDouble1Vec(1, 6953.0), TDouble1Vec(1, 10090.0), 0.0, 0.1, 0.1, -5.0,
            fieldName, influences, false, true, 1, 100, eventDescriptions, {});

        BOOST_TEST_REQUIRE(writer.acceptResult(result2));
        BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 1U));
    }

    json::error_code ec;
    json::value doc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    // Debug print record
    LOG_DEBUG(<< "Results:\n" << doc_);
    BOOST_TEST_REQUIRE(doc_.is_array());
    const json::array& doc = doc_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(2), doc.size());
    // the first bucket has no events
    const json::value& bucket_ = doc.at(std::size_t(1)).as_object().at("bucket");
    const json::object& bucket = bucket_.as_object();
    BOOST_TEST_REQUIRE(bucket.contains("scheduled_event") == false);

    const json::value& bucketWithEvents_ = doc.at(std::size_t(1)).at("bucket");
    const json::object& bucketWithEvents = bucketWithEvents_.as_object();
    BOOST_TEST_REQUIRE(bucketWithEvents.contains("scheduled_events"));
    const json::value& events_ = bucketWithEvents.at("scheduled_events");
    BOOST_TEST_REQUIRE(events_.is_array());
    const json::array& events = events_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(2), events.size());
    BOOST_REQUIRE_EQUAL("event-foo", events.at(std::size_t(0)).as_string());
    BOOST_REQUIRE_EQUAL("event-bar", events.at(std::size_t(1)).as_string());
}

BOOST_AUTO_TEST_CASE(testThroughputWithScopedAllocator) {
    testThroughputHelper(true);
}

BOOST_AUTO_TEST_CASE(testThroughputWithoutScopedAllocator) {
    testThroughputHelper(false);
}

BOOST_AUTO_TEST_CASE(testRareAnomalyScoreExplanation) {
    // Ensure that anomaly score explanation fields for rare events
    // are outputted.
    std::ostringstream sstream;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        std::string partitionFieldName("Carrier");
        std::string partitionFieldValue("JetBeats");
        std::string overFieldName("pfn");
        std::string overFieldValue("pfv");
        std::string byFieldName("Dest");
        std::string byFieldValue("Adelaide International Airport");
        std::string correlatedByFieldValue("BAW");
        std::string fieldName("clientip");
        std::string function("rare");
        std::string functionDescription("rare");
        std::string emptyString;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        ml::api::CHierarchicalResultsWriter::TAnomalyScoreExplanation anomalyScoreExplanation;

        {
            anomalyScoreExplanation.s_ByFieldFirstOccurrence = true;
            anomalyScoreExplanation.s_ByFieldActualConcentration = 0.1;
            anomalyScoreExplanation.s_ByFieldTypicalConcentration = 0.5;
            ml::api::CHierarchicalResultsWriter::SResults result(
                ml::api::CHierarchicalResultsWriter::E_Result, partitionFieldName,
                partitionFieldValue, byFieldName, byFieldValue, correlatedByFieldValue,
                1, function, functionDescription, 42.0, 79, TDouble1Vec(1, 6953.0),
                TDouble1Vec(1, 10090.0), 2.24, 0.8, 0.0, -5.0, fieldName, influences,
                false, false, 2, 100, EMPTY_STRING_LIST, anomalyScoreExplanation);

            // 1st bucket
            BOOST_TEST_REQUIRE(writer.acceptResult(result));
        }
        // Finished adding results
        BOOST_TEST_REQUIRE(writer.endOutputBatch(false, 10U));
    }

    json::error_code ec;
    json::value arrayDoc_ = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    // Debug print record
    LOG_DEBUG(<< "Results:\n" << arrayDoc_);
    BOOST_TEST_REQUIRE(arrayDoc_.is_array());
    const json::array& arrayDoc = arrayDoc_.as_array();
    BOOST_REQUIRE_EQUAL(std::size_t(2), arrayDoc.size());
    BOOST_TEST_REQUIRE(arrayDoc.at(0).as_object().contains("records"));
    BOOST_TEST_REQUIRE(arrayDoc.at(0).as_object().at("records").is_array());
    BOOST_REQUIRE_EQUAL(std::size_t(1),
                        arrayDoc.at(0).as_object().at("records").as_array().size());
    const auto& record =
        arrayDoc.at(0).as_object().at("records").as_array().at(0).as_object();
    BOOST_TEST_REQUIRE(record.contains("anomaly_score_explanation"));
    BOOST_TEST_REQUIRE(record.at("anomaly_score_explanation").as_object().contains("by_field_first_occurrence"));
    BOOST_REQUIRE_EQUAL(true, record.at("anomaly_score_explanation")
                                  .as_object()
                                  .at("by_field_first_occurrence")
                                  .as_bool());
    BOOST_TEST_REQUIRE(record.at("anomaly_score_explanation").as_object().contains("by_field_relative_rarity"));
    BOOST_REQUIRE_EQUAL(5.0, record.at("anomaly_score_explanation")
                                 .as_object()
                                 .at("by_field_relative_rarity")
                                 .to_number<double>());
}
BOOST_AUTO_TEST_SUITE_END()
