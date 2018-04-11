/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CJsonOutputWriterTest.h"

#include <maths/CTools.h>

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CScopedRapidJsonPoolAllocator.h>
#include <core/CSmallVector.h>
#include <core/CTimeUtils.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CStringStore.h>
#include <model/ModelTypes.h>

#include <api/CJsonOutputWriter.h>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <boost/ref.hpp>

#include <set>
#include <sstream>
#include <string>

using TDouble1Vec = ml::core::CSmallVector<double, 1>;
using TStr1Vec = ml::core::CSmallVector<std::string, 1>;
const TStr1Vec EMPTY_STRING_LIST;

CppUnit::Test* CJsonOutputWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CJsonOutputWriterTest");
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testSimpleWrite", &CJsonOutputWriterTest::testSimpleWrite));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testWriteNonAnomalousBucket",
                                                                         &CJsonOutputWriterTest::testWriteNonAnomalousBucket));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testBucketWrite", &CJsonOutputWriterTest::testBucketWrite));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testBucketWriteInterim",
                                                                         &CJsonOutputWriterTest::testBucketWriteInterim));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testLimitedRecordsWrite",
                                                                         &CJsonOutputWriterTest::testLimitedRecordsWrite));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testLimitedRecordsWriteInterim",
                                                                         &CJsonOutputWriterTest::testLimitedRecordsWriteInterim));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testFlush", &CJsonOutputWriterTest::testFlush));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testWriteCategoryDefinition",
                                                                         &CJsonOutputWriterTest::testWriteCategoryDefinition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testWriteWithInfluences",
                                                                         &CJsonOutputWriterTest::testWriteWithInfluences));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testWriteInfluencers",
                                                                         &CJsonOutputWriterTest::testWriteInfluencers));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testWriteInfluencersWithLimit",
                                                                         &CJsonOutputWriterTest::testWriteInfluencersWithLimit));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testPersistNormalizer",
                                                                         &CJsonOutputWriterTest::testPersistNormalizer));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testPartitionScores",
                                                                         &CJsonOutputWriterTest::testPartitionScores));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testReportMemoryUsage",
                                                                         &CJsonOutputWriterTest::testReportMemoryUsage));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testWriteScheduledEvent",
                                                                         &CJsonOutputWriterTest::testWriteScheduledEvent));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testThroughputWithScopedAllocator",
                                                                         &CJsonOutputWriterTest::testThroughputWithScopedAllocator));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonOutputWriterTest>("CJsonOutputWriterTest::testThroughputWithoutScopedAllocator",
                                                                         &CJsonOutputWriterTest::testThroughputWithoutScopedAllocator));
    return suiteOfTests;
}

void CJsonOutputWriterTest::testSimpleWrite() {
    // Data isn't grouped by bucket/detector record it
    // is written straight through and everything is a string
    ml::api::CJsonOutputWriter::TStrStrUMap dataFields;

    dataFields["anomalyFactor"] = "2.24";
    dataFields["by_field_name"] = "airline";
    dataFields["by_field_value"] = "GAL";
    dataFields["typical"] = "6953";
    dataFields["actual"] = "10090";
    dataFields["probability"] = "0";
    dataFields["field_name"] = "responsetime";

    ml::api::CJsonOutputWriter::TStrStrUMap emptyFields;

    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        writer.writeRow(emptyFields, dataFields);

        dataFields["by_field_name"] = "busroute";
        dataFields["by_field_value"] = "No 32";
        writer.writeRow(emptyFields, dataFields);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), arrayDoc.Size());

    const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(object.IsObject());

    CPPUNIT_ASSERT(object.HasMember("by_field_name"));
    CPPUNIT_ASSERT_EQUAL(std::string("airline"), std::string(object["by_field_name"].GetString()));
    CPPUNIT_ASSERT(object.HasMember("by_field_value"));
    CPPUNIT_ASSERT_EQUAL(std::string("GAL"), std::string(object["by_field_value"].GetString()));
    CPPUNIT_ASSERT(object.HasMember("typical"));
    CPPUNIT_ASSERT_EQUAL(std::string("6953"), std::string(object["typical"].GetString()));
    CPPUNIT_ASSERT(object.HasMember("actual"));
    CPPUNIT_ASSERT_EQUAL(std::string("10090"), std::string(object["actual"].GetString()));
    CPPUNIT_ASSERT(object.HasMember("probability"));
    CPPUNIT_ASSERT_EQUAL(std::string("0"), std::string(object["probability"].GetString()));
    CPPUNIT_ASSERT(object.HasMember("field_name"));
    CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), std::string(object["field_name"].GetString()));

    const rapidjson::Value& object2 = arrayDoc[rapidjson::SizeType(1)];
    CPPUNIT_ASSERT(object.IsObject());

    CPPUNIT_ASSERT(object2.HasMember("by_field_name"));
    CPPUNIT_ASSERT_EQUAL(std::string("busroute"), std::string(object2["by_field_name"].GetString()));
    CPPUNIT_ASSERT(object2.HasMember("by_field_value"));
    CPPUNIT_ASSERT_EQUAL(std::string("No 32"), std::string(object2["by_field_value"].GetString()));
    CPPUNIT_ASSERT(object2.HasMember("typical"));
    CPPUNIT_ASSERT_EQUAL(std::string("6953"), std::string(object2["typical"].GetString()));
    CPPUNIT_ASSERT(object2.HasMember("actual"));
    CPPUNIT_ASSERT_EQUAL(std::string("10090"), std::string(object2["actual"].GetString()));
    CPPUNIT_ASSERT(object2.HasMember("probability"));
    CPPUNIT_ASSERT_EQUAL(std::string("0"), std::string(object2["probability"].GetString()));
    CPPUNIT_ASSERT(object2.HasMember("field_name"));
    CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), std::string(object2["field_name"].GetString()));
}

void CJsonOutputWriterTest::testWriteNonAnomalousBucket() {
    std::ostringstream sstream;

    std::string function("mean");
    std::string functionDescription("mean(responsetime)");
    std::string emptyString;
    ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::api::CHierarchicalResultsWriter::SResults result(false,
                                                             false,
                                                             emptyString,
                                                             emptyString,
                                                             emptyString,
                                                             emptyString,
                                                             emptyString,
                                                             emptyString,
                                                             emptyString,
                                                             1,
                                                             function,
                                                             functionDescription,
                                                             TDouble1Vec(1, 42.0),
                                                             TDouble1Vec(1, 42.0),
                                                             0.0,
                                                             0.0,
                                                             1.0,
                                                             30,
                                                             emptyString,
                                                             influences,
                                                             false,
                                                             false,
                                                             1,
                                                             100);

        CPPUNIT_ASSERT(writer.acceptResult(result));
        writer.acceptBucketTimeInfluencer(1, 1.0, 0.0, 0.0);
        CPPUNIT_ASSERT(writer.endOutputBatch(false, 10U));
        writer.finalise();
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter prettyPrinter(strbuf);
    arrayDoc.Accept(prettyPrinter);
    LOG_DEBUG(<< "Results:\n" << strbuf.GetString());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& bucketWrapper = arrayDoc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(bucketWrapper.IsObject());
    CPPUNIT_ASSERT(bucketWrapper.HasMember("bucket"));

    const rapidjson::Value& bucket = bucketWrapper["bucket"];
    CPPUNIT_ASSERT(bucket.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(bucket["job_id"].GetString()));
    CPPUNIT_ASSERT_EQUAL(1000, bucket["timestamp"].GetInt());
    CPPUNIT_ASSERT(bucket.HasMember("bucket_influencers") == false);
    CPPUNIT_ASSERT_EQUAL(0, bucket["event_count"].GetInt());
    CPPUNIT_ASSERT(bucket.HasMember("anomaly_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, bucket["anomaly_score"].GetDouble(), 0.00001);
}

void CJsonOutputWriterTest::testFlush() {
    std::string testId("testflush");
    ml::core_t::TTime lastFinalizedBucketEnd(123456789);
    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        writer.acknowledgeFlush(testId, lastFinalizedBucketEnd);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter writer(strbuf);
    arrayDoc.Accept(writer);
    LOG_DEBUG(<< "Flush:\n" << strbuf.GetString());

    const rapidjson::Value& flushWrapper = arrayDoc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(flushWrapper.IsObject());
    CPPUNIT_ASSERT(flushWrapper.HasMember("flush"));

    const rapidjson::Value& flush = flushWrapper["flush"];
    CPPUNIT_ASSERT(flush.IsObject());
    CPPUNIT_ASSERT(flush.HasMember("id"));
    CPPUNIT_ASSERT_EQUAL(testId, std::string(flush["id"].GetString()));
    CPPUNIT_ASSERT(flush.HasMember("last_finalized_bucket_end"));
    CPPUNIT_ASSERT_EQUAL(lastFinalizedBucketEnd * 1000, static_cast<ml::core_t::TTime>(flush["last_finalized_bucket_end"].GetInt64()));
}

void CJsonOutputWriterTest::testWriteCategoryDefinition() {
    int categoryId(42);
    std::string terms("foo bar");
    std::string regex(".*?foo.+?bar.*");
    std::size_t maxMatchingLength(132);
    using TStrSet = std::set<std::string>;
    TStrSet examples;
    examples.insert("User foo failed to log in");
    examples.insert("User bar failed to log in");

    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        writer.writeCategoryDefinition(categoryId, terms, regex, maxMatchingLength, examples);
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter writer(strbuf);
    arrayDoc.Accept(writer);
    LOG_DEBUG(<< "CategoryDefinition:\n" << strbuf.GetString());

    const rapidjson::Value& categoryWrapper = arrayDoc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(categoryWrapper.IsObject());
    CPPUNIT_ASSERT(categoryWrapper.HasMember("category_definition"));

    const rapidjson::Value& category = categoryWrapper["category_definition"];
    CPPUNIT_ASSERT(category.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(category["job_id"].GetString()));
    CPPUNIT_ASSERT(category.IsObject());
    CPPUNIT_ASSERT(category.HasMember("category_id"));
    CPPUNIT_ASSERT_EQUAL(categoryId, category["category_id"].GetInt());
    CPPUNIT_ASSERT(category.HasMember("terms"));
    CPPUNIT_ASSERT_EQUAL(terms, std::string(category["terms"].GetString()));
    CPPUNIT_ASSERT(category.HasMember("regex"));
    CPPUNIT_ASSERT_EQUAL(regex, std::string(category["regex"].GetString()));
    CPPUNIT_ASSERT(category.HasMember("max_matching_length"));
    CPPUNIT_ASSERT_EQUAL(maxMatchingLength, static_cast<std::size_t>(category["max_matching_length"].GetInt()));
    CPPUNIT_ASSERT(category.HasMember("examples"));

    TStrSet writtenExamplesSet;
    const rapidjson::Value& writtenExamples = category["examples"];
    for (rapidjson::SizeType i = 0; i < writtenExamples.Size(); i++) {
        writtenExamplesSet.insert(std::string(writtenExamples[i].GetString()));
    }
    CPPUNIT_ASSERT(writtenExamplesSet == examples);
}

void CJsonOutputWriterTest::testBucketWrite() {
    this->testBucketWriteHelper(false);
}

void CJsonOutputWriterTest::testBucketWriteInterim() {
    this->testBucketWriteHelper(true);
}

void CJsonOutputWriterTest::testLimitedRecordsWrite() {
    this->testLimitedRecordsWriteHelper(false);
}

void CJsonOutputWriterTest::testLimitedRecordsWriteInterim() {
    this->testLimitedRecordsWriteHelper(true);
}

void CJsonOutputWriterTest::testBucketWriteHelper(bool isInterim) {
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
            ml::api::CHierarchicalResultsWriter::SResults result11(false,
                                                                   false,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   overFieldName,
                                                                   overFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   1,
                                                                   function,
                                                                   functionDescription,
                                                                   TDouble1Vec(1, 10090.0),
                                                                   TDouble1Vec(1, 6953.0),
                                                                   2.24,
                                                                   0.5,
                                                                   0.0,
                                                                   79,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   1,
                                                                   100);

            ml::api::CHierarchicalResultsWriter::SResults result112(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    correlatedByFieldValue,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    2.24,
                                                                    0.5,
                                                                    0.0,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    false,
                                                                    1,
                                                                    100);

            ml::api::CHierarchicalResultsWriter::SResults result12(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   1,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.8,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   true,
                                                                   2,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            ml::api::CHierarchicalResultsWriter::SResults result13(ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   1,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.5,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   3,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            ml::api::CHierarchicalResultsWriter::SResults result14(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   1,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.0,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   4,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            // 1st bucket
            CPPUNIT_ASSERT(writer.acceptResult(result11));
            CPPUNIT_ASSERT(writer.acceptResult(result11));
            CPPUNIT_ASSERT(writer.acceptResult(result112));
            CPPUNIT_ASSERT(writer.acceptResult(result12));
            CPPUNIT_ASSERT(writer.acceptResult(result12));
            CPPUNIT_ASSERT(writer.acceptResult(result13));
            CPPUNIT_ASSERT(writer.acceptResult(result13));
            CPPUNIT_ASSERT(writer.acceptResult(result14));
            CPPUNIT_ASSERT(writer.acceptResult(result14));
            writer.acceptBucketTimeInfluencer(1, 0.01, 13.44, 70.0);
        }

        {
            ml::api::CHierarchicalResultsWriter::SResults result21(false,
                                                                   false,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   overFieldName,
                                                                   overFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   2,
                                                                   function,
                                                                   functionDescription,
                                                                   TDouble1Vec(1, 10090.0),
                                                                   TDouble1Vec(1, 6953.0),
                                                                   2.24,
                                                                   0.6,
                                                                   0.0,
                                                                   79,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   1,
                                                                   100);

            ml::api::CHierarchicalResultsWriter::SResults result212(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    correlatedByFieldValue,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    2.24,
                                                                    0.6,
                                                                    0.0,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    false,
                                                                    1,
                                                                    100);

            ml::api::CHierarchicalResultsWriter::SResults result22(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   2,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.8,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   true,
                                                                   2,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            ml::api::CHierarchicalResultsWriter::SResults result23(ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   2,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.0,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   3,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            ml::api::CHierarchicalResultsWriter::SResults result24(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   2,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.0,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   4,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            // 2nd bucket
            CPPUNIT_ASSERT(writer.acceptResult(result21));
            CPPUNIT_ASSERT(writer.acceptResult(result21));
            CPPUNIT_ASSERT(writer.acceptResult(result212));
            CPPUNIT_ASSERT(writer.acceptResult(result22));
            CPPUNIT_ASSERT(writer.acceptResult(result22));
            CPPUNIT_ASSERT(writer.acceptResult(result23));
            CPPUNIT_ASSERT(writer.acceptResult(result23));
            CPPUNIT_ASSERT(writer.acceptResult(result24));
            CPPUNIT_ASSERT(writer.acceptResult(result24));
            writer.acceptBucketTimeInfluencer(2, 0.01, 13.44, 70.0);
        }

        {
            ml::api::CHierarchicalResultsWriter::SResults result31(false,
                                                                   false,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   overFieldName,
                                                                   overFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   3,
                                                                   function,
                                                                   functionDescription,
                                                                   TDouble1Vec(1, 10090.0),
                                                                   TDouble1Vec(1, 6953.0),
                                                                   2.24,
                                                                   0.8,
                                                                   0.0,
                                                                   79,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   1,
                                                                   100);

            ml::api::CHierarchicalResultsWriter::SResults result312(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    correlatedByFieldValue,
                                                                    3,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    2.24,
                                                                    0.8,
                                                                    0.0,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    false,
                                                                    1,
                                                                    100);

            ml::api::CHierarchicalResultsWriter::SResults result32(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   3,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.0,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   true,
                                                                   2,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            ml::api::CHierarchicalResultsWriter::SResults result33(ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   3,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.0,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   3,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            ml::api::CHierarchicalResultsWriter::SResults result34(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                   partitionFieldName,
                                                                   partitionFieldValue,
                                                                   byFieldName,
                                                                   byFieldValue,
                                                                   correlatedByFieldValue,
                                                                   3,
                                                                   function,
                                                                   functionDescription,
                                                                   42.0,
                                                                   79,
                                                                   TDouble1Vec(1, 6953.0),
                                                                   TDouble1Vec(1, 10090.0),
                                                                   2.24,
                                                                   0.0,
                                                                   0.0,
                                                                   fieldName,
                                                                   influences,
                                                                   false,
                                                                   false,
                                                                   4,
                                                                   100,
                                                                   EMPTY_STRING_LIST);

            // 3rd bucket
            CPPUNIT_ASSERT(writer.acceptResult(result31));
            CPPUNIT_ASSERT(writer.acceptResult(result31));
            CPPUNIT_ASSERT(writer.acceptResult(result312));
            CPPUNIT_ASSERT(writer.acceptResult(result32));
            CPPUNIT_ASSERT(writer.acceptResult(result32));
            CPPUNIT_ASSERT(writer.acceptResult(result33));
            CPPUNIT_ASSERT(writer.acceptResult(result33));
            CPPUNIT_ASSERT(writer.acceptResult(result34));
            CPPUNIT_ASSERT(writer.acceptResult(result34));
            writer.acceptBucketTimeInfluencer(3, 0.01, 13.44, 70.0);
        }

        // Finished adding results
        CPPUNIT_ASSERT(writer.endOutputBatch(isInterim, 10U));
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str());
    CPPUNIT_ASSERT(!arrayDoc.HasParseError());

    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter writer(strbuf);
    arrayDoc.Accept(writer);
    LOG_DEBUG(<< "Results:\n" << strbuf.GetString());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    // There are 3 buckets and 3 record arrays in the order: r1, b1, r2, b2, r3, b3
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(6), arrayDoc.Size());

    int bucketTimes[] = {1000, 1000, 2000, 2000, 3000, 3000};

    // Assert buckets
    for (rapidjson::SizeType i = 1; i < arrayDoc.Size(); i = i + 2) {
        int buckettime = bucketTimes[i];
        const rapidjson::Value& bucketWrapper = arrayDoc[i];
        CPPUNIT_ASSERT(bucketWrapper.HasMember("bucket"));

        const rapidjson::Value& bucket = bucketWrapper["bucket"];
        CPPUNIT_ASSERT(bucket.IsObject());
        CPPUNIT_ASSERT(bucket.HasMember("job_id"));
        CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(bucket["job_id"].GetString()));

        // 3 detectors each have 2 records (simple count detector isn't added)
        // except the population detector which has a single record and clauses
        CPPUNIT_ASSERT_EQUAL(buckettime, bucket["timestamp"].GetInt());
        CPPUNIT_ASSERT(bucket.HasMember("bucket_influencers"));
        const rapidjson::Value& bucketInfluencers = bucket["bucket_influencers"];
        CPPUNIT_ASSERT(bucketInfluencers.IsArray());
        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), bucketInfluencers.Size());
        const rapidjson::Value& bucketInfluencer = bucketInfluencers[rapidjson::SizeType(0)];
        CPPUNIT_ASSERT_DOUBLES_EQUAL(13.44, bucketInfluencer["raw_anomaly_score"].GetDouble(), 0.00001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.01, bucketInfluencer["probability"].GetDouble(), 0.00001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(70.0, bucketInfluencer["initial_anomaly_score"].GetDouble(), 0.00001);
        CPPUNIT_ASSERT(bucketInfluencer.HasMember("anomaly_score"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(70.0, bucketInfluencer["anomaly_score"].GetDouble(), 0.00001);
        CPPUNIT_ASSERT_EQUAL(std::string("bucket_time"), std::string(bucketInfluencer["influencer_field_name"].GetString()));

        CPPUNIT_ASSERT_EQUAL(79, bucket["event_count"].GetInt());
        CPPUNIT_ASSERT(bucket.HasMember("anomaly_score"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(70.0, bucket["anomaly_score"].GetDouble(), 0.00001);
        CPPUNIT_ASSERT(bucket.HasMember("initial_anomaly_score"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(70.0, bucket["initial_anomaly_score"].GetDouble(), 0.00001);
        if (isInterim) {
            CPPUNIT_ASSERT(bucket.HasMember("is_interim"));
            CPPUNIT_ASSERT_EQUAL(isInterim, bucket["is_interim"].GetBool());
        } else {
            CPPUNIT_ASSERT(!bucket.HasMember("is_interim"));
        }

        CPPUNIT_ASSERT_EQUAL(uint64_t(10ll), bucket["processing_time_ms"].GetUint64());
    }

    for (rapidjson::SizeType i = 0; i < arrayDoc.Size(); i = i + 2) {
        int buckettime = bucketTimes[i];

        const rapidjson::Value& recordsWrapper = arrayDoc[i];
        CPPUNIT_ASSERT(recordsWrapper.HasMember("records"));

        const rapidjson::Value& records = recordsWrapper["records"];
        CPPUNIT_ASSERT(records.IsArray());
        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(5), records.Size());

        // 1st record is for population detector
        {
            const rapidjson::Value& record = records[rapidjson::SizeType(0)];
            CPPUNIT_ASSERT(record.HasMember("job_id"));
            CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(record["job_id"].GetString()));
            CPPUNIT_ASSERT(record.HasMember("detector_index"));
            CPPUNIT_ASSERT_EQUAL(1, record["detector_index"].GetInt());
            CPPUNIT_ASSERT(record.HasMember("timestamp"));
            CPPUNIT_ASSERT_EQUAL(buckettime, record["timestamp"].GetInt());
            CPPUNIT_ASSERT(record.HasMember("probability"));
            CPPUNIT_ASSERT_EQUAL(0.0, record["probability"].GetDouble());
            CPPUNIT_ASSERT(record.HasMember("by_field_name"));
            CPPUNIT_ASSERT_EQUAL(std::string("airline"), std::string(record["by_field_name"].GetString()));
            CPPUNIT_ASSERT(!record.HasMember("by_field_value"));
            CPPUNIT_ASSERT(!record.HasMember("correlated_by_field_value"));
            CPPUNIT_ASSERT(record.HasMember("function"));
            CPPUNIT_ASSERT_EQUAL(std::string("mean"), std::string(record["function"].GetString()));
            CPPUNIT_ASSERT(record.HasMember("function_description"));
            CPPUNIT_ASSERT_EQUAL(std::string("mean(responsetime)"), std::string(record["function_description"].GetString()));
            CPPUNIT_ASSERT(record.HasMember("over_field_name"));
            CPPUNIT_ASSERT_EQUAL(std::string("pfn"), std::string(record["over_field_name"].GetString()));
            CPPUNIT_ASSERT(record.HasMember("over_field_value"));
            CPPUNIT_ASSERT_EQUAL(std::string("pfv"), std::string(record["over_field_value"].GetString()));
            CPPUNIT_ASSERT(record.HasMember("bucket_span"));
            CPPUNIT_ASSERT_EQUAL(100, record["bucket_span"].GetInt());
            // It's hard to predict what these will be, so just assert their
            // presence
            CPPUNIT_ASSERT(record.HasMember("initial_record_score"));
            CPPUNIT_ASSERT(record.HasMember("record_score"));
            if (isInterim) {
                CPPUNIT_ASSERT(record.HasMember("is_interim"));
                CPPUNIT_ASSERT_EQUAL(isInterim, record["is_interim"].GetBool());
            } else {
                CPPUNIT_ASSERT(!record.HasMember("is_interim"));
            }

            CPPUNIT_ASSERT(record.HasMember("causes"));
            const rapidjson::Value& causes = record["causes"];
            CPPUNIT_ASSERT(causes.IsArray());
            CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), causes.Size());
            for (rapidjson::SizeType k = 0; k < causes.Size(); k++) {
                const rapidjson::Value& cause = causes[k];
                CPPUNIT_ASSERT(cause.HasMember("probability"));
                CPPUNIT_ASSERT_EQUAL(0.0, cause["probability"].GetDouble());
                CPPUNIT_ASSERT(cause.HasMember("field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), std::string(cause["field_name"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("by_field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("airline"), std::string(cause["by_field_name"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("by_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string("GAL"), std::string(cause["by_field_value"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("correlated_by_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string("BAW"), std::string(cause["correlated_by_field_value"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("partition_field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("tfn"), std::string(cause["partition_field_name"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("partition_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(cause["partition_field_value"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("function"));
                CPPUNIT_ASSERT_EQUAL(std::string("mean"), std::string(cause["function"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("function_description"));
                CPPUNIT_ASSERT_EQUAL(std::string("mean(responsetime)"), std::string(cause["function_description"].GetString()));
                CPPUNIT_ASSERT(cause.HasMember("typical"));
                CPPUNIT_ASSERT(cause["typical"].IsArray());
                CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), cause["typical"].Size());
                CPPUNIT_ASSERT_EQUAL(6953.0, cause["typical"][rapidjson::SizeType(0)].GetDouble());
                CPPUNIT_ASSERT(cause.HasMember("actual"));
                CPPUNIT_ASSERT(cause["actual"].IsArray());
                CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), cause["actual"].Size());
                CPPUNIT_ASSERT_EQUAL(10090.0, cause["actual"][rapidjson::SizeType(0)].GetDouble());
                CPPUNIT_ASSERT(cause.HasMember("function"));
            }
        }

        // Next 2 records are for metric detector
        {
            for (rapidjson::SizeType k = 1; k < 3; k++) {
                const rapidjson::Value& record = records[k];
                CPPUNIT_ASSERT(record.HasMember("job_id"));
                CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(record["job_id"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("detector_index"));
                CPPUNIT_ASSERT_EQUAL(2, record["detector_index"].GetInt());
                CPPUNIT_ASSERT(record.HasMember("timestamp"));
                CPPUNIT_ASSERT_EQUAL(buckettime, record["timestamp"].GetInt());
                CPPUNIT_ASSERT(record.HasMember("probability"));
                CPPUNIT_ASSERT_EQUAL(0.0, record["probability"].GetDouble());
                CPPUNIT_ASSERT(record.HasMember("by_field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("airline"), std::string(record["by_field_name"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("by_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string("GAL"), std::string(record["by_field_value"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("correlated_by_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string("BAW"), std::string(record["correlated_by_field_value"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("typical"));
                CPPUNIT_ASSERT(record["typical"].IsArray());
                CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), record["typical"].Size());
                CPPUNIT_ASSERT_EQUAL(6953.0, record["typical"][rapidjson::SizeType(0)].GetDouble());
                CPPUNIT_ASSERT(record.HasMember("actual"));
                CPPUNIT_ASSERT(record["actual"].IsArray());
                CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), record["actual"].Size());
                CPPUNIT_ASSERT_EQUAL(10090.0, record["actual"][rapidjson::SizeType(0)].GetDouble());
                CPPUNIT_ASSERT(record.HasMember("field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), std::string(record["field_name"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("function"));
                CPPUNIT_ASSERT_EQUAL(std::string("mean"), std::string(record["function"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("function_description"));
                CPPUNIT_ASSERT_EQUAL(std::string("mean(responsetime)"), std::string(record["function_description"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("partition_field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("tfn"), std::string(record["partition_field_name"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("partition_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(record["partition_field_value"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("bucket_span"));
                CPPUNIT_ASSERT_EQUAL(100, record["bucket_span"].GetInt());
                // It's hard to predict what these will be, so just assert their
                // presence
                CPPUNIT_ASSERT(record.HasMember("initial_record_score"));
                CPPUNIT_ASSERT(record.HasMember("record_score"));
                if (isInterim) {
                    CPPUNIT_ASSERT(record.HasMember("is_interim"));
                    CPPUNIT_ASSERT_EQUAL(isInterim, record["is_interim"].GetBool());
                } else {
                    CPPUNIT_ASSERT(!record.HasMember("is_interim"));
                }
            }
        }

        // Last 2 records are for event rate detector
        {
            for (rapidjson::SizeType k = 3; k < 5; k++) {
                const rapidjson::Value& record = records[k];
                CPPUNIT_ASSERT(record.HasMember("job_id"));
                CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(record["job_id"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("detector_index"));
                CPPUNIT_ASSERT_EQUAL(4, record["detector_index"].GetInt());
                CPPUNIT_ASSERT(record.HasMember("timestamp"));
                CPPUNIT_ASSERT_EQUAL(buckettime, record["timestamp"].GetInt());
                CPPUNIT_ASSERT(record.HasMember("probability"));
                CPPUNIT_ASSERT_EQUAL(0.0, record["probability"].GetDouble());
                CPPUNIT_ASSERT(record.HasMember("by_field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("airline"), std::string(record["by_field_name"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("by_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string("GAL"), std::string(record["by_field_value"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("correlated_by_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string("BAW"), std::string(record["correlated_by_field_value"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("typical"));
                CPPUNIT_ASSERT(record["typical"].IsArray());
                CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), record["typical"].Size());
                CPPUNIT_ASSERT_EQUAL(6953.0, record["typical"][rapidjson::SizeType(0)].GetDouble());
                CPPUNIT_ASSERT(record.HasMember("actual"));
                CPPUNIT_ASSERT(record["actual"].IsArray());
                CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(1), record["actual"].Size());
                CPPUNIT_ASSERT_EQUAL(10090.0, record["actual"][rapidjson::SizeType(0)].GetDouble());
                CPPUNIT_ASSERT(record.HasMember("function"));
                // This would be count in the real case with properly generated input data
                CPPUNIT_ASSERT_EQUAL(std::string("mean"), std::string(record["function"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("function_description"));
                CPPUNIT_ASSERT_EQUAL(std::string("mean(responsetime)"), std::string(record["function_description"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("partition_field_name"));
                CPPUNIT_ASSERT_EQUAL(std::string("tfn"), std::string(record["partition_field_name"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("partition_field_value"));
                CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(record["partition_field_value"].GetString()));
                CPPUNIT_ASSERT(record.HasMember("bucket_span"));
                CPPUNIT_ASSERT_EQUAL(100, record["bucket_span"].GetInt());
                // It's hard to predict what these will be, so just assert their
                // presence
                CPPUNIT_ASSERT(record.HasMember("initial_record_score"));
                CPPUNIT_ASSERT(record.HasMember("record_score"));
                if (isInterim) {
                    CPPUNIT_ASSERT(record.HasMember("is_interim"));
                    CPPUNIT_ASSERT_EQUAL(isInterim, record["is_interim"].GetBool());
                } else {
                    CPPUNIT_ASSERT(!record.HasMember("is_interim"));
                }
            }
        }
    }
}

void CJsonOutputWriterTest::testLimitedRecordsWriteHelper(bool isInterim) {
    // Tests CJsonOutputWriter::limitNumberRecords(size_t)
    // set the record limit for each detector to 2

    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
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
            ml::api::CHierarchicalResultsWriter::SResults result111(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    0.0,
                                                                    0.1,
                                                                    0.1,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result111));

            ml::api::CHierarchicalResultsWriter::SResults result112(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    0.0,
                                                                    0.1,
                                                                    0.2,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result112));

            ml::api::CHierarchicalResultsWriter::SResults result113(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    2.0,
                                                                    0.0,
                                                                    0.4,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result113));

            ml::api::CHierarchicalResultsWriter::SResults result114(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    12.0,
                                                                    0.0,
                                                                    0.4,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result114));
            CPPUNIT_ASSERT(writer.acceptResult(result114));

            overFieldName = "ofn";
            overFieldValue = "ofv";

            ml::api::CHierarchicalResultsWriter::SResults result121(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    emptyString,
                                                                    emptyString,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    12.0,
                                                                    0.0,
                                                                    0.01,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result121));

            ml::api::CHierarchicalResultsWriter::SResults result122(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    12.0,
                                                                    0.0,
                                                                    0.01,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result122));

            ml::api::CHierarchicalResultsWriter::SResults result123(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    0.5,
                                                                    0.0,
                                                                    0.5,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result123));

            ml::api::CHierarchicalResultsWriter::SResults result124(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    emptyString,
                                                                    emptyString,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    0.5,
                                                                    0.0,
                                                                    0.5,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result124));

            ml::api::CHierarchicalResultsWriter::SResults result125(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    6.0,
                                                                    0.0,
                                                                    0.5,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result125));

            ml::api::CHierarchicalResultsWriter::SResults result126(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    emptyString,
                                                                    emptyString,
                                                                    emptyString,
                                                                    1,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    6.0,
                                                                    0.0,
                                                                    0.05,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result126));
        }

        {
            // 2nd bucket
            overFieldName.clear();
            overFieldValue.clear();

            ml::api::CHierarchicalResultsWriter::SResults result211(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    1.0,
                                                                    0.0,
                                                                    0.05,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result211));

            ml::api::CHierarchicalResultsWriter::SResults result212(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    7.0,
                                                                    0.0,
                                                                    0.001,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result212));

            ml::api::CHierarchicalResultsWriter::SResults result213(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    0.6,
                                                                    0.0,
                                                                    0.1,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result213));
            CPPUNIT_ASSERT(writer.acceptResult(result213));

            overFieldName = "ofn";
            overFieldValue = "ofv";

            ml::api::CHierarchicalResultsWriter::SResults result221(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    0.6,
                                                                    0.0,
                                                                    0.1,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result221));
            CPPUNIT_ASSERT(writer.acceptResult(result221));

            ml::api::CHierarchicalResultsWriter::SResults result222(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    emptyString,
                                                                    emptyString,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    0.6,
                                                                    0.0,
                                                                    0.1,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result222));

            ml::api::CHierarchicalResultsWriter::SResults result223(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    3.0,
                                                                    0.0,
                                                                    0.02,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result223));

            ml::api::CHierarchicalResultsWriter::SResults result224(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    emptyString,
                                                                    emptyString,
                                                                    emptyString,
                                                                    2,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    20.0,
                                                                    0.0,
                                                                    0.02,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result224));
        }

        {
            // 3rd bucket
            overFieldName.clear();
            overFieldValue.clear();

            ml::api::CHierarchicalResultsWriter::SResults result311(ml::api::CHierarchicalResultsWriter::E_Result,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    3,
                                                                    function,
                                                                    functionDescription,
                                                                    42.0,
                                                                    79,
                                                                    TDouble1Vec(1, 6953.0),
                                                                    TDouble1Vec(1, 10090.0),
                                                                    30.0,
                                                                    0.0,
                                                                    0.02,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    1,
                                                                    100,
                                                                    EMPTY_STRING_LIST);
            CPPUNIT_ASSERT(writer.acceptResult(result311));

            overFieldName = "ofn";
            overFieldValue = "ofv";

            ml::api::CHierarchicalResultsWriter::SResults result321(false,
                                                                    false,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    byFieldName,
                                                                    byFieldValue,
                                                                    emptyString,
                                                                    3,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    31.0,
                                                                    0.0,
                                                                    0.0002,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result321));

            ml::api::CHierarchicalResultsWriter::SResults result322(false,
                                                                    true,
                                                                    partitionFieldName,
                                                                    partitionFieldValue,
                                                                    overFieldName,
                                                                    overFieldValue,
                                                                    emptyString,
                                                                    emptyString,
                                                                    emptyString,
                                                                    3,
                                                                    function,
                                                                    functionDescription,
                                                                    TDouble1Vec(1, 10090.0),
                                                                    TDouble1Vec(1, 6953.0),
                                                                    31.0,
                                                                    0.0,
                                                                    0.0002,
                                                                    79,
                                                                    fieldName,
                                                                    influences,
                                                                    false,
                                                                    true,
                                                                    2,
                                                                    100);
            CPPUNIT_ASSERT(writer.acceptResult(result322));
        }

        // Finished adding results
        CPPUNIT_ASSERT(writer.endOutputBatch(isInterim, 10U));
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter writer(strbuf);
    arrayDoc.Accept(writer);
    LOG_DEBUG(<< "Results:\n" << strbuf.GetString());

    CPPUNIT_ASSERT(arrayDoc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(6), arrayDoc.Size());

    // buckets and records are the top level objects
    // records corresponding to a bucket appear first. The bucket follows.
    // each bucket has max 2 records from either both or
    // one or the other of the 2 detectors used.
    // records are sorted by probability.
    // bucket total anomaly score is the sum of all anomalies not just those printed.
    {
        const rapidjson::Value& bucketWrapper = arrayDoc[rapidjson::SizeType(1)];
        CPPUNIT_ASSERT(bucketWrapper.IsObject());
        CPPUNIT_ASSERT(bucketWrapper.HasMember("bucket"));

        const rapidjson::Value& bucket = bucketWrapper["bucket"];
        CPPUNIT_ASSERT(bucket.IsObject());
        // It's hard to predict what these will be, so just assert their presence
        CPPUNIT_ASSERT(bucket.HasMember("anomaly_score"));
        if (isInterim) {
            CPPUNIT_ASSERT(bucket.HasMember("is_interim"));
            CPPUNIT_ASSERT_EQUAL(isInterim, bucket["is_interim"].GetBool());
        } else {
            CPPUNIT_ASSERT(!bucket.HasMember("is_interim"));
        }

        const rapidjson::Value& recordsWrapper = arrayDoc[rapidjson::SizeType(0)];
        CPPUNIT_ASSERT(recordsWrapper.IsObject());
        CPPUNIT_ASSERT(recordsWrapper.HasMember("records"));
        const rapidjson::Value& records = recordsWrapper["records"];
        CPPUNIT_ASSERT(records.IsArray());

        double EXPECTED_PROBABILITIES[] = {0.01, 0.05, 0.001, 0.02, 0.0002};

        int probIndex = 0;
        for (rapidjson::SizeType i = 0; i < records.Size(); i++) {
            CPPUNIT_ASSERT(records[i].HasMember("detector_index"));
            CPPUNIT_ASSERT(records[i].HasMember("initial_record_score"));
            CPPUNIT_ASSERT(records[i].HasMember("record_score"));
            CPPUNIT_ASSERT(records[i].HasMember("probability"));
            CPPUNIT_ASSERT_EQUAL(EXPECTED_PROBABILITIES[probIndex], records[i]["probability"].GetDouble());
            ++probIndex;

            if (isInterim) {
                CPPUNIT_ASSERT(records[i].HasMember("is_interim"));
                CPPUNIT_ASSERT_EQUAL(isInterim, records[i]["is_interim"].GetBool());
            } else {
                CPPUNIT_ASSERT(!records[i].HasMember("is_interim"));
            }
        }

        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), records.Size());
    }
    {
        const rapidjson::Value& bucketWrapper = arrayDoc[rapidjson::SizeType(3)];
        CPPUNIT_ASSERT(bucketWrapper.IsObject());
        CPPUNIT_ASSERT(bucketWrapper.HasMember("bucket"));

        const rapidjson::Value& bucket = bucketWrapper["bucket"];
        CPPUNIT_ASSERT(bucket.IsObject());
        // It's hard to predict what these will be, so just assert their presence
        CPPUNIT_ASSERT(bucket.HasMember("anomaly_score"));
        if (isInterim) {
            CPPUNIT_ASSERT(bucket.HasMember("is_interim"));
            CPPUNIT_ASSERT_EQUAL(isInterim, bucket["is_interim"].GetBool());
        } else {
            CPPUNIT_ASSERT(!bucket.HasMember("is_interim"));
        }

        const rapidjson::Value& recordsWrapper = arrayDoc[rapidjson::SizeType(2)];
        CPPUNIT_ASSERT(recordsWrapper.IsObject());
        CPPUNIT_ASSERT(recordsWrapper.HasMember("records"));
        const rapidjson::Value& records = recordsWrapper["records"];
        CPPUNIT_ASSERT(records.IsArray());

        for (rapidjson::SizeType i = 0; i < records.Size(); i++) {
            //CPPUNIT_ASSERT_EQUAL(0.1, records1[rapidjson::SizeType(0)]["probability"].GetDouble());
            CPPUNIT_ASSERT(records[i].HasMember("detector_index"));
            CPPUNIT_ASSERT(records[i].HasMember("initial_record_score"));
            CPPUNIT_ASSERT(records[i].HasMember("record_score"));
            if (isInterim) {
                CPPUNIT_ASSERT(records[i].HasMember("is_interim"));
                CPPUNIT_ASSERT_EQUAL(isInterim, records[i]["is_interim"].GetBool());
            } else {
                CPPUNIT_ASSERT(!records[i].HasMember("is_interim"));
            }
        }

        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), records.Size());
    }
    {
        const rapidjson::Value& bucketWrapper = arrayDoc[rapidjson::SizeType(5)];
        CPPUNIT_ASSERT(bucketWrapper.IsObject());
        CPPUNIT_ASSERT(bucketWrapper.HasMember("bucket"));

        const rapidjson::Value& bucket = bucketWrapper["bucket"];
        CPPUNIT_ASSERT(bucket.IsObject());
        // It's hard to predict what these will be, so just assert their presence
        CPPUNIT_ASSERT(bucket.HasMember("anomaly_score"));
        if (isInterim) {
            CPPUNIT_ASSERT(bucket.HasMember("is_interim"));
            CPPUNIT_ASSERT_EQUAL(isInterim, bucket["is_interim"].GetBool());
        } else {
            CPPUNIT_ASSERT(!bucket.HasMember("is_interim"));
        }

        const rapidjson::Value& recordsWrapper = arrayDoc[rapidjson::SizeType(4)];
        CPPUNIT_ASSERT(recordsWrapper.IsObject());
        CPPUNIT_ASSERT(recordsWrapper.HasMember("records"));
        const rapidjson::Value& records = recordsWrapper["records"];
        CPPUNIT_ASSERT(records.IsArray());

        for (rapidjson::SizeType i = 0; i < records.Size(); i++) {
            CPPUNIT_ASSERT(records[i].HasMember("detector_index"));
            //CPPUNIT_ASSERT_EQUAL(0.1, records1[rapidjson::SizeType(0)]["probability"].GetDouble());
            CPPUNIT_ASSERT(records[i].HasMember("initial_record_score"));
            CPPUNIT_ASSERT(records[i].HasMember("record_score"));
            if (isInterim) {
                CPPUNIT_ASSERT(records[i].HasMember("is_interim"));
                CPPUNIT_ASSERT_EQUAL(isInterim, records[i]["is_interim"].GetBool());
            } else {
                CPPUNIT_ASSERT(!records[i].HasMember("is_interim"));
            }
        }

        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), records.Size());
    }
}

ml::model::CHierarchicalResults::TNode
createInfluencerNode(const std::string& personName, const std::string& personValue, double probability, double normalisedAnomalyScore) {
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
createBucketInfluencerNode(const std::string& personName, double probability, double normalisedAnomalyScore, double rawAnomalyScore) {
    ml::model::CHierarchicalResults::TResultSpec spec;
    spec.s_PersonFieldName = ml::model::CStringStore::names().get(personName);

    ml::model::CHierarchicalResults::TNode node;
    node.s_AnnotatedProbability.s_Probability = probability;
    node.s_NormalizedAnomalyScore = normalisedAnomalyScore;
    node.s_RawAnomalyScore = rawAnomalyScore;
    node.s_Spec = spec;

    return node;
}

void CJsonOutputWriterTest::testWriteInfluencers() {
    std::ostringstream sstream;

    {
        std::string user("user");
        std::string daisy("daisy");
        std::string jim("jim");

        ml::model::CHierarchicalResults::TNode node1 = createInfluencerNode(user, daisy, 0.5, 10.0);
        ml::model::CHierarchicalResults::TNode node2 = createInfluencerNode(user, jim, 0.9, 100.0);

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(42), node1, false));
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(42), node2, false));

        // Finished adding results
        CPPUNIT_ASSERT(writer.endOutputBatch(true, 1U));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    // Debug print record
    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter writer(strbuf);
    doc.Accept(writer);
    LOG_DEBUG(<< "influencers:\n" << strbuf.GetString());

    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), doc.Size());

    const rapidjson::Value& influencers = doc[rapidjson::SizeType(0)]["influencers"];
    CPPUNIT_ASSERT(influencers.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), influencers.Size());

    const rapidjson::Value& influencer = influencers[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(influencer.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(influencer["job_id"].GetString()));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, influencer["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, influencer["initial_influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(influencer.HasMember("influencer_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, influencer["influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("user"), std::string(influencer["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("daisy"), std::string(influencer["influencer_field_value"].GetString()));
    CPPUNIT_ASSERT_EQUAL(42000, influencer["timestamp"].GetInt());
    CPPUNIT_ASSERT(influencer["is_interim"].GetBool());
    CPPUNIT_ASSERT(influencer.HasMember("bucket_span"));

    const rapidjson::Value& influencer2 = influencers[rapidjson::SizeType(1)];
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9, influencer2["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, influencer2["initial_influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(influencer2.HasMember("influencer_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, influencer2["influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("user"), std::string(influencer2["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("jim"), std::string(influencer2["influencer_field_value"].GetString()));
    CPPUNIT_ASSERT_EQUAL(42000, influencer2["timestamp"].GetInt());
    CPPUNIT_ASSERT(influencer2["is_interim"].GetBool());
    CPPUNIT_ASSERT(influencer2.HasMember("bucket_span"));

    const rapidjson::Value& bucket = doc[rapidjson::SizeType(1)]["bucket"];
    CPPUNIT_ASSERT(bucket.HasMember("influencers") == false);
}

void CJsonOutputWriterTest::testWriteInfluencersWithLimit() {
    std::ostringstream sstream;

    {
        std::string user("user");
        std::string computer("computer");
        std::string monitor("monitor");
        std::string daisy("daisy");
        std::string jim("jim");
        std::string bob("bob");
        std::string laptop("laptop");

        ml::model::CHierarchicalResults::TNode node1 = createInfluencerNode(user, daisy, 0.5, 10.0);
        ml::model::CHierarchicalResults::TNode node2 = createInfluencerNode(user, jim, 0.9, 100.0);
        ml::model::CHierarchicalResults::TNode node3 = createInfluencerNode(user, bob, 0.3, 9.0);
        ml::model::CHierarchicalResults::TNode node4 = createInfluencerNode(computer, laptop, 0.3, 12.0);

        ml::model::CHierarchicalResults::TNode bnode1 = createBucketInfluencerNode(user, 0.5, 10.0, 1.0);
        ml::model::CHierarchicalResults::TNode bnode2 = createBucketInfluencerNode(computer, 0.9, 100.0, 10.0);
        ml::model::CHierarchicalResults::TNode bnode3 = createBucketInfluencerNode(monitor, 0.3, 9.0, 0.9);

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        writer.limitNumberRecords(2);

        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), node1, false));
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), node2, false));
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), node3, false));
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), node4, false));

        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), bnode1, true));
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), bnode2, true));
        CPPUNIT_ASSERT(writer.acceptInfluencer(ml::core_t::TTime(0), bnode3, true));

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
        ml::api::CHierarchicalResultsWriter::SResults result(ml::api::CHierarchicalResultsWriter::E_Result,
                                                             pfn,
                                                             pfv,
                                                             bfn,
                                                             bfv,
                                                             emptyStr,
                                                             0,
                                                             fun,
                                                             fund,
                                                             42.0,
                                                             79,
                                                             TDouble1Vec(1, 6953.0),
                                                             TDouble1Vec(1, 10090.0),
                                                             0.0,
                                                             0.1,
                                                             0.1,
                                                             fn,
                                                             influences,
                                                             false,
                                                             true,
                                                             1,
                                                             100,
                                                             EMPTY_STRING_LIST);

        CPPUNIT_ASSERT(writer.acceptResult(result));

        writer.acceptBucketTimeInfluencer(ml::core_t::TTime(0), 0.6, 1.0, 10.0);

        // Finished adding results
        CPPUNIT_ASSERT(writer.endOutputBatch(false, 1U));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    rapidjson::StringBuffer strbuf;
    using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
    TStringBufferPrettyWriter writer(strbuf);
    doc.Accept(writer);

    LOG_DEBUG(<< "limited write influencers:\n" << strbuf.GetString());

    const rapidjson::Value& influencers = doc[rapidjson::SizeType(1)]["influencers"];
    CPPUNIT_ASSERT(influencers.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), influencers.Size());

    const rapidjson::Value& influencer = influencers[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9, influencer["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, influencer["initial_influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(influencer.HasMember("influencer_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, influencer["influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("user"), std::string(influencer["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("jim"), std::string(influencer["influencer_field_value"].GetString()));
    CPPUNIT_ASSERT(influencer.HasMember("bucket_span"));

    const rapidjson::Value& influencer2 = influencers[rapidjson::SizeType(1)];
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.3, influencer2["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, influencer2["initial_influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(influencer2.HasMember("influencer_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, influencer2["influencer_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("computer"), std::string(influencer2["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("laptop"), std::string(influencer2["influencer_field_value"].GetString()));
    CPPUNIT_ASSERT(influencer2.HasMember("bucket_span"));

    // bucket influencers
    const rapidjson::Value& bucketResult = doc[rapidjson::SizeType(2)]["bucket"];
    CPPUNIT_ASSERT(bucketResult.HasMember("bucket_influencers"));
    const rapidjson::Value& bucketInfluencers = bucketResult["bucket_influencers"];
    CPPUNIT_ASSERT(bucketInfluencers.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(3), bucketInfluencers.Size());

    const rapidjson::Value& binf = bucketInfluencers[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9, binf["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, binf["initial_anomaly_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(binf.HasMember("anomaly_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, binf["anomaly_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("computer"), std::string(binf["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, binf["raw_anomaly_score"].GetDouble(), 0.001);

    const rapidjson::Value& binf2 = bucketInfluencers[rapidjson::SizeType(1)];
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, binf2["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, binf2["initial_anomaly_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(binf2.HasMember("anomaly_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, binf2["anomaly_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("user"), std::string(binf2["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, binf2["raw_anomaly_score"].GetDouble(), 0.001);

    const rapidjson::Value& binf3 = bucketInfluencers[rapidjson::SizeType(2)];
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6, binf3["probability"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, binf3["initial_anomaly_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT(binf3.HasMember("anomaly_score"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, binf3["anomaly_score"].GetDouble(), 0.001);
    CPPUNIT_ASSERT_EQUAL(std::string("bucket_time"), std::string(binf3["influencer_field_name"].GetString()));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, binf3["raw_anomaly_score"].GetDouble(), 0.001);
}

void CJsonOutputWriterTest::testWriteWithInfluences() {
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
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(ml::model::CStringStore::names().get(user),
                                                                                   ml::model::CStringStore::names().get(dave));
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr field2 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(ml::model::CStringStore::names().get(user),
                                                                                   ml::model::CStringStore::names().get(cat));
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr field3 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(ml::model::CStringStore::names().get(user),
                                                                                   ml::model::CStringStore::names().get(jo));

        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr hostField1 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(ml::model::CStringStore::names().get(host),
                                                                                   ml::model::CStringStore::names().get(localhost));
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr hostField2 =
            ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPr(ml::model::CStringStore::names().get(host),
                                                                                   ml::model::CStringStore::names().get(webserver));

        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(field1, 0.4));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(field2, 1.0));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(hostField1, 0.7));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(field3, 0.1));
        influences.push_back(ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePr(hostField2, 0.8));

        // The output writer won't close the JSON structures until is is destroyed

        ml::api::CHierarchicalResultsWriter::SResults result(ml::api::CHierarchicalResultsWriter::E_Result,
                                                             partitionFieldName,
                                                             partitionFieldValue,
                                                             byFieldName,
                                                             byFieldValue,
                                                             emptyString,
                                                             1,
                                                             function,
                                                             functionDescription,
                                                             42.0,
                                                             79,
                                                             TDouble1Vec(1, 6953.0),
                                                             TDouble1Vec(1, 10090.0),
                                                             0.0,
                                                             0.1,
                                                             0.1,
                                                             fieldName,
                                                             influences,
                                                             false,
                                                             true,
                                                             1,
                                                             100,
                                                             EMPTY_STRING_LIST);

        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);
        CPPUNIT_ASSERT(writer.acceptResult(result));

        // Finished adding results
        CPPUNIT_ASSERT(writer.endOutputBatch(false, 1U));
    }

    rapidjson::Document doc;
    std::string out = sstream.str();
    doc.Parse<rapidjson::kParseDefaultFlags>(out);

    // Debug print record
    {
        rapidjson::StringBuffer strbuf;
        using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
        TStringBufferPrettyWriter writer(strbuf);
        doc.Accept(writer);
        LOG_DEBUG(<< "Results:\n" << strbuf.GetString());
    }

    CPPUNIT_ASSERT(doc[rapidjson::SizeType(1)].HasMember("bucket"));
    const rapidjson::Value& bucket = doc[rapidjson::SizeType(1)]["bucket"];
    CPPUNIT_ASSERT(bucket.HasMember("records") == false);

    CPPUNIT_ASSERT(doc[rapidjson::SizeType(0)].HasMember("records"));
    const rapidjson::Value& records = doc[rapidjson::SizeType(0)]["records"];

    CPPUNIT_ASSERT(records[rapidjson::SizeType(0)].HasMember("influencers"));
    const rapidjson::Value& influences = records[rapidjson::SizeType(0)]["influencers"];

    CPPUNIT_ASSERT(influences.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), influences.Size());

    {
        const rapidjson::Value& influence = influences[rapidjson::SizeType(0)];
        CPPUNIT_ASSERT(influence.HasMember("influencer_field_name"));
        CPPUNIT_ASSERT_EQUAL(std::string("host"), std::string(influence["influencer_field_name"].GetString()));
        CPPUNIT_ASSERT(influence.HasMember("influencer_field_values"));
        const rapidjson::Value& influencerFieldValues = influence["influencer_field_values"];
        CPPUNIT_ASSERT(influencerFieldValues.IsArray());
        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), influencerFieldValues.Size());

        // Check influencers are ordered
        CPPUNIT_ASSERT_EQUAL(std::string("web-server"), std::string(influencerFieldValues[rapidjson::SizeType(0)].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("localhost"), std::string(influencerFieldValues[rapidjson::SizeType(1)].GetString()));
    }
    {
        const rapidjson::Value& influence = influences[rapidjson::SizeType(1)];
        CPPUNIT_ASSERT(influence.HasMember("influencer_field_name"));
        CPPUNIT_ASSERT_EQUAL(std::string("user"), std::string(influence["influencer_field_name"].GetString()));
        CPPUNIT_ASSERT(influence.HasMember("influencer_field_values"));
        const rapidjson::Value& influencerFieldValues = influence["influencer_field_values"];
        CPPUNIT_ASSERT(influencerFieldValues.IsArray());
        CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(3), influencerFieldValues.Size());

        // Check influencers are ordered
        CPPUNIT_ASSERT_EQUAL(std::string("cat"), std::string(influencerFieldValues[rapidjson::SizeType(0)].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("dave"), std::string(influencerFieldValues[rapidjson::SizeType(1)].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("jo"), std::string(influencerFieldValues[rapidjson::SizeType(2)].GetString()));
    }
}

void CJsonOutputWriterTest::testPersistNormalizer() {
    ml::model::CAnomalyDetectorModelConfig modelConfig = ml::model::CAnomalyDetectorModelConfig::defaultConfig();

    std::ostringstream sstream;
    ml::core_t::TTime persistTime(1);
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::model::CHierarchicalResultsNormalizer normalizer(modelConfig);
        writer.persistNormalizer(normalizer, persistTime);
        writer.finalise();
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    CPPUNIT_ASSERT(persistTime <= ml::core::CTimeUtils::now());
    CPPUNIT_ASSERT(persistTime > ml::core::CTimeUtils::now() - 10);

    CPPUNIT_ASSERT(doc.IsArray());

    const rapidjson::Value& quantileWrapper = doc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(quantileWrapper.HasMember("quantiles"));
    const rapidjson::Value& quantileState = quantileWrapper["quantiles"];
    CPPUNIT_ASSERT(quantileState.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(quantileState["job_id"].GetString()));
    CPPUNIT_ASSERT(quantileState.HasMember("quantile_state"));
    CPPUNIT_ASSERT(quantileState.HasMember("timestamp"));
}

void CJsonOutputWriterTest::testPartitionScores() {
    ml::model::CAnomalyDetectorModelConfig modelConfig = ml::model::CAnomalyDetectorModelConfig::defaultConfig();

    std::ostringstream sstream;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        std::string emptyString;
        ml::api::CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec influences;

        std::string partitionFieldName("part1");

        for (int i = 0; i < 4; ++i) {
            // For the first iteration use an empty string for the value
            std::string partitionFieldValue;
            if (i > 0) {
                partitionFieldValue = 'p' + ml::core::CStringUtils::typeToString(i);
            }
            ml::api::CHierarchicalResultsWriter::SResults result(ml::api::CHierarchicalResultsWriter::E_PartitionResult,
                                                                 partitionFieldName,
                                                                 partitionFieldValue,
                                                                 emptyString,
                                                                 emptyString,
                                                                 emptyString,
                                                                 1,
                                                                 emptyString,
                                                                 emptyString,
                                                                 42.0,
                                                                 79,
                                                                 TDouble1Vec(1, 6953.0),
                                                                 TDouble1Vec(1, 10090.0),
                                                                 0.0,
                                                                 double(i), // normalised anomaly score
                                                                 0.1,
                                                                 emptyString,
                                                                 influences,
                                                                 false,
                                                                 true,
                                                                 1,
                                                                 100,
                                                                 EMPTY_STRING_LIST);

            writer.acceptResult(result);
        }

        writer.endOutputBatch(false, 1ul);
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    LOG_DEBUG(<< sstream.str());

    const rapidjson::Value& bucketWrapper = doc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(bucketWrapper.HasMember("bucket"));
    const rapidjson::Value& bucket = bucketWrapper["bucket"];
    CPPUNIT_ASSERT(bucket.HasMember("partition_scores"));
    const rapidjson::Value& partitionScores = bucket["partition_scores"];

    CPPUNIT_ASSERT(partitionScores.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(4), partitionScores.Size());

    for (rapidjson::SizeType i = 0; i < partitionScores.Size(); ++i) {
        const rapidjson::Value& pDoc = partitionScores[i];
        CPPUNIT_ASSERT(pDoc.IsObject());
        CPPUNIT_ASSERT(pDoc.HasMember("probability"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1, pDoc["probability"].GetDouble(), 0.01);
        CPPUNIT_ASSERT(pDoc.HasMember("record_score"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(i), pDoc["record_score"].GetDouble(), 0.01);
        CPPUNIT_ASSERT(pDoc.HasMember("initial_record_score"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(i), pDoc["initial_record_score"].GetDouble(), 0.01);

        CPPUNIT_ASSERT(pDoc.HasMember("partition_field_name"));
        CPPUNIT_ASSERT_EQUAL(std::string("part1"), std::string(pDoc["partition_field_name"].GetString()));
        std::string fieldValue;
        if (i > 0) {
            fieldValue = 'p' + ml::core::CStringUtils::typeToString(i);
        }
        CPPUNIT_ASSERT(pDoc.HasMember("partition_field_value"));
        CPPUNIT_ASSERT_EQUAL(fieldValue, std::string(pDoc["partition_field_value"].GetString()));
    }
}

void CJsonOutputWriterTest::testReportMemoryUsage() {
    std::ostringstream sstream;
    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CJsonOutputWriter writer("job", outputStream);

        ml::model::CResourceMonitor::SResults resourceUsage;
        resourceUsage.s_Usage = 1;
        resourceUsage.s_ByFields = 2;
        resourceUsage.s_PartitionFields = 3;
        resourceUsage.s_OverFields = 4;
        resourceUsage.s_AllocationFailures = 5;
        resourceUsage.s_MemoryStatus = ml::model_t::E_MemoryStatusHardLimit;
        resourceUsage.s_BucketStartTime = 6;

        writer.reportMemoryUsage(resourceUsage);
        writer.endOutputBatch(false, 1ul);
    }

    LOG_DEBUG(<< sstream.str());

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    const rapidjson::Value& resourceWrapper = doc[rapidjson::SizeType(0)];
    CPPUNIT_ASSERT(resourceWrapper.HasMember("model_size_stats"));
    const rapidjson::Value& sizeStats = resourceWrapper["model_size_stats"];

    CPPUNIT_ASSERT(sizeStats.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job"), std::string(sizeStats["job_id"].GetString()));
    CPPUNIT_ASSERT(sizeStats.HasMember("model_bytes"));
    CPPUNIT_ASSERT_EQUAL(2, sizeStats["model_bytes"].GetInt());
    CPPUNIT_ASSERT(sizeStats.HasMember("total_by_field_count"));
    CPPUNIT_ASSERT_EQUAL(2, sizeStats["total_by_field_count"].GetInt());
    CPPUNIT_ASSERT(sizeStats.HasMember("total_partition_field_count"));
    CPPUNIT_ASSERT_EQUAL(3, sizeStats["total_partition_field_count"].GetInt());
    CPPUNIT_ASSERT(sizeStats.HasMember("total_over_field_count"));
    CPPUNIT_ASSERT_EQUAL(4, sizeStats["total_over_field_count"].GetInt());
    CPPUNIT_ASSERT(sizeStats.HasMember("bucket_allocation_failures_count"));
    CPPUNIT_ASSERT_EQUAL(5, sizeStats["bucket_allocation_failures_count"].GetInt());
    CPPUNIT_ASSERT(sizeStats.HasMember("timestamp"));
    CPPUNIT_ASSERT_EQUAL(6000, sizeStats["timestamp"].GetInt());
    CPPUNIT_ASSERT(sizeStats.HasMember("memory_status"));
    CPPUNIT_ASSERT_EQUAL(std::string("hard_limit"), std::string(sizeStats["memory_status"].GetString()));
    CPPUNIT_ASSERT(sizeStats.HasMember("log_time"));
    int64_t nowMs = ml::core::CTimeUtils::now() * 1000ll;
    CPPUNIT_ASSERT(nowMs >= sizeStats["log_time"].GetInt64());
    CPPUNIT_ASSERT(nowMs + 1000ll >= sizeStats["log_time"].GetInt64());
}

void CJsonOutputWriterTest::testWriteScheduledEvent() {
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
        ml::api::CHierarchicalResultsWriter::SResults result(ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                                                             partitionFieldName,
                                                             partitionFieldValue,
                                                             byFieldName,
                                                             byFieldValue,
                                                             emptyString,
                                                             100,
                                                             function,
                                                             functionDescription,
                                                             42.0,
                                                             79,
                                                             TDouble1Vec(1, 6953.0),
                                                             TDouble1Vec(1, 10090.0),
                                                             0.0,
                                                             0.1,
                                                             0.1,
                                                             fieldName,
                                                             influences,
                                                             false,
                                                             true,
                                                             1,
                                                             100,
                                                             EMPTY_STRING_LIST);
        CPPUNIT_ASSERT(writer.acceptResult(result));

        // This result has 2 scheduled events
        std::vector<std::string> eventDescriptions{"event-foo", "event-bar"};
        ml::api::CHierarchicalResultsWriter::SResults result2(ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                                                              partitionFieldName,
                                                              partitionFieldValue,
                                                              byFieldName,
                                                              byFieldValue,
                                                              emptyString,
                                                              200,
                                                              function,
                                                              functionDescription,
                                                              42.0,
                                                              79,
                                                              TDouble1Vec(1, 6953.0),
                                                              TDouble1Vec(1, 10090.0),
                                                              0.0,
                                                              0.1,
                                                              0.1,
                                                              fieldName,
                                                              influences,
                                                              false,
                                                              true,
                                                              1,
                                                              100,
                                                              eventDescriptions);

        CPPUNIT_ASSERT(writer.acceptResult(result2));
        CPPUNIT_ASSERT(writer.endOutputBatch(false, 1U));
    }

    rapidjson::Document doc;
    std::string out = sstream.str();
    doc.Parse<rapidjson::kParseDefaultFlags>(out);

    // Debug print record
    {
        rapidjson::StringBuffer strbuf;
        using TStringBufferPrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;
        TStringBufferPrettyWriter writer(strbuf);
        doc.Accept(writer);
        LOG_DEBUG(<< "Results:\n" << strbuf.GetString());
    }

    CPPUNIT_ASSERT(doc.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), doc.Size());
    // the first bucket has no events
    const rapidjson::Value& bucket = doc[rapidjson::SizeType(1)]["bucket"];
    CPPUNIT_ASSERT(bucket.HasMember("scheduled_event") == false);

    const rapidjson::Value& bucketWithEvents = doc[rapidjson::SizeType(1)]["bucket"];
    CPPUNIT_ASSERT(bucketWithEvents.HasMember("scheduled_events"));
    const rapidjson::Value& events = bucketWithEvents["scheduled_events"];
    CPPUNIT_ASSERT(events.IsArray());
    CPPUNIT_ASSERT_EQUAL(rapidjson::SizeType(2), events.Size());
    CPPUNIT_ASSERT_EQUAL(std::string("event-foo"), std::string(events[rapidjson::SizeType(0)].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("event-bar"), std::string(events[rapidjson::SizeType(1)].GetString()));
}

void CJsonOutputWriterTest::testThroughputWithScopedAllocator() {
    this->testThroughputHelper(true);
}

void CJsonOutputWriterTest::testThroughputWithoutScopedAllocator() {
    this->testThroughputHelper(false);
}

void CJsonOutputWriterTest::testThroughputHelper(bool useScopedAllocator) {
    // Write to /dev/null (Unix) or nul (Windows)
    std::ofstream ofs(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(ofs.is_open());

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

    ml::api::CHierarchicalResultsWriter::SResults result11(false,
                                                           false,
                                                           partitionFieldName,
                                                           partitionFieldValue,
                                                           overFieldName,
                                                           overFieldValue,
                                                           byFieldName,
                                                           byFieldValue,
                                                           correlatedByFieldValue,
                                                           1,
                                                           function,
                                                           functionDescription,
                                                           TDouble1Vec(1, 10090.0),
                                                           TDouble1Vec(1, 6953.0),
                                                           2.24,
                                                           0.5,
                                                           0.0,
                                                           79,
                                                           fieldName,
                                                           influences,
                                                           false,
                                                           false,
                                                           1,
                                                           100);

    ml::api::CHierarchicalResultsWriter::SResults result112(false,
                                                            true,
                                                            partitionFieldName,
                                                            partitionFieldValue,
                                                            overFieldName,
                                                            overFieldValue,
                                                            byFieldName,
                                                            byFieldValue,
                                                            correlatedByFieldValue,
                                                            1,
                                                            function,
                                                            functionDescription,
                                                            TDouble1Vec(1, 10090.0),
                                                            TDouble1Vec(1, 6953.0),
                                                            2.24,
                                                            0.5,
                                                            0.0,
                                                            79,
                                                            fieldName,
                                                            influences,
                                                            false,
                                                            false,
                                                            1,
                                                            100);

    ml::api::CHierarchicalResultsWriter::SResults result12(ml::api::CHierarchicalResultsWriter::E_Result,
                                                           partitionFieldName,
                                                           partitionFieldValue,
                                                           byFieldName,
                                                           byFieldValue,
                                                           correlatedByFieldValue,
                                                           1,
                                                           function,
                                                           functionDescription,
                                                           42.0,
                                                           79,
                                                           TDouble1Vec(1, 6953.0),
                                                           TDouble1Vec(1, 10090.0),
                                                           2.24,
                                                           0.8,
                                                           0.0,
                                                           fieldName,
                                                           influences,
                                                           false,
                                                           true,
                                                           2,
                                                           100,
                                                           EMPTY_STRING_LIST);

    ml::api::CHierarchicalResultsWriter::SResults result13(ml::api::CHierarchicalResultsWriter::E_SimpleCountResult,
                                                           partitionFieldName,
                                                           partitionFieldValue,
                                                           byFieldName,
                                                           byFieldValue,
                                                           correlatedByFieldValue,
                                                           1,
                                                           function,
                                                           functionDescription,
                                                           42.0,
                                                           79,
                                                           TDouble1Vec(1, 6953.0),
                                                           TDouble1Vec(1, 10090.0),
                                                           2.24,
                                                           0.5,
                                                           0.0,
                                                           fieldName,
                                                           influences,
                                                           false,
                                                           false,
                                                           3,
                                                           100,
                                                           EMPTY_STRING_LIST);

    ml::api::CHierarchicalResultsWriter::SResults result14(ml::api::CHierarchicalResultsWriter::E_Result,
                                                           partitionFieldName,
                                                           partitionFieldValue,
                                                           byFieldName,
                                                           byFieldValue,
                                                           correlatedByFieldValue,
                                                           1,
                                                           function,
                                                           functionDescription,
                                                           42.0,
                                                           79,
                                                           TDouble1Vec(1, 6953.0),
                                                           TDouble1Vec(1, 10090.0),
                                                           2.24,
                                                           0.0,
                                                           0.0,
                                                           fieldName,
                                                           influences,
                                                           false,
                                                           false,
                                                           4,
                                                           100,
                                                           EMPTY_STRING_LIST);

    // 1st bucket
    writer.acceptBucketTimeInfluencer(1, 0.01, 13.44, 70.0);

    // Write the record this many times
    static const size_t TEST_SIZE(75000);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        if (useScopedAllocator) {
            using TScopedAllocator = ml::core::CScopedRapidJsonPoolAllocator<ml::api::CJsonOutputWriter>;
            static const std::string ALLOCATOR_ID("CAnomalyJob::writeOutResults");
            TScopedAllocator scopedAllocator(ALLOCATOR_ID, writer);

            CPPUNIT_ASSERT(writer.acceptResult(result11));
            CPPUNIT_ASSERT(writer.acceptResult(result11));
            CPPUNIT_ASSERT(writer.acceptResult(result112));
            CPPUNIT_ASSERT(writer.acceptResult(result12));
            CPPUNIT_ASSERT(writer.acceptResult(result12));
            CPPUNIT_ASSERT(writer.acceptResult(result13));
            CPPUNIT_ASSERT(writer.acceptResult(result13));
            CPPUNIT_ASSERT(writer.acceptResult(result14));
            CPPUNIT_ASSERT(writer.acceptResult(result14));

            // Finished adding results
            CPPUNIT_ASSERT(writer.endOutputBatch(false, 1U));
        } else {
            CPPUNIT_ASSERT(writer.acceptResult(result11));
            CPPUNIT_ASSERT(writer.acceptResult(result11));
            CPPUNIT_ASSERT(writer.acceptResult(result112));
            CPPUNIT_ASSERT(writer.acceptResult(result12));
            CPPUNIT_ASSERT(writer.acceptResult(result12));
            CPPUNIT_ASSERT(writer.acceptResult(result13));
            CPPUNIT_ASSERT(writer.acceptResult(result13));
            CPPUNIT_ASSERT(writer.acceptResult(result14));
            CPPUNIT_ASSERT(writer.acceptResult(result14));

            // Finished adding results
            CPPUNIT_ASSERT(writer.endOutputBatch(false, 1U));
        }
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Writing " << TEST_SIZE << " records took " << (end - start) << " seconds");
}
