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
#include "CCsvOutputWriterTest.h"

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <api/CCsvOutputWriter.h>

#include <fstream>
#include <sstream>


CppUnit::Test *CCsvOutputWriterTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CCsvOutputWriterTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvOutputWriterTest>(
                               "CCsvOutputWriterTest::testAdd",
                               &CCsvOutputWriterTest::testAdd) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvOutputWriterTest>(
                               "CCsvOutputWriterTest::testOverwrite",
                               &CCsvOutputWriterTest::testOverwrite) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvOutputWriterTest>(
                               "CCsvOutputWriterTest::testThroughput",
                               &CCsvOutputWriterTest::testThroughput) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvOutputWriterTest>(
                               "CCsvOutputWriterTest::testExcelQuoting",
                               &CCsvOutputWriterTest::testExcelQuoting) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvOutputWriterTest>(
                               "CCsvOutputWriterTest::testNonExcelQuoting",
                               &CCsvOutputWriterTest::testNonExcelQuoting) );

    return suiteOfTests;
}

void CCsvOutputWriterTest::testAdd(void) {
    // In this test, the output is the input plus an extra field - no input
    // fields are changed

    ml::api::CCsvOutputWriter writer;

    ml::api::CCsvOutputWriter::TStrVec fieldNames;
    fieldNames.push_back("_cd");
    fieldNames.push_back("_indextime");
    fieldNames.push_back("_kv");
    fieldNames.push_back("_raw");
    fieldNames.push_back("_serial");
    fieldNames.push_back("_si");
    fieldNames.push_back("_sourcetype");
    fieldNames.push_back("_time");
    fieldNames.push_back("date_hour");
    fieldNames.push_back("date_mday");
    fieldNames.push_back("date_minute");
    fieldNames.push_back("date_month");
    fieldNames.push_back("date_second");
    fieldNames.push_back("date_wday");
    fieldNames.push_back("date_year");
    fieldNames.push_back("date_zone");
    fieldNames.push_back("eventtype");
    fieldNames.push_back("host");
    fieldNames.push_back("index");
    fieldNames.push_back("linecount");
    fieldNames.push_back("punct");
    fieldNames.push_back("source");
    fieldNames.push_back("sourcetype");
    fieldNames.push_back("server");
    fieldNames.push_back("timeendpos");
    fieldNames.push_back("timestartpos");

    ml::api::CCsvOutputWriter::TStrVec mlFieldNames;
    mlFieldNames.push_back("ml_type");

    CPPUNIT_ASSERT(writer.fieldNames(fieldNames, mlFieldNames));

    ml::api::CCsvOutputWriter::TStrStrUMap originalFields;
    originalFields["_cd"] = "0:3933689";
    originalFields["_indextime"] = "1337698174";
    originalFields["_kv"] = "1";
    originalFields["_raw"] = "2010-02-11 16:11:19+00,service has started,160198,24";
    originalFields["_serial"] = "14";
    originalFields["_si"] = "linux.prelert.com\nmain";
    originalFields["_sourcetype"] = "rmds";
    originalFields["_time"] = "1265904679";
    originalFields["date_hour"] = "16";
    originalFields["date_mday"] = "11";
    originalFields["date_minute"] = "11";
    originalFields["date_month"] = "february";
    originalFields["date_second"] = "19";
    originalFields["date_wday"] = "thursday";
    originalFields["date_year"] = "2010";
    originalFields["date_zone"] = "local";
    originalFields["eventtype"] = "";
    originalFields["host"] = "linux.prelert.com";
    originalFields["index"] = "main";
    originalFields["linecount"] = "1";
    originalFields["punct"] = "--_::+,__,,";
    originalFields["source"] = "/export/temp/cs_10feb_reload";
    originalFields["sourcetype"] = "rmds";
    originalFields["server"] = "linux.prelert.com";
    originalFields["timeendpos"] = "22";
    originalFields["timestartpos"] = "0";

    ml::api::CCsvOutputWriter::TStrStrUMap mlFields;
    mlFields["ml_type"] = "75";

    CPPUNIT_ASSERT(writer.writeRow(originalFields, mlFields));

    std::string output(writer.internalString());

    LOG_DEBUG("Output is:\n" << output);

    for (ml::api::CCsvOutputWriter::TStrVecCItr iter = fieldNames.begin();
         iter != fieldNames.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << *iter << "'");
        CPPUNIT_ASSERT(output.find(*iter) != std::string::npos);
    }

    for (ml::api::CCsvOutputWriter::TStrVecCItr iter = mlFieldNames.begin();
         iter != mlFieldNames.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << *iter << "'");
        CPPUNIT_ASSERT(output.find(*iter) != std::string::npos);
    }

    for (ml::api::CCsvOutputWriter::TStrStrUMapCItr iter = originalFields.begin();
         iter != originalFields.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << iter->second << "'");
        CPPUNIT_ASSERT(output.find(iter->second) != std::string::npos);
    }

    for (ml::api::CCsvOutputWriter::TStrStrUMapCItr iter = mlFields.begin();
         iter != mlFields.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << iter->second << "'");
        CPPUNIT_ASSERT(output.find(iter->second) != std::string::npos);
    }
}

void CCsvOutputWriterTest::testOverwrite(void) {
    // In this test, some fields from the input are changed in the output

    ml::api::CCsvOutputWriter writer;

    ml::api::CCsvOutputWriter::TStrVec fieldNames;
    fieldNames.push_back("_cd");
    fieldNames.push_back("_indextime");
    fieldNames.push_back("_kv");
    fieldNames.push_back("_raw");
    fieldNames.push_back("_serial");
    fieldNames.push_back("_si");
    fieldNames.push_back("_sourcetype");
    fieldNames.push_back("_time");
    fieldNames.push_back("date_hour");
    fieldNames.push_back("date_mday");
    fieldNames.push_back("date_minute");
    fieldNames.push_back("date_month");
    fieldNames.push_back("date_second");
    fieldNames.push_back("date_wday");
    fieldNames.push_back("date_year");
    fieldNames.push_back("date_zone");
    fieldNames.push_back("eventtype");
    fieldNames.push_back("host");
    fieldNames.push_back("index");
    fieldNames.push_back("linecount");
    fieldNames.push_back("punct");
    fieldNames.push_back("source");
    fieldNames.push_back("sourcetype");
    fieldNames.push_back("server");
    fieldNames.push_back("timeendpos");
    fieldNames.push_back("timestartpos");

    ml::api::CCsvOutputWriter::TStrVec mlFieldNames;
    mlFieldNames.push_back("eventtype");
    mlFieldNames.push_back("ml_type");

    CPPUNIT_ASSERT(writer.fieldNames(fieldNames, mlFieldNames));

    ml::api::CCsvOutputWriter::TStrStrUMap originalFields;
    originalFields["_cd"] = "0:3933689";
    originalFields["_indextime"] = "1337698174";
    originalFields["_kv"] = "1";
    originalFields["_raw"] = "2010-02-11 16:11:19+00,service has started,160198,24";
    originalFields["_serial"] = "14";
    originalFields["_si"] = "linux.prelert.com\nmain";
    originalFields["_sourcetype"] = "rmds";
    originalFields["_time"] = "1265904679";
    originalFields["date_hour"] = "16";
    originalFields["date_mday"] = "11";
    originalFields["date_minute"] = "11";
    originalFields["date_month"] = "february";
    originalFields["date_second"] = "19";
    originalFields["date_wday"] = "thursday";
    originalFields["date_year"] = "2010";
    originalFields["date_zone"] = "local";
    originalFields["eventtype"] = "";
    originalFields["host"] = "linux.prelert.com";
    originalFields["index"] = "main";
    originalFields["linecount"] = "1";
    originalFields["punct"] = "--_::+,__,,";
    originalFields["source"] = "/export/temp/cs_10feb_reload";
    originalFields["sourcetype"] = "rmds";
    originalFields["server"] = "linux.prelert.com";
    originalFields["timeendpos"] = "22";
    originalFields["timestartpos"] = "0";

    ml::api::CCsvOutputWriter::TStrStrUMap mlFields;
    mlFields["_cd"] = "2:8934689";
    mlFields["date_zone"] = "GMT";
    mlFields["ml_type"] = "42";

    CPPUNIT_ASSERT(writer.writeRow(originalFields, mlFields));

    std::string output(writer.internalString());

    LOG_DEBUG("Output is:\n" << output);

    for (ml::api::CCsvOutputWriter::TStrVecCItr iter = fieldNames.begin();
         iter != fieldNames.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << *iter << "'");
        CPPUNIT_ASSERT(output.find(*iter) != std::string::npos);
    }

    for (ml::api::CCsvOutputWriter::TStrVecCItr iter = mlFieldNames.begin();
         iter != mlFieldNames.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << *iter << "'");
        CPPUNIT_ASSERT(output.find(*iter) != std::string::npos);
    }

    for (ml::api::CCsvOutputWriter::TStrStrUMapCItr iter = originalFields.begin();
         iter != originalFields.end();
         ++iter) {
        // The Ml fields should override the originals
        if (mlFields.find(iter->first) == mlFields.end()) {
            LOG_DEBUG("Checking output contains '" << iter->second << "'");
            CPPUNIT_ASSERT(output.find(iter->second) != std::string::npos);
        } else {
            LOG_DEBUG("Checking output does not contain '" << iter->second << "'");
            CPPUNIT_ASSERT(output.find(iter->second) == std::string::npos);
        }
    }

    for (ml::api::CCsvOutputWriter::TStrStrUMapCItr iter = mlFields.begin();
         iter != mlFields.end();
         ++iter) {
        LOG_DEBUG("Checking output contains '" << iter->second << "'");
        CPPUNIT_ASSERT(output.find(iter->second) != std::string::npos);
    }
}

void CCsvOutputWriterTest::testThroughput(void) {
    // In this test, some fields from the input are changed in the output

    // Write to /dev/null (Unix) or nul (Windows)
    std::ofstream ofs(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(ofs.is_open());

    ml::api::CCsvOutputWriter writer(ofs);

    ml::api::CCsvOutputWriter::TStrVec fieldNames;
    fieldNames.push_back("_cd");
    fieldNames.push_back("_indextime");
    fieldNames.push_back("_kv");
    fieldNames.push_back("_raw");
    fieldNames.push_back("_serial");
    fieldNames.push_back("_si");
    fieldNames.push_back("_sourcetype");
    fieldNames.push_back("_time");
    fieldNames.push_back("date_hour");
    fieldNames.push_back("date_mday");
    fieldNames.push_back("date_minute");
    fieldNames.push_back("date_month");
    fieldNames.push_back("date_second");
    fieldNames.push_back("date_wday");
    fieldNames.push_back("date_year");
    fieldNames.push_back("date_zone");
    fieldNames.push_back("eventtype");
    fieldNames.push_back("host");
    fieldNames.push_back("index");
    fieldNames.push_back("linecount");
    fieldNames.push_back("punct");
    fieldNames.push_back("source");
    fieldNames.push_back("sourcetype");
    fieldNames.push_back("server");
    fieldNames.push_back("timeendpos");
    fieldNames.push_back("timestartpos");

    ml::api::CCsvOutputWriter::TStrVec mlFieldNames;
    mlFieldNames.push_back("eventtype");
    mlFieldNames.push_back("ml_type");

    ml::api::CCsvOutputWriter::TStrStrUMap originalFields;
    originalFields["_cd"] = "0:3933689";
    originalFields["_indextime"] = "1337698174";
    originalFields["_kv"] = "1";
    originalFields["_raw"] = "2010-02-11 16:11:19+00,service has started,160198,24";
    originalFields["_serial"] = "14";
    originalFields["_si"] = "linux.prelert.com\nmain";
    originalFields["_sourcetype"] = "rmds";
    originalFields["_time"] = "1265904679";
    originalFields["date_hour"] = "16";
    originalFields["date_mday"] = "11";
    originalFields["date_minute"] = "11";
    originalFields["date_month"] = "february";
    originalFields["date_second"] = "19";
    originalFields["date_wday"] = "thursday";
    originalFields["date_year"] = "2010";
    originalFields["date_zone"] = "local";
    originalFields["eventtype"] = "";
    originalFields["host"] = "linux.prelert.com";
    originalFields["index"] = "main";
    originalFields["linecount"] = "1";
    originalFields["punct"] = "--_::+,__,,";
    originalFields["source"] = "/export/temp/cs_10feb_reload";
    originalFields["sourcetype"] = "rmds";
    originalFields["server"] = "linux.prelert.com";
    originalFields["timeendpos"] = "22";
    originalFields["timestartpos"] = "0";

    ml::api::CCsvOutputWriter::TStrStrUMap mlFields;
    mlFields["_cd"] = "2:8934689";
    mlFields["date_zone"] = "GMT";
    mlFields["ml_type"] = "42";

    // Write the record this many times
    static const size_t TEST_SIZE(75000);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    CPPUNIT_ASSERT(writer.fieldNames(fieldNames, mlFieldNames));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        CPPUNIT_ASSERT(writer.writeRow(originalFields, mlFields));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO("Writing " << TEST_SIZE <<
             " records took " << (end - start) << " seconds");
}

void CCsvOutputWriterTest::testExcelQuoting(void) {
    ml::api::CCsvOutputWriter writer;

    ml::api::CCsvOutputWriter::TStrVec fieldNames;
    fieldNames.push_back("no_special");
    fieldNames.push_back("contains_quote");
    fieldNames.push_back("contains_quote_quote");
    fieldNames.push_back("contains_separator");
    fieldNames.push_back("contains_quote_separator");
    fieldNames.push_back("contains_newline");
    fieldNames.push_back("contains_quote_newline");

    CPPUNIT_ASSERT(writer.fieldNames(fieldNames));

    ml::api::CCsvOutputWriter::TStrStrUMap fieldValues;
    fieldValues["no_special"] = "a";
    fieldValues["contains_quote"] = "\"";
    fieldValues["contains_quote_quote"] = "\"\"";
    fieldValues["contains_separator"] = ",";
    fieldValues["contains_quote_separator"] = "\",";
    fieldValues["contains_newline"] = "\n";
    fieldValues["contains_quote_newline"] = "\"\n";

    CPPUNIT_ASSERT(writer.writeRow(fieldValues));

    std::string output(writer.internalString());

    LOG_DEBUG("Output is:\n" << output);

    CPPUNIT_ASSERT_EQUAL(std::string(
                             "no_special,"
                             "contains_quote,"
                             "contains_quote_quote,"
                             "contains_separator,"
                             "contains_quote_separator,"
                             "contains_newline,"
                             "contains_quote_newline\n"
                             "a,"
                             "\"\"\"\","
                             "\"\"\"\"\"\","
                             "\",\","
                             "\"\"\",\","
                             "\"\n\","
                             "\"\"\"\n\"\n"
                             ),
                         output);
}

void CCsvOutputWriterTest::testNonExcelQuoting(void) {
    ml::api::CCsvOutputWriter writer(false,
                                     true,
                                     '\\');

    ml::api::CCsvOutputWriter::TStrVec fieldNames;
    fieldNames.push_back("no_special");
    fieldNames.push_back("contains_quote");
    fieldNames.push_back("contains_escape");
    fieldNames.push_back("contains_escape_quote");
    fieldNames.push_back("contains_separator");
    fieldNames.push_back("contains_escape_separator");
    fieldNames.push_back("contains_newline");
    fieldNames.push_back("contains_escape_newline");

    CPPUNIT_ASSERT(writer.fieldNames(fieldNames));

    ml::api::CCsvOutputWriter::TStrStrUMap fieldValues;
    fieldValues["no_special"] = "a";
    fieldValues["contains_quote"] = "\"";
    fieldValues["contains_escape"] = "\\";
    fieldValues["contains_escape_quote"] = "\\\"";
    fieldValues["contains_separator"] = ",";
    fieldValues["contains_escape_separator"] = "\\,";
    fieldValues["contains_newline"] = "\n";
    fieldValues["contains_escape_newline"] = "\\\n";

    CPPUNIT_ASSERT(writer.writeRow(fieldValues));

    std::string output(writer.internalString());

    LOG_DEBUG("Output is:\n" << output);

    CPPUNIT_ASSERT_EQUAL(std::string(
                             "no_special,"
                             "contains_quote,"
                             "contains_escape,"
                             "contains_escape_quote,"
                             "contains_separator,"
                             "contains_escape_separator,"
                             "contains_newline,"
                             "contains_escape_newline\n"
                             "a,"
                             "\"\\\"\","
                             "\"\\\\\","
                             "\"\\\\\\\"\","
                             "\",\","
                             "\"\\\\,\","
                             "\"\n\","
                             "\"\\\\\n\"\n"
                             ),
                         output);
}

