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

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <api/CCsvOutputWriter.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CCsvOutputWriterTest)

BOOST_AUTO_TEST_CASE(testAdd) {
    // In this test, the output is the input plus an extra field - no input
    // fields are changed

    ml::api::CCsvOutputWriter writer;

    ml::api::CCsvOutputWriter::TStrVec fieldNames{
        "_cd",         "_indextime",  "_kv",         "_raw",      "_serial",
        "_si",         "_sourcetype", "_time",       "date_hour", "date_mday",
        "date_minute", "date_month",  "date_second", "date_wday", "date_year",
        "date_zone",   "eventtype",   "host",        "index",     "linecount",
        "punct",       "source",      "sourcetype",  "server",    "timeendpos",
        "timestartpos"};

    ml::api::CCsvOutputWriter::TStrVec mlFieldNames{"mlcategory"};

    BOOST_TEST_REQUIRE(writer.fieldNames(fieldNames, mlFieldNames));

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
    mlFields["mlcategory"] = "75";

    BOOST_TEST_REQUIRE(writer.writeRow(originalFields, mlFields));

    std::string output{writer.internalString()};

    LOG_DEBUG(<< "Output is:\n" << output);

    for (const auto& fieldName : fieldNames) {
        BOOST_TEST_REQUIRE(output.find(fieldName) != std::string::npos);
    }

    for (const auto& fieldName : mlFieldNames) {
        BOOST_TEST_REQUIRE(output.find(fieldName) != std::string::npos);
    }

    for (const auto& field : originalFields) {
        BOOST_TEST_REQUIRE(output.find(field.second) != std::string::npos);
    }

    for (const auto& field : mlFields) {
        BOOST_TEST_REQUIRE(output.find(field.second) != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(testOverwrite) {
    // In this test, some fields from the input are changed in the output

    ml::api::CCsvOutputWriter writer;

    ml::api::CCsvOutputWriter::TStrVec fieldNames{
        "_cd",         "_indextime",  "_kv",         "_raw",      "_serial",
        "_si",         "_sourcetype", "_time",       "date_hour", "date_mday",
        "date_minute", "date_month",  "date_second", "date_wday", "date_year",
        "date_zone",   "eventtype",   "host",        "index",     "linecount",
        "punct",       "source",      "sourcetype",  "server",    "timeendpos",
        "timestartpos"};

    ml::api::CCsvOutputWriter::TStrVec mlFieldNames{"eventtype", "mlcategory"};

    BOOST_TEST_REQUIRE(writer.fieldNames(fieldNames, mlFieldNames));

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
    mlFields["mlcategory"] = "42";

    BOOST_TEST_REQUIRE(writer.writeRow(originalFields, mlFields));

    std::string output{writer.internalString()};

    LOG_DEBUG(<< "Output is:\n" << output);

    for (const auto& fieldName : fieldNames) {
        BOOST_TEST_REQUIRE(output.find(fieldName) != std::string::npos);
    }

    for (const auto& fieldName : mlFieldNames) {
        BOOST_TEST_REQUIRE(output.find(fieldName) != std::string::npos);
    }

    for (const auto& field : originalFields) {
        // The ML fields should override the originals
        if (mlFields.find(field.first) == mlFields.end()) {
            BOOST_TEST_REQUIRE(output.find(field.second) != std::string::npos);
        } else {
            BOOST_TEST_REQUIRE(output.find(field.second) == std::string::npos);
        }
    }

    for (const auto& field : mlFields) {
        BOOST_TEST_REQUIRE(output.find(field.second) != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(testThroughput) {
    // In this test, some fields from the input are changed in the output

    // Write to /dev/null (Unix) or nul (Windows)
    std::ofstream ofs{ml::core::COsFileFuncs::NULL_FILENAME};
    BOOST_TEST_REQUIRE(ofs.is_open());

    ml::api::CCsvOutputWriter writer{ofs};

    ml::api::CCsvOutputWriter::TStrVec fieldNames{
        "_cd",         "_indextime",  "_kv",         "_raw",      "_serial",
        "_si",         "_sourcetype", "_time",       "date_hour", "date_mday",
        "date_minute", "date_month",  "date_second", "date_wday", "date_year",
        "date_zone",   "eventtype",   "host",        "index",     "linecount",
        "punct",       "source",      "sourcetype",  "server",    "timeendpos",
        "timestartpos"};

    ml::api::CCsvOutputWriter::TStrVec mlFieldNames{"eventtype", "mlcategory"};

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
    mlFields["mlcategory"] = "42";

    // Write the record this many times
    static const std::size_t TEST_SIZE{75000};

    ml::core_t::TTime start{ml::core::CTimeUtils::now()};
    LOG_INFO(<< "Starting throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    BOOST_TEST_REQUIRE(writer.fieldNames(fieldNames, mlFieldNames));

    for (std::size_t count = 0; count < TEST_SIZE; ++count) {
        BOOST_TEST_REQUIRE(writer.writeRow(originalFields, mlFields));
    }

    ml::core_t::TTime end{ml::core::CTimeUtils::now()};
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Writing " << TEST_SIZE << " records took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testExcelQuoting) {
    ml::api::CCsvOutputWriter writer;

    ml::api::CCsvOutputWriter::TStrVec fieldNames{"no_special",
                                                  "contains_quote",
                                                  "contains_quote_quote",
                                                  "contains_separator",
                                                  "contains_quote_separator",
                                                  "contains_newline",
                                                  "contains_quote_newline"};

    BOOST_TEST_REQUIRE(writer.fieldNames(fieldNames));

    ml::api::CCsvOutputWriter::TStrStrUMap fieldValues;
    fieldValues["no_special"] = "a";
    fieldValues["contains_quote"] = "\"";
    fieldValues["contains_quote_quote"] = "\"\"";
    fieldValues["contains_separator"] = ",";
    fieldValues["contains_quote_separator"] = "\",";
    fieldValues["contains_newline"] = "\n";
    fieldValues["contains_quote_newline"] = "\"\n";

    BOOST_TEST_REQUIRE(writer.writeRow(fieldValues));

    std::string output{writer.internalString()};

    LOG_DEBUG(<< "Output is:\n" << output);

    BOOST_REQUIRE_EQUAL("no_special,"
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
                        "\"\"\"\n\"\n",
                        output);
}

BOOST_AUTO_TEST_CASE(testNonExcelQuoting) {
    ml::api::CCsvOutputWriter writer{true, '\\'};

    ml::api::CCsvOutputWriter::TStrVec fieldNames{
        "no_special",         "contains_quote",
        "contains_escape",    "contains_escape_quote",
        "contains_separator", "contains_escape_separator",
        "contains_newline",   "contains_escape_newline"};

    BOOST_TEST_REQUIRE(writer.fieldNames(fieldNames));

    ml::api::CCsvOutputWriter::TStrStrUMap fieldValues;
    fieldValues["no_special"] = "a";
    fieldValues["contains_quote"] = "\"";
    fieldValues["contains_escape"] = "\\";
    fieldValues["contains_escape_quote"] = "\\\"";
    fieldValues["contains_separator"] = ",";
    fieldValues["contains_escape_separator"] = "\\,";
    fieldValues["contains_newline"] = "\n";
    fieldValues["contains_escape_newline"] = "\\\n";

    BOOST_TEST_REQUIRE(writer.writeRow(fieldValues));

    std::string output{writer.internalString()};

    LOG_DEBUG(<< "Output is:\n" << output);

    BOOST_REQUIRE_EQUAL("no_special,"
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
                        "\"\\\\\n\"\n",
                        output);
}

BOOST_AUTO_TEST_SUITE_END()
