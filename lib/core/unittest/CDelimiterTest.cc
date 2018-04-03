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
#include "CDelimiterTest.h"

#include <core/CDelimiter.h>
#include <core/CLogger.h>

#include <algorithm>
#include <sstream>


CppUnit::Test *CDelimiterTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CDelimiterTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CDelimiterTest>(
                                   "CDelimiterTest::testSimpleTokenise",
                                   &CDelimiterTest::testSimpleTokenise) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDelimiterTest>(
                                   "CDelimiterTest::testRegexTokenise",
                                   &CDelimiterTest::testRegexTokenise) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDelimiterTest>(
                                   "CDelimiterTest::testQuotedTokenise",
                                   &CDelimiterTest::testQuotedTokenise) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDelimiterTest>(
                                   "CDelimiterTest::testQuotedEscapedTokenise",
                                   &CDelimiterTest::testQuotedEscapedTokenise) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDelimiterTest>(
                                   "CDelimiterTest::testInvalidQuotedTokenise",
                                   &CDelimiterTest::testInvalidQuotedTokenise) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDelimiterTest>(
                                   "CDelimiterTest::testQuoteEqualsEscapeTokenise",
                                   &CDelimiterTest::testQuoteEqualsEscapeTokenise) );
    return suiteOfTests;
}

void CDelimiterTest::testSimpleTokenise()
{
    std::string testData(
        "Oct 12, 2008 8:38:51 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
        "Oct 12, 2008 8:38:52 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
        "Oct 12, 2008 8:38:53 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
        "Oct 12, 2008 8:39:03 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
        "Oct 12, 2008 8:39:04 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
    );

    LOG_DEBUG("Input data:\n" << testData << '\n');

    ml::core::CDelimiter delimiter("\n", "\\w+\\s+\\d+,\\s+\\d+\\s+\\d+:\\d+:\\d+\\s+\\w+", true);

    ml::core::CStringUtils::TStrVec delimited;
    std::string remainder;

    delimiter.tokenise(testData, false, delimited, remainder);

    std::ostringstream strm1;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm1, "\n"));
    LOG_DEBUG("First output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm1.str());
    LOG_DEBUG("First remainder:\n" << remainder << '\n');

    CPPUNIT_ASSERT_EQUAL(size_t(4), delimited.size());
    CPPUNIT_ASSERT(remainder.size() > 0);

    delimited.clear();

    delimiter.tokenise(testData, true, delimited, remainder);

    std::ostringstream strm2;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm2, "\n"));
    LOG_DEBUG("Second output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm2.str());
    LOG_DEBUG("Second remainder:\n" << remainder << '\n');

    CPPUNIT_ASSERT_EQUAL(size_t(5), delimited.size());
    CPPUNIT_ASSERT_EQUAL(size_t(0), remainder.size());
}

void CDelimiterTest::testRegexTokenise()
{
    // Some of the lines here are Windows text format, and others Unix text
    std::string testData(
        "Oct 12, 2008 8:38:51 AM org.apache.tomcat.util.http.Parameters processParameters\r\n"
        "WARNING: Parameters: Invalid chunk ignored.\r\n"
        "Oct 12, 2008 8:38:52 AM org.apache.tomcat.util.http.Parameters processParameters\r\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
        "Oct 12, 2008 8:38:53 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\r\n"
        "Oct 12, 2008 8:39:03 AM org.apache.tomcat.util.http.Parameters processParameters\r\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
        "Oct 12, 2008 8:39:04 AM org.apache.tomcat.util.http.Parameters processParameters\n"
        "WARNING: Parameters: Invalid chunk ignored.\n"
    );

    LOG_DEBUG("Input data:\n" << testData << '\n');

    // Regex matches line terminator for either Windows or Unix text
    ml::core::CDelimiter delimiter("\r?\n", "\\w+\\s+\\d+,\\s+\\d+\\s+\\d+:\\d+:\\d+\\s+\\w+", true);

    ml::core::CStringUtils::TStrVec delimited;
    std::string remainder;

    delimiter.tokenise(testData, false, delimited, remainder);

    std::ostringstream strm1;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm1, "\n"));
    LOG_DEBUG("First output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm1.str());
    LOG_DEBUG("First remainder:\n" << remainder << '\n');

    CPPUNIT_ASSERT_EQUAL(size_t(4), delimited.size());
    CPPUNIT_ASSERT(remainder.size() > 0);

    delimited.clear();

    delimiter.tokenise(testData, true, delimited, remainder);

    std::ostringstream strm2;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm2, "\n"));
    LOG_DEBUG("Second output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm2.str());
    LOG_DEBUG("Second remainder:\n" << remainder << '\n');

    CPPUNIT_ASSERT_EQUAL(size_t(5), delimited.size());
    CPPUNIT_ASSERT_EQUAL(size_t(0), remainder.size());
}

void CDelimiterTest::testQuotedTokenise()
{
    // NB: The backslashes here escape the quotes for the benefit of the C++ compiler
    std::string testData(
        "3,1,5415.1132,56135135,0x00000001,0x00000002,\"SOME_STRING\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",0x0000000000000000,0x0000000000000000,\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\""
    );

    LOG_DEBUG("Input data:\n" << testData << '\n');

    ml::core::CDelimiter delimiter(",");
    delimiter.quote('"');

    ml::core::CStringUtils::TStrVec delimited;
    std::string remainder;

    delimiter.tokenise(testData, false, delimited, remainder);

    delimited.push_back(remainder);

    std::ostringstream strm;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm, "\n"));
    LOG_DEBUG("Quoted output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm.str());

    // 40 fields (most blank)
    CPPUNIT_ASSERT_EQUAL(size_t(40), delimited.size());
}

void CDelimiterTest::testQuotedEscapedTokenise()
{
    // Similar to previous test, but there are four values with escaped quotes in AFTER
    // pre-processing by the C++ compiler
    std::string testData(
        "3,1,5415.1132,56135135,0x00000001,0x00000002,\"SOME_STRING\",\"\",\"\\\"\",\"\",\"\",\"\",\"\",\"\",\"A \\\"middling\\\" one\",\"\",\"\",\"\",\"\",0x0000000000000000,0x0000000000000000,\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\\\"start\",\"\",\"\",\"end\\\"\",\"\",\"\",\"\",\"\",\"\",\"\\\"both\\\"\",\"\",\"\""
    );

    LOG_DEBUG("Input data:\n" << testData << '\n');

    ml::core::CDelimiter delimiter(",");
    delimiter.quote('"');

    ml::core::CStringUtils::TStrVec delimited;
    std::string remainder;

    delimiter.tokenise(testData, false, delimited, remainder);

    delimited.push_back(remainder);

    std::ostringstream strm;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm, "\n"));
    LOG_DEBUG("Quoted output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm.str());

    // 40 fields (most blank)
    CPPUNIT_ASSERT_EQUAL(size_t(40), delimited.size());
}

void CDelimiterTest::testInvalidQuotedTokenise()
{
    // Invalid quoting (e.g. mismatched) mustn't cause the tokeniser to go into
    // an infinite loop
    std::string testData(
        "4/26/2011 4:19,aaa.bbbbbb@cc.ddddd.com,\"64222\",\"/some_action.do?param1=foo&param2=Sljahfej+kfejhafef/3931nfV"
    );

    LOG_DEBUG("Input data:\n" << testData << '\n');

    ml::core::CDelimiter delimiter(",");
    delimiter.quote('"');

    ml::core::CStringUtils::TStrVec delimited;
    std::string remainder;

    delimiter.tokenise(testData, false, delimited, remainder);

    CPPUNIT_ASSERT_EQUAL(size_t(3), delimited.size());
    CPPUNIT_ASSERT_EQUAL(std::string("/some_action.do?param1=foo&param2=Sljahfej+kfejhafef/3931nfV"), remainder);
}

void CDelimiterTest::testQuoteEqualsEscapeTokenise()
{
    // In this example, double quotes are used for quoting, but they are escaped
    // by doubling them up, so the escape character is the same as the quote
    // character
    std::string testData(
        "May 24 22:02:13 1,2012/05/24 22:02:13,724747467,SOME_STRING,url,1,2012/04/10 02:53:17,192.168.0.3,192.168.0.1,0.0.0.0,0.0.0.0,aaa,bbbbbb,,ccccc,dddd1,eeee,ffffff,gggggggg1/2,ggggggg1/1,aA,2012/04/10 02:53:19,27555,1,8450,80,0,0,0x200000,hhh,jjjjjjj,\"www.somesite.com/ajax/home.php/Pane?__a=1&data={\"\"pid\"\":34,\"\"data\"\":[\"\"a.163624624.35636.13135\"\",true,false]}&__user=6625141\",(9999),yetuth-atrat,info,client-to-server,0,0x0,192.168.0.0-192.168.255.255,Some Country,0,application/x-javascript"
    );

    LOG_DEBUG("Input data:\n" << testData << '\n');

    ml::core::CDelimiter delimiter(",");
    delimiter.quote('"', '"');

    ml::core::CStringUtils::TStrVec delimited;
    std::string remainder;

    delimiter.tokenise(testData, false, delimited, remainder);

    delimited.push_back(remainder);

    std::ostringstream strm;
    std::copy(delimited.begin(), delimited.end(), TStrOStreamItr(strm, "\n"));
    LOG_DEBUG("Quoted output data:\nNumber of lines = " << delimited.size() << "\nLines are:\n" << strm.str());

    // 42 fields - in particular, the JSON data at index 31 in the vector should
    // still contain commas and double quotes
    CPPUNIT_ASSERT_EQUAL(size_t(42), delimited.size());
    CPPUNIT_ASSERT(delimited[31].find(',') != std::string::npos);
    CPPUNIT_ASSERT(delimited[31].find('"') != std::string::npos);
}
