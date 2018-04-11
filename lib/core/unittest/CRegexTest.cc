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
#include "CRegexTest.h"

#include <core/CLogger.h>
#include <core/CRegex.h>

CppUnit::Test* CRegexTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CRegexTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testInit", &CRegexTest::testInit));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testSearch", &CRegexTest::testSearch));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testSplit", &CRegexTest::testSplit));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testTokenise1", &CRegexTest::testTokenise1));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testTokenise2", &CRegexTest::testTokenise2));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testEscape", &CRegexTest::testEscape));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexTest>(
        "CRegexTest::testLiteralCount", &CRegexTest::testLiteralCount));

    return suiteOfTests;
}

void CRegexTest::testInit() {
    {
        std::string regexStr = "[[:digit: ] )";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(!regex.init(regexStr));
    }
    {
        std::string regexStr = "*[[:digit:]]a*[a-z]";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(!regex.init(regexStr));
    }
    {
        std::string regexStr = "[[:digit:]]a*[a-z]";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(regexStr, regex.str());
    }
    {
        // Test init twice
        std::string regexStr1 = "\\d+";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr1));
        CPPUNIT_ASSERT(regex.matches("6371"));
        CPPUNIT_ASSERT(!regex.matches("test"));

        std::string regexStr2 = "\\D+";

        CPPUNIT_ASSERT(regex.init(regexStr2));
        CPPUNIT_ASSERT(!regex.matches("6371"));
        CPPUNIT_ASSERT(regex.matches("test"));
    }
    {
        std::string regexStr = "<.*";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(regexStr, regex.str());
        CPPUNIT_ASSERT(regex.matches("<Jan 19, 2011 1:58:42 PM EST> <Notice> "
                                     "<WebLogicServer> <BEA-000365> <Server "
                                     "state changed to STARTING>"));
    }
    {
        // Uninitialised
        std::string regexStr = "<.*";
        ml::core::CRegex regex;
        CPPUNIT_ASSERT(!regex.matches("<Jan 19, 2011 1:58:42 PM EST> <Notice> "
                                      "<WebLogicServer> <BEA-000365> <Server "
                                      "state changed to STARTING>"));
    }
}

void CRegexTest::testSearch() {
    {
        // Uninitialised
        ml::core::CRegex regex;
        CPPUNIT_ASSERT_EQUAL(std::string(""), regex.str());
    }
    {
        // Uninitialised
        ml::core::CRegex regex;
        size_t position(0);
        size_t length(0);

        CPPUNIT_ASSERT(!regex.search("671", position, length));
    }
    {
        std::string regexStr = "\\d+";

        ml::core::CRegex regex;
        size_t position(0);
        size_t length(0);

        CPPUNIT_ASSERT(regex.init(regexStr));
        CPPUNIT_ASSERT(regex.search("671", position, length));
        CPPUNIT_ASSERT_EQUAL(size_t(0), position);
        CPPUNIT_ASSERT_EQUAL(size_t(3), length);
        CPPUNIT_ASSERT(regex.search("abc 76371", position, length));
        CPPUNIT_ASSERT_EQUAL(size_t(4), position);
        CPPUNIT_ASSERT_EQUAL(size_t(5), length);
        CPPUNIT_ASSERT(regex.search("68 abc", position, length));
        CPPUNIT_ASSERT_EQUAL(size_t(0), position);
        CPPUNIT_ASSERT_EQUAL(size_t(2), length);
        CPPUNIT_ASSERT(regex.search("abc 6371 def", position, length));
        CPPUNIT_ASSERT_EQUAL(size_t(4), position);
        CPPUNIT_ASSERT_EQUAL(size_t(4), length);
        CPPUNIT_ASSERT(!regex.search("test", position, length));
    }

    {
        std::string regexStr = "(\\d+\\s+\\w+\\s+\\d+\\s+\\d+:\\d+:\\d+,\\d+)";

        ml::core::CRegex regex;
        size_t position(0);

        CPPUNIT_ASSERT(regex.init(regexStr));
        CPPUNIT_ASSERT(regex.search("03 Nov 2009 09:22:58,289", position));
        CPPUNIT_ASSERT_EQUAL(size_t(0), position);
        CPPUNIT_ASSERT(regex.search("abc 03 Nov 2009 09:22:58,289", position));
        CPPUNIT_ASSERT_EQUAL(size_t(4), position);
        CPPUNIT_ASSERT(regex.search("03 Nov 2009 09:22:58,289 abc", position));
        CPPUNIT_ASSERT_EQUAL(size_t(0), position);
        CPPUNIT_ASSERT(regex.search("abc 03 Nov 2009 09:22:58,289 def", position));
        CPPUNIT_ASSERT_EQUAL(size_t(4), position);
        CPPUNIT_ASSERT(!regex.search("test", position));
    }
}

void CRegexTest::testTokenise1() {
    std::string str1("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:53:44 2008> "
                     "Transport node error on node 0x1234<END>");
    std::string str2("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:30:14 2008> "
                     "Transport read error (8) on node 0x1235<END>");

    {
        // Uninitialised
        std::string regexStr;
        regexStr += "((.+?) )+";
        ml::core::CRegex regex;
        ml::core::CRegex::TStrVec tokens;
        CPPUNIT_ASSERT(!regex.tokenise(str1, tokens));
    }
    {
        // An invalid regex
        std::string regexStr;

        regexStr += "((.+?) )+";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        CPPUNIT_ASSERT(!regex.matches(str1));
        CPPUNIT_ASSERT(!regex.tokenise(str1, tokens));
    }
    {
        std::string regexStr;

        regexStr += "^<(.+?):";
        regexStr += "\\s*(\\w+):";
        regexStr +=
            ".+?(\\w+)\\s+(\\w+)\\s+(\\d+)\\s+(\\d+:\\d+:\\d+)\\s+(\\d+)";
        regexStr += ">\\s+(?:Transport node error)";
        regexStr += ".+?node\\s+(0x\\d+|\\d+)";
        regexStr += ".*$";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        CPPUNIT_ASSERT(regex.matches(str1));
        CPPUNIT_ASSERT(regex.tokenise(str1, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }

        CPPUNIT_ASSERT(!regex.matches(str2));
        CPPUNIT_ASSERT(!regex.tokenise(str2, tokens));
    }
    {
        std::string regexStr;

        regexStr += "^<(.+?):";
        regexStr += "\\s*(\\w+):";
        regexStr += ".+?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+("
                    "\\d+)\\s+(\\d+:\\d+:\\d+)\\s+(\\d+)";
        regexStr += ">\\s+(?:Transport read error)";
        regexStr += ".+?node\\s+(0x\\d+|\\d+)";
        regexStr += ".*$";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        CPPUNIT_ASSERT(regex.matches(str2));
        CPPUNIT_ASSERT(regex.tokenise(str2, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }

        CPPUNIT_ASSERT(!regex.matches(str1));
        CPPUNIT_ASSERT(!regex.tokenise(str1, tokens));
    }

    std::string str3(
        "Sep 10, 2009 3:54:12 AM org.apache.tomcat.util.http.Parameters "
        "processParameters\r\nWARNING: Parameters: Invalid chunk ignored.");

    {
        std::string regexStr(
            "(\\w+\\s+\\d+,\\s+\\d+\\s+\\d+:\\d+:\\d+\\s+\\w+)\\s*([[:alnum:].]"
            "+)\\s*(\\w+)\\r?\\n(INFO|WARNING|SEVERE|"
            "DEBUG|FATAL): Parameters: Invalid chunk ignored\\.\\s*");

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        CPPUNIT_ASSERT(regex.matches(str3));
        CPPUNIT_ASSERT(regex.tokenise(str3, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }
    }

    std::string str4("dataview[(@name=\"Snoozed\")]/rows/"
                     "row[(@name=\"796480523\")]/"
                     "cell[(@column=\"managedEntity\")]");

    {
        std::string regexStr(".*dataview\\[\\(@name=\"(.*)\"\\)\\]/rows/"
                             "row\\[\\(@name=\"(.*)\"\\)\\]/"
                             "cell\\[\\(@column=\"(.*)\"\\)\\].*");

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        CPPUNIT_ASSERT(regex.matches(str4));
        CPPUNIT_ASSERT(regex.tokenise(str4, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }
    }
}

void CRegexTest::testTokenise2() {
    std::string regexStr("(.+?)(?:\\((.*?)\\))?");

    ml::core::CRegex regex;

    CPPUNIT_ASSERT(regex.init(regexStr));

    ml::core::CRegex::TStrVec tokens;

    CPPUNIT_ASSERT(regex.tokenise("count", tokens));
    CPPUNIT_ASSERT(tokens.size() == 2);
    CPPUNIT_ASSERT(tokens[0] == "count");
    CPPUNIT_ASSERT(tokens[1].empty());

    CPPUNIT_ASSERT(regex.tokenise("count(category)", tokens));
    CPPUNIT_ASSERT(tokens.size() == 2);
    CPPUNIT_ASSERT(tokens[0] == "count");
    CPPUNIT_ASSERT(tokens[1] == "category");

    CPPUNIT_ASSERT(regex.tokenise("sdcasc asc(sddscv)(sdcsc)", tokens));
    CPPUNIT_ASSERT(tokens.size() == 2);
    CPPUNIT_ASSERT(tokens[0] == "sdcasc asc");
    CPPUNIT_ASSERT(tokens[1] == "sddscv)(sdcsc");

    CPPUNIT_ASSERT(regex.tokenise("dc(category)", tokens));
    CPPUNIT_ASSERT(tokens.size() == 2);
    CPPUNIT_ASSERT(tokens[0] == "dc");
    CPPUNIT_ASSERT(tokens[1] == "category");

    CPPUNIT_ASSERT(regex.tokenise("count()", tokens));
    CPPUNIT_ASSERT(tokens.size() == 2);
    LOG_DEBUG(<< tokens[0] << " " << tokens[1]);
    CPPUNIT_ASSERT(tokens[0] == "count");
    CPPUNIT_ASSERT(tokens[1].empty());
}

void CRegexTest::testSplit() {
    std::string str1("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:53:44 2008> "
                     "Transport node error on node 0x1234<END>");
    std::string str2("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:30:14 2008> "
                     "Transport read error (8) on node 0x1235<END>");

    {
        // Uninitialised
        std::string regexStr;
        regexStr += "\\s+";
        ml::core::CRegex regex;
        ml::core::CRegex::TStrVec tokens;
        CPPUNIT_ASSERT(!regex.split(str1, tokens));
    }
    {
        std::string regexStr;

        regexStr += "\\s+";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        CPPUNIT_ASSERT(regex.split(str1, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }
    }
}

void CRegexTest::testEscape() {
    CPPUNIT_ASSERT_EQUAL(std::string("\\.\\.\\."),
                         ml::core::CRegex::escapeRegexSpecial("..."));
    CPPUNIT_ASSERT_EQUAL(std::string("hello"),
                         ml::core::CRegex::escapeRegexSpecial("hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("\\)hello\\(\\n\\^"),
                         ml::core::CRegex::escapeRegexSpecial(")hello(\n^"));
    CPPUNIT_ASSERT_EQUAL(std::string("\\)hello\\(\\r?\\n\\^"),
                         ml::core::CRegex::escapeRegexSpecial(")hello(\r\n^"));
}

void CRegexTest::testLiteralCount() {
    {
        // Uninitialised
        ml::core::CRegex regex;
        CPPUNIT_ASSERT_EQUAL(size_t(0), regex.literalCount());
    }
    {
        std::string regexStr = "[[:digit:]]a*[a-z]";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(0), regex.literalCount());
    }
    {
        std::string regexStr = "hello";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(5), regex.literalCount());
    }
    {
        std::string regexStr = "hello.*";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(5), regex.literalCount());
    }
    {
        std::string regexStr = "(hello.*|goodbye.*)my friend";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(14), regex.literalCount());
    }
    {
        std::string regexStr = "number\\s+(\\d+,\\d+\\.\\d+|\\d+\\.\\d+)";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));
        CPPUNIT_ASSERT_EQUAL(size_t(7), regex.literalCount());
    }
    {
        std::string regexStr = "(cpu\\d+)";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(3), regex.literalCount());
    }
    {
        std::string regexStr =
            "ip = (\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(8), regex.literalCount());
    }
    {
        std::string regexStr = "[[:space:][:alpha:]_]+(\\d+)";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(0), regex.literalCount());
    }
    {
        std::string regexStr = "[[:space:][:alpha:]_]+(abc|\\*)";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(1), regex.literalCount());
    }
    {
        std::string regexStr = "[[:space:][:alpha:]_]+(\\d+|\\*)";

        ml::core::CRegex regex;

        CPPUNIT_ASSERT(regex.init(regexStr));

        CPPUNIT_ASSERT_EQUAL(size_t(0), regex.literalCount());
    }
}
