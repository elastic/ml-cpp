/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRegex.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CRegexTest)

BOOST_AUTO_TEST_CASE(testInit) {
    {
        std::string regexStr = "[[:digit: ] )";

        ml::core::CRegex regex;

        BOOST_TEST(!regex.init(regexStr));
    }
    {
        std::string regexStr = "*[[:digit:]]a*[a-z]";

        ml::core::CRegex regex;

        BOOST_TEST(!regex.init(regexStr));
    }
    {
        std::string regexStr = "[[:digit:]]a*[a-z]";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(regexStr, regex.str());
    }
    {
        // Test init twice
        std::string regexStr1 = "\\d+";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr1));
        BOOST_TEST(regex.matches("6371"));
        BOOST_TEST(!regex.matches("test"));

        std::string regexStr2 = "\\D+";

        BOOST_TEST(regex.init(regexStr2));
        BOOST_TEST(!regex.matches("6371"));
        BOOST_TEST(regex.matches("test"));
    }
    {
        std::string regexStr = "<.*";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(regexStr, regex.str());
        BOOST_TEST(regex.matches("<Jan 19, 2011 1:58:42 PM EST> <Notice> <WebLogicServer> <BEA-000365> <Server state changed to STARTING>"));
    }
    {
        // Uninitialised
        std::string regexStr = "<.*";
        ml::core::CRegex regex;
        BOOST_TEST(!regex.matches("<Jan 19, 2011 1:58:42 PM EST> <Notice> <WebLogicServer> <BEA-000365> <Server state changed to STARTING>"));
    }
}

BOOST_AUTO_TEST_CASE(testSearch) {
    {
        // Uninitialised
        ml::core::CRegex regex;
        BOOST_CHECK_EQUAL(std::string(""), regex.str());
    }
    {
        // Uninitialised
        ml::core::CRegex regex;
        size_t position(0);
        size_t length(0);

        BOOST_TEST(!regex.search("671", position, length));
    }
    {
        std::string regexStr = "\\d+";

        ml::core::CRegex regex;
        size_t position(0);
        size_t length(0);

        BOOST_TEST(regex.init(regexStr));
        BOOST_TEST(regex.search("671", position, length));
        BOOST_CHECK_EQUAL(size_t(0), position);
        BOOST_CHECK_EQUAL(size_t(3), length);
        BOOST_TEST(regex.search("abc 76371", position, length));
        BOOST_CHECK_EQUAL(size_t(4), position);
        BOOST_CHECK_EQUAL(size_t(5), length);
        BOOST_TEST(regex.search("68 abc", position, length));
        BOOST_CHECK_EQUAL(size_t(0), position);
        BOOST_CHECK_EQUAL(size_t(2), length);
        BOOST_TEST(regex.search("abc 6371 def", position, length));
        BOOST_CHECK_EQUAL(size_t(4), position);
        BOOST_CHECK_EQUAL(size_t(4), length);
        BOOST_TEST(!regex.search("test", position, length));
    }

    {
        std::string regexStr = "(\\d+\\s+\\w+\\s+\\d+\\s+\\d+:\\d+:\\d+,\\d+)";

        ml::core::CRegex regex;
        size_t position(0);

        BOOST_TEST(regex.init(regexStr));
        BOOST_TEST(regex.search("03 Nov 2009 09:22:58,289", position));
        BOOST_CHECK_EQUAL(size_t(0), position);
        BOOST_TEST(regex.search("abc 03 Nov 2009 09:22:58,289", position));
        BOOST_CHECK_EQUAL(size_t(4), position);
        BOOST_TEST(regex.search("03 Nov 2009 09:22:58,289 abc", position));
        BOOST_CHECK_EQUAL(size_t(0), position);
        BOOST_TEST(regex.search("abc 03 Nov 2009 09:22:58,289 def", position));
        BOOST_CHECK_EQUAL(size_t(4), position);
        BOOST_TEST(!regex.search("test", position));
    }
}

BOOST_AUTO_TEST_CASE(testTokenise1) {
    std::string str1("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:53:44 2008> Transport node error on node 0x1234<END>");
    std::string str2("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:30:14 2008> Transport read error (8) on node 0x1235<END>");

    {
        // Uninitialised
        std::string regexStr;
        regexStr += "((.+?) )+";
        ml::core::CRegex regex;
        ml::core::CRegex::TStrVec tokens;
        BOOST_TEST(!regex.tokenise(str1, tokens));
    }
    {
        // An invalid regex
        std::string regexStr;

        regexStr += "((.+?) )+";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        BOOST_TEST(!regex.matches(str1));
        BOOST_TEST(!regex.tokenise(str1, tokens));
    }
    {
        std::string regexStr;

        regexStr += "^<(.+?):";
        regexStr += "\\s*(\\w+):";
        regexStr += ".+?(\\w+)\\s+(\\w+)\\s+(\\d+)\\s+(\\d+:\\d+:\\d+)\\s+(\\d+)";
        regexStr += ">\\s+(?:Transport node error)";
        regexStr += ".+?node\\s+(0x\\d+|\\d+)";
        regexStr += ".*$";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        BOOST_TEST(regex.matches(str1));
        BOOST_TEST(regex.tokenise(str1, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }

        BOOST_TEST(!regex.matches(str2));
        BOOST_TEST(!regex.tokenise(str2, tokens));
    }
    {
        std::string regexStr;

        regexStr += "^<(.+?):";
        regexStr += "\\s*(\\w+):";
        regexStr += ".+?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+(\\d+)\\s+(\\d+:\\d+:\\d+)\\s+(\\d+)";
        regexStr += ">\\s+(?:Transport read error)";
        regexStr += ".+?node\\s+(0x\\d+|\\d+)";
        regexStr += ".*$";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        BOOST_TEST(regex.matches(str2));
        BOOST_TEST(regex.tokenise(str2, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }

        BOOST_TEST(!regex.matches(str1));
        BOOST_TEST(!regex.tokenise(str1, tokens));
    }

    std::string str3("Sep 10, 2009 3:54:12 AM org.apache.tomcat.util.http.Parameters processParameters\r\nWARNING: Parameters: Invalid chunk ignored.");

    {
        std::string regexStr("(\\w+\\s+\\d+,\\s+\\d+\\s+\\d+:\\d+:\\d+\\s+\\w+)\\s*([[:alnum:].]+)\\s*(\\w+)\\r?\\n(INFO|WARNING|SEVERE|"
                             "DEBUG|FATAL): Parameters: Invalid chunk ignored\\.\\s*");

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        BOOST_TEST(regex.matches(str3));
        BOOST_TEST(regex.tokenise(str3, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }
    }

    std::string str4("dataview[(@name=\"Snoozed\")]/rows/row[(@name=\"796480523\")]/cell[(@column=\"managedEntity\")]");

    {
        std::string regexStr(".*dataview\\[\\(@name=\"(.*)\"\\)\\]/rows/row\\[\\(@name=\"(.*)\"\\)\\]/cell\\[\\(@column=\"(.*)\"\\)\\].*");

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        BOOST_TEST(regex.matches(str4));
        BOOST_TEST(regex.tokenise(str4, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }
    }
}

BOOST_AUTO_TEST_CASE(testTokenise2) {
    std::string regexStr("(.+?)(?:\\((.*?)\\))?");

    ml::core::CRegex regex;

    BOOST_TEST(regex.init(regexStr));

    ml::core::CRegex::TStrVec tokens;

    BOOST_TEST(regex.tokenise("count", tokens));
    BOOST_TEST(tokens.size() == 2);
    BOOST_TEST(tokens[0] == "count");
    BOOST_TEST(tokens[1].empty());

    BOOST_TEST(regex.tokenise("count(category)", tokens));
    BOOST_TEST(tokens.size() == 2);
    BOOST_TEST(tokens[0] == "count");
    BOOST_TEST(tokens[1] == "category");

    BOOST_TEST(regex.tokenise("sdcasc asc(sddscv)(sdcsc)", tokens));
    BOOST_TEST(tokens.size() == 2);
    BOOST_TEST(tokens[0] == "sdcasc asc");
    BOOST_TEST(tokens[1] == "sddscv)(sdcsc");

    BOOST_TEST(regex.tokenise("dc(category)", tokens));
    BOOST_TEST(tokens.size() == 2);
    BOOST_TEST(tokens[0] == "dc");
    BOOST_TEST(tokens[1] == "category");

    BOOST_TEST(regex.tokenise("count()", tokens));
    BOOST_TEST(tokens.size() == 2);
    LOG_DEBUG(<< tokens[0] << " " << tokens[1]);
    BOOST_TEST(tokens[0] == "count");
    BOOST_TEST(tokens[1].empty());
}

BOOST_AUTO_TEST_CASE(testSplit) {
    std::string str1("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:53:44 2008> Transport node error on node 0x1234<END>");
    std::string str2("<ml00-4203.1.p2ps: Error: Fri Apr 11  15:30:14 2008> Transport read error (8) on node 0x1235<END>");

    {
        // Uninitialised
        std::string regexStr;
        regexStr += "\\s+";
        ml::core::CRegex regex;
        ml::core::CRegex::TStrVec tokens;
        BOOST_TEST(!regex.split(str1, tokens));
    }
    {
        std::string regexStr;

        regexStr += "\\s+";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        ml::core::CRegex::TStrVec tokens;

        BOOST_TEST(regex.split(str1, tokens));

        for (ml::core::CRegex::TStrVec::iterator itr = tokens.begin();
             itr != tokens.end(); ++itr) {
            LOG_DEBUG(<< "'" << *itr << "'");
        }
    }
}

BOOST_AUTO_TEST_CASE(testEscape) {
    BOOST_CHECK_EQUAL(std::string("\\.\\.\\."), ml::core::CRegex::escapeRegexSpecial("..."));
    BOOST_CHECK_EQUAL(std::string("hello"), ml::core::CRegex::escapeRegexSpecial("hello"));
    BOOST_CHECK_EQUAL(std::string("\\)hello\\(\\n\\^"),
                      ml::core::CRegex::escapeRegexSpecial(")hello(\n^"));
    BOOST_CHECK_EQUAL(std::string("\\)hello\\(\\r?\\n\\^"),
                      ml::core::CRegex::escapeRegexSpecial(")hello(\r\n^"));
}

BOOST_AUTO_TEST_CASE(testLiteralCount) {
    {
        // Uninitialised
        ml::core::CRegex regex;
        BOOST_CHECK_EQUAL(size_t(0), regex.literalCount());
    }
    {
        std::string regexStr = "[[:digit:]]a*[a-z]";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(0), regex.literalCount());
    }
    {
        std::string regexStr = "hello";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(5), regex.literalCount());
    }
    {
        std::string regexStr = "hello.*";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(5), regex.literalCount());
    }
    {
        std::string regexStr = "(hello.*|goodbye.*)my friend";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(14), regex.literalCount());
    }
    {
        std::string regexStr = "number\\s+(\\d+,\\d+\\.\\d+|\\d+\\.\\d+)";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));
        BOOST_CHECK_EQUAL(size_t(7), regex.literalCount());
    }
    {
        std::string regexStr = "(cpu\\d+)";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(3), regex.literalCount());
    }
    {
        std::string regexStr = "ip = (\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(8), regex.literalCount());
    }
    {
        std::string regexStr = "[[:space:][:alpha:]_]+(\\d+)";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(0), regex.literalCount());
    }
    {
        std::string regexStr = "[[:space:][:alpha:]_]+(abc|\\*)";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(1), regex.literalCount());
    }
    {
        std::string regexStr = "[[:space:][:alpha:]_]+(\\d+|\\*)";

        ml::core::CRegex regex;

        BOOST_TEST(regex.init(regexStr));

        BOOST_CHECK_EQUAL(size_t(0), regex.literalCount());
    }
}

BOOST_AUTO_TEST_SUITE_END()
