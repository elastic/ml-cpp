/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "../CCommandParser.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CCommandParserTest)

BOOST_AUTO_TEST_CASE(testParsingStream) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [1, 2, 3]}"
    "{\"request_id\": \"bar\", \"tokens\": [4, 5]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};    
    BOOST_TEST_REQUIRE(
        processor.ioLoop([&parsed](const ml::torch::CCommandParser::SRequest& request){
            parsed.push_back(request); 
            return true;
    } ));

    BOOST_REQUIRE_EQUAL(2, parsed.size());    
    {
        BOOST_REQUIRE_EQUAL("foo", parsed[0].s_RequestId);
        ml::torch::CCommandParser::TUint32Vec expected{1, 2, 3};
        BOOST_REQUIRE_EQUAL_COLLECTIONS(parsed[0].s_Tokens.begin(), parsed[0].s_Tokens.end(), 
            expected.begin(), expected.end());
    }
    {
        BOOST_REQUIRE_EQUAL("bar", parsed[1].s_RequestId);    
        ml::torch::CCommandParser::TUint32Vec expected{4, 5};
        BOOST_REQUIRE_EQUAL_COLLECTIONS(parsed[1].s_Tokens.begin(), parsed[1].s_Tokens.end(), 
            expected.begin(), expected.end());
    }
}

BOOST_AUTO_TEST_CASE(testParsingInvalidDoc) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"foo\": 1, }"};
    
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};    
    BOOST_TEST_REQUIRE(
        processor.ioLoop([&parsed](const ml::torch::CCommandParser::SRequest& request){
            parsed.push_back(request); 
            return true;
    }) == false);

    BOOST_REQUIRE_EQUAL(0, parsed.size());
}

BOOST_AUTO_TEST_CASE(testParsingWhitespaceSeparatedDocs) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [1, 2, 3]}\t"
        "{\"request_id\": \"bar\", \"tokens\": [1, 2, 3]}\n"
        "{\"request_id\": \"foo2\", \"tokens\": [1, 2, 3]} "
        "{\"request_id\": \"bar2\", \"tokens\": [1, 2, 3]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};    
    BOOST_TEST_REQUIRE(
        processor.ioLoop([&parsed](const ml::torch::CCommandParser::SRequest& request){
            parsed.push_back(request); 
            return true;
    } ));

    BOOST_REQUIRE_EQUAL(4, parsed.size());
    BOOST_REQUIRE_EQUAL("foo", parsed[0].s_RequestId);
    BOOST_REQUIRE_EQUAL("bar", parsed[1].s_RequestId);
    BOOST_REQUIRE_EQUAL("foo2", parsed[2].s_RequestId);
    BOOST_REQUIRE_EQUAL("bar2", parsed[3].s_RequestId);    
}

BOOST_AUTO_TEST_CASE(testParsingVariableArguments) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [1, 2], \"arg_1\": [0, 0], \"arg_2\": [0, 1]}"
                            "{\"request_id\": \"bar\", \"tokens\": [3, 4], \"arg_1\": [1, 0], \"arg_2\": [1, 1]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};    
    BOOST_TEST_REQUIRE(
        processor.ioLoop([&parsed](const ml::torch::CCommandParser::SRequest& request){
            parsed.push_back(request); 
            return true;
    } ));

    BOOST_REQUIRE_EQUAL(2, parsed.size());
    {    
        ml::torch::CCommandParser::TUint32Vec expectedArg1{0, 0};
        ml::torch::CCommandParser::TUint32Vec expectedArg2{0, 1};

        ml::torch::CCommandParser::TUint32VecVec extraArgs = parsed[0].s_SecondaryArguments;
        BOOST_REQUIRE_EQUAL(2, extraArgs.size());

        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[0].begin(), extraArgs[0].end(), 
            expectedArg1.begin(), expectedArg1.end());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[1].begin(), extraArgs[1].end(), 
            expectedArg2.begin(), expectedArg2.end());        
    }        
    {        
        ml::torch::CCommandParser::TUint32Vec expectedArg1{1, 0};
        ml::torch::CCommandParser::TUint32Vec expectedArg2{1, 1};

        ml::torch::CCommandParser::TUint32VecVec extraArgs = parsed[1].s_SecondaryArguments;
        BOOST_REQUIRE_EQUAL(2, extraArgs.size());

        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[0].begin(), extraArgs[0].end(), 
            expectedArg1.begin(), expectedArg1.end());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[1].begin(), extraArgs[1].end(), 
            expectedArg2.begin(), expectedArg2.end());        
    }        
}

BOOST_AUTO_TEST_CASE(testParsingInvalidVarArg) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [1, 2], \"arg_1\": \"not_an_array\"}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};    
    BOOST_TEST_REQUIRE(
        processor.ioLoop([&parsed](const ml::torch::CCommandParser::SRequest& request){
            parsed.push_back(request); 
            return true;
    } ));

    BOOST_REQUIRE_EQUAL(0, parsed.size());
}

BOOST_AUTO_TEST_SUITE_END()
