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

#include "../CCommandParser.h"

#include <boost/test/unit_test.hpp>
#include <string>

namespace {
void unexpectedError(const std::string&, const std::string& message) {
    BOOST_TEST_FAIL(message);
}

bool unexpectedRequest(const ml::torch::CCommandParser::SRequest& request) {
    BOOST_TEST_FAIL("Unexpected request " + request.s_RequestId);
    return true;
}
}

BOOST_AUTO_TEST_SUITE(CCommandParserTest)

BOOST_AUTO_TEST_CASE(testParsingStream) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [[1, 2, 3]]}"
                        "{\"request_id\": \"bar\", \"tokens\": [[4, 5]]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        [&parsed](const ml::torch::CCommandParser::SRequest& request) {
            parsed.push_back(request);
            return true;
        },
        unexpectedError));

    BOOST_REQUIRE_EQUAL(2, parsed.size());
    {
        BOOST_REQUIRE_EQUAL("foo", parsed[0].s_RequestId);
        ml::torch::CCommandParser::TUint64Vec expected{1, 2, 3};
        BOOST_REQUIRE_EQUAL_COLLECTIONS(parsed[0].s_Tokens.begin(),
                                        parsed[0].s_Tokens.end(),
                                        expected.begin(), expected.end());
        BOOST_TEST_REQUIRE(parsed[0].hasTokens());
    }
    {
        BOOST_REQUIRE_EQUAL("bar", parsed[1].s_RequestId);
        ml::torch::CCommandParser::TUint64Vec expected{4, 5};
        BOOST_REQUIRE_EQUAL_COLLECTIONS(parsed[1].s_Tokens.begin(),
                                        parsed[1].s_Tokens.end(),
                                        expected.begin(), expected.end());
        BOOST_TEST_REQUIRE(parsed[0].hasTokens());
    }
}

BOOST_AUTO_TEST_CASE(testParsingInvalidDoc) {

    std::vector<std::string> errors;

    std::string command{"{\"foo\": 1, }"};

    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(
        processor.ioLoop(unexpectedRequest, [&errors](const std::string& id,
                                                      const ::std::string& message) {
            errors.push_back(message);
            BOOST_REQUIRE_EQUAL(ml::torch::CCommandParser::UNKNOWN_ID, id);
        }) == false);

    BOOST_REQUIRE_EQUAL(1, errors.size());
}

BOOST_AUTO_TEST_CASE(testParsingInvalidRequestId) {

    std::vector<std::string> errors;

    std::string command{"{\"request_id\": 1}"};

    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        unexpectedRequest, [&errors](const std::string& id, const ::std::string& message) {
            BOOST_REQUIRE_EQUAL(ml::torch::CCommandParser::UNKNOWN_ID, id);
            errors.push_back(message);
        }));

    BOOST_REQUIRE_EQUAL(1, errors.size());
}

BOOST_AUTO_TEST_CASE(testParsingTokenArrayNotInts) {

    std::vector<std::string> errors;

    std::string command{R"({"request_id": "tokens_should_be_uints", "tokens": [["a", "b", "c"]]})"};

    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        unexpectedRequest, [&errors](const std::string& id, const ::std::string& message) {
            BOOST_REQUIRE_EQUAL(id, "tokens_should_be_uints");
            errors.push_back(message);
        }));

    BOOST_REQUIRE_EQUAL(1, errors.size());
}

BOOST_AUTO_TEST_CASE(testParsingTokenVarArgsNotInts) {

    std::vector<std::string> errors;

    std::string command{R"({"request_id": "bad", "tokens": [[1, 2]], "arg_1": [["a", "b"]]})"};

    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        unexpectedRequest, [&errors](const std::string& id, const ::std::string& message) {
            BOOST_REQUIRE_EQUAL("bad", id);
            errors.push_back(message);
        }));

    BOOST_REQUIRE_EQUAL(1, errors.size());
}

BOOST_AUTO_TEST_CASE(testParsingWhitespaceSeparatedDocs) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [[1, 2, 3]]}\t"
                        "{\"request_id\": \"bar\", \"tokens\": [[1, 2, 3]]}\n"
                        "{\"request_id\": \"foo2\", \"tokens\": [[1, 2, 3]]} "
                        "{\"request_id\": \"bar2\", \"tokens\": [[1, 2, 3]]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        [&parsed](const ml::torch::CCommandParser::SRequest& request) {
            parsed.push_back(request);
            return true;
        },
        unexpectedError));

    BOOST_REQUIRE_EQUAL(4, parsed.size());
    BOOST_REQUIRE_EQUAL("foo", parsed[0].s_RequestId);
    BOOST_REQUIRE_EQUAL("bar", parsed[1].s_RequestId);
    BOOST_REQUIRE_EQUAL("foo2", parsed[2].s_RequestId);
    BOOST_REQUIRE_EQUAL("bar2", parsed[3].s_RequestId);
}

BOOST_AUTO_TEST_CASE(testParsingVariableArguments) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{
        "{\"request_id\": \"foo\", \"tokens\": [[1, 2]], \"arg_1\": [[0, 0]], \"arg_2\": [[0, 1]]}"
        "{\"request_id\": \"bar\", \"tokens\": [[3, 4]], \"arg_1\": [[1, 0]], \"arg_2\": [[1, 1]]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        [&parsed](const ml::torch::CCommandParser::SRequest& request) {
            parsed.push_back(request);
            return true;
        },
        unexpectedError));

    BOOST_REQUIRE_EQUAL(2, parsed.size());
    {
        ml::torch::CCommandParser::TUint64Vec expectedArg1{0, 0};
        ml::torch::CCommandParser::TUint64Vec expectedArg2{0, 1};

        ml::torch::CCommandParser::TUint64VecVec extraArgs = parsed[0].s_SecondaryArguments;
        BOOST_REQUIRE_EQUAL(2, extraArgs.size());

        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[0].begin(), extraArgs[0].end(),
                                        expectedArg1.begin(), expectedArg1.end());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[1].begin(), extraArgs[1].end(),
                                        expectedArg2.begin(), expectedArg2.end());
    }
    {
        ml::torch::CCommandParser::TUint64Vec expectedArg1{1, 0};
        ml::torch::CCommandParser::TUint64Vec expectedArg2{1, 1};

        ml::torch::CCommandParser::TUint64VecVec extraArgs = parsed[1].s_SecondaryArguments;
        BOOST_REQUIRE_EQUAL(2, extraArgs.size());

        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[0].begin(), extraArgs[0].end(),
                                        expectedArg1.begin(), expectedArg1.end());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[1].begin(), extraArgs[1].end(),
                                        expectedArg2.begin(), expectedArg2.end());
    }
}

BOOST_AUTO_TEST_CASE(testParsingInvalidVarArg) {

    std::vector<std::string> errors;

    std::string command{R"({"request_id": "foo", "tokens": [[1, 2]], "arg_1": "not_an_array"})"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        unexpectedRequest, [&errors](const std::string& id, const ::std::string& message) {
            BOOST_REQUIRE_EQUAL("foo", id);
            errors.push_back(message);
        }));

    BOOST_REQUIRE_EQUAL(1, errors.size());
}

BOOST_AUTO_TEST_CASE(testRequestHandlerExitsLoop) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{"{\"request_id\": \"foo\", \"tokens\": [[1, 2, 3]]}"
                        "{\"request_id\": \"bar\", \"tokens\": [[4, 5]]}"};
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    // handler returns false
    BOOST_TEST_REQUIRE(false == processor.ioLoop(
                                    [&parsed](const ml::torch::CCommandParser::SRequest& request) {
                                        parsed.push_back(request);
                                        return false;
                                    },
                                    unexpectedError));

    // ioloop should exit after the first call to the handler
    BOOST_REQUIRE_EQUAL(1, parsed.size());
}

BOOST_AUTO_TEST_CASE(testParsingBatch) {

    std::vector<ml::torch::CCommandParser::SRequest> parsed;

    std::string command{
        R"({"request_id": "foo", "tokens": [[1, 2], [3, 4], [5, 6]], "arg_1": [[0, 0], [0, 1], [0, 2]], "arg_2": [[1, 0], [1, 1], [1, 2]]}
        {"request_id": "bar", "tokens": [[1, 2], [3, 4]], "arg_1": [[0, 0], [0, 1]], "arg_2": [[1, 0], [1, 1]]}"})"
    };
    std::istringstream commandStream{command};

    ml::torch::CCommandParser processor{commandStream};
    BOOST_TEST_REQUIRE(processor.ioLoop(
        [&parsed](const ml::torch::CCommandParser::SRequest& request) {
            parsed.push_back(request);
            return true;
        },
        unexpectedError));

    BOOST_REQUIRE_EQUAL(2, parsed.size());
    {
        ml::torch::CCommandParser::TUint64Vec expectedTokens{1, 2, 3, 4, 5, 6};
        ml::torch::CCommandParser::TUint64Vec expectedArg1{0, 0, 0, 1, 0, 2};
        ml::torch::CCommandParser::TUint64Vec expectedArg2{1, 0, 1, 1, 1, 2};      

        BOOST_REQUIRE_EQUAL_COLLECTIONS(parsed[0].s_Tokens.begin(),
                                        parsed[0].s_Tokens.end(),
                                        expectedTokens.begin(), expectedTokens.end());

        ml::torch::CCommandParser::TUint64VecVec extraArgs = parsed[0].s_SecondaryArguments;
        BOOST_REQUIRE_EQUAL(2, extraArgs.size());

        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[0].begin(), extraArgs[0].end(),
                                        expectedArg1.begin(), expectedArg1.end());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[1].begin(), extraArgs[1].end(),
                                        expectedArg2.begin(), expectedArg2.end());
    }
    {
        ml::torch::CCommandParser::TUint64Vec expectedTokens{1, 2, 3, 4};
        ml::torch::CCommandParser::TUint64Vec expectedArg1{0, 0, 0, 1};
        ml::torch::CCommandParser::TUint64Vec expectedArg2{1, 0, 1, 1};      

        BOOST_REQUIRE_EQUAL_COLLECTIONS(parsed[1].s_Tokens.begin(),
                                        parsed[1].s_Tokens.end(),
                                        expectedTokens.begin(), expectedTokens.end());

        ml::torch::CCommandParser::TUint64VecVec extraArgs = parsed[1].s_SecondaryArguments;
        BOOST_REQUIRE_EQUAL(2, extraArgs.size());

        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[0].begin(), extraArgs[0].end(),
                                        expectedArg1.begin(), expectedArg1.end());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(extraArgs[1].begin(), extraArgs[1].end(),
                                        expectedArg2.begin(), expectedArg2.end());
    }
}

BOOST_AUTO_TEST_SUITE_END()
