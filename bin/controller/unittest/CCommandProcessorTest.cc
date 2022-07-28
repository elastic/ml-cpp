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

#include <core/CProcess.h>
#include <core/CStringUtils.h>

#include "../CCommandProcessor.h"

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

BOOST_AUTO_TEST_SUITE(CCommandProcessorTest)

namespace {
const std::string OUTPUT_FILE{"slogan1.txt"};
#ifdef Windows
// Unlike Windows NT system calls, copy's command line cannot cope with
// forward slash path separators
const std::string INPUT_FILE1{"testfiles\\slogan1.txt"};
const std::string INPUT_FILE2{"testfiles\\slogan2.txt"};
const char* winDir{std::getenv("windir")};
const std::string PROCESS_PATH{winDir != nullptr
                                   ? std::string{winDir} + "\\System32\\cmd"
                                   : std::string{"C:\\Windows\\System32\\cmd"}};
const std::string PROCESS_ARGS1[]{"/C", "copy " + INPUT_FILE1 + " ."};
const std::string PROCESS_ARGS2[]{"/C", "del " + INPUT_FILE2};
#else
const std::string INPUT_FILE1{"testfiles/slogan1.txt"};
const std::string INPUT_FILE2{"testfiles/slogan2.txt"};
const std::string PROCESS_PATH{"/bin/sh"};
const std::string PROCESS_ARGS1[]{"-c", "cp " + INPUT_FILE1 + " ."};
const std::string PROCESS_ARGS2[]{"-c", "rm " + INPUT_FILE2};
#endif
const std::string SLOGAN1{"Elastic is great!"};
const std::string SLOGAN2{"You know, for search!"};
}

BOOST_AUTO_TEST_CASE(testStartPermitted) {
    // Remove any output file left behind by a previous failed test, but don't
    // check the return code as this will usually fail
    std::remove(OUTPUT_FILE.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{"1\t" + ml::controller::CCommandProcessor::START + '\t' + PROCESS_PATH};
        for (std::size_t index = 0; index < std::size(PROCESS_ARGS1); ++index) {
            command += '\t';
            command += PROCESS_ARGS1[index];
        }

        std::istringstream commandStream{command + '\n'};
        processor.processCommands(commandStream);

        // Expect the copy to complete in less than 1 second
        std::this_thread::sleep_for(std::chrono::seconds{1});

        std::ifstream ifs{OUTPUT_FILE};
        BOOST_TEST_REQUIRE(ifs.is_open());
        std::string content;
        std::getline(ifs, content);
        ifs.close();

        BOOST_REQUIRE_EQUAL(SLOGAN1, content);
    }

    std::string jsonEscapedProcessPath{PROCESS_PATH};
    ml::core::CStringUtils::replace("\\", "\\\\", jsonEscapedProcessPath);
    BOOST_REQUIRE_EQUAL("[{\"id\":1,\"success\":true,\"reason\":\"Process '" + jsonEscapedProcessPath +
                            "' started\"}\n"
                            "]",
                        responseStream.str());

    BOOST_REQUIRE_EQUAL(0, std::remove(OUTPUT_FILE.c_str()));
}

BOOST_AUTO_TEST_CASE(testStartNonPermitted) {
    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{"some other process"};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{"2\t" + ml::controller::CCommandProcessor::START + '\t' + PROCESS_PATH};
        for (std::size_t index = 0; index < std::size(PROCESS_ARGS2); ++index) {
            command += '\t';
            command += PROCESS_ARGS2[index];
        }

        std::istringstream commandStream{command + '\n'};
        processor.processCommands(commandStream);

        // The delete should have been rejected, so the second input file should
        // still exist and have the expected contents

        std::ifstream ifs{INPUT_FILE2};
        BOOST_TEST_REQUIRE(ifs.is_open());
        std::string content;
        std::getline(ifs, content);
        ifs.close();

        BOOST_REQUIRE_EQUAL(SLOGAN2, content);
    }

    std::string jsonEscapedProcessPath{PROCESS_PATH};
    ml::core::CStringUtils::replace("\\", "\\\\", jsonEscapedProcessPath);
    BOOST_REQUIRE_EQUAL("[{\"id\":2,\"success\":false,\"reason\":\"Failed to start process '" +
                            jsonEscapedProcessPath +
                            "'\"}\n"
                            "]",
                        responseStream.str());
}

BOOST_AUTO_TEST_CASE(testStartNonExistent) {
    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{"some other process"};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{"3\t" + ml::controller::CCommandProcessor::START + "\tsome other process"};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    BOOST_REQUIRE_EQUAL("[{\"id\":3,\"success\":false,\"reason\":\"Failed to start process 'some other process'\"}\n"
                        "]",
                        responseStream.str());
}

BOOST_AUTO_TEST_CASE(testKillDisallowed) {
    // Attempt to kill a process that exists but isn't allowed to be killed,
    // namely the unit test program
    std::string pidStr{
        ml::core::CStringUtils::typeToString(ml::core::CProcess::instance().id())};

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{"4\t" + ml::controller::CCommandProcessor::KILL + '\t' + pidStr};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    BOOST_REQUIRE_EQUAL("[{\"id\":4,\"success\":false,\"reason\":\"Failed to kill process with PID " +
                            pidStr +
                            "\"}\n"
                            "]",
                        responseStream.str());
}

BOOST_AUTO_TEST_CASE(testInvalidVerb) {
    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{"some other process"};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{"5\tdrive\tsome other process"};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    BOOST_REQUIRE_EQUAL("[{\"id\":5,\"success\":false,\"reason\":\"Did not understand verb 'drive'\"}\n"
                        "]",
                        responseStream.str());
}

BOOST_AUTO_TEST_CASE(testTooFewTokens) {
    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{"some other process"};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{ml::controller::CCommandProcessor::START + "\tsome other process"};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    // It's not possible to respond without an ID
    BOOST_REQUIRE_EQUAL("[]", responseStream.str());
}

BOOST_AUTO_TEST_CASE(testMissingId) {
    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{"some other process"};
        ml::controller::CCommandProcessor processor{permittedPaths, responseStream};

        std::string command{ml::controller::CCommandProcessor::START +
                            "\tsome other process\targ1\targ2"};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    // It's not possible to respond without an ID
    BOOST_REQUIRE_EQUAL("[]", responseStream.str());
}

BOOST_AUTO_TEST_SUITE_END()
