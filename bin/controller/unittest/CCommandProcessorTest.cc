/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CProcess.h>
#include <core/CSleep.h>
#include <core/CStringUtils.h>

#include "../CCommandProcessor.h"

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>
#include <string>

#include <stdio.h>

BOOST_AUTO_TEST_SUITE(CCommandProcessorTest)

namespace {
const std::string OUTPUT_FILE("slogan1.txt");
#ifdef Windows
// Unlike Windows NT system calls, copy's command line cannot cope with
// forward slash path separators
const std::string INPUT_FILE1("testfiles\\slogan1.txt");
const std::string INPUT_FILE2("testfiles\\slogan2.txt");
const char* winDir(::getenv("windir"));
const std::string PROCESS_PATH(winDir != 0 ? std::string(winDir) + "\\System32\\cmd"
                                           : std::string("C:\\Windows\\System32\\cmd"));
const std::string PROCESS_ARGS1[] = {"/C", "copy " + INPUT_FILE1 + " ."};
const std::string PROCESS_ARGS2[] = {"/C", "del " + INPUT_FILE2};
#else
const std::string INPUT_FILE1("testfiles/slogan1.txt");
const std::string INPUT_FILE2("testfiles/slogan2.txt");
const std::string PROCESS_PATH("/bin/sh");
const std::string PROCESS_ARGS1[] = {"-c", "cp " + INPUT_FILE1 + " ."};
const std::string PROCESS_ARGS2[] = {"-c", "rm " + INPUT_FILE2};
#endif
const std::string SLOGAN1("Elastic is great!");
const std::string SLOGAN2("You know, for search!");
}

BOOST_AUTO_TEST_CASE(testStartPermitted) {
    // Remove any output file left behind by a previous failed test, but don't
    // check the return code as this will usually fail
    ::remove(OUTPUT_FILE.c_str());

    ml::controller::CCommandProcessor::TStrVec permittedPaths(1, PROCESS_PATH);
    ml::controller::CCommandProcessor processor(permittedPaths);

    std::string command(ml::controller::CCommandProcessor::START);
    command += '\t';
    command += PROCESS_PATH;
    for (size_t index = 0; index < boost::size(PROCESS_ARGS1); ++index) {
        command += '\t';
        command += PROCESS_ARGS1[index];
    }

    std::istringstream commandStream(command + '\n');
    processor.processCommands(commandStream);

    // Expect the copy to complete in less than 1 second
    ml::core::CSleep::sleep(1000);

    std::ifstream ifs(OUTPUT_FILE.c_str());
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();

    BOOST_REQUIRE_EQUAL(SLOGAN1, content);

    BOOST_REQUIRE_EQUAL(0, ::remove(OUTPUT_FILE.c_str()));
}

BOOST_AUTO_TEST_CASE(testStartNonPermitted) {
    ml::controller::CCommandProcessor::TStrVec permittedPaths(1, "some other process");
    ml::controller::CCommandProcessor processor(permittedPaths);

    std::string command(ml::controller::CCommandProcessor::START);
    command += '\t';
    command += PROCESS_PATH;
    for (size_t index = 0; index < boost::size(PROCESS_ARGS2); ++index) {
        command += '\t';
        command += PROCESS_ARGS2[index];
    }

    std::istringstream commandStream(command + '\n');
    processor.processCommands(commandStream);

    // The delete should have been rejected, so the second input file should
    // still exist and have the expected contents

    std::ifstream ifs(INPUT_FILE2.c_str());
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();

    BOOST_REQUIRE_EQUAL(SLOGAN2, content);
}

BOOST_AUTO_TEST_CASE(testStartNonExistent) {
    ml::controller::CCommandProcessor::TStrVec permittedPaths(1, "some other process");
    ml::controller::CCommandProcessor processor(permittedPaths);

    std::string command(ml::controller::CCommandProcessor::START);
    command += "\tsome other process";

    BOOST_TEST_REQUIRE(!processor.handleCommand(command));
}

BOOST_AUTO_TEST_CASE(testKillDisallowed) {
    // Attempt to kill a process that exists but isn't allowed to be killed,
    // namely the unit test program

    ml::controller::CCommandProcessor::TStrVec permittedPaths(1, PROCESS_PATH);
    ml::controller::CCommandProcessor processor(permittedPaths);

    std::string command(ml::controller::CCommandProcessor::KILL);
    command += '\t';
    command +=
        ml::core::CStringUtils::typeToString(ml::core::CProcess::instance().id());

    BOOST_TEST_REQUIRE(!processor.handleCommand(command));
}

BOOST_AUTO_TEST_CASE(testInvalidVerb) {
    ml::controller::CCommandProcessor::TStrVec permittedPaths(1, "some other process");
    ml::controller::CCommandProcessor processor(permittedPaths);

    std::string command("drive");
    command += "\tsome other process";

    BOOST_TEST_REQUIRE(!processor.handleCommand(command));
}

BOOST_AUTO_TEST_SUITE_END()
