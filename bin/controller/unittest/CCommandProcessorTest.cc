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
#include <core/CProcess.h>
#include <core/CSetEnv.h>
#include <core/CStringUtils.h>
#include <core/CUnSetEnv.h>

#include "../CCommandProcessor.h"

#include <boost/make_shared.hpp>
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

//! Sets ML_SANDBOX2_DEFAULT_ENFORCED for the duration of a scope and
//! restores the (unset) state afterwards. CCommandProcessor reads the
//! variable once in its constructor, so it must be set before the processor
//! under test is constructed.
class CScopedSandbox2DefaultEnforced {
public:
    explicit CScopedSandbox2DefaultEnforced(const char* value) {
        BOOST_REQUIRE_EQUAL(
            0, ml::core::CSetEnv::setEnv("ML_SANDBOX2_DEFAULT_ENFORCED", value, 1));
    }
    ~CScopedSandbox2DefaultEnforced() {
        ml::core::CUnSetEnv::unSetEnv("ML_SANDBOX2_DEFAULT_ENFORCED");
    }
    CScopedSandbox2DefaultEnforced(const CScopedSandbox2DefaultEnforced&) = delete;
    CScopedSandbox2DefaultEnforced& operator=(const CScopedSandbox2DefaultEnforced&) = delete;
};

//! Redirect the logger to a string stream for the duration of \p fn, so a
//! test can assert on the router's H4 sandbox2_launch signal (the same
//! capture style bin/controller/unittest/CProcessSpawnerRouterTest.cc uses).

//! RAII guard ensuring ml::core::CLogger::instance().reset() always runs,
//! even if the captured function throws (e.g. a failed BOOST_REQUIRE*
//! inside it) - without this, an exception mid-fn() would leave the global
//! logger redirected into a stream nobody reads for the rest of the test
//! binary process, causing misleading cascading failures/log loss in later,
//! unrelated tests.
class CScopedLoggerReset {
public:
    ~CScopedLoggerReset() { ml::core::CLogger::instance().reset(); }
};

template<typename FN>
std::string captureLogged(FN&& fn) {
    auto stream = boost::make_shared<std::ostringstream>();
    BOOST_TEST_REQUIRE(ml::core::CLogger::instance().reconfigure(stream));
    CScopedLoggerReset resetOnExit;
    fn();
    return stream->str();
}
}

BOOST_AUTO_TEST_CASE(testStartPermitted) {
    // Remove any output file left behind by a previous failed test, but don't
    // check the return code as this will usually fail
    std::remove(OUTPUT_FILE.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

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
    std::string expected = "[{\"id\":1,\"success\":true,\"reason\":\"Process '" +
                           jsonEscapedProcessPath +
                           "' started\"}\n"
                           "]";
    std::string actual = responseStream.str();
    LOG_INFO(<< "expected: " << expected);
    LOG_INFO(<< "actual  : " << actual);
    BOOST_REQUIRE_EQUAL(expected, actual);

    BOOST_REQUIRE_EQUAL(0, std::remove(OUTPUT_FILE.c_str()));
}

BOOST_AUTO_TEST_CASE(testStartNonPermitted) {
    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{"some other process"};
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

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
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

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
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

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
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

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
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

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
        ml::controller::CCommandProcessor processor{permittedPaths, {}, responseStream};

        std::string command{ml::controller::CCommandProcessor::START +
                            "\tsome other process\targ1\targ2"};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    // It's not possible to respond without an ID
    BOOST_REQUIRE_EQUAL("[]", responseStream.str());
}

namespace {
//! Build a tab-separated "start" command for \p processPath with \p args.
std::string startCommand(std::uint32_t id,
                         const std::string& processPath,
                         const std::vector<std::string>& args) {
    std::string command{ml::core::CStringUtils::typeToString(id) + '\t' +
                        ml::controller::CCommandProcessor::START + '\t' + processPath};
    for (const auto& arg : args) {
        command += '\t';
        command += arg;
    }
    return command;
}

//! \return true if \p file does not exist / could not be opened.
bool fileAbsent(const std::string& file) {
    std::ifstream ifs{file};
    return ifs.is_open() == false;
}
}

BOOST_AUTO_TEST_CASE(testStartRejectsDuplicateDisableSandboxTokenOnSandboxedPath) {
    // Two occurrences of the token must be rejected outright, even when
    // processPath IS the configured sandboxed path - never "last one
    // wins"/"first one wins".
    const std::string OUT{"duplicate_reject_sandboxed_out.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(10, PROCESS_PATH,
                                         {"-c", "cp " + INPUT_FILE1 + " " + OUT,
                                          "--disableSandbox", "--disableSandbox"})};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    // Rejected before any spawn: the copy must never have happened.
    BOOST_REQUIRE_EQUAL(true, fileAbsent(OUT));

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":10,\"success\":false") != std::string::npos);
    BOOST_TEST_REQUIRE(response.find("specified 2 times") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testStartRejectsDuplicateDisableSandboxTokenOnNonSandboxedPath) {
    // Duplicate-token rejection applies regardless of whether processPath
    // matches a configured sandboxed path.
    const std::string OUT{"duplicate_reject_nonsandboxed_out.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths; // empty
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(11, PROCESS_PATH,
                                         {"-c", "cp " + INPUT_FILE1 + " " + OUT,
                                          "--disableSandbox", "--disableSandbox"})};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    BOOST_REQUIRE_EQUAL(true, fileAbsent(OUT));

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":11,\"success\":false") != std::string::npos);
    BOOST_TEST_REQUIRE(response.find("specified 2 times") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testStartRejectsDisableSandboxTokenOnNonSandboxedPath) {
    // A single --disableSandbox token is only meaningful for the exact
    // configured sandboxed path; on any other permitted process it must be
    // rejected rather than silently ignored or passed through.
    const std::string OUT{"single_reject_nonsandboxed_out.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths; // empty: PROCESS_PATH not sandboxed
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(
            12, PROCESS_PATH, {"-c", "cp " + INPUT_FILE1 + " " + OUT, "--disableSandbox"})};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    BOOST_REQUIRE_EQUAL(true, fileAbsent(OUT));

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":12,\"success\":false") != std::string::npos);
    BOOST_TEST_REQUIRE(response.find("only valid for the configured sandboxed process") !=
                       std::string::npos);
}

BOOST_AUTO_TEST_CASE(testStartStripsDisableSandboxTokenForConfiguredSandboxedPath) {
    // A single --disableSandbox token on the configured sandboxed path must
    // be stripped before the underlying spawner ever sees it. Verified via
    // an observable side effect (arg count reaching the shell), not just
    // the response: if the token leaked through, $# would be 1 instead of 0.
    const std::string OUT{"strip_token_arg_count.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(
            13, PROCESS_PATH, {"-c", "echo $# > " + OUT, "argv0name", "--disableSandbox"})};

        BOOST_REQUIRE_EQUAL(true, processor.handleCommand(command));
    }

    std::this_thread::sleep_for(std::chrono::seconds{1});

    std::ifstream ifs{OUT};
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();
    std::remove(OUT.c_str());

    // If the token had NOT been stripped, argv0name and --disableSandbox
    // would both reach the shell as positional args and $# would be 1.
    BOOST_REQUIRE_EQUAL(std::string{"0"}, content);

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":13,\"success\":true") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testStartLeavesArgsUntouchedWhenTokenAbsent) {
    // With zero occurrences of --disableSandbox, args must reach the
    // spawner completely unmodified (default route is Sandbox2, but this
    // processPath isn't configured as sandboxed so it still dispatches to
    // the legacy spawner, same as pre-existing behaviour).
    const std::string OUT{"absent_token_arg_count.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths; // empty
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(
            14, PROCESS_PATH, {"-c", "echo $# > " + OUT, "argv0name", "extraArg"})};

        BOOST_REQUIRE_EQUAL(true, processor.handleCommand(command));
    }

    std::this_thread::sleep_for(std::chrono::seconds{1});

    std::ifstream ifs{OUT};
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();
    std::remove(OUT.c_str());

    BOOST_REQUIRE_EQUAL(std::string{"1"}, content);

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":14,\"success\":true") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testStartDefaultsToLegacyRouteWhenTokenAbsentOnSandboxedPath) {
    // SHIPS DORMANT: with ML_SANDBOX2_DEFAULT_ENFORCED unset (the shipped
    // default), a no-token start command for the configured sandboxed path
    // must take the *legacy* route - i.e. behave exactly as it did before
    // typed routing existed. Observed here as the copy succeeding: had the
    // route been E_Sandbox2, this build (no Sandbox2 support / no real
    // Sandbox2 policy for /bin/sh) would have failed closed instead.
    //
    // Deliberately not gated on !SANDBOX2_AVAILABLE: the dormant default is
    // platform-independent, and on a Sandbox2 build this still proves the
    // legacy dispatch (a Sandbox2 launch of /bin/sh with these args would
    // not produce the file).
    ml::core::CUnSetEnv::unSetEnv("ML_SANDBOX2_DEFAULT_ENFORCED");

    const std::string OUT{"sandbox2_default_dormant_out.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(16, PROCESS_PATH,
                                         {"-c", "cp " + INPUT_FILE1 + " " + OUT})};

        BOOST_REQUIRE_EQUAL(true, processor.handleCommand(command));
    }

    std::this_thread::sleep_for(std::chrono::seconds{1});

    std::ifstream ifs{OUT};
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();
    std::remove(OUT.c_str());
    BOOST_REQUIRE_EQUAL(SLOGAN1, content);

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":16,\"success\":true") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testLegacyReasonProvenanceReachesH4Signal) {
    // The two legacy-route provenances must arrive at the H4 signal
    // distinguishable: mode == "degraded" alone cannot separate a deliberate
    // operator kill switch from the dormant default that is in effect for
    // the whole rollout window. This asserts the wiring from the route
    // decision in handleStart() through to the emitted signal.
    ml::core::CUnSetEnv::unSetEnv("ML_SANDBOX2_DEFAULT_ENFORCED");

    const std::string OUT{"sandbox2_legacy_reason_out.txt"};

    // (a) No token, option off -> dormant_default.
    std::remove(OUT.c_str());
    std::ostringstream dormantResponses;
    std::string dormantLogged{captureLogged([&] {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    dormantResponses};
        BOOST_REQUIRE_EQUAL(
            true, processor.handleCommand(startCommand(
                      20, PROCESS_PATH, {"-c", "cp " + INPUT_FILE1 + " " + OUT})));
    })};
    std::this_thread::sleep_for(std::chrono::seconds{1});
    std::remove(OUT.c_str());

    BOOST_REQUIRE(dormantLogged.find("\"route\":\"legacy\"") != std::string::npos);
    BOOST_REQUIRE(dormantLogged.find("\"legacy_reason\":\"dormant_default\"") !=
                  std::string::npos);
    BOOST_REQUIRE(dormantLogged.find("\"legacy_reason\":\"kill_switch\"") ==
                  std::string::npos);

    // (b) Validated --disableSandbox token -> kill_switch, whatever the
    // option's state (here explicitly on, so the token is the only reason
    // the legacy route could have been selected).
    CScopedSandbox2DefaultEnforced enforced{"1"};
    std::remove(OUT.c_str());
    std::ostringstream killSwitchResponses;
    std::string killSwitchLogged{captureLogged([&] {
        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    killSwitchResponses};
        BOOST_REQUIRE_EQUAL(
            true, processor.handleCommand(startCommand(
                      21, PROCESS_PATH,
                      {"-c", "cp " + INPUT_FILE1 + " " + OUT, "--disableSandbox"})));
    })};
    std::this_thread::sleep_for(std::chrono::seconds{1});
    std::remove(OUT.c_str());

    BOOST_REQUIRE(killSwitchLogged.find("\"route\":\"legacy\"") != std::string::npos);
    BOOST_REQUIRE(killSwitchLogged.find("\"legacy_reason\":\"kill_switch\"") !=
                  std::string::npos);
    BOOST_REQUIRE(killSwitchLogged.find("\"legacy_reason\":\"dormant_default\"") ==
                  std::string::npos);
}

#ifndef SANDBOX2_AVAILABLE
BOOST_AUTO_TEST_CASE(testStartSelectsSandbox2RouteWhenTokenAbsentAndDefaultEnforced) {
    // The opt-in half of the dormant default: with the internal option
    // explicitly on, no token present on the configured sandboxed path
    // selects the Sandbox2 route (V2, no automatic legacy fallback). On a
    // build with no Sandbox2 support, CProcessSpawnerRouter fails closed for
    // that route - observed here as the command failing rather than the copy
    // succeeding, which is exactly how we know Sandbox2 (not legacy) was
    // selected: had the route been E_Legacy, this copy would have succeeded
    // (see testStartDefaultsToLegacyRouteWhenTokenAbsentOnSandboxedPath,
    // which is the same vector with the option off).
    const std::string OUT{"sandbox2_route_selected_out.txt"};
    std::remove(OUT.c_str());

    std::ostringstream responseStream;
    {
        CScopedSandbox2DefaultEnforced enforced{"1"};

        ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
        ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                    responseStream};

        std::string command{startCommand(15, PROCESS_PATH,
                                         {"-c", "cp " + INPUT_FILE1 + " " + OUT})};

        BOOST_REQUIRE_EQUAL(false, processor.handleCommand(command));
    }

    BOOST_REQUIRE_EQUAL(true, fileAbsent(OUT));

    std::string response{responseStream.str()};
    BOOST_TEST_REQUIRE(response.find("\"id\":15,\"success\":false") != std::string::npos);
    BOOST_TEST_REQUIRE(response.find("Failed to start process") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testNonCanonicalTruthyValuesLeaveDefaultDormant) {
    // Exactly "1" is the one canonical truthy spelling. Anything else must
    // leave the option off, i.e. keep the legacy default - proven with the
    // same fail-closed vector as above: with the option genuinely on the
    // command fails, so a *succeeding* command is proof it stayed off.
    for (const char* value : {"true", "TRUE", "yes", "0", ""}) {
        const std::string OUT{"sandbox2_default_non_canonical_out.txt"};
        std::remove(OUT.c_str());

        std::ostringstream responseStream;
        {
            CScopedSandbox2DefaultEnforced notEnforced{value};

            ml::controller::CCommandProcessor::TStrVec permittedPaths{PROCESS_PATH};
            ml::controller::CCommandProcessor::TStrVec sandboxedPaths{PROCESS_PATH};
            ml::controller::CCommandProcessor processor{permittedPaths, sandboxedPaths,
                                                        responseStream};

            std::string command{startCommand(
                17, PROCESS_PATH, {"-c", "cp " + INPUT_FILE1 + " " + OUT})};

            BOOST_REQUIRE_EQUAL(true, processor.handleCommand(command));
        }

        std::this_thread::sleep_for(std::chrono::seconds{1});
        BOOST_REQUIRE_EQUAL(false, fileAbsent(OUT));
        std::remove(OUT.c_str());
    }
}
#endif // !SANDBOX2_AVAILABLE

BOOST_AUTO_TEST_SUITE_END()
