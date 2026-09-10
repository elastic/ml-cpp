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

#include <core/CDetachedProcessSpawner.h>
#include <core/COsFileFuncs.h>
#include <core/CSetEnv.h>
#include <core/CStringUtils.h>
#include <core/CUnSetEnv.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDetachedProcessSpawnerTest)

namespace {
const std::string OUTPUT_FILE("withNs.xml");
#ifdef Windows
// Unlike Windows NT system calls, copy's command line cannot cope with
// forward slash path separators
const std::string INPUT_FILE("testfiles\\withNs.xml");
// File size is different on Windows due to CRLF line endings
const size_t EXPECTED_FILE_SIZE(585);
const char* winDir(std::getenv("windir"));
const std::string PROCESS_PATH1(winDir != 0 ? std::string(winDir) + "\\System32\\cmd"
                                            : std::string("C:\\Windows\\System32\\cmd"));
const std::string PROCESS_ARGS1[] = {"/C", "copy " + INPUT_FILE + " ."};
const std::string& PROCESS_PATH2 = PROCESS_PATH1;
const std::string PROCESS_ARGS2[] = {"/C", "ping 127.0.0.1 -n 11"};
#else
const std::string INPUT_FILE("testfiles/withNs.xml");
const size_t EXPECTED_FILE_SIZE(563);
const std::string PROCESS_PATH1("/bin/dd");
const std::string PROCESS_ARGS1[] = {
    "if=" + INPUT_FILE, "of=" + OUTPUT_FILE, "bs=1",
    "count=" + ml::core::CStringUtils::typeToString(EXPECTED_FILE_SIZE)};
const std::string PROCESS_PATH2("/bin/sleep");
const std::string PROCESS_ARGS2[] = {"10"};
#endif
}

BOOST_AUTO_TEST_CASE(testSpawn) {
    // The intention of this test is to copy a file by spawning an external
    // program and then make sure the file has been copied

    // Remove any output file left behind by a previous failed test, but don't
    // check the return code as this will usually fail
    std::remove(OUTPUT_FILE.c_str());

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(
        PROCESS_ARGS1, PROCESS_ARGS1 + std::size(PROCESS_ARGS1));

    BOOST_TEST_REQUIRE(spawner.spawn(PROCESS_PATH1, args));

    // Expect the copy to complete in less than 1 second
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ml::core::COsFileFuncs::TStat statBuf;
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(OUTPUT_FILE.c_str(), &statBuf));
    BOOST_REQUIRE_EQUAL(EXPECTED_FILE_SIZE, static_cast<size_t>(statBuf.st_size));

    BOOST_REQUIRE_EQUAL(0, std::remove(OUTPUT_FILE.c_str()));
}

BOOST_AUTO_TEST_CASE(testKill) {
    // The intention of this test is to spawn a process that sleeps for 10
    // seconds, but kill it before it exits by itself and prove that its death
    // has been detected

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH2);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(
        PROCESS_ARGS2, PROCESS_ARGS2 + std::size(PROCESS_ARGS2));

    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(spawner.spawn(PROCESS_PATH2, args, childPid));

    BOOST_TEST_REQUIRE(spawner.hasChild(childPid));
    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));

    // The spawner should detect the death of the process within half a second
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));

    // We shouldn't be able to kill an already killed process
    BOOST_TEST_REQUIRE(!spawner.terminateChild(childPid));

    // We shouldn't be able to kill processes we didn't start
    BOOST_TEST_REQUIRE(!spawner.terminateChild(1));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(0));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(static_cast<ml::core::CProcess::TPid>(-1)));
}

BOOST_AUTO_TEST_CASE(testPermitted) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    // Should fail as ml_test is not on the permitted processes list
    BOOST_TEST_REQUIRE(
        !spawner.spawn("./ml_test", ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_CASE(testNonExistent) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, "./does_not_exist");
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    // Should fail as even though it's a permitted process as the file doesn't exist
    BOOST_TEST_REQUIRE(!spawner.spawn(
        "./does_not_exist", ml::core::CDetachedProcessSpawner::TStrVec()));
}

#ifndef Windows
BOOST_AUTO_TEST_CASE(testMlSandboxedStrippedFromChildEnvironment) {
    // ML_SANDBOXED=1 is the Sandbox2 sandboxee marker
    // (lib/sandbox/CSandboxedProcessSpawner_Linux.cc) and pytorch_inference
    // skips its mandatory in-process seccomp filter when it sees it
    // (include/seccomp/CSystemCallFilter.h sandbox2LaunchedChild()). A child
    // spawned by this class is never inside Sandbox2, so it must never
    // inherit the marker - not even when the spawning process's own
    // environment carries it.
    BOOST_REQUIRE_EQUAL(0, ml::core::CSetEnv::setEnv("ML_SANDBOXED", "1", 1));
    BOOST_REQUIRE_EQUAL(0, ml::core::CSetEnv::setEnv("ML_SANDBOXED_KEEP_ME", "1", 1));

    // Pure form: the array handed to posix_spawn() drops ML_SANDBOXED,
    // keeps everything else in order, and is NULL terminated. Exact-name
    // match only, so a different variable sharing the prefix survives.
    {
        std::vector<std::string> parentEntries{"PATH=/bin", "ML_SANDBOXED=1",
                                               "ML_SANDBOXED_KEEP_ME=1", "TMPDIR=/tmp"};
        std::vector<char*> parentEnv;
        for (auto& entry : parentEntries) {
            parentEnv.push_back(const_cast<char*>(entry.c_str()));
        }
        parentEnv.push_back(static_cast<char*>(nullptr));

        auto childEnv = ml::core::detail::buildChildEnvironment(&parentEnv[0]);
        BOOST_REQUIRE_EQUAL(std::size_t(4), childEnv.size());
        BOOST_REQUIRE_EQUAL(std::string("PATH=/bin"), std::string(childEnv[0]));
        BOOST_REQUIRE_EQUAL(std::string("ML_SANDBOXED_KEEP_ME=1"), std::string(childEnv[1]));
        BOOST_REQUIRE_EQUAL(std::string("TMPDIR=/tmp"), std::string(childEnv[2]));
        BOOST_REQUIRE_EQUAL(static_cast<char*>(nullptr), childEnv[3]);
    }

    BOOST_REQUIRE_EQUAL(true, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOXED=1"));
    BOOST_REQUIRE_EQUAL(true, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOXED="));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOXED_KEEP_ME=1"));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOX=1"));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry(nullptr));

    // End to end: a real spawned child reports what it actually inherited.
    // Its stdout is redirected to /dev/null by the spawner, so the shell
    // writes the value to a file instead.
    const std::string envDumpFile{"child_ml_sandboxed.txt"};
    std::remove(envDumpFile.c_str());

    const std::string shell{"/bin/sh"};
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, shell);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args{
        "-c", "echo \"[${ML_SANDBOXED-unset}][${ML_SANDBOXED_KEEP_ME-unset}]\" > " + envDumpFile};
    BOOST_TEST_REQUIRE(spawner.spawn(shell, args));

    std::string dumped;
    for (int attempt = 0; attempt < 20 && dumped.empty(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::ifstream ifs{envDumpFile};
        if (ifs.is_open()) {
            std::getline(ifs, dumped);
        }
    }

    BOOST_REQUIRE_EQUAL(std::string("[unset][1]"), dumped);

    std::remove(envDumpFile.c_str());
    ml::core::CUnSetEnv::unSetEnv("ML_SANDBOXED");
    ml::core::CUnSetEnv::unSetEnv("ML_SANDBOXED_KEEP_ME");
}
#endif // !Windows

#ifdef Windows
BOOST_AUTO_TEST_CASE(testMlSandboxedStrippedFromChildEnvironmentBlock) {
    // Windows analog of testMlSandboxedStrippedFromChildEnvironment above:
    // ML_SANDBOXED=1 is the Sandbox2 sandboxee marker and pytorch_inference
    // skips its mandatory in-process seccomp filter when it sees it (see
    // include/seccomp/CSystemCallFilter.h sandbox2LaunchedChild()). A child
    // spawned by this class is never inside Sandbox2, so it must never
    // inherit the marker via the environment block passed to
    // CreateProcess()'s lpEnvironment parameter - not even when the
    // spawning process's own environment carries it.
    BOOST_REQUIRE_EQUAL(true, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOXED=1"));
    BOOST_REQUIRE_EQUAL(true, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOXED="));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOXED_KEEP_ME=1"));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry("ML_SANDBOX=1"));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry(nullptr));
    // Windows environment variable names are case-INSENSITIVE OS-wide, and
    // the child-side reader (std::getenv, via CSystemCallFilter's
    // sandbox2LaunchedChild()) matches case-insensitively too. A
    // differently-cased marker must still be recognised and stripped here,
    // or it would survive the filter and still be found by the child.
    BOOST_REQUIRE_EQUAL(true, ml::core::detail::isStrippedChildEnvEntry("ml_sandboxed=1"));
    BOOST_REQUIRE_EQUAL(true, ml::core::detail::isStrippedChildEnvEntry("Ml_Sandboxed=1"));
    BOOST_REQUIRE_EQUAL(false, ml::core::detail::isStrippedChildEnvEntry("ml_sandboxed_keep_me=1"));

    // Build a synthetic Windows environment block: NUL-terminated
    // "NAME=VALUE" strings back to back, with an extra terminating NUL after
    // the last entry's own NUL.
    auto appendEntry = [](std::string& block, const std::string& entry) {
        block.append(entry);
        block.push_back('\0');
    };
    std::string parentBlock;
    appendEntry(parentBlock, "PATH=C:\\Windows");
    appendEntry(parentBlock, "ML_SANDBOXED=1");
    appendEntry(parentBlock, "ML_SANDBOXED_KEEP_ME=1");
    appendEntry(parentBlock, "ml_sandboxed=2");
    appendEntry(parentBlock, "TMP=C:\\Temp");
    parentBlock.push_back('\0');

    std::string childBlock{
        ml::core::detail::buildChildEnvironmentBlock(parentBlock.c_str())};

    // Walk the resulting block and confirm ML_SANDBOXED is gone but
    // everything else survives, in order, and the block is still
    // double-NUL-terminated.
    std::vector<std::string> childEntries;
    const char* entry{childBlock.c_str()};
    while (*entry != '\0') {
        std::string entryStr(entry);
        childEntries.push_back(entryStr);
        entry += entryStr.length() + 1;
    }

    // Both the canonically-cased and the differently-cased marker
    // ("ml_sandboxed=2") must be stripped: Windows env var lookups are
    // case-insensitive, so either form would still be visible to the
    // child's std::getenv("ML_SANDBOXED") if it survived here.
    BOOST_REQUIRE_EQUAL(std::size_t(3), childEntries.size());
    BOOST_REQUIRE_EQUAL(std::string("PATH=C:\\Windows"), childEntries[0]);
    BOOST_REQUIRE_EQUAL(std::string("ML_SANDBOXED_KEEP_ME=1"), childEntries[1]);
    BOOST_REQUIRE_EQUAL(std::string("TMP=C:\\Temp"), childEntries[2]);
    // Two-NUL block terminator: the last byte and the one before it are NUL.
    BOOST_TEST_REQUIRE(childBlock.size() >= 2);
    BOOST_REQUIRE_EQUAL('\0', childBlock[childBlock.size() - 1]);
    BOOST_REQUIRE_EQUAL('\0', childBlock[childBlock.size() - 2]);

    // Empty-environment edge case still produces a valid double-NUL block.
    std::string emptyParentBlock;
    emptyParentBlock.push_back('\0');
    std::string emptyChildBlock{
        ml::core::detail::buildChildEnvironmentBlock(emptyParentBlock.c_str())};
    BOOST_REQUIRE_EQUAL(std::size_t(2), emptyChildBlock.size());
    BOOST_REQUIRE_EQUAL('\0', emptyChildBlock[0]);
    BOOST_REQUIRE_EQUAL('\0', emptyChildBlock[1]);
}
#endif // Windows

BOOST_AUTO_TEST_SUITE_END()
