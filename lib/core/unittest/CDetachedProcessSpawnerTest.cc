/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CDetachedProcessSpawnerTest.h"

#include <core/CDetachedProcessSpawner.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>
#include <core/CStringUtils.h>

#include <boost/range.hpp>

#include <stdio.h>
#include <stdlib.h>

namespace {
const std::string OUTPUT_FILE("withNs.xml");
#ifdef Windows
// Unlike Windows NT system calls, copy's command line cannot cope with
// forward slash path separators
const std::string INPUT_FILE("testfiles\\withNs.xml");
// File size is different on Windows due to CRLF line endings
const size_t EXPECTED_FILE_SIZE(585);
const char* winDir(::getenv("windir"));
const std::string PROCESS_PATH1(winDir != 0 ? std::string(winDir) + "\\System32\\cmd" : std::string("C:\\Windows\\System32\\cmd"));
const std::string PROCESS_ARGS1[] = {"/C", "copy " + INPUT_FILE + " ."};
const std::string& PROCESS_PATH2 = PROCESS_PATH1;
const std::string PROCESS_ARGS2[] = {"/C", "ping 127.0.0.1 -n 11"};
#else
const std::string INPUT_FILE("testfiles/withNs.xml");
const size_t EXPECTED_FILE_SIZE(563);
const std::string PROCESS_PATH1("/bin/dd");
const std::string PROCESS_ARGS1[] = {"if=" + INPUT_FILE,
                                     "of=" + OUTPUT_FILE,
                                     "bs=1",
                                     "count=" + ml::core::CStringUtils::typeToString(EXPECTED_FILE_SIZE)};
const std::string PROCESS_PATH2("/bin/sleep");
const std::string PROCESS_ARGS2[] = {"10"};
#endif
}

CppUnit::Test* CDetachedProcessSpawnerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDetachedProcessSpawnerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDetachedProcessSpawnerTest>("CDetachedProcessSpawnerTest::testSpawn",
                                                                               &CDetachedProcessSpawnerTest::testSpawn));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetachedProcessSpawnerTest>("CDetachedProcessSpawnerTest::testKill",
                                                                               &CDetachedProcessSpawnerTest::testKill));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetachedProcessSpawnerTest>("CDetachedProcessSpawnerTest::testPermitted",
                                                                               &CDetachedProcessSpawnerTest::testPermitted));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetachedProcessSpawnerTest>("CDetachedProcessSpawnerTest::testNonExistent",
                                                                               &CDetachedProcessSpawnerTest::testNonExistent));

    return suiteOfTests;
}

void CDetachedProcessSpawnerTest::testSpawn() {
    // The intention of this test is to copy a file by spawning an external
    // program and then make sure the file has been copied

    // Remove any output file left behind by a previous failed test, but don't
    // check the return code as this will usually fail
    ::remove(OUTPUT_FILE.c_str());

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(PROCESS_ARGS1, PROCESS_ARGS1 + boost::size(PROCESS_ARGS1));

    CPPUNIT_ASSERT(spawner.spawn(PROCESS_PATH1, args));

    // Expect the copy to complete in less than 1 second
    ml::core::CSleep::sleep(1000);

    ml::core::COsFileFuncs::TStat statBuf;
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::stat(OUTPUT_FILE.c_str(), &statBuf));
    CPPUNIT_ASSERT_EQUAL(EXPECTED_FILE_SIZE, static_cast<size_t>(statBuf.st_size));

    CPPUNIT_ASSERT_EQUAL(0, ::remove(OUTPUT_FILE.c_str()));
}

void CDetachedProcessSpawnerTest::testKill() {
    // The intention of this test is to spawn a process that sleeps for 10
    // seconds, but kill it before it exits by itself and prove that its death
    // has been detected

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH2);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(PROCESS_ARGS2, PROCESS_ARGS2 + boost::size(PROCESS_ARGS2));

    ml::core::CProcess::TPid childPid = 0;
    CPPUNIT_ASSERT(spawner.spawn(PROCESS_PATH2, args, childPid));

    CPPUNIT_ASSERT(spawner.hasChild(childPid));
    CPPUNIT_ASSERT(spawner.terminateChild(childPid));

    // The spawner should detect the death of the process within half a second
    ml::core::CSleep::sleep(500);

    CPPUNIT_ASSERT(!spawner.hasChild(childPid));

    // We shouldn't be able to kill an already killed process
    CPPUNIT_ASSERT(!spawner.terminateChild(childPid));

    // We shouldn't be able to kill processes we didn't start
    CPPUNIT_ASSERT(!spawner.terminateChild(1));
    CPPUNIT_ASSERT(!spawner.terminateChild(0));
    CPPUNIT_ASSERT(!spawner.terminateChild(static_cast<ml::core::CProcess::TPid>(-1)));
}

void CDetachedProcessSpawnerTest::testPermitted() {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    // Should fail as ml_test is not on the permitted processes list
    CPPUNIT_ASSERT(!spawner.spawn("./ml_test", ml::core::CDetachedProcessSpawner::TStrVec()));
}

void CDetachedProcessSpawnerTest::testNonExistent() {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, "./does_not_exist");
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    // Should fail as even though it's a permitted process as the file doesn't exist
    CPPUNIT_ASSERT(!spawner.spawn("./does_not_exist", ml::core::CDetachedProcessSpawner::TStrVec()));
}
