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

#include <core/CStateFileRemover.h>

#include <boost/test/unit_test.hpp>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

BOOST_AUTO_TEST_SUITE(CStateFileRemoverTest)

namespace {

const std::string TEST_FILE{"CStateFileRemoverTest_quantiles_state"};

bool fileExists(const std::string& filename) {
    std::ifstream stream(filename);
    return stream.good();
}

void createTestFile(const std::string& filename) {
    std::ofstream stream(filename);
    stream << "test quantiles state data";
}

void removeTestFile(const std::string& filename) {
    std::remove(filename.c_str());
}

} // unnamed namespace

BOOST_AUTO_TEST_CASE(testNoDeleteWhenFilenameEmpty) {
    // Mirrors the case in both normalize and autodetect where quantilesStateFile
    // is empty, so CStateFileRemover is not created. Here we test the guard
    // condition directly: an empty filename should cause the destructor to
    // return early without attempting any file operation.
    { ml::core::CStateFileRemover remover(std::string{}, true); }
}

BOOST_AUTO_TEST_CASE(testNoDeleteWhenFlagFalse) {
    // Mirrors both apps when --deleteStateFiles is not passed. The file should
    // remain after CStateFileRemover is destroyed.
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    { ml::core::CStateFileRemover remover(TEST_FILE, false); }

    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));
    removeTestFile(TEST_FILE);
}

BOOST_AUTO_TEST_CASE(testDeleteOnDestruction) {
    // Mirrors the failure path in both normalize and autodetect: the
    // CStateFileRemover goes out of scope and its destructor deletes the
    // quantiles state file.
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    { ml::core::CStateFileRemover remover(TEST_FILE, true); }

    BOOST_TEST_REQUIRE(!fileExists(TEST_FILE));
}

BOOST_AUTO_TEST_CASE(testDeleteNonExistentFile) {
    // Edge case: the file does not exist at destruction time. The destructor
    // should handle this gracefully (logging a warning but not crashing).
    const std::string nonExistentFile{"CStateFileRemoverTest_does_not_exist"};
    removeTestFile(nonExistentFile);
    BOOST_TEST_REQUIRE(!fileExists(nonExistentFile));

    { ml::core::CStateFileRemover remover(nonExistentFile, true); }
}

BOOST_AUTO_TEST_CASE(testDisarmPreventsDelete) {
    // Calling disarm() sets the internal flag to false so the destructor
    // becomes a no-op. This is used when the caller wants to suppress the
    // RAII cleanup without leaking memory (as unique_ptr::release() would).
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    {
        ml::core::CStateFileRemover remover(TEST_FILE, true);
        remover.disarm();
    }

    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));
    removeTestFile(TEST_FILE);
}

BOOST_AUTO_TEST_CASE(testDisarmWithUniquePointer) {
    // Mirrors a safe alternative to the old happy path that used
    // unique_ptr::release() (which leaked). With disarm() the unique_ptr
    // destructor runs normally, no memory is leaked, and the file is
    // not deleted.
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    {
        auto remover = std::make_unique<ml::core::CStateFileRemover>(TEST_FILE, true);
        remover->disarm();
    }

    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));
    removeTestFile(TEST_FILE);
}

BOOST_AUTO_TEST_CASE(testUniquePointerDestructorTriggersDelete) {
    // Mirrors the failure path in both normalize and autodetect when a
    // unique_ptr<CStateFileRemover> goes out of scope without disarm()
    // being called. The destructor fires and deletes the file.
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    {
        auto remover = std::make_unique<ml::core::CStateFileRemover>(TEST_FILE, true);
    }

    BOOST_TEST_REQUIRE(!fileExists(TEST_FILE));
}

BOOST_AUTO_TEST_CASE(testHappyPathDestructorDeletesFile) {
    // Mirrors the happy path in both normalize and autodetect: the
    // unique_ptr is simply allowed to destruct at the end of main().
    // When deleteStateFiles is true the destructor deletes the file.
    // There is no need for separate manual deletion logic.
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    {
        auto remover = std::make_unique<ml::core::CStateFileRemover>(TEST_FILE, true);
    }

    BOOST_TEST_REQUIRE(!fileExists(TEST_FILE));
}

BOOST_AUTO_TEST_CASE(testHappyPathDestructorKeepsFileWhenFlagFalse) {
    // When deleteStateFiles is false, the destructor is a no-op and
    // the file remains on disk.
    removeTestFile(TEST_FILE);
    createTestFile(TEST_FILE);
    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));

    {
        auto remover = std::make_unique<ml::core::CStateFileRemover>(TEST_FILE, false);
    }

    BOOST_TEST_REQUIRE(fileExists(TEST_FILE));
    removeTestFile(TEST_FILE);
}

BOOST_AUTO_TEST_SUITE_END()
