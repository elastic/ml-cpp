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

#include <core/COsFileFuncs.h>
#include <core/CProcess.h>
#include <core/CResourceLocator.h>
#include <core/CStringUtils.h>

#include "../CProcessSpawnerRouter.h"

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <unistd.h>

BOOST_AUTO_TEST_SUITE(CProcessSpawnerRouterTest)

namespace {
const std::string INPUT_FILE{ml::core::CResourceLocator::cppRootDir() +
                             "/bin/controller/unittest/testfiles/slogan1.txt"};
const std::string SLOGAN1{"Elastic is great!"};
}

BOOST_AUTO_TEST_CASE(testDisableSandboxUsesLegacyPath) {
    const std::string dir{"/tmp/ml_spawn_test_" +
                          ml::core::CStringUtils::typeToString(::getpid())};
    ::mkdir(dir.c_str(), 0700);

    const std::string linkPath{dir + "/pytorch_inference"};
    std::remove(linkPath.c_str());
    BOOST_TEST_REQUIRE(::symlink("/bin/sh", linkPath.c_str()) == 0);

    const std::string outputFile{dir + "/out.txt"};
    std::remove(outputFile.c_str());

    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{linkPath};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{linkPath};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{
        "--disableSandbox",
        "-c",
        "cp " + INPUT_FILE + " " + outputFile,
    };

    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(router.spawn(linkPath, args, childPid));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ml::core::COsFileFuncs::TStat statBuf;
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(outputFile.c_str(), &statBuf));

    std::ifstream ifs{outputFile};
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();
    BOOST_REQUIRE_EQUAL(SLOGAN1, content);

    std::remove(outputFile.c_str());
    std::remove(linkPath.c_str());
    ::rmdir(dir.c_str());
}

BOOST_AUTO_TEST_SUITE_END()
