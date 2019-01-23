/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestRunner.h>

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CResourceLocator.h>
#include <core/CStringUtils.h>

#include <test/CTestTimer.h>
#include <test/CTimingXmlOutputterHook.h>

#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/extensions/HelperMacros.h>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <fstream>
#include <ios>
#include <stdexcept>

#include <errno.h>
#include <stdio.h>
#include <string.h>

namespace {
// Since the unit tests log to STDERR, we can improve performance, especially on
// Windows, by buffering it.  The file scope variable ensures the setvbuf() gets
// called to set the buffer size during the static initialisation phase of the
// program, before main() is called.  By putting this in the libMlTest library,
// STDERR remains in its standard unbuffered state in Ml non-test programs.
// (Note that Linux always uses a 1024 byte buffer size for line-buffered files,
// i.e. ignores the 16384 specified in this call, and Windows treats a request
// for line buffering as a request for full buffering, so this doesn't do what
// it says on the tin on all platforms.  However, it still has a beneficial
// effect on unit test performance.)
const int DO_NOT_USE_THIS_VARIABLE(::setvbuf(stderr, nullptr, _IOLBF, 16384));

const std::string LIB_DIR("lib");
const std::string BIN_DIR("bin");
const std::string UNKNOWN_DIR("unknown");
}

namespace ml {
namespace test {

// Initialise statics
const std::string CTestRunner::SKIP_FILE_NAME("unit_test_skip_dirs.csv");
const std::string CTestRunner::XML_RESULT_FILE_NAME("cppunit_results.xml");

CTestRunner::CTestRunner(int argc, const char** argv) {
    ml::core::CLogger::instance().fatalErrorHandler([](std::string message) {
        LOG_FATAL(<< message);
        CPPUNIT_ASSERT(false);
    });
    this->processCmdLine(argc, argv);
}

CTestRunner::~CTestRunner() {
}

void CTestRunner::processCmdLine(int argc, const char** argv) {
    std::string exeName(argv[0]);

    size_t pos(exeName.rfind('/'));
    if (pos != std::string::npos) {
        m_ExeName.assign(exeName, pos + 1, exeName.length() - pos - 1);
    } else {
        m_ExeName = exeName;
    }

    if (argc > 1) {
        static const std::string SRC_EXT(".cc");
        static const std::string HDR_EXT(".h");
        m_TestCases.reserve(argc - 1);
        size_t numSrcStrips(0);
        size_t numHdrStrips(0);
        int lastSrcIndex(0);
        int lastHdrIndex(0);
        for (int i = 1; i < argc; ++i) {
            m_TestCases.push_back(argv[i]);
            std::string& testName = m_TestCases.back();
            if (testName.length() > SRC_EXT.length() &&
                testName.rfind(SRC_EXT) == testName.length() - SRC_EXT.length()) {
                testName.erase(testName.length() - SRC_EXT.length());
                ++numSrcStrips;
                lastSrcIndex = i;
            } else if (testName.length() > HDR_EXT.length() &&
                       testName.rfind(HDR_EXT) == testName.length() - HDR_EXT.length()) {
                testName.erase(testName.length() - HDR_EXT.length());
                ++numHdrStrips;
                lastHdrIndex = i;
            }
        }
        if (numSrcStrips == 1) {
            LOG_INFO(<< "Source file extension " << SRC_EXT
                     << " stripped from supplied test name " << argv[lastSrcIndex]);
        } else if (numSrcStrips > 0) {
            LOG_INFO(<< "Source file extension " << SRC_EXT << " stripped from "
                     << numSrcStrips << " supplied test names");
        }
        if (numHdrStrips == 1) {
            LOG_INFO(<< "Header file extension " << HDR_EXT
                     << " stripped from supplied test name " << argv[lastHdrIndex]);
        } else if (numHdrStrips > 0) {
            LOG_INFO(<< "Header file extension " << HDR_EXT << " stripped from "
                     << numHdrStrips << " supplied test names");
        }
        std::sort(m_TestCases.begin(), m_TestCases.end());
        size_t numDuplicates(m_TestCases.size());
        m_TestCases.erase(std::unique(m_TestCases.begin(), m_TestCases.end()),
                          m_TestCases.end());
        numDuplicates -= m_TestCases.size();
        if (numDuplicates > 0) {
            LOG_WARN(<< numDuplicates
                     << " of the supplied test names were "
                        "duplicates - each test case will only be run once");
        }
    }
}

bool CTestRunner::runTests() {
    boost::filesystem::path cwd;
    try {
        cwd = boost::filesystem::current_path();
    } catch (std::exception& e) {
        LOG_ERROR(<< "Unable to determine current directory: " << e.what());
        return false;
    }

    bool passed(false);
    if (this->checkSkipFile(cwd.string(), passed) == true) {
        LOG_WARN(<< "Skipping tests for directory " << cwd
                 << " and using previous test result " << std::boolalpha << passed);
        return passed;
    }

    // Take the "test path" to be the directory above the one called "unittest",
    // as this should make clear to the human reader which library/program is
    // being tested.  The "top path" is either "bin" or "lib", whichever is
    // found first when continuing to search backwards through the path.
    std::string topPath(UNKNOWN_DIR);
    std::string testPath(UNKNOWN_DIR);
    boost::filesystem::path::iterator iter = cwd.end();
    if (--iter != cwd.begin() && --iter != cwd.begin()) {
        testPath = iter->string();
        while (--iter != cwd.begin()) {
            if (iter->string() == LIB_DIR) {
                topPath = LIB_DIR;
                break;
            }
            if (iter->string() == BIN_DIR) {
                topPath = BIN_DIR;
                break;
            }
        }
    }

    passed = this->timeTests(topPath, testPath);

    if (this->updateSkipFile(cwd.string(), passed) == true) {
        LOG_INFO(<< "Added directory " << cwd << " to skip file with result "
                 << std::boolalpha << passed);
    }

    return passed;
}

bool CTestRunner::timeTests(const std::string& topPath, const std::string& testPath) {
    bool allPassed(true);

    CTestTimer testTimer;
    CppUnit::TestResultCollector resultCollector;

    // m_eventManager is a protected member in the base class
    if (m_eventManager != nullptr) {
        m_eventManager->addListener(&testTimer);
        m_eventManager->addListener(&resultCollector);
    } else {
        LOG_ERROR(<< "Unexpected NULL pointer");
    }

    if (m_TestCases.empty()) {
        allPassed = this->run();
    } else {
        for (TStrVecItr itr = m_TestCases.begin();
             itr != m_TestCases.end() && allPassed; ++itr) {
            try {
                allPassed = this->run(*itr);
            } catch (std::invalid_argument&) {
                LOG_ERROR(<< "No Test called " << *itr << " in testsuite");
            }
        }
    }

    if (m_eventManager != nullptr) {
        std::ofstream xmlResultFile(XML_RESULT_FILE_NAME.c_str());
        if (xmlResultFile.is_open()) {
            CppUnit::XmlOutputter xmlOutputter(&resultCollector, xmlResultFile);
            CTimingXmlOutputterHook hook(testTimer, topPath, testPath);
            xmlOutputter.addHook(&hook);
            xmlOutputter.write();
            xmlOutputter.removeHook(&hook);
        }

        m_eventManager->removeListener(&resultCollector);
        m_eventManager->removeListener(&testTimer);
    }

    return allPassed;
}

bool CTestRunner::checkSkipFile(const std::string& cwd, bool& passed) const {
    std::string fullPath(core::CResourceLocator::cppRootDir() + '/' + SKIP_FILE_NAME);
    std::ifstream strm(fullPath.c_str());

    std::string line;
    while (std::getline(strm, line)) {
        size_t commaPos(line.rfind(','));
        if (commaPos != std::string::npos && line.compare(0, commaPos, cwd) == 0 &&
            core::CStringUtils::stringToType(line.substr(commaPos + 1), passed) == true) {
            return true;
        }
    }

    return false;
}

bool CTestRunner::updateSkipFile(const std::string& cwd, bool passed) const {
    std::string fullPath(core::CResourceLocator::cppRootDir() + '/' + SKIP_FILE_NAME);

    // Don't create the file if it doesn't already exist, and don't write to it
    // if it's not writable
    if (core::COsFileFuncs::access(fullPath.c_str(), core::COsFileFuncs::READABLE |
                                                         core::COsFileFuncs::WRITABLE) == -1) {
        LOG_TRACE(<< "Will not update skip file " << fullPath << " : " << ::strerror(errno));
        return false;
    }

    // Append to any existing contents
    std::ofstream strm(fullPath.c_str(), std::ios::out | std::ios::app);
    strm << cwd << ',' << std::boolalpha << passed << std::endl;

    return strm.good();
}
}
}
