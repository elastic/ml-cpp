/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTimingXmlOutputterHook.h>

#include <core/CStringUtils.h>

#include <test/CTestTimer.h>

#include <cppunit/Test.h>
#include <cppunit/tools/XmlElement.h>

namespace ml {
namespace test {

namespace {

const std::string NAME_TAG("Name");
const std::string TEST_PATH_TAG("TestPath");
const std::string TIME_TAG("Time");
const std::string TOTAL_ELAPSED_TIME_TAG("TotalElapsedTime");
const std::string AVERAGE_TEST_CASE_TIME_TAG("AverageTestCaseTime");
}

CTimingXmlOutputterHook::CTimingXmlOutputterHook(const CTestTimer& testTimer, const std::string& topPath, const std::string& testPath)
    : m_TestTimer(testTimer), m_TopPath(topPath), m_TestPath(testPath) {
}

void CTimingXmlOutputterHook::failTestAdded(CppUnit::XmlDocument* /*document*/,
                                            CppUnit::XmlElement* testElement,
                                            CppUnit::Test* test,
                                            CppUnit::TestFailure* /*failure*/) {
    if (testElement == 0 || test == 0) {
        return;
    }

    const std::string& testName = test->getName();

    testElement->elementFor(NAME_TAG)->setContent(m_TopPath + '.' + m_TestPath + '.' + testName);
}

void CTimingXmlOutputterHook::successfulTestAdded(CppUnit::XmlDocument* /*document*/,
                                                  CppUnit::XmlElement* testElement,
                                                  CppUnit::Test* test) {
    if (testElement == 0 || test == 0) {
        return;
    }

    const std::string& testName = test->getName();

    testElement->elementFor(NAME_TAG)->setContent(m_TopPath + '.' + m_TestPath + '.' + testName);

    testElement->addElement(new CppUnit::XmlElement(TEST_PATH_TAG, m_TestPath + '/' + testName));
    testElement->addElement(new CppUnit::XmlElement(TIME_TAG, this->toSecondsStr(m_TestTimer.timeForTest(testName))));
}

void CTimingXmlOutputterHook::statisticsAdded(CppUnit::XmlDocument* /*document*/, CppUnit::XmlElement* statisticsElement) {
    if (statisticsElement == 0) {
        return;
    }

    statisticsElement->addElement(new CppUnit::XmlElement(TOTAL_ELAPSED_TIME_TAG, this->toSecondsStr(m_TestTimer.totalTime())));
    statisticsElement->addElement(new CppUnit::XmlElement(AVERAGE_TEST_CASE_TIME_TAG, this->toSecondsStr(m_TestTimer.averageTime())));
}

std::string CTimingXmlOutputterHook::toSecondsStr(uint64_t ms) {
    std::string result(core::CStringUtils::typeToString(ms));

    if (result.length() < 4) {
        result.insert(0, "0.000", 5 - result.length());
    } else {
        result.insert(result.length() - 3, 1, '.');
    }

    return result;
}
}
}
