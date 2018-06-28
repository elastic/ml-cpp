/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CTimingXmlOutputterHook_h
#define INCLUDED_ml_test_CTimingXmlOutputterHook_h

#include <test/ImportExport.h>

#include <cppunit/XmlOutputterHook.h>

#include <string>

#include <stdint.h>

namespace ml {
namespace test {
class CTestTimer;

//! \brief
//! A class to add timings to XML results.
//!
//! DESCRIPTION:\n
//! A class to add the time taken by each cppunit test to
//! XML results.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Outputs in the same format as the ClockerPlugin example
//! because the xUnit Jenkins plugin's default XSL transform
//! knows this format.
//!
class TEST_EXPORT CTimingXmlOutputterHook : public CppUnit::XmlOutputterHook {
public:
    CTimingXmlOutputterHook(const CTestTimer& testTimer,
                            const std::string& topPath,
                            const std::string& testPath);

    virtual void failTestAdded(CppUnit::XmlDocument* document,
                               CppUnit::XmlElement* testElement,
                               CppUnit::Test* test,
                               CppUnit::TestFailure* failure);

    virtual void successfulTestAdded(CppUnit::XmlDocument* document,
                                     CppUnit::XmlElement* testElement,
                                     CppUnit::Test* test);

    virtual void statisticsAdded(CppUnit::XmlDocument* document,
                                 CppUnit::XmlElement* statisticsElement);

private:
    //! Convert a time in ms to a time in seconds in string form
    static std::string toSecondsStr(uint64_t ms);

private:
    //! Reference to test timer that we can query
    const CTestTimer& m_TestTimer;

    //! "bin" or "lib", to make the Jenkins output nicer
    const std::string& m_TopPath;

    //! Name of the directory above the "unittest" directory being tested
    const std::string& m_TestPath;
};
}
}

#endif // INCLUDED_ml_test_CTimingXmlOutputterHook_h
