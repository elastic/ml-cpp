/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CTestTimer_h
#define INCLUDED_ml_test_CTestTimer_h

#include <core/CStopWatch.h>

#include <test/ImportExport.h>

#include <cppunit/TestListener.h>

#include <map>
#include <string>

#include <stdint.h>


namespace ml
{
namespace test
{

//! \brief
//! A class to time cppunit tests.
//!
//! DESCRIPTION:\n
//! A class to log the time taken by each cppunit test.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Timing is done to millisecond accuracy using a monotonic
//! timer where available.
//!
//! There is an assumption that each call to startTest() will
//! be followed by a call to endTest() FOR THE SAME TEST.  If
//! this assumption ceases to hold, for example if parallisation
//! is added to cppunit, then this class will need to be
//! rewritten.
//!
class TEST_EXPORT CTestTimer : public CppUnit::TestListener
{
    public:
        //! Called at the start of each test
        virtual void startTest(CppUnit::Test *test);

        //! Called at the end of each test
        virtual void endTest(CppUnit::Test *test);

        //! Get the time taken for a given test
        uint64_t timeForTest(const std::string &testName) const;

        //! Get the total time taken for all tests
        uint64_t totalTime(void) const;

        //! Get the average time taken for the tests
        uint64_t averageTime(void) const;

    private:
        //! Used to time each test
        core::CStopWatch m_StopWatch;

        using TStrUInt64Map = std::map<std::string, uint64_t>;
        using TStrUInt64MapCItr = TStrUInt64Map::const_iterator;

        //! Map of test name to time taken (in ms)
        TStrUInt64Map    m_TestTimes;
};


}
}

#endif // INCLUDED_ml_test_CTestTimer_h

