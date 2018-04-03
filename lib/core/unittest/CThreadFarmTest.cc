/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CThreadFarmTest.h"

#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>
#include <core/CThreadFarm.h>

#include <boost/thread.hpp>

#include <set>
#include <string>


CppUnit::Test *CThreadFarmTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CThreadFarmTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CThreadFarmTest>(
                               "CThreadFarmTest::testNumCpus",
                               &CThreadFarmTest::testNumCpus) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CThreadFarmTest>(
                               "CThreadFarmTest::testSendReceive",
                               &CThreadFarmTest::testSendReceive) );

    return suiteOfTests;
}

void CThreadFarmTest::testNumCpus(void) {
    unsigned int numCpus(boost::thread::hardware_concurrency());

    LOG_INFO("Number of CPUs on this machine is " << numCpus);
}

namespace {
class CString {
    public:
        CString(void)
        {}

        CString(const std::string &str) : m_Str(str)
        {}

        CString(const CString &arg) : m_Str(arg.m_Str)
        {}

        CString &operator=(const CString &arg) {
            m_Str = arg.m_Str;
            return *this;
        }

        CString &operator=(const std::string &str) {
            m_Str = str;
            return *this;
        }

        const std::string   &str(void) const {
            return m_Str;
        }

    private:
        std::string m_Str;
};

class CHandler {
    public:
        void processResult(const CString &result) {
            LOG_DEBUG("Process result " << result.str() <<
                      " in thread " << ml::core::CThread::currentThreadId());

            ml::core::CScopedLock lock(m_Mutex);
            m_OutstandingOutput.erase(result.str());
        }

        void addExpectedOutput(const std::string &expected) {
            ml::core::CScopedLock lock(m_Mutex);
            m_OutstandingOutput.insert(expected);
        }

        bool haveAllExpected(void) {
            ml::core::CScopedLock lock(m_Mutex);

            TStrSet::iterator iter = m_OutstandingOutput.begin();
            if (iter != m_OutstandingOutput.end()) {
                LOG_WARN("Result: " << *iter << " is still outstanding");
            }

            return m_OutstandingOutput.empty();
        }

    private:
        typedef std::set<std::string> TStrSet;

        TStrSet m_OutstandingOutput;
        ml::core::CMutex m_Mutex;
};

class CProcessor {
    public:
        CProcessor(const std::string &id) : m_Id(id)
        {}

        void msgToResult(const std::string &str, CString &result) {
            LOG_DEBUG("messageToResult " << str);

            result = (str + ' ' + m_Id);

            LOG_DEBUG("messageToResult " << result.str());
        }

    private:
        std::string m_Id;
};
}

void CThreadFarmTest::testSendReceive(void) {
    CHandler handler;

    ml::core::CThreadFarm<CHandler, CProcessor, std::string, CString> farm(handler, "test");

    CProcessor proc1("1");
    CPPUNIT_ASSERT(farm.addProcessor(proc1));
    CProcessor proc2("2");
    CPPUNIT_ASSERT(farm.addProcessor(proc2));
    CProcessor proc3("3");
    CPPUNIT_ASSERT(farm.addProcessor(proc3));
    CProcessor proc4("4");
    CPPUNIT_ASSERT(farm.addProcessor(proc4));

    CPPUNIT_ASSERT(farm.start());

    size_t max(10);

    LOG_DEBUG("Sending " << max << " strings");

    char id = 'A';
    for (size_t i = 0; i < max; ++i) {
        std::string message("Test string ");
        message += id;

        // Need to add the expected output before sending the
        // message to the thread farm in case it's processed
        // really quickly!
        handler.addExpectedOutput(message + " 1");
        handler.addExpectedOutput(message + " 2");
        handler.addExpectedOutput(message + " 3");
        handler.addExpectedOutput(message + " 4");

        LOG_DEBUG("Add message " << message);
        CPPUNIT_ASSERT(farm.addMessage(message));

        ++id;
    }

    LOG_DEBUG("Sent all strings");

    CPPUNIT_ASSERT(farm.stop());

    LOG_DEBUG("Farm stopped");

    CPPUNIT_ASSERT(handler.haveAllExpected());
}

