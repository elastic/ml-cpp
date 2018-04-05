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
#include "CThreadMutexConditionTest.h"

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CSleep.h>
#include <core/CThread.h>

CppUnit::Test* CThreadMutexConditionTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CThreadMutexConditionTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CThreadMutexConditionTest>("CThreadMutexConditionTest::testThread",
                                                                             &CThreadMutexConditionTest::testThread));
    suiteOfTests->addTest(new CppUnit::TestCaller<CThreadMutexConditionTest>("CThreadMutexConditionTest::testThreadCondition",
                                                                             &CThreadMutexConditionTest::testThreadCondition));

    return suiteOfTests;
}

void CThreadMutexConditionTest::testThread() {
    class CThread : public ml::core::CThread {
    public:
        CThread() : m_Running(false) {}

        bool isRunning() {
            m_Mutex.lock();
            bool ret = m_Running;
            m_Mutex.unlock();
            return ret;
        }

    private:
        void run() {
            LOG_DEBUG("Thread running");
            m_Mutex.lock();
            m_Running = true;
            m_Mutex.unlock();

            for (;;) {
                m_Mutex.lock();
                if (m_Running == false) {
                    m_Mutex.unlock();
                    break;
                }
                m_Mutex.unlock();
            }

            LOG_DEBUG("Thread exiting");
        }

        void shutdown() {
            LOG_DEBUG("Thread shutdown");
            m_Mutex.lock();
            m_Running = false;
            m_Mutex.unlock();
        }

    private:
        ml::core::CMutex m_Mutex;
        bool m_Running;
    };

    CThread thread;

    CPPUNIT_ASSERT(thread.isRunning() == false);

    // Start thread
    CPPUNIT_ASSERT(thread.start());

    // Wait for thread to initialise
    ml::core::CSleep::sleep(1000);

    CPPUNIT_ASSERT(thread.isRunning() == true);

    thread.stop();

    CPPUNIT_ASSERT(thread.isRunning() == false);
}

void CThreadMutexConditionTest::testThreadCondition() {
    class CThread : public ml::core::CThread {
    public:
        CThread() : m_Condition(m_Mutex) {}

        void lock() {
            LOG_DEBUG("lock start " << this->currentThreadId());
            m_Mutex.lock();
            LOG_DEBUG("lock end " << this->currentThreadId());
        }

        void unlock() {
            LOG_DEBUG("unlock " << this->currentThreadId());
            m_Mutex.unlock();
        }

        void wait() {
            LOG_DEBUG("wait start " << this->currentThreadId());
            m_Condition.wait();
            LOG_DEBUG("wait end " << this->currentThreadId());
        }

        void signal() {
            LOG_DEBUG("signal " << this->currentThreadId());
            m_Condition.signal();
        }

    private:
        void run() {
            LOG_DEBUG("Thread running");
            this->lock();
            this->signal();
            this->wait();
            this->unlock();
            LOG_DEBUG("Thread exiting");
        }

        void shutdown() {
            LOG_DEBUG("Thread shutting down");
            this->lock();
            this->signal();
            this->unlock();
            LOG_DEBUG("Thread shutdown");
        }

    private:
        ml::core::CMutex m_Mutex;
        ml::core::CCondition m_Condition;
    };

    CThread thread;

    thread.lock();
    CPPUNIT_ASSERT(thread.start());
    thread.wait();
    thread.unlock();

    CPPUNIT_ASSERT(thread.stop());
}
