/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CThreadMutexConditionTest)

BOOST_AUTO_TEST_CASE(testThread) {
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
            LOG_DEBUG(<< "Thread running");
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

            LOG_DEBUG(<< "Thread exiting");
        }

        void shutdown() {
            LOG_DEBUG(<< "Thread shutdown");
            m_Mutex.lock();
            m_Running = false;
            m_Mutex.unlock();
        }

    private:
        ml::core::CMutex m_Mutex;
        bool m_Running;
    };

    CThread thread;

    BOOST_TEST(thread.isRunning() == false);

    // Start thread
    BOOST_TEST(thread.start());

    // Wait for thread to initialise
    ml::core::CSleep::sleep(1000);

    BOOST_TEST(thread.isRunning() == true);

    thread.stop();

    BOOST_TEST(thread.isRunning() == false);
}

BOOST_AUTO_TEST_CASE(testThreadCondition) {
    class CThread : public ml::core::CThread {
    public:
        CThread() : m_Condition(m_Mutex) {}

        void lock() {
            LOG_DEBUG(<< "lock start " << this->currentThreadId());
            m_Mutex.lock();
            LOG_DEBUG(<< "lock end " << this->currentThreadId());
        }

        void unlock() {
            LOG_DEBUG(<< "unlock " << this->currentThreadId());
            m_Mutex.unlock();
        }

        void wait() {
            LOG_DEBUG(<< "wait start " << this->currentThreadId());
            m_Condition.wait();
            LOG_DEBUG(<< "wait end " << this->currentThreadId());
        }

        void signal() {
            LOG_DEBUG(<< "signal " << this->currentThreadId());
            m_Condition.signal();
        }

    private:
        void run() {
            LOG_DEBUG(<< "Thread running");
            this->lock();
            this->signal();
            this->wait();
            this->unlock();
            LOG_DEBUG(<< "Thread exiting");
        }

        void shutdown() {
            LOG_DEBUG(<< "Thread shutting down");
            this->lock();
            this->signal();
            this->unlock();
            LOG_DEBUG(<< "Thread shutdown");
        }

    private:
        ml::core::CMutex m_Mutex;
        ml::core::CCondition m_Condition;
    };

    CThread thread;

    thread.lock();
    BOOST_TEST(thread.start());
    thread.wait();
    thread.unlock();

    BOOST_TEST(thread.stop());
}

BOOST_AUTO_TEST_SUITE_END()
