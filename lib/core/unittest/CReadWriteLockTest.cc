/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CFastMutex.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CReadWriteLock.h>
#include <core/CScopedFastLock.h>
#include <core/CScopedLock.h>
#include <core/CScopedReadLock.h>
#include <core/CScopedWriteLock.h>
#include <core/CThread.h>
#include <core/CTimeUtils.h>

#include <boost/test/unit_test.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

BOOST_AUTO_TEST_SUITE(CReadWriteLockTest)

namespace {

class CUnprotectedAdder : public ml::core::CThread {
public:
    CUnprotectedAdder(std::uint32_t sleepTime,
                      std::uint32_t iterations,
                      std::uint32_t increment,
                      volatile std::uint32_t& variable)
        : m_SleepTime(sleepTime), m_Iterations(iterations),
          m_Increment(increment), m_Variable(variable) {}

protected:
    void run() {
        for (std::uint32_t count = 0; count < m_Iterations; ++count) {
            m_Variable += m_Increment;
            std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTime));
        }
    }

    void shutdown() {
        // Always just wait for run() to complete
    }

private:
    std::uint32_t m_SleepTime;
    std::uint32_t m_Iterations;
    std::uint32_t m_Increment;
    volatile std::uint32_t& m_Variable;
};

class CAtomicAdder : public ml::core::CThread {
public:
    CAtomicAdder(std::uint32_t sleepTime,
                 std::uint32_t iterations,
                 std::uint32_t increment,
                 std::atomic_uint_fast32_t& variable)
        : m_SleepTime(sleepTime), m_Iterations(iterations),
          m_Increment(increment), m_Variable(variable) {}

protected:
    void run() {
        for (std::uint32_t count = 0; count < m_Iterations; ++count) {
            m_Variable.fetch_add(m_Increment);
            std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTime));
        }
    }

    void shutdown() {
        // Always just wait for run() to complete
    }

private:
    std::uint32_t m_SleepTime;
    std::uint32_t m_Iterations;
    std::uint32_t m_Increment;
    std::atomic_uint_fast32_t& m_Variable;
};

class CFastMutexProtectedAdder : public ml::core::CThread {
public:
    CFastMutexProtectedAdder(ml::core::CFastMutex& mutex,
                             std::uint32_t sleepTime,
                             std::uint32_t iterations,
                             std::uint32_t increment,
                             volatile std::uint32_t& variable)
        : m_Mutex(mutex), m_SleepTime(sleepTime), m_Iterations(iterations),
          m_Increment(increment), m_Variable(variable) {}

protected:
    void run() {
        for (std::uint32_t count = 0; count < m_Iterations; ++count) {
            ml::core::CScopedFastLock lock(m_Mutex);

            m_Variable += m_Increment;
            std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTime));
        }
    }

    void shutdown() {
        // Always just wait for run() to complete
    }

private:
    ml::core::CFastMutex& m_Mutex;
    std::uint32_t m_SleepTime;
    std::uint32_t m_Iterations;
    std::uint32_t m_Increment;
    volatile std::uint32_t& m_Variable;
};

class CMutexProtectedAdder : public ml::core::CThread {
public:
    CMutexProtectedAdder(ml::core::CMutex& mutex,
                         std::uint32_t sleepTime,
                         std::uint32_t iterations,
                         std::uint32_t increment,
                         volatile std::uint32_t& variable)
        : m_Mutex(mutex), m_SleepTime(sleepTime), m_Iterations(iterations),
          m_Increment(increment), m_Variable(variable) {}

protected:
    void run() {
        for (std::uint32_t count = 0; count < m_Iterations; ++count) {
            ml::core::CScopedLock lock(m_Mutex);

            m_Variable += m_Increment;
            std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTime));
        }
    }

    void shutdown() {
        // Always just wait for run() to complete
    }

private:
    ml::core::CMutex& m_Mutex;
    std::uint32_t m_SleepTime;
    std::uint32_t m_Iterations;
    std::uint32_t m_Increment;
    volatile std::uint32_t& m_Variable;
};

class CWriteLockProtectedAdder : public ml::core::CThread {
public:
    CWriteLockProtectedAdder(ml::core::CReadWriteLock& readWriteLock,
                             std::uint32_t sleepTime,
                             std::uint32_t iterations,
                             std::uint32_t increment,
                             volatile std::uint32_t& variable)
        : m_ReadWriteLock(readWriteLock), m_SleepTime(sleepTime),
          m_Iterations(iterations), m_Increment(increment), m_Variable(variable) {}

protected:
    void run() {
        for (std::uint32_t count = 0; count < m_Iterations; ++count) {
            ml::core::CScopedWriteLock lock(m_ReadWriteLock);

            m_Variable += m_Increment;
            std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTime));
        }
    }

    void shutdown() {
        // Always just wait for run() to complete
    }

private:
    ml::core::CReadWriteLock& m_ReadWriteLock;
    std::uint32_t m_SleepTime;
    std::uint32_t m_Iterations;
    std::uint32_t m_Increment;
    volatile std::uint32_t& m_Variable;
};

class CReadLockProtectedReader : public ml::core::CThread {
public:
    CReadLockProtectedReader(ml::core::CReadWriteLock& readWriteLock,
                             std::uint32_t sleepTime,
                             std::uint32_t iterations,
                             volatile std::uint32_t& variable)
        : m_ReadWriteLock(readWriteLock), m_SleepTime(sleepTime),
          m_Iterations(iterations), m_Variable(variable), m_LastRead(variable) {}

    std::uint32_t lastRead() const { return m_LastRead; }

protected:
    void run() {
        for (std::uint32_t count = 0; count < m_Iterations; ++count) {
            ml::core::CScopedReadLock lock(m_ReadWriteLock);

            m_LastRead = m_Variable;
            std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTime));
        }
    }

    void shutdown() {
        // Always just wait for run() to complete
    }

private:
    ml::core::CReadWriteLock& m_ReadWriteLock;
    std::uint32_t m_SleepTime;
    std::uint32_t m_Iterations;
    volatile std::uint32_t& m_Variable;
    std::uint32_t m_LastRead;
};
}

BOOST_AUTO_TEST_CASE(testReadLock) {
    std::uint32_t testVariable(0);
    ml::core::CReadWriteLock readWriteLock;

    // Each reader will do 1 second of "work" inside a read lock.  If they all
    // work at the same time (which they should) then the test will take around
    // 1 second.  If they block each other, the test will take around 3
    // seconds.
    CReadLockProtectedReader reader1(readWriteLock, 100, 10, testVariable);
    CReadLockProtectedReader reader2(readWriteLock, 100, 10, testVariable);
    CReadLockProtectedReader reader3(readWriteLock, 100, 10, testVariable);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());

    reader1.start();
    reader2.start();
    reader3.start();

    testVariable = 42;

    reader1.stop();
    reader2.stop();
    reader3.stop();

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    ml::core_t::TTime duration(end - start);

    LOG_INFO(<< "Reader concurrency test took " << duration << " seconds");

    // Allow the test to run up to 3 seconds, as there is processing
    // other than the sleeping, and also sleeps are not very accurate
    // under Jenkins on Apple M1.
    BOOST_TEST_REQUIRE(duration <= 3);
    BOOST_TEST_REQUIRE(duration >= 1);

    BOOST_REQUIRE_EQUAL(testVariable, reader1.lastRead());
    BOOST_REQUIRE_EQUAL(testVariable, reader2.lastRead());
    BOOST_REQUIRE_EQUAL(testVariable, reader3.lastRead());
}

BOOST_AUTO_TEST_CASE(testWriteLock) {
    static const std::uint32_t TEST_SIZE(50000);

    std::uint32_t testVariable(0);
    ml::core::CReadWriteLock readWriteLock;

    CWriteLockProtectedAdder writer1(readWriteLock, 0, TEST_SIZE, 1, testVariable);
    CWriteLockProtectedAdder writer2(readWriteLock, 0, TEST_SIZE, 5, testVariable);
    CWriteLockProtectedAdder writer3(readWriteLock, 0, TEST_SIZE, 9, testVariable);

    writer1.start();
    writer2.start();
    writer3.start();

    writer1.stop();
    writer2.stop();
    writer3.stop();

    LOG_INFO(<< "Write lock protected variable incremented to " << testVariable);

    BOOST_REQUIRE_EQUAL(TEST_SIZE * (1 + 5 + 9), testVariable);
}

BOOST_AUTO_TEST_CASE(testPerformanceVersusMutex) {
    static const std::uint32_t TEST_SIZE(1000000);

    {
        std::uint32_t testVariable(0);

        ml::core_t::TTime start(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Starting unlocked throughput test at "
                 << ml::core::CTimeUtils::toTimeString(start));

        CUnprotectedAdder writer1(0, TEST_SIZE, 1, testVariable);
        CUnprotectedAdder writer2(0, TEST_SIZE, 5, testVariable);
        CUnprotectedAdder writer3(0, TEST_SIZE, 9, testVariable);

        writer1.start();
        writer2.start();
        writer3.start();

        writer1.stop();
        writer2.stop();
        writer3.stop();

        ml::core_t::TTime end(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Finished unlocked throughput test at "
                 << ml::core::CTimeUtils::toTimeString(end));

        LOG_INFO(<< "Unlocked throughput test with test size " << TEST_SIZE
                 << " took " << (end - start) << " seconds");

        LOG_INFO(<< "Unlocked variable incremented to " << testVariable);

        if (testVariable != TEST_SIZE * (1 + 5 + 9)) {
            // Obviously this would be unacceptable in production code, but this
            // unit test is showing the cost of different types of lock compared
            // to the unlocked case
            LOG_INFO(<< "Lack of locking caused race condition");
        }
    }
    {
        std::atomic_uint_fast32_t testVariable(0);

        ml::core_t::TTime start(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Starting atomic throughput test at "
                 << ml::core::CTimeUtils::toTimeString(start));

        CAtomicAdder writer1(0, TEST_SIZE, 1, testVariable);
        CAtomicAdder writer2(0, TEST_SIZE, 5, testVariable);
        CAtomicAdder writer3(0, TEST_SIZE, 9, testVariable);

        writer1.start();
        writer2.start();
        writer3.start();

        writer1.stop();
        writer2.stop();
        writer3.stop();

        ml::core_t::TTime end(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Finished atomic throughput test at "
                 << ml::core::CTimeUtils::toTimeString(end));

        LOG_INFO(<< "Atomic throughput test with test size " << TEST_SIZE
                 << " took " << (end - start) << " seconds");

        LOG_INFO(<< "Atomic variable incremented to " << testVariable.load());

        BOOST_REQUIRE_EQUAL(uint_fast32_t(TEST_SIZE * (1 + 5 + 9)), testVariable.load());
    }
    {
        std::uint32_t testVariable(0);
        ml::core::CFastMutex mutex;

        ml::core_t::TTime start(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Starting fast mutex lock throughput test at "
                 << ml::core::CTimeUtils::toTimeString(start));

        CFastMutexProtectedAdder writer1(mutex, 0, TEST_SIZE, 1, testVariable);
        CFastMutexProtectedAdder writer2(mutex, 0, TEST_SIZE, 5, testVariable);
        CFastMutexProtectedAdder writer3(mutex, 0, TEST_SIZE, 9, testVariable);

        writer1.start();
        writer2.start();
        writer3.start();

        writer1.stop();
        writer2.stop();
        writer3.stop();

        ml::core_t::TTime end(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Finished fast mutex lock throughput test at "
                 << ml::core::CTimeUtils::toTimeString(end));

        LOG_INFO(<< "Fast mutex lock throughput test with test size "
                 << TEST_SIZE << " took " << (end - start) << " seconds");

        LOG_INFO(<< "Fast mutex lock protected variable incremented to " << testVariable);

        BOOST_REQUIRE_EQUAL(TEST_SIZE * (1 + 5 + 9), testVariable);
    }
    {
        std::uint32_t testVariable(0);
        ml::core::CMutex mutex;

        ml::core_t::TTime start(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Starting mutex lock throughput test at "
                 << ml::core::CTimeUtils::toTimeString(start));

        CMutexProtectedAdder writer1(mutex, 0, TEST_SIZE, 1, testVariable);
        CMutexProtectedAdder writer2(mutex, 0, TEST_SIZE, 5, testVariable);
        CMutexProtectedAdder writer3(mutex, 0, TEST_SIZE, 9, testVariable);

        writer1.start();
        writer2.start();
        writer3.start();

        writer1.stop();
        writer2.stop();
        writer3.stop();

        ml::core_t::TTime end(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Finished mutex lock throughput test at "
                 << ml::core::CTimeUtils::toTimeString(end));

        LOG_INFO(<< "Mutex lock throughput test with test size " << TEST_SIZE
                 << " took " << (end - start) << " seconds");

        LOG_INFO(<< "Mutex lock protected variable incremented to " << testVariable);

        BOOST_REQUIRE_EQUAL(TEST_SIZE * (1 + 5 + 9), testVariable);
    }
    {
        std::uint32_t testVariable(0);
        ml::core::CReadWriteLock readWriteLock;

        ml::core_t::TTime start(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Starting read-write lock throughput test at "
                 << ml::core::CTimeUtils::toTimeString(start));

        CWriteLockProtectedAdder writer1(readWriteLock, 0, TEST_SIZE, 1, testVariable);
        CWriteLockProtectedAdder writer2(readWriteLock, 0, TEST_SIZE, 5, testVariable);
        CWriteLockProtectedAdder writer3(readWriteLock, 0, TEST_SIZE, 9, testVariable);

        writer1.start();
        writer2.start();
        writer3.start();

        writer1.stop();
        writer2.stop();
        writer3.stop();

        ml::core_t::TTime end(ml::core::CTimeUtils::now());
        LOG_INFO(<< "Finished read-write lock throughput test at "
                 << ml::core::CTimeUtils::toTimeString(end));

        LOG_INFO(<< "Read-write lock throughput test with test size "
                 << TEST_SIZE << " took " << (end - start) << " seconds");

        LOG_INFO(<< "Write lock protected variable incremented to " << testVariable);

        BOOST_REQUIRE_EQUAL(TEST_SIZE * (1 + 5 + 9), testVariable);
    }
}

BOOST_AUTO_TEST_SUITE_END()
