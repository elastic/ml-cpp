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

#include "CStringStoreTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CThread.h>

#include <model/CStringStore.h>

#include <cppunit/Exception.h>

#include <boost/shared_ptr.hpp>

using namespace ml;
using namespace model;

namespace {
typedef std::vector<std::size_t> TSizeVec;
typedef std::vector<std::string> TStrVec;
typedef std::vector<core::CStoredStringPtr> TStoredStringPtrVec;
typedef boost::unordered_set<const std::string *> TStrCPtrUSet;

class CStringThread : public core::CThread {
public:
    typedef boost::shared_ptr<CppUnit::Exception> TCppUnitExceptionP;

public:
    CStringThread(std::size_t i, const TStrVec &strings) : m_I(i), m_Strings(strings) {}

    void uniques(TStrCPtrUSet &result) const {
        result.insert(m_UniquePtrs.begin(), m_UniquePtrs.end());
    }

    void propagateLastThreadAssert(void) {
        if (m_LastException != 0) {
            throw *m_LastException;
        }
    }

    void clearPtrs(void) {
        m_UniquePtrs.clear();
        m_Ptrs.clear();
    }

private:
    virtual void run(void) {
        try {
            std::size_t n = m_Strings.size();
            for (std::size_t i = m_I; i < 1000; ++i) {
                m_Ptrs.push_back(core::CStoredStringPtr());
                m_Ptrs.back() = CStringStore::names().get(m_Strings[i % n]);
                m_UniquePtrs.insert(m_Ptrs.back().get());
                CPPUNIT_ASSERT_EQUAL(m_Strings[i % n], *m_Ptrs.back());
            }
            for (std::size_t i = m_I; i < 1000000; ++i) {
                core::CStoredStringPtr p = CStringStore::names().get(m_Strings[i % n]);
                m_UniquePtrs.insert(p.get());
                CPPUNIT_ASSERT_EQUAL(m_Strings[i % n], *p);
            }
        }
        // CppUnit won't automatically catch the exceptions thrown by
        // assertions in newly created threads, so propagate manually
        catch (CppUnit::Exception &e) {
            m_LastException.reset(new CppUnit::Exception(e));
        }
    }

    virtual void shutdown(void) {}

private:
    std::size_t m_I;
    TStrVec m_Strings;
    TStoredStringPtrVec m_Ptrs;
    TStrCPtrUSet m_UniquePtrs;
    TCppUnitExceptionP m_LastException;
};
}

void CStringStoreTest::setUp(void) {
    // Other test suites also use the string store, and it will mess up the
    // tests in this suite if the string store is not empty when they start
    CStringStore::names().clearEverythingTestOnly();
    CStringStore::influencers().clearEverythingTestOnly();
}

void CStringStoreTest::testStringStore(void) {
    TStrVec strings;
    strings.emplace_back("Milano");
    strings.emplace_back("Monza");
    strings.emplace_back("Amalfi");
    strings.emplace_back("Pompei");
    strings.emplace_back("Gragnano");
    strings.emplace_back("Roma");
    strings.emplace_back("Bologna");
    strings.emplace_back("Torina");
    strings.emplace_back("Napoli");
    strings.emplace_back("Rimini");
    strings.emplace_back("Genova");
    strings.emplace_back("Capri");
    strings.emplace_back("Ravello");
    strings.emplace_back("Reggio");
    strings.emplace_back("Palermo");
    strings.emplace_back("Focaccino");

    LOG_DEBUG("*** CStringStoreTest ***");
    {
        LOG_DEBUG("Testing basic insert");
        core::CStoredStringPtr pG = CStringStore::names().get("Gragnano");
        CPPUNIT_ASSERT(pG);
        CPPUNIT_ASSERT_EQUAL(std::string("Gragnano"), *pG);

        core::CStoredStringPtr pG2 = CStringStore::names().get("Gragnano");
        CPPUNIT_ASSERT(pG2);
        CPPUNIT_ASSERT_EQUAL(std::string("Gragnano"), *pG2);
        CPPUNIT_ASSERT_EQUAL(pG.get(), pG2.get());
        CPPUNIT_ASSERT_EQUAL(*pG, *pG2);

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), CStringStore::names().m_Strings.size());
    }
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), CStringStore::names().m_Strings.size());
    CStringStore::names().pruneNotThreadSafe();
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), CStringStore::names().m_Strings.size());

    {
        LOG_DEBUG("Testing multi-threaded");

        typedef boost::shared_ptr<CStringThread> TThreadPtr;
        typedef std::vector<TThreadPtr> TThreadVec;
        TThreadVec threads;
        for (std::size_t i = 0; i < 20; ++i) {
            threads.emplace_back(new CStringThread(i, strings));
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            CPPUNIT_ASSERT(threads[i]->start());
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            CPPUNIT_ASSERT(threads[i]->waitForFinish());
        }

        CPPUNIT_ASSERT_EQUAL(strings.size(), CStringStore::names().m_Strings.size());
        CStringStore::names().pruneNotThreadSafe();
        CPPUNIT_ASSERT_EQUAL(strings.size(), CStringStore::names().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), CStringStore::influencers().m_Strings.size());

        for (std::size_t i = 0; i < threads.size(); ++i) {
            // CppUnit won't automatically catch the exceptions thrown by
            // assertions in newly created threads, so propagate manually
            threads[i]->propagateLastThreadAssert();
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i]->clearPtrs();
        }

        CPPUNIT_ASSERT_EQUAL(strings.size(), CStringStore::names().m_Strings.size());
        CStringStore::names().pruneNotThreadSafe();
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), CStringStore::names().m_Strings.size());
        threads.clear();
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), CStringStore::names().m_Strings.size());
    }
    {
        LOG_DEBUG("Testing multi-threaded string duplication rate");

        TStrVec lotsOfStrings;
        for (std::size_t i = 0u; i < 1000; ++i) {
            lotsOfStrings.push_back(core::CStringUtils::typeToString(i));
        }

        typedef boost::shared_ptr<CStringThread> TThreadPtr;
        typedef std::vector<TThreadPtr> TThreadVec;
        TThreadVec threads;
        for (std::size_t i = 0; i < 20; ++i) {
            threads.emplace_back(new CStringThread(i * 50, lotsOfStrings));
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            CPPUNIT_ASSERT(threads[i]->start());
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            CPPUNIT_ASSERT(threads[i]->waitForFinish());
        }

        for (std::size_t i = 0; i < threads.size(); ++i) {
            // CppUnit won't automatically catch the exceptions thrown by
            // assertions in newly created threads, so propagate manually
            threads[i]->propagateLastThreadAssert();
        }

        TStrCPtrUSet uniques;
        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i]->uniques(uniques);
        }
        LOG_DEBUG("unique counts = " << uniques.size());
        CPPUNIT_ASSERT(uniques.size() < 20000);

        // Tidy up
        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i]->clearPtrs();
        }
        CStringStore::names().pruneNotThreadSafe();
    }
}

void CStringStoreTest::testMemUsage(void) {
    std::string shortStr("short");
    std::string longStr("much much longer than the short string");

    std::size_t origMemUse = CStringStore::names().memoryUsage();
    LOG_DEBUG("Original memory usage: " << origMemUse);
    CPPUNIT_ASSERT(origMemUse > 0);

    std::size_t inUseMemUse{0};
    {
        core::CStoredStringPtr shortPtr = CStringStore::names().get(shortStr);
        core::CStoredStringPtr longPtr = CStringStore::names().get(longStr);

        inUseMemUse = CStringStore::names().memoryUsage();
        LOG_DEBUG("Memory usage with in-use pointers: " << inUseMemUse);
        CPPUNIT_ASSERT(inUseMemUse > origMemUse + shortStr.length() + longStr.length());

        // This pruning should have no effect, as there are external pointers to
        // the contents
        CStringStore::names().pruneNotThreadSafe();
        CPPUNIT_ASSERT_EQUAL(inUseMemUse, CStringStore::names().memoryUsage());
    }

    // Previously obtained pointers being destroyed should not on its own reduce
    // the memory usage of the string store
    CPPUNIT_ASSERT_EQUAL(inUseMemUse, CStringStore::names().memoryUsage());

    // There are no external references, so this should remove values
    CStringStore::names().pruneNotThreadSafe();
    std::size_t prunedMemUse = CStringStore::names().memoryUsage();
    LOG_DEBUG("Pruned memory usage: " << prunedMemUse);
    CPPUNIT_ASSERT(prunedMemUse < inUseMemUse - shortStr.length() - longStr.length());

    CStringStore::names().clearEverythingTestOnly();
    CPPUNIT_ASSERT_EQUAL(origMemUse, CStringStore::names().memoryUsage());
}

CppUnit::Test *CStringStoreTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStringStoreTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CStringStoreTest>(
        "CStringStoreTest::testStringStore", &CStringStoreTest::testStringStore));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringStoreTest>(
        "CStringStoreTest::testMemUsage", &CStringStoreTest::testMemUsage));

    return suiteOfTests;
}
