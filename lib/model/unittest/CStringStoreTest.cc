/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CThread.h>

#include <model/CStringStore.h>

#include <boost/optional.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <tuple>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::core::CStoredStringPtr)

BOOST_AUTO_TEST_SUITE(CStringStoreTest)

using namespace ml;
using namespace model;

namespace {
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;
using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;
using TStrCPtrUSet = boost::unordered_set<const std::string*>;

class CStringThread : public core::CThread {
private:
    using TSizeSizeStrTr = std::tuple<std::size_t, std::size_t, core::CStoredStringPtr>;
    using TOptionalSizeSizeStrTr = boost::optional<TSizeSizeStrTr>;

public:
    CStringThread(std::size_t i, const TStrVec& strings)
        : m_I(i), m_Strings(strings) {}

    void uniques(TStrCPtrUSet& result) const {
        result.insert(m_UniquePtrs.begin(), m_UniquePtrs.end());
    }

    void propagateLastDetectedMismatch() {
        if (m_LastMismatch.has_value()) {
            // Reconstruct the original local variables to make any
            // assertion failure message easier to understand
            std::size_t i{0};
            std::size_t n{0};
            core::CStoredStringPtr p;
            std::tie(i, n, p) = *m_LastMismatch;
            BOOST_REQUIRE_EQUAL(m_Strings[i % n], *p);
        }
    }

    void clearPtrs() {
        m_UniquePtrs.clear();
        m_Ptrs.clear();
    }

private:
    virtual void run() {
        std::size_t n = m_Strings.size();
        for (std::size_t i = m_I; i < 1000; ++i) {
            m_Ptrs.push_back(core::CStoredStringPtr());
            m_Ptrs.back() = CStringStore::names().get(m_Strings[i % n]);
            m_UniquePtrs.insert(m_Ptrs.back().get());
            if (isMismatch(i, n, m_Ptrs.back())) {
                break;
            }
        }
        for (std::size_t i = m_I; i < 1000000; ++i) {
            core::CStoredStringPtr p = CStringStore::names().get(m_Strings[i % n]);
            m_UniquePtrs.insert(p.get());
            if (isMismatch(i, n, p)) {
                break;
            }
        }
    }

    virtual void shutdown() {}

    bool isMismatch(std::size_t i, std::size_t n, const core::CStoredStringPtr& p) {
        if (m_Strings[i % n] != *p) {
            m_LastMismatch = TSizeSizeStrTr{i, n, p};
            return true;
        }
        return false;
    }

private:
    const std::size_t m_I;
    const TStrVec m_Strings;
    TStoredStringPtrVec m_Ptrs;
    TStrCPtrUSet m_UniquePtrs;
    TOptionalSizeSizeStrTr m_LastMismatch;
};
}

class CTestFixture {
public:
    CTestFixture() {
        // Other test suites also use the string store, and it will mess up the
        // tests in this suite if the string store is not empty when they start
        CStringStore::names().clearEverythingTestOnly();
        CStringStore::influencers().clearEverythingTestOnly();
    }
};

BOOST_FIXTURE_TEST_CASE(testStringStore, CTestFixture) {
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

    LOG_DEBUG(<< "*** CStringStoreTest ***");
    {
        LOG_DEBUG(<< "Testing basic insert");
        core::CStoredStringPtr pG = CStringStore::names().get("Gragnano");
        BOOST_TEST_REQUIRE(pG);
        BOOST_REQUIRE_EQUAL(std::string("Gragnano"), *pG);

        core::CStoredStringPtr pG2 = CStringStore::names().get("Gragnano");
        BOOST_TEST_REQUIRE(pG2);
        BOOST_REQUIRE_EQUAL(std::string("Gragnano"), *pG2);
        BOOST_REQUIRE_EQUAL(pG.get(), pG2.get());
        BOOST_REQUIRE_EQUAL(*pG, *pG2);

        BOOST_REQUIRE_EQUAL(std::size_t(1), CStringStore::names().m_Strings.size());
    }
    BOOST_REQUIRE_EQUAL(std::size_t(1), CStringStore::names().m_Strings.size());
    CStringStore::names().pruneNotThreadSafe();
    BOOST_REQUIRE_EQUAL(std::size_t(0), CStringStore::names().m_Strings.size());

    {
        LOG_DEBUG(<< "Testing multi-threaded");

        using TThreadPtr = std::shared_ptr<CStringThread>;
        using TThreadVec = std::vector<TThreadPtr>;
        TThreadVec threads;
        for (std::size_t i = 0; i < 20; ++i) {
            threads.emplace_back(new CStringThread(i, strings));
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            BOOST_TEST_REQUIRE(threads[i]->start());
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            BOOST_TEST_REQUIRE(threads[i]->waitForFinish());
        }

        BOOST_REQUIRE_EQUAL(strings.size(), CStringStore::names().m_Strings.size());
        CStringStore::names().pruneNotThreadSafe();
        BOOST_REQUIRE_EQUAL(strings.size(), CStringStore::names().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                            CStringStore::influencers().m_Strings.size());

        for (std::size_t i = 0; i < threads.size(); ++i) {
            // Propagate problems to main testing thread
            threads[i]->propagateLastDetectedMismatch();
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i]->clearPtrs();
        }

        BOOST_REQUIRE_EQUAL(strings.size(), CStringStore::names().m_Strings.size());
        CStringStore::names().pruneNotThreadSafe();
        BOOST_REQUIRE_EQUAL(std::size_t(0), CStringStore::names().m_Strings.size());
        threads.clear();
        BOOST_REQUIRE_EQUAL(std::size_t(0), CStringStore::names().m_Strings.size());
    }
    {
        LOG_DEBUG(<< "Testing multi-threaded string duplication rate");

        TStrVec lotsOfStrings;
        for (std::size_t i = 0u; i < 1000; ++i) {
            lotsOfStrings.push_back(core::CStringUtils::typeToString(i));
        }

        using TThreadPtr = std::shared_ptr<CStringThread>;
        using TThreadVec = std::vector<TThreadPtr>;
        TThreadVec threads;
        for (std::size_t i = 0; i < 20; ++i) {
            threads.emplace_back(new CStringThread(i * 50, lotsOfStrings));
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            BOOST_TEST_REQUIRE(threads[i]->start());
        }
        for (std::size_t i = 0; i < threads.size(); ++i) {
            BOOST_TEST_REQUIRE(threads[i]->waitForFinish());
        }

        for (std::size_t i = 0; i < threads.size(); ++i) {
            // Propagate exceptions to main testing thread
            threads[i]->propagateLastDetectedMismatch();
        }

        TStrCPtrUSet uniques;
        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i]->uniques(uniques);
        }
        LOG_DEBUG(<< "unique counts = " << uniques.size());
        BOOST_TEST_REQUIRE(uniques.size() < 20000);

        // Tidy up
        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i]->clearPtrs();
        }
        CStringStore::names().pruneNotThreadSafe();
    }
}

BOOST_FIXTURE_TEST_CASE(testMemUsage, CTestFixture) {
    std::string shortStr("short");
    std::string longStr("much much longer than the short string");

    std::size_t origMemUse = CStringStore::names().memoryUsage();
    LOG_DEBUG(<< "Original memory usage: " << origMemUse);
    BOOST_TEST_REQUIRE(origMemUse > 0);

    std::size_t inUseMemUse{0};
    {
        core::CStoredStringPtr shortPtr = CStringStore::names().get(shortStr);
        core::CStoredStringPtr longPtr = CStringStore::names().get(longStr);

        inUseMemUse = CStringStore::names().memoryUsage();
        LOG_DEBUG(<< "Memory usage with in-use pointers: " << inUseMemUse);
        BOOST_TEST_REQUIRE(inUseMemUse >
                           origMemUse + shortStr.length() + longStr.length());

        // This pruning should have no effect, as there are external pointers to
        // the contents
        CStringStore::names().pruneNotThreadSafe();
        BOOST_REQUIRE_EQUAL(inUseMemUse, CStringStore::names().memoryUsage());
    }

    // Previously obtained pointers being destroyed should not on its own reduce
    // the memory usage of the string store
    BOOST_REQUIRE_EQUAL(inUseMemUse, CStringStore::names().memoryUsage());

    // There are no external references, so this should remove values
    CStringStore::names().pruneNotThreadSafe();
    std::size_t prunedMemUse = CStringStore::names().memoryUsage();
    LOG_DEBUG(<< "Pruned memory usage: " << prunedMemUse);
    BOOST_TEST_REQUIRE(prunedMemUse < inUseMemUse - shortStr.length() - longStr.length());

    CStringStore::names().clearEverythingTestOnly();
    BOOST_REQUIRE_EQUAL(origMemUse, CStringStore::names().memoryUsage());
}

BOOST_AUTO_TEST_SUITE_END()
