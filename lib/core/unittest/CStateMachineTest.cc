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

#include "CStateMachineTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStateMachine.h>
#include <core/CThread.h>

#include <test/CRandomNumbers.h>

#include <algorithm>
#include <vector>

#include <boost/range.hpp>

using namespace ml;

namespace {

typedef std::vector<std::size_t> TSizeVec;
typedef std::vector<TSizeVec> TSizeVecVec;
typedef std::vector<std::string> TStrVec;

class CStateMachineClearer : core::CStateMachine {
    public:
        static void clear(void) {
            core::CStateMachine::clear();
        }
};

struct SMachine {
    TStrVec s_Alphabet;
    TStrVec s_States;
    TSizeVecVec s_TransitionFunction;

    bool operator==(const SMachine &rhs) const {
        return s_Alphabet == rhs.s_Alphabet &&
               s_States == rhs.s_States &&
               s_TransitionFunction == rhs.s_TransitionFunction;
    }

    bool operator<(const SMachine &rhs) const {
        return s_Alphabet < rhs.s_Alphabet ||
               (s_Alphabet == rhs.s_Alphabet && s_States < rhs.s_States) ||
               (   s_Alphabet == rhs.s_Alphabet &&
                   s_States == rhs.s_States &&
                   s_TransitionFunction < rhs.s_TransitionFunction);
    }
};

typedef std::vector<SMachine> TMachineVec;

class CTestThread : public core::CThread {
    public:
        typedef boost::shared_ptr<CppUnit::Exception> TCppUnitExceptionP;

    public:
        CTestThread(const TMachineVec &machines) :
            m_Machines(machines),
            m_Failures(0) {
        }

        std::size_t failures(void) const {
            return m_Failures;
        }

        const TSizeVec &states(void) const {
            return m_States;
        }

    private:
        virtual void run(void) {
            std::size_t n = 10000;
            m_States.reserve(n);
            TSizeVec machine;
            for (std::size_t i = 0u; i < n; ++i) {
                m_Rng.generateUniformSamples(0, m_Machines.size(), 1, machine);
                core::CStateMachine sm = core::CStateMachine::create(m_Machines[machine[0]].s_Alphabet,
                                                                     m_Machines[machine[0]].s_States,
                                                                     m_Machines[machine[0]].s_TransitionFunction,
                                                                     0); // initial state
                if (!sm.apply(0)) {
                    ++m_Failures;
                }
                m_States.push_back(sm.state());
            }
        }

        virtual void shutdown(void) {
        }

    private:
        test::CRandomNumbers m_Rng;
        TMachineVec m_Machines;
        std::size_t m_Failures;
        TSizeVec m_States;
};

void randomMachines(std::size_t n, TMachineVec &result) {
    std::string states[] =
    {
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"
    };
    std::string alphabet[] =
    {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    };

    test::CRandomNumbers rng;

    TSizeVec ns;
    rng.generateUniformSamples(2, boost::size(states), n, ns);

    TSizeVec na;
    rng.generateUniformSamples(1, boost::size(alphabet), n, na);

    result.resize(n);
    for (std::size_t i = 0u; i < n; ++i) {
        result[i].s_States.assign(states, states + ns[i]);
        result[i].s_Alphabet.assign(alphabet, alphabet + na[i]);
        result[i].s_TransitionFunction.resize(na[i]);
        for (std::size_t j = 0u; j < na[i]; ++j) {
            rng.generateUniformSamples(0, ns[i], ns[i], result[i].s_TransitionFunction[j]);
        }

        std::next_permutation(boost::begin(states), boost::end(states));
        std::next_permutation(boost::begin(alphabet), boost::end(alphabet));
    }
}

}

void CStateMachineTest::testBasics(void) {
    // Test errors on create.

    // Test transitions.

    TMachineVec machines;
    randomMachines(5, machines);

    for (std::size_t m = 0u; m < machines.size(); ++m) {
        LOG_DEBUG("machine " << m);
        for (std::size_t i = 0u; i < machines[m].s_Alphabet.size(); ++i) {
            for (std::size_t j = 0u; j < machines[m].s_States.size(); ++j) {
                core::CStateMachine sm = core::CStateMachine::create(machines[m].s_Alphabet,
                                                                     machines[m].s_States,
                                                                     machines[m].s_TransitionFunction,
                                                                     j); // initial state

                const std::string &oldState = machines[m].s_States[j];

                sm.apply(i);

                const std::string &newState = machines[m].s_States[sm.state()];

                LOG_DEBUG("  " << oldState << " -> " << newState);
                CPPUNIT_ASSERT_EQUAL(machines[m].s_States[machines[m].s_TransitionFunction[i][j]],
                                     sm.printState(sm.state()));
            }
        }
    }
}

void CStateMachineTest::testPersist(void) {
    // Check persist maintains the checksum and is idempotent.

    TMachineVec machine;
    randomMachines(2, machine);

    core::CStateMachine original = core::CStateMachine::create(machine[0].s_Alphabet,
                                                               machine[0].s_States,
                                                               machine[0].s_TransitionFunction,
                                                               1); // initial state
    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        original.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("State machine XML representation:\n" << origXml);

    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    core::CStateMachine restored = core::CStateMachine::create(machine[1].s_Alphabet,
                                                               machine[1].s_States,
                                                               machine[1].s_TransitionFunction,
                                                               0); // initial state
    traverser.traverseSubLevel(boost::bind(&core::CStateMachine::acceptRestoreTraverser, &restored, _1));

    CPPUNIT_ASSERT_EQUAL(original.checksum(), restored.checksum());
    std::string newXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        restored.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CStateMachineTest::testMultithreaded(void) {
    // Check that we create each machine once and we don't get any
    // errors updating due to stale reads.

    CStateMachineClearer::clear();

    TMachineVec machines;
    randomMachines(100, machines);

    LOG_DEBUG("# machines = " << machines.size());

    std::sort(machines.begin(), machines.end());
    machines.erase(std::unique(machines.begin(), machines.end()), machines.end());

    typedef boost::shared_ptr<CTestThread> TThreadPtr;
    typedef std::vector<TThreadPtr> TThreadVec;
    TThreadVec threads;
    for (std::size_t i = 0u; i < 20; ++i) {
        threads.push_back(TThreadPtr(new CTestThread(machines)));
    }
    for (std::size_t i = 0u; i < threads.size(); ++i) {
        CPPUNIT_ASSERT(threads[i]->start());
    }
    for (std::size_t i = 0u; i < threads.size(); ++i) {
        CPPUNIT_ASSERT(threads[i]->waitForFinish());
    }
    for (std::size_t i = 0u; i < threads.size(); ++i) {
        // No failed reads.
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), threads[i]->failures());
    }
    for (std::size_t i = 1u; i < threads.size(); ++i) {
        // No wrong reads.
        CPPUNIT_ASSERT(threads[i]->states() == threads[i-1]->states());
    }
    // No duplicates.
    CPPUNIT_ASSERT_EQUAL(machines.size(), core::CStateMachine::numberMachines());
}

CppUnit::Test *CStateMachineTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStateMachineTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CStateMachineTest>(
                               "CStateMachineTest::testBasics",
                               &CStateMachineTest::testBasics) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStateMachineTest>(
                               "CStateMachineTest::testPersist",
                               &CStateMachineTest::testPersist) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStateMachineTest>(
                               "CStateMachineTest::testMultithreaded",
                               &CStateMachineTest::testMultithreaded) );

    return suiteOfTests;
}
