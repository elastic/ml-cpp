/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CStateMachine.h>

#include <core/CContainerPrinter.h>
#include <core/CFastMutex.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>
#include <core/UnwrapRef.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <sstream>

namespace ml {
namespace core {
namespace {

// CStateMachine
//const std::string MACHINE_TAG("a"); No longer used
const core::TPersistenceTag STATE_TAG("b", "state");

// CStateMachine::SMachine
const std::string ALPHABET_TAG("a");
const std::string STATES_TAG("b");
const std::string TRANSITION_FUNCTION_TAG("c");

std::size_t BAD_MACHINE = boost::numeric::bounds<std::size_t>::highest();
CFastMutex mutex;
}

void CStateMachine::expectedNumberMachines(std::size_t number) {
    CScopedFastLock lock(mutex);
    ms_Machines.capacity(number);
}

CStateMachine CStateMachine::create(const TStrVec& alphabet,
                                    const TStrVec& states,
                                    const TSizeVecVec& transitionFunction,
                                    std::size_t state) {
    // Validate that the alphabet, states, transition function,
    // and initial state are consistent.

    CStateMachine result;

    if (state >= states.size()) {
        LOG_ERROR(<< "Invalid initial state: " << state);
        return result;
    }
    if (alphabet.empty() || alphabet.size() != transitionFunction.size()) {
        LOG_ERROR(<< "Bad alphabet: " << core::CContainerPrinter::print(alphabet));
        return result;
    }
    for (const auto& function : transitionFunction) {
        if (states.size() != function.size()) {
            LOG_ERROR(<< "Bad transition function row: "
                      << core::CContainerPrinter::print(function));
            return result;
        }
    }

    // We use the standard double lock pattern with an atomic size to
    // indicate that a machine is ready to use. Because we are storing
    // the machines in a custom deque container a concurrent push_back
    // doesn't invalidate access to any other existing machine.

    SLookupMachine machine(alphabet, states, transitionFunction);
    std::size_t size = ms_Machines.size();
    std::size_t m = find(0, size, machine);
    if (m == size || machine != ms_Machines[m]) {
        CScopedFastLock lock(mutex);
        m = find(0, ms_Machines.size(), machine);
        if (m == ms_Machines.size()) {
            ms_Machines.push_back(SMachine(alphabet, states, transitionFunction));
        }
    }

    result.m_Machine = m;
    result.m_State = state;
    return result;
}

bool CStateMachine::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                           const TSizeSizeMap& mapping) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(STATE_TAG, m_State)
    } while (traverser.next());
    if (mapping.size() > 0) {
        auto mapped = mapping.find(m_State);
        if (mapped != mapping.end()) {
            m_State = mapped->second;
        } else {
            LOG_ERROR(<< "Bad mapping '" << core::CContainerPrinter::print(mapping)
                      << "' state = " << m_State);
            return false;
        }
    }
    return true;
}

void CStateMachine::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(STATE_TAG, m_State);
}

bool CStateMachine::bad() const {
    return m_Machine == BAD_MACHINE;
}

bool CStateMachine::apply(std::size_t symbol) {
    const TSizeVecVec& table = ms_Machines[m_Machine].s_TransitionFunction;

    if (symbol >= table.size()) {
        LOG_ERROR(<< "Bad symbol " << symbol << " not in alphabet [" << table.size() << "]");
        return false;
    }
    if (m_State >= table[symbol].size()) {
        LOG_ERROR(<< "Bad state " << m_State << " not in states ["
                  << table[symbol].size() << "]");
        return false;
    }

    m_State = table[symbol][m_State];
    return true;
}

std::size_t CStateMachine::state() const {
    return m_State;
}

std::string CStateMachine::printState(std::size_t state) const {
    if (state >= ms_Machines[m_Machine].s_States.size()) {
        return "State Not Found";
    }
    return ms_Machines[m_Machine].s_States[state];
}

std::string CStateMachine::printSymbol(std::size_t symbol) const {
    if (symbol >= ms_Machines[m_Machine].s_Alphabet.size()) {
        return "Symbol Not Found";
    }
    return ms_Machines[m_Machine].s_Alphabet[symbol];
}

uint64_t CStateMachine::checksum() const {
    return CHashing::hashCombine(static_cast<uint64_t>(m_Machine),
                                 static_cast<uint64_t>(m_State));
}

std::size_t CStateMachine::numberMachines() {
    CScopedFastLock lock(mutex);
    return ms_Machines.size();
}

void CStateMachine::clear() {
    CScopedFastLock lock(mutex);
    ms_Machines.clear();
}

std::size_t CStateMachine::find(std::size_t begin, std::size_t end, const SLookupMachine& machine) {
    for (std::size_t i = begin; i < end; ++i) {
        if (machine == ms_Machines[i]) {
            return i;
        }
    }
    return end;
}

CStateMachine::CStateMachine() : m_Machine(BAD_MACHINE), m_State(0) {
}

CStateMachine::SMachine::SMachine(const TStrVec& alphabet,
                                  const TStrVec& states,
                                  const TSizeVecVec& transitionFunction)
    : s_Alphabet(alphabet), s_States(states), s_TransitionFunction(transitionFunction) {
}

CStateMachine::SMachine::SMachine(const SMachine& other)
    : s_Alphabet(other.s_Alphabet), s_States(other.s_States),
      s_TransitionFunction(other.s_TransitionFunction) {
}

CStateMachine::SLookupMachine::SLookupMachine(const TStrVec& alphabet,
                                              const TStrVec& states,
                                              const TSizeVecVec& transitionFunction)
    : s_Alphabet(alphabet), s_States(states), s_TransitionFunction(transitionFunction) {
}

bool CStateMachine::SLookupMachine::operator==(const SMachine& rhs) const {
    return ml::core::unwrap_ref(s_TransitionFunction) == rhs.s_TransitionFunction &&
           ml::core::unwrap_ref(s_Alphabet) == rhs.s_Alphabet &&
           ml::core::unwrap_ref(s_States) == rhs.s_States;
}

CStateMachine::CMachineDeque::CMachineDeque()
    : m_Capacity(DEFAULT_CAPACITY), m_NumberMachines(0) {
    m_Machines.push_back(TMachineVec());
    m_Machines.back().reserve(m_Capacity);
}

void CStateMachine::CMachineDeque::capacity(std::size_t capacity) {
    m_Capacity = capacity;
}

const CStateMachine::SMachine& CStateMachine::CMachineDeque::operator[](std::size_t pos_) const {
    std::size_t pos{pos_};
    for (const auto& machines : m_Machines) {
        if (pos < machines.size()) {
            return machines[pos];
        }
        pos -= machines.size();
    }
    LOG_ABORT(<< "Invalid index '" << pos_ << "'");
}

std::size_t CStateMachine::CMachineDeque::size() const {
    return m_NumberMachines.load(std::memory_order_acquire);
}

void CStateMachine::CMachineDeque::push_back(const SMachine& machine) {
    if (m_Machines.back().size() == m_Capacity) {
        m_Machines.push_back(TMachineVec());
        m_Machines.back().reserve(m_Capacity);
    }
    m_Machines.back().push_back(machine);
    m_NumberMachines.store(this->size() + 1, std::memory_order_release);
}

void CStateMachine::CMachineDeque::clear() {
    m_NumberMachines.store(0);
    m_Machines.clear();
    m_Machines.push_back(TMachineVec());
    m_Machines.back().reserve(m_Capacity);
}

CStateMachine::CMachineDeque CStateMachine::ms_Machines;
}
}
