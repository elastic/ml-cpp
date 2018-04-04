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

#ifndef INCLUDED_ml_core_CStateMachine_h
#define INCLUDED_ml_core_CStateMachine_h

#include <core/AtomicTypes.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <list>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;

//! \brief A simple state machine implementation.
//!
//! DESCRIPTION:\n
//! This defines an alphabet of symbols, a set of states and a transition
//! table.
//!
//! A state machine is the tuple \f$\left(\Sigma, S, \delta, s_0\right)\f$.
//! Here,
//!   -# \f$\Sigma\f$ is an alphabet of symbols which are passed to the
//!      machine.
//!   -# \f$S\f$ are the collection of state.
//!   -# \f$\delta\f$ is the transition function \f$\delta : S \cross \Sigma \leftarrow S\f$.
//!   -# \f$s_0\f$ Is the initial state.
//!
//! IMPLEMENTATION:\n
//! Most of the state is stored statically and unique machines are identified
//! by their alphabet, states, and transition table. This state can be shared
//! by many instances of the machine and so the only state that needs to be
//! stored is the machine identifier, which is managed by the create function,
//! and the current state.
//!
//! Note I purposely haven't bothered to allow the user to register actions to
//! perform when a symbol is applied. This is because these typically need
//! additional information to be supplied and there is no way of doing that
//! at the level of this interface without using void* or equivalent. Instead
//! it is intended that the user of this class holds an instance of the state
//! machine and uses it to implement an apply function which receives additional
//! state.
//!
//! This implementation is mostly lock free since after it is created the shared
//! state of a machine is effectively constant and we can use the double locking
//! pattern in create with an atomic to pass a message to other threads that a
//! new machine is ready to use.
//!
//! \warning Currently this class is NOT thread safe.  The unlocked accesses
//! to ms_Machines can occur at the same time as ms_Machines is being expanded
//! leading to segmentation faults as the accessors traverse the deque data
//! structure while it's being changed.  Do not use this class in multithreaded
//! code until this is fixed.  The bug reference is:
//! https://github.com/elastic/machine-learning-cpp/issues/10
class CORE_EXPORT CStateMachine {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TStrVec = std::vector<std::string>;

public:
    //! Set the number of machines we expect the program to use.
    static void expectedNumberMachines(std::size_t number);

    //! Create a machine with a specified alphabet, set of states and
    //! transition function and initialize its state to \p state.
    //!
    //! \note This can fail if the supplied data are inconsistent in
    //! which case the state is set to bad.
    static CStateMachine create(const TStrVec& alphabet, const TStrVec& states, const TSizeVecVec& transitionFunction, std::size_t state);

    //! \name Persistence
    //@{
    //! Initialize by reading state from \p traverser.
    bool acceptRestoreTraverser(CStateRestoreTraverser& traverser);

    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(CStatePersistInserter& inserter) const;
    //@}

    //! Check if the machine is bad, i.e. not a valid state machine.
    bool bad() const;

    //! Apply \p symbol to the machine.
    bool apply(std::size_t symbol);

    //! Get the current state of the machine.
    std::size_t state() const;

    //! Print \p state.
    std::string printState(std::size_t state) const;

    //! Print \p symbol.
    std::string printSymbol(std::size_t symbol) const;

    //! Get a checksum of this object.
    uint64_t checksum() const;

    //! Print all the state machines.
    static std::size_t numberMachines();

protected:
    //! Clear all machines (for test only).
    static void clear();

private:
    //! \brief The state of a single machine.
    struct CORE_EXPORT SMachine {
        SMachine(const TStrVec& alphabet, const TStrVec& states, const TSizeVecVec& transitionFunction);
        SMachine(const SMachine& other);

        //! The alphabet of action symbols \f$\Sigma\f$.
        TStrVec s_Alphabet;
        //! The possible states \f$S\f$.
        TStrVec s_States;
        //! The transition table \f$\delta : \Sigma \times S \rightarrow S\f$.
        TSizeVecVec s_TransitionFunction;
    };

    //! \brief A lightweight object to lookup a single machine.
    struct CORE_EXPORT SLookupMachine : boost::equality_comparable2<SLookupMachine, SMachine> {
        SLookupMachine(const TStrVec& alphabet, const TStrVec& states, const TSizeVecVec& transitionFunction);

        //! Test if two machines are equal.
        bool operator==(const SMachine& rhs) const;

        //! The alphabet of action symbols \f$\Sigma\f$.
        const TStrVec& s_Alphabet;
        //! The possible states \f$S\f$.
        const TStrVec& s_States;
        //! The transition table \f$\delta : \Sigma \times S \rightarrow S\f$.
        const TSizeVecVec& s_TransitionFunction;
    };

    //! \brief A custom paired down std::deque like container.
    //!
    //! DESCRIPTION:\n
    //! We have rather specific requirements for this container:
    //!   -# It must be (as) random access (as possible),
    //!   -# It must be possible for push_back to occur concurrently with
    //!      lookup of an existing item in the container.
    //!
    //! IMPLEMENTATION:\n
    //! With std::deque implementations any invocation of operator[] can
    //! fail if there is a concurrent push_back. The code using this class
    //! ensures that it only ever asks for an element which already exists
    //! in the container at the start of push_back. This is possible safely
    //! with a std::list. However, since this also needs to be random access,
    //! using a vanilla std::list, which has \f$O(N)\f$ complexity for
    //! accessing the \f$N^{th}\f$ item and poor locality of reference,
    //! is not suitable. Instead we prefer to use a list of preallocated
    //! std::vectors, on which it is also safe to call push_back, for our
    //! use case, provided doing so doesn't cause them to reallocate.
    class CORE_EXPORT CMachineDeque {
    private:
        //! The default vector capacity.
        static const std::size_t DEFAULT_CAPACITY = 20;

    public:
        CMachineDeque();

        //! Set the vector capacity to \p capacity.
        void capacity(std::size_t capacity);

        //! Access the element at \p pos.
        const SMachine& operator[](std::size_t pos) const;

        //! Get the number of elements in this container.
        std::size_t size() const;

        //! Add a new element to the back of the collection.
        void push_back(const SMachine& machine);

        //! Remove all elements.
        void clear();

    private:
        using TMachineVec = std::vector<SMachine>;
        using TMachineVecList = std::list<TMachineVec>;

    private:
        //! The vector capacity.
        //!
        //! \note This should be set to slightly more than the number
        //! of distinct machines which are created by the program
        //! which uses this class.
        std::size_t m_Capacity;

        //! Get the number of available machines.
        atomic_t::atomic<std::size_t> m_NumberMachines;

        //! The actual machines.
        TMachineVecList m_Machines;
    };

private:
    CStateMachine();

    //! Try to find \p machine in the range [\p begin, \p end).
    static std::size_t find(std::size_t begin, std::size_t end, const SLookupMachine& machine);

private:
    //! The machine identifier.
    std::size_t m_Machine;
    //! The current state of the machine.
    std::size_t m_State;
    //! A complete list of available machines.
    static CMachineDeque ms_Machines;
};
}
}

#endif // INCLUDE_CORE_CSTATEMACHINE_H_
