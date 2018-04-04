/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStat_h
#define INCLUDED_ml_core_CStat_h

#include <core/CNonCopyable.h>

#include <atomic>


namespace ml
{
namespace core
{

//! \brief
//! An atomic statistic object
//!
//! DESCRIPTION:\n
//! A wrapper for an atomic uint64_t
//!
//! This provides a thread-safe way to aggregate runtime counts
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable - it only makes sense to have one instance of this
//!
class CORE_EXPORT CStat : private CNonCopyable
{
    public:
        //! Default constructor
        CStat();

        //! Add the value 1 to this stat
        void increment();

        //! Add some value to this stat
        void increment(uint64_t value);

        //! Remove the value 1 from this stat
        void decrement();

        //! Set the stat to this new value
        void set(uint64_t value);

        //! Get the value of this stat
        uint64_t value() const;

    private:
        //! The counter value of this stat
        std::atomic_uint_fast64_t m_Value;
};


} // core
} // ml

#endif // INCLUDED_ml_core_CStat_h
