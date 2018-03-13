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
#ifndef INCLUDED_ml_core_CStat_h
#define INCLUDED_ml_core_CStat_h

#include <core/CNonCopyable.h>

#include <atomic>

namespace ml {
namespace core {

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
class CORE_EXPORT CStat : private CNonCopyable {
public:
    //! Default constructor
    CStat(void);

    //! Add the value 1 to this stat
    void increment(void);

    //! Add some value to this stat
    void increment(uint64_t value);

    //! Remove the value 1 from this stat
    void decrement(void);

    //! Set the stat to this new value
    void set(uint64_t value);

    //! Get the value of this stat
    uint64_t value(void) const;

private:
    //! The counter value of this stat
    std::atomic_uint_fast64_t m_Value;
};

}// core
}// ml

#endif// INCLUDED_ml_core_CStat_h
