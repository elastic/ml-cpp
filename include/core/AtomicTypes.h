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
#ifndef INCLUDED_ml_core_AtomicTypes_h
#define INCLUDED_ml_core_AtomicTypes_h

//! \brief
//! Header file that defines atomic types.
//!
//! Depending on the platform, these are either brought from the
//! C++ standard library header <atomic> or the Boost header
//! <boost/atomic.hpp>.

#ifdef Windows

// Windows cannot yet use std::atomic because Atomic support is buggy prior to
// Visual Studio 2015

#include <boost/atomic.hpp>
#define atomic_t boost

#else

#include <atomic>
#define atomic_t std

#endif

#endif // INCLUDED_ml_core_AtomicTypes_h
