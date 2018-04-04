/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
