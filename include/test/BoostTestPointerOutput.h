/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_BoostTestPointerOutput_h
#define INCLUDED_ml_test_BoostTestPointerOutput_h

#include <memory>
#include <ostream>

//! \brief
//! Operators to output unique_ptr and nullptr_t.
//!
//! DESCRIPTION:\n
//! Boost.Test likes to print values that fail assertions.
//! It does this using operator<<.  It is common that we
//! assert on pointer values, yet there is no operator<<
//! for unique_ptr until C++20 and no operator<< for
//! nullptr until C++17.  This file adds these until such
//! time as we are using a sufficiently modern standard.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Memory addresses are printed, not the dereferenced
//! object.  Functions need to be in the std namespace so
//! that they can be found via argument-dependent lookup
//! (since unique_ptr and nullptr_t are in the std
//! namespace).  This is not ideal, hence this functionality
//! is in the test library to limit usage to unit tests.
//!

// TODO: delete this source file once all compilers are using C++20
namespace std {

// TODO: remove once all compilers are using C++17
template<class C, class T>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     std::nullptr_t p) {
    return os << static_cast<void*>(p);
}

// TODO: remove once all compilers are using C++20
template <class C, class T, class Y, class D>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const std::unique_ptr<Y, D>& p) {
    return os << p.get();
}
}

#endif // INCLUDED_ml_test_BoostTestPointerOutput_h
