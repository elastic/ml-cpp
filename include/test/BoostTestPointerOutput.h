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
//! Functions to output unique_ptr and nullptr_t.
//!
//! DESCRIPTION:\n
//! Boost.Test likes to print values that fail assertions.  It
//! can do this when operator<< is defined for the type being
//! asserted on.  It is common that we assert on pointer values,
//! yet there is no operator<< for unique_ptr until C++20 and
//! no operator<< for nullptr until C++17.  This file adds
//! replacements for these until such time as we are using a
//! sufficiently modern standard.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Memory addresses are printed, not the dereferenced object.
//! The intermediate boost_test_print_type function is
//! overloaded rather than operator<< itself, as it will not
//! cause an overloading ambiguity as newer compiler versions
//! add the operators as extensions to the standard (for example
//! Xcode 10.3 has added operator<< for unique_ptr even when the
//! selected standard is C++14).  Functions need to be in the
//! std namespace so that they can be found via argument-dependent
//! lookup (since unique_ptr and nullptr_t are in the std
//! namespace).  This is not ideal but the intention is that this
//! file is a temporary measure.
//!

// TODO: delete this source file once all compilers are using C++20
namespace std {

// TODO: remove once all compilers are using C++17
inline std::ostream& boost_test_print_type(std::ostream& os, std::nullptr_t p) {
    return os << static_cast<void*>(p);
}

// TODO: remove once all compilers are using C++20
template<class T, class D>
std::ostream& boost_test_print_type(std::ostream& os, const std::unique_ptr<T, D>& p) {
    return os << p.get();
}
}

#endif // INCLUDED_ml_test_BoostTestPointerOutput_h
