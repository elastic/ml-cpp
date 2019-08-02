/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

//! Temporary substitute for std::unwrap_ref. This will be replaced by
//! std::unwrap_ref once all our compilers support it. The names in this
//! file do not conform to the coding style to ease the eventual transition
//! to std::unwrap_ref.

#ifndef INCLUDED_ml_core_UnwrapRef_h
#define INCLUDED_ml_core_UnwrapRef_h

#include <functional>

namespace ml {
namespace core {

template<typename T>
struct unwrap_reference {
    typedef T type;
};

template<typename T>
struct unwrap_reference<std::reference_wrapper<T>> {
    typedef T type;
};

template<typename T>
struct unwrap_reference<const std::reference_wrapper<T>> {
    typedef T type;
};

template<typename T>
struct unwrap_reference<volatile std::reference_wrapper<T>> {
    typedef T type;
};

template<typename T>
typename unwrap_reference<T>::type& unwrap_ref(T& t) {
    return t;
}
}
}

#endif // INCLUDED_ml_core_UnwrapRef_h
