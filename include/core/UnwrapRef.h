/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_UNWRAPREF_H_
#define INCLUDED_ml_core_UNWRAPREF_H_

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

#endif /* INCLUDED_ml_core_UNWRAPREF_H_ */
