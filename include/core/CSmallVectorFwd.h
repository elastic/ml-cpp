/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_core_CSmallVectorFwd_h
#define INCLUDED_ml_core_CSmallVectorFwd_h

#include <boost/container/container_fwd.hpp>

#include <cstddef>

namespace ml {
namespace core {
//! Map boost::container::small_vector_base for consistent naming.
template<typename T>
using CSmallVectorBase = boost::container::small_vector_base<T>;
template<typename T, std::size_t N>
class CSmallVector;
}
}

#endif // INCLUDED_ml_core_CSmallVectorFwd_h
