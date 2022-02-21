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
#ifndef INCLUDED_ml_core_CNonInstantiatable_h
#define INCLUDED_ml_core_CNonInstantiatable_h

#include <core/ImportExport.h>

namespace ml {
namespace core {

//! \brief
//! Similar idea to boost::noncopyable, but for instantiation.
//!
//! DESCRIPTION:\n
//! Classes for which instantiation is not allowed, i.e. those that
//! have only static methods, should inherit privately from this
//! class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The class is (seemingly pointlessly) exported from the DLL to
//! prevent Visual C++ warning C4275.
//!
class CORE_EXPORT CNonInstantiatable {
private:
    //! Prevent instantiation
    CNonInstantiatable();
    CNonInstantiatable(const CNonInstantiatable&);
};
}
}

#endif // INCLUDED_ml_core_CNonInstantiatable_h
