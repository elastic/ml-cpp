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
#ifndef INCLUDED_ml_core_CNonCopyable_h
#define INCLUDED_ml_core_CNonCopyable_h

#include <core/ImportExport.h>

namespace ml {
namespace core {

//! \brief
//! Equivalent to boost::noncopyable.
//!
//! DESCRIPTION:\n
//! This class is designed to be used wherever boost::noncopyable would
//! be used.  It has the advantage over boost::noncopyable that it does
//! not trigger Visual C++ warning C4275.
//!
//! Classes for which copying is not allowed should inherit privately
//! from this class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The class is exported from the DLL, which means that non-inlined
//! versions of its methods will be generated even though they're all
//! inlined.  This is the difference compared to boost::noncopyable,
//! and what prevents Visual C++ warning C4275.
//!
class CORE_EXPORT CNonCopyable {
protected:
    //! Inlined in the hope that the compiler will optimise it away
    CNonCopyable() = default;

    //! Inlined in the hope that the compiler will optimise it away
    ~CNonCopyable() = default;

private:
    //! Prevent copying
    CNonCopyable(const CNonCopyable&);
    CNonCopyable& operator=(const CNonCopyable&);
};
}
}

#endif // INCLUDED_ml_core_CNonCopyable_h
