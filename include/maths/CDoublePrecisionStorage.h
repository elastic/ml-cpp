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
#ifndef INCLUDED_ml_maths_CDoublePrecisionStorage_h
#define INCLUDED_ml_maths_CDoublePrecisionStorage_h

#include <maths/ImportExport.h>

namespace ml {
namespace maths {

//! \brief A wrapper around double to enable double-precision persisting
//!
//! DESCRIPTION:\n
//! A wrapper around double to enable double-precision persisting
//!
//! Doubles are usually persisted with single precision, but in certain
//! cases this leads to an unacceptable loss of precision, for example
//! when a bucket time value is stored in a double (~1e9)
class CDoublePrecisionStorage {
public:
    CDoublePrecisionStorage() : m_Value(0) {}

    CDoublePrecisionStorage(double v) : m_Value(v) {}

    //! Implicit conversion to a double.
    operator double() const { return m_Value; }

    //! Assign from a double.
    CDoublePrecisionStorage& operator=(double value) {
        m_Value = value;
        return *this;
    }

    //! Plus assign from double.
    CDoublePrecisionStorage& operator+=(double value) {
        m_Value += value;
        return *this;
    }

private:
    //! The underlying value
    double m_Value;
};

} // maths
} // ml

#endif // INCLUDED_ml_maths_CDoublePrecisionStorage_h
