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
#ifndef INCLUDED_ml_maths_CDoublePrecisionStorage_h
#define INCLUDED_ml_maths_CDoublePrecisionStorage_h

#include <maths/ImportExport.h>

namespace ml
{
namespace maths
{


//! \brief A wrapper around double to enable double-precision persisting
//!
//! DESCRIPTION:\n
//! A wrapper around double to enable double-precision persisting
//!
//! Doubles are usually persisted with single precision, but in certain
//! cases this leads to an unacceptable loss of precision, for example
//! when a bucket time value is stored in a double (~1e9)
class CDoublePrecisionStorage
{
    public:
        CDoublePrecisionStorage() : m_Value(0)
        {}

        CDoublePrecisionStorage(double v) : m_Value(v)
        {}

        //! Implicit conversion to a double.
        operator double (void) const
        {
            return m_Value;
        }

        //! Assign from a double.
        CDoublePrecisionStorage &operator=(double value)
        {
            m_Value = value;
            return *this;
        }

        //! Plus assign from double.
        CDoublePrecisionStorage &operator+=(double value)
        {
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
