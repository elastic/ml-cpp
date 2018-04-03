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

#ifndef INCLUDED_ml_core_CFloatStorage_h
#define INCLUDED_ml_core_CFloatStorage_h

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/ImportExport.h>

#include <limits>

#include <math.h>

namespace ml {
namespace core {

namespace {
const int MAX_PRECISE_INTEGER_FLOAT(static_cast<int>(::pow(10.0, static_cast<double>(std::numeric_limits<float>::digits10))) - 1);
}

//! \brief This class should be used in place of float whenever
//! that class is required.
//!
//! DESCRIPTION:\n
//! We STRONGLY DISCOURAGE any use of float whatsoever in the
//! code base. However, there are occasions where using it in the
//! maths library gives significant overall space improvements.
//! That said it should only be used for storage: all calculations
//! should be double precision. This is important because, for
//! example, it is much easier to overflow and underflow float.
//! There are no significant runtime advantages in using float
//! with modern FPUs, so there is essentially no reason ever to
//! use float in calculations.
//!
//! This class provides the necessary silent conversion operations
//! between float and double and provides optional, at compile time,
//! bounds checking via the build flag CFLOATSTORAGE_BOUNDS_CHECK.
//!
//! IMPLEMENTATION DECISIONS:\n
//! All the operations are intentionally inline to minimize conversion
//! costs.
//!
//! Note that no operators are defined taking CFloatStorage so any
//! expression will be forced to promote to double. Furthermore,
//! all intermediate results will also be double. This means that
//! something like:
//!
//! <code>
//!   CFloatStorage a(1f);
//!   CFloatStorage b(2f);
//!   CFloatStorage c(4f);
//!   CFloatStorage d = a * b + 2.0 * c * c;
//! </code>
//!
//! Will use exactly one conversion from double to float to assign
//! the value of a * b + 2.0 * c * c to d.
class CORE_EXPORT CFloatStorage {
public:
    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero(void) { return true; }

public:
    //! Default construction of the floating point value.
    CFloatStorage(void) : m_Value() {}

    //! Integer promotion. So one can write things like CFloatStorage(1).
    CFloatStorage(int value) : m_Value(float(value)) {
#ifdef CFLOATSTORAGE_BOUNDS_CHECK
        if (value > MAX_PRECISE_INTEGER_FLOAT || -value < MAX_PRECISE_INTEGER_FLOAT) {
            LOG_WARN("Loss of precision assigning int " << value << " to float");
        }
#endif // CFLOATSTORAGE_BOUNDS_CHECK
    }

    //! Implicit construction from a float.
    CFloatStorage(float value) : m_Value(value) {}

    //! Implicit construction from a double.
    CFloatStorage(double value) : m_Value() { this->set(value); }

    //! Set from a string.
    bool fromString(const std::string& string) {
        double value;
        if (CStringUtils::stringToType(string, value)) {
            this->set(value);
            return true;
        }
        return false;
    }

    //! Convert to a string.
    std::string toString(void) const {
        return CStringUtils::typeToStringPrecise(static_cast<double>(m_Value), CIEEE754::E_SinglePrecision);
    }

    //! \name Double Assignment
    //@{
    //! Assign from a double.
    CFloatStorage& operator=(double value) {
        this->set(value);
        return *this;
    }
    //! Plus assign from double.
    CFloatStorage& operator+=(double value) {
        this->set(static_cast<double>(m_Value) + value);
        return *this;
    }
    //! Minus assign from double.
    CFloatStorage& operator-=(double value) {
        this->set(static_cast<double>(m_Value) - value);
        return *this;
    }
    //! Multiply assign from double.
    CFloatStorage& operator*=(double value) {
        this->set(static_cast<double>(m_Value) * value);
        return *this;
    }
    //! Divide assign from double.
    CFloatStorage& operator/=(double value) {
        this->set(static_cast<double>(m_Value) / value);
        return *this;
    }
    //@}

    //! Implicit conversion to a double.
    operator double(void) const { return static_cast<double>(m_Value); }

private:
    //! Utility to actually set the floating point value.
    void set(double value) {
#ifdef CFLOATSTORAGE_BOUNDS_CHECK
        if (value > std::numeric_limits<float>::max() || -value > std::numeric_limits<float>::max()) {
            LOG_WARN("Value overflows float " << value);
        }
        if (value < std::numeric_limits<float>::min() && -value < std::numeric_limits<float>::min()) {
            LOG_WARN("Value underflows float " << value);
        } else if (value < 100 * std::numeric_limits<float>::min() && -value < 100 * std::numeric_limits<float>::min()) {
            LOG_WARN("Less than 3 s.f. precision retained for " << value);
        }
#endif // CFLOATSTORAGE_BOUNDS_CHECK
        m_Value = static_cast<float>(value);
    }

private:
    float m_Value;
};
}
}

#endif // INCLUDED_ml_core_CFloatStorage_h
