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

#ifndef INCLUDED_ml_maths_COrdinal_h
#define INCLUDED_ml_maths_COrdinal_h

#include <maths/ImportExport.h>

#include <boost/operators.hpp>

#include <iosfwd>
#include <stdint.h>

namespace ml {
namespace maths {

//! \brief A representation of an ordinal type.
//!
//! DESCRIPTION:\n
//! This deals with floating point and integer values and works
//! around the loss of precision converting 64 bit integers to
//! doubles.
class MATHS_EXPORT COrdinal : private boost::equality_comparable<COrdinal, boost::partially_ordered<COrdinal>> {
public:
    //! Create an unset value.
    COrdinal(void);
    COrdinal(int64_t value);
    COrdinal(uint64_t value);
    COrdinal(double value);

    //! Check if two ordinals are equal.
    bool operator==(COrdinal rhs) const;

    //! Check if one ordinal is less than another.
    bool operator<(COrdinal rhs) const;

    //! Check if the value has been set.
    bool isNan(void) const;

    //! Convert to a double (accepting possible loss in precision).
    double asDouble(void) const;

    //! Get a hash of the value.
    uint64_t hash(void);

private:
    //! Enumeration of the types which can be stored.
    enum EType {
        E_Integer,
        E_PositiveInteger,
        E_Real,
        E_Nan // Semantics are same as Nan.
    };

    union Value {
        int64_t integer;
        uint64_t positiveInteger;
        double real;
    };

private:
    bool equal(int64_t lhs, uint64_t rhs) const;
    bool equal(int64_t lhs, double rhs) const;
    bool equal(uint64_t lhs, double rhs) const;
    bool less(int64_t lhs, uint64_t rhs) const;
    bool less(int64_t lhs, double rhs) const;
    bool less(uint64_t lhs, double rhs) const;

private:
    //! The type of value stored.
    EType m_Type;
    //! The value.
    Value m_Value;

    MATHS_EXPORT
    friend std::ostream& operator<<(std::ostream& o, COrdinal ord);
};
}
}

#endif // INCLUDED_ml_maths_COrdinal_h
