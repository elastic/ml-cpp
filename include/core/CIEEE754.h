/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CIEEE754_h
#define INCLUDED_ml_core_CIEEE754_h

#include <core/ImportExport.h>

#include <cstdint>
#include <cstring>

namespace ml {
namespace core {

//! \brief A collection of utility functionality that understands
//! IEEE 754 floating point representation.
class CORE_EXPORT CIEEE754 {
public:
    //! Enumeration of possible precision types:
    //!   -# Half precision:   10 bit mantissa, 5 bit exponent,  1 sign bit.
    //!   -# Single precision: 23 bit mantissa, 8 bit exponent,  1 sign bit.
    //!   -# Double precision: 52 bit mantissa, 11 bit exponent, 1 sign bit.
    enum EPrecision { E_HalfPrecision, E_SinglePrecision, E_DoublePrecision };

    //! This emulates rounding to a specified precision (based on the number
    //! of bits in the mantissa corresponding to \p precision). It uses round
    //! to nearest ties away from zero. It never converts the values to less
    //! than double precision so leaves the exponent unmodified and handles
    //! the case that the exponent would overflow.
    static double round(double value, EPrecision precision);

    //! Used to extract the bits corresponding to the mantissa, exponent
    //! and sign of an IEEE754 double.
    //!
    //! \warning You need to be careful using these bits since the mantissa
    //! corresponding to a given double is not endian neutral when interpreted
    //! as an integer.
    //! \note The actual "exponent" is "exponent - 1022" in two's complement.
    struct SDoubleRep {
#ifdef __sparc                         // Add any other big endian architectures
        std::uint64_t s_Sign : 1;      // sign bit
        std::uint64_t s_Exponent : 11; // exponent
        std::uint64_t s_Mantissa : 52; // mantissa
#else
        std::uint64_t s_Mantissa : 52; // mantissa
        std::uint64_t s_Exponent : 11; // exponent
        std::uint64_t s_Sign : 1;      // sign bit
#endif
    };

    static const uint64_t IEEE754_MANTISSA_MASK = 0xFFFFFFFFFFFFF;

    //! Decompose a double in to its mantissa and exponent.
    //!
    //! \note This is closely related to std::frexp for double but returns
    //! the mantissa interpreted as an integer.
    static void decompose(double value, std::uint64_t& mantissa, int& exponent) {
        SDoubleRep parsed;
        static_assert(sizeof(double) == sizeof(SDoubleRep),
                      "SDoubleRep definition unsuitable for memcpy to double");
        // Use memcpy() rather than union to adhere to strict aliasing rules
        std::memcpy(&parsed, &value, sizeof(double));
        exponent = static_cast<int>(parsed.s_Exponent) - 1022;
        mantissa = parsed.s_Mantissa;
    }

    //! Drop \p bits trailing bits from the mantissa of \p value in a stable way
    //! for different operating systems.
    static double dropbits(double value, int bits);
};
}
}

#endif // INCLUDED_ml_core_CIEEE754_h
