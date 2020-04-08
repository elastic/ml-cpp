/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CIEEE754.h>

#include <cmath>
#include <cstring>

namespace ml {
namespace core {

double CIEEE754::round(double value, EPrecision precision) {
    // First decomposes the value into the mantissa and exponent to avoid the
    // problem with overflow if the values are close to max double.

    int exponent;
    double mantissa = std::frexp(value, &exponent);

    switch (precision) {
    case E_HalfPrecision: {
        static const double PRECISION = 2048.0;
        mantissa = mantissa < 0.0 ? std::ceil(mantissa * PRECISION - 0.5) / PRECISION
                                  : std::floor(mantissa * PRECISION + 0.5) / PRECISION;
        break;
    }
    case E_SinglePrecision: {
        static const double PRECISION = 16777216.0;
        mantissa = mantissa < 0.0 ? std::ceil(mantissa * PRECISION - 0.5) / PRECISION
                                  : std::floor(mantissa * PRECISION + 0.5) / PRECISION;
        break;
    }
    case E_DoublePrecision:
        // Nothing to do.
        break;
    }

    return std::ldexp(mantissa, exponent);
}

double CIEEE754::dropbits(double value, int bits) {
    SDoubleRep parsed;
    static_assert(sizeof(double) == sizeof(SDoubleRep),
                  "SDoubleRep definition unsuitable for memcpy to double");
    std::memcpy(&parsed, &value, sizeof(double));
    parsed.s_Mantissa &= ((IEEE754_MANTISSA_MASK << bits) & IEEE754_MANTISSA_MASK);
    std::memcpy(&value, &parsed, sizeof(double));
    return value;
}
}
}
