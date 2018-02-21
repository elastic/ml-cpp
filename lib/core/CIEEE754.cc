/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CIEEE754.h>

#include <math.h>

namespace ml
{
namespace core
{

double CIEEE754::round(double value, EPrecision precision)
{
    // This first decomposes the value into the mantissa
    // and exponent to avoid the problem with overflow if
    // the values are close to max double.

    int exponent;
    double mantissa = ::frexp(value, &exponent);

    switch (precision)
    {
    case E_HalfPrecision:
    {
        static double PRECISION = 2048.0;
        mantissa = mantissa < 0.0 ?
                   ::ceil(mantissa * PRECISION - 0.5) / PRECISION :
                   ::floor(mantissa * PRECISION + 0.5) / PRECISION;
        break;
    }
    case E_SinglePrecision:
    {
        static double PRECISION = 16777216.0;
        mantissa = mantissa < 0.0 ?
                   ::ceil(mantissa * PRECISION - 0.5) / PRECISION :
                   ::floor(mantissa * PRECISION + 0.5) / PRECISION;
        break;
    }
    case E_DoublePrecision:
        // Nothing to do.
        break;
    }

    return ::ldexp(mantissa, exponent);
}

}
}

