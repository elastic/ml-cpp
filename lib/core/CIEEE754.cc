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

