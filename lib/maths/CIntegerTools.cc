/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CIntegerTools.h>

#include <core/CLogger.h>

#include <algorithm>
#include <cmath>

namespace ml {
namespace maths {

bool CIntegerTools::isInteger(double value, double tolerance) {
    double integerPart;
    double remainder = std::modf(value, &integerPart);
    return remainder <= tolerance * integerPart;
}

std::size_t CIntegerTools::nextPow2(uint64_t x) {
    // This is just a binary search for the highest non-zero bit.

    static const std::size_t SHIFTS[] = {32u, 16u, 8u, 4u, 2u, 1u};
    static const uint64_t MASKS[] = {0xffffffff, 0xffff, 0xff, 0xf, 0x3, 0x1};

    std::size_t result = 0u;
    for (std::size_t i = 0; i < 6; ++i) {
        uint64_t y = (x >> SHIFTS[i]);
        if (y & MASKS[i]) {
            result += SHIFTS[i];
            x = y;
        }
    }
    return result + static_cast<std::size_t>(x);
}

uint64_t CIntegerTools::reverseBits(uint64_t x) {
    // Uses the standard "parallel" approach of swapping adjacent bits, then
    // adjacent pairs, quadruples, etc.
    x = ((x >> 1) & 0x5555555555555555) | ((x << 1) & 0xaaaaaaaaaaaaaaaa);
    x = ((x >> 2) & 0x3333333333333333) | ((x << 2) & 0xcccccccccccccccc);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0f) | ((x << 4) & 0xf0f0f0f0f0f0f0f0);
    x = ((x >> 8) & 0x00ff00ff00ff00ff) | ((x << 8) & 0xff00ff00ff00ff00);
    x = ((x >> 16) & 0x0000ffff0000ffff) | ((x << 16) & 0xffff0000ffff0000);
    x = ((x >> 32) & 0x00000000ffffffff) | ((x << 32) & 0xffffffff00000000);
    return x;
}

double CIntegerTools::binomial(unsigned int n, unsigned int k) {
    if (n < k) {
        LOG_ERROR(<< "Bad coefficient : (n k) = (" << n << " " << k << ")");
        return 0.0;
    }

    double result = 1.0;
    k = std::min(k, n - k);
    for (unsigned int k_ = k; k_ > 0; --k_, --n) {
        result *= static_cast<double>(n) / static_cast<double>(k_);
    }
    return result;
}
}
}
