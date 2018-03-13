/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_maths_CMathsFuncsForMatrixAndVectorTypes_h
#define INCLUDED_ml_maths_CMathsFuncsForMatrixAndVectorTypes_h

#include <maths/CLinearAlgebra.h>
#include <maths/CMathsFuncs.h>

namespace ml {
namespace maths {

template <typename VECTOR, typename F> bool CMathsFuncs::aComponent(const F &f, const VECTOR &val) {
    for (std::size_t i = 0u; i < val.dimension(); ++i) {
        if (f(val(i))) {
            return true;
        }
    }
    return false;
}

template <typename VECTOR, typename F>
bool CMathsFuncs::everyComponent(const F &f, const VECTOR &val) {
    for (std::size_t i = 0u; i < val.dimension(); ++i) {
        if (!f(val(i))) {
            return false;
        }
    }
    return true;
}

template <typename SYMMETRIC_MATRIX, typename F>
bool CMathsFuncs::anElement(const F &f, const SYMMETRIC_MATRIX &val) {
    for (std::size_t i = 0u; i < val.rows(); ++i) {
        for (std::size_t j = i; j < val.columns(); ++j) {
            if (f(val(i, j))) {
                return true;
            }
        }
    }
    return false;
}

template <typename SYMMETRIC_MATRIX, typename F>
bool CMathsFuncs::everyElement(const F &f, const SYMMETRIC_MATRIX &val) {
    for (std::size_t i = 0u; i < val.rows(); ++i) {
        for (std::size_t j = i; j < val.columns(); ++j) {
            if (!f(val(i, j))) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N> bool CMathsFuncs::isNan(const CVectorNx1<double, N> &val) {
    return aComponent(static_cast<bool (*)(double)>(&isNan), val);
}

template <std::size_t N> bool CMathsFuncs::isNan(const CSymmetricMatrixNxN<double, N> &val) {
    return anElement(static_cast<bool (*)(double)>(&isNan), val);
}

template <std::size_t N> bool CMathsFuncs::isInf(const CVectorNx1<double, N> &val) {
    return aComponent(static_cast<bool (*)(double)>(&isInf), val);
}

template <std::size_t N> bool CMathsFuncs::isInf(const CSymmetricMatrixNxN<double, N> &val) {
    return anElement(static_cast<bool (*)(double)>(&isInf), val);
}

template <std::size_t N> bool CMathsFuncs::isFinite(const CVectorNx1<double, N> &val) {
    return everyComponent(static_cast<bool (*)(double)>(&isFinite), val);
}

template <std::size_t N> bool CMathsFuncs::isFinite(const CSymmetricMatrixNxN<double, N> &val) {
    return everyElement(static_cast<bool (*)(double)>(&isFinite), val);
}
}
}

#endif// INCLUDED_ml_maths_CMathsFuncsForMatrixAndVectorTypes_h
