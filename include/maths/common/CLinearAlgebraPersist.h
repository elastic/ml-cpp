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

#ifndef INCLUDED_ml_maths_common_CLinearAlgebraPersist_h
#define INCLUDED_ml_maths_common_CLinearAlgebraPersist_h

#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

#include <maths/common/CLinearAlgebra.h>

namespace ml {
namespace maths {
namespace common {
namespace linear_algebra_detail {

//! \brief Extracts a vector component / matrix element from a string.
struct SFromString {
    template<typename T>
    bool operator()(const std::string& token, T& value) const {
        return core::CStringUtils::stringToType(token, value);
    }
    bool operator()(const std::string& token, CFloatStorage& value) const {
        return value.fromString(token);
    }
};

//! \brief Converts a vector component / matrix element to a string.
struct SToString {
    template<typename T>
    std::string operator()(const T& value) const {
        return core::CStringUtils::typeToString(value);
    }
    std::string operator()(double value) const {
        return core::CStringUtils::typeToStringPrecise(value, core::CIEEE754::E_DoublePrecision);
    }
    std::string operator()(CFloatStorage value) const {
        return value.toString();
    }
};

template<typename STORAGE>
bool SSymmetricMatrix<STORAGE>::fromDelimited(const std::string& str) {
    return core::CPersistUtils::fromString(str, SFromString(), m_LowerTriangle,
                                           CLinearAlgebra::DELIMITER);
}

template<typename STORAGE>
std::string SSymmetricMatrix<STORAGE>::toDelimited() const {
    return core::CPersistUtils::toString(m_LowerTriangle, SToString(),
                                         CLinearAlgebra::DELIMITER);
}

template<typename STORAGE>
bool SVector<STORAGE>::fromDelimited(const std::string& str) {
    return core::CPersistUtils::fromString(str, SFromString(), m_X, CLinearAlgebra::DELIMITER);
}

template<typename STORAGE>
std::string SVector<STORAGE>::toDelimited() const {
    return core::CPersistUtils::toString(m_X, SToString(), CLinearAlgebra::DELIMITER);
}
}
}
}
}

#endif // INCLUDED_ml_maths_common_CLinearAlgebraPersist_h
