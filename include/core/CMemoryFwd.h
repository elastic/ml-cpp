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

#ifndef INCLUDED_ml_core_CMemoryFwd_h
#define INCLUDED_ml_core_CMemoryFwd_h

#include <core/CMemoryUsage.h>

#include <functional>
#include <optional>
#include <type_traits>

namespace ml {
namespace core {
namespace memory_detail {
//! \brief Base implementation checks for POD.
template<typename T, typename = void>
struct SDynamicSizeAlwaysZero {
    static constexpr inline bool value() { return std::is_pod<T>::value; }
};

//! \brief Checks types in pair.
template<typename U, typename V>
struct SDynamicSizeAlwaysZero<std::pair<U, V>> {
    static constexpr inline bool value() {
        return SDynamicSizeAlwaysZero<U>::value() && SDynamicSizeAlwaysZero<V>::value();
    }
};

//! \brief Specialisation for std::less always true.
template<typename T>
struct SDynamicSizeAlwaysZero<std::less<T>> {
    static constexpr inline bool value() { return true; }
};

//! \brief Specialisation for std::greater always true.
template<typename T>
struct SDynamicSizeAlwaysZero<std::greater<T>> {
    static constexpr inline bool value() { return true; }
};

//! \brief Checks type in optional.
template<typename T>
struct SDynamicSizeAlwaysZero<std::optional<T>> {
    static constexpr inline bool value() {
        return SDynamicSizeAlwaysZero<T>::value();
    }
};

//! \brief Check for member dynamicSizeAlwaysZero function.
// clang-format off
template<typename T>
struct SDynamicSizeAlwaysZero<T, std::enable_if_t<
            std::is_same_v<decltype(&T::dynamicSizeAlwaysZero), bool (*)()>>> {
    static constexpr inline bool value() { return T::dynamicSizeAlwaysZero(); }
};
// clang-format on
}
}
}

#endif // INCLUDED_ml_core_CMemoryFwd_h
