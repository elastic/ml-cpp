/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CAlignment_h
#define INCLUDED_ml_core_CAlignment_h

#include <core/ImportExport.h>

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <vector>

namespace ml {
namespace core {

class CORE_EXPORT CAlignment {
public:
    //! Alignment types.
    enum EType {
    E_Unaligned = 1,
    E_Aligned8 = 8,
    E_Aligned16 = 16,
    E_Aligned32 = 32
    };

    //! This is an ordering by inclusion, i.e. \p lhs < \p rhs if an address is
    //! \p rhs aligned implies it is lhs aligned but not vice versa.
    static bool less(EType lhs, EType rhs) {
        return bytes(lhs) < bytes(rhs);
    }

    //! Get the alignment of \p address.
    template<typename T>
    static EType maxAlignment(const T* address) {
        // clang-format off
        return (isAligned(address, E_Aligned32) ? E_Aligned32 :
               (isAligned(address, E_Aligned16) ? E_Aligned16 :
               (isAligned(address, E_Aligned8)  ? E_Aligned8  :
               (E_Unaligned))));
        // clang-format on
    }

    //! Check if \p address has \p alignment.
    template<typename T>
    static bool isAligned(const T* address, EType alignment) {
        return offset(address, alignment) == 0;
    }

    //! Get the next index in \p buffer which is aligned to \p alignment.
    template<typename T, std::size_t N>
    static std::size_t nextAligned(const std::array<T, N>& buffer,
                                   std::size_t index,
                                   EType alignment) {
        std::size_t offset_{offset(&buffer[index], alignment)};
        return offset_ == 0 ? index : index + (bytes(alignment) - offset_) / sizeof(T);
    }

    //! Get the next index in \p buffer which is aligned to \p alignment.
    template<typename T>
    static std::size_t nextAligned(const std::vector<T>& buffer,
                                   std::size_t index,
                                   EType alignment) {
        std::size_t offset_{offset(&buffer[index], alignment)};
        return offset_ == 0 ? index : index + (bytes(alignment) - offset_) / sizeof(T);
    }

    //! Round up n items of T so they use a multiple of \p alignment size memory.
    template<typename T>
    static std::size_t roundup(EType alignment, std::size_t n) {
        return roundupSizeof<T>(alignment, n) / sizeof(T);
    }

    //! Round up sizeof(T) up to multiple of \p alignment bytes.
    template<typename T>
    static std::size_t roundupSizeof(EType alignment, std::size_t n = 1) {
        std::size_t bytes_{bytes(alignment)};
        return ((n * sizeof(T) + bytes_ - 1) / bytes_) * bytes_;
    }

    //! Print the type.
    static std::string print(EType type) {
    switch (type) {
    case E_Unaligned: return "unaligned";
    case E_Aligned8:  return "aligned 8";
    case E_Aligned16: return "aligned 16";
    case E_Aligned32: return "aligned 32";
    }
    return "";
    }

private:
    template<typename T>
    static std::size_t offset(const T* address, EType alignment) {
        return reinterpret_cast<std::size_t>(address) & mask(alignment);
    }

    static std::size_t mask(EType alignment) {
        return bytes(alignment) - 1;
    }

    static std::size_t bytes(EType alignment) {
        return static_cast<std::size_t>(alignment);
    }
};

template<typename T>
using CAlignedAllocator = Eigen::aligned_allocator<T>;
}
}

#endif // INCLUDED_ml_core_CAlignment_h
