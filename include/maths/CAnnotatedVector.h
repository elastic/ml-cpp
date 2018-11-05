/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CAnnotatedVector_h
#define INCLUDED_ml_maths_CAnnotatedVector_h

#include <core/CMemory.h>

#include <maths/CTypeTraits.h>

#include <cstddef>

namespace ml {
namespace maths {

//! \brief A vector on to which data have been annotated.
//!
//! \tparam VECTOR The vector type.
//! \tparam ANNOTATION The annotated data type.
template<typename VECTOR, typename ANNOTATION>
class CAnnotatedVector : public VECTOR {
public:
    using TAnnotation = ANNOTATION;
    using TCoordinate = typename SCoordinate<VECTOR>::Type;

    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() {
        return core::memory_detail::SDynamicSizeAlwaysZero<VECTOR>::value() &&
               core::memory_detail::SDynamicSizeAlwaysZero<ANNOTATION>::value();
    }

public:
    //! Construct with a vector and annotation data.
    CAnnotatedVector(const VECTOR& vector = VECTOR(),
                     const ANNOTATION& annotation = ANNOTATION())
        : VECTOR(vector), m_Annotation(annotation) {}

    //! Construct with a vector initialized with \p coordinate
    //! and some default constructed annotation data.
    //! TODO this needs to be removed since it assumes the dimension
    //! is available at compile time.
    explicit CAnnotatedVector(TCoordinate coordinate) : VECTOR(coordinate) {}

    //! Assign from \p rhs.
    const CAnnotatedVector& operator=(const VECTOR& rhs) {
        static_cast<VECTOR&>(*this) = rhs;
        return *this;
    }

    //! Get the annotation data by constant reference.
    const ANNOTATION& annotation() const { return m_Annotation; }

    //! Get the annotation data by reference.
    ANNOTATION& annotation() { return m_Annotation; }

private:
    //! The data which has been annotated onto the vector.
    ANNOTATION m_Annotation;
};

//! Get the concomitant constant annotated vector.
template<typename VECTOR, typename ANNOTATION>
struct SConstant<CAnnotatedVector<VECTOR, ANNOTATION>> {
    static auto get(std::size_t dimension,
                    typename CAnnotatedVector<VECTOR, ANNOTATION>::TCoordinate constant)
        -> CAnnotatedVector<decltype(SConstant<VECTOR>::get(dimension, constant)), ANNOTATION> {
        return {SConstant<VECTOR>::get(dimension, constant)};
    }
};
}
}

#endif // INCLUDED_ml_maths_CAnnotatedVector_h
