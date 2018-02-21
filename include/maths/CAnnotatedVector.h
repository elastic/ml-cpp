/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CAnnotatedVector_h
#define INCLUDED_ml_maths_CAnnotatedVector_h

#include <core/CMemory.h>

#include <maths/CTypeConversions.h>

#include <cstddef>

namespace ml
{
namespace maths
{

//! \brief A vector on to which data have been annotated.
//!
//! \tparam VECTOR The vector type.
//! \tparam ANNOTATION The annotated data type.
template<typename VECTOR, typename ANNOTATION>
class CAnnotatedVector : public VECTOR
{
    public:
        typedef ANNOTATION TAnnotation;
        typedef typename SCoordinate<VECTOR>::Type TCoordinate;

        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero(void)
        {
            return   core::memory_detail::SDynamicSizeAlwaysZero<VECTOR>::value()
                  && core::memory_detail::SDynamicSizeAlwaysZero<ANNOTATION>::value();
        }

    public:
        //! Construct with a vector and annotation data.
        CAnnotatedVector(const VECTOR &vector = VECTOR(),
                         const ANNOTATION &annotation = ANNOTATION()) :
                VECTOR(vector),
                m_Annotation(annotation)
        {}

        //! Construct with a vector initialized with \p coordinate
        //! and some default constructed annotation data.
        explicit CAnnotatedVector(TCoordinate coordinate) :
                VECTOR(coordinate),
                m_Annotation()
        {}

        //! Get the annotation data by constant reference.
        const ANNOTATION &annotation(void) const
        {
            return m_Annotation;
        }

        //! Get the annotation data by reference.
        ANNOTATION &annotation(void)
        {
            return m_Annotation;
        }

    private:
        //! The data which has been annotated onto the vector.
        ANNOTATION m_Annotation;
};

}
}

#endif // INCLUDED_ml_maths_CAnnotatedVector_h
