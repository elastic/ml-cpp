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

#ifndef INCLUDED_ml_maths_CBoundingBox_h
#define INCLUDED_ml_maths_CBoundingBox_h

#include <core/CMemory.h>

#include <maths/CTypeConversions.h>

#include <cstddef>
#include <ostream>
#include <string>

namespace ml {
namespace maths {

//! \brief An axis aligned bounding box.
//!
//! DESCRIPTION\n
//! Holds the bottom left and top right corners and provides
//! various utility functions need by the x-means algorithm.
template<typename POINT>
class CBoundingBox {
    public:
        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero(void) {
            return core::memory_detail::SDynamicSizeAlwaysZero<POINT>::value();
        }
        typedef typename SFloatingPoint<POINT, double>::Type TPointPrecise;

    public:
        CBoundingBox(void) : m_Empty(true), m_A(), m_B() {
        }

        CBoundingBox(const POINT &x) : m_Empty(false), m_A(x), m_B(x) {
        }

        //! Clear the bounding box.
        void clear(void) {
            m_Empty = true;
            m_A = m_B = POINT();
        }

        //! Add \p p point, i.e. find the bounding box of the point
        //! and this bounding box.
        void add(const POINT &p) {
            if (m_Empty) {
                m_A = m_B = p;
            } else {
                m_A = min(m_A, p);
                m_B = max(m_B, p);
            }
            m_Empty = false;
        }

        //! Add \p other bounding box, i.e. find the bounding box of
        //! the two bounding boxes.
        void add(const CBoundingBox &other) {
            if (m_Empty) {
                *this = other;
            } else if (!other.m_Empty) {
                m_A = min(m_A, other.m_A);
                m_B = max(m_B, other.m_B);
                m_Empty = false;
            }
        }

        //! Get the bottom left corner.
        const POINT &blc(void) const {
            return m_A;
        }

        //! Get the top right corner.
        const POINT &trc(void) const {
            return m_B;
        }

        //! Get the centre of the bounding box.
        POINT centre(void) const {
            return POINT(TPointPrecise(m_A + m_B) / 2.0);
        }

        //! Check if \p x is everywhere closer to the bounding box
        //! than \p y.
        //!
        //! The idea is fairly simple: find the corner which is closest
        //! to the hyperplane bisecting the vector from \p x to \p y.
        //! Note that this plane is boundary of the region where every
        //! point is closer to \p x or \p y. If the corner is in the
        //! region closer to \p x then the bounding box must necessarily
        //! be in this region too and no point can therefore be closer
        //! to \p y than \p x.
        bool closerToX(const POINT &x, const POINT &y) const {
            POINT xy = y - x;
            POINT f(0);
            for (std::size_t i = 0u; i < x.dimension(); ++i) {
                f(i) = xy(i) < 0 ? m_A(i) : m_B(i);
            }
            return (f - x).euclidean() <= (f - y).euclidean();
        }

        //! Print this bounding box.
        std::string print(void) const {
            std::ostringstream result;
            result << "{" << m_A << ", " << m_B << "}";
            return result.str();
        }

    private:
        //! True if this is empty and false otherwise.
        bool  m_Empty;

        //! The bottom left and top right corner of the bounding box.
        POINT m_A, m_B;
};


}
}

#endif // INCLUDED_ml_maths_CBoundingBox_h
