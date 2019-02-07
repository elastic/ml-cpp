/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoundingBox_h
#define INCLUDED_ml_maths_CBoundingBox_h

#include <core/CMemory.h>

#include <maths/CLinearAlgebraShims.h>
#include <maths/CTypeTraits.h>

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
    static bool dynamicSizeAlwaysZero() {
        return core::memory_detail::SDynamicSizeAlwaysZero<POINT>::value();
    }
    using TPointPrecise = typename SFloatingPoint<POINT, double>::Type;

public:
    CBoundingBox() : m_Empty(true), m_A(), m_B() {}
    CBoundingBox(const POINT& x) : m_Empty(false), m_A(x), m_B(x) {}

    //! Clear the bounding box.
    void clear() {
        m_Empty = true;
        m_A = m_B = POINT();
    }

    //! Add \p p point, i.e. find the bounding box of the point
    //! and this bounding box.
    void add(const POINT& p) {
        if (m_Empty) {
            m_A = m_B = p;
        } else {
            las::min(p, m_A);
            las::max(p, m_B);
        }
        m_Empty = false;
    }

    //! Add \p other bounding box, i.e. find the bounding box of
    //! the two bounding boxes.
    void add(const CBoundingBox& other) {
        if (m_Empty) {
            *this = other;
        } else if (!other.m_Empty) {
            las::min(other.m_A, m_A);
            las::max(other.m_B, m_B);
            m_Empty = false;
        }
    }

    //! Get the bottom left corner.
    const POINT& blc() const { return m_A; }

    //! Get the top right corner.
    const POINT& trc() const { return m_B; }

    //! Get the centre of the bounding box.
    POINT centre() const { return POINT(TPointPrecise(m_A + m_B) / 2.0); }

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
    bool closerToX(const POINT& x, const POINT& y) const {
        POINT xy = y - x;
        POINT f(m_B);
        for (std::size_t i = 0u; i < las::dimension(x); ++i) {
            if (xy(i) < 0) {
                f(i) = m_A(i);
            }
        }
        return las::distance(f, x) <= las::distance(f, y);
    }

    //! Print this bounding box.
    std::string print() const {
        std::ostringstream result;
        result << "{" << m_A << ", " << m_B << "}";
        return result.str();
    }

private:
    //! True if this is empty and false otherwise.
    bool m_Empty;

    //! The bottom left and top right corner of the bounding box.
    POINT m_A, m_B;
};
}
}

#endif // INCLUDED_ml_maths_CBoundingBox_h
