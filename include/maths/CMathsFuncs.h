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

#ifndef INCLUDED_ml_maths_CMathsFuncs_h
#define INCLUDED_ml_maths_CMathsFuncs_h

#include <core/CNonInstantiatable.h>

#include <maths/CLinearAlgebraFwd.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <functional>
#include <iterator>

namespace ml {
namespace maths {

//! \brief
//! Portable maths functions
//!
//! DESCRIPTION:\n
//! Portable maths functions
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses double - it's best that we DON'T use long double, as its size varies
//! between platforms and compilers, e.g. on SPARC long double is 128 bits
//! but has to be manipulated in software (which is slow), on Intel long
//! double is 80 bits natively, but Visual Studio treats long double as 64
//! bits, i.e. the same as double, whereas in gcc long double maps to the 80
//! bit CPU type.
//!
//! Where maths functions have different names on different platforms,
//! they should be added to this file.
//!
class MATHS_EXPORT CMathsFuncs : private core::CNonInstantiatable {
public:
    //! Wrapper around boost::math::isnan() which avoids the need to add
    //! cryptic brackets everywhere to deal with macros.
    static bool isNan(double val);
    //! Check if any of the components are NaN.
    template<std::size_t N>
    static bool isNan(const CVectorNx1<double, N>& val);
    //! Check if any of the components are NaN.
    static bool isNan(const CVector<double>& val);
    //! Check if any of the elements are NaN.
    template<std::size_t N>
    static bool isNan(const CSymmetricMatrixNxN<double, N>& val);
    //! Check if any of the elements are NaN.
    static bool isNan(const CSymmetricMatrix<double>& val);
    //! Check if an element is NaN.
    static bool isNan(const core::CSmallVectorBase<double>& val);

    //! Wrapper around boost::math::isinf() which avoids the need to add
    //! cryptic brackets everywhere to deal with macros.
    static bool isInf(double val);
    //! Check if any of the components are infinite.
    template<std::size_t N>
    static bool isInf(const CVectorNx1<double, N>& val);
    //! Check if any of the components are infinite.
    static bool isInf(const CVector<double>& val);
    //! Check if any of the elements are infinite.
    template<std::size_t N>
    static bool isInf(const CSymmetricMatrixNxN<double, N>& val);
    //! Check if any of the elements are infinite.
    static bool isInf(const CSymmetricMatrix<double>& val);
    //! Check if an element is NaN.
    static bool isInf(const core::CSmallVectorBase<double>& val);

    //! Neither infinite nor NaN.
    static bool isFinite(double val);
    //! Check if all of the components are finite.
    template<std::size_t N>
    static bool isFinite(const CVectorNx1<double, N>& val);
    //! Check if all of the components are finite.
    static bool isFinite(const CVector<double>& val);
    //! Check if all of the components are NaN.
    template<std::size_t N>
    static bool isFinite(const CSymmetricMatrixNxN<double, N>& val);
    //! Check if all of the components are NaN.
    static bool isFinite(const CSymmetricMatrix<double>& val);
    //! Check if an element is NaN.
    static bool isFinite(const core::CSmallVectorBase<double>& val);

    //! Check the floating point status of \p value.
    static maths_t::EFloatingPointErrorStatus fpStatus(double val);

    //! Unary function object to check if a value is finite.
    struct SIsFinite : std::unary_function<double, bool> {
        bool operator()(double val) const { return isFinite(val); }
    };

    //! \brief Wrapper around an iterator over a collection of doubles,
    //! which must implement the forward iterator concepts, that skips
    //! non-finite values.
    template<typename ITR>
    class CFiniteIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = typename std::iterator_traits<ITR>::value_type;
        using difference_type = typename std::iterator_traits<ITR>::difference_type;
        using pointer = typename std::iterator_traits<ITR>::pointer;
        using reference = typename std::iterator_traits<ITR>::reference;

    public:
        CFiniteIterator() : m_Base(), m_End() {}
        CFiniteIterator(const ITR& base, const ITR& end) : m_Base(base), m_End(end) {
            if (m_Base != m_End && !isFinite(*m_Base)) {
                this->increment();
            }
        }

        //! Equal.
        bool operator==(const CFiniteIterator& rhs) const { return m_Base == rhs.m_Base; }
        //! Different.
        bool operator!=(const CFiniteIterator& rhs) const { return m_Base != rhs.m_Base; }

        //! Dereference.
        reference operator*() const { return *m_Base; }
        //! Pointer.
        pointer operator->() const { return m_Base.operator->(); }

        //! Prefix increment.
        const CFiniteIterator& operator++() {
            this->increment();
            return *this;
        }
        //! Post-fix increment.
        CFiniteIterator operator++(int) {
            CFiniteIterator result(*this);
            this->increment();
            return result;
        }

    private:
        //! Implements increment.
        void increment() {
            while (++m_Base != m_End) {
                if (isFinite(*m_Base)) {
                    break;
                }
            }
        }

    private:
        ITR m_Base;
        ITR m_End;
    };

    //! Get an iterator over the finite values of a double container.
    template<typename T>
    static CFiniteIterator<typename T::iterator> beginFinite(T& container) {
        return CFiniteIterator<typename T::iterator>(container.begin(), container.end());
    }

    //! Get a const_iterator over the finite values of a double container.
    template<typename T>
    static CFiniteIterator<typename T::const_iterator> beginFinite(const T& container) {
        return CFiniteIterator<typename T::const_iterator>(container.begin(), container.end());
    }

    //! Get a finite values iterator at the end of a double container.
    template<typename T>
    static CFiniteIterator<typename T::iterator> endFinite(T& container) {
        return CFiniteIterator<typename T::iterator>(container.end(), container.end());
    }

    //! Get a finite values const_iterator at the end of a double container.
    template<typename T>
    static CFiniteIterator<typename T::const_iterator> endFinite(const T& container) {
        return CFiniteIterator<typename T::const_iterator>(container.end(), container.end());
    }

private:
    //! Check if any of the components return true for \p f.
    template<typename VECTOR, typename F>
    static bool aComponent(const F& f, const VECTOR& val);

    //! Check if all the components return true for \p f.
    template<typename VECTOR, typename F>
    static bool everyComponent(const F& f, const VECTOR& val);

    //! Check if any of the elements return true for \p f.
    template<typename SYMMETRIC_MATRIX, typename F>
    static bool anElement(const F& f, const SYMMETRIC_MATRIX& val);

    //! Check if all the elements return true for \p f.
    template<typename SYMMETRIC_MATRIX, typename F>
    static bool everyElement(const F& f, const SYMMETRIC_MATRIX& val);
};
}
}

#endif // INCLUDED_ml_maths_CMathsFuncs_h
