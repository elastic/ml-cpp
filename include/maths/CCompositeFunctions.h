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

#ifndef INCLUDED_ml_maths_CCompositeFunctions_h
#define INCLUDED_ml_maths_CCompositeFunctions_h

#include <maths/ImportExport.h>

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <cmath>
#include <limits>

namespace ml {
namespace maths {

namespace composition_detail {

//! Type used to deduce the result type for a function.
template<typename T>
struct function_result_type {};

//! Vanilla function type 1: "result type" is the return type.
template<typename R, typename A1>
struct function_result_type<R (*)(A1)> {
    using type = typename boost::remove_reference<R>::type;
};

//! Vanilla function type 2: "result type" is the second argument type.
template<typename R, typename A1, typename A2>
struct function_result_type<R (*)(A1, A2)> {
    using type = typename boost::remove_reference<A2>::type;
};

using true_ = boost::true_type;
using false_ = boost::false_type;

//! \brief Auxiliary type used by has_result_type to test for
//! a nested typedef.
template<typename T, typename R = void>
struct enable_if_type {
    using type = R;
};

//! Checks for a nested typedef called result_type.
template<typename T, typename ENABLE = void>
struct has_result_type {
    using value = false_;
};

//! Has a nested typedef called result_type.
template<typename T>
struct has_result_type<T, typename enable_if_type<typename T::result_type>::type> {
    using value = true_;
};

//! Extracts the result type of a function (object) for composition.
template<typename F, typename SELECTOR>
struct result_type_impl {};

//! \brief Read the typedef from the function.
//!
//! This is needed to get result type for function objects: they must
//! define a nested typedef called result_type as per our compositions.
template<typename F>
struct result_type_impl<F, true_> {
    using type = typename F::result_type;
};

//! Deduce result type from function (object).
template<typename F>
struct result_type_impl<F, false_> {
    using type = typename function_result_type<F>::type;
};

//! \brief Tries to deduce the result type of a function (object)
//! in various ways.
template<typename F>
struct result_type : public result_type_impl<typename boost::remove_reference<F>::type,
                                             typename has_result_type<typename boost::remove_reference<F>::type>::value> {};

} // composition_detail::

//! \brief A collection of useful compositions of functions for the solver
//! and numerical integration functions.
//!
//! DESCRIPTION:\n
//! These provide the composition of generic functions with some useful
//! functions which crop up repeatedly. In particular,
//! <pre class="fragment">
//!   \f$f(x) = x - c\f$
//!   \f$f(x) = -x\f$
//!   \f$f(x) = e^x\f$
//!   \f$h(x) = f(x)g(x)\f$
//! </pre>
//!
//! For example, \f$x - c\f$ is often used to find the point at which a
//! function is equal to some particular value in conjunction with standard
//! root finding algorithms.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Unfortunately, because we use the boost solver we don't have a consistent
//! function signature for the two places we want to use these adapters:
//! specifically solving and integration. These overload operator() to deal
//! with this. Since a member function of a template is only instantiated
//! when needed, the functions supplied don't need to support both.
class MATHS_EXPORT CCompositeFunctions {
public:
    //! Function composition with minus a constant.
    template<typename F_, typename T = typename composition_detail::result_type<F_>::type>
    class CMinusConstant {
    public:
        using F = typename boost::remove_reference<F_>::type;
        using result_type = T;

    public:
        CMinusConstant(const F& f, double offset) : m_F(f), m_Offset(offset) {}

        //! For function returning value.
        inline T operator()(double x) const { return m_F(x) - m_Offset; }

        //! For function return success/fail and taking result as argument.
        inline bool operator()(double x, T& result) const {
            if (m_F(x, result)) {
                result -= m_Offset;
                return true;
            }
            return false;
        }

    private:
        F_ m_F;
        double m_Offset;
    };

    //! Function composition with negation.
    template<typename F_, typename T = typename composition_detail::result_type<F_>::type>
    class CMinus {
    public:
        using F = typename boost::remove_reference<F_>::type;
        using result_type = T;

    public:
        explicit CMinus(const F& f = F()) : m_F(f) {}

        //! For function returning value.
        inline T operator()(double x) const { return -m_F(x); }

        //! For function return success/fail and taking result as argument.
        inline bool operator()(double x, T& result) const {
            if (m_F(x, result)) {
                result = -result;
                return true;
            }
            return false;
        }

    private:
        F_ m_F;
    };

    //! Composition with exponentiation.
    template<typename F_, typename T = typename composition_detail::result_type<F_>::type>
    class CExp {
    public:
        using F = typename boost::remove_reference<F_>::type;
        using result_type = T;

    public:
        explicit CExp(const F& f = F()) : m_F(f) {}

        //! For function returning value.
        inline T operator()(double x) const {
            static const double LOG_MIN_DOUBLE = std::log(std::numeric_limits<double>::min());
            double fx = m_F(x);
            return fx < LOG_MIN_DOUBLE ? 0.0 : std::exp(fx);
        }

        //! For function return success/fail and taking result as argument.
        inline bool operator()(double x, T& result) const {
            static const double LOG_MIN_DOUBLE = std::log(std::numeric_limits<double>::min());
            if (m_F(x, result)) {
                result = result < LOG_MIN_DOUBLE ? 0.0 : std::exp(result);
                return true;
            }
            return false;
        }

    private:
        F_ m_F;
    };

    //! Composition of two functions by multiplication.
    template<typename F_,
             typename G_,
             typename U = typename composition_detail::result_type<F_>::type,
             typename V = typename composition_detail::result_type<G_>::type>
    class CProduct {
    public:
        using F = typename boost::remove_reference<F_>::type;
        using G = typename boost::remove_reference<G_>::type;
        using result_type = U;

    public:
        explicit CProduct(const F& f = F(), const G& g = G()) : m_F(f), m_G(g) {}

        //! For function returning value.
        inline U operator()(double x) const { return m_F(x) * m_G(x); }

        //! For function return success/fail and taking result as argument.
        inline bool operator()(double x, U& result) const {
            U fx;
            V gx;
            if (m_F(x, fx) && m_G(x, gx)) {
                result = fx * gx;
                return true;
            }
            return false;
        }

        //! Retrieve the component function f.
        const F& f() const { return m_F; }

        //! Retrieve the component function g.
        const G& g() const { return m_G; }

    private:
        F_ m_F;
        G_ m_G;
    };
};
}
}

#endif // INCLUDED_ml_maths_CCompositeFunctions_h
