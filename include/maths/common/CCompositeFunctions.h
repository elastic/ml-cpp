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

#ifndef INCLUDED_ml_maths_common_CCompositeFunctions_h
#define INCLUDED_ml_maths_common_CCompositeFunctions_h

#include <core/Constants.h>

#include <maths/common/ImportExport.h>

#include <cmath>

namespace ml {
namespace maths {
namespace common {

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
class MATHS_COMMON_EXPORT CCompositeFunctions {
public:
    //! Function composition with minus a constant.
    template<typename F_>
    class CMinusConstant {
    public:
        using F = std::remove_reference_t<F_>;

    public:
        CMinusConstant(const F& f, double offset) : m_F(f), m_Offset(offset) {}

        //! For function returning value.
        inline auto operator()(double x) const { return m_F(x) - m_Offset; }

        //! For function return success/fail and taking result as argument.
        template<typename R>
        inline bool operator()(double x, R& result) const {
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
    template<typename F_>
    class CMinus {
    public:
        using F = std::remove_reference_t<F_>;

    public:
        explicit CMinus(const F& f = F()) : m_F(f) {}

        //! For function returning value.
        inline auto operator()(double x) const { return -m_F(x); }

        //! For function return success/fail and taking result as argument.
        template<typename R>
        inline bool operator()(double x, R& result) const {
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
    template<typename F_>
    class CExp {
    public:
        using F = std::remove_reference_t<F_>;

    public:
        explicit CExp(const F& f = F()) : m_F(f) {}

        //! For function returning value.
        inline auto operator()(double x) const {
            double fx = m_F(x);
            return fx < core::constants::LOG_MIN_DOUBLE ? 0.0 : std::exp(fx);
        }

        //! For function return success/fail and taking result as argument.
        template<typename R>
        inline bool operator()(double x, R& result) const {
            if (m_F(x, result)) {
                result = result < core::constants::LOG_MIN_DOUBLE ? 0.0 : std::exp(result);
                return true;
            }
            return false;
        }

    private:
        F_ m_F;
    };

    //! Composition of two functions by multiplication.
    template<typename F_, typename G_>
    class CProduct {
    public:
        using F = std::remove_reference_t<F_>;
        using G = std::remove_reference_t<G_>;

    public:
        explicit CProduct(const F& f = F(), const G& g = G())
            : m_F(f), m_G(g) {}

        //! For function returning value.
        inline auto operator()(double x) const { return m_F(x) * m_G(x); }

        //! For function return success/fail and taking result as argument.
        template<typename R>
        inline bool operator()(double x, R& result) const {
            R fx;
            R gx;
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
}

#endif // INCLUDED_ml_maths_common_CCompositeFunctions_h
