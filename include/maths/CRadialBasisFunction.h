/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CRadialBasisFunction_h
#define INCLUDED_ml_maths_CRadialBasisFunction_h

#include <maths/ImportExport.h>

namespace ml
{
namespace maths
{

//! \brief Common interface implemented by all our radial basis
//! functions.
//!
//! DESCRIPTION:\n
//! This implements hierarchy implements some common
//! <a href="http://en.wikipedia.org/wiki/Radial_basis_function_network">radial basis functions</a>,
//! for use with our interpolation algorithms.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The radial basis function hierarchy is stateless to keep the
//! representation as compact as possible. (All functions are
//! supplied the centre.)
class MATHS_EXPORT CRadialBasisFunction
{
    public:
        virtual ~CRadialBasisFunction();

        //! Create a copy of this object.
        //!
        //! \warning The caller owns this copy.
        virtual CRadialBasisFunction *clone() const = 0;

        //! \brief Evaluate the basis function with centre \p centre
        //! at the point \p x.
        virtual double value(double x,
                             double centre,
                             double scale = 1.0) const = 0;

        //! \brief Evaluate the derivative of the basis function w.r.t.
        //! its argument x, with centre \p centre at the point
        //! \p x.
        virtual double derivative(double x,
                                  double centre,
                                  double scale = 1.0) const = 0;

        //! \brief Solves for the scale that gives the \p value at a
        //! distance \p distance from the centre of the radial basis
        //! function, i.e. the value \f$\epsilon^*\f$ s.t.
        //! <pre class="fragment">
        //!   \f$\displaystyle \phi_{\epsilon^*}(\left \|d - c \right \|) = v\f$
        //! </pre>
        //!
        //! \note That \p value must be in the range (0, 1).
        virtual bool scale(double distance,
                           double value,
                           double &result) const = 0;

        //! \brief Get the mean value of this function on the interval
        //! [\p a, \p b], i.e. the result of:
        //! <pre class="fragment">
        //!   \f$\displaystyle \frac{1}{b - a}\int_{[a,b]}{\phi_{\epsilon}(\left \|u - c \right \|)}du\f$
        //! </pre>
        //!
        //! \note \p b should be greater than or equal to \p a.
        virtual double mean(double a,
                            double b,
                            double centre,
                            double scale = 1.0) const = 0;

        //! \brief Get the mean square derivative of the basis function
        //! on the interval [\p a, \p b], i.e. the result of:
        //! <pre class="fragment">
        //!  \f$\displaystyle \frac{1}{b - a}\int_{[a,b]}{\phi_{\epsilon}'(\left \|u - c \right \|)^2}du\f$
        //! </pre>
        //!
        //! \note \p b should be greater than or equal to \p a.
        virtual double meanSquareDerivative(double a,
                                            double b,
                                            double centre,
                                            double scale = 1.0) const = 0;

        //! \brief Get the integral of the product of two basis functions
        //! on the interval \f$[a,b]\f$, i.e.
        //! <pre class="fragment">
        //!   \f$\displaystyle \frac{1}{b - a} \int_a^b{\phi_{\epsilon}(\left \|u - c_1 \right \|)\phi_{\epsilon}(\left \|u - c_2 \right \|)}du\f$
        //! </pre>
        virtual double product(double a,
                               double b,
                               double centre1,
                               double centre2,
                               double scale1 = 1.0,
                               double scale2 = 1.0) const = 0;
};

//! \brief The Gaussian radial basis function.
//!
//! DESCRIPTION:\n
//! Implements
//! <pre class="fragment">
//!  \f$\displaystyle \phi_{\epsilon}(x) = e^{-(\epsilon \left \|x - c \right \|)^2}\f$
//! </pre>
//!
//! Here, \f$\epsilon\f$ denotes the scale and \f$c\f$ the centre
//! of the basis function.
class MATHS_EXPORT CGaussianBasisFunction : public CRadialBasisFunction
{
    public:
        //! Create a copy of this object.
        //!
        //! \warning The caller owns this copy.
        virtual CGaussianBasisFunction *clone() const;

        //! \brief Evaluate the basis function with centre \p centre
        //! at the point \p x.
        virtual double value(double x,
                             double centre,
                             double scale = 1.0) const;

        //! \brief Evaluate the derivative of the basis function w.r.t.
        //! its argument x, with centre \p centre at the point
        //! \p x.
        virtual double derivative(double x,
                                  double centre,
                                  double scale = 1.0) const;

        //! \brief Solves for the scale that gives the \p value at a
        //! distance \p distance from the centre of the radial basis
        //! function.
        virtual bool scale(double distance,
                           double value,
                           double &result) const;

        //! \brief Get the mean value of this function on the specified
        //! interval [\p a, \p b].
        virtual double mean(double a,
                            double b,
                            double centre,
                            double scale = 1.0) const;

        //! \brief Get the mean square derivative of the basis function
        //! on the interval [\p a, \p b], i.e. the result of:
        virtual double meanSquareDerivative(double a,
                                            double b,
                                            double centre,
                                            double scale = 1.0) const;

        //! \brief Get the integral of the product of two basis functions
        //! on the interval [\p a, \p b].
        virtual double product(double a,
                               double b,
                               double centre1,
                               double centre2,
                               double scale1 = 1.0,
                               double scale2 = 1.0) const;
};

//! \brief The inverse quadratic radial basis function.
//!
//! DESCRIPTION:\n
//! Implements
//! <pre class="fragment">
//!  \f$\displaystyle \phi_{\epsilon}(x) = \frac{1}{1 + (\epsilon \left \|x - c \right \|)^2}\f$
//! </pre>
//!
//! Here, \f$\epsilon\f$ denotes the scale and \f$c\f$ the centre
//! of the basis function.
class MATHS_EXPORT CInverseQuadraticBasisFunction : public CRadialBasisFunction
{
    public:
        //! Create a copy of this object.
        //!
        //! \warning The caller owns this copy.
        CInverseQuadraticBasisFunction *clone() const;

        //! \brief Evaluate the basis function with centre \p centre
        //! at the point \p x.
        virtual double value(double x,
                             double centre,
                             double scale = 1.0) const;

        //! \brief Evaluate the derivative of the basis function w.r.t.
        //! its argument x, with centre \p centre at the point
        //! \p x.
        virtual double derivative(double x,
                                  double centre,
                                  double scale = 1.0) const;

        //! \brief Solves for the scale that gives the \p value at a
        //! distance \p distance from the centre of the radial basis
        //! function.
        virtual bool scale(double distance,
                           double value,
                           double &result) const;

        //! \brief Get the mean value of this function on the specified
        //! interval [\p a, \p b].
        virtual double mean(double a,
                            double b,
                            double centre,
                            double scale = 1.0) const;

        //! \brief Get the mean square derivative of the basis function
        //! on the interval [\p a, \p b], i.e. the result of:
        virtual double meanSquareDerivative(double a,
                                            double b,
                                            double centre,
                                            double scale = 1.0) const;

        //! \brief Get the integral of the product of two basis functions
        //! on the interval [\p a, \p b].
        virtual double product(double a,
                               double b,
                               double centre1,
                               double centre2,
                               double scale1 = 1.0,
                               double scale2 = 1.0) const;
};

}
}

#endif // INCLUDED_ml_maths_CRadialBasisFunction_h
