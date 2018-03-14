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

#ifndef INCLUDED_ml_maths_CSpline_h
#define INCLUDED_ml_maths_CSpline_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>

#include <maths/CChecksum.h>
#include <maths/CTools.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <algorithm>
#include <string>
#include <vector>

#include <stdint.h>


namespace ml {
namespace maths {

namespace spline_detail {

typedef std::vector<double> TDoubleVec;
typedef std::vector<CFloatStorage> TFloatVec;

//! Solves \f$Ax = y\f$ where \f$A\f$ is a tridiagonal matrix
//! consisting of vectors \f$a\f$, \f$b\f$ and \f$c\f$.
//!
//! \param[in] a The subdiagonal indexed from [0, ..., n - 2].
//! \param[in] b The main diagonal, indexed from [0, ..., n - 1].
//! \param[in] c The superdiagonal, indexed from [0, ..., N - 2].
//! \param[in,out] x Initially contains the input vector \f$y\f$,
//! and returns the solution \f$x\f$, indexed from [0, ..., n - 1].
//! \note The contents of input vector c will be modified.
bool MATHS_EXPORT solveTridiagonal(const TDoubleVec &a,
                                   const TDoubleVec &b,
                                   TDoubleVec &c,
                                   TDoubleVec &x);

//! Solves:
//! <pre class="fragment">
//!   \f$\displaystyle \left(A + \begin{bmatrix} 1\\ 0\\ 0\\ \cdot\\u \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 & \cdot & v \end{bmatrix}\right)x=y\f$
//! </pre>
//!
//! Here, \f$A\f$ is a tridiagonal matrix consisting of vectors
//! \f$a\f$, \f$b\f$ and \f$c\f$.
//!
//! \param[in] a The subdiagonal indexed from [0, ..., n - 2].
//! \param[in] b The main diagonal, indexed from [0, ..., n - 1].
//! \param[in] c The superdiagonal, indexed from [0, ..., N - 2].
//! \param[in,out] x Initially contains the input vector \f$y\f$,
//! and returns the solution \f$x\f$, indexed from [0, ..., n - 1].
//! \note The contents of input vector c will be modified.
bool MATHS_EXPORT solvePeturbedTridiagonal(const TDoubleVec &a,
                                           const TDoubleVec &b,
                                           TDoubleVec &c,
                                           TDoubleVec &u,
                                           const TDoubleVec &v,
                                           TDoubleVec &x);

}

//! \brief Defines types used by the spline implementation.
class MATHS_EXPORT CSplineTypes {
    public:
        typedef std::vector<double> TDoubleVec;
        typedef CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

        //! Types of spline interpolation that this will perform.
        //!
        //! -# Linear interpolation of the knot points.
        //! -# Standard cubic spline. The boundary conditions are
        //!    chosen by a separate enumeration.
        enum EType {
            E_Linear,
            E_Cubic
        };

        //! Types of boundary condition for the spline.
        //!
        //! -# The natural boundary condition sets curvature to zero
        //!    at the start and end of the interpolated interval.
        //! -# The parabolic run out sets curvature at the start and
        //!    end of the interval to the curvature at the second and
        //!    penultimate knot point, respectively.
        //! -# For the periodic boundary condition we identify the
        //!    start and end of the interval and apply the usual
        //!    spline condition that the first derivative is equal
        //!    across the knot point.
        enum EBoundaryCondition {
            E_Natural,
            E_ParabolicRunout,
            E_Periodic
        };
};

//! \brief Implements a standard cubic spline.
//!
//! DESCRIPTION:\n
//! The standard cubic spline is an interpolation of a set of function
//! points, \f$(x, y(x))\f$. A set of knot points \f$x_i\f$ are defined
//! and the function is interpolated between these points using piecewise
//! cubic function, i.e.
//! <pre class="fragment">
//!   \f$y_i(x) = a_i(x-x_i)^3 + b_i(x-x_i)^2 + c_i(x-x_i) + d_i\f$
//! </pre>
//!
//! The spline conditions require that the function is continuous and
//! conditionally differentiable to order 2. In addition, the user must
//! specify some boundary conditions at the first and last knot points.
//! We support some standard types. However, for our particular use
//! cases it is also desirable to support a slightly non-standard boundary
//! condition, namely periodic. This corresponds to a function which is
//! periodic with period equal to the interpolation interval.
//!
//! \see http://en.wikipedia.org/wiki/Spline_(mathematics) for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The types of the knot, value and curvatures collections are templated
//! principally so that splines can be defined on top of different precision
//! floating point numbers and also so that reference splines can be created.
//! Reference splines work on collections that are stored elsewhere, so that,
//! for example knot points can be shared between several splines.
template<typename KNOTS = std::vector<CFloatStorage>,
         typename VALUES = std::vector<CFloatStorage>,
         typename CURVATURES = std::vector<double> >
class CSpline : public CSplineTypes {
    public:
        typedef typename boost::unwrap_reference<KNOTS>::type TKnots;
        typedef typename boost::unwrap_reference<VALUES>::type TValues;
        typedef typename boost::unwrap_reference<CURVATURES>::type TCurvatures;
        typedef typename boost::remove_const<TKnots>::type TNonConstKnots;
        typedef typename boost::remove_const<TValues>::type TNonConstValues;
        typedef typename boost::remove_const<TCurvatures>::type TNonConstCurvatures;

    public:
        CSpline(EType type) : m_Type(type) {
        }

        CSpline(EType type,
                const KNOTS &knots,
                const VALUES &values,
                const CURVATURES &curvatures) :
            m_Type(type),
            m_Knots(knots),
            m_Values(values),
            m_Curvatures(curvatures) {
        }

        //! Efficiently swap the contents of two spline objects.
        void swap(CSpline &other) {
            using std::swap;
            swap(m_Type, other.m_Type);
            swap(m_Knots, other.m_Knots);
            swap(m_Values, other.m_Values);
            swap(m_Curvatures, other.m_Curvatures);
        }

        //! Check if the spline has been initialized.
        bool initialized(void) const {
            return this->knots().size() > 0;
        }

        //! Clear the contents of this spline and recover any
        //! allocated memory.
        void clear(void) {
            TNonConstKnots noKnots;
            this->knotsRef().swap(noKnots);
            TNonConstValues noValues;
            this->valuesRef().swap(noValues);
            TNonConstCurvatures noCurvatures;
            this->curvaturesRef().swap(noCurvatures);
        }

        //! Evaluate the value of the spline at the point \p x.
        //!
        //! \warning \p x should be in the interpolation interval as
        //! defined by the last set of knot points supplied to the
        //! interpolate function.
        double value(double x) const {
            if (this->knots().empty()) {
                return 0.0;
            }

            std::size_t k = CTools::truncate(
                std::size_t(std::lower_bound(this->knots().begin(),
                                             this->knots().end(), x)
                            - this->knots().begin()),
                std::size_t(1), this->knots().size() - 1);

            if (x == this->knots()[k]) {
                return this->values()[k];
            }

            switch (m_Type) {
                case E_Linear: {
                    double h = this->knots()[k] - this->knots()[k-1];
                    double c = (this->values()[k] - this->values()[k-1]) / h;
                    double d = this->values()[k-1];
                    double r = x - this->knots()[k-1];
                    return c * r + d;
                }
                case E_Cubic: {
                    double h = this->knots()[k] - this->knots()[k-1];
                    double a = (this->curvatures()[k] - this->curvatures()[k-1]) / 6.0 / h;
                    double b = this->curvatures()[k-1] / 2.0;
                    double c = (this->values()[k] - this->values()[k-1]) / h
                               - (this->curvatures()[k] / 6.0 + this->curvatures()[k-1] / 3.0) * h;
                    double d = this->values()[k-1];
                    double r = x - this->knots()[k-1];
                    return ((a * r + b) * r + c) * r + d;
                }
            }

            LOG_ABORT("Unexpected type " << m_Type);
        }

        //! Get the mean value of the spline.
        double mean(void) const {
            if (this->knots().empty()) {
                return 0.0;
            }

            std::size_t n = this->knots().size();
            double      interval = (this->knots()[n-1] - this->knots()[0]);

            TMeanAccumulator result;
            switch (m_Type) {
                case E_Linear:
                    for (std::size_t i = 1u; i < this->knots().size(); ++i) {
                        double h = this->knots()[i] - this->knots()[i-1];
                        double c = (this->values()[i] - this->values()[i-1]) / h;
                        double d = this->values()[i-1];
                        result.add(c / 2.0 * h + d,  h / interval);
                    }
                    break;

                case E_Cubic:
                    for (std::size_t i = 1u; i < this->knots().size(); ++i) {
                        double h = this->knots()[i] - this->knots()[i-1];
                        double a = (this->curvatures()[i] - this->curvatures()[i-1]) / 6.0 / h;
                        double b = this->curvatures()[i-1] / 2.0;
                        double c = (this->values()[i] - this->values()[i-1]) / h
                                   - (this->curvatures()[i] / 6.0 + this->curvatures()[i-1] / 3.0) * h;
                        double d = this->values()[i-1];
                        result.add(((a * h / 4.0 + b / 3.0) * h + c / 2.0) * h + d, h / interval);
                    }
                    break;
            }

            return CBasicStatistics::mean(result);
        }

        //! Evaluate the slope of the spline at the point \p x.
        //!
        //! \warning \p x should be in the interpolation interval as
        //! defined by the last set of knot points supplied to the
        //! interpolate function.
        double slope(double x) const {
            if (this->knots().empty()) {
                return 0.0;
            }

            std::size_t k = CTools::truncate(
                std::size_t(std::lower_bound(this->knots().begin(),
                                             this->knots().end(), x)
                            - this->knots().begin()),
                std::size_t(1), this->knots().size() - 1);

            switch (m_Type) {
                case E_Linear: {
                    double h = this->knots()[k] - this->knots()[k-1];
                    return (this->values()[k] - this->values()[k-1]) / h;
                }
                case E_Cubic: {
                    double h = this->knots()[k] - this->knots()[k-1];
                    double a = (this->curvatures()[k] - this->curvatures()[k-1]) / 6.0 / h;
                    double b = this->curvatures()[k-1] / 2.0;
                    double c = (this->values()[k] - this->values()[k-1]) / h
                               - (this->curvatures()[k] / 6.0 + this->curvatures()[k-1] / 3.0) * h;
                    double r = x - this->knots()[k-1];
                    return ((3.0 * a * r + 2.0 * b) * r + c);
                }
            }

            LOG_ABORT("Unexpected type " << m_Type);
        }

        //! Compute the mean absolute slope of the spline.
        //!
        //! This is defined as
        //! <pre class="fragment">
        //!   \f$\frac{1}{|b-a|}\int_{[a,b]}{\left|\frac{df(s)}{ds}\right|}ds\f$
        //! </pre>
        double absSlope(void) const {
            double result = 0.0;

            std::size_t n = this->knots().size();

            switch (m_Type) {
                case E_Linear:
                    for (std::size_t i = 1u; i < n; ++i) {
                        result += ::fabs((this->values()[i] - this->values()[i-1]));
                    }
                    break;

                case E_Cubic:
                    for (std::size_t i = 1u; i < n; ++i) {
                        double a = this->knots()[i-1];
                        double b = this->knots()[i];
                        double h = b - a;
                        double ai = (this->curvatures()[i] - this->curvatures()[i-1]) / 6.0 / h;
                        double bi = this->curvatures()[i-1] / 2.0;
                        double ci = (this->values()[i] - this->values()[i-1]) / h
                                    - (this->curvatures()[i] / 6.0 + this->curvatures()[i-1] / 3.0) * h;

                        double descriminant = bi * bi - 3.0 * ai * ci;
                        if (descriminant < 0.0) {
                            result += ::fabs(((ai * h + bi) * h + ci) * h);
                            continue;
                        }
                        double rl = CTools::truncate(a - ( bi + descriminant) / 3.0 / ai, a, b);
                        double rr = CTools::truncate(a + (-bi + descriminant) / 3.0 / ai, a, b);
                        if (rl > rr) {
                            std::swap(rl, rr);
                        }
                        result += ::fabs(((ai * (rl - a)  + bi) * (rl - a)  + ci) * (rl - a))
                                  + ::fabs(((ai * (rr - rl) + bi) * (rr - rl) + ci) * (rr - rl))
                                  + ::fabs(((ai * (b - rr)  + bi) * (b - rr)  + ci) * (b - rr));
                    }
                    break;
            }

            return result / (this->knots()[n-1] - this->knots()[0]);
        }

        //! Get the specified curvatures.
        //!
        //! \param[out] a Filled in with the cubic coefficient.
        //! \param[out] b Filled in with the quadratic coefficient.
        //! \param[out] c Filled in with the linear coefficient.
        //! \param[out] d Filled in with the constant.
        //! \note Null pointers are ignored.
        void coefficients(TDoubleVec *a = 0,
                          TDoubleVec *b = 0,
                          TDoubleVec *c = 0,
                          TDoubleVec *d = 0) const {
            if (a) a->reserve(this->values().size());
            if (b) b->reserve(this->values().size());
            if (c) c->reserve(this->values().size());
            if (d) d->reserve(this->values().size());

            switch (m_Type) {
                case E_Linear:
                    for (std::size_t i = 1u; i < this->knots().size(); ++i) {
                        double h = this->knots()[i] - this->knots()[i-1];
                        if (a) a->push_back(0.0);
                        if (b) b->push_back(0.0);
                        if (c) c->push_back((this->values()[i] - this->values()[i-1]) / h);
                        if (d) d->push_back(this->values()[i-1]);
                    }
                    break;

                case E_Cubic:
                    for (std::size_t i = 1u; i < this->knots().size(); ++i) {
                        double h = this->knots()[i] - this->knots()[i-1];
                        if (a) a->push_back((this->curvatures()[i] - this->curvatures()[i-1]) / 6.0 / h);
                        if (b) b->push_back(this->curvatures()[i-1] / 2.0);
                        if (c) c->push_back(  (this->values()[i] - this->values()[i-1]) / h
                                              - (this->curvatures()[i] / 6.0 + this->curvatures()[i-1] / 3.0) * h);
                        if (d) d->push_back(this->values()[i-1]);
                    }
                    break;
            }
        }

        //! Interpolate the function, using the selected spline style,
        //! on the knot points \p knots, with the values \p values,
        //! and applying the boundary conditions \p boundary.
        //!
        //! \param[in] knots The knot points for the spline.
        //! \param[in] values The values of the function at \p knots.
        //! \param[in] boundary Selects the boundary condition to use
        //! for the interpolation. See EBoundaryCondition for more
        //! details on this argument.
        //! \warning \p knots must be ordered increasing.
        //! \warning There must be a one-to-one correspondence between
        //! \p knots and \p values.
        //! \note If \p knots contain duplicates the standard spline
        //! is ill posed. This implementation removes duplicates and
        //! sets the function value to the mean of function values
        //! over the duplicates.
        bool interpolate(const TDoubleVec &knots,
                         const TDoubleVec &values,
                         EBoundaryCondition boundary) {
            if (knots.size() < 2) {
                LOG_ERROR("Insufficient knot points supplied");
                return false;
            }
            if (knots.size() != values.size()) {
                LOG_ERROR("Number knots not equal to number of values: "
                          << " knots = " << core::CContainerPrinter::print(knots)
                          << " values = " << core::CContainerPrinter::print(values));
                return false;
            }

            TNonConstKnots  oldKnots;
            TNonConstValues oldValues;
            oldKnots.swap(this->knotsRef());
            oldValues.swap(this->valuesRef());
            this->knotsRef().assign(knots.begin(), knots.end());
            this->valuesRef().assign(values.begin(), values.end());

            // If two knots are equal to working precision then we
            // de-duplicate and use the average of the values at the
            // duplicate knots. The exit condition from this loop is
            // rather subtle and ensures that "last" always points
            // to the last element in the reduced knot set.
            std::size_t last = std::numeric_limits<std::size_t>::max();
            std::size_t n = this->knots().size();
            for (std::size_t i = 1u; i <= n; ++i) {
                std::size_t i_ = i-1;
                double      knot = this->knots()[i_];
                for (/**/; i < n && this->knots()[i] == knot; ++i) {
                }
                if (i - i_ > 1) {
                    TMeanAccumulator value;
                    for (std::size_t j = i_; j < i; ++j) {
                        value.add(this->values()[j]);
                    }
                    this->valuesRef()[i_] = CBasicStatistics::mean(value);
                }
                if (++last != i_) {
                    this->knotsRef()[last] = this->knots()[i_];
                    this->valuesRef()[last] = this->values()[i_];
                }
            }
            this->knotsRef().erase(this->knotsRef().begin() + last + 1, this->knotsRef().end());
            this->valuesRef().erase(this->valuesRef().begin() + last + 1, this->valuesRef().end());
            n = this->knots().size();
            LOG_TRACE("knots = " << core::CContainerPrinter::print(this->knots()));
            LOG_TRACE("values = " << core::CContainerPrinter::print(this->values()));

            if (this->knots().size() < 2) {
                LOG_ERROR("Insufficient distinct knot points supplied");
                this->knotsRef().swap(oldKnots);
                this->valuesRef().swap(oldValues);
                return false;
            }

            switch (m_Type) {
                case E_Linear:
                    // Curvatures are all zero and we don't bother to store them.
                    break;

                case E_Cubic: {
                    this->curvaturesRef().clear();
                    this->curvaturesRef().reserve(n);

                    // Construct the diagonals: a is the subdiagonal, b is the
                    // main diagonal and c is the superdiagonal.

                    TDoubleVec a;
                    TDoubleVec b;
                    TDoubleVec c;
                    a.reserve(n - 1);
                    b.reserve(n);
                    c.reserve(n - 1);

                    double h = this->knots()[1] - this->knots()[0];
                    double h_ = this->knots()[n-1] - this->knots()[n-2];

                    switch (boundary) {
                        case E_Natural:
                            b.push_back(1.0);
                            c.push_back(0.0);
                            this->curvaturesRef().push_back(0.0);
                            break;

                        case E_ParabolicRunout:
                            b.push_back( 1.0);
                            c.push_back(-1.0);
                            this->curvaturesRef().push_back(0.0);
                            break;

                        case E_Periodic:
                            b.push_back(2.0 * (h + h_));
                            c.push_back(h - 1.0);
                            this->curvaturesRef().push_back(6.0 * (  (this->values()[1] - this->values()[0]) / h
                                                                     - (this->values()[0] - this->values()[n-2]) / h_));
                            break;
                    }

                    for (std::size_t i = 1u; i + 1 < n; ++i) {
                        h_ = h;
                        h = this->knots()[i+1] - this->knots()[i];
                        a.push_back(h_);
                        b.push_back(2.0 * (h + h_));
                        c.push_back(h);
                        this->curvaturesRef().push_back(6.0 * (  (this->values()[i+1] - this->values()[i]) / h
                                                                 - (this->values()[i] - this->values()[i-1]) / h_));
                    }

                    h_ = h;
                    h = this->knots()[1] - this->knots()[0];

                    switch (boundary) {
                        case E_Natural:
                            a.push_back(0.0);
                            b.push_back(1.0);
                            this->curvaturesRef().push_back(0.0);
                            if (!spline_detail::solveTridiagonal(a, b, c, this->curvaturesRef())) {
                                LOG_ERROR("Failed to calculate curvatures");
                                return false;
                            }
                            break;

                        case E_ParabolicRunout:
                            a.push_back(-1.0);
                            b.push_back( 1.0);
                            this->curvaturesRef().push_back(0.0);
                            if (!spline_detail::solveTridiagonal(a, b, c, this->curvaturesRef())) {
                                LOG_ERROR("Failed to calculate curvatures");
                                return false;
                            }
                            break;

                        case E_Periodic: {
                            a.push_back(h_ * (1.0 - h));
                            b.push_back(2.0 * (h + h_));
                            TDoubleVec u(n, 0.0);
                            u[0] = 1.0;
                            u[n-1] = h;
                            TDoubleVec v(n, 0.0);
                            v[1] = 1.0;
                            v[n-2] = h_;
                            this->curvaturesRef().push_back(6.0 * (  (this->values()[1] - this->values()[n-1]) / h
                                                                     - (this->values()[n-1] - this->values()[n-2]) / h_));
                            if (!spline_detail::solvePeturbedTridiagonal(a, b, c, u, v, this->curvaturesRef())) {
                                LOG_ERROR("Failed to calculate curvatures");
                                return false;
                            }
                        }
                        break;
                    }
                    break;
                }
            }

            return true;
        }

        //! Get a human readable description of this spline.
        //!
        //! \param[in] indent The indent to use at the start of new lines.
        //! \param[in,out] result Filled in with the description.
        void describe(const std::string &indent, std::string &result) const {
            result += "\n" + indent + "cubic spline";
            if (!this->initialized()) {
                result += " zero everywhere";
                return;
            }

            result += ":";
            switch (m_Type) {
                case E_Linear:
                    for (std::size_t i = 1u; i < this->knots().size(); ++i) {
                        double      h = this->knots()[i] - this->knots()[i-1];
                        double      c =  (this->values()[i] - this->values()[i-1]) / h;
                        double      d = this->values()[i-1];
                        std::string kl = core::CStringUtils::typeToStringPretty(this->knots()[i-1]);
                        result += "\n" + indent + core::CStringUtils::typeToStringPretty(c) + " (x - " + kl + ") + "
                                  + core::CStringUtils::typeToStringPretty(d)
                                  + "   x in [" + kl + "," + core::CStringUtils::typeToStringPretty(this->knots()[i]) + ")";
                    }
                    break;

                case E_Cubic:
                    for (std::size_t i = 1u; i < this->knots().size(); ++i) {
                        double h = this->knots()[i] - this->knots()[i-1];
                        double a = (this->curvatures()[i] - this->curvatures()[i-1]) / 6.0 / h;
                        double b = this->curvatures()[i-1] / 2.0;
                        double c = (this->values()[i] - this->values()[i-1]) / h
                                   - (this->curvatures()[i] / 6.0 + this->curvatures()[i-1] / 3.0) * h;
                        double      d = this->values()[i-1];
                        std::string kl = core::CStringUtils::typeToStringPretty(this->knots()[i-1]);
                        result += "\n" + indent + core::CStringUtils::typeToStringPretty(a) + " (x - " + kl + ")^3 + "
                                  + core::CStringUtils::typeToStringPretty(b) + " (x - " + kl + ")^2 + "
                                  + core::CStringUtils::typeToStringPretty(c) + " (x - " + kl + ") + "
                                  + core::CStringUtils::typeToStringPretty(d)
                                  + "   x in [" + kl + "," + core::CStringUtils::typeToStringPretty(this->knots()[i]) + ")";
                    }
                    break;
            }
        }

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const {
            seed = CChecksum::calculate(seed, m_Type);
            seed = CChecksum::calculate(seed, m_Knots);
            seed = CChecksum::calculate(seed, m_Values);
            return CChecksum::calculate(seed, m_Curvatures);
        }

        //! Get the memory used by this component
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
            mem->setName("CSpline");
            core::CMemoryDebug::dynamicSize("m_Knots", m_Knots, mem);
            core::CMemoryDebug::dynamicSize("m_Values", m_Values, mem);
            core::CMemoryDebug::dynamicSize("m_Curvatures", m_Curvatures, mem);
        }

        //! Get the memory used by this component
        std::size_t memoryUsage(void) const {
            std::size_t mem = core::CMemory::dynamicSize(m_Knots);
            mem += core::CMemory::dynamicSize(m_Values);
            mem += core::CMemory::dynamicSize(m_Curvatures);
            return mem;
        }

        //! Get the knot points of the spline.
        inline const TNonConstKnots &knots(void) const {
            return boost::unwrap_ref(m_Knots);
        }

        //! Get the values at the knot points of the spline.
        inline const TNonConstValues &values(void) const {
            return boost::unwrap_ref(m_Values);
        }

        //! Get the curvatures at the knot points of the spline.
        inline const TNonConstCurvatures &curvatures(void) const {
            return boost::unwrap_ref(m_Curvatures);
        }

    private:
        //! Get the knot points of the spline.
        inline TKnots &knotsRef(void) {
            return boost::unwrap_ref(m_Knots);
        }

        //! Get the values at the knot points of the spline.
        inline TNonConstValues &valuesRef(void) {
            return boost::unwrap_ref(m_Values);
        }

        //! Get the curvatures at the knot points of the spline.
        inline TCurvatures &curvaturesRef(void) {
            return boost::unwrap_ref(m_Curvatures);
        }

    private:
        //! The type of spline.
        EType m_Type;
        //! The spline knot points.
        KNOTS m_Knots;
        //! The spline values at the knot points.
        VALUES m_Values;
        //! The spline curvatures at the knot points.
        CURVATURES m_Curvatures;
};

}
}

#endif // INCLUDED_ml_maths_CSpline_h
