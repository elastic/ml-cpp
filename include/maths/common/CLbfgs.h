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

#ifndef INCLUDED_ml_maths_common_CLbfgs_h
#define INCLUDED_ml_maths_common_CLbfgs_h

#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CPRNG.h>
#include <maths/common/CSampling.h>
#include <maths/common/CSolvers.h>
#include <maths/common/CTools.h>

#include <boost/circular_buffer.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace common {
//! \brief The limited memory BFGS algorithm.
//!
//! DESCRIPTION:\n
//! The basic implementation uses a low rank approximation to the Hessian to compute
//! the search direction and line search with back tracking.
//!
//! For more information \see https://en.wikipedia.org/wiki/Limited-memory_BFGS
//!
//! This also provides a constrained version which uses an ADMM scheme for minimizing
//! a function f in box. Normal L-BFGS is used in the inner loop to minimize the
//! augmented Lagrangian w.r.t. the domain of f, the minimization w.r.t. the slack
//! variables is possible in closed form.
template<typename VECTOR>
class CLbfgs {
public:
    //! The scale to apply to apply in the Armijo condition in back tracking.
    static const double BACKTRACKING_MIN_DECREASE;

    //! The scale which is applied to the step size in back tracking.
    static const double STEP_SCALE;

    //! The maximum number of iterations to use backtracking.
    static const std::size_t MAXIMUM_BACK_TRACKING_ITERATIONS;

    //! The maximum number of random probes we'll use to look for a direction in which
    //! the objective decreases if the gradient function returns zero.
    static const std::size_t MAXIMUM_RANDOM_PROBES;

public:
    explicit CLbfgs(std::size_t rank,
                    double decrease = BACKTRACKING_MIN_DECREASE,
                    double scale = STEP_SCALE)
        : m_Rank{rank}, m_StepScale{scale}, m_BacktrackingMinDecrease{decrease} {}

    //! Minimise \p f using the starting point \p x0.
    //!
    //! \param f The function to minimise.
    //! \param g The gradient of the function to minimise.
    //! \param x0 The point in the domain of f from which to start the search.
    //! \param eps The convergence tolerance applied to the ratio between the function
    //! decrease in the last step and the function decrease since the start. The main
    //! loop will exit when \f$|f(x_k) - f(x_{k-1})| < \epsilon |f(x_k) - f(x_0)|\f$.
    //! \param iterations The maximum number of iterations of the main loop to perform.
    //!
    //! \tparam F must be a Callable with single argument type VECTOR and returning
    //! a scalar.
    //! \tparam G must be a Callable with single argument type VECTOR and returning
    //! a VECTOR.
    //!
    //! \warning If you use lambdas for F and G you must ensure you don't return
    //! Eigen expressions which can easily reference temporary variables which are
    //! out of scope. You can avoid this for example by defining the lambda return
    //! type explicitly.
    template<typename F, typename G>
    std::pair<VECTOR, double>
    minimize(const F& f, const G& g, VECTOR x0, double eps = 1e-8, std::size_t iterations = 50) {

        this->reinitialize(f, x0);

        VECTOR x{std::move(x0)};

        for (std::size_t i = 0; i < iterations; ++i) {
            this->updateDescentDirection(g, x);
            x = this->lineSearch(f);
            if (this->converged(eps)) {
                break;
            }
        }

        return {std::move(x), m_Fx};
    }

    //! Minimize \p f in the bounding box with corners \p a and \p b.
    //!
    //! \param f The function to minimise.
    //! \param g The gradient of the function to minimise.
    //! \param a The bottom left corner of the constraint bounding box.
    //! \param b The top right corner of the constraint bounding box.
    //! \param x0 The point in the domain of f from which to start the search.
    //! \param eps The convergence tolerance applied to the ratio between the function
    //! decrease in the last step and the function decrease since the start. The main
    //! loop will exit when \f$|f(x_k) - f(x_{k-1})| < \epsilon |f(x_k) - f(x_0)|\f$.
    //! \param iterations The maximum number of iterations of the main loop to perform.
    //!
    //! \tparam F must be a Callable with single argument type VECTOR and returning
    //! a scalar.
    //! \tparam G must be a Callable with single argument type VECTOR and returning
    //! a VECTOR.
    template<typename F, typename G>
    std::pair<VECTOR, double> constrainedMinimize(const F& f,
                                                  const G& g,
                                                  const VECTOR& a,
                                                  const VECTOR& b,
                                                  VECTOR x0,
                                                  double rho,
                                                  double eps = 1e-8,
                                                  std::size_t iterations = 50) {

        // We use an ADMM update scheme to compute the minimum with respect to the
        // bounding box constraints. This is less efficient but significantly simpler
        // than L-BFGS-B.
        //
        // In particular, we want to solve
        //   argmin_x f(x)
        //  s.t. a <= x <= b                                                   (1)
        //
        // We convert the inequality to an affine constraint (suitable for ADMM) by
        // introducing slack variables, i.e. we transform (1) as follows
        //   argmin_x,z f(x) + g(z_1) g(z_2)
        //     s.t. x - z_1 = a
        //          x + z_2 = b
        //
        // where the function g(z) is infinite for z < 0 and 0 otherwise. Note that
        // the minimization over z_1 and z_2 can be done in closed form and are just
        // the closest non-negative points to x - a + w_1 and b - x - w_2, respectively.

        // Get the closest feasible point to x0.
        VECTOR x{std::move(x0)};
        las::max(a, x);
        las::min(b, x);

        VECTOR zero{las::zero(x)};

        VECTOR z1{zero};
        VECTOR z2{zero};
        VECTOR w1{zero};
        VECTOR w2{zero};
        VECTOR r1;
        VECTOR r2;

        // Functions to compute the augmented Lagrangian and its gradient w.r.t. x.
        auto al = [&](const VECTOR& x_) {
            r1 = x_ - z1 + w1 - a;
            r2 = x_ + z2 + w2 - b;
            double n1{las::norm(r1)};
            double n2{las::norm(r2)};
            // Explicitly construct the return type before returning from the lambda,
            // otherwise the auto return type may be an Eigen closure type which
            // references temporaries
            return double{f(x_) + 0.5 * rho * (n1 * n1 + n2 * n2)};
        };
        auto gal = [&](const VECTOR& x_) {
            // Explicitly construct the return type before returning from the lambda,
            // otherwise the auto return type may be an Eigen closure type which
            // references temporaries
            return VECTOR{g(x_) + rho * (2 * x_ - z1 + z2 - a - b + w1 + w2)};
        };

        VECTOR xmin;
        double fmin;

        double f0{f(x)};
        double fl{f0};
        for (std::size_t i = 0; i < iterations; ++i) {
            // x-minimization.
            std::tie(x, std::ignore) = this->minimize(al, gal, x, eps, iterations);

            // z-minimization.
            z1 = x - a + w1;
            z2 = b - x - w2;
            las::max(zero, z1);
            las::max(zero, z2);

            // Dual update.
            w1 += x - z1 - a;
            w2 += x + z2 - b;

            // Snap to nearest feasible point to test for convergence.
            xmin = x;
            las::max(a, xmin);
            las::min(b, xmin);
            fmin = f(xmin);

            if (converged(fmin, fl, f0, eps)) {
                break;
            }

            fl = fmin;
        }

        return {xmin, fmin};
    }

private:
    using TDoubleVec = std::vector<double>;
    using TVectorBuf = boost::circular_buffer<VECTOR>;

private:
    template<typename F>
    void reinitialize(const F& f, const VECTOR& x0) {
        m_Initial = true;
        m_F0 = m_Fl = m_Fx = f(x0);
        m_X = x0;
        m_Dx.clear();
        m_Dg.clear();
        m_Dx.set_capacity(std::min(m_Rank, las::dimension(x0)));
        m_Dg.set_capacity(std::min(m_Rank, las::dimension(x0)));
        m_Rho.clear();
        m_Alpha.clear();
    }

    bool converged(double eps) const {
        return converged(m_Fx, m_Fl, m_F0, eps);
    }

    static bool converged(double fx, double fl, double f0, double eps) {
        // We use less than or equal so that this is true if fx == fl == f0.
        return std::fabs(fx - fl) <= eps * std::fabs(fx - f0);
    }

    template<typename F>
    VECTOR lineSearch(const F& f) {

        // This uses the Armijo condition that the decrease is bounded below by
        // some multiple of the decrease promised by the gradient for the step
        // size.

        double s{1.0};
        VECTOR xs{m_X - m_P};
        double fs{f(xs)};

        for (std::size_t i = 0; i < MAXIMUM_BACK_TRACKING_ITERATIONS &&
                                fs - m_Fx > this->minimumDecrease(s);
             ++i) {
            s *= m_StepScale;
            xs = m_X - s * m_P;
            fs = f(xs);
        }

        // This can happen if s |m_P| < epsilon |m_X|.
        if (las::distance(xs, m_X) == 0.0) {
            this->fullLineSearch(f, xs, fs);
        }

        m_Fl = m_Fx;
        m_Fx = fs;

        return xs;
    }

    template<typename F>
    void fullLineSearch(const F& f, VECTOR& xs, double& fs) {
        VECTOR step{las::norm(m_P) == 0.0 ? this->randomStep(f)
                                          : this->minimumStepSize() * las::norm(m_X) /
                                                las::norm(m_P) * m_P};

        if (las::norm(step) == 0.0) {
            xs = m_X;
            fs = m_Fx;
            return;
        }

        // We probe a selection of points along the step direction looking for
        // a minimum and polish up with a few iterations of Brent's method.

        TDoubleVec probes(MAXIMUM_BACK_TRACKING_ITERATIONS);
        probes[0] = 1.0;
        for (std::size_t i = 1; i < MAXIMUM_BACK_TRACKING_ITERATIONS; ++i) {
            probes[i] = probes[i - 1] / m_StepScale;
        }

        auto f_ = [&](double s_) { return f(m_X - s_ * step); };

        double smin;
        double fmin;
        CSolvers::globalMinimize(probes, f_, smin, fmin);
        xs = m_X - smin * step;
        fs = f(xs);
    }

    template<typename F>
    VECTOR randomStep(const F& f) {
        double eps{this->minimumStepSize() * las::norm(m_X)};
        for (std::size_t i = 0; eps > 0.0 && i < MAXIMUM_RANDOM_PROBES; ++i) {
            // Randomly probe looking for a decrease in the function.
            VECTOR step{las::zero(m_X)};
            for (std::size_t j = 0; j < las::dimension(m_X); ++j) {
                step(j) = CSampling::uniformSample(m_Rng, -eps, eps);
            }
            if (f(m_X + step) < f(m_X)) {
                return -step;
            }
        }
        // We couldn't find a direction in which the fuction decreases. Returning
        // zero means we'll exit on the convergence condition.
        return las::zero(m_X);
    }

    template<typename G>
    void updateDescentDirection(const G& g, const VECTOR& x) {

        // This uses the L-BFGS Hessian approximation scheme.

        m_P = g(x);

        if (m_Initial == false) {
            m_Dx.push_back(x - m_X);
            m_Dg.push_back(m_P - m_Gx);
        }

        m_X = x;
        m_Gx = m_P;

        if (m_Initial == false && las::norm(m_Gx) > 0.0) {
            double eps{std::numeric_limits<double>::epsilon() * las::norm(m_Gx)};

            std::size_t k{m_Dx.size()};
            m_Rho.resize(k);
            m_Alpha.resize(k);
            for (std::size_t i = k; i > 0; --i) {
                m_Rho[i - 1] = 1.0 / (las::inner(m_Dg[i - 1], m_Dx[i - 1]) + eps);
                m_Alpha[i - 1] = m_Rho[i - 1] * las::inner(m_Dx[i - 1], m_P);
                m_P -= m_Alpha[i - 1] * m_Dg[i - 1];
            }

            // The initialisation choice is free, this is an estimate for the
            // size of the true Hessian along the most recent search direction
            // and tends to mean that the initial step size is chosen in the
            // line search.

            double hmax{las::norm(m_Gx) / this->minimumStepSize()};

            double lambda{(las::inner(m_Dg[k - 1], m_Dx[k - 1]) + eps) /
                          (las::inner(m_Dg[k - 1], m_Dg[k - 1]) + eps * eps)};
            double h0{std::copysign(std::min(std::fabs(lambda), hmax), lambda)};

            m_P *= h0;

            for (std::size_t i = 0; i < k; ++i) {
                double beta{m_Rho[i] * (las::inner(m_Dg[i], m_P) + eps)};
                double gk{m_Alpha[i] - beta};
                double gmax{hmax / las::norm(m_Dx[i])};
                m_P += std::copysign(std::min(std::fabs(gk), gmax), gk) * m_Dx[i];
            }

            if (las::inner(m_Gx, m_P) <= 0.0) {
                m_P = (las::norm(m_P) / (las::norm(m_Gx) + eps)) * m_Gx;
            }
        } else {
            m_Initial = false;
        }
    }

    double minimumDecrease(double s) const {
        double normP{las::norm(m_P)};
        return m_BacktrackingMinDecrease * s *
               (normP == 0.0 ? las::norm(m_Gx) : las::inner(m_Gx, m_P) / normP);
    }

    double minimumStepSize() const {
        return CTools::stable(std::pow(
            m_StepScale, static_cast<double>(MAXIMUM_BACK_TRACKING_ITERATIONS)));
    }

private:
    std::size_t m_Rank;
    double m_StepScale;
    double m_BacktrackingMinDecrease;
    CPRNG::CXorOShiro128Plus m_Rng;
    bool m_Initial = true;
    double m_F0;
    double m_Fl;
    double m_Fx;
    VECTOR m_X;
    VECTOR m_Gx;
    VECTOR m_P;
    TVectorBuf m_Dx;
    TVectorBuf m_Dg;
    TDoubleVec m_Rho;
    TDoubleVec m_Alpha;
};

template<typename VECTOR>
const double CLbfgs<VECTOR>::BACKTRACKING_MIN_DECREASE{1e-4};
template<typename VECTOR>
const double CLbfgs<VECTOR>::STEP_SCALE{0.3};
template<typename VECTOR>
const std::size_t CLbfgs<VECTOR>::MAXIMUM_BACK_TRACKING_ITERATIONS{20};
template<typename VECTOR>
const std::size_t CLbfgs<VECTOR>::MAXIMUM_RANDOM_PROBES{10};
}
}
}

#endif // INCLUDED_ml_maths_common_CLbfgs_h
