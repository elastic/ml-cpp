/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLbfgs_h
#define INCLUDED_ml_maths_CLbfgs_h

#include <maths/CLinearAlgebraShims.h>

#include <boost/circular_buffer.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief The limited memory BFGS algorithm.
//!
//! DESCRIPTION:\n
//! This uses a low rank approximation to the Hessian to compute the search direction
//! and line search with back tracking.
//!
//! For more information \see https://en.wikipedia.org/wiki/Limited-memory_BFGS
template<typename VECTOR>
class CLbfgs {
public:
    //! The scale to apply to the expected decrease from the gradient to decrease
    //! in the test to continue backtracking.
    static const double BACKTRACKING_MIN_DECREASE;
    //! The scale which is applied to the step size in backtracking.
    static const double STEP_SCALE;
    //! The maximum number of iterations to use backtracking.
    static const std::size_t MAXIMUM_BACK_TRACKING_ITERATIONS;

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
    }

    bool converged(double eps) const {
        return std::fabs(m_Fx - m_Fl) < eps * std::fabs(m_Fx - m_F0);
    }

    template<typename F>
    VECTOR lineSearch(const F& f) {
        // This uses the Armijo condition that the decrease is bounded below by
        // some multiple of the decrease promised by the gradient for the step
        // size.

        double s{1.0};
        double fs{f(m_X - s * m_P)};
        double fb{fs};
        double sb{s};

        for (std::size_t i = 0; i < MAXIMUM_BACK_TRACKING_ITERATIONS &&
                                fs - m_Fx > this->minimumDecrease(s);
             ++i) {
            s *= m_StepScale;
            fs = f(m_X - s * m_P);
            fb = fs;
            sb = s;
        }

        m_Fl = m_Fx;
        m_Fx = fb;

        return m_X - s * m_P;
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

        if (m_Initial == false) {
            double eps{std::numeric_limits<double>::epsilon() * las::norm(m_Gx)};

            std::size_t k{m_Dx.size()};
            TDoubleVec rho(k);
            TDoubleVec alpha(k);
            for (std::size_t i = k; i > 0; --i) {
                rho[i - 1] = 1.0 / (las::inner(m_Dg[i - 1], m_Dx[i - 1]) + eps);
                alpha[i - 1] = rho[i - 1] * las::inner(m_Dx[i - 1], m_P);
                m_P.noalias() -= alpha[i - 1] * m_Dg[i - 1];
            }

            // The initialisation choice is free, this is an estimate for the
            // size of the true Hessian along the most recent search direction
            // and tends to mean that the initial step size is choosen in the
            // line search.

            double hmax{las::norm(m_Gx) / this->minimumStepSize()};

            double lambda{las::inner(m_Dg[k - 1], m_Dx[k - 1]) /
                          las::inner(m_Dg[k - 1], m_Dg[k - 1])};
            double h0{std::copysign(std::min(std::fabs(lambda), hmax), lambda)};

            m_P *= h0;

            for (std::size_t i = 0; i < k; ++i) {
                double beta{rho[i] * (las::inner(m_Dg[i], m_P) + eps)};
                double gk{alpha[i] - beta};
                double gmax{hmax / las::norm(m_Dx[i])};
                m_P.noalias() += std::copysign(std::min(std::fabs(gk), gmax), gk) *
                                 m_Dx[i];
            }
        } else {
            m_Initial = false;
        }
    }

    double minimumDecrease(double s) const {
        return -m_BacktrackingMinDecrease * s * las::inner(m_Gx, m_P);
    }

    constexpr double minimumStepSize() const {
        return std::pow(m_StepScale, static_cast<double>(MAXIMUM_BACK_TRACKING_ITERATIONS));
    }

private:
    std::size_t m_Rank;
    double m_StepScale;
    double m_BacktrackingMinDecrease;
    bool m_Initial = true;
    double m_F0;
    double m_Fl;
    double m_Fx;
    VECTOR m_X;
    VECTOR m_Gx;
    VECTOR m_P;
    TVectorBuf m_Dx;
    TVectorBuf m_Dg;
};

template<typename VECTOR>
const double CLbfgs<VECTOR>::BACKTRACKING_MIN_DECREASE{1e-4};
template<typename VECTOR>
const double CLbfgs<VECTOR>::STEP_SCALE{0.3};
template<typename VECTOR>
const std::size_t CLbfgs<VECTOR>::MAXIMUM_BACK_TRACKING_ITERATIONS{20};
}
}

#endif // INCLUDED_ml_maths_CLbfgs_h
