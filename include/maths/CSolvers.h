/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSolvers_h
#define INCLUDED_ml_maths_CSolvers_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCompositeFunctions.h>
#include <maths/CEqualWithTolerance.h>
#include <maths/CMathsFuncs.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>

#include <boost/math/tools/roots.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ml {
namespace maths {

//! \brief Collection of root solving functions.
//!
//! DESCRIPTION:\n
//! This implements a collection of root solving functions as static
//! member functions. In particular, it implements Brent's method and
//! interval bisection.
class MATHS_EXPORT CSolvers {
private:
    using TDoubleDoublePr = std::pair<double, double>;

    //! \name Helpers
    //@{
    //! An inverse quadratic interpolation of three distinct
    //! function values. WARNING the caller must ensure that
    //! the \p fa != \p fb != \p fc.
    static inline double inverseQuadraticInterpolate(const double a,
                                                     const double b,
                                                     const double c,
                                                     const double fa,
                                                     const double fb,
                                                     const double fc) {
        return a * fb * fc / (fa - fb) / (fa - fc) +
               b * fa * fc / (fb - fa) / (fb - fc) + c * fa * fb / (fc - fa) / (fc - fb);
    }

    //! A secant interpolation of two distinct function values.
    //! WARNING the caller must ensure that \p fa != \p fb.
    static inline double
    secantInterpolate(const double a, const double b, const double fa, const double fb) {
        return b - fb * (b - a) / (fb - fa);
    }

    //! Bisect the interval [\p a, \p b].
    static inline double bisect(const double a, const double b) {
        return (a + b) / 2.0;
    }

    //! Shift the values such that a = b and b = c.
    static inline void shift(double& a, double& b, const double c) {
        a = b;
        b = c;
    }

    //! Shift the values such that a = b, b = c and c = d.
    static inline void shift(double& a, double& b, double& c, const double d) {
        a = b;
        b = c;
        c = d;
    }
    //@}

    //! Function wrapper which checks for NaN argument.
    template<typename F>
    class CTrapNaNArgument {
    public:
        CTrapNaNArgument(const F& f) : m_F(f) {}

        inline double operator()(const double x) const {
            if (CMathsFuncs::isNan(x)) {
                throw std::invalid_argument("x is nan");
            }
            return m_F(x);
        }

    private:
        F m_F;
    };

private:
    //! Minimizes \p f in the interval [\p a, \p b] subject
    //! to the constraint that \p fx > \p lb.
    template<typename F>
    static void minimize(double a,
                         double b,
                         double fa,
                         double fb,
                         const F& f,
                         double tolerance,
                         std::size_t& maxIterations,
                         double lb,
                         double& x,
                         double& fx) {
        tolerance = std::max(tolerance, std::sqrt(std::numeric_limits<double>::epsilon()));
        const double golden = 0.3819660;

        if (fa < fb) {
            x = a;
            fx = fa;
        } else {
            x = b;
            fx = fb;
        }

        double w = x, v = x;
        double fw = fx, fv = fx;
        double s = 0.0, sLast = 0.0;

        std::size_t n = maxIterations;

        do {
            double xm = bisect(a, b);

            double t1 = tolerance * (std::fabs(x) + 0.25);
            double t2 = 2.0 * t1;
            if (fx <= lb || std::fabs(x - xm) <= (t2 - (b - a) / 2.0)) {
                break;
            }

            double sign = (x >= xm) ? -1.0 : 1.0;

            if (sLast > t1) {
                // Fit parabola to abscissa.
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2 * (q - r);
                if (q > 0) {
                    p = -p;
                }
                q = std::fabs(q);

                double td = sLast;
                sLast = std::fabs(s);

                if (std::fabs(p) >= q * td / 2.0 || p <= q * (a - x) || p >= q * (b - x)) {
                    // Minimum not in range or converging too slowly.
                    sLast = (x >= xm) ? x - a : b - x;
                    s = sign * std::max(golden * sLast, t1);
                } else {
                    s = p / q;
                    double u = x + s;
                    if ((u - a) < t2 || (b - u) < t2) {
                        s = sign * t1;
                    }
                }
            } else {
                // Don't have a suitable abscissa so just use golden
                // section in to the larger of [a, x] and [x, b].
                sLast = (x >= xm) ? x - a : b - x;
                s = sign * std::max(golden * sLast, t1);
            }

            double u = x + s;
            double fu = f(u);
            LOG_TRACE(<< "s = " << s << ", u = " << u);
            LOG_TRACE(<< "f(u) = " << fu << ", f(x) = " << fx);

            if (fu <= fx) {
                u >= x ? a = x : b = x;
                shift(v, w, x, u);
                shift(fv, fw, fx, fu);
            } else {
                u < x ? a = u : b = u;
                if (fu <= fw || w == x) {
                    shift(v, w, u);
                    shift(fv, fw, fu);
                } else if (fu <= fv || v == x || v == w) {
                    v = u;
                    fv = fu;
                }
            }
            LOG_TRACE(<< "a = " << a << ", b = " << b);
            LOG_TRACE(<< "x = " << x << ", v = " << v << ", w = " << w);
            LOG_TRACE(<< "f(x) = " << fx << ", f(v) = " << fv << ", f(w) = " << fw);
        } while (--n > 0);

        maxIterations -= n;
    }

    //! Attempt to bracket a root of \p f using [\p a, \p b]
    //! as a starting point.
    template<typename F>
    static bool bracket(double& a,
                        double& b,
                        double& fa,
                        double& fb,
                        const F& f,
                        const double direction,
                        std::size_t& maxIterations,
                        const double min,
                        const double max) {
        if (a > b) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        // Simple doubling search which switches to the secant
        // method if this strategy fails to produce a bracket
        // quickly.

        double step = b - a;
        if (step == 0.0) {
            step = 1.0;
        }

        std::size_t n = maxIterations;
        for (/**/; n > 0; --n) {
            if (fa * fb <= 0.0) {
                break;
            }

            step *= 2.0;
            if (n < (3 * maxIterations) / 4) {
                double minStep = step;
                double maxStep = step * step;
                step = fa == fb
                           ? maxStep
                           : std::min(std::max(std::fabs(b - a) / std::fabs(fb - fa) *
                                                   std::fabs(fb),
                                               minStep),
                                      maxStep);
            }
            a = b;
            fa = fb;
            b += direction * step;
            b = std::max(std::min(b, max), min);
            if (a == b) {
                // We've hit our domain constraints.
                break;
            }

            fb = f(b);
        }

        if (a > b) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        maxIterations = maxIterations - n;
        return fa * fb <= 0.0;
    }

public:
    //! Find a bracket for a root of \p f searching left.
    //! WARNING to be guaranteed to work the function must
    //! only have a single root which is guaranteed if it
    //! is monotonic, for example.
    //!
    //! \param[in,out] a The bracket starting left end point.
    //! Set to the bracketing left end point if a root could
    //! be bracketed.
    //! \param[in,out] b The bracket starting right end point.
    //! Set to the bracketing right end point if a root could
    //! be bracketed.
    //! \param[in,out] fa The value of f(a). Set to the value
    //! of f at the output value of a if a root could be
    //! bracketed.
    //! \param[in,out] fb The value of f(b). Set to the value
    //! of f at the output value of b if a root could be
    //! bracketed.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] min The minimum value in the function domain.
    //! \param[in] max The maximum value in the function domain.
    template<typename F>
    static inline bool leftBracket(double& a,
                                   double& b,
                                   double& fa,
                                   double& fb,
                                   const F& f,
                                   std::size_t& maxIterations,
                                   double min = -std::numeric_limits<double>::max(),
                                   double max = std::numeric_limits<double>::max()) {
        return bracket(a, b, fa, fb, f, -1.0, maxIterations, min, max);
    }

    //! Find a bracket for a root of \p f searching right.
    //! WARNING to be guaranteed to work the function must
    //! only have a single root which is guaranteed if it
    //! is monotonic, for example.
    //!
    //! \param[in,out] a The bracket starting left end point.
    //! Set to the bracketing left end point if a root could
    //! be bracketed.
    //! \param[in,out] b The bracket starting right end point.
    //! Set to the bracketing right end point if a root could
    //! be bracketed.
    //! \param[in,out] fa The value of f(a). Set to the value
    //! of f at the output value of a if a root could be
    //! bracketed.
    //! \param[in,out] fb The value of f(b). Set to the value
    //! of f at the output value of b if a root could be
    //! bracketed.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] min The minimum value in the function domain.
    //! \param[in] max The maximum value in the function domain.
    template<typename F>
    static inline bool rightBracket(double& a,
                                    double& b,
                                    double& fa,
                                    double& fb,
                                    const F& f,
                                    std::size_t& maxIterations,
                                    double min = -std::numeric_limits<double>::max(),
                                    double max = std::numeric_limits<double>::max()) {
        return bracket(a, b, fa, fb, f, +1.0, maxIterations, min, max);
    }

    //! \name Bracketed Solvers
    //@{
    //! The preferred solver implementation. This uses the
    //! TOMS 748 algorithm implemented by boost trapping
    //! the case that it produces a bad interpolant, in which
    //! case it falls back to Brent's method). WARNING this
    //! follows the boost::math::tools::toms748_solve policy
    //! and throws if a and b don't bracket the root.
    //!
    //! \param[in,out] a The left bracket. Set to the solution
    //! interval left end point if [a,b] brackets a root.
    //! \param[in,out] b The right bracket. Set to the solution
    //! interval right end point if [a,b] brackets a root.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] equal The predicate to decide when to terminate
    //! The test is applied to the interval end points a and b.
    //! \param[out] bestGuess Filled in with the best estimate
    //! of the root.
    template<typename F, typename EQUAL>
    static inline void
    solve(double& a, double& b, const F& f, std::size_t& maxIterations, const EQUAL& equal, double& bestGuess) {
        if (equal(a, b)) {
            bestGuess = bisect(a, b);
            maxIterations = 0u;
        } else if (maxIterations < 3) {
            bestGuess = bisect(a, b);
            maxIterations = 0u;
        } else {
            maxIterations -= 2;
            solve(a, b, f(a), f(b), f, maxIterations, equal, bestGuess);
            maxIterations += 2;
        }
    }

    //! The preferred solver implementation. This uses the
    //! TOMS 748 algorithm implemented by boost trapping
    //! the case that it produces a bad interpolant, in which
    //! case it falls back to Brent's method). WARNING this
    //! follows the boost::math::tools::toms748_solve policy
    //! and throws if a and b don't bracket the root.
    //!
    //! \param[in,out] a The left bracket. Set to the solution
    //! interval left end point if [a,b] brackets a root.
    //! \param[in,out] b The right bracket. Set to the solution
    //! interval right end point if [a,b] brackets a root.
    //! \param[in] fa The value of \p f at \p a.
    //! \param[in] fb The value of \p f at \p b.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] equal The predicate to decide when to terminate
    //! The test is applied to the interval end points a and b.
    //! \param[out] bestGuess Filled in with the best estimate
    //! of the root.
    template<typename F, typename EQUAL>
    static void solve(double& a,
                      double& b,
                      double fa,
                      double fb,
                      const F& f,
                      std::size_t& maxIterations,
                      const EQUAL& equal,
                      double& bestGuess) {
        if (equal(a, b)) {
            // There is a bug in boost's solver for the case that
            // a == b so trap and return early.
            maxIterations = 0;
            bestGuess = (a + b) / 2.0;
            return;
        }
        try {
            CTrapNaNArgument<F> fSafe(f);
            // Need at least one step or the boost solver underflows
            // size_t.
            boost::uintmax_t n = std::max(maxIterations, std::size_t(1));
            TDoubleDoublePr bracket =
                boost::math::tools::toms748_solve<const CTrapNaNArgument<F>&>(
                    fSafe, a, b, fa, fb, equal, n);
            a = bracket.first;
            b = bracket.second;
            bestGuess = bisect(a, b);
            maxIterations = static_cast<std::size_t>(n);
            return;
        } catch (const std::exception& e) {
            LOG_TRACE(<< "Falling back to Brent's solver: " << e.what());
            // Avoid compiler warning in the case of LOG_TRACE being compiled out
            static_cast<void>(&e);
        }
        if (!brent(a, b, fa, fb, f, maxIterations, equal, bestGuess)) {
            throw std::invalid_argument("doesn't bracket root");
        }
    }

    //! Implements Brent's method for root finding.
    //!
    //! \see <a href="http://en.wikipedia.org/wiki/Brent%27s_method">here</a>
    //! for details.
    //!
    //! \param[in,out] a The left bracket. Set to the solution
    //! interval left end point if [a,b] brackets a root.
    //! \param[in,out] b The right bracket. Set to the solution
    //! interval right end point if [a,b] brackets a root.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] equal The predicate to decide when to terminate
    //! The test is applied to the interval end points a and b.
    //! \param[out] bestGuess Filled in with the best estimate
    //! of the root.
    //! \return True if a, b bracket the root.
    template<typename F, typename EQUAL>
    static bool
    brent(double& a, double& b, const F& f, std::size_t& maxIterations, const EQUAL& equal, double& bestGuess) {
        if (equal(a, b)) {
            bestGuess = bisect(a, b);
            maxIterations = 0u;
            return true;
        }
        if (maxIterations < 3) {
            bestGuess = bisect(a, b);
            maxIterations = 0u;
            return true;
        }
        maxIterations -= 2;
        bool result = brent(a, b, f(a), f(b), f, maxIterations, equal, bestGuess);
        maxIterations += 2;
        return result;
    }

    //! Implements Brent's method for root finding.
    //!
    //! \see http://en.wikipedia.org/wiki/Brent%27s_method for details.
    //!
    //! \param[in,out] a The left bracket. Set to the solution
    //! interval left end point if [a,b] brackets a root.
    //! \param[in,out] b The right bracket. Set to the solution
    //! interval right end point if [a,b] brackets a root.
    //! \param[in] fa The value of \p f at \p a.
    //! \param[in] fb The value of \p f at \p b.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] equal The predicate to decide when to terminate
    //! The test is applied to the interval end points a and b.
    //! \param[out] bestGuess Filled in with the best estimate
    //! of the root.
    //! \return True if a, b bracket the root.
    template<typename F, typename EQUAL>
    static bool brent(double& a,
                      double& b,
                      double fa,
                      double fb,
                      const F& f,
                      std::size_t& maxIterations,
                      const EQUAL& equal,
                      double& bestGuess) {
        std::size_t n = maxIterations;

        if (fa == 0.0) {
            // Root at left bracket.
            bestGuess = b = a;
            maxIterations -= n;
            return true;
        }
        if (fb == 0.0) {
            // Root at right bracket.
            bestGuess = a = b;
            maxIterations -= n;
            return true;
        }
        if (fa * fb > 0.0) {
            // Not bracketed.
            maxIterations -= n;
            return false;
        }

        if (std::fabs(fa) < std::fabs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        bool bisected = true;
        double c = a;
        double fc = fa;
        double d = std::numeric_limits<double>::max();

        do {
            double s = (fa != fc) && (fb != fc)
                           ? inverseQuadraticInterpolate(a, b, c, fa, fb, fc)
                           : secantInterpolate(a, b, fa, fb);

            double e = (3.0 * a + b) / 4.0;

            if ((!(((s > e) && (s < b)) || ((s < e) && (s > b)))) ||
                (bisected && ((std::fabs(s - b) >= std::fabs(b - c) / 2.0) || equal(b, c))) ||
                (!bisected &&
                 ((std::fabs(s - b) >= std::fabs(c - d) / 2.0) || equal(c, d)))) {
                // Use bisection.
                s = bisect(a, b);
                bisected = true;
            } else {
                bisected = false;
            }

            double fs = f(s);
            shift(d, c, b);
            fc = fb;

            if (fs == 0.0) {
                // Root at s.
                bestGuess = a = b = s;
                maxIterations -= (n - 1);
                return true;
            }

            if (fa * fs > 0.0) {
                a = s;
                fa = fs;
            } else {
                b = s;
                fb = fs;
            }

            if (std::fabs(fa) < std::fabs(fb)) {
                std::swap(a, b);
                std::swap(fa, fb);
            }
        } while (--n > 0 && !equal(a, b));

        if (b < a) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
        bestGuess = (fa != fc) && (fb != fc)
                        ? inverseQuadraticInterpolate(a, b, c, fa, fb, fc)
                        : (fa != fb ? secantInterpolate(a, b, fa, fb) : bisect(a, b));
        bestGuess = std::min(std::max(a, bestGuess), b);
        maxIterations -= n;

        return true;
    }

    //! Bisection for which the function has not been evaluated
    //! at the interval end points. WARNING this is worse than
    //! Brent's method (although extremely numerically robust)
    //! and is primarily intended for testing.
    //!
    //! \param[in,out] a The left bracket. Set to the solution
    //! interval left end point if [a,b] brackets a root.
    //! \param[in,out] b The right bracket. Set to the solution
    //! interval right end point if [a,b] brackets a root.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] equal The predicate to decide when to terminate
    //! The test is applied to the interval end points a and b.
    //! \param[out] bestGuess Filled in with the best estimate
    //! of the root.
    //! \return True if a, b bracket the root and equal(a, b).
    template<typename F, typename EQUAL>
    static bool bisection(double& a,
                          double& b,
                          const F& f,
                          std::size_t& maxIterations,
                          const EQUAL& equal,
                          double& bestGuess) {
        if (equal(a, b)) {
            bestGuess = bisect(a, b);
            maxIterations = 0u;
            return true;
        }
        if (maxIterations < 3) {
            bestGuess = bisect(a, b);
            maxIterations = 0u;
            return true;
        }
        maxIterations -= 2;
        bool result = bisection(a, b, f(a), f(b), f, maxIterations, equal, bestGuess);
        maxIterations += 2;
        return result;
    }

    //! Bisection for which the function *has* been evaluated
    //! at the interval end points. This means we can save
    //! ourselves two function calls. WARNING this is worse than
    //! Brent's method (although extremely numerically robust)
    //! and is primarily intended for testing.
    //!
    //! \param[in,out] a The left bracket. Set to the solution
    //! interval left end point if [a,b] brackets a root.
    //! \param[in,out] b The right bracket. Set to the solution
    //! interval right end point if [a,b] brackets a root.
    //! \param[in] fa The value of \p f at \p a.
    //! \param[in] fb The value of \p f at \p b.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[in] equal The predicate to decide when to terminate
    //! The test is applied to the interval end points a and b.
    //! \param[out] bestGuess Filled in with the best estimate
    //! of the root.
    //! \return True if a, b bracket the root and equal(a, b).
    template<typename F, typename EQUAL>
    static bool bisection(double& a,
                          double& b,
                          double fa,
                          double fb,
                          const F& f,
                          std::size_t& maxIterations,
                          const EQUAL& equal,
                          double& bestGuess) {
        std::size_t n = maxIterations;
        if (fa == 0.0) {
            // Root at left bracket.
            bestGuess = b = a;
            maxIterations -= n;
            return true;
        }
        if (fb == 0.0) {
            // Root at right bracket.
            bestGuess = a = b;
            maxIterations -= n;
            return true;
        }
        if (fa * fb > 0.0) {
            // Not bracketed.
            maxIterations -= n;
            return false;
        }

        do {
            const double c = bisect(a, b);
            const double fc = f(c);
            if (fc == 0.0) {
                // Root at s.
                bestGuess = a = b = c;
                maxIterations -= (n - 1);
                return true;
            }

            if (fa * fc > 0.0) {
                a = c;
                fa = fc;
            } else {
                b = c;
                fb = fc;
            }
        } while (--n > 0 && !equal(a, b));

        bestGuess = fa != fb ? secantInterpolate(a, b, fa, fb) : bisect(a, b);
        bestGuess = std::min(std::max(bestGuess, a), b);
        maxIterations -= n;

        return true;
    }
    //@}

    //! Minimize the function \p f on the interval [\p a, \p b]
    //! This terminates if it has converged on a local minimum
    //! or it has run out of iterations. This implements Brent's
    //! method which combines golden section search and quadratic
    //! minimization (see for example numerical recipes in C).
    //!
    //! Note that this converges to the unique minimum if there
    //! is one. If the function \p f has several minima then you
    //! could consider using globalMaximize.
    //!
    //! \param[in] a The left end of the interval on which to
    //! minimize \p f.
    //! \param[in] b The right end of the interval on which to
    //! minimize \p f.
    //! \param[in] fa The value of \p f at \p a.
    //! \param[in] fb The value of \p f at \p b.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in] tolerance The convergence threshold ignored
    //! if too small so feel free to set to zero.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[out] x Set to argmin of f on [\p a, \p b].
    //! \param[out] fx Set to the value of f at \p x.
    template<typename F>
    static inline void minimize(double a,
                                double b,
                                double fa,
                                double fb,
                                const F& f,
                                double tolerance,
                                std::size_t& maxIterations,
                                double& x,
                                double& fx) {
        minimize(a, b, fa, fb, f, tolerance, maxIterations,
                 -std::numeric_limits<double>::max(), x, fx);
    }

    //! Maximize the function \p f on the interval [\p a, \p b]
    //! This terminates if it has converged on a local maximum
    //! or it has run out of iterations. This minimizes minus
    //! \p f using our standard minimization algorithm.
    //!
    //! Note that this converges to the unique maximum if there
    //! is one. If the function \p f has several maxima then you
    //! could consider using globalMaximize.
    //!
    //! \param[in] a The left end of the interval on which to
    //! maximize \p f.
    //! \param[in] b The right end of the interval on which to
    //! maximize \p f.
    //! \param[in] fa The value of \p f at \p a.
    //! \param[in] fb The value of \p f at \p b.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in] tolerance The convergence threshold ignored
    //! if too small so feel free to set to zero.
    //! \param[in,out] maxIterations The maximum number of times
    //! \p f is evaluated and set to the number of times it was
    //! evaluated.
    //! \param[out] x Set to argmax of f on [\p a, \p b].
    //! \param[out] fx Set to the value of f at \p x.
    template<typename F>
    static inline void maximize(double a,
                                double b,
                                double fa,
                                double fb,
                                const F& f,
                                double tolerance,
                                std::size_t& maxIterations,
                                double& x,
                                double& fx) {
        CCompositeFunctions::CMinus<F> f_(f);
        minimize(a, b, -fa, -fb, f_, tolerance, maxIterations, x, fx);
        fx = -fx;
    }

    //! Try and find a global minimum for the function evaluating
    //! it at the points \p p and then searching for a local
    //! minimum.
    //!
    //! \param[in] p The points at which to evaluate f looking
    //! for a global minimum.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[out] x Set to argmin of f on [\p a, \p b].
    //! \param[out] fx Set to the value of f at \p x.
    template<typename T, typename F>
    static bool globalMinimize(const T& p, const F& f, double& x, double& fx) {
        using TDoubleSizePr = std::pair<double, std::size_t>;
        using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleSizePr, 1>;

        std::size_t n = p.size();

        if (n == 0) {
            LOG_ERROR(<< "Must provide points at which to evaluate function");
            return false;
        }

        TMinAccumulator min;
        T fp(p.size());
        for (std::size_t i = 0u; i < p.size(); ++i) {
            double fi = f(p[i]);
            fp[i] = fi;
            min.add(TDoubleSizePr(fi, i));
        }
        LOG_TRACE(<< "p    = " << core::CContainerPrinter::print(p));
        LOG_TRACE(<< "f(p) = " << core::CContainerPrinter::print(fp));

        std::size_t i = min[0].second;
        std::size_t maxIterations = 5;
        if (i == 0) {
            minimize(p[0], p[1], fp[0], fp[1], f, 0.0, maxIterations, x, fx);
        } else if (i == n - 1) {
            minimize(p[n - 2], p[n - 1], fp[n - 2], fp[n - 1], f, 0.0,
                     maxIterations, x, fx);
        } else {
            std::size_t ai = i - 1;
            std::size_t bi = i + 1;
            minimize(p[ai], p[bi], fp[ai], fp[bi], f, 0.0, maxIterations, x, fx);
            if (fp[i] < fx) {
                x = p[i];
                fx = fp[i];
            }
        }
        LOG_TRACE(<< "x = " << x << " fx = " << fx);
        return true;
    }

    //! Try and find a global minimum for the function evaluating
    //! it at the points \p p and then searching for a local
    //! minimum.
    //!
    //! \param[in] p The points at which to evaluate f looking
    //! for a global minimum.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[out] x Set to argmin of f on [\p a, \p b].
    //! \param[out] fx Set to the value of f at \p x.
    template<typename T, typename F>
    static bool globalMaximize(const T& p, const F& f, double& x, double& fx) {
        CCompositeFunctions::CMinus<F> f_(f);
        bool result = globalMinimize(p, f_, x, fx);
        fx = -fx;
        return result;
    }

    //! Find the sublevel set of \p fc the function \p f on the
    //! interval [\p a, \p b]. The sublevel set is defined as:
    //! <pre class="fragment">
    //!   \f$L^-_{fc}(f) = {x : f(x) <= fc}\f$
    //! </pre>
    //!
    //! It is assumed that the sublevel set is a closed interval.
    //! WARNING this is equivalent to assuming that the function
    //! has a unique minimum in the interval [\p a, \p b] and
    //! the caller should ensure this condition is satisfied.
    //!
    //! \param[in] a The left end of the interval on which to
    //! compute the sublevel set of \p fmax.
    //! \param[in] b The right end of the interval on which to
    //! compute the sublevel set of \p fmax.
    //! \param[in] fa The value of \p f at \p a.
    //! \param[in] fb The value of \p f at \p b.
    //! \param[in] f The function to evaluate. This is expected
    //! to implement a function signature taking a double and
    //! returning a double.
    //! \param[in] fc The function value, f(c), for which to
    //! compute the sublevel set.
    //! \param[in] maxIterations The maximum number of times
    //! \p f is evaluated.
    //! \param[out] result Filled in with the sublevel set of
    //! \p fc if it isn't empty or the point which minimizes
    //! \p f otherwise.
    //! \return True if the sublevel set could be computed and
    //! false otherwise.
    //! \note This will evaluate \p f at most 3 * \p maxIterations.
    template<typename F>
    static bool sublevelSet(double a,
                            double b,
                            double fa,
                            double fb,
                            const F& f,
                            const double fc,
                            std::size_t maxIterations,
                            TDoubleDoublePr& result) {
        if (a > b) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        double x, fx;
        {
            std::size_t n = maxIterations;
            minimize(a, b, fa, fb, f, 0.0, n, fc, x, fx);
        }

        result = TDoubleDoublePr(x, x);
        if (fx > fc) {
            return false;
        }

        // [a, x] and [b, r] bracket the sublevel set end points.

        CCompositeFunctions::CMinusConstant<F> f_(f, fc);

        LOG_TRACE(<< "a = " << a << ", x = " << x << ", b = " << b);
        LOG_TRACE(<< "f_(a) = " << fa - fc << ", f_(x) = " << fx - fc
                  << ", f_(b) = " << fb - fc);

        const double eps = std::sqrt(std::numeric_limits<double>::epsilon()) * b;
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, eps);
        LOG_TRACE(<< "eps = " << eps);

        try {
            std::size_t n = maxIterations;
            solve(a, x, fa - fc, fx - fc, f_, n, equal, result.first);
            LOG_TRACE(<< "iterations = " << n);
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to find left end point: " << e.what());
            return false;
        }

        try {
            std::size_t n = maxIterations;
            solve(x, b, fx - fc, fb - fc, f_, n, equal, result.second);
            LOG_TRACE(<< "iterations = " << n);
        } catch (std::exception& e) {
            LOG_ERROR(<< "Failed to find right end point: " << e.what());
            return false;
        }

        return true;
    }
};
}
}

#endif // INCLUDED_ml_maths_CSolvers_h
