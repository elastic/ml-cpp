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

#ifndef INCLUDED_ml_maths_CIntegration_h
#define INCLUDED_ml_maths_CIntegration_h

#include <core/CContainerPrinter.h>
#include <core/CFastMutex.h>
#include <core/CLogger.h>
#include <core/CNonCopyable.h>
#include <core/CScopedFastLock.h>
#include <core/Constants.h>

#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/ImportExport.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <numeric>

#include <math.h>

namespace ml {
namespace maths {

//! \brief Place holder for numerical integration schemes.
//!
//! DESCRIPTION:\n
//! This class is intended to define various quadrature schemes as member functions.
//! Currently, only Gauss-Legendre quadrature is implemented (up to order order five).
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is intended to implement the bare quadratures (as routines which take the
//! function to integrate and the integration range). It is up to the user to choose
//! how to subdivide the true integration range to achieve the necessary precision.
//!
//! The quadratures should take a function (object) as a template parameter which
//! supplies a function like interface with the signature:
//!   bool (double, double &)
//!
//! where the second argument is filled in with the value of the function at the
//! first argument and a return value of false means that the function could not
//! be evaluated.
class MATHS_EXPORT CIntegration {
public:
    using TDoubleVec = std::vector<double>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TLogFunc = std::function<bool(double, double &)>;

public:
    //! Enumeration of order of quadrature.
    enum EOrder {
        OrderOne = 1,
        OrderTwo = 2,
        OrderThree = 3,
        OrderFour = 4,
        OrderFive = 5,
        OrderSix = 6,
        OrderSeven = 7,
        OrderEight = 8,
        OrderNine = 9,
        OrderTen = 10
    };

    //! Enumerations of the dimension of multivariate integrals we support
    //! using sparse quadrature.
    //!
    //! \note The limit is imposed by curse of dimensionality. If we need
    //! more dimensions we'll have to implement a quasi Monte-Carlo method.
    enum EDimension {
        OneDimension = 1,
        TwoDimensions = 2,
        ThreeDimensions = 3,
        FourDimensions = 4,
        FiveDimensions = 5,
        SixDimensions = 6,
        SevenDimensions = 7,
        EightDimensions = 8,
        NineDimensions = 9,
        TenDimensions = 10
    };

public:
    //! Gauss-Legendre quadrature.
    //!
    //! This implements Gauss-Legendre numerical integration of the function
    //! f(x) on the interval [a, b].
    //!
    //! \param[in] function The function to integrate.
    //! \param[in] a The start of the integration interval.
    //! \param[in] b The end of the integration interval.
    //! \param[out] result Filled with the integral of \p function over [\p a, \p b].
    //!
    //! \tparam ORDER The order of quadrature to use.
    //! \tparam F It is assumed that this has the signature:
    //!   bool function(double x, T &f)
    //! where f is filled in with the value of the function at x and returning
    //! false means that the function could not be evaluated at x.
    //! \tparam T The type of range of \p f. This must have a meaningful default
    //! constructor, support multiplication by a double and addition.
    template <EOrder ORDER, typename F, typename T>
    static bool gaussLegendre(const F &function, double a, double b, T &result) {
        result = T();

        const double *weights = CGaussLegendreQuadrature::weights(ORDER);
        const double *abscissas = CGaussLegendreQuadrature::abscissas(ORDER);

        // Evaluate f(x) at the abscissas and compute the weighted sum
        // of the quadrature.
        double centre = (a + b) / 2.0;
        double range = (b - a) / 2.0;
        for (unsigned int i = 0; i < ORDER; ++i) {
            T fx;
            if (!function(centre + range * abscissas[i], fx)) {
                return false;
            }
            fx *= weights[i];
            result += fx;
        }
        result *= range;

        return true;
    }

    //! Gauss-Legendre quadrature of the product of two functions.
    //!
    //! This implements Gauss-Legendre numerical integration of the function
    //! f(x), g(x) and f(x) * g(x) on the interval [a, b] which can be computed
    //! simultaneously with only a total of ORDER function evaluations of both.
    //!
    //! \param[in] f A function to integrate.
    //! \param[in] g Another function to integrate.
    //! \param[in] a The start of the integration interval.
    //! \param[in] b The end of the integration interval.
    //! \param[out] productIntegral Filled with the integral of \p f * \p g
    //! over [\p a, \p b].
    //! \param[out] fIntegral Filled with the integral of \p f over [\p a, \p b].
    //! \param[out] gIntegral Filled with the integral of \p g over [\p a, \p b].
    //!
    //! \tparam ORDER The order of quadrature to use.
    //! \tparam F It is assumed that this has the signature:
    //!   bool function(double x, T &f)
    //! where f is filled in with the value of the function at x and returning
    //! false means that the function could not be evaluated at x.
    //! \tparam G It is assumed that this has the signature:
    //!   bool function(double x, T &f)
    //! where f is filled in with the value of the function at x and returning
    //! false means that the function could not be evaluated at x.
    //! \tparam U The type of range of \p f. This must have a meaningful default
    //! constructor, support multiplication by a double and addition.
    //! \tparam V The type of range of \p g. This must have a meaningful default
    //! constructor, support multiplication by a double and addition.
    template <EOrder ORDER, typename F, typename G, typename U, typename V>
    static bool productGaussLegendre(const F &f,
                                     const G &g,
                                     double a,
                                     double b,
                                     U &productIntegral,
                                     U &fIntegral,
                                     V &gIntegral) {
        productIntegral = U();
        fIntegral = U();
        gIntegral = V();

        const double *weights = CGaussLegendreQuadrature::weights(ORDER);
        const double *abscissas = CGaussLegendreQuadrature::abscissas(ORDER);

        // Evaluate f(x) at the abscissas and compute the weighted sum
        // of the quadrature.
        double centre = (a + b) / 2.0;
        double range = (b - a) / 2.0;
        for (unsigned int i = 0; i < ORDER; ++i) {
            U fx;
            V gx;
            if (!f(centre + range * abscissas[i], fx) || !g(centre + range * abscissas[i], gx)) {
                return false;
            }
            double weight = weights[i];
            productIntegral += fx * gx * weight;
            fIntegral += fx * weight;
            gIntegral += gx * weight;
        }

        productIntegral *= range;
        fIntegral *= range;
        gIntegral *= range;

        return true;
    }

    //! Gauss-Legendre quadrature using logarithms.
    //!
    //! This implements Gauss-Legendre numerical integration of the function
    //! f(x) on the interval [a, b] where log(f(x)) is computed and the log
    //! of the integral is returned. This is intended to support integration
    //! where the integrand is very small and may underflow double. This is
    //! handled by renormalizing the integral so only underflows if values
    //! are less than ::exp(-std::numeric_limits<double>::max()), so really
    //! very small!
    //!
    //! \param[in] function The log of the function to integrate.
    //! \param[in] a The start of the integration interval.
    //! \param[in] b The end of the integration interval.
    //! \param[out] result Filled with the log of the integral of \p function
    //! over [\p a, \p b].
    //!
    //! \tparam ORDER The order of quadrature to use.
    //! \tparam F It is assumed that this has the signature:
    //!   bool function(double x, double &f)
    //! where f is filled in with the value of the function at x and returning
    //! false means that the function could not be evaluated at x.
    template <EOrder ORDER, typename F>
    static bool logGaussLegendre(const F &function, double a, double b, double &result) {
        result = 0.0;

        if (b <= a) {
            std::swap(a, b);
        }

        const double *weights = CGaussLegendreQuadrature::weights(ORDER);
        const double *abscissas = CGaussLegendreQuadrature::abscissas(ORDER);

        double fx[ORDER] = {0.0};

        // Evaluate f(x) at the abscissas.
        double centre = (a + b) / 2.0;
        double range = (b - a) / 2.0;
        for (unsigned int i = 0; i < ORDER; ++i) {
            if (!function(centre + range * abscissas[i], fx[i])) {
                return false;
            }
        }

        // Re-normalize and then take exponentials to avoid underflow.
        double fmax = *std::max_element(fx, fx + ORDER);
        for (unsigned int i = 0; i < ORDER; ++i) {
            fx[i] = ::exp(fx[i] - fmax);
        }

        // Quadrature.
        for (unsigned int i = 0; i < ORDER; ++i) {
            result += weights[i] * fx[i];
        }
        result *= range;
        result = result <= 0.0 ? core::constants::LOG_MIN_DOUBLE : fmax + ::log(result);

        return true;
    }

    //! An adaptive Gauss-Legendre scheme for univariate integration.
    //!
    //! This evaluates the integral of \p f over successive refinements of
    //! the initial intervals \p intervals. An interval is only refined if
    //! it is determined that doing so will affect the result by the relative
    //! \p tolerance. Note that the worst case complexity is
    //! O(\p splitsPerRefinement ^ refinements) although typical complexity
    //! are much smaller, because many intervals are quickly pruned.
    //!
    //! \param[in] f The function to integrate.
    //! \param[in,out] intervals The seed intervals over which to evaluate
    //! the function integral.
    //! \param[in,out] fIntervals The integral of \p f over each of the seed
    //! intervals \p intervals.
    //! \param[in] refinements The maximum number of times this will split
    //! the seed intervals.
    //! \param[in] splitsPerRefinement The number of times an interval is
    //! is split  when it is refined.
    //! \param[in] tolerance The relative (to \p result) tolerance in the
    //! error in \p result.
    //! \param[in,out] result The integral of \p f over \p intervals is added
    //! to this value.
    //! \note \p intervals and \p fIntervals are modified in order to avoid
    //! the copy if it isn't needed. If it is make copies yourself and pass
    //! these in.
    template <EOrder ORDER, typename F>
    static bool adaptiveGaussLegendre(const F &f,
                                      TDoubleDoublePrVec &intervals,
                                      TDoubleVec &fIntervals,
                                      std::size_t refinements,
                                      std::size_t splitsPerRefinement,
                                      double tolerance,
                                      double &result) {
        if (intervals.size() != fIntervals.size()) {
            LOG_ERROR("Inconsistent intervals and function integrals: "
                      << core::CContainerPrinter::print(intervals) << " "
                      << core::CContainerPrinter::print(fIntervals));
            return false;
        }

        result += std::accumulate(fIntervals.begin(), fIntervals.end(), 0.0);
        LOG_TRACE("initial = " << result);

        TDoubleVec corrections;
        corrections.reserve(fIntervals.size());
        for (std::size_t i = 0u; i < fIntervals.size(); ++i) {
            corrections.push_back(::fabs(fIntervals[i]));
        }

        for (std::size_t i = 0u; !intervals.empty() && i < refinements; ++i) {
            std::size_t n = intervals.size();
            double cutoff = tolerance * ::fabs(result) / static_cast<double>(n);

            std::size_t end = 0u;
            for (std::size_t j = 0u; j < corrections.size(); ++j) {
                if (corrections[j] > cutoff) {
                    std::swap(intervals[end], intervals[j]);
                    std::swap(fIntervals[end], fIntervals[j]);
                    std::swap(corrections[end], corrections[j]);
                    ++end;
                }
            }
            if (end != corrections.size()) {
                intervals.erase(intervals.begin() + end, intervals.end());
                fIntervals.erase(fIntervals.begin() + end, fIntervals.end());
                corrections.erase(corrections.begin() + end, corrections.end());
            }
            n = intervals.size();

            if (i + 1 < refinements) {
                intervals.reserve(splitsPerRefinement * n);
                fIntervals.reserve(splitsPerRefinement * n);
                corrections.reserve(splitsPerRefinement * n);
            }

            for (std::size_t j = 0u; j < n; ++j) {
                if (corrections[j] <= cutoff) {
                    corrections[j] = 0.0;
                    continue;
                }

                double fjOld = fIntervals[j];
                double fjNew = 0.0;

                double aj = intervals[j].first;
                double dj = (intervals[j].second - intervals[j].first) /
                            static_cast<double>(splitsPerRefinement);
                for (std::size_t k = 0u; k < splitsPerRefinement; ++k, aj += dj) {
                    double df;
                    if (CIntegration::gaussLegendre<ORDER>(f, aj, aj + dj, df)) {
                        fjNew += df;
                        if (i + 1 < refinements) {
                            if (k == 0) {
                                intervals[j] = TDoubleDoublePr(aj, aj + dj);
                                fIntervals[j] = df;
                            } else {
                                intervals.push_back(TDoubleDoublePr(aj, aj + dj));
                                fIntervals.push_back(df);
                            }
                        }
                    } else {
                        LOG_ERROR("Couldn't integrate f over [" << aj << "," << aj + dj << "]");
                        return false;
                    }
                }

                LOG_TRACE("fjNew = " << fjNew << ", fjOld = " << fjOld);
                double correction = fjNew - fjOld;
                if (i + 1 < refinements) {
                    corrections[j] = ::fabs(correction);
                    corrections.resize(corrections.size() + splitsPerRefinement - 1,
                                       ::fabs(correction));
                }

                result += correction;
                cutoff = tolerance * ::fabs(result) / static_cast<double>(n);
            }
        }

        return true;
    }

    //! \brief Implements Smolyak's construction to create the sparse
    //! grid points and weights from Gauss-Legendre quadrature.
    //!
    //! DESCRIPTION:\n
    //! For more details on this process see:
    //! http://www.cims.nyu.edu/~kellen/research/sparsegrids/%20Numerical%20integration%20using%20sparse%20grids.pdf
    //!
    //! \note This approach still suffers the curse of dimensionality,
    //! albeit with significantly reduced growth rate \f$O(2^l l^d)\f$ for
    //! order \f$2^l\f$ and dimension \f$d\f$ rather than \f$O(2^{ld}\f$
    //! for the standard tensor product approach. So, this only supports
    //! up to ten dimensions.
    //!
    //! IMPLEMENTATION DECISIONS:\n
    //! In general, we won't need every combination of order and dimension.
    //! Therefore, although the points and weights can be precomputed and
    //! could in principle be loaded at start time, doing this preemptively
    //! would bloat the process footprint unnecessarily. This class therefore
    //! implements the singleton pattern, one for each template parameter
    //! combination, and computes the points and weights in the constructor.
    //! This means any calling code only ever pays the runtime cost for
    //! computing them once and means that the amortized cost of integration
    //! is as low as possible.
    template <EOrder O, EDimension D>
    class CSparseGaussLegendreQuadrature : private core::CNonCopyable {
    private:
        static const unsigned int ORDER = static_cast<unsigned int>(O);
        static const unsigned int DIMENSION = static_cast<unsigned int>(D);

    public:
        using TVector = CVectorNx1<double, DIMENSION>;
        using TVectorVec = std::vector<TVector>;

    public:
        static const CSparseGaussLegendreQuadrature &instance(void) {
            const CSparseGaussLegendreQuadrature *tmp = ms_Instance.load(std::memory_order_acquire);
            if (!tmp) {
                core::CScopedFastLock scopedLock(CIntegration::ms_Mutex);
                tmp = ms_Instance.load(std::memory_order_relaxed);
                if (!tmp) {
                    tmp = new CSparseGaussLegendreQuadrature();
                    ms_Instance.store(tmp, std::memory_order_release);
                }
            }
            return *tmp;
        }

        //! The sparse grid point weights.
        const TDoubleVec &weights(void) const { return m_Weights; }

        //! The sparse grid point points.
        const TVectorVec &points(void) const { return m_Points; }

    private:
        using TUIntVec = std::vector<unsigned int>;

    private:
        //! Iterates through the combinations such that \f$\|I\|_1 = l\f$
        //! for the indices \f$I\f$ and some fixed monomial order \f$l\f$.
        static bool next(std::size_t d, TUIntVec &indices, TUIntVec &stop) {
            for (;;) {
                ++indices[d];
                if (indices[d] > stop[d]) {
                    if (d == DIMENSION - 1) {
                        break;
                    }
                    indices[d] = 1;
                    ++d;
                } else {
                    for (std::size_t j = 0; j < d; ++j) {
                        stop[j] = stop[d] - indices[d] + 1;
                    }
                    indices[0] = stop[0];
                    d = 0u;
                    return true;
                }
            }
            return false;
        }

        CSparseGaussLegendreQuadrature(void) {
            // Generate the weights. We don't exploit the weight and
            // abscissa symmetries to reduce the static storage since
            // this reduces the speed of integration and since we limit
            // the dimension the maximum number of points will be 8761.
            //
            // Note this uses the construction:
            //   Q^d_l = \sum{l <= ||k||_1 <= l+d-1}{ (-1)^(l+d-||k||_1-1) (d-1 ||k||_1-l) (Q^1_k_1
            //   x ... x Q^1_k_d) }

            using TVectorDoubleMap = std::map<TVector, double>;

            TVectorDoubleMap ordered;

            for (unsigned int l = ORDER > DIMENSION ? ORDER - DIMENSION : 0; l < ORDER; ++l) {
                LOG_TRACE("order = " << l);
                std::size_t d = 0u;
                TUIntVec indices(DIMENSION, 1);
                indices[0] = l + 1;
                TUIntVec stop(DIMENSION, l + 1);

                double sign = (ORDER - l - 1) % 2 == 1 ? -1.0 : 1.0;
                double scale = sign * CIntegerTools::binomial(DIMENSION - 1, DIMENSION + l - ORDER);
                LOG_TRACE("scale = " << scale);

                do {
                    LOG_TRACE("indices = " << core::CContainerPrinter::print(indices));

                    unsigned int n = 1u;
                    for (std::size_t i = 0u; i < indices.size(); ++i) {
                        n *= indices[i];
                    }
                    LOG_TRACE("Number of points = " << n);

                    TDoubleVec weights(n, 1.0);
                    TVectorVec points(n, TVector(0.0));
                    for (unsigned int i = 0u; i < n; ++i) {
                        for (unsigned int i_ = i, j = 0u; j < indices.size();
                             i_ /= indices[j], ++j) {
                            EOrder order = static_cast<EOrder>(indices[j]);
                            const double *w = CGaussLegendreQuadrature::weights(order);
                            const double *a = CGaussLegendreQuadrature::abscissas(order);
                            std::size_t k = i_ % indices[j];
                            weights[i] *= w[k];
                            points[i](j) = a[k];
                        }
                    }
                    LOG_TRACE("weights = " << core::CContainerPrinter::print(weights));
                    LOG_TRACE("points =  " << core::CContainerPrinter::print(points));
                    for (std::size_t i = 0u; i < n; ++i) {
                        ordered[points[i]] += scale * weights[i];
                    }
                } while (next(d, indices, stop));
            }

            m_Weights.reserve(ordered.size());
            m_Points.reserve(ordered.size());
            for (const auto &i : ordered) {
                m_Weights.push_back(i.second);
                m_Points.push_back(i.first);
            }
        }

        static std::atomic<const CSparseGaussLegendreQuadrature *> ms_Instance;

        TDoubleVec m_Weights;
        TVectorVec m_Points;
    };

    //! Sparse grid Gauss-Legendre quadrature.
    //!
    //! This implements Smolyak's sparse grid construction using Gauss-
    //! Legendre quadrature to numerically integrate a function f(x) on
    //! the box [a_1, b_1] x ... x [a_n, b_n].
    //!
    //! \param[in] function The function to integrate.
    //! \param[in] a The lower integration limits.
    //! \param[in] b The upper integration limits.
    //! \param[out] result Filled with the integral of \p function over [\p a, \p b].
    //! \note There must be exactly one upper and lower integration limit for
    //! each dimension.
    //! \note This only supports up to ten dimensions.
    //!
    //! \tparam ORDER The order of quadrature to use.
    //! \tparam DIMENSION The number of dimensions.
    //! \tparam F It is assumed that this has the signature:
    //!   bool function(const CVectorNx1<double, DIMENSION> &x, T &f)
    //! where f is filled in with the value of the function at x and returning
    //! false means that the function could not be evaluated at x.
    //! \tparam T The type of range of \p f. This must have a meaningful
    //! default constructor, support multiplication by a double and addition.
    template <EOrder ORDER, EDimension DIMENSION, typename F, typename T>
    static bool
    sparseGaussLegendre(const F &function, const TDoubleVec &a, const TDoubleVec &b, T &result) {
        using TSparseQuadrature = CSparseGaussLegendreQuadrature<ORDER, DIMENSION>;
        using TVector = typename TSparseQuadrature::TVector;
        using TVectorVec = typename TSparseQuadrature::TVectorVec;

        result = T();

        if (a.size() != static_cast<std::size_t>(DIMENSION)) {
            LOG_ERROR("Bad lower limits: " << core::CContainerPrinter::print(a));
            return false;
        }
        if (b.size() != static_cast<std::size_t>(DIMENSION)) {
            LOG_ERROR("Bad upper limits: " << core::CContainerPrinter::print(b));
            return false;
        }

        const TDoubleVec &weights = TSparseQuadrature::instance().weights();
        const TVectorVec &points = TSparseQuadrature::instance().points();

        // Evaluate f(x) at the abscissas and compute the weighted sum
        // of the quadrature.

        TVector a_(a.begin(), a.end());
        TVector b_(b.begin(), b.end());
        TVector centre = (a_ + b_) / 2.0;
        TVector range = (b_ - a_) / 2.0;

        for (std::size_t i = 0; i < weights.size(); ++i) {
            T fx;
            if (!function(centre + range * points[i], fx)) {
                return false;
            }
            fx *= weights[i];
            result += fx;
        }

        for (std::size_t i = 0u; i < DIMENSION; ++i) {
            result *= range(i);
        }

        return true;
    }

private:
    //! \brief Definitions of the weights and abscissas for different orders
    //! of Gauss-Legendre quadrature.
    class MATHS_EXPORT CGaussLegendreQuadrature {
    public:
        static const double *weights(EOrder order);
        static const double *abscissas(EOrder order);

    private:
        //! Order one.
        static const double WEIGHTS1[1];
        static const double ABSCISSAS1[1];

        //! Order two.
        static const double WEIGHTS2[2];
        static const double ABSCISSAS2[2];

        //! Order three.
        static const double WEIGHTS3[3];
        static const double ABSCISSAS3[3];

        //! Order four.
        static const double WEIGHTS4[4];
        static const double ABSCISSAS4[4];

        //! Order five.
        static const double WEIGHTS5[5];
        static const double ABSCISSAS5[5];

        //! Order six.
        static const double WEIGHTS6[6];
        static const double ABSCISSAS6[6];

        //! Order seven.
        static const double WEIGHTS7[7];
        static const double ABSCISSAS7[7];

        //! Order eight.
        static const double WEIGHTS8[8];
        static const double ABSCISSAS8[8];

        //! Order nine.
        static const double WEIGHTS9[9];
        static const double ABSCISSAS9[9];

        //! Order ten.
        static const double WEIGHTS10[10];
        static const double ABSCISSAS10[10];
    };

private:
    //! This is used to protect initialisation of the
    //! CSparseGaussLegendreQuadrature singleton.  There's a chicken-and-egg
    //! situation here with static initialisation, which means that the
    //! CSparseGaussLegendreQuadrature cannot be used before main() has run.
    static core::CFastMutex ms_Mutex;
};

template <CIntegration::EOrder O, CIntegration::EDimension D>
std::atomic<const CIntegration::CSparseGaussLegendreQuadrature<O, D> *>
    CIntegration::CSparseGaussLegendreQuadrature<O, D>::ms_Instance;
}
}

#endif// INCLUDED_ml_maths_CIntegration_h
