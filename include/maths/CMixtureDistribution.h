/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMixtureDistribution_h
#define INCLUDED_ml_maths_CMixtureDistribution_h

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CMathsFuncs.h>
#include <maths/CSolvers.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/fwd.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/variant.hpp>

#include <exception>
#include <vector>

#include <math.h>

namespace ml
{
namespace maths
{

namespace mixture_detail
{

typedef std::pair<double, double> TDoubleDoublePr;

//! \brief Implements the "polymorphic" mixture mode.
class MATHS_EXPORT CMixtureModeImpl
{
    public:
        CMixtureModeImpl(const boost::math::normal_distribution<> &normal);
        CMixtureModeImpl(const boost::math::gamma_distribution<> &gamma);
        CMixtureModeImpl(const boost::math::lognormal_distribution<> &lognormal);

        template<typename F>
        typename F::result_type visit(const F &f, double x) const
        {
            return boost::apply_visitor(boost::bind(f, _1, x), m_Distribution);
        }

        template<typename F>
        typename F::result_type visit(const F &f) const
        {
            return boost::apply_visitor(f, m_Distribution);
        }

    private:
        typedef boost::variant<boost::math::normal_distribution<>,
                               boost::math::gamma_distribution<>,
                               boost::math::lognormal_distribution<> > TDistribution;

    private:
        //! The actual distribution.
        TDistribution m_Distribution;
};

}

template<bool COMPLEMENT>
class CMixtureMode;

//! \brief A wrapper around one of the standard mode distributions.
//!
//! DESCRIPTION:\n
//! This holds a variant which contains the actual boost::math distribution
//! which can be one of normal, gamma or log-normal. It provides a callback
//! interface to invoke the various functions such as mode, pdf, cdf, etc.
//! It is used to support mixtures with different distributions describing
//! each mode.
//!
//! IMPLEMENTATION:\n
//! This uses a variant because we know the distributions we can use to model
//! a mode up front and it avoids heap allocation. The complement concept is
//! encoded in a type parameter to avoid condition checking.
template<>
class MATHS_EXPORT CMixtureMode<false> : public mixture_detail::CMixtureModeImpl
{
    public:
        CMixtureMode(const boost::math::normal_distribution<> &normal);
        CMixtureMode(const boost::math::gamma_distribution<> &gamma);
        CMixtureMode(const boost::math::lognormal_distribution<> &lognormal);
};

//! \brief A wrapper around the complement of one of the standard mode
//! distributions.
template<>
class MATHS_EXPORT CMixtureMode<true> : public mixture_detail::CMixtureModeImpl
{
    public:
        CMixtureMode(const CMixtureMode<false> &other);
};

//! Compute the distribution support.
MATHS_EXPORT
mixture_detail::TDoubleDoublePr support(const CMixtureMode<false> &mode);

//! Compute the distribution mode.
MATHS_EXPORT
double mode(const CMixtureMode<false> &mode);

//! Compute the distribution mean.
MATHS_EXPORT
double mean(const CMixtureMode<false> &mode);

//! Compute the distribution probability density at \p x.
MATHS_EXPORT
double pdf(const CMixtureMode<false> &mode, double x);

//! Compute the distribution cumulative density at \p x.
MATHS_EXPORT
double cdf(const CMixtureMode<false> &mode, double x);

//! Compute one minus the distribution cumulative density at \p x.
MATHS_EXPORT
double cdf(const CMixtureMode<true> &mode, double x);

//! Compute the distribution quantile at \p x.
//!
//! \note x must be in the range (0, 1).
MATHS_EXPORT
double quantile(const CMixtureMode<false> &mode, double x);

//! Get the complement distribution of \p mode.
MATHS_EXPORT
CMixtureMode<true> complement(const CMixtureMode<false> &mode);

//! \brief A mixture distribution.
//!
//! DESCRIPTION:\n
//! This is a wrapper round a vector of weights and a vector of
//! distributions conforming to boost::math::distribution objects
//! which describes a mixture distribution.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This follows the pattern used by boost::math::distributions
//! which defines lightweight objects to represent distributions
//! and free functions for computing various properties of the
//! distribution. In order to get this to support mixtures of
//! different distributions use the CMixtureMode object.
template<typename T>
class CMixtureDistribution
{
    public:
        typedef std::vector<double> TDoubleVec;
        typedef std::vector<T> TModeVec;

    public:
        CMixtureDistribution(void) {}

        //! \note The length of \p weights should match \p modes.
        CMixtureDistribution(const TDoubleVec &weights, const TModeVec &modes) :
                m_Weights(weights),
                m_Modes(modes)
        {
            std::size_t w = m_Weights.size();
            if (w != m_Modes.size())
            {
                LOG_ERROR("# weights = " << w << ", # modes = " << m_Modes.size());
                m_Weights.resize(m_Modes.size(), 0.0);
            }

            // Normalize the weights.
            double weightSum = 0.0;
            for (std::size_t i = 0u; i < w; ++i)
            {
                weightSum += m_Weights[i];
            }
            if (weightSum == 0.0)
            {
                LOG_ERROR("Expected non-zero weight sum");
            }
            for (std::size_t i = 0u; i < w; ++i)
            {
                m_Weights[i] = weightSum == 0.0 ?
                               1.0 / static_cast<double>(w) : m_Weights[i] / weightSum;
            }
        }

        void swap(CMixtureDistribution &other)
        {
            m_Weights.swap(other.m_Weights);
            m_Modes.swap(other.m_Modes);
        }

        inline const TDoubleVec &weights(void) const
        {
            return m_Weights;
        }
        inline TDoubleVec &weights(void)
        {
            return m_Weights;
        }

        inline const TModeVec &modes(void) const
        {
            return m_Modes;
        }
        inline TModeVec &modes(void)
        {
            return m_Modes;
        }

        std::string print(void) const
        {
            std::string result;
            for (std::size_t i = 0u; i < m_Weights.size(); ++i)
            {
                result +=  ' ' + core::CStringUtils::typeToStringPretty(m_Weights[i])
                         + '/' + core::CStringUtils::typeToStringPretty(mean(m_Modes[i]))
                         + '/' + core::CStringUtils::typeToStringPretty(standard_deviation(m_Modes[i]));

            }
            result += (m_Weights.empty() ? "" : " ");
            return result;
        }


    private:
        TDoubleVec m_Weights;
        TModeVec m_Modes;
};


namespace mixture_detail
{

//! Adapts the free p.d.f. function for use with the solver.
template<typename T>
class CPdfAdpater
{
    public:
        typedef double result_type;

    public:
        CPdfAdpater(const CMixtureDistribution<T> &distribution) :
            m_Distribution(&distribution)
        {
        }

        double operator()(double x) const
        {
            return pdf(*m_Distribution, x);
        }

    private:
        const CMixtureDistribution<T> *m_Distribution;
};

}

//! Get the support for \p distribution.
template<typename T>
mixture_detail::TDoubleDoublePr support(const CMixtureDistribution<T> &distribution)
{
    typedef typename CMixtureDistribution<T>::TModeVec TModeVec;

    const TModeVec &modes = distribution.modes();

    if (modes.empty())
    {
        return mixture_detail::TDoubleDoublePr(boost::numeric::bounds<double>::lowest(),
                                               boost::numeric::bounds<double>::highest());
    }

    mixture_detail::TDoubleDoublePr result(boost::numeric::bounds<double>::highest(),
                                           boost::numeric::bounds<double>::lowest());

    for (std::size_t i = 0u; i < modes.size(); ++i)
    {
        try
        {
            mixture_detail::TDoubleDoublePr modeSupport = support(modes[i]);
            result.first = std::min(result.first, modeSupport.first);
            result.second = std::max(result.second, modeSupport.second);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed to compute support for mode: " << e.what());
        }
    }

    return result;
}


//! Compute the mode for \p distribution.
//!
//! \warning This propagates boost exceptions.
template<typename T>
double mode(const CMixtureDistribution<T> &distribution)
{
    typedef typename CMixtureDistribution<T>::TDoubleVec TDoubleVec;
    typedef typename CMixtureDistribution<T>::TModeVec TModeVec;

    static const std::size_t MAX_ITERATIONS = 20u;

    double result = 0.0;

    const TDoubleVec &weights = distribution.weights();
    const TModeVec &modes = distribution.modes();

    if (weights.empty())
    {
        return result;
    }
    if (weights.size() == 1)
    {
        return mode(modes[0]);
    }

    mixture_detail::CPdfAdpater<T> f(distribution);
    double fMax = 0.0;
    for (std::size_t i = 0u; i < weights.size(); ++i)
    {
        try
        {
            double x25 = quantile(modes[i], 0.25);
            double x75 = quantile(modes[i], 0.75);
            std::size_t maxIterations = MAX_ITERATIONS;
            double x;
            double fx;
            CSolvers::maximize(x25, x75, f(x25), f(x75), f, 0.0, maxIterations, x, fx);
            if (fx > fMax)
            {
                result = x;
                fMax = fx;
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed to compute f(x) at mode: " << e.what());
            throw e;
        }
    }


    return result;
}

//! Compute the p.d.f. at \p x for \p distribution.
//!
//! \warning This propagates boost exceptions.
template<typename T>
double pdf(const CMixtureDistribution<T> &distribution, double x)
{
    typedef typename CMixtureDistribution<T>::TDoubleVec TDoubleVec;
    typedef typename CMixtureDistribution<T>::TModeVec TModeVec;

    if (CMathsFuncs::isNan(x))
    {
        LOG_ERROR("Bad value x = " << x);
        return 0.0;
    }

    double result = 0.0;

    const TDoubleVec &weights = distribution.weights();
    const TModeVec &modes = distribution.modes();

    if (weights.empty())
    {
        return result;
    }

    for (std::size_t i = 0u; i < weights.size(); ++i)
    {
        mixture_detail::TDoubleDoublePr ms = support(modes[i]);
        if (x >= ms.first && x <= ms.second)
        {
            try
            {
                double fx = pdf(modes[i], x);
                LOG_TRACE("x = " << x
                          << ", w(" << i << ") = " << weights[i]
                          << ", f(x, " << i << ") " << fx);
                result += weights[i] * fx;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to compute f(x) for mode at " << x << ": " << e.what());
                throw e;
            }
        }
        else
        {
            LOG_TRACE("x = " << x << ", support = (" << ms.first << "," << ms.second << ")");
        }
    }

    return result;
}

//! Compute the c.d.f. at \p x for \p distribution.
//!
//! \warning This propagates boost exceptions.
template<typename T>
double cdf(const CMixtureDistribution<T> &distribution, double x)
{
    typedef typename CMixtureDistribution<T>::TDoubleVec TDoubleVec;
    typedef typename CMixtureDistribution<T>::TModeVec TModeVec;

    if (CMathsFuncs::isNan(x))
    {
        LOG_ERROR("Bad value x = " << x);
        return 1.0;
    }

    const TDoubleVec &weights = distribution.weights();
    const TModeVec &modes = distribution.modes();

    if (weights.empty())
    {
        return 0.0;
    }

    double result = 0.0;
    for (std::size_t i = 0u; i < modes.size(); ++i)
    {
        mixture_detail::TDoubleDoublePr ms = support(modes[i]);
        if (x >= ms.second)
        {
            result += weights[i];
        }
        else if (x >= ms.first)
        {
            try
            {
                double fx = cdf(modes[i], x);
                LOG_TRACE("x = " << x
                          << ", w(" << i << ") = " << weights[i]
                          << ", f(x, " << i << ") " << fx);
                result += weights[i] * fx;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to compute f(x) for mode at "
                          << x << ": " << e.what());
                throw e;
            }
        }
        else
        {
            LOG_TRACE("x = " << x
                      << ", support = (" << ms.first << "," << ms.second << ")");
        }
    }

    return result;
}

//! Compute one minus the c.d.f. at \p x for \p distribution.
//!
//! \warning This propagates boost exceptions.
template<typename T>
double cdfComplement(const CMixtureDistribution<T> &distribution, double x)
{
    typedef typename CMixtureDistribution<T>::TDoubleVec TDoubleVec;
    typedef typename CMixtureDistribution<T>::TModeVec TModeVec;

    if (CMathsFuncs::isNan(x))
    {
        LOG_ERROR("Bad value x = " << x);
        return 1.0;
    }

    const TDoubleVec &weights = distribution.weights();
    const TModeVec &modes = distribution.modes();

    if (weights.empty())
    {
        return 1.0;
    }

    double result = 0.0;
    for (std::size_t i = 0u; i < modes.size(); ++i)
    {
        mixture_detail::TDoubleDoublePr ms = support(modes[i]);
        if (x < ms.first)
        {
            result += weights[i];
        }
        else if (x < ms.second)
        {
            try
            {
                double fx = cdf(complement(modes[i], x));
                LOG_TRACE("x = " << x
                          << ", w(" << i << ") = " << weights[i]
                          << ", f(x, " << i << ") " << fx);
                result += weights[i] * fx;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to compute f(x) for mode at "
                          << x << ": " << e.what());
                throw e;
            }
        }
        else
        {
            LOG_TRACE("x = " << x
                      << ", support = (" << ms.first << "," << ms.second << ")");
        }
    }

    return result;

}

namespace mixture_detail
{

//! Adapts the free c.d.f. function for use with the solver.
template<typename T>
class CCdfAdapter
{
    public:
        typedef double result_type;

    public:
        CCdfAdapter(const CMixtureDistribution<T> &distribution) :
            m_Distribution(&distribution)
        {
        }

        double operator()(const double x) const
        {
            return cdf(*m_Distribution, x);
        }

    private:
        const CMixtureDistribution<T> *m_Distribution;
};

}

//! Compute the \p q'th quantile for \p distribution.
//!
//! \warning This propagates boost exceptions.
template<typename T>
double quantile(const CMixtureDistribution<T> &distribution, const double q)
{
    typedef typename CMixtureDistribution<T>::TModeVec TModeVec;

    mixture_detail::TDoubleDoublePr s = support(distribution);

    if (q <= 0.0)
    {
        if (q < 0.0)
        {
            LOG_ERROR("Bad quantile " << q);
        }
        return s.first;
    }
    else if (q >= 1.0)
    {
        if (q > 1.0)
        {
            LOG_ERROR("Bad quantile " << q);
        }
        return s.second;
    }

    const TModeVec &modes = distribution.modes();
    if (modes.empty())
    {
        return q < 0.5 ? s.first : (q > 0.5 ? s.second : 0.0);
    }
    else if (modes.size() == 1)
    {
        return quantile(modes[0], q);
    }

    mixture_detail::CCdfAdapter<T> f(distribution);
    CCompositeFunctions::CMinusConstant<mixture_detail::CCdfAdapter<T> > fq(f, q);

    static const std::size_t MAX_ITERATIONS = 100u;
    static const double EPS = 1e-3;

    double x0 = mode(distribution);
    double result = x0;

    try
    {
        double f0 = fq(x0);
        double a = x0, b = x0, fa = f0, fb = f0;
        LOG_TRACE("(a,b) = [" << a << "," << b << "], "
                  << ", (f(a),f(b)) = [" << fa << "," << fb << "]");

        std::size_t maxIterations = MAX_ITERATIONS;
        if (   (f0 < 0 && !CSolvers::rightBracket(a, b, fa, fb, fq,
                                                  maxIterations,
                                                  s.first, s.second))
            || (f0 >= 0 && !CSolvers::leftBracket(a, b, fa, fb, fq,
                                                  maxIterations,
                                                  s.first, s.second)))
        {
            LOG_ERROR("Unable to bracket quantile = " << q
                      << ", (a,b) = (" << a << "," << b << ")"
                      << ", (f(a),f(b)) = (" << fa << "," << fb << ")");
            result = ::fabs(fa) < ::fabs(fb) ? a : b;
        }
        else
        {
            LOG_TRACE("(a,b) = (" << a << "," << b << ")"
                      << ", (f(a),f(b)) = (" << fa << "," << fb << ")");
            maxIterations = MAX_ITERATIONS - maxIterations;
            CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance,
                                              std::min(std::numeric_limits<double>::epsilon() * b,
                                                       EPS * q / std::max(fa, fb)));
            CSolvers::solve(a, b, fa, fb, fq, maxIterations, equal, result);
            LOG_TRACE("q = " << q
                      << ", x = " << result
                      << ", f(x) = " << fq(result)
                      << ", iterations = " << maxIterations);
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to compute quantile " << q);
        throw e;
    }

    return result;
}

}
}

#endif // INCLUDED_ml_maths_CMixtureDistribution_h
