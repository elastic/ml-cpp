/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLogTDistribution.h>

#include <maths/CMathsFuncs.h>
#include <maths/CTools.h>

#include <boost/math/distributions/students_t.hpp>
#include <boost/optional.hpp>

#include <cmath>

namespace ml
{
namespace maths
{

namespace
{

inline double square(double x)
{
    return x * x;
}

}

CLogTDistribution::CLogTDistribution(double degreesFreedom,
                                     double location,
                                     double scale) :
        m_DegreesFreedom(degreesFreedom),
        m_Location(location),
        m_Scale(scale)
{}

double CLogTDistribution::degreesFreedom() const
{
    return m_DegreesFreedom;
}

double CLogTDistribution::location() const
{
    return m_Location;
}

double CLogTDistribution::scale() const
{
    return m_Scale;
}

CLogTDistribution::TDoubleDoublePr support(const CLogTDistribution &/*distribution*/)
{
    return CLogTDistribution::TDoubleDoublePr(0.0, boost::numeric::bounds<double>::highest());
}

double mode(const CLogTDistribution &distribution)
{
    // The mode of a log t distribution is found by taking the derivative
    // of the p.d.f. In particular,
    //   f(x) ~ 1 / x * (1 + 1 / (n * s^2) * (log(x) - m)^2) ^ -((n+1)/2)
    //
    // where,
    //   n are the degrees freedom,
    //   m is the location and
    //   s is the scale.
    //
    // The maximum occurs when:
    //     ( -1 / x^2 ) * ( (1 + 1 / (n * s^2) * (log(x) - m)^2) ^ -((n+1)/2) )
    //   + ( 1 / x ) * ( 1 / x * 2 / (n * s^2) * (log(x) - m) * -(n+1)/2
    //                   * (1 + 1 / (n * s^2) * (log(x) - m)^2) ^ -((n+3)/2) ) = 0
    //
    // Canceling common factors this reduces to the quadratic:
    //   0 = 1 + 1 / (n * s^2) * (log(x) - m)^2 + (n+1) / (n * s^2) * (log(x) - m)
    //     = (log(x) - m)^2 + (n+1) * (log(x) - m) + n * s^2
    //
    // This has solutions provided:
    //   (n+1) ^ 2 > 4 * n * s^2
    //
    // Otherwise, the distribution is single sided with mode at the origin.
    // The root of interest is:
    //   log(x) - m = -(n+1) / 2 + ((n+1)^2 / 4 - n * s^2) ^ (1/2)
    //
    // The mode is at:
    //   x = exp(m - (n+1) / 2 + ((n+1)^2 / 4 - n * s^2) ^ (1/2))

    double degreesFreedom = distribution.degreesFreedom();
    double squareScale = square(distribution.scale());

    if (square(degreesFreedom + 1.0) < 4.0 * degreesFreedom * squareScale)
    {
        return 0.0;
    }

    double location = distribution.location();

    return std::exp(location - (degreesFreedom + 1.0) / 2.0
                          + std::sqrt(square(degreesFreedom + 1.0) / 4.0
                                   - degreesFreedom * squareScale));
}

CLogTDistribution::TOptionalDouble localMinimum(const CLogTDistribution &distribution)
{
    // The distribution has a local minimum at:
    //   x = exp(m - (n+1) / 2 - ((n+1)^2 / 4 - n*s^2) ^ (1/2))
    //
    // provided:
    //   (n+1) ^ 2 > 4 * n * s^2
    //
    // See the documentation in the mode function for more details.

    double degreesFreedom = distribution.degreesFreedom();
    double squareScale = square(distribution.scale());

    if (square(degreesFreedom + 1.0) < 4.0 * degreesFreedom * squareScale)
    {
        return CLogTDistribution::TOptionalDouble();
    }

    double location = distribution.location();

    return std::exp(location - (degreesFreedom + 1.0) / 2.0
                          - std::sqrt(square(degreesFreedom + 1.0) / 4.0
                                   - degreesFreedom * squareScale));
}

double pdf(const CLogTDistribution &distribution, double x)
{
    // It can be shown that the p.d.f. is related to the student's t
    // p.d.f. by:
    //   f(x) = 1 / (s * x) * f((log(x) - m) / s | n)
    //
    // where,
    //   f( . | n) is a student's t p.d.f. with n degrees of freedom,
    //   s is the scale and
    //   m is the location.

    if (x < 0.0)
    {
        return 0.0;
    }
    else if (x == 0.0)
    {
        // In limit x tends down to 0 it can be shown that the density
        // function tends to:
        //   f(x) = f(e^l) * (v^(1/2) * s)^(v+1) / (y * log(y)^(v+1))
        //
        // where, y = x / e^l. Repeated application of l'Hopital's rule
        // shows that:
        //   lim_{y -> 0}{ y * log(y)^(v+1) } = 0
        //
        // So the density function is actually infinite at x = 0. We'll use
        // the p.d.f evaluated at the smallest positive double as a proxy.
        x = std::numeric_limits<double>::min();
    }

    double degreesFreedom = distribution.degreesFreedom();
    boost::math::students_t_distribution<> students(degreesFreedom);

    double scale = distribution.scale();
    double location = distribution.location();
    double value = (std::log(x) - location) / scale;

    return CTools::safePdf(students, value) / scale / x;
}

double cdf(const CLogTDistribution &distribution, double x)
{
    // It can be shown that the c.d.f. is related to the student's t
    // c.d.f. by:
    //   F(x) = F((log(x) - m) / s | n)
    //
    // where,
    //   F( . | n) is a student's t c.d.f. with n degrees of freedom,
    //   s is the scale and
    //   m is the location.

    if (CMathsFuncs::isNan(x))
    {
        LOG_ERROR("Bad argument x = " << x);
        return 0.0;
    }
    else if (x <= 0.0)
    {
        return 0.0;
    }

    double degreesFreedom = distribution.degreesFreedom();
    boost::math::students_t_distribution<> students(degreesFreedom);

    double scale = distribution.scale();
    double location = distribution.location();
    double value = (std::log(x) - location) / scale;

    return CTools::safeCdf(students, value);
}

double cdfComplement(const CLogTDistribution &distribution, double x)
{
    // This is just 1 - F(x) but uses boost::math::complement to
    // avoid cancellation errors.

    if (CMathsFuncs::isNan(x))
    {
        LOG_ERROR("Bad argument x = " << x);
        return 0.0;
    }
    else if (x <= 0.0)
    {
        return 1.0;
    }

    double degreesFreedom = distribution.degreesFreedom();
    boost::math::students_t_distribution<> students(degreesFreedom);

    double scale = distribution.scale();
    double location = distribution.location();
    double value = (std::log(x) - location) / scale;

    return CTools::safeCdfComplement(students, value);
}

double quantile(const CLogTDistribution &distribution, double q)
{
    // The distribution describes X = exp(s * Y + m) where Y is student's
    // t. This implies that the quantile's are obtained from the student's t
    // distribution by the transformation x_q = exp(s * y_q + m).

    double degreesFreedom = distribution.degreesFreedom();
    boost::math::students_t_distribution<> students(degreesFreedom);
    double y_q = boost::math::quantile(students, q);

    double scale = distribution.scale();
    double location = distribution.location();
    return std::exp(scale * y_q + location);
}

}
}
