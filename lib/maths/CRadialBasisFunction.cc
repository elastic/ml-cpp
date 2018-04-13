/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CRadialBasisFunction.h>

#include <core/CLogger.h>

#include <maths/CTools.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>

#include <cmath>

namespace ml {
namespace maths {

namespace {

//! Checks of the interval [\p a, \p b] contains the point \p x.
inline bool contains(double a, double b, double x) {
    return x >= a && x <= b;
}

//! The indefinite integral
//! <pre class="fragment">
//!   \f$\displaystyle \int_{-\infty}^x{(2\epsilon(u - c))^2e^{-2(\epsilon(u-c))^2}}du\f$
//! </pre>
double gaussianSquareDerivative(double x, double centre, double scale) {
    double r = scale * (x - centre);
    return scale *
           (boost::math::double_constants::root_two_pi * boost::math::erf(boost::math::double_constants::root_two * r) -
            4.0 * r * std::exp(-2.0 * r * r)) /
           4.0;
}

//! The indefinite integral
//! <pre class="fragment">
//!   \f$\displaystyle \int_{-\infty}^x{e^{-\epsilon_1(u - c_1))^2 - (\epsilon_2(u - c_2))^2}}du\f$
//! </pre>
double gaussianProduct(double x, double centre1, double centre2, double scale1, double scale2) {
    double ss = scale1 + scale2;
    double sd = scale2 - scale1;
    double scale = std::sqrt((ss * ss + sd * sd) / 2.0);

    double m = (scale1 * scale1 * centre1 + scale2 * scale2 * centre2) / (scale * scale);
    double d = scale1 * scale2 * (centre2 - centre1);

    return boost::math::double_constants::root_pi * std::exp(-d * d / (scale * scale)) * boost::math::erf(scale * (x - m)) / (2.0 * scale);
}

//! The indefinite integral
//! <pre class="fragment">
//!   \f$\displaystyle \int_{-\infty}^x{\frac{(2\epsilon(u - c))^2}{(1+(\epsilon(u - c)^2))^2}}du\f$
//! </pre>
double inverseQuadraticSquareDerivative(double x, double centre, double scale) {
    double r = scale * (x - centre);
    double d = (1.0 + r * r);
    return scale * (3.0 * r / d + 2.0 * r / (d * d) - 8.0 * r / (d * d * d) + 3.0 * std::atan(r)) / 12.0;
}

//! The indefinite integral
//! <pre class="fragment">
//!   \f$\displaystyle \int_{-\infty}^x{\frac{1}{(1+(\epsilon_1(u-c_1)^2)(1+(\epsilon_2(u-c_2)^2)}}du\f$
//! </pre>
double inverseQuadraticProduct(double x, double centre1, double centre2, double scale1, double scale2) {
    double r1 = scale1 * (x - centre1);
    double r2 = scale2 * (x - centre2);
    double ss = scale1 + scale2;
    double sd = scale2 - scale1;
    double d = scale1 * scale2 * (centre2 - centre1);

    if (sd == 0.0 && d == 0.0) {
        return (r1 / (1.0 + r1 * r1) + std::atan(r1)) / (2.0 * scale1);
    }

    if ((d * d) > 1.0) {
        return (scale1 * scale2 / d * std::log((1.0 + r1 * r1) / (1.0 + r2 * r2)) + scale1 * (1.0 - (ss * sd) / (d * d)) * std::atan(r1) +
                scale2 * (1.0 + (ss * sd) / (d * d)) * std::atan(r2)) /
               ((1.0 + (ss * ss) / (d * d)) * (d * d + sd * sd));
    }
    return (scale1 * scale2 * d * std::log((1.0 + r1 * r1) / (1.0 + r2 * r2)) + (d * d - ss * sd) * scale1 * std::atan(r1) +
            (d * d + ss * sd) * scale2 * std::atan(r2)) /
           ((d * d + ss * ss) * (d * d + sd * sd));
}
}

CRadialBasisFunction::~CRadialBasisFunction() {
}

CGaussianBasisFunction* CGaussianBasisFunction::clone() const {
    return new CGaussianBasisFunction();
}

double CGaussianBasisFunction::value(double x, double centre, double scale) const {
    double r = x - centre;
    double y = scale * r;
    return std::exp(-y * y);
}

double CGaussianBasisFunction::derivative(double x, double centre, double scale) const {
    double r = x - centre;
    double y = scale * r;
    return -2.0 * scale * y * std::exp(-y * y);
}

bool CGaussianBasisFunction::scale(double distance, double value, double& result) const {
    if (value <= 0.0 || value >= 1.0) {
        return false;
    }
    result = std::sqrt(-std::log(value)) / distance;
    return true;
}

double CGaussianBasisFunction::mean(double a, double b, double centre, double scale) const {
    // The maximum function value is at the minimum of |x - c|
    // in the range [a,b] and the maximum is at the maximum of
    // |x - c|. Denoting these x+ and x-, respectively, we can
    // bound the mean by:
    //    f(x-, c) <= m <= f(x+, c)
    //
    // So the maximum error taking the mid point is:
    //    1/2 * (f(x+, c) - f(x-, c))
    //
    // If this is smaller than the maximum precision available
    // for the mean we return this mid point.

    static const double EPS = std::numeric_limits<double>::epsilon();

    double m = (a + b) / 2.0;
    double fmin = this->value(centre < m ? b : a, centre, scale);
    double fmax = this->value(CTools::truncate(centre, a, b), centre, scale);

    if (fmax - fmin <= 2.0 * EPS * fmin * (b - a)) {
        return (fmax + fmin) / 2.0;
    }

    return std::max(boost::math::double_constants::root_pi / 2.0 / scale *
                        (boost::math::erf(scale * (b - centre)) - boost::math::erf(scale * (a - centre))) / (b - a),
                    0.0);
}

double CGaussianBasisFunction::meanSquareDerivative(double a, double b, double centre, double scale) const {
    // The maximum of the derivative function is at the point
    // c +/- 1 / sqrt(2) / s. To find the maximum and minimum
    // values of the derivative function x+ and x- we need to
    // check if the range [a,b] contains c and either of these
    // points. Having found x+ and x- we can bound the mean by:
    //    f(x-, c) <= m <= f(x+, c)
    //
    // So the maximum error taking the mid point is:
    //    1/2 * (f(x+, c) - f(x-, c))
    //
    // If this is smaller than the maximum precision available
    // for the derivative we return this mid point.

    static const double EPS = std::numeric_limits<double>::epsilon();

    double maxima[] = {centre - 1.0 / (boost::math::double_constants::root_two * scale),
                       centre + 1.0 / (boost::math::double_constants::root_two * scale)};

    double fa = this->derivative(a, centre, scale);
    double fb = this->derivative(b, centre, scale);
    double fmin = contains(a, b, centre) ? 0.0 : std::min(fa, fb);
    double fmax = (contains(a, b, maxima[0]) || contains(a, b, maxima[1])) ? this->derivative(maxima[0], centre, scale) : std::max(fa, fb);

    double smin = fmin * fmin;
    double smax = fmax * fmax;

    if (smax - smin <= 2.0 * EPS * smin * (b - a)) {
        return (smin + smax) / 2.0;
    }

    return std::max((gaussianSquareDerivative(b, centre, scale) - gaussianSquareDerivative(a, centre, scale)) / (b - a), 0.0);
}

double CGaussianBasisFunction::product(double a, double b, double centre1, double centre2, double scale1, double scale2) const {
    // The maximum function value is at the minimum of |x - c|
    // in the range [a,b] and the maximum is at the maximum of
    // |x - c|. Denoting these x+ and x-, respectively, we can
    // bound the product by:
    //    f(x-|c1, c1) * f(x-|c2, c2) <= p <= f(x+|c1, c1) * f(x-|c2, c2)
    //
    // So the maximum error taking the mid point is:
    //    1/2 * (f(x+|c1, c1) * f(x-|c2, c2)
    //           - f(x-|c1, c1) * f(x-|c2, c2))
    //
    // If this is smaller than the maximum precision available
    // for the product we return this mid point.

    static const double EPS = std::numeric_limits<double>::epsilon();

    double m = (a + b) / 2.0;
    double f1min = this->value(centre1 < m ? b : a, centre1, scale1);
    double f1max = this->value(CTools::truncate(centre1, a, b), centre1, scale1);
    double f2min = this->value(centre2 < m ? b : a, centre1, scale2);
    double f2max = this->value(CTools::truncate(centre2, a, b), centre2, scale2);

    double pmin = f1min * f2min;
    double pmax = f1max * f2max;

    if (pmax - pmin <= 2.0 * EPS * pmin * (b - a)) {
        return (pmin + pmax) / 2.0;
    }

    return std::max((gaussianProduct(b, centre1, centre2, scale1, scale2) - gaussianProduct(a, centre1, centre2, scale1, scale2)) / (b - a),
                    0.0);
}

CInverseQuadraticBasisFunction* CInverseQuadraticBasisFunction::clone() const {
    return new CInverseQuadraticBasisFunction();
}

double CInverseQuadraticBasisFunction::value(double x, double centre, double scale) const {
    double r = x - centre;
    double y = scale * r;
    return 1.0 / (1.0 + y * y);
}

double CInverseQuadraticBasisFunction::derivative(double x, double centre, double scale) const {
    double r = x - centre;
    double y = scale * r;
    double yy = (1.0 + y * y);
    return -2.0 * scale * y / yy / yy;
}

double CInverseQuadraticBasisFunction::mean(double a, double b, double centre, double scale) const {
    // The maximum function value is at the minimum of |x - c|
    // in the range [a,b] and the maximum is at the maximum of
    // |x - c|. Denoting these x+ and x-, respectively, we can
    // bound the mean by:
    //    f(x-, c) <= m <= f(x+, c)
    //
    // So the maximum error taking the mid point is:
    //    1/2 * (f(x+, c) - f(x-, c))
    //
    // If this is smaller than the maximum precision available
    // for the mean we return this mid point.

    static const double EPS = std::numeric_limits<double>::epsilon();

    double m = (a + b) / 2.0;
    double fmin = this->value(centre < m ? b : a, centre, scale);
    double fmax = this->value(CTools::truncate(centre, a, b), centre, scale);

    if (fmax - fmin <= 2.0 * EPS * fmin * (b - a)) {
        return (fmax + fmin) / 2.0;
    }

    return std::max((std::atan(scale * (b - centre)) - std::atan(scale * (a - centre))) / scale / (b - a), 0.0);
}

double CInverseQuadraticBasisFunction::meanSquareDerivative(double a, double b, double centre, double scale) const {
    // The maximum of the derivative function is at the point
    // c +/- 1 / sqrt(3) / s. To find the maximum and minimum
    // values of the derivative function x+ and x- we need to
    // check if the range [a,b] contains c and either of these
    // points. Having found x+ and x- we can bound the mean by:
    //    f(x-, c) <= m <= f(x+, c)
    //
    // So the maximum error taking the mid point is:
    //    1/2 * (f(x+, c) - f(x-, c))
    //
    // If this is smaller than the maximum precision available
    // for the derivative we return this mid point.

    static const double EPS = std::numeric_limits<double>::epsilon();

    double maxima[] = {centre - 1.0 / (boost::math::double_constants::root_three * scale),
                       centre + 1.0 / (boost::math::double_constants::root_three * scale)};

    double fa = this->derivative(a, centre, scale);
    double fb = this->derivative(b, centre, scale);
    double fmin = contains(a, b, centre) ? 0.0 : std::min(fa, fb);
    double fmax = (contains(a, b, maxima[0]) || contains(a, b, maxima[1])) ? this->derivative(maxima[0], centre, scale) : std::max(fa, fb);

    double smin = fmin * fmin;
    double smax = fmax * fmax;

    if (smax - smin <= 2.0 * EPS * smin * (b - a)) {
        return (smin + smax) / 2.0;
    }

    return std::max((inverseQuadraticSquareDerivative(b, centre, scale) - inverseQuadraticSquareDerivative(a, centre, scale)) / (b - a),
                    0.0);
}

bool CInverseQuadraticBasisFunction::scale(double distance, double value, double& result) const {
    if (value <= 0.0 || value >= 1.0) {
        return false;
    }
    result = std::sqrt((1.0 - value) / value) / distance;
    return true;
}

double CInverseQuadraticBasisFunction::product(double a, double b, double centre1, double centre2, double scale1, double scale2) const {
    // The maximum function value is at the minimum of |x - c|
    // in the range [a,b] and the maximum is at the maximum of
    // |x - c|. Denoting these x+ and x-, respectively, we can
    // bound the product by:
    //    f(x-|c1, c1) * f(x-|c2, c2) <= p <= f(x+|c1, c1) * f(x-|c2, c2)
    //
    // So the maximum error taking the mid point is:
    //    1/2 * (f(x+|c1, c1) * f(x-|c2, c2)
    //           - f(x-|c1, c1) * f(x-|c2, c2))
    //
    // If this is smaller than the maximum precision available
    // for the product we return this mid point.

    static const double EPS = std::numeric_limits<double>::epsilon();

    double m = (a + b) / 2.0;
    double f1min = this->value(centre1 < m ? b : a, centre1, scale1);
    double f1max = this->value(CTools::truncate(centre1, a, b), centre1, scale1);
    double f2min = this->value(centre2 < m ? b : a, centre1, scale2);
    double f2max = this->value(CTools::truncate(centre2, a, b), centre2, scale2);

    double pmin = f1min * f2min;
    double pmax = f1max * f2max;

    if (pmax - pmin <= 2.0 * EPS * pmin * (b - a)) {
        return (pmin + pmax) / 2.0;
    }

    return std::max(
        (inverseQuadraticProduct(b, centre1, centre2, scale1, scale2) - inverseQuadraticProduct(a, centre1, centre2, scale1, scale2)) /
            (b - a),
        0.0);
}
}
}
