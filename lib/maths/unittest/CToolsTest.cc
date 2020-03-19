/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CCompositeFunctions.h>
#include <maths/CIntegration.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CLogTDistribution.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <array>
#include <numeric>

BOOST_AUTO_TEST_SUITE(CToolsTest)

using namespace ml;
using namespace maths;
using namespace test;

namespace {

using TDoubleDoublePr = std::pair<double, double>;
using TDoubleBoolPr = std::pair<double, bool>;
using TDoubleVec = std::vector<double>;

namespace adapters {

template<typename DISTRIBUTION>
bool isDiscrete(const DISTRIBUTION&) {
    return false;
}
bool isDiscrete(const boost::math::negative_binomial_distribution<>&) {
    return true;
}

template<typename DISTRIBUTION>
TDoubleDoublePr support(const DISTRIBUTION& distribution) {
    return boost::math::support(distribution);
}
TDoubleDoublePr support(const CLogTDistribution& logt) {
    CLogTDistribution::TOptionalDouble minimum = localMinimum(logt);
    return TDoubleDoublePr(minimum ? *minimum : 0.0,
                           boost::math::tools::max_value<double>());
}

template<typename DISTRIBUTION>
TDoubleBoolPr stationaryPoint(const DISTRIBUTION& distribution) {
    return TDoubleBoolPr(boost::math::mode(distribution), true);
}
TDoubleBoolPr stationaryPoint(const CLogTDistribution& logt) {
    return TDoubleBoolPr(ml::maths::mode(logt), true);
}
TDoubleBoolPr stationaryPoint(const boost::math::beta_distribution<>& beta) {
    if (beta.alpha() < 1.0 && beta.beta() < 1.0) {
        return TDoubleBoolPr((beta.alpha() - 1.0) / (beta.alpha() + beta.beta() - 2.0), false);
    }
    return TDoubleBoolPr(boost::math::mode(beta), true);
}

template<typename DISTRIBUTION>
double pdf(const DISTRIBUTION& distribution, const double& x) {
    return CTools::safePdf(distribution, x);
}
double pdf(const CLogTDistribution& logt, const double& x) {
    return ml::maths::pdf(logt, x);
}

template<typename DISTRIBUTION>
double cdf(const DISTRIBUTION& distribution, const double& x) {
    return CTools::safeCdf(distribution, x);
}
double cdf(const CLogTDistribution& logt, const double& x) {
    return ml::maths::cdf(logt, x);
}

template<typename DISTRIBUTION>
double cdfComplement(const DISTRIBUTION& distribution, const double& x) {
    return CTools::safeCdfComplement(distribution, x);
}
double cdfComplement(const CLogTDistribution& logt, const double& x) {
    return ml::maths::cdfComplement(logt, x);
}

} // adapters::

template<typename DISTRIBUTION>
double numericalProbabilityOfLessLikelySampleImpl(const DISTRIBUTION& distribution, double x) {
    TDoubleBoolPr stationaryPoint = adapters::stationaryPoint(distribution);

    double eps = 1e-8;

    double pdf = adapters::pdf(distribution, x);
    LOG_TRACE(<< "x = " << x << ", f(x) = " << pdf
              << ", stationaryPoint = " << stationaryPoint.first);

    double x1 = stationaryPoint.first;
    if (x > stationaryPoint.first) {
        // Search for lower bound.
        double minX = adapters::support(distribution).first + eps;
        for (double increment = std::max(x1 / 2.0, 1.0);
             x1 > minX &&
             ((stationaryPoint.second && adapters::pdf(distribution, x1) > pdf) ||
              (!stationaryPoint.second && adapters::pdf(distribution, x1) < pdf));
             x1 = std::max(x1 - increment, minX), increment *= 2.0) {
            // Empty.
        }
    }

    double x2 = stationaryPoint.first;
    if (x < stationaryPoint.first) {
        // Search for upper bound.
        double maxX = adapters::support(distribution).second - eps;
        for (double increment = std::max(x2 / 2.0, 1.0);
             x2 < maxX &&
             ((stationaryPoint.second && adapters::pdf(distribution, x2) > pdf) ||
              (!stationaryPoint.second && adapters::pdf(distribution, x2) < pdf));
             x2 = std::min(x2 + increment, maxX), increment *= 2.0) {
            // Empty.
        }
    }
    LOG_TRACE(<< "1) x1 = " << x1 << ", x2 = " << x2);

    if (adapters::pdf(distribution, x1) > adapters::pdf(distribution, x2)) {
        std::swap(x1, x2);
    }
    LOG_TRACE(<< "2) x1 = " << x1 << ", x2 = " << x2);

    // Binary search.
    while (std::fabs(x1 - x2) > eps) {
        double x3 = (x1 + x2) / 2.0;
        if (adapters::pdf(distribution, x3) > pdf) {
            x2 = x3;
        } else {
            x1 = x3;
        }
    }
    LOG_TRACE(<< "3) x1 = " << x1 << ", x2 = " << x2);

    double y = (x > (x1 + x2) / 2.0) ? std::min(x1, x2) : std::max(x1, x2);
    if (x > y) {
        std::swap(x, y);
    }

    LOG_TRACE(<< "x = " << x << ", y = " << y
              << ", f(x) = " << adapters::pdf(distribution, x)
              << ", f(y) = " << adapters::pdf(distribution, y));

    if (stationaryPoint.second) {
        double cdfy =
            adapters::cdfComplement(distribution, y) +
            (adapters::isDiscrete(distribution) ? adapters::pdf(distribution, y) : 0.0);
        double cdfx = adapters::cdf(distribution, x);

        LOG_TRACE(<< "F(x) = " << cdfx << ", 1 - F(y) = " << cdfy);

        return cdfx + cdfy;
    }

    double cdfy = adapters::cdf(distribution, y) +
                  (adapters::isDiscrete(distribution) ? adapters::pdf(distribution, y) : 0.0);
    double cdfx = adapters::cdf(distribution, x);

    LOG_TRACE(<< "F(x) = " << cdfx << ", F(y) = " << cdfy);

    return cdfy - cdfx;
}

template<typename DISTRIBUTION>
double numericalProbabilityOfLessLikelySample(const DISTRIBUTION& distribution, double x) {
    return numericalProbabilityOfLessLikelySampleImpl(distribution, x);
}

double numericalProbabilityOfLessLikelySample(const boost::math::negative_binomial_distribution<>& negativeBinomial,
                                              double x) {
    double fx = CTools::safePdf(negativeBinomial, x);

    double m = boost::math::mode(negativeBinomial);
    double fm = CTools::safePdf(negativeBinomial, m);
    if (fx >= fm) {
        return 1.0;
    }
    double f0 = CTools::safePdf(negativeBinomial, 0.0);
    if (x > m && fx < f0) {
        return CTools::safeCdfComplement(negativeBinomial, x) +
               CTools::safePdf(negativeBinomial, x);
    }
    return numericalProbabilityOfLessLikelySampleImpl(negativeBinomial, x);
}

double numericalProbabilityOfLessLikelySample(const CLogTDistribution& logt, double x) {
    // We need special handling for the case that the p.d.f. is
    // single sided and if it is greater at x than the local "mode".

    double m = mode(logt);
    if (m == 0.0) {
        return cdfComplement(logt, x);
    }

    double fx = pdf(logt, x);
    double fm = pdf(logt, m);
    if (fx > fm) {
        return cdfComplement(logt, x);
    }

    CLogTDistribution::TOptionalDouble xmin = localMinimum(logt);
    if (xmin && (pdf(logt, *xmin) > fx || *xmin == m)) {
        return cdfComplement(logt, x);
    }

    return numericalProbabilityOfLessLikelySampleImpl(logt, x);
}

double numericalProbabilityOfLessLikelySample(const boost::math::beta_distribution<>& beta,
                                              double x) {
    // We need special handling of the case that the equal p.d.f.
    // point is very close to 0 or 1.

    double fx = CTools::safePdf(beta, x);

    double a = beta.alpha();
    double b = beta.beta();

    double xmin = 1000.0 * std::numeric_limits<double>::min();
    if (a >= 1.0 && fx < CTools::safePdf(beta, xmin)) {
        return std::exp(a * std::log(xmin) - std::log(a) + std::lgamma(a + b) -
                        std::lgamma(a) - std::lgamma(b)) +
               CTools::safeCdfComplement(beta, x);
    }

    double xmax = 1.0 - std::numeric_limits<double>::epsilon();
    if (b >= 1.0 && fx < CTools::safePdf(beta, xmax)) {
        double y = std::exp(std::log(boost::math::beta(a, b) * fx) / b);
        return std::pow(y, b) / b / boost::math::beta(a, b) + CTools::safeCdf(beta, x);
    }

    return numericalProbabilityOfLessLikelySampleImpl(beta, x);
}

template<typename DISTRIBUTION>
class CPdf {
public:
    using result_type = double;

public:
    CPdf(const DISTRIBUTION& distribution) : m_Distribution(distribution) {}

    bool operator()(double x, double& result) const {
        result = boost::math::pdf(m_Distribution, x);
        return true;
    }

private:
    DISTRIBUTION m_Distribution;
};

class CIdentity {
public:
    using result_type = double;

public:
    bool operator()(double x, double& result) const {
        result = x;
        return true;
    }
};

template<typename DISTRIBUTION>
double numericalIntervalExpectation(const DISTRIBUTION& distribution, double a, double b) {
    double numerator = 0.0;
    double denominator = 0.0;

    CPdf<DISTRIBUTION> fx(distribution);
    maths::CCompositeFunctions::CProduct<CPdf<DISTRIBUTION>, CIdentity> xfx(fx);
    double dx = (b - a) / 10.0;
    for (std::size_t i = 0u; i < 10; ++i, a += dx) {
        double fxi;
        BOOST_TEST_REQUIRE(maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(
            fx, a, a + dx, fxi));
        double xfxi;
        BOOST_TEST_REQUIRE(maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(
            xfx, a, a + dx, xfxi));
        numerator += xfxi;
        denominator += fxi;
    }

    return numerator / denominator;
}

template<typename T>
class CTruncatedPdf {
public:
    CTruncatedPdf(const maths::CMixtureDistribution<T>& mixture, double cutoff)
        : m_Mixture(mixture), m_Cutoff(cutoff) {}

    bool operator()(double x, double& fx) const {
        fx = maths::pdf(m_Mixture, x);
        if (fx > m_Cutoff) {
            fx = 0.0;
        }
        return true;
    }

private:
    const maths::CMixtureDistribution<T>& m_Mixture;
    double m_Cutoff;
};

template<typename T>
class CLogPdf {
public:
    CLogPdf(const maths::CMixtureDistribution<T>& mixture)
        : m_Mixture(mixture) {}

    double operator()(double x) const {
        return std::log(maths::pdf(m_Mixture, x));
    }

    bool operator()(double x, double& fx) const {
        fx = std::log(maths::pdf(m_Mixture, x));
        return true;
    }

private:
    const maths::CMixtureDistribution<T>& m_Mixture;
};
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySample) {
    // The probability of a lower likelihood sample x from a single
    // mode distribution is:
    //   F(a) + 1 - F(b)
    //
    // where,
    //   a satisfies f(a) = f(x) and a < m,
    //   b satisfies f(b) = f(x) and b > m,
    //   F(.) denotes the c.d.f. of the distribution,
    //   f(.) denotes the p.d.f. of the distribution and
    //   m is the mode of the distribution.
    //
    // We'll use this relation to test our probabilities.

    CTools::CProbabilityOfLessLikelySample probabilityOfLessLikelySample(maths_t::E_TwoSided);

    double p1, p2;
    maths_t::ETail tail = maths_t::E_UndeterminedTail;
    double m;

    LOG_DEBUG(<< "******** normal ********");

    boost::math::normal_distribution<> normal(3.0, 5.0);

    p1 = numericalProbabilityOfLessLikelySample(normal, -1.0);
    p2 = probabilityOfLessLikelySample(normal, -1.0, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.01 * std::max(p1, p2));
    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

    m = adapters::stationaryPoint(normal).first;
    tail = maths_t::E_UndeterminedTail;
    p1 = 1.0;
    p2 = probabilityOfLessLikelySample(normal, m, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_EQUAL(p1, p2);
    BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

    tail = maths_t::E_UndeterminedTail;
    p1 = numericalProbabilityOfLessLikelySample(normal, 8.0);
    p2 = probabilityOfLessLikelySample(normal, 8.0, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.01 * std::max(p1, p2));
    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);

    LOG_DEBUG(<< "******** student's t ********");

    boost::math::students_t_distribution<> students(2.0);

    tail = maths_t::E_UndeterminedTail;
    p1 = numericalProbabilityOfLessLikelySample(students, -4.0);
    p2 = probabilityOfLessLikelySample(students, -4.0, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.01 * std::max(p1, p2));
    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

    m = adapters::stationaryPoint(students).first;
    tail = maths_t::E_UndeterminedTail;
    p1 = 1.0;
    p2 = probabilityOfLessLikelySample(students, m, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_EQUAL(p1, p2);
    BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

    tail = maths_t::E_UndeterminedTail;
    p1 = numericalProbabilityOfLessLikelySample(students, 3.0);
    p2 = probabilityOfLessLikelySample(students, 3.0, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.01 * std::max(p1, p2));
    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);

    LOG_DEBUG(<< "******** negative binomial ********");

    {
        double successFraction[] = {0.5, 1.0, 10.0, 100.0, 1000.0};
        double successProbability[] = {1e-3, 0.25, 0.5, 0.75, 1.0 - 1e-3};

        for (size_t i = 0; i < boost::size(successFraction); ++i) {
            for (size_t j = 0; j < boost::size(successProbability); ++j) {
                LOG_DEBUG(<< "**** r = " << successFraction[i]
                          << ", p = " << successProbability[j] << " ****");

                boost::math::negative_binomial_distribution<> negativeBinomial(
                    successFraction[i], successProbability[j]);

                if (successFraction[i] <= 1.0) {
                    // Monotone decreasing.
                    double x = std::fabs(std::log(successProbability[j]));
                    for (int l = 0; l < 10; ++l) {
                        tail = maths_t::E_UndeterminedTail;
                        x = std::floor(2.0 * x + 0.5);
                        p1 = CTools::safeCdfComplement(negativeBinomial, x) +
                             CTools::safePdf(negativeBinomial, x);
                        p2 = probabilityOfLessLikelySample(negativeBinomial, x, tail);
                        LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 1e-3 * std::max(p1, p2));
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    }
                    continue;
                }

                double m1 = boost::math::mode(negativeBinomial);

                BOOST_REQUIRE_EQUAL(
                    1.0, probabilityOfLessLikelySample(negativeBinomial, m1, tail));
                BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

                double offset = m1;
                for (int l = 0; l < 5; ++l) {
                    offset /= 2.0;

                    double x = std::floor(m1 - offset);
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(negativeBinomial, x);
                    p2 = probabilityOfLessLikelySample(negativeBinomial, x, tail);
                    LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.02 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.02 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    if (offset > 0.0)
                        BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                    if (offset == 0.0)
                        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

                    x = std::ceil(m1 + offset);
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(negativeBinomial, x);
                    p2 = probabilityOfLessLikelySample(negativeBinomial, x, tail);
                    LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.02 * std::max(p1, p2));
                    if (offset > 0.0)
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    if (offset == 0.0)
                        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                }

                double factor = 1.0;
                for (int l = 0; l < 5; ++l) {
                    factor *= 2.0;

                    double x = std::floor(m1 / factor);
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(negativeBinomial, x);
                    p2 = probabilityOfLessLikelySample(negativeBinomial, x, tail);
                    LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.01 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.05 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    if (x != m1)
                        BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                    if (x == m1)
                        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

                    x = std::ceil(m1 * factor);
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(negativeBinomial, x);
                    p2 = probabilityOfLessLikelySample(negativeBinomial, x, tail);
                    LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.01 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.05 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    if (x != m1)
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    if (x == m1)
                        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                }
            }
        }
    }

    // Note when we are testing the tails of the distribution we
    // use the relative error in the log.

    LOG_DEBUG(<< "******** log normal ********");

    boost::math::lognormal_distribution<> logNormal(1.0, 0.5);

    tail = maths_t::E_UndeterminedTail;
    p1 = -std::log(numericalProbabilityOfLessLikelySample(logNormal, 0.3));
    p2 = -std::log(probabilityOfLessLikelySample(logNormal, 0.3, tail));
    LOG_DEBUG(<< "-log(p1) = " << p1 << ", -log(p2) = " << p2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.05 * std::max(p1, p2));
    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

    m = adapters::stationaryPoint(logNormal).first;
    tail = maths_t::E_UndeterminedTail;
    p1 = 1.0;
    p2 = probabilityOfLessLikelySample(logNormal, m, tail);
    LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
    BOOST_REQUIRE_EQUAL(p1, p2);
    BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

    tail = maths_t::E_UndeterminedTail;
    p1 = -std::log(numericalProbabilityOfLessLikelySample(logNormal, 12.0));
    p2 = -std::log(probabilityOfLessLikelySample(logNormal, 12.0, tail));
    LOG_DEBUG(<< "-log(p1) = " << p1 << ", -log(p2) = " << p2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.01 * std::max(p1, p2));
    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);

    LOG_DEBUG(<< "******** log t ********");

    {
        double degreesFreedom[] = {1.0, 5.0, 50.0};
        double locations[] = {-0.5, 0.1, 1.0, 5.0};
        double scales[] = {0.1, 1.0, 5.0};

        for (size_t i = 0; i < boost::size(degreesFreedom); ++i) {
            for (size_t j = 0; j < boost::size(locations); ++j) {
                for (size_t k = 0; k < boost::size(scales); ++k) {
                    LOG_DEBUG(<< "**** v = " << degreesFreedom[i] << ", l = "
                              << locations[j] << ", s = " << scales[k] << " ****");

                    CLogTDistribution logt(degreesFreedom[i], locations[j], scales[k]);

                    double m1 = mode(logt);
                    if (m1 == 0.0) {
                        // Monotone decreasing.
                        double x = std::exp(locations[j]) / 32.0;
                        tail = maths_t::E_UndeterminedTail;
                        for (int l = 0; l < 10; ++l) {
                            x *= 2.0;
                            p1 = cdfComplement(logt, x);
                            p2 = probabilityOfLessLikelySample(logt, x, tail);
                            LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2);
                            BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 1e-3 * std::max(p1, p2));
                        }
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                        continue;
                    }

                    double offset = m1;
                    for (int l = 0; l < 5; ++l) {
                        offset /= 2.0;

                        double x = m1 - offset;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = numericalProbabilityOfLessLikelySample(logt, x);
                        p2 = probabilityOfLessLikelySample(logt, x, tail);
                        LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.02 * std::max(p1, p2));
                        BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

                        x = m1 + offset;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = numericalProbabilityOfLessLikelySample(logt, x);
                        p2 = probabilityOfLessLikelySample(logt, x, tail);
                        LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.02 * std::max(p1, p2));
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    }

                    double factor = 1.0;
                    for (int l = 0; l < 5; ++l) {
                        factor *= 2.0;

                        double x = m1 / factor;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = numericalProbabilityOfLessLikelySample(logt, x);
                        p2 = probabilityOfLessLikelySample(logt, x, tail);
                        LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                                  << ", log(p1) = " << std::log(p1)
                                  << ", log(p2) = " << std::log(p2));
                        BOOST_TEST_REQUIRE(
                            (std::fabs(p1 - p2) <= 0.01 * std::max(p1, p2) ||
                             std::fabs(std::log(p1) - std::log(p2)) <=
                                 0.05 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                        BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

                        x = m1 * factor;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = numericalProbabilityOfLessLikelySample(logt, x);
                        p2 = probabilityOfLessLikelySample(logt, x, tail);
                        LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                                  << ", log(p1) = " << std::log(p1)
                                  << ", log(p2) = " << std::log(p2));

                        BOOST_TEST_REQUIRE(
                            (std::fabs(p1 - p2) <= 0.01 * std::max(p1, p2) ||
                             std::fabs(std::log(p1) - std::log(p2)) <=
                                 0.05 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    }
                }
            }
        }
    }

    LOG_DEBUG(<< "******** gamma ********");

    {
        double shapes[] = {0.1, 1.0, 1.1, 100.0, 10000.0};
        double scales[] = {0.0001, 0.01, 1.0, 10.0};

        for (size_t i = 0; i < boost::size(shapes); ++i) {
            for (size_t j = 0; j < boost::size(scales); ++j) {
                LOG_DEBUG(<< "***** shape = " << shapes[i]
                          << ", scale = " << scales[j] << " *****");

                boost::math::gamma_distribution<> gamma(shapes[i], scales[j]);

                if (shapes[i] <= 1.0) {
                    double x = boost::math::mean(gamma);
                    tail = maths_t::E_UndeterminedTail;
                    for (int k = 0; k < 10; ++k) {
                        x *= 2.0;
                        p1 = CTools::safeCdfComplement(gamma, x);
                        p2 = probabilityOfLessLikelySample(gamma, x, tail);
                        LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 1e-3 * std::max(p1, p2));
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    }
                    continue;
                }

                double m1 = boost::math::mode(gamma);
                LOG_DEBUG(<< "mode = " << m1);

                double offset = 1.0;
                for (int k = 0; k < 5; ++k) {
                    offset /= 2.0;

                    double x = (1.0 - offset) * m1;
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(gamma, x);
                    p2 = probabilityOfLessLikelySample(gamma, x, tail);
                    LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.06 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.01 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

                    double y = (1.0 + offset) * m1;
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(gamma, y);
                    p2 = probabilityOfLessLikelySample(gamma, y, tail);
                    LOG_TRACE(<< "y = " << y << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.06 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.01 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                }

                double factor = 1.0;
                for (int k = 0; k < 5; ++k) {
                    factor *= 2.0;

                    double x = m1 / factor;
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(gamma, x);
                    p2 = probabilityOfLessLikelySample(gamma, x, tail);
                    LOG_TRACE(<< "x = " << x << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.1 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.01 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);

                    double y = factor * m1;
                    tail = maths_t::E_UndeterminedTail;
                    p1 = numericalProbabilityOfLessLikelySample(gamma, y);
                    p2 = probabilityOfLessLikelySample(gamma, y, tail);
                    LOG_TRACE(<< "y = " << y << ", p1 = " << p1 << ", p2 = " << p2
                              << ", log(p1) = " << std::log(p1)
                              << ", log(p2) = " << std::log(p2));
                    BOOST_TEST_REQUIRE(
                        (std::fabs(p1 - p2) <= 0.1 * std::max(p1, p2) ||
                         std::fabs(std::log(p1) - std::log(p2)) <=
                             0.01 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                    BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                }
            }
        }
    }

    LOG_DEBUG(<< "******** beta ********");

    {
        double alphas[] = {0.01, 0.98, 1.0, 1.01, 1000.0};
        double betas[] = {0.01, 0.98, 1.0, 1.01, 1000.0};

        for (size_t i = 0; i < boost::size(alphas); ++i) {
            for (size_t j = 0; j < boost::size(betas); ++j) {
                LOG_DEBUG(<< "**** alpha = " << alphas[i]
                          << ", beta = " << betas[j] << " ****");

                boost::math::beta_distribution<> beta(alphas[i], betas[j]);

                if (alphas[i] == 1.0 && betas[j] == 1.0) {
                    // Constant.
                    for (int k = 0; k < 6; ++k) {
                        double x = static_cast<double>(k) / 5.0;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = 1.0;
                        p2 = probabilityOfLessLikelySample(beta, x, tail);
                        LOG_TRACE(<< "x = " << x << ", f(x) = " << CTools::safePdf(beta, x)
                                  << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_EQUAL(p1, p2);
                        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                    }
                } else if (alphas[i] <= 1.0 && betas[j] >= 1.0) {
                    // Monotone decreasing.
                    for (int k = 0; k < 6; ++k) {
                        double x = static_cast<double>(k) / 5.0;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = CTools::safeCdfComplement(beta, x);
                        p2 = probabilityOfLessLikelySample(beta, x, tail);
                        LOG_TRACE(<< "x = " << x << ", f(x) = " << CTools::safePdf(beta, x)
                                  << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 1e-3 * std::max(p1, p2));
                        BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                    }
                } else if (alphas[i] >= 1.0 && betas[j] <= 1.0) {
                    // Monotone increasing.
                    for (int k = 0; k < 6; ++k) {
                        double x = static_cast<double>(k) / 5.0;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = CTools::safeCdf(beta, x);
                        p2 = probabilityOfLessLikelySample(beta, x, tail);
                        LOG_TRACE(<< "x = " << x << ", f(x) = " << CTools::safePdf(beta, x)
                                  << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 1e-3 * std::max(p1, p2));
                        BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                    }
                } else {
                    double stationaryPoint = adapters::stationaryPoint(beta).first;
                    bool maximum = adapters::stationaryPoint(beta).second;
                    LOG_DEBUG(<< "stationary point = " << stationaryPoint);

                    double epsMinus = stationaryPoint;
                    double epsPlus = 1.0 - stationaryPoint;
                    for (int k = 0; k < 5; ++k) {
                        epsMinus /= 2.0;
                        double xMinus = stationaryPoint - epsMinus;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = numericalProbabilityOfLessLikelySample(beta, xMinus);
                        p2 = probabilityOfLessLikelySample(beta, xMinus, tail);
                        LOG_TRACE(<< "x- = " << xMinus << ", p1 = " << p1
                                  << ", p2 = " << p2 << ", log(p1) = " << log(p1)
                                  << ", log(p2) = " << std::log(p2));
                        BOOST_TEST_REQUIRE(
                            (std::fabs(p1 - p2) <= 0.05 * std::max(p1, p2) ||
                             std::fabs(std::log(p1) - std::log(p2)) <
                                 0.25 * std::fabs(std::min(std::log(p1), std::log(p2)))));
                        if (maximum)
                            BOOST_REQUIRE_EQUAL(maths_t::E_LeftTail, tail);
                        if (!maximum)
                            BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);

                        epsPlus /= 2.0;
                        double xPlus = stationaryPoint + epsPlus;
                        tail = maths_t::E_UndeterminedTail;
                        p1 = numericalProbabilityOfLessLikelySample(beta, xPlus);
                        p2 = probabilityOfLessLikelySample(beta, xPlus, tail);
                        LOG_TRACE(<< "x+ = " << xPlus << ", p1 = " << p1 << ", p2 = " << p2);
                        BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p2, 0.05 * std::max(p1, p2));
                        if (maximum)
                            BOOST_REQUIRE_EQUAL(maths_t::E_RightTail, tail);
                        if (!maximum)
                            BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
                    }
                }
            }
        }
    }
    // Special cases:
    {
        // x at maximum
        boost::math::beta_distribution<> beta(2.0, 8.0);
        double mode = adapters::stationaryPoint(beta).first;
        tail = maths_t::E_UndeterminedTail;
        p1 = 1.0;
        p2 = probabilityOfLessLikelySample(beta, mode, tail);
        LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
        BOOST_REQUIRE_EQUAL(p1, p2);
        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
    }
    {
        // x at minimum
        boost::math::beta_distribution<> beta(0.6, 0.3);
        double mode = adapters::stationaryPoint(beta).first;
        tail = maths_t::E_UndeterminedTail;
        p1 = 0.0;
        p2 = probabilityOfLessLikelySample(beta, mode, tail);
        LOG_DEBUG(<< "p1 = " << p1 << ", p2 = " << p2);
        BOOST_REQUIRE_EQUAL(p1, p2);
        BOOST_REQUIRE_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
    }
}

BOOST_AUTO_TEST_CASE(testIntervalExpectation) {
    // We check the expectations agree with numerical integration
    // and also some corner cases. Specifically, that we handle
    // +/- infinity correctly and the also the case that a and b
    // are very close.

    maths::CTools::SIntervalExpectation expectation;

    double expected;
    double actual;

    LOG_DEBUG(<< "*** Normal ***");
    {
        boost::math::normal_distribution<> normal(10.0, 5.0);
        expected = numericalIntervalExpectation(normal, 0.0, 12.0);
        actual = expectation(normal, 0.0, 12.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);

        expected = numericalIntervalExpectation(normal, -40.0, 13.0);
        actual = expectation(normal, boost::numeric::bounds<double>::lowest(), 13.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);

        expected = 7.0;
        actual = expectation(normal, 7.0, 7.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-12 * expected);

        expected = 8.1;
        actual = expectation(normal, 8.1,
                             8.1 * (1.0 + std::numeric_limits<double>::epsilon()));
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-12 * expected);
    }

    LOG_DEBUG(<< "*** Log-Normal ***");
    {
        boost::math::lognormal_distribution<> logNormal(1.5, 0.8);
        expected = numericalIntervalExpectation(logNormal, 0.5, 7.0);
        actual = expectation(logNormal, 0.5, 7.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);

        expected = numericalIntervalExpectation(logNormal, 0.0, 9.0);
        actual = expectation(logNormal, boost::numeric::bounds<double>::lowest(), 9.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);

        expected = 6.0;
        actual = expectation(logNormal, 6.0, 6.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-12 * expected);

        expected = 8.1;
        actual = expectation(logNormal, 8.1,
                             8.1 * (1.0 + std::numeric_limits<double>::epsilon()));
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-12 * expected);
    }

    LOG_DEBUG(<< "*** Gamma ***");
    {
        boost::math::gamma_distribution<> gamma(5.0, 3.0);
        expected = numericalIntervalExpectation(gamma, 0.5, 4.0);
        actual = expectation(gamma, 0.5, 4.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);

        expected = numericalIntervalExpectation(gamma, 0.0, 5.0);
        actual = expectation(gamma, boost::numeric::bounds<double>::lowest(), 5.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);

        expected = 6.0;
        actual = expectation(gamma, 6.0, 6.0);
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-12 * expected);

        expected = 8.1;
        actual = expectation(gamma, 8.1,
                             8.1 * (1.0 + std::numeric_limits<double>::epsilon()));
        LOG_DEBUG(<< "expected = " << expected << ", actual = " << actual);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-12 * expected);
    }
}

BOOST_AUTO_TEST_CASE(testMixtureProbabilityOfLessLikelySample) {
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    std::size_t n = 500;

    TDoubleVec x;
    rng.generateUniformSamples(0.0, 1000.0, 20, x);
    TDoubleVec means;
    rng.generateUniformSamples(200.0, 800.0, n, means);
    TDoubleVec sd;
    rng.generateUniformSamples(0.1, 50.0, n, sd);
    TDoubleVec weights;
    rng.generateUniformSamples(1.0, 10.0, n, weights);

    for (std::size_t i = 4u; i <= 20; i += 4) {
        LOG_DEBUG(<< "*** modes = " << i << " ***");

        TMeanAccumulator meanError;
        TMeanAccumulator meanLogError;
        for (std::size_t j = 0u; j < n - i; j += i) {
            TDoubleVec modeWeights;
            std::vector<boost::math::normal> modes;
            double a = std::numeric_limits<double>::max();
            double b = -std::numeric_limits<double>::max();
            for (std::size_t k = 0u; k < i; ++k) {
                modeWeights.push_back(weights[j + k]);
                modes.push_back(boost::math::normal(means[j + k], sd[j + k]));
                a = std::min(a, means[j + k]);
                b = std::max(b, means[j + k]);
            }

            maths::CMixtureDistribution<boost::math::normal> mixture(modeWeights, modes);
            for (std::size_t k = 0u; k < x.size(); ++k) {
                double logFx = maths::pdf(mixture, x[k]);
                if (logFx == 0.0) {
                    logFx = 10.0 * core::constants::LOG_MIN_DOUBLE;
                } else {
                    logFx = std::log(logFx);
                }

                maths::CTools::CMixtureProbabilityOfLessLikelySample calculator(
                    i, x[k], logFx, a, b);
                for (std::size_t l = 0u; l < modeWeights.size(); ++l) {
                    calculator.addMode((mixture.weights())[l],
                                       boost::math::mean(modes[l]),
                                       boost::math::standard_deviation(modes[l]));
                }

                double pTails = 0.0;

                CLogPdf<boost::math::normal> logPdf(mixture);
                maths::CEqualWithTolerance<double> equal(
                    maths::CToleranceTypes::E_AbsoluteTolerance, 0.5);
                double xleft;
                BOOST_TEST_REQUIRE(calculator.leftTail(logPdf, 10, equal, xleft));
                pTails += maths::cdf(mixture, xleft);
                double xright;
                BOOST_TEST_REQUIRE(calculator.rightTail(logPdf, 10, equal, xright));
                pTails += maths::cdfComplement(mixture, xright);

                double p = pTails + calculator.calculate(logPdf, pTails);

                double pExpected = pTails;
                CTruncatedPdf<boost::math::normal> pdf(mixture, std::exp(logFx));
                for (double xi = a, l = 0, step = 0.5 * (b - a) / std::floor(b - a);
                     l < 2 * static_cast<std::size_t>(b - a); xi += step, ++l) {
                    double pi;
                    maths::CIntegration::gaussLegendre<maths::CIntegration::OrderThree>(
                        pdf, xi, xi + step, pi);
                    pExpected += pi;
                }

                LOG_TRACE(<< "pTails = " << pTails);
                LOG_TRACE(<< "x = " << x[k] << ", log(f(x)) = " << logFx
                          << ", P(x) = " << p << ", expected P(x) = " << pExpected);

                BOOST_TEST_REQUIRE(pExpected > 0.0);
                if (pExpected > 0.1) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(pExpected, p, 0.12);
                } else if (pExpected > 1e-10) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(std::log(pExpected), std::log(p),
                                                 0.15 * std::fabs(std::log(pExpected)));
                } else {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(std::log(pExpected), std::log(p),
                                                 0.015 * std::fabs(std::log(pExpected)));
                }
                meanError.add(std::fabs(p - pExpected));
                meanLogError.add(std::fabs(std::log(p) - std::log(pExpected)) /
                                 std::max(std::fabs(std::log(pExpected)),
                                          std::fabs(std::log(p))));
            }
        }

        LOG_DEBUG(<< "meanError    = " << maths::CBasicStatistics::mean(meanError));
        LOG_DEBUG(<< "meanLogError = " << maths::CBasicStatistics::mean(meanLogError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.005);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanLogError) < 0.03);
    }
}

BOOST_AUTO_TEST_CASE(testAnomalyScore) {
    // Test p = inverseAnomalyScore(anomalyScore(p))

    double p = 0.04;
    for (std::size_t i = 0u; i < 305; ++i, p *= 0.1) {
        double anomalyScore = CTools::anomalyScore(p);
        LOG_DEBUG(<< "p = " << p << ", anomalyScore = " << anomalyScore);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(p, CTools::inverseAnomalyScore(anomalyScore), 1e-3 * p);
    }
}

BOOST_AUTO_TEST_CASE(testSpread) {
    double period = 86400.0;
    {
        double raw[] = {15.0,    120.0,   4500.0,  9000.0, 25700.0,
                        43100.0, 73000.0, 74000.0, 84300.0};
        double separation = 20.0;
        TDoubleVec points(std::begin(raw), std::end(raw));
        std::string expected = core::CContainerPrinter::print(points);
        CTools::spread(0.0, period, separation, points);
        BOOST_REQUIRE_EQUAL(expected, core::CContainerPrinter::print(points));
        separation = 200.0;
        expected = "[0, 200, 4500, 9000, 25700, 43100, 73000, 74000, 84300]";
        CTools::spread(0.0, period, separation, points);
        LOG_DEBUG(<< "spread = " << core::CContainerPrinter::print(points));
        BOOST_REQUIRE_EQUAL(expected, core::CContainerPrinter::print(points));
    }
    {
        double raw[] = {150.0,   170.0,   4500.0,  4650.0,  4700.0,  4800.0,
                        73000.0, 73150.0, 73500.0, 73600.0, 73800.0, 74000.0};
        double separation = 126.0;
        std::string expected = "[97, 223, 4473.5, 4599.5, 4725.5, 4851.5, 73000, 73150, 73487, 73613, 73800, 74000]";
        TDoubleVec points(std::begin(raw), std::end(raw));
        CTools::spread(0.0, period, separation, points);
        LOG_DEBUG(<< "spread = " << core::CContainerPrinter::print(points));
        BOOST_REQUIRE_EQUAL(expected, core::CContainerPrinter::print(points));
    }
    {
        CRandomNumbers rng;
        for (std::size_t i = 0u; i < 100; ++i) {
            TDoubleVec origSamples;
            rng.generateUniformSamples(1000.0, static_cast<double>(period) - 1000.0,
                                       100, origSamples);
            TDoubleVec samples(origSamples);
            CTools::spread(0.0, period, 150.0, samples);

            std::sort(origSamples.begin(), origSamples.end());
            double eps = 1e-3;
            double dcost = (samples[0] + eps - origSamples[0]) *
                               (samples[0] + eps - origSamples[0]) -
                           (samples[0] - eps - origSamples[0]) *
                               (samples[0] - eps - origSamples[0]);
            for (std::size_t j = 1u; j < samples.size(); ++j) {
                BOOST_TEST_REQUIRE(samples[j] - samples[j - 1] >= 150.0 - eps);
                dcost += (samples[j] + eps - origSamples[j]) *
                             (samples[j] + eps - origSamples[j]) -
                         (samples[j] - eps - origSamples[j]) *
                             (samples[j] - eps - origSamples[j]);
            }
            dcost /= 2.0 * eps;
            LOG_DEBUG(<< "d(cost)/dx = " << dcost);
            BOOST_TEST_REQUIRE(std::fabs(dcost) < 2e-6);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFastLog) {
    test::CRandomNumbers rng;

    // Small
    {
        TDoubleVec x;
        rng.generateUniformSamples(-100.0, 0.0, 10000, x);
        for (std::size_t i = 0u; i < x.size(); ++i) {
            LOG_TRACE(<< "x = " << std::exp(x[i]) << ", log(x) = " << x[i]
                      << ", fast log(x) = " << maths::CTools::fastLog(std::exp(x[i])));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(x[i], maths::CTools::fastLog(std::exp(x[i])), 5e-5);
        }
    }
    // Mid
    {
        TDoubleVec x;
        rng.generateUniformSamples(1.0, 1e6, 10000, x);
        for (std::size_t i = 0u; i < x.size(); ++i) {
            LOG_TRACE(<< "x = " << x[i] << ", log(x) = " << std::log(x[i])
                      << ", fast log(x) = " << maths::CTools::fastLog(x[i]));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(std::log(x[i]),
                                         maths::CTools::fastLog(x[i]), 5e-5);
        }
    }
    // Large
    {
        TDoubleVec x;
        rng.generateUniformSamples(20.0, 80.0, 10000, x);
        for (std::size_t i = 0u; i < x.size(); ++i) {
            LOG_TRACE(<< "x = " << std::exp(x[i]) << ", log(x) = " << x[i]
                      << ", fast log(x) = " << maths::CTools::fastLog(std::exp(x[i])));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(x[i], maths::CTools::fastLog(std::exp(x[i])), 5e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMiscellaneous) {
    double x_[] = {0.0, 3.2, 2.1, -1.8, 4.5};

    maths::CVectorNx1<double, 5> x(x_, x_ + 5);
    maths::CVectorNx1<double, 5> a(-2.0);
    maths::CVectorNx1<double, 5> b(5.0);

    double expected[][5] = {
        {0.0, 3.2, 2.1, -1.8, 4.5}, {0.0, 3.2, 2.1, -1.5, 4.5},
        {0.0, 3.2, 2.1, -1.0, 4.0}, {0.0, 3.2, 2.1, -0.5, 3.5},
        {0.0, 3.0, 2.1, 0.0, 3.0},  {0.5, 2.5, 2.1, 0.5, 2.5},
        {1.0, 2.0, 2.0, 1.0, 2.0},  {1.5, 1.5, 1.5, 1.5, 1.5}};

    for (std::size_t i = 0u; a <= b; ++i) {
        maths::CVectorNx1<double, 5> expect(expected[i]);
        maths::CVectorNx1<double, 5> actual = maths::CTools::truncate(x, a, b);

        BOOST_REQUIRE_EQUAL(expect, actual);

        a = a + maths::CVectorNx1<double, 5>(0.5);
        b = b - maths::CVectorNx1<double, 5>(0.5);
    }
}

BOOST_AUTO_TEST_CASE(testLgamma) {
    std::array<double, 8> testData = {{3.5, 0.125, -0.125, 0.000244140625, 1.3552527156068805e-20,
                                       4.95547e+25, 5.01753e+25, 8.64197e+25}};

    std::array<double, 8> expectedData = {
        {1.2009736023470742248160218814507129957702389154682,
         2.0194183575537963453202905211670995899482809521344,
         2.1653002489051702517540619481440174064962195287626,
         8.3176252939431805089043336920440196990966796875000,
         45.7477139169563926657247066032141447067260742187500,
         2.882355039447984e+27, 2.919076782442754e+27, 5.074673490557339e+27}};

    for (std::size_t i = 0u; i < testData.size(); ++i) {
        double actual;
        double expected = expectedData[i];
        BOOST_TEST_REQUIRE(maths::CTools::lgamma(testData[i], actual, true));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, actual, 1e-5 * expected);
    }

    double result;
    BOOST_TEST_REQUIRE(maths::CTools::lgamma(0, result, false));
    BOOST_REQUIRE_EQUAL(result, std::numeric_limits<double>::infinity());

    BOOST_TEST_REQUIRE((maths::CTools::lgamma(0, result, true) == false));
    BOOST_REQUIRE_EQUAL(result, std::numeric_limits<double>::infinity());

    BOOST_TEST_REQUIRE((maths::CTools::lgamma(-1, result, false)));
    BOOST_REQUIRE_EQUAL(result, std::numeric_limits<double>::infinity());

    BOOST_TEST_REQUIRE((maths::CTools::lgamma(-1, result, true) == false));
    BOOST_REQUIRE_EQUAL(result, std::numeric_limits<double>::infinity());

    BOOST_TEST_REQUIRE(
        (maths::CTools::lgamma(std::numeric_limits<double>::max() - 1, result, false)));
    BOOST_REQUIRE_EQUAL(result, std::numeric_limits<double>::infinity());

    BOOST_TEST_REQUIRE((maths::CTools::lgamma(std::numeric_limits<double>::max() - 1,
                                              result, true) == false));
    BOOST_REQUIRE_EQUAL(result, std::numeric_limits<double>::infinity());
}

BOOST_AUTO_TEST_CASE(testSoftMax) {
    // Test some invariants and that std::vector and maths::CDenseVector versions agree.

    using TDoubleVector = maths::CDenseVector<double>;

    test::CRandomNumbers rng;

    TDoubleVec z;
    for (std::size_t t = 0; t < 100; ++t) {

        rng.generateUniformSamples(-3.0, 3.0, 5, z);
        TDoubleVec p{CTools::softmax(z)};

        BOOST_REQUIRE_CLOSE(1.0, std::accumulate(p.begin(), p.end(), 0.0), 1e-6);
        BOOST_TEST_REQUIRE(*std::min_element(p.begin(), p.end()) >= 0.0);

        TDoubleVector p_{CTools::softmax(TDoubleVector::fromStdVector(z))};
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_CLOSE(p[i], p_[i], 1e-6);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
