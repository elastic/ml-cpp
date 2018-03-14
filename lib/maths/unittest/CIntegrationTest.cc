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

#include "CIntegrationTest.h"

#include <core/CLogger.h>

#include <maths/CCompositeFunctions.h>
#include <maths/CIntegration.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>

#include <test/CRandomNumbers.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <functional>
#include <numeric>

#include <math.h>

using namespace ml;
using namespace maths;

namespace {

template<unsigned int ORDER>
class CPolynomialFunction : public std::unary_function<double, double> {
    public:
        CPolynomialFunction(const double (&coefficients)[ORDER + 1]) {
            std::copy(coefficients,
                      coefficients + ORDER + 1,
                      m_Coefficients);
        }

        bool operator()(const double &x, double &result) const {
            result = 0.0;
            for (unsigned int i = 0u; i < ORDER + 1; ++i) {
                result += m_Coefficients[i] * ::pow(x, static_cast<double>(i));
            }
            return true;
        }

        const double &coefficient(unsigned int i) const {
            return m_Coefficients[i];
        }

    private:
        double m_Coefficients[ORDER + 1];
};

std::ostream &operator<<(std::ostream &o, const CPolynomialFunction<0u> &f) {
    return o << f.coefficient(0);
}

template<unsigned int ORDER>
std::ostream &operator<<(std::ostream &o, const CPolynomialFunction<ORDER> &f) {
    o << f.coefficient(0) << " + ";
    for (unsigned int i = 1u; i < ORDER; ++i) {
        o << f.coefficient(i) << "x^" << i << " + ";
    }
    if (ORDER > 0) {
        o << f.coefficient(ORDER) << "x^" << ORDER;
    }
    return o;
}

template<unsigned int ORDER>
double integrate(const CPolynomialFunction<ORDER> &f,
                 const double &a,
                 const double &b) {
    double result = 0.0;
    for (unsigned int i = 0; i < ORDER + 1; ++i) {
        double n = static_cast<double>(i) + 1.0;
        result += f.coefficient(i) / n * (::pow(b, n) - ::pow(a, n));
    }
    return result;
}

template<unsigned int DIMENSION>
class CMultivariatePolynomialFunction {
    public:
        typedef CVectorNx1<double, DIMENSION> TVector;

        struct SMonomial {
            bool operator<(const SMonomial &rhs) const {
                return std::accumulate(s_Powers, s_Powers + DIMENSION, 0.0)
                       < std::accumulate(rhs.s_Powers, rhs.s_Powers + DIMENSION, 0.0);
            }
            double s_Coefficient;
            double s_Powers[DIMENSION];
        };

        typedef std::vector<SMonomial> TMonomialVec;

    public:
        void add(double coefficient, double powers[DIMENSION]) {
            m_Terms.push_back(SMonomial());
            m_Terms.back().s_Coefficient = coefficient;
            std::copy(powers, powers + DIMENSION, m_Terms.back().s_Powers);
        }

        void finalize(void) {
            std::sort(m_Terms.begin(), m_Terms.end());
        }

        bool operator()(const TVector &x, double &result) const {
            result = 0.0;
            for (std::size_t i = 0u; i < m_Terms.size(); ++i) {
                const SMonomial &monomial = m_Terms[i];
                double          term = monomial.s_Coefficient;
                for (unsigned int j = 0u; j < DIMENSION; ++j) {
                    if (monomial.s_Powers[j] > 0.0) {
                        term *= ::pow(x(j), monomial.s_Powers[j]);
                    }
                }
                result += term;
            }
            return true;
        }

        const TMonomialVec &terms(void) const {
            return m_Terms;
        }

    private:
        TMonomialVec       m_Terms;
};

template<unsigned int DIMENSION>
std::ostream &operator<<(std::ostream &o, const CMultivariatePolynomialFunction<DIMENSION> &f) {
    if (!f.terms().empty()) {
        o << (f.terms())[0].s_Coefficient;
        for (unsigned int j = 0u; j < DIMENSION; ++j) {
            if ((f.terms())[0].s_Powers[j] > 0.0) {
                o << ".x" << j << "^" << (f.terms())[0].s_Powers[j];
            }
        }
    }
    for (std::size_t i = 1u; i < f.terms().size(); ++i) {
        o << " + " << (f.terms())[i].s_Coefficient;
        for (unsigned int j = 0u; j < DIMENSION; ++j) {
            if ((f.terms())[i].s_Powers[j] > 0.0) {
                o << ".x" << j << "^" << (f.terms())[i].s_Powers[j];
            }
        }
    }
    return o;
}

typedef std::vector<double> TDoubleVec;

template<unsigned int DIMENSION>
double integrate(const CMultivariatePolynomialFunction<DIMENSION> &f,
                 const TDoubleVec &a,
                 const TDoubleVec &b) {
    double result = 0.0;
    for (std::size_t i = 0u; i < f.terms().size(); ++i) {
        double term = (f.terms())[i].s_Coefficient;
        for (unsigned int j = 0; j < DIMENSION; ++j) {
            double n = (f.terms())[i].s_Powers[j] + 1.0;
            term *= (::pow(b[j], n) - ::pow(a[j], n)) / n;
        }
        result += term;
    }
    return result;
}


typedef std::vector<TDoubleVec> TDoubleVecVec;

bool readGrid(const std::string &file,
              TDoubleVec &weights,
              TDoubleVecVec &points) {
    typedef std::vector<std::string> TStrVec;
    std::ifstream d2_l1;
    d2_l1.open(file.c_str());
    if (!d2_l1) {
        LOG_ERROR("Bad file: " << file);
        return false;
    }

    std::string line;
    while (std::getline(d2_l1, line)) {
        TStrVec     point;
        std::string weight;
        core::CStringUtils::tokenise(", ", line, point, weight);
        core::CStringUtils::trimWhitespace(weight);

        points.push_back(TDoubleVec());
        for (std::size_t i = 0u; i < point.size(); ++i) {
            core::CStringUtils::trimWhitespace(point[i]);
            double xi;
            if (!core::CStringUtils::stringToType(point[i], xi)) {
                LOG_ERROR("Bad point: " << core::CContainerPrinter::print(point));
                return false;
            }
            points.back().push_back(xi);
        }

        double w;
        if (!core::CStringUtils::stringToType(weight, w)) {
            LOG_ERROR("Bad weight: " << weight);
            return false;
        }
        weights.push_back(w);
    }

    return true;
}

class CSmoothHeavySide {
    public:
        typedef double result_type;

    public:
        CSmoothHeavySide(double slope, double offset) :
            m_Slope(slope),
            m_Offset(offset) {
        }

        bool operator()(double x, double &result) const {
            result =    ::exp(m_Slope * (x - m_Offset))
                     / (::exp(m_Slope * (x - m_Offset)) + 1.0);
            return true;
        }

    private:
        double m_Slope;
        double m_Offset;
};

class CNormal {
    public:
        typedef double result_type;

    public:
        CNormal(double mean, double std) :
            m_Mean(mean),
            m_Std(std) {
        }

        bool operator()(double x, double &result) const {
            if (m_Std <= 0.0) {
                return false;
            }
            boost::math::normal_distribution<> normal(m_Mean, m_Std);
            result = boost::math::pdf(normal, x);
            return true;
        }

    private:
        double m_Mean;
        double m_Std;
};

}

void CIntegrationTest::testAllSingleVariate(void) {
    LOG_DEBUG("+-------------------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testAllSingleVariate  |");
    LOG_DEBUG("+-------------------------------------------+");

    // Test that "low" order polynomials are integrated exactly
    // (as they should be for a higher order quadrature).

    typedef CPolynomialFunction<0u> TConstant;
    typedef CPolynomialFunction<1u> TLinear;
    typedef CPolynomialFunction<2u> TQuadratic;
    typedef CPolynomialFunction<3u> TCubic;
    typedef CPolynomialFunction<4u> TQuartic;
    typedef CPolynomialFunction<5u> TQuintic;
    typedef CPolynomialFunction<6u> THexic;
    typedef CPolynomialFunction<7u> THeptic;
    typedef CPolynomialFunction<8u> TOctic;
    typedef CPolynomialFunction<9u> TNonic;
    typedef CPolynomialFunction<10u> TDecic;

    static const double EPS = 1e-6;

    double ranges[][2] =
    {
        { -3.0, -1.0 },
        { -1.0,  5.0 },
        {  0.0,  8.0 }
    };

    {
        double coeffs[][1] =
        {
            { -3.2 },
            {  0.0 },
            {  1.0 },
            {  5.0 },
            { 12.1 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TConstant f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderOne>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTwo>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderThree>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFour>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFive>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][2] =
        {
            { -3.2, -1.2 },
            {  0.0,  2.0 },
            {  1.0, -1.0 },
            {  5.0,  6.4 },
            { 12.1, -8.3 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TLinear f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderOne>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTwo>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderThree>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFour>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFive>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][3] =
        {
            { -3.2, -1.2, -3.5 },
            {  0.1,  2.0,  4.6 },
            {  1.0, -1.0,  1.0 },
            {  5.0,  6.4, -4.1 },
            { 12.1, -8.3,  10.1 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TQuadratic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTwo>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderThree>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFour>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFive>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][4] =
        {
            { -1.2, -1.9, -3.0, -3.2 },
            {  0.4,  2.0,  4.6,  2.3 },
            {  1.0, -1.0,  1.0, -1.0 },
            {  4.0,  2.4, -8.1, -2.1 },
            { 10.1, -6.3,  1.1,  8.3 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TCubic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderThree>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFour>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFive>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][5] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2 },
            { 20.4,  2.0,  4.6,  2.3,  0.7 },
            {  1.0, -1.0,  1.0, -1.0,  1.0 },
            {  4.0,  2.4, -8.1, -2.1,  1.4 },
            { 10.1, -6.3,  1.1,  8.3, -5.1 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TQuartic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFour>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFive>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][6] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2, -1.1 },
            { 20.4,  6.0,  2.6,  0.3,  0.7,  2.3 },
            {  1.0, -1.0,  1.0, -1.0,  1.0, -1.0 },
            {  3.0,  2.4, -8.1, -2.1,  1.4, -3.1 },
            { 10.1, -5.3,  2.1,  4.3, -7.1,  0.4 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TQuintic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderFive>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][7] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2, -1.1, -0.1 },
            { 20.4,  6.0,  2.6,  0.3,  0.7,  2.3,  1.0 },
            {  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0 },
            {  3.0,  2.4, -8.1, -2.1,  1.4, -3.1,  2.1 },
            { 10.1, -5.3,  2.1,  4.3, -7.1,  0.4, -0.5 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                THexic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSix>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][8] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2, -1.1, -0.1, -2.0 },
            { 20.4,  6.0,  2.6,  0.3,  0.7,  2.3,  1.0,  3.0 },
            {  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0 },
            {  3.0,  2.4, -8.1, -2.1,  1.4, -3.1,  2.1, -2.1 },
            { 10.1, -5.3,  2.1,  4.3, -7.1,  0.4, -0.5,  0.3 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                THeptic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderSeven>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][9] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2, -1.1, -0.1, -2.0, -0.1 },
            { 20.4,  6.0,  2.6,  0.3,  0.7,  2.3,  1.0,  3.0, 10.0 },
            {  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0 },
            {  3.0,  2.4, -8.1, -2.1,  1.4, -3.1,  2.1, -2.1, -1.0 },
            { 10.1, -5.3,  2.1,  4.3, -7.1,  0.4, -0.5,  0.3,  0.3 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TOctic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderEight>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][10] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2, -1.1, -0.1, -2.0, -0.1, -3.4 },
            { 20.4,  6.0,  2.6,  0.3,  0.7,  2.3,  1.0,  3.0, 10.0,  1.3 },
            {  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0 },
            {  3.0,  2.4, -8.1, -2.1,  1.4, -3.1,  2.1, -2.1, -1.0,  1.1 },
            { 10.1, -5.3,  2.1,  4.3, -7.1,  0.4, -0.5,  0.3,  0.3, -5.0 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TNonic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderNine>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }

    {
        double coeffs[][11] =
        {
            { -1.1, -0.9, -4.0, -1.2, -0.2, -1.1, -0.1, -2.0, -0.1, -3.4, -0.9 },
            { 20.4,  6.0,  2.6,  0.3,  0.7,  2.3,  1.0,  3.0, 10.0,  1.3,  2.0 },
            {  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0 },
            {  3.0,  2.4, -8.1, -2.1,  1.4, -3.1,  2.1, -2.1, -1.0,  1.1,  3.1 },
            { 10.1, -5.3,  2.1,  4.3, -7.1,  0.4, -0.5,  0.3,  0.3, -5.0, -0.1 }
        };

        for (unsigned int i = 0; i < sizeof(ranges)/sizeof(ranges[0]); ++i) {
            for (unsigned int j = 0; j < sizeof(coeffs)/sizeof(coeffs[0]); ++j) {
                TDecic f(coeffs[j]);
                LOG_DEBUG("range = [" << ranges[i][0] << "," << ranges[i][1] << "]"
                                      << ", f(x) = " << f);

                double expected = integrate(f, ranges[i][0], ranges[i][1]);

                double actual;

                CPPUNIT_ASSERT(CIntegration::gaussLegendre<CIntegration::OrderTen>(
                                   f, ranges[i][0], ranges[i][1], actual));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, EPS);
            }
        }
    }
}



void CIntegrationTest::testAdaptive(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testAdaptive  |");
    LOG_DEBUG("+-----------------------------------+");

    typedef std::pair<double, double> TDoubleDoublePr;
    typedef std::vector<TDoubleDoublePr> TDoubleDoublePrVec;

    {
        LOG_DEBUG("*** Smooth unit step at 20 ***");

        CSmoothHeavySide heavySide(10.0, 20.0);

        TDoubleDoublePr intervals_[] =
        {
            TDoubleDoublePr(0.0,  10.0),
            TDoubleDoublePr(10.0, 20.0),
            TDoubleDoublePr(20.0, 30.0),
            TDoubleDoublePr(30.0, 40.0)
        };
        TDoubleDoublePrVec intervals(boost::begin(intervals_),
                                     boost::end(intervals_));
        TDoubleVec fIntervals(intervals.size());
        for (std::size_t i = 0u; i < intervals.size(); ++i) {
            CIntegration::gaussLegendre<CIntegration::OrderThree>(heavySide,
                                                                  intervals[i].first,
                                                                  intervals[i].second,
                                                                  fIntervals[i]);
        }

        double result = 0.0;
        CIntegration::adaptiveGaussLegendre<CIntegration::OrderThree>(heavySide,
                                                                      intervals,
                                                                      fIntervals,
                                                                      3, 5, 0.01,
                                                                      result);
        LOG_DEBUG("expectedResult = 20.0");
        LOG_DEBUG("result = " << result);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, result, 0.01 * 20.0);
    }

    {
        LOG_DEBUG("*** N(21, 3) ***");
        CNormal normal(21.0, 3.0);

        double expectedResult = 0.0;
        for (std::size_t i = 0u; i < 400; ++i) {
            double fi;
            CIntegration::gaussLegendre<CIntegration::OrderThree>(normal,
                                                                  0.1 * static_cast<double>(i),
                                                                  0.1 * static_cast<double>(i+1),
                                                                  fi);
            expectedResult += fi;
        }

        TDoubleDoublePr intervals_[] =
        {
            TDoubleDoublePr(0.0,  10.0),
            TDoubleDoublePr(10.0, 20.0),
            TDoubleDoublePr(20.0, 30.0),
            TDoubleDoublePr(30.0, 40.0)
        };
        TDoubleDoublePrVec intervals(boost::begin(intervals_),
                                     boost::end(intervals_));
        TDoubleVec fIntervals(intervals.size());
        for (std::size_t i = 0u; i < intervals.size(); ++i) {
            CIntegration::gaussLegendre<CIntegration::OrderThree>(normal,
                                                                  intervals[i].first,
                                                                  intervals[i].second,
                                                                  fIntervals[i]);
        }

        double result = 0.0;
        CIntegration::adaptiveGaussLegendre<CIntegration::OrderThree>(normal,
                                                                      intervals,
                                                                      fIntervals,
                                                                      3, 5, 0.0001,
                                                                      result);
        LOG_DEBUG("expectedResult = " << expectedResult);
        LOG_DEBUG("result = " << result);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedResult, result, 0.0001 * expectedResult);
    }

    {
        LOG_DEBUG("*** \"Smooth unit step at 20\" x N(21, 3) ***");

        CSmoothHeavySide                                         heavySide(4.0, 20.0);
        CNormal                                                  normal(21.0, 4.0);
        CCompositeFunctions::CProduct<CSmoothHeavySide, CNormal> f(heavySide, normal);

        double expectedResult = 0.0;
        for (std::size_t i = 0u; i < 400; ++i) {
            double fi;
            CIntegration::gaussLegendre<CIntegration::OrderThree>(f,
                                                                  0.1 * static_cast<double>(i),
                                                                  0.1 * static_cast<double>(i+1),
                                                                  fi);
            expectedResult += fi;
        }

        TDoubleDoublePr intervals_[] =
        {
            TDoubleDoublePr(0.0,  20.0),
            TDoubleDoublePr(20.0, 40.0)
        };
        TDoubleDoublePrVec intervals(boost::begin(intervals_),
                                     boost::end(intervals_));
        TDoubleVec fIntervals(intervals.size());
        for (std::size_t i = 0u; i < intervals.size(); ++i) {
            CIntegration::gaussLegendre<CIntegration::OrderThree>(f,
                                                                  intervals[i].first,
                                                                  intervals[i].second,
                                                                  fIntervals[i]);
        }

        double result = 0.0;
        CIntegration::adaptiveGaussLegendre<CIntegration::OrderThree>(f,
                                                                      intervals,
                                                                      fIntervals,
                                                                      3, 5, 0.0001,
                                                                      result);
        LOG_DEBUG("expectedResult = " << expectedResult);
        LOG_DEBUG("result = " << result);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedResult, result, 0.0001 * expectedResult);
    }
}

void CIntegrationTest::testSparseGrid(void) {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testSparseGrid  |");
    LOG_DEBUG("+-------------------------------------+");

    // Compare against known grid characteristics. These are available
    // at http://www.sparse-grids.de/#Nodes.

    {
        LOG_DEBUG("*** 2D, order 1 ***");

        TDoubleVec    expectedWeights;
        TDoubleVecVec expectedPoints;
        CPPUNIT_ASSERT(readGrid("testfiles/sparse_guass_quadrature_test_d2_l1",
                                expectedWeights,
                                expectedPoints));

        typedef CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderOne,
                                                             CIntegration::TwoDimensions> Sparse2do1;

        const Sparse2do1 &sparse = Sparse2do1::instance();

        LOG_DEBUG("# points = " << sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedWeights.size(), sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedPoints.size(), sparse.points().size());

        for (std::size_t i = 0u; i < expectedWeights.size(); ++i) {
            LOG_DEBUG("weight = " << (sparse.weights())[i]);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedWeights[i],
                                         (sparse.weights())[i] / 4.0, 1e-6);

            LOG_DEBUG("point = " << (sparse.points())[i]);
            for (std::size_t j = 0u; j < expectedPoints[i].size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedPoints[i][j],
                                             0.5 + (sparse.points())[i](j) / 2.0, 1e-6);
            }
        }
    }

    {
        LOG_DEBUG("*** 2D, order 2 ***");

        TDoubleVec    expectedWeights;
        TDoubleVecVec expectedPoints;
        CPPUNIT_ASSERT(readGrid("testfiles/sparse_guass_quadrature_test_d2_l2",
                                expectedWeights,
                                expectedPoints));

        typedef CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderTwo,
                                                             CIntegration::TwoDimensions> Sparse2do2;

        const Sparse2do2 &sparse = Sparse2do2::instance();

        LOG_DEBUG("# points = " << sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedWeights.size(), sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedPoints.size(), sparse.points().size());

        for (std::size_t i = 0u; i < expectedWeights.size(); ++i) {
            LOG_DEBUG("weight = " << (sparse.weights())[i]);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedWeights[i],
                                         (sparse.weights())[i] / 4.0, 1e-6);

            LOG_DEBUG("point = " << (sparse.points())[i]);
            for (std::size_t j = 0u; j < expectedPoints[i].size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedPoints[i][j],
                                             0.5 + (sparse.points())[i](j) / 2.0, 1e-6);
            }
        }
    }

    {
        LOG_DEBUG("*** 2D, order 4 ***");

        TDoubleVec    expectedWeights;
        TDoubleVecVec expectedPoints;
        CPPUNIT_ASSERT(readGrid("testfiles/sparse_guass_quadrature_test_d2_l4",
                                expectedWeights,
                                expectedPoints));

        typedef CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderFour,
                                                             CIntegration::TwoDimensions> Sparse2do4;

        const Sparse2do4 &sparse = Sparse2do4::instance();

        LOG_DEBUG("# points = " << sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedWeights.size(), sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedPoints.size(), sparse.points().size());

        for (std::size_t i = 0u; i < expectedWeights.size(); ++i) {
            LOG_DEBUG("weight = " << (sparse.weights())[i]);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedWeights[i],
                                         (sparse.weights())[i] / 4.0, 1e-6);

            LOG_DEBUG("point = " << (sparse.points())[i]);
            for (std::size_t j = 0u; j < expectedPoints[i].size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedPoints[i][j],
                                             0.5 + (sparse.points())[i](j) / 2.0, 1e-6);
            }
        }
    }

    {
        LOG_DEBUG("*** 7D, order 3 ***");

        TDoubleVec    expectedWeights;
        TDoubleVecVec expectedPoints;
        CPPUNIT_ASSERT(readGrid("testfiles/sparse_guass_quadrature_test_d7_l3",
                                expectedWeights,
                                expectedPoints));

        typedef CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderThree,
                                                             CIntegration::SevenDimensions> Sparse7do3;

        const Sparse7do3 &sparse = Sparse7do3::instance();

        LOG_DEBUG("# points = " << sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedWeights.size(), sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedPoints.size(), sparse.points().size());

        for (std::size_t i = 0u; i < expectedWeights.size(); ++i) {
            LOG_DEBUG("weight = " << (sparse.weights())[i]);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedWeights[i],
                                         (sparse.weights())[i] / ::pow(2.0, 7.0), 1e-6);

            LOG_DEBUG("point = " << (sparse.points())[i]);
            for (std::size_t j = 0u; j < expectedPoints[i].size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedPoints[i][j],
                                             0.5 + (sparse.points())[i](j) / 2.0, 1e-6);
            }
        }
    }

    {
        LOG_DEBUG("*** 7D, order 5 ***");

        TDoubleVec    expectedWeights;
        TDoubleVecVec expectedPoints;
        CPPUNIT_ASSERT(readGrid("testfiles/sparse_guass_quadrature_test_d7_l5",
                                expectedWeights,
                                expectedPoints));

        typedef CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderFive,
                                                             CIntegration::SevenDimensions> Sparse7do5;

        const Sparse7do5 &sparse = Sparse7do5::instance();

        LOG_DEBUG("# points = " << sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedWeights.size(), sparse.weights().size());
        CPPUNIT_ASSERT_EQUAL(expectedPoints.size(), sparse.points().size());

        for (std::size_t i = 0u; i < expectedWeights.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG("weight = " << (sparse.weights())[i]);
            }
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedWeights[i],
                                         (sparse.weights())[i] / ::pow(2.0, 7.0), 1e-6);

            if (i % 10 == 0) {
                LOG_DEBUG("point = " << (sparse.points())[i]);
            }
            for (std::size_t j = 0u; j < expectedPoints[i].size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedPoints[i][j],
                                             0.5 + (sparse.points())[i](j) / 2.0, 1e-6);
            }
        }
    }

    unsigned int dimensions[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    unsigned int order[] = { 1, 2, 3, 4, 5 };

    std::size_t expectedNumberPoints[][5] =
    {
        { 1, 2,  3,   4,    5    },
        { 1, 5,  13,  29,   53   },
        { 1, 7,  25,  69,   165  },
        { 1, 9,  41,  137,  385  },
        { 1, 11, 61,  241,  781  },
        { 1, 13, 85,  389,  1433 },
        { 1, 15, 113, 589,  2437 },
        { 1, 17, 145, 849,  3905 },
        { 1, 19, 181, 1177, 5965 },
        { 1, 21, 221, 1581, 8761 }
    };

    for (std::size_t i = 0u; i < boost::size(dimensions); ++i) {
        LOG_DEBUG("DIMENSION = " << dimensions[i]);

#define NUMBER_POINTS(dimension, n)                                                                           \
    switch (order[j])                                                                                 \
    {                                                                                                 \
        case 1:  n = CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderOne,                 \
                                                                  dimension>::instance().points().size(); \
            break;                                                                                   \
        case 2:  n = CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderTwo,                 \
                                                                  dimension>::instance().points().size(); \
            break;                                                                                   \
        case 3:  n = CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderThree,               \
                                                                  dimension>::instance().points().size(); \
            break;                                                                                   \
        case 4:  n = CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderFour,                \
                                                                  dimension>::instance().points().size(); \
            break;                                                                                   \
        case 5:  n = CIntegration::CSparseGaussLegendreQuadrature<CIntegration::OrderFive,                \
                                                                  dimension>::instance().points().size(); \
            break;                                                                                   \
        default: n = 0;                                                                                   \
            break;                                                                                   \
    }
        for (std::size_t j = 0u; j < boost::size(order); ++j) {
            LOG_DEBUG("ORDER = " << order[j]);

            std::size_t numberPoints = 0u;
            switch (dimensions[i]) {
                case 1:  NUMBER_POINTS(CIntegration::OneDimension,    numberPoints); break;
                case 2:  NUMBER_POINTS(CIntegration::TwoDimensions,   numberPoints); break;
                case 3:  NUMBER_POINTS(CIntegration::ThreeDimensions, numberPoints); break;
                case 4:  NUMBER_POINTS(CIntegration::FourDimensions,  numberPoints); break;
                case 5:  NUMBER_POINTS(CIntegration::FiveDimensions,  numberPoints); break;
                case 6:  NUMBER_POINTS(CIntegration::SixDimensions,   numberPoints); break;
                case 7:  NUMBER_POINTS(CIntegration::SevenDimensions, numberPoints); break;
                case 8:  NUMBER_POINTS(CIntegration::EightDimensions, numberPoints); break;
                case 9:  NUMBER_POINTS(CIntegration::NineDimensions,  numberPoints); break;
                case 10: NUMBER_POINTS(CIntegration::TenDimensions,   numberPoints); break;
                default: numberPoints = 0u;                                          break;
            }
#undef NUMBER_POINTS

            LOG_DEBUG("number points: actual = " << numberPoints
                                                 << ", expected = " << expectedNumberPoints[i][j]);
            CPPUNIT_ASSERT_EQUAL(expectedNumberPoints[i][j], numberPoints);
        }
    }
}

void CIntegrationTest::testMultivariateSmooth(void) {
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testMultivariateSmooth  |");
    LOG_DEBUG("+---------------------------------------------+");

    // Test that "low" order polynomials are integrated exactly.
    // A sparse grid of order l will integrate a polynomial exactly
    // if its order is no more than 2l - 1. A polynomial order is
    // the largest sum of exponents in any term.

    typedef std::vector<std::size_t> TSizeVec;

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("*** Dimension = 2 ***");

        static const std::size_t DIMENSION = 2u;

        for (std::size_t l = 2; l < 5; ++l) {
            LOG_DEBUG("ORDER = " << 1 + l);

            for (std::size_t t = 0u; t < 20; ++t) {
                std::size_t n = 3u;

                TSizeVec coefficients;
                rng.generateUniformSamples(1, 50, n, coefficients);

                TSizeVec powers;
                rng.generateUniformSamples(0, l, DIMENSION * n, powers);

                CMultivariatePolynomialFunction<DIMENSION> polynomial;
                for (std::size_t i = 0u; i < n; ++i) {
                    double c = static_cast<double>(coefficients[i]);
                    double p[] =
                    {
                        static_cast<double>(powers[DIMENSION*i + 0]),
                        static_cast<double>(powers[DIMENSION*i + 1])
                    };
                    if (std::accumulate(p, p + DIMENSION, 0.0) > (2.0 * static_cast<double>(l) - 1.0)) {
                        continue;
                    }
                    polynomial.add(c, p);
                }

                TDoubleVec limits;
                rng.generateUniformSamples(-10.0, 10.0, 2 * DIMENSION, limits);

                std::sort(limits.begin(), limits.end());
                TDoubleVec a(limits.begin(), limits.begin() + DIMENSION);
                TDoubleVec b(limits.begin() + DIMENSION, limits.end());

                LOG_DEBUG("Testing " << polynomial);
                LOG_DEBUG("a = " << core::CContainerPrinter::print(a));
                LOG_DEBUG("b = " << core::CContainerPrinter::print(b));

                double expected = integrate(polynomial, a, b);

                double actual = 0.0;
                bool   successful = false;
                switch (l) {
                    case 2:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderTwo,
                                                                       CIntegration::TwoDimensions>(polynomial, a, b, actual);
                        break;
                    case 3:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderThree,
                                                                       CIntegration::TwoDimensions>(polynomial, a, b, actual);
                        break;
                    case 4:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderFour,
                                                                       CIntegration::TwoDimensions>(polynomial, a, b, actual);
                        break;
                    case 5:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderFive,
                                                                       CIntegration::TwoDimensions>(polynomial, a, b, actual);
                        break;
                    default:
                        break;
                }

                LOG_DEBUG("expected = " << expected);
                LOG_DEBUG("actual   = " << actual);
                CPPUNIT_ASSERT(successful);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-6);
            }
        }
    }

    {
        LOG_DEBUG("*** Dimension = 5 ***");

        static const std::size_t DIMENSION = 5u;

        for (std::size_t l = 2; l < 5; ++l) {
            LOG_DEBUG("ORDER = " << l);

            for (std::size_t t = 0u; t < 20; ++t) {
                std::size_t n = 10u;

                TSizeVec coefficients;
                rng.generateUniformSamples(1, 50, n, coefficients);

                TSizeVec powers;
                rng.generateUniformSamples(0, l, DIMENSION * n, powers);

                CMultivariatePolynomialFunction<DIMENSION> polynomial;
                for (std::size_t i = 0u; i < n; ++i) {
                    double c = static_cast<double>(coefficients[i]);
                    double p[] =
                    {
                        static_cast<double>(powers[5*i + 0]),
                        static_cast<double>(powers[5*i + 1]),
                        static_cast<double>(powers[5*i + 2]),
                        static_cast<double>(powers[5*i + 3]),
                        static_cast<double>(powers[5*i + 4])
                    };
                    if (std::accumulate(p, p + DIMENSION, 0.0) > (2.0 * static_cast<double>(l) - 1.0)) {
                        continue;
                    }
                    polynomial.add(c, p);
                }

                TDoubleVec limits;
                rng.generateUniformSamples(-10.0, 10.0, 2 * DIMENSION, limits);

                std::sort(limits.begin(), limits.end());
                TDoubleVec a(limits.begin(), limits.begin() + DIMENSION);
                TDoubleVec b(limits.begin() + DIMENSION, limits.end());

                LOG_DEBUG("Testing " << polynomial);
                LOG_DEBUG("a = " << core::CContainerPrinter::print(a));
                LOG_DEBUG("b = " << core::CContainerPrinter::print(b));

                double expected = integrate(polynomial, a, b);

                double actual = 0.0;
                bool   successful = false;
                switch (l) {
                    case 2:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderTwo,
                                                                       CIntegration::FiveDimensions>(polynomial, a, b, actual);
                        break;
                    case 3:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderThree,
                                                                       CIntegration::FiveDimensions>(polynomial, a, b, actual);
                        break;
                    case 4:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderFour,
                                                                       CIntegration::FiveDimensions>(polynomial, a, b, actual);
                        break;
                    case 5:
                        successful = CIntegration::sparseGaussLegendre<CIntegration::OrderFive,
                                                                       CIntegration::FiveDimensions>(polynomial, a, b, actual);
                        break;
                    default:
                        break;
                }

                LOG_DEBUG("expected = " << expected);
                LOG_DEBUG("actual   = " << actual);
                CPPUNIT_ASSERT(successful);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-2);
            }
        }
    }
}

CppUnit::Test *CIntegrationTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CIntegrationTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegrationTest>(
                               "CIntegrationTest::testAllSingleVariate",
                               &CIntegrationTest::testAllSingleVariate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegrationTest>(
                               "CIntegrationTest::testAdaptive",
                               &CIntegrationTest::testAdaptive) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegrationTest>(
                               "CIntegrationTest::testSparseGrid",
                               &CIntegrationTest::testSparseGrid) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegrationTest>(
                               "CIntegrationTest::testMultivariateSmooth",
                               &CIntegrationTest::testMultivariateSmooth) );

    return suiteOfTests;
}
