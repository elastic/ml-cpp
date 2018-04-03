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

#include "CRadialBasisFunctionTest.h"

#include <core/CLogger.h>

#include <maths/CCompositeFunctions.h>
#include <maths/CIntegration.h>
#include <maths/CRadialBasisFunction.h>

#include <boost/range.hpp>

using namespace ml;

namespace
{

class CValueAdaptor
{
    public:
        using result_type = double;

    public:
        CValueAdaptor(const maths::CRadialBasisFunction &function,
                      double centre,
                      double scale) :
            m_Function(&function),
            m_Centre(centre),
            m_Scale(scale)
        {
        }

        bool operator()(double x, double &result) const
        {
            result = m_Function->value(x, m_Centre, m_Scale);
            return true;
        }

    private:
        const maths::CRadialBasisFunction *m_Function;
        double m_Centre;
        double m_Scale;
};

class CSquareDerivativeAdaptor
{
    public:
        CSquareDerivativeAdaptor(const maths::CRadialBasisFunction &function,
                                 double centre,
                                 double scale) :
            m_Function(&function),
            m_Centre(centre),
            m_Scale(scale)
        {
        }

        bool operator()(double x, double &result) const
        {
            double d = m_Function->derivative(x, m_Centre, m_Scale);
            result = d * d;
            return true;
        }

    private:
        const maths::CRadialBasisFunction *m_Function;
        double m_Centre;
        double m_Scale;
};

}

void CRadialBasisFunctionTest::testDerivative(void)
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CRadialBasisFunctionTest::testDerivative  |");
    LOG_DEBUG("+--------------------------------------------+");

    const double a = 0.0;
    const double b = 10.0;
    const double centres[] = { 0.0, 5.0, 10.0 };
    const double scales[] = { 5.0, 1.0, 0.1 };

    const double eps = 1e-3;

    LOG_DEBUG("*** Gaussian ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(scales); ++j)
        {
            LOG_DEBUG("centre = " << centres[i] << ", scale = " << scales[j]);

            maths::CGaussianBasisFunction gaussian;
            for (std::size_t k = 0u; k < 10; ++k)
            {
                double x = a + static_cast<double>(k) / 10.0 * (b - a);
                double d = gaussian.derivative(x, centres[i], scales[j]);
                double e = (gaussian.value(x + eps, centres[i], scales[j])
                            - gaussian.value(x - eps, centres[i], scales[j]))
                           / 2.0 / eps;

                // Centred difference nuemrical derivative should
                // be accurate to o(eps^2).
                CPPUNIT_ASSERT_DOUBLES_EQUAL(d, e, eps * eps);
            }
        }
    }

    LOG_DEBUG("*** Inverse Quadratic ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(scales); ++j)
        {
            LOG_DEBUG("centre = " << centres[i] << ", scale = " << scales[j]);

            maths::CInverseQuadraticBasisFunction inverseQuadratic;
            for (std::size_t k = 0u; k < 10; ++k)
            {
                double x = a + static_cast<double>(k) / 10.0 * (b - a);
                double d = inverseQuadratic.derivative(x, centres[i], scales[j]);
                double e = (inverseQuadratic.value(x + eps, centres[i], scales[j])
                            - inverseQuadratic.value(x - eps, centres[i], scales[j]))
                           / 2.0 / eps;

                // Centred difference nuemrical derivative should
                // be accurate to o(eps^2).
                CPPUNIT_ASSERT_DOUBLES_EQUAL(d, e, eps * eps);
            }
        }
    }

}

void CRadialBasisFunctionTest::testMean(void)
{
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CRadialBasisFunctionTest::testMean  |");
    LOG_DEBUG("+--------------------------------------+");

    const double a = 0.0;
    const double b = 10.0;
    const double centres[] = { 0.0, 5.0, 10.0 };
    const double scales[] = { 5.0, 1.0, 0.1 };

    const double eps = 1e-3;

    LOG_DEBUG("*** Gaussian ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(scales); ++j)
        {
            LOG_DEBUG("centre = " << centres[i] << ", scale = " << scales[j]);

            maths::CGaussianBasisFunction gaussian;
            CValueAdaptor f(gaussian, centres[i], scales[j]);
            double expectedMean = 0.0;
            for (std::size_t k = 0u; k < 20; ++k)
            {
                double aa = a + static_cast<double>(k) / 20.0 * (b - a);
                double bb = a + static_cast<double>(k + 1) / 20.0 * (b - a);
                double interval;
                maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(f, aa, bb, interval);
                expectedMean += interval;
            }
            expectedMean /= (b - a);

            double mean = gaussian.mean(a, b, centres[i], scales[j]);
            LOG_DEBUG("expectedMean = " << expectedMean
                      << ", mean = " << mean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, mean, eps * mean);
        }
    }

    LOG_DEBUG("*** Inverse Quadratic ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(scales); ++j)
        {
            LOG_DEBUG("centre = " << centres[i] << ", scale = " << scales[j]);

            maths::CInverseQuadraticBasisFunction inverseQuadratic;
            CValueAdaptor f(inverseQuadratic, centres[i], scales[j]);
            double expectedMean = 0.0;
            for (std::size_t k = 0u; k < 20; ++k)
            {
                double aa = a + static_cast<double>(k) / 20.0 * (b - a);
                double bb = a + static_cast<double>(k + 1) / 20.0 * (b - a);
                double interval;
                maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(f, aa, bb, interval);
                expectedMean += interval;
            }
            expectedMean /= (b - a);

            double mean = inverseQuadratic.mean(a, b, centres[i], scales[j]);
            LOG_DEBUG("expectedMean = " << expectedMean
                      << ", mean = " << mean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, mean, eps * mean);
        }
    }
}

void CRadialBasisFunctionTest::testMeanSquareDerivative(void)
{
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CRadialBasisFunctionTest::testMean  |");
    LOG_DEBUG("+--------------------------------------+");

    const double a = 0.0;
    const double b = 10.0;
    const double centres[] = { 0.0, 5.0, 10.0 };
    const double scales[] = { 5.0, 1.0, 0.1 };

    const double eps = 1e-3;

    LOG_DEBUG("*** Gaussian ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(scales); ++j)
        {
            LOG_DEBUG("centre = " << centres[i] << ", scale = " << scales[j]);

            maths::CGaussianBasisFunction gaussian;
            CSquareDerivativeAdaptor f(gaussian, centres[i], scales[j]);
            double expectedMean = 0.0;
            for (std::size_t k = 0u; k < 50; ++k)
            {
                double aa = a + static_cast<double>(k) / 50.0 * (b - a);
                double bb = a + static_cast<double>(k + 1) / 50.0 * (b - a);
                double interval;
                maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(f, aa, bb, interval);
                expectedMean += interval;
            }
            expectedMean /= (b - a);

            double mean = gaussian.meanSquareDerivative(a, b, centres[i], scales[j]);
            LOG_DEBUG("expectedMean = " << expectedMean
                      << ", mean = " << mean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, mean, eps * mean);
        }
    }

    LOG_DEBUG("*** Inverse Quadratic ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(scales); ++j)
        {
            LOG_DEBUG("centre = " << centres[i] << ", scale = " << scales[j]);

            maths::CInverseQuadraticBasisFunction inverseQuadratic;
            CSquareDerivativeAdaptor f(inverseQuadratic, centres[i], scales[j]);
            double expectedMean = 0.0;
            for (std::size_t k = 0u; k < 50; ++k)
            {
                double aa = a + static_cast<double>(k) / 50.0 * (b - a);
                double bb = a + static_cast<double>(k + 1) / 50.0 * (b - a);
                double interval;
                maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(f, aa, bb, interval);
                expectedMean += interval;
            }
            expectedMean /= (b - a);

            double mean = inverseQuadratic.meanSquareDerivative(a, b, centres[i], scales[j]);
            LOG_DEBUG("expectedMean = " << expectedMean
                      << ", mean = " << mean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, mean, eps * mean);
        }
    }
}

void CRadialBasisFunctionTest::testProduct(void)
{
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CRadialBasisFunctionTest::testProduct  |");
    LOG_DEBUG("+-----------------------------------------+");

    const double a = 0.0;
    const double b = 10.0;
    const double centres[] = { 0.0, 5.0, 10.0 };
    const double scales[] = { 5.0, 1.0, 0.1 };

    const double eps = 1e-3;

    LOG_DEBUG("*** Gaussian ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(centres); ++j)
        {
            for (std::size_t k = 0u; k < boost::size(scales); ++k)
            {
                for (std::size_t l = 0u; l < boost::size(scales); ++l)
                {
                    LOG_DEBUG("centre1 = " << centres[i]
                              << ", centre2 = " << centres[j]
                              << ", scale1 = " << scales[k]
                              << ", scale2 = " << scales[l]);

                    maths::CGaussianBasisFunction gaussian;
                    CValueAdaptor f1(gaussian, centres[i], scales[k]);
                    CValueAdaptor f2(gaussian, centres[j], scales[l]);
                    maths::CCompositeFunctions::CProduct<CValueAdaptor,
                                                         CValueAdaptor> f(f1, f2);
                    double expectedProduct = 0.0;
                    for (std::size_t m = 0u; m < 50; ++m)
                    {
                        double aa = a + static_cast<double>(m) / 50.0 * (b - a);
                        double bb = a + static_cast<double>(m + 1) / 50.0 * (b - a);
                        double interval;
                        maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(f, aa, bb, interval);
                        expectedProduct += interval;
                    }
                    expectedProduct /= (b - a);

                    double product = gaussian.product(a, b,
                                                      centres[i],
                                                      centres[j],
                                                      scales[k],
                                                      scales[l]);
                    LOG_DEBUG("expectedMean = " << expectedProduct
                              << ", mean = " << product);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProduct, product, eps * product);
                }
            }
        }
    }

    LOG_DEBUG("*** Inverse Quadratic ***");

    for (std::size_t i = 0u; i < boost::size(centres); ++i)
    {
        for (std::size_t j = 0u; j < boost::size(centres); ++j)
        {
            for (std::size_t k = 0u; k < boost::size(scales); ++k)
            {
                for (std::size_t l = 0u; l < boost::size(scales); ++l)
                {
                    LOG_DEBUG("centre1 = " << centres[i]
                              << ", centre2 = " << centres[j]
                              << ", scale1 = " << scales[k]
                              << ", scale2 = " << scales[l]);

                    maths::CInverseQuadraticBasisFunction inverseQuadratic;
                    CValueAdaptor f1(inverseQuadratic, centres[i], scales[k]);
                    CValueAdaptor f2(inverseQuadratic, centres[j], scales[l]);
                    double expectedProduct = 0.0;
                    maths::CCompositeFunctions::CProduct<CValueAdaptor,
                                                         CValueAdaptor> f(f1, f2);
                    for (std::size_t m = 0u; m < 50; ++m)
                    {
                        double aa = a + static_cast<double>(m) / 50.0 * (b - a);
                        double bb = a + static_cast<double>(m + 1) / 50.0 * (b - a);
                        double interval;
                        maths::CIntegration::gaussLegendre<maths::CIntegration::OrderFive>(f, aa, bb, interval);
                        expectedProduct += interval;
                    }
                    expectedProduct /= (b - a);

                    double product = inverseQuadratic.product(a, b,
                                                              centres[i],
                                                              centres[j],
                                                              scales[k],
                                                              scales[l]);
                    LOG_DEBUG("expectedProduct = " << expectedProduct
                              << ", product = " << product);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProduct, product, eps * product);
                }
            }
        }
    }
}

CppUnit::Test *CRadialBasisFunctionTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CRadialBasisFunctionTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CRadialBasisFunctionTest>(
                                   "CRadialBasisFunctionTest::testDerivative",
                                   &CRadialBasisFunctionTest::testDerivative) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CRadialBasisFunctionTest>(
                                   "CRadialBasisFunctionTest::testMean",
                                   &CRadialBasisFunctionTest::testMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CRadialBasisFunctionTest>(
                                   "CRadialBasisFunctionTest::testMeanSquareDerivative",
                                   &CRadialBasisFunctionTest::testMeanSquareDerivative) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CRadialBasisFunctionTest>(
                                   "CRadialBasisFunctionTest::testProduct",
                                   &CRadialBasisFunctionTest::testProduct) );

    return suiteOfTests;
}
