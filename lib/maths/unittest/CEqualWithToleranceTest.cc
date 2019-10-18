/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CLinearAlgebra.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CEqualWithToleranceTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testScalar) {
    {
        maths::CEqualWithTolerance<double> abs(
            maths::CToleranceTypes::E_AbsoluteTolerance, 0.31);
        maths::CEqualWithTolerance<double> rel(
            maths::CToleranceTypes::E_RelativeTolerance, 0.01);
        maths::CEqualWithTolerance<double> absAndRel(
            maths::CToleranceTypes::E_AbsoluteTolerance & maths::CToleranceTypes::E_RelativeTolerance,
            0.31, 0.01);
        maths::CEqualWithTolerance<double> absOrRel(
            maths::CToleranceTypes::E_AbsoluteTolerance | maths::CToleranceTypes::E_RelativeTolerance,
            0.31, 0.01);
        {
            double a = 1.1;
            double b = 1.4;
            double c = 200.6;
            double d = 202.61;
            BOOST_CHECK_EQUAL(true, abs(a, b));
            BOOST_CHECK_EQUAL(false, abs(c, d));
            BOOST_CHECK_EQUAL(false, rel(a, b));
            BOOST_CHECK_EQUAL(true, rel(c, d));
            BOOST_CHECK_EQUAL(false, absAndRel(a, b));
            BOOST_CHECK_EQUAL(false, absAndRel(c, d));
            BOOST_CHECK_EQUAL(true, absOrRel(a, b));
            BOOST_CHECK_EQUAL(true, absOrRel(c, d));
        }
        {
            double a = -1.1;
            double b = -1.4;
            double c = -200.6;
            double d = -202.61;
            BOOST_CHECK_EQUAL(true, abs(a, b));
            BOOST_CHECK_EQUAL(false, abs(c, d));
            BOOST_CHECK_EQUAL(false, rel(a, b));
            BOOST_CHECK_EQUAL(true, rel(c, d));
            BOOST_CHECK_EQUAL(false, absAndRel(a, b));
            BOOST_CHECK_EQUAL(false, absAndRel(c, d));
            BOOST_CHECK_EQUAL(true, absOrRel(a, b));
            BOOST_CHECK_EQUAL(true, absOrRel(c, d));
        }
    }
    {
        maths::CEqualWithTolerance<float> abs(maths::CToleranceTypes::E_AbsoluteTolerance, 0.31f);
        maths::CEqualWithTolerance<float> rel(maths::CToleranceTypes::E_RelativeTolerance, 0.01f);
        maths::CEqualWithTolerance<float> absAndRel(
            maths::CToleranceTypes::E_AbsoluteTolerance & maths::CToleranceTypes::E_RelativeTolerance,
            0.31f, 0.01f);
        maths::CEqualWithTolerance<float> absOrRel(
            maths::CToleranceTypes::E_AbsoluteTolerance | maths::CToleranceTypes::E_RelativeTolerance,
            0.31f, 0.01f);

        float a = 1.1f;
        float b = 1.4f;
        float c = 200.6f;
        float d = 202.61f;
        BOOST_CHECK_EQUAL(true, abs(a, b));
        BOOST_CHECK_EQUAL(false, abs(c, d));
        BOOST_CHECK_EQUAL(false, rel(a, b));
        BOOST_CHECK_EQUAL(true, rel(c, d));
        BOOST_CHECK_EQUAL(false, absAndRel(a, b));
        BOOST_CHECK_EQUAL(false, absAndRel(c, d));
        BOOST_CHECK_EQUAL(true, absOrRel(a, b));
        BOOST_CHECK_EQUAL(true, absOrRel(c, d));
    }
}

BOOST_AUTO_TEST_CASE(testVector) {
    float a_[] = {1.1f, 1.2f};
    float b_[] = {1.2f, 1.3f};
    float c_[] = {201.1f, 202.2f};
    float d_[] = {202.1f, 203.2f};

    maths::CVector<double> epsAbs(2, 0.15 / std::sqrt(2.0));
    maths::CVector<double> epsRel(2, 0.0062 / std::sqrt(2.0));

    maths::CEqualWithTolerance<maths::CVector<double>> abs(
        maths::CToleranceTypes::E_AbsoluteTolerance, epsAbs);
    maths::CEqualWithTolerance<maths::CVector<double>> rel(
        maths::CToleranceTypes::E_RelativeTolerance, epsRel);
    maths::CEqualWithTolerance<maths::CVector<double>> absAndRel(
        maths::CToleranceTypes::E_AbsoluteTolerance & maths::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);
    maths::CEqualWithTolerance<maths::CVector<double>> absOrRel(
        maths::CToleranceTypes::E_AbsoluteTolerance | maths::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);

    {
        maths::CVector<double> a(a_, a_ + 2);
        maths::CVector<double> b(b_, b_ + 2);
        maths::CVector<double> c(c_, c_ + 2);
        maths::CVector<double> d(d_, d_ + 2);
        BOOST_CHECK_EQUAL(true, abs(a, b));
        BOOST_CHECK_EQUAL(false, abs(c, d));
        BOOST_CHECK_EQUAL(false, rel(a, b));
        BOOST_CHECK_EQUAL(true, rel(c, d));
        BOOST_CHECK_EQUAL(false, absAndRel(a, b));
        BOOST_CHECK_EQUAL(false, absAndRel(c, d));
        BOOST_CHECK_EQUAL(true, absOrRel(a, b));
        BOOST_CHECK_EQUAL(true, absOrRel(c, d));
    }
    {
        maths::CVector<double> a(a_, a_ + 2);
        maths::CVector<double> b(b_, b_ + 2);
        maths::CVector<double> c(c_, c_ + 2);
        maths::CVector<double> d(d_, d_ + 2);
        BOOST_CHECK_EQUAL(true, abs(-a, -b));
        BOOST_CHECK_EQUAL(false, abs(-c, -d));
        BOOST_CHECK_EQUAL(false, rel(-a, -b));
        BOOST_CHECK_EQUAL(true, rel(-c, -d));
        BOOST_CHECK_EQUAL(false, absAndRel(-a, -b));
        BOOST_CHECK_EQUAL(false, absAndRel(-c, -d));
        BOOST_CHECK_EQUAL(true, absOrRel(-a, -b));
        BOOST_CHECK_EQUAL(true, absOrRel(-c, -d));
    }
    {
        maths::CVector<float> a(a_, a_ + 2);
        maths::CVector<float> b(b_, b_ + 2);
        maths::CVector<float> c(c_, c_ + 2);
        maths::CVector<float> d(d_, d_ + 2);
        BOOST_CHECK_EQUAL(true, abs(a, b));
        BOOST_CHECK_EQUAL(false, abs(c, d));
        BOOST_CHECK_EQUAL(false, rel(a, b));
        BOOST_CHECK_EQUAL(true, rel(c, d));
        BOOST_CHECK_EQUAL(false, absAndRel(a, b));
        BOOST_CHECK_EQUAL(false, absAndRel(c, d));
        BOOST_CHECK_EQUAL(true, absOrRel(a, b));
        BOOST_CHECK_EQUAL(true, absOrRel(c, d));
    }
}

BOOST_AUTO_TEST_CASE(testMatrix) {
    float a_[] = {1.1f, 1.2f, 1.3f};
    float b_[] = {1.2f, 1.3f, 1.4f};
    float c_[] = {201.1f, 202.2f, 203.4f};
    float d_[] = {202.1f, 203.2f, 204.4f};

    maths::CSymmetricMatrix<double> epsAbs(2, 0.21 / 2.0);
    maths::CSymmetricMatrix<double> epsRel(2, 0.005 / 2.0);

    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> abs(
        maths::CToleranceTypes::E_AbsoluteTolerance, epsAbs);
    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> rel(
        maths::CToleranceTypes::E_RelativeTolerance, epsRel);
    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> absAndRel(
        maths::CToleranceTypes::E_AbsoluteTolerance & maths::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);
    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> absOrRel(
        maths::CToleranceTypes::E_AbsoluteTolerance | maths::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);

    {
        maths::CSymmetricMatrix<double> a(a_, a_ + 3);
        maths::CSymmetricMatrix<double> b(b_, b_ + 3);
        maths::CSymmetricMatrix<double> c(c_, c_ + 3);
        maths::CSymmetricMatrix<double> d(d_, d_ + 3);
        BOOST_CHECK_EQUAL(true, abs(a, b));
        BOOST_CHECK_EQUAL(false, abs(c, d));
        BOOST_CHECK_EQUAL(false, rel(a, b));
        BOOST_CHECK_EQUAL(true, rel(c, d));
        BOOST_CHECK_EQUAL(false, absAndRel(a, b));
        BOOST_CHECK_EQUAL(false, absAndRel(c, d));
        BOOST_CHECK_EQUAL(true, absOrRel(a, b));
        BOOST_CHECK_EQUAL(true, absOrRel(c, d));
    }
    {
        maths::CSymmetricMatrix<double> a(a_, a_ + 3);
        maths::CSymmetricMatrix<double> b(b_, b_ + 3);
        maths::CSymmetricMatrix<double> c(c_, c_ + 3);
        maths::CSymmetricMatrix<double> d(d_, d_ + 3);
        BOOST_CHECK_EQUAL(true, abs(-a, -b));
        BOOST_CHECK_EQUAL(false, abs(-c, -d));
        BOOST_CHECK_EQUAL(false, rel(-a, -b));
        BOOST_CHECK_EQUAL(true, rel(-c, -d));
        BOOST_CHECK_EQUAL(false, absAndRel(-a, -b));
        BOOST_CHECK_EQUAL(false, absAndRel(-c, -d));
        BOOST_CHECK_EQUAL(true, absOrRel(a, b));
        BOOST_CHECK_EQUAL(true, absOrRel(c, d));
    }
    {
        maths::CSymmetricMatrix<float> a(a_, a_ + 3);
        maths::CSymmetricMatrix<float> b(b_, b_ + 3);
        maths::CSymmetricMatrix<float> c(c_, c_ + 3);
        maths::CSymmetricMatrix<float> d(d_, d_ + 3);
        BOOST_CHECK_EQUAL(true, abs(a, b));
        BOOST_CHECK_EQUAL(false, abs(c, d));
        BOOST_CHECK_EQUAL(false, rel(a, b));
        BOOST_CHECK_EQUAL(true, rel(c, d));
        BOOST_CHECK_EQUAL(false, absAndRel(a, b));
        BOOST_CHECK_EQUAL(false, absAndRel(c, d));
        BOOST_CHECK_EQUAL(true, absOrRel(a, b));
        BOOST_CHECK_EQUAL(true, absOrRel(c, d));
    }
}

BOOST_AUTO_TEST_SUITE_END()
