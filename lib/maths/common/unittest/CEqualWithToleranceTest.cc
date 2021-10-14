/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CLogger.h>

#include <maths/common/CEqualWithTolerance.h>
#include <maths/common/CLinearAlgebra.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CEqualWithToleranceTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testScalar) {
    {
        maths::common::CEqualWithTolerance<double> abs(
            maths::common::CToleranceTypes::E_AbsoluteTolerance, 0.31);
        maths::common::CEqualWithTolerance<double> rel(
            maths::common::CToleranceTypes::E_RelativeTolerance, 0.01);
        maths::common::CEqualWithTolerance<double> absAndRel(
            maths::common::CToleranceTypes::E_AbsoluteTolerance &
                maths::common::CToleranceTypes::E_RelativeTolerance,
            0.31, 0.01);
        maths::common::CEqualWithTolerance<double> absOrRel(
            maths::common::CToleranceTypes::E_AbsoluteTolerance |
                maths::common::CToleranceTypes::E_RelativeTolerance,
            0.31, 0.01);
        {
            double a = 1.1;
            double b = 1.4;
            double c = 200.6;
            double d = 202.61;
            BOOST_REQUIRE_EQUAL(true, abs(a, b));
            BOOST_REQUIRE_EQUAL(false, abs(c, d));
            BOOST_REQUIRE_EQUAL(false, rel(a, b));
            BOOST_REQUIRE_EQUAL(true, rel(c, d));
            BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
            BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
            BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
            BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
        }
        {
            double a = -1.1;
            double b = -1.4;
            double c = -200.6;
            double d = -202.61;
            BOOST_REQUIRE_EQUAL(true, abs(a, b));
            BOOST_REQUIRE_EQUAL(false, abs(c, d));
            BOOST_REQUIRE_EQUAL(false, rel(a, b));
            BOOST_REQUIRE_EQUAL(true, rel(c, d));
            BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
            BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
            BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
            BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
        }
    }
    {
        maths::common::CEqualWithTolerance<float> abs(
            maths::common::CToleranceTypes::E_AbsoluteTolerance, 0.31f);
        maths::common::CEqualWithTolerance<float> rel(
            maths::common::CToleranceTypes::E_RelativeTolerance, 0.01f);
        maths::common::CEqualWithTolerance<float> absAndRel(
            maths::common::CToleranceTypes::E_AbsoluteTolerance &
                maths::common::CToleranceTypes::E_RelativeTolerance,
            0.31f, 0.01f);
        maths::common::CEqualWithTolerance<float> absOrRel(
            maths::common::CToleranceTypes::E_AbsoluteTolerance |
                maths::common::CToleranceTypes::E_RelativeTolerance,
            0.31f, 0.01f);

        float a = 1.1f;
        float b = 1.4f;
        float c = 200.6f;
        float d = 202.61f;
        BOOST_REQUIRE_EQUAL(true, abs(a, b));
        BOOST_REQUIRE_EQUAL(false, abs(c, d));
        BOOST_REQUIRE_EQUAL(false, rel(a, b));
        BOOST_REQUIRE_EQUAL(true, rel(c, d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
    }
}

BOOST_AUTO_TEST_CASE(testVector) {
    float a_[] = {1.1f, 1.2f};
    float b_[] = {1.2f, 1.3f};
    float c_[] = {201.1f, 202.2f};
    float d_[] = {202.1f, 203.2f};

    maths::common::CVector<double> epsAbs(2, 0.15 / std::sqrt(2.0));
    maths::common::CVector<double> epsRel(2, 0.0062 / std::sqrt(2.0));

    maths::common::CEqualWithTolerance<maths::common::CVector<double>> abs(
        maths::common::CToleranceTypes::E_AbsoluteTolerance, epsAbs);
    maths::common::CEqualWithTolerance<maths::common::CVector<double>> rel(
        maths::common::CToleranceTypes::E_RelativeTolerance, epsRel);
    maths::common::CEqualWithTolerance<maths::common::CVector<double>> absAndRel(
        maths::common::CToleranceTypes::E_AbsoluteTolerance &
            maths::common::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);
    maths::common::CEqualWithTolerance<maths::common::CVector<double>> absOrRel(
        maths::common::CToleranceTypes::E_AbsoluteTolerance |
            maths::common::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);

    {
        maths::common::CVector<double> a(a_, a_ + 2);
        maths::common::CVector<double> b(b_, b_ + 2);
        maths::common::CVector<double> c(c_, c_ + 2);
        maths::common::CVector<double> d(d_, d_ + 2);
        BOOST_REQUIRE_EQUAL(true, abs(a, b));
        BOOST_REQUIRE_EQUAL(false, abs(c, d));
        BOOST_REQUIRE_EQUAL(false, rel(a, b));
        BOOST_REQUIRE_EQUAL(true, rel(c, d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
    }
    {
        maths::common::CVector<double> a(a_, a_ + 2);
        maths::common::CVector<double> b(b_, b_ + 2);
        maths::common::CVector<double> c(c_, c_ + 2);
        maths::common::CVector<double> d(d_, d_ + 2);
        BOOST_REQUIRE_EQUAL(true, abs(-a, -b));
        BOOST_REQUIRE_EQUAL(false, abs(-c, -d));
        BOOST_REQUIRE_EQUAL(false, rel(-a, -b));
        BOOST_REQUIRE_EQUAL(true, rel(-c, -d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(-a, -b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(-c, -d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(-a, -b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(-c, -d));
    }
    {
        maths::common::CVector<float> a(a_, a_ + 2);
        maths::common::CVector<float> b(b_, b_ + 2);
        maths::common::CVector<float> c(c_, c_ + 2);
        maths::common::CVector<float> d(d_, d_ + 2);
        BOOST_REQUIRE_EQUAL(true, abs(a, b));
        BOOST_REQUIRE_EQUAL(false, abs(c, d));
        BOOST_REQUIRE_EQUAL(false, rel(a, b));
        BOOST_REQUIRE_EQUAL(true, rel(c, d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
    }
}

BOOST_AUTO_TEST_CASE(testMatrix) {
    float a_[] = {1.1f, 1.2f, 1.3f};
    float b_[] = {1.2f, 1.3f, 1.4f};
    float c_[] = {201.1f, 202.2f, 203.4f};
    float d_[] = {202.1f, 203.2f, 204.4f};

    maths::common::CSymmetricMatrix<double> epsAbs(2, 0.21 / 2.0);
    maths::common::CSymmetricMatrix<double> epsRel(2, 0.005 / 2.0);

    maths::common::CEqualWithTolerance<maths::common::CSymmetricMatrix<double>> abs(
        maths::common::CToleranceTypes::E_AbsoluteTolerance, epsAbs);
    maths::common::CEqualWithTolerance<maths::common::CSymmetricMatrix<double>> rel(
        maths::common::CToleranceTypes::E_RelativeTolerance, epsRel);
    maths::common::CEqualWithTolerance<maths::common::CSymmetricMatrix<double>> absAndRel(
        maths::common::CToleranceTypes::E_AbsoluteTolerance &
            maths::common::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);
    maths::common::CEqualWithTolerance<maths::common::CSymmetricMatrix<double>> absOrRel(
        maths::common::CToleranceTypes::E_AbsoluteTolerance |
            maths::common::CToleranceTypes::E_RelativeTolerance,
        epsAbs, epsRel);

    {
        maths::common::CSymmetricMatrix<double> a(a_, a_ + 3);
        maths::common::CSymmetricMatrix<double> b(b_, b_ + 3);
        maths::common::CSymmetricMatrix<double> c(c_, c_ + 3);
        maths::common::CSymmetricMatrix<double> d(d_, d_ + 3);
        BOOST_REQUIRE_EQUAL(true, abs(a, b));
        BOOST_REQUIRE_EQUAL(false, abs(c, d));
        BOOST_REQUIRE_EQUAL(false, rel(a, b));
        BOOST_REQUIRE_EQUAL(true, rel(c, d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
    }
    {
        maths::common::CSymmetricMatrix<double> a(a_, a_ + 3);
        maths::common::CSymmetricMatrix<double> b(b_, b_ + 3);
        maths::common::CSymmetricMatrix<double> c(c_, c_ + 3);
        maths::common::CSymmetricMatrix<double> d(d_, d_ + 3);
        BOOST_REQUIRE_EQUAL(true, abs(-a, -b));
        BOOST_REQUIRE_EQUAL(false, abs(-c, -d));
        BOOST_REQUIRE_EQUAL(false, rel(-a, -b));
        BOOST_REQUIRE_EQUAL(true, rel(-c, -d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(-a, -b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(-c, -d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
    }
    {
        maths::common::CSymmetricMatrix<float> a(a_, a_ + 3);
        maths::common::CSymmetricMatrix<float> b(b_, b_ + 3);
        maths::common::CSymmetricMatrix<float> c(c_, c_ + 3);
        maths::common::CSymmetricMatrix<float> d(d_, d_ + 3);
        BOOST_REQUIRE_EQUAL(true, abs(a, b));
        BOOST_REQUIRE_EQUAL(false, abs(c, d));
        BOOST_REQUIRE_EQUAL(false, rel(a, b));
        BOOST_REQUIRE_EQUAL(true, rel(c, d));
        BOOST_REQUIRE_EQUAL(false, absAndRel(a, b));
        BOOST_REQUIRE_EQUAL(false, absAndRel(c, d));
        BOOST_REQUIRE_EQUAL(true, absOrRel(a, b));
        BOOST_REQUIRE_EQUAL(true, absOrRel(c, d));
    }
}

BOOST_AUTO_TEST_SUITE_END()
