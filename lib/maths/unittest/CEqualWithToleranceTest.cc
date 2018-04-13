/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CEqualWithToleranceTest.h"

#include <core/CLogger.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CLinearAlgebra.h>

using namespace ml;

void CEqualWithToleranceTest::testScalar()
{
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CEqualWithToleranceTest::testScalar  |");
    LOG_DEBUG("+---------------------------------------+");

    {
        maths::CEqualWithTolerance<double> abs(maths::CToleranceTypes::E_AbsoluteTolerance, 0.31);
        maths::CEqualWithTolerance<double> rel(maths::CToleranceTypes::E_RelativeTolerance, 0.01);
        maths::CEqualWithTolerance<double> absAndRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                     & maths::CToleranceTypes::E_RelativeTolerance,
                                                     0.31, 0.01);
        maths::CEqualWithTolerance<double> absOrRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                    | maths::CToleranceTypes::E_RelativeTolerance,
                                                    0.31, 0.01);
        {
            double a = 1.1;
            double b = 1.4;
            double c = 200.6;
            double d = 202.61;
            CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
            CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
            CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
            CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
            CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
            CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
            CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
            CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
        }
        {
            double a = -1.1;
            double b = -1.4;
            double c = -200.6;
            double d = -202.61;
            CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
            CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
            CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
            CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
            CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
            CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
            CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
            CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
        }
    }
    {
        maths::CEqualWithTolerance<float> abs(maths::CToleranceTypes::E_AbsoluteTolerance, 0.31f);
        maths::CEqualWithTolerance<float> rel(maths::CToleranceTypes::E_RelativeTolerance, 0.01f);
        maths::CEqualWithTolerance<float> absAndRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                    & maths::CToleranceTypes::E_RelativeTolerance,
                                                    0.31f, 0.01f);
        maths::CEqualWithTolerance<float> absOrRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                   | maths::CToleranceTypes::E_RelativeTolerance,
                                                   0.31f, 0.01f);

        float a = 1.1f;
        float b = 1.4f;
        float c = 200.6f;
        float d = 202.61f;
        CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
        CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
        CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
    }
}

void CEqualWithToleranceTest::testVector()
{
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CEqualWithToleranceTest::testVector  |");
    LOG_DEBUG("+---------------------------------------+");

    float a_[] = { 1.1f, 1.2f };
    float b_[] = { 1.2f, 1.3f };
    float c_[] = { 201.1f, 202.2f };
    float d_[] = { 202.1f, 203.2f };

    maths::CVector<double> epsAbs(2, 0.15 / std::sqrt(2.0));
    maths::CVector<double> epsRel(2, 0.0062 / std::sqrt(2.0));

    maths::CEqualWithTolerance<maths::CVector<double>> abs(maths::CToleranceTypes::E_AbsoluteTolerance, epsAbs);
    maths::CEqualWithTolerance<maths::CVector<double>> rel(maths::CToleranceTypes::E_RelativeTolerance, epsRel);
    maths::CEqualWithTolerance<maths::CVector<double>> absAndRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                                 & maths::CToleranceTypes::E_RelativeTolerance,
                                                                 epsAbs, epsRel);
    maths::CEqualWithTolerance<maths::CVector<double>> absOrRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                                | maths::CToleranceTypes::E_RelativeTolerance,
                                                                epsAbs, epsRel);

    {
        maths::CVector<double> a(a_, a_ + 2);
        maths::CVector<double> b(b_, b_ + 2);
        maths::CVector<double> c(c_, c_ + 2);
        maths::CVector<double> d(d_, d_ + 2);
        CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
        CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
        CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
    }
    {
        maths::CVector<double> a(a_, a_ + 2);
        maths::CVector<double> b(b_, b_ + 2);
        maths::CVector<double> c(c_, c_ + 2);
        maths::CVector<double> d(d_, d_ + 2);
        CPPUNIT_ASSERT_EQUAL(true,  abs(-a, -b));
        CPPUNIT_ASSERT_EQUAL(false, abs(-c, -d));
        CPPUNIT_ASSERT_EQUAL(false, rel(-a, -b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(-c, -d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(-a, -b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(-c, -d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(-a, -b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(-c, -d));
    }
    {
        maths::CVector<float> a(a_, a_ + 2);
        maths::CVector<float> b(b_, b_ + 2);
        maths::CVector<float> c(c_, c_ + 2);
        maths::CVector<float> d(d_, d_ + 2);
        CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
        CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
        CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
    }
}

void CEqualWithToleranceTest::testMatrix()
{
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CEqualWithToleranceTest::testMatrix  |");
    LOG_DEBUG("+---------------------------------------+");

    float a_[] = { 1.1f, 1.2f, 1.3f };
    float b_[] = { 1.2f, 1.3f, 1.4f };
    float c_[] = { 201.1f, 202.2f, 203.4f };
    float d_[] = { 202.1f, 203.2f, 204.4f };

    maths::CSymmetricMatrix<double> epsAbs(2, 0.21 / 2.0);
    maths::CSymmetricMatrix<double> epsRel(2, 0.005 / 2.0);

    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> abs(maths::CToleranceTypes::E_AbsoluteTolerance, epsAbs);
    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> rel(maths::CToleranceTypes::E_RelativeTolerance, epsRel);
    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> absAndRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                                          & maths::CToleranceTypes::E_RelativeTolerance,
                                                                          epsAbs, epsRel);
    maths::CEqualWithTolerance<maths::CSymmetricMatrix<double>> absOrRel(  maths::CToleranceTypes::E_AbsoluteTolerance
                                                                         | maths::CToleranceTypes::E_RelativeTolerance,
                                                                         epsAbs, epsRel);

    {
        maths::CSymmetricMatrix<double> a(a_, a_ + 3);
        maths::CSymmetricMatrix<double> b(b_, b_ + 3);
        maths::CSymmetricMatrix<double> c(c_, c_ + 3);
        maths::CSymmetricMatrix<double> d(d_, d_ + 3);
        CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
        CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
        CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
    }
    {
        maths::CSymmetricMatrix<double> a(a_, a_ + 3);
        maths::CSymmetricMatrix<double> b(b_, b_ + 3);
        maths::CSymmetricMatrix<double> c(c_, c_ + 3);
        maths::CSymmetricMatrix<double> d(d_, d_ + 3);
        CPPUNIT_ASSERT_EQUAL(true,  abs(-a, -b));
        CPPUNIT_ASSERT_EQUAL(false, abs(-c, -d));
        CPPUNIT_ASSERT_EQUAL(false, rel(-a, -b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(-c, -d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(-a, -b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(-c, -d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
    }
    {
        maths::CSymmetricMatrix<float> a(a_, a_ + 3);
        maths::CSymmetricMatrix<float> b(b_, b_ + 3);
        maths::CSymmetricMatrix<float> c(c_, c_ + 3);
        maths::CSymmetricMatrix<float> d(d_, d_ + 3);
        CPPUNIT_ASSERT_EQUAL(true,  abs(a, b));
        CPPUNIT_ASSERT_EQUAL(false, abs(c, d));
        CPPUNIT_ASSERT_EQUAL(false, rel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  rel(c, d));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(a, b));
        CPPUNIT_ASSERT_EQUAL(false, absAndRel(c, d));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(a, b));
        CPPUNIT_ASSERT_EQUAL(true,  absOrRel(c, d));
    }
}

CppUnit::Test *CEqualWithToleranceTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CEqualWithToleranceTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CEqualWithToleranceTest>(
                                   "CEqualWithToleranceTest::testScalar",
                                   &CEqualWithToleranceTest::testScalar) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CEqualWithToleranceTest>(
                                   "CEqualWithToleranceTest::testVector",
                                   &CEqualWithToleranceTest::testVector) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CEqualWithToleranceTest>(
                                   "CEqualWithToleranceTest::testMatrix",
                                   &CEqualWithToleranceTest::testMatrix) );

    return suiteOfTests;
}
