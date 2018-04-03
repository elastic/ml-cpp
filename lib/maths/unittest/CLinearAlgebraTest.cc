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

#include "CLinearAlgebraTest.h"

#include <core/CLogger.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraTools.h>

#include <boost/range.hpp>

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;

void CLinearAlgebraTest::testSymmetricMatrixNxN(void)
{
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testSymmetricMatrixNxN  |");
    LOG_DEBUG("+----------------------------------------------+");

    // Construction.
    {
        maths::CSymmetricMatrixNxN<double, 3> matrix;
        LOG_DEBUG("matrix = " << matrix);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j < 3; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(0.0, matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(0.0, matrix.trace());
   }
    {
        maths::CSymmetricMatrixNxN<double, 3> matrix(3.0);
        LOG_DEBUG("matrix = " << matrix);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j < 3; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(3.0, matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(9.0, matrix.trace());
    }
    {
        double m[][5] =
            {
                { 1.1, 2.4, 1.4, 3.7, 4.0 },
                { 2.4, 3.2, 1.8, 0.7, 1.0 },
                { 1.4, 1.8, 0.8, 4.7, 3.1 },
                { 3.7, 0.7, 4.7, 4.7, 1.1 },
                { 4.0, 1.0, 3.1, 1.1, 1.0 }
            };
        maths::CSymmetricMatrixNxN<double, 5> matrix(m);
        LOG_DEBUG("matrix = " << matrix);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(m[i][j], matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(10.8, matrix.trace());
    }
    {
        double v[] = { 1.0, 2.0, 3.0, 4.0 };
        maths::CVectorNx1<double, 4> vector(v);
        maths::CSymmetricMatrixNxN<double, 4> matrix(maths::E_OuterProduct, vector);
        LOG_DEBUG("matrix = " << matrix);
        for (std::size_t i = 0u; i < 4; ++i)
        {
            for (std::size_t j = 0u; j < 4; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i+1) * (j+1)), matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(30.0, matrix.trace());
    }
    {
        double v[] = { 1.0, 2.0, 3.0, 4.0 };
        maths::CVectorNx1<double, 4> vector(v);
        maths::CSymmetricMatrixNxN<double, 4> matrix(maths::E_Diagonal, vector);
        LOG_DEBUG("matrix = " << matrix);
        for (std::size_t i = 0u; i < 4; ++i)
        {
            for (std::size_t j = 0u; j < 4; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(i == j ? vector(i) : 0.0, matrix(i, j));
            }
        }
    }

    // Operators
    {
        LOG_DEBUG("Sum");

        double m[][5] =
            {
                { 1.1, 2.4, 1.4, 3.7, 4.0 },
                { 2.4, 3.2, 1.8, 0.7, 1.0 },
                { 1.4, 1.8, 0.8, 4.7, 3.1 },
                { 3.7, 0.7, 4.7, 4.7, 1.1 },
                { 4.0, 1.0, 3.1, 1.1, 1.0 }
            };
        maths::CSymmetricMatrixNxN<double, 5> matrix(m);
        maths::CSymmetricMatrixNxN<double, 5> sum = matrix + matrix;
        LOG_DEBUG("sum = " << sum);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(2.0 * m[i][j], sum(i, j));
            }
        }
    }
    {
        LOG_DEBUG("Difference");

        double m1[][3] =
            {
                { 1.1, 0.4, 1.4 },
                { 0.4, 1.2, 1.8 },
                { 1.4, 1.8, 0.8 }
            };
        double m2[][3] =
            {
                { 2.1, 0.3, 0.4 },
                { 0.3, 1.2, 3.8 },
                { 0.4, 3.8, 0.2 }
            };
        maths::CSymmetricMatrixNxN<double, 3> matrix1(m1);
        maths::CSymmetricMatrixNxN<double, 3> matrix2(m2);
        maths::CSymmetricMatrixNxN<double, 3> difference = matrix1 - matrix2;
        LOG_DEBUG("difference = " << difference);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j < 3; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(m1[i][j] - m2[i][j], difference(i, j));
            }
        }
    }
    {
        LOG_DEBUG("Matrix Scalar Multiplication");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVectorNx1<double, 5> vector(v);
        maths::CSymmetricMatrixNxN<double, 5> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrixNxN<double, 5> ms = m * 3.0;
        LOG_DEBUG("3 * m = " << ms);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i+1) * (j+1)), ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG("Matrix Scalar Division");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVectorNx1<double, 5> vector(v);
        maths::CSymmetricMatrixNxN<double, 5> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrixNxN<double, 5> ms = m / 4.0;
        LOG_DEBUG("m / 4.0 = " << ms);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i+1) * (j+1)) / 4.0, ms(i, j));
            }
        }
    }
}

void CLinearAlgebraTest::testVectorNx1(void)
{
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testVectorNx1  |");
    LOG_DEBUG("+-------------------------------------+");

    // Construction.
    {
        maths::CVectorNx1<double, 3> vector;
        LOG_DEBUG("vector = " << vector);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(0.0, vector(i));
        }
    }
    {
        maths::CVectorNx1<double, 3> vector(3.0);
        LOG_DEBUG("vector = " << vector);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(3.0, vector(i));
        }
    }
    {
        double v[] = { 1.1, 2.4, 1.4, 3.7, 4.0 };
        maths::CVectorNx1<double, 5> vector(v);
        LOG_DEBUG("vector = " << vector);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(v[i], vector(i));
        }
    }

    // Operators
    {
        LOG_DEBUG("Sum");

        double v[] = { 1.1, 2.4, 1.4, 3.7, 4.0 };
        maths::CVectorNx1<double, 5> vector(v);
        maths::CVectorNx1<double, 5> sum = vector + vector;
        LOG_DEBUG("vector = " << sum);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(2.0 * v[i], sum(i));
        }
    }
    {
        LOG_DEBUG("Difference");

        double v1[] = { 1.1, 0.4, 1.4 };
        double v2[] = { 2.1, 0.3, 0.4 };
        maths::CVectorNx1<double, 3> vector1(v1);
        maths::CVectorNx1<double, 3> vector2(v2);
        maths::CVectorNx1<double, 3> difference = vector1 - vector2;
        LOG_DEBUG("vector = " << difference);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(v1[i] - v2[i], difference(i));
        }
    }
    {
        LOG_DEBUG("Matrix Vector Multiplication");

        Eigen::Matrix4d randomMatrix;
        Eigen::Vector4d randomVector;
        for (std::size_t t = 0u; t < 20; ++t)
        {
            randomMatrix.setRandom();
            Eigen::Matrix4d a = randomMatrix.selfadjointView<Eigen::Lower>();
            LOG_DEBUG("A   =\n" << a);
            randomVector.setRandom();
            LOG_DEBUG("x   =\n" << randomVector);
            Eigen::Vector4d expected = a * randomVector;
            LOG_DEBUG("Ax =\n" << expected);

            maths::CSymmetricMatrixNxN<double, 4> s(maths::fromDenseMatrix(randomMatrix));
            LOG_DEBUG("S   = " << s);
            maths::CVectorNx1<double, 4> y(maths::fromDenseVector(randomVector));
            LOG_DEBUG("y   =\n" << y);
            maths::CVectorNx1<double, 4> sy = s * y;
            LOG_DEBUG("Sy = " << sy);
            for (std::size_t i = 0u; i < 4; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected(i), sy(i));
            }
        }
    }
    {
        LOG_DEBUG("Vector Scalar Multiplication");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVectorNx1<double, 5> vector(v);
        maths::CVectorNx1<double, 5> vs = vector * 3.0;
        LOG_DEBUG("3 * v = " << vs);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i+1)), vs(i));
        }
    }
    {
        LOG_DEBUG("Matrix Scalar Division");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVectorNx1<double, 5> vector(v);
        maths::CVectorNx1<double, 5> vs = vector / 4.0;
        LOG_DEBUG("v / 4.0 = " << vs);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(static_cast<double>((i+1)) / 4.0, vs(i));
        }
    }
}

void CLinearAlgebraTest::testSymmetricMatrix(void)
{
    LOG_DEBUG("+-------------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testSymmetricMatrix  |");
    LOG_DEBUG("+-------------------------------------------+");

    // Construction.
    {
        maths::CSymmetricMatrix<double> matrix(3);
        LOG_DEBUG("matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), matrix.columns());
        for (std::size_t i = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j < 3; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(0.0, matrix(i, j));
            }
        }
    }
    {
        maths::CSymmetricMatrix<double> matrix(4, 3.0);
        LOG_DEBUG("matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.columns());
        for (std::size_t i = 0u; i < 4; ++i)
        {
            for (std::size_t j = 0u; j < 4; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(3.0, matrix(i, j));
            }
        }
    }
    {
        double m_[][5] =
            {
                { 1.1, 2.4, 1.4, 3.7, 4.0 },
                { 2.4, 3.2, 1.8, 0.7, 1.0 },
                { 1.4, 1.8, 0.8, 4.7, 3.1 },
                { 3.7, 0.7, 4.7, 4.7, 1.1 },
                { 4.0, 1.0, 3.1, 1.1, 1.0 }
            };
        TDoubleVecVec m(5, TDoubleVec(5));
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                m[i][j] = m_[i][j];
            }
        }
        maths::CSymmetricMatrix<double> matrix(m);
        LOG_DEBUG("matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.columns());
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(m[i][j], matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(10.8, matrix.trace());
    }
    {
        double m[] =
            {
                1.1,
                2.4, 3.2,
                1.4, 1.8, 0.8,
                3.7, 0.7, 4.7, 4.7,
                4.0, 1.0, 3.1, 1.1, 1.0
            };
        maths::CSymmetricMatrix<double> matrix(boost::begin(m), boost::end(m));
        LOG_DEBUG("matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.columns());
        for (std::size_t i = 0u, k = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j, ++k)
            {
                CPPUNIT_ASSERT_EQUAL(m[k], matrix(i, j));
                CPPUNIT_ASSERT_EQUAL(m[k], matrix(j, i));
            }
        }
        CPPUNIT_ASSERT_EQUAL(10.8, matrix.trace());
    }
    {
        double v[] = { 1.0, 2.0, 3.0, 4.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> matrix(maths::E_OuterProduct, vector);
        LOG_DEBUG("matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.columns());
        for (std::size_t i = 0u; i < 4; ++i)
        {
            for (std::size_t j = 0u; j < 4; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i+1) * (j+1)), matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(30.0, matrix.trace());
    }
    {
        double v[] = { 1.0, 2.0, 3.0, 4.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> matrix(maths::E_Diagonal, vector);
        LOG_DEBUG("matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.columns());
        for (std::size_t i = 0u; i < 4; ++i)
        {
            for (std::size_t j = 0u; j < 4; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(i == j ? vector(i) : 0.0, matrix(i, j));
            }
        }
    }

    // Operators
    {
        LOG_DEBUG("Sum");

        double m[] =
            {
                1.1,
                2.4, 3.2,
                1.4, 1.8, 0.8,
                3.7, 0.7, 4.7, 4.7,
                4.0, 1.0, 3.1, 1.1, 1.0
            };
        maths::CSymmetricMatrix<double> matrix(boost::begin(m), boost::end(m));
        maths::CSymmetricMatrix<double> sum = matrix + matrix;
        LOG_DEBUG("sum = " << sum);
        for (std::size_t i = 0u, k = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j, ++k)
            {
                CPPUNIT_ASSERT_EQUAL(2.0 * m[k], sum(i, j));
                CPPUNIT_ASSERT_EQUAL(2.0 * m[k], sum(j, i));
            }
        }
    }
    {
        LOG_DEBUG("Difference");

        double m1[] =
            {
                1.1,
                0.4, 1.2,
                1.4, 1.8, 0.8
            };
        double m2[] =
            {
                2.1,
                0.3, 1.2,
                0.4, 3.8, 0.2
            };
        maths::CSymmetricMatrix<double> matrix1(boost::begin(m1), boost::end(m1));
        maths::CSymmetricMatrix<double> matrix2(boost::begin(m2), boost::end(m2));
        maths::CSymmetricMatrix<double> difference = matrix1 - matrix2;
        LOG_DEBUG("difference = " << difference);
        for (std::size_t i = 0u, k = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j, ++k)
            {
                CPPUNIT_ASSERT_EQUAL(m1[k] - m2[k], difference(i, j));
                CPPUNIT_ASSERT_EQUAL(m1[k] - m2[k], difference(j, i));
            }
        }
    }
    {
        LOG_DEBUG("Matrix Scalar Multiplication");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrix<double> ms = m * 3.0;
        LOG_DEBUG("3 * m = " << ms);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i+1) * (j+1)), ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG("Matrix Scalar Division");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrix<double> ms = m / 4.0;
        LOG_DEBUG("m / 4.0 = " << ms);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            for (std::size_t j = 0u; j < 5; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i+1) * (j+1)) / 4.0, ms(i, j));
            }
        }
    }
}

void CLinearAlgebraTest::testVector(void)
{
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testVector  |");
    LOG_DEBUG("+----------------------------------+");

    // Construction.
    {
        maths::CVector<double> vector(3);
        LOG_DEBUG("vector = " << vector);
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), vector.dimension());
        for (std::size_t i = 0u; i < 3; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(0.0, vector(i));
        }
    }
    {
        maths::CVector<double> vector(4, 3.0);
        LOG_DEBUG("vector = " << vector);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), vector.dimension());
        for (std::size_t i = 0u; i < 4; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(3.0, vector(i));
        }
    }
    {
        double v_[] = { 1.1, 2.4, 1.4, 3.7, 4.0 };
        TDoubleVec v(boost::begin(v_), boost::end(v_));
        maths::CVector<double> vector(v);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), vector.dimension());
        LOG_DEBUG("vector = " << vector);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(v[i], vector(i));
        }
    }
    {
        double v[] = { 1.1, 2.4, 1.4, 3.7, 4.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), vector.dimension());
        LOG_DEBUG("vector = " << vector);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(v[i], vector(i));
        }
    }

    // Operators
    {
        LOG_DEBUG("Sum");

        double v[] = { 1.1, 2.4, 1.4, 3.7, 4.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CVector<double> sum = vector + vector;
        LOG_DEBUG("vector = " << sum);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(2.0 * v[i], sum(i));
        }
    }
    {
        LOG_DEBUG("Difference");

        double v1[] = { 1.1, 0.4, 1.4 };
        double v2[] = { 2.1, 0.3, 0.4 };
        maths::CVector<double> vector1(boost::begin(v1), boost::end(v1));
        maths::CVector<double> vector2(boost::begin(v2), boost::end(v2));
        maths::CVector<double> difference = vector1 - vector2;
        LOG_DEBUG("vector = " << difference);
        for (std::size_t i = 0u; i < 3; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(v1[i] - v2[i], difference(i));
        }
    }
    {
        LOG_DEBUG("Matrix Vector Multiplication");

        Eigen::MatrixXd randomMatrix(std::size_t(4), std::size_t(4));
        Eigen::VectorXd randomVector(4);
        for (std::size_t t = 0u; t < 20; ++t)
        {
            randomMatrix.setRandom();
            Eigen::MatrixXd a = randomMatrix.selfadjointView<Eigen::Lower>();
            LOG_DEBUG("A   =\n" << a);
            randomVector.setRandom();
            LOG_DEBUG("x   =\n" << randomVector);
            Eigen::VectorXd expected = a * randomVector;
            LOG_DEBUG("Ax =\n" << expected);

            maths::CSymmetricMatrix<double> s(maths::fromDenseMatrix(randomMatrix));
            LOG_DEBUG("S   = " << s);
            maths::CVector<double> y(maths::fromDenseVector(randomVector));
            LOG_DEBUG("y   =\n" << y);
            maths::CVector<double> sy = s * y;
            LOG_DEBUG("Sy = " << sy);
            for (std::size_t i = 0u; i < 4; ++i)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected(i), sy(i), 1e-10);
            }
        }
    }
    {
        LOG_DEBUG("Vector Scalar Multiplication");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CVector<double> vs = vector * 3.0;
        LOG_DEBUG("3 * v = " << vs);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i+1)), vs(i));
        }
    }
    {
        LOG_DEBUG("Matrix Scalar Division");

        double v[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CVector<double> vs = vector / 4.0;
        LOG_DEBUG("v / 4.0 = " << vs);
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(static_cast<double>((i+1)) / 4.0, vs(i));
        }
    }
}

void CLinearAlgebraTest::testNorms(void)
{
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testNorms  |");
    LOG_DEBUG("+---------------------------------+");

    double v[][5] =
        {
            {  1.0,  2.1,  3.2, 1.7,  0.1 },
            {  0.0, -2.1,  1.2, 1.9,  4.1 },
            { -1.0,  7.1,  5.2, 1.7, -0.1 },
            { -3.0,  1.1, -3.3, 1.8,  6.1 }
        };
    double expectedEuclidean[] =
        {
            4.30697,
            5.12543,
            9.01942,
            7.84538
        };

    for (std::size_t i = 0u; i < boost::size(v); ++i)
    {
        maths::CVectorNx1<double, 5> v_(v[i]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedEuclidean[i], v_.euclidean(), 5e-6);
    }

    double m[][15] =
        {
            {  1.0,
               2.1,  3.2,
               1.7,  0.1,  4.2,
               0.3,  2.8,  4.1, 0.1,
               0.4,  1.2,  5.2, 0.2, 6.3 },
            {  0.0,
              -2.1,  1.2,
               1.9,  4.1,  4.5,
              -3.1,  0.0,  1.3, 7.5,
               0.2,  1.0,  4.5, 8.1, 0.3 },
            { -1.0,
               7.1,  5.2,
               1.7, -0.1,  3.2,
               1.8, -3.2,  4.2, 9.1,
               0.2,  0.4,  4.1, 7.2, 1.3 },
            { -3.0,
               1.1, -3.3,
               1.8,  6.1, -1.3,
               1.3,  4.2,  3.1, 1.9,
              -2.3,  3.1,  2.4, 2.3, 1.0 }
        };
    double expectedFrobenius[] =
        {
            13.78550,
            18.00250,
            20.72052,
            14.80844
        };

    for (std::size_t i = 0u; i < boost::size(m); ++i)
    {
        maths::CSymmetricMatrixNxN<double, 5> m_(boost::begin(m[i]),
                                                 boost::end(m[i]));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFrobenius[i], m_.frobenius(), 5e-6);
    }
}

void CLinearAlgebraTest::testUtils(void)
{
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testUtils  |");
    LOG_DEBUG("+---------------------------------+");

    // Test component min, max, sqrt and fabs.
    {
        LOG_DEBUG("Vector min, max, fabs, sqrt");

        const double v1_[] = { 1.0, 3.1, 2.2, 4.9, 12.0 };
        maths::CVectorNx1<double, 5> v1(v1_);
        const double v2_[] = { 1.5, 3.0, 2.7, 5.2, 8.0 };
        maths::CVectorNx1<double, 5> v2(v2_);
        const double v3_[] = { -1.0, 3.1, -2.2, -4.9, 12.0 };
        maths::CVectorNx1<double, 5> v3(v3_);

        {
            double expected[] = { 1.0, 3.1, 2.2, 4.0, 4.0 };
            LOG_DEBUG("min(v1, 4.0) = " << maths::min(v1, 4.0));
            for (std::size_t i = 0u; i < 5; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected[i],
                                     (maths::min(v1, 4.0))(i));
            }
        }
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL((maths::min(v1, 4.0))(i),
                                 (maths::min(4.0, v1))(i));
        }
        {
            double expected[] = { 1.0, 3.0, 2.2, 4.9, 8.0 };
            LOG_DEBUG("min(v1, v2) = " << maths::min(v1, v2));
            for (std::size_t i = 0u; i < 5; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected[i],
                                     (maths::min(v1, v2))(i));
            }
        }

        {
            double expected[] = { 3.0, 3.1, 3.0, 4.9, 12.0 };
            LOG_DEBUG("max(v1, 3.0) = " << maths::max(v1, 3.0));
            for (std::size_t i = 0u; i < 5; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected[i],
                                     (maths::max(v1, 3.0))(i));
            }
        }
        for (std::size_t i = 0u; i < 5; ++i)
        {
            CPPUNIT_ASSERT_EQUAL((maths::max(v1, 3.0))(i),
                                 (maths::max(3.0, v1))(i));
        }
        {
            double expected[] = { 1.5, 3.1, 2.7, 5.2, 12.0 };
            LOG_DEBUG("max(v1, v2) = " << maths::max(v1, v2));
            for (std::size_t i = 0u; i < 5; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected[i],
                                     (maths::max(v1, v2))(i));
            }
        }

        {
            double expected[] = { 1.0, ::sqrt(3.1), ::sqrt(2.2), ::sqrt(4.9), ::sqrt(12.0) };
            LOG_DEBUG("sqrt(v1) = " << maths::sqrt(v1));
            for (std::size_t i = 0u; i < 5; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected[i],
                                     (maths::sqrt(v1))(i));
            }
        }

        {
            const double expected[] = { 1.0, 3.1, 2.2, 4.9, 12.0 };
            LOG_DEBUG("fabs(v3) = " << maths::fabs(v3));
            for (std::size_t i = 0u; i < 5; ++i)
            {
                CPPUNIT_ASSERT_EQUAL(expected[i],
                                     (maths::fabs(v3))(i));
            }
        }
    }
    {
        LOG_DEBUG("Matrix min, max, fabs, sqrt");

        double m1_[][3] =
            {
                { 2.1, 0.3, 0.4 },
                { 0.3, 1.2, 3.8 },
                { 0.4, 3.8, 0.2 }
            };
        maths::CSymmetricMatrixNxN<double, 3> m1(m1_);
        double m2_[][3] =
            {
                { 1.1, 0.4, 1.4 },
                { 0.4, 1.2, 1.8 },
                { 1.4, 1.8, 0.8 }
            };
        maths::CSymmetricMatrixNxN<double, 3> m2(m2_);
        double m3_[][3] =
            {
                { -2.1,  0.3,  0.4 },
                {  0.3, -1.2, -3.8 },
                {  0.4, -3.8,  0.2 }
            };
        maths::CSymmetricMatrixNxN<double, 3> m3(m3_);

        {
            double expected[][3] =
                {
                    { 2.1, 0.3, 0.4 },
                    { 0.3, 1.2, 3.0 },
                    { 0.4, 3.0, 0.2 }
                };
            LOG_DEBUG("min(m1, 3.0) = " << maths::min(m1, 3.0));
            for (std::size_t i = 0u; i < 3; ++i)
            {
                for (std::size_t j = 0u; j < 3; ++j)
                {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j],
                                         (maths::min(m1, 3.0))(i, j));
                }
            }
        }
        for (std::size_t i = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j < 3; ++j)
            {
                CPPUNIT_ASSERT_EQUAL((maths::min(m1, 3.0))(i, j),
                                     (maths::min(3.0, m1))(i, j));
            }
        }
        {
            double expected[][3] =
                {
                    { 1.1, 0.3, 0.4 },
                    { 0.3, 1.2, 1.8 },
                    { 0.4, 1.8, 0.2 }
                };
            LOG_DEBUG("min(m1, m2) = " << maths::min(m1, m2));
            for (std::size_t i = 0u; i < 3; ++i)
            {
                for (std::size_t j = 0u; j < 3; ++j)
                {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j],
                                         (maths::min(m1, m2))(i, j));
                }
            }
        }

        {
            double expected[][3] =
                {
                    { 2.1, 2.0, 2.0 },
                    { 2.0, 2.0, 3.8 },
                    { 2.0, 3.8, 2.0 }
                };
            LOG_DEBUG("max(m1, 2.0) = " << maths::max(m1, 2.0));
            for (std::size_t i = 0u; i < 3; ++i)
            {
                for (std::size_t j = 0u; j < 3; ++j)
                {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j],
                                         (maths::max(m1, 2.0))(i, j));
                }
            }
        }
        for (std::size_t i = 0u; i < 3; ++i)
        {
            for (std::size_t j = 0u; j < 3; ++j)
            {
                CPPUNIT_ASSERT_EQUAL((maths::max(m1, 2.0))(i, j),
                                     (maths::max(2.0, m1))(i, j));
            }
        }
        {
            double expected[][3] =
                {
                    { 2.1, 0.4, 1.4 },
                    { 0.4, 1.2, 3.8 },
                    { 1.4, 3.8, 0.8 }
                };
            LOG_DEBUG("max(m1, m2) = " << maths::max(m1, m2));
            for (std::size_t i = 0u; i < 3; ++i)
            {
                for (std::size_t j = 0u; j < 3; ++j)
                {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j],
                                         (maths::max(m1, m2))(i, j));
                }
            }
        }

        {
            double expected[][3] =
                {
                    { ::sqrt(2.1), ::sqrt(0.3), ::sqrt(0.4) },
                    { ::sqrt(0.3), ::sqrt(1.2), ::sqrt(3.8) },
                    { ::sqrt(0.4), ::sqrt(3.8), ::sqrt(0.2) }
                };
            LOG_DEBUG("sqrt(m1) = " << maths::sqrt(m1));
            for (std::size_t i = 0u; i < 3; ++i)
            {
                for (std::size_t j = 0u; j < 3; ++j)
                {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j],
                                         (maths::sqrt(m1))(i, j));
                }
            }
        }

        {
            double expected[][3] =
                {
                    { 2.1, 0.3, 0.4 },
                    { 0.3, 1.2, 3.8 },
                    { 0.4, 3.8, 0.2 }
                };
            LOG_DEBUG("fabs(m3) = " << maths::fabs(m3));
            for (std::size_t i = 0u; i < 3; ++i)
            {
                for (std::size_t j = 0u; j < 3; ++j)
                {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j],
                                         (maths::fabs(m3))(i, j));
                }
            }
        }
    }
}

void CLinearAlgebraTest::testGaussianLogLikelihood(void)
{
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testGaussianLogLikelihood  |");
    LOG_DEBUG("+-------------------------------------------------+");

    // Test the log likelihood (expected from octave).
    {
        const double covariance_[][4] =
            {
                { 10.70779,  0.14869,  1.44263,  2.26889 },
                {  0.14869, 10.70919,  2.56363,  1.87805 },
                {  1.44263,  2.56363, 11.90966,  2.44121 },
                {  2.26889,  1.87805,  2.44121, 11.53904 }
            };
        maths::CSymmetricMatrixNxN<double, 4> covariance(covariance_);

        const double x_[][4] =
            {
                { -1.335028, -0.222988, -0.174935, -0.480772 },
                {  0.137550,  1.286252,  0.027043,  1.349709 },
                { -0.445561,  2.390953,  0.302770,  0.084871 },
                {  0.275802,  0.408910, -2.247157,  0.196043 },
                {  0.179101,  0.177340, -0.456634,  5.314863 },
                {  0.260426,  0.325159,  1.214650, -1.267697 },
                { -0.363917, -0.422225,  0.360000,  0.401383 },
                {  1.492814,  3.257986,  0.065441, -0.187108 },
                {  1.214063,  0.067988, -0.241846, -0.425730 },
                { -0.306693, -0.188497, -1.092719,  1.288093 }
            };

        const double expected[] =
            {
               -8.512128, -8.569778, -8.706920, -8.700537, -9.794163,
               -8.602336, -8.462027, -9.096402, -8.521042, -8.590054
            };

        for (std::size_t i = 0u; i < boost::size(x_); ++i)
        {
            maths::CVectorNx1<double, 4> x(x_[i]);
            double likelihood;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, x, likelihood));
            LOG_DEBUG("expected log(L(x)) = " << expected[i]);
            LOG_DEBUG("got      log(L(x)) = " << likelihood);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[i], likelihood, 1e-6);
        }
    }

    // Test log likelihood singular matrix.
    {
        double e1_[] = {  1.0,  1.0,  1.0, 1.0 };
        double e2_[] = { -1.0,  1.0,  0.0, 0.0 };
        double e3_[] = { -1.0, -1.0,  2.0, 0.0 };
        double e4_[] = { -1.0, -1.0, -1.0, 3.0 };
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
                  10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e1 / e1.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e2 / e2.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e3 / e3.euclidean()));

        double likelihood;
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e1, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  3.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0)
                                             + 4.0 / 10.0),
                                     likelihood,
                                     1e-10);

        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e2, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  3.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0)
                                             + 2.0 / 5.0),
                                     likelihood,
                                     1e-10);

        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e3, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  3.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0)
                                             + 6.0 / 5.0),
                                     likelihood,
                                     1e-10);

        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpOverflowed, maths::gaussianLogLikelihood(covariance, e1, likelihood, false));
        CPPUNIT_ASSERT(likelihood > 0.0);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpOverflowed, maths::gaussianLogLikelihood(covariance, e4, likelihood, false));
        CPPUNIT_ASSERT(likelihood < 0.0);
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double e1_[] = {  1.0,  1.0,  1.0, 1.0 };
        double e2_[] = { -1.0,  1.0,  0.0, 0.0 };
        double e3_[] = { -1.0, -1.0,  2.0, 0.0 };
        double e4_[] = { -1.0, -1.0, -1.0, 3.0 };
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
                  10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e1 / e1.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e2 / e2.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e3 / e3.euclidean())
                +  2.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e4 / e4.euclidean()));

        double likelihood;
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e1, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  4.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0 * 2.0)
                                             + 4.0 / 10.0),
                                     likelihood,
                                     1e-10);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e2, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  4.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0 * 2.0)
                                             + 2.0 / 5.0),
                                     likelihood,
                                     1e-10);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e3, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  4.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0 * 2.0)
                                             + 6.0 / 5.0),
                                     likelihood,
                                     1e-10);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, maths::gaussianLogLikelihood(covariance, e4, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5 * (  4.0 * ::log(boost::math::double_constants::two_pi)
                                             + ::log(10.0 * 5.0 * 5.0 * 2.0)
                                             + 12.0 / 2.0),
                                     likelihood,
                                     1e-10);
    }
}

void CLinearAlgebraTest::testSampleGaussian(void)
{
    LOG_DEBUG("+------------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testSampleGaussian  |");
    LOG_DEBUG("+------------------------------------------+");

    // Test singular matrix.
    {
        double m[] = { 1.0, 2.0, 3.0, 4.0 };
        maths::CVectorNx1<double, 4> mean(m);

        double e1_[] = {  1.0,  1.0,  1.0, 1.0 };
        double e2_[] = { -1.0,  1.0,  0.0, 0.0 };
        double e3_[] = { -1.0, -1.0,  2.0, 0.0 };
        double e4_[] = { -1.0, -1.0, -1.0, 3.0 };
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
                  10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e1 / e1.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e2 / e2.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e3 / e3.euclidean()));

        std::vector<maths::CVectorNx1<double, 4>> samples;
        maths::sampleGaussian(100, mean, covariance, samples);

        CPPUNIT_ASSERT_EQUAL(std::size_t(99), samples.size());

        maths::CBasicStatistics::SSampleCovariances<double, 4> covariances;

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            covariances.add(samples[i]);
        }

        LOG_DEBUG("mean       = " << mean);
        LOG_DEBUG("covariance = " << covariance);
        LOG_DEBUG("sample mean       = " << maths::CBasicStatistics::mean(covariances));
        LOG_DEBUG("sample covariance = " << maths::CBasicStatistics::maximumLikelihoodCovariances(covariances));

        maths::CVectorNx1<double, 4> meanError =
                  maths::CVectorNx1<double, 4>(mean)
                - maths::CBasicStatistics::mean(covariances);
        maths::CSymmetricMatrixNxN<double, 4> covarianceError =
                  maths::CSymmetricMatrixNxN<double, 4>(covariance)
                - maths::CBasicStatistics::maximumLikelihoodCovariances(covariances);

        LOG_DEBUG("|error| / |mean| = "
                  << meanError.euclidean() / mean.euclidean());
        LOG_DEBUG("|error| / |covariance| = "
                  << covarianceError.frobenius() / covariance.frobenius());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meanError.euclidean() / mean.euclidean(), 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, covarianceError.frobenius() / covariance.frobenius(), 0.01);
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double m[] = { 15.0, 0.0, 1.0, 5.0 };
        maths::CVectorNx1<double, 4> mean(m);

        double e1_[] = {  1.0,  1.0,  1.0, 1.0 };
        double e2_[] = { -1.0,  1.0,  0.0, 0.0 };
        double e3_[] = { -1.0, -1.0,  2.0, 0.0 };
        double e4_[] = { -1.0, -1.0, -1.0, 3.0 };
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
                  10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e1 / e1.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e2 / e2.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e3 / e3.euclidean())
                +  2.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e4 / e4.euclidean()));

        std::vector<maths::CVectorNx1<double, 4>> samples;
        maths::sampleGaussian(100, mean, covariance, samples);

        CPPUNIT_ASSERT_EQUAL(std::size_t(100), samples.size());

        maths::CBasicStatistics::SSampleCovariances<double, 4> covariances;

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            covariances.add(samples[i]);
        }

        LOG_DEBUG("mean       = " << mean);
        LOG_DEBUG("covariance = " << covariance);
        LOG_DEBUG("sample mean       = " << maths::CBasicStatistics::mean(covariances));
        LOG_DEBUG("sample covariance = " << maths::CBasicStatistics::maximumLikelihoodCovariances(covariances));

        maths::CVectorNx1<double, 4> meanError =
                  maths::CVectorNx1<double, 4>(mean)
                - maths::CBasicStatistics::mean(covariances);
        maths::CSymmetricMatrixNxN<double, 4> covarianceError =
                  maths::CSymmetricMatrixNxN<double, 4>(covariance)
                - maths::CBasicStatistics::maximumLikelihoodCovariances(covariances);

        LOG_DEBUG("|error| / |mean| = "
                  << meanError.euclidean() / mean.euclidean());
        LOG_DEBUG("|error| / |covariance| = "
                  << covarianceError.frobenius() / covariance.frobenius());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meanError.euclidean() / mean.euclidean(), 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, covarianceError.frobenius() / covariance.frobenius(), 0.02);
    }
}

void CLinearAlgebraTest::testLogDeterminant(void)
{
    LOG_DEBUG("+------------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testLogDeterminant  |");
    LOG_DEBUG("+------------------------------------------+");

    // Test the determinant (expected from octave).
    {
        const double matrices[][3][3] =
            {
                { { 0.25451, 0.52345, 0.61308 },
                  { 0.52345, 1.19825, 1.12804 },
                  { 0.61308, 1.12804, 1.78833 } },
                { { 0.83654, 0.24520, 0.80310 },
                  { 0.24520, 0.38368, 0.30554 },
                  { 0.80310, 0.30554, 0.78936 } },
                { { 0.73063, 0.87818, 0.85836 },
                  { 0.87818, 1.50305, 1.17931 },
                  { 0.85836, 1.17931, 1.05850 } },
                { { 0.38947, 0.61062, 0.34423 },
                  { 0.61062, 1.60437, 0.91664 },
                  { 0.34423, 0.91664, 0.52448 } },
                { { 1.79563, 1.78751, 2.17200 },
                  { 1.78751, 1.83443, 2.17340 },
                  { 2.17200, 2.17340, 2.62958 } },
                { { 0.57023, 0.47992, 0.71581 },
                  { 0.47992, 1.09182, 0.97989 },
                  { 0.71581, 0.97989, 1.32316 } },
                { { 2.31264, 0.72098, 2.38050 },
                  { 0.72098, 0.28103, 0.78025 },
                  { 2.38050, 0.78025, 2.49219 } },
                { { 0.83678, 0.45230, 0.74564 },
                  { 0.45230, 0.26482, 0.33491 },
                  { 0.74564, 0.33491, 1.29216 } },
                { { 0.84991, 0.85443, 0.36922 },
                  { 0.85443, 1.12737, 0.83074 },
                  { 0.36922, 0.83074, 1.01195 } },
                { { 0.27156, 0.26441, 0.29726 },
                  { 0.26441, 0.32388, 0.18895 },
                  { 0.29726, 0.18895, 0.47884 } }
            };

        const double expected[] =
            {
                5.1523e-03, 6.7423e-04, 4.5641e-04, 1.5880e-04, 3.1654e-06,
                8.5319e-02, 2.0840e-03, 6.8008e-03, 1.4755e-02, 2.6315e-05
            };

        for (std::size_t i = 0u; i < boost::size(matrices); ++i)
        {
            maths::CSymmetricMatrixNxN<double, 3> M(matrices[i]);
            double logDeterminant;
            maths::logDeterminant(M, logDeterminant);
            LOG_DEBUG("expected |M| = " << expected[i]);
            LOG_DEBUG("got      |M| = " << ::exp(logDeterminant));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[i], ::exp(logDeterminant), 1e-4 * expected[i]);
        }
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double e1_[] = {  1.0,  1.0,  1.0, 1.0 };
        double e2_[] = { -1.0,  1.0,  0.0, 0.0 };
        double e3_[] = { -1.0, -1.0,  2.0, 0.0 };
        double e4_[] = { -1.0, -1.0, -1.0, 3.0 };
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> M(
                  10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e1 / e1.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e2 / e2.euclidean())
                +  5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e3 / e3.euclidean())
                +  2.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct, e4 / e4.euclidean()));
        double logDeterminant;
        maths::logDeterminant(M, logDeterminant);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(::log(10.0 * 5.0 * 5.0 * 2.0), logDeterminant, 1e-10);
    }
}

namespace
{

template<typename MATRIX>
std::string print(const MATRIX &m)
{
    std::ostringstream result;
    result << m;
    return result.str();
}

}

void CLinearAlgebraTest::testProjected(void)
{
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testProjected  |");
    LOG_DEBUG("+-------------------------------------+");

    using TSizeVec = std::vector<std::size_t>;

    const double m[][5] =
        {
            { 1.2, 2.4, 1.9, 3.8, 8.3 },
            { 2.4, 1.0, 0.2, 1.6, 3.1 },
            { 1.9, 0.2, 8.1, 1.1, 0.1 },
            { 3.8, 1.6, 1.1, 3.7, 7.3 },
            { 8.3, 3.1, 0.1, 7.3, 0.9 }
        };
    const double v[] =
        {
            0.3, 3.4, 10.6, 0.9, 5.7
        };

    maths::CSymmetricMatrixNxN<double, 5> matrix(m);
    maths::CVectorNx1<double, 5> vector(v);

    {
        std::size_t ss[] = { 0, 1 };
        TSizeVec subspace(boost::begin(ss), boost::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::projectedVector(subspace, vector);
        LOG_DEBUG("projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG("projectedVector =\n" << projectedVector);
        CPPUNIT_ASSERT_EQUAL(std::string("1.2 2.4\n2.4   1"), print(projectedMatrix));
        CPPUNIT_ASSERT_EQUAL(std::string("0.3\n3.4"), print(projectedVector));
    }
    {
        std::size_t ss[] = { 1, 0 };
        TSizeVec subspace(boost::begin(ss), boost::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::projectedVector(subspace, vector);
        LOG_DEBUG("projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG("projectedVector =\n" << projectedVector);
        CPPUNIT_ASSERT_EQUAL(std::string("  1 2.4\n2.4 1.2"), print(projectedMatrix));
        CPPUNIT_ASSERT_EQUAL(std::string("3.4\n0.3"), print(projectedVector));
    }
    {
        std::size_t ss[] = { 1, 0, 4 };
        TSizeVec subspace(boost::begin(ss), boost::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::projectedVector(subspace, vector);
        LOG_DEBUG("projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG("projectedVector =\n" << projectedVector);
        CPPUNIT_ASSERT_EQUAL(std::string("  1 2.4 3.1\n2.4 1.2 8.3\n3.1 8.3 0.9"), print(projectedMatrix));
        CPPUNIT_ASSERT_EQUAL(std::string("3.4\n0.3\n5.7"), print(projectedVector));
    }
}

void CLinearAlgebraTest::testPersist(void)
{
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CLinearAlgebraTest::testPersist  |");
    LOG_DEBUG("+-----------------------------------+");

    // Check conversion to and from delimited is idempotent.

    {
        double matrix_[][4] =
            {
                { 1.0, 2.1, 1.5, 0.1 },
                { 2.1, 2.2, 3.7, 0.6 },
                { 1.5, 3.7, 0.4, 8.1 },
                { 0.1, 0.6, 8.1, 4.3 }
            };

        maths::CSymmetricMatrixNxN<double, 4> matrix(matrix_);

        std::string str = matrix.toDelimited();

        maths::CSymmetricMatrixNxN<double, 4> restoredMatrix;
        CPPUNIT_ASSERT(restoredMatrix.fromDelimited(str));

        LOG_DEBUG("delimited = " << str);

        for (std::size_t i = 0u; i < 4; ++i)
        {
            for (std::size_t j = 0u; j < 4; ++j)
            {
                CPPUNIT_ASSERT_EQUAL(matrix(i, j), restoredMatrix(i, j));
            }
        }

        CPPUNIT_ASSERT(!restoredMatrix.fromDelimited(std::string()));

        std::string bad("0.1,0.3,0.5,3");
        CPPUNIT_ASSERT(!restoredMatrix.fromDelimited(bad));
        bad = "0.1,0.3,a,0.1,0.3,0.1,0.3,0.1,1.0,3";
        CPPUNIT_ASSERT(!restoredMatrix.fromDelimited(bad));
    }
    {
        double vector_[] = { 11.2, 2.1, 1.5 };

        maths::CVectorNx1<double, 3> vector(vector_);

        std::string str = vector.toDelimited();

        maths::CVectorNx1<double, 3> restoredVector;
        CPPUNIT_ASSERT(restoredVector.fromDelimited(str));

        LOG_DEBUG("delimited = " << str);

        for (std::size_t i = 0u; i < 3; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(vector(i), restoredVector(i));
        }

        CPPUNIT_ASSERT(!restoredVector.fromDelimited(std::string()));

        std::string bad("0.1,0.3,0.5,3");
        CPPUNIT_ASSERT(!restoredVector.fromDelimited(bad));
        bad = "0.1,0.3,a";
        CPPUNIT_ASSERT(!restoredVector.fromDelimited(bad));
    }
}

CppUnit::Test *CLinearAlgebraTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CLinearAlgebraTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testSymmetricMatrixNxN",
                                   &CLinearAlgebraTest::testSymmetricMatrixNxN) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testVectorNx1",
                                   &CLinearAlgebraTest::testVectorNx1) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testSymmetricMatrix",
                                   &CLinearAlgebraTest::testSymmetricMatrix) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testVector",
                                   &CLinearAlgebraTest::testVector) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testNorms",
                                   &CLinearAlgebraTest::testNorms) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testUtils",
                                   &CLinearAlgebraTest::testUtils) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testGaussianLogLikelihood",
                                   &CLinearAlgebraTest::testGaussianLogLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testSampleGaussian",
                                   &CLinearAlgebraTest::testSampleGaussian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testLogDeterminant",
                                   &CLinearAlgebraTest::testLogDeterminant) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testProjected",
                                   &CLinearAlgebraTest::testProjected) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLinearAlgebraTest>(
                                   "CLinearAlgebraTest::testPersist",
                                   &CLinearAlgebraTest::testPersist) );

    return suiteOfTests;
}
