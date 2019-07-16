/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CLinearAlgebraTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsCovariances.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraTools.h>

#include <boost/range.hpp>

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;

void CLinearAlgebraTest::testSymmetricMatrixNxN() {
    // Construction.
    {
        maths::CSymmetricMatrixNxN<double, 3> matrix;
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_EQUAL(0.0, matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(0.0, matrix.trace());
    }
    {
        maths::CSymmetricMatrixNxN<double, 3> matrix(3.0);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_EQUAL(3.0, matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(9.0, matrix.trace());
    }
    {
        double m[][5]{{1.1, 2.4, 1.4, 3.7, 4.0},
                      {2.4, 3.2, 1.8, 0.7, 1.0},
                      {1.4, 1.8, 0.8, 4.7, 3.1},
                      {3.7, 0.7, 4.7, 4.7, 1.1},
                      {4.0, 1.0, 3.1, 1.1, 1.0}};
        maths::CSymmetricMatrixNxN<double, 5> matrix(m);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(m[i][j], matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(10.8, matrix.trace());
    }
    {
        double v[]{1.0, 2.0, 3.0, 4.0};
        maths::CVectorNx1<double, 4> vector(v);
        maths::CSymmetricMatrixNxN<double, 4> matrix(maths::E_OuterProduct, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i + 1) * (j + 1)), matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(30.0, matrix.trace());
    }
    {
        double v[]{1.0, 2.0, 3.0, 4.0};
        maths::CVectorNx1<double, 4> vector(v);
        maths::CSymmetricMatrixNxN<double, 4> matrix(maths::E_Diagonal, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_EQUAL(i == j ? vector(i) : 0.0, matrix(i, j));
            }
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        double m[][5]{{1.1, 2.4, 1.4, 3.7, 4.0},
                      {2.4, 3.2, 1.8, 0.7, 1.0},
                      {1.4, 1.8, 0.8, 4.7, 3.1},
                      {3.7, 0.7, 4.7, 4.7, 1.1},
                      {4.0, 1.0, 3.1, 1.1, 1.0}};
        maths::CSymmetricMatrixNxN<double, 5> matrix(m);
        maths::CSymmetricMatrixNxN<double, 5> sum = matrix + matrix;
        LOG_DEBUG(<< "sum = " << sum);
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(2.0 * m[i][j], sum(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        double m1[][3]{{1.1, 0.4, 1.4}, {0.4, 1.2, 1.8}, {1.4, 1.8, 0.8}};
        double m2[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.8}, {0.4, 3.8, 0.2}};
        maths::CSymmetricMatrixNxN<double, 3> matrix1(m1);
        maths::CSymmetricMatrixNxN<double, 3> matrix2(m2);
        maths::CSymmetricMatrixNxN<double, 3> difference = matrix1 - matrix2;
        LOG_DEBUG(<< "difference = " << difference);
        for (std::size_t i = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_EQUAL(m1[i][j] - m2[i][j], difference(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Multiplication");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVectorNx1<double, 5> vector(v);
        maths::CSymmetricMatrixNxN<double, 5> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrixNxN<double, 5> ms = m * 3.0;
        LOG_DEBUG(<< "3 * m = " << ms);
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i + 1) * (j + 1)),
                                     ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVectorNx1<double, 5> vector(v);
        maths::CSymmetricMatrixNxN<double, 5> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrixNxN<double, 5> ms = m / 4.0;
        LOG_DEBUG(<< "m / 4.0 = " << ms);
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i + 1) * (j + 1)) / 4.0,
                                     ms(i, j));
            }
        }
    }
}

void CLinearAlgebraTest::testVectorNx1() {
    // Construction.
    {
        maths::CVectorNx1<double, 3> vector;
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_EQUAL(0.0, vector(i));
        }
    }
    {
        maths::CVectorNx1<double, 3> vector(3.0);
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_EQUAL(3.0, vector(i));
        }
    }
    {
        double v[]{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::CVectorNx1<double, 5> vector(v);
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(v[i], vector(i));
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        double v[]{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::CVectorNx1<double, 5> vector(v);
        maths::CVectorNx1<double, 5> sum = vector + vector;
        LOG_DEBUG(<< "vector = " << sum);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(2.0 * v[i], sum(i));
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        double v1[]{1.1, 0.4, 1.4};
        double v2[]{2.1, 0.3, 0.4};
        maths::CVectorNx1<double, 3> vector1(v1);
        maths::CVectorNx1<double, 3> vector2(v2);
        maths::CVectorNx1<double, 3> difference = vector1 - vector2;
        LOG_DEBUG(<< "vector = " << difference);
        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_EQUAL(v1[i] - v2[i], difference(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Vector Multiplication");

        Eigen::Matrix4d randomMatrix;
        Eigen::Vector4d randomVector;
        for (std::size_t t = 0u; t < 20; ++t) {
            randomMatrix.setRandom();
            Eigen::Matrix4d a = randomMatrix.selfadjointView<Eigen::Lower>();
            LOG_DEBUG(<< "A   =\n" << a);
            randomVector.setRandom();
            LOG_DEBUG(<< "x   =\n" << randomVector);
            Eigen::Vector4d expected = a * randomVector;
            LOG_DEBUG(<< "Ax =\n" << expected);

            maths::CSymmetricMatrixNxN<double, 4> s(maths::fromDenseMatrix(randomMatrix));
            LOG_DEBUG(<< "S   = " << s);
            maths::CVectorNx1<double, 4> y(maths::fromDenseVector(randomVector));
            LOG_DEBUG(<< "y   =\n" << y);
            maths::CVectorNx1<double, 4> sy = s * y;
            LOG_DEBUG(<< "Sy = " << sy);
            for (std::size_t i = 0u; i < 4; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected(i), sy(i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Vector Scalar Multiplication");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVectorNx1<double, 5> vector(v);
        maths::CVectorNx1<double, 5> vs = vector * 3.0;
        LOG_DEBUG(<< "3 * v = " << vs);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i + 1)), vs(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVectorNx1<double, 5> vector(v);
        maths::CVectorNx1<double, 5> vs = vector / 4.0;
        LOG_DEBUG(<< "v / 4.0 = " << vs);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(static_cast<double>((i + 1)) / 4.0, vs(i));
        }
    }
}

void CLinearAlgebraTest::testSymmetricMatrix() {
    // Construction.
    {
        maths::CSymmetricMatrix<double> matrix(3);
        LOG_DEBUG(<< "matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), matrix.columns());
        for (std::size_t i = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_EQUAL(0.0, matrix(i, j));
            }
        }
    }
    {
        maths::CSymmetricMatrix<double> matrix(4, 3.0);
        LOG_DEBUG(<< "matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.columns());
        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_EQUAL(3.0, matrix(i, j));
            }
        }
    }
    {
        double m_[][5]{{1.1, 2.4, 1.4, 3.7, 4.0},
                       {2.4, 3.2, 1.8, 0.7, 1.0},
                       {1.4, 1.8, 0.8, 4.7, 3.1},
                       {3.7, 0.7, 4.7, 4.7, 1.1},
                       {4.0, 1.0, 3.1, 1.1, 1.0}};
        TDoubleVecVec m(5, TDoubleVec(5));
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                m[i][j] = m_[i][j];
            }
        }
        maths::CSymmetricMatrix<double> matrix(m);
        LOG_DEBUG(<< "matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.columns());
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(m[i][j], matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(10.8, matrix.trace());
    }
    {
        double m[]{1.1, 2.4, 3.2, 1.4, 1.8, 0.8, 3.7, 0.7,
                   4.7, 4.7, 4.0, 1.0, 3.1, 1.1, 1.0};
        maths::CSymmetricMatrix<double> matrix(boost::begin(m), boost::end(m));
        LOG_DEBUG(<< "matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), matrix.columns());
        for (std::size_t i = 0u, k = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++k) {
                CPPUNIT_ASSERT_EQUAL(m[k], matrix(i, j));
                CPPUNIT_ASSERT_EQUAL(m[k], matrix(j, i));
            }
        }
        CPPUNIT_ASSERT_EQUAL(10.8, matrix.trace());
    }
    {
        double v[]{1.0, 2.0, 3.0, 4.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> matrix(maths::E_OuterProduct, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.columns());
        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i + 1) * (j + 1)), matrix(i, j));
            }
        }
        CPPUNIT_ASSERT_EQUAL(30.0, matrix.trace());
    }
    {
        double v[]{1.0, 2.0, 3.0, 4.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> matrix(maths::E_Diagonal, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.rows());
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), matrix.columns());
        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_EQUAL(i == j ? vector(i) : 0.0, matrix(i, j));
            }
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        double m[]{1.1, 2.4, 3.2, 1.4, 1.8, 0.8, 3.7, 0.7,
                   4.7, 4.7, 4.0, 1.0, 3.1, 1.1, 1.0};
        maths::CSymmetricMatrix<double> matrix(boost::begin(m), boost::end(m));
        maths::CSymmetricMatrix<double> sum = matrix + matrix;
        LOG_DEBUG(<< "sum = " << sum);
        for (std::size_t i = 0u, k = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++k) {
                CPPUNIT_ASSERT_EQUAL(2.0 * m[k], sum(i, j));
                CPPUNIT_ASSERT_EQUAL(2.0 * m[k], sum(j, i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        double m1[]{1.1, 0.4, 1.2, 1.4, 1.8, 0.8};
        double m2[]{2.1, 0.3, 1.2, 0.4, 3.8, 0.2};
        maths::CSymmetricMatrix<double> matrix1(boost::begin(m1), boost::end(m1));
        maths::CSymmetricMatrix<double> matrix2(boost::begin(m2), boost::end(m2));
        maths::CSymmetricMatrix<double> difference = matrix1 - matrix2;
        LOG_DEBUG(<< "difference = " << difference);
        for (std::size_t i = 0u, k = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++k) {
                CPPUNIT_ASSERT_EQUAL(m1[k] - m2[k], difference(i, j));
                CPPUNIT_ASSERT_EQUAL(m1[k] - m2[k], difference(j, i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Multiplication");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrix<double> ms = m * 3.0;
        LOG_DEBUG(<< "3 * m = " << ms);
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i + 1) * (j + 1)),
                                     ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CSymmetricMatrix<double> m(maths::E_OuterProduct, vector);
        maths::CSymmetricMatrix<double> ms = m / 4.0;
        LOG_DEBUG(<< "m / 4.0 = " << ms);
        for (std::size_t i = 0u; i < 5; ++i) {
            for (std::size_t j = 0u; j < 5; ++j) {
                CPPUNIT_ASSERT_EQUAL(static_cast<double>((i + 1) * (j + 1)) / 4.0,
                                     ms(i, j));
            }
        }
    }
}

void CLinearAlgebraTest::testVector() {
    // Construction.
    {
        maths::CVector<double> vector(3);
        LOG_DEBUG(<< "vector = " << vector);
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), vector.dimension());
        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_EQUAL(0.0, vector(i));
        }
    }
    {
        maths::CVector<double> vector(4, 3.0);
        LOG_DEBUG(<< "vector = " << vector);
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), vector.dimension());
        for (std::size_t i = 0u; i < 4; ++i) {
            CPPUNIT_ASSERT_EQUAL(3.0, vector(i));
        }
    }
    {
        double v_[]{1.1, 2.4, 1.4, 3.7, 4.0};
        TDoubleVec v(boost::begin(v_), boost::end(v_));
        maths::CVector<double> vector(v);
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), vector.dimension());
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(v[i], vector(i));
        }
    }
    {
        double v[]{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), vector.dimension());
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(v[i], vector(i));
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        double v[]{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CVector<double> sum = vector + vector;
        LOG_DEBUG(<< "vector = " << sum);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(2.0 * v[i], sum(i));
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        double v1[]{1.1, 0.4, 1.4};
        double v2[]{2.1, 0.3, 0.4};
        maths::CVector<double> vector1(boost::begin(v1), boost::end(v1));
        maths::CVector<double> vector2(boost::begin(v2), boost::end(v2));
        maths::CVector<double> difference = vector1 - vector2;
        LOG_DEBUG(<< "vector = " << difference);
        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_EQUAL(v1[i] - v2[i], difference(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Vector Multiplication");

        Eigen::MatrixXd randomMatrix(std::size_t(4), std::size_t(4));
        Eigen::VectorXd randomVector(4);
        for (std::size_t t = 0u; t < 20; ++t) {
            randomMatrix.setRandom();
            Eigen::MatrixXd a = randomMatrix.selfadjointView<Eigen::Lower>();
            LOG_DEBUG(<< "A   =\n" << a);
            randomVector.setRandom();
            LOG_DEBUG(<< "x   =\n" << randomVector);
            Eigen::VectorXd expected = a * randomVector;
            LOG_DEBUG(<< "Ax =\n" << expected);

            maths::CSymmetricMatrix<double> s(maths::fromDenseMatrix(randomMatrix));
            LOG_DEBUG(<< "S   = " << s);
            maths::CVector<double> y(maths::fromDenseVector(randomVector));
            LOG_DEBUG(<< "y   =\n" << y);
            maths::CVector<double> sy = s * y;
            LOG_DEBUG(<< "Sy = " << sy);
            for (std::size_t i = 0u; i < 4; ++i) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected(i), sy(i), 1e-10);
            }
        }
    }
    {
        LOG_DEBUG(<< "Vector Scalar Multiplication");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CVector<double> vs = vector * 3.0;
        LOG_DEBUG(<< "3 * v = " << vs);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(3.0 * static_cast<double>((i + 1)), vs(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        double v[]{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::CVector<double> vector(boost::begin(v), boost::end(v));
        maths::CVector<double> vs = vector / 4.0;
        LOG_DEBUG(<< "v / 4.0 = " << vs);
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL(static_cast<double>((i + 1)) / 4.0, vs(i));
        }
    }
}

void CLinearAlgebraTest::testNorms() {
    double v[][5]{{1.0, 2.1, 3.2, 1.7, 0.1},
                  {0.0, -2.1, 1.2, 1.9, 4.1},
                  {-1.0, 7.1, 5.2, 1.7, -0.1},
                  {-3.0, 1.1, -3.3, 1.8, 6.1}};
    double expectedEuclidean[]{4.30697, 5.12543, 9.01942, 7.84538};

    for (std::size_t i = 0u; i < boost::size(v); ++i) {
        maths::CVectorNx1<double, 5> v_(v[i]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedEuclidean[i], v_.euclidean(), 5e-6);
    }

    double m[][15]{
        {1.0, 2.1, 3.2, 1.7, 0.1, 4.2, 0.3, 2.8, 4.1, 0.1, 0.4, 1.2, 5.2, 0.2, 6.3},
        {0.0, -2.1, 1.2, 1.9, 4.1, 4.5, -3.1, 0.0, 1.3, 7.5, 0.2, 1.0, 4.5, 8.1, 0.3},
        {-1.0, 7.1, 5.2, 1.7, -0.1, 3.2, 1.8, -3.2, 4.2, 9.1, 0.2, 0.4, 4.1, 7.2, 1.3},
        {-3.0, 1.1, -3.3, 1.8, 6.1, -1.3, 1.3, 4.2, 3.1, 1.9, -2.3, 3.1, 2.4, 2.3, 1.0}};
    double expectedFrobenius[]{13.78550, 18.00250, 20.72052, 14.80844};

    for (std::size_t i = 0u; i < boost::size(m); ++i) {
        maths::CSymmetricMatrixNxN<double, 5> m_(boost::begin(m[i]), boost::end(m[i]));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFrobenius[i], m_.frobenius(), 5e-6);
    }
}

void CLinearAlgebraTest::testUtils() {
    // Test component min, max, sqrt and fabs.
    {
        LOG_DEBUG(<< "Vector min, max, fabs, sqrt");

        const double v1_[]{1.0, 3.1, 2.2, 4.9, 12.0};
        maths::CVectorNx1<double, 5> v1(v1_);
        const double v2_[]{1.5, 3.0, 2.7, 5.2, 8.0};
        maths::CVectorNx1<double, 5> v2(v2_);
        const double v3_[]{-1.0, 3.1, -2.2, -4.9, 12.0};
        maths::CVectorNx1<double, 5> v3(v3_);

        {
            double expected[]{1.0, 3.1, 2.2, 4.0, 4.0};
            LOG_DEBUG(<< "min(v1, 4.0) = " << maths::min(v1, 4.0));
            for (std::size_t i = 0u; i < 5; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected[i], (maths::min(v1, 4.0))(i));
            }
        }
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL((maths::min(v1, 4.0))(i), (maths::min(4.0, v1))(i));
        }
        {
            double expected[]{1.0, 3.0, 2.2, 4.9, 8.0};
            LOG_DEBUG(<< "min(v1, v2) = " << maths::min(v1, v2));
            for (std::size_t i = 0u; i < 5; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected[i], (maths::min(v1, v2))(i));
            }
        }

        {
            double expected[]{3.0, 3.1, 3.0, 4.9, 12.0};
            LOG_DEBUG(<< "max(v1, 3.0) = " << maths::max(v1, 3.0));
            for (std::size_t i = 0u; i < 5; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected[i], (maths::max(v1, 3.0))(i));
            }
        }
        for (std::size_t i = 0u; i < 5; ++i) {
            CPPUNIT_ASSERT_EQUAL((maths::max(v1, 3.0))(i), (maths::max(3.0, v1))(i));
        }
        {
            double expected[]{1.5, 3.1, 2.7, 5.2, 12.0};
            LOG_DEBUG(<< "max(v1, v2) = " << maths::max(v1, v2));
            for (std::size_t i = 0u; i < 5; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected[i], (maths::max(v1, v2))(i));
            }
        }

        {
            double expected[]{1.0, std::sqrt(3.1), std::sqrt(2.2),
                              std::sqrt(4.9), std::sqrt(12.0)};
            LOG_DEBUG(<< "sqrt(v1) = " << maths::sqrt(v1));
            for (std::size_t i = 0u; i < 5; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected[i], (maths::sqrt(v1))(i));
            }
        }

        {
            const double expected[]{1.0, 3.1, 2.2, 4.9, 12.0};
            LOG_DEBUG(<< "fabs(v3) = " << maths::fabs(v3));
            for (std::size_t i = 0u; i < 5; ++i) {
                CPPUNIT_ASSERT_EQUAL(expected[i], (maths::fabs(v3))(i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix min, max, fabs, sqrt");

        double m1_[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.8}, {0.4, 3.8, 0.2}};
        maths::CSymmetricMatrixNxN<double, 3> m1(m1_);
        double m2_[][3]{{1.1, 0.4, 1.4}, {0.4, 1.2, 1.8}, {1.4, 1.8, 0.8}};
        maths::CSymmetricMatrixNxN<double, 3> m2(m2_);
        double m3_[][3]{{-2.1, 0.3, 0.4}, {0.3, -1.2, -3.8}, {0.4, -3.8, 0.2}};
        maths::CSymmetricMatrixNxN<double, 3> m3(m3_);

        {
            double expected[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.0}, {0.4, 3.0, 0.2}};
            LOG_DEBUG(<< "min(m1, 3.0) = " << maths::min(m1, 3.0));
            for (std::size_t i = 0u; i < 3; ++i) {
                for (std::size_t j = 0u; j < 3; ++j) {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j], (maths::min(m1, 3.0))(i, j));
                }
            }
        }
        for (std::size_t i = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_EQUAL((maths::min(m1, 3.0))(i, j),
                                     (maths::min(3.0, m1))(i, j));
            }
        }
        {
            double expected[][3]{{1.1, 0.3, 0.4}, {0.3, 1.2, 1.8}, {0.4, 1.8, 0.2}};
            LOG_DEBUG(<< "min(m1, m2) = " << maths::min(m1, m2));
            for (std::size_t i = 0u; i < 3; ++i) {
                for (std::size_t j = 0u; j < 3; ++j) {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j], (maths::min(m1, m2))(i, j));
                }
            }
        }

        {
            double expected[][3]{{2.1, 2.0, 2.0}, {2.0, 2.0, 3.8}, {2.0, 3.8, 2.0}};
            LOG_DEBUG(<< "max(m1, 2.0) = " << maths::max(m1, 2.0));
            for (std::size_t i = 0u; i < 3; ++i) {
                for (std::size_t j = 0u; j < 3; ++j) {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j], (maths::max(m1, 2.0))(i, j));
                }
            }
        }
        for (std::size_t i = 0u; i < 3; ++i) {
            for (std::size_t j = 0u; j < 3; ++j) {
                CPPUNIT_ASSERT_EQUAL((maths::max(m1, 2.0))(i, j),
                                     (maths::max(2.0, m1))(i, j));
            }
        }
        {
            double expected[][3]{{2.1, 0.4, 1.4}, {0.4, 1.2, 3.8}, {1.4, 3.8, 0.8}};
            LOG_DEBUG(<< "max(m1, m2) = " << maths::max(m1, m2));
            for (std::size_t i = 0u; i < 3; ++i) {
                for (std::size_t j = 0u; j < 3; ++j) {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j], (maths::max(m1, m2))(i, j));
                }
            }
        }

        {
            double expected[][3]{{std::sqrt(2.1), std::sqrt(0.3), std::sqrt(0.4)},
                                 {std::sqrt(0.3), std::sqrt(1.2), std::sqrt(3.8)},
                                 {std::sqrt(0.4), std::sqrt(3.8), std::sqrt(0.2)}};
            LOG_DEBUG(<< "sqrt(m1) = " << maths::sqrt(m1));
            for (std::size_t i = 0u; i < 3; ++i) {
                for (std::size_t j = 0u; j < 3; ++j) {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j], (maths::sqrt(m1))(i, j));
                }
            }
        }

        {
            double expected[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.8}, {0.4, 3.8, 0.2}};
            LOG_DEBUG(<< "fabs(m3) = " << maths::fabs(m3));
            for (std::size_t i = 0u; i < 3; ++i) {
                for (std::size_t j = 0u; j < 3; ++j) {
                    CPPUNIT_ASSERT_EQUAL(expected[i][j], (maths::fabs(m3))(i, j));
                }
            }
        }
    }
}

void CLinearAlgebraTest::testScaleCovariances() {
    const double scale_[]{0.8, 1.5, 2.0, 0.5};
    const double covariance_[][4]{{10.0, 0.1, 1.5, 2.0},
                                  {0.1, 11.0, 2.5, 1.9},
                                  {1.5, 2.5, 12.0, 2.4},
                                  {2.0, 1.9, 2.4, 11.5}};

    LOG_DEBUG(<< "CVectorNx1");
    {
        maths::CVectorNx1<double, 4> scale(scale_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(covariance_);
        maths::scaleCovariances(scale, covariance);
        LOG_DEBUG(<< "scaled covariances =" << covariance);

        for (std::size_t i = 0u; i < 4; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(scale_[i] * covariance_[i][i],
                                         covariance(i, i), 1e-10);
            for (std::size_t j = i + 1; j < 4; ++j) {
                double expected = ::sqrt(scale_[i] * scale_[j]) * covariance_[i][j];
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, covariance(i, j), 1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, covariance(j, i), 1e-10);
            }
        }
    }
    LOG_DEBUG(<< "CDenseVector");
    {
        maths::CDenseVector<double> scale(4);
        scale << scale_[0], scale_[1], scale_[2], scale_[3];
        maths::CDenseMatrix<double> covariance(4, 4);
        covariance << covariance_[0][0], covariance_[0][1], covariance_[0][2],
            covariance_[0][3], covariance_[1][0], covariance_[1][1],
            covariance_[1][2], covariance_[1][3], covariance_[2][0],
            covariance_[2][1], covariance_[2][2], covariance_[2][3],
            covariance_[3][0], covariance_[3][1], covariance_[3][2],
            covariance_[3][3];
        maths::scaleCovariances(scale, covariance);
        LOG_DEBUG(<< "scaled covariances =\n" << covariance);

        for (std::size_t i = 0u; i < 4; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(scale_[i] * covariance_[i][i],
                                         covariance(i, i), 1e-10);
            for (std::size_t j = i + 1; j < 4; ++j) {
                double expected = ::sqrt(scale_[i] * scale_[j]) * covariance_[i][j];
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, covariance(i, j), 1e-10);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, covariance(j, i), 1e-10);
            }
        }
    }
}

void CLinearAlgebraTest::testGaussianLogLikelihood() {
    // Test the log likelihood (expected from octave).
    {
        const double covariance_[][4]{{10.70779, 0.14869, 1.44263, 2.26889},
                                      {0.14869, 10.70919, 2.56363, 1.87805},
                                      {1.44263, 2.56363, 11.90966, 2.44121},
                                      {2.26889, 1.87805, 2.44121, 11.53904}};
        maths::CSymmetricMatrixNxN<double, 4> covariance(covariance_);

        const double x_[][4]{{-1.335028, -0.222988, -0.174935, -0.480772},
                             {0.137550, 1.286252, 0.027043, 1.349709},
                             {-0.445561, 2.390953, 0.302770, 0.084871},
                             {0.275802, 0.408910, -2.247157, 0.196043},
                             {0.179101, 0.177340, -0.456634, 5.314863},
                             {0.260426, 0.325159, 1.214650, -1.267697},
                             {-0.363917, -0.422225, 0.360000, 0.401383},
                             {1.492814, 3.257986, 0.065441, -0.187108},
                             {1.214063, 0.067988, -0.241846, -0.425730},
                             {-0.306693, -0.188497, -1.092719, 1.288093}};

        const double expected[]{-8.512128, -8.569778, -8.706920, -8.700537,
                                -9.794163, -8.602336, -8.462027, -9.096402,
                                -8.521042, -8.590054};

        for (std::size_t i = 0u; i < boost::size(x_); ++i) {
            maths::CVectorNx1<double, 4> x(x_[i]);
            double likelihood;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                 maths::gaussianLogLikelihood(covariance, x, likelihood));
            LOG_DEBUG(<< "expected log(L(x)) = " << expected[i]);
            LOG_DEBUG(<< "got      log(L(x)) = " << likelihood);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[i], likelihood, 1e-6);
        }
    }

    // Test log likelihood singular matrix.
    {
        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                         e1 / e1.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e2 / e2.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e3 / e3.euclidean()));

        double likelihood;
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e1, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (3.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0) + 4.0 / 10.0),
            likelihood, 1e-10);

        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e2, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (3.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0) + 2.0 / 5.0),
            likelihood, 1e-10);

        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e3, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (3.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0) + 6.0 / 5.0),
            likelihood, 1e-10);

        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpOverflowed,
                             maths::gaussianLogLikelihood(covariance, e1, likelihood, false));
        CPPUNIT_ASSERT(likelihood > 0.0);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpOverflowed,
                             maths::gaussianLogLikelihood(covariance, e4, likelihood, false));
        CPPUNIT_ASSERT(likelihood < 0.0);
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                         e1 / e1.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e2 / e2.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e3 / e3.euclidean()) +
            2.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e4 / e4.euclidean()));

        double likelihood;
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e1, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 4.0 / 10.0),
            likelihood, 1e-10);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e2, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 2.0 / 5.0),
            likelihood, 1e-10);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e3, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 6.0 / 5.0),
            likelihood, 1e-10);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                             maths::gaussianLogLikelihood(covariance, e4, likelihood));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 12.0 / 2.0),
            likelihood, 1e-10);
    }
}

void CLinearAlgebraTest::testSampleGaussian() {
    // Test singular matrix.
    {
        double m[]{1.0, 2.0, 3.0, 4.0};
        maths::CVectorNx1<double, 4> mean(m);

        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                         e1 / e1.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e2 / e2.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e3 / e3.euclidean()));

        std::vector<maths::CVectorNx1<double, 4>> samples;
        maths::sampleGaussian(100, mean, covariance, samples);

        CPPUNIT_ASSERT_EQUAL(std::size_t(99), samples.size());

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 4>> covariances(4);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            covariances.add(samples[i]);
        }

        LOG_DEBUG(<< "mean       = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);
        LOG_DEBUG(<< "sample mean       = " << maths::CBasicStatistics::mean(covariances));
        LOG_DEBUG(<< "sample covariance = "
                  << maths::CBasicStatistics::maximumLikelihoodCovariances(covariances));

        maths::CVectorNx1<double, 4> meanError =
            maths::CVectorNx1<double, 4>(mean) - maths::CBasicStatistics::mean(covariances);
        maths::CSymmetricMatrixNxN<double, 4> covarianceError =
            maths::CSymmetricMatrixNxN<double, 4>(covariance) -
            maths::CBasicStatistics::maximumLikelihoodCovariances(covariances);

        LOG_DEBUG(<< "|error| / |mean| = " << meanError.euclidean() / mean.euclidean());
        LOG_DEBUG(<< "|error| / |covariance| = "
                  << covarianceError.frobenius() / covariance.frobenius());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meanError.euclidean() / mean.euclidean(), 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0.0, covarianceError.frobenius() / covariance.frobenius(), 0.01);
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double m[]{15.0, 0.0, 1.0, 5.0};
        maths::CVectorNx1<double, 4> mean(m);

        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                         e1 / e1.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e2 / e2.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e3 / e3.euclidean()) +
            2.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e4 / e4.euclidean()));

        std::vector<maths::CVectorNx1<double, 4>> samples;
        maths::sampleGaussian(100, mean, covariance, samples);

        CPPUNIT_ASSERT_EQUAL(std::size_t(100), samples.size());

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 4>> covariances(4);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            covariances.add(samples[i]);
        }

        LOG_DEBUG(<< "mean       = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);
        LOG_DEBUG(<< "sample mean       = " << maths::CBasicStatistics::mean(covariances));
        LOG_DEBUG(<< "sample covariance = "
                  << maths::CBasicStatistics::maximumLikelihoodCovariances(covariances));

        maths::CVectorNx1<double, 4> meanError =
            maths::CVectorNx1<double, 4>(mean) - maths::CBasicStatistics::mean(covariances);
        maths::CSymmetricMatrixNxN<double, 4> covarianceError =
            maths::CSymmetricMatrixNxN<double, 4>(covariance) -
            maths::CBasicStatistics::maximumLikelihoodCovariances(covariances);

        LOG_DEBUG(<< "|error| / |mean| = " << meanError.euclidean() / mean.euclidean());
        LOG_DEBUG(<< "|error| / |covariance| = "
                  << covarianceError.frobenius() / covariance.frobenius());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meanError.euclidean() / mean.euclidean(), 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0.0, covarianceError.frobenius() / covariance.frobenius(), 0.02);
    }
}

void CLinearAlgebraTest::testLogDeterminant() {
    // Test the determinant (expected from octave).
    {
        const double matrices[][3][3]{
            {{0.25451, 0.52345, 0.61308}, {0.52345, 1.19825, 1.12804}, {0.61308, 1.12804, 1.78833}},
            {{0.83654, 0.24520, 0.80310}, {0.24520, 0.38368, 0.30554}, {0.80310, 0.30554, 0.78936}},
            {{0.73063, 0.87818, 0.85836}, {0.87818, 1.50305, 1.17931}, {0.85836, 1.17931, 1.05850}},
            {{0.38947, 0.61062, 0.34423}, {0.61062, 1.60437, 0.91664}, {0.34423, 0.91664, 0.52448}},
            {{1.79563, 1.78751, 2.17200}, {1.78751, 1.83443, 2.17340}, {2.17200, 2.17340, 2.62958}},
            {{0.57023, 0.47992, 0.71581}, {0.47992, 1.09182, 0.97989}, {0.71581, 0.97989, 1.32316}},
            {{2.31264, 0.72098, 2.38050}, {0.72098, 0.28103, 0.78025}, {2.38050, 0.78025, 2.49219}},
            {{0.83678, 0.45230, 0.74564}, {0.45230, 0.26482, 0.33491}, {0.74564, 0.33491, 1.29216}},
            {{0.84991, 0.85443, 0.36922}, {0.85443, 1.12737, 0.83074}, {0.36922, 0.83074, 1.01195}},
            {{0.27156, 0.26441, 0.29726}, {0.26441, 0.32388, 0.18895}, {0.29726, 0.18895, 0.47884}}};

        const double expected[]{5.1523e-03, 6.7423e-04, 4.5641e-04, 1.5880e-04,
                                3.1654e-06, 8.5319e-02, 2.0840e-03, 6.8008e-03,
                                1.4755e-02, 2.6315e-05};

        for (std::size_t i = 0u; i < boost::size(matrices); ++i) {
            maths::CSymmetricMatrixNxN<double, 3> M(matrices[i]);
            double logDeterminant;
            maths::logDeterminant(M, logDeterminant);
            LOG_DEBUG(<< "expected |M| = " << expected[i]);
            LOG_DEBUG(<< "got      |M| = " << std::exp(logDeterminant));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[i], std::exp(logDeterminant),
                                         1e-4 * expected[i]);
        }
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::CVectorNx1<double, 4> e1(e1_);
        maths::CVectorNx1<double, 4> e2(e2_);
        maths::CVectorNx1<double, 4> e3(e3_);
        maths::CVectorNx1<double, 4> e4(e4_);
        maths::CSymmetricMatrixNxN<double, 4> M(
            10.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                         e1 / e1.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e2 / e2.euclidean()) +
            5.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e3 / e3.euclidean()) +
            2.0 * maths::CSymmetricMatrixNxN<double, 4>(maths::E_OuterProduct,
                                                        e4 / e4.euclidean()));
        double logDeterminant;
        maths::logDeterminant(M, logDeterminant);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(10.0 * 5.0 * 5.0 * 2.0), logDeterminant, 1e-10);
    }
}

namespace {

template<typename MATRIX>
std::string print(const MATRIX& m) {
    std::ostringstream result;
    result << m;
    return result.str();
}
}

void CLinearAlgebraTest::testProjected() {
    using TSizeVec = std::vector<std::size_t>;

    const double m[][5]{{1.2, 2.4, 1.9, 3.8, 8.3},
                        {2.4, 1.0, 0.2, 1.6, 3.1},
                        {1.9, 0.2, 8.1, 1.1, 0.1},
                        {3.8, 1.6, 1.1, 3.7, 7.3},
                        {8.3, 3.1, 0.1, 7.3, 0.9}};
    const double v[]{0.3, 3.4, 10.6, 0.9, 5.7};

    maths::CSymmetricMatrixNxN<double, 5> matrix(m);
    maths::CVectorNx1<double, 5> vector(v);

    {
        std::size_t ss[]{0, 1};
        TSizeVec subspace(boost::begin(ss), boost::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::projectedVector(subspace, vector);
        LOG_DEBUG(<< "projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG(<< "projectedVector =\n" << projectedVector);
        CPPUNIT_ASSERT_EQUAL(std::string("1.2 2.4\n2.4   1"), print(projectedMatrix));
        CPPUNIT_ASSERT_EQUAL(std::string("0.3\n3.4"), print(projectedVector));
    }
    {
        std::size_t ss[]{1, 0};
        TSizeVec subspace(boost::begin(ss), boost::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::projectedVector(subspace, vector);
        LOG_DEBUG(<< "projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG(<< "projectedVector =\n" << projectedVector);
        CPPUNIT_ASSERT_EQUAL(std::string("  1 2.4\n2.4 1.2"), print(projectedMatrix));
        CPPUNIT_ASSERT_EQUAL(std::string("3.4\n0.3"), print(projectedVector));
    }
    {
        std::size_t ss[]{1, 0, 4};
        TSizeVec subspace(boost::begin(ss), boost::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::projectedVector(subspace, vector);
        LOG_DEBUG(<< "projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG(<< "projectedVector =\n" << projectedVector);
        CPPUNIT_ASSERT_EQUAL(std::string("  1 2.4 3.1\n2.4 1.2 8.3\n3.1 8.3 0.9"),
                             print(projectedMatrix));
        CPPUNIT_ASSERT_EQUAL(std::string("3.4\n0.3\n5.7"), print(projectedVector));
    }
}

void CLinearAlgebraTest::testShims() {
    using TVector4 = maths::CVectorNx1<double, 4>;
    using TMatrix4 = maths::CSymmetricMatrixNxN<double, 4>;
    using TVector = maths::CVector<double>;
    using TMatrix = maths::CSymmetricMatrix<double>;
    using TDenseVector = maths::CDenseVector<double>;
    using TDenseMatrix = maths::CDenseMatrix<double>;
    using TMappedVector = maths::CMemoryMappedDenseVector<double>;
    using TMappedMatrix = maths::CMemoryMappedDenseMatrix<double>;

    double components[][4]{{1.0, 3.1, 4.0, 1.5},
                           {0.9, 3.2, 2.1, 1.7},
                           {1.3, 1.6, 8.9, 0.2},
                           {-1.3, 2.7, -4.7, 3.1}};
    TVector4 vector1(components[0]);
    TVector vector2(boost::begin(components[0]), boost::end(components[0]));
    TDenseVector vector3(4);
    vector3 << components[0][0], components[0][1], components[0][2], components[0][3];
    TMappedVector vector4(components[0], 4);

    LOG_DEBUG(<< "Test dimension");
    {
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), maths::las::dimension(vector1));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), maths::las::dimension(vector2));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), maths::las::dimension(vector3));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), maths::las::dimension(vector4));
    }
    LOG_DEBUG(<< "Test zero");
    {
        CPPUNIT_ASSERT(TVector4(0.0) == maths::las::zero(vector1));
        CPPUNIT_ASSERT(TVector(4, 0.0) == maths::las::zero(vector2));
        CPPUNIT_ASSERT(TDenseVector::Zero(4) == maths::las::zero(vector3));
        CPPUNIT_ASSERT(TDenseVector::Zero(4) == maths::las::zero(vector4));
    }
    LOG_DEBUG(<< "Test conformableZeroMatrix");
    {
        CPPUNIT_ASSERT(TMatrix4(0.0) == maths::las::conformableZeroMatrix(vector1));
        CPPUNIT_ASSERT(TMatrix(4, 0.0) == maths::las::conformableZeroMatrix(vector2));
        CPPUNIT_ASSERT(TDenseMatrix::Zero(4, 4) == maths::las::conformableZeroMatrix(vector3));
        CPPUNIT_ASSERT(TDenseMatrix::Zero(4, 4) == maths::las::conformableZeroMatrix(vector4));
    }
    LOG_DEBUG(<< "Test isZero");
    {
        CPPUNIT_ASSERT(maths::las::isZero(vector1) == false);
        CPPUNIT_ASSERT(maths::las::isZero(vector2) == false);
        CPPUNIT_ASSERT(maths::las::isZero(vector3) == false);
        CPPUNIT_ASSERT(maths::las::isZero(vector4) == false);
        CPPUNIT_ASSERT(maths::las::isZero(maths::las::zero(vector1)));
        CPPUNIT_ASSERT(maths::las::isZero(maths::las::zero(vector2)));
        CPPUNIT_ASSERT(maths::las::isZero(maths::las::zero(vector3)));
        CPPUNIT_ASSERT(maths::las::isZero(maths::las::zero(vector4)));
    }
    LOG_DEBUG(<< "Test ones");
    {
        CPPUNIT_ASSERT(TVector4(1.0) == maths::las::ones(vector1));
        CPPUNIT_ASSERT(TVector(4, 1.0) == maths::las::ones(vector2));
        CPPUNIT_ASSERT(TDenseVector::Ones(4) == maths::las::ones(vector3));
        CPPUNIT_ASSERT(TDenseVector::Ones(4) == maths::las::ones(vector4));
    }
    LOG_DEBUG(<< "Test constant");
    {
        CPPUNIT_ASSERT(TVector4(5.1) == maths::las::constant(vector1, 5.1));
        CPPUNIT_ASSERT(TVector(4, 5.1) == maths::las::constant(vector2, 5.1));
        CPPUNIT_ASSERT(5.1 * TDenseVector::Ones(4) == maths::las::constant(vector3, 5.1));
        CPPUNIT_ASSERT(5.1 * TDenseVector::Ones(4) == maths::las::constant(vector4, 5.1));
    }
    LOG_DEBUG(<< "Test min");
    {
        double placeholder[2][4];
        std::copy(boost::begin(components[1]), boost::end(components[1]),
                  boost::begin(placeholder[0]));
        std::copy(boost::begin(components[2]), boost::end(components[2]),
                  boost::begin(placeholder[1]));
        TVector4 vector5(components[1]);
        TVector vector6(boost::begin(components[1]), boost::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(placeholder[0], 4);
        TVector4 vector9(boost::begin(components[2]), boost::end(components[2]));
        TVector vector10(boost::begin(components[2]), boost::end(components[2]));
        TDenseVector vector11(4);
        vector11 << components[2][0], components[2][1], components[2][2],
            components[2][3];
        TMappedVector vector12(placeholder[1], 4);
        maths::las::min(vector1, vector5);
        maths::las::min(vector2, vector6);
        maths::las::min(vector3, vector7);
        maths::las::min(vector4, vector8);
        maths::las::min(vector1, vector9);
        maths::las::min(vector2, vector10);
        maths::las::min(vector3, vector11);
        maths::las::min(vector4, vector12);
        LOG_DEBUG(<< " min1 = " << core::CContainerPrinter::print(vector5));
        LOG_DEBUG(<< " min1 = " << core::CContainerPrinter::print(vector6));
        LOG_DEBUG(<< " min1 = " << vector7.transpose());
        LOG_DEBUG(<< " min1 = " << vector8.transpose());
        LOG_DEBUG(<< " min2 = " << core::CContainerPrinter::print(vector9));
        LOG_DEBUG(<< " min2 = " << core::CContainerPrinter::print(vector10));
        LOG_DEBUG(<< " min2 = " << vector11.transpose());
        LOG_DEBUG(<< " min2 = " << vector12.transpose());
        std::ostringstream o7;
        o7 << vector7.transpose();
        std::ostringstream o8;
        o8 << vector8.transpose();
        std::ostringstream o11;
        o11 << vector11.transpose();
        std::ostringstream o12;
        o12 << vector12.transpose();
        CPPUNIT_ASSERT_EQUAL(std::string("[0.9, 3.1, 2.1, 1.5]"),
                             core::CContainerPrinter::print(vector5));
        CPPUNIT_ASSERT_EQUAL(std::string("[0.9, 3.1, 2.1, 1.5]"),
                             core::CContainerPrinter::print(vector6));
        CPPUNIT_ASSERT_EQUAL(std::string("0.9 3.1 2.1 1.5"), o7.str());
        CPPUNIT_ASSERT_EQUAL(std::string("0.9 3.1 2.1 1.5"), o8.str());
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 1.6, 4, 0.2]"),
                             core::CContainerPrinter::print(vector9));
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 1.6, 4, 0.2]"),
                             core::CContainerPrinter::print(vector10));
        CPPUNIT_ASSERT_EQUAL(std::string("  1 1.6   4 0.2"), o11.str());
        CPPUNIT_ASSERT_EQUAL(std::string("  1 1.6   4 0.2"), o12.str());
    }
    LOG_DEBUG(<< "Test max");
    {
        double placeholder[2][4];
        std::copy(boost::begin(components[1]), boost::end(components[1]),
                  boost::begin(placeholder[0]));
        std::copy(boost::begin(components[2]), boost::end(components[2]),
                  boost::begin(placeholder[1]));
        TVector4 vector5(components[1]);
        TVector vector6(boost::begin(components[1]), boost::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(placeholder[0], 4);
        TVector4 vector9(components[2]);
        TVector vector10(boost::begin(components[2]), boost::end(components[2]));
        TDenseVector vector11(4);
        vector11 << components[2][0], components[2][1], components[2][2],
            components[2][3];
        TMappedVector vector12(placeholder[1], 4);
        maths::las::max(vector1, vector5);
        maths::las::max(vector2, vector6);
        maths::las::max(vector3, vector7);
        maths::las::max(vector4, vector8);
        maths::las::max(vector1, vector9);
        maths::las::max(vector2, vector10);
        maths::las::max(vector3, vector11);
        maths::las::max(vector4, vector12);
        LOG_DEBUG(<< " max1 = " << core::CContainerPrinter::print(vector5));
        LOG_DEBUG(<< " max1 = " << core::CContainerPrinter::print(vector6));
        LOG_DEBUG(<< " max1 = " << vector7.transpose());
        LOG_DEBUG(<< " max1 = " << vector8.transpose());
        LOG_DEBUG(<< " max2 = " << core::CContainerPrinter::print(vector9));
        LOG_DEBUG(<< " max2 = " << core::CContainerPrinter::print(vector10));
        LOG_DEBUG(<< " max2 = " << vector11.transpose());
        LOG_DEBUG(<< " max2 = " << vector12.transpose());
        std::ostringstream o7;
        o7 << vector7.transpose();
        std::ostringstream o8;
        o8 << vector8.transpose();
        std::ostringstream o11;
        o11 << vector11.transpose();
        std::ostringstream o12;
        o12 << vector12.transpose();
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 3.2, 4, 1.7]"),
                             core::CContainerPrinter::print(vector5));
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 3.2, 4, 1.7]"),
                             core::CContainerPrinter::print(vector6));
        CPPUNIT_ASSERT_EQUAL(std::string("  1 3.2   4 1.7"), o7.str());
        CPPUNIT_ASSERT_EQUAL(std::string("  1 3.2   4 1.7"), o8.str());
        CPPUNIT_ASSERT_EQUAL(std::string("[1.3, 3.1, 8.9, 1.5]"),
                             core::CContainerPrinter::print(vector9));
        CPPUNIT_ASSERT_EQUAL(std::string("[1.3, 3.1, 8.9, 1.5]"),
                             core::CContainerPrinter::print(vector10));
        CPPUNIT_ASSERT_EQUAL(std::string("1.3 3.1 8.9 1.5"), o11.str());
        CPPUNIT_ASSERT_EQUAL(std::string("1.3 3.1 8.9 1.5"), o12.str());
    }
    LOG_DEBUG(<< "Test componentwise divide");
    {
        TVector4 vector5(components[1]);
        TVector vector6(boost::begin(components[1]), boost::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(components[1], 4);
        TVector4 d1 = maths::las::componentwise(vector1) / maths::las::componentwise(vector5);
        TVector d2 = maths::las::componentwise(vector2) / maths::las::componentwise(vector6);
        TDenseVector d3 = maths::las::componentwise(vector3) /
                          maths::las::componentwise(vector7);
        TDenseVector d4 = maths::las::componentwise(vector4) /
                          maths::las::componentwise(vector8);
        LOG_DEBUG(<< " v1 / v2 = " << core::CContainerPrinter::print(d1));
        LOG_DEBUG(<< " v1 / v2 = " << core::CContainerPrinter::print(d2));
        LOG_DEBUG(<< " v1 / v2 = " << d3.transpose());
        LOG_DEBUG(<< " v1 / v2 = " << d4.transpose());
        std::ostringstream o3;
        o3 << d3.transpose();
        std::ostringstream o4;
        o4 << d4.transpose();
        CPPUNIT_ASSERT_EQUAL(std::string("[1.111111, 0.96875, 1.904762, 0.8823529]"),
                             core::CContainerPrinter::print(d1));
        CPPUNIT_ASSERT_EQUAL(std::string("[1.111111, 0.96875, 1.904762, 0.8823529]"),
                             core::CContainerPrinter::print(d2));
        CPPUNIT_ASSERT_EQUAL(std::string(" 1.11111  0.96875  1.90476 0.882353"), o3.str());
        CPPUNIT_ASSERT_EQUAL(std::string(" 1.11111  0.96875  1.90476 0.882353"), o4.str());
    }
    LOG_DEBUG(<< "Test distance");
    {
        TVector4 vector5(components[2]);
        TVector vector6(boost::begin(components[2]), boost::end(components[2]));
        TDenseVector vector7(4);
        vector7 << components[2][0], components[2][1], components[2][2],
            components[2][3];
        TMappedVector vector8(components[2], 4);
        double d1 = maths::las::distance(vector1, vector5);
        double d2 = maths::las::distance(vector2, vector6);
        double d3 = maths::las::distance(vector3, vector7);
        double d4 = maths::las::distance(vector4, vector8);
        LOG_DEBUG(<< " d1 = " << d1);
        LOG_DEBUG(<< " d1 = " << d2);
        LOG_DEBUG(<< " d1 = " << d3);
        LOG_DEBUG(<< " d1 = " << d4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.29528091794949, d1, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.29528091794949, d2, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.29528091794949, d3, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.29528091794949, d4, 1e-10);
    }
    LOG_DEBUG(<< "Test Euclidean norm");
    {
        double n1 = maths::las::norm(vector1);
        double n2 = maths::las::norm(vector2);
        double n3 = maths::las::norm(vector3);
        double n4 = maths::las::norm(vector4);
        LOG_DEBUG(<< " n1 = " << n1);
        LOG_DEBUG(<< " n1 = " << n2);
        LOG_DEBUG(<< " n1 = " << n3);
        LOG_DEBUG(<< " n1 = " << n4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.37215040742532, n1, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.37215040742532, n2, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.37215040742532, n3, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.37215040742532, n4, 1e-10);
    }
    LOG_DEBUG(<< "Test L1");
    {
        TVector4 vector5(components[3]);
        TVector vector6(boost::begin(components[3]), boost::end(components[3]));
        TDenseVector vector7(4);
        vector7 << components[3][0], components[3][1], components[3][2],
            components[3][3];
        TMappedVector vector8(components[3], 4);
        double l1 = maths::las::L1(vector5);
        double l2 = maths::las::L1(vector6);
        double l3 = maths::las::L1(vector7);
        double l4 = maths::las::L1(vector8);
        LOG_DEBUG(<< " l1 = " << l1);
        LOG_DEBUG(<< " l1 = " << l2);
        LOG_DEBUG(<< " l1 = " << l3);
        LOG_DEBUG(<< " l1 = " << l4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(11.8, l1, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(11.8, l2, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(11.8, l3, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(11.8, l4, 1e-10);
    }
    LOG_DEBUG(<< "Test Frobenius");
    {
        double elements[][4]{{1.0, 2.3, 2.1, -1.3},
                             {2.3, 5.3, 0.1, -0.8},
                             {2.1, 0.1, 3.1, 0.0},
                             {-1.3, -0.8, 0.0, 0.3}};
        double flat[16];
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                flat[4 * j + i] = elements[i][j];
            }
        }
        TDoubleVecVec elements_;
        elements_.emplace_back(boost::begin(elements[0]), boost::end(elements[0]));
        elements_.emplace_back(boost::begin(elements[1]), boost::end(elements[1]));
        elements_.emplace_back(boost::begin(elements[2]), boost::end(elements[2]));
        elements_.emplace_back(boost::begin(elements[3]), boost::end(elements[3]));
        TMatrix4 matrix1(elements);
        TMatrix matrix2(elements_);
        TDenseMatrix matrix3(4, 4);
        matrix3 << elements[0][0], elements[0][1], elements[0][2], elements[0][3],
            elements[1][0], elements[1][1], elements[1][2], elements[1][3],
            elements[2][0], elements[2][1], elements[2][2], elements[2][3],
            elements[3][0], elements[3][1], elements[3][2], elements[3][3];
        TMappedMatrix matrix4(flat, 4, 4);
        LOG_DEBUG(<< matrix4);
        double f1 = maths::las::frobenius(matrix1);
        double f2 = maths::las::frobenius(matrix2);
        double f3 = maths::las::frobenius(matrix3);
        double f4 = maths::las::frobenius(matrix4);
        LOG_DEBUG(<< " f1 = " << f1);
        LOG_DEBUG(<< " f1 = " << f2);
        LOG_DEBUG(<< " f1 = " << f3);
        LOG_DEBUG(<< " f1 = " << f4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(7.92906047397799, f1, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(7.92906047397799, f2, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(7.92906047397799, f3, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(7.92906047397799, f4, 1e-10);
    }
    LOG_DEBUG(<< "Test inner");
    {
        TVector4 vector5(components[1]);
        TVector vector6(boost::begin(components[1]), boost::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(components[1], 4);
        double i1 = maths::las::inner(vector1, vector5);
        double i2 = maths::las::inner(vector2, vector6);
        double i3 = maths::las::inner(vector3, vector7);
        double i4 = maths::las::inner(vector4, vector8);
        LOG_DEBUG(<< "i1 = " << i1);
        LOG_DEBUG(<< "i1 = " << i2);
        LOG_DEBUG(<< "i1 = " << i3);
        LOG_DEBUG(<< "i1 = " << i4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(21.77, i1, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(21.77, i2, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(21.77, i3, 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(21.77, i4, 1e-10);
    }
    LOG_DEBUG(<< "Test outer");
    {
        TMatrix4 outer1 = maths::las::outer(vector1);
        TMatrix outer2 = maths::las::outer(vector2);
        TDenseMatrix outer3 = maths::las::outer(vector3);
        TDenseMatrix outer4 = maths::las::outer(vector4);
        LOG_DEBUG(<< "outer = " << outer1);
        LOG_DEBUG(<< "outer = " << outer2);
        LOG_DEBUG(<< "outer =\n" << outer3);
        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
                CPPUNIT_ASSERT_EQUAL(vector1(i) * vector1(j), outer1(i, j));
                CPPUNIT_ASSERT_EQUAL(vector2(i) * vector2(j), outer2(i, j));
                CPPUNIT_ASSERT_EQUAL(vector3(i) * vector3(j), outer3(i, j));
                CPPUNIT_ASSERT_EQUAL(vector4(i) * vector4(j), outer4(i, j));
            }
        }
    }
}

void CLinearAlgebraTest::testPersist() {
    // Check conversion to and from delimited is idempotent and parsing
    // bad input produces an error.

    {
        double matrix_[][4]{{1.0, 2.1, 1.5, 0.1},
                            {2.1, 2.2, 3.7, 0.6},
                            {1.5, 3.7, 0.4, 8.1},
                            {0.1, 0.6, 8.1, 4.3}};

        maths::CSymmetricMatrixNxN<double, 4> matrix(matrix_);

        std::string str = matrix.toDelimited();

        maths::CSymmetricMatrixNxN<double, 4> restoredMatrix;
        CPPUNIT_ASSERT(restoredMatrix.fromDelimited(str));

        LOG_DEBUG(<< "delimited = " << str);

        for (std::size_t i = 0u; i < 4; ++i) {
            for (std::size_t j = 0u; j < 4; ++j) {
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
        double vector_[]{11.2, 2.1, 1.5};

        maths::CVectorNx1<double, 3> vector(vector_);

        std::string str = vector.toDelimited();

        maths::CVectorNx1<double, 3> restoredVector;
        CPPUNIT_ASSERT(restoredVector.fromDelimited(str));

        LOG_DEBUG(<< "delimited = " << str);

        for (std::size_t i = 0u; i < 3; ++i) {
            CPPUNIT_ASSERT_EQUAL(vector(i), restoredVector(i));
        }

        CPPUNIT_ASSERT(!restoredVector.fromDelimited(std::string()));

        std::string bad("0.1,0.3,0.5,3");
        CPPUNIT_ASSERT(!restoredVector.fromDelimited(bad));
        bad = "0.1,0.3,a";
        CPPUNIT_ASSERT(!restoredVector.fromDelimited(bad));
    }
    {
        maths::CDenseVector<double> origVector(4);
        origVector << 1.3, 2.4, 3.1, 5.1;

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origVector.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "vector XML representation:\n" << origXml);

        // Restore the XML into a new regression.
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CDenseVector<double> restoredVector;
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            std::bind(&maths::CDenseVector<double>::acceptRestoreTraverser,
                      &restoredVector, std::placeholders::_1)));

        CPPUNIT_ASSERT_EQUAL(origVector.checksum(0), restoredVector.checksum(0));
    }
}

CppUnit::Test* CLinearAlgebraTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CLinearAlgebraTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testSymmetricMatrixNxN", &CLinearAlgebraTest::testSymmetricMatrixNxN));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testVectorNx1", &CLinearAlgebraTest::testVectorNx1));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testSymmetricMatrix", &CLinearAlgebraTest::testSymmetricMatrix));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testVector", &CLinearAlgebraTest::testVector));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testNorms", &CLinearAlgebraTest::testNorms));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testUtils", &CLinearAlgebraTest::testUtils));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testScaleCovariances", &CLinearAlgebraTest::testScaleCovariances));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testGaussianLogLikelihood",
        &CLinearAlgebraTest::testGaussianLogLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testSampleGaussian", &CLinearAlgebraTest::testSampleGaussian));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testLogDeterminant", &CLinearAlgebraTest::testLogDeterminant));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testProjected", &CLinearAlgebraTest::testProjected));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testShims", &CLinearAlgebraTest::testShims));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLinearAlgebraTest>(
        "CLinearAlgebraTest::testPersist", &CLinearAlgebraTest::testPersist));

    return suiteOfTests;
}
