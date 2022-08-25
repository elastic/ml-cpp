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
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsCovariances.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CLinearAlgebraPersist.h>
#include <maths/common/CLinearAlgebraTools.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CLinearAlgebraTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;

BOOST_AUTO_TEST_CASE(testSymmetricMatrixNxN) {
    // Construction.
    {
        maths::common::CSymmetricMatrixNxN<double, 3> matrix;
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_REQUIRE_EQUAL(0.0, matrix(i, j));
            }
        }
        BOOST_REQUIRE_EQUAL(0.0, matrix.trace());
    }
    {
        maths::common::CSymmetricMatrixNxN<double, 3> matrix(3.0);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_REQUIRE_EQUAL(3.0, matrix(i, j));
            }
        }
        BOOST_REQUIRE_EQUAL(9.0, matrix.trace());
    }
    {
        double m[][5]{{1.1, 2.4, 1.4, 3.7, 4.0},
                      {2.4, 3.2, 1.8, 0.7, 1.0},
                      {1.4, 1.8, 0.8, 4.7, 3.1},
                      {3.7, 0.7, 4.7, 4.7, 1.1},
                      {4.0, 1.0, 3.1, 1.1, 1.0}};
        maths::common::CSymmetricMatrixNxN<double, 5> matrix(m);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(m[i][j], matrix(i, j));
            }
        }
        BOOST_REQUIRE_EQUAL(10.8, matrix.trace());
    }
    {
        maths::common::CVectorNx1<double, 4> vector{{1.0, 2.0, 3.0, 4.0}};
        maths::common::CSymmetricMatrixNxN<double, 4> matrix(
            maths::common::E_OuterProduct, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(static_cast<double>((i + 1) * (j + 1)), matrix(i, j));
            }
        }
        BOOST_REQUIRE_EQUAL(30.0, matrix.trace());
    }
    {
        maths::common::CVectorNx1<double, 4> vector{{1.0, 2.0, 3.0, 4.0}};
        maths::common::CSymmetricMatrixNxN<double, 4> matrix(maths::common::E_Diagonal, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(i == j ? vector(i) : 0.0, matrix(i, j));
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
        maths::common::CSymmetricMatrixNxN<double, 5> matrix(m);
        maths::common::CSymmetricMatrixNxN<double, 5> sum = matrix + matrix;
        LOG_DEBUG(<< "sum = " << sum);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(2.0 * m[i][j], sum(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        double m1[][3]{{1.1, 0.4, 1.4}, {0.4, 1.2, 1.8}, {1.4, 1.8, 0.8}};
        double m2[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.8}, {0.4, 3.8, 0.2}};
        maths::common::CSymmetricMatrixNxN<double, 3> matrix1(m1);
        maths::common::CSymmetricMatrixNxN<double, 3> matrix2(m2);
        maths::common::CSymmetricMatrixNxN<double, 3> difference = matrix1 - matrix2;
        LOG_DEBUG(<< "difference = " << difference);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_REQUIRE_EQUAL(m1[i][j] - m2[i][j], difference(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Multiplication");

        maths::common::CVectorNx1<double, 5> vector{{1.0, 2.0, 3.0, 4.0, 5.0}};
        maths::common::CSymmetricMatrixNxN<double, 5> m(maths::common::E_OuterProduct, vector);
        maths::common::CSymmetricMatrixNxN<double, 5> ms = m * 3.0;
        LOG_DEBUG(<< "3 * m = " << ms);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(3.0 * static_cast<double>((i + 1) * (j + 1)),
                                    ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        maths::common::CVectorNx1<double, 5> vector{{1.0, 2.0, 3.0, 4.0, 5.0}};
        maths::common::CSymmetricMatrixNxN<double, 5> m(maths::common::E_OuterProduct, vector);
        maths::common::CSymmetricMatrixNxN<double, 5> ms = m / 4.0;
        LOG_DEBUG(<< "m / 4.0 = " << ms);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(static_cast<double>((i + 1) * (j + 1)) / 4.0,
                                    ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Mean");

        double m[][5]{{1.1, 2.4, 1.4, 3.7, 4.0},
                      {2.4, 3.2, 1.8, 0.7, 1.0},
                      {1.4, 1.8, 0.8, 4.7, 3.1},
                      {3.7, 0.7, 4.7, 4.7, 1.1},
                      {4.0, 1.0, 3.1, 1.1, 1.0}};
        maths::common::CSymmetricMatrixNxN<double, 5> matrix(m);

        double expectedMean{
            (5.0 * maths::common::CBasicStatistics::mean({1.1, 3.2, 0.8, 4.7, 1.0}) +
             20.0 * maths::common::CBasicStatistics::mean(
                        {2.4, 1.4, 3.7, 4.0, 1.8, 0.7, 1.0, 4.7, 3.1, 1.1})) /
            25.0};
        BOOST_REQUIRE_CLOSE(expectedMean, matrix.mean(), 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(testVectorNx1) {
    // Construction.
    {
        maths::common::CVectorNx1<double, 3> vector;
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0; i < 3; ++i) {
            BOOST_REQUIRE_EQUAL(0.0, vector(i));
        }
    }
    {
        maths::common::CVectorNx1<double, 3> vector(3.0);
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0; i < 3; ++i) {
            BOOST_REQUIRE_EQUAL(3.0, vector(i));
        }
    }
    {
        double v[]{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::common::CVectorNx1<double, 5> vector(v);
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(v[i], vector(i));
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        maths::common::CVectorNx1<double, 5> vector{{1.1, 2.4, 1.4, 3.7, 4.0}};
        maths::common::CVectorNx1<double, 5> sum = vector + vector;
        LOG_DEBUG(<< "vector = " << sum);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(2.0 * vector(i), sum(i));
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        maths::common::CVectorNx1<double, 3> vector1{{1.1, 0.4, 1.4}};
        maths::common::CVectorNx1<double, 3> vector2{{2.1, 0.3, 0.4}};
        maths::common::CVectorNx1<double, 3> difference = vector1 - vector2;
        LOG_DEBUG(<< "vector = " << difference);
        for (std::size_t i = 0; i < 3; ++i) {
            BOOST_REQUIRE_EQUAL(vector1(i) - vector2(i), difference(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Vector Multiplication");

        Eigen::Matrix4d randomMatrix;
        Eigen::Vector4d randomVector;
        for (std::size_t t = 0; t < 20; ++t) {
            randomMatrix.setRandom();
            Eigen::Matrix4d a = randomMatrix.selfadjointView<Eigen::Lower>();
            LOG_DEBUG(<< "A   =\n" << a);
            randomVector.setRandom();
            LOG_DEBUG(<< "x   =\n" << randomVector);
            Eigen::Vector4d expected = a * randomVector;
            LOG_DEBUG(<< "Ax =\n" << expected);

            maths::common::CSymmetricMatrixNxN<double, 4> s(
                maths::common::fromDenseMatrix(randomMatrix));
            LOG_DEBUG(<< "S   = " << s);
            maths::common::CVectorNx1<double, 4> y(maths::common::fromDenseVector(randomVector));
            LOG_DEBUG(<< "y   =\n" << y);
            maths::common::CVectorNx1<double, 4> sy = s * y;
            LOG_DEBUG(<< "Sy = " << sy);
            for (std::size_t i = 0; i < 4; ++i) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected(i), sy(i), 1e-14);
            }
        }
    }
    {
        LOG_DEBUG(<< "Vector Scalar Multiplication");

        maths::common::CVectorNx1<double, 5> vector{{1.0, 2.0, 3.0, 4.0, 5.0}};
        maths::common::CVectorNx1<double, 5> vs = vector * 3.0;
        LOG_DEBUG(<< "3 * v = " << vs);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(3.0 * static_cast<double>((i + 1)), vs(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        maths::common::CVectorNx1<double, 5> vector({1.0, 2.0, 3.0, 4.0, 5.0});
        maths::common::CVectorNx1<double, 5> vs = vector / 4.0;
        LOG_DEBUG(<< "v / 4.0 = " << vs);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(static_cast<double>((i + 1)) / 4.0, vs(i));
        }
    }
    {
        LOG_DEBUG(<< "Mean");

        maths::common::CVectorNx1<double, 5> vector{{1.0, 2.0, 3.0, 4.0, 5.0}};

        double expectedMean{maths::common::CBasicStatistics::mean({1.0, 2.0, 3.0, 4.0, 5.0})};
        BOOST_REQUIRE_EQUAL(expectedMean, vector.mean());
    }
}

BOOST_AUTO_TEST_CASE(testSymmetricMatrix) {
    // Construction.
    {
        maths::common::CSymmetricMatrix<double> matrix(3);
        LOG_DEBUG(<< "matrix = " << matrix);
        BOOST_REQUIRE_EQUAL(3, matrix.rows());
        BOOST_REQUIRE_EQUAL(3, matrix.columns());
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_REQUIRE_EQUAL(0.0, matrix(i, j));
            }
        }
    }
    {
        maths::common::CSymmetricMatrix<double> matrix(4, 3.0);
        LOG_DEBUG(<< "matrix = " << matrix);
        BOOST_REQUIRE_EQUAL(4, matrix.rows());
        BOOST_REQUIRE_EQUAL(4, matrix.columns());
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(3.0, matrix(i, j));
            }
        }
    }
    {
        TDoubleVecVec m{{1.1, 2.4, 1.4, 3.7, 4.0},
                        {2.4, 3.2, 1.8, 0.7, 1.0},
                        {1.4, 1.8, 0.8, 4.7, 3.1},
                        {3.7, 0.7, 4.7, 4.7, 1.1},
                        {4.0, 1.0, 3.1, 1.1, 1.0}};
        maths::common::CSymmetricMatrix<double> matrix(m);
        LOG_DEBUG(<< "matrix = " << matrix);
        BOOST_REQUIRE_EQUAL(5, matrix.rows());
        BOOST_REQUIRE_EQUAL(5, matrix.columns());
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(m[i][j], matrix(i, j));
            }
        }
        BOOST_REQUIRE_EQUAL(10.8, matrix.trace());
    }
    {
        double m[]{1.1, 2.4, 3.2, 1.4, 1.8, 0.8, 3.7, 0.7,
                   4.7, 4.7, 4.0, 1.0, 3.1, 1.1, 1.0};
        maths::common::CSymmetricMatrix<double> matrix(std::begin(m), std::end(m));
        LOG_DEBUG(<< "matrix = " << matrix);
        BOOST_REQUIRE_EQUAL(5, matrix.rows());
        BOOST_REQUIRE_EQUAL(5, matrix.columns());
        for (std::size_t i = 0, k = 0; i < 5; ++i) {
            for (std::size_t j = 0; j <= i; ++j, ++k) {
                BOOST_REQUIRE_EQUAL(m[k], matrix(i, j));
                BOOST_REQUIRE_EQUAL(m[k], matrix(j, i));
            }
        }
        BOOST_REQUIRE_EQUAL(10.8, matrix.trace());
    }
    {
        TDoubleVec v{1.0, 2.0, 3.0, 4.0};
        maths::common::CVector<double> vector{v};
        maths::common::CSymmetricMatrix<double> matrix(maths::common::E_OuterProduct, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        BOOST_REQUIRE_EQUAL(4, matrix.rows());
        BOOST_REQUIRE_EQUAL(4, matrix.columns());
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(static_cast<double>((i + 1) * (j + 1)), matrix(i, j));
            }
        }
        BOOST_REQUIRE_EQUAL(30.0, matrix.trace());
    }
    {
        TDoubleVec v{1.0, 2.0, 3.0, 4.0};
        maths::common::CVector<double> vector{v};
        maths::common::CSymmetricMatrix<double> matrix(maths::common::E_Diagonal, vector);
        LOG_DEBUG(<< "matrix = " << matrix);
        BOOST_REQUIRE_EQUAL(4, matrix.rows());
        BOOST_REQUIRE_EQUAL(4, matrix.columns());
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(i == j ? vector(i) : 0.0, matrix(i, j));
            }
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        double m[]{1.1, 2.4, 3.2, 1.4, 1.8, 0.8, 3.7, 0.7,
                   4.7, 4.7, 4.0, 1.0, 3.1, 1.1, 1.0};
        maths::common::CSymmetricMatrix<double> matrix(std::begin(m), std::end(m));
        maths::common::CSymmetricMatrix<double> sum = matrix + matrix;
        LOG_DEBUG(<< "sum = " << sum);
        for (std::size_t i = 0, k = 0; i < 5; ++i) {
            for (std::size_t j = 0; j <= i; ++j, ++k) {
                BOOST_REQUIRE_EQUAL(2.0 * m[k], sum(i, j));
                BOOST_REQUIRE_EQUAL(2.0 * m[k], sum(j, i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        double m1[]{1.1, 0.4, 1.2, 1.4, 1.8, 0.8};
        double m2[]{2.1, 0.3, 1.2, 0.4, 3.8, 0.2};
        maths::common::CSymmetricMatrix<double> matrix1(std::begin(m1), std::end(m1));
        maths::common::CSymmetricMatrix<double> matrix2(std::begin(m2), std::end(m2));
        maths::common::CSymmetricMatrix<double> difference = matrix1 - matrix2;
        LOG_DEBUG(<< "difference = " << difference);
        for (std::size_t i = 0u, k = 0; i < 3; ++i) {
            for (std::size_t j = 0; j <= i; ++j, ++k) {
                BOOST_REQUIRE_EQUAL(m1[k] - m2[k], difference(i, j));
                BOOST_REQUIRE_EQUAL(m1[k] - m2[k], difference(j, i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Multiplication");

        TDoubleVec v{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::common::CVector<double> vector{v};
        maths::common::CSymmetricMatrix<double> m(maths::common::E_OuterProduct, vector);
        maths::common::CSymmetricMatrix<double> ms = m * 3.0;
        LOG_DEBUG(<< "3 * m = " << ms);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(3.0 * static_cast<double>((i + 1) * (j + 1)),
                                    ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        TDoubleVec v{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::common::CVector<double> vector{v};
        maths::common::CSymmetricMatrix<double> m(maths::common::E_OuterProduct, vector);
        maths::common::CSymmetricMatrix<double> ms = m / 4.0;
        LOG_DEBUG(<< "m / 4.0 = " << ms);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                BOOST_REQUIRE_EQUAL(static_cast<double>((i + 1) * (j + 1)) / 4.0,
                                    ms(i, j));
            }
        }
    }
    {
        LOG_DEBUG(<< "Mean");

        TDoubleVecVec m{{1.1, 2.4, 1.4, 3.7, 4.0},
                        {2.4, 3.2, 1.8, 0.7, 1.0},
                        {1.4, 1.8, 0.8, 4.7, 3.1},
                        {3.7, 0.7, 4.7, 4.7, 1.1},
                        {4.0, 1.0, 3.1, 1.1, 1.0}};
        maths::common::CSymmetricMatrix<double> matrix(m);

        double expectedMean{
            (5.0 * maths::common::CBasicStatistics::mean({1.1, 3.2, 0.8, 4.7, 1.0}) +
             20.0 * maths::common::CBasicStatistics::mean(
                        {2.4, 1.4, 3.7, 4.0, 1.8, 0.7, 1.0, 4.7, 3.1, 1.1})) /
            25.0};
        BOOST_REQUIRE_CLOSE(expectedMean, matrix.mean(), 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(testVector) {
    // Construction.
    {
        maths::common::CVector<double> vector(3);
        LOG_DEBUG(<< "vector = " << vector);
        BOOST_REQUIRE_EQUAL(3, vector.dimension());
        for (std::size_t i = 0; i < 3; ++i) {
            BOOST_REQUIRE_EQUAL(0.0, vector(i));
        }
    }
    {
        maths::common::CVector<double> vector(4, 3.0);
        LOG_DEBUG(<< "vector = " << vector);
        BOOST_REQUIRE_EQUAL(4, vector.dimension());
        for (std::size_t i = 0; i < 4; ++i) {
            BOOST_REQUIRE_EQUAL(3.0, vector(i));
        }
    }
    {
        TDoubleVec v{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::common::CVector<double> vector(v);
        BOOST_REQUIRE_EQUAL(5, vector.dimension());
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(v[i], vector(i));
        }
    }
    {
        TDoubleVec v{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::common::CVector<double> vector{v.begin(), v.end()};
        BOOST_REQUIRE_EQUAL(5, vector.dimension());
        LOG_DEBUG(<< "vector = " << vector);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(v[i], vector(i));
        }
    }

    // Operators
    {
        LOG_DEBUG(<< "Sum");

        TDoubleVec v{1.1, 2.4, 1.4, 3.7, 4.0};
        maths::common::CVector<double> vector{v};
        maths::common::CVector<double> sum = vector + vector;
        LOG_DEBUG(<< "vector = " << sum);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(2.0 * v[i], sum(i));
        }
    }
    {
        LOG_DEBUG(<< "Difference");

        TDoubleVec v1{1.1, 0.4, 1.4};
        TDoubleVec v2{2.1, 0.3, 0.4};
        maths::common::CVector<double> vector1{v1};
        maths::common::CVector<double> vector2{v2};
        maths::common::CVector<double> difference = vector1 - vector2;
        LOG_DEBUG(<< "vector = " << difference);
        for (std::size_t i = 0; i < 3; ++i) {
            BOOST_REQUIRE_EQUAL(v1[i] - v2[i], difference(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Vector Multiplication");

        Eigen::MatrixXd randomMatrix(4, 4);
        Eigen::VectorXd randomVector(4);
        for (std::size_t t = 0; t < 20; ++t) {
            randomMatrix.setRandom();
            Eigen::MatrixXd a = randomMatrix.selfadjointView<Eigen::Lower>();
            LOG_DEBUG(<< "A   =\n" << a);
            randomVector.setRandom();
            LOG_DEBUG(<< "x   =\n" << randomVector);
            Eigen::VectorXd expected = a * randomVector;
            LOG_DEBUG(<< "Ax =\n" << expected);

            maths::common::CSymmetricMatrix<double> s(
                maths::common::fromDenseMatrix(randomMatrix));
            LOG_DEBUG(<< "S   = " << s);
            maths::common::CVector<double> y(maths::common::fromDenseVector(randomVector));
            LOG_DEBUG(<< "y   =\n" << y);
            maths::common::CVector<double> sy = s * y;
            LOG_DEBUG(<< "Sy = " << sy);
            for (std::size_t i = 0; i < 4; ++i) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected(i), sy(i), 1e-10);
            }
        }
    }
    {
        LOG_DEBUG(<< "Vector Scalar Multiplication");

        TDoubleVec v{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::common::CVector<double> vector{v};
        maths::common::CVector<double> vs = vector * 3.0;
        LOG_DEBUG(<< "3 * v = " << vs);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(3.0 * static_cast<double>((i + 1)), vs(i));
        }
    }
    {
        LOG_DEBUG(<< "Matrix Scalar Division");

        TDoubleVec v{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::common::CVector<double> vector{v};
        maths::common::CVector<double> vs = vector / 4.0;
        LOG_DEBUG(<< "v / 4.0 = " << vs);
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL(static_cast<double>((i + 1)) / 4.0, vs(i));
        }
    }
    {
        LOG_DEBUG(<< "Mean");

        TDoubleVec v{1.0, 2.0, 3.0, 4.0, 5.0};
        maths::common::CVector<double> vector{v};

        double expectedMean{maths::common::CBasicStatistics::mean({1.0, 2.0, 3.0, 4.0, 5.0})};
        BOOST_REQUIRE_EQUAL(expectedMean, vector.mean());
    }
}

BOOST_AUTO_TEST_CASE(testNorms) {
    double v[][5]{{1.0, 2.1, 3.2, 1.7, 0.1},
                  {0.0, -2.1, 1.2, 1.9, 4.1},
                  {-1.0, 7.1, 5.2, 1.7, -0.1},
                  {-3.0, 1.1, -3.3, 1.8, 6.1}};
    double expectedEuclidean[]{4.30697, 5.12543, 9.01942, 7.84538};

    for (std::size_t i = 0; i < std::size(v); ++i) {
        maths::common::CVectorNx1<double, 5> v_(v[i]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedEuclidean[i], v_.euclidean(), 5e-6);
    }

    double m[][15]{
        {1.0, 2.1, 3.2, 1.7, 0.1, 4.2, 0.3, 2.8, 4.1, 0.1, 0.4, 1.2, 5.2, 0.2, 6.3},
        {0.0, -2.1, 1.2, 1.9, 4.1, 4.5, -3.1, 0.0, 1.3, 7.5, 0.2, 1.0, 4.5, 8.1, 0.3},
        {-1.0, 7.1, 5.2, 1.7, -0.1, 3.2, 1.8, -3.2, 4.2, 9.1, 0.2, 0.4, 4.1, 7.2, 1.3},
        {-3.0, 1.1, -3.3, 1.8, 6.1, -1.3, 1.3, 4.2, 3.1, 1.9, -2.3, 3.1, 2.4, 2.3, 1.0}};
    double expectedFrobenius[]{13.78550, 18.00250, 20.72052, 14.80844};

    for (std::size_t i = 0; i < std::size(m); ++i) {
        maths::common::CSymmetricMatrixNxN<double, 5> m_(std::begin(m[i]),
                                                         std::end(m[i]));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedFrobenius[i], m_.frobenius(), 5e-6);
    }
}

BOOST_AUTO_TEST_CASE(testUtils) {
    // Test component min, max, sqrt and fabs.
    {
        LOG_DEBUG(<< "Vector min, max, fabs, sqrt");

        const double v1_[]{1.0, 3.1, 2.2, 4.9, 12.0};
        maths::common::CVectorNx1<double, 5> v1(v1_);
        const double v2_[]{1.5, 3.0, 2.7, 5.2, 8.0};
        maths::common::CVectorNx1<double, 5> v2(v2_);
        const double v3_[]{-1.0, 3.1, -2.2, -4.9, 12.0};
        maths::common::CVectorNx1<double, 5> v3(v3_);

        {
            double expected[]{1.0, 3.1, 2.2, 4.0, 4.0};
            LOG_DEBUG(<< "min(v1, 4.0) = " << maths::common::min(v1, 4.0));
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_EQUAL(expected[i], (maths::common::min(v1, 4.0))(i));
            }
        }
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL((maths::common::min(v1, 4.0))(i),
                                (maths::common::min(4.0, v1))(i));
        }
        {
            double expected[]{1.0, 3.0, 2.2, 4.9, 8.0};
            LOG_DEBUG(<< "min(v1, v2) = " << maths::common::min(v1, v2));
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_EQUAL(expected[i], (maths::common::min(v1, v2))(i));
            }
        }

        {
            double expected[]{3.0, 3.1, 3.0, 4.9, 12.0};
            LOG_DEBUG(<< "max(v1, 3.0) = " << maths::common::max(v1, 3.0));
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_EQUAL(expected[i], (maths::common::max(v1, 3.0))(i));
            }
        }
        for (std::size_t i = 0; i < 5; ++i) {
            BOOST_REQUIRE_EQUAL((maths::common::max(v1, 3.0))(i),
                                (maths::common::max(3.0, v1))(i));
        }
        {
            double expected[]{1.5, 3.1, 2.7, 5.2, 12.0};
            LOG_DEBUG(<< "max(v1, v2) = " << maths::common::max(v1, v2));
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_EQUAL(expected[i], (maths::common::max(v1, v2))(i));
            }
        }

        {
            double expected[]{1.0, std::sqrt(3.1), std::sqrt(2.2),
                              std::sqrt(4.9), std::sqrt(12.0)};
            LOG_DEBUG(<< "sqrt(v1) = " << maths::common::sqrt(v1));
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_EQUAL(expected[i], (maths::common::sqrt(v1))(i));
            }
        }

        {
            const double expected[]{1.0, 3.1, 2.2, 4.9, 12.0};
            LOG_DEBUG(<< "fabs(v3) = " << maths::common::fabs(v3));
            for (std::size_t i = 0; i < 5; ++i) {
                BOOST_REQUIRE_EQUAL(expected[i], (maths::common::fabs(v3))(i));
            }
        }
    }
    {
        LOG_DEBUG(<< "Matrix min, max, fabs, sqrt");

        double m1_[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.8}, {0.4, 3.8, 0.2}};
        maths::common::CSymmetricMatrixNxN<double, 3> m1(m1_);
        double m2_[][3]{{1.1, 0.4, 1.4}, {0.4, 1.2, 1.8}, {1.4, 1.8, 0.8}};
        maths::common::CSymmetricMatrixNxN<double, 3> m2(m2_);
        double m3_[][3]{{-2.1, 0.3, 0.4}, {0.3, -1.2, -3.8}, {0.4, -3.8, 0.2}};
        maths::common::CSymmetricMatrixNxN<double, 3> m3(m3_);

        {
            double expected[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.0}, {0.4, 3.0, 0.2}};
            LOG_DEBUG(<< "min(m1, 3.0) = " << maths::common::min(m1, 3.0));
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    BOOST_REQUIRE_EQUAL(expected[i][j],
                                        (maths::common::min(m1, 3.0))(i, j));
                }
            }
        }
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_REQUIRE_EQUAL((maths::common::min(m1, 3.0))(i, j),
                                    (maths::common::min(3.0, m1))(i, j));
            }
        }
        {
            double expected[][3]{{1.1, 0.3, 0.4}, {0.3, 1.2, 1.8}, {0.4, 1.8, 0.2}};
            LOG_DEBUG(<< "min(m1, m2) = " << maths::common::min(m1, m2));
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    BOOST_REQUIRE_EQUAL(expected[i][j],
                                        (maths::common::min(m1, m2))(i, j));
                }
            }
        }

        {
            double expected[][3]{{2.1, 2.0, 2.0}, {2.0, 2.0, 3.8}, {2.0, 3.8, 2.0}};
            LOG_DEBUG(<< "max(m1, 2.0) = " << maths::common::max(m1, 2.0));
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    BOOST_REQUIRE_EQUAL(expected[i][j],
                                        (maths::common::max(m1, 2.0))(i, j));
                }
            }
        }
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                BOOST_REQUIRE_EQUAL((maths::common::max(m1, 2.0))(i, j),
                                    (maths::common::max(2.0, m1))(i, j));
            }
        }
        {
            double expected[][3]{{2.1, 0.4, 1.4}, {0.4, 1.2, 3.8}, {1.4, 3.8, 0.8}};
            LOG_DEBUG(<< "max(m1, m2) = " << maths::common::max(m1, m2));
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    BOOST_REQUIRE_EQUAL(expected[i][j],
                                        (maths::common::max(m1, m2))(i, j));
                }
            }
        }

        {
            double expected[][3]{{std::sqrt(2.1), std::sqrt(0.3), std::sqrt(0.4)},
                                 {std::sqrt(0.3), std::sqrt(1.2), std::sqrt(3.8)},
                                 {std::sqrt(0.4), std::sqrt(3.8), std::sqrt(0.2)}};
            LOG_DEBUG(<< "sqrt(m1) = " << maths::common::sqrt(m1));
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    BOOST_REQUIRE_EQUAL(expected[i][j], (maths::common::sqrt(m1))(i, j));
                }
            }
        }

        {
            double expected[][3]{{2.1, 0.3, 0.4}, {0.3, 1.2, 3.8}, {0.4, 3.8, 0.2}};
            LOG_DEBUG(<< "fabs(m3) = " << maths::common::fabs(m3));
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    BOOST_REQUIRE_EQUAL(expected[i][j], (maths::common::fabs(m3))(i, j));
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testScaleCovariances) {
    const double scale_[]{0.8, 1.5, 2.0, 0.5};
    const double covariance_[][4]{{10.0, 0.1, 1.5, 2.0},
                                  {0.1, 11.0, 2.5, 1.9},
                                  {1.5, 2.5, 12.0, 2.4},
                                  {2.0, 1.9, 2.4, 11.5}};

    LOG_DEBUG(<< "CVectorNx1");
    {
        maths::common::CVectorNx1<double, 4> scale(scale_);
        maths::common::CSymmetricMatrixNxN<double, 4> covariance(covariance_);
        maths::common::scaleCovariances(scale, covariance);
        LOG_DEBUG(<< "scaled covariances =" << covariance);

        for (std::size_t i = 0; i < 4; ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(scale_[i] * covariance_[i][i],
                                         covariance(i, i), 1e-10);
            for (std::size_t j = i + 1; j < 4; ++j) {
                double expected = ::sqrt(scale_[i] * scale_[j]) * covariance_[i][j];
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, covariance(i, j), 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, covariance(j, i), 1e-10);
            }
        }
    }
    LOG_DEBUG(<< "CDenseVector");
    {
        maths::common::CDenseVector<double> scale(4);
        scale << scale_[0], scale_[1], scale_[2], scale_[3];
        maths::common::CDenseMatrix<double> covariance(4, 4);
        covariance << covariance_[0][0], covariance_[0][1], covariance_[0][2],
            covariance_[0][3], covariance_[1][0], covariance_[1][1],
            covariance_[1][2], covariance_[1][3], covariance_[2][0],
            covariance_[2][1], covariance_[2][2], covariance_[2][3],
            covariance_[3][0], covariance_[3][1], covariance_[3][2],
            covariance_[3][3];
        maths::common::scaleCovariances(scale, covariance);
        LOG_DEBUG(<< "scaled covariances =\n" << covariance);

        for (std::size_t i = 0; i < 4; ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(scale_[i] * covariance_[i][i],
                                         covariance(i, i), 1e-10);
            for (std::size_t j = i + 1; j < 4; ++j) {
                double expected = ::sqrt(scale_[i] * scale_[j]) * covariance_[i][j];
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, covariance(i, j), 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, covariance(j, i), 1e-10);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testGaussianLogLikelihood) {
    // Test the log likelihood (expected from octave).
    {
        const double covariance_[][4]{{10.70779, 0.14869, 1.44263, 2.26889},
                                      {0.14869, 10.70919, 2.56363, 1.87805},
                                      {1.44263, 2.56363, 11.90966, 2.44121},
                                      {2.26889, 1.87805, 2.44121, 11.53904}};
        maths::common::CSymmetricMatrixNxN<double, 4> covariance(covariance_);

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

        for (std::size_t i = 0; i < std::size(x_); ++i) {
            maths::common::CVectorNx1<double, 4> x(x_[i]);
            double likelihood;
            BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                maths::common::gaussianLogLikelihood(covariance, x, likelihood));
            LOG_DEBUG(<< "expected log(L(x)) = " << expected[i]);
            LOG_DEBUG(<< "got      log(L(x)) = " << likelihood);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected[i], likelihood, 1e-6);
        }
    }

    // Test log likelihood singular matrix.
    {
        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::common::CVectorNx1<double, 4> e1(e1_);
        maths::common::CVectorNx1<double, 4> e2(e2_);
        maths::common::CVectorNx1<double, 4> e3(e3_);
        maths::common::CVectorNx1<double, 4> e4(e4_);
        maths::common::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                       maths::common::E_OuterProduct, e1 / e1.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e2 / e2.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e3 / e3.euclidean()));

        double likelihood;
        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e1, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (3.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0) + 4.0 / 10.0),
            likelihood, 1e-10);

        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e2, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (3.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0) + 2.0 / 5.0),
            likelihood, 1e-10);

        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e3, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (3.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0) + 6.0 / 5.0),
            likelihood, 1e-10);

        BOOST_REQUIRE_EQUAL(
            maths_t::E_FpOverflowed,
            maths::common::gaussianLogLikelihood(covariance, e1, likelihood, false));
        BOOST_TEST_REQUIRE(likelihood > 0.0);
        BOOST_REQUIRE_EQUAL(
            maths_t::E_FpOverflowed,
            maths::common::gaussianLogLikelihood(covariance, e4, likelihood, false));
        BOOST_TEST_REQUIRE(likelihood < 0.0);
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::common::CVectorNx1<double, 4> e1(e1_);
        maths::common::CVectorNx1<double, 4> e2(e2_);
        maths::common::CVectorNx1<double, 4> e3(e3_);
        maths::common::CVectorNx1<double, 4> e4(e4_);
        maths::common::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                       maths::common::E_OuterProduct, e1 / e1.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e2 / e2.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e3 / e3.euclidean()) +
            2.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e4 / e4.euclidean()));

        double likelihood;
        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e1, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 4.0 / 10.0),
            likelihood, 1e-10);
        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e2, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 2.0 / 5.0),
            likelihood, 1e-10);
        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e3, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 6.0 / 5.0),
            likelihood, 1e-10);
        BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                            maths::common::gaussianLogLikelihood(covariance, e4, likelihood));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            -0.5 * (4.0 * std::log(boost::math::double_constants::two_pi) +
                    std::log(10.0 * 5.0 * 5.0 * 2.0) + 12.0 / 2.0),
            likelihood, 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testSampleGaussian) {
    // Test singular matrix.
    {
        double m[]{1.0, 2.0, 3.0, 4.0};
        maths::common::CVectorNx1<double, 4> mean(m);

        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::common::CVectorNx1<double, 4> e1(e1_);
        maths::common::CVectorNx1<double, 4> e2(e2_);
        maths::common::CVectorNx1<double, 4> e3(e3_);
        maths::common::CVectorNx1<double, 4> e4(e4_);
        maths::common::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                       maths::common::E_OuterProduct, e1 / e1.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e2 / e2.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e3 / e3.euclidean()));

        std::vector<maths::common::CVectorNx1<double, 4>> samples;
        maths::common::sampleGaussian(100, mean, covariance, samples);

        BOOST_REQUIRE_EQUAL(99, samples.size());

        maths::common::CBasicStatistics::SSampleCovariances<maths::common::CVectorNx1<double, 4>> covariances(
            4);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            covariances.add(samples[i]);
        }

        LOG_DEBUG(<< "mean       = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);
        LOG_DEBUG(<< "sample mean       = "
                  << maths::common::CBasicStatistics::mean(covariances));
        LOG_DEBUG(<< "sample covariance = "
                  << maths::common::CBasicStatistics::maximumLikelihoodCovariances(covariances));

        maths::common::CVectorNx1<double, 4> meanError =
            maths::common::CVectorNx1<double, 4>(mean) -
            maths::common::CBasicStatistics::mean(covariances);
        maths::common::CSymmetricMatrixNxN<double, 4> covarianceError =
            maths::common::CSymmetricMatrixNxN<double, 4>(covariance) -
            maths::common::CBasicStatistics::maximumLikelihoodCovariances(covariances);

        LOG_DEBUG(<< "|error| / |mean| = " << meanError.euclidean() / mean.euclidean());
        LOG_DEBUG(<< "|error| / |covariance| = "
                  << covarianceError.frobenius() / covariance.frobenius());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, meanError.euclidean() / mean.euclidean(), 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, covarianceError.frobenius() / covariance.frobenius(), 0.01);
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double m[]{15.0, 0.0, 1.0, 5.0};
        maths::common::CVectorNx1<double, 4> mean(m);

        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::common::CVectorNx1<double, 4> e1(e1_);
        maths::common::CVectorNx1<double, 4> e2(e2_);
        maths::common::CVectorNx1<double, 4> e3(e3_);
        maths::common::CVectorNx1<double, 4> e4(e4_);
        maths::common::CSymmetricMatrixNxN<double, 4> covariance(
            10.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                       maths::common::E_OuterProduct, e1 / e1.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e2 / e2.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e3 / e3.euclidean()) +
            2.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e4 / e4.euclidean()));

        std::vector<maths::common::CVectorNx1<double, 4>> samples;
        maths::common::sampleGaussian(100, mean, covariance, samples);

        BOOST_REQUIRE_EQUAL(100, samples.size());

        maths::common::CBasicStatistics::SSampleCovariances<maths::common::CVectorNx1<double, 4>> covariances(
            4);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            covariances.add(samples[i]);
        }

        LOG_DEBUG(<< "mean       = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);
        LOG_DEBUG(<< "sample mean       = "
                  << maths::common::CBasicStatistics::mean(covariances));
        LOG_DEBUG(<< "sample covariance = "
                  << maths::common::CBasicStatistics::maximumLikelihoodCovariances(covariances));

        maths::common::CVectorNx1<double, 4> meanError =
            maths::common::CVectorNx1<double, 4>(mean) -
            maths::common::CBasicStatistics::mean(covariances);
        maths::common::CSymmetricMatrixNxN<double, 4> covarianceError =
            maths::common::CSymmetricMatrixNxN<double, 4>(covariance) -
            maths::common::CBasicStatistics::maximumLikelihoodCovariances(covariances);

        LOG_DEBUG(<< "|error| / |mean| = " << meanError.euclidean() / mean.euclidean());
        LOG_DEBUG(<< "|error| / |covariance| = "
                  << covarianceError.frobenius() / covariance.frobenius());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, meanError.euclidean() / mean.euclidean(), 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            0.0, covarianceError.frobenius() / covariance.frobenius(), 0.02);
    }
}

BOOST_AUTO_TEST_CASE(testLogDeterminant) {
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

        for (std::size_t i = 0; i < std::size(matrices); ++i) {
            maths::common::CSymmetricMatrixNxN<double, 3> M(matrices[i]);
            double logDeterminant;
            maths::common::logDeterminant(M, logDeterminant);
            LOG_DEBUG(<< "expected |M| = " << expected[i]);
            LOG_DEBUG(<< "got      |M| = " << std::exp(logDeterminant));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected[i], std::exp(logDeterminant),
                                         1e-4 * expected[i]);
        }
    }

    // Construct a matrix whose eigenvalues and vectors are known.
    {
        double e1_[]{1.0, 1.0, 1.0, 1.0};
        double e2_[]{-1.0, 1.0, 0.0, 0.0};
        double e3_[]{-1.0, -1.0, 2.0, 0.0};
        double e4_[]{-1.0, -1.0, -1.0, 3.0};
        maths::common::CVectorNx1<double, 4> e1(e1_);
        maths::common::CVectorNx1<double, 4> e2(e2_);
        maths::common::CVectorNx1<double, 4> e3(e3_);
        maths::common::CVectorNx1<double, 4> e4(e4_);
        maths::common::CSymmetricMatrixNxN<double, 4> M(
            10.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                       maths::common::E_OuterProduct, e1 / e1.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e2 / e2.euclidean()) +
            5.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e3 / e3.euclidean()) +
            2.0 * maths::common::CSymmetricMatrixNxN<double, 4>(
                      maths::common::E_OuterProduct, e4 / e4.euclidean()));
        double logDeterminant;
        maths::common::logDeterminant(M, logDeterminant);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(std::log(10.0 * 5.0 * 5.0 * 2.0), logDeterminant, 1e-10);
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

BOOST_AUTO_TEST_CASE(testProjected) {
    using TSizeVec = std::vector<std::size_t>;

    const double m[][5]{{1.2, 2.4, 1.9, 3.8, 8.3},
                        {2.4, 1.0, 0.2, 1.6, 3.1},
                        {1.9, 0.2, 8.1, 1.1, 0.1},
                        {3.8, 1.6, 1.1, 3.7, 7.3},
                        {8.3, 3.1, 0.1, 7.3, 0.9}};
    const double v[]{0.3, 3.4, 10.6, 0.9, 5.7};

    maths::common::CSymmetricMatrixNxN<double, 5> matrix(m);
    maths::common::CVectorNx1<double, 5> vector(v);

    {
        std::size_t ss[]{0, 1};
        TSizeVec subspace(std::begin(ss), std::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::common::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::common::projectedVector(subspace, vector);
        LOG_DEBUG(<< "projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG(<< "projectedVector =\n" << projectedVector);
        BOOST_REQUIRE_EQUAL(std::string("1.2 2.4\n2.4   1"), print(projectedMatrix));
        BOOST_REQUIRE_EQUAL(std::string("0.3\n3.4"), print(projectedVector));
    }
    {
        std::size_t ss[]{1, 0};
        TSizeVec subspace(std::begin(ss), std::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::common::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::common::projectedVector(subspace, vector);
        LOG_DEBUG(<< "projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG(<< "projectedVector =\n" << projectedVector);
        BOOST_REQUIRE_EQUAL(std::string("  1 2.4\n2.4 1.2"), print(projectedMatrix));
        BOOST_REQUIRE_EQUAL(std::string("3.4\n0.3"), print(projectedVector));
    }
    {
        std::size_t ss[]{1, 0, 4};
        TSizeVec subspace(std::begin(ss), std::end(ss));

        Eigen::MatrixXd projectedMatrix = maths::common::projectedMatrix(subspace, matrix);
        Eigen::MatrixXd projectedVector = maths::common::projectedVector(subspace, vector);
        LOG_DEBUG(<< "projectedMatrix =\n" << projectedMatrix);
        LOG_DEBUG(<< "projectedVector =\n" << projectedVector);
        BOOST_REQUIRE_EQUAL(std::string("  1 2.4 3.1\n2.4 1.2 8.3\n3.1 8.3 0.9"),
                            print(projectedMatrix));
        BOOST_REQUIRE_EQUAL(std::string("3.4\n0.3\n5.7"), print(projectedVector));
    }
}

BOOST_AUTO_TEST_CASE(testShims) {
    using TVector4 = maths::common::CVectorNx1<double, 4>;
    using TMatrix4 = maths::common::CSymmetricMatrixNxN<double, 4>;
    using TVector = maths::common::CVector<double>;
    using TMatrix = maths::common::CSymmetricMatrix<double>;
    using TDenseVector = maths::common::CDenseVector<double>;
    using TDenseMatrix = maths::common::CDenseMatrix<double>;
    using TMappedVector = maths::common::CMemoryMappedDenseVector<double>;
    using TMappedMatrix = maths::common::CMemoryMappedDenseMatrix<double>;

    double components[][4]{{1.0, 3.1, 4.0, 1.5},
                           {0.9, 3.2, 2.1, 1.7},
                           {1.3, 1.6, 8.9, 0.2},
                           {-1.3, 2.7, -4.7, 3.1}};
    TVector4 vector1(components[0]);
    TVector vector2(std::begin(components[0]), std::end(components[0]));
    TDenseVector vector3(4);
    vector3 << components[0][0], components[0][1], components[0][2], components[0][3];
    TMappedVector vector4(components[0], 4);

    LOG_DEBUG(<< "Test dimension");
    {
        BOOST_REQUIRE_EQUAL(4, maths::common::las::dimension(vector1));
        BOOST_REQUIRE_EQUAL(4, maths::common::las::dimension(vector2));
        BOOST_REQUIRE_EQUAL(4, maths::common::las::dimension(vector3));
        BOOST_REQUIRE_EQUAL(4, maths::common::las::dimension(vector4));
    }
    LOG_DEBUG(<< "Test zero");
    {
        BOOST_TEST_REQUIRE(TVector4(0.0) == maths::common::las::zero(vector1));
        BOOST_TEST_REQUIRE(TVector(4, 0.0) == maths::common::las::zero(vector2));
        BOOST_TEST_REQUIRE(TDenseVector::Zero(4) == maths::common::las::zero(vector3));
        BOOST_TEST_REQUIRE(TDenseVector::Zero(4) == maths::common::las::zero(vector4));
    }
    LOG_DEBUG(<< "Test conformableZeroMatrix");
    {
        BOOST_TEST_REQUIRE(TMatrix4(0.0) ==
                           maths::common::las::conformableZeroMatrix(vector1));
        BOOST_TEST_REQUIRE((TMatrix(4, 0.0) ==
                            maths::common::las::conformableZeroMatrix(vector2)));
        BOOST_TEST_REQUIRE((TDenseMatrix::Zero(4, 4) ==
                            maths::common::las::conformableZeroMatrix(vector3)));
        BOOST_TEST_REQUIRE((TDenseMatrix::Zero(4, 4) ==
                            maths::common::las::conformableZeroMatrix(vector4)));
    }
    LOG_DEBUG(<< "Test isZero");
    {
        BOOST_TEST_REQUIRE(maths::common::las::isZero(vector1) == false);
        BOOST_TEST_REQUIRE(maths::common::las::isZero(vector2) == false);
        BOOST_TEST_REQUIRE(maths::common::las::isZero(vector3) == false);
        BOOST_TEST_REQUIRE(maths::common::las::isZero(vector4) == false);
        BOOST_TEST_REQUIRE(maths::common::las::isZero(maths::common::las::zero(vector1)));
        BOOST_TEST_REQUIRE(maths::common::las::isZero(maths::common::las::zero(vector2)));
        BOOST_TEST_REQUIRE(maths::common::las::isZero(maths::common::las::zero(vector3)));
        BOOST_TEST_REQUIRE(maths::common::las::isZero(maths::common::las::zero(vector4)));
    }
    LOG_DEBUG(<< "Test ones");
    {
        BOOST_TEST_REQUIRE(TVector4(1.0) == maths::common::las::ones(vector1));
        BOOST_TEST_REQUIRE(TVector(4, 1.0) == maths::common::las::ones(vector2));
        BOOST_TEST_REQUIRE(TDenseVector::Ones(4) == maths::common::las::ones(vector3));
        BOOST_TEST_REQUIRE(TDenseVector::Ones(4) == maths::common::las::ones(vector4));
    }
    LOG_DEBUG(<< "Test constant");
    {
        BOOST_TEST_REQUIRE(TVector4(5.1) == maths::common::las::constant(vector1, 5.1));
        BOOST_TEST_REQUIRE(TVector(4, 5.1) == maths::common::las::constant(vector2, 5.1));
        const auto expected = 5.1 * TDenseVector::Ones(4);
        BOOST_TEST_REQUIRE(expected == maths::common::las::constant(vector3, 5.1));
        BOOST_TEST_REQUIRE(expected == maths::common::las::constant(vector4, 5.1));
    }
    LOG_DEBUG(<< "Test min");
    {
        double placeholder[2][4];
        std::copy(std::begin(components[1]), std::end(components[1]),
                  std::begin(placeholder[0]));
        std::copy(std::begin(components[2]), std::end(components[2]),
                  std::begin(placeholder[1]));
        TVector4 vector5(components[1]);
        TVector vector6(std::begin(components[1]), std::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(placeholder[0], 4);
        TVector4 vector9(std::begin(components[2]), std::end(components[2]));
        TVector vector10(std::begin(components[2]), std::end(components[2]));
        TDenseVector vector11(4);
        vector11 << components[2][0], components[2][1], components[2][2],
            components[2][3];
        TMappedVector vector12(placeholder[1], 4);
        maths::common::las::min(vector1, vector5);
        maths::common::las::min(vector2, vector6);
        maths::common::las::min(vector3, vector7);
        maths::common::las::min(vector4, vector8);
        maths::common::las::min(vector1, vector9);
        maths::common::las::min(vector2, vector10);
        maths::common::las::min(vector3, vector11);
        maths::common::las::min(vector4, vector12);
        LOG_DEBUG(<< " min1 = " << vector5);
        LOG_DEBUG(<< " min1 = " << vector6);
        LOG_DEBUG(<< " min1 = " << vector7.transpose());
        LOG_DEBUG(<< " min1 = " << vector8.transpose());
        LOG_DEBUG(<< " min2 = " << vector9);
        LOG_DEBUG(<< " min2 = " << vector10);
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
        BOOST_REQUIRE_EQUAL(std::string("[0.9, 3.1, 2.1, 1.5]"),
                            core::CContainerPrinter::print(vector5));
        BOOST_REQUIRE_EQUAL(std::string("[0.9, 3.1, 2.1, 1.5]"),
                            core::CContainerPrinter::print(vector6));
        BOOST_REQUIRE_EQUAL(std::string("0.9 3.1 2.1 1.5"), o7.str());
        BOOST_REQUIRE_EQUAL(std::string("0.9 3.1 2.1 1.5"), o8.str());
        BOOST_REQUIRE_EQUAL(std::string("[1, 1.6, 4, 0.2]"),
                            core::CContainerPrinter::print(vector9));
        BOOST_REQUIRE_EQUAL(std::string("[1, 1.6, 4, 0.2]"),
                            core::CContainerPrinter::print(vector10));
        BOOST_REQUIRE_EQUAL(std::string("  1 1.6   4 0.2"), o11.str());
        BOOST_REQUIRE_EQUAL(std::string("  1 1.6   4 0.2"), o12.str());
    }
    LOG_DEBUG(<< "Test max");
    {
        double placeholder[2][4];
        std::copy(std::begin(components[1]), std::end(components[1]),
                  std::begin(placeholder[0]));
        std::copy(std::begin(components[2]), std::end(components[2]),
                  std::begin(placeholder[1]));
        TVector4 vector5(components[1]);
        TVector vector6(std::begin(components[1]), std::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(placeholder[0], 4);
        TVector4 vector9(components[2]);
        TVector vector10(std::begin(components[2]), std::end(components[2]));
        TDenseVector vector11(4);
        vector11 << components[2][0], components[2][1], components[2][2],
            components[2][3];
        TMappedVector vector12(placeholder[1], 4);
        maths::common::las::max(vector1, vector5);
        maths::common::las::max(vector2, vector6);
        maths::common::las::max(vector3, vector7);
        maths::common::las::max(vector4, vector8);
        maths::common::las::max(vector1, vector9);
        maths::common::las::max(vector2, vector10);
        maths::common::las::max(vector3, vector11);
        maths::common::las::max(vector4, vector12);
        LOG_DEBUG(<< " max1 = " << vector5);
        LOG_DEBUG(<< " max1 = " << vector6);
        LOG_DEBUG(<< " max1 = " << vector7.transpose());
        LOG_DEBUG(<< " max1 = " << vector8.transpose());
        LOG_DEBUG(<< " max2 = " << vector9);
        LOG_DEBUG(<< " max2 = " << vector10);
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
        BOOST_REQUIRE_EQUAL(std::string("[1, 3.2, 4, 1.7]"),
                            core::CContainerPrinter::print(vector5));
        BOOST_REQUIRE_EQUAL(std::string("[1, 3.2, 4, 1.7]"),
                            core::CContainerPrinter::print(vector6));
        BOOST_REQUIRE_EQUAL(std::string("  1 3.2   4 1.7"), o7.str());
        BOOST_REQUIRE_EQUAL(std::string("  1 3.2   4 1.7"), o8.str());
        BOOST_REQUIRE_EQUAL(std::string("[1.3, 3.1, 8.9, 1.5]"),
                            core::CContainerPrinter::print(vector9));
        BOOST_REQUIRE_EQUAL(std::string("[1.3, 3.1, 8.9, 1.5]"),
                            core::CContainerPrinter::print(vector10));
        BOOST_REQUIRE_EQUAL(std::string("1.3 3.1 8.9 1.5"), o11.str());
        BOOST_REQUIRE_EQUAL(std::string("1.3 3.1 8.9 1.5"), o12.str());
    }
    LOG_DEBUG(<< "Test componentwise divide");
    {
        TVector4 vector5(components[1]);
        TVector vector6(std::begin(components[1]), std::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(components[1], 4);
        TVector4 d1 = maths::common::las::componentwise(vector1) /
                      maths::common::las::componentwise(vector5);
        TVector d2 = maths::common::las::componentwise(vector2) /
                     maths::common::las::componentwise(vector6);
        TDenseVector d3 = maths::common::las::componentwise(vector3) /
                          maths::common::las::componentwise(vector7);
        TDenseVector d4 = maths::common::las::componentwise(vector4) /
                          maths::common::las::componentwise(vector8);
        LOG_DEBUG(<< " v1 / v2 = " << d1);
        LOG_DEBUG(<< " v1 / v2 = " << d2);
        LOG_DEBUG(<< " v1 / v2 = " << d3.transpose());
        LOG_DEBUG(<< " v1 / v2 = " << d4.transpose());
        std::ostringstream o3;
        o3 << d3.transpose();
        std::ostringstream o4;
        o4 << d4.transpose();
        BOOST_REQUIRE_EQUAL(std::string("[1.111111, 0.96875, 1.904762, 0.8823529]"),
                            core::CContainerPrinter::print(d1));
        BOOST_REQUIRE_EQUAL(std::string("[1.111111, 0.96875, 1.904762, 0.8823529]"),
                            core::CContainerPrinter::print(d2));
        BOOST_REQUIRE_EQUAL(std::string(" 1.11111  0.96875  1.90476 0.882353"), o3.str());
        BOOST_REQUIRE_EQUAL(std::string(" 1.11111  0.96875  1.90476 0.882353"), o4.str());
    }
    LOG_DEBUG(<< "Test distance");
    {
        TVector4 vector5(components[2]);
        TVector vector6(std::begin(components[2]), std::end(components[2]));
        TDenseVector vector7(4);
        vector7 << components[2][0], components[2][1], components[2][2],
            components[2][3];
        TMappedVector vector8(components[2], 4);
        double d1 = maths::common::las::distance(vector1, vector5);
        double d2 = maths::common::las::distance(vector2, vector6);
        double d3 = maths::common::las::distance(vector3, vector7);
        double d4 = maths::common::las::distance(vector4, vector8);
        LOG_DEBUG(<< " d1 = " << d1);
        LOG_DEBUG(<< " d1 = " << d2);
        LOG_DEBUG(<< " d1 = " << d3);
        LOG_DEBUG(<< " d1 = " << d4);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.29528091794949, d1, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.29528091794949, d2, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.29528091794949, d3, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.29528091794949, d4, 1e-10);
    }
    LOG_DEBUG(<< "Test Euclidean norm");
    {
        double n1 = maths::common::las::norm(vector1);
        double n2 = maths::common::las::norm(vector2);
        double n3 = maths::common::las::norm(vector3);
        double n4 = maths::common::las::norm(vector4);
        LOG_DEBUG(<< " n1 = " << n1);
        LOG_DEBUG(<< " n1 = " << n2);
        LOG_DEBUG(<< " n1 = " << n3);
        LOG_DEBUG(<< " n1 = " << n4);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.37215040742532, n1, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.37215040742532, n2, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.37215040742532, n3, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5.37215040742532, n4, 1e-10);
    }
    LOG_DEBUG(<< "Test L1");
    {
        TVector4 vector5(components[3]);
        TVector vector6(std::begin(components[3]), std::end(components[3]));
        TDenseVector vector7(4);
        vector7 << components[3][0], components[3][1], components[3][2],
            components[3][3];
        TMappedVector vector8(components[3], 4);
        double l1 = maths::common::las::L1(vector5);
        double l2 = maths::common::las::L1(vector6);
        double l3 = maths::common::las::L1(vector7);
        double l4 = maths::common::las::L1(vector8);
        LOG_DEBUG(<< " l1 = " << l1);
        LOG_DEBUG(<< " l1 = " << l2);
        LOG_DEBUG(<< " l1 = " << l3);
        LOG_DEBUG(<< " l1 = " << l4);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(11.8, l1, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(11.8, l2, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(11.8, l3, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(11.8, l4, 1e-10);
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
        elements_.emplace_back(std::begin(elements[0]), std::end(elements[0]));
        elements_.emplace_back(std::begin(elements[1]), std::end(elements[1]));
        elements_.emplace_back(std::begin(elements[2]), std::end(elements[2]));
        elements_.emplace_back(std::begin(elements[3]), std::end(elements[3]));
        TMatrix4 matrix1(elements);
        TMatrix matrix2(elements_);
        TDenseMatrix matrix3(4, 4);
        matrix3 << elements[0][0], elements[0][1], elements[0][2], elements[0][3],
            elements[1][0], elements[1][1], elements[1][2], elements[1][3],
            elements[2][0], elements[2][1], elements[2][2], elements[2][3],
            elements[3][0], elements[3][1], elements[3][2], elements[3][3];
        TMappedMatrix matrix4(flat, 4, 4);
        LOG_DEBUG(<< matrix4);
        double f1 = maths::common::las::frobenius(matrix1);
        double f2 = maths::common::las::frobenius(matrix2);
        double f3 = maths::common::las::frobenius(matrix3);
        double f4 = maths::common::las::frobenius(matrix4);
        LOG_DEBUG(<< " f1 = " << f1);
        LOG_DEBUG(<< " f1 = " << f2);
        LOG_DEBUG(<< " f1 = " << f3);
        LOG_DEBUG(<< " f1 = " << f4);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(7.92906047397799, f1, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(7.92906047397799, f2, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(7.92906047397799, f3, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(7.92906047397799, f4, 1e-10);
    }
    LOG_DEBUG(<< "Test inner");
    {
        TVector4 vector5(components[1]);
        TVector vector6(std::begin(components[1]), std::end(components[1]));
        TDenseVector vector7(4);
        vector7 << components[1][0], components[1][1], components[1][2],
            components[1][3];
        TMappedVector vector8(components[1], 4);
        double i1 = maths::common::las::inner(vector1, vector5);
        double i2 = maths::common::las::inner(vector2, vector6);
        double i3 = maths::common::las::inner(vector3, vector7);
        double i4 = maths::common::las::inner(vector4, vector8);
        LOG_DEBUG(<< "i1 = " << i1);
        LOG_DEBUG(<< "i1 = " << i2);
        LOG_DEBUG(<< "i1 = " << i3);
        LOG_DEBUG(<< "i1 = " << i4);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(21.77, i1, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(21.77, i2, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(21.77, i3, 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(21.77, i4, 1e-10);
    }
    LOG_DEBUG(<< "Test outer");
    {
        TMatrix4 outer1 = maths::common::las::outer(vector1);
        TMatrix outer2 = maths::common::las::outer(vector2);
        TDenseMatrix outer3 = maths::common::las::outer(vector3);
        TDenseMatrix outer4 = maths::common::las::outer(vector4);
        LOG_DEBUG(<< "outer = " << outer1);
        LOG_DEBUG(<< "outer = " << outer2);
        LOG_DEBUG(<< "outer =\n" << outer3);
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(vector1(i) * vector1(j), outer1(i, j));
                BOOST_REQUIRE_EQUAL(vector2(i) * vector2(j), outer2(i, j));
                BOOST_REQUIRE_EQUAL(vector3(i) * vector3(j), outer3(i, j));
                BOOST_REQUIRE_EQUAL(vector4(i) * vector4(j), outer4(i, j));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMemoryMapped) {
    using TDenseVector = maths::common::CDenseVector<double>;
    using TDenseMatrix = maths::common::CDenseMatrix<double>;
    using TMappedFloatVector = maths::common::CMemoryMappedDenseVector<float>;
    using TMappedFloatMatrix = maths::common::CMemoryMappedDenseMatrix<float>;

    {
        float components[4];
        TMappedFloatVector mappedVector(components, 4);

        TDenseVector vector{4};
        vector << 1.1, 2.7, 0.1, -3.0;
        mappedVector = vector;

        for (int i = 0; i < 4; ++i) {
            BOOST_REQUIRE_CLOSE(vector(i), mappedVector(i), 1e-4);
        }
    }
    {
        float components[9];
        TMappedFloatMatrix mappedMatrix(components, 3, 3);

        TDenseMatrix matrix{3, 3};
        matrix << 1.1, 2.7, 0.1, 3.0, 1.2, 0.4, 5.1, 0.2, 9.3;
        mappedMatrix = matrix;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                BOOST_REQUIRE_CLOSE(matrix(i, j), mappedMatrix(i, j), 1e-4);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check conversion to and from delimited is idempotent and parsing
    // bad input produces an error.

    {
        double matrix_[][4]{{1.0, 2.1, 1.5, 0.1},
                            {2.1, 2.2, 3.7, 0.6},
                            {1.5, 3.7, 0.4, 8.1},
                            {0.1, 0.6, 8.1, 4.3}};

        maths::common::CSymmetricMatrixNxN<double, 4> matrix(matrix_);

        std::string str = matrix.toDelimited();

        maths::common::CSymmetricMatrixNxN<double, 4> restoredMatrix;
        BOOST_TEST_REQUIRE(restoredMatrix.fromDelimited(str));

        LOG_DEBUG(<< "delimited = " << str);

        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                BOOST_REQUIRE_EQUAL(matrix(i, j), restoredMatrix(i, j));
            }
        }

        BOOST_TEST_REQUIRE(!restoredMatrix.fromDelimited(std::string()));

        std::string bad("0.1,0.3,0.5,3");
        BOOST_TEST_REQUIRE(!restoredMatrix.fromDelimited(bad));
        bad = "0.1,0.3,a,0.1,0.3,0.1,0.3,0.1,1.0,3";
        BOOST_TEST_REQUIRE(!restoredMatrix.fromDelimited(bad));
    }
    {
        double vector_[]{11.2, 2.1, 1.5};

        maths::common::CVectorNx1<double, 3> vector(vector_);

        std::string str = vector.toDelimited();

        maths::common::CVectorNx1<double, 3> restoredVector;
        BOOST_TEST_REQUIRE(restoredVector.fromDelimited(str));

        LOG_DEBUG(<< "delimited = " << str);

        for (std::size_t i = 0; i < 3; ++i) {
            BOOST_REQUIRE_EQUAL(vector(i), restoredVector(i));
        }

        BOOST_TEST_REQUIRE(!restoredVector.fromDelimited(std::string()));

        std::string bad("0.1,0.3,0.5,3");
        BOOST_TEST_REQUIRE(!restoredVector.fromDelimited(bad));
        bad = "0.1,0.3,a";
        BOOST_TEST_REQUIRE(!restoredVector.fromDelimited(bad));
    }
    {
        maths::common::CDenseVector<double> origVector(4);
        origVector << 1.3, 2.4, 3.1, 5.1;

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origVector.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "vector XML representation:\n" << origXml);

        // Restore the XML into a new vector.
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::common::CDenseVector<double> restoredVector;
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(
            std::bind(&maths::common::CDenseVector<double>::acceptRestoreTraverser,
                      &restoredVector, std::placeholders::_1)));

        BOOST_REQUIRE_EQUAL(origVector.checksum(), restoredVector.checksum());
    }
}

BOOST_AUTO_TEST_SUITE_END()
