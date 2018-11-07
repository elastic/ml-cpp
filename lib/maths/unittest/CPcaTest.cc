/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CPcaTest.h"

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CGramSchmidt.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPca.h>

#include <test/CRandomNumbers.h>

#include <boost/unordered_set.hpp>

#include <cmath>
#include <numeric>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;

void CPcaTest::testProjectOntoPrincipleComponents() {
    using TMeanAcculator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Standard");
    {
        // Test random data are projected on to the basis vectors
        // corresponding to the largest variances.

        TDoubleVec variances{128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0};
        std::size_t dimension{variances.size()};
        TSizeVec support;

        TMeanAcculator meanMeanError;
        for (std::size_t t = 0u; t < 10; ++t) {
            maths::CPca::TDenseVectorVec basis;
            {
                TDoubleVec components;
                rng.generateUniformSamples(-1.0, 1.0, dimension * dimension, components);
                for (std::size_t i = 0u; i < components.size(); /**/) {
                    basis.emplace_back(dimension);
                    for (std::size_t j = 0u; j < dimension; ++i, ++j) {
                        basis.back()(j) = components[i];
                    }
                }
            }
            maths::CGramSchmidt::basis(basis);

            std::size_t points{500};
            support.resize(points);
            std::iota(support.begin(), support.end(), 0);
            maths::CPca::TDenseVectorVec data(
                points, maths::SConstant<maths::CPca::TDenseVector>::get(dimension, 0));
            maths::CPca::TDenseVectorVec expected(points, maths::CPca::TDenseVector(3));

            for (std::size_t i = 0u; i < dimension; ++i) {
                TDoubleVec x;
                rng.generateNormalSamples(0.0, variances[i], data.size(), x);
                for (std::size_t j = 0u; j < points; ++j) {
                    data[j] += x[j] * basis[i];
                    if (i < 3) {
                        expected[j][i] = x[j];
                    }
                }
            }

            maths::CPca::projectOntoPrincipleComponents(3, support, data);

            TMeanAcculator meanError;
            for (std::size_t i = 0u; i < data.size(); ++i) {
                double error{0.0};
                for (std::size_t j = 0u; j < 3; ++j) {
                    error += std::pow(
                        std::fabs(data[i](j)) - std::fabs(expected[i](j)), 2.0);
                }
                meanError.add(std::sqrt(error) / expected[i].norm());
            }

            LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.11);

            meanMeanError += meanError;
        }

        LOG_DEBUG(<< "mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.09);
    }
    {
        // Test with incomplete support.
    }
}

void CPcaTest::testSparseProjectOntoPrincipleComponents() {
    using TDenseMatrix = maths::CDenseMatrix<double>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "Sampled Exact");

        // Check we get sampling loses no information when the non-zero
        // columns can fit in the same memory as the sparse representation.

        TDoubleVec sd{100.0, 50.0, 40.0, 4.0, 2.0};
        TDoubleVecVec basis{{1.0, 1.0, 1.0, 1.0, 1.0},
                            {1.0, -1.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 1.0, -1.0, 0.0},
                            {1.0, 1.0, 1.0, 1.0, -4.0},
                            {-1.0, -1.0, 1.0, 1.0, 0.0}};
        for (auto&& e : basis) {
            double norm{std::sqrt(
                std::accumulate(e.begin(), e.end(), 0.0,
                                [](double n, double x) { return n + x * x; }))};
            std::for_each(e.begin(), e.end(), [norm](double& x) { x /= norm; });
        }
        std::size_t dimension{20};

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 10.0, 1000, samples);
        TSizeVec dimensions;
        rng.generateUniformSamples(0, dimension, 7, dimensions);
        std::sort(dimensions.begin(), dimensions.end());
        dimensions.erase(std::unique(dimensions.begin(), dimensions.end()),
                         dimensions.end());
        rng.random_shuffle(dimensions.begin(), dimensions.end());

        maths::CPca::TIntDoublePrVecVec sparse;
        maths::CPca::TDenseVectorVec dense;
        for (std::size_t i = 0u; i < samples.size(); /**/) {
            std::size_t working{sparse.size()};
            sparse.emplace_back(5);
            dense.push_back(maths::SConstant<maths::CPca::TDenseVector>::get(dimension, 0.0));
            TDoubleVec values(5, 0.0);
            for (std::size_t j = 0u; j < 5; ++i, ++j) {
                for (std::size_t k = 0u; k < 5; ++k) {
                    values[k] += sd[j] * basis[j][k] * samples[i];
                }
            }
            for (std::size_t j = 0u; j < 5; ++j) {
                std::ptrdiff_t k{static_cast<std::ptrdiff_t>(dimensions[j])};
                sparse[working][j] = {k, values[j]};
                dense[working](k) = values[j];
            }
            std::sort(sparse[working].begin(), sparse[working].end());
        }

        TSizeVec support;
        support.resize(sparse.size());
        std::iota(support.begin(), support.end(), 0);

        maths::CPca::TDenseVectorVec projected;
        maths::CPca::projectOntoPrincipleComponents(3, support, dense);
        maths::CPca::projectOntoPrincipleComponentsRandom(3, dimension, support,
                                                          sparse, projected);

        for (std::size_t i = 0u; i < dense.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "exact       = " << dense[i].transpose());
                LOG_DEBUG(<< "approximate = " << projected[i].transpose());
            }
            CPPUNIT_ASSERT((dense[i] - projected[i]).norm() < 1e-6);
        }
    }
    {
        LOG_DEBUG(<< "Sampled Approximate");

        // For a large sparse data matrix we test how good the low
        // rank approximation calculated using sampling verses the
        // best achievable calculated using exact SVD.

        TDoubleVec componentProbabilities{
            0.01, 0.1,  0.04, 0.2,  0.05, 0.02, 0.04, 0.01, 0.04,
            0.1,  0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.01, 0.01,
            0.01, 0.04, 0.03, 0.04, 0.1,  0.01, 0.01};
        TDoubleVec componentMaxComponents{
            5.0,  25.0, 15.0, 50.0, 5.0,  1.0, 5.0, 40.0, 1.0,
            15.0, 5.0,  10.0, 1.0,  1.0,  1.0, 5.0, 10.0, 10.0,
            5.0,  10.0, 5.0,  10.0, 20.0, 1.0, 3.0};
        std::ptrdiff_t dimension{
            static_cast<std::ptrdiff_t>(componentProbabilities.size())};
        std::size_t numberVectors{1000};

        TMeanAccumulator meanError;
        for (std::size_t t = 0u; t < 10; ++t) {
            maths::CPca::TIntDoublePrVecVec sparse(numberVectors);
            maths::CPca::TDenseVectorVec dense(
                numberVectors,
                maths::SConstant<maths::CPca::TDenseVector>::get(dimension, 0.0));
            TDoubleVec uniform01;
            TDoubleVec component;
            for (std::size_t i = 0u; i < numberVectors; ++i) {
                rng.generateUniformSamples(0.0, 1.0, componentProbabilities.size(), uniform01);
                for (std::size_t j = 0u; j < uniform01.size(); ++j) {
                    if (uniform01[j] < componentProbabilities[j]) {
                        rng.generateUniformSamples(-0.5 * componentMaxComponents[j],
                                                   0.5 * componentMaxComponents[j],
                                                   1, component);
                        sparse[i].emplace_back(j, component[0]);
                        dense[i](j) = component[0];
                    }
                }
            }

            TSizeVec support;
            support.resize(sparse.size());
            std::iota(support.begin(), support.end(), 0);

            maths::CPca::TDenseVectorVec approx;
            maths::CPca::projectOntoPrincipleComponents(4, support, dense);
            maths::CPca::projectOntoPrincipleComponentsRandom(4, dimension, support,
                                                              sparse, approx);

            TDenseMatrix exactProjection(dense.size(), 4);
            TDenseMatrix approxProjection(dense.size(), 4);
            for (std::size_t i = 0u; i < dense.size(); ++i) {
                exactProjection.row(i) = dense[i];
                approxProjection.row(i) = approx[i];
            }
            auto exactSvd = exactProjection.jacobiSvd();
            auto approxSvd = approxProjection.jacobiSvd();
            LOG_DEBUG(<< "exact = " << exactSvd.singularValues().transpose());
            LOG_DEBUG(<< "approx = " << approxSvd.singularValues().transpose());
            double error{
                (approxSvd.singularValues() - exactSvd.singularValues()).norm() /
                exactSvd.singularValues().norm()};
            LOG_DEBUG(<< "error = " << error);
            CPPUNIT_ASSERT(error < 0.15);
            meanError.add(std::log(error));
        }
        LOG_DEBUG(<< "mean error = " << std::exp(maths::CBasicStatistics::mean(meanError)));
        CPPUNIT_ASSERT(std::exp(maths::CBasicStatistics::mean(meanError)) < 0.05);
    }
    {
        // Test with incomplete support.
    }
}

void CPcaTest::testNumericRank() {
}

CppUnit::Test* CPcaTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CPcaTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CPcaTest>(
        "CPcaTest::testProjectOntoPrincipleComponents",
        &CPcaTest::testProjectOntoPrincipleComponents));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPcaTest>(
        "CPcaTest::testSparseProjectOntoPrincipleComponents",
        &CPcaTest::testSparseProjectOntoPrincipleComponents));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPcaTest>(
        "CPcaTest::testNumericRank", &CPcaTest::testNumericRank));

    return suiteOfTests;
}
