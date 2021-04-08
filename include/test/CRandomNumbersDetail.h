/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_test_CRandomNumbersDetail_h
#define INCLUDED_ml_test_CRandomNumbersDetail_h

#include <core/CLogger.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraTools.h>

#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>

namespace ml {
namespace test {

template<typename RNG, typename Distribution, typename Container>
void CRandomNumbers::generateSamples(RNG& randomNumberGenerator,
                                     const Distribution& distribution,
                                     std::size_t numberSamples,
                                     Container& samples) {
    samples.clear();
    samples.reserve(numberSamples);
    std::generate_n(std::back_inserter(samples), numberSamples,
                    [&distribution, &randomNumberGenerator]() {
                        return const_cast<Distribution&>(distribution)(randomNumberGenerator);
                    });
}

template<typename T, std::size_t N>
void CRandomNumbers::generateRandomMultivariateNormals(
    const TSizeVec& sizes,
    std::vector<maths::CVectorNx1<T, N>>& means,
    std::vector<maths::CSymmetricMatrixNxN<T, N>>& covariances,
    std::vector<std::vector<maths::CVectorNx1<T, N>>>& points) {
    means.clear();
    covariances.clear();
    points.clear();

    std::size_t k = sizes.size();

    TDoubleVec means_;
    this->generateUniformSamples(-100.0, 100.0, N * k, means_);
    for (std::size_t i = 0; i < N * k; i += N) {
        maths::CVectorNx1<T, N> mean(&means_[i], &means_[i + N]);
        means.push_back(mean);
    }

    TDoubleVec variances;
    this->generateUniformSamples(10.0, 100.0, N * k, variances);
    for (std::size_t i = 0; i < k; ++i) {
        constexpr int N_{static_cast<int>(N)};
        Eigen::Matrix<T, N_, N_> covariance = Eigen::Matrix<T, N_, N_>::Zero();

        for (std::size_t j = 0; j < N; ++j) {
            covariance(j, j) = variances[i * N + j];
        }

        // Generate random rotations in two planes.
        TSizeVec coordinates;
        this->generateUniformSamples(0, N, 4, coordinates);
        std::sort(coordinates.begin(), coordinates.end());
        coordinates.erase(std::unique(coordinates.begin(), coordinates.end()),
                          coordinates.end());

        TDoubleVec thetas;
        this->generateUniformSamples(0.0, boost::math::constants::two_pi<double>(), 2, thetas);

        Eigen::Matrix<T, N_, N_> rotation = Eigen::Matrix<T, N_, N_>::Identity();
        for (std::size_t j = 1; j < coordinates.size(); j += 2) {
            double ct = std::cos(thetas[j / 2]);
            double st = std::sin(thetas[j / 2]);

            Eigen::Matrix<T, N_, N_> r = Eigen::Matrix<T, N_, N_>::Identity();
            r(coordinates[j / 2], coordinates[j / 2]) = ct;
            r(coordinates[j / 2], coordinates[j / 2 + 1]) = -st;
            r(coordinates[j / 2 + 1], coordinates[j / 2]) = st;
            r(coordinates[j / 2 + 1], coordinates[j / 2 + 1]) = ct;
            rotation *= r;
        }
        covariance = rotation.transpose() * covariance * rotation;

        covariances.emplace_back(maths::fromDenseMatrix(covariance));
    }

    points.resize(k);
    TDoubleVecVec pointsi;
    for (std::size_t i = 0; i < k; ++i) {
        LOG_TRACE(<< "mean = " << means[i]);
        LOG_TRACE(<< "covariance = " << covariances[i]);
        this->generateMultivariateNormalSamples(
            means[i].template toVector<TDoubleVec>(),
            covariances[i].template toVectors<TDoubleVecVec>(), sizes[i], pointsi);
        for (std::size_t j = 0; j < pointsi.size(); ++j) {
            points[i].emplace_back(pointsi[j]);
        }
    }
}
}
}

#endif // INCLUDED_ml_test_CRandomNumbersDetail_h
