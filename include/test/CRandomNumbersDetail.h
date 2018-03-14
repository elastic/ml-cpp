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
#include <iterator>

namespace ml {
namespace test {

template<typename RNG,
         typename Distribution,
         typename Container>
void CRandomNumbers::generateSamples(RNG &randomNumberGenerator,
                                     const Distribution &distribution,
                                     std::size_t numberSamples,
                                     Container &samples) {
    samples.clear();
    samples.reserve(numberSamples);
    std::generate_n(std::back_inserter(samples),
                    numberSamples,
                    boost::bind(distribution, boost::ref(randomNumberGenerator)));
}

template<typename ITR>
void CRandomNumbers::random_shuffle(ITR first, ITR last) {
    CUniform0nGenerator rand(m_Generator);
    auto                d = last - first;
    if (d > 1) {
        for (--last; first < last; ++first, --d) {
            auto i = rand(d);
            if (i > 0) {
                std::iter_swap(first, first + i);
            }
        }
    }
}

template<typename T, std::size_t N>
void CRandomNumbers::generateRandomMultivariateNormals(const TSizeVec &sizes,
                                                       std::vector<maths::CVectorNx1<T, N> > &means,
                                                       std::vector<maths::CSymmetricMatrixNxN<T, N> > &covariances,
                                                       std::vector<std::vector<maths::CVectorNx1<T, N> > > &points) {
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
        Eigen::Matrix<T, N, N> covariance = Eigen::Matrix<T, N, N>::Zero();

        for (std::size_t j = 0u; j < N; ++j) {
            covariance(j, j) = variances[i * N + j];
        }

        // Generate random rotations in two planes.
        TSizeVec coordinates;
        this->generateUniformSamples(0, N, 4, coordinates);
        std::sort(coordinates.begin(), coordinates.end());
        coordinates.erase(std::unique(coordinates.begin(),
                                      coordinates.end()), coordinates.end());

        TDoubleVec thetas;
        this->generateUniformSamples(0.0, boost::math::constants::two_pi<double>(), 2, thetas);

        Eigen::Matrix<T, N, N> rotation = Eigen::Matrix<T, N, N>::Identity();
        for (std::size_t j = 1u; j < coordinates.size(); j += 2) {
            double ct = ::cos(thetas[j/2]);
            double st = ::sin(thetas[j/2]);

            Eigen::Matrix<T, N, N> r = Eigen::Matrix<T, N, N>::Identity();
            r(coordinates[j/2],   coordinates[j/2])   =  ct;
            r(coordinates[j/2],   coordinates[j/2+1]) = -st;
            r(coordinates[j/2+1], coordinates[j/2])   =  st;
            r(coordinates[j/2+1], coordinates[j/2+1]) =  ct;
            rotation *= r;
        }
        covariance = rotation.transpose() * covariance * rotation;

        covariances.emplace_back(maths::fromDenseMatrix(covariance));
    }

    points.resize(k);
    TDoubleVecVec pointsi;
    for (std::size_t i = 0u; i < k; ++i) {
        LOG_TRACE("mean = " << means[i]);
        LOG_TRACE("covariance = " << covariances[i]);
        this->generateMultivariateNormalSamples(means[i].template toVector<TDoubleVec>(),
                                                covariances[i].template toVectors<TDoubleVecVec>(),
                                                sizes[i], pointsi);
        for (std::size_t j = 0u; j < pointsi.size(); ++j) {
            points[i].emplace_back(pointsi[j]);
        }
    }
}

}
}

#endif // INCLUDED_ml_test_CRandomNumbersDetail_h
