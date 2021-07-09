/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLowessDetail_h
#define INCLUDED_ml_maths_CLowessDetail_h

#include <maths/CLowess.h>

#include <core/CContainerPrinter.h>

#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COrderings.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

template<std::size_t N>
void CLowess<N>::fit(TDoubleDoublePrVec data, std::size_t numberFolds) {

    m_K = 0.0;
    m_Data = std::move(data);
    std::sort(m_Data.begin(), m_Data.end(), COrderings::SFirstLess{});

    if (m_Data.size() < 4) {
        return;
    }

    // We use exponential decay in the weights and cross-validated maximum likelihood
    // to choose the decay constant. Formally, we are fitting
    //
    //   f(x | p^*) = poly(x | p^*(x))
    //
    //   p^*(x) = argmin_p{ sum_i{ w_i (Y_i - poly(X_i | p))^2 } }              (1)
    //
    // where w = exp(-k (x - X_i)), (X, Y) are the data to fit and p is the vector
    // of parameters for the polynomial function poly(. | p), i.e. the coefficients
    // p_0 + p_1 x + p_2 x^2 ... (which are determined by minimizing the weighted
    // least square prediction errors as in (1)).
    //
    // We determine k by solving
    //
    //   k^* = argmin_k{ sum_{Yi in H}{ L(Yi | f(x | p^*(k))) } }
    //
    // where H is a hold out set and we assume Y_i ~ N(poly(X_i | p^*(k)), sigma)
    // with sigma estimated from the training data prediction residuals to compute
    // the likelihood function L(Yi | f(x | p^*(k))).

    m_Mask.resize(m_Data.size());
    std::iota(m_Mask.begin(), m_Mask.end(), 0);

    TSizeVecVec trainingMasks;
    TSizeVecVec testingMasks;
    this->setupMasks(numberFolds, trainingMasks, testingMasks);

    // Here, we line search different values of m_K. We aim to cover the case we have
    // a lot of smoothing, m_K is 0, to the case m_K is large compared to the data
    // range so most points have very low weight and don't constrain the polynomial
    // parameters. We finish up by polishing up the minimum on the best candidate
    // interval using Brent's method. See CSolvers::globalMaximize for details.

    TDoubleVec K(17);
    double range{m_Data.back().first - m_Data.front().first};
    for (std::size_t i = 0; i < K.size(); ++i) {
        K[i] = 2.0 * static_cast<double>(i) / range;
    }
    LOG_TRACE(<< "range = " << range << ", K = " << core::CContainerPrinter::print(K));

    double kmax;
    double likelihoodMax;
    CSolvers::globalMaximize(K,
                             [&](double k) {
                                 return this->likelihood(trainingMasks, testingMasks, k);
                             },
                             kmax, likelihoodMax);
    LOG_TRACE(<< "kmax = " << kmax << " likelihood(kmax) = " << likelihoodMax);

    m_K = kmax;
}

template<std::size_t N>
double CLowess<N>::predict(double x) const {
    if (m_Data.empty()) {
        return 0.0;
    }
    auto poly = this->fit(m_Mask.begin(), m_Mask.end(), m_K, x);
    return poly.predict(x);
}

template<std::size_t N>
typename CLowess<N>::TDoubleDoublePr CLowess<N>::minimum() const {

    if (m_Data.empty()) {
        return {0.0, 0.0};
    }

    // There is no guaranty the function is convex so we need a global method.
    // We choose something simple:
    //   1. Find (local) minimum near a data point.
    //   2. Search around here for the true local minimum.
    //
    // All in all this has complexity O(2 |data| function evaluations).

    TDoubleVec X;

    double xa;
    double xb;
    std::tie(xa, xb) = this->extrapolationInterval();

    // Coarse.
    X.reserve(m_Data.size() + 2);
    X.push_back(xa);
    for (const auto& xi : m_Data) {
        X.push_back(xi.first);
    }
    X.push_back(xb);
    double xmin;
    double fmin;
    CSolvers::globalMinimize(X, [&](double x) { return this->predict(x); }, xmin, fmin);

    // Refine.
    double range{(xb - xa) / static_cast<double>(X.size())};
    xa = std::max(xa, xmin - 0.5 * range);
    xb = std::min(xb, xmin + 0.5 * range);
    double dx{2.0 * (xb - xa) / static_cast<double>(X.size())};
    X.clear();
    for (double x = xa; x < xb; x += dx) {
        X.push_back(x);
    }
    double xcand;
    double fcand;
    CSolvers::globalMinimize(X, [&](double x) { return this->predict(x); }, xcand, fcand);

    if (fcand < fmin) {
        xmin = xcand;
        fmin = fcand;
    }

    return {xmin, fmin};
}

template<std::size_t N>
double CLowess<N>::residualVariance() const {

    if (m_Data.empty()) {
        return 0.0;
    }

    TMeanVarAccumulator moments;

    std::size_t n{m_Data.size()};

    TSizeVec mask(n);
    std::iota(mask.begin(), mask.end(), 1);
    for (std::size_t i = 0; i < n; ++i) {
        double xi;
        double yi;
        std::tie(xi, yi) = m_Data[i];
        auto poly = this->fit(mask.begin(), mask.begin() + n - 1, m_K, xi);
        moments.add(yi - poly.predict(xi));
        mask[i] -= 1;
    }

    return CBasicStatistics::variance(moments);
}

template<std::size_t N>
typename CLowess<N>::TDoubleDoublePr CLowess<N>::extrapolationInterval() const {
    double xa{m_Data.front().first};
    double xb{m_Data.back().first};
    xa -= std::min(0.1 * (xb - xa), 0.5 / m_K);
    xb += std::min(0.1 * (xb - xa), 0.5 / m_K);
    return {xa, xb};
}

template<std::size_t N>
void CLowess<N>::setupMasks(std::size_t numberFolds,
                            TSizeVecVec& trainingMasks,
                            TSizeVecVec& testingMasks) const {

    numberFolds = CTools::truncate(numberFolds, std::size_t{2}, m_Data.size());

    trainingMasks.resize(numberFolds);
    testingMasks.resize(numberFolds);

    if (numberFolds == m_Data.size()) {
        // Leave-out-one cross-validation.
        trainingMasks[0].resize(m_Data.size() - 1);
        std::iota(trainingMasks[0].begin(), trainingMasks[0].end(), 1);
        testingMasks[0].push_back(0);
        for (std::size_t i = 1; i < numberFolds; ++i) {
            trainingMasks[i] = trainingMasks[0];
            trainingMasks[i][i - 1] = 0;
            std::sort(trainingMasks[i].begin(), trainingMasks[i].end());
            testingMasks[i].push_back(i);
        }
    } else {
        // K-fold cross-validation.
        CPRNG::CXorOShiro128Plus rng;
        TSizeVec all(m_Data.size());
        TSizeVec remaining;
        TSizeVec sample;
        TDoubleVec probabilities;

        std::iota(all.begin(), all.end(), 0);
        remaining = all;

        for (std::size_t i = 0; i < numberFolds; ++i) {
            std::size_t n{std::min((m_Data.size() + numberFolds - 1) / numberFolds,
                                   remaining.size())};
            probabilities.assign(remaining.size(), 1.0);
            CSampling::categoricalSampleWithoutReplacement(rng, probabilities, n, sample);

            testingMasks[i].reserve(sample.size());
            for (auto j : sample) {
                testingMasks[i].push_back(remaining[j]);
            }
            std::sort(testingMasks[i].begin(), testingMasks[i].end());

            trainingMasks[i].reserve(all.size() - testingMasks[i].size());
            std::set_difference(all.begin(), all.end(), testingMasks[i].begin(),
                                testingMasks[i].end(),
                                std::back_inserter(trainingMasks[i]));

            CSetTools::inplace_set_difference(remaining, testingMasks[i].begin(),
                                              testingMasks[i].end());
            rng.discard(100000);
        }
    }

    LOG_TRACE(<< "training masks = " << core::CContainerPrinter::print(trainingMasks));
    LOG_TRACE(<< "testing masks = " << core::CContainerPrinter::print(testingMasks));
}

template<std::size_t N>
double CLowess<N>::likelihood(TSizeVecVec& trainingMasks, TSizeVecVec& testingMasks, double k) const {

    double result{0.0};

    CNormalMeanPrecConjugate::TDouble1Vec testResiduals;
    CNormalMeanPrecConjugate::TDoubleWeightsAry1Vec weights;

    for (std::size_t i = 0; i < trainingMasks.size(); ++i) {

        CNormalMeanPrecConjugate residuals{
            CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};

        std::size_t last{trainingMasks[i].size() - 1};

        for (auto& j : trainingMasks[i]) {
            double xj;
            double yj;
            std::tie(xj, yj) = m_Data[j];
            // Here we wish to leave out the j'th fold training mask. Since this
            // is a vector we do this efficiently by temporarily swaping to the
            // back of the collection so we can pass the masks as a contiguous
            // range.
            std::swap(j, trainingMasks[i][last]);
            auto poly = this->fit(trainingMasks[i].cbegin(),
                                  trainingMasks[i].cbegin() + last, k, xj);
            std::swap(j, trainingMasks[i][last]);
            residuals.addSamples({yj - poly.predict(xj)}, maths_t::CUnitWeights::SINGLE_UNIT);
        }
        LOG_TRACE(<< "residual distribution = " << residuals.print());

        testResiduals.clear();
        testResiduals.reserve(testingMasks[i].size());
        for (auto j : testingMasks[i]) {
            double xj;
            double yj;
            std::tie(xj, yj) = m_Data[j];
            auto poly = this->fit(trainingMasks[i].cbegin(),
                                  trainingMasks[i].cend(), k, xj);
            testResiduals.push_back(yj - poly.predict(xj));
        }
        weights.assign(testingMasks[i].size(), maths_t::CUnitWeights::UNIT);
        LOG_TRACE(<< "test residuals = " << testResiduals);

        double likelihood;
        residuals.jointLogMarginalLikelihood(testResiduals, weights, likelihood);
        result += likelihood;
    }
    LOG_TRACE(<< "k = " << k << ", likelihood = " << result);

    return result;
}

template<std::size_t N>
typename CLowess<N>::TPolynomial
CLowess<N>::fit(TSizeVecCItr beginMask, TSizeVecCItr endMask, double k, double x) const {
    TPolynomial poly;
    for (auto i = beginMask; i != endMask; ++i) {
        double xi;
        double yi;
        std::tie(xi, yi) = m_Data[*i];
        poly.add(xi, yi, this->weight(k, xi, x));
    }
    return poly;
}

template<std::size_t N>
double CLowess<N>::weight(double k, double x1, double x2) const {
    return std::exp(-k * std::fabs(x2 - x1));
}
}
}

#endif // INCLUDED_ml_maths_CLowessDetail_h
