/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLassoLogisticRegression.h>

#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(CLassoLogisticRegressionTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrDoublePr = std::pair<TSizeSizePr, double>;
using TSizeSizePrDoublePrVec = std::vector<TSizeSizePrDoublePr>;

template<typename ARRAY>
void initializeMatrix(const ARRAY& x_, TDoubleVecVec& x) {
    x.resize(boost::size(x_[0]), TDoubleVec(boost::size(x_), 0.0));
    for (std::size_t i = 0; i < boost::size(x_); ++i) {
        for (std::size_t j = 0; j < boost::size(x_[i]); ++j) {
            x[j][i] = x_[i][j];
        }
    }
}

template<typename ARRAY>
void initializeMatrix(const ARRAY& x_, TSizeSizePrDoublePrVec& x) {
    for (std::size_t i = 0; i < boost::size(x_); ++i) {
        for (std::size_t j = 0; j < boost::size(x_[i]); ++j) {
            if (x_[i][j] > 0.0) {
                x.push_back(TSizeSizePrDoublePr(TSizeSizePr(j, i), x_[i][j]));
            }
        }
    }
}

double inner(const TDoubleVec& x, const TDoubleVec& y) {
    double result = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }
    return result;
}

double logLikelihood(const TDoubleVecVec& x,
                     const TDoubleVec& y,
                     const TDoubleVec& lambda,
                     const TDoubleVec& beta) {
    double result = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        double f = 0.0;
        for (std::size_t j = 0; j < beta.size(); ++j) {
            f += beta[j] * x[j][i];
        }
        result -= std::log(1.0 + std::exp(-f * y[i]));
    }
    for (std::size_t j = 0; j < beta.size(); ++j) {
        result -= lambda[j] * std::fabs(beta[j]);
    }
    return result;
}
}

BOOST_AUTO_TEST_CASE(testCyclicCoordinateDescent) {
    static const double EPS = 5e-3;

    test::CRandomNumbers rng;

    // The training data are linearly separable.
    //
    // Expect p ~= exp( C * (x - 0.55) ) / (1 + exp( C * (x - 0.55) ))
    // where C = C(lambda).
    //
    // Check that:
    //   1) Dense and sparse solutions are the same,
    //   2) We are at a local minimum of the objective function.
    {
        maths::lasso_logistic_regression_detail::CCyclicCoordinateDescent clg(50, 0.001);

        TDoubleVec lambda(2, 0.25);
        double x_[][2] = {{0.1, 1.0}, {0.3, 1.0}, {0.4, 1.0}, {0.0, 1.0},
                          {1.0, 1.0}, {0.6, 1.0}, {0.7, 1.0}, {0.45, 1.0}};
        double y_[] = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0};

        TDoubleVecVec x;
        initializeMatrix(x_, x);
        TSizeSizePrDoublePrVec xs_;
        initializeMatrix(x_, xs_);
        TDoubleVec y(std::begin(y_), std::end(y_));

        TDoubleVec beta1;
        std::size_t numberIterations;
        clg.run(x, y, lambda, beta1, numberIterations);
        LOG_DEBUG(<< "dense beta = " << core::CContainerPrinter::print(beta1)
                  << ", numberIterations = " << numberIterations);

        TDoubleVec beta2;
        maths::lasso_logistic_regression_detail::CLrSparseMatrix xs(
            boost::size(x_), boost::size(x_[0]), xs_);
        clg.run(xs, y, lambda, beta2, numberIterations);
        LOG_DEBUG(<< "sparse beta = " << core::CContainerPrinter::print(beta2)
                  << ", numberIterations = " << numberIterations);

        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(beta1),
                            core::CContainerPrinter::print(beta2));

        initializeMatrix(x_, x);
        double ll = logLikelihood(x, y, lambda, beta1);
        LOG_DEBUG(<< "log-likelihood = " << ll);

        double llMinusEps = 0.0;
        double llPlusEps = 0.0;
        for (std::size_t i = 0; i < 10; ++i) {
            TDoubleVec step;
            rng.generateUniformSamples(0.0, EPS, beta1.size(), step);

            TDoubleVec betaMinusEps;
            TDoubleVec betaPlusEps;
            double length = 0.0;
            for (std::size_t j = 0; j < beta1.size(); ++j) {
                betaMinusEps.push_back(beta1[j] - step[j]);
                betaPlusEps.push_back(beta1[j] + step[j]);
                length += step[j] * step[j];
            }
            length = 2.0 * std::sqrt(length);

            llMinusEps += logLikelihood(x, y, lambda, betaMinusEps);
            llPlusEps += logLikelihood(x, y, lambda, betaPlusEps);
            LOG_DEBUG(<< "log-likelihood minus eps = "
                      << llMinusEps / static_cast<double>(i + 1));
            LOG_DEBUG(<< "log-likelihood plus eps  = "
                      << llPlusEps / static_cast<double>(i + 1));

            double slope = (llPlusEps - llMinusEps) / length;
            LOG_DEBUG(<< "slope = " << slope);
            BOOST_TEST_REQUIRE(slope < 0.015);
        }
        BOOST_TEST_REQUIRE(ll > llMinusEps / 10.0);
        BOOST_TEST_REQUIRE(ll > llPlusEps / 10.0);
    }

    double lambdas[] = {2.5, 5.0, 10.0, 15.0, 20.0};

    maths::lasso_logistic_regression_detail::CCyclicCoordinateDescent clg(100, 0.001);

    // Generate linearly separable data along a random direction
    // in R^5.

    double decision = 1.0;
    TDoubleVec decisionNormal;
    rng.generateUniformSamples(0.0, 1.0, 5, decisionNormal);
    double length = std::sqrt(inner(decisionNormal, decisionNormal));
    for (std::size_t j = 0; j < decisionNormal.size(); ++j) {
        decisionNormal[j] /= length;
    }
    LOG_DEBUG(<< "decisionNormal = " << core::CContainerPrinter::print(decisionNormal));

    TDoubleVecVec x_(6, TDoubleVec(100, 0.0));
    TDoubleVec y_(100, 0.0);
    for (std::size_t i = 0; i < 100; ++i) {
        TDoubleVec xi;
        rng.generateUniformSamples(-20.0, 20.0, 5, xi);
        double yi = std::sqrt(inner(decisionNormal, xi)) > decision ? 1.0 : -1.0;
        for (std::size_t j = 0; j < xi.size(); ++j) {
            x_[j][i] = xi[j];
        }
        x_[5][i] = 1.0;
        y_[i] = yi;
    }

    for (std::size_t k = 0; k < boost::size(lambdas); ++k) {
        TDoubleVec lambda(6, lambdas[k]);
        TDoubleVecVec x(x_);
        TDoubleVec y(y_);

        TDoubleVec beta;
        std::size_t numberIterations;
        clg.run(x, y, lambda, beta, numberIterations);
        LOG_DEBUG(<< "beta = " << core::CContainerPrinter::print(beta)
                  << ", numberIterations = " << numberIterations);

        TDoubleVec effectiveDecisionNormal;
        for (std::size_t j = 0; j < decisionNormal.size(); ++j) {
            effectiveDecisionNormal.push_back(beta[j]);
        }

        double theta =
            std::acos(inner(effectiveDecisionNormal, decisionNormal) /
                      std::sqrt(inner(effectiveDecisionNormal, effectiveDecisionNormal))) *
            360.0 / boost::math::double_constants::two_pi;
        LOG_DEBUG(<< "angular error = " << theta << " deg");
        BOOST_TEST_REQUIRE(theta < 7.5);
    }

    // Generate three features with monotonically increasing
    // correlation to the labels. We should eliminate them
    // in order as we increase lambda.
}

BOOST_AUTO_TEST_CASE(testCyclicCoordinateDescentLargeSparse, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testCyclicCoordinateDescentIncremental, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testNormBasedLambda, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testCrossValidatedLambda, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_SUITE_END()
