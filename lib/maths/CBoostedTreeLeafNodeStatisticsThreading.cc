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

#include <maths/CBoostedTreeLeafNodeStatisticsThreading.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CTools.h>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;

std::size_t CBoostedTreeLeafNodeStatisticsThreading::numberThreadsForAggregateLossDerivatives(
    std::size_t numberThreads,
    std::size_t features,
    std::size_t rows) {

    // The number of threads we'll use breaks down as follows:
    //   - We need a minimum number of rows per thread to ensure reasonable
    //     load balancing.
    //   - We need a minimum amount of work per thread to make the overheads
    //     of distributing worthwhile.

    std::size_t rowsPerThreadConstraint{rows / 64};
    std::size_t workPerThreadConstraint{(features * rows) / (8 * 64)};
    return std::min(numberThreads, std::max(std::min(rowsPerThreadConstraint, workPerThreadConstraint),
                                            std::size_t{1}));
}

std::size_t CBoostedTreeLeafNodeStatisticsThreading::numberThreadsForAddSplitsDerivatives(
    std::size_t numberThreads,
    std::size_t features,
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double totalWork{addSplitsDerivativesTotalWork(numberLossParameters, numberDerivatives)};
    return CBasicStatistics::min(maximumThroughputNumberThreads(totalWork),
                                 features, numberThreads);
}

std::size_t CBoostedTreeLeafNodeStatisticsThreading::numberThreadsForSubtractSplitsDerivatives(
    std::size_t numberThreads,
    std::size_t features,
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double totalWork{subtractSplitsDerivativesTotalWork(numberLossParameters, numberDerivatives)};
    return CBasicStatistics::min(maximumThroughputNumberThreads(totalWork),
                                 features, numberThreads);
}

std::size_t CBoostedTreeLeafNodeStatisticsThreading::numberThreadsForRemapSplitsDerivatives(
    std::size_t numberThreads,
    std::size_t features,
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double totalWork{remapSplitsDerivativesTotalWork(numberLossParameters, numberDerivatives)};
    return CBasicStatistics::min(maximumThroughputNumberThreads(totalWork),
                                 features, numberThreads);
}

std::size_t CBoostedTreeLeafNodeStatisticsThreading::numberThreadsForComputeBestSplitStatistics(
    std::size_t numberThreads,
    std::size_t features,
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double totalWork{computeBestSplitStatisticsTotalWork(numberLossParameters, numberDerivatives)};
    return CBasicStatistics::min(maximumThroughputNumberThreads(totalWork),
                                 features, numberThreads);
}

double CBoostedTreeLeafNodeStatisticsThreading::addSplitsDerivativesTotalWork(
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {

    // Interestingly, you get a much better fit using a function of the form
    // a n + b n p (p + 1) / 2 even though the number of operations is just
    // proportional to n p (p + 1) / 2.

    double n{static_cast<double>(numberDerivatives)};
    double p{static_cast<double>(numberLossParameters * (numberLossParameters + 3) / 2)};
    return n * (1.0 / 2500.0 + p / 5000.0);
}

double CBoostedTreeLeafNodeStatisticsThreading::subtractSplitsDerivativesTotalWork(
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double n{static_cast<double>(numberDerivatives)};
    double p{static_cast<double>(numberLossParameters * (numberLossParameters + 1))};
    return n * (1.0 / 400.0 + p / 20000.0);
}

double CBoostedTreeLeafNodeStatisticsThreading::remapSplitsDerivativesTotalWork(
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double n{static_cast<double>(numberDerivatives)};
    double p{static_cast<double>(numberLossParameters * (numberLossParameters + 1) / 2)};
    return n * (1.0 / 500.0 + p / 4000.0);
}

double CBoostedTreeLeafNodeStatisticsThreading::computeBestSplitStatisticsTotalWork(
    std::size_t numberLossParameters,
    std::size_t numberDerivatives) {
    double d{static_cast<double>(numberLossParameters)};
    double n{static_cast<double>(numberDerivatives)};
    if (d == 1.0) {
        return n / 50.0;
    }
    if (d <= 5.0) {
        return n * (1.0 / 24.0 + static_cast<double>(d * d) / 30.0);
    }
    return n * (1.0 / 2.5 + static_cast<double>(d * d) / 180.0);
}

std::size_t CBoostedTreeLeafNodeStatisticsThreading::maximumThroughputNumberThreads(double totalWork) {

    // We assume that the total work is expressed as a function of cost
    // of parallel_for_each. The cost model we use is 1 + 0.6 * t for the
    // threading overhead for thread count t. The time to execute total
    // work w is then w / t. We seek an integer t^* satisfying
    //
    //   t^* = argmin_t{ (w / t + 1 + 0.6 t) 1{t > 1} + w 1{t == 1} }
    //
    // For t > 1 the function is smooth and we can differentiate to get the
    // optimal value, i.e. t = (w / 0.6)^(1/2). We need to check integers
    // either side and also the case t is 1.

    using TDoubleAry = std::array<double, 3>;

    double threads{std::sqrt(totalWork / 0.6)};

    if (threads < 1.0) {
        return 1;
    }

    auto duration = [&](double t) {
        return t > 1 ? totalWork / t + threadCost(t) : totalWork;
    };
    TDoubleAry result{1.0, std::ceil(threads), std::floor(threads)};
    TDoubleAry durations{duration(result[0]), duration(result[1]), duration(result[2])};
    return static_cast<std::size_t>(
        result[std::min_element(durations.begin(), durations.end()) - durations.begin()]);
}

double CBoostedTreeLeafNodeStatisticsThreading::threadCost(double numberThreads) {
    return 1.0 + 0.6 * numberThreads;
}

template<int d>
CBoostedTreeLeafNodeStatisticsThreading::TMinimumLoss
CBoostedTreeLeafNodeStatisticsThreading::makeThreadLocalMinimumLossFunction(double lambda) {
    static_assert(d <= 5, "Dimension too large");
    if (d == 1) {
        return [lambda](const Eigen::VectorXd& g, const Eigen::MatrixXd& h) -> double {
            return CTools::pow2(g(0)) / (h(0, 0) + lambda);
        };
    }
    return [
        lambda, hessian = Eigen::Matrix<double, d, d>{},
        hessianInvg = Eigen::Matrix<double, d, 1>{},
        llt = Eigen::LLT<Eigen::Matrix<double, d, d>>{}
    ](const Eigen::VectorXd& g, const Eigen::MatrixXd& h) mutable->double {
        hessian = (h + lambda * Eigen::Matrix<double, d, d>::Identity(d, d))
                      .template selfadjointView<Eigen::Lower>();
        // Since the Hessian is positive semidefinite, the trace is larger than the
        // largest eigenvalue. Therefore, H_eps = H + eps * trace(H) * I will have
        // condition number at least eps. As long as eps >> double epsilon we should
        // be able to invert it accurately.
        double eps{std::max(1e-5 * hessian.trace(), 1e-10)};
        for (std::size_t i = 0; i < 2; ++i) {
            llt.compute(hessian);
            hessianInvg.noalias() = g;
            llt.solveInPlace(hessianInvg);
            if ((hessian * hessianInvg - g).norm() < 1e-2 * g.norm()) {
                return g.transpose() * hessianInvg;
            }
            hessian.diagonal().array() += eps;
        }
        return -INF / 2.0; // We couldn't invert the Hessian: discard this split.
    };
}

CBoostedTreeLeafNodeStatisticsThreading::TMinimumLoss
CBoostedTreeLeafNodeStatisticsThreading::makeThreadLocalMinimumLossFunction(int numberLossParameters,
                                                                            double lambda) {

    // For very small numbers of classes it is significantly faster to use stack
    // based matrices and vectors.
    switch (numberLossParameters) {
    case 1:
        return makeThreadLocalMinimumLossFunction<1>(lambda);
    case 2:
        return makeThreadLocalMinimumLossFunction<2>(lambda);
    case 3:
        return makeThreadLocalMinimumLossFunction<3>(lambda);
    case 4:
        return makeThreadLocalMinimumLossFunction<4>(lambda);
    case 5:
        return makeThreadLocalMinimumLossFunction<5>(lambda);
    default:
        break;
    }

    // We preallocate all temporary objects so no allocations are made in the
    // hot loop where this is executed. The Eigen allocator consistently shows
    // up as a bottleneck and creates contention which largely kills threading
    // advantages. Note that this means one copy of the function must be used
    // per thread.
    int d{static_cast<int>(numberLossParameters)};
    return [
        hessian = Eigen::MatrixXd{d, d}, hessianInvg = Eigen::VectorXd{d},
        llt = Eigen::LLT<Eigen::MatrixXd>{d}, lambda, d
    ](const Eigen::VectorXd& g, const Eigen::MatrixXd& h) mutable->double {
        hessian = (h + lambda * Eigen::MatrixXd::Identity(d, d)).selfadjointView<Eigen::Lower>();
        // Since the Hessian is positive semidefinite, the trace is larger than the
        // largest eigenvalue. Therefore, H_eps = H + eps * trace(H) * I will have
        // condition number at least eps. As long as eps >> double epsilon we should
        // be able to invert it accurately.
        double eps{std::max(1e-5 * hessian.trace(), 1e-10)};
        for (std::size_t i = 0; i < 2; ++i) {
            llt.compute(hessian);
            hessianInvg.noalias() = g;
            llt.solveInPlace(hessianInvg);
            // Matrix products can allocate memory and using lazyProduct here to
            // force no allocations was a large performance win (especially for
            // for very small numbers of loss parameters).
            if ((hessian.lazyProduct(hessianInvg) - g).norm() < 1e-2 * g.norm()) {
                return g.transpose().lazyProduct(hessianInvg);
            }
            hessian.diagonal().array() += eps;
        }
        return -INF / 2.0; // We couldn't invert the Hessian: discard this split.
    };
}
}
}
