/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMic.h>

#include <core/CLogger.h>

#include <maths/CTools.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
namespace {
const std::size_t X{0};
const std::size_t Y{1};
const double INF{std::numeric_limits<double>::max()};
const double EPS{std::numeric_limits<double>::epsilon()};
}

void CMic::reserve(std::size_t n) {
    m_Samples.reserve(n);
}

const CMic& CMic::operator+=(const CMic& other) {
    m_Samples.insert(m_Samples.end(), other.m_Samples.begin(), other.m_Samples.end());
    return *this;
}

void CMic::add(double x, double y) {
    TVector2d p;
    p(0) = x;
    p(1) = y;
    m_Samples.push_back(p);
}

double CMic::compute() {

    if (m_Samples.size() < 4) {
        return 0.0;
    }

    // Compute the MICe statistic.

    this->setup();

    // We follow the paper naming convention here denoting the maximum grid size B,
    // the equipartition of the Y-axis Q and the maximum number of rows and columns
    // in the grid by l and k, respectively.

    std::size_t b{this->maximumGridSize()};
    std::size_t rootb{static_cast<std::size_t>(std::sqrt(static_cast<double>(b)))};
    LOG_TRACE(<< "B(n) = " << b << ", sqrt(B(n)) = " << rootb);

    double mic{0.0};
    for (std::size_t i = 0; i < 2; ++i) {

        for (std::size_t l = 2; l < rootb; ++l) {
            TDoubleVec q(this->equipartitionAxis(Y, l));
            LOG_TRACE(<< "Q = " << core::CContainerPrinter::print(q));
            // Note in the case of points with duplicate Y- values we still choose
            // k equal to the target size for Q here (since overall we still have
            // that kl < B(n) and we want to test additional grids).
            std::size_t ldistinct{q.size()};
            std::size_t k{l};
            TDoubleVec mi{this->optimizeXAxis(q, ldistinct, k)};
            for (std::size_t p = 0; p < mi.size(); ++p) {
                mic = std::max(mic, mi[p] / CTools::fastLog(static_cast<double>(p + 2)));
            }
        }

        for (std::size_t l = rootb; l < b / 2; ++l) {
            TDoubleVec q(this->equipartitionAxis(Y, l));
            LOG_TRACE(<< "Q = " << core::CContainerPrinter::print(q));
            // The same rational for choice of k in the presence of duplicates.
            std::size_t ldistinct{q.size()};
            std::size_t k{b / l};
            TDoubleVec mi{this->optimizeXAxis(q, ldistinct, k)};
            for (std::size_t p = 0; p < mi.size(); ++p) {
                mic = std::max(mic, mi[p] / CTools::fastLog(static_cast<double>(p + 2)));
            }
        }

        for (auto& sample : m_Samples) {
            std::swap(sample(0), sample(1));
        }
        std::swap(m_Order[0], m_Order[1]);
    }

    return mic;
}

void CMic::setup() {
    for (auto variable : {X, Y}) {
        m_Order[variable].resize(m_Samples.size());
        std::iota(m_Order[variable].begin(), m_Order[variable].end(), 0);
        std::sort(m_Order[variable].begin(), m_Order[variable].end(),
                  [variable, this](std::size_t lhs, std::size_t rhs) {
                      return m_Samples[lhs](variable) < m_Samples[rhs](variable);
                  });
    }
}

std::size_t CMic::maximumGridSize() const {

    // We use the parameters for the maximum grid size given in "An Empirical Study
    // of Leading Measures of Dependence" Reshef et al. See table E-4.

    const double MINIMUM_SAMPLE_SIZE{50.0};
    const double MAXIMUM_SAMPLE_SIZE{1000000.0};
    const double knotSampleSizes[]{MINIMUM_SAMPLE_SIZE - 1.0,
                                   MINIMUM_SAMPLE_SIZE,
                                   100.0,
                                   500.0,
                                   1000.0,
                                   5000.0,
                                   10000.0,
                                   50000.0,
                                   MAXIMUM_SAMPLE_SIZE,
                                   MAXIMUM_SAMPLE_SIZE + 1.0};
    const double knotGridSizeExponents[]{0.85, 0.85, 0.8, 0.8,  0.75,
                                         0.65, 0.6,  0.5, 0.45, 0.45};

    double clampedSampleSize{std::max(
        std::min(static_cast<double>(m_Samples.size()), MAXIMUM_SAMPLE_SIZE), MINIMUM_SAMPLE_SIZE)};

    auto rightKnot = std::lower_bound(std::begin(knotSampleSizes),
                                      std::end(knotSampleSizes), clampedSampleSize);
    auto leftKnot = rightKnot - 1;

    double exponentAtRightKnot{knotGridSizeExponents[rightKnot - std::begin(knotSampleSizes)]};
    double exponentAtLeftKnot{knotGridSizeExponents[leftKnot - std::begin(knotSampleSizes)]};

    return static_cast<std::size_t>(
        std::pow(static_cast<double>(m_Samples.size()),
                 CTools::linearlyInterpolate(*leftKnot, *rightKnot, exponentAtLeftKnot,
                                             exponentAtRightKnot, clampedSampleSize)) +
        0.5);
}

double CMic::maximumXAxisPartitionSizeToSearch() const {

    // We use the parameters for the partition to search for the maximum mutual
    // information given in "An Empirical Study of Leading Measures of Dependence"
    // Reshef et al. See table E-4.

    const double MINIMUM_SAMPLE_SIZE{50.0};
    const double MAXIMUM_SAMPLE_SIZE{5000.0};
    double knotSampleSizes[]{MINIMUM_SAMPLE_SIZE - 1.0,
                             MINIMUM_SAMPLE_SIZE,
                             100.0,
                             500.0,
                             1000.0,
                             MAXIMUM_SAMPLE_SIZE,
                             MAXIMUM_SAMPLE_SIZE + 1.0};
    double knotPartitionSizes[]{5.0, 5.0, 5.0, 5.0, 4.0, 1.0, 1.0};

    double clampedSampleSize{std::max(
        std::min(static_cast<double>(m_Samples.size()), MAXIMUM_SAMPLE_SIZE), MINIMUM_SAMPLE_SIZE)};

    auto rightKnot = std::lower_bound(std::begin(knotSampleSizes),
                                      std::end(knotSampleSizes), clampedSampleSize);
    auto leftKnot = rightKnot - 1;

    double sizeAtRightKnot{knotPartitionSizes[rightKnot - std::begin(knotSampleSizes)]};
    double sizeAtLeftKnot{knotPartitionSizes[leftKnot - std::begin(knotSampleSizes)]};

    return CTools::linearlyInterpolate(*leftKnot, *rightKnot, sizeAtLeftKnot,
                                       sizeAtRightKnot, clampedSampleSize);
}

CMic::TDoubleVec CMic::equipartitionAxis(std::size_t variable, std::size_t l) const {

    // We seek an l-partition of "variable"-axis into buckets of approximately
    // equal numbers of points. This calculates the values of "variable" for
    // the partition boundaries.

    LOG_TRACE(<< "Equipartition " << (variable == 0 ? "x" : "y") << "-axis");

    TDoubleVec partitionBoundaries;
    partitionBoundaries.reserve(l + 1);

    std::size_t desired{std::max((m_Samples.size() + l - 1) / l, std::size_t{1})};
    LOG_TRACE(<< "l = " << l << ", desired bucket size = " << desired);

    for (std::size_t i = desired; i < m_Order[variable].size(); i += desired) {
        std::size_t variableIthOrderStatisticIndex{m_Order[variable][i]};
        partitionBoundaries.push_back(m_Samples[variableIthOrderStatisticIndex](variable));
    }
    partitionBoundaries.erase(
        std::unique(partitionBoundaries.begin(), partitionBoundaries.end()),
        partitionBoundaries.end());
    double max{m_Samples[m_Order[variable].back()](variable)};
    partitionBoundaries.push_back((1.0 + std::copysign(EPS, max)) * max);

    return partitionBoundaries;
}

CMic::TDoubleVec CMic::optimizeXAxis(const TDoubleVec& q, std::size_t l, std::size_t k) const {

    // Compute the k-partition of the X-axis which maximises the mutual information
    // between the variables given a fixed l-partition of the Y-axis.

    using TDoubleVecVec = std::vector<TDoubleVec>;

    LOG_TRACE(<< "Optimize x-axis, k = " << k);

    std::size_t ck{static_cast<std::size_t>(this->maximumXAxisPartitionSizeToSearch() * k)};
    LOG_TRACE(<< "c * k = " << ck);

    TDoubleVec pi(this->equipartitionAxis(X, ck));
    LOG_TRACE(<< "pi = " << core::CContainerPrinter::print(pi));
    ck = std::min(ck, pi.size());

    double n{static_cast<double>(m_Samples.size())};
    double w{1.0 / n};

    // Compute grid cell and row and column probabilities.
    TVectorXdVec cump(ck, TVectorXd{l});
    for (const auto& sample : m_Samples) {
        std::ptrdiff_t i{std::upper_bound(pi.begin(), pi.end(), sample(X)) - pi.begin()};
        std::ptrdiff_t j{std::upper_bound(q.begin(), q.end(), sample(Y)) - q.begin()};
        cump[i](j) += w;
    }

    // We need cumulative probabilities in each column for the dynamic program.
    for (std::size_t i = 1; i < ck; ++i) {
        cump[i] += cump[i - 1];
    }

    TDoubleVecVec M(ck);

    // Compute the optimal 2-subset of the X-axis partition pi.
    //
    // In the following we use H_X(s,t) to denote the entropy of the two partition
    // of the X-axis at s and t and H_XY(s,t) to denote the entropy of the 2 x l
    // grid generated by merging X- cells [0,...,s) and [s,...,t). Note this is
    // why we work with cumulative probabilities. (This differs only by a constant
    // H_Y from the mutual information so shares the same maximum. We add on H_Y
    // at the end.)
    //
    // for t = 2 to c k do
    //   Find s in {1,...,t} maximizing H_X(s,t) − H_XY(s,t)
    //   M(t,2) <- H_X(s,t) − H_XY(s,t)
    // end for

    for (std::size_t t = 2; t <= ck; ++t) {

        double Fmax{-INF};

        double Z{cump[t - 1].L1()};

        TVectorXd p{2};
        for (std::size_t s = 1; s < t; ++s) {
            TVectorXd pq0{cump[s - 1] / Z};
            TVectorXd pq1{(cump[t - 1] - cump[s - 1]) / Z};
            p(0) = pq0.L1();
            p(1) = pq1.L1();
            double F{entropy(p) - entropy(pq0) - entropy(pq1)};
            Fmax = std::max(Fmax, F);
        }

        M[t - 1].reserve(k);
        M[t - 1].push_back(Fmax);
    }
    LOG_TRACE(<< "mutual information = " << core::CContainerPrinter::print(M));

    // A dynamic program to compute the optimal 3- to k-subsets of partition pi.
    //
    // Let M(t,x) denote the maximum of H_X(P_x) - H_XY(P_x) for all partitions of
    // the X-axis that satisfy |P_x| = x. One can show that M(t,x) satisfies:
    //
    // M(t,x) = max_{x-1<=s<=t}{ Z_s / Z_t M(s,x-1) + (Z_t-Z_s) / Z_t H_XY(s,t) }
    //
    // where Z_s is the count of samples whose X- value is less than s and Z_t is
    // the count of samples whose X- value is less than t whence:
    //
    // for x = 3 to k do
    //   for t = x to c k do
    //     Find s in {x-1,...,t} maximizing Z_s / Z_t M(s,x-1) + (Z_t-Z_s) / Z_t H_XY(s,t)
    //     M(t,x) <- Z_s / Z_t M(s,x-1) + (Z_t-Z_s) / Z_t H_XY(s,t)

    for (std::size_t x = 3; x <= k; ++x) {
        for (std::size_t t = x; t <= ck; ++t) {

            double Fmax{-INF};

            double Zt{cump[t - 1].L1()};
            for (std::size_t s = x - 1; s < t; ++s) {
                double Zs{cump[s - 1].L1()};
                TVectorXd pst{cump[t - 1] - cump[s - 1]};
                pst /= Zt - Zs;
                double F{Zs / Zt * M[s - 1][x - 3] - (Zt - Zs) / Zt * entropy(pst)};
                Fmax = std::max(Fmax, F);
            }

            M[t - 1].push_back(Fmax);
        }
    }

    // Add H_Y to recover mutual information, which we need when comparing grids.

    LOG_TRACE(<< "Z = " << cump.back().L1());
    double Hq{entropy(cump.back())};
    TDoubleVec result(M.back());
    for (auto& r : result) {
        r += Hq;
    }

    LOG_TRACE(<< "max mutual information(k = " << k
              << ") = " << core::CContainerPrinter::print(result));

    return result;
}

double CMic::entropy(const TVectorXd& dist) {
    double result{0.0};
    for (auto p : dist) {
        if (p > 0.0) {
            result -= p * CTools::fastLog(p);
        }
    }
    return result;
}
}
}
