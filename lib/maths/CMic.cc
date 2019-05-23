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

    // Compute the MICe statistic. See "Measuring Dependence Powerfully and Equitably"
    // Reshef et al.

    this->setup();

    std::size_t b{this->b()};
    std::size_t rootb{static_cast<std::size_t>(std::sqrt(static_cast<double>(b)))};
    LOG_TRACE(<< "B(n) = " << b << ", sqrt(B(n)) = " << rootb);

    double mic{0.0};
    for (std::size_t i = 0; i < 2; ++i) {

        for (std::size_t l = 2; l < rootb; ++l) {
            TDoubleVec q(this->equipartitionAxis(Y, l));
            TDoubleVec mi{this->optimizeXAxis(q, q.size(), q.size())};
            for (std::size_t p = 0; p < mi.size(); ++p) {
                mic = std::max(mic, mi[p] / CTools::fastLog(static_cast<double>(p + 2)));
            }
        }

        for (std::size_t l = rootb; l < b / 2; ++l) {
            TDoubleVec q(this->equipartitionAxis(Y, l));
            TDoubleVec mi{this->optimizeXAxis(q, q.size(), b / q.size())};
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

std::size_t CMic::b() const {

    // We use the parameters for the maximum grid size given in "An Empirical Study
    // of Leading Measures of Dependence" Reshef et al. See table E-4.

    double n{std::max(std::min(static_cast<double>(m_Samples.size()), 100000.0), 50.0)};

    double x[]{49.0,   50.0,    100.0,   500.0,    1000.0,
               5000.0, 10000.0, 50000.0, 100000.0, 100001.0};
    double f[]{0.85, 0.85, 0.8, 0.8, 0.75, 0.65, 0.6, 0.5, 0.45, 0.45};

    auto b = std::lower_bound(std::begin(x), std::end(x), n);
    auto a = b - 1;

    double fb{f[b - std::begin(x)]};
    double fa{f[a - std::begin(x)]};

    return static_cast<std::size_t>(
        std::pow(n, CTools::linearlyInterpolate(*a, *b, fa, fb, n)) + 0.5);
}

double CMic::c() const {

    // We use the parameters for the partition to search for the maximum mutual
    // information given in "An Empirical Study of Leading Measures of Dependence"
    // Reshef et al. See table E-4.

    double n{std::max(std::min(static_cast<double>(m_Samples.size()), 5000.0), 50.0)};

    double x[]{49.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 5001.0};
    double f[]{5.0, 5.0, 5.0, 5.0, 4.0, 1.0, 1.0};

    auto b = std::lower_bound(std::begin(x), std::end(x), n);
    auto a = b - 1;

    double fb{f[b - std::begin(x)]};
    double fa{f[a - std::begin(x)]};

    return CTools::linearlyInterpolate(*a, *b, fa, fb, n);
}

CMic::TDoubleVec CMic::equipartitionAxis(std::size_t variable, std::size_t l) const {

    // We seek an l-partition of the y-axis into buckets of approximately equal
    // numbers of points.

    LOG_TRACE(<< "Equipartition " << (variable == 0 ? "x" : "y") << "-axis");

    TDoubleVec result;
    result.reserve(l + 1);

    std::size_t desired{std::max((m_Samples.size() + l - 1) / l, std::size_t{1})};
    LOG_TRACE(<< "l = " << l << ", desired bucket size = " << desired);

    for (std::size_t i = desired; i < m_Order[variable].size(); i += desired) {
        result.push_back(m_Samples[m_Order[variable][i]](variable));
    }
    result.erase(std::unique(result.begin(), result.end()), result.end());
    double max{m_Samples[m_Order[variable].back()](variable)};
    result.push_back((1.0 + std::copysign(EPS, max)) * max);
    LOG_TRACE(<< "partition = " << core::CContainerPrinter::print(result));

    return result;
}

CMic::TDoubleVec CMic::optimizeXAxis(const TDoubleVec& q, std::size_t l, std::size_t k) const {

    // Compute the split k-partition of the x-axis which maximises the mutual
    // information between the variables given a fixed partition of the y-axis.

    using TDoubleVecVec = std::vector<TDoubleVec>;

    LOG_TRACE(<< "Optimize x-axis, k = " << k);

    std::size_t ck{static_cast<std::size_t>(this->c() * k)};
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

    // We need to cumulative probabilities in each column for the dynamic program.
    for (std::size_t i = 1; i < ck; ++i) {
        cump[i] += cump[i - 1];
    }

    TDoubleVecVec M(ck);

    // Compute the optimal 2-subset of the x-axis partition pi.
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

    // Dynamic program to compute the optimal 3- to k-subsets of partition pi.
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
