/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CAssignment.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <vector>

BOOST_AUTO_TEST_SUITE(CAssignmentTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrVec = std::vector<TSizeSizePr>;

template<std::size_t N, std::size_t M>
void fill(const double (&costs)[N][M], TDoubleVecVec& result) {
    for (std::size_t i = 0; i < N; ++i) {
        result.push_back(TDoubleVec());
        for (std::size_t j = 0; j < M; ++j) {
            result.back().push_back(costs[i][j]);
        }
    }
}

void fill(const TDoubleVec& costs, TDoubleVecVec& result) {
    std::size_t n =
        static_cast<std::size_t>(std::sqrt(static_cast<double>(costs.size())));
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(TDoubleVec());
        result.back().reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            result.back().push_back(costs[i * n + j]);
        }
    }
}

double cost(const TDoubleVecVec& costs, const TSizeSizePrVec& matching) {
    double result = 0.0;
    for (std::size_t i = 0; i < matching.size(); ++i) {
        result += costs[matching[i].first][matching[i].second];
    }
    return result;
}

double match(const TDoubleVecVec& costs, TSizeSizePrVec& matching) {
    std::size_t n = costs.size();
    TSizeVec permutation;
    for (std::size_t i = 0; i < n; ++i) {
        permutation.push_back(i);
    }

    double minCost = std::numeric_limits<double>::max();
    do {
        double cost = 0.0;
        for (std::size_t i = 0; i < costs.size(); ++i) {
            cost += costs[i][permutation[i]];
        }
        if (cost < minCost) {
            minCost = cost;
            matching.clear();
            for (std::size_t i = 0; i < permutation.size(); ++i) {
                matching.push_back(TSizeSizePr(i, permutation[i]));
            }
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    return minCost;
}
}

BOOST_AUTO_TEST_CASE(testKuhnMunkres) {
    {
        LOG_DEBUG(<< "test 1: bad input");
        const double test11[][5] = {
            {2.0, 1.0, 1.0, 2.0, 2.0},
            {1.0, 2.0, 2.0, 2.0, 2.0},
        };
        const double test12[][6] = {{1.0, 1.0, 2.0, 2.0, 2.0, 3.0}};
        TDoubleVecVec costs;
        fill(test11, costs);
        fill(test12, costs);

        TSizeSizePrVec matching;
        BOOST_TEST_REQUIRE(!maths::CAssignment::kuhnMunkres(costs, matching));
    }
    {
        LOG_DEBUG(<< "test 2: 5x5");
        const double test2[][5] = {{2.0, 1.0, 1.0, 2.0, 2.0},
                                   {1.0, 2.0, 2.0, 2.0, 2.0},
                                   {2.0, 2.0, 2.0, 1.0, 2.0},
                                   {1.0, 1.0, 2.0, 2.0, 2.0},
                                   {2.0, 2.0, 2.0, 2.0, 1.0}};
        TDoubleVecVec costs;
        fill(test2, costs);

        TSizeSizePrVec matching;
        BOOST_TEST_REQUIRE(maths::CAssignment::kuhnMunkres(costs, matching));

        LOG_DEBUG(<< "matching = " << core::CContainerPrinter::print(matching));
        BOOST_REQUIRE_EQUAL(5.0, cost(costs, matching));
    }
    {
        LOG_DEBUG(<< "test 3: 5x4");
        const double test3[][4] = {{2.0, 1.0, 1.0, 2.0},
                                   {1.0, 2.0, 2.0, 2.0},
                                   {2.0, 2.0, 2.0, 1.0},
                                   {1.0, 1.0, 2.0, 2.0},
                                   {2.0, 2.0, 2.0, 2.0}};
        TDoubleVecVec costs;
        fill(test3, costs);

        TSizeSizePrVec matching;
        BOOST_TEST_REQUIRE(maths::CAssignment::kuhnMunkres(costs, matching));

        LOG_DEBUG(<< "matching = " << core::CContainerPrinter::print(matching));
        BOOST_REQUIRE_EQUAL(4.0, cost(costs, matching));
    }
    {
        LOG_DEBUG(<< "test 4: 4x5");
        const double test4[][5] = {
            {2.0, 1.0, 1.0, 2.0, 2.0},
            {1.0, 2.0, 2.0, 2.0, 2.0},
            {2.0, 2.0, 2.0, 1.0, 2.0},
            {1.0, 1.0, 2.0, 2.0, 2.0},
        };
        TDoubleVecVec costs;
        fill(test4, costs);

        TSizeSizePrVec matching;
        BOOST_TEST_REQUIRE(maths::CAssignment::kuhnMunkres(costs, matching));

        LOG_DEBUG(<< "matching = " << core::CContainerPrinter::print(matching));
        BOOST_REQUIRE_EQUAL(4.0, cost(costs, matching));
    }

    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "test 5: small random");
        for (std::size_t i = 2; i < 9; ++i) {
            LOG_DEBUG(<< "***" << i << "x" << i);
            for (std::size_t test = 0; test < 100; ++test) {
                TDoubleVec samples;
                rng.generateUniformSamples(0.1, 1000.0, i * i, samples);

                TDoubleVecVec costs;
                fill(samples, costs);

                if (test % 10 == 0) {
                    LOG_DEBUG(<< "costs = " << core::CContainerPrinter::print(costs));
                }

                TSizeSizePrVec expectedMatching;
                double expectedCost = match(costs, expectedMatching);
                if (test % 10 == 0) {
                    LOG_DEBUG(<< "expectedCost = " << expectedCost);
                }

                TSizeSizePrVec matching;
                maths::CAssignment::kuhnMunkres(costs, matching);
                if (test % 10 == 0) {
                    LOG_DEBUG(<< "cost = " << cost(costs, matching));
                }

                BOOST_REQUIRE_EQUAL(expectedCost, cost(costs, matching));
            }
        }
    }

    {
        LOG_DEBUG(<< "test 6: large");

        // Sanity check for non-square: if we embed a non-square
        // matrix in a larger non-square matrix whose additional
        // values are all greater than we should find the same
        // cost assignment.

        TDoubleVecVec costs;
        TDoubleVec samples;
        rng.generateUniformSamples(0.1, 2.0, 10000u, samples);
        fill(samples, costs);

        TSizeSizePrVec matching;
        BOOST_TEST_REQUIRE(maths::CAssignment::kuhnMunkres(costs, matching));

        double optimalCost = cost(costs, matching);
        LOG_DEBUG(<< "cost = " << optimalCost);

        // Try some random permutations random permutations and check
        // we don't find a lower cost solution.
        TSizeVec randomMatching;
        for (std::size_t i = 0; i < costs.size(); ++i) {
            randomMatching.push_back(i);
        }

        double lowestRandomCost = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < 1000; ++i) {
            rng.random_shuffle(randomMatching.begin(), randomMatching.end());
            double cost = 0.0;
            for (std::size_t j = 0; j < costs.size(); ++j) {
                cost += costs[j][randomMatching[j]];
            }
            lowestRandomCost = std::min(lowestRandomCost, cost);
        }

        LOG_DEBUG(<< "optimal cost = " << optimalCost
                  << ", lowest random cost = " << lowestRandomCost);
        BOOST_TEST_REQUIRE(lowestRandomCost >= optimalCost);

        // Check adding higher cost row has no effect.
        {
            costs.push_back(TDoubleVec(100, 2.1));
            BOOST_TEST_REQUIRE(maths::CAssignment::kuhnMunkres(costs, matching));
            BOOST_REQUIRE_EQUAL(optimalCost, cost(costs, matching));
        }
    }

    {
        LOG_DEBUG(<< "test 7: euler 345");
        const double euler345[][15] = {
            {7, 53, 183, 439, 863, 497, 383, 563, 79, 973, 287, 63, 343, 169, 583},
            {627, 343, 773, 959, 943, 767, 473, 103, 699, 303, 957, 703, 583, 639, 913},
            {447, 283, 463, 29, 23, 487, 463, 993, 119, 883, 327, 493, 423, 159, 743},
            {217, 623, 3, 399, 853, 407, 103, 983, 89, 463, 290, 516, 212, 462, 350},
            {960, 376, 682, 962, 300, 780, 486, 502, 912, 800, 250, 346, 172, 812, 350},
            {870, 456, 192, 162, 593, 473, 915, 45, 989, 873, 823, 965, 425, 329, 803},
            {973, 965, 905, 919, 133, 673, 665, 235, 509, 613, 673, 815, 165, 992, 326},
            {322, 148, 972, 962, 286, 255, 941, 541, 265, 323, 925, 281, 601, 95, 973},
            {445, 721, 11, 525, 473, 65, 511, 164, 138, 672, 18, 428, 154, 448, 848},
            {414, 456, 310, 312, 798, 104, 566, 520, 302, 248, 694, 976, 430, 392, 198},
            {184, 829, 373, 181, 631, 101, 969, 613, 840, 740, 778, 458, 284, 760, 390},
            {821, 461, 843, 513, 17, 901, 711, 993, 293, 157, 274, 94, 192, 156, 574},
            {34, 124, 4, 878, 450, 476, 712, 914, 838, 669, 875, 299, 823, 329, 699},
            {815, 559, 813, 459, 522, 788, 168, 586, 966, 232, 308, 833, 251, 631, 107},
            {813, 883, 451, 509, 615, 77, 281, 613, 459, 205, 380, 274, 302, 35, 805}};
        TDoubleVecVec costs;
        fill(euler345, costs);
        for (std::size_t i = 0; i < costs.size(); ++i) {
            for (std::size_t j = 0; j < costs[i].size(); ++j) {
                costs[i][j] = 1000 - costs[i][j];
            }
        }

        TSizeSizePrVec matching;
        maths::CAssignment::kuhnMunkres(costs, matching);
        LOG_DEBUG(<< "cost = " << 15000.0 - cost(costs, matching));
        BOOST_REQUIRE_EQUAL(13938.0, 15000.0 - cost(costs, matching));
    }
}

BOOST_AUTO_TEST_SUITE_END()
