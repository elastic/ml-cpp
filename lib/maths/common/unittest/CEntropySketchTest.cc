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

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CEntropySketch.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <cmath>
#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(CEntropySketchTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testAll) {
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoubleUMap = boost::unordered_map<std::size_t, double>;
    using TSizeDoubleUMapCItr = TSizeDoubleUMap::const_iterator;

    test::CRandomNumbers rng;

    TSizeVec numberCategories;
    rng.generateUniformSamples(500, 1001, 1000, numberCategories);

    maths::common::CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>> maxError[3];
    maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator meanError[3];

    double K[] = {20.0, 40.0, 60.0};
    double eps[] = {0.2, 0.4, 0.6};
    double epsDeviations[][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    TSizeVec counts;
    for (std::size_t t = 0; t < numberCategories.size(); ++t) {
        rng.generateUniformSamples(1, 10, numberCategories[t], counts);
        std::size_t Z = std::accumulate(counts.begin(), counts.end(), 0);

        maths::common::CEntropySketch entropy[] = {
            maths::common::CEntropySketch(static_cast<std::size_t>(K[0])),
            maths::common::CEntropySketch(static_cast<std::size_t>(K[1])),
            maths::common::CEntropySketch(static_cast<std::size_t>(K[2]))};

        for (std::size_t i = 0; i < 3; ++i) {
            TSizeDoubleUMap p;
            for (std::size_t j = 0; j < numberCategories[t]; ++j) {
                entropy[i].add(j, counts[j]);
                p[j] += static_cast<double>(counts[j]) / static_cast<double>(Z);
            }

            double ha = entropy[i].calculate();
            double h = 0.0;
            for (TSizeDoubleUMapCItr j = p.begin(); j != p.end(); ++j) {
                h -= j->second * std::log(j->second);
            }
            if (t % 30 == 0) {
                LOG_DEBUG(<< "H_approx = " << ha << ", H_exact = " << h);
            }

            meanError[i].add(std::fabs(ha - h) / h);
            maxError[i].add(std::fabs(ha - h) / h);
            for (std::size_t k = 0; k < 3; ++k) {
                if (std::fabs(ha - h) > eps[k]) {
                    epsDeviations[i][k] += 1.0;
                }
            }
        }
    }

    double maxMaxErrors[] = {0.14, 0.11, 0.08};
    double maxMeanErrors[] = {0.05, 0.04, 0.03};
    for (std::size_t i = 0; i < 3; ++i) {
        LOG_DEBUG(<< "max error  = " << maxError[i][0]);
        LOG_DEBUG(<< "mean error = "
                  << maths::common::CBasicStatistics::mean(meanError[i]));
        LOG_DEBUG(<< "large deviations = "
                  << core::CContainerPrinter::print(epsDeviations[i]));
        BOOST_TEST_REQUIRE(maxError[i][0] < maxMaxErrors[i]);
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError[i]) <
                           maxMeanErrors[i]);
        // Test additive approximation bounds.
        for (std::size_t j = 0; j < 3; ++j) {
            BOOST_TEST_REQUIRE(epsDeviations[i][j] / 1000.0 <
                               2.0 * std::exp(-K[i] * eps[j] * eps[j] / 6.0));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
