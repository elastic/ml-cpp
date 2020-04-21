/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>

#include <maths/CAdaptiveBucketing.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <vector>

BOOST_AUTO_TEST_SUITE(CAdaptiveBucketingTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testSpread) {
    using TDoubleVec = std::vector<double>;
    using TFloatVec = std::vector<maths::CFloatStorage>;

    double period{86400.0};
    {
        TFloatVec points{15.0,    120.0,   4500.0,  9000.0, 25700.0,
                         43100.0, 73000.0, 74000.0, 84300.0};
        double separation{20.0};
        std::string expected{core::CContainerPrinter::print(points)};
        maths::CAdaptiveBucketing::spread(0.0, period, separation, points);
        BOOST_REQUIRE_EQUAL(expected, core::CContainerPrinter::print(points));
        separation = 200.0;
        expected = "[0, 200, 4500, 9000, 25700, 43100, 73000, 74000, 84300]";
        maths::CAdaptiveBucketing::spread(0.0, period, separation, points);
        LOG_DEBUG(<< "spread = " << core::CContainerPrinter::print(points));
        BOOST_REQUIRE_EQUAL(expected, core::CContainerPrinter::print(points));
    }
    {
        TFloatVec points{150.0,   170.0,   4500.0,  4650.0,  4700.0,  4800.0,
                         73000.0, 73150.0, 73500.0, 73600.0, 73800.0, 74000.0};
        double separation{126.0};
        std::string expected = "[97, 223, 4473.5, 4599.5, 4725.5, 4851.5, 73000, 73150, 73487, 73613, 73800, 74000]";
        maths::CAdaptiveBucketing::spread(0.0, period, separation, points);
        LOG_DEBUG(<< "spread = " << core::CContainerPrinter::print(points));
        BOOST_REQUIRE_EQUAL(expected, core::CContainerPrinter::print(points));
    }
    test::CRandomNumbers rng;
    double dcdxMean{0.0};
    for (std::size_t i = 0; i < 100; ++i) {
        TDoubleVec random;
        rng.generateUniformSamples(1000.0, period - 1000.0, 100, random);
        std::sort(random.begin(), random.end());
        TFloatVec origSamples(random.begin(), random.end());
        TFloatVec samples{origSamples};
        maths::CAdaptiveBucketing::spread(0.0, period, 150.0, samples);
        double eps{1e-2};
        for (std::size_t j = 1; j < samples.size(); ++j) {
            BOOST_TEST_REQUIRE(samples[j] - samples[j - 1] >= 150.0 - eps);
        }
        double dcdx{0.0};
        for (std::size_t j = 0; j < samples.size(); ++j) {
            dcdx += maths::CTools::pow2(samples[j] + eps - origSamples[j]) -
                    maths::CTools::pow2(samples[j] - eps - origSamples[j]);
        }
        dcdx /= 2.0 * eps;
        LOG_DEBUG(<< "d(cost)/dx = " << dcdx);
        BOOST_TEST_REQUIRE(std::fabs(dcdx) < 0.1);
        dcdxMean += std::fabs(dcdx);
    }
    dcdxMean /= 100.0;
    LOG_DEBUG(<< "mean d(cost)/dx = " << dcdxMean);
    BOOST_TEST_REQUIRE(dcdxMean < 0.01);
}

BOOST_AUTO_TEST_SUITE_END()
