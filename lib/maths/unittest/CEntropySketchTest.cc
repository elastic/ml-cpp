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

#include "CEntropySketchTest.h"

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CEntropySketch.h>

#include <test/CRandomNumbers.h>

#include <boost/unordered_map.hpp>

#include <math.h>
#include <numeric>
#include <vector>

using namespace ml;

void CEntropySketchTest::testAll(void) {
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testPersist  |");
    LOG_DEBUG("+---------------------------------------+");

    typedef std::vector<std::size_t> TSizeVec;
    typedef boost::unordered_map<std::size_t, double> TSizeDoubleUMap;
    typedef TSizeDoubleUMap::const_iterator TSizeDoubleUMapCItr;

    test::CRandomNumbers rng;

    TSizeVec numberCategories;
    rng.generateUniformSamples(500, 1001, 1000, numberCategories);

    maths::CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>> maxError[3];
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanError[3];

    double K[] = {20.0, 40.0, 60.0};
    double eps[] = {0.2, 0.4, 0.6};
    double epsDeviations[][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    TSizeVec counts;
    for (std::size_t t = 0u; t < numberCategories.size(); ++t) {
        rng.generateUniformSamples(1, 10, numberCategories[t], counts);
        std::size_t Z = std::accumulate(counts.begin(), counts.end(), 0);

        maths::CEntropySketch entropy[] = {maths::CEntropySketch(static_cast<std::size_t>(K[0])),
                                           maths::CEntropySketch(static_cast<std::size_t>(K[1])),
                                           maths::CEntropySketch(static_cast<std::size_t>(K[2]))};

        for (std::size_t i = 0u; i < 3; ++i) {
            TSizeDoubleUMap p;
            for (std::size_t j = 0u; j < numberCategories[t]; ++j) {
                entropy[i].add(j, counts[j]);
                p[j] += static_cast<double>(counts[j]) / static_cast<double>(Z);
            }

            double ha = entropy[i].calculate();
            double h = 0.0;
            for (TSizeDoubleUMapCItr j = p.begin(); j != p.end(); ++j) {
                h -= j->second * ::log(j->second);
            }
            if (t % 30 == 0) {
                LOG_DEBUG("H_approx = " << ha << ", H_exact = " << h);
            }

            meanError[i].add(::fabs(ha - h) / h);
            maxError[i].add(::fabs(ha - h) / h);
            for (std::size_t k = 0u; k < 3; ++k) {
                if (::fabs(ha - h) > eps[k]) {
                    epsDeviations[i][k] += 1.0;
                }
            }
        }
    }

    double maxMaxErrors[] = {0.14, 0.11, 0.08};
    double maxMeanErrors[] = {0.05, 0.04, 0.03};
    for (std::size_t i = 0u; i < 3; ++i) {
        LOG_DEBUG("max error  = " << maxError[i][0]);
        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError[i]));
        LOG_DEBUG("large deviations = " << core::CContainerPrinter::print(epsDeviations[i]));
        CPPUNIT_ASSERT(maxError[i][0] < maxMaxErrors[i]);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError[i]) < maxMeanErrors[i]);
        // Test additive approximation bounds.
        for (std::size_t j = 0u; j < 3; ++j) {
            CPPUNIT_ASSERT(epsDeviations[i][j] / 1000.0 <
                           2.0 * ::exp(-K[i] * eps[j] * eps[j] / 6.0));
        }
    }
}

CppUnit::Test *CEntropySketchTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CEntropySketchTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CEntropySketchTest>(
        "CEntropySketchTest::testAll", &CEntropySketchTest::testAll));

    return suiteOfTests;
}
