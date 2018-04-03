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

#include "CNaturalBreaksClassifierTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CNaturalBreaksClassifier.h>
#include <maths/CRestoreParams.h>

#include <test/CRandomNumbers.h>

#include <algorithm>
#include <vector>

using namespace ml;
using namespace maths;

namespace
{

using TDoubleVec = std::vector<double>;
using TTuple = CNaturalBreaksClassifier::TTuple;
using TTupleVec = std::vector<TTuple>;

//! Computes the deviation of a category.mac1Password

bool computeDeviation(const TTuple &category,
                      std::size_t p,
                      double &result)
{
    double count = CBasicStatistics::count(category);
    double variance = CBasicStatistics::variance(category);
    result = ::sqrt((count - 1.0) * variance);
    return count >= static_cast<double>(p);
}

//! Branch and bound exhaustive search for the optimum split.
bool naturalBreaksBranchAndBound(const TTupleVec &categories,
                                 std::size_t n,
                                 std::size_t p,
                                 TTupleVec &result)
{
    using TSizeVec = std::vector<std::size_t>;

    // Find the minimum variance partition.
    //
    // This uses branch and bound in the inner loop over all
    // partitions. Note, however, this still has worst case
    // exponential complexity so shouldn't be called with
    // parameters which could cause it to have very large
    // runtime.

    static const double INF = std::numeric_limits<double>::max();

    std::size_t N = categories.size();
    std::size_t m = n - 1u;
    LOG_TRACE("m = " << m);

    TSizeVec split;
    split.reserve(m + 1);
    for (std::size_t i = 1u; i < n; ++i)
    {
        split.push_back(i);
    }
    split.push_back(N);
    LOG_TRACE("split = " << core::CContainerPrinter::print(split));

    TSizeVec end;
    end.reserve(m + 1);
    for (std::size_t i = N - m; i < N; ++i)
    {
        end.push_back(i);
    }
    end.push_back(N);
    LOG_TRACE("end = " << core::CContainerPrinter::print(end));

    TSizeVec bestSplit;
    double deviationMin = INF;
    for (;;)
    {
        LOG_TRACE("split = " << core::CContainerPrinter::print(split));

        double deviation = 0.0;

        for (std::size_t i = 0u, j = 0u; i < split.size(); ++i)
        {
            TTuple category;
            for (/**/; j < split[i]; ++j)
            {
                category += categories[j];
            }

            double categoryDeviation;
            if (!computeDeviation(category, p, categoryDeviation)
                || (deviation >= deviationMin && i < m - 1))
            {
                // We can prune all possible solutions which have
                // sub-split (split[0], ... split[i]) since their
                // deviation is necessarily larger than the minimum
                // we've found so far. We do this by incrementing
                // split[j], for j > i, such that we'll increment
                // split[i] below.
                for (++i; i < m; ++i)
                {
                    split[i] = N - (m - i);
                }
                deviation = INF;
                LOG_TRACE("Pruning solutions variation = " << deviation
                          << ", deviationMin = " << deviationMin
                          << ", split = " << core::CContainerPrinter::print(split));
            }
            else
            {
                deviation += categoryDeviation;
            }
        }

        if (deviation < deviationMin)
        {
            bestSplit = split;
            deviationMin = deviation;
            LOG_TRACE("splitMin = " << core::CContainerPrinter::print(result)
                      << ", deviationMin = " << deviationMin);
        }

        if (split == end)
        {
            break;
        }

        for (std::size_t i = 1; i <= m; ++i)
        {
            if (split[m - i] < N - i)
            {
                ++split[m - i];
                for (--i; i > 0; --i)
                {
                    split[m - i] = split[m - i - 1] + 1;
                }
                break;
            }
        }
    }

    if (deviationMin == INF)
    {
        return false;
    }

    result.reserve(n);
    for (std::size_t i = 0u, j = 0u; i < bestSplit.size(); ++i)
    {
        TTuple category;
        for (/**/; j < bestSplit[i]; ++j)
        {
            category += categories[j];
        }
        result.push_back(category);
    }

    return true;
}

}

void CNaturalBreaksClassifierTest::testCategories(void)
{
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CNaturalBreaksClassifierTest::testCategories  |");
    LOG_DEBUG("+------------------------------------------------+");

    // Check that we correctly find the optimum solution.
    {
        test::CRandomNumbers rng;

        CNaturalBreaksClassifier classifier(10);

        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 15.0, 5000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            classifier.add(samples[i]);
            if (i > 0 && i % 50 == 0)
            {
                for (std::size_t j = 3u; j < 7; ++j)
                {
                    LOG_DEBUG("# samples = " << i << ", # splits = " << j);

                    TTupleVec split;
                    classifier.categories(j, 0, split);

                    TTupleVec all;
                    classifier.categories(10, 0, all);
                    TTupleVec expectedSplit;
                    naturalBreaksBranchAndBound(all, j, 0, expectedSplit);

                    LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedSplit));
                    LOG_DEBUG("actual =   " << core::CContainerPrinter::print(split));

                    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSplit),
                                         core::CContainerPrinter::print(split));
                }
            }
        }
    }

    // Check we correctly find the optimum solution subject to
    // a constraint on the minimum count.
    {
        test::CRandomNumbers rng;

        CNaturalBreaksClassifier classifier(10);

        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 15.0, 500, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            classifier.add(samples[i]);
            if (i > 0 && i % 10 == 0)
            {
                for (std::size_t j = 3u; j < 7; ++j)
                {
                    std::size_t k = 1u;
                    do
                    {
                        k *= 2;

                        LOG_DEBUG("# samples = " << i
                                  << ", # splits = " << j
                                  << ", minimum cluster size = " << k);

                        TTupleVec split;
                        bool haveSplit = classifier.categories(j, k, split);

                        TTupleVec all;
                        classifier.categories(10, 0, all);
                        TTupleVec expectedSplit;
                        bool expectSplit = naturalBreaksBranchAndBound(all, j, k, expectedSplit);

                        CPPUNIT_ASSERT_EQUAL(expectSplit, haveSplit);

                        if (expectSplit && haveSplit)
                        {
                            LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedSplit));
                            LOG_DEBUG("actual =   " << core::CContainerPrinter::print(split));

                            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSplit),
                                                 core::CContainerPrinter::print(split));
                        }
                    }
                    while (k < i / j);
                }
            }
        }
    }

    // All tests test a variety of random permutations of the
    // input data. Good data stream clustering algorithms should
    // be able to handle different data in different orders.

    test::CRandomNumbers rng;

    // Test Gaussian clusters.
    // There are three test cases:
    //   1) A single cluster
    //   2) Fewer clusters than tuples we maintain.
    //   3) More clusters than tuples we maintain.
    {
        CNaturalBreaksClassifier classifier(4);

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 3.0, 5000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            classifier.add(samples[i]);
        }

        TTupleVec twoSplit;
        classifier.categories(2u, 0, twoSplit);

        // We expect two groups of roughly equal size which
        // overlap significantly.
        double c1 = CBasicStatistics::count(twoSplit[0]);
        double c2 = CBasicStatistics::count(twoSplit[1]);
        LOG_DEBUG("count ratio = " << c1/c2);
        CPPUNIT_ASSERT(::fabs(c1/c2 - 1.0) < 0.8);
        double separation = ::fabs(CBasicStatistics::mean(twoSplit[0])
                                   - CBasicStatistics::mean(twoSplit[1]))
                            / (::sqrt(CBasicStatistics::variance(twoSplit[0]))
                               + ::sqrt(CBasicStatistics::variance(twoSplit[1])));
        LOG_DEBUG("separation = " << separation);
        CPPUNIT_ASSERT(::fabs(separation - 1.0) < 0.4);
    }
    {
        const double mean1 = 10.0;
        const double var1 = 1.0;
        const std::size_t n1 = 1500;

        const double mean2 = 13.5;
        const double var2 = 2.0;
        const std::size_t n2 = 4500;

        TDoubleVec samples1;
        rng.generateNormalSamples(mean1, var1, n1, samples1);
        TDoubleVec samples2;
        rng.generateNormalSamples(mean2, var2, n2, samples2);

        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());

        {
            CNaturalBreaksClassifier classifier(6);

            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                classifier.add(samples[i]);
            }
            {
                TTupleVec twoSplit;
                classifier.categories(2u, 0, twoSplit);

                LOG_DEBUG("split 1 = " << CBasicStatistics::print(twoSplit[0])
                          << ", split 2 = " << CBasicStatistics::print(twoSplit[1])
                          << ", (mean1,var1) = (" << mean1 << "," << var1 << ")"
                          << ", (mean2,var2) = (" << mean2 << "," << var2 << ")");

                CPPUNIT_ASSERT(::fabs(CBasicStatistics::mean(twoSplit[0]) - mean1) < 0.5);
                CPPUNIT_ASSERT(::fabs(CBasicStatistics::variance(twoSplit[0]) - var1) < 0.6);
                CPPUNIT_ASSERT(::fabs(CBasicStatistics::count(twoSplit[0])
                                      - static_cast<double>(n1))
                                   / static_cast<double>(n1) < 0.33);
                CPPUNIT_ASSERT(::fabs(CBasicStatistics::mean(twoSplit[1]) - mean2) < 0.4);
                CPPUNIT_ASSERT(::fabs(CBasicStatistics::variance(twoSplit[1]) - var2) < 0.63);
                CPPUNIT_ASSERT(::fabs(CBasicStatistics::count(twoSplit[1])
                                      - static_cast<double>(n2))
                                   / static_cast<double>(n2) < 0.11);
            }
        }

        double totalMeanError1 = 0.0;
        double totalVarError1 = 0.0;
        double totalCountError1 = 0.0;

        double totalMeanError2 = 0.0;
        double totalVarError2 = 0.0;
        double totalCountError2 = 0.0;

        for (int i = 0; i < 500; ++i)
        {
            rng.random_shuffle(samples.begin(), samples.end());

            CNaturalBreaksClassifier classifier(12);

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                classifier.add(samples[j]);
            }

            TTupleVec twoSplit;
            classifier.categories(2u, 0, twoSplit);

            LOG_DEBUG("split 1 = " << CBasicStatistics::print(twoSplit[0])
                      << ", split 2 = " << CBasicStatistics::print(twoSplit[1]));

            CPPUNIT_ASSERT(::fabs(CBasicStatistics::mean(twoSplit[0]) - mean1) < 0.7);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::variance(twoSplit[0]) - var1) < 0.4);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::count(twoSplit[0])
                                  - static_cast<double>(n1)) / static_cast<double>(n1) < 0.7);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::mean(twoSplit[1]) - mean2) < 0.6);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::variance(twoSplit[1]) - var2) < 1.0);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::count(twoSplit[1])
                                  - static_cast<double>(n2)) / static_cast<double>(n2) < 0.3);

            totalMeanError1 += ::fabs(CBasicStatistics::mean(twoSplit[0]) - mean1);
            totalVarError1 += ::fabs(CBasicStatistics::variance(twoSplit[0]) - var1);
            totalCountError1 += ::fabs(CBasicStatistics::count(twoSplit[0])
                                       - static_cast<double>(n1)) / static_cast<double>(n1);
            totalMeanError2 += ::fabs(CBasicStatistics::mean(twoSplit[1]) - mean2);
            totalVarError2 += ::fabs(CBasicStatistics::variance(twoSplit[1]) - var2);
            totalCountError2 += ::fabs(CBasicStatistics::count(twoSplit[1])
                                       - static_cast<double>(n2)) / static_cast<double>(n2);
        }

        totalMeanError1 /= 500.0;
        totalVarError1 /= 500.0;
        totalCountError1 /= 500.0;
        totalMeanError2 /= 500.0;
        totalVarError2 /= 500.0;
        totalCountError2 /= 500.0;

        LOG_DEBUG("mean mean error 1 = " << totalMeanError1
                  << ", mean variance error 1 = " << totalVarError1
                  << ", mean count error 1 = " <<  totalCountError1);
        CPPUNIT_ASSERT(totalMeanError1 < 0.21);
        CPPUNIT_ASSERT(totalVarError1 < 0.2);
        CPPUNIT_ASSERT(totalCountError1 < 0.3);

        LOG_DEBUG("mean mean error 2 = " << totalMeanError2
                  << ", mean variance error 2 = " << totalVarError2
                  << ", mean count error 2 = " <<  totalCountError2);
        CPPUNIT_ASSERT(totalMeanError2 < 0.3);
        CPPUNIT_ASSERT(totalVarError2 < 0.56);
        CPPUNIT_ASSERT(totalCountError2 < 0.1);
    }

    // Test Gaussian clusters plus noise.
    // We have two well separated clusters containing 90% of all data
    // points plus uniform high background noise.
    {
        const double mean1 = 5.0;
        const double var1 = 1.0;
        const std::size_t n1 = 900;

        const double mean2 = 12.0;
        const double var2 = 2.0;
        const std::size_t n2 = 1800;

        const double a = 10.0;
        const double b = 30.0;
        const std::size_t n3 = 300;

        TDoubleVec samples1;
        rng.generateNormalSamples(mean1, var1, n1, samples1);
        TDoubleVec samples2;
        rng.generateNormalSamples(mean2, var2, n2, samples2);
        TDoubleVec samples3;
        rng.generateUniformSamples(a, b, n3, samples3);

        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        samples.insert(samples.end(), samples3.begin(), samples3.end());
        LOG_DEBUG("# samples = " << samples.size());

        double totalMeanError1 = 0.0;
        double totalVarError1 = 0.0;
        double totalCountError1 = 0.0;

        double totalMeanError2 = 0.0;
        double totalVarError2 = 0.0;
        double totalCountError2 = 0.0;

        for (int i = 0; i < 500; ++i)
        {
            rng.random_shuffle(samples.begin(), samples.end());

            CNaturalBreaksClassifier classifier(12);

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                classifier.add(samples[j]);
            }

            TTupleVec twoSplit;
            classifier.categories(3u, 0, twoSplit);

            LOG_DEBUG("split 1 = " << CBasicStatistics::print(twoSplit[0])
                      << ", split 2 = " << CBasicStatistics::print(twoSplit[1])
                      << ", split 3 = " << CBasicStatistics::print(twoSplit[2]));

            CPPUNIT_ASSERT(::fabs(CBasicStatistics::mean(twoSplit[0]) - mean1) < 0.15);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::variance(twoSplit[0]) - var1) < 0.4);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::count(twoSplit[0])
                                  - static_cast<double>(n1)) / static_cast<double>(n1) < 0.05);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::mean(twoSplit[1]) - mean2) < 0.5);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::variance(twoSplit[1]) - var2) < 2.5);
            CPPUNIT_ASSERT(::fabs(CBasicStatistics::count(twoSplit[1])
                                  - static_cast<double>(n2)) / static_cast<double>(n2) < 0.15);

            totalMeanError1 += ::fabs(CBasicStatistics::mean(twoSplit[0]) - mean1);
            totalVarError1 += ::fabs(CBasicStatistics::variance(twoSplit[0]) - var1);
            totalCountError1 += ::fabs(CBasicStatistics::count(twoSplit[0])
                                       - static_cast<double>(n1)) / static_cast<double>(n1);
            totalMeanError2 += ::fabs(CBasicStatistics::mean(twoSplit[1]) - mean2);
            totalVarError2 += ::fabs(CBasicStatistics::variance(twoSplit[1]) - var2);
            totalCountError2 += ::fabs(CBasicStatistics::count(twoSplit[1])
                                       - static_cast<double>(n2)) / static_cast<double>(n2);
        }

        totalMeanError1 /= 500.0;
        totalVarError1 /= 500.0;
        totalCountError1 /= 500.0;
        totalMeanError2 /= 500.0;
        totalVarError2 /= 500.0;
        totalCountError2 /= 500.0;

        LOG_DEBUG("mean mean error 1 = " << totalMeanError1
                  << ", mean variance error 1 = " << totalVarError1
                  << ", mean count error 1 = " <<  totalCountError1);
        CPPUNIT_ASSERT(totalMeanError1 < 0.05);
        CPPUNIT_ASSERT(totalVarError1 < 0.1);
        CPPUNIT_ASSERT(totalCountError1 < 0.01);

        LOG_DEBUG("mean mean error 2 = " << totalMeanError2
                  << ", mean variance error 2 = " << totalVarError2
                  << ", mean count error 2 = " <<  totalCountError2);
        CPPUNIT_ASSERT(totalMeanError2 < 0.15);
        CPPUNIT_ASSERT(totalVarError2 < 1.0);
        CPPUNIT_ASSERT(totalCountError2 < 0.1);
    }
}

void CNaturalBreaksClassifierTest::testPropagateForwardsByTime(void)
{
    LOG_DEBUG("+-------------------------------------------------------------+");
    LOG_DEBUG("|  CNaturalBreaksClassifierTest::testPropagateForwardsByTime  |");
    LOG_DEBUG("+-------------------------------------------------------------+");

    // Check pruning of dead categories.

    test::CRandomNumbers rng;

    TDoubleVec samples1;
    rng.generateNormalSamples(10.0, 3.0, 100, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(20.0, 1.0, 100, samples2);

    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    samples.push_back(100.0);

    CNaturalBreaksClassifier classifier(4, 0.1);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        classifier.add(samples[i]);
    }

    TTupleVec categories;
    classifier.categories(4u, 0, categories);
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), categories.size());

    LOG_DEBUG("categories = " << core::CContainerPrinter::print(categories));

    classifier.propagateForwardsByTime(10.0);
    classifier.categories(4u, 0, categories);

    LOG_DEBUG("categories = " << core::CContainerPrinter::print(categories));

    // We expect the category with count of 1 to have been pruned out.
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), categories.size());
    for (std::size_t i = 0u; i < categories.size(); ++i)
    {
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(categories[i]) != 100.0);
    }
}

void CNaturalBreaksClassifierTest::testSample(void)
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CNaturalBreaksClassifierTest::testSample  |");
    LOG_DEBUG("+--------------------------------------------+");

    // We test that for a small number of samples we get back exactly
    // the points we have added and for a large number of samples we
    // sample the modes of the mixture correctly.

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    static const double NEG_INF = boost::numeric::bounds<double>::lowest();
    static const double POS_INF = boost::numeric::bounds<double>::highest();

    test::CRandomNumbers rng;

    TMeanVarAccumulator expectedMeanVar1;
    TDoubleVec samples1;
    rng.generateNormalSamples(10.0, 3.0, 500, samples1);
    expectedMeanVar1.add(samples1);

    TMeanVarAccumulator expectedMeanVar2;
    TDoubleVec samples2;
    rng.generateNormalSamples(20.0, 1.0, 500, samples2);
    expectedMeanVar2.add(samples2);

    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());

    CNaturalBreaksClassifier classifier(8, 0.1);

    TDoubleVec expectedSampled;
    for (std::size_t i = 0u; i < 5; ++i)
    {
        classifier.add(samples[i]);
        expectedSampled.push_back(samples[i]);
    }

    std::sort(expectedSampled.begin(), expectedSampled.end());

    TDoubleVec sampled;
    classifier.sample(10u, NEG_INF, POS_INF, sampled);
    std::sort(sampled.begin(), sampled.end());

    LOG_DEBUG("expected = " << core::CContainerPrinter::print(expectedSampled));
    LOG_DEBUG("sampled  = " << core::CContainerPrinter::print(sampled));

    double error = 0.0;
    for (std::size_t i = 0u; i < expectedSampled.size(); ++i)
    {
        error += ::fabs(expectedSampled[i] - sampled[i]);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, error, 2e-6);

    for (std::size_t i = 5u; i < samples.size(); ++i)
    {
        classifier.add(samples[i]);
    }

    classifier.sample(50u, NEG_INF, POS_INF, sampled);
    std::sort(sampled.begin(), sampled.end());
    LOG_DEBUG("sampled = " << core::CContainerPrinter::print(sampled));

    TMeanVarAccumulator meanVar1;
    TMeanVarAccumulator meanVar2;
    for (std::size_t i = 0u; i < sampled.size(); ++i)
    {
        if (sampled[i] < 15.0)
        {
            meanVar1.add(sampled[i]);
        }
        else
        {
            meanVar2.add(sampled[i]);
        }
    }

    LOG_DEBUG("expected mean, variance 1 = " << expectedMeanVar1);
    LOG_DEBUG("mean, variance 1          = " << meanVar1);
    LOG_DEBUG("expected mean, variance 2 = " << expectedMeanVar2);
    LOG_DEBUG("mean, variance 2          = " << meanVar2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(CBasicStatistics::mean(expectedMeanVar1),
                                 CBasicStatistics::mean(meanVar1),
                                 0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(CBasicStatistics::variance(expectedMeanVar1),
                                 CBasicStatistics::variance(meanVar1),
                                 0.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(CBasicStatistics::mean(expectedMeanVar2),
                                 CBasicStatistics::mean(meanVar2),
                                 0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(CBasicStatistics::variance(expectedMeanVar2),
                                 CBasicStatistics::variance(meanVar2),
                                 0.1);
}

void CNaturalBreaksClassifierTest::testPersist(void)
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CNaturalBreaksClassifierTest::testSample  |");
    LOG_DEBUG("+--------------------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec samples1;
    rng.generateNormalSamples(10.0, 3.0, 500, samples1);

    TDoubleVec samples2;
    rng.generateNormalSamples(20.0, 1.0, 500, samples2);

    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());

    CNaturalBreaksClassifier origClassifier(8, 0.1);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        origClassifier.add(samples[i]);
    }

    uint64_t checksum = origClassifier.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origClassifier.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("Classifier XML representation:\n" << origXml);

    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    // Restore the XML into a new classifier.
    CNaturalBreaksClassifier restoredClassifier(8);
    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData, 0.2,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(&CNaturalBreaksClassifier::acceptRestoreTraverser,
                                                          &restoredClassifier, boost::cref(params), _1)));

    LOG_DEBUG("orig checksum = " << checksum
              << " restored checksum = " << restoredClassifier.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum, restoredClassifier.checksum());

    // The XML representation of the new filter should be the same
    // as the original.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredClassifier.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test *CNaturalBreaksClassifierTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CNaturalBreaksClassifierTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CNaturalBreaksClassifierTest>(
                                   "CNaturalBreaksClassifierTest::testCategories",
                                   &CNaturalBreaksClassifierTest::testCategories) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CNaturalBreaksClassifierTest>(
                                   "CNaturalBreaksClassifierTest::testPropagateForwardsByTime",
                                   &CNaturalBreaksClassifierTest::testPropagateForwardsByTime) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CNaturalBreaksClassifierTest>(
                                   "CNaturalBreaksClassifierTest::testSample",
                                   &CNaturalBreaksClassifierTest::testSample) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CNaturalBreaksClassifierTest>(
                                   "CNaturalBreaksClassifierTest::testPersist",
                                   &CNaturalBreaksClassifierTest::testPersist) );

    return suiteOfTests;
}
