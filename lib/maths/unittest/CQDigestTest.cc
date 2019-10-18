/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CQDigest.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <set>

BOOST_AUTO_TEST_SUITE(CQDigestTest)

using namespace ml;
using namespace maths;
using namespace test;

using TDoubleVec = std::vector<double>;
using TUInt32UInt64Pr = std::pair<uint32_t, uint64_t>;
using TUInt32UInt64PrVec = std::vector<TUInt32UInt64Pr>;

BOOST_AUTO_TEST_CASE(testAdd) {
    // We test the space and error bounds on the quantile calculations
    // for various inputs.

    {
        // All one value.
        CQDigest qDigest(10u);

        for (std::size_t i = 0u; i < 50u; ++i) {
            qDigest.add(5);
        }

        LOG_DEBUG(<< qDigest.print());

        BOOST_TEST(qDigest.checkInvariants());

        BOOST_CHECK_EQUAL(std::string("50 | 10 | { \"[5,5],50,50\" \"[0,7],0,50\" }"),
                          qDigest.print());
    }

    {
        // Less than or equal k unique values.
        CQDigest qDigest(5u);

        std::string expectedDigests[] = {
            std::string("1 | 5 | { \"[0,0],1,1\" \"[0,1],0,1\" }"),
            std::string("2 | 5 | { \"[0,0],1,1\" \"[1,1],1,1\" \"[0,1],0,2\" }"),
            std::string("3 | 5 | { \"[0,0],1,1\" \"[1,1],1,1\" \"[2,2],1,1\" \"[0,3],0,3\" }"),
            std::string("4 | 5 | { \"[0,0],1,1\" \"[1,1],1,1\" \"[2,2],1,1\" \"[3,3],1,1\" \"[0,3],0,4\" }"),
            std::string("5 | 5 | { \"[0,0],1,1\" \"[1,1],1,1\" \"[2,2],1,1\" \"[3,3],1,1\" \"[4,4],1,1\" \"[0,7],0,5\" }"),
        };

        for (std::size_t i = 0u; i < 5u; ++i) {
            qDigest.add(static_cast<uint32_t>(i));

            LOG_DEBUG(<< qDigest.print());

            BOOST_TEST(qDigest.checkInvariants());

            BOOST_CHECK_EQUAL(expectedDigests[i], qDigest.print());
        }
    }

    // Large n uniform random.
    {
        using TUInt64Set = std::multiset<uint64_t>;

        const double expectedMaxErrors[] = {0.007, 0.01,  0.12,  0.011, 0.016,
                                            0.018, 0.023, 0.025, 0.02};

        CRandomNumbers generator;

        CQDigest qDigest(100u);

        TDoubleVec samples;
        generator.generateUniformSamples(0.0, 5000.0, 10000u, samples);

        double totalErrors[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        TUInt64Set orderedSamples;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            uint32_t sample = static_cast<uint32_t>(std::floor(samples[i]));

            qDigest.add(sample);
            orderedSamples.insert(sample);

            BOOST_TEST(qDigest.checkInvariants());

            double n = static_cast<double>(i + 1u);

            if (i > 99) {
                for (unsigned int j = 1; j < 10; ++j) {
                    double q = static_cast<double>(j) / 10.0;

                    uint32_t quantile;
                    qDigest.quantile(q, quantile);

                    std::size_t rank = std::distance(
                        orderedSamples.begin(), orderedSamples.lower_bound(quantile));

                    double error = (static_cast<double>(rank) - q * n) / n;

                    if ((i + 1) % 1000 == 0) {
                        LOG_DEBUG(<< "q = " << q << ", quantile = " << quantile
                                  << ", rank = " << rank << ", n = " << n
                                  << ", error " << error);
                    }

                    BOOST_TEST(std::fabs(error) < 0.06);

                    totalErrors[j - 1u] += error;
                }
            }
        }

        for (size_t i = 0; i < boost::size(totalErrors); ++i) {
            totalErrors[i] /= static_cast<double>(samples.size());
        }

        LOG_DEBUG(<< "total errors = " << core::CContainerPrinter::print(totalErrors));

        for (size_t i = 0; i < boost::size(totalErrors); ++i) {
            BOOST_TEST(totalErrors[i] < expectedMaxErrors[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMerge) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testCdf) {
    // We check the relationship that c.d.f. is the approximate inverse
    // of quantile. We also test the quality of the approximation versus
    // the true c.d.f. of the data.

    const std::size_t k = 100u;
    CQDigest qDigest(k + 1);

    const std::size_t nSamples = 5000u;
    TDoubleVec samples;
    CRandomNumbers generator;
    generator.generateUniformSamples(0.0, 500.0, nSamples, samples);

    std::size_t s = 0u;
    for (/**/; s < std::min(k, samples.size()); ++s) {
        uint32_t sample = static_cast<uint32_t>(std::floor(samples[s]));
        qDigest.add(sample);
    }

    TUInt32UInt64PrVec summary;
    qDigest.summary(summary);
    LOG_DEBUG(<< "summary = " << core::CContainerPrinter::print(summary));

    for (std::size_t i = 0u; i < summary.size(); ++i) {
        double lowerBound;
        double upperBound;
        qDigest.cdf(summary[i].first, 0.0, lowerBound, upperBound);

        LOG_DEBUG(<< "x = " << summary[i].first << ", F(x) >= " << lowerBound
                  << ", F(x) <= " << upperBound);

        double fx = static_cast<double>(summary[i].second) / 100.0;

        BOOST_TEST(fx >= lowerBound);
        BOOST_TEST(fx <= upperBound);
    }

    for (/**/; s < samples.size(); ++s) {
        uint32_t sample = static_cast<uint32_t>(std::floor(samples[s]));
        qDigest.add(sample);
    }

    qDigest.summary(summary);
    LOG_DEBUG(<< "summary = " << core::CContainerPrinter::print(summary));

    for (std::size_t i = 0u; i < summary.size(); ++i) {
        double lowerBound;
        double upperBound;
        qDigest.cdf(summary[i].first, 0.0, lowerBound, upperBound);

        // The expected lower bound.
        double fx = static_cast<double>(summary[i].second) / static_cast<double>(nSamples);

        // Get the true c.d.f. value.
        double ft = std::min(static_cast<double>(summary[i].first) / 500.0, 1.0);

        LOG_DEBUG(<< "x = " << summary[i].first << ", F(x) = " << ft
                  << ", F(x) >= " << lowerBound << ", F(x) <= " << upperBound);

        BOOST_TEST(fx >= lowerBound);
        BOOST_TEST(fx <= upperBound);
        BOOST_TEST(ft >= lowerBound - 0.01);
        BOOST_TEST(ft <= upperBound + 0.01);
    }
}

BOOST_AUTO_TEST_CASE(testSummary) {
    // Check that quantiles of the summary agree with the digest.
    {
        CQDigest qDigest(20u);

        TDoubleVec samples;
        CRandomNumbers generator;
        generator.generateUniformSamples(0.0, 500.0, 100u, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            uint32_t sample = static_cast<uint32_t>(std::floor(samples[i]));
            qDigest.add(sample);
        }

        BOOST_TEST(qDigest.checkInvariants());

        TUInt32UInt64PrVec summary;
        qDigest.summary(summary);
        LOG_DEBUG(<< "summary = " << core::CContainerPrinter::print(summary));

        for (std::size_t i = 0u; i < summary.size(); ++i) {
            double q = static_cast<double>(summary[i].second) / 100.0;

            uint32_t xq;
            qDigest.quantile(q, xq);

            LOG_DEBUG(<< "q = " << q << ", x(q) = " << summary[i].first
                      << ", expected x(q) = " << xq);

            BOOST_CHECK_EQUAL(xq, summary[i].first);
        }
    }

    // Edge case: all the values are in one node.
    {
        CQDigest qDigest(20u);

        qDigest.add(3);

        TUInt32UInt64PrVec summary;
        qDigest.summary(summary);

        BOOST_CHECK_EQUAL(std::string("[(3, 1)]"), core::CContainerPrinter::print(summary));
    }

    // Edge case: non-zero count at the root.
    {
        CQDigest qDigest(2u);

        qDigest.add(0);
        qDigest.add(0);
        qDigest.add(5);
        qDigest.add(0);

        TUInt32UInt64PrVec summary;
        qDigest.summary(summary);

        BOOST_CHECK_EQUAL(std::string("[(0, 3), (7, 4)]"),
                          core::CContainerPrinter::print(summary));
    }
}

BOOST_AUTO_TEST_CASE(testPropagateForwardByTime) {
    using TMeanAccumlator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    {
        // Check a simple case where exact aging is possible.

        CQDigest qDigest(10u, 1.0);

        for (std::size_t i = 0; i < 10; ++i) {
            qDigest.add(0);
            qDigest.add(3);
            qDigest.add(2);
            qDigest.add(15);
            qDigest.add(7);
            qDigest.add(0);
            qDigest.add(1);
            qDigest.add(1);
            qDigest.add(5);
            qDigest.add(8);
        }

        LOG_DEBUG(<< "Before propagation " << qDigest.print());
        qDigest.propagateForwardsByTime(-std::log(0.9));
        LOG_DEBUG(<< "After propagation " << qDigest.print());
        BOOST_TEST(qDigest.checkInvariants());

        BOOST_CHECK_EQUAL(std::string("90 | 10 | { \"[0,0],18,18\" \"[1,1],18,18\" \"[2,2],9,9\""
                                      " \"[3,3],9,9\" \"[5,5],9,9\" \"[7,7],9,9\" \"[8,8],9,9\""
                                      " \"[15,15],9,9\" \"[0,15],0,90\" }"),
                          qDigest.print());
    }

    double intrinsicError;
    {
        // Check that the error introduced into the quantiles by aging
        // by a small amount are small.

        CQDigest qDigest(201, 0.001);

        CRandomNumbers rng;

        double mean = 10000.0;
        double std = 100.0;

        TDoubleVec samples;
        rng.generateNormalSamples(mean, std * std, 200000, samples);
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            qDigest.add(static_cast<uint32_t>(samples[i] + 0.5));
        }

        uint64_t n = qDigest.n();
        LOG_DEBUG(<< "n = " << n);

        TDoubleVec cdfLower;
        TDoubleVec cdfUpper;

        TMeanAccumlator error;
        boost::math::normal_distribution<> normal(mean, std);
        for (double x = mean - 5.0 * std; x <= mean + 5 * std; x += 5.0) {
            double lb, ub;
            BOOST_TEST(qDigest.cdf(static_cast<uint32_t>(x), 0.0, lb, ub));
            cdfLower.push_back(lb);
            double f = boost::math::cdf(normal, x);
            cdfUpper.push_back(ub);
            error.add(std::fabs(f - (lb + ub) / 2.0));
        }
        LOG_DEBUG(<< "error = " << CBasicStatistics::mean(error));
        intrinsicError = CBasicStatistics::mean(error);

        qDigest.propagateForwardsByTime(1.0);
        BOOST_TEST(qDigest.checkInvariants());

        uint64_t nAged = qDigest.n();
        LOG_DEBUG(<< "nAged = " << nAged);

        BOOST_CHECK_CLOSE_ABSOLUTE(0.001, double(n - nAged) / double(n), 5e-4);

        TDoubleVec cdfLowerAged;
        TDoubleVec cdfUpperAged;
        for (double x = mean - 5.0 * std; x <= mean + 5 * std; x += 5.0) {
            double lb, ub;
            BOOST_TEST(qDigest.cdf(static_cast<uint32_t>(x), 0.0, lb, ub));
            cdfLowerAged.push_back(lb);
            cdfUpperAged.push_back(ub);
        }

        TMeanAccumlator diff;
        for (std::size_t i = 0; i < cdfLower.size(); ++i) {
            BOOST_CHECK_CLOSE_ABSOLUTE(cdfLower[i], cdfLowerAged[i],
                                       std::min(5e-5, 2e-3 * cdfLower[i]));
            BOOST_CHECK_CLOSE_ABSOLUTE(cdfUpper[i], cdfUpperAged[i],
                                       std::min(5e-5, 2e-3 * cdfUpper[i]));
            diff.add(std::fabs(cdfLower[i] - cdfLowerAged[i]));
            diff.add(std::fabs(cdfUpper[i] - cdfUpperAged[i]));
        }
        LOG_DEBUG(<< "diff = " << CBasicStatistics::mean(diff));
        BOOST_TEST(CBasicStatistics::mean(diff) < 1e-5);
    }

    {
        // Check no systematic aging accumulate over multiple rounds
        // of aging.

        CQDigest qDigest(201, 0.001);

        CRandomNumbers rng;

        double mean = 10000.0;
        double std = 100.0;

        TDoubleVec samples;
        for (std::size_t i = 0u; i < 500; ++i) {
            rng.generateNormalSamples(mean, std * std, 2000, samples);
            for (std::size_t j = 0u; j < samples.size(); ++j) {
                qDigest.add(static_cast<uint32_t>(samples[j] + 0.5));
            }
            if (i % 10 == 0) {
                LOG_DEBUG(<< "iteration = " << i);
            }
            qDigest.propagateForwardsByTime(1.0);
        }
        LOG_DEBUG(<< "n = " << qDigest.n());

        TMeanAccumlator error;
        boost::math::normal_distribution<> normal(mean, std);
        for (double x = mean - 5.0 * std; x <= mean + 5 * std; x += 5.0) {
            double lb, ub;
            BOOST_TEST(qDigest.cdf(static_cast<uint32_t>(x), 0.0, lb, ub));
            double f = boost::math::cdf(normal, x);
            BOOST_TEST(lb <= f);
            BOOST_TEST(ub >= f);
            BOOST_CHECK_CLOSE_ABSOLUTE(f, (lb + ub) / 2.0, 0.015);
            error.add(std::fabs(f - (lb + ub) / 2.0));
        }
        LOG_DEBUG(<< "error = " << CBasicStatistics::mean(error));
        BOOST_TEST(CBasicStatistics::mean(error) < 0.006);
        BOOST_TEST(CBasicStatistics::mean(error) < 1.5 * intrinsicError);
    }
}

BOOST_AUTO_TEST_CASE(testScale) {
    {
        CQDigest qDigest(10u, 1.0);

        for (std::size_t i = 0; i < 10; ++i) {
            qDigest.add(0);
            qDigest.add(3);
            qDigest.add(2);
            qDigest.add(15);
            qDigest.add(7);
            qDigest.add(0);
            qDigest.add(1);
            qDigest.add(1);
            qDigest.add(5);
            qDigest.add(8);
        }

        LOG_DEBUG(<< "Before scaling " << qDigest.print());
        qDigest.scale(3.0);
        LOG_DEBUG(<< "After scaling " << qDigest.print());
        BOOST_TEST(qDigest.checkInvariants());

        BOOST_CHECK_EQUAL(std::string("100 | 10 | { \"[0,0],20,20\" \"[3,3],20,20\" \"[6,6],10,10\""
                                      " \"[9,9],10,10\" \"[15,15],10,10\" \"[21,21],10,10\" \"[24,24],10,10\""
                                      " \"[45,45],10,10\" \"[0,63],0,100\" }"),
                          qDigest.print());

        // Test that adding more values after scaling works
        for (std::size_t i = 0; i < 10; ++i) {
            qDigest.add(0);
            qDigest.add(7);
            qDigest.add(5);
            qDigest.add(25);
            qDigest.add(17);
            qDigest.add(0);
            qDigest.add(2);
            qDigest.add(1);
            qDigest.add(5);
            qDigest.add(38);
        }

        LOG_DEBUG(<< "After adding more values " << qDigest.print());
        BOOST_TEST(qDigest.checkInvariants());

        // Add a new value that will expand the root
        qDigest.add(65);

        LOG_DEBUG(<< "After expanding root " << qDigest.print());
        BOOST_TEST(qDigest.checkInvariants());
    }

    {
        const double scales[] = {1.5, 1.7, 2.2, 3.1, 4.0, 5.0};
        const double maxMaxType1[] = {0.17, 0.19, 0.32, 0.31, 0.38, 0.33};
        const double maxTotalType1[] = {2.0, 2.5, 9.6, 6.8, 8.7, 12.8};
        const double maxMaxType2[] = {0.11, 0.1, 0.15, 0.18, 0.19, 0.22};
        const double maxTotalType2[] = {1.9, 1.1, 1.1, 3.3, 2.9, 10.1};

        TDoubleVec samples;
        CRandomNumbers generator;
        generator.generateNormalSamples(50.0, 5.0, 500u, samples);

        for (std::size_t i = 0u; i < boost::size(scales); ++i) {
            LOG_DEBUG(<< "*** Testing scale = " << scales[i] << " ***");

            CQDigest qDigest(20u);
            CQDigest qDigestScaled(20u);

            for (std::size_t j = 0; j < samples.size(); ++j) {
                qDigest.add(static_cast<uint32_t>(samples[j]));
                qDigestScaled.add(static_cast<uint32_t>(scales[i] * samples[j]));
            }

            qDigest.scale(scales[i]);
            BOOST_CHECK_EQUAL(qDigestScaled.n(), qDigest.n());

            double maxType1 = 0.0;
            double totalType1 = 0.0;
            double maxType2 = 0.0;
            double totalType2 = 0.0;

            uint32_t end =
                static_cast<uint32_t>(
                    scales[i] * *std::max_element(samples.begin(), samples.end())) +
                1;
            for (uint32_t j = 0; j < end; ++j) {
                double expectedLowerBound;
                double expectedUpperBound;
                qDigestScaled.cdf(j, 0.0, expectedLowerBound, expectedUpperBound);

                double lowerBound;
                double upperBound;
                qDigest.cdf(j, 0.0, lowerBound, upperBound);
                double type1 = std::fabs(expectedLowerBound - lowerBound) +
                               std::fabs(expectedUpperBound - upperBound);
                double type2 = std::max(lowerBound - expectedLowerBound, 0.0) +
                               std::max(expectedUpperBound - upperBound, 0.0);
                maxType1 = std::max(maxType1, type1);
                totalType1 += type1;
                maxType2 = std::max(maxType2, type2);
                totalType2 += type2;
            }
            LOG_DEBUG(<< "maxType1 = " << maxType1);
            LOG_DEBUG(<< "totalType1 = " << totalType1);
            LOG_DEBUG(<< "maxType2 = " << maxType2);
            LOG_DEBUG(<< "totalType2 = " << totalType2);
            BOOST_TEST(maxType1 < maxMaxType1[i]);
            BOOST_TEST(totalType1 < maxTotalType1[i]);
            BOOST_TEST(maxType2 < maxMaxType2[i]);
            BOOST_TEST(totalType2 < maxTotalType2[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Large n uniform random.
    CRandomNumbers generator;

    CQDigest origQDigest(100u);

    TDoubleVec samples;
    generator.generateUniformSamples(0.0, 5000.0, 1000u, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        uint32_t sample = static_cast<uint32_t>(std::floor(samples[i]));

        origQDigest.add(sample);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origQDigest.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "q-digest XML representation:\n" << origXml);

    // Restore the XML into a new filter
    CQDigest restoredQDigest(100u);
    {
        core::CRapidXmlParser parser;
        BOOST_TEST(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST(traverser.traverseSubLevel(std::bind(
            &CQDigest::acceptRestoreTraverser, &restoredQDigest, std::placeholders::_1)));
    }

    BOOST_TEST(restoredQDigest.checkInvariants());
    BOOST_CHECK_EQUAL(origQDigest.print(), restoredQDigest.print());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredQDigest.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_CHECK_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_SUITE_END()
