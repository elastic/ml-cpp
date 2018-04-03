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

#include "CStatisticalTestsTest.h"

#include <core/CoreTypes.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CStatisticalTests.h>

#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>

#include <boost/bind.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;

void CStatisticalTestsTest::testCramerVonMises(void)
{
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CStatisticalTestsTest::testCramerVonMises  |");
    LOG_DEBUG("+---------------------------------------------+");

    // These test that the test statistic p value percentiles
    // are correct if the random variable and the distribution
    // function are perfectly matched.

    const std::size_t n[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 100, 200, 500 };

    test::CRandomNumbers rng;

    double averageMeanError = 0.0;

    for (std::size_t i = 0u; i < boost::size(n); ++i)
    {
        LOG_DEBUG("*** n = " << n[i] << " ***");
        {
            LOG_DEBUG("N(" << 5.0 << "," << ::sqrt(2.0) << ")");
            TDoubleVec samples;
            rng.generateNormalSamples(5.0, 2.0, n[i] * 1000, samples);
            boost::math::normal_distribution<> normal(5.0, ::sqrt(2.0));

            TDoubleVec p;
            for (std::size_t j = 0u; j < samples.size()/n[i]; ++j)
            {
                maths::CStatisticalTests::CCramerVonMises cvm(n[i]-1);
                for (std::size_t k = n[i]*j; k < n[i]*(j+1); ++k)
                {
                    cvm.addF(boost::math::cdf(normal, samples[k]));
                }
                p.push_back(cvm.pValue());
            }
            std::sort(p.begin(), p.end());

            double meanError = 0.0;
            for (std::size_t j = 0; j < 21; ++j)
            {
                double percentile = static_cast<double>(j) / 20.0;
                double pp = static_cast<double>(std::lower_bound(p.begin(),
                                                                 p.end(),
                                                                 percentile)
                                                - p.begin())
                            / static_cast<double>(p.size());
                LOG_DEBUG("percentile = " << percentile
                          << ", p value percentile = " << pp
                          << ", error = " << ::fabs(pp - percentile));
                meanError += ::fabs(pp - percentile);
                CPPUNIT_ASSERT(::fabs(pp - percentile) < 0.055);
            }
            meanError /= 21.0;
            LOG_DEBUG("meanError = " << meanError);
            CPPUNIT_ASSERT(meanError < 0.026);
            averageMeanError += meanError;
        }
        {
            LOG_DEBUG("ln(N(" << 2.0 << "," << 1.0 << "))");
            TDoubleVec samples;
            rng.generateLogNormalSamples(2.0, 1.0, n[i] * 1000, samples);
            boost::math::lognormal_distribution<> lognormal(2.0, 1.0);

            TDoubleVec p;
            for (std::size_t j = 0u; j < samples.size()/n[i]; ++j)
            {
                maths::CStatisticalTests::CCramerVonMises cvm(n[i]-1);
                for (std::size_t k = n[i]*j; k < n[i]*(j+1); ++k)
                {
                    cvm.addF(boost::math::cdf(lognormal, samples[k]));
                }
                p.push_back(cvm.pValue());
            }
            std::sort(p.begin(), p.end());

            double meanError = 0.0;
            for (std::size_t j = 0; j < 21; ++j)
            {
                double percentile = static_cast<double>(j) / 20.0;
                double pp = static_cast<double>(std::lower_bound(p.begin(),
                                                                 p.end(),
                                                                 percentile)
                                                - p.begin())
                            / static_cast<double>(p.size());
                LOG_DEBUG("percentile = " << percentile
                          << ", p value percentile = " << pp
                          << ", error = " << ::fabs(pp - percentile));
                meanError += ::fabs(pp - percentile);
                CPPUNIT_ASSERT(::fabs(pp - percentile) < 0.055);
            }
            meanError /= 21.0;
            LOG_DEBUG("meanError = " << meanError);
            CPPUNIT_ASSERT(meanError < 0.025);
            averageMeanError += meanError;
        }
    }

    averageMeanError /= 2.0 * static_cast<double>(boost::size(n));
    LOG_DEBUG("averageMeanError = " << averageMeanError);
    CPPUNIT_ASSERT(averageMeanError < 0.011);
}

void CStatisticalTestsTest::testPersist(void)
{
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CStatisticalTestsTest::testPersist  |");
    LOG_DEBUG("+--------------------------------------+");

    // Check that serialization is idempotent.

    {
        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateNormalSamples(5.0, 2.0, 25, samples);
        boost::math::normal_distribution<> normal(5.0, ::sqrt(2.0));

        maths::CStatisticalTests::CCramerVonMises origCvm(9);
        TDoubleVec p;
        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            origCvm.addF(boost::math::cdf(normal, samples[i]));
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origCvm.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG("seasonal component XML representation:\n" << origXml);

        // Restore the XML into a new filter
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CStatisticalTests::CCramerVonMises restoredCvm(traverser);
        CPPUNIT_ASSERT_EQUAL(origCvm.checksum(),
                             restoredCvm.checksum());

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredCvm.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test *CStatisticalTestsTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStatisticalTestsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CStatisticalTestsTest>(
                                   "CStatisticalTestsTest::testCramerVonMises",
                                   &CStatisticalTestsTest::testCramerVonMises) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStatisticalTestsTest>(
                                   "CStatisticalTestsTest::testPersist",
                                   &CStatisticalTestsTest::testPersist) );

    return suiteOfTests;
}
