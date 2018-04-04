/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CModelTest.h"

#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CModel.h>
#include <maths/Constants.h>

using namespace ml;

void CModelTest::testAll(void)
{
    LOG_DEBUG("+-----------------------+");
    LOG_DEBUG("|  CModelTest::testAll  |");
    LOG_DEBUG("+-----------------------+");

    // Test that the various parameter classes work as expected.

    {
        core_t::TTime bucketLength{600};
        double learnRate{0.5};
        double decayRate{0.001};
        double minimumSeasonalVarianceScale{0.3};
        maths::CModelParams params(bucketLength, learnRate, decayRate,
                                   minimumSeasonalVarianceScale,
                                   6 * core::constants::HOUR, core::constants::DAY);
        CPPUNIT_ASSERT_EQUAL(bucketLength, params.bucketLength());
        CPPUNIT_ASSERT_EQUAL(learnRate, params.learnRate());
        CPPUNIT_ASSERT_EQUAL(decayRate, params.decayRate());
        CPPUNIT_ASSERT_EQUAL(minimumSeasonalVarianceScale, params.minimumSeasonalVarianceScale());
        CPPUNIT_ASSERT_EQUAL(0.0, params.probabilityBucketEmpty());
        params.probabilityBucketEmpty(0.2);
        CPPUNIT_ASSERT_EQUAL(0.2, params.probabilityBucketEmpty());
        CPPUNIT_ASSERT_EQUAL(6 * core::constants::HOUR, params.minimumTimeToDetectChange());
        CPPUNIT_ASSERT_EQUAL(core::constants::DAY, params.maximumTimeToTestForChange());
    }
    {
        maths::CModelAddSamplesParams::TDouble2Vec weight1(2, 0.4);
        maths::CModelAddSamplesParams::TDouble2Vec weight2(2, 0.7);
        maths::CModelAddSamplesParams::TDouble2Vec4Vec weights1(1, weight1);
        maths::CModelAddSamplesParams::TDouble2Vec4Vec weights2(1, weight2);
        maths::CModelAddSamplesParams::TDouble2Vec4VecVec trendWeights(1, weights1);
        maths::CModelAddSamplesParams::TDouble2Vec4VecVec priorWeights(1, weights2);
        maths::CModelAddSamplesParams params;
        params.integer(true)
              .propagationInterval(1.5)
              .weightStyles(maths::CConstantWeights::SEASONAL_VARIANCE)
              .trendWeights(trendWeights)
              .priorWeights(priorWeights);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_IntegerData, params.type());
        CPPUNIT_ASSERT_EQUAL(1.5, params.propagationInterval());
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(maths::CConstantWeights::SEASONAL_VARIANCE),
                             core::CContainerPrinter::print(params.weightStyles()));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(trendWeights),
                             core::CContainerPrinter::print(params.trendWeights()));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(priorWeights),
                             core::CContainerPrinter::print(params.priorWeights()));
    }
    {
        maths::CModelProbabilityParams::TDouble2Vec weight1(2, 0.4);
        maths::CModelProbabilityParams::TDouble2Vec weight2(2, 0.7);
        maths::CModelProbabilityParams::TDouble2Vec4Vec weights1(1, weight1);
        maths::CModelProbabilityParams::TDouble2Vec4Vec weights2(1, weight2);
        maths::CModelProbabilityParams params;
        CPPUNIT_ASSERT(!params.mostAnomalousCorrelate());
        CPPUNIT_ASSERT(params.coordinates().empty());
        params.addCalculation(maths_t::E_OneSidedAbove)
              .addCalculation(maths_t::E_TwoSided)
              .seasonalConfidenceInterval(50.0)
              .addBucketEmpty(maths::CModelProbabilityParams::TBool2Vec{true, true})
              .addBucketEmpty(maths::CModelProbabilityParams::TBool2Vec{false, true})
              .weightStyles(maths::CConstantWeights::COUNT_VARIANCE)
              .addWeights(weights1)
              .addWeights(weights2)
              .mostAnomalousCorrelate(1)
              .addCoordinate(1)
              .addCoordinate(0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), params.calculations());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove, params.calculation(0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, params.calculation(1));
        CPPUNIT_ASSERT_EQUAL(50.0, params.seasonalConfidenceInterval());
        CPPUNIT_ASSERT_EQUAL(std::string("[[true, true], [false, true]]"),
                             core::CContainerPrinter::print(params.bucketEmpty()));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(maths::CConstantWeights::COUNT_VARIANCE),
                             core::CContainerPrinter::print(params.weightStyles()));
        CPPUNIT_ASSERT_EQUAL(std::string("[[[0.4, 0.4]], [[0.7, 0.7]]]"),
                             core::CContainerPrinter::print(params.weights()));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), *params.mostAnomalousCorrelate());
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 0]"), core::CContainerPrinter::print(params.coordinates()));
    }
}

CppUnit::Test *CModelTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CModelTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CModelTest>(
                                   "CModelTest::testAll",
                                   &CModelTest::testAll) );

    return suiteOfTests;
}
