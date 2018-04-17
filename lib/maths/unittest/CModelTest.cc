/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CModelTest.h"

#include <core/CLogger.h>

#include <maths/CModel.h>
#include <maths/Constants.h>

using namespace ml;

void CModelTest::testAll() {
    // Test that the various parameter classes work as expected.

    using TDouble2Vec = maths_t::TDouble2Vec;
    using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;

    {
        core_t::TTime bucketLength{600};
        double learnRate{0.5};
        double decayRate{0.001};
        double minimumSeasonalVarianceScale{0.3};
        maths::CModelParams params(bucketLength, learnRate, decayRate,
                                   minimumSeasonalVarianceScale);
        CPPUNIT_ASSERT_EQUAL(bucketLength, params.bucketLength());
        CPPUNIT_ASSERT_EQUAL(learnRate, params.learnRate());
        CPPUNIT_ASSERT_EQUAL(decayRate, params.decayRate());
        CPPUNIT_ASSERT_EQUAL(minimumSeasonalVarianceScale,
                             params.minimumSeasonalVarianceScale());
        CPPUNIT_ASSERT_EQUAL(0.0, params.probabilityBucketEmpty());
        params.probabilityBucketEmpty(0.2);
        CPPUNIT_ASSERT_EQUAL(0.2, params.probabilityBucketEmpty());
    }
    {
        maths_t::TDouble2VecWeightsAry weight1(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        maths_t::TDouble2VecWeightsAry weight2(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        maths_t::setSeasonalVarianceScale(TDouble2Vec(2, 0.4), weight1);
        maths_t::setSeasonalVarianceScale(TDouble2Vec(2, 0.7), weight2);
        TDouble2VecWeightsAryVec trendWeights{weight1};
        TDouble2VecWeightsAryVec priorWeights{weight2};
        maths::CModelAddSamplesParams params;
        params.integer(true).propagationInterval(1.5).trendWeights(trendWeights).priorWeights(priorWeights);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_IntegerData, params.type());
        CPPUNIT_ASSERT_EQUAL(1.5, params.propagationInterval());
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(trendWeights),
                             core::CContainerPrinter::print(params.trendWeights()));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(priorWeights),
                             core::CContainerPrinter::print(params.priorWeights()));
    }
    {
        maths_t::TDouble2VecWeightsAry weight1(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        maths_t::TDouble2VecWeightsAry weight2(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        maths_t::setCountVarianceScale(TDouble2Vec(2, 0.4), weight1);
        maths_t::setCountVarianceScale(TDouble2Vec(2, 0.7), weight2);
        TDouble2VecWeightsAryVec weights{weight1, weight2};
        maths::CModelProbabilityParams params;
        CPPUNIT_ASSERT(!params.mostAnomalousCorrelate());
        CPPUNIT_ASSERT(params.coordinates().empty());
        params.addCalculation(maths_t::E_OneSidedAbove)
            .addCalculation(maths_t::E_TwoSided)
            .seasonalConfidenceInterval(50.0)
            .addBucketEmpty({true, true})
            .addBucketEmpty({false, true})
            .addWeights(weight1)
            .addWeights(weight2)
            .mostAnomalousCorrelate(1)
            .addCoordinate(1)
            .addCoordinate(0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), params.calculations());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove, params.calculation(0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, params.calculation(1));
        CPPUNIT_ASSERT_EQUAL(50.0, params.seasonalConfidenceInterval());
        CPPUNIT_ASSERT_EQUAL(std::string("[[true, true], [false, true]]"),
                             core::CContainerPrinter::print(params.bucketEmpty()));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(weights),
                             core::CContainerPrinter::print(params.weights()));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), *params.mostAnomalousCorrelate());
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 0]"),
                             core::CContainerPrinter::print(params.coordinates()));
    }
}

CppUnit::Test* CModelTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CModelTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CModelTest>(
        "CModelTest::testAll", &CModelTest::testAll));

    return suiteOfTests;
}
