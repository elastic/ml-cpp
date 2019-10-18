/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CModel.h>
#include <maths/Constants.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CModelTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testAll) {
    // Test that the various parameter classes work as expected.

    using TDouble2Vec = maths_t::TDouble2Vec;
    using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;

    {
        core_t::TTime bucketLength{600};
        double learnRate{0.5};
        double decayRate{0.001};
        double minimumSeasonalVarianceScale{0.3};
        maths::CModelParams params(bucketLength, learnRate, decayRate, minimumSeasonalVarianceScale,
                                   6 * core::constants::HOUR, core::constants::DAY);
        BOOST_CHECK_EQUAL(bucketLength, params.bucketLength());
        BOOST_CHECK_EQUAL(learnRate, params.learnRate());
        BOOST_CHECK_EQUAL(decayRate, params.decayRate());
        BOOST_CHECK_EQUAL(minimumSeasonalVarianceScale,
                             params.minimumSeasonalVarianceScale());
        BOOST_CHECK_EQUAL(6 * core::constants::HOUR, params.minimumTimeToDetectChange());
        BOOST_CHECK_EQUAL(core::constants::DAY, params.maximumTimeToTestForChange());
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
        BOOST_CHECK_EQUAL(maths_t::E_IntegerData, params.type());
        BOOST_CHECK_EQUAL(1.5, params.propagationInterval());
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(trendWeights),
                             core::CContainerPrinter::print(params.trendWeights()));
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(priorWeights),
                             core::CContainerPrinter::print(params.priorWeights()));
    }
    {
        maths_t::TDouble2VecWeightsAry weight1(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        maths_t::TDouble2VecWeightsAry weight2(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        maths_t::setCountVarianceScale(TDouble2Vec(2, 0.4), weight1);
        maths_t::setCountVarianceScale(TDouble2Vec(2, 0.7), weight2);
        TDouble2VecWeightsAryVec weights{weight1, weight2};
        maths::CModelProbabilityParams params;
        BOOST_TEST(!params.mostAnomalousCorrelate());
        BOOST_TEST(params.coordinates().empty());
        params.addCalculation(maths_t::E_OneSidedAbove)
            .addCalculation(maths_t::E_TwoSided)
            .seasonalConfidenceInterval(50.0)
            .addWeights(weight1)
            .addWeights(weight2)
            .mostAnomalousCorrelate(1)
            .addCoordinate(1)
            .addCoordinate(0);
        BOOST_CHECK_EQUAL(std::size_t(2), params.calculations());
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove, params.calculation(0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, params.calculation(1));
        BOOST_CHECK_EQUAL(50.0, params.seasonalConfidenceInterval());
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(weights),
                             core::CContainerPrinter::print(params.weights()));
        BOOST_CHECK_EQUAL(std::size_t(1), *params.mostAnomalousCorrelate());
        BOOST_CHECK_EQUAL(std::string("[1, 0]"),
                             core::CContainerPrinter::print(params.coordinates()));
    }
}


BOOST_AUTO_TEST_SUITE_END()
