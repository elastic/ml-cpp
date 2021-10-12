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

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <maths/CBoostedTreeHyperparameters.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CBoostedTreeHyperparametersTest)

using namespace ml;

using TDoubleParameter = maths::CBoostedTreeParameter<double>;
using TMeanAccumulator = maths::CBoostedTreeHyperparameters::TMeanAccumulator;
using TMeanVarAccumulator = maths::CBoostedTreeHyperparameters::TMeanVarAccumulator;
using TAddInitialRangeFunc = maths::CBoostedTreeHyperparameters::TAddInitialRangeFunc;

BOOST_AUTO_TEST_CASE(testBoostedTreeParameter) {

    TDoubleParameter parameter{10.0};

    BOOST_REQUIRE_EQUAL(10.0, parameter.value());

    parameter.save();
    parameter.set(5.0);

    BOOST_REQUIRE_EQUAL(5.0, parameter.value());

    parameter.load();

    BOOST_REQUIRE_EQUAL(10.0, parameter.value());

    parameter.fixTo(12.0);

    BOOST_REQUIRE_EQUAL(12.0, parameter.value());

    parameter.set(10.0);

    BOOST_REQUIRE_EQUAL(12.0, parameter.value());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeParameterPersist) {

    // Test that checksums of the original and restored parameters agree.

    TDoubleParameter origParameter{10.0};
    origParameter.save();
    origParameter.set(5.0);

    std::stringstream state;
    {
        // We need to call the inserter destructor to finish writing the state.
        core::CJsonStatePersistInserter inserter{state};
        origParameter.acceptPersistInserter(inserter);
        state.flush();
    }

    core::CJsonStateRestoreTraverser traverser(state);

    TDoubleParameter restoredParameter{1.0};
    BOOST_TEST_REQUIRE(restoredParameter.acceptRestoreTraverser(traverser));

    BOOST_REQUIRE_EQUAL(origParameter.checksum(), restoredParameter.checksum());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersAccessors) {

    // Check that setters and getters work for all hyperparameters.

    auto asConst = [](const maths::CBoostedTreeHyperparameters& parameters) {
        return &parameters;
    };

    maths::CBoostedTreeHyperparameters origHyperaparameters;
    origHyperaparameters.treeSizePenaltyMultiplier().set(1.0);
    BOOST_REQUIRE_EQUAL(
        1.0, asConst(origHyperaparameters)->treeSizePenaltyMultiplier().value());
    origHyperaparameters.leafWeightPenaltyMultiplier().set(5.0);
    BOOST_REQUIRE_EQUAL(
        5.0, asConst(origHyperaparameters)->leafWeightPenaltyMultiplier().value());
    origHyperaparameters.softTreeDepthLimit().set(8.0);
    BOOST_REQUIRE_EQUAL(8.0, asConst(origHyperaparameters)->softTreeDepthLimit().value());
    origHyperaparameters.treeTopologyChangePenalty().set(1.5);
    BOOST_REQUIRE_EQUAL(
        1.5, asConst(origHyperaparameters)->treeTopologyChangePenalty().value());
    origHyperaparameters.downsampleFactor().set(0.2);
    BOOST_REQUIRE_EQUAL(0.2, asConst(origHyperaparameters)->downsampleFactor().value());
    origHyperaparameters.featureBagFraction().set(0.5);
    BOOST_REQUIRE_EQUAL(0.5, asConst(origHyperaparameters)->featureBagFraction().value());
    origHyperaparameters.eta().set(0.3);
    BOOST_REQUIRE_EQUAL(0.3, asConst(origHyperaparameters)->eta().value());
    origHyperaparameters.etaGrowthRatePerTree().set(0.05);
    BOOST_REQUIRE_EQUAL(
        0.05, asConst(origHyperaparameters)->etaGrowthRatePerTree().value());
    origHyperaparameters.retrainedTreeEta().set(0.9);
    BOOST_REQUIRE_EQUAL(0.9, asConst(origHyperaparameters)->retrainedTreeEta().value());
    origHyperaparameters.predictionChangeCost().set(20.0);
    BOOST_REQUIRE_EQUAL(
        20.0, asConst(origHyperaparameters)->predictionChangeCost().value());
    origHyperaparameters.maximumNumberTrees().set(100);
    BOOST_REQUIRE_EQUAL(100, asConst(origHyperaparameters)->maximumNumberTrees().value());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisation) {

    // Check that hyperparameter optimisation generates expected point sequences.
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisationWithOverrides) {

    // Check that fixed parameters are not adjusted.

    maths::CBoostedTreeHyperparameters hyperaparameters;

    hyperaparameters.maximumOptimisationRoundsPerHyperparameter(2);

    std::size_t numberToTune{hyperaparameters.numberToTune()};

    hyperaparameters.treeSizePenaltyMultiplier().fixTo(10.0);

    BOOST_REQUIRE_EQUAL(numberToTune - 1, hyperaparameters.numberToTune());

    auto addInitialRange = [](maths::boosted_tree_detail::EHyperparameter,
                              maths::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
        bb.emplace_back(0.1, 1.0);
    };

    hyperaparameters.initializeSearch(addInitialRange);

    BOOST_REQUIRE_EQUAL(2 * hyperaparameters.numberToTune(),
                        hyperaparameters.numberRounds());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersPersistWithOverrides) {

    // Test that checksums of the original and restored parameters agree when
    // overrides are defined.

    maths::CBoostedTreeHyperparameters origHyperaparameters;
    origHyperaparameters.treeSizePenaltyMultiplier().set(1.0);
    origHyperaparameters.leafWeightPenaltyMultiplier().set(5.0);
    origHyperaparameters.softTreeDepthLimit().set(8.0);
    origHyperaparameters.treeTopologyChangePenalty().set(1.5);
    origHyperaparameters.downsampleFactor().set(0.2);
    origHyperaparameters.featureBagFraction().set(0.5);
    origHyperaparameters.eta().set(0.3);
    origHyperaparameters.etaGrowthRatePerTree().set(0.05);
    origHyperaparameters.retrainedTreeEta().set(0.9);
    origHyperaparameters.predictionChangeCost().set(20.0);
    origHyperaparameters.maximumNumberTrees().set(100);

    TMeanVarAccumulator testLossMoments;
    testLossMoments.add(1.0);
    origHyperaparameters.captureBest(testLossMoments, 0.0, 0.0, 100.0, 500);

    std::stringstream state;
    {
        // We need to call the inserter destructor to finish writing the state.
        core::CJsonStatePersistInserter inserter{state};
        origHyperaparameters.acceptPersistInserter(inserter);
        state.flush();
    }

    core::CJsonStateRestoreTraverser traverser(state);

    maths::CBoostedTreeHyperparameters restoredHyperaparameters;
    BOOST_TEST_REQUIRE(restoredHyperaparameters.acceptRestoreTraverser(traverser));

    BOOST_REQUIRE_EQUAL(origHyperaparameters.checksum(),
                        restoredHyperaparameters.checksum());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersPersistWithOptimisation) {

    // Test that checksums of the original and restored parameters agree when
    // after starting hyperparameter search.

    maths::CBoostedTreeHyperparameters origHyperaparameters;

    auto addInitialRange = [](maths::boosted_tree_detail::EHyperparameter,
                              maths::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
        bb.emplace_back(0.1, 1.0);
    };
    origHyperaparameters.initializeSearch(addInitialRange);

    origHyperaparameters.startSearch();

    maths::CBoostedTreeHyperparameters::TMeanAccumulator forestSize;
    forestSize.add(103.0);
    double meanTestLoss{1.0};
    origHyperaparameters.addRoundStats(forestSize, meanTestLoss);

    TMeanVarAccumulator testLossMoments;
    testLossMoments.add(1.0);
    testLossMoments.add(1.2);
    testLossMoments.add(1.1);
    origHyperaparameters.captureBest(testLossMoments, 0.0, 0.0, 100.0, 500);

    origHyperaparameters.selectNext(testLossMoments);

    std::stringstream state;
    {
        // We need to call the inserter destructor to finish writing the state.
        core::CJsonStatePersistInserter inserter{state};
        origHyperaparameters.acceptPersistInserter(inserter);
        state.flush();
    }

    core::CJsonStateRestoreTraverser traverser(state);

    maths::CBoostedTreeHyperparameters restoredHyperaparameters;
    BOOST_TEST_REQUIRE(restoredHyperaparameters.acceptRestoreTraverser(traverser));

    restoredHyperaparameters.startSearch();

    BOOST_REQUIRE_EQUAL(origHyperaparameters.checksum(),
                        restoredHyperaparameters.checksum());
}

BOOST_AUTO_TEST_SUITE_END()
