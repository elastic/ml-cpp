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

#include <maths/analytics/CBoostedTreeHyperparameters.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <limits>

BOOST_AUTO_TEST_SUITE(CBoostedTreeHyperparametersTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleParameter = maths::analytics::CBoostedTreeParameter<double>;
using TMeanAccumulator = maths::analytics::CBoostedTreeHyperparameters::TMeanAccumulator;
using TMeanVarAccumulator = maths::analytics::CBoostedTreeHyperparameters::TMeanVarAccumulator;
using TAddInitialRangeFunc = maths::analytics::CBoostedTreeHyperparameters::TAddInitialRangeFunc;

BOOST_AUTO_TEST_CASE(testBoostedTreeParameter) {

    TDoubleParameter parameter1{10.0};

    BOOST_REQUIRE_EQUAL(10.0, parameter1.value());

    parameter1.save();
    parameter1.set(5.0);

    BOOST_REQUIRE_EQUAL(5.0, parameter1.value());

    parameter1.load();

    BOOST_REQUIRE_EQUAL(10.0, parameter1.value());

    parameter1.fixTo(12.0);

    BOOST_TEST_REQUIRE(parameter1.fixed());
    BOOST_REQUIRE_EQUAL(12.0, parameter1.value());

    parameter1.set(10.0);

    BOOST_REQUIRE_EQUAL(12.0, parameter1.value());

    TDoubleParameter parameter2{10.0};

    parameter2.fixToRange(8.0, 12.0);

    BOOST_TEST_REQUIRE(parameter2.rangeFixed());

    parameter2.set(11.0);

    BOOST_REQUIRE_EQUAL(11.0, parameter2.value());

    parameter2.set(7.0);

    BOOST_REQUIRE_EQUAL(8.0, parameter2.value());

    parameter2.set(15.0);

    BOOST_REQUIRE_EQUAL(12.0, parameter2.value());

    TDoubleParameter parameter3{10.0};

    parameter3.fixTo(TDoubleVec{11.0});

    BOOST_TEST_REQUIRE(parameter3.fixed());
    BOOST_REQUIRE_EQUAL(11.0, parameter3.value());

    TDoubleParameter parameter4{10.0};

    parameter4.fixTo(TDoubleVec{12.0, 14.0});

    BOOST_TEST_REQUIRE(parameter4.rangeFixed());
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

    auto asConst = [](const maths::analytics::CBoostedTreeHyperparameters& parameters) {
        return &parameters;
    };

    maths::analytics::CBoostedTreeHyperparameters origHyperaparameters;
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

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisationCaptureBest) {

    // Check that we do recover best hyperparameters with minimum loss.

    maths::analytics::CBoostedTreeHyperparameters hyperaparameters;

    hyperaparameters.stopHyperparameterOptimizationEarly(false);

    auto addInitialRange =
        [](maths::analytics::boosted_tree_detail::EHyperparameter,
           maths::analytics::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
            bb.emplace_back(0.1, 1.0);
        };

    hyperaparameters.initializeSearch(addInitialRange);

    test::CRandomNumbers rng;
    TDoubleVec losses;

    double minimumLoss{std::numeric_limits<double>::max()};
    TDoubleVec expectedBestParameters;

    for (hyperaparameters.startSearch(); hyperaparameters.searchNotFinished();
         hyperaparameters.startNextSearchRound()) {

        TMeanVarAccumulator testLossMoments;
        rng.generateUniformSamples(0.5, 1.5, 3, losses);
        testLossMoments.add(losses);

        hyperaparameters.captureBest(testLossMoments, 0.0, 100, 0, 50);

        double loss{maths::analytics::CBoostedTreeHyperparameters::lossAtNSigma(1, testLossMoments)};
        if (loss < minimumLoss) {
            expectedBestParameters.assign(
                {hyperaparameters.depthPenaltyMultiplier().value(),
                 hyperaparameters.treeSizePenaltyMultiplier().value(),
                 hyperaparameters.leafWeightPenaltyMultiplier().value(),
                 hyperaparameters.softTreeDepthLimit().value(),
                 hyperaparameters.softTreeDepthTolerance().value(),
                 hyperaparameters.downsampleFactor().value(),
                 hyperaparameters.featureBagFraction().value(),
                 hyperaparameters.etaGrowthRatePerTree().value(),
                 hyperaparameters.eta().value()});
        }
    }

    hyperaparameters.restoreBest();

    BOOST_REQUIRE_EQUAL(expectedBestParameters[0],
                        hyperaparameters.depthPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[1],
                        hyperaparameters.treeSizePenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[2],
                        hyperaparameters.leafWeightPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[3],
                        hyperaparameters.softTreeDepthLimit().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[4],
                        hyperaparameters.softTreeDepthTolerance().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[5],
                        hyperaparameters.downsampleFactor().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[6],
                        hyperaparameters.featureBagFraction().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[7],
                        hyperaparameters.etaGrowthRatePerTree().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[8], hyperaparameters.eta().value());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisationWithOverrides) {

    // Check that fixed parameters are not adjusted.

    maths::analytics::CBoostedTreeHyperparameters hyperaparameters;

    hyperaparameters.maximumOptimisationRoundsPerHyperparameter(2);

    auto testFix = [&](TDoubleParameter& parameter, std::size_t& numberToTune) {

        parameter.fixTo(TDoubleVec{0.5});

        BOOST_REQUIRE_EQUAL(--numberToTune, hyperaparameters.numberToTune());

        auto addInitialRange =
            [](maths::analytics::boosted_tree_detail::EHyperparameter,
               maths::analytics::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
                bb.emplace_back(0.1, 1.0);
            };

        hyperaparameters.initializeSearch(addInitialRange);

        BOOST_REQUIRE_EQUAL(2 * hyperaparameters.numberToTune(),
                            hyperaparameters.numberRounds());

        test::CRandomNumbers rng;
        TDoubleVec losses;

        for (hyperaparameters.startSearch(); hyperaparameters.searchNotFinished();
             hyperaparameters.startNextSearchRound()) {

            TMeanVarAccumulator testLossMoments;
            rng.generateUniformSamples(0.1, 1.0, 3, losses);
            testLossMoments.add(losses);
            hyperaparameters.selectNext(testLossMoments);

            BOOST_REQUIRE_EQUAL(0.5, parameter.value());
        }
    };

    std::size_t numberToTune{hyperaparameters.numberToTune()};

    testFix(hyperaparameters.depthPenaltyMultiplier(), numberToTune);
    testFix(hyperaparameters.treeSizePenaltyMultiplier(), numberToTune);
    testFix(hyperaparameters.leafWeightPenaltyMultiplier(), numberToTune);
    testFix(hyperaparameters.softTreeDepthLimit(), numberToTune);
    testFix(hyperaparameters.softTreeDepthTolerance(), numberToTune);
    testFix(hyperaparameters.downsampleFactor(), numberToTune);
    testFix(hyperaparameters.featureBagFraction(), numberToTune);
    testFix(hyperaparameters.etaGrowthRatePerTree(), numberToTune);
    testFix(hyperaparameters.eta(), numberToTune);
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisationWithRangeOverrides) {

    // Check that fixed parameters are not adjusted.

    maths::analytics::CBoostedTreeHyperparameters hyperaparameters;

    hyperaparameters.maximumOptimisationRoundsPerHyperparameter(2);

    auto testFixToRange = [&](TDoubleParameter& parameter, std::size_t& numberToTune) {

        parameter.fixTo(TDoubleVec{0.25, 0.75});

        BOOST_REQUIRE_EQUAL(numberToTune, hyperaparameters.numberToTune());

        auto addInitialRange =
            [](maths::analytics::boosted_tree_detail::EHyperparameter,
               maths::analytics::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
                bb.emplace_back(0.1, 1.0);
            };

        hyperaparameters.initializeSearch(addInitialRange);

        BOOST_REQUIRE_EQUAL(2 * hyperaparameters.numberToTune(),
                            hyperaparameters.numberRounds());

        test::CRandomNumbers rng;
        TDoubleVec losses;

        for (hyperaparameters.startSearch(); hyperaparameters.searchNotFinished();
             hyperaparameters.startNextSearchRound()) {

            TMeanVarAccumulator testLossMoments;
            rng.generateUniformSamples(0.1, 1.0, 3, losses);
            testLossMoments.add(losses);
            hyperaparameters.selectNext(testLossMoments);

            BOOST_TEST_REQUIRE(parameter.value() >= 0.25);
            BOOST_TEST_REQUIRE(parameter.value() <= 0.75);
        }
    };

    std::size_t numberToTune{hyperaparameters.numberToTune()};

    testFixToRange(hyperaparameters.depthPenaltyMultiplier(), numberToTune);
    testFixToRange(hyperaparameters.treeSizePenaltyMultiplier(), numberToTune);
    testFixToRange(hyperaparameters.leafWeightPenaltyMultiplier(), numberToTune);
    testFixToRange(hyperaparameters.softTreeDepthLimit(), numberToTune);
    testFixToRange(hyperaparameters.softTreeDepthTolerance(), numberToTune);
    testFixToRange(hyperaparameters.downsampleFactor(), numberToTune);
    testFixToRange(hyperaparameters.featureBagFraction(), numberToTune);
    testFixToRange(hyperaparameters.etaGrowthRatePerTree(), numberToTune);
    testFixToRange(hyperaparameters.eta(), numberToTune);
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersResetSearch) {

    // Check that we generate exactly the same sequence of hyperparameters after reset
    // and the same best values.

    maths::analytics::CBoostedTreeHyperparameters hyperaparameters;

    hyperaparameters.stopHyperparameterOptimizationEarly(false);

    auto addInitialRange =
        [](maths::analytics::boosted_tree_detail::EHyperparameter,
           maths::analytics::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
            bb.emplace_back(0.1, 1.0);
        };

    hyperaparameters.initializeSearch(addInitialRange);

    auto initHyperaparameters = [&] {
        hyperaparameters.depthPenaltyMultiplier().set(0.5);
        hyperaparameters.treeSizePenaltyMultiplier().set(0.5);
        hyperaparameters.leafWeightPenaltyMultiplier().set(0.5);
        hyperaparameters.softTreeDepthLimit().set(0.5);
        hyperaparameters.softTreeDepthTolerance().set(0.5);
        hyperaparameters.downsampleFactor().set(0.5);
        hyperaparameters.featureBagFraction().set(0.5);
        hyperaparameters.etaGrowthRatePerTree().set(0.5);
        hyperaparameters.eta().set(0.5);
    };

    test::CRandomNumbers rng;

    TDoubleVec losses;
    rng.generateUniformSamples(0.1, 1.0, 3 * hyperaparameters.numberRounds(), losses);

    TDoubleVecVec previousHyperparameters;
    double minimumLoss{std::numeric_limits<double>::max()};

    initHyperaparameters();

    for (hyperaparameters.startSearch(); hyperaparameters.searchNotFinished();
         hyperaparameters.startNextSearchRound()) {
        TMeanVarAccumulator testLossMoments;
        testLossMoments.add(losses[3 * hyperaparameters.currentRound() + 0]);
        testLossMoments.add(losses[3 * hyperaparameters.currentRound() + 1]);
        testLossMoments.add(losses[3 * hyperaparameters.currentRound() + 2]);

        hyperaparameters.selectNext(testLossMoments);
        hyperaparameters.captureBest(testLossMoments, 0.0, 100, 0, 50);

        previousHyperparameters.push_back(
            TDoubleVec{hyperaparameters.depthPenaltyMultiplier().value(),
                       hyperaparameters.treeSizePenaltyMultiplier().value(),
                       hyperaparameters.leafWeightPenaltyMultiplier().value(),
                       hyperaparameters.softTreeDepthLimit().value(),
                       hyperaparameters.softTreeDepthTolerance().value(),
                       hyperaparameters.downsampleFactor().value(),
                       hyperaparameters.featureBagFraction().value(),
                       hyperaparameters.etaGrowthRatePerTree().value(),
                       hyperaparameters.eta().value()});
    }

    hyperaparameters.restoreBest();

    TDoubleVec previousBestHyperparameters;
    previousBestHyperparameters.assign(
        {hyperaparameters.depthPenaltyMultiplier().value(),
         hyperaparameters.treeSizePenaltyMultiplier().value(),
         hyperaparameters.leafWeightPenaltyMultiplier().value(),
         hyperaparameters.softTreeDepthLimit().value(),
         hyperaparameters.softTreeDepthTolerance().value(),
         hyperaparameters.downsampleFactor().value(),
         hyperaparameters.featureBagFraction().value(),
         hyperaparameters.etaGrowthRatePerTree().value(), hyperaparameters.eta().value()});

    hyperaparameters.resetSearch();
    hyperaparameters.initializeSearch(addInitialRange);

    initHyperaparameters();
    minimumLoss = std::numeric_limits<double>::max();
    TDoubleVec bestParameters;

    for (hyperaparameters.startSearch(); hyperaparameters.searchNotFinished();
         hyperaparameters.startNextSearchRound()) {

        TMeanVarAccumulator testLossMoments;
        testLossMoments.add(losses[3 * hyperaparameters.currentRound() + 0]);
        testLossMoments.add(losses[3 * hyperaparameters.currentRound() + 1]);
        testLossMoments.add(losses[3 * hyperaparameters.currentRound() + 2]);

        hyperaparameters.selectNext(testLossMoments);
        hyperaparameters.captureBest(testLossMoments, 0.0, 100, 0, 50);

        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][0],
                            hyperaparameters.depthPenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][1],
                            hyperaparameters.treeSizePenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][2],
                            hyperaparameters.leafWeightPenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][3],
                            hyperaparameters.softTreeDepthLimit().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][4],
                            hyperaparameters.softTreeDepthTolerance().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][5],
                            hyperaparameters.downsampleFactor().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][6],
                            hyperaparameters.featureBagFraction().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][7],
                            hyperaparameters.etaGrowthRatePerTree().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperaparameters.currentRound()][8],
                            hyperaparameters.eta().value());
    }

    hyperaparameters.restoreBest();

    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[0],
                        hyperaparameters.depthPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[1],
                        hyperaparameters.treeSizePenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[2],
                        hyperaparameters.leafWeightPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[3],
                        hyperaparameters.softTreeDepthLimit().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[4],
                        hyperaparameters.softTreeDepthTolerance().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[5],
                        hyperaparameters.downsampleFactor().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[6],
                        hyperaparameters.featureBagFraction().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[7],
                        hyperaparameters.etaGrowthRatePerTree().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[8], hyperaparameters.eta().value());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersPersistWithOverrides) {

    // Test that checksums of the original and restored parameters agree when
    // overrides are defined.

    maths::analytics::CBoostedTreeHyperparameters origHyperaparameters;
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

    maths::analytics::CBoostedTreeHyperparameters restoredHyperaparameters;
    BOOST_TEST_REQUIRE(restoredHyperaparameters.acceptRestoreTraverser(traverser));

    BOOST_REQUIRE_EQUAL(origHyperaparameters.checksum(),
                        restoredHyperaparameters.checksum());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersPersistWithOptimisation) {

    // Test that checksums of the original and restored parameters agree when
    // after starting hyperparameter search.

    maths::analytics::CBoostedTreeHyperparameters origHyperaparameters;

    auto addInitialRange =
        [](maths::analytics::boosted_tree_detail::EHyperparameter,
           maths::analytics::CBoostedTreeHyperparameters::TDoubleDoublePrVec& bb) {
            bb.emplace_back(0.1, 1.0);
        };
    origHyperaparameters.initializeSearch(addInitialRange);

    origHyperaparameters.startSearch();

    maths::analytics::CBoostedTreeHyperparameters::TMeanAccumulator forestSize;
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

    maths::analytics::CBoostedTreeHyperparameters restoredHyperaparameters;
    BOOST_TEST_REQUIRE(restoredHyperaparameters.acceptRestoreTraverser(traverser));

    restoredHyperaparameters.startSearch();

    BOOST_REQUIRE_EQUAL(origHyperaparameters.checksum(),
                        restoredHyperaparameters.checksum());
}

BOOST_AUTO_TEST_SUITE_END()
