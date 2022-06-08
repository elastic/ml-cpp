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
#include <maths/analytics/CBoostedTreeImpl.h>
#include <maths/analytics/CBoostedTreeLoss.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <limits>
#include <utility>

BOOST_AUTO_TEST_SUITE(CBoostedTreeHyperparametersTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleParameter = maths::analytics::CBoostedTreeParameter<double>;
using TMeanAccumulator = maths::analytics::CBoostedTreeHyperparameters::TMeanAccumulator;
using TMeanVarAccumulator = maths::analytics::CBoostedTreeHyperparameters::TMeanVarAccumulator;

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

BOOST_AUTO_TEST_CASE(testBoostedTreeParameterScaling) {

    TDoubleParameter parameter{10.0};

    parameter.scale(1.5);
    BOOST_REQUIRE_EQUAL(1.5, parameter.scale());
    BOOST_REQUIRE_EQUAL(15.0, parameter.value());

    parameter.scale(1.2).captureScale();
    BOOST_REQUIRE_EQUAL(1.0, parameter.scale());
    BOOST_REQUIRE_EQUAL(12.0, parameter.value());

    parameter.fixToRange(12.0, 15.0);

    parameter.scale(0.1);
    BOOST_REQUIRE_CLOSE(1.2, parameter.value(), 1e-6);
    BOOST_REQUIRE_EQUAL(12.0, parameter.toSearchValue());
    BOOST_REQUIRE_EQUAL(12.0, parameter.searchRange().first);
    BOOST_REQUIRE_EQUAL(15.0, parameter.searchRange().second);

    parameter.captureScale();
    BOOST_REQUIRE_CLOSE(1.2, parameter.value(), 1e-6);
    BOOST_REQUIRE_EQUAL(1.0, parameter.scale());
    BOOST_REQUIRE_CLOSE(1.2, parameter.toSearchValue(), 1e-6);
    BOOST_REQUIRE_CLOSE(1.2, parameter.searchRange().first, 1e-6);
    BOOST_REQUIRE_CLOSE(1.5, parameter.searchRange().second, 1e-6);
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

    maths::analytics::CBoostedTreeHyperparameters hyperparameters;

    hyperparameters.stopHyperparameterOptimizationEarly(false);

    hyperparameters.depthPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.treeSizePenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.leafWeightPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthLimit().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthTolerance().fixToRange(0.1, 1.0);
    hyperparameters.downsampleFactor().fixToRange(0.1, 1.0);
    hyperparameters.featureBagFraction().fixToRange(0.1, 1.0);
    hyperparameters.eta().fixToRange(0.1, 1.0);
    hyperparameters.etaGrowthRatePerTree().fixToRange(0.1, 1.0);

    hyperparameters.initializeFineTuneSearch(0.0, 0);

    test::CRandomNumbers rng;
    TDoubleVec losses;

    double minimumLoss{std::numeric_limits<double>::max()};
    TDoubleVec expectedBestParameters;

    for (hyperparameters.startFineTuneSearch();
         hyperparameters.fineTuneSearchNotFinished(); hyperparameters.startNextRound()) {

        TMeanVarAccumulator testLossMoments;
        rng.generateUniformSamples(0.5, 1.5, 3, losses);
        testLossMoments.add(losses);

        hyperparameters.captureBest(testLossMoments, 0.0, 100, 0, 50);

        double loss{maths::analytics::CBoostedTreeHyperparameters::lossAtNSigma(1, testLossMoments)};
        if (loss < minimumLoss) {
            expectedBestParameters.assign(
                {hyperparameters.depthPenaltyMultiplier().value(),
                 hyperparameters.treeSizePenaltyMultiplier().value(),
                 hyperparameters.leafWeightPenaltyMultiplier().value(),
                 hyperparameters.softTreeDepthLimit().value(),
                 hyperparameters.softTreeDepthTolerance().value(),
                 hyperparameters.downsampleFactor().value(),
                 hyperparameters.featureBagFraction().value(),
                 hyperparameters.etaGrowthRatePerTree().value(),
                 hyperparameters.eta().value()});
        }
    }

    hyperparameters.restoreBest();

    BOOST_REQUIRE_EQUAL(expectedBestParameters[0],
                        hyperparameters.depthPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[1],
                        hyperparameters.treeSizePenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[2],
                        hyperparameters.leafWeightPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[3],
                        hyperparameters.softTreeDepthLimit().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[4],
                        hyperparameters.softTreeDepthTolerance().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[5],
                        hyperparameters.downsampleFactor().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[6],
                        hyperparameters.featureBagFraction().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[7],
                        hyperparameters.etaGrowthRatePerTree().value());
    BOOST_REQUIRE_EQUAL(expectedBestParameters[8], hyperparameters.eta().value());
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisationWithOverrides) {

    // Check that fixed parameters are not adjusted.

    maths::analytics::CBoostedTreeHyperparameters hyperparameters;

    hyperparameters.maximumOptimisationRoundsPerHyperparameter(2);

    hyperparameters.depthPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.treeSizePenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.leafWeightPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthLimit().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthTolerance().fixToRange(0.1, 1.0);
    hyperparameters.downsampleFactor().fixToRange(0.1, 1.0);
    hyperparameters.featureBagFraction().fixToRange(0.1, 1.0);
    hyperparameters.eta().fixToRange(0.1, 1.0);
    hyperparameters.etaGrowthRatePerTree().fixToRange(0.1, 1.0);

    auto testFix = [&](TDoubleParameter& parameter, std::size_t& numberToTune) {

        parameter.fixTo(TDoubleVec{0.5});

        hyperparameters.initializeFineTuneSearch(0.0, 0);

        BOOST_REQUIRE_EQUAL(--numberToTune, hyperparameters.numberToTune());
        BOOST_REQUIRE_EQUAL(2 * hyperparameters.numberToTune(),
                            hyperparameters.numberRounds());

        test::CRandomNumbers rng;
        TDoubleVec losses;
        for (hyperparameters.startFineTuneSearch();
             hyperparameters.fineTuneSearchNotFinished();
             hyperparameters.startNextRound()) {

            TMeanVarAccumulator testLossMoments;
            rng.generateUniformSamples(0.1, 1.0, 3, losses);
            testLossMoments.add(losses);
            hyperparameters.selectNext(testLossMoments);

            BOOST_REQUIRE_EQUAL(0.5, parameter.value());
        }
    };

    std::size_t numberToTune{hyperparameters.numberToTune()};

    testFix(hyperparameters.downsampleFactor(), numberToTune);
    testFix(hyperparameters.depthPenaltyMultiplier(), numberToTune);
    testFix(hyperparameters.treeSizePenaltyMultiplier(), numberToTune);
    testFix(hyperparameters.leafWeightPenaltyMultiplier(), numberToTune);
    testFix(hyperparameters.softTreeDepthLimit(), numberToTune);
    testFix(hyperparameters.softTreeDepthTolerance(), numberToTune);
    testFix(hyperparameters.featureBagFraction(), numberToTune);
    testFix(hyperparameters.etaGrowthRatePerTree(), numberToTune);
    testFix(hyperparameters.eta(), numberToTune);
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersOptimisationWithRangeOverrides) {

    // Check that fixed parameters are not adjusted.

    maths::analytics::CBoostedTreeHyperparameters hyperparameters;

    hyperparameters.maximumOptimisationRoundsPerHyperparameter(2);

    // Since downsample factor acts exactly like a divisor of the regularisation
    // multipliers we adjust them as we vary this parameter to keep the amount
    // of regularisation fixed. As such we override it so we can assess we really
    // do keep all parameters in range subject without scaling.
    hyperparameters.downsampleFactor().fixToRange(0.5, 0.5);

    hyperparameters.depthPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.treeSizePenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.leafWeightPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthLimit().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthTolerance().fixToRange(0.1, 1.0);
    hyperparameters.featureBagFraction().fixToRange(0.1, 1.0);
    hyperparameters.eta().fixToRange(0.1, 1.0);
    hyperparameters.etaGrowthRatePerTree().fixToRange(0.1, 1.0);

    auto testFixToRange = [&](TDoubleParameter& parameter, std::size_t& numberToTune) {

        parameter.fixTo(TDoubleVec{0.25, 0.75});

        BOOST_REQUIRE_EQUAL(numberToTune, hyperparameters.numberToTune());

        hyperparameters.initializeFineTuneSearch(0.0, 0);

        BOOST_REQUIRE_EQUAL(2 * hyperparameters.numberToTune(),
                            hyperparameters.numberRounds());

        test::CRandomNumbers rng;
        TDoubleVec losses;

        for (hyperparameters.startFineTuneSearch();
             hyperparameters.fineTuneSearchNotFinished();
             hyperparameters.startNextRound()) {

            TMeanVarAccumulator testLossMoments;
            rng.generateUniformSamples(0.1, 1.0, 3, losses);
            testLossMoments.add(losses);
            hyperparameters.selectNext(testLossMoments);

            BOOST_TEST_REQUIRE(hyperparameters.downsampleFactor().value(), 0.5);
            BOOST_TEST_REQUIRE(parameter.value() >= 0.25);
            BOOST_TEST_REQUIRE(parameter.value() <= 0.75);
        }
    };

    std::size_t numberToTune{hyperparameters.numberToTune()};

    testFixToRange(hyperparameters.depthPenaltyMultiplier(), numberToTune);
    testFixToRange(hyperparameters.treeSizePenaltyMultiplier(), numberToTune);
    testFixToRange(hyperparameters.leafWeightPenaltyMultiplier(), numberToTune);
    testFixToRange(hyperparameters.softTreeDepthLimit(), numberToTune);
    testFixToRange(hyperparameters.softTreeDepthTolerance(), numberToTune);
    testFixToRange(hyperparameters.featureBagFraction(), numberToTune);
    testFixToRange(hyperparameters.etaGrowthRatePerTree(), numberToTune);
    testFixToRange(hyperparameters.eta(), numberToTune);
}

BOOST_AUTO_TEST_CASE(testBoostedTreeHyperparametersResetSearch) {

    // Check that we generate exactly the same sequence of hyperparameters after reset
    // and the same best values.

    maths::analytics::CBoostedTreeHyperparameters hyperparameters;

    auto initHyperaparameters = [&] {
        hyperparameters.depthPenaltyMultiplier().set(0.5).scale(1.0);
        hyperparameters.treeSizePenaltyMultiplier().set(0.5).scale(1.0);
        hyperparameters.leafWeightPenaltyMultiplier().set(0.5).scale(1.0);
        hyperparameters.softTreeDepthLimit().set(0.5).scale(1.0);
        hyperparameters.softTreeDepthTolerance().set(0.5).scale(1.0);
        hyperparameters.downsampleFactor().set(0.5).scale(1.0);
        hyperparameters.featureBagFraction().set(0.5).scale(1.0);
        hyperparameters.etaGrowthRatePerTree().set(0.5).scale(1.0);
        hyperparameters.eta().set(0.5).scale(1.0);
    };

    hyperparameters.stopHyperparameterOptimizationEarly(false);
    hyperparameters.depthPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.treeSizePenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.leafWeightPenaltyMultiplier().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthLimit().fixToRange(0.1, 1.0);
    hyperparameters.softTreeDepthTolerance().fixToRange(0.1, 1.0);
    hyperparameters.downsampleFactor().fixToRange(0.1, 1.0);
    hyperparameters.featureBagFraction().fixToRange(0.1, 1.0);
    hyperparameters.eta().fixToRange(0.1, 1.0);
    hyperparameters.etaGrowthRatePerTree().fixToRange(0.1, 1.0);

    hyperparameters.initializeFineTuneSearch(0.0, 0);

    test::CRandomNumbers rng;

    TDoubleVec losses;
    rng.generateUniformSamples(0.1, 1.0, 3 * hyperparameters.numberRounds(), losses);

    TDoubleVecVec previousHyperparameters;

    initHyperaparameters();

    for (hyperparameters.startFineTuneSearch();
         hyperparameters.fineTuneSearchNotFinished(); hyperparameters.startNextRound()) {
        TMeanVarAccumulator testLossMoments;
        testLossMoments.add(losses[3 * hyperparameters.currentRound() + 0]);
        testLossMoments.add(losses[3 * hyperparameters.currentRound() + 1]);
        testLossMoments.add(losses[3 * hyperparameters.currentRound() + 2]);

        hyperparameters.selectNext(testLossMoments);
        hyperparameters.captureBest(testLossMoments, 0.0, 100, 0, 50);

        previousHyperparameters.push_back(
            TDoubleVec{hyperparameters.depthPenaltyMultiplier().value(),
                       hyperparameters.treeSizePenaltyMultiplier().value(),
                       hyperparameters.leafWeightPenaltyMultiplier().value(),
                       hyperparameters.softTreeDepthLimit().value(),
                       hyperparameters.softTreeDepthTolerance().value(),
                       hyperparameters.downsampleFactor().value(),
                       hyperparameters.featureBagFraction().value(),
                       hyperparameters.etaGrowthRatePerTree().value(),
                       hyperparameters.eta().value()});
    }

    hyperparameters.restoreBest();

    TDoubleVec previousBestHyperparameters;
    previousBestHyperparameters.assign(
        {hyperparameters.depthPenaltyMultiplier().value(),
         hyperparameters.treeSizePenaltyMultiplier().value(),
         hyperparameters.leafWeightPenaltyMultiplier().value(),
         hyperparameters.softTreeDepthLimit().value(),
         hyperparameters.softTreeDepthTolerance().value(),
         hyperparameters.downsampleFactor().value(),
         hyperparameters.featureBagFraction().value(),
         hyperparameters.etaGrowthRatePerTree().value(), hyperparameters.eta().value()});

    hyperparameters.resetFineTuneSearch();
    hyperparameters.initializeFineTuneSearch(0.0, 0);

    initHyperaparameters();
    TDoubleVec bestParameters;

    for (hyperparameters.startFineTuneSearch();
         hyperparameters.fineTuneSearchNotFinished(); hyperparameters.startNextRound()) {

        TMeanVarAccumulator testLossMoments;
        testLossMoments.add(losses[3 * hyperparameters.currentRound() + 0]);
        testLossMoments.add(losses[3 * hyperparameters.currentRound() + 1]);
        testLossMoments.add(losses[3 * hyperparameters.currentRound() + 2]);

        hyperparameters.selectNext(testLossMoments);
        hyperparameters.captureBest(testLossMoments, 0.0, 100, 0, 50);

        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][0],
                            hyperparameters.depthPenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][1],
                            hyperparameters.treeSizePenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][2],
                            hyperparameters.leafWeightPenaltyMultiplier().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][3],
                            hyperparameters.softTreeDepthLimit().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][4],
                            hyperparameters.softTreeDepthTolerance().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][5],
                            hyperparameters.downsampleFactor().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][6],
                            hyperparameters.featureBagFraction().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][7],
                            hyperparameters.etaGrowthRatePerTree().value());
        BOOST_REQUIRE_EQUAL(previousHyperparameters[hyperparameters.currentRound()][8],
                            hyperparameters.eta().value());
    }

    hyperparameters.restoreBest();

    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[0],
                        hyperparameters.depthPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[1],
                        hyperparameters.treeSizePenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[2],
                        hyperparameters.leafWeightPenaltyMultiplier().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[3],
                        hyperparameters.softTreeDepthLimit().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[4],
                        hyperparameters.softTreeDepthTolerance().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[5],
                        hyperparameters.downsampleFactor().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[6],
                        hyperparameters.featureBagFraction().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[7],
                        hyperparameters.etaGrowthRatePerTree().value());
    BOOST_REQUIRE_EQUAL(previousBestHyperparameters[8], hyperparameters.eta().value());
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

    origHyperaparameters.depthPenaltyMultiplier().fixToRange(0.1, 1.0);
    origHyperaparameters.treeSizePenaltyMultiplier().fixToRange(0.1, 1.0);
    origHyperaparameters.leafWeightPenaltyMultiplier().fixToRange(0.1, 1.0);
    origHyperaparameters.softTreeDepthLimit().fixToRange(0.1, 1.0);
    origHyperaparameters.softTreeDepthTolerance().fixToRange(0.1, 1.0);
    origHyperaparameters.downsampleFactor().fixToRange(0.1, 1.0);
    origHyperaparameters.featureBagFraction().fixToRange(0.1, 1.0);
    origHyperaparameters.eta().fixToRange(0.1, 1.0);
    origHyperaparameters.etaGrowthRatePerTree().fixToRange(0.1, 1.0);

    origHyperaparameters.initializeFineTuneSearch(0.0, 0);

    origHyperaparameters.startFineTuneSearch();

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

    restoredHyperaparameters.startFineTuneSearch();

    BOOST_REQUIRE_EQUAL(origHyperaparameters.checksum(),
                        restoredHyperaparameters.checksum());
}

BOOST_AUTO_TEST_SUITE_END()
