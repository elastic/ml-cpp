/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CBoostedTreeUtils.h>

#include <test/CRandomNumbers.h>

#include "BoostedTreeTestData.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CBoostedTreeUtilsTest)

using namespace ml;
using TDoubleVec = std::vector<double>;

BOOST_AUTO_TEST_CASE(testRetrainTreeSelectionProbabilities) {

    // 1. Edge cases such as empty forest,
    // 2. If all the new training data is applies to a particular leaf node
    //    that tree is selected with high probability.

    std::size_t rows{100};
    std::size_t cols{4};
    TDoubleVec m{10.0, 2.0, 5.0};
    TDoubleVec s{1.0, -0.9, 0.8};

    TTargetFunc target{[&](const core::CDataFrame::TRowRef& row) {
        double result{0.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += m[i] + s[i] * row[i];
        }
        return result;
    }};

    test::CRandomNumbers rng;

    // Edge cases
    {
        auto frame = setupRegressionProblem(rng, target, 1.0, rows, cols);

        auto regression = maths::CBoostedTreeFactory::constructFromParameters(
                              1, std::make_unique<maths::boosted_tree::CMse>())
                              .buildFor(*frame, cols - 1);
        regression->train();

        const auto& hyperparameters = regression->impl().bestHyperparameters();
        core::CPackedBitVector allRows{rows, true};
        core::CPackedBitVector noRows;

        std::vector<std::vector<maths::CBoostedTreeNode>> emptyForest;
        auto probabilities = maths::boosted_tree_detail::retrainTreeSelectionProbabilities(
            1, *frame, regression->impl().extraColumns(), cols - 1,
            regression->impl().encoder(), allRows, noRows, regression->impl().loss(),
            hyperparameters.regularization(), hyperparameters.eta(),
            hyperparameters.etaGrowthRatePerTree(), emptyForest);
        BOOST_TEST_REQUIRE(probabilities.empty());

        probabilities = maths::boosted_tree_detail::retrainTreeSelectionProbabilities(
            1, *frame, regression->impl().extraColumns(), cols - 1,
            regression->impl().encoder(), allRows, noRows, regression->impl().loss(),
            hyperparameters.regularization(), hyperparameters.eta(),
            hyperparameters.etaGrowthRatePerTree(), regression->impl().trainedModel());
        BOOST_REQUIRE_EQUAL(regression->impl().trainedModel().size(),
                            probabilities.size());
        BOOST_REQUIRE_EQUAL(0.0, probabilities[0]);
        for (std::size_t i = 1; i < probabilities.size(); ++i) {
            BOOST_REQUIRE_EQUAL(1.0 / static_cast<double>(probabilities.size() - 1),
                                probabilities[i]);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
