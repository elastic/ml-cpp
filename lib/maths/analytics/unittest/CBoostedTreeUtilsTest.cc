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

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeFactory.h>
#include <maths/analytics/CBoostedTreeImpl.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CBoostedTreeUtils.h>

#include <test/CRandomNumbers.h>

#include "BoostedTreeTestData.h"
#include <core/CStopWatch.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CBoostedTreeUtilsTest)

using namespace ml;
using maths::analytics::boosted_tree_detail::CSearchTree;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;

BOOST_AUTO_TEST_CASE(testRetrainTreeSelectionProbabilities) {

    std::size_t rows{100};
    std::size_t cols{4};
    TDoubleVec m{10.0, 2.0, 5.0};
    TDoubleVec s{1.0, -0.9, 0.8};
    double noiseVariance{1.0};

    TTargetFunc target{[=](const core::CDataFrame::TRowRef& row) {
        double result{0.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += m[i] + s[i] * row[i];
        }
        return result;
    }};

    test::CRandomNumbers rng;

    auto frame = setupRegressionProblem(rng, target, noiseVariance, rows, cols);

    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
                          .buildForTrain(*frame, cols - 1);
    regression->train();

    const auto& impl = regression->impl();

    // Edge case.
    {
        core::CPackedBitVector allRowsMask{rows, true};
        maths::analytics::CBoostedTree::TNodeVecVec emptyForest;
        auto probabilities =
            impl.retrainTreeSelectionProbabilities(*frame, allRowsMask, emptyForest);
        BOOST_TEST_REQUIRE(probabilities.empty());
    }

    // Test some invariants of the probabilities:
    //   1. That number probabilities is equal to the number of trees,
    //   2. That the probabilities are non-negative, and
    //   3. That the probabilities are normalized.
    for (std::size_t test = 0; test < 5; ++test) {
        m[0] += 5.0;
        s[0] -= 0.1;
        m[2] += 1.0;
        s[2] += 0.1;
        TTargetFunc deltaTarget{[=](const core::CDataFrame::TRowRef& row) {
            double result{0.0};
            for (std::size_t i = 0; i < cols - 1; ++i) {
                result += m[i] + s[i] * row[i];
            }
            return result;
        }};

        TDoubleVecVec x(cols - 1);
        for (std::size_t i = 0; i < cols - 1; ++i) {
            rng.generateUniformSamples(0.0, 10.0, 10, x[i]);
        }
        addData(rng, deltaTarget, x, noiseVariance, *frame);

        core::CPackedBitVector allRowsMask{frame->numberRows(), true};
        core::CPackedBitVector newRowsMask{frame->numberRows() - 10, false};
        for (std::size_t i = 0; i < 10; ++i) {
            newRowsMask.extend(true);
        }

        regression->predict();

        auto probabilities = impl.retrainTreeSelectionProbabilities(
            *frame, allRowsMask, impl.trainedModel());

        BOOST_REQUIRE_EQUAL(impl.trainedModel().size(), probabilities.size());
        BOOST_TEST_REQUIRE(*std::max_element(probabilities.begin(),
                                             probabilities.end()) >= 0.0);
        BOOST_REQUIRE_CLOSE(
            1.0, std::accumulate(probabilities.begin(), probabilities.end(), 0.0), 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(testSearchTree) {

    // Check that the result of upperBound is identical to std::upper_bound on some
    // edge cases and random data.

    using TFloatVec = std::vector<maths::common::CFloatStorage>;

    TSizeVec size;
    TDoubleVec set;
    TDoubleVec probes;
    TDoubleVec extraProbes;
    TFloatVec fset;

    LOG_DEBUG(<< "Empty");
    {
        CSearchTree tree{{}};
        BOOST_REQUIRE_EQUAL(0, tree.upperBound(0.0F));
        BOOST_REQUIRE_EQUAL(0, tree.upperBound(-1000.0F));
        BOOST_REQUIRE_EQUAL(0, tree.upperBound(10000.0F));
    }
    LOG_DEBUG(<< "Before start");
    {
        CSearchTree tree{{0.0}};
        BOOST_REQUIRE_EQUAL(1, tree.upperBound(0.0));
        BOOST_REQUIRE_EQUAL(0, tree.upperBound(-1.0));
    }
    LOG_DEBUG(<< "Duplicate");
    for (std::size_t i = 0; i < 5; ++i) {
        fset.push_back(0.0);
        CSearchTree tree{fset};
        BOOST_REQUIRE_EQUAL(0, tree.upperBound(-1.0));
        BOOST_REQUIRE_EQUAL(i + 1, tree.upperBound(0.0));
        BOOST_REQUIRE_EQUAL(i + 1, tree.upperBound(1.0));
    }
    LOG_DEBUG(<< "Infinity");
    {
        CSearchTree tree{{0.0}};
        BOOST_REQUIRE_EQUAL(1, tree.upperBound(std::numeric_limits<float>::infinity()));
    }

    LOG_DEBUG(<< "Small");

    for (std::size_t i = 0; i < 5; ++i) {
        set.push_back(static_cast<double>(i + 1));
        CSearchTree tree{{set.begin(), set.end()}};
        for (std::size_t j = 0; j <= i; ++j) {
            BOOST_REQUIRE_EQUAL(j, tree.upperBound(static_cast<double>(j) + 0.5));
        }
    }

    // Random small sets.
    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < 10000; ++i) {
        if (i % 500 == 0) {
            LOG_DEBUG(<< static_cast<double>(i) / 100.0 << "%");
        }
        rng.generateUniformSamples(1, 100, 1, size);
        rng.generateUniformSamples(-1000.0, 1000.0, size[0], set);
        rng.generateUniformSamples(-2000.0, 2000.0, 10, extraProbes);

        fset.assign(set.begin(), set.end());
        std::sort(fset.begin(), fset.end());
        probes.clear();
        probes.insert(probes.end(), set.begin(), set.end());
        probes.insert(probes.end(), extraProbes.begin(), extraProbes.end());

        CSearchTree tree({fset.begin(), fset.end()});
        for (auto probe : probes) {
            maths::common::CFloatStorage fprobe{probe};
            auto expected = std::upper_bound(fset.begin(), fset.end(), fprobe) -
                            fset.begin();
            BOOST_REQUIRE_EQUAL(expected, tree.upperBound(fprobe));
        }
    }

    LOG_DEBUG(<< "Large");

    rng.generateUniformSamples(-1000.0, 1000.0, 100000, set);
    rng.generateUniformSamples(-2000.0, 2000.0, 1000, extraProbes);

    fset.assign(set.begin(), set.end());
    std::sort(fset.begin(), fset.end());

    probes.clear();
    probes.insert(probes.end(), set.begin(), set.end());
    probes.insert(probes.end(), extraProbes.begin(), extraProbes.end());

    CSearchTree tree({fset.begin(), fset.end()});
    for (auto probe : probes) {
        maths::common::CFloatStorage fprobe{probe};
        auto expected = std::upper_bound(fset.begin(), fset.end(), fprobe) - fset.begin();
        BOOST_REQUIRE_EQUAL(expected, tree.upperBound(probe));
    }
}

BOOST_AUTO_TEST_SUITE_END()
