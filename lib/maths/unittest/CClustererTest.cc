/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <maths/CClusterer.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <set>
#include <vector>

BOOST_AUTO_TEST_SUITE(CClustererTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testIndexGenerator) {
    // We test the invariants that:
    //   1) It never produces duplicate index.
    //   2) The highest index in the set is less than the
    //      maximum set size to date.

    using TSizeVec = std::vector<std::size_t>;
    using TDoubleVec = std::vector<double>;
    using TSizeSet = std::set<std::size_t, std::greater<std::size_t>>;
    using TSizeSetItr = TSizeSet::iterator;

    test::CRandomNumbers rng;

    std::size_t numberOperations = 100000;

    TDoubleVec tmp;
    rng.generateUniformSamples(0.0, 1.0, numberOperations, tmp);
    TSizeVec nexts;
    nexts.reserve(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        nexts.push_back(static_cast<std::size_t>(tmp[i] + 0.5));
    }

    maths::CClusterer1d::CIndexGenerator generator;

    TSizeSet indices;
    std::size_t maxSetSize = 0;

    for (std::size_t i = 0; i < numberOperations; ++i) {
        if (i % 1000 == 0) {
            LOG_DEBUG(<< "maxSetSize = " << maxSetSize);
            LOG_DEBUG(<< "indices = " << core::CContainerPrinter::print(indices));
        }
        if (nexts[i] == 1) {
            BOOST_TEST_REQUIRE(indices.insert(generator.next()).second);
            maxSetSize = std::max(maxSetSize, indices.size());
            if (*indices.begin() >= maxSetSize) {
                LOG_DEBUG(<< "index = " << *indices.begin() << ", maxSetSize = " << maxSetSize);
            }
            BOOST_TEST_REQUIRE(*indices.begin() < maxSetSize);
        } else if (!indices.empty()) {
            TDoubleVec indexToErase;
            double max = static_cast<double>(indices.size()) - 1e-3;
            rng.generateUniformSamples(0.0, max, 1u, indexToErase);
            std::size_t index = static_cast<std::size_t>(indexToErase[0]);
            TSizeSetItr itr = indices.begin();
            std::advance(itr, index);
            generator.recycle(*itr);
            indices.erase(itr);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
