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
#include <core/CLogger.h>
#include <core/CSmallVector.h>

#include <maths/common/CMultivariateConstantPrior.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>

#include <limits>

BOOST_AUTO_TEST_SUITE(CMultivariateConstantPriorTest)

using namespace ml;
using namespace handy_typedefs;

BOOST_AUTO_TEST_CASE(testAddSamples) {
    // Test error cases.

    maths::common::CMultivariateConstantPrior filter(2);

    double wrongDimension[] = {1.3, 2.1, 7.9};

    filter.addSamples({TDouble10Vec(std::begin(wrongDimension), std::end(wrongDimension))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
    BOOST_TEST_REQUIRE(filter.isNonInformative());

    double nans[] = {1.3, std::numeric_limits<double>::quiet_NaN()};

    filter.addSamples({TDouble10Vec(std::begin(nans), std::end(nans))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
    BOOST_TEST_REQUIRE(filter.isNonInformative());

    double constant[] = {1.4, 1.0};

    filter.addSamples({TDouble10Vec(std::begin(constant), std::end(constant))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
    BOOST_TEST_REQUIRE(!filter.isNonInformative());
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihood) {
    // Check that the marginal likelihood is 0 for non informative, otherwise
    // either 0 or infinity depending on whether the value is equal to the
    // constant or not.

    maths::common::CMultivariateConstantPrior filter(2);

    double constant[] = {1.3, 17.3};
    double different[] = {1.1, 17.3};

    double likelihood;

    BOOST_REQUIRE_EQUAL(maths_t::E_FpFailed,
                        filter.jointLogMarginalLikelihood(
                            {}, maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(
        maths_t::E_FpFailed,
        filter.jointLogMarginalLikelihood(
            TDouble10Vec1Vec(2, TDouble10Vec(std::begin(constant), std::end(constant))),
            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(maths_t::E_FpOverflowed,
                        filter.jointLogMarginalLikelihood(
                            {TDouble10Vec(std::begin(constant), std::end(constant))},
                            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(std::numeric_limits<double>::lowest(), likelihood);

    filter.addSamples(
        TDouble10Vec1Vec(2, TDouble10Vec(std::begin(constant), std::end(constant))),
        maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));

    BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                        filter.jointLogMarginalLikelihood(
                            {TDouble10Vec(std::begin(constant), std::end(constant))},
                            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(std::log(std::numeric_limits<double>::max()), likelihood);

    BOOST_REQUIRE_EQUAL(
        maths_t::E_FpOverflowed,
        filter.jointLogMarginalLikelihood(
            {TDouble10Vec(std::begin(different), std::end(different))},
            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(std::numeric_limits<double>::lowest(), likelihood);
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMean) {
    // Check that the marginal likelihood mean is 0 for non informative,
    // otherwise equal to the constant.

    maths::common::CMultivariateConstantPrior filter(3);

    BOOST_REQUIRE_EQUAL(std::string("[0, 0, 0]"),
                        core::CContainerPrinter::print(filter.marginalLikelihoodMean()));

    double constant[] = {1.2, 6.0, 14.1};
    filter.addSamples({TDouble10Vec(std::begin(constant), std::end(constant))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));

    BOOST_REQUIRE_EQUAL(std::string("[1.2, 6, 14.1]"),
                        core::CContainerPrinter::print(filter.marginalLikelihoodMean()));
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMode) {
    // Check that the marginal likelihood mode is 0 for non informative,
    // otherwise equal to the constant.

    maths::common::CMultivariateConstantPrior filter(4);

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(filter.marginalLikelihoodMean()),
                        core::CContainerPrinter::print(filter.marginalLikelihoodMode(
                            maths_t::CUnitWeights::unit<TDouble10Vec>(4))));

    double constant[] = {1.1, 6.5, 12.3, 14.1};
    filter.addSamples({TDouble10Vec(std::begin(constant), std::end(constant))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(4));

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(filter.marginalLikelihoodMean()),
                        core::CContainerPrinter::print(filter.marginalLikelihoodMode(
                            maths_t::CUnitWeights::unit<TDouble10Vec>(4))));
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodCovariance) {
    // Check that the marginal likelihood mode is infinite diagonal for
    // non informative, otherwise the zero matrix.

    maths::common::CMultivariateConstantPrior filter(4);

    TDouble10Vec10Vec covariance = filter.marginalLikelihoodCovariance();
    BOOST_REQUIRE_EQUAL(4, covariance.size());
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(4, covariance[i].size());
        BOOST_REQUIRE_EQUAL(std::numeric_limits<double>::max(), covariance[i][i]);
        for (std::size_t j = 0; j < i; ++j) {
            BOOST_REQUIRE_EQUAL(0.0, covariance[i][j]);
        }
        for (std::size_t j = i + 1; j < 4; ++j) {
            BOOST_REQUIRE_EQUAL(0.0, covariance[i][j]);
        }
    }

    double constant[] = {1.1, 6.5, 12.3, 14.1};
    filter.addSamples({TDouble10Vec(std::begin(constant), std::end(constant))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));

    covariance = filter.marginalLikelihoodCovariance();
    BOOST_REQUIRE_EQUAL(4, covariance.size());
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(4, covariance[i].size());
        for (std::size_t j = 0; j < 4; ++j) {
            BOOST_REQUIRE_EQUAL(0.0, covariance[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihood) {
    // Check we get zero samples for non-informative and sample of the
    // constant otherwise.

    maths::common::CMultivariateConstantPrior filter(2);

    TDouble10Vec1Vec samples;
    filter.sampleMarginalLikelihood(3, samples);
    BOOST_REQUIRE_EQUAL(0, samples.size());

    double constant[] = {1.2, 4.1};

    filter.addSamples({TDouble10Vec(std::begin(constant), std::end(constant))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));

    filter.sampleMarginalLikelihood(4, samples);
    BOOST_REQUIRE_EQUAL(4, samples.size());
    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(std::string("[1.2, 4.1]"),
                            core::CContainerPrinter::print(samples[i]));
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySamples) {
    // Check we get one for non-informative and the constant and zero
    // otherwise.

    maths::common::CMultivariateConstantPrior filter(2);

    double samples_[][2] = {{1.3, 1.4}, {1.1, 1.6}, {1.0, 5.4}};
    TDouble10Vec1Vec samples[] = {
        TDouble10Vec1Vec(1, TDouble10Vec(std::begin(samples_[0]), std::end(samples_[0]))),
        TDouble10Vec1Vec(1, TDouble10Vec(std::begin(samples_[1]), std::end(samples_[1]))),
        TDouble10Vec1Vec(1, TDouble10Vec(std::begin(samples_[2]), std::end(samples_[2])))};
    for (std::size_t i = 0; i < std::size(samples); ++i) {
        double lb, ub;
        maths::common::CMultivariateConstantPrior::TTail10Vec tail;
        filter.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, samples[i],
            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
        BOOST_REQUIRE_EQUAL(1.0, lb);
        BOOST_REQUIRE_EQUAL(1.0, ub);
        LOG_DEBUG(<< "tail = " << tail);
        BOOST_REQUIRE_EQUAL(std::string("[0, 0]"), core::CContainerPrinter::print(tail));
    }

    filter.addSamples(samples[0], maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
    BOOST_TEST_REQUIRE(!filter.isNonInformative());

    std::string expectedTails[] = {"[0, 0]", "[1, 2]", "[1, 2]"};
    for (std::size_t i = 0; i < std::size(samples); ++i) {
        double lb, ub;
        maths::common::CMultivariateConstantPrior::TTail10Vec tail;
        filter.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, samples[i],
            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
        BOOST_REQUIRE_EQUAL(i == 0 ? 1.0 : 0.0, lb);
        BOOST_REQUIRE_EQUAL(i == 0 ? 1.0 : 0.0, ub);
        LOG_DEBUG(<< "tail = " << tail);
        BOOST_REQUIRE_EQUAL(expectedTails[i], core::CContainerPrinter::print(tail));
    }
}

void testPersistFilter(const maths::common::CMultivariateConstantPrior& origFilter) {
    std::uint64_t checksum = origFilter.checksum();

    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, std::bind_front(&maths::common::CMultivariateConstantPrior::acceptPersistInserter,
                                  &origFilter));
    LOG_DEBUG(<< "Constant JSON representation:\n" << origJson.str());

    // Restore the JSON into a new filter
    std::istringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
    core::CJsonStateRestoreTraverser traverser(origJsonStrm);

    maths::common::CMultivariateConstantPrior restoredFilter(3, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    BOOST_REQUIRE_EQUAL(checksum, restoredFilter.checksum());

    // The JSON representation of the new filter should be the same as the original
    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, std::bind_front(&maths::common::CMultivariateConstantPrior::acceptPersistInserter,
                                 &restoredFilter));
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check persistence is idempotent.

    LOG_DEBUG(<< "*** Non-informative ***");
    {
        maths::common::CMultivariateConstantPrior origFilter(3);

        testPersistFilter(origFilter);
    }

    LOG_DEBUG(<< "*** Constant ***");
    {
        double constant[] = {1.2, 4.1, 1.0 / 3.0};

        maths::common::CMultivariateConstantPrior origFilter(
            3, TDouble10Vec(std::begin(constant), std::end(constant)));

        testPersistFilter(origFilter);
    }
}

BOOST_AUTO_TEST_SUITE_END()
