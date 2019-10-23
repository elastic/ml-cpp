/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>

#include <maths/CMultivariateConstantPrior.h>

#include "TestUtils.h"

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <limits>

BOOST_AUTO_TEST_SUITE(CMultivariateConstantPriorTest)

using namespace ml;
using namespace handy_typedefs;

BOOST_AUTO_TEST_CASE(testAddSamples) {
    // Test error cases.

    maths::CMultivariateConstantPrior filter(2);

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

    maths::CMultivariateConstantPrior filter(2);

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
    BOOST_REQUIRE_EQUAL(boost::numeric::bounds<double>::lowest(), likelihood);

    filter.addSamples(
        TDouble10Vec1Vec(2, TDouble10Vec(std::begin(constant), std::end(constant))),
        maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));

    BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                      filter.jointLogMarginalLikelihood(
                          {TDouble10Vec(std::begin(constant), std::end(constant))},
                          maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(std::log(boost::numeric::bounds<double>::highest()), likelihood);

    BOOST_REQUIRE_EQUAL(maths_t::E_FpOverflowed,
                      filter.jointLogMarginalLikelihood(
                          {TDouble10Vec(std::begin(different), std::end(different))},
                          maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), likelihood));
    BOOST_REQUIRE_EQUAL(boost::numeric::bounds<double>::lowest(), likelihood);
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihoodMean) {
    // Check that the marginal likelihood mean is 0 for non informative,
    // otherwise equal to the constant.

    maths::CMultivariateConstantPrior filter(3);

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

    maths::CMultivariateConstantPrior filter(4);

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

    maths::CMultivariateConstantPrior filter(4);

    TDouble10Vec10Vec covariance = filter.marginalLikelihoodCovariance();
    BOOST_REQUIRE_EQUAL(std::size_t(4), covariance.size());
    for (std::size_t i = 0u; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(std::size_t(4), covariance[i].size());
        BOOST_REQUIRE_EQUAL(boost::numeric::bounds<double>::highest(), covariance[i][i]);
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
    BOOST_REQUIRE_EQUAL(std::size_t(4), covariance.size());
    for (std::size_t i = 0u; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(std::size_t(4), covariance[i].size());
        for (std::size_t j = 0; j < 4; ++j) {
            BOOST_REQUIRE_EQUAL(0.0, covariance[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihood) {
    // Check we get zero samples for non-informative and sample of the
    // constant otherwise.

    maths::CMultivariateConstantPrior filter(2);

    TDouble10Vec1Vec samples;
    filter.sampleMarginalLikelihood(3, samples);
    BOOST_REQUIRE_EQUAL(std::size_t(0), samples.size());

    double constant[] = {1.2, 4.1};

    filter.addSamples({TDouble10Vec(std::begin(constant), std::end(constant))},
                      maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));

    filter.sampleMarginalLikelihood(4, samples);
    BOOST_REQUIRE_EQUAL(std::size_t(4), samples.size());
    for (std::size_t i = 0u; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(std::string("[1.2, 4.1]"),
                          core::CContainerPrinter::print(samples[i]));
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySamples) {
    // Check we get one for non-informative and the constant and zero
    // otherwise.

    maths::CMultivariateConstantPrior filter(2);

    double samples_[][2] = {{1.3, 1.4}, {1.1, 1.6}, {1.0, 5.4}};
    TDouble10Vec1Vec samples[] = {
        TDouble10Vec1Vec(1, TDouble10Vec(std::begin(samples_[0]), std::end(samples_[0]))),
        TDouble10Vec1Vec(1, TDouble10Vec(std::begin(samples_[1]), std::end(samples_[1]))),
        TDouble10Vec1Vec(1, TDouble10Vec(std::begin(samples_[2]), std::end(samples_[2])))};
    for (std::size_t i = 0u; i < boost::size(samples); ++i) {
        double lb, ub;
        maths::CMultivariateConstantPrior::TTail10Vec tail;
        filter.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, samples[i],
            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
        BOOST_REQUIRE_EQUAL(1.0, lb);
        BOOST_REQUIRE_EQUAL(1.0, ub);
        LOG_DEBUG(<< "tail = " << core::CContainerPrinter::print(tail));
        BOOST_REQUIRE_EQUAL(std::string("[0, 0]"), core::CContainerPrinter::print(tail));
    }

    filter.addSamples(samples[0], maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
    BOOST_TEST_REQUIRE(!filter.isNonInformative());

    std::string expectedTails[] = {"[0, 0]", "[1, 2]", "[1, 2]"};
    for (std::size_t i = 0u; i < boost::size(samples); ++i) {
        double lb, ub;
        maths::CMultivariateConstantPrior::TTail10Vec tail;
        filter.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, samples[i],
            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
        BOOST_REQUIRE_EQUAL(i == 0 ? 1.0 : 0.0, lb);
        BOOST_REQUIRE_EQUAL(i == 0 ? 1.0 : 0.0, ub);
        LOG_DEBUG(<< "tail = " << core::CContainerPrinter::print(tail));
        BOOST_REQUIRE_EQUAL(expectedTails[i], core::CContainerPrinter::print(tail));
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check persistence is idempotent.

    LOG_DEBUG(<< "*** Non-informative ***");
    {
        maths::CMultivariateConstantPrior origFilter(3);
        uint64_t checksum = origFilter.checksum();

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origFilter.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "Constant XML representation:\n" << origXml);

        // Restore the XML into a new filter
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CMultivariateConstantPrior restoredFilter(3, traverser);

        LOG_DEBUG(<< "orig checksum = " << checksum
                  << " restored checksum = " << restoredFilter.checksum());
        BOOST_REQUIRE_EQUAL(checksum, restoredFilter.checksum());

        // The XML representation of the new filter should be the same as the original
        std::string newXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            restoredFilter.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }

    LOG_DEBUG(<< "*** Constant ***");
    {
        double constant[] = {1.2, 4.1, 1.0 / 3.0};

        maths::CMultivariateConstantPrior origFilter(
            3, TDouble10Vec(std::begin(constant), std::end(constant)));
        uint64_t checksum = origFilter.checksum();

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origFilter.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "Constant XML representation:\n" << origXml);

        // Restore the XML into a new filter
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CMultivariateConstantPrior restoredFilter(3, traverser);

        LOG_DEBUG(<< "orig checksum = " << checksum
                  << " restored checksum = " << restoredFilter.checksum());
        BOOST_REQUIRE_EQUAL(checksum, restoredFilter.checksum());

        // The XML representation of the new filter should be the same as the original
        std::string newXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            restoredFilter.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }
}

BOOST_AUTO_TEST_SUITE_END()
