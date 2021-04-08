/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CMultinomialConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CMultinomialConjugateTest)

using namespace ml;
using namespace handy_typedefs;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TUIntVec = std::vector<unsigned int>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using CMultinomialConjugate = CPriorTestInterfaceMixin<maths::CMultinomialConjugate>;

BOOST_AUTO_TEST_CASE(testMultipleUpdate) {
    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    const double rawCategories[] = {-1.2, 5.1, 2.0, 18.0, 10.3};
    const double rawProbabilities[] = {0.17, 0.13, 0.35, 0.3, 0.05};
    const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
    const TDoubleVec probabilities(std::begin(rawProbabilities), std::end(rawProbabilities));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateMultinomialSamples(categories, probabilities, 100, samples);

    CMultinomialConjugate filter1(CMultinomialConjugate::nonInformativePrior(5u));
    CMultinomialConjugate filter2(filter1);

    for (std::size_t j = 0; j < samples.size(); ++j) {
        filter1.addSamples(TDouble1Vec(1, samples[j]));
    }
    filter2.addSamples(samples);

    using TEqual = maths::CEqualWithTolerance<double>;
    TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5);
    BOOST_TEST_REQUIRE(filter1.equalTolerance(filter2, equal));
}

BOOST_AUTO_TEST_CASE(testPropagation) {
    // Test that propagation doesn't affect the expected values
    // of probabilities.

    const double rawCategories[] = {0.0, 1.1, 2.0};
    const double rawProbabilities[] = {0.27, 0.13, 0.6};
    const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
    const TDoubleVec probabilities(std::begin(rawProbabilities), std::end(rawProbabilities));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateMultinomialSamples(categories, probabilities, 500, samples);

    CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(5u));

    for (std::size_t i = 0; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
    }

    TDoubleVec expectedProbabilities = filter.probabilities();

    filter.propagateForwardsByTime(5.0);

    TDoubleVec propagatedExpectedProbabilities = filter.probabilities();

    LOG_DEBUG(<< "expectedProbabilities = " << core::CContainerPrinter::print(expectedProbabilities)
              << ", propagatedExpectedProbabilities = "
              << core::CContainerPrinter::print(propagatedExpectedProbabilities));

    using TEqual = maths::CEqualWithTolerance<double>;
    TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-12);
    BOOST_TEST_REQUIRE(std::equal(expectedProbabilities.begin(),
                                  expectedProbabilities.end(),
                                  propagatedExpectedProbabilities.begin(), equal));
}

BOOST_AUTO_TEST_CASE(testProbabilityEstimation) {
    // We are going to test that we correctly estimate the distribution
    // for the probabilities of a multinomial process by checking that
    // the true probabilities lie in various confidence intervals the
    // correct percentage of the times.

    const double rawCategories[] = {0.0, 1.1, 2.0, 5.0, 12.0, 15.0};
    const double rawProbabilities[] = {0.1, 0.15, 0.12, 0.31, 0.03, 0.29};
    const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
    const TDoubleVec probabilities(std::begin(rawProbabilities), std::end(rawProbabilities));

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nTests = 5000;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    for (size_t i = 0; i < boost::size(decayRates); ++i) {
        test::CRandomNumbers rng;

        TUIntVec errors[] = {TUIntVec(6, 0), TUIntVec(6, 0), TUIntVec(6, 0),
                             TUIntVec(6, 0), TUIntVec(6, 0), TUIntVec(6, 0),
                             TUIntVec(6, 0), TUIntVec(6, 0)};

        for (unsigned int test = 0; test < nTests; ++test) {
            TDoubleVec samples;
            rng.generateMultinomialSamples(categories, probabilities, 500, samples);

            CMultinomialConjugate filter(
                CMultinomialConjugate::nonInformativePrior(6, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                filter.addSamples(TDouble1Vec(1, samples[j]));
                filter.propagateForwardsByTime(1.0);
            }

            for (size_t j = 0; j < boost::size(testIntervals); ++j) {
                TDoubleDoublePrVec confidenceIntervals =
                    filter.confidenceIntervalProbabilities(testIntervals[j]);
                BOOST_REQUIRE_EQUAL(confidenceIntervals.size(), probabilities.size());

                for (std::size_t k = 0; k < probabilities.size(); ++k) {
                    if (probabilities[k] < confidenceIntervals[k].first ||
                        probabilities[k] > confidenceIntervals[k].second) {
                        ++errors[j][k];
                    }
                }
            }
        }

        for (size_t j = 0; j < boost::size(testIntervals); ++j) {
            TDoubleVec intervals;
            intervals.reserve(errors[j].size());
            for (std::size_t k = 0; k < errors[j].size(); ++k) {
                intervals.push_back(100.0 * errors[j][k] / static_cast<double>(nTests));
            }
            LOG_DEBUG(<< "interval = " << core::CContainerPrinter::print(intervals)
                      << ", expectedInterval = " << (100.0 - testIntervals[j]));

            // If the decay rate is zero the intervals should be accurate.
            // Otherwise, they should be an upper bound.
            double meanError = 0.0;
            for (std::size_t k = 0; k < intervals.size(); ++k) {
                if (decayRates[i] == 0.0) {
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        intervals[k], 100.0 - testIntervals[j],
                        std::min(5.0, 0.4 * (100.0 - testIntervals[j])));
                    meanError += std::fabs(intervals[k] - (100.0 - testIntervals[j]));
                } else {
                    BOOST_TEST_REQUIRE(intervals[k] <= (100.0 - testIntervals[j]));
                }
            }
            meanError /= static_cast<double>(intervals.size());
            LOG_DEBUG(<< "meanError = " << meanError);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                0.0, meanError, std::min(2.0, 0.2 * (100.0 - testIntervals[j])));
        }
    }
}

BOOST_AUTO_TEST_CASE(testMarginalLikelihood) {
    {
        // For a single sample the log likelihood of the i'th category is
        // equal to log(p(i)) where p(i) is the i'th category expected
        // probability.

        test::CRandomNumbers rng;

        const double rawCategories[] = {0.0, 1.0, 2.0};
        const double rawProbabilities[] = {0.15, 0.5, 0.35};
        const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
        const TDoubleVec probabilities(std::begin(rawProbabilities),
                                       std::end(rawProbabilities));

        TDoubleVec samples;
        rng.generateMultinomialSamples(categories, probabilities, 50, samples);

        const double decayRates[] = {0.0, 0.001, 0.01};

        for (size_t i = 0; i < boost::size(decayRates); ++i) {
            LOG_DEBUG(<< "**** Decay rate = " << decayRates[i] << " ****");

            CMultinomialConjugate filter(
                CMultinomialConjugate::nonInformativePrior(3, decayRates[i]));

            for (std::size_t j = 0; j < samples.size(); ++j) {
                TDouble1Vec sample(1, samples[j]);

                filter.addSamples(sample);
                filter.propagateForwardsByTime(1.0);

                double logp;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    filter.jointLogMarginalLikelihood(sample, logp));

                const TDoubleVec& filterCategories = filter.categories();
                std::size_t k = std::lower_bound(filterCategories.begin(),
                                                 filterCategories.end(), samples[j]) -
                                filterCategories.begin();
                TDoubleVec filterProbabilities(filter.probabilities());
                BOOST_TEST_REQUIRE(k < filterProbabilities.size());
                double p = filterProbabilities[k];

                LOG_DEBUG(<< "sample = " << samples[j] << ", expected likelihood = " << p
                          << ", likelihood = " << std::exp(logp));

                BOOST_REQUIRE_CLOSE_ABSOLUTE(p, std::exp(logp), 1e-12);
            }
        }
    }

    {
        // The idea of this is to test the joint likelihood accurately predicts
        // the relative frequencies of combinations of categories.
        //
        // In particular, if we initialize the prior with the correct
        // concentrations for a particular multinomial distribution (using the
        // limit corresponding to a large number of updates, i.e. large sum).
        // Then the law of large numbers applied to the indicator functions
        // of the events {c(i)} x {c(i)} x...x {c(i)}, where {c(i)} are the
        // sets of categories and "x" denotes the outer product, implies that
        // the relative frequencies of these events should tend to the joint
        // marginal likelihoods for each event.

        test::CRandomNumbers rng;

        const double rawCategories[] = {0.0, 1.0, 2.0};
        const double rawProbabilities[] = {0.1, 0.6, 0.3};
        const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
        const TDoubleVec probabilities(std::begin(rawProbabilities),
                                       std::end(rawProbabilities));

        // Compute the outer products of size 2 and 3.
        TDoubleVecVec o2, o3;
        for (std::size_t i = 0; i < categories.size(); ++i) {
            for (std::size_t j = i; j < categories.size(); ++j) {
                o2.push_back(TDoubleVec());
                o2.back().push_back(categories[i]);
                o2.back().push_back(categories[j]);
                for (std::size_t k = j; k < categories.size(); ++k) {
                    o3.push_back(TDoubleVec());
                    o3.back().push_back(categories[i]);
                    o3.back().push_back(categories[j]);
                    o3.back().push_back(categories[k]);
                }
            }
        }
        LOG_DEBUG(<< "o2 = " << core::CContainerPrinter::print(o2));
        LOG_DEBUG(<< "o3 = " << core::CContainerPrinter::print(o3));

        double rawConcentrations[] = {1000.0, 6000.0, 3000.0};
        TDoubleVec concentrations(std::begin(rawConcentrations), std::end(rawConcentrations));

        CMultinomialConjugate filter(maths::CMultinomialConjugate(3, categories, concentrations));

        const unsigned int nTests = 100000;

        {
            // Compute the likelihoods of the various 2-category combinations.

            TDoubleVec p2;
            for (std::size_t i = 0; i < o2.size(); ++i) {
                double p;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    filter.jointLogMarginalLikelihood(o2[i], p));
                p = std::exp(p);
                p2.push_back(p);
                LOG_DEBUG(<< "categories = " << core::CContainerPrinter::print(o2[i])
                          << ", p = " << p);
            }
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.0, std::accumulate(p2.begin(), p2.end(), 0.0), 1e-10);

            TDoubleVec frequencies(o2.size(), 0.0);

            TDoubleVec samples;
            rng.generateMultinomialSamples(categories, probabilities, 2 * nTests, samples);

            for (unsigned int test = 0; test < nTests; ++test) {
                TDoubleVec sample;
                sample.push_back(samples[2 * test]);
                sample.push_back(samples[2 * test + 1]);
                std::sort(sample.begin(), sample.end());

                std::size_t i = std::lower_bound(o2.begin(), o2.end(), sample) -
                                o2.begin();
                BOOST_TEST_REQUIRE(i < o2.size());
                frequencies[i] += 1.0;
            }

            for (std::size_t i = 0; i < o2.size(); ++i) {
                double p = frequencies[i] / static_cast<double>(nTests);

                LOG_DEBUG(<< "category = " << core::CContainerPrinter::print(o2[i])
                          << ", p = " << p << ", expected p = " << p2[i]);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(p, p2[i], 0.05 * std::max(p, p2[i]));
            }
        }
        {
            // Compute the likelihoods of the various 3-category combinations.

            TDoubleVec p3;
            for (std::size_t i = 0; i < o3.size(); ++i) {
                double p;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    filter.jointLogMarginalLikelihood(o3[i], p));
                p = std::exp(p);
                p3.push_back(p);
                LOG_DEBUG(<< "categories = " << core::CContainerPrinter::print(o3[i])
                          << ", p = " << p);
            }
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.0, std::accumulate(p3.begin(), p3.end(), 0.0), 1e-10);

            TDoubleVec frequencies(o3.size(), 0.0);

            TDoubleVec samples;
            rng.generateMultinomialSamples(categories, probabilities, 3 * nTests, samples);

            for (unsigned int test = 0; test < nTests; ++test) {
                TDoubleVec sample;
                sample.push_back(samples[3 * test]);
                sample.push_back(samples[3 * test + 1]);
                sample.push_back(samples[3 * test + 2]);
                std::sort(sample.begin(), sample.end());

                std::size_t i = std::lower_bound(o3.begin(), o3.end(), sample) -
                                o3.begin();
                BOOST_TEST_REQUIRE(i < o3.size());
                frequencies[i] += 1.0;
            }

            for (std::size_t i = 0; i < o3.size(); ++i) {
                double p = frequencies[i] / static_cast<double>(nTests);

                LOG_DEBUG(<< "category = " << core::CContainerPrinter::print(o3[i])
                          << ", p = " << p << ", expected p = " << p3[i]);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(p, p3[i], 0.05 * std::max(p, p3[i]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleMarginalLikelihood) {
    // Test that we sample categories in proportion to their marginal
    // probabilities. We test two cases:
    //   1) The probabilities exactly divide the requested number of samples n.
    //      In this case number of samples category i should equal n * P(i).
    //   2) The probabilities don't exactly divide the requested number of
    //      samples n. In this case we should get back n samples and the
    //      number of samples of each category should be floor(n * P(i)) or
    //      ceil(n * P(i)) and the sum over |n * P(i) - n(i)| should be as
    //      small as possible.

    {
        const double rawCategories[] = {1.1, 1.2, 2.1, 2.2};
        const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));

        // The probabilities {P(i)} are proportional to the number of samples
        // of each category we add to the filter.

        CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(4u));

        filter.addSamples(TDouble1Vec(30, categories[0])); // P = 0.3
        filter.addSamples(TDouble1Vec(10, categories[1])); // P = 0.1
        filter.addSamples(TDouble1Vec(20, categories[2])); // P = 0.2
        filter.addSamples(TDouble1Vec(40, categories[3])); // P = 0.4

        TDouble1Vec samples;
        filter.sampleMarginalLikelihood(10, samples);
        std::sort(samples.begin(), samples.end());

        LOG_DEBUG(<< "samples = " << core::CContainerPrinter::print(samples));

        BOOST_REQUIRE_EQUAL(std::string("[1.1, 1.1, 1.1, 1.2, 2.1, 2.1, 2.2, 2.2, 2.2, 2.2]"),
                            core::CContainerPrinter::print(samples));
    }

    {
        const double rawCategories[] = {1.1, 1.2, 2.1, 2.2, 3.2, 5.1};
        const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));

        CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(6u));

        filter.addSamples(TDouble1Vec(11, categories[0])); // P = 0.11
        filter.addSamples(TDouble1Vec(19, categories[1])); // P = 0.19
        filter.addSamples(TDouble1Vec(23, categories[2])); // P = 0.23
        filter.addSamples(TDouble1Vec(37, categories[3])); // P = 0.37
        filter.addSamples(TDouble1Vec(5, categories[4]));  // P = 0.05
        filter.addSamples(TDouble1Vec(5, categories[5]));  // P = 0.05

        TDouble1Vec samples;
        filter.sampleMarginalLikelihood(10, samples);
        std::sort(samples.begin(), samples.end());

        LOG_DEBUG(<< "samples = " << core::CContainerPrinter::print(samples));

        BOOST_REQUIRE_EQUAL(std::string("[1.1, 1.2, 1.2, 2.1, 2.1, 2.2, 2.2, 2.2, 2.2, 5.1]"),
                            core::CContainerPrinter::print(samples));
    }

    {
        const double rawCategories[] = {1.1, 1.2, 2.1, 2.2, 3.2, 5.1};
        const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));

        CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(6u));

        filter.addSamples(TDouble1Vec(11, categories[0])); // P = 0.11
        filter.addSamples(TDouble1Vec(18, categories[1])); // P = 0.18
        filter.addSamples(TDouble1Vec(23, categories[2])); // P = 0.23
        filter.addSamples(TDouble1Vec(37, categories[3])); // P = 0.37
        filter.addSamples(TDouble1Vec(6, categories[4]));  // P = 0.06
        filter.addSamples(TDouble1Vec(5, categories[5]));  // P = 0.05

        TDouble1Vec samples;
        filter.sampleMarginalLikelihood(10, samples);
        std::sort(samples.begin(), samples.end());

        LOG_DEBUG(<< "samples = " << core::CContainerPrinter::print(samples));

        BOOST_REQUIRE_EQUAL(std::string("[1.1, 1.2, 1.2, 2.1, 2.1, 2.2, 2.2, 2.2, 2.2, 3.2]"),
                            core::CContainerPrinter::print(samples));
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelySamples) {
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TDoubleSizePrVec = std::vector<TDoubleSizePr>;

    // We test the definition of the various sided calculations:
    //   - one sided below: P(R) = P(y <= x)
    //   - two sided:       P(R) = P({y : f(y) <= f(x))
    //   - one sided above: P(R) = P(y >= x)

    LOG_DEBUG(<< "**** one sided below ****");
    {
        // TODO
    }

    LOG_DEBUG(<< "**** two sided ****");
    {
        const double rawCategories[] = {1.1, 1.2, 2.1, 2.2, 3.2, 5.1};
        const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));

        {
            CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(6u));

            // Large update limit.
            filter.addSamples({categories[0]}, {maths_t::countWeight(10000.0)}); // P = 0.10
            filter.addSamples({categories[1]}, {maths_t::countWeight(12000.0)}); // P = 0.12
            filter.addSamples({categories[2]}, {maths_t::countWeight(29000.0)}); // P = 0.29
            filter.addSamples({categories[3]}, {maths_t::countWeight(39000.0)}); // P = 0.39
            filter.addSamples({categories[4]}, {maths_t::countWeight(4000.0)}); // P = 0.04
            filter.addSamples({categories[5]}, {maths_t::countWeight(6000.0)}); // P = 0.06

            // We expect the following probabilities for each category:
            //   P(1.1) = 0.20
            //   P(1.2) = 0.32
            //   P(2.1) = 0.61
            //   P(2.2) = 1.00
            //   P(3.2) = 0.04
            //   P(5.1) = 0.10
            double expectedProbabilities[] = {0.20, 0.32, 0.61,
                                              1.0,  0.04, 0.10};

            for (size_t i = 0; i < boost::size(categories); ++i) {
                double lowerBound, upperBound;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      TDouble1Vec(1, categories[i]),
                                                      lowerBound, upperBound);

                LOG_DEBUG(<< "category = " << categories[i] << ", lower bound = " << lowerBound
                          << ", upper bound = " << upperBound
                          << ", expected probability = " << expectedProbabilities[i]);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(lowerBound, expectedProbabilities[i], 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(upperBound, expectedProbabilities[i], 1e-10);
            }
        }

        {
            CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(6u));

            // Large update limit.
            filter.addSamples({categories[0]}, {maths_t::countWeight(11000.0)}); // P = 0.11
            filter.addSamples({categories[1]}, {maths_t::countWeight(11000.0)}); // P = 0.11
            filter.addSamples({categories[2]}, {maths_t::countWeight(29000.0)}); // P = 0.29
            filter.addSamples({categories[3]}, {maths_t::countWeight(39000.0)}); // P = 0.39
            filter.addSamples({categories[4]}, {maths_t::countWeight(5000.0)}); // P = 0.05
            filter.addSamples({categories[5]}, {maths_t::countWeight(5000.0)}); // P = 0.05

            // We expect the following probabilities for each category:
            //   P(1.1) = P(1.2) = 0.32
            //   P(2.1) = 0.61
            //   P(2.2) = 1.00
            //   P(3.2) = P(5.1) = 0.10
            double expectedProbabilities[] = {0.32, 0.32, 0.61, 1.0, 0.1, 0.1};

            for (size_t i = 0; i < boost::size(categories); ++i) {
                double lowerBound, upperBound;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      TDouble1Vec(1, categories[i]),
                                                      lowerBound, upperBound);

                LOG_DEBUG(<< "category = " << categories[i] << ", lower bound = " << lowerBound
                          << ", upper bound = " << upperBound
                          << ", expected probability = " << expectedProbabilities[i]);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(lowerBound, expectedProbabilities[i], 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(upperBound, expectedProbabilities[i], 1e-10);
            }
        }

        {
            CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(6u));

            // Large update limit.
            filter.addSamples({categories[0]}, {maths_t::countWeight(15000.0)}); // P = 0.15
            filter.addSamples({categories[1]}, {maths_t::countWeight(15000.0)}); // P = 0.15
            filter.addSamples({categories[2]}, {maths_t::countWeight(30000.0)}); // P = 0.30
            filter.addSamples({categories[3]}, {maths_t::countWeight(30000.0)}); // P = 0.30
            filter.addSamples({categories[4]}, {maths_t::countWeight(5000.0)}); // P = 0.05
            filter.addSamples({categories[5]}, {maths_t::countWeight(5000.0)}); // P = 0.05

            // We expect the following probabilities for each category:
            //   P(1.1) = P(1.2) = 0.40
            //   P(2.1) = P(2.2) = 1.00
            //   P(3.2) = P(5.1) = 0.10
            double expectedProbabilities[] = {0.4, 0.4, 1.0, 1.0, 0.1, 0.1};

            for (size_t i = 0; i < boost::size(categories); ++i) {
                double lowerBound, upperBound;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      TDouble1Vec(1, categories[i]),
                                                      lowerBound, upperBound);

                LOG_DEBUG(<< "category = " << categories[i] << ", lower bound = " << lowerBound
                          << ", upper bound = " << upperBound
                          << ", expected probability = " << expectedProbabilities[i]);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(lowerBound, expectedProbabilities[i], 1e-10);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(upperBound, expectedProbabilities[i], 1e-10);
            }
        }

        {
            CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(4u));
            filter.addSamples(TDouble1Vec(25, categories[0])); // P = 0.25
            filter.addSamples(TDouble1Vec(25, categories[1])); // P = 0.25
            filter.addSamples(TDouble1Vec(25, categories[2])); // P = 0.25
            filter.addSamples(TDouble1Vec(25, categories[3])); // P = 0.25

            // We expect the following probabilities for each category:
            //   P(1.1) = P(1.2) = P(2.1) = P(2.2) = 1.0
            double expectedProbabilities[] = {0.95, 0.95, 0.95, 0.95};

            for (size_t i = 0; i < boost::size(expectedProbabilities); ++i) {
                double lowerBound, upperBound;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      TDouble1Vec(1, categories[i]),
                                                      lowerBound, upperBound);

                LOG_DEBUG(<< "category = " << categories[i] << ", lower bound = " << lowerBound
                          << ", upper bound = " << upperBound
                          << ", expected probability = " << expectedProbabilities[i]);

                BOOST_TEST_REQUIRE(lowerBound > expectedProbabilities[i]);
                BOOST_TEST_REQUIRE(upperBound > expectedProbabilities[i]);
            }
        }
        {
            using TDoubleDoubleVecMap = std::map<double, TDoubleVec>;
            using TDoubleDoubleVecMapCItr = TDoubleDoubleVecMap::const_iterator;
            using TDoubleVecDoubleMap = std::map<TDoubleVec, double>;
            using TDoubleVecDoubleMapCItr = TDoubleVecDoubleMap::const_iterator;

            double categoryProbabilities[] = {0.10, 0.12, 0.29,
                                              0.39, 0.04, 0.06};
            TDoubleDoubleVecMap categoryPairProbabilities;
            for (size_t i = 0; i < boost::size(categories); ++i) {
                for (size_t j = i; j < boost::size(categories); ++j) {
                    double p = (i != j ? 2.0 : 1.0) * categoryProbabilities[i] *
                               categoryProbabilities[j];

                    TDoubleVec& categoryPair =
                        categoryPairProbabilities
                            .insert(TDoubleDoubleVecMap::value_type(p, TDoubleVec()))
                            .first->second;
                    categoryPair.push_back(categories[i]);
                    categoryPair.push_back(categories[j]);
                }
            }
            LOG_DEBUG(<< "category pair probabilities = "
                      << core::CContainerPrinter::print(categoryPairProbabilities));

            double pc = 0.0;
            TDoubleVecDoubleMap trueProbabilities;
            for (TDoubleDoubleVecMapCItr itr = categoryPairProbabilities.begin();
                 itr != categoryPairProbabilities.end(); ++itr) {
                pc += itr->first * static_cast<double>(itr->second.size() / 2u);
                for (std::size_t i = 0; i < itr->second.size(); i += 2u) {
                    TDoubleVec categoryPair;
                    categoryPair.push_back(itr->second[i]);
                    categoryPair.push_back(itr->second[i + 1u]);
                    trueProbabilities.insert(
                        TDoubleVecDoubleMap::value_type(categoryPair, pc));
                }
            }
            LOG_DEBUG(<< "true probabilities = "
                      << core::CContainerPrinter::print(trueProbabilities));

            CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(6u));

            // Large update limit.
            filter.addSamples({categories[0]}, {maths_t::countWeight(10000.0)}); // P = 0.10
            filter.addSamples({categories[1]}, {maths_t::countWeight(12000.0)}); // P = 0.12
            filter.addSamples({categories[2]}, {maths_t::countWeight(29000.0)}); // P = 0.29
            filter.addSamples({categories[3]}, {maths_t::countWeight(39000.0)}); // P = 0.39
            filter.addSamples({categories[4]}, {maths_t::countWeight(4000.0)}); // P = 0.04
            filter.addSamples({categories[5]}, {maths_t::countWeight(6000.0)}); // P = 0.06

            double expectedProbabilities[] = {0.2, 0.32, 0.61, 1.0, 0.04, 0.1};

            for (TDoubleVecDoubleMapCItr itr = trueProbabilities.begin();
                 itr != trueProbabilities.end(); ++itr) {
                TDoubleVec categoryPair;
                categoryPair.push_back(itr->first[0]);
                categoryPair.push_back(itr->first[1]);
                double lowerBound, upperBound;
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, categoryPair, lowerBound, upperBound);
                BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
                double probability = lowerBound;

                maths::CJointProbabilityOfLessLikelySamples expectedProbabilityCalculator;
                {
                    ptrdiff_t i = std::lower_bound(categories.begin(),
                                                   categories.end(), itr->first[0]) -
                                  categories.begin();
                    ptrdiff_t j = std::lower_bound(categories.begin(),
                                                   categories.end(), itr->first[1]) -
                                  categories.begin();
                    expectedProbabilityCalculator.add(expectedProbabilities[i]);
                    expectedProbabilityCalculator.add(expectedProbabilities[j]);
                }
                double expectedProbability;
                BOOST_TEST_REQUIRE(expectedProbabilityCalculator.calculate(expectedProbability));

                LOG_DEBUG(<< "category pair = " << core::CContainerPrinter::print(itr->first)
                          << ", probability = " << probability
                          << ", expected probability = " << expectedProbability
                          << ", true probability = " << itr->second);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedProbability, probability, 1e-10);
            }
        }
    }
    {
        // Test the function to compute all category probabilities.
        const double rawCategories[] = {
            1.1,  1.2,  2.1,  2.2,  3.2,  5.1,  5.5,  6.0,  6.2,  6.6,
            7.8,  8.0,  9.0,  9.9,  10.0, 10.1, 10.2, 12.0, 12.1, 12.8,
            13.1, 13.7, 15.2, 17.1, 17.5, 17.9, 18.2, 19.6, 20.0, 20.2};
        const double rawProbabilities[] = {
            0.02, 0.05, 0.01,  0.2,   0.001, 0.03,  0.02,  0.005, 0.1,  0.03,
            0.04, 0.01, 0.001, 0.006, 0.02,  0.05,  0.001, 0.001, 0.01, 0.01,
            0.2,  0.01, 0.02,  0.07,  0.01,  0.002, 0.01,  0.02,  0.03, 0.013};

        BOOST_REQUIRE_EQUAL(boost::size(rawCategories), boost::size(rawProbabilities));

        test::CRandomNumbers rng;
        const std::size_t numberSamples = 10000;

        // Generate samples from the Dirichlet prior.
        TDoubleVecVec dirichletSamples(boost::size(rawProbabilities));
        for (size_t i = 0; i < boost::size(rawProbabilities); ++i) {
            TDoubleVec& samples = dirichletSamples[i];
            rng.generateGammaSamples(rawProbabilities[i] * 100.0, 1.0, numberSamples, samples);
        }
        for (std::size_t i = 0; i < numberSamples; ++i) {
            double n = 0.0;
            for (std::size_t j = 0; j < dirichletSamples.size(); ++j) {
                n += dirichletSamples[j][i];
            }
            for (std::size_t j = 0; j < dirichletSamples.size(); ++j) {
                dirichletSamples[j][i] /= n;
            }
        }

        // Compute the expected probabilities w.r.t. the Dirichlet prior.
        TDoubleVec expectedProbabilities(boost::size(rawCategories), 0.0);
        for (std::size_t i = 0; i < numberSamples; ++i) {
            TDoubleSizePrVec probabilities;
            probabilities.reserve(dirichletSamples.size() + 1);
            for (std::size_t j = 0; j < dirichletSamples.size(); ++j) {
                probabilities.push_back(TDoubleSizePr(dirichletSamples[j][i], j));
            }
            std::sort(probabilities.begin(), probabilities.end());
            for (std::size_t j = 1; j < probabilities.size(); ++j) {
                probabilities[j].first += probabilities[j - 1].first;
            }
            probabilities.push_back(TDoubleSizePr(1.0, probabilities.size()));
            for (std::size_t j = 0; j < probabilities.size() - 1; ++j) {
                expectedProbabilities[probabilities[j].second] +=
                    probabilities[j + 1].first;
            }
        }
        for (std::size_t i = 0; i < expectedProbabilities.size(); ++i) {
            expectedProbabilities[i] /= static_cast<double>(numberSamples);
        }
        LOG_DEBUG(<< "expectedProbabilities = "
                  << core::CContainerPrinter::print(expectedProbabilities));

        TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
        CMultinomialConjugate filter(
            CMultinomialConjugate::nonInformativePrior(categories.size()));
        for (std::size_t i = 0; i < categories.size(); ++i) {
            filter.addSamples({categories[i]},
                              {maths_t::countWeight(rawProbabilities[i] * 100.0)});
        }

        TDoubleVec lowerBounds, upperBounds;
        filter.probabilitiesOfLessLikelyCategories(maths_t::E_TwoSided,
                                                   lowerBounds, upperBounds);
        LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(lowerBounds));
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(lowerBounds),
                            core::CContainerPrinter::print(upperBounds));

        double totalError = 0.0;
        for (std::size_t i = 0; i < lowerBounds.size(); ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedProbabilities[i], lowerBounds[i], 0.1);
            totalError += std::fabs(lowerBounds[i] - expectedProbabilities[i]);
        }
        LOG_DEBUG(<< "totalError = " << totalError);
        BOOST_TEST_REQUIRE(totalError < 0.7);

        for (std::size_t i = 0; i < categories.size(); ++i) {
            double lowerBound, upperBound;
            BOOST_TEST_REQUIRE(filter.probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, TDouble1Vec(1, categories[i]), lowerBound, upperBound));
            BOOST_REQUIRE_EQUAL(lowerBound, upperBound);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(lowerBounds[i], lowerBound, 1e-10);
        }
    }

    LOG_DEBUG(<< "**** one sided above ****");
    {
        // TODO
    }
}

BOOST_AUTO_TEST_CASE(testAnomalyScore, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testRemoveCategories) {
    double rawCategories[] = {1.0, 3.0, 15.0, 17.0, 19.0, 20.0};
    double rawConcentrations[] = {1.0, 2.0, 1.5, 12.0, 10.0, 2.0};

    TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
    TDoubleVec concentrationParameters(std::begin(rawConcentrations),
                                       std::end(rawConcentrations));

    {
        CMultinomialConjugate prior(
            maths::CMultinomialConjugate(100, categories, concentrationParameters));

        TDoubleVec categoriesToRemove;
        categoriesToRemove.push_back(3.0);
        categoriesToRemove.push_back(17.0);
        categoriesToRemove.push_back(19.0);
        prior.removeCategories(categoriesToRemove);

        TDoubleVec expectedCategories;
        expectedCategories.push_back(1.0);
        expectedCategories.push_back(15.0);
        expectedCategories.push_back(20.0);
        TDoubleVec expectedConcentrations;
        expectedConcentrations.push_back(1.0);
        expectedConcentrations.push_back(1.5);
        expectedConcentrations.push_back(2.0);
        CMultinomialConjugate expectedPrior(maths::CMultinomialConjugate(
            100, expectedCategories, expectedConcentrations));

        LOG_DEBUG(<< "expectedCategories = "
                  << core::CContainerPrinter::print(expectedCategories));
        LOG_DEBUG(<< "expectedConcentrations = "
                  << core::CContainerPrinter::print(expectedConcentrations));

        BOOST_REQUIRE_EQUAL(expectedPrior.checksum(), prior.checksum());
    }
    {
        CMultinomialConjugate prior(
            maths::CMultinomialConjugate(90, categories, concentrationParameters));

        TDoubleVec categoriesToRemove;
        categoriesToRemove.push_back(1.0);
        categoriesToRemove.push_back(15.0);
        categoriesToRemove.push_back(20.0);
        prior.removeCategories(categoriesToRemove);

        TDoubleVec expectedCategories;
        expectedCategories.push_back(3.0);
        expectedCategories.push_back(17.0);
        expectedCategories.push_back(19.0);
        TDoubleVec expectedConcentrations;
        expectedConcentrations.push_back(2.0);
        expectedConcentrations.push_back(12.0);
        expectedConcentrations.push_back(10.0);
        CMultinomialConjugate expectedPrior(maths::CMultinomialConjugate(
            90, expectedCategories, expectedConcentrations));

        LOG_DEBUG(<< "expectedCategories = "
                  << core::CContainerPrinter::print(expectedCategories));
        LOG_DEBUG(<< "expectedConcentrations = "
                  << core::CContainerPrinter::print(expectedConcentrations));

        BOOST_REQUIRE_EQUAL(expectedPrior.checksum(), prior.checksum());
    }
    {
        CMultinomialConjugate prior(
            maths::CMultinomialConjugate(10, categories, concentrationParameters));

        prior.removeCategories(categories);

        CMultinomialConjugate expectedPrior = CMultinomialConjugate::nonInformativePrior(10);

        BOOST_REQUIRE_EQUAL(expectedPrior.checksum(), prior.checksum());
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    const double rawCategories[] = {-1.0, 5.0, 2.1, 78.0, 15.3};
    const double rawProbabilities[] = {0.1, 0.2, 0.35, 0.3, 0.05};
    const TDoubleVec categories(std::begin(rawCategories), std::end(rawCategories));
    const TDoubleVec probabilities(std::begin(rawProbabilities), std::end(rawProbabilities));

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateMultinomialSamples(categories, probabilities, 100, samples);
    maths::CMultinomialConjugate origFilter(CMultinomialConjugate::nonInformativePrior(5));
    for (std::size_t i = 0; i < samples.size(); ++i) {
        origFilter.addSamples({samples[i]}, maths_t::CUnitWeights::SINGLE_UNIT);
    }
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Multinomial conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(
        maths_t::E_ContinuousData, decayRate + 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    maths::CMultinomialConjugate restoredFilter(params, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());

    BOOST_REQUIRE_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testOverflow, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testConcentration) {
    CMultinomialConjugate filter(CMultinomialConjugate::nonInformativePrior(5u));
    for (std::size_t i = 1; i <= 5; ++i) {
        filter.addSamples(TDouble1Vec(i, static_cast<double>(i)));
    }

    double concentration;
    for (std::size_t i = 1; i <= 5; ++i) {
        double category = static_cast<double>(i);
        filter.concentration(category, concentration);
        BOOST_REQUIRE_EQUAL(category, concentration);
    }
}

BOOST_AUTO_TEST_SUITE_END()
