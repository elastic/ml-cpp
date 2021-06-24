/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCategoricalTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CCategoricalToolsTest)

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

using namespace ml;

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelyMultinomialSample,
                     *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testProbabilityOfLessLikelyCategoryCount,
                     *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testExpectedDistinctCategories) {
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    static const std::size_t nTrials = 4000;

    test::CRandomNumbers rng;

    {
        double categories[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        {
            double probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};

            TMeanVarAccumulator expectedDistinctCategories;
            for (std::size_t i = 0; i < nTrials; ++i) {
                TDoubleVec samples;
                rng.generateMultinomialSamples(
                    TDoubleVec(std::begin(categories), std::end(categories)),
                    TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                    boost::size(probabilities), samples);
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()),
                              samples.end());
                expectedDistinctCategories.add(static_cast<double>(samples.size()));
            }
            LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
            LOG_DEBUG(<< "expectedDistinctCategories = "
                      << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                      << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                   static_cast<double>(nTrials))
                      << ")");

            double distinctCategories;
            maths::CCategoricalTools::expectedDistinctCategories(
                TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                static_cast<double>(boost::size(probabilities)), distinctCategories);
            LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                2.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                static_cast<double>(nTrials)));
        }

        {
            double probabilities[] = {0.1, 0.3, 0.4, 0.1, 0.1};

            TMeanVarAccumulator expectedDistinctCategories;
            for (std::size_t i = 0; i < nTrials; ++i) {
                TDoubleVec samples;
                rng.generateMultinomialSamples(
                    TDoubleVec(std::begin(categories), std::end(categories)),
                    TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                    boost::size(probabilities), samples);
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()),
                              samples.end());
                expectedDistinctCategories.add(static_cast<double>(samples.size()));
            }
            LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
            LOG_DEBUG(<< "expectedDistinctCategories = "
                      << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                      << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                   static_cast<double>(nTrials))
                      << ")");

            double distinctCategories;
            maths::CCategoricalTools::expectedDistinctCategories(
                TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                static_cast<double>(boost::size(probabilities)), distinctCategories);
            LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                2.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                static_cast<double>(nTrials)));
        }
        {
            double probabilities[] = {0.35, 0.1, 0.25, 0.25, 0.05};

            TMeanVarAccumulator expectedDistinctCategories;
            for (std::size_t i = 0; i < nTrials; ++i) {
                TDoubleVec samples;
                rng.generateMultinomialSamples(
                    TDoubleVec(std::begin(categories), std::end(categories)),
                    TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                    boost::size(probabilities), samples);
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()),
                              samples.end());
                expectedDistinctCategories.add(static_cast<double>(samples.size()));
            }
            LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
            LOG_DEBUG(<< "expectedDistinctCategories = "
                      << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                      << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                   static_cast<double>(nTrials))
                      << ")");

            double distinctCategories;
            maths::CCategoricalTools::expectedDistinctCategories(
                TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                static_cast<double>(boost::size(probabilities)), distinctCategories);
            LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                2.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                static_cast<double>(nTrials)));
        }
    }

    {
        double categories[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                               6.0, 7.0, 8.0, 9.0, 10.0};
        {
            double probabilities[] = {0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1};

            TMeanVarAccumulator expectedDistinctCategories;
            for (std::size_t i = 0; i < nTrials; ++i) {
                TDoubleVec samples;
                rng.generateMultinomialSamples(
                    TDoubleVec(std::begin(categories), std::end(categories)),
                    TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                    boost::size(probabilities), samples);
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()),
                              samples.end());
                expectedDistinctCategories.add(static_cast<double>(samples.size()));
            }
            LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
            LOG_DEBUG(<< "expectedDistinctCategories = "
                      << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                      << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                   static_cast<double>(nTrials))
                      << ")");

            double distinctCategories;
            maths::CCategoricalTools::expectedDistinctCategories(
                TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                static_cast<double>(boost::size(probabilities)), distinctCategories);
            LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                2.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                static_cast<double>(nTrials)));
        }

        {
            double probabilities[] = {0.05, 0.3,  0.4,  0.02, 0.03,
                                      0.05, 0.05, 0.01, 0.02, 0.07};

            TMeanVarAccumulator expectedDistinctCategories;
            for (std::size_t i = 0; i < nTrials; ++i) {
                TDoubleVec samples;
                rng.generateMultinomialSamples(
                    TDoubleVec(std::begin(categories), std::end(categories)),
                    TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                    boost::size(probabilities), samples);
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()),
                              samples.end());
                expectedDistinctCategories.add(static_cast<double>(samples.size()));
            }
            LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
            LOG_DEBUG(<< "expectedDistinctCategories = "
                      << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                      << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                   static_cast<double>(nTrials))
                      << ")");

            double distinctCategories;
            maths::CCategoricalTools::expectedDistinctCategories(
                TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                static_cast<double>(boost::size(probabilities)), distinctCategories);
            LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                2.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                static_cast<double>(nTrials)));
        }
        {
            double probabilities[] = {0.05, 0.1, 0.15, 0.15, 0.05,
                                      0.05, 0.1, 0.15, 0.15, 0.05};

            TMeanVarAccumulator expectedDistinctCategories;
            for (std::size_t i = 0; i < nTrials; ++i) {
                TDoubleVec samples;
                rng.generateMultinomialSamples(
                    TDoubleVec(std::begin(categories), std::end(categories)),
                    TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                    boost::size(probabilities), samples);
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()),
                              samples.end());
                expectedDistinctCategories.add(static_cast<double>(samples.size()));
            }
            LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
            LOG_DEBUG(<< "expectedDistinctCategories = "
                      << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                      << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                   static_cast<double>(nTrials))
                      << ")");

            double distinctCategories;
            maths::CCategoricalTools::expectedDistinctCategories(
                TDoubleVec(std::begin(probabilities), std::end(probabilities)),
                static_cast<double>(boost::size(probabilities)), distinctCategories);
            LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                2.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                static_cast<double>(nTrials)));
        }
    }
    {
        TDoubleVec categories;
        for (std::size_t i = 1; i < 101; ++i) {
            categories.push_back(static_cast<double>(i));
        }

        {
            TDoubleVec concentrations;
            concentrations.resize(100u, 1.0);

            TDoubleVecVec probabilities;
            rng.generateDirichletSamples(concentrations, 50u, probabilities);
            for (std::size_t i = 0; i < 50; ++i) {
                TMeanVarAccumulator expectedDistinctCategories;
                for (std::size_t j = 0; j < nTrials; ++j) {
                    TDoubleVec samples;
                    rng.generateMultinomialSamples(categories, probabilities[i],
                                                   categories.size(), samples);
                    std::sort(samples.begin(), samples.end());
                    samples.erase(std::unique(samples.begin(), samples.end()),
                                  samples.end());
                    expectedDistinctCategories.add(static_cast<double>(samples.size()));
                }
                LOG_DEBUG(<< "expectedDistinctCategories = "
                          << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                          << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                       static_cast<double>(nTrials))
                          << ")");

                double distinctCategories;
                maths::CCategoricalTools::expectedDistinctCategories(
                    probabilities[i], static_cast<double>(categories.size()), distinctCategories);
                LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                    3.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                    static_cast<double>(nTrials)));
            }
        }
        {
            TDoubleVec concentrations;
            concentrations.resize(100u, 1.0);
            concentrations[20] = 70.0;
            concentrations[35] = 20.0;
            concentrations[36] = 25.0;

            TDoubleVecVec probabilities;
            rng.generateDirichletSamples(concentrations, 50u, probabilities);
            for (std::size_t i = 0; i < 50; ++i) {
                TMeanVarAccumulator expectedDistinctCategories;
                for (std::size_t j = 0; j < nTrials; ++j) {
                    TDoubleVec samples;
                    rng.generateMultinomialSamples(categories, probabilities[i],
                                                   categories.size(), samples);
                    std::sort(samples.begin(), samples.end());
                    samples.erase(std::unique(samples.begin(), samples.end()),
                                  samples.end());
                    expectedDistinctCategories.add(static_cast<double>(samples.size()));
                }
                LOG_DEBUG(<< "expectedDistinctCategories = "
                          << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                          << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                       static_cast<double>(nTrials))
                          << ")");

                double distinctCategories;
                maths::CCategoricalTools::expectedDistinctCategories(
                    probabilities[i], static_cast<double>(categories.size()), distinctCategories);
                LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                    3.0 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                    static_cast<double>(nTrials)));
            }
        }
        {
            TDoubleVec concentrations;
            concentrations.resize(100u, 2.0);
            concentrations[20] = 30.0;
            concentrations[35] = 20.0;
            concentrations[36] = 25.0;
            concentrations[37] = 25.0;
            concentrations[38] = 25.0;
            concentrations[46] = 45.0;
            concentrations[65] = 25.0;

            TDoubleVecVec probabilities;
            rng.generateDirichletSamples(concentrations, 50u, probabilities);
            for (std::size_t i = 0; i < 50; ++i) {
                TMeanVarAccumulator expectedDistinctCategories;
                for (std::size_t j = 0; j < nTrials; ++j) {
                    TDoubleVec samples;
                    rng.generateMultinomialSamples(categories, probabilities[i],
                                                   categories.size(), samples);
                    std::sort(samples.begin(), samples.end());
                    samples.erase(std::unique(samples.begin(), samples.end()),
                                  samples.end());
                    expectedDistinctCategories.add(static_cast<double>(samples.size()));
                }
                LOG_DEBUG(<< "expectedDistinctCategories = "
                          << maths::CBasicStatistics::mean(expectedDistinctCategories) << " (deviation = "
                          << std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                       static_cast<double>(nTrials))
                          << ")");

                double distinctCategories;
                maths::CCategoricalTools::expectedDistinctCategories(
                    probabilities[i], static_cast<double>(categories.size()), distinctCategories);
                LOG_DEBUG(<< "distinctCategories = " << distinctCategories);

                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    maths::CBasicStatistics::mean(expectedDistinctCategories), distinctCategories,
                    2.5 * std::sqrt(maths::CBasicStatistics::variance(expectedDistinctCategories) /
                                    static_cast<double>(nTrials)));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testLogBinomialProbability) {
    // Test the calculation matches the boost::binomial_distribution.

    double n[] = {10, 100, 10000};
    double p[] = {0.1, 0.5, 0.9};

    for (std::size_t i = 0; i < boost::size(n); ++i) {
        for (std::size_t j = 0; j < boost::size(p); ++j) {
            LOG_DEBUG(<< "n = " << n[i] << ", p = " << p[j]);

            boost::math::binomial_distribution<> binomial(n[i], p[j]);
            double median = boost::math::median(binomial);
            for (std::size_t f = 1; f < 10; ++f) {
                double f_ = static_cast<double>(f) / 10.0;
                double m = std::floor(f_ * median);
                double pdf = boost::math::pdf(binomial, m);
                double logpdf;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    maths::CCategoricalTools::logBinomialProbability(
                                        static_cast<std::size_t>(n[i]), p[j],
                                        static_cast<std::size_t>(m), logpdf));
                LOG_DEBUG(<< "f(" << m << "), expected = " << pdf
                          << ", actual = " << std::exp(logpdf));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, std::exp(logpdf), 1e-6 * pdf);
            }
            for (std::size_t f = 1; f < 10; ++f) {
                double f_ = static_cast<double>(f) / 10.0;
                double m = median + std::floor(f_ * (n[i] - median));
                double pdf = boost::math::pdf(binomial, m);
                double logpdf;
                BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                    maths::CCategoricalTools::logBinomialProbability(
                                        static_cast<std::size_t>(n[i]), p[j],
                                        static_cast<std::size_t>(m), logpdf));
                LOG_DEBUG(<< "f(" << m << "), expected = " << pdf
                          << ", actual = " << std::exp(logpdf));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, std::exp(logpdf), 1e-6 * pdf);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testLogMultinomialProbability) {
    // Test:
    //   1) The two category case matches the binomial.
    //   2) The marginal matches the binomial.

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Test two categories ***");
    {
        double n[] = {10, 100, 10000};
        double p[] = {0.1, 0.5, 0.9};

        for (std::size_t i = 0; i < boost::size(n); ++i) {
            for (std::size_t j = 0; j < boost::size(p); ++j) {
                LOG_DEBUG(<< "n = " << n[i] << ", p = " << p[j]);

                boost::math::binomial_distribution<> binomial(n[i], p[j]);
                double median = boost::math::median(binomial);
                for (std::size_t f = 1; f < 10; ++f) {
                    double f_ = static_cast<double>(f) / 10.0;
                    double m = std::floor(f_ * median);
                    double pdf = boost::math::pdf(binomial, m);
                    double logpdf;
                    TDoubleVec pi;
                    pi.push_back(p[j]);
                    pi.push_back(1.0 - p[j]);
                    TSizeVec ni;
                    ni.push_back(static_cast<std::size_t>(m));
                    ni.push_back(static_cast<std::size_t>(n[i] - m));
                    BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                        maths::CCategoricalTools::logMultinomialProbability(
                                            pi, ni, logpdf));
                    LOG_DEBUG(<< "f(" << m << "), expected = " << pdf
                              << ", actual = " << std::exp(logpdf));
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, std::exp(logpdf), 1e-6 * pdf);
                }
                for (std::size_t f = 1; f < 10; ++f) {
                    double f_ = static_cast<double>(f) / 10.0;
                    double m = median + std::floor(f_ * (n[i] - median));
                    double pdf = boost::math::pdf(binomial, m);
                    double logpdf;
                    TDoubleVec pi;
                    pi.push_back(p[j]);
                    pi.push_back(1.0 - p[j]);
                    TSizeVec ni;
                    ni.push_back(static_cast<std::size_t>(m));
                    ni.push_back(static_cast<std::size_t>(n[i] - m));
                    BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors,
                                        maths::CCategoricalTools::logMultinomialProbability(
                                            pi, ni, logpdf));
                    LOG_DEBUG(<< "f(" << m << "), expected = " << pdf
                              << ", actual = " << std::exp(logpdf));
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, std::exp(logpdf), 1e-6 * pdf);
                }
            }
        }
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** test marginal ***");
    {
        TDoubleVec pi;
        pi.push_back(0.1);
        pi.push_back(0.3);
        pi.push_back(0.6);

        std::size_t n = 10;
        for (std::size_t m = 0; m <= n; ++m) {
            double marginal = 0.0;
            for (std::size_t i = 0; i <= n - m; ++i) {
                double logpdf;
                TSizeVec ni;
                ni.push_back(m);
                ni.push_back(i);
                ni.push_back(n - m - i);
                BOOST_REQUIRE_EQUAL(
                    maths_t::E_FpNoErrors,
                    maths::CCategoricalTools::logMultinomialProbability(pi, ni, logpdf));
                marginal += std::exp(logpdf);
            }

            boost::math::binomial_distribution<> binomial(static_cast<double>(n), pi[0]);
            double pdf = boost::math::pdf(binomial, static_cast<double>(m));
            LOG_DEBUG(<< "f = " << pdf << ", marginal = " << marginal);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(pdf, marginal, 1e-6 * pdf);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
