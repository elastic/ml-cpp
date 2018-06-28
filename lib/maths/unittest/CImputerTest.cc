/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CImputerTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CImputer.h>
#include <maths/CStatisticalTests.h>

#include <test/CRandomNumbers.h>

#include <cstddef>
#include <utility>
#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleVecVecVec = std::vector<TDoubleVecVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TSizeVecVecVec = std::vector<TSizeVecVec>;
using TIntDoublePr = std::pair<std::ptrdiff_t, double>;
using TIntDoublePrVec = std::vector<TIntDoublePr>;
using TIntDoublePrVecVec = std::vector<TIntDoublePrVec>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

void generateWithMissingFraction(test::CRandomNumbers& rng,
                                 std::size_t a,
                                 std::size_t b,
                                 double fraction,
                                 const TDoubleVecVec& attributes,
                                 TSizeVecVec& present,
                                 TSizeVecVec& missing,
                                 TIntDoublePrVecVec& values) {
    for (std::size_t i = a; i < b; ++i) {
        TDoubleVec mi;
        rng.generateUniformSamples(0.0, 1.0, attributes.size(), mi);

        TIntDoublePrVec value;
        for (std::size_t j = 0u; j < attributes.size(); ++j) {
            if (mi[j] > fraction) {
                TSizeVec index;
                rng.generateUniformSamples(0, attributes[j].size(), 1, index);
                value.emplace_back(static_cast<std::ptrdiff_t>(j),
                                   attributes[j][index[0]]);
                present[j].push_back(static_cast<std::ptrdiff_t>(i));
            } else {
                missing[j].push_back(static_cast<std::ptrdiff_t>(i));
            }
        }
        values[i] = std::move(value);
    }
}
}

void CImputerTest::testRandom(void) {
    LOG_DEBUG("+----------------------------+");
    LOG_DEBUG("|  CImputerTest::testRandom  |");
    LOG_DEBUG("+----------------------------+");

    // Test that the missing values are generated according to the law
    // which describes the values present for each attribute.

    test::CRandomNumbers rng;

    TDoubleVecVec attributes(6);
    TDoubleVec means{10.0, 4.0, 8.8, 1.0, 3.2, 12.5};
    TDoubleVec variances{4.0, 2.0, 1.5, 1.2, 4.1, 5.5};
    std::size_t n{100};
    for (std::size_t i = 0u; i < attributes.size(); ++i) {
        rng.generateNormalSamples(means[i], variances[i], n, attributes[i]);
    }

    std::size_t dimension{attributes.size()};

    TSizeVecVec present(dimension);
    TSizeVecVec missing(dimension);
    TIntDoublePrVecVec values(n);
    generateWithMissingFraction(rng, 0, n, 0.15, attributes, present, missing, values);

    maths::CImputer::TDenseVectorVec complete;
    maths::CImputer imputer;
    imputer.impute(maths::CImputer::E_NoFiltering, maths::CImputer::E_Random,
                   static_cast<std::ptrdiff_t>(dimension), values, complete);

    TDoubleVecVec donors(dimension);
    TDoubleVecVec imputed(dimension);
    for (std::size_t i = 0u; i < dimension; ++i) {
        for (auto j : present[i]) {
            donors[i].push_back(complete[j](i));
        }
    }
    for (std::size_t i = 0u; i < dimension; ++i) {
        for (auto j : missing[i]) {
            imputed[i].push_back(complete[j](i));
        }
    }

    TMeanAccumulator meanSignificance;

    for (std::size_t i = 0u; i < dimension; ++i) {
        // Check we've sampled one of the original values.
        std::sort(donors[i].begin(), donors[i].end());
        for (auto inputed_ : imputed[i]) {
            CPPUNIT_ASSERT(std::binary_search(donors[i].begin(), donors[i].end(), inputed_));
        }

        // Get the distribution significance.
        double significance{maths::CStatisticalTests::twoSampleKS(donors[i], imputed[i])};
        LOG_DEBUG("significance = " << significance);
        CPPUNIT_ASSERT(significance > 0.05);
        meanSignificance.add(significance);
    }

    LOG_DEBUG("mean significance = " << maths::CBasicStatistics::mean(meanSignificance));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanSignificance) > 0.4);
}

void CImputerTest::testNearestNeighbourPlain(void) {
    LOG_DEBUG("+-------------------------------------------+");
    LOG_DEBUG("|  CImputerTest::testNearestNeighbourPlain  |");
    LOG_DEBUG("+-------------------------------------------+");

    // Test we impute from nearest neighbours if on hand crafted data set.
    {
        TIntDoublePrVecVec values{
            TIntDoublePrVec{{0, 1.0}, {1, 3.0}, {2, -1.0}, {3, 4.0}},
            TIntDoublePrVec{{1, 2.0}, {2, -1.1}, {3, 5.0}},
            TIntDoublePrVec{{0, 5.0}, {1, 8.0}, {2, -7.1}, {3, 1.0}},
            TIntDoublePrVec{{0, 4.0}, {2, -6.1}, {3, 0.0}},
            TIntDoublePrVec{{0, -1.0}, {1, 3.3}, {2, 3.1}, {3, 9.0}},
            TIntDoublePrVec{{0, -1.0}, {1, 3.0}, {3, 9.0}},
            TIntDoublePrVec{{0, 3.0}, {1, 5.0}, {2, -1.1}}};

        maths::CImputer imputer{maths::CImputer::FILTER_FRACTION,
                                maths::CImputer::FRACTION_OF_VALUES_IN_BAG,
                                maths::CImputer::NUMBER_BAGS, 1}; // Number nearest neighbours.

        maths::CImputer::TDenseVectorVec complete;
        imputer.impute(maths::CImputer::E_NoFiltering,
                       maths::CImputer::E_NNPlain, 4, values, complete);

        CPPUNIT_ASSERT_EQUAL(1.0, complete[1](0));
        CPPUNIT_ASSERT_EQUAL(8.0, complete[3](1));
        CPPUNIT_ASSERT_EQUAL(3.1, complete[5](2));
        CPPUNIT_ASSERT_EQUAL(4.0, complete[6](3));
    }

    // Create some clustered data and check that samples for missing values
    // come from the appropriate cluster.
    {
        test::CRandomNumbers rng;

        TDoubleVecVec means{TDoubleVec{12.0, 9.0, 6.8}, TDoubleVec{1.0, 3.2, 12.5}};
        TDoubleVecVec variances{TDoubleVec{4.0, 2.0, 1.5}, TDoubleVec{1.2, 4.1, 5.5}};
        std::size_t n{100};

        std::size_t dimension{3};

        TSizeVecVecVec present(means.size(), TSizeVecVec(dimension));
        TSizeVecVecVec missing(means.size(), TSizeVecVec(dimension));
        TIntDoublePrVecVec values(means.size() * n);

        for (std::size_t i = 0u; i < means.size(); ++i) {
            TDoubleVecVec attributes(dimension);
            for (std::size_t j = 0u; j < dimension; ++j) {
                rng.generateNormalSamples(means[i][j], variances[i][j], n, attributes[j]);
            }
            generateWithMissingFraction(rng, i * n, (i + 1) * n, 0.15, attributes,
                                        present[i], missing[i], values);
        }

        TIntDoublePrVecVec values_(values);

        maths::CImputer::TDenseVectorVec complete;
        maths::CImputer imputer;
        imputer.impute(maths::CImputer::E_NoFiltering, maths::CImputer::E_NNPlain,
                       static_cast<std::ptrdiff_t>(dimension), values, complete);

        TMeanAccumulator meanError;
        for (std::size_t i = 0u; i < missing.size(); ++i) {
            for (std::size_t j = 0u; j < dimension; ++j) {
                double mean{means[i][j]};
                double sd{::sqrt(variances[i][j])};
                for (auto k : missing[i][j]) {
                    double x{complete[k](static_cast<std::ptrdiff_t>(j))};
                    double error{::fabs(x - mean) / sd};
                    LOG_DEBUG("error = " << error);
                    CPPUNIT_ASSERT(error < 2.5);
                    meanError.add(error);
                }
            }
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.6);
    }
}

void CImputerTest::testNearestNeighbourBaggedSamples(void) {
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CImputerTest::testNearestNeighbourBaggedSamples  |");
    LOG_DEBUG("+---------------------------------------------------+");
}

void CImputerTest::testNearestNeighbourBaggedAttributes(void) {
    LOG_DEBUG("+------------------------------------------------------+");
    LOG_DEBUG("|  CImputerTest::testNearestNeighbourBaggedAttributes  |");
    LOG_DEBUG("+------------------------------------------------------+");
}

void CImputerTest::testNearestNeighbourRandom(void) {
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CImputerTest::testNearestNeighbourRandom  |");
    LOG_DEBUG("+--------------------------------------------+");
}

CppUnit::Test* CImputerTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CImputerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CImputerTest>(
        "CImputerTest::testRandom", &CImputerTest::testRandom));
    suiteOfTests->addTest(new CppUnit::TestCaller<CImputerTest>(
        "CImputerTest::testNearestNeighbourPlain", &CImputerTest::testNearestNeighbourPlain));
    suiteOfTests->addTest(new CppUnit::TestCaller<CImputerTest>(
        "CImputerTest::testNearestNeighbourBaggedSamples",
        &CImputerTest::testNearestNeighbourBaggedSamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CImputerTest>(
        "CImputerTest::testNearestNeighbourBaggedAttributes",
        &CImputerTest::testNearestNeighbourBaggedAttributes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CImputerTest>(
        "CImputerTest::testNearestNeighbourRandom", &CImputerTest::testNearestNeighbourRandom));

    return suiteOfTests;
}
