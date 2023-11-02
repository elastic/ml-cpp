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

#include <core/CLogger.h>
#include <core/CMemoryDef.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/common/CNaiveBayes.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/CRestoreParams.h>
#include <maths/common/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <memory>

namespace {
BOOST_AUTO_TEST_SUITE(CNaiveBayesTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble1VecVec = std::vector<TDouble1Vec>;
using TDoubleSizePr = std::pair<double, std::size_t>;
using TDoubleSizePrVec = std::vector<TDoubleSizePr>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

class CTestFeatureWeight : public maths::common::CNaiveBayesFeatureWeight {
public:
    void add(std::size_t, double) override {}
    double calculate() const override { return 1e-3; }
};

BOOST_AUTO_TEST_CASE(testClassification) {
    // We'll test classification using Gaussian naive Bayes. We
    // test:
    //   - We get the probabilities we expect using if the underlying
    //     classes are consistent with the assumptions,
    //   - Test with missing data

    // We test two features with true density
    //   - x(1) ~ N(0,12)  | C(1),
    //   - x(2) ~ N(10,16) | C(1),
    //   - x(1) ~ N(3,14)  | C(2),
    //   - x(2) ~ N(-5,24) | C(2)

    test::CRandomNumbers rng;

    TDoubleVecVec trainingData(4);
    rng.generateNormalSamples(0.0, 12.0, 100, trainingData[0]);
    rng.generateNormalSamples(10.0, 16.0, 100, trainingData[1]);
    rng.generateNormalSamples(3.0, 14.0, 200, trainingData[2]);
    rng.generateNormalSamples(-5.0, 24.0, 200, trainingData[3]);

    TMeanAccumulator meanMeanError;

    for (auto initialCount : {0.0, 100.0}) {
        maths::common::CNormalMeanPrecConjugate normal{
            maths::common::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};
        maths::common::CNaiveBayes nb{
            maths::common::CNaiveBayesFeatureDensityFromPrior(normal)};

        if (initialCount > 0) {
            nb.initialClassCounts({{initialCount, 1}, {initialCount, 2}});
        }

        for (std::size_t i = 0; i < 100; ++i) {
            nb.addTrainingDataPoint(1, {{trainingData[0][i]}, {trainingData[1][i]}});
        }
        for (std::size_t i = 0; i < 200; ++i) {
            nb.addTrainingDataPoint(2, {{trainingData[2][i]}, {trainingData[3][i]}});
        }

        TMeanVarAccumulator moments[4];
        moments[0].add(trainingData[0]);
        moments[1].add(trainingData[1]);
        moments[2].add(trainingData[2]);
        moments[3].add(trainingData[3]);

        // The training data sizes are 100 and 200 so we expect the
        // class probabilities to be:
        //   - P(1) = (initialCount + 100) / (2*initialCount + 300)
        //   - P(2) = (initialCount + 200) / (2*initialCount + 300)

        auto[probabilities, confidence](nb.highestClassProbabilities(2, {{}, {}}));

        double P1{(initialCount + 100.0) / (2.0 * initialCount + 300.0)};
        double P2{(initialCount + 200.0) / (2.0 * initialCount + 300.0)};

        BOOST_REQUIRE_EQUAL(2, probabilities.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(P1, probabilities[1].first, 1e-5);
        BOOST_REQUIRE_EQUAL(1, probabilities[1].second);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(P2, probabilities[0].first, 1e-5);
        BOOST_REQUIRE_EQUAL(2, probabilities[0].second);

        // If we supply feature values we should approximately
        // get these modulated by the product of the true density
        // ratios for those feature values.

        boost::math::normal class1[]{
            boost::math::normal{
                maths::common::CBasicStatistics::mean(moments[0]),
                std::sqrt(maths::common::CBasicStatistics::variance(moments[0]))},
            boost::math::normal{
                maths::common::CBasicStatistics::mean(moments[1]),
                std::sqrt(maths::common::CBasicStatistics::variance(moments[1]))}};
        boost::math::normal class2[]{
            boost::math::normal{
                maths::common::CBasicStatistics::mean(moments[2]),
                std::sqrt(maths::common::CBasicStatistics::variance(moments[2]))},
            boost::math::normal{
                maths::common::CBasicStatistics::mean(moments[3]),
                std::sqrt(maths::common::CBasicStatistics::variance(moments[3]))}};

        TDoubleVec xtest;
        rng.generateNormalSamples(0.0, 64.0, 40, xtest);

        TMeanAccumulator meanErrors[3];

        for (std::size_t i = 0; i < xtest.size(); i += 2) {
            auto test = [i](double p1, double p2, const TDoubleSizePrVec& p,
                            TMeanAccumulator& meanError) {
                double Z{p1 + p2};
                p1 /= Z;
                p2 /= Z;
                double p1_{p[0].second == 1 ? p[0].first : p[1].first};
                double p2_{p[0].second == 1 ? p[1].first : p[0].first};

                if (i % 10 == 0) {
                    LOG_DEBUG(<< i << ") expected P(1) = " << p1 << ", P(2) = " << p2
                              << " got P(1) = " << p1_ << ", P(2) = " << p2_);
                }

                BOOST_REQUIRE_EQUAL(2, p.size());
                BOOST_REQUIRE_CLOSE_ABSOLUTE(p1, p1_, 0.03);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(p2, p2_, 0.03);
                if (p1 > 0.001) {
                    meanError.add(std::fabs((p1 - p1_) / p1));
                }
                if (p2 > 0.001) {
                    meanError.add(std::fabs((p2 - p2_) / p2));
                }
            };

            // Supply both feature values.
            double p1{P1 * maths::common::CTools::safePdf(class1[0], xtest[i]) *
                      maths::common::CTools::safePdf(class1[1], xtest[i + 1])};
            double p2{P2 * maths::common::CTools::safePdf(class2[0], xtest[i]) *
                      maths::common::CTools::safePdf(class2[1], xtest[i + 1])};
            std::tie(probabilities, confidence) =
                nb.highestClassProbabilities(2, {{xtest[i]}, {xtest[i + 1]}});
            test(p1, p2, probabilities, meanErrors[0]);

            // Miss out the first feature value.
            p1 = P1 * maths::common::CTools::safePdf(class1[1], xtest[i + 1]);
            p2 = P2 * maths::common::CTools::safePdf(class2[1], xtest[i + 1]);
            std::tie(probabilities, confidence) =
                nb.highestClassProbabilities(2, {{}, {xtest[i + 1]}});
            test(p1, p2, probabilities, meanErrors[1]);

            // Miss out the second feature value.
            p1 = P1 * maths::common::CTools::safePdf(class1[0], xtest[i]);
            p2 = P2 * maths::common::CTools::safePdf(class2[0], xtest[i]);
            std::tie(probabilities, confidence) =
                nb.highestClassProbabilities(2, {{xtest[i]}, {}});
            test(p1, p2, probabilities, meanErrors[2]);
        }

        for (std::size_t i = 0; i < 3; ++i) {
            LOG_DEBUG(<< "Mean relative error = "
                      << maths::common::CBasicStatistics::mean(meanErrors[i]));
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanErrors[i]) < 0.05);
            meanMeanError += meanErrors[i];
        }
    }
}

BOOST_AUTO_TEST_CASE(testUninitialized) {
    // Check that the classifier remains uninitialized while the class
    // conditional densities are improper.

    test::CRandomNumbers rng;

    maths::common::CNormalMeanPrecConjugate normal{
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, 0.05)};
    maths::common::CNaiveBayes nb{maths::common::CNaiveBayes{
        maths::common::CNaiveBayesFeatureDensityFromPrior(normal), 0.05}};

    TDoubleVecVec trainingData(2);

    for (std::size_t i = 0; i < 2; ++i) {
        BOOST_REQUIRE_EQUAL(false, nb.initialized());

        double x{static_cast<double>(i)};
        rng.generateNormalSamples(0.02 * x - 14.0, 16.0, 1, trainingData[0]);
        rng.generateNormalSamples(0.02 * x - 14.0, 16.0, 1, trainingData[1]);

        nb.addTrainingDataPoint(1, {{trainingData[0][0]}, {trainingData[1][0]}});
        nb.propagateForwardsByTime(1.0);
    }

    BOOST_REQUIRE_EQUAL(true, nb.initialized());
}

BOOST_AUTO_TEST_CASE(testPropagationByTime) {
    // Make feature distributions drift over time and verify that
    // the classifier adapts.

    test::CRandomNumbers rng;

    maths::common::CNormalMeanPrecConjugate normal{
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, 0.05)};
    maths::common::CNaiveBayes nb[]{
        maths::common::CNaiveBayes{
            maths::common::CNaiveBayesFeatureDensityFromPrior(normal), 0.05},
        maths::common::CNaiveBayes{
            maths::common::CNaiveBayesFeatureDensityFromPrior(normal), 0.05}};

    TDoubleVecVec trainingData(4);
    for (std::size_t i = 0; i < 1000; ++i) {
        double x{static_cast<double>(i)};
        rng.generateNormalSamples(0.02 * x - 14.0, 16.0, 1, trainingData[0]);
        rng.generateNormalSamples(0.02 * x - 14.0, 16.0, 1, trainingData[1]);
        rng.generateNormalSamples(-0.02 * x + 14.0, 16.0, 1, trainingData[2]);
        rng.generateNormalSamples(-0.02 * x + 14.0, 16.0, 1, trainingData[3]);

        nb[0].addTrainingDataPoint(1, {{trainingData[0][0]}, {trainingData[1][0]}});
        nb[0].addTrainingDataPoint(2, {{trainingData[2][0]}, {trainingData[3][0]}});
        nb[0].propagateForwardsByTime(1.0);

        nb[1].addTrainingDataPoint(1, {{trainingData[0][0]}, {trainingData[1][0]}});
        nb[1].addTrainingDataPoint(2, {{trainingData[2][0]}, {trainingData[3][0]}});
    }

    // Check that the value:
    //    - (-10,-10) gets assigned to class 2
    //    - ( 10, 10) gets assigned to class 1
    // for the aged classifier and vice versa.

    {
        TDoubleSizePrVec probabilities[]{
            nb[0].highestClassProbabilities(2, {{-10.0}, {-10.0}}).first,
            nb[1].highestClassProbabilities(2, {{-10.0}, {-10.0}}).first};
        LOG_DEBUG(<< "Aged class probabilities = " << probabilities[0]);
        LOG_DEBUG(<< "Class probabilities = " << probabilities[1]);
        BOOST_REQUIRE_EQUAL(2, probabilities[0][0].second);
        BOOST_TEST_REQUIRE(probabilities[0][0].first > 0.99);
        BOOST_REQUIRE_EQUAL(1, probabilities[1][0].second);
        BOOST_TEST_REQUIRE(probabilities[1][0].first > 0.95);
    }
    {
        TDoubleSizePrVec probabilities[]{
            nb[0].highestClassProbabilities(2, {{10.0}, {10.0}}).first,
            nb[1].highestClassProbabilities(2, {{10.0}, {10.0}}).first};
        LOG_DEBUG(<< "Aged class probabilities = " << probabilities[0]);
        LOG_DEBUG(<< "Class probabilities = " << probabilities[1]);
        BOOST_REQUIRE_EQUAL(1, probabilities[0][0].second);
        BOOST_TEST_REQUIRE(probabilities[0][0].first > 0.99);
        BOOST_REQUIRE_EQUAL(2, probabilities[1][0].second);
        BOOST_TEST_REQUIRE(probabilities[1][0].first > 0.95);
    }
}

BOOST_AUTO_TEST_CASE(testExtrapolation) {
    // Test that:
    //   1. Applying low feature weights means the conditional probabilies
    //      of all classes tend to 1 / "number of classes",
    //   2. That the number returned as a confidence in the probabilities
    //      is the minimum feature weight.

    test::CRandomNumbers rng;

    TDoubleVecVec trainingData(2);
    rng.generateNormalSamples(0.0, 12.0, 100, trainingData[0]);
    rng.generateNormalSamples(10.0, 16.0, 100, trainingData[1]);

    maths::common::CNormalMeanPrecConjugate normal{
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};
    maths::common::CNaiveBayes nb{maths::common::CNaiveBayesFeatureDensityFromPrior(normal)};

    for (auto x : trainingData[0]) {
        nb.addTrainingDataPoint(0, {{x}});
    }
    for (auto x : trainingData[1]) {
        nb.addTrainingDataPoint(1, {{x}});
    }

    auto weightProvider = [weight = CTestFeatureWeight()]() mutable->maths::common::CNaiveBayesFeatureWeight& {
        return weight;
    };
    auto[probabilities, confidence] = nb.classProbabilities({{30.0}}, weightProvider);
    LOG_DEBUG(<< "p = " << probabilities << ", confidence = " << confidence);

    BOOST_REQUIRE_EQUAL(2, probabilities.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.5, probabilities[0].first, 1e-2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.5, probabilities[1].first, 1e-2);
    BOOST_REQUIRE_EQUAL(1e-3, confidence);
}

BOOST_AUTO_TEST_CASE(testMemoryUsage) {
    // Check invariants.

    using TNaiveBayesUPtr = std::unique_ptr<maths::common::CNaiveBayes>;

    test::CRandomNumbers rng;

    TDoubleVec trainingData[4];
    rng.generateNormalSamples(0.0, 12.0, 100, trainingData[0]);
    rng.generateNormalSamples(10.0, 16.0, 100, trainingData[1]);
    rng.generateNormalSamples(3.0, 14.0, 200, trainingData[2]);
    rng.generateNormalSamples(-5.0, 24.0, 200, trainingData[3]);

    TMeanAccumulator meanMeanError;

    maths::common::CNormalMeanPrecConjugate normal{
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, 0.1)};
    TNaiveBayesUPtr nb{std::make_unique<maths::common::CNaiveBayes>(
        maths::common::CNaiveBayesFeatureDensityFromPrior(normal), 0.1)};

    for (std::size_t i = 0; i < 100; ++i) {
        nb->addTrainingDataPoint(1, {{trainingData[0][i]}, {trainingData[1][i]}});
    }
    for (std::size_t i = 0; i < 200; ++i) {
        nb->addTrainingDataPoint(2, {{trainingData[2][i]}, {trainingData[3][i]}});
    }

    std::size_t memoryUsage{nb->memoryUsage()};
    auto mem{std::make_shared<core::CMemoryUsage>()};
    nb->debugMemoryUsage(mem);

    LOG_DEBUG(<< "Memory = " << memoryUsage);
    BOOST_REQUIRE_EQUAL(memoryUsage, mem->usage());

    LOG_DEBUG(<< "Memory = " << core::memory::dynamicSize(nb));
    BOOST_REQUIRE_EQUAL(memoryUsage + sizeof(maths::common::CNaiveBayes),
                        core::memory::dynamicSize(nb));
}

BOOST_AUTO_TEST_CASE(testPersist) {
    test::CRandomNumbers rng;

    TDoubleVecVec trainingData(4);
    rng.generateNormalSamples(0.0, 12.0, 100, trainingData[0]);
    rng.generateNormalSamples(10.0, 16.0, 100, trainingData[1]);
    rng.generateNormalSamples(3.0, 14.0, 200, trainingData[2]);
    rng.generateNormalSamples(-5.0, 24.0, 200, trainingData[3]);

    TMeanAccumulator meanMeanError;

    maths::common::CNormalMeanPrecConjugate normal{
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, 0.1)};
    maths::common::CNaiveBayes origNb{
        maths::common::CNaiveBayesFeatureDensityFromPrior(normal), 0.1};

    for (std::size_t i = 0; i < 100; ++i) {
        origNb.addTrainingDataPoint(1, {{trainingData[0][i]}, {trainingData[1][i]}});
    }
    for (std::size_t i = 0; i < 200; ++i) {
        origNb.addTrainingDataPoint(2, {{trainingData[2][i]}, {trainingData[3][i]}});
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origNb.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Naive Bayes XML representation:\n" << origXml);

    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::common::SDistributionRestoreParams params{maths_t::E_ContinuousData,
                                                     0.1, 0.0, 0.0, 0.0};
    maths::common::CNaiveBayes restoredNb{
        maths::common::CNaiveBayesFeatureDensityFromPrior(normal), params, traverser};

    BOOST_REQUIRE_EQUAL(origNb.checksum(), restoredNb.checksum());

    std::string restoredXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origNb.acceptPersistInserter(inserter);
        inserter.toXml(restoredXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, restoredXml);
}

BOOST_AUTO_TEST_SUITE_END()
}
