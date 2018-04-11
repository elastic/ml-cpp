/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CMultimodalPriorTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMixtureDistribution.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPriorDetail.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>
#include <maths/CXMeansOnline1d.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

using namespace ml;
using namespace handy_typedefs;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TPriorPtr = boost::shared_ptr<maths::CPrior>;
using CGammaRateConjugate = CPriorTestInterfaceMixin<maths::CGammaRateConjugate>;
using CLogNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CLogNormalMeanPrecConjugate>;
using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;
using CMultimodalPrior = CPriorTestInterfaceMixin<maths::CMultimodalPrior>;
using COneOfNPrior = CPriorTestInterfaceMixin<maths::COneOfNPrior>;

//! Make the default mode prior.
COneOfNPrior makeModePrior(const double& decayRate = 0.0) {
    CGammaRateConjugate gamma(maths::CGammaRateConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.01, decayRate, 0.0));
    CLogNormalMeanPrecConjugate logNormal(
        maths::CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.01, decayRate, 0.0));
    CNormalMeanPrecConjugate normal(maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, decayRate));

    COneOfNPrior::TPriorPtrVec priors;
    priors.push_back(COneOfNPrior::TPriorPtr(gamma.clone()));
    priors.push_back(COneOfNPrior::TPriorPtr(logNormal.clone()));
    priors.push_back(COneOfNPrior::TPriorPtr(normal.clone()));
    return COneOfNPrior(maths::COneOfNPrior(priors, maths_t::E_ContinuousData, decayRate));
}

//! Make a vanilla multimodal prior.
CMultimodalPrior makePrior(const maths::CPrior* modePrior, const double& decayRate) {
    maths::CXMeansOnline1d clusterer(
        maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, decayRate);

    if (modePrior) {
        return maths::CMultimodalPrior(maths_t::E_ContinuousData, clusterer, *modePrior, decayRate);
    }
    return maths::CMultimodalPrior(maths_t::E_ContinuousData, clusterer, makeModePrior(decayRate), decayRate);
}
CMultimodalPrior makePrior(const maths::CPrior* modePrior) {
    return makePrior(modePrior, 0.0);
}
CMultimodalPrior makePrior(double decayRate) {
    return makePrior(nullptr, decayRate);
}
CMultimodalPrior makePrior() {
    return makePrior(nullptr, 0.0);
}

test::CRandomNumbers RNG;

void sample(const boost::math::normal_distribution<>& normal, std::size_t numberSamples, TDoubleVec& result) {
    RNG.generateNormalSamples(boost::math::mean(normal), boost::math::variance(normal), numberSamples, result);
}

void sample(const boost::math::lognormal_distribution<>& lognormal, std::size_t numberSamples, TDoubleVec& result) {
    RNG.generateLogNormalSamples(lognormal.location(), lognormal.scale() * lognormal.scale(), numberSamples, result);
}

void sample(const boost::math::gamma_distribution<>& gamma, std::size_t numberSamples, TDoubleVec& result) {
    RNG.generateGammaSamples(gamma.shape(), gamma.scale(), numberSamples, result);
}
template<typename T>
void probabilityOfLessLikelySample(const maths::CMixtureDistribution<T>& mixture, const double& x, double& probability, double& deviation) {
    using TModeVec = typename maths::CMixtureDistribution<T>::TModeVec;

    static const double NUMBER_SAMPLES = 10000.0;

    probability = 0.0;

    double fx = pdf(mixture, x);
    const TDoubleVec& weights = mixture.weights();
    const TModeVec& modes = mixture.modes();
    for (std::size_t i = 0u; i < modes.size(); ++i) {
        TDoubleVec samples;
        sample(modes[i], static_cast<std::size_t>(NUMBER_SAMPLES * weights[i]), samples);
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            if (pdf(mixture, samples[j]) < fx) {
                probability += 1.0 / NUMBER_SAMPLES;
            }
        }
    }

    // For a discussion of the deviation see the paper:
    //   "Anomaly Detection in Application Performance Monitoring Data"
    deviation = std::sqrt(probability * (1.0 - probability) / NUMBER_SAMPLES);
}
}

void CMultimodalPriorTest::testMultipleUpdate() {
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testMultipleUpdate  |");
    LOG_DEBUG("+--------------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double shape = 2.0;
    const double scale = 3.0;

    const double decayRate = 0.0;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(shape, scale, 100, samples);

    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        maths::CXMeansOnline1d clusterer(dataTypes[i], maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight);

        CMultimodalPrior filter1(maths::CMultimodalPrior(
            dataTypes[i], clusterer, maths::CNormalMeanPrecConjugate::nonInformativePrior(dataTypes[i], decayRate)));
        CMultimodalPrior filter2(filter1);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            filter1.addSamples(TDouble1Vec(1, samples[j]));
        }
        filter2.addSamples(samples);

        LOG_DEBUG("checksum 1 " << filter1.checksum());
        LOG_DEBUG("checksum 2 " << filter2.checksum());
        CPPUNIT_ASSERT_EQUAL(filter1.checksum(), filter2.checksum());
    }
}

void CMultimodalPriorTest::testPropagation() {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testPropagation  |");
    LOG_DEBUG("+-----------------------------------------+");

    // Test that propagation doesn't affect the marginal likelihood
    // mean and the marginal likelihood confidence intervals increase
    // (due to influence of the prior uncertainty) after propagation.

    double eps = 0.01;

    test::CRandomNumbers rng;

    TDoubleVec samples1;
    rng.generateNormalSamples(3.0, 1.0, 200, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(10.0, 4.0, 200, samples2);
    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());

    rng.random_shuffle(samples.begin(), samples.end());

    const double decayRate = 0.1;
    CMultimodalPrior filter(makePrior(decayRate));

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter.addSamples(TDouble1Vec(1, static_cast<double>(samples[i])));
        CPPUNIT_ASSERT(filter.checkInvariants());
    }

    double mean = filter.marginalLikelihoodMean();
    TDoubleDoublePr percentiles[] = {filter.marginalLikelihoodConfidenceInterval(60.0),
                                     filter.marginalLikelihoodConfidenceInterval(70.0),
                                     filter.marginalLikelihoodConfidenceInterval(80.0),
                                     filter.marginalLikelihoodConfidenceInterval(90.0)};

    filter.propagateForwardsByTime(40.0);
    CPPUNIT_ASSERT(filter.checkInvariants());

    double propagatedMean = filter.marginalLikelihoodMean();
    TDoubleDoublePr propagatedPercentiles[] = {filter.marginalLikelihoodConfidenceInterval(60.0),
                                               filter.marginalLikelihoodConfidenceInterval(70.0),
                                               filter.marginalLikelihoodConfidenceInterval(80.0),
                                               filter.marginalLikelihoodConfidenceInterval(90.0)};

    LOG_DEBUG("mean = " << mean << ", propagatedMean = " << propagatedMean);
    LOG_DEBUG("percentiles           = " << core::CContainerPrinter::print(percentiles));
    LOG_DEBUG("propagatedPercentiles = " << core::CContainerPrinter::print(propagatedPercentiles));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(mean, propagatedMean, eps * mean);
    for (std::size_t i = 0u; i < boost::size(percentiles); ++i) {
        CPPUNIT_ASSERT(propagatedPercentiles[i].first < percentiles[i].first);
        CPPUNIT_ASSERT(propagatedPercentiles[i].second > percentiles[i].second);
    }
}

void CMultimodalPriorTest::testSingleMode() {
    LOG_DEBUG("+----------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testSingleMode  |");
    LOG_DEBUG("+----------------------------------------+");

    // We test the log likelihood of the data for the estimated
    // distributions versus the generating distributions. Note
    // that the generating distribution doesn't necessarily have
    // a larger likelihood because we are using a finite sample.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    LOG_DEBUG("Gaussian");
    {
        COneOfNPrior modePrior(makeModePrior());
        CMultimodalPrior filter1(makePrior(&modePrior));
        COneOfNPrior filter2 = modePrior;

        const double mean = 10.0;
        const double variance = 2.0;

        TDoubleVec samples;
        rng.generateNormalSamples(mean, std::sqrt(variance), 1000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            TDouble1Vec sample(1, samples[i]);
            filter1.addSamples(sample);
            filter2.addSamples(sample);
            CPPUNIT_ASSERT(filter1.checkInvariants());
        }

        TMeanAccumulator L1G;
        TMeanAccumulator L12;
        TMeanAccumulator differentialEntropy;

        boost::math::normal_distribution<> f(mean, std::sqrt(variance));
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double fx = boost::math::pdf(f, samples[i]);
            TDouble1Vec sample(1, samples[i]);
            double l1;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter1.jointLogMarginalLikelihood(sample, l1));
            L1G.add(std::log(fx) - l1);
            double l2;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter2.jointLogMarginalLikelihood(sample, l2));
            L12.add(l2 - l1);
            differentialEntropy.add(-std::log(fx));
        }

        LOG_DEBUG("L1G = " << maths::CBasicStatistics::mean(L1G) << ", L12 = " << maths::CBasicStatistics::mean(L12)
                           << ", differential entropy " << differentialEntropy);

        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(L1G) / maths::CBasicStatistics::mean(differentialEntropy) < 0.0);
    }
    LOG_DEBUG("Log-Normal");
    {
        COneOfNPrior modePrior(makeModePrior());
        CMultimodalPrior filter1(makePrior(&modePrior));
        COneOfNPrior filter2 = modePrior;

        const double location = 1.5;
        const double squareScale = 0.9;

        TDoubleVec samples;
        rng.generateLogNormalSamples(location, squareScale, 1000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            TDouble1Vec sample(1, samples[i]);
            filter1.addSamples(sample);
            filter2.addSamples(sample);
            CPPUNIT_ASSERT(filter1.checkInvariants());
        }

        TMeanAccumulator L1G;
        TMeanAccumulator L12;
        TMeanAccumulator differentialEntropy;

        boost::math::lognormal_distribution<> f(location, std::sqrt(squareScale));

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double fx = boost::math::pdf(f, samples[i]);
            TDouble1Vec sample(1, samples[i]);
            double l1;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter1.jointLogMarginalLikelihood(sample, l1));
            L1G.add(std::log(fx) - l1);
            double l2;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter2.jointLogMarginalLikelihood(sample, l2));
            L12.add(l2 - l1);
            differentialEntropy.add(-std::log(fx));
        }

        LOG_DEBUG("L1G = " << maths::CBasicStatistics::mean(L1G) << ", L12 = " << maths::CBasicStatistics::mean(L12)
                           << ", differential entropy " << differentialEntropy);

        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(L1G) / maths::CBasicStatistics::mean(differentialEntropy) < 0.0);
    }
    LOG_DEBUG("Gamma");
    {
        COneOfNPrior modePrior(makeModePrior());
        CMultimodalPrior filter1(makePrior(&modePrior));
        COneOfNPrior filter2 = modePrior;

        const double shape = 1.0;
        const double scale = 0.5;

        TDoubleVec samples;
        rng.generateGammaSamples(shape, scale, 1000, samples);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            TDouble1Vec sample(1, samples[i]);
            filter1.addSamples(sample);
            filter2.addSamples(sample);
            CPPUNIT_ASSERT(filter1.checkInvariants());
        }

        TMeanAccumulator L1G;
        TMeanAccumulator L12;
        TMeanAccumulator differentialEntropy;

        boost::math::gamma_distribution<> f(shape, scale);

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double fx = boost::math::pdf(f, samples[i]);
            TDouble1Vec sample(1, samples[i]);
            double l1;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter1.jointLogMarginalLikelihood(sample, l1));
            L1G.add(std::log(fx) - l1);
            double l2;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter2.jointLogMarginalLikelihood(sample, l2));
            L12.add(l2 - l1);
            differentialEntropy.add(-std::log(fx));
        }

        LOG_DEBUG("L1G = " << maths::CBasicStatistics::mean(L1G) << ", L12 = " << maths::CBasicStatistics::mean(L12)
                           << ", differential entropy " << differentialEntropy);

        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(L1G) / maths::CBasicStatistics::mean(differentialEntropy) < 0.1);
    }
}

void CMultimodalPriorTest::testMultipleModes() {
    LOG_DEBUG("+-------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testMultipleModes  |");
    LOG_DEBUG("+-------------------------------------------+");

    // We check that for data generated from multiple modes
    // we get something close to the generating distribution.
    // In particular, we test the log likelihood of the data
    // for the estimated distribution versus the generating
    // distribution and versus an unclustered distribution.
    // Note that the generating distribution doesn't necessarily
    // have a larger likelihood because we are using a finite
    // sample.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("Mixture Normals");

        const std::size_t n1 = 400u;
        const double mean1 = 10.0;
        const double variance1 = 2.0;

        const std::size_t n2 = 600u;
        const double mean2 = 20.0;
        const double variance2 = 5.0;

        TDoubleVec samples1;
        rng.generateNormalSamples(mean1, variance1, n1, samples1);
        TDoubleVec samples2;
        rng.generateNormalSamples(mean2, variance2, n2, samples2);

        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());

        LOG_DEBUG("# samples = " << samples.size());

        double w1 = n1 / static_cast<double>(n1 + n2);
        double w2 = n2 / static_cast<double>(n1 + n2);
        boost::math::normal_distribution<> mode1Distribution(mean1, std::sqrt(variance1));
        boost::math::normal_distribution<> mode2Distribution(mean2, std::sqrt(variance2));

        double loss = 0.0;
        TMeanAccumulator differentialEntropy_;
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            double fx = w1 * boost::math::pdf(mode1Distribution, samples[j]) + w2 * boost::math::pdf(mode2Distribution, samples[j]);
            differentialEntropy_.add(-std::log(fx));
        }
        double differentialEntropy = maths::CBasicStatistics::mean(differentialEntropy_);

        for (std::size_t i = 0; i < 10; ++i) {
            rng.random_shuffle(samples.begin(), samples.end());

            COneOfNPrior modePrior(makeModePrior());
            CMultimodalPrior filter1(makePrior(&modePrior));
            COneOfNPrior filter2 = modePrior;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                TDouble1Vec sample(1, samples[j]);
                filter1.addSamples(sample);
                filter2.addSamples(sample);
                CPPUNIT_ASSERT(filter1.checkInvariants());
            }

            CPPUNIT_ASSERT_EQUAL(std::size_t(2), filter1.numberModes());

            TMeanAccumulator loss1G;
            TMeanAccumulator loss12;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                double fx = w1 * boost::math::pdf(mode1Distribution, samples[j]) + w2 * boost::math::pdf(mode2Distribution, samples[j]);
                TDouble1Vec sample(1, samples[j]);
                double l1;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter1.jointLogMarginalLikelihood(sample, l1));
                loss1G.add(std::log(fx) - l1);
                double l2;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter2.jointLogMarginalLikelihood(sample, l2));
                loss12.add(l2 - l1);
            }

            LOG_DEBUG("loss1G = " << maths::CBasicStatistics::mean(loss1G) << ", loss12 = " << maths::CBasicStatistics::mean(loss12)
                                  << ", differential entropy " << differentialEntropy);

            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss12) < 0.0);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss1G) / differentialEntropy < 0.0);
            loss += maths::CBasicStatistics::mean(loss1G);
        }

        loss /= 10.0;
        LOG_DEBUG("loss = " << loss << ", differential entropy = " << differentialEntropy);
        CPPUNIT_ASSERT(loss / differentialEntropy < 0.0);
    }
    {
        LOG_DEBUG("Mixture Log-Normals");

        const std::size_t n1 = 600u;
        const double location1 = 2.0;
        const double squareScale1 = 0.04;

        const std::size_t n2 = 300u;
        const double location2 = 3.0;
        const double squareScale2 = 0.08;

        const std::size_t n3 = 100u;
        const double location3 = 4.0;
        const double squareScale3 = 0.01;

        TDoubleVec samples1;
        rng.generateLogNormalSamples(location1, squareScale1, n1, samples1);
        TDoubleVec samples2;
        rng.generateLogNormalSamples(location2, squareScale2, n2, samples2);
        TDoubleVec samples3;
        rng.generateLogNormalSamples(location3, squareScale3, n3, samples3);

        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        samples.insert(samples.end(), samples3.begin(), samples3.end());

        LOG_DEBUG("# samples = " << samples.size());

        double w1 = n1 / static_cast<double>(n1 + n2 + n3);
        double w2 = n2 / static_cast<double>(n1 + n2 + n3);
        double w3 = n3 / static_cast<double>(n1 + n2 + n3);
        boost::math::lognormal_distribution<> mode1Distribution(location1, std::sqrt(squareScale1));
        boost::math::lognormal_distribution<> mode2Distribution(location2, std::sqrt(squareScale2));
        boost::math::lognormal_distribution<> mode3Distribution(location3, std::sqrt(squareScale3));

        double loss = 0.0;
        TMeanAccumulator differentialEntropy_;
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            double fx = w1 * boost::math::pdf(mode1Distribution, samples[j]) + w2 * boost::math::pdf(mode2Distribution, samples[j]) +
                        w3 * boost::math::pdf(mode3Distribution, samples[j]);
            differentialEntropy_.add(-std::log(fx));
        }
        double differentialEntropy = maths::CBasicStatistics::mean(differentialEntropy_);

        for (std::size_t i = 0; i < 10; ++i) {
            rng.random_shuffle(samples.begin(), samples.end());

            COneOfNPrior modePrior(makeModePrior());
            CMultimodalPrior filter1(makePrior(&modePrior));
            COneOfNPrior filter2 = modePrior;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                TDouble1Vec sample(1, samples[j]);
                filter1.addSamples(sample);
                filter2.addSamples(sample);
                CPPUNIT_ASSERT(filter1.checkInvariants());
            }

            CPPUNIT_ASSERT_EQUAL(std::size_t(3), filter1.numberModes());

            TMeanAccumulator loss1G;
            TMeanAccumulator loss12;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                double fx = w1 * boost::math::pdf(mode1Distribution, samples[j]) + w2 * boost::math::pdf(mode2Distribution, samples[j]) +
                            w3 * boost::math::pdf(mode3Distribution, samples[j]);
                TDouble1Vec sample(1, samples[j]);
                double l1;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter1.jointLogMarginalLikelihood(sample, l1));
                loss1G.add(std::log(fx) - l1);
                double l2;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter2.jointLogMarginalLikelihood(sample, l2));
                loss12.add(l2 - l1);
            }

            LOG_DEBUG("loss1G = " << maths::CBasicStatistics::mean(loss1G) << ", loss12 = " << maths::CBasicStatistics::mean(loss12)
                                  << ", differential entropy " << differentialEntropy);

            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss12) < 0.0);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss1G) / differentialEntropy < 0.001);
            loss += maths::CBasicStatistics::mean(loss1G);
        }

        loss /= 10.0;
        LOG_DEBUG("loss = " << loss << ", differential entropy = " << differentialEntropy);
        CPPUNIT_ASSERT(loss / differentialEntropy < 0.0);
    }
    {
        LOG_DEBUG("Mixed Modes");

        const std::size_t n1 = 400u;
        const double mean1 = 10.0;
        const double variance1 = 1.0;

        const std::size_t n2 = 200u;
        const double location2 = 3.0;
        const double squareScale2 = 0.08;

        const std::size_t n3 = 400u;
        const double shape3 = 120.0;
        const double scale3 = 0.3;

        TDoubleVec samples1;
        rng.generateNormalSamples(mean1, variance1, n1, samples1);
        TDoubleVec samples2;
        rng.generateLogNormalSamples(location2, squareScale2, n2, samples2);
        TDoubleVec samples3;
        rng.generateGammaSamples(shape3, scale3, n3, samples3);

        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        samples.insert(samples.end(), samples3.begin(), samples3.end());

        LOG_DEBUG("# samples = " << samples.size());

        double w1 = n1 / static_cast<double>(n1 + n2 + n3);
        double w2 = n2 / static_cast<double>(n1 + n2 + n3);
        double w3 = n3 / static_cast<double>(n1 + n2 + n3);
        boost::math::normal_distribution<> mode1Distribution(mean1, std::sqrt(variance1));
        boost::math::lognormal_distribution<> mode2Distribution(location2, std::sqrt(squareScale2));
        boost::math::gamma_distribution<> mode3Distribution(shape3, scale3);

        double loss = 0.0;
        TMeanAccumulator differentialEntropy_;
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            double fx = w1 * boost::math::pdf(mode1Distribution, samples[j]) + w2 * boost::math::pdf(mode2Distribution, samples[j]) +
                        w3 * boost::math::pdf(mode3Distribution, samples[j]);
            differentialEntropy_.add(-std::log(fx));
        }
        double differentialEntropy = maths::CBasicStatistics::mean(differentialEntropy_);

        for (std::size_t i = 0; i < 10; ++i) {
            rng.random_shuffle(samples.begin(), samples.end());

            COneOfNPrior modePrior(makeModePrior());
            CMultimodalPrior filter1(makePrior(&modePrior));
            COneOfNPrior filter2 = modePrior;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                TDouble1Vec sample(1, samples[j]);
                filter1.addSamples(sample);
                filter2.addSamples(sample);
                CPPUNIT_ASSERT(filter1.checkInvariants());
            }

            CPPUNIT_ASSERT_EQUAL(std::size_t(3), filter1.numberModes());

            TMeanAccumulator loss1G;
            TMeanAccumulator loss12;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                double fx = w1 * boost::math::pdf(mode1Distribution, samples[j]) + w2 * boost::math::pdf(mode2Distribution, samples[j]) +
                            w3 * boost::math::pdf(mode3Distribution, samples[j]);
                TDouble1Vec sample(1, samples[j]);
                double l1;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter1.jointLogMarginalLikelihood(sample, l1));
                loss1G.add(std::log(fx) - l1);
                double l2;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter2.jointLogMarginalLikelihood(sample, l2));
                loss12.add(l2 - l1);
            }

            LOG_DEBUG("loss1G = " << maths::CBasicStatistics::mean(loss1G) << ", loss12 = " << maths::CBasicStatistics::mean(loss12)
                                  << ", differential entropy " << differentialEntropy);

            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss12) < 0.0);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss1G) / differentialEntropy < 0.01);
            loss += maths::CBasicStatistics::mean(loss1G);
        }

        loss /= 10.0;
        LOG_DEBUG("loss = " << loss << ", differential entropy = " << differentialEntropy);
        CPPUNIT_ASSERT(loss / differentialEntropy < 0.003);
    }
}

void CMultimodalPriorTest::testMarginalLikelihood() {
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testMarginalLikelihood  |");
    LOG_DEBUG("+------------------------------------------------+");

    using TNormalVec = std::vector<boost::math::normal_distribution<>>;

    // Check that the c.d.f. <= 1 at extreme.
    {
        CMultimodalPrior filter(makePrior());

        const double shape = 1.0;
        const double scale = 1.0;
        const double location = 2.0;
        const double squareScale = 0.5;

        test::CRandomNumbers rng;

        TDoubleVec samples;
        rng.generateGammaSamples(shape, scale, 100, samples);
        filter.addSamples(samples);
        rng.generateLogNormalSamples(location, squareScale, 100, samples);
        filter.addSamples(samples);

        maths_t::ESampleWeightStyle weightStyles[] = {
            maths_t::E_SampleCountWeight, maths_t::E_SampleWinsorisationWeight, maths_t::E_SampleCountWeight};
        double weights[] = {0.1, 1.0, 10.0};

        for (std::size_t i = 0u; i < boost::size(weightStyles); ++i) {
            for (std::size_t j = 0u; j < boost::size(weights); ++j) {
                double lb, ub;
                filter.minusLogJointCdf(maths_t::TWeightStyleVec(1, weightStyles[i]),
                                        TDouble1Vec(1, 20000.0),
                                        TDouble4Vec1Vec(1, TDouble4Vec(1, weights[j])),
                                        lb,
                                        ub);
                LOG_DEBUG("-log(c.d.f) = " << (lb + ub) / 2.0);
                CPPUNIT_ASSERT(lb >= 0.0);
                CPPUNIT_ASSERT(ub >= 0.0);
            }
        }
    }

    // Check that the marginal likelihood and c.d.f. agree for some
    // test data and that the c.d.f. <= 1 and that the expected value
    // of the log likelihood tends to the differential entropy.

    const double decayRates[] = {0.0, 0.001, 0.01};

    unsigned int numberSamples[] = {2u, 20u, 500u};
    const double tolerance = 0.01;

    test::CRandomNumbers rng;

    const double w1 = 0.5;
    const double mean1 = 10.0;
    const double variance1 = 1.0;
    const double w2 = 0.3;
    const double mean2 = 15.0;
    const double variance2 = 2.0;
    const double w3 = 0.2;
    const double mean3 = 25.0;
    const double variance3 = 3.0;
    TDoubleVec samples1;
    rng.generateNormalSamples(mean1, variance1, static_cast<std::size_t>(w1 * 500.0), samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(mean2, variance2, static_cast<std::size_t>(w2 * 500.0), samples2);
    TDoubleVec samples3;
    rng.generateNormalSamples(mean3, variance3, static_cast<std::size_t>(w3 * 500.0), samples3);
    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    samples.insert(samples.end(), samples3.begin(), samples3.end());
    rng.random_shuffle(samples.begin(), samples.end());

    for (size_t i = 0; i < boost::size(numberSamples); ++i) {
        for (size_t j = 0; j < boost::size(decayRates); ++j) {
            CMultimodalPrior filter(makePrior(decayRates[j]));

            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(TDouble1Vec(1, samples[k]));
                filter.propagateForwardsByTime(1.0);
                CPPUNIT_ASSERT(filter.checkInvariants());
            }
            LOG_DEBUG("# modes = " << filter.numberModes());

            // We'll check that the p.d.f. is close to the derivative of the
            // c.d.f. at a range of points on the p.d.f.

            const double eps = 1e-4;

            for (size_t k = 5; k < 31; ++k) {
                TDouble1Vec sample(1, static_cast<double>(k));

                LOG_DEBUG("number = " << numberSamples[i] << ", sample = " << sample[0]);

                double logLikelihood = 0.0;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter.jointLogMarginalLikelihood(sample, logLikelihood));
                double pdf = std::exp(logLikelihood);

                double lowerBound = 0.0, upperBound = 0.0;
                sample[0] -= eps;
                CPPUNIT_ASSERT(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(lowerBound, upperBound, 1e-3);
                double minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtMinusEps = std::exp(-minusLogCdf);
                CPPUNIT_ASSERT(minusLogCdf >= 0.0);

                sample[0] += 2.0 * eps;
                CPPUNIT_ASSERT(filter.minusLogJointCdf(sample, lowerBound, upperBound));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(lowerBound, upperBound, 1e-3);
                minusLogCdf = (lowerBound + upperBound) / 2.0;
                double cdfAtPlusEps = std::exp(-minusLogCdf);
                CPPUNIT_ASSERT(minusLogCdf >= 0.0);

                double dcdfdx = (cdfAtPlusEps - cdfAtMinusEps) / 2.0 / eps;

                LOG_DEBUG("pdf(x) = " << pdf << ", d(cdf)/dx = " << dcdfdx);

                CPPUNIT_ASSERT_DOUBLES_EQUAL(pdf, dcdfdx, tolerance);
            }
        }
    }

    {
        // Test that the sample expectation of the log likelihood tends
        // to the expected log likelihood, which is just the differential
        // entropy.

        CMultimodalPrior filter(makePrior());
        filter.addSamples(samples);
        LOG_DEBUG("# modes = " << filter.numberModes());

        TDoubleVec manySamples1;
        rng.generateNormalSamples(mean1, variance1, static_cast<std::size_t>(w1 * 100000.0), manySamples1);
        TDoubleVec manySamples2;
        rng.generateNormalSamples(mean2, variance2, static_cast<std::size_t>(w2 * 100000.0), manySamples2);
        TDoubleVec manySamples3;
        rng.generateNormalSamples(mean3, variance3, static_cast<std::size_t>(w3 * 100000.0), manySamples3);
        TDoubleVec manySamples;
        manySamples.insert(manySamples.end(), manySamples1.begin(), manySamples1.end());
        manySamples.insert(manySamples.end(), manySamples2.begin(), manySamples2.end());
        manySamples.insert(manySamples.end(), manySamples3.begin(), manySamples3.end());
        rng.random_shuffle(manySamples.begin(), manySamples.end());

        TDoubleVec weights;
        weights.push_back(w1);
        weights.push_back(w2);
        weights.push_back(w3);
        TNormalVec modes;
        modes.push_back(boost::math::normal_distribution<>(mean1, variance1));
        modes.push_back(boost::math::normal_distribution<>(mean2, variance2));
        modes.push_back(boost::math::normal_distribution<>(mean3, variance3));
        maths::CMixtureDistribution<boost::math::normal_distribution<>> f(weights, modes);
        double expectedDifferentialEntropy = maths::CTools::differentialEntropy(f);

        double differentialEntropy = 0.0;
        for (std::size_t i = 0u; i < manySamples.size(); ++i) {
            if (i % 1000 == 0) {
                LOG_DEBUG("Processed " << i << " samples");
            }
            TDouble1Vec sample(1, manySamples[i]);
            filter.addSamples(sample);
            double logLikelihood = 0.0;
            CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors, filter.jointLogMarginalLikelihood(sample, logLikelihood));
            differentialEntropy -= logLikelihood;
        }

        differentialEntropy /= static_cast<double>(manySamples.size());

        LOG_DEBUG("differentialEntropy = " << differentialEntropy << ", expectedDifferentialEntropy = " << expectedDifferentialEntropy);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedDifferentialEntropy, differentialEntropy, 0.05 * expectedDifferentialEntropy);
    }
}

void CMultimodalPriorTest::testMarginalLikelihoodMode() {
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testMarginalLikelihoodMode  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that the marginal likelihood mode is at a local
    // minimum of the likelihood function. And we don't find
    // a higher likelihood location with high probability.

    test::CRandomNumbers rng;

    double w1 = 0.1;
    double mean1 = 1.0;
    double variance1 = 1.0;
    double w2 = 0.9;
    double mean2 = 8.0;
    double variance2 = 1.5;
    TDoubleVec samples1;
    rng.generateNormalSamples(mean1, variance1, static_cast<std::size_t>(w1 * 500.0), samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(mean2, variance2, static_cast<std::size_t>(w2 * 500.0), samples2);
    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    rng.random_shuffle(samples.begin(), samples.end());

    const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};

    CMultimodalPrior filter(makePrior());
    filter.addSamples(samples);

    maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleCountVarianceScaleWeight);
    TDouble4Vec weight(1, 1.0);
    TDouble4Vec1Vec weights(1, weight);

    std::size_t totalCount = 0u;
    for (std::size_t i = 0u; i < boost::size(varianceScales); ++i) {
        double vs = varianceScales[i];
        weight[0] = vs;
        weights[0][0] = vs;
        LOG_DEBUG("*** vs = " << vs << " ***");
        double mode = filter.marginalLikelihoodMode(weightStyle, weight);
        LOG_DEBUG("marginalLikelihoodMode = " << mode);
        // Should be near 8.
        CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, filter.marginalLikelihoodMode(weightStyle, weight), 2.0);
        double eps = 0.01;
        double modeMinusEps = mode - eps;
        double modePlusEps = mode + eps;
        double fMode, fModeMinusEps, fModePlusEps;
        filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, mode), weights, fMode);
        filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, modeMinusEps), weights, fModeMinusEps);
        filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, modePlusEps), weights, fModePlusEps);
        fMode = std::exp(fMode);
        fModeMinusEps = std::exp(fModeMinusEps);
        fModePlusEps = std::exp(fModePlusEps);
        double gradient = (fModePlusEps - fModeMinusEps) / 2.0 / eps;
        LOG_DEBUG("f(mode) = " << fMode << ", f(mode-eps) = " << fModeMinusEps << ", f(mode + eps) = " << fModePlusEps);
        LOG_DEBUG("gradient = " << gradient);
        CPPUNIT_ASSERT(std::fabs(gradient) < 0.05);
        CPPUNIT_ASSERT(fMode > 0.999 * fModeMinusEps);
        CPPUNIT_ASSERT(fMode > 0.999 * fModePlusEps);
        TDoubleVec trials;
        rng.generateUniformSamples(mean1, mean2, 500, trials);
        std::size_t count = 0u;
        TDoubleVec fTrials;
        for (std::size_t j = 0u; j < trials.size(); ++j) {
            double fTrial;
            filter.jointLogMarginalLikelihood(weightStyle, TDouble1Vec(1, trials[j]), weights, fTrial);
            fTrial = std::exp(fTrial);
            if (fTrial > fMode) {
                LOG_DEBUG("f(" << trials[j] << ") = " << fTrial << " > " << fMode);
                ++count;
            }
            fTrials.push_back(fTrial);
        }
        LOG_DEBUG("count = " << count);
        CPPUNIT_ASSERT(count < 6);
        totalCount += count;
    }

    LOG_DEBUG("totalCount = " << totalCount);
    CPPUNIT_ASSERT(totalCount < 11);
}

void CMultimodalPriorTest::testMarginalLikelihoodConfidenceInterval() {
    LOG_DEBUG("+------------------------------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testMarginalLikelihoodConfidenceInterval  |");
    LOG_DEBUG("+------------------------------------------------------------------+");

    // Test that marginal likelihood confidence intervals are
    // what we'd expect for various variance scales.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    LOG_DEBUG("Synthetic");
    {
        test::CRandomNumbers rng;

        double w1 = 0.2;
        double location1 = 0.1;
        double squareScale1 = 0.2;
        double w2 = 0.8;
        double mean2 = 8.0;
        double variance2 = 2.0;
        TDoubleVec samples1;
        rng.generateLogNormalSamples(location1, squareScale1, static_cast<std::size_t>(w1 * 2000.0), samples1);
        TDoubleVec samples2;
        rng.generateNormalSamples(mean2, variance2, static_cast<std::size_t>(w2 * 2000.0), samples2);
        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        rng.random_shuffle(samples.begin(), samples.end());

        const double varianceScales[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};

        const double percentages[] = {5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 95.0, 99.0, 99.9, 99.99};

        CMultimodalPrior filter(makePrior());
        filter.addSamples(samples);

        for (std::size_t i = 0u; i < boost::size(varianceScales); ++i) {
            LOG_DEBUG("*** vs = " << varianceScales[i] << " ***");
            TMeanAccumulator error;
            for (std::size_t j = 0u; j < boost::size(percentages); ++j) {
                LOG_DEBUG("** percentage = " << percentages[j] << " **");
                double q1, q2;
                filter.marginalLikelihoodQuantileForTest(50.0 - percentages[j] / 2.0, 1e-3, q1);
                filter.marginalLikelihoodQuantileForTest(50.0 + percentages[j] / 2.0, 1e-3, q2);
                TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(percentages[j]);
                LOG_DEBUG("[q1, q2] = [" << q1 << ", " << q2 << "]"
                                         << ", interval = " << core::CContainerPrinter::print(interval));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, 0.1);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.05);
                error.add(std::fabs(interval.first - q1));
                error.add(std::fabs(interval.second - q2));
            }
            LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 5e-3);
        }

        std::sort(samples.begin(), samples.end());
        TMeanAccumulator error;
        for (std::size_t i = 0u; i < boost::size(percentages); ++i) {
            LOG_DEBUG("** percentage = " << percentages[i] << " **");
            std::size_t i1 = static_cast<std::size_t>(static_cast<double>(samples.size()) * (50.0 - percentages[i] / 2.0) / 100.0 + 0.5);
            std::size_t i2 = static_cast<std::size_t>(static_cast<double>(samples.size()) * (50.0 + percentages[i] / 2.0) / 100.0 + 0.5);
            double q1 = samples[i1];
            double q2 = samples[std::min(i2, samples.size() - 1)];
            TDoubleDoublePr interval = filter.marginalLikelihoodConfidenceInterval(percentages[i]);
            LOG_DEBUG("[q1, q2] = [" << q1 << ", " << q2 << "]"
                                     << ", interval = " << core::CContainerPrinter::print(interval));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(q1, interval.first, std::max(0.1 * q1, 0.15));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(q2, interval.second, 0.1 * q2);
            error.add(std::fabs(interval.first - q1) / q1);
            error.add(std::fabs(interval.second - q2) / q2);
        }
        LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.05);
    }

    LOG_DEBUG("Problem Case (Issue 439)") {
        std::ifstream file;
        file.open("testfiles/poorly_conditioned_multimodal.txt");
        std::ostringstream state;
        state << file.rdbuf();

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(state.str()));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                                 0.0,
                                                 maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                                 maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                                 maths::MINIMUM_CATEGORY_COUNT);
        TPriorPtr prior;
        maths::CPriorStateSerialiser restorer;
        CPPUNIT_ASSERT(restorer(params, prior, traverser));
        TDoubleDoublePr median =
            prior->marginalLikelihoodConfidenceInterval(0, maths::CConstantWeights::COUNT, maths::CConstantWeights::UNIT);
        TDoubleDoublePr i90 =
            prior->marginalLikelihoodConfidenceInterval(90, maths::CConstantWeights::COUNT, maths::CConstantWeights::UNIT);

        LOG_DEBUG("median = " << maths::CBasicStatistics::mean(median));
        LOG_DEBUG("confidence interval = " << core::CContainerPrinter::print(i90));

        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(median) > i90.first);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(median) < i90.second);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-112.0, i90.first, 0.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(158952.0, i90.second, 0.5);
    }
}

void CMultimodalPriorTest::testSampleMarginalLikelihood() {
    LOG_DEBUG("+------------------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG("+------------------------------------------------------+");

    // We're going to test two properties of the sampling:
    //   1) That the sample mean is equal to the marginal likelihood
    //      mean.
    //   2) That the sample percentiles match the distribution
    //      percentiles.
    //   3) That the sample mean, variance and skew are all close to
    //      the corresponding quantities in the training data.
    //
    // I want to cross check these with the implementations of the
    // jointLogMarginalLikelihood and minusLogJointCdf so use these
    // to compute the mean and percentiles.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarSkewAccumulator = maths::CBasicStatistics::SSampleMeanVarSkew<double>::TAccumulator;

    const double eps = 1e-3;

    test::CRandomNumbers rng;

    TDoubleVec samples1;
    rng.generateNormalSamples(50.0, 1.0, 150, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(57.0, 1.0, 100, samples2);
    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    rng.random_shuffle(samples.begin(), samples.end());

    CMultimodalPrior filter(makePrior());

    TDouble1Vec sampled;

    TMeanVarSkewAccumulator sampleMoments;

    for (std::size_t i = 0u; i < 3u; ++i) {
        LOG_DEBUG("sample = " << samples[i]);

        sampleMoments.add(samples[i]);
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(10, sampled);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), sampled.size());
    }

    TMeanAccumulator meanMeanError;
    TMeanAccumulator meanVarError;

    std::size_t numberSampled = 20u;
    for (std::size_t i = 3u; i < samples.size(); ++i) {
        LOG_DEBUG("sample = " << samples[i]);

        sampleMoments.add(samples[i]);
        filter.addSamples(TDouble1Vec(1, samples[i]));

        sampled.clear();
        filter.sampleMarginalLikelihood(numberSampled, sampled);
        CPPUNIT_ASSERT_EQUAL(numberSampled, sampled.size());

        {
            TMeanVarAccumulator sampledMoments;
            sampledMoments = std::for_each(sampled.begin(), sampled.end(), sampledMoments);

            LOG_DEBUG("expectedMean = " << filter.marginalLikelihoodMean()
                                        << ", sampledMean = " << maths::CBasicStatistics::mean(sampledMoments));
            LOG_DEBUG("expectedVariance = " << filter.marginalLikelihoodVariance()
                                            << ", sampledVariance = " << maths::CBasicStatistics::variance(sampledMoments));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                filter.marginalLikelihoodMean(), maths::CBasicStatistics::mean(sampledMoments), 0.005 * filter.marginalLikelihoodMean());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodVariance(),
                                         maths::CBasicStatistics::variance(sampledMoments),
                                         0.2 * filter.marginalLikelihoodVariance());
            meanMeanError.add(std::fabs(filter.marginalLikelihoodMean() - maths::CBasicStatistics::mean(sampledMoments)) /
                              filter.marginalLikelihoodMean());
            meanVarError.add(std::fabs(filter.marginalLikelihoodVariance() - maths::CBasicStatistics::variance(sampledMoments)) /
                             filter.marginalLikelihoodVariance());
        }

        std::sort(sampled.begin(), sampled.end());
        for (std::size_t j = 1u; j < sampled.size(); ++j) {
            double q = 100.0 * static_cast<double>(j) / static_cast<double>(sampled.size());

            double expectedQuantile;
            CPPUNIT_ASSERT(filter.marginalLikelihoodQuantileForTest(q, eps, expectedQuantile));

            LOG_DEBUG("quantile = " << q << ", x_quantile = " << expectedQuantile << ", quantile range = [" << sampled[j - 1] << ","
                                    << sampled[j] << "]");

            CPPUNIT_ASSERT(expectedQuantile >= 0.98 * sampled[j - 1]);
            CPPUNIT_ASSERT(expectedQuantile <= 1.02 * sampled[j]);
        }
    }

    LOG_DEBUG("mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.0015);
    LOG_DEBUG("mean variance error = " << maths::CBasicStatistics::mean(meanVarError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanVarError) < 0.04);

    sampled.clear();
    filter.sampleMarginalLikelihood(numberSampled, sampled);
    TMeanVarSkewAccumulator sampledMoments;
    for (std::size_t i = 0u; i < sampled.size(); ++i) {
        sampledMoments.add(sampled[i]);
    }
    LOG_DEBUG("Sample moments = " << sampledMoments << ", sampled moments = " << sampleMoments);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(sampleMoments),
                                 maths::CBasicStatistics::mean(sampledMoments),
                                 1e-4 * maths::CBasicStatistics::mean(sampleMoments));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::variance(sampleMoments),
                                 maths::CBasicStatistics::variance(sampledMoments),
                                 0.05 * maths::CBasicStatistics::variance(sampleMoments));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::skewness(sampleMoments),
                                 maths::CBasicStatistics::skewness(sampledMoments),
                                 0.1 * maths::CBasicStatistics::skewness(sampleMoments));
}

void CMultimodalPriorTest::testCdf() {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testCdf  |");
    LOG_DEBUG("+---------------------------------+");

    // Test error cases.
    //
    // Test some invariants:
    //   "cdf" + "cdf complement" = 1
    //    cdf x for x < 0 = 1
    //    cdf complement x for x < 0 = 0

    const double locations[] = {1.0, 3.0};
    const double squareScales[] = {0.5, 0.3};
    const std::size_t n[] = {100u, 100u};

    test::CRandomNumbers rng;

    CGammaRateConjugate gamma(maths::CGammaRateConjugate::nonInformativePrior(maths_t::E_ContinuousData));
    CLogNormalMeanPrecConjugate logNormal(maths::CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData));
    COneOfNPrior::TPriorPtrVec priors;
    priors.push_back(COneOfNPrior::TPriorPtr(gamma.clone()));
    priors.push_back(COneOfNPrior::TPriorPtr(logNormal.clone()));
    COneOfNPrior modePrior(maths::COneOfNPrior(priors, maths_t::E_ContinuousData));
    CMultimodalPrior filter(makePrior(&modePrior));

    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        TDoubleVec samples;
        rng.generateLogNormalSamples(locations[i], squareScales[i], n[i], samples);
        filter.addSamples(samples);
    }

    double lowerBound;
    double upperBound;
    CPPUNIT_ASSERT(!filter.minusLogJointCdf(TDouble1Vec(), lowerBound, upperBound));
    CPPUNIT_ASSERT(!filter.minusLogJointCdfComplement(TDouble1Vec(), lowerBound, upperBound));

    CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, -1.0), lowerBound, upperBound));
    double f = (lowerBound + upperBound) / 2.0;
    CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, -1.0), lowerBound, upperBound));
    double fComplement = (lowerBound + upperBound) / 2.0;
    LOG_DEBUG("log(F(x)) = " << -f << ", log(1 - F(x)) = " << fComplement);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::log(std::numeric_limits<double>::min()), -f, 1e-8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, std::exp(-fComplement), 1e-8);

    for (std::size_t j = 1u; j < 1000; ++j) {
        double x = static_cast<double>(j) / 2.0;

        CPPUNIT_ASSERT(filter.minusLogJointCdf(TDouble1Vec(1, x), lowerBound, upperBound));
        f = (lowerBound + upperBound) / 2.0;
        CPPUNIT_ASSERT(filter.minusLogJointCdfComplement(TDouble1Vec(1, x), lowerBound, upperBound));
        fComplement = (lowerBound + upperBound) / 2.0;

        LOG_DEBUG("log(F(x)) = " << (f == 0.0 ? f : -f) << ", log(1 - F(x)) = " << (fComplement == 0.0 ? fComplement : -fComplement));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, std::exp(-f) + std::exp(-fComplement), 1e-8);
    }
}

void CMultimodalPriorTest::testProbabilityOfLessLikelySamples() {
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    using TNormalVec = std::vector<boost::math::normal_distribution<>>;
    using TLogNormalVec = std::vector<boost::math::lognormal_distribution<>>;
    using TGammaVec = std::vector<boost::math::gamma_distribution<>>;

    test::CRandomNumbers rng;

    {
        double weight1 = 0.5, weight2 = 0.5;
        double mean1 = 50.0, mean2 = 57.0;
        double variance1 = 1.0, variance2 = 1.0;

        TDoubleVec samples1;
        rng.generateNormalSamples(mean1, variance1, static_cast<std::size_t>(10000.0 * weight1), samples1);
        TDoubleVec samples2;
        rng.generateNormalSamples(mean2, variance2, static_cast<std::size_t>(10000.0 * weight2), samples2);
        TDoubleVec samples;
        samples.insert(samples.end(), samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        rng.random_shuffle(samples.begin(), samples.end());

        TDoubleVec weights;
        weights.push_back(weight1);
        weights.push_back(weight2);
        TNormalVec modes;
        modes.push_back(boost::math::normal_distribution<>(mean1, variance1));
        modes.push_back(boost::math::normal_distribution<>(mean2, variance2));
        maths::CMixtureDistribution<boost::math::normal_distribution<>> mixture(weights, modes);

        CMultimodalPrior filter(makePrior());
        filter.addSamples(samples);
        LOG_DEBUG("# modes = " << filter.numberModes());

        double x[] = {46.0, 49.0, 54.0, 55.0, 68.0};

        double error = 0.0;

        for (std::size_t i = 0u; i < boost::size(x); ++i) {
            double expectedProbability;
            double deviation;
            probabilityOfLessLikelySample(mixture, x[i], expectedProbability, deviation);

            double lowerBound;
            double upperBound;
            filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, TDouble1Vec(1, x[i]), lowerBound, upperBound);
            LOG_DEBUG("lowerBound = " << lowerBound << ", upperBound = " << upperBound << ", expectedProbability = " << expectedProbability
                                      << ", deviation = " << deviation);

            double probability = (lowerBound + upperBound) / 2.0;
            error +=
                probability < expectedProbability - 2.0 * deviation
                    ? (expectedProbability - 2.0 * deviation) - probability
                    : (probability > expectedProbability + 2.0 * deviation ? probability - (expectedProbability + 2.0 * deviation) : 0.0);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProbability, probability, std::max(3.0 * deviation, 3e-5));
        }

        error /= static_cast<double>(boost::size(x));
        LOG_DEBUG("error = " << error);
        CPPUNIT_ASSERT(error < 0.001);

        double lb, ub;
        maths_t::ETail tail;
        filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                              maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight),
                                              TDouble1Vec(1, 49.0),
                                              TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)),
                                              lb,
                                              ub,
                                              tail);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_LeftTail, tail);
        filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                              maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight),
                                              TDouble1Vec(1, 54.0),
                                              TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)),
                                              lb,
                                              ub,
                                              tail);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedOrNeitherTail, tail);
        filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                              maths_t::TWeightStyleVec(1, maths_t::E_SampleCountVarianceScaleWeight),
                                              TDouble1Vec(1, 59.0),
                                              TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)),
                                              lb,
                                              ub,
                                              tail);
        CPPUNIT_ASSERT_EQUAL(maths_t::E_RightTail, tail);
    }
    {
        double weights[] = {0.6, 0.2, 0.2};
        double locations[] = {1.0, 2.5, 4.0};
        double squareScales[] = {0.1, 0.05, 0.3};

        TDoubleVec samples;
        samples.reserve(20000u);
        for (std::size_t i = 0u; i < boost::size(weights); ++i) {
            TDoubleVec modeSamples;
            rng.generateLogNormalSamples(locations[i], squareScales[i], static_cast<std::size_t>(20000.0 * weights[i]), modeSamples);
            samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        }
        rng.random_shuffle(samples.begin(), samples.end());

        TDoubleVec mixtureWeights(boost::begin(weights), boost::end(weights));
        TLogNormalVec modes;
        modes.push_back(boost::math::lognormal_distribution<>(locations[0], std::sqrt(squareScales[0])));
        modes.push_back(boost::math::lognormal_distribution<>(locations[1], std::sqrt(squareScales[1])));
        modes.push_back(boost::math::lognormal_distribution<>(locations[2], std::sqrt(squareScales[2])));
        maths::CMixtureDistribution<boost::math::lognormal_distribution<>> mixture(mixtureWeights, modes);

        CMultimodalPrior filter(makePrior());
        filter.addSamples(samples);
        LOG_DEBUG("# modes = " << filter.numberModes());

        double x[] = {2.0, 3.0, 9.0, 15.0, 18.0, 22.0, 40.0, 60.0, 80.0, 110.0};

        double error = 0.0;

        for (std::size_t i = 0u; i < boost::size(x); ++i) {
            double expectedProbability;
            double deviation;
            probabilityOfLessLikelySample(mixture, x[i], expectedProbability, deviation);

            double lowerBound;
            double upperBound;
            filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, TDouble1Vec(1, x[i]), lowerBound, upperBound);
            LOG_DEBUG("lowerBound = " << lowerBound << ", upperBound = " << upperBound << ", expectedProbability = " << expectedProbability
                                      << ", deviation = " << deviation);

            double probability = (lowerBound + upperBound) / 2.0;
            error +=
                probability < expectedProbability - 2.0 * deviation
                    ? (expectedProbability - 2.0 * deviation) - probability
                    : (probability > expectedProbability + 2.0 * deviation ? probability - (expectedProbability + 2.0 * deviation) : 0.0);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                expectedProbability, probability, std::min(0.2 * expectedProbability + std::max(3.0 * deviation, 1e-10), 0.06));
        }

        error /= static_cast<double>(boost::size(x));
        LOG_DEBUG("error = " << error);
        CPPUNIT_ASSERT(error < 0.009);
    }
    {
        double weights[] = {0.6, 0.4};
        double shapes[] = {2.0, 300.0};
        double scales[] = {0.5, 1.5};

        TDoubleVec samples;
        samples.reserve(20000u);
        for (std::size_t i = 0u; i < boost::size(weights); ++i) {
            TDoubleVec modeSamples;
            rng.generateGammaSamples(shapes[i], scales[i], static_cast<std::size_t>(20000.0 * weights[i]), modeSamples);
            samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        }
        rng.random_shuffle(samples.begin(), samples.end());

        TDoubleVec mixtureWeights(boost::begin(weights), boost::end(weights));
        TGammaVec modes;
        modes.push_back(boost::math::gamma_distribution<>(shapes[0], scales[0]));
        modes.push_back(boost::math::gamma_distribution<>(shapes[1], scales[1]));
        maths::CMixtureDistribution<boost::math::gamma_distribution<>> mixture(mixtureWeights, modes);

        CMultimodalPrior filter(makePrior());
        filter.addSamples(samples);
        LOG_DEBUG("# modes = " << filter.numberModes());

        double x[] = {0.5, 1.5, 3.0, 35.0, 100.0, 320.0, 340.0, 360.0, 380.0, 410.0};

        double error = 0.0;

        for (std::size_t i = 0u; i < boost::size(x); ++i) {
            double expectedProbability;
            double deviation;
            probabilityOfLessLikelySample(mixture, x[i], expectedProbability, deviation);

            double lowerBound;
            double upperBound;
            filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, TDouble1Vec(1, x[i]), lowerBound, upperBound);
            LOG_DEBUG("lowerBound = " << lowerBound << ", upperBound = " << upperBound << ", expectedProbability = " << expectedProbability
                                      << ", deviation = " << deviation);

            double probability = (lowerBound + upperBound) / 2.0;
            error +=
                probability < expectedProbability - 2.0 * deviation
                    ? (expectedProbability - 2.0 * deviation) - probability
                    : (probability > expectedProbability + 2.0 * deviation ? probability - (expectedProbability + 2.0 * deviation) : 0.0);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProbability, probability, 0.18 * expectedProbability + std::max(2.5 * deviation, 1e-3));
        }

        error /= static_cast<double>(boost::size(x));
        LOG_DEBUG("error = " << error);
        CPPUNIT_ASSERT(error < 0.02);
    }
}

void CMultimodalPriorTest::testLargeValues() {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testLargeValues  |");
    LOG_DEBUG("+-----------------------------------------+");

    // Check that the confidence interval calculation stays
    // well conditioned for very large values.

    TDoubleVec values{
        7.324121e+10,  7.251927e+10,  7.152208e+10,  7.089604e+10,  7.018831e+10,  6.94266e+10,   6.890659e+10,  6.837292e+10,
        6.794372e+10,  6.793463e+10,  6.785385e+10,  6.773589e+10,  6.787609e+10,  6.760049e+10,  6.709596e+10,  6.701824e+10,
        6.672568e+10,  6.617609e+10,  6.620431e+10,  6.627069e+10,  6.617393e+10,  6.633176e+10,  6.600326e+10,  6.530363e+10,
        6.494482e+10,  6.433443e+10,  6.362233e+10,  6.317814e+10,  6.296127e+10,  6.272491e+10,  6.243567e+10,  6.19567e+10,
        6.13123e+10,   6.150823e+10,  6.160438e+10,  6.106396e+10,  6.128276e+10,  6.13318e+10,   6.161243e+10,  6.182719e+10,
        6.177156e+10,  6.174539e+10,  6.216147e+10,  6.272091e+10,  6.286637e+10,  6.310137e+10,  6.315882e+10,  6.312109e+10,
        6.312296e+10,  6.312432e+10,  6.328676e+10,  6.37708e+10,   6.421867e+10,  6.490675e+10,  6.547625e+10,  6.593425e+10,
        6.67186e+10,   6.755033e+10,  6.754501e+10,  6.730381e+10,  6.76163e+10,   6.761007e+10,  6.745505e+10,  6.777796e+10,
        6.783472e+10,  6.779558e+10,  6.787643e+10,  6.800003e+10,  6.840413e+10,  6.856255e+10,  6.939239e+10,  6.907512e+10,
        6.914988e+10,  6.901868e+10,  6.884531e+10,  6.934499e+10,  6.955862e+10,  6.938019e+10,  6.942022e+10,  6.950912e+10,
        6.979618e+10,  7.064871e+10,  7.152501e+10,  7.178129e+10,  7.2239e+10,    7.257321e+10,  7.28913e+10,   7.365193e+10,
        7.432521e+10,  7.475098e+10,  7.553025e+10,  7.654561e+10,  7.698032e+10,  7.768267e+10,  7.826669e+10,  7.866854e+10,
        7.924608e+10,  7.998602e+10,  8.038091e+10,  8.094976e+10,  8.145126e+10,  8.132123e+10,  8.142747e+10,  8.148276e+10,
        8.118588e+10,  8.122279e+10,  8.078815e+10,  8.008936e+10,  7.991103e+10,  7.981722e+10,  7.932372e+10,  7.900164e+10,
        7.881053e+10,  7.837734e+10,  7.847101e+10,  7.816575e+10,  7.789224e+10,  7.803634e+10,  7.827226e+10,  7.812112e+10,
        7.814848e+10,  7.812407e+10,  7.779805e+10,  7.783394e+10,  7.768365e+10,  7.74484e+10,   7.740301e+10,  7.725512e+10,
        7.666682e+10,  7.635862e+10,  7.592468e+10,  7.539656e+10,  7.529974e+10,  7.501661e+10,  7.442706e+10,  7.406878e+10,
        7.347894e+10,  7.268775e+10,  7.23729e+10,   7.171337e+10,  7.146626e+10,  7.130693e+10,  7.066356e+10,  6.977915e+10,
        6.915126e+10,  6.830462e+10,  6.73021e+10,   6.67686e+10,   6.600806e+10,  6.504958e+10,  6.427045e+10,  6.35093e+10,
        6.277891e+10,  6.258429e+10,  6.184866e+10,  6.114754e+10,  6.093035e+10,  6.063859e+10,  5.999596e+10,  5.952608e+10,
        5.927059e+10,  5.831014e+10,  5.763428e+10,  5.77239e+10,   5.82414e+10,   5.911797e+10,  5.987076e+10,  5.976584e+10,
        6.017487e+10,  6.023042e+10,  6.029144e+10,  6.068466e+10,  6.139924e+10,  6.208432e+10,  6.259237e+10,  6.300856e+10,
        6.342197e+10,  6.423638e+10,  6.494938e+10,  6.478293e+10,  6.444705e+10,  6.432593e+10,  6.437474e+10,  6.447832e+10,
        6.450247e+10,  6.398122e+10,  6.399681e+10,  6.406744e+10,  6.404553e+10,  6.417746e+10,  6.39819e+10,   6.389218e+10,
        6.453242e+10,  6.491168e+10,  6.493824e+10,  6.524365e+10,  6.537463e+10,  6.543864e+10,  6.583769e+10,  6.596521e+10,
        6.641129e+10,  6.718787e+10,  6.741177e+10,  6.776819e+10,  6.786579e+10,  6.783788e+10,  6.790788e+10,  6.77233e+10,
        6.738099e+10,  6.718351e+10,  6.739131e+10,  6.752051e+10,  6.747344e+10,  6.757187e+10,  6.739908e+10,  6.702725e+10,
        6.70474e+10,   6.708783e+10,  6.72989e+10,   6.75298e+10,   6.727323e+10,  6.677787e+10,  6.686342e+10,  6.687026e+10,
        6.714555e+10,  6.750766e+10,  6.807156e+10,  6.847816e+10,  6.915895e+10,  6.958225e+10,  6.970934e+10,  6.972807e+10,
        6.973312e+10,  6.970858e+10,  6.962325e+10,  6.968693e+10,  6.965446e+10,  6.983768e+10,  6.974386e+10,  6.992195e+10,
        7.010707e+10,  7.004337e+10,  7.006336e+10,  7.06312e+10,   7.078169e+10,  7.080609e+10,  7.107845e+10,  7.084754e+10,
        7.032667e+10,  7.052029e+10,  7.031464e+10,  7.006906e+10,  7.018558e+10,  7.022278e+10,  7.012379e+10,  7.043974e+10,
        7.016036e+10,  6.975801e+10,  6.95197e+10,   6.92444e+10,   6.85828e+10,   6.808828e+10,  6.74055e+10,   6.663602e+10,
        6.588224e+10,  6.52747e+10,   6.412303e+10,  6.315978e+10,  6.268569e+10,  6.219346e+10,  6.177174e+10,  6.101807e+10,
        6.018369e+10,  5.97554e+10,   5.924427e+10,  5.867325e+10,  5.814079e+10,  5.745633e+10,  5.641881e+10,  5.608709e+10,
        5.529503e+10,  5.450575e+10,  5.383054e+10,  5.297568e+10,  5.210389e+10,  5.139513e+10,  5.03026e+10,   4.922761e+10,
        4.839502e+10,  4.739353e+10,  4.605013e+10,  4.486422e+10,  4.369101e+10,  4.241115e+10,  4.128026e+10,  4.025775e+10,
        3.915851e+10,  3.819004e+10,  3.700971e+10,  3.581475e+10,  3.498126e+10,  3.384422e+10,  3.224959e+10,  3.108637e+10,
        2.997983e+10,  2.86439e+10,   2.774108e+10,  2.682793e+10,  2.590098e+10,  2.500665e+10,  2.368987e+10,  2.24582e+10,
        2.158596e+10,  2.062636e+10,  1.942922e+10,  1.873734e+10,  1.823214e+10,  1.726518e+10,  1.665115e+10,  1.582729e+10,
        1.477715e+10,  1.406265e+10,  1.285904e+10,  1.145722e+10,  1.038312e+10,  9.181713e+09,  8.141138e+09,  7.45358e+09,
        6.59996e+09,   5.72857e+09,   5.136189e+09,  4.51829e+09,   3.649536e+09,  2.990132e+09,  2.29392e+09,   1.390141e+09,
        5.611192e+08,  -1.62469e+08,  -1.041465e+09, -1.804217e+09, -2.923116e+09, -4.205691e+09, -5.09832e+09,  -6.12155e+09,
        -7.10503e+09,  -7.957297e+09, -9.107372e+09, -1.039097e+10, -1.133152e+10, -1.221205e+10, -1.318018e+10, -1.402195e+10,
        -1.512e+10,    -1.634369e+10, -1.710999e+10, -1.786548e+10, -1.866482e+10, -1.938912e+10, -2.039964e+10, -2.160603e+10,
        -2.259855e+10, -2.353314e+10, -2.449689e+10, -2.52005e+10,  -2.627104e+10, -2.730019e+10, -2.815777e+10, -2.920027e+10,
        -3.03507e+10,  -3.126021e+10, -3.212383e+10, -3.329089e+10, -3.402306e+10, -3.475361e+10, -3.572698e+10, -3.644467e+10,
        -3.721484e+10, -3.800023e+10, -3.865459e+10, -3.918282e+10, -3.983764e+10, -4.051065e+10, -4.119051e+10, -4.202436e+10,
        -4.24868e+10,  -4.340278e+10, -4.418258e+10, -4.490206e+10, -4.587365e+10, -4.697342e+10, -4.778222e+10, -4.882614e+10,
        -4.984197e+10, -5.051089e+10, -5.143766e+10, -5.252824e+10, -5.353136e+10, -5.436329e+10, -5.533555e+10, -5.623246e+10,
        -5.689744e+10, -5.798439e+10, -5.882786e+10, -5.96284e+10,  -6.061507e+10, -6.145417e+10, -6.235327e+10, -6.335978e+10,
        -6.405788e+10, -6.496648e+10, -6.600807e+10, -6.686964e+10, -6.782611e+10, -6.890904e+10, -6.941638e+10, -7.012465e+10,
        -7.113145e+10, -7.186233e+10, -7.2293e+10,   -7.313894e+10, -7.394114e+10, -7.475566e+10, -7.572029e+10, -7.660066e+10,
        -7.738602e+10, -7.846013e+10, -7.921084e+10, -7.986093e+10, -8.07113e+10,  -8.159104e+10, -8.243174e+10, -8.305353e+10,
        -8.346367e+10, -8.402575e+10, -8.482895e+10, -8.536747e+10, -8.581526e+10, -8.640365e+10, -8.683093e+10, -8.724777e+10,
        -8.746026e+10, -8.760338e+10, -8.809235e+10, -8.870936e+10, -8.905536e+10, -8.953669e+10, -9.031665e+10, -9.090067e+10,
        -9.135409e+10, -9.185499e+10, -9.225697e+10, -9.253896e+10, -9.314785e+10, -9.354807e+10, -9.391591e+10, -9.436751e+10,
        -9.471133e+10, -9.517393e+10, -9.587184e+10, -9.619209e+10, -9.607482e+10, -9.593427e+10, -9.604743e+10, -9.619758e+10,
        -9.62449e+10,  -9.61466e+10,  -9.636941e+10, -9.692289e+10, -9.735416e+10, -9.774056e+10, -9.828883e+10, -9.859253e+10,
        -9.888183e+10, -9.95351e+10,  -1.001142e+11};

    maths::CGammaRateConjugate gammaPrior = maths::CGammaRateConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.2, 0.001);
    maths::CNormalMeanPrecConjugate normalPrior = maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.001);
    maths::CLogNormalMeanPrecConjugate logNormalPrior =
        maths::CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.2, 0.001);

    maths::COneOfNPrior::TPriorPtrVec modePriors;
    modePriors.reserve(3u);
    modePriors.push_back(TPriorPtr(gammaPrior.clone()));
    modePriors.push_back(TPriorPtr(logNormalPrior.clone()));
    modePriors.push_back(TPriorPtr(normalPrior.clone()));
    maths::COneOfNPrior modePrior(modePriors, maths_t::E_ContinuousData, 0.001);
    maths::CXMeansOnline1d clusterer(
        maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight, 0.001, 0.05, 12, 0.8 / 3.0);
    maths::CMultimodalPrior multimodalPrior(maths_t::E_ContinuousData, clusterer, modePrior, 0.001);

    for (auto value : values) {

        multimodalPrior.addSamples(maths::CConstantWeights::COUNT, TDouble1Vec(1, value), TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0 / 3.0)));
        if (!multimodalPrior.isNonInformative()) {
            TDoubleDoublePr interval =
                multimodalPrior.marginalLikelihoodConfidenceInterval(95.0, maths::CConstantWeights::COUNT, maths::CConstantWeights::UNIT);
            if (interval.second - interval.first >= 3e11) {
                LOG_DEBUG("interval = " << interval.second - interval.first);
                LOG_DEBUG(multimodalPrior.print());
            }
            CPPUNIT_ASSERT(interval.second - interval.first < 3e11);
        }
    }
}

void CMultimodalPriorTest::testSeasonalVarianceScale() {
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testSeasonalVarianceScale  |");
    LOG_DEBUG("+---------------------------------------------------+");

    // We are test:
    //   1) The marginal likelihood is normalized.
    //   2) E[(X - m)^2] w.r.t. the log-likelihood is scaled.
    //   3) E[(X - m)^2] is close to marginalLikelihoodVariance.
    //   4) dF/dx = exp(log-likelihood) with different scales.
    //   5) The probability of less likely sample transforms as
    //      expected.
    //   6) Updating with scaled samples behaves as expected.

    const double mean1 = 6.0;
    const double variance1 = 4.0;
    const double mean2 = 20.0;
    const double variance2 = 20.0;
    const double mean3 = 50.0;
    const double variance3 = 20.0;

    test::CRandomNumbers rng;

    TDoubleVec samples1;
    rng.generateNormalSamples(mean1, variance1, 100, samples1);
    TDoubleVec samples2;
    rng.generateNormalSamples(mean2, variance2, 100, samples2);
    TDoubleVec samples3;
    rng.generateNormalSamples(mean3, variance3, 100, samples3);

    double varianceScales[] = {0.2, 0.5, 1.0, 2.0, 5.0};
    maths_t::TWeightStyleVec weightStyle(1, maths_t::E_SampleSeasonalVarianceScaleWeight);
    TDouble4Vec weight(1, 1.0);
    TDouble4Vec1Vec weights(1, weight);

    double m;
    double v;

    {
        CMultimodalPrior filter(makePrior());
        filter.addSamples(samples1);
        filter.addSamples(samples2);
        filter.addSamples(samples3);

        m = filter.marginalLikelihoodMean();
        v = filter.marginalLikelihoodVariance();
        LOG_DEBUG("v = " << v);

        double points[] = {0.5, 4.0, 12.0, 20.0, 40.0, 50.0, 60.0};

        double unscaledExpectationVariance;
        filter.expectation(CVarianceKernel(filter.marginalLikelihoodMean()), 50, unscaledExpectationVariance);
        LOG_DEBUG("unscaledExpectationVariance = " << unscaledExpectationVariance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(v, unscaledExpectationVariance, 1e-2 * unscaledExpectationVariance);

        for (std::size_t i = 0u; i < boost::size(varianceScales); ++i) {
            double vs = varianceScales[i];
            weight[0] = vs;
            weights[0][0] = vs;
            LOG_DEBUG("*** variance scale = " << vs << " ***");

            double Z;
            filter.expectation(C1dUnitKernel(), 50, Z, weightStyle, weight);
            LOG_DEBUG("Z = " << Z);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, Z, 1e-3);

            LOG_DEBUG("sv = " << filter.marginalLikelihoodVariance(weightStyle, weight));
            double expectationVariance;
            filter.expectation(CVarianceKernel(filter.marginalLikelihoodMean()), 50, expectationVariance, weightStyle, weight);
            LOG_DEBUG("expectationVariance = " << expectationVariance);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(vs * unscaledExpectationVariance, expectationVariance, 1e-3 * vs * unscaledExpectationVariance);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(filter.marginalLikelihoodVariance(weightStyle, weight),
                                         expectationVariance,
                                         1e-3 * filter.marginalLikelihoodVariance(weightStyle, weight));

            TDouble1Vec sample(1, 0.0);
            for (std::size_t j = 0u; j < boost::size(points); ++j) {
                TDouble1Vec x(1, points[j]);
                double fx;
                filter.jointLogMarginalLikelihood(weightStyle, x, weights, fx);
                TDouble1Vec xMinusEps(1, points[j] - 1e-3);
                TDouble1Vec xPlusEps(1, points[j] + 1e-3);
                double lb, ub;
                filter.minusLogJointCdf(weightStyle, xPlusEps, weights, lb, ub);
                double FxPlusEps = std::exp(-(lb + ub) / 2.0);
                filter.minusLogJointCdf(weightStyle, xMinusEps, weights, lb, ub);
                double FxMinusEps = std::exp(-(lb + ub) / 2.0);
                LOG_DEBUG("x = " << points[j] << ", log(f(x)) = " << fx << ", log(dF/dx)) = " << std::log((FxPlusEps - FxMinusEps) / 2e-3));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(fx, std::log((FxPlusEps - FxMinusEps) / 2e-3), 0.05 * std::fabs(fx));

                sample[0] = m + (points[j] - m) / std::sqrt(vs);
                weights[0][0] = 1.0;
                double expectedLowerBound;
                double expectedUpperBound;
                maths_t::ETail expectedTail;
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, weightStyle, sample, weights, expectedLowerBound, expectedUpperBound, expectedTail);

                sample[0] = points[j];
                weights[0][0] = vs;
                double lowerBound;
                double upperBound;
                maths_t::ETail tail;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided, weightStyle, sample, weights, lowerBound, upperBound, tail);

                LOG_DEBUG("expectedLowerBound = " << expectedLowerBound);
                LOG_DEBUG("lowerBound         = " << lowerBound);
                LOG_DEBUG("expectedUpperBound = " << expectedUpperBound);
                LOG_DEBUG("upperBound         = " << upperBound);
                LOG_DEBUG("expectedTail       = " << expectedTail);
                LOG_DEBUG("tail               = " << tail);

                if ((expectedLowerBound + expectedUpperBound) < 0.02) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        std::log(expectedLowerBound), std::log(lowerBound), 0.1 * std::fabs(std::log(expectedLowerBound)));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        std::log(expectedUpperBound), std::log(upperBound), 0.1 * std::fabs(std::log(expectedUpperBound)));
                } else {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedLowerBound, lowerBound, 0.05 * expectedLowerBound);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedUpperBound, upperBound, 0.05 * expectedUpperBound);
                }
                CPPUNIT_ASSERT_EQUAL(expectedTail, tail);
            }
        }
    }
    for (std::size_t i = 0u; i < boost::size(varianceScales); ++i) {
        double vs = varianceScales[i];

        TDouble1Vec samples(samples1.begin(), samples1.end());
        samples.insert(samples.end(), samples2.begin(), samples2.end());
        samples.insert(samples.end(), samples3.begin(), samples3.end());
        rng.random_shuffle(samples.begin(), samples.end());

        CMultimodalPrior filter(makePrior());
        weights[0][0] = vs;
        for (std::size_t j = 0u; j < samples.size(); ++j) {
            filter.addSamples(weightStyle, TDouble1Vec(1, samples[j]), weights);
        }

        double sm = filter.marginalLikelihoodMean();
        double sv = filter.marginalLikelihoodVariance();
        LOG_DEBUG("m  = " << m << ", v  = " << v);
        LOG_DEBUG("sm = " << sm << ", sv = " << sv);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(m, sm, 0.12 * m);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(v / vs, sv, 0.07 * v / vs);
    }
}

void CMultimodalPriorTest::testPersist() {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CMultimodalPriorTest::testPersist  |");
    LOG_DEBUG("+-------------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec samples1;
    rng.generateNormalSamples(5.0, 1.0, 100, samples1);
    TDoubleVec samples2;
    rng.generateLogNormalSamples(3.0, 0.1, 200, samples2);
    TDoubleVec samples;
    samples.insert(samples.end(), samples1.begin(), samples1.end());
    samples.insert(samples.end(), samples2.begin(), samples2.end());
    rng.random_shuffle(samples.begin(), samples.end());

    maths::CXMeansOnline1d clusterer(maths_t::E_ContinuousData, maths::CAvailableModeDistributions::ALL, maths_t::E_ClustersFractionWeight);
    maths::CGammaRateConjugate gamma = maths::CGammaRateConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.01);
    maths::CLogNormalMeanPrecConjugate logNormal = maths::CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.01);
    maths::CNormalMeanPrecConjugate normal = maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);

    COneOfNPrior::TPriorPtrVec priors;
    priors.push_back(COneOfNPrior::TPriorPtr(gamma.clone()));
    priors.push_back(COneOfNPrior::TPriorPtr(logNormal.clone()));
    priors.push_back(COneOfNPrior::TPriorPtr(normal.clone()));
    COneOfNPrior modePrior(maths::COneOfNPrior(priors, maths_t::E_ContinuousData));

    maths::CMultimodalPrior origFilter(maths_t::E_ContinuousData, clusterer, modePrior);
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        origFilter.addSamples(
            maths_t::TWeightStyleVec(1, maths_t::E_SampleCountWeight), TDouble1Vec(1, samples[i]), TDouble4Vec1Vec(1, TDouble4Vec(1, 1.0)));
    }
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("Multimodal XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(maths_t::E_ContinuousData,
                                             decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::CMultimodalPrior restoredFilter(params, traverser);

    LOG_DEBUG("orig checksum = " << checksum << " restored checksum = " << restoredFilter.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test* CMultimodalPriorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMultimodalPriorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testMultipleUpdate",
                                                                        &CMultimodalPriorTest::testMultipleUpdate));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testPropagation", &CMultimodalPriorTest::testPropagation));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testSingleMode", &CMultimodalPriorTest::testSingleMode));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testMultipleModes", &CMultimodalPriorTest::testMultipleModes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testMarginalLikelihood",
                                                                        &CMultimodalPriorTest::testMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testMarginalLikelihoodMode",
                                                                        &CMultimodalPriorTest::testMarginalLikelihoodMode));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testMarginalLikelihoodConfidenceInterval",
                                                                        &CMultimodalPriorTest::testMarginalLikelihoodConfidenceInterval));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testSampleMarginalLikelihood",
                                                                        &CMultimodalPriorTest::testSampleMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testCdf", &CMultimodalPriorTest::testCdf));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testProbabilityOfLessLikelySamples",
                                                                        &CMultimodalPriorTest::testProbabilityOfLessLikelySamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testSeasonalVarianceScale",
                                                                        &CMultimodalPriorTest::testSeasonalVarianceScale));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testLargeValues", &CMultimodalPriorTest::testLargeValues));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMultimodalPriorTest>("CMultimodalPriorTest::testPersist", &CMultimodalPriorTest::testPersist));

    return suiteOfTests;
}
