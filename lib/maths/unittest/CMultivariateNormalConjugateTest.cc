/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include "CMultivariateNormalConjugateTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CIntegration.h>
#include <maths/CMultivariateNormalConjugate.h>
#include <maths/CRestoreParams.h>

#include "TestUtils.h"

#include <test/CRandomNumbers.h>

using namespace ml;
using namespace handy_typedefs;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

namespace {

const maths_t::TWeightStyleVec COUNT_WEIGHT(1, maths_t::E_SampleCountWeight);
const maths_t::TWeightStyleVec VARIANCE_WEIGHT(1, maths_t::E_SampleCountVarianceScaleWeight);
const TDouble10Vec4Vec UNIT_WEIGHT_2(1, TDouble10Vec(2, 1.0));
const TDouble10Vec4Vec1Vec
    SINGLE_UNIT_WEIGHT_2(1, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));

void empiricalProbabilityOfLessLikelySamples(const TDoubleVec& mean,
                                             const TDoubleVecVec& covariance,
                                             TDoubleVec& result) {
    test::CRandomNumbers rng;
    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);
    result.resize(samples.size());
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        maths::gaussianLogLikelihood(
            TMatrix2(covariance), TVector2(samples[i]) - TVector2(mean), result[i]);
    }
    std::sort(result.begin(), result.end());
}

std::string print(maths_t::EDataType dataType) {
    switch (dataType) {
    case maths_t::E_DiscreteData:
        return "Discrete";
    case maths_t::E_IntegerData:
        return "Integer";
    case maths_t::E_ContinuousData:
        return "Continuous";
    case maths_t::E_MixedData:
        return "Mixed";
    }
    return "";
}

void gaussianSamples(test::CRandomNumbers& rng,
                     std::size_t n,
                     const double (&means)[2],
                     const double (&covariances)[3],
                     TDouble10Vec1Vec& samples) {
    TVector2 mean(means, means + 2);
    TMatrix2 covariance(covariances, covariances + 3);
    TDoubleVecVec samples_;
    rng.generateMultivariateNormalSamples(
        mean.toVector<TDoubleVec>(), covariance.toVectors<TDoubleVecVec>(), n, samples_);
    samples.reserve(samples.size() + samples_.size());
    for (std::size_t j = 0u; j < samples_.size(); ++j) {
        samples.push_back(TDouble10Vec(samples_[j].begin(), samples_[j].end()));
    }
    LOG_DEBUG(<< "# samples = " << samples.size());
}
}

void CMultivariateNormalConjugateTest::testMultipleUpdate() {
    LOG_DEBUG(<< "+--------------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testMultipleUpdate  |");
    LOG_DEBUG(<< "+--------------------------------------------------------+");

    maths::CSampling::seed();

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double mean[] = {10.0, 20.0};
    const double covariance[] = {3.0, 1.0, 2.0};

    test::CRandomNumbers rng;

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, 100, mean, covariance, samples);

    LOG_DEBUG(<< "****** Test vanilla ******");
    for (std::size_t i = 0; i < boost::size(dataTypes); ++i) {
        LOG_DEBUG(<< "*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateNormalConjugate<2> filter1(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(dataTypes[i]));
        maths::CMultivariateNormalConjugate<2> filter2(filter1);

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            filter1.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[j]),
                               SINGLE_UNIT_WEIGHT_2);
        }
        TDouble10Vec4Vec1Vec weights(samples.size(), UNIT_WEIGHT_2);
        filter2.addSamples(COUNT_WEIGHT, samples, weights);

        CPPUNIT_ASSERT(filter1.equalTolerance(
            filter2, maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5));
    }

    LOG_DEBUG(<< "****** Test with variance scale ******");
    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        LOG_DEBUG(<< "*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateNormalConjugate<2> filter1(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(dataTypes[i]));
        maths::CMultivariateNormalConjugate<2> filter2(filter1);

        TDouble10Vec4Vec1Vec weights;
        weights.resize(samples.size() / 2, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.5)));
        weights.resize(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 2.0)));

        for (std::size_t j = 0u; j < samples.size(); ++j) {
            TDouble10Vec1Vec sample(1, samples[j]);
            TDouble10Vec4Vec1Vec weight(1, weights[j]);
            filter1.addSamples(VARIANCE_WEIGHT, sample, weight);
        }
        filter2.addSamples(VARIANCE_WEIGHT, samples, weights);

        CPPUNIT_ASSERT(filter1.equalTolerance(
            filter2, maths::CToleranceTypes::E_RelativeTolerance, 1e-5));
    }

    // Test the count weight is equivalent to adding repeated samples.

    LOG_DEBUG(<< "****** Test count weight ******");
    for (size_t i = 0; i < boost::size(dataTypes); ++i) {
        LOG_DEBUG(<< "*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateNormalConjugate<2> filter1(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(dataTypes[i]));
        maths::CMultivariateNormalConjugate<2> filter2(filter1);

        double x = 3.0;
        std::size_t count = 10;
        for (std::size_t j = 0u; j < count; ++j) {
            filter1.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, TDouble10Vec(2, x)),
                               SINGLE_UNIT_WEIGHT_2);
        }
        TDouble10Vec1Vec sample(1, TDouble10Vec(2, x));
        TDouble10Vec4Vec1Vec weight(
            1, TDouble10Vec4Vec(1, TDouble10Vec(2, static_cast<double>(count))));
        filter2.addSamples(COUNT_WEIGHT, sample, weight);

        CPPUNIT_ASSERT(filter1.equalTolerance(
            filter2, maths::CToleranceTypes::E_AbsoluteTolerance, 1e-5));
    }
}

void CMultivariateNormalConjugateTest::testPropagation() {
    LOG_DEBUG(<< "+-----------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testPropagation  |");
    LOG_DEBUG(<< "+-----------------------------------------------------+");

    // Test that propagation doesn't affect the marginal likelihood
    // mean and expected precision.

    maths::CSampling::seed();

    const maths_t::EDataType dataTypes[] = {maths_t::E_IntegerData, maths_t::E_ContinuousData};

    const double eps = 1e-12;

    const double mean[] = {10.0, 20.0};
    const double covariance[] = {3.0, 1.0, 2.0};

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, 100, mean, covariance, samples);

    for (std::size_t i = 0u; i < boost::size(dataTypes); ++i) {
        LOG_DEBUG(<< "*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateNormalConjugate<2> filter(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(dataTypes[i], 0.1));

        TDouble10Vec4Vec1Vec weights(samples.size(), UNIT_WEIGHT_2);
        filter.addSamples(COUNT_WEIGHT, samples, weights);

        TVector2 initialMean = filter.mean();
        TMatrix2 initialPrecision = filter.precision();

        filter.propagateForwardsByTime(5.0);

        TVector2 propagatedMean = filter.mean();
        TMatrix2 propagatedPrecision = filter.precision();

        LOG_DEBUG(<< "initial mean    = " << initialMean);
        LOG_DEBUG(<< "propagated mean = " << propagatedMean);
        LOG_DEBUG(<< "initial precision    = " << initialPrecision);
        LOG_DEBUG(<< "propagated precision = " << propagatedPrecision);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, (propagatedMean - initialMean).euclidean(), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0.0, (propagatedPrecision - initialPrecision).frobenius(), eps);
    }
}

void CMultivariateNormalConjugateTest::testMeanVectorEstimation() {
    LOG_DEBUG(<< "+--------------------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testMeanVectorEstimation  |");
    LOG_DEBUG(<< "+--------------------------------------------------------------+");

    // We are going to test that we correctly estimate a distribution
    // for the mean of a multivariate normal by checking that the true
    // mean lies in various confidence intervals the correct percentage
    // of the times.

    maths::CSampling::seed();

    const double decayRates[] = {0.0, 0.001, 0.01};

    const unsigned int nt = 500u;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        LOG_DEBUG(<< "decay rate = " << decayRates[i]);

        unsigned int errors[][8] = {{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                    {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}};

        for (unsigned int t = 0; t < nt; ++t) {
            if ((t % 50) == 0) {
                LOG_DEBUG(<< "test = " << t << " / " << nt);
            }

            // Generate the samples.
            double mean_[] = {0.5 * (t + 1.0), t + 1.0};
            double covariances_[] = {40.0, 12.0, 20.0};
            TDoubleVec mean(mean_, mean_ + 2);
            TDoubleVecVec covariances;
            covariances.push_back(TDoubleVec(covariances_, covariances_ + 2));
            covariances.push_back(TDoubleVec(covariances_ + 1, covariances_ + 3));
            TDoubleVecVec samples;
            rng.generateMultivariateNormalSamples(mean, covariances, 500, samples);

            // Create the posterior.
            maths::CMultivariateNormalConjugate<2> filter(
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData, decayRates[i]));
            for (std::size_t j = 0u; j < samples.size(); ++j) {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[j]),
                                  SINGLE_UNIT_WEIGHT_2);
                filter.propagateForwardsByTime(1.0);
            }

            // Get sorted samples for each component of the mean vector.
            std::size_t n = 500;
            TVector2Vec meanSamples;
            filter.randomSampleMeanPrior(n, meanSamples);
            TDoubleVecVec componentSamples(2);
            for (std::size_t j = 0; j < meanSamples.size(); ++j) {
                componentSamples[0].push_back(meanSamples[j](0));
                componentSamples[1].push_back(meanSamples[j](1));
            }
            std::sort(componentSamples[0].begin(), componentSamples[0].end());
            std::sort(componentSamples[1].begin(), componentSamples[1].end());

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                std::size_t l = static_cast<std::size_t>(
                    static_cast<double>(n) * (0.5 - testIntervals[j] / 200.0));
                std::size_t u = static_cast<std::size_t>(
                    static_cast<double>(n) * (0.5 + testIntervals[j] / 200.0));
                for (std::size_t k = 0u; k < 2; ++k) {
                    double a = componentSamples[k][l];
                    double b = componentSamples[k][u];
                    if (mean_[k] < a || mean_[k] > b) {
                        ++errors[k][j];
                    }
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            for (std::size_t k = 0u; k < 2; ++k) {
                double interval = 100.0 * errors[k][j] / static_cast<double>(nt);

                LOG_DEBUG(<< "interval = " << interval
                          << ", expectedInterval = " << (100.0 - testIntervals[j]));

                // If the decay rate is zero the intervals should be accurate.
                // Otherwise, they should be an upper bound.
                if (decayRates[i] == 0.0) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[j]), 5.0);
                } else {
                    CPPUNIT_ASSERT(interval <= (100.0 - testIntervals[j]) + 4.0);
                }
            }
        }
    }
}

void CMultivariateNormalConjugateTest::testPrecisionMatrixEstimation() {
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "-------+");
    LOG_DEBUG(<< "|  "
                 "CMultivariateNormalConjugateTest::"
                 "testPrecisionMatrixEstimation  |");
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "-------+");

    // We are going to test that we correctly estimate a distribution
    // for the precision of a multivariate normal by checking that the
    // true precision lies in various confidence intervals the correct
    // percentage of the times.

    maths::CSampling::seed();

    const double decayRates[] = {0.0, 0.004, 0.04};

    const unsigned int nt = 500u;
    const double testIntervals[] = {50.0, 60.0, 70.0, 80.0,
                                    85.0, 90.0, 95.0, 99.0};

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < boost::size(decayRates); ++i) {
        LOG_DEBUG(<< "decay rate = " << decayRates[i]);

        unsigned int errors[][8] = {{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                    {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                                    {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}};
        std::size_t ij[][2] = {{0, 0}, {0, 1}, {1, 1}};

        TDoubleVec covariancesii;
        rng.generateUniformSamples(10.0, 20.0, 2 * nt, covariancesii);
        TDoubleVec covariancesij;
        rng.generateUniformSamples(-5.0, 5.0, 1 * nt, covariancesij);

        for (unsigned int t = 0; t < nt; ++t) {
            if ((t % 50) == 0) {
                LOG_DEBUG(<< "test = " << t << " / " << nt);
            }

            // Generate the samples.
            double mean_[] = {10.0, 10.0};
            double covariances_[] = {covariancesii[2 * t], covariancesij[t],
                                     covariancesii[2 * t + 1]};
            TDoubleVec mean(mean_, mean_ + 2);
            TDoubleVecVec covariances;
            covariances.push_back(TDoubleVec(covariances_, covariances_ + 2));
            covariances.push_back(TDoubleVec(covariances_ + 1, covariances_ + 3));
            TDoubleVecVec samples;
            rng.generateMultivariateNormalSamples(mean, covariances, 500, samples);

            // Create the posterior.
            maths::CMultivariateNormalConjugate<2> filter(
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData, decayRates[i]));
            for (std::size_t j = 0u; j < samples.size(); ++j) {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[j]),
                                  SINGLE_UNIT_WEIGHT_2);
                filter.propagateForwardsByTime(1.0);
            }

            // Get sorted samples for each element of the precision prior.
            std::size_t n = 500;
            TMatrix2Vec precisionSamples;
            filter.randomSamplePrecisionMatrixPrior(n, precisionSamples);
            TDouble10Vec4Vec elementSamples(3);
            for (std::size_t j = 0; j < precisionSamples.size(); ++j) {
                elementSamples[0].push_back(precisionSamples[j](0, 0));
                elementSamples[1].push_back(precisionSamples[j](1, 0));
                elementSamples[2].push_back(precisionSamples[j](1, 1));
            }
            std::sort(elementSamples[0].begin(), elementSamples[0].end());
            std::sort(elementSamples[1].begin(), elementSamples[1].end());
            std::sort(elementSamples[2].begin(), elementSamples[2].end());

            TMatrix2 covarianceMatrix(covariances_, covariances_ + 3);
            TMatrix2 precisionMatrix(maths::fromDenseMatrix(
                maths::toDenseMatrix(covarianceMatrix).inverse()));

            for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
                std::size_t l = static_cast<std::size_t>(
                    static_cast<double>(n) * (0.5 - testIntervals[j] / 200.0));
                std::size_t u = static_cast<std::size_t>(
                    static_cast<double>(n) * (0.5 + testIntervals[j] / 200.0));
                for (std::size_t k = 0u; k < elementSamples.size(); ++k) {
                    double a = elementSamples[k][l];
                    double b = elementSamples[k][u];
                    if (precisionMatrix(ij[k][0], ij[k][1]) < a ||
                        precisionMatrix(ij[k][0], ij[k][1]) > b) {
                        ++errors[k][j];
                    }
                }
            }
        }

        for (std::size_t j = 0; j < boost::size(testIntervals); ++j) {
            for (std::size_t k = 0u; k < boost::size(errors); ++k) {
                double interval = 100.0 * errors[k][j] / static_cast<double>(nt);

                LOG_DEBUG(<< "interval = " << interval
                          << ", expectedInterval = " << (100.0 - testIntervals[j]));

                // If the decay rate is zero the intervals should be accurate.
                // Otherwise, they should be an upper bound.
                if (decayRates[i] == 0.0) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(interval, (100.0 - testIntervals[j]), 4.0);
                } else {
                    CPPUNIT_ASSERT(interval <= (100.0 - testIntervals[j]));
                }
            }
        }
    }
}

void CMultivariateNormalConjugateTest::testMarginalLikelihood() {
    LOG_DEBUG(<< "+------------------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testMarginalLikelihood  |");
    LOG_DEBUG(<< "+------------------------------------------------------------+");

    // Test that:
    //   1) The likelihood is normalized.
    //   2) E[X] w.r.t. the likelihood is equal to the predictive distribution mean.
    //   3) E[(X - m)^2] w.r.t. the marginal likelihood is equal to the predictive
    //      distribution covariance matrix.

    maths::CSampling::seed();

    std::size_t nt = 2;

    test::CRandomNumbers rng;

    TDoubleVec meani;
    rng.generateUniformSamples(0.0, 50.0, 2 * nt, meani);
    TDoubleVec covariancesii;
    rng.generateUniformSamples(20.0, 500.0, 2 * nt, covariancesii);
    TDoubleVec covariancesij;
    rng.generateUniformSamples(-10.0, 10.0, 1 * nt, covariancesij);

    for (std::size_t t = 0u; t < nt; ++t) {
        LOG_DEBUG(<< "*** Test " << t + 1 << " ***");

        // Generate the samples.
        double mean_[] = {meani[2 * t], meani[2 * t + 1]};
        double covariances_[] = {covariancesii[2 * t], covariancesij[t],
                                 covariancesii[2 * t + 1]};
        TDoubleVec mean(mean_, mean_ + 2);
        TDoubleVecVec covariances;
        covariances.push_back(TDoubleVec(covariances_, covariances_ + 2));
        covariances.push_back(TDoubleVec(covariances_ + 1, covariances_ + 3));
        TDoubleVecVec samples;
        maths::CSampling::multivariateNormalSample(mean, covariances, 20, samples);

        maths::CMultivariateNormalConjugate<2> filter(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData));

        TMeanAccumulator meanMeanError;
        TMeanAccumulator meanCovarianceError;

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);

            if (!filter.isNonInformative()) {
                TDouble10Vec m = filter.marginalLikelihoodMean();
                TDouble10Vec10Vec v = filter.marginalLikelihoodCovariance();
                LOG_DEBUG(<< "m = " << core::CContainerPrinter::print(m));
                LOG_DEBUG(<< "v = " << core::CContainerPrinter::print(v));
                double trace = 0.0;
                for (std::size_t j = 0u; j < v.size(); ++j) {
                    trace += v[j][j];
                }
                double intervals[][2] = {
                    {m[0] - 3.0 * std::sqrt(trace), m[1] - 3.0 * std::sqrt(trace)},
                    {m[0] - 3.0 * std::sqrt(trace), m[1] - 1.0 * std::sqrt(trace)},
                    {m[0] - 3.0 * std::sqrt(trace), m[1] + 1.0 * std::sqrt(trace)},
                    {m[0] - 1.0 * std::sqrt(trace), m[1] - 3.0 * std::sqrt(trace)},
                    {m[0] - 1.0 * std::sqrt(trace), m[1] - 1.0 * std::sqrt(trace)},
                    {m[0] - 1.0 * std::sqrt(trace), m[1] + 1.0 * std::sqrt(trace)},
                    {m[0] + 1.0 * std::sqrt(trace), m[1] - 3.0 * std::sqrt(trace)},
                    {m[0] + 1.0 * std::sqrt(trace), m[1] - 1.0 * std::sqrt(trace)},
                    {m[0] + 1.0 * std::sqrt(trace), m[1] + 1.0 * std::sqrt(trace)}};

                TVector2 expectedMean(m.begin(), m.end());
                double elements[] = {v[0][0], v[0][1], v[1][1]};
                TMatrix2 expectedCovariance(elements, elements + 3);

                CUnitKernel<2> likelihoodKernel(filter);
                CMeanKernel<2> meanKernel(filter);
                CCovarianceKernel<2> covarianceKernel(filter, expectedMean);

                double z = 0.0;
                TVector2 actualMean(0.0);
                TMatrix2 actualCovariance(0.0);
                for (std::size_t j = 0u; j < boost::size(intervals); ++j) {
                    TDoubleVec a(boost::begin(intervals[j]), boost::end(intervals[j]));
                    TDoubleVec b(a);
                    b[0] += 2.0 * std::sqrt(trace);
                    b[1] += 2.0 * std::sqrt(trace);

                    double zj;
                    maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix, maths::CIntegration::TwoDimensions>(
                        likelihoodKernel, a, b, zj);
                    TVector2 mj;
                    maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix, maths::CIntegration::TwoDimensions>(
                        meanKernel, a, b, mj);
                    TMatrix2 cj;
                    maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix, maths::CIntegration::TwoDimensions>(
                        covarianceKernel, a, b, cj);

                    z += zj;
                    actualMean += mj;
                    actualCovariance += cj;
                }

                LOG_DEBUG(<< "Z = " << z);
                LOG_DEBUG(<< "mean = " << actualMean);
                LOG_DEBUG(<< "covariance = " << actualCovariance);

                TVector2 meanError = actualMean - expectedMean;
                TMatrix2 covarianceError = actualCovariance - expectedCovariance;
                CPPUNIT_ASSERT(meanError.euclidean() < expectedMean.euclidean());
                CPPUNIT_ASSERT(covarianceError.frobenius() < expectedCovariance.frobenius());

                meanMeanError.add(meanError.euclidean() / expectedMean.euclidean());
                meanCovarianceError.add(covarianceError.frobenius() /
                                        expectedCovariance.frobenius());
            }
        }

        LOG_DEBUG(<< "Mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
        LOG_DEBUG(<< "Mean covariance error = "
                  << maths::CBasicStatistics::mean(meanCovarianceError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.12);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanCovarianceError) < 0.07);
    }
}

void CMultivariateNormalConjugateTest::testMarginalLikelihoodMode() {
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "----+");
    LOG_DEBUG(<< "|  "
                 "CMultivariateNormalConjugateTest::testMarginalLikelihoodMode "
                 " |");
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "----+");

    // Test that the marginal likelihood mode is at a stationary maximum
    // of the likelihood function.

    const double mean[] = {10.0, 20.0};
    const double covariance[] = {3.0, 1.0, 2.0};

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, 100, mean, covariance, samples);

    maths::CMultivariateNormalConjugate<2> filter(
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData));
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
    }
    LOG_DEBUG(<< "prior = " << filter.print());

    TDouble10Vec mode = filter.marginalLikelihoodMode(COUNT_WEIGHT, UNIT_WEIGHT_2);

    TDoubleVec epsilons;
    rng.generateUniformSamples(-0.01, 0.01, 10, epsilons);
    for (std::size_t i = 0u; i < epsilons.size(); i += 2) {
        TDouble10Vec1Vec modeMinusEps(1, TDouble10Vec(2));
        TDouble10Vec1Vec modePlusEps(1, TDouble10Vec(2));
        double norm = 0.0;
        for (std::size_t j = 0u; j < 2; ++j) {
            double eps = epsilons[i + j];
            modeMinusEps[0][j] = mode[j] - eps;
            modePlusEps[0][j] = mode[j] + eps;
            norm += eps * eps;
        }
        LOG_DEBUG(<< "mode - eps = " << core::CContainerPrinter::print(modeMinusEps));
        LOG_DEBUG(<< "mode + eps = " << core::CContainerPrinter::print(modePlusEps));
        norm = std::sqrt(norm);

        double llm, ll, llp;
        filter.jointLogMarginalLikelihood(COUNT_WEIGHT, modeMinusEps,
                                          SINGLE_UNIT_WEIGHT_2, llm);
        filter.jointLogMarginalLikelihood(COUNT_WEIGHT, TDouble10Vec1Vec(1, mode),
                                          SINGLE_UNIT_WEIGHT_2, ll);
        filter.jointLogMarginalLikelihood(COUNT_WEIGHT, modePlusEps,
                                          SINGLE_UNIT_WEIGHT_2, llp);
        double gradient = std::fabs(std::exp(llp) - std::exp(llm)) / norm;
        LOG_DEBUG(<< "gradient = " << gradient);
        CPPUNIT_ASSERT(gradient < 1e-6);
        CPPUNIT_ASSERT(ll > llm && ll > llp);
    }
}

void CMultivariateNormalConjugateTest::testSampleMarginalLikelihood() {
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "------+");
    LOG_DEBUG(<< "|  "
                 "CMultivariateNormalConjugateTest::"
                 "testSampleMarginalLikelihood  |");
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "------+");

    // We're going to test three properties of the sampling:
    //   1) That the sample mean is equal to the marginal likelihood mean.
    //   2) The sample variance is close to the marginal likelihood variance.
    //   3) The number of samples whose likelihood exceeds a threshold is near
    //      the expected value for the true distribution.

    test::CRandomNumbers rng;

    const double mean_[] = {50.0, 20.0};
    const double covariance_[] = {8.0, 3.0, 5.0};

    TVector2 mean(mean_);
    TMatrix2 covariance(covariance_, covariance_ + 3);

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, 50, mean_, covariance_, samples);

    maths::CMultivariateNormalConjugate<2> filter(
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData));

    std::size_t i = 0u;
    for (/**/; i < samples.size(); ++i) {
        if (!filter.isNonInformative()) {
            break;
        }

        TDouble10Vec1Vec resamples;
        filter.sampleMarginalLikelihood(40, resamples);
        if (filter.numberSamples() == 0) {
            CPPUNIT_ASSERT(resamples.empty());
        } else {
            CPPUNIT_ASSERT(resamples.size() == 1);
            CPPUNIT_ASSERT_EQUAL(
                core::CContainerPrinter::print(filter.marginalLikelihoodMean()),
                core::CContainerPrinter::print(resamples[0]));
        }

        filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
    }

    TDoubleVec p;
    empiricalProbabilityOfLessLikelySamples(
        mean.toVector<TDoubleVec>(), covariance.toVectors<TDoubleVecVec>(), p);

    TMeanAccumulator pAbsError;
    TMeanAccumulator pRelError;

    for (/**/; i < samples.size(); ++i) {
        maths::CBasicStatistics::SSampleCovariances<double, 2> covariances;

        TVector2 likelihoodMean(filter.marginalLikelihoodMean());
        TMatrix2 likelihoodCov(filter.marginalLikelihoodCovariance());

        TDouble10Vec1Vec resamples;
        filter.sampleMarginalLikelihood(40, resamples);
        for (std::size_t j = 0u; j < resamples.size(); ++j) {
            covariances.add(TVector2(resamples[j]));
        }

        TVector2 sampleMean = maths::CBasicStatistics::mean(covariances);
        TMatrix2 sampleCov = maths::CBasicStatistics::covariances(covariances);

        LOG_DEBUG(<< "likelihood mean = " << likelihoodMean);
        LOG_DEBUG(<< "sample mean     = " << sampleMean);
        LOG_DEBUG(<< "likelihood cov  = " << likelihoodCov);
        LOG_DEBUG(<< "sample cov      = " << sampleCov);

        CPPUNIT_ASSERT(
            (sampleMean - likelihoodMean).euclidean() / likelihoodMean.euclidean() < 1e-6);
        CPPUNIT_ASSERT((sampleCov - likelihoodCov).frobenius() / likelihoodCov.frobenius() < 0.01);

        TDoubleVec sampleProbabilities;
        for (std::size_t j = 0u; j < resamples.size(); ++j) {
            double ll;
            filter.jointLogMarginalLikelihood(COUNT_WEIGHT,
                                              TDouble10Vec1Vec(1, resamples[j]),
                                              SINGLE_UNIT_WEIGHT_2, ll);
            sampleProbabilities.push_back(
                static_cast<double>(std::lower_bound(p.begin(), p.end(), ll) - p.begin()) /
                static_cast<double>(p.size()));
        }
        std::sort(sampleProbabilities.begin(), sampleProbabilities.end());
        LOG_DEBUG(<< "sample p = " << core::CContainerPrinter::print(sampleProbabilities));

        for (std::size_t j = 0u; j < sampleProbabilities.size(); ++j) {
            double expectedProbability = static_cast<double>(j + 1) /
                                         static_cast<double>(sampleProbabilities.size());
            double error = std::fabs(sampleProbabilities[j] - expectedProbability);
            pAbsError.add(error);
            pRelError.add(error / expectedProbability);
        }
        filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
    }

    LOG_DEBUG(<< "pAbsError = " << maths::CBasicStatistics::mean(pAbsError));
    LOG_DEBUG(<< "pRelError = " << maths::CBasicStatistics::mean(pRelError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(pAbsError) < 0.15);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(pRelError) < 0.3);
}

void CMultivariateNormalConjugateTest::testProbabilityOfLessLikelySamples() {
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "------------+");
    LOG_DEBUG(<< "|  "
                 "CMultivariateNormalConjugateTest::"
                 "testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG(<< "+------------------------------------------------------------"
                 "------------+");

    // Test that the probability is approximately equal to the chance of drawing
    // a less likely sample from generating distribution.

    maths::CSampling::seed();

    const double means[][2] = {
        {-10.0, -100.0},
        {0.0, 0.0},
        {100.0, 50.0},
    };
    const double covariances[][3] = {
        {10.0, 0.0, 10.0}, {10.0, 9.0, 10.0}, {10.0, -9.0, 10.0}};
    const double offsets[][2] = {{0.0, 0.0},  {0.0, 6.0},  {4.0, 0.0},
                                 {6.0, 6.0},  {6.0, -6.0}, {-8.0, 8.0},
                                 {-8.0, -8.0}};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        TDoubleVec mean(means[i], means[i] + 2);
        LOG_DEBUG(<< "mean = " << core::CContainerPrinter::print(mean));

        for (std::size_t j = 0u; j < boost::size(covariances); ++j) {
            TDoubleVecVec covariance;
            covariance.push_back(TDoubleVec(covariances[j], covariances[j] + 2));
            covariance.push_back(TDoubleVec(covariances[j] + 1, covariances[j] + 3));
            LOG_DEBUG(<< "covariances = " << core::CContainerPrinter::print(covariance));

            TDoubleVecVec samples;
            rng.generateMultivariateNormalSamples(mean, covariance, 500, samples);

            maths::CMultivariateNormalConjugate<2> filter(
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData));
            for (std::size_t k = 0u; k < samples.size(); ++k) {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[k]),
                                  SINGLE_UNIT_WEIGHT_2);
            }

            TDoubleVec p;
            empiricalProbabilityOfLessLikelySamples(mean, covariance, p);

            TMeanAccumulator meanAbsError;
            TMeanAccumulator meanRelError;

            for (std::size_t k = 0u; k < boost::size(offsets); ++k) {
                TVector2 x = TVector2(mean) + TVector2(offsets[k]);

                double ll;
                maths::gaussianLogLikelihood(TMatrix2(covariance),
                                             TVector2(offsets[k]), ll);
                double px = static_cast<double>(
                                std::lower_bound(p.begin(), p.end(), ll) - p.begin()) /
                            static_cast<double>(p.size());

                double lb, ub;
                maths::CMultivariatePrior::TTail10Vec tail;
                filter.probabilityOfLessLikelySamples(
                    maths_t::E_TwoSided, COUNT_WEIGHT,
                    TDouble10Vec1Vec(1, x.toVector<TDouble10Vec>()),
                    SINGLE_UNIT_WEIGHT_2, lb, ub, tail);
                double pa = (lb + ub) / 2.0;

                LOG_DEBUG(<< "  p(" << x << "), actual = " << pa << ", expected = " << px);
                meanAbsError.add(std::fabs(px - pa));
                if (px < 1.0 && px > 0.0) {
                    meanRelError.add(std::fabs(std::log(px) - std::log(pa)) /
                                     std::fabs(std::log(px)));
                }
            }

            LOG_DEBUG(<< "mean absolute error = "
                      << maths::CBasicStatistics::mean(meanAbsError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanAbsError) < 0.018);

            LOG_DEBUG(<< "mean relative error = "
                      << maths::CBasicStatistics::mean(meanRelError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelError) < 0.15);
        }
    }
}

void CMultivariateNormalConjugateTest::testIntegerData() {
    LOG_DEBUG(<< "+-----------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testIntegerData  |");
    LOG_DEBUG(<< "+-----------------------------------------------------+");

    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // The idea of this test is to check that the inferred model agrees in the
    // limit (large n) with the model inferred from such data.

    const double means[][2] = {
        {-10.0, -100.0},
        {0.0, 0.0},
        {100.0, 50.0},
    };
    const double covariances[][3] = {{10.0, 0.0, 10.0}, {10.0, 9.0, 10.0}};
    const std::size_t n = 10000u;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        TVector2 mean(means[i], means[i] + 2);
        for (std::size_t j = 0u; j < boost::size(covariances); ++j) {
            TMatrix2 covariance(covariances[j], covariances[j] + 3);

            TDoubleVecVec samples;
            rng.generateMultivariateNormalSamples(
                mean.toVector<TDoubleVec>(),
                covariance.toVectors<TDoubleVecVec>(), n, samples);

            TDoubleVecVec uniform;
            TDoubleVec uniform_;
            rng.generateUniformSamples(0.0, 1.0, 2 * n, uniform_);
            for (std::size_t k = 0u; k < uniform_.size(); k += 2) {
                uniform.push_back(TDoubleVec(&uniform_[k], &uniform_[k] + 2));
            }

            maths::CMultivariateNormalConjugate<2> filter1(
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_IntegerData));
            maths::CMultivariateNormalConjugate<2> filter2(
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData));

            for (std::size_t k = 0u; k < n; ++k) {
                TVector2 x(samples[k]);
                TDouble10Vec1Vec sample(1, x.toVector<TDouble10Vec>());
                filter1.addSamples(COUNT_WEIGHT, sample, SINGLE_UNIT_WEIGHT_2);
                sample[0] = (x + TVector2(uniform[k])).toVector<TDouble10Vec>();
                filter2.addSamples(COUNT_WEIGHT, sample, SINGLE_UNIT_WEIGHT_2);
            }

            CPPUNIT_ASSERT(filter1.equalTolerance(
                filter2, maths::CToleranceTypes::E_RelativeTolerance | maths::CToleranceTypes::E_AbsoluteTolerance,
                0.005));

            TMeanAccumulator meanLogLikelihood1;
            TMeanAccumulator meanLogLikelihood2;
            for (std::size_t k = 0u; k < n; ++k) {
                TVector2 x(samples[k]);

                TDouble10Vec1Vec sample(1, x.toVector<TDouble10Vec>());
                double ll1;
                filter1.jointLogMarginalLikelihood(COUNT_WEIGHT, sample,
                                                   SINGLE_UNIT_WEIGHT_2, ll1);
                meanLogLikelihood1.add(-ll1);

                sample[0] = (x + TVector2(uniform[k])).toVector<TDouble10Vec>();
                double ll2;
                filter2.jointLogMarginalLikelihood(COUNT_WEIGHT, sample,
                                                   SINGLE_UNIT_WEIGHT_2, ll2);
                meanLogLikelihood2.add(-ll2);
            }

            LOG_DEBUG(<< "meanLogLikelihood1 = " << maths::CBasicStatistics::mean(meanLogLikelihood1)
                      << ", meanLogLikelihood2 = "
                      << maths::CBasicStatistics::mean(meanLogLikelihood2));

            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                maths::CBasicStatistics::mean(meanLogLikelihood1),
                maths::CBasicStatistics::mean(meanLogLikelihood2), 0.03);
        }
    }
}

void CMultivariateNormalConjugateTest::testLowVariationData() {
    LOG_DEBUG(<< "+----------------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testLowVariationData  |");
    LOG_DEBUG(<< "+----------------------------------------------------------+");

    {
        maths::CMultivariateNormalConjugate<2> filter(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_IntegerData));
        for (std::size_t i = 0u; i < 100; ++i) {
            filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, TDouble10Vec(2, 430.0)),
                              SINGLE_UNIT_WEIGHT_2);
        }

        TDouble10Vec10Vec covariances = filter.marginalLikelihoodCovariance();
        LOG_DEBUG(<< "covariance matrix " << core::CContainerPrinter::print(covariances));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            12.0, 2.0 / (covariances[0][0] + covariances[1][1]), 0.3);
    }
    {
        maths::CMultivariateNormalConjugate<2> filter(
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData));
        for (std::size_t i = 0u; i < 100; ++i) {
            filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, TDouble10Vec(2, 430.0)),
                              SINGLE_UNIT_WEIGHT_2);
        }

        TDouble10Vec10Vec covariances = filter.marginalLikelihoodCovariance();
        LOG_DEBUG(<< "covariance matrix " << core::CContainerPrinter::print(covariances));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            1.0 / maths::MINIMUM_COEFFICIENT_OF_VARIATION / std::sqrt(2.0) / 430.5,
            std::sqrt(2.0 / (covariances[0][0] + covariances[1][1])), 0.4);
    }
}

void CMultivariateNormalConjugateTest::testPersist() {
    LOG_DEBUG(<< "+-------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::testPersist  |");
    LOG_DEBUG(<< "+-------------------------------------------------+");

    // Check that persist/restore is idempotent.

    const double mean[] = {10.0, 20.0};
    const double covariance[] = {3.0, 1.0, 2.0};

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, 100, mean, covariance, samples);

    maths_t::EDataType dataType = maths_t::E_ContinuousData;

    maths::CMultivariateNormalConjugate<2> origFilter(
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(dataType));
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        origFilter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
    }
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Normal mean conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(
        dataType, decayRate + 0.1, maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
        maths::MINIMUM_CLUSTER_SPLIT_COUNT, maths::MINIMUM_CATEGORY_COUNT);
    maths::CMultivariateNormalConjugate<2> restoredFilter(params, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredFilter.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum, restoredFilter.checksum());

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredFilter.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CMultivariateNormalConjugateTest::calibrationExperiment() {
    LOG_DEBUG(<< "+-----------------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::calibrationExperiment  |");
    LOG_DEBUG(<< "+-----------------------------------------------------------+");

    using TVector10 = maths::CVectorNx1<double, 10>;
    using TMatrix10 = maths::CSymmetricMatrixNxN<double, 10>;

    double means[] = {10.0, 10.0, 20.0, 20.0, 30.0,
                      20.0, 10.0, 40.0, 30.0, 20.0};
    double covariances[] = {
        10.0, 9.0, 10.0, -5.0, 1.0,  6.0,  -8.0, 9.0, 4.0, 20.0, 8.0,
        3.0,  1.0, 12.0, 12.0, -4.0, 2.0,  1.0,  1.0, 4.0, 4.0,  5.0,
        1.0,  3.0, 8.0,  10.0, 3.0,  10.0, 9.0,  9.0, 5.0, 19.0, 11.0,
        3.0,  9.0, 25.0, 5.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0,  0.0,
        20.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0,  1.0};
    TVector10 mean(means, means + boost::size(means));
    TMatrix10 covariance(covariances, covariances + boost::size(covariances));

    test::CRandomNumbers rng;
    TDoubleVecVec samples_;
    rng.generateMultivariateNormalSamples(mean.toVector<TDoubleVec>(),
                                          covariance.toVectors<TDoubleVecVec>(),
                                          2000, samples_);

    TDouble10Vec1Vec samples;
    samples.reserve(samples.size() + samples_.size());
    for (std::size_t j = 0u; j < samples_.size(); ++j) {
        samples.push_back(TDouble10Vec(samples_[j].begin(), samples_[j].end()));
    }

    maths::CMultivariateNormalConjugate<2> filters[] = {
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData),
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData)};
    std::size_t indices[][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5},
                                {0, 6}, {0, 7}, {0, 8}, {0, 9}};

    for (std::size_t i = 0u; i < 200; ++i) {
        for (std::size_t j = 0u; j < boost::size(filters); ++j) {
            TDouble10Vec1Vec sample(1, TDouble10Vec(2));
            sample[0][0] = samples[i][indices[j][0]];
            sample[0][1] = samples[i][indices[j][1]];
            filters[j].addSamples(COUNT_WEIGHT, sample, SINGLE_UNIT_WEIGHT_2);
        }
    }

    TDoubleVecVec p(boost::size(filters));
    TDoubleVec mp;
    TDoubleVec ep;
    for (std::size_t i = 200u; i < 2000; ++i) {
        double mpi = 1.0;
        maths::CProbabilityOfExtremeSample epi;
        for (std::size_t j = 0u; j < boost::size(filters); ++j) {
            TDouble10Vec1Vec sample(1, TDouble10Vec(2));
            sample[0][0] = samples[i][indices[j][0]];
            sample[0][1] = samples[i][indices[j][1]];
            double lb, ub;
            maths::CMultivariatePrior::TTail10Vec tail;
            filters[j].probabilityOfLessLikelySamples(maths_t::E_TwoSided, COUNT_WEIGHT,
                                                      sample, SINGLE_UNIT_WEIGHT_2,
                                                      lb, ub, tail);
            p[j].push_back((lb + ub) / 2.0);
            mpi = std::min(mpi, (lb + ub) / 2.0);
            epi.add((lb + ub) / 2.0, 0.5);
        }
        mp.push_back(mpi);
        double pi;
        epi.calculate(pi);
        ep.push_back(pi);
    }

    for (std::size_t i = 0u; i < p.size(); ++i) {
        std::sort(p[i].begin(), p[i].end());
    }
    std::sort(mp.begin(), mp.end());
    std::sort(ep.begin(), ep.end());

    double test[] = {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};
    for (std::size_t i = 0u; i < boost::size(test); ++i) {
        for (std::size_t j = 0u; j < p.size(); ++j) {
            LOG_DEBUG(<< j << ") " << test[i] << " "
                      << static_cast<double>(
                             std::lower_bound(p[j].begin(), p[j].end(), test[i]) -
                             p[j].begin()) /
                             static_cast<double>(p[j].size()));
        }
        LOG_DEBUG(<< "min " << test[i] << " "
                  << static_cast<double>(
                         std::lower_bound(mp.begin(), mp.end(), test[i]) - mp.begin()) /
                         static_cast<double>(mp.size()));
        LOG_DEBUG(<< "corrected min " << test[i] << " "
                  << static_cast<double>(
                         std::lower_bound(ep.begin(), ep.end(), test[i]) - ep.begin()) /
                         static_cast<double>(ep.size()));
    }
}

void CMultivariateNormalConjugateTest::dataGenerator() {
    LOG_DEBUG(<< "+---------------------------------------------------+");
    LOG_DEBUG(<< "|  CMultivariateNormalConjugateTest::dataGenerator  |");
    LOG_DEBUG(<< "+---------------------------------------------------+");

    const double means[][2] = {{10.0, 20.0}, {30.0, 25.0}, {50.0, 5.0}, {100.0, 50.0}};
    const double covariances[][3] = {
        {3.0, 2.0, 2.0}, {6.0, -4.0, 5.0}, {4.0, 1.0, 3.0}, {20.0, -12.0, 12.0}};

    double anomalies[][4] = {{7000.0, 0.0, 2.8, -2.8}, {7001.0, 0.0, 2.8, -2.8},
                             {7002.0, 0.0, 2.8, -2.8}, {7003.0, 0.0, 2.8, -2.8},
                             {8000.0, 3.0, 3.5, 4.9},  {8001.0, 3.0, 3.5, 4.9},
                             {8002.0, 3.0, 3.5, 4.9},  {8003.0, 3.0, 3.5, 4.9},
                             {8004.0, 3.0, 3.5, 4.9},  {8005.0, 3.0, 3.5, 4.9}};

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples[4];
    for (std::size_t i = 0u; i < boost::size(means); ++i) {
        gaussianSamples(rng, 10000, means[i], covariances[i], samples[i]);
    }
    for (std::size_t i = 0u; i < boost::size(anomalies); ++i) {
        std::size_t j = static_cast<std::size_t>(anomalies[i][1]);
        std::size_t k = static_cast<std::size_t>(anomalies[i][0]);
        samples[j][k][0] += anomalies[i][2];
        samples[j][k][1] += anomalies[i][3];
    }

    std::ofstream f;
    f.open("four_2d_gaussian.csv");
    core_t::TTime time = 1451606400;
    for (std::size_t i = 0u; i < 10000; ++i, time += 30) {
        for (std::size_t j = 0u; j < boost::size(samples); ++j) {
            f << time << ",x" << 2 * j << "," << samples[j][i][0] << "\n";
            f << time << ",x" << 2 * j + 1 << "," << samples[j][i][1] << "\n";
        }
    }
    f.close();
}

CppUnit::Test* CMultivariateNormalConjugateTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMultivariateNormalConjugateTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testMultipleUpdate",
        &CMultivariateNormalConjugateTest::testMultipleUpdate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testPropagation",
        &CMultivariateNormalConjugateTest::testPropagation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testMeanVectorEstimation",
        &CMultivariateNormalConjugateTest::testMeanVectorEstimation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testPrecisionMatrixEstimation",
        &CMultivariateNormalConjugateTest::testPrecisionMatrixEstimation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testMarginalLikelihood",
        &CMultivariateNormalConjugateTest::testMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testMarginalLikelihoodMode",
        &CMultivariateNormalConjugateTest::testMarginalLikelihoodMode));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testSampleMarginalLikelihood",
        &CMultivariateNormalConjugateTest::testSampleMarginalLikelihood));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testProbabilityOfLessLikelySamples",
        &CMultivariateNormalConjugateTest::testProbabilityOfLessLikelySamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testIntegerData",
        &CMultivariateNormalConjugateTest::testIntegerData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testLowVariationData",
        &CMultivariateNormalConjugateTest::testLowVariationData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
        "CMultivariateNormalConjugateTest::testPersist",
        &CMultivariateNormalConjugateTest::testPersist));
    //suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
    //                               "CMultivariateNormalConjugateTest::calibrationExperiment",
    //                               &CMultivariateNormalConjugateTest::calibrationExperiment) );
    //suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateNormalConjugateTest>(
    //                               "CMultivariateNormalConjugateTest::dataGenerator",
    //                               &CMultivariateNormalConjugateTest::dataGenerator) );

    return suiteOfTests;
}
