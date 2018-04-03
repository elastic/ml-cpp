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

#include "CMultivariateOneOfNPriorTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CGradientDescent.h>
#include <maths/CIntegration.h>
#include <maths/CMultivariateMultimodalPrior.h>
#include <maths/CMultivariateNormalConjugate.h>
#include <maths/CMultivariateOneOfNPrior.h>
#include <maths/CRestoreParams.h>
#include <maths/CXMeansOnline.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include "TestUtils.h"

#include <cmath>
#include <vector>

using namespace ml;
using namespace handy_typedefs;

namespace
{

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TPriorPtr = maths::CMultivariateOneOfNPrior::TPriorPtr;
using TPriorPtrVec = maths::CMultivariateOneOfNPrior::TPriorPtrVec;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

const maths_t::TWeightStyleVec COUNT_WEIGHT(1, maths_t::E_SampleCountWeight);
const maths_t::TWeightStyleVec VARIANCE_WEIGHT(1, maths_t::E_SampleCountVarianceScaleWeight);
const TDouble10Vec4Vec UNIT_WEIGHT_2(1, TDouble10Vec(2, 1.0));
const TDouble10Vec4Vec1Vec SINGLE_UNIT_WEIGHT_2(1, UNIT_WEIGHT_2);

class CMinusLogLikelihood : public maths::CGradientDescent::CFunction
{
    public:
        CMinusLogLikelihood(const maths::CMultivariateOneOfNPrior &prior) :
                m_Prior(&prior)
        {}

        bool operator()(const maths::CGradientDescent::TVector &x, double &result) const
        {
            if (m_Prior->jointLogMarginalLikelihood(
                                 COUNT_WEIGHT,
                                 TDouble10Vec1Vec(1, TDouble10Vec(x.toVector<TDouble10Vec>())),
                                 SINGLE_UNIT_WEIGHT_2,
                                 result) == maths_t::E_FpNoErrors)
            {
                result = -result;
                return true;
            }
            return false;
        }

    private:
        const maths::CMultivariateOneOfNPrior *m_Prior;
};

template<std::size_t N>
maths::CMultivariateMultimodalPrior<N> makeMultimodal(maths_t::EDataType dataType,
                                                      double decayRate = 0.0)
{
    maths::CXMeansOnline<maths::CFloatStorage, N> clusterer(dataType,
                                                            maths_t::E_ClustersFractionWeight,
                                                            decayRate);
    return maths::CMultivariateMultimodalPrior<N>(
                   dataType,
                   clusterer,
                   maths::CMultivariateNormalConjugate<N>::nonInformativePrior(dataType, decayRate),
                   decayRate);
}

template<std::size_t N>
maths::CMultivariateOneOfNPrior makeOneOfN(maths_t::EDataType dataType,
                                           double decayRate = 0.0)
{
    TPriorPtrVec priors;
    priors.push_back(TPriorPtr(maths::CMultivariateNormalConjugate<N>::nonInformativePrior(dataType, decayRate).clone()));
    priors.push_back(TPriorPtr(makeMultimodal<N>(dataType, decayRate).clone()));
    return maths::CMultivariateOneOfNPrior(N, priors, dataType, decayRate);
}

void gaussianSamples(test::CRandomNumbers &rng,
                     std::size_t modes,
                     const std::size_t *n,
                     const double (*means)[2],
                     const double (*covariances)[3],
                     TDouble10Vec1Vec &samples)
{
    for (std::size_t i = 0u; i < modes; ++i)
    {
        TVector2 mean(means[i], means[i] + 2);
        TMatrix2 covariance(covariances[i], covariances[i] + 3);
        TDoubleVecVec samples_;
        rng.generateMultivariateNormalSamples(mean.toVector<TDoubleVec>(),
                                              covariance.toVectors<TDoubleVecVec>(),
                                              n[i],
                                              samples_);
        samples.reserve(samples.size() + samples_.size());
        for (std::size_t j = 0u; j < samples_.size(); ++j)
        {
            samples.push_back(TDouble10Vec(samples_[j].begin(), samples_[j].end()));
        }
    }
    LOG_DEBUG("# samples = " << samples.size());
}

double sum(const TDoubleVec &x)
{
    return std::accumulate(x.begin(), x.end(), 0.0);
}

std::string print(maths_t::EDataType dataType)
{
    switch (dataType)
    {
    case maths_t::E_DiscreteData:   return "Discrete";
    case maths_t::E_IntegerData:    return "Integer";
    case maths_t::E_ContinuousData: return "Continuous";
    case maths_t::E_MixedData:      return "Mixed";
    }
    return "";
}

}

void CMultivariateOneOfNPriorTest::testMultipleUpdate(void)
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testMultipleUpdate  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    maths::CSampling::CScopeMockRandomNumberGenerator scopeMockRng;

    const maths_t::EDataType dataTypes[] =
        {
            maths_t::E_IntegerData,
            maths_t::E_ContinuousData
        };

    const double mean_[] = { 10.0, 20.0 };
    const double covariance_[] = { 20.0, 1.0, 16.0 };

    TVector2 mean(mean_);
    TMatrix2 covariance(covariance_, covariance_ + 3);

    test::CRandomNumbers rng;

    TDouble10Vec1Vec seedSamples;
    TDouble10Vec1Vec samples;
    {
        TDoubleVecVec samples_;
        rng.generateMultivariateNormalSamples(mean.toVector<TDoubleVec>(),
                                              covariance.toVectors<TDoubleVecVec>(),
                                              100,
                                              samples_);
        seedSamples.reserve(10);
        for (std::size_t i = 0u; i < 10; ++i)
        {
            seedSamples.push_back(TDouble10Vec(samples_[i].begin(), samples_[i].end()));
        }
        samples.reserve(samples_.size() - 10);
        for (std::size_t i = 10u; i < samples_.size(); ++i)
        {
            samples.push_back(TDouble10Vec(samples_[i].begin(), samples_[i].end()));
        }
    }

    LOG_DEBUG("****** Test vanilla ******");
    for (std::size_t i = 0u; i < boost::size(dataTypes); ++i)
    {
        LOG_DEBUG("*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateOneOfNPrior filter1(makeOneOfN<2>(dataTypes[i]));
        maths::CMultivariateOneOfNPrior filter2(filter1);

        for (std::size_t j = 0u; j < seedSamples.size(); ++j)
        {
            TDouble10Vec1Vec sample(1, seedSamples[j]);
            TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
            filter1.addSamples(COUNT_WEIGHT, sample, weight);
            filter2.addSamples(COUNT_WEIGHT, sample, weight);
        }
        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            TDouble10Vec1Vec sample(1, samples[j]);
            TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
            filter1.addSamples(COUNT_WEIGHT, sample, weight);
        }
        TDouble10Vec4Vec1Vec weights(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
        filter2.addSamples(COUNT_WEIGHT, samples, weights);

        LOG_DEBUG("checksum 1 " << filter1.checksum());
        LOG_DEBUG("checksum 2 " << filter2.checksum());
        CPPUNIT_ASSERT_EQUAL(filter1.checksum(), filter2.checksum());
    }

    LOG_DEBUG("****** Test with variance scale ******");
    for (size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        LOG_DEBUG("*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateOneOfNPrior filter1(makeOneOfN<2>(dataTypes[i]));
        maths::CMultivariateOneOfNPrior filter2(filter1);

        for (std::size_t j = 0u; j < seedSamples.size(); ++j)
        {
            TDouble10Vec1Vec sample(1, seedSamples[j]);
            TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
            filter1.addSamples(COUNT_WEIGHT, sample, weight);
            filter2.addSamples(COUNT_WEIGHT, sample, weight);
        }
        TDouble10Vec4Vec1Vec weights;
        weights.resize(samples.size() / 2, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.5)));
        weights.resize(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 2.0)));
        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            TDouble10Vec1Vec sample(1, samples[j]);
            TDouble10Vec4Vec1Vec weight(1, weights[j]);
            filter1.addSamples(VARIANCE_WEIGHT, sample, weight);
        }
        filter2.addSamples(VARIANCE_WEIGHT, samples, weights);

        LOG_DEBUG("checksum 1 " << filter1.checksum());
        LOG_DEBUG("checksum 2 " << filter2.checksum());
        CPPUNIT_ASSERT_EQUAL(filter1.checksum(), filter2.checksum());
    }
}

void CMultivariateOneOfNPriorTest::testPropagation(void)
{
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testPropagation  |");
    LOG_DEBUG("+-------------------------------------------------+");

    // Test that propagation doesn't affect the marginal likelihood
    // mean and the marginal likelihood variance increases (due to
    // influence of the prior uncertainty) after propagation.

    maths::CSampling::seed();

    const double eps = 2e-3;

    const std::size_t n[] = { 400, 600 };
    const double means[][2] =
        {
            { 10.0, 10.0 },
            { 20.0, 20.0 }
        };
    const double covariances[][3] =
        {
            {  8.0,  1.0, 8.0 },
            { 20.0, -4.0, 10.0 }
        };

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);
    rng.random_shuffle(samples.begin(), samples.end());
    LOG_DEBUG("# samples = " << samples.size());

    const double decayRate = 0.1;

    maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData, decayRate));
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(COUNT_WEIGHT,
                          TDouble10Vec1Vec(1, samples[i]),
                          TDouble10Vec4Vec1Vec(1, UNIT_WEIGHT_2));
    }

    double numberSamples = filter.numberSamples();
    TDouble10Vec mean = filter.marginalLikelihoodMean();
    TDouble10Vec10Vec covariance = filter.marginalLikelihoodCovariance();
    double logWeightRatio = std::fabs(filter.logWeights()[0] - filter.logWeights()[1]);

    filter.propagateForwardsByTime(40.0);

    double propagatedNumberSamples = filter.numberSamples();
    TDouble10Vec propagatedMean = filter.marginalLikelihoodMean();
    TDouble10Vec10Vec propagatedCovariance = filter.marginalLikelihoodCovariance();
    double propagatedLogWeightRatio = std::fabs(filter.logWeights()[0] - filter.logWeights()[1]);

    LOG_DEBUG("numberSamples           = " << numberSamples);
    LOG_DEBUG("propagatedNumberSamples = " << propagatedNumberSamples);
    LOG_DEBUG("mean           = " << core::CContainerPrinter::print(mean));
    LOG_DEBUG("propagatedMean = " << core::CContainerPrinter::print(propagatedMean));
    LOG_DEBUG("covariance           = " << core::CContainerPrinter::print(covariance));
    LOG_DEBUG("propagatedCovariance = " << core::CContainerPrinter::print(propagatedCovariance));
    LOG_DEBUG("logWeightRatio           = " << logWeightRatio);
    LOG_DEBUG("propagatedLogWeightRatio = " << propagatedLogWeightRatio);

    CPPUNIT_ASSERT(propagatedNumberSamples < numberSamples);
    CPPUNIT_ASSERT((TVector2(propagatedMean) - TVector2(mean)).euclidean() < eps * TVector2(mean).euclidean());
    Eigen::MatrixXd c(2, 2);
    Eigen::MatrixXd cp(2, 2);
    for (std::size_t i = 0u; i < 2; ++i)
    {
        for (std::size_t j = 0u; j < 2; ++j)
        {
            c(i,j)  = covariance[i][j];
            cp(i,j) = propagatedCovariance[i][j];
        }
    }
    Eigen::VectorXd sv  = c.jacobiSvd().singularValues();
    Eigen::VectorXd svp = cp.jacobiSvd().singularValues();
    LOG_DEBUG("singular values            = " << sv.transpose());
    LOG_DEBUG("propagated singular values = " << svp.transpose());
    for (std::size_t i = 0u; i < 2; ++i)
    {
        CPPUNIT_ASSERT(svp(i) > sv(i));
    }
    CPPUNIT_ASSERT(propagatedLogWeightRatio < logWeightRatio);
}

void CMultivariateOneOfNPriorTest::testWeightUpdate(void)
{
    LOG_DEBUG("+--------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testWeightUpdate  |");
    LOG_DEBUG("+--------------------------------------------------+");

    // Test that the weights stay normalized over update.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    {
        const std::size_t n[] = { 100 };
        const double mean[][2] = { { 10.0, 20.0 } };
        const double covariance[][3] = { { 3.0, 1.0, 2.0 } };

        TDouble10Vec1Vec samples;
        gaussianSamples(rng, boost::size(n), n, mean, covariance, samples);

        using TEqual = maths::CEqualWithTolerance<double>;
        TEqual equal(maths::CToleranceTypes::E_AbsoluteTolerance, 1e-10);
        const double decayRates[] = { 0.0, 0.004, 0.04 };

        for (std::size_t i = 0; i < boost::size(decayRates); ++i)
        {
            maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData, decayRates[i]));
            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[j]), SINGLE_UNIT_WEIGHT_2);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, sum(filter.weights()), 1e-6);
                filter.propagateForwardsByTime(1.0);
                CPPUNIT_ASSERT(equal(sum(filter.weights()), 1.0));
            }
        }
    }

    {
        // Test that non-zero decay rate behaves as expected.

        const std::size_t n[] = { 4000, 6000 };
        const double means[][2] =
            {
                { 10.0, 10.0 },
                { 20.0, 20.0 }
            };
        const double covariances[][3] =
            {
                {  8.0,  1.0, 8.0 },
                { 20.0, -4.0, 10.0 }
            };

        TDouble10Vec1Vec samples;
        gaussianSamples(rng, boost::size(n), n, means, covariances, samples);
        rng.random_shuffle(samples.begin(), samples.end());

        const double decayRates[] = { 0.0008, 0.004, 0.02 };

        double previousLogWeightRatio = -6700;

        for (std::size_t i = 0u; i < boost::size(decayRates); ++i)
        {
            maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData, decayRates[i]));

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[j]), SINGLE_UNIT_WEIGHT_2);
                filter.propagateForwardsByTime(1.0);
            }

            TDoubleVec logWeights = filter.logWeights();
            LOG_DEBUG("log weight ratio = " << logWeights[0] - logWeights[1]);

            // Should be approximately 0.2: we reduce the filter memory
            // by a factor of 5 each iteration.
            CPPUNIT_ASSERT((logWeights[0] - logWeights[1]) / previousLogWeightRatio > 0.15);
            CPPUNIT_ASSERT((logWeights[0] - logWeights[1]) / previousLogWeightRatio < 0.3);
            previousLogWeightRatio = logWeights[0] - logWeights[1];
        }
    }
}

void CMultivariateOneOfNPriorTest::testModelUpdate(void)
{
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testModelUpdate  |");
    LOG_DEBUG("+-------------------------------------------------+");

    maths::CSampling::CScopeMockRandomNumberGenerator scopeMockRng;

    const std::size_t n[] = { 400, 600 };
    const double means[][2] =
        {
            { 10.0, 10.0 },
            { 20.0, 20.0 }
        };
    const double covariances[][3] =
        {
            {  8.0,  1.0, 8.0 },
            { 20.0, -4.0, 10.0 }
        };

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);
    rng.random_shuffle(samples.begin(), samples.end());

    const maths_t::EDataType dataTypes[] =
        {
            maths_t::E_IntegerData,
            maths_t::E_ContinuousData
        };

    for (std::size_t i = 0u; i < boost::size(dataTypes); ++i)
    {
        maths::CMultivariateNormalConjugate<2> normal =
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(dataTypes[i]);
        maths::CMultivariateMultimodalPrior<2> multimodal = makeMultimodal<2>(dataTypes[i]);
        maths::CMultivariateOneOfNPrior oneOfN(makeOneOfN<2>(dataTypes[i]));

        normal.addSamples(COUNT_WEIGHT, samples, TDouble10Vec4Vec1Vec(samples.size(), UNIT_WEIGHT_2));
        multimodal.addSamples(COUNT_WEIGHT, samples, TDouble10Vec4Vec1Vec(samples.size(), UNIT_WEIGHT_2));
        oneOfN.addSamples(COUNT_WEIGHT, samples, TDouble10Vec4Vec1Vec(samples.size(), UNIT_WEIGHT_2));

        CPPUNIT_ASSERT_EQUAL(normal.checksum(), oneOfN.models()[0]->checksum());
        CPPUNIT_ASSERT_EQUAL(multimodal.checksum(), oneOfN.models()[1]->checksum());
    }
}

void CMultivariateOneOfNPriorTest::testModelSelection(void)
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testModelSelection  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // TODO When copula models are available.
}

void CMultivariateOneOfNPriorTest::testMarginalLikelihood(void)
{
    LOG_DEBUG("+--------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testMarginalLikelihood  |");
    LOG_DEBUG("+--------------------------------------------------------+");

    // Test that:
    //   1) The likelihood is normalized.
    //   2) E[X] w.r.t. the likelihood is equal to the predictive distribution mean.
    //   3) E[(X - m)^2] w.r.t. the marginal likelihood is equal to the predictive
    //      distribution covariance matrix.

    using TSizeVec = std::vector<std::size_t>;

    maths::CSampling::seed();

    std::size_t nt = 3;

    test::CRandomNumbers rng;
    {
        LOG_DEBUG("*** Normal ***");

        TDoubleVec meani;
        rng.generateUniformSamples(  0.0,  50.0, 2 * nt, meani);
        TDoubleVec covariancesii;
        rng.generateUniformSamples( 20.0, 500.0, 2 * nt, covariancesii);
        TDoubleVec covariancesij;
        rng.generateUniformSamples(-10.0,  10.0, 1 * nt, covariancesij);

        for (std::size_t t = 0u; t < nt; ++t)
        {
            LOG_DEBUG("*** Test " << t+1 << " ***");

            // Generate the samples.
            double mean_[] = { meani[2 * t], meani[2 * t + 1] };
            double covariances_[] =
                {
                    covariancesii[2 * t], covariancesij[t], covariancesii[2 * t + 1]
                };
            TDoubleVec mean(mean_, mean_ + 2);
            TDoubleVecVec covariances;
            covariances.push_back(TDoubleVec(covariances_,     covariances_ + 2));
            covariances.push_back(TDoubleVec(covariances_ + 1, covariances_ + 3));
            TDoubleVecVec samples;
            maths::CSampling::multivariateNormalSample(mean, covariances, 20, samples);

            maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));

            TMeanAccumulator meanMeanError;
            TMeanAccumulator meanCovarianceError;

            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);

                if (!filter.isNonInformative())
                {
                    TDouble10Vec m = filter.marginalLikelihoodMean();
                    TDouble10Vec10Vec v = filter.marginalLikelihoodCovariance();
                    LOG_DEBUG("m = " << core::CContainerPrinter::print(m));
                    LOG_DEBUG("v = " << core::CContainerPrinter::print(v));
                    double trace = 0.0;
                    for (std::size_t j = 0u; j < v.size(); ++j)
                    {
                        trace += v[j][j];
                    }
                    double intervals[][2] =
                        {
                            { m[0] - 3.0 * std::sqrt(trace), m[1] - 3.0 * std::sqrt(trace) },
                            { m[0] - 3.0 * std::sqrt(trace), m[1] - 1.0 * std::sqrt(trace) },
                            { m[0] - 3.0 * std::sqrt(trace), m[1] + 1.0 * std::sqrt(trace) },
                            { m[0] - 1.0 * std::sqrt(trace), m[1] - 3.0 * std::sqrt(trace) },
                            { m[0] - 1.0 * std::sqrt(trace), m[1] - 1.0 * std::sqrt(trace) },
                            { m[0] - 1.0 * std::sqrt(trace), m[1] + 1.0 * std::sqrt(trace) },
                            { m[0] + 1.0 * std::sqrt(trace), m[1] - 3.0 * std::sqrt(trace) },
                            { m[0] + 1.0 * std::sqrt(trace), m[1] - 1.0 * std::sqrt(trace) },
                            { m[0] + 1.0 * std::sqrt(trace), m[1] + 1.0 * std::sqrt(trace) }
                        };

                    TVector2 expectedMean(m.begin(), m.end());
                    double elements[] = { v[0][0], v[0][1], v[1][1] };
                    TMatrix2 expectedCovariance(elements, elements + 3);

                    CUnitKernel<2> likelihoodKernel(filter);
                    CMeanKernel<2> meanKernel(filter);
                    CCovarianceKernel<2> covarianceKernel(filter, expectedMean);

                    double z = 0.0;
                    TVector2 actualMean(0.0);
                    TMatrix2 actualCovariance(0.0);
                    for (std::size_t j = 0u; j < boost::size(intervals); ++j)
                    {
                        TDoubleVec a(boost::begin(intervals[j]), boost::end(intervals[j]));
                        TDoubleVec b(a);
                        b[0] += 2.0 * std::sqrt(trace);
                        b[1] += 2.0 * std::sqrt(trace);

                        double zj;
                        maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix,
                                                                 maths::CIntegration::TwoDimensions>(likelihoodKernel, a, b, zj);
                        TVector2 mj;
                        maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix,
                                                                 maths::CIntegration::TwoDimensions>(meanKernel, a, b, mj);
                        TMatrix2 cj;
                        maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix,
                                                                 maths::CIntegration::TwoDimensions>(covarianceKernel, a, b, cj);

                        z += zj;
                        actualMean += mj;
                        actualCovariance += cj;
                    }

                    LOG_DEBUG("Z = " << z);
                    LOG_DEBUG("mean = " << actualMean);
                    LOG_DEBUG("covariance = " << actualCovariance);

                    TVector2 meanError = actualMean - expectedMean;
                    TMatrix2 covarianceError = actualCovariance - expectedCovariance;
                    CPPUNIT_ASSERT(meanError.euclidean() < expectedMean.euclidean());
                    CPPUNIT_ASSERT(covarianceError.frobenius() < expectedCovariance.frobenius());

                    meanMeanError.add(meanError.euclidean() / expectedMean.euclidean());
                    meanCovarianceError.add(covarianceError.frobenius() / expectedCovariance.frobenius());
                }
            }

            LOG_DEBUG("Mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
            LOG_DEBUG("Mean covariance error = " << maths::CBasicStatistics::mean(meanCovarianceError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.16);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanCovarianceError) < 0.09);
        }
    }
    {
        LOG_DEBUG("*** Multimodal ***");

        std::size_t sizes_[] = { 200, 150, 300 };
        TSizeVec sizes(boost::begin(sizes_), boost::end(sizes_));

        TMeanAccumulator meanZ;
        TMeanAccumulator meanMeanError;
        TMeanAccumulator meanCovarianceError;

        for (std::size_t t = 0u; t < nt; ++t)
        {
            LOG_DEBUG("*** Test " << t+1 << " ***");

            TVector2Vec means;
            TMatrix2Vec covariances;
            TVector2VecVec samples_;
            rng.generateRandomMultivariateNormals(sizes, means, covariances, samples_);
            TDouble10Vec1Vec samples;
            for (std::size_t i = 0u; i < means.size(); ++i)
            {
                means[i] += TVector2(20.0);
            }
            for (std::size_t i = 0u; i < samples_.size(); ++i)
            {
                for (std::size_t j = 0u; j < samples_[i].size(); ++j)
                {
                    samples.push_back((TVector2(20.0) + samples_[i][j]).toVector<TDouble10Vec>());
                }
            }
            rng.random_shuffle(samples.begin(), samples.end());

            maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));
            for (std::size_t i = 0u; i < samples.size(); ++i)
            {
                filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
            }

            TDouble10Vec m = filter.marginalLikelihoodMean();
            TDouble10Vec10Vec v = filter.marginalLikelihoodCovariance();

            TVector2 expectedMean(m.begin(), m.end());
            double elements[] = { v[0][0], v[0][1], v[1][1] };
            TMatrix2 expectedCovariance(elements, elements + 3);

            double z = 0.0;
            TVector2 actualMean(0.0);
            TMatrix2 actualCovariance(0.0);
            for (std::size_t i = 0u; i < means.size(); ++i)
            {
                double trace = covariances[i].trace();
                LOG_DEBUG("m = " << means[i]);
                LOG_DEBUG("v = " << trace);

                double intervals[][2] =
                    {
                        { means[i](0) - 3.0 * std::sqrt(trace), means[i](1) - 3.0 * std::sqrt(trace) },
                        { means[i](0) - 3.0 * std::sqrt(trace), means[i](1) - 1.0 * std::sqrt(trace) },
                        { means[i](0) - 3.0 * std::sqrt(trace), means[i](1) + 1.0 * std::sqrt(trace) },
                        { means[i](0) - 1.0 * std::sqrt(trace), means[i](1) - 3.0 * std::sqrt(trace) },
                        { means[i](0) - 1.0 * std::sqrt(trace), means[i](1) - 1.0 * std::sqrt(trace) },
                        { means[i](0) - 1.0 * std::sqrt(trace), means[i](1) + 1.0 * std::sqrt(trace) },
                        { means[i](0) + 1.0 * std::sqrt(trace), means[i](1) - 3.0 * std::sqrt(trace) },
                        { means[i](0) + 1.0 * std::sqrt(trace), means[i](1) - 1.0 * std::sqrt(trace) },
                        { means[i](0) + 1.0 * std::sqrt(trace), means[i](1) + 1.0 * std::sqrt(trace) }
                    };
                CUnitKernel<2> likelihoodKernel(filter);
                CMeanKernel<2> meanKernel(filter);
                CCovarianceKernel<2> covarianceKernel(filter, expectedMean);

                for (std::size_t j = 0u; j < boost::size(intervals); ++j)
                {
                    TDoubleVec a(boost::begin(intervals[j]), boost::end(intervals[j]));
                    TDoubleVec b(a);
                    b[0] += 2.0 * std::sqrt(trace);
                    b[1] += 2.0 * std::sqrt(trace);

                    double zj;
                    maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix,
                                                             maths::CIntegration::TwoDimensions>(likelihoodKernel, a, b, zj);
                    TVector2 mj;
                    maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix,
                                                             maths::CIntegration::TwoDimensions>(meanKernel, a, b, mj);
                    TMatrix2 cj;
                    maths::CIntegration::sparseGaussLegendre<maths::CIntegration::OrderSix,
                                                             maths::CIntegration::TwoDimensions>(covarianceKernel, a, b, cj);

                    z += zj;
                    actualMean += mj;
                    actualCovariance += cj;
                }
            }

            actualMean /= z;
            actualCovariance /= z;

            LOG_DEBUG("Z = " << z);
            LOG_DEBUG("expected mean = " << expectedMean);
            LOG_DEBUG("expected covariance = " << expectedCovariance);
            LOG_DEBUG("actual mean = " << actualMean);
            LOG_DEBUG("actual covariance = " << actualCovariance);

            TVector2 meanError = actualMean - expectedMean;
            TMatrix2 covarianceError = actualCovariance - expectedCovariance;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, z, 0.7);
            CPPUNIT_ASSERT(meanError.euclidean() < 0.3 * expectedMean.euclidean());
            CPPUNIT_ASSERT(covarianceError.frobenius() < 0.25 * expectedCovariance.frobenius());

            meanZ.add(z);
            meanMeanError.add(meanError.euclidean() / expectedMean.euclidean());
            meanCovarianceError.add(covarianceError.frobenius() / expectedCovariance.frobenius());
        }

        LOG_DEBUG("Mean Z = " << maths::CBasicStatistics::mean(meanZ));
        LOG_DEBUG("Mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
        LOG_DEBUG("Mean covariance error = " << maths::CBasicStatistics::mean(meanCovarianceError));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, maths::CBasicStatistics::mean(meanZ), 0.3);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.1);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanCovarianceError) < 0.16);
    }
}

void CMultivariateOneOfNPriorTest::testMarginalLikelihoodMean(void)
{
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testMarginalLikelihoodMean  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    // Test that the marginal likelihood mean is close to the sample
    // mean for a variety of models.

    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMean2Accumulator = maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator;

    maths::CSampling::seed();

    TSizeVecVec sizes;
    sizes.push_back(TSizeVec(1, 100));
    sizes.push_back(TSizeVec());
    sizes.back().push_back(100);
    sizes.back().push_back(100);
    sizes.back().push_back(100);

    test::CRandomNumbers rng;

    double expectedMeanErrors[] = { 1e-6, 0.05 };

    for (std::size_t i = 0u; i < sizes.size(); ++i)
    {
        LOG_DEBUG("# modes = " << sizes[i].size());

        TVector2Vec means;
        TMatrix2Vec covariances;
        TVector2VecVec samples_;
        rng.generateRandomMultivariateNormals(sizes[i], means, covariances, samples_);
        TDouble10Vec1Vec samples;
        for (std::size_t j = 0u; j < samples_.size(); ++j)
        {
            for (std::size_t k = 0u; k < samples_[j].size(); ++k)
            {
                samples.push_back(samples_[j][k].toVector<TDouble10Vec>());
            }
        }
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));

        TMeanAccumulator meanError;

        TMean2Accumulator expectedMean;
        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[j]), SINGLE_UNIT_WEIGHT_2);
            expectedMean.add(TVector2(samples[j]));

            if (!filter.isNonInformative())
            {
                if (j % 10 == 0)
                {
                    LOG_DEBUG("expected = " << maths::CBasicStatistics::mean(expectedMean)
                              << " actual = " << core::CContainerPrinter::print(filter.marginalLikelihoodMean()));
                }
                double error =  (TVector2(filter.marginalLikelihoodMean()) - maths::CBasicStatistics::mean(expectedMean)).euclidean()
                              / maths::CBasicStatistics::mean(expectedMean).euclidean();
                meanError.add(error);
                CPPUNIT_ASSERT(error < 0.2);
            }
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < expectedMeanErrors[i]);
    }
}

void CMultivariateOneOfNPriorTest::testMarginalLikelihoodMode(void)
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testMultipleUpdate  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that the marginal likelihood mode is near the maximum
    // of the marginal likelihood.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("****** Normal ******");

        const double means[][2] =
            {
                { 10.0, 10.0 },
                { 50.0, 50.0 },
            };
        const double covariances[][3] =
            {
                {  1.0, 0.0,  1.0 },
                { 10.0, 0.0, 10.0 },
            };
        double learnRates[] = { 0.1, 0.3 };

        for (std::size_t i = 0u; i < boost::size(means); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(covariances); ++j)
            {
                std::size_t n[] = { 100 };
                const double mean[][2] = { { means[i][0],  means[i][1] } };
                const double covariance[][3] =
                    {
                        { covariances[i][0], covariances[i][1], covariances[i][2] }
                    };
                LOG_DEBUG("*** mean = " << core::CContainerPrinter::print(mean[0], mean[0] + 2)
                          << ", variance = " << covariance[0][0] << " ***");

                TDouble10Vec1Vec samples;
                gaussianSamples(rng, 1, n, mean, covariance, samples);

                maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));
                for (std::size_t k = 0u; k < samples.size(); ++k)
                {
                    filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[k]), SINGLE_UNIT_WEIGHT_2);
                }

                CMinusLogLikelihood likelihood(filter);
                maths::CGradientDescent::CEmpiricalCentralGradient gradientOfLikelihood(likelihood, 1e-3);
                maths::CGradientDescent gd(learnRates[j], 0.75);
                maths::CVector<double> expectedMode;
                TDoubleVec likelihoods;
                gd.run(20, // iterations
                       maths::CVector<double>(mean[0], mean[0] + 2),
                       likelihood,
                       gradientOfLikelihood,
                       expectedMode, likelihoods);

                TDouble10Vec mode = filter.marginalLikelihoodMode(COUNT_WEIGHT, UNIT_WEIGHT_2);

                LOG_DEBUG("marginalLikelihoodMode = " << core::CContainerPrinter::print(mode)
                          << ", expectedMode = " << expectedMode);

                for (std::size_t k = 0u; k < 2; ++k)
                {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode(k), mode[k], 0.01 * expectedMode(k));
                }
            }
        }
    }

    {
        LOG_DEBUG("****** Multimodal ******");

        const std::size_t n[] = { 100, 100 };
        const double means[][2] =
            {
                { 10.0, 10.0 },
                { 16.0, 18.0 },
            };
        const double covariances[][3] =
            {
                {  4.0, 0.0,  4.0 },
                { 10.0, 0.0, 10.0 },
            };

        TDouble10Vec1Vec samples;
        gaussianSamples(rng, boost::size(n), n, means, covariances, samples);

        maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));
        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
        }

        CMinusLogLikelihood likelihood(filter);
        maths::CGradientDescent::CEmpiricalCentralGradient gradientOfLikelihood(likelihood, 1e-3);
        maths::CGradientDescent gd(0.2, 0.75);
        maths::CVector<double> expectedMode;
        TDoubleVec likelihoods;
        gd.run(20, // iterations
               maths::CVector<double>(means[0], means[0] + 2),
               likelihood,
               gradientOfLikelihood,
               expectedMode, likelihoods);

        TDouble10Vec mode = filter.marginalLikelihoodMode(COUNT_WEIGHT, UNIT_WEIGHT_2);

        LOG_DEBUG("marginalLikelihoodMode = " << core::CContainerPrinter::print(mode)
                  << ", expectedMode = " << expectedMode);

        for (std::size_t i = 0u; i < 2; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode(i), mode[i], 0.2 * expectedMode(i));
        }
    }
}

void CMultivariateOneOfNPriorTest::testSampleMarginalLikelihood(void)
{
    LOG_DEBUG("+--------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG("+--------------------------------------------------------------+");

    // Test we sample the constitute priors in proportion to their weights.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    const std::size_t n[] = { 50, 50 };
    const double means[][2] =
        {
            { 10.0, 10.0 },
            { 25.0, 25.0 },
        };
    const double covariances[][3] =
        {
            {  4.0, 0.0,  4.0 },
            { 10.0, 0.0, 10.0 },
        };

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);
    rng.random_shuffle(samples.begin(), samples.end());

    maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);

        if (!filter.isNonInformative())
        {
            TDoubleVec weights = filter.weights();
            LOG_DEBUG("weights = " << core::CContainerPrinter::print(weights));

            TDouble10Vec1Vec sampled;
            filter.sampleMarginalLikelihood(20, sampled);
            std::sort(sampled.begin(), sampled.end());

            // We modes to be sampled according to their weights.
            maths::CSampling::TSizeVec counts;
            maths::CSampling::weightedSample(20, weights, counts);
            LOG_DEBUG("counts = " << core::CContainerPrinter::print(counts));

            maths::CMultivariateOneOfNPrior::TPriorCPtr3Vec posteriorModels = filter.models();
            TDouble10Vec1Vec normalSamples;
            posteriorModels[0]->sampleMarginalLikelihood(counts[0], normalSamples);
            TDouble10Vec1Vec multimodalSamples;
            posteriorModels[1]->sampleMarginalLikelihood(counts[1], multimodalSamples);

            TDouble10Vec1Vec expectedSampled(normalSamples);
            expectedSampled.insert(expectedSampled.end(),
                                   multimodalSamples.begin(),
                                   multimodalSamples.end());
            std::sort(expectedSampled.begin(), expectedSampled.end());
            LOG_DEBUG("expected samples = " << core::CContainerPrinter::print(expectedSampled));
            LOG_DEBUG("samples          = " << core::CContainerPrinter::print(sampled));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSampled),
                                 core::CContainerPrinter::print(sampled));
        }
    }
}

void CMultivariateOneOfNPriorTest::testProbabilityOfLessLikelySamples(void)
{
    LOG_DEBUG("+--------------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+--------------------------------------------------------------------+");

    // We simply test that the calculation is close to the weighted
    // sum of component model calculations.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    const std::size_t n[] = { 100, 100 };
    const double means[][2] =
        {
            { 10.0, 10.0 },
            { 16.0, 18.0 },
        };
    const double covariances[][3] =
        {
            {  4.0, 0.0,  4.0 },
            { 10.0, 0.0, 10.0 },
        };

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);
    rng.random_shuffle(samples.begin(), samples.end());

    maths::CMultivariateOneOfNPrior filter(makeOneOfN<2>(maths_t::E_ContinuousData));

    TMeanAccumulator error;

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        TDouble10Vec1Vec sample(1, samples[i]);
        filter.addSamples(COUNT_WEIGHT, sample, SINGLE_UNIT_WEIGHT_2);

        double lowerBound, upperBound;
        maths::CMultivariatePrior::TTail10Vec tail;
        CPPUNIT_ASSERT(filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                             COUNT_WEIGHT,
                                                             sample,
                                                             SINGLE_UNIT_WEIGHT_2,
                                                             lowerBound,
                                                             upperBound,
                                                             tail));

        CPPUNIT_ASSERT_EQUAL(lowerBound, upperBound);
        double probability = (lowerBound + upperBound) / 2.0;

        double expectedProbability = 0.0;

        TDoubleVec weights(filter.weights());
        maths::CMultivariateOneOfNPrior::TPriorCPtr3Vec models(filter.models());
        for (std::size_t j = 0u; j < weights.size(); ++j)
        {
            double modelLowerBound, modelUpperBound;
            double weight = weights[j];
            CPPUNIT_ASSERT(models[j]->probabilityOfLessLikelySamples(
                                              maths_t::E_TwoSided,
                                              COUNT_WEIGHT,
                                              sample,
                                              SINGLE_UNIT_WEIGHT_2,
                                              modelLowerBound,
                                              modelUpperBound,
                                              tail));
            CPPUNIT_ASSERT_EQUAL(modelLowerBound, modelUpperBound);
            double modelProbability = (modelLowerBound + modelUpperBound) / 2.0;
            expectedProbability += weight * modelProbability;
        }

        LOG_DEBUG("weights = " << core::CContainerPrinter::print(weights)
                  << ", expectedProbability = " << expectedProbability
                  << ", probability = " << probability);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedProbability,
                                     probability,
                                     0.3 * std::max(expectedProbability, probability));
        error.add(std::fabs(probability - expectedProbability));
    }

    LOG_DEBUG("error = " << maths::CBasicStatistics::mean(error));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.01);
}

void CMultivariateOneOfNPriorTest::testPersist(void)
{
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CMultivariateOneOfNPriorTest::testPersist  |");
    LOG_DEBUG("+---------------------------------------------+");

    // Check that persist/restore is idempotent.

    const std::size_t n[] = { 100 };
    const double mean[][2] = { { 10.0, 20.0 } };
    const double covariance[][3] = { { 3.0, 1.0, 2.0 } };

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, mean, covariance, samples);

    maths_t::EDataType dataType = maths_t::E_ContinuousData;

    maths::CMultivariateOneOfNPrior origFilter(makeOneOfN<2>(dataType));
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        origFilter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
    }
    std::size_t dimension = origFilter.dimension();
    double decayRate = origFilter.decayRate();
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("Multivariate one-of-n XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(dataType, decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::CMultivariateOneOfNPrior restoredFilter(dimension, params, traverser);

    LOG_DEBUG("orig checksum = " << checksum
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

CppUnit::Test *CMultivariateOneOfNPriorTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMultivariateOneOfNPriorTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testMultipleUpdate",
                                   &CMultivariateOneOfNPriorTest::testMultipleUpdate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testPropagation",
                                   &CMultivariateOneOfNPriorTest::testPropagation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testWeightUpdate",
                                   &CMultivariateOneOfNPriorTest::testWeightUpdate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testModelUpdate",
                                   &CMultivariateOneOfNPriorTest::testModelUpdate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testModelSelection",
                                   &CMultivariateOneOfNPriorTest::testModelSelection) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testMarginalLikelihood",
                                   &CMultivariateOneOfNPriorTest::testMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testMarginalLikelihoodMean",
                                   &CMultivariateOneOfNPriorTest::testMarginalLikelihoodMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testMarginalLikelihoodMode",
                                   &CMultivariateOneOfNPriorTest::testMarginalLikelihoodMode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testSampleMarginalLikelihood",
                                   &CMultivariateOneOfNPriorTest::testSampleMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testProbabilityOfLessLikelySamples",
                                   &CMultivariateOneOfNPriorTest::testProbabilityOfLessLikelySamples) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateOneOfNPriorTest>(
                                   "CMultivariateOneOfNPriorTest::testPersist",
                                   &CMultivariateOneOfNPriorTest::testPersist) );

    return suiteOfTests;
}
