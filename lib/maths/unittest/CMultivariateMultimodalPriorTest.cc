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

#include "CMultivariateMultimodalPriorTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CIntegration.h>
#include <maths/CMultivariateMultimodalPrior.h>
#include <maths/CMultivariateMultimodalPriorFactory.h>
#include <maths/CMultivariateNormalConjugate.h>
#include <maths/CMultivariateNormalConjugateFactory.h>
#include <maths/CXMeansOnline.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

using namespace ml;
using namespace handy_typedefs;

typedef std::vector<double> TDoubleVec;
typedef std::vector<TDoubleVec> TDoubleVecVec;
typedef std::vector<TDoubleVecVec> TDoubleVecVecVec;
typedef std::vector<std::size_t> TSizeVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
typedef maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator TMean2Accumulator;
typedef maths::CBasicStatistics::SSampleCovariances<double, 2> TCovariances2;

namespace
{

const maths_t::TWeightStyleVec COUNT_WEIGHT(1, maths_t::E_SampleCountWeight);
const maths_t::TWeightStyleVec VARIANCE_WEIGHT(1, maths_t::E_SampleCountVarianceScaleWeight);
const TDouble10Vec UNIT_WEIGHT_2(2, 1.0);
const TDouble10Vec4Vec1Vec SINGLE_UNIT_WEIGHT_2(1, TDouble10Vec4Vec(1, UNIT_WEIGHT_2));

template<std::size_t N>
class CMultivariateMultimodalPriorForTest : public maths::CMultivariateMultimodalPrior<N>
{
    public:
        typedef typename maths::CMultivariateMultimodalPrior<N>::TClusterer TClusterer;
        typedef typename maths::CMultivariateMultimodalPrior<N>::TModeVec TModeVec;

    public:
        CMultivariateMultimodalPriorForTest(const maths::CMultivariateMultimodalPrior<N> &prior) :
                 maths::CMultivariateMultimodalPrior<N>(prior)
        {}

        const TModeVec &modes(void) const
        {
            return this->maths::CMultivariateMultimodalPrior<N>::modes();
        }
};

template<std::size_t N>
maths::CMultivariateMultimodalPrior<N> makePrior(maths_t::EDataType dataType,
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

template<std::size_t N>
double logLikelihood(const double w[N],
                     const double means[N][2],
                     const double covariances[N][3],
                     const TDouble10Vec &x)
{
    double lx = 0.0;
    for (std::size_t i = 0u; i < N; ++i)
    {
        TVector2 mean(means[i]);
        TMatrix2 covariance(covariances[i], covariances[i] + 3);
        double ll;
        maths::gaussianLogLikelihood(covariance, TVector2(x) - mean, ll);
        lx += w[i] * ::exp(ll);
    }
    return ::log(lx);
}

double logLikelihood(const TDoubleVec &w,
                     const TDoubleVecVec &means,
                     const TDoubleVecVecVec &covariances,
                     const TDoubleVec &x)
{
    double lx = 0.0;
    for (std::size_t i = 0u; i < w.size(); ++i)
    {
        double ll;
        maths::gaussianLogLikelihood(TMatrix2(covariances[i]),
                                     TVector2(x) - TVector2(means[i]),
                                     ll);
        lx += w[i] * ::exp(ll);
    }
    return ::log(lx);
}

void empiricalProbabilityOfLessLikelySamples(const TDoubleVec &w,
                                             const TDoubleVecVec &means,
                                             const TDoubleVecVecVec &covariances,
                                             TDoubleVec &result)
{
    test::CRandomNumbers rng;

    double m = static_cast<double>(w.size());

    for (std::size_t i = 0u; i < w.size(); ++i)
    {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(means[i],
                                              covariances[i],
                                              static_cast<std::size_t>(w[i] * 1000.0 * m),
                                              samples);
        result.reserve(samples.size());
        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            result.push_back(logLikelihood(w, means, covariances, samples[j]));
        }
    }

    std::sort(result.begin(), result.end());
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

void CMultivariateMultimodalPriorTest::testMultipleUpdate(void)
{
    LOG_DEBUG("+--------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testMultipleUpdate  |");
    LOG_DEBUG("+--------------------------------------------------------+");

    // Test that we get the same result updating once with a vector of 100
    // samples of an R.V. versus updating individually 100 times.

    const std::size_t n[] = { 100 };
    const double means[][2] = { { 10.0, 20.0 } };
    const double covariances[][3] = { { 3.0, 1.0, 2.0 } };

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);

    const maths_t::EDataType dataTypes[] =
        {
            maths_t::E_IntegerData,
            maths_t::E_ContinuousData
        };

    LOG_DEBUG("****** Test vanilla ******");
    for (size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        LOG_DEBUG("*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateMultimodalPrior<2> filter1(makePrior<2>(dataTypes[i]));
        maths::CMultivariateMultimodalPrior<2> filter2(filter1);

        maths::CSampling::seed();
        for (std::size_t j = 0; j < samples.size(); ++j)
        {
            filter1.addSamples(COUNT_WEIGHT,
                               TDouble10Vec1Vec(1, samples[j]),
                               SINGLE_UNIT_WEIGHT_2);
        }
        maths::CSampling::seed();
        filter2.addSamples(COUNT_WEIGHT,
                           samples,
                           TDouble10Vec4Vec1Vec(samples.size(),
                                                TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));

        LOG_DEBUG("checksum 1 " << filter1.checksum());
        LOG_DEBUG("checksum 2 " << filter2.checksum());
        CPPUNIT_ASSERT_EQUAL(filter1.checksum(), filter2.checksum());
    }

    LOG_DEBUG("****** Test with variance scale ******");
    for (size_t i = 0; i < boost::size(dataTypes); ++i)
    {
        LOG_DEBUG("*** data type = " << print(dataTypes[i]) << " ***");

        maths::CMultivariateMultimodalPrior<2> filter1(makePrior<2>(dataTypes[i]));
        maths::CMultivariateMultimodalPrior<2> filter2(filter1);

        TDouble10Vec4Vec1Vec weights;
        weights.resize(samples.size() / 2, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.5)));
        weights.resize(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 2.0)));
        maths::CSampling::seed();
        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            TDouble10Vec1Vec sample(1, samples[j]);
            TDouble10Vec4Vec1Vec weight(1, weights[j]);
            filter1.addSamples(VARIANCE_WEIGHT, sample, weight);
        }
        maths::CSampling::seed();
        filter2.addSamples(VARIANCE_WEIGHT, samples, weights);

        LOG_DEBUG("checksum 1 " << filter1.checksum());
        LOG_DEBUG("checksum 2 " << filter2.checksum());
        CPPUNIT_ASSERT_EQUAL(filter1.checksum(), filter2.checksum());
    }
}

void CMultivariateMultimodalPriorTest::testPropagation(void)
{
    LOG_DEBUG("+-----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testPropagation  |");
    LOG_DEBUG("+-----------------------------------------------------+");

    // Test that propagation doesn't affect the marginal likelihood
    // mean and the marginal likelihood variance increases (due to
    // influence of the prior uncertainty) after propagation.

    maths::CSampling::seed();

    const double eps = 1e-3;

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

    const double decayRate = 0.1;

    maths::CMultivariateMultimodalPrior<2> filter(makePrior<2>(maths_t::E_ContinuousData, decayRate));
    filter.addSamples(COUNT_WEIGHT,
                      samples,
                      TDouble10Vec4Vec1Vec(samples.size(), TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));

    double numberSamples = filter.numberSamples();
    TDouble10Vec mean = filter.marginalLikelihoodMean();
    TDouble10Vec10Vec covariance = filter.marginalLikelihoodCovariance();

    filter.propagateForwardsByTime(40.0);

    double propagatedNumberSamples = filter.numberSamples();
    TDouble10Vec propagatedMean = filter.marginalLikelihoodMean();
    TDouble10Vec10Vec propagatedCovariance = filter.marginalLikelihoodCovariance();

    LOG_DEBUG("numberSamples           = " << numberSamples);
    LOG_DEBUG("propagatedNumberSamples = " << propagatedNumberSamples);
    LOG_DEBUG("mean           = " << core::CContainerPrinter::print(mean));
    LOG_DEBUG("propagatedMean = " << core::CContainerPrinter::print(propagatedMean));
    LOG_DEBUG("covariance           = " << core::CContainerPrinter::print(covariance));
    LOG_DEBUG("propagatedCovariance = " << core::CContainerPrinter::print(propagatedCovariance));

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
}

void CMultivariateMultimodalPriorTest::testSingleMode(void)
{
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testSingleMode  |");
    LOG_DEBUG("+----------------------------------------------------+");

    // Test that we stably get one cluster.

    maths::CSampling::seed();

    const std::size_t n[] = { 500 };
    const double means[][2] = { { 20.0, 20.0 } };
    const double covariances[][3] = { { 40.0, 10.0, 20.0 } };

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);

    maths::CMultivariateMultimodalPrior<2> filter(makePrior<2>(maths_t::E_ContinuousData));

    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        filter.addSamples(COUNT_WEIGHT,
                          TDouble10Vec1Vec(1, samples[i]),
                          SINGLE_UNIT_WEIGHT_2);
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), filter.numberModes());
    }
}

void CMultivariateMultimodalPriorTest::testMultipleModes(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testMultipleModes  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    // We check that for data generated from multiple modes
    // we get something close to the generating distribution.
    // In particular, we test the log likelihood of the data
    // for the estimated distribution versus the generating
    // distribution and versus an unclustered distribution.
    // Note that the generating distribution doesn't necessarily
    // have a larger likelihood because we are using a finite
    // sample.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    LOG_DEBUG("Mixture Normals");
    {
        const std::size_t n[] = { 400, 600 };
        const double means[][2] =
            {
                { 10.0, 10.0 },
                { 20.0, 20.0 }
            };
        const double covariances[][3] =
            {
                {  4.0,  1.0, 4.0 },
                { 10.0, -4.0, 6.0 }
            };

        TDouble10Vec1Vec samples;
        gaussianSamples(rng, boost::size(n), n, means, covariances, samples);

        double w[] =
            {
                n[0] / static_cast<double>(n[0] + n[1]),
                n[1] / static_cast<double>(n[0] + n[1])
            };

        double loss = 0.0;
        TMeanAccumulator differentialEntropy_;
        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            differentialEntropy_.add(-logLikelihood<2>(w, means, covariances, samples[i]));
        }
        double differentialEntropy = maths::CBasicStatistics::mean(differentialEntropy_);

        for (std::size_t i = 0; i < 10; ++i)
        {
            rng.random_shuffle(samples.begin(), samples.end());

            maths::CMultivariateMultimodalPrior<2> filter1(makePrior<2>(maths_t::E_ContinuousData));
            maths::CMultivariateNormalConjugate<2> filter2 =
                    maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData);

            filter1.addSamples(COUNT_WEIGHT,
                               samples,
                               TDouble10Vec4Vec1Vec(samples.size(), TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));
            filter2.addSamples(COUNT_WEIGHT,
                               samples,
                               TDouble10Vec4Vec1Vec(samples.size(), TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));

            CPPUNIT_ASSERT_EQUAL(std::size_t(2), filter1.numberModes());

            TMeanAccumulator loss1G;
            TMeanAccumulator loss12;

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                double ll = logLikelihood<2>(w, means, covariances, samples[j]);

                TDouble10Vec1Vec sample(1, samples[j]);
                double l1;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                     filter1.jointLogMarginalLikelihood(COUNT_WEIGHT,
                                                                        sample,
                                                                        SINGLE_UNIT_WEIGHT_2,
                                                                        l1));
                loss1G.add(ll - l1);
                double l2;
                CPPUNIT_ASSERT_EQUAL(maths_t::E_FpNoErrors,
                                     filter2.jointLogMarginalLikelihood(COUNT_WEIGHT,
                                                                        sample,
                                                                        SINGLE_UNIT_WEIGHT_2,
                                                                        l2));
                loss12.add(l2 - l1);
            }

            LOG_DEBUG("loss1G = " << maths::CBasicStatistics::mean(loss1G)
                      << ", loss12 = " << maths::CBasicStatistics::mean(loss12)
                      << ", differential entropy " << differentialEntropy);

            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss12) < 0.0);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(loss1G) / differentialEntropy < 0.0);
            loss += maths::CBasicStatistics::mean(loss1G);
        }

        loss /= 10.0;
        LOG_DEBUG("loss = " << loss
                  << ", differential entropy = " << differentialEntropy);
        CPPUNIT_ASSERT(loss / differentialEntropy < 0.0);
    }
}

void CMultivariateMultimodalPriorTest::testSplitAndMerge(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testSplitAndMerge  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    // Test clustering which changes over time.

    typedef std::vector<TDoubleVecVec> TDoubleVecVecVec;

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    double means_[][2] =
        {
            { 10, 15 },
            { 30, 10 },
            { 10, 15 },
            { 30, 10 }
        };
    double covariances_[][2][2] =
         {
             { { 10, 2}, {2, 15} },
             { { 30, 8}, {8, 15} },
             { {100, 2}, {2, 15} },
             { {100, 2}, {2, 15} }
         };

    TDoubleVecVec means(boost::size(means_));
    TDoubleVecVecVec covariances(boost::size(means_));
    for (std::size_t i = 0u; i < boost::size(means_); ++i)
    {
        means[i].assign(&means_[i][0], &means_[i][2]);
        for (std::size_t j = 0u; j < 2; ++j)
        {
            covariances[i].push_back(TDoubleVec(&covariances_[i][j][0],
                                                &covariances_[i][j][2]));
        }
    }

    LOG_DEBUG("Clusters Split and Merge")
    {
        std::size_t n[][4] =
            {
                { 200,   0,   0,   0 },
                { 100, 100,   0,   0 },
                {   0,   0, 300, 300 }
            };

        TCovariances2 totalCovariances;
        TCovariances2 modeCovariances[4];

        CMultivariateMultimodalPriorForTest<2> filter(makePrior<2>(maths_t::E_ContinuousData));

        maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMeanError;
        maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanCovError;

        //std::ofstream f;
        //f.open("results.m");
        //std::size_t subplot = 0u;
        //std::size_t subplotCounts[] = { 50, 200, 250, 450, 500, 550, 585, 615, 650, 750, 800, 1000, 10000 };
        //TDouble10Vec1Vec pointsToDate;

        for (std::size_t i = 0u; i < boost::size(n); ++i)
        {
            TDoubleVecVec samples;
            for (std::size_t j = 0u; j < boost::size(n[i]); ++j)
            {
                TDoubleVecVec samples_;
                rng.generateMultivariateNormalSamples(means[j], covariances[j], n[i][j], samples_);
                for (std::size_t k = 0u; k < samples_.size(); ++k)
                {
                    modeCovariances[j].add(TVector2(samples_[k]));
                    totalCovariances.add(TVector2(samples_[k]));
                }
                samples.insert(samples.end(), samples_.begin(), samples_.end());
            }
            rng.random_shuffle(samples.begin(), samples.end());
            LOG_DEBUG("# samples = " << samples.size());

            for (std::size_t j = 0u; j < samples.size(); ++j)
            {
                filter.addSamples(COUNT_WEIGHT,
                                  TDouble10Vec1Vec(1, samples[j]),
                                  TDouble10Vec4Vec1Vec(1, TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));

                //pointsToDate.push_back(samples[j]);
                //if (pointsToDate.size() == subplotCounts[subplot])
                //{
                //    f << "subplot(4, 3, " << ++subplot << ")\n";
                //    std::ostringstream points;
                //    points << "points = [";
                //    std::size_t last = pointsToDate.size() - 1;
                //    for (std::size_t k = 0u; k < last; ++k)
                //    {
                //        points << pointsToDate[k][0] << " " << pointsToDate[k][1] << ";";
                //    }
                //    points << pointsToDate[last][0] << " " << pointsToDate[last][1] << "];";
                //    f << points.str() << "\nhold on; scatter(points(:,1), points(:,2), 'b', 'x')\n";
                //    f << filter.printMarginalLikelihoodFunction(0, 1) << "\n";
                //    f << "axis([-8 55 0 25])\n";
                //}
            }

            const CMultivariateMultimodalPriorForTest<2>::TModeVec &modes = filter.modes();
            LOG_DEBUG("# modes = " << modes.size());
            LOG_DEBUG("prior = " << filter.print());

            for (std::size_t j = 0u; j < modes.size(); ++j)
            {
                maths::CBasicStatistics::COrderStatisticsStack<double, 1> meanError;
                maths::CBasicStatistics::COrderStatisticsStack<double, 1> covError;

                if (modes.size() == 1)
                {
                    meanError.add((  TVector2(modes[j].s_Prior->marginalLikelihoodMean())
                                   - maths::CBasicStatistics::mean(totalCovariances)).euclidean());
                    TMatrix2 mlc(modes[j].s_Prior->marginalLikelihoodCovariance());
                    TMatrix2 tcm = maths::CBasicStatistics::covariances(totalCovariances);
                    covError.add((mlc - tcm).frobenius() / tcm.frobenius());
                }
                else
                {
                    for (std::size_t k = 0u; k < boost::size(modeCovariances); ++k)
                    {
                        meanError.add( (  TVector2(modes[j].s_Prior->marginalLikelihoodMean())
                                        - maths::CBasicStatistics::mean(modeCovariances[k])).euclidean()
                                      / maths::CBasicStatistics::mean(modeCovariances[k]).euclidean());
                        covError.add( (  TMatrix2(modes[j].s_Prior->marginalLikelihoodCovariance())
                                       - maths::CBasicStatistics::covariances(modeCovariances[k])).frobenius()
                                     / maths::CBasicStatistics::covariances(modeCovariances[k]).frobenius());
                    }
                }

                LOG_DEBUG("mean error = " << meanError[0]);
                LOG_DEBUG("cov error  = " << covError[0]);
                CPPUNIT_ASSERT(meanError[0] < 0.05);
                CPPUNIT_ASSERT(covError[0] < 0.1);

                meanMeanError.add(meanError[0]);
                meanCovError.add(covError[0]);
            }
        }

        LOG_DEBUG("mean meanError = " << maths::CBasicStatistics::mean(meanMeanError));
        LOG_DEBUG("mean covError  = " << maths::CBasicStatistics::mean(meanCovError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.013);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanCovError) < 0.030);
    }
}

void CMultivariateMultimodalPriorTest::testMarginalLikelihood(void)
{
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testMarginalLikelihood  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    // Test that:
    //   1) The likelihood is normalized.
    //   2) E[X] w.r.t. the likelihood is equal to the predictive distribution mean.
    //   3) E[(X - m)^2] w.r.t. the marginal likelihood is equal to the predictive
    //      distribution covariance matrix.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    std::size_t sizes_[] = { 200, 150, 300 };
    TSizeVec sizes(boost::begin(sizes_), boost::end(sizes_));

    TMeanAccumulator meanZ;
    TMeanAccumulator meanMeanError;
    TMeanAccumulator meanCovarianceError;

    for (std::size_t t = 0u; t < 10; /**/)
    {
        TVector2Vec means;
        TMatrix2Vec covariances;
        TVector2VecVec samples_;
        rng.generateRandomMultivariateNormals(sizes, means, covariances, samples_);
        TDouble10Vec1Vec samples;
        for (std::size_t i = 0u; i < samples_.size(); ++i)
        {
            for (std::size_t j = 0u; j < samples_[i].size(); ++j)
            {
                samples.push_back(samples_[i][j].toVector<TDouble10Vec>());
            }
        }
        rng.random_shuffle(samples.begin(), samples.end());

        maths::CMultivariateMultimodalPrior<2> filter(makePrior<2>(maths_t::E_ContinuousData));
        filter.addSamples(COUNT_WEIGHT, samples, TDouble10Vec4Vec1Vec(samples.size(), TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));
        LOG_DEBUG("# modes = " << filter.numberModes());
        if (filter.numberModes() != 3)
        {
            continue;
        }
        LOG_DEBUG("*** Test " << t+1 << " ***");
        ++t;

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
                    { means[i](0) - 3.0 * ::sqrt(trace), means[i](1) - 3.0 * ::sqrt(trace) },
                    { means[i](0) - 3.0 * ::sqrt(trace), means[i](1) - 1.0 * ::sqrt(trace) },
                    { means[i](0) - 3.0 * ::sqrt(trace), means[i](1) + 1.0 * ::sqrt(trace) },
                    { means[i](0) - 1.0 * ::sqrt(trace), means[i](1) - 3.0 * ::sqrt(trace) },
                    { means[i](0) - 1.0 * ::sqrt(trace), means[i](1) - 1.0 * ::sqrt(trace) },
                    { means[i](0) - 1.0 * ::sqrt(trace), means[i](1) + 1.0 * ::sqrt(trace) },
                    { means[i](0) + 1.0 * ::sqrt(trace), means[i](1) - 3.0 * ::sqrt(trace) },
                    { means[i](0) + 1.0 * ::sqrt(trace), means[i](1) - 1.0 * ::sqrt(trace) },
                    { means[i](0) + 1.0 * ::sqrt(trace), means[i](1) + 1.0 * ::sqrt(trace) }
                };
            CUnitKernel<2> likelihoodKernel(filter);
            CMeanKernel<2> meanKernel(filter);
            CCovarianceKernel<2> covarianceKernel(filter, expectedMean);

            for (std::size_t j = 0u; j < boost::size(intervals); ++j)
            {
                TDoubleVec a(boost::begin(intervals[j]), boost::end(intervals[j]));
                TDoubleVec b(a);
                b[0] += 2.0 * ::sqrt(trace);
                b[1] += 2.0 * ::sqrt(trace);

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
        CPPUNIT_ASSERT(covarianceError.frobenius() < 0.2 * expectedCovariance.frobenius());

        meanZ.add(z);
        meanMeanError.add(meanError.euclidean() / expectedMean.euclidean());
        meanCovarianceError.add(covarianceError.frobenius() / expectedCovariance.frobenius());
    }

    LOG_DEBUG("Mean Z = " << maths::CBasicStatistics::mean(meanZ));
    LOG_DEBUG("Mean mean error = " << maths::CBasicStatistics::mean(meanMeanError));
    LOG_DEBUG("Mean covariance error = " << maths::CBasicStatistics::mean(meanCovarianceError));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, maths::CBasicStatistics::mean(meanZ), 0.1);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMeanError) < 0.1);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanCovarianceError) < 0.04);
}

void CMultivariateMultimodalPriorTest::testMarginalLikelihoodMean(void)
{
    LOG_DEBUG("+----------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testMarginalLikelihoodMean  |");
    LOG_DEBUG("+----------------------------------------------------------------+");

    // Test that the marginal likelihood mean is close to the sample
    // mean for a multimodal distribution.

    maths::CSampling::seed();

    const double eps = 0.05;

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

    maths::CMultivariateMultimodalPrior<2> filter(makePrior<2>(maths_t::E_ContinuousData));
    TMean2Accumulator expectedMean;
    TMeanAccumulator meanError;

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[i]), SINGLE_UNIT_WEIGHT_2);
        expectedMean.add(samples[i]);

        if (i % 10 == 0)
        {
            LOG_DEBUG("sample mean = " << maths::CBasicStatistics::mean(expectedMean));
            LOG_DEBUG("distribution mean = " << core::CContainerPrinter::print(filter.marginalLikelihoodMean()));
        }

        double error = (  maths::CBasicStatistics::mean(expectedMean)
                        - TVector2(filter.marginalLikelihoodMean())).euclidean()
                      / maths::CBasicStatistics::mean(expectedMean).euclidean();
        CPPUNIT_ASSERT(error < eps);
        meanError.add(error);
    }

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.002);
}

void CMultivariateMultimodalPriorTest::testMarginalLikelihoodMode(void)
{
    LOG_DEBUG("+----------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testMarginalLikelihoodMode  |");
    LOG_DEBUG("+----------------------------------------------------------------+");

    // Test that the sample mode is close to the generating distribution mode.

    typedef maths::CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double> > TMaxAccumulator;

    maths::CSampling::seed();

    double eps = 1e-6;

    std::size_t sizes_[] = { 150, 200, 100 };
    TSizeVec sizes(boost::begin(sizes_), boost::end(sizes_));

    test::CRandomNumbers rng;

    TMeanAccumulator meanError;

    for (std::size_t t = 0u; t < 50; ++t)
    {
        TVector2Vec means;
        TMatrix2Vec covariances;
        TVector2VecVec samples_;
        rng.generateRandomMultivariateNormals(sizes, means, covariances, samples_);
        TDouble10Vec1Vec samples;
        for (std::size_t i = 0u; i < samples_.size(); ++i)
        {
            for (std::size_t j = 0u; j < samples_[i].size(); ++j)
            {
                samples.push_back(samples_[i][j].toVector<TDouble10Vec>());
            }
        }
        rng.random_shuffle(samples.begin(), samples.end());

        CMultivariateMultimodalPriorForTest<2> filter(makePrior<2>(maths_t::E_ContinuousData));
        filter.addSamples(COUNT_WEIGHT,
                          samples,
                          TDouble10Vec4Vec1Vec(samples.size(), SINGLE_UNIT_WEIGHT_2[0]));
        TDouble10Vec mode = filter.marginalLikelihoodMode(COUNT_WEIGHT, SINGLE_UNIT_WEIGHT_2[0]);

        TVector2 expectedMode;
        TMaxAccumulator maxLikelihood;
        for (std::size_t i = 0u; i < filter.modes().size(); ++i)
        {
            TDouble10Vec mi = (filter.modes())[i].s_Prior->marginalLikelihoodMode(COUNT_WEIGHT,
                                                                                  SINGLE_UNIT_WEIGHT_2[0]);
            double likelihood;
            filter.jointLogMarginalLikelihood(COUNT_WEIGHT,
                                              TDouble10Vec1Vec(1, mi),
                                              SINGLE_UNIT_WEIGHT_2,
                                              likelihood);
            if (maxLikelihood.add(likelihood))
            {
                expectedMode = TVector2(mi);
            }
        }

        LOG_DEBUG("# modes = " << filter.numberModes());
        LOG_DEBUG("mode          = " << core::CContainerPrinter::print(mode));
        LOG_DEBUG("expected mode = " << expectedMode);
        double error = (TVector2(mode) - expectedMode).euclidean() / expectedMode.euclidean();
        CPPUNIT_ASSERT(error < eps);
        meanError.add(error);
    }

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.02);
}

void CMultivariateMultimodalPriorTest::testSampleMarginalLikelihood(void)
{
    LOG_DEBUG("+------------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testSampleMarginalLikelihood  |");
    LOG_DEBUG("+------------------------------------------------------------------+");

    // We're going to test the following properties of the sampling:
    //   1) That the sampled mean and covariance are close to the marginal
    //      likelihood mean and covariance.
    //   2) That each mode is sampled according to the number of samples
    //      it contains in the training data.
    //   3) That the sampled mean and covariance of each mode are close
    //      to the corresponding quantities in the training data.

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    const std::size_t n[] = { 400, 600 };
    const double means_[][2] =
        {
            { 10.0, 10.0 },
            { 20.0, 20.0 }
        };
    const double covariances_[][3] =
        {
            {  8.0,  1.0, 8.0 },
            { 20.0, -4.0, 10.0 }
        };

    TVector2Vec means;
    TMatrix2Vec covariances;
    TDouble10Vec1Vec samples;
    for (std::size_t i = 0u; i < boost::size(n); ++i)
    {
        TVector2 mean(means_[i]);
        TMatrix2 covariance(covariances_[i], covariances_[i] + 3);
        means.push_back(mean);
        covariances.push_back(covariance);
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
    rng.random_shuffle(samples.begin(), samples.end());
    LOG_DEBUG("# samples = " << samples.size());

    maths::CMultivariateMultimodalPrior<2> filter(makePrior<2>(maths_t::E_ContinuousData));
    filter.addSamples(COUNT_WEIGHT,
                      samples,
                      TDouble10Vec4Vec1Vec(samples.size(), TDouble10Vec4Vec(1, UNIT_WEIGHT_2)));

    TDouble10Vec1Vec sampled;
    filter.sampleMarginalLikelihood(300, sampled);

    TDouble10Vec expectedMean_ = filter.marginalLikelihoodMean();
    TDouble10Vec10Vec expectedCovariance_ = filter.marginalLikelihoodCovariance();

    TCovariances2 sampledCovariances;
    for (std::size_t i = 0u; i < sampled.size(); ++i)
    {
        sampledCovariances.add(TVector2(sampled[i]));
    }

    TVector2 expectedMean(expectedMean_);
    TMatrix2 expectedCovariance(expectedCovariance_);
    TVector2 sampledMean = maths::CBasicStatistics::mean(sampledCovariances);
    TMatrix2 sampledCovariance = maths::CBasicStatistics::covariances(sampledCovariances);
    LOG_DEBUG("expected mean = " << expectedMean);
    LOG_DEBUG("expected covariance = " << expectedCovariance);
    LOG_DEBUG("sampled mean = " << sampledMean);
    LOG_DEBUG("sampled covariance = " << sampledCovariance);
    CPPUNIT_ASSERT((sampledMean - expectedMean).euclidean() < 1e-3 * expectedMean.euclidean());
    CPPUNIT_ASSERT((sampledCovariance - expectedCovariance).frobenius() < 5e-3 * expectedCovariance.frobenius());

    TCovariances2 modeSampledCovariances[2];
    for (std::size_t i = 0u; i < sampled.size(); ++i)
    {
        double l1, l2;
        maths::gaussianLogLikelihood(covariances[0], TVector2(sampled[i]) - means[0], l1);
        maths::gaussianLogLikelihood(covariances[1], TVector2(sampled[i]) - means[1], l2);
        modeSampledCovariances[l1 > l2 ? 0 : 1].add(TVector2(sampled[i]));
    }

    for (std::size_t i = 0u; i < 2; ++i)
    {
        TVector2 modeSampledMean = maths::CBasicStatistics::mean(modeSampledCovariances[i]);
        TMatrix2 modeSampledCovariance = maths::CBasicStatistics::covariances(modeSampledCovariances[i]);
        LOG_DEBUG("sample mean = " << means[i]);
        LOG_DEBUG("sample covariance = " << covariances[i]);
        LOG_DEBUG("sampled mean = " << modeSampledMean);
        LOG_DEBUG("sampled covariance = " << modeSampledCovariance);
        CPPUNIT_ASSERT((modeSampledMean - means[i]).euclidean() < 0.03 * means[i].euclidean());
        CPPUNIT_ASSERT((modeSampledCovariance - covariances[i]).frobenius() < 0.2 * covariances[i].frobenius());
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(n[0]) / static_cast<double>(n[1]),
                                   maths::CBasicStatistics::count(modeSampledCovariances[0])
                                 / maths::CBasicStatistics::count(modeSampledCovariances[1]),
                                 0.02);
}

void CMultivariateMultimodalPriorTest::testProbabilityOfLessLikelySamples(void)
{
    LOG_DEBUG("+------------------------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testProbabilityOfLessLikelySamples  |");
    LOG_DEBUG("+------------------------------------------------------------------------+");

    // Test that the probability is approximately equal to the chance of drawing
    // a less likely sample from generating distribution.

    maths::CSampling::seed();

    const double w_[][3] =
        {
            { 0.25, 0.3, 0.45 },
            { 0.1,  0.3, 0.6 }
        };
    const double means_[][3][2] =
        {
            { {10, 10}, { 15,  18}, {10, 60} },
            { { 0,  0}, {-20, -30}, {40, 15} }
        };
    const double covariances_[][3][2][2] =
        {
            {{ {10,  0}, { 0, 10} }, { {10,  9}, { 9, 10} }, { {10, -9},  {-9,  10} }},
            {{ { 5,  0}, { 0,  5} }, { {40,  9}, { 9, 40} }, { {30, -27}, {-27, 30} }}
        };
    const double offsets[][2] =
        {
            {  0.0,  0.0 },
            {  0.0,  6.0 },
            {  4.0,  0.0 },
            {  6.0,  6.0 },
            {  6.0, -6.0 },
            { -8.0,  8.0 },
            { -8.0, -8.0 }
        };

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(w_); ++i)
    {
        std::size_t n = (boost::size(w_[i]));

        TDoubleVec w(n);
        TDoubleVecVec means(n);
        TDoubleVecVecVec covariances(n);
        for (std::size_t j = 0u; j < boost::size(w_[i]); ++j)
        {
            w[j] = w_[i][j];
            means[j].assign(means_[i][j], means_[i][j] + 2);
            covariances[j].resize(2);
            for (std::size_t k = 0u; k < 2; ++k)
            {
                covariances[j][k].assign(covariances_[i][j][k], covariances_[i][j][k] + 2);
            }
        }
        LOG_DEBUG("means = " << core::CContainerPrinter::print(means));
        LOG_DEBUG("covariances = " << core::CContainerPrinter::print(covariances));

        TDoubleVecVec samples;
        for (std::size_t j = 0u; j < w.size(); ++j)
        {
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(means[j],
                                                  covariances[j],
                                                  static_cast<std::size_t>(w[j] * 1000.0),
                                                  samples_);
            samples.insert(samples.end(), samples_.begin(), samples_.end());
        }
        rng.random_shuffle(samples.begin(), samples.end());

        CMultivariateMultimodalPriorForTest<2> filter(makePrior<2>(maths_t::E_ContinuousData));
        for (std::size_t k = 0u; k < samples.size(); ++k)
        {
            filter.addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, samples[k]), SINGLE_UNIT_WEIGHT_2);
        }
        LOG_DEBUG("# modes = " << filter.numberModes());

        TDoubleVec p;
        empiricalProbabilityOfLessLikelySamples(w, means, covariances, p);

        for (std::size_t j = 0u; j < means.size(); ++j)
        {
            TMeanAccumulator meanAbsError;
            TMeanAccumulator meanRelError;

            for (std::size_t k = 0u; k < boost::size(offsets); ++k)
            {
                TVector2 x = TVector2(means[j]) + TVector2(offsets[k]);

                double ll = logLikelihood(w, means, covariances, x.toVector<TDoubleVec>());
                double px =  static_cast<double>(std::lower_bound(p.begin(), p.end(), ll) - p.begin())
                           / static_cast<double>(p.size());

                double lb, ub;
                maths::CMultivariatePrior::TTail10Vec tail;
                filter.probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      COUNT_WEIGHT,
                                                      TDouble10Vec1Vec(1, x.toVector<TDouble10Vec>()),
                                                      SINGLE_UNIT_WEIGHT_2,
                                                      lb, ub, tail);
                double pa = (lb + ub) / 2.0;

                LOG_DEBUG("  p(" << x << "), actual = " << pa << ", expected = " << px);
                meanAbsError.add(::fabs(px - pa));
                if (px < 1.0 && px > 0.0)
                {
                    meanRelError.add(::fabs(::log(px) - ::log(pa)) / ::fabs(::log(px)));
                }
            }

            LOG_DEBUG("mean absolute error = " << maths::CBasicStatistics::mean(meanAbsError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanAbsError) < 0.25);

            LOG_DEBUG("mean relative error = " << maths::CBasicStatistics::mean(meanRelError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelError) < 0.6);
        }
    }
}

void CMultivariateMultimodalPriorTest::testIntegerData(void)
{
    LOG_DEBUG("+-----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testIntegerData  |");
    LOG_DEBUG("+-----------------------------------------------------+");

    // TODO
}

void CMultivariateMultimodalPriorTest::testLowVariationData(void)
{
    LOG_DEBUG("+----------------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testLowVariationData  |");
    LOG_DEBUG("+----------------------------------------------------------+");

    // TODO
}

void CMultivariateMultimodalPriorTest::testLatLongData(void)
{
    LOG_DEBUG("+-----------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testLatLongData  |");
    LOG_DEBUG("+-----------------------------------------------------+");

    typedef std::pair<core_t::TTime, TDoubleVec> TTimeDoubleVecPr;
    typedef std::vector<TTimeDoubleVecPr> TTimeDoubleVecPrVec;

    TTimeDoubleVecPrVec timeseries;
    CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/lat_lng.csv",
                                                    timeseries,
                                                    test::CTimeSeriesTestData::CSV_UNIX_BIVALUED_REGEX));
    CPPUNIT_ASSERT(!timeseries.empty());

    LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                timeseries.begin() + 10)
              << " ...");

    maths_t::EDataType dataType = maths_t::E_ContinuousData;
    boost::shared_ptr<maths::CMultivariatePrior> modePrior =
            maths::CMultivariateNormalConjugateFactory::nonInformative(2, dataType, 0.001);
    boost::shared_ptr<maths::CMultivariatePrior> filter =
            maths::CMultivariateMultimodalPriorFactory::nonInformative(2,    // dimension
                                                                       dataType, 0.0005,
                                                                       maths_t::E_ClustersFractionWeight,
                                                                       0.02, // minimumClusterFraction
                                                                       4,    // minimumClusterCount
                                                                       0.8,  // minimumCategoryCount
                                                                       *modePrior);

    for (std::size_t i = 0u; i < timeseries.size(); ++i)
    {
        filter->addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, timeseries[i].second), SINGLE_UNIT_WEIGHT_2);
        filter->propagateForwardsByTime(1.0);
    }
    LOG_DEBUG(filter->print());

    // TODO Finish

    //TDoubleVec p;
    //for (std::size_t i = 0u; i < timeseries.size(); ++i)
    //{
    //    double lb, ub;
    //    maths::CMultivariatePrior::TTail10Vec tail;
    //    filter->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
    //                                           COUNT_WEIGHT,
    //                                           TDouble10Vec1Vec(1, timeseries[i].second),
    //                                           SINGLE_UNIT_WEIGHT_2,
    //                                           lb, ub, tail);
    //    p.push_back((lb + ub) / 2.0);
    //}
    //LOG_DEBUG("p = " << core::CContainerPrinter::print(p) << ";");

    //std::ofstream f;
    //f.open("results.m");
    //f << prior->printMarginalLikelihoodFunction(0, 1);
}

void CMultivariateMultimodalPriorTest::testPersist(void)
{
    LOG_DEBUG("+-------------------------------------------------+");
    LOG_DEBUG("|  CMultivariateMultimodalPriorTest::testPersist  |");
    LOG_DEBUG("+-------------------------------------------------+");

    // Check that persist/restore is idempotent.

    maths::CSampling::seed();

    std::size_t n[] = { 100, 100 };
    const double means[][2] =
        {
            {  10.0, 20.0 },
            { 100.0, 30.0 }
        };
    const double covariances[][3] =
        {
            {  3.0,  1.0,  2.0 },
            { 60.0, 20.0, 70.0 }
        };

    test::CRandomNumbers rng;

    TDouble10Vec1Vec samples;
    gaussianSamples(rng, boost::size(n), n, means, covariances, samples);
    rng.random_shuffle(samples.begin(), samples.end());

    maths_t::EDataType dataType = maths_t::E_ContinuousData;
    double decayRate = 0.0;

    maths::CMultivariateMultimodalPrior<2> origFilter(makePrior<2>(dataType));

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        origFilter.addSamples(COUNT_WEIGHT,
                              TDouble10Vec1Vec(1, samples[i]),
                              SINGLE_UNIT_WEIGHT_2);
    }
    uint64_t checksum = origFilter.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origFilter.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("Normal mean conjugate XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::SDistributionRestoreParams params(dataType, decayRate + 0.1,
                                             maths::MINIMUM_CLUSTER_SPLIT_FRACTION,
                                             maths::MINIMUM_CLUSTER_SPLIT_COUNT,
                                             maths::MINIMUM_CATEGORY_COUNT);
    maths::CMultivariateMultimodalPrior<2> restoredFilter(params, traverser);

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

CppUnit::Test *CMultivariateMultimodalPriorTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMultivariateMultimodalPriorTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testMultipleUpdate",
                                   &CMultivariateMultimodalPriorTest::testMultipleUpdate) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testPropagation",
                                   &CMultivariateMultimodalPriorTest::testPropagation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testSingleMode",
                                   &CMultivariateMultimodalPriorTest::testSingleMode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testMultipleModes",
                                   &CMultivariateMultimodalPriorTest::testMultipleModes) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testSplitAndMerge",
                                   &CMultivariateMultimodalPriorTest::testSplitAndMerge) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testMarginalLikelihood",
                                   &CMultivariateMultimodalPriorTest::testMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testMarginalLikelihoodMean",
                                   &CMultivariateMultimodalPriorTest::testMarginalLikelihoodMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testMarginalLikelihoodMode",
                                   &CMultivariateMultimodalPriorTest::testMarginalLikelihoodMode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testSampleMarginalLikelihood",
                                   &CMultivariateMultimodalPriorTest::testSampleMarginalLikelihood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testProbabilityOfLessLikelySamples",
                                   &CMultivariateMultimodalPriorTest::testProbabilityOfLessLikelySamples) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testIntegerData",
                                   &CMultivariateMultimodalPriorTest::testIntegerData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testLowVariationData",
                                   &CMultivariateMultimodalPriorTest::testLowVariationData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testLatLongData",
                                   &CMultivariateMultimodalPriorTest::testLatLongData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMultivariateMultimodalPriorTest>(
                                   "CMultivariateMultimodalPriorTest::testPersist",
                                   &CMultivariateMultimodalPriorTest::testPersist) );

    return suiteOfTests;
}
