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

#include "CInformationCriteriaTest.h"

#include <maths/CInformationCriteria.h>
#include <maths/CKMeansFast.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSampling.h>

#include <test/CRandomNumbers.h>

using namespace ml;

namespace {

typedef std::vector<std::size_t> TSizeVec;
typedef std::vector<TSizeVec> TSizeVecVec;
typedef maths::CVectorNx1<double, 2> TVector2;
typedef std::vector<TVector2> TVector2Vec;
typedef TVector2Vec::const_iterator TVector2VecCItr;
typedef std::vector<TVector2Vec> TVector2VecVec;
typedef maths::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator TMeanVar2Accumulator;
typedef maths::CSymmetricMatrixNxN<double, 2> TMatrix2;
typedef std::vector<TMatrix2> TMatrix2Vec;
typedef maths::CVectorNx1<double, 4> TVector4;
typedef std::vector<TVector4> TVector4Vec;
typedef maths::CBasicStatistics::SSampleMeanVar<TVector4>::TAccumulator TMeanVar4Accumulator;
typedef maths::CSymmetricMatrixNxN<double, 4> TMatrix4;
typedef std::vector<TMatrix4> TMatrix4Vec;

template<typename POINT>
double logfSphericalGaussian(const POINT &mean,
                             double variance,
                             const POINT &x) {
    double d = static_cast<double>(x.dimension());
    double r = (x - mean).euclidean();
    return -0.5 * (  d * ::log(boost::math::double_constants::two_pi * variance)
                     + r * r / variance);
}



template<typename POINT, typename MATRIX>
double logfGaussian(const POINT &mean,
                    const MATRIX &covariance,
                    const POINT &x) {
    double result;
    maths::gaussianLogLikelihood(covariance, x - mean, result);
    return result;
}

}

void CInformationCriteriaTest::testSphericalGaussian(void) {
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CInformationCriteriaTest::testSphericalGaussian  |");
    LOG_DEBUG("+---------------------------------------------------+");

    // Check that the information criterion values are the expected
    // values for the generating distribution.

    maths::CSampling::seed();

    {
        double variance        = 5.0;
        double mean_[]         = { 10.0, 20.0 };
        double lowerTriangle[] = { variance, 0.0, variance };

        TVector2 mean(boost::begin(mean_), boost::end(mean_));
        TMatrix2 covariance(boost::begin(lowerTriangle),
                            boost::end(lowerTriangle));
        LOG_DEBUG("mean = " << mean);
        LOG_DEBUG("covariance = " << covariance);

        TVector2Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            likelihood += -2.0 * logfSphericalGaussian(mean, variance, samples[i])
                          + 2.0 * ::log(upper);
        }
        double expectedAICc = likelihood + 6.0 + 12.0 / (n - 4.0);
        double expectedBIC  = likelihood + 3.0 * ::log(n);

        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG("expected BIC  = " << expectedBIC);
        LOG_DEBUG("BIC           = " << bic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedBIC,
                                     bic.calculate(),
                                     2e-3 * expectedBIC);

        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG("expected AICc = " << expectedAICc);
        LOG_DEBUG("AICc          = " << aic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedAICc,
                                     aic.calculate(),
                                     2e-3 * expectedAICc);
    }
    {
        double variance        = 8.0;
        double mean_[]         = { -5.0, 30.0, 2.0, 7.9 };
        double lowerTriangle[] = { variance, 0.0, variance, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance };

        TVector4 mean(boost::begin(mean_), boost::end(mean_));
        TMatrix4 covariance(boost::begin(lowerTriangle),
                            boost::end(lowerTriangle));
        LOG_DEBUG("mean = " << mean);
        LOG_DEBUG("covariance = " << covariance);

        TVector4Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 500, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            likelihood += -2.0 * logfSphericalGaussian(mean, variance, samples[i])
                          + 4.0 * ::log(upper);
        }
        double expectedAICc = likelihood + 10.0 + 30.0 / (n - 6.0);
        double expectedBIC  = likelihood + 5.0 * ::log(n);

        maths::CSphericalGaussianInfoCriterion<TVector4, maths::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG("expected BIC = " << expectedBIC);
        LOG_DEBUG("BIC          = " << bic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedBIC,
                                     bic.calculate(),
                                     2e-3 * expectedBIC);

        maths::CSphericalGaussianInfoCriterion<TVector4, maths::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG("expected AICc = " << expectedAICc);
        LOG_DEBUG("AICc          = " << aic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedAICc,
                                     aic.calculate(),
                                     2e-3 * expectedAICc);
    }

    // Check that they correctly distinguish the best fit model.

    test::CRandomNumbers rng;

    {
        double variance        = 40.0;
        double mean_[]         = { 15.0, 27.0 };
        double lowerTriangle[] = { variance, 0.0, variance };

        TVector2 mean(boost::begin(mean_), boost::end(mean_));
        TMatrix2 covariance(boost::begin(lowerTriangle),
                            boost::end(lowerTriangle));
        LOG_DEBUG("mean = " << mean);
        LOG_DEBUG("covariance = " << covariance);

        TVector2Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC>  bic1(samples);
        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aic1(samples);

        for (std::size_t t = 0u; t < 100; ++t) {
            rng.random_shuffle(samples.begin(), samples.end());

            TSizeVec split;
            rng.generateUniformSamples(100, 900, 1, split);

            TVector2Vec samples1(&samples[0], &samples[split[0]]);
            TVector2Vec samples2(&samples[split[0]], &samples[999]);

            maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC>  bic2(samples);
            maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aic2(samples);
            bic2.add(samples1);
            bic2.add(samples2);
            aic2.add(samples1);
            aic2.add(samples2);

            LOG_DEBUG("1 cluster BIC = " << bic1.calculate());
            LOG_DEBUG("2 cluster BIC = " << bic2.calculate());
            CPPUNIT_ASSERT(bic1.calculate() < bic2.calculate());

            LOG_DEBUG("1 cluster AIC = " << aic1.calculate());
            LOG_DEBUG("2 cluster AIC = " << aic2.calculate());
            CPPUNIT_ASSERT(aic1.calculate() < aic2.calculate());
        }

        TVector2Vec centres;
        maths::CSampling::multivariateNormalSample(mean, covariance, 200, centres);

        maths::CKMeansFast<TVector2> kmeans;
        kmeans.setPoints(samples);
        for (std::size_t t = 0u; t < centres.size(); t += 2) {
            TVector2Vec tcentres(&centres[t], &centres[t+2]);
            kmeans.setCentres(tcentres);

            kmeans.run(10);
            TVector2VecVec clusters;
            kmeans.clusters(clusters);

            maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC>  bic2(clusters);
            maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aic2(clusters);

            LOG_DEBUG("1 cluster BIC = " << bic1.calculate());
            LOG_DEBUG("2 cluster BIC = " << bic2.calculate());
            CPPUNIT_ASSERT(bic1.calculate() < bic2.calculate());

            LOG_DEBUG("1 cluster AIC = " << aic1.calculate());
            LOG_DEBUG("2 cluster AIC = " << aic2.calculate());
            CPPUNIT_ASSERT(aic1.calculate() < aic2.calculate());
        }
    }
}

void CInformationCriteriaTest::testSphericalGaussianWithSphericalCluster(void) {
    LOG_DEBUG("+-----------------------------------------------------------------------+");
    LOG_DEBUG("|  CInformationCriteriaTest::testSphericalGaussianWithSphericalCluster  |");
    LOG_DEBUG("+-----------------------------------------------------------------------+");

    // The idea of this test is simply to check that we get the
    // same result working with clusters of points or their
    // spherical cluster representation.

    typedef maths::CSphericalCluster<TVector2>::Type TSphericalCluster2;
    typedef std::vector<TSphericalCluster2> TSphericalCluster2Vec;
    typedef std::vector<TSphericalCluster2Vec> TSphericalCluster2VecVec;

    maths::CSampling::seed();

    double means_[][2]     = { { 10.0, 20.0 }, { 12.0, 30.0 } };
    double lowerTriangle[] = { 5.0, 0.0, 5.0 };
    TVector2Vec means;
    for (std::size_t i = 0u; i < boost::size(means_); ++i) {
        means.push_back(TVector2(boost::begin(means_[i]), boost::end(means_[i])));
    }
    TMatrix2 covariance(boost::begin(lowerTriangle),
                        boost::end(lowerTriangle));
    LOG_DEBUG("means = " << core::CContainerPrinter::print(means));
    LOG_DEBUG("covariance = " << covariance);

    for (std::size_t t = 0u; t < 10; ++t) {
        LOG_DEBUG("*** trial = " << t+1 << " ***");

        TVector2VecVec points(means.size());
        TSphericalCluster2VecVec clusters;

        for (std::size_t i = 0u; i < means.size(); ++i) {
            maths::CSampling::multivariateNormalSample(means[i], covariance, 1000, points[i]);
            TMeanVar2Accumulator moments;
            moments.add(points[i]);
            double   n = maths::CBasicStatistics::count(moments);
            TVector2 m = maths::CBasicStatistics::mean(moments);
            TVector2 v = maths::CBasicStatistics::maximumLikelihoodVariance(moments);
            TSphericalCluster2::TAnnotation countAndVariance(n, (v(0) + v(1)) / 2.0);
            TSphericalCluster2 cluster(m, countAndVariance);
            clusters.push_back(TSphericalCluster2Vec(1, cluster));
        }

        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC> bicPoints;
        bicPoints.add(points);
        maths::CSphericalGaussianInfoCriterion<TSphericalCluster2, maths::E_BIC> bicClusters;
        bicClusters.add(clusters);
        LOG_DEBUG("BIC points  = " << bicPoints.calculate());
        LOG_DEBUG("BIC clusters  = " << bicClusters.calculate());

        CPPUNIT_ASSERT_DOUBLES_EQUAL(bicPoints.calculate(),
                                     bicClusters.calculate(),
                                     1e-10 * bicPoints.calculate());

        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aicPoints;
        aicPoints.add(points);
        maths::CSphericalGaussianInfoCriterion<TSphericalCluster2, maths::E_AICc> aicClusters;
        aicClusters.add(clusters);
        LOG_DEBUG("AICc points   = " << aicPoints.calculate());
        LOG_DEBUG("AICc clusters = " << aicClusters.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(aicPoints.calculate(),
                                     aicClusters.calculate(),
                                     1e-10 * aicPoints.calculate());
    }
}

void CInformationCriteriaTest::testGaussian(void) {
    LOG_DEBUG("+------------------------------------------+");
    LOG_DEBUG("|  CInformationCriteriaTest::testGaussian  |");
    LOG_DEBUG("+------------------------------------------+");

    maths::CSampling::seed();

    {
        double mean_[]         = { 10.0, 20.0 };
        double lowerTriangle[] = { 5.0, 1.0, 5.0 };

        TVector2 mean(boost::begin(mean_), boost::end(mean_));
        TMatrix2 covariance(boost::begin(lowerTriangle),
                            boost::end(lowerTriangle));
        LOG_DEBUG("mean = " << mean);
        LOG_DEBUG("covariance = " << covariance);

        TVector2Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            likelihood += -2.0 * logfGaussian(mean, covariance, samples[i])
                          + 2.0 * ::log(upper);
        }
        double expectedAICc = likelihood + 10.0 + 30.0 / (n - 6.0);
        double expectedBIC  = likelihood + 5.0 * ::log(n);

        maths::CGaussianInfoCriterion<TVector2, maths::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG("expected BIC  = " << expectedBIC);
        LOG_DEBUG("BIC           = " << bic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedBIC,
                                     bic.calculate(),
                                     2e-3 * expectedBIC);

        maths::CGaussianInfoCriterion<TVector2, maths::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG("expected AICc = " << expectedAICc);
        LOG_DEBUG("AICc          = " << aic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedAICc,
                                     aic.calculate(),
                                     2e-3 * expectedAICc);
    }
    {
        double mean_[]         = { -5.0, 30.0, 2.0, 7.9 };
        double lowerTriangle[] = { 8.0, 1.0, 8.0, 0.0, 0.0, 8.0, 0.0, 2.0, 0.5, 8.0 };

        TVector4 mean(boost::begin(mean_), boost::end(mean_));
        TMatrix4 covariance(boost::begin(lowerTriangle),
                            boost::end(lowerTriangle));
        LOG_DEBUG("mean = " << mean);
        LOG_DEBUG("covariance = " << covariance);

        TVector4Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 500, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            likelihood += -2.0 * logfGaussian(mean, covariance, samples[i])
                          + 4.0 * ::log(upper);
        }
        double expectedAICc = likelihood + 28.0 + 210.0 / (n - 15.0);
        double expectedBIC  = likelihood + 14.0 * ::log(n);

        maths::CGaussianInfoCriterion<TVector4, maths::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG("expected BIC = " << expectedBIC);
        LOG_DEBUG("BIC          = " << bic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedBIC,
                                     bic.calculate(),
                                     2e-3 * expectedBIC);

        maths::CGaussianInfoCriterion<TVector4, maths::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG("expected AICc = " << expectedAICc);
        LOG_DEBUG("AICc          = " << aic.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedAICc,
                                     aic.calculate(),
                                     2e-3 * expectedAICc);
    }

    // Check that they correctly distinguish the best fit model.

    test::CRandomNumbers rng;

    {
        double mean_[]         = { 15.0, 27.0 };
        double lowerTriangle[] = { 40.0, 5.0, 40.0 };

        TVector2 mean(boost::begin(mean_), boost::end(mean_));
        TMatrix2 covariance(boost::begin(lowerTriangle),
                            boost::end(lowerTriangle));
        LOG_DEBUG("mean = " << mean);
        LOG_DEBUG("covariance = " << covariance);

        TVector2Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC>  bic1(samples);
        maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aic1(samples);

        for (std::size_t t = 0u; t < 100; ++t) {
            rng.random_shuffle(samples.begin(), samples.end());

            TSizeVec split;
            rng.generateUniformSamples(100, 900, 1, split);

            TVector2Vec samples1(&samples[0], &samples[split[0]]);
            TVector2Vec samples2(&samples[split[0]], &samples[999]);

            maths::CGaussianInfoCriterion<TVector2, maths::E_BIC>  bic2(samples);
            maths::CGaussianInfoCriterion<TVector2, maths::E_AICc> aic2(samples);
            bic2.add(samples1);
            bic2.add(samples2);
            aic2.add(samples1);
            aic2.add(samples2);

            LOG_DEBUG("1 cluster BIC = " << bic1.calculate());
            LOG_DEBUG("2 cluster BIC = " << bic2.calculate());
            CPPUNIT_ASSERT(bic1.calculate() < bic2.calculate());

            LOG_DEBUG("1 cluster AIC = " << aic1.calculate());
            LOG_DEBUG("2 cluster AIC = " << aic2.calculate());
            CPPUNIT_ASSERT(aic1.calculate() < aic2.calculate());
        }

        TVector2Vec centres;
        maths::CSampling::multivariateNormalSample(mean, covariance, 200, centres);

        maths::CKMeansFast<TVector2> kmeans;
        kmeans.setPoints(samples);
        for (std::size_t t = 0u; t < centres.size(); t += 2) {
            TVector2Vec tcentres(&centres[t], &centres[t+2]);
            kmeans.setCentres(tcentres);

            kmeans.run(10);
            TVector2VecVec clusters;
            kmeans.clusters(clusters);

            maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_BIC>  bic2(clusters);
            maths::CSphericalGaussianInfoCriterion<TVector2, maths::E_AICc> aic2(clusters);

            LOG_DEBUG("1 cluster BIC = " << bic1.calculate());
            LOG_DEBUG("2 cluster BIC = " << bic2.calculate());
            CPPUNIT_ASSERT(bic1.calculate() < bic2.calculate());

            LOG_DEBUG("1 cluster AIC = " << aic1.calculate());
            LOG_DEBUG("2 cluster AIC = " << aic2.calculate());
            CPPUNIT_ASSERT(aic1.calculate() < aic2.calculate());
        }
    }
}

void CInformationCriteriaTest::testGaussianWithSphericalCluster(void) {
    LOG_DEBUG("+--------------------------------------------------------------+");
    LOG_DEBUG("|  CInformationCriteriaTest::testGaussianWithSphericalCluster  |");
    LOG_DEBUG("+--------------------------------------------------------------+");

    typedef maths::CSphericalCluster<TVector2>::Type TSphericalCluster2;
    typedef std::vector<TSphericalCluster2> TSphericalCluster2Vec;
    typedef std::vector<TSphericalCluster2Vec> TSphericalCluster2VecVec;

    maths::CSampling::seed();

    double means_[][2]     = { { 10.0, 20.0 }, { 12.0, 30.0 } };
    double lowerTriangle[] = { 5.0, 0.0, 5.0 };
    TVector2Vec means;
    for (std::size_t i = 0u; i < boost::size(means_); ++i) {
        means.push_back(TVector2(boost::begin(means_[i]), boost::end(means_[i])));
    }
    TMatrix2 covariance(boost::begin(lowerTriangle),
                        boost::end(lowerTriangle));
    LOG_DEBUG("means = " << core::CContainerPrinter::print(means));
    LOG_DEBUG("covariance = " << covariance);

    for (std::size_t t = 0u; t < 10; ++t) {
        LOG_DEBUG("*** trial = " << t+1 << " ***");

        TVector2VecVec points(means.size());
        TSphericalCluster2VecVec clusters;

        for (std::size_t i = 0u; i < means.size(); ++i) {
            maths::CSampling::multivariateNormalSample(means[i], covariance, 1000, points[i]);
            TMeanVar2Accumulator moments;
            moments.add(points[i]);
            double   n = maths::CBasicStatistics::count(moments);
            TVector2 m = maths::CBasicStatistics::mean(moments);
            TVector2 v = maths::CBasicStatistics::maximumLikelihoodVariance(moments);
            TSphericalCluster2::TAnnotation countAndVariance(n, (v(0) + v(1)) / 2.0);
            TSphericalCluster2 cluster(m, countAndVariance);
            clusters.push_back(TSphericalCluster2Vec(1, cluster));
        }

        maths::CGaussianInfoCriterion<TVector2, maths::E_BIC> bicPoints;
        bicPoints.add(points);
        maths::CGaussianInfoCriterion<TSphericalCluster2, maths::E_BIC> bicClusters;
        bicClusters.add(clusters);
        LOG_DEBUG("BIC points  = " << bicPoints.calculate());
        LOG_DEBUG("BIC clusters  = " << bicClusters.calculate());

        CPPUNIT_ASSERT_DOUBLES_EQUAL(bicPoints.calculate(),
                                     bicClusters.calculate(),
                                     2e-3 * bicPoints.calculate());

        maths::CGaussianInfoCriterion<TVector2, maths::E_AICc> aicPoints;
        aicPoints.add(points);
        maths::CGaussianInfoCriterion<TSphericalCluster2, maths::E_AICc> aicClusters;
        aicClusters.add(clusters);
        LOG_DEBUG("AICc points   = " << aicPoints.calculate());
        LOG_DEBUG("AICc clusters = " << aicClusters.calculate());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(aicPoints.calculate(),
                                     aicClusters.calculate(),
                                     2e-3 * aicPoints.calculate());
    }
}

CppUnit::Test *CInformationCriteriaTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CInformationCriteriaTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CInformationCriteriaTest>(
                               "CInformationCriteriaTest::testSphericalGaussian",
                               &CInformationCriteriaTest::testSphericalGaussian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CInformationCriteriaTest>(
                               "CInformationCriteriaTest::testSphericalGaussianWithSphericalCluster",
                               &CInformationCriteriaTest::testSphericalGaussianWithSphericalCluster) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CInformationCriteriaTest>(
                               "CInformationCriteriaTest::testGaussian",
                               &CInformationCriteriaTest::testGaussian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CInformationCriteriaTest>(
                               "CInformationCriteriaTest::testGaussianWithSphericalCluster",
                               &CInformationCriteriaTest::testGaussianWithSphericalCluster) );

    return suiteOfTests;
}
