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

#include <maths/common/CInformationCriteria.h>
#include <maths/common/CKMeans.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraTools.h>
#include <maths/common/CSampling.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CInformationCriteriaTest)

using namespace ml;

namespace {

using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TVector2 = maths::common::CVectorNx1<double, 2>;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecCItr = TVector2Vec::const_iterator;
using TVector2VecVec = std::vector<TVector2Vec>;
using TMeanVar2Accumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator;
using TMatrix2 = maths::common::CSymmetricMatrixNxN<double, 2>;
using TMatrix2Vec = std::vector<TMatrix2>;
using TVector4 = maths::common::CVectorNx1<double, 4>;
using TVector4Vec = std::vector<TVector4>;
using TMeanVar4Accumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<TVector4>::TAccumulator;
using TMatrix4 = maths::common::CSymmetricMatrixNxN<double, 4>;
using TMatrix4Vec = std::vector<TMatrix4>;

template<typename POINT>
double logfSphericalGaussian(const POINT& mean, double variance, const POINT& x) {
    double d = static_cast<double>(x.dimension());
    double r = (x - mean).euclidean();
    return -0.5 * (d * std::log(boost::math::double_constants::two_pi * variance) +
                   r * r / variance);
}

template<typename POINT, typename MATRIX>
double logfGaussian(const POINT& mean, const MATRIX& covariance, const POINT& x) {
    double result;
    maths::common::gaussianLogLikelihood(covariance, x - mean, result);
    return result;
}
}

BOOST_AUTO_TEST_CASE(testSphericalGaussian) {
    // Check that the information criterion values are the expected
    // values for the generating distribution.

    maths::common::CSampling::seed();

    {
        double variance = 5.0;
        double mean_[] = {10.0, 20.0};
        double lowerTriangle[] = {variance, 0.0, variance};

        TVector2 mean(std::begin(mean_), std::end(mean_));
        TMatrix2 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
        LOG_DEBUG(<< "mean = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);

        TVector2Vec samples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::common::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            likelihood += -2.0 * logfSphericalGaussian(mean, variance, samples[i]) +
                          2.0 * std::log(upper);
        }
        double expectedAICc = likelihood + 6.0 + 12.0 / (n - 4.0);
        double expectedBIC = likelihood + 3.0 * std::log(n);

        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG(<< "expected BIC  = " << expectedBIC);
        LOG_DEBUG(<< "BIC           = " << bic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedBIC, bic.calculate(), 2e-3 * expectedBIC);

        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG(<< "expected AICc = " << expectedAICc);
        LOG_DEBUG(<< "AICc          = " << aic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedAICc, aic.calculate(), 2e-3 * expectedAICc);
    }
    {
        double variance = 8.0;
        double mean_[] = {-5.0, 30.0, 2.0, 7.9};
        double lowerTriangle[] = {variance, 0.0, variance, 0.0, 0.0,
                                  variance, 0.0, 0.0,      0.0, variance};

        TVector4 mean(std::begin(mean_), std::end(mean_));
        TMatrix4 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
        LOG_DEBUG(<< "mean = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);

        TVector4Vec samples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 500, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::common::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            likelihood += -2.0 * logfSphericalGaussian(mean, variance, samples[i]) +
                          4.0 * std::log(upper);
        }
        double expectedAICc = likelihood + 10.0 + 30.0 / (n - 6.0);
        double expectedBIC = likelihood + 5.0 * std::log(n);

        maths::common::CSphericalGaussianInfoCriterion<TVector4, maths::common::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG(<< "expected BIC = " << expectedBIC);
        LOG_DEBUG(<< "BIC          = " << bic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedBIC, bic.calculate(), 2e-3 * expectedBIC);

        maths::common::CSphericalGaussianInfoCriterion<TVector4, maths::common::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG(<< "expected AICc = " << expectedAICc);
        LOG_DEBUG(<< "AICc          = " << aic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedAICc, aic.calculate(), 2e-3 * expectedAICc);
    }

    // Check that they correctly distinguish the best fit model.

    test::CRandomNumbers rng;

    {
        double variance = 40.0;
        double mean_[] = {15.0, 27.0};
        double lowerTriangle[] = {variance, 0.0, variance};

        TVector2 mean(std::begin(mean_), std::end(mean_));
        TMatrix2 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
        LOG_DEBUG(<< "mean = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);

        TVector2Vec samples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic1(samples);
        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic1(samples);

        for (std::size_t t = 0; t < 100; ++t) {
            rng.random_shuffle(samples.begin(), samples.end());

            TSizeVec split;
            rng.generateUniformSamples(100, 900, 1, split);

            TVector2Vec samples1(&samples[0], &samples[split[0]]);
            TVector2Vec samples2(&samples[split[0]], &samples[999]);

            maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic2(samples);
            maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic2(
                samples);
            bic2.add(samples1);
            bic2.add(samples2);
            aic2.add(samples1);
            aic2.add(samples2);

            LOG_TRACE(<< "1 cluster BIC = " << bic1.calculate());
            LOG_TRACE(<< "2 cluster BIC = " << bic2.calculate());
            BOOST_TEST_REQUIRE(bic1.calculate() < bic2.calculate());

            LOG_TRACE(<< "1 cluster AIC = " << aic1.calculate());
            LOG_TRACE(<< "2 cluster AIC = " << aic2.calculate());
            BOOST_TEST_REQUIRE(aic1.calculate() < aic2.calculate());
        }

        TVector2Vec centres;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 200, centres);

        maths::common::CKMeans<TVector2> kmeans;
        kmeans.setPoints(samples);
        for (std::size_t t = 0; t < centres.size(); t += 2) {
            TVector2Vec tcentres(&centres[t], &centres[t + 2]);
            kmeans.setCentres(tcentres);

            kmeans.run(10);
            TVector2VecVec clusters;
            kmeans.clusters(clusters);

            maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic2(
                clusters);
            maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic2(
                clusters);

            LOG_TRACE(<< "1 cluster BIC = " << bic1.calculate());
            LOG_TRACE(<< "2 cluster BIC = " << bic2.calculate());
            BOOST_TEST_REQUIRE(bic1.calculate() < bic2.calculate());

            LOG_TRACE(<< "1 cluster AIC = " << aic1.calculate());
            LOG_TRACE(<< "2 cluster AIC = " << aic2.calculate());
            BOOST_TEST_REQUIRE(aic1.calculate() < aic2.calculate());
        }
    }
}

BOOST_AUTO_TEST_CASE(testSphericalGaussianWithSphericalCluster) {
    // The idea of this test is simply to check that we get the
    // same result working with clusters of points or their
    // spherical cluster representation.

    using TSphericalCluster2 = maths::common::CSphericalCluster<TVector2>::Type;
    using TSphericalCluster2Vec = std::vector<TSphericalCluster2>;
    using TSphericalCluster2VecVec = std::vector<TSphericalCluster2Vec>;

    maths::common::CSampling::seed();

    double means_[][2] = {{10.0, 20.0}, {12.0, 30.0}};
    double lowerTriangle[] = {5.0, 0.0, 5.0};
    TVector2Vec means;
    for (std::size_t i = 0; i < boost::size(means_); ++i) {
        means.push_back(TVector2(std::begin(means_[i]), std::end(means_[i])));
    }
    TMatrix2 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
    LOG_DEBUG(<< "means = " << core::CContainerPrinter::print(means));
    LOG_DEBUG(<< "covariance = " << covariance);

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "*** trial = " << t + 1 << " ***");

        TVector2VecVec points(means.size());
        TSphericalCluster2VecVec clusters;

        for (std::size_t i = 0; i < means.size(); ++i) {
            maths::common::CSampling::multivariateNormalSample(means[i], covariance,
                                                               1000, points[i]);
            TMeanVar2Accumulator moments;
            moments.add(points[i]);
            double n = maths::common::CBasicStatistics::count(moments);
            TVector2 m = maths::common::CBasicStatistics::mean(moments);
            TVector2 v = maths::common::CBasicStatistics::maximumLikelihoodVariance(moments);
            TSphericalCluster2::TAnnotation countAndVariance(n, (v(0) + v(1)) / 2.0);
            TSphericalCluster2 cluster(m, countAndVariance);
            clusters.push_back(TSphericalCluster2Vec(1, cluster));
        }

        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bicPoints;
        bicPoints.add(points);
        maths::common::CSphericalGaussianInfoCriterion<TSphericalCluster2, maths::common::E_BIC> bicClusters;
        bicClusters.add(clusters);
        LOG_DEBUG(<< "BIC points  = " << bicPoints.calculate());
        LOG_DEBUG(<< "BIC clusters  = " << bicClusters.calculate());

        BOOST_REQUIRE_CLOSE_ABSOLUTE(bicPoints.calculate(), bicClusters.calculate(),
                                     1e-10 * bicPoints.calculate());

        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aicPoints;
        aicPoints.add(points);
        maths::common::CSphericalGaussianInfoCriterion<TSphericalCluster2, maths::common::E_AICc> aicClusters;
        aicClusters.add(clusters);
        LOG_DEBUG(<< "AICc points   = " << aicPoints.calculate());
        LOG_DEBUG(<< "AICc clusters = " << aicClusters.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(aicPoints.calculate(), aicClusters.calculate(),
                                     1e-10 * aicPoints.calculate());
    }
}

BOOST_AUTO_TEST_CASE(testGaussian) {
    maths::common::CSampling::seed();

    {
        double mean_[] = {10.0, 20.0};
        double lowerTriangle[] = {5.0, 1.0, 5.0};

        TVector2 mean(std::begin(mean_), std::end(mean_));
        TMatrix2 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
        LOG_DEBUG(<< "mean = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);

        TVector2Vec samples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::common::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            likelihood += -2.0 * logfGaussian(mean, covariance, samples[i]) +
                          2.0 * std::log(upper);
        }
        double expectedAICc = likelihood + 10.0 + 30.0 / (n - 6.0);
        double expectedBIC = likelihood + 5.0 * std::log(n);

        maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG(<< "expected BIC  = " << expectedBIC);
        LOG_DEBUG(<< "BIC           = " << bic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedBIC, bic.calculate(), 2e-3 * expectedBIC);

        maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG(<< "expected AICc = " << expectedAICc);
        LOG_DEBUG(<< "AICc          = " << aic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedAICc, aic.calculate(), 2e-3 * expectedAICc);
    }
    {
        double mean_[] = {-5.0, 30.0, 2.0, 7.9};
        double lowerTriangle[] = {8.0, 1.0, 8.0, 0.0, 0.0,
                                  8.0, 0.0, 2.0, 0.5, 8.0};

        TVector4 mean(std::begin(mean_), std::end(mean_));
        TMatrix4 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
        LOG_DEBUG(<< "mean = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);

        TVector4Vec samples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 500, samples);

        double n = static_cast<double>(samples.size());
        double upper = maths::common::information_criteria_detail::confidence(n - 1.0);

        double likelihood = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            likelihood += -2.0 * logfGaussian(mean, covariance, samples[i]) +
                          4.0 * std::log(upper);
        }
        double expectedAICc = likelihood + 28.0 + 210.0 / (n - 15.0);
        double expectedBIC = likelihood + 14.0 * std::log(n);

        maths::common::CGaussianInfoCriterion<TVector4, maths::common::E_BIC> bic;
        bic.add(samples);
        LOG_DEBUG(<< "expected BIC = " << expectedBIC);
        LOG_DEBUG(<< "BIC          = " << bic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedBIC, bic.calculate(), 2e-3 * expectedBIC);

        maths::common::CGaussianInfoCriterion<TVector4, maths::common::E_AICc> aic;
        aic.add(samples);
        LOG_DEBUG(<< "expected AICc = " << expectedAICc);
        LOG_DEBUG(<< "AICc          = " << aic.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedAICc, aic.calculate(), 2e-3 * expectedAICc);
    }

    // Check that they correctly distinguish the best fit model.

    test::CRandomNumbers rng;

    {
        double mean_[] = {15.0, 27.0};
        double lowerTriangle[] = {40.0, 5.0, 40.0};

        TVector2 mean(std::begin(mean_), std::end(mean_));
        TMatrix2 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
        LOG_DEBUG(<< "mean = " << mean);
        LOG_DEBUG(<< "covariance = " << covariance);

        TVector2Vec samples;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 1000, samples);

        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic1(samples);
        maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic1(samples);

        for (std::size_t t = 0; t < 100; ++t) {
            rng.random_shuffle(samples.begin(), samples.end());

            TSizeVec split;
            rng.generateUniformSamples(100, 900, 1, split);

            TVector2Vec samples1(&samples[0], &samples[split[0]]);
            TVector2Vec samples2(&samples[split[0]], &samples[999]);

            maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic2(samples);
            maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic2(samples);
            bic2.add(samples1);
            bic2.add(samples2);
            aic2.add(samples1);
            aic2.add(samples2);

            LOG_TRACE(<< "1 cluster BIC = " << bic1.calculate());
            LOG_TRACE(<< "2 cluster BIC = " << bic2.calculate());
            BOOST_TEST_REQUIRE(bic1.calculate() < bic2.calculate());

            LOG_TRACE(<< "1 cluster AIC = " << aic1.calculate());
            LOG_TRACE(<< "2 cluster AIC = " << aic2.calculate());
            BOOST_TEST_REQUIRE(aic1.calculate() < aic2.calculate());
        }

        TVector2Vec centres;
        maths::common::CSampling::multivariateNormalSample(mean, covariance, 200, centres);

        maths::common::CKMeans<TVector2> kmeans;
        kmeans.setPoints(samples);
        for (std::size_t t = 0; t < centres.size(); t += 2) {
            TVector2Vec tcentres(&centres[t], &centres[t + 2]);
            kmeans.setCentres(tcentres);

            kmeans.run(10);
            TVector2VecVec clusters;
            kmeans.clusters(clusters);

            maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_BIC> bic2(
                clusters);
            maths::common::CSphericalGaussianInfoCriterion<TVector2, maths::common::E_AICc> aic2(
                clusters);

            LOG_TRACE(<< "1 cluster BIC = " << bic1.calculate());
            LOG_TRACE(<< "2 cluster BIC = " << bic2.calculate());
            BOOST_TEST_REQUIRE(bic1.calculate() < bic2.calculate());

            LOG_TRACE(<< "1 cluster AIC = " << aic1.calculate());
            LOG_TRACE(<< "2 cluster AIC = " << aic2.calculate());
            BOOST_TEST_REQUIRE(aic1.calculate() < aic2.calculate());
        }
    }
}

BOOST_AUTO_TEST_CASE(testGaussianWithSphericalCluster) {
    using TSphericalCluster2 = maths::common::CSphericalCluster<TVector2>::Type;
    using TSphericalCluster2Vec = std::vector<TSphericalCluster2>;
    using TSphericalCluster2VecVec = std::vector<TSphericalCluster2Vec>;

    maths::common::CSampling::seed();

    double means_[][2] = {{10.0, 20.0}, {12.0, 30.0}};
    double lowerTriangle[] = {5.0, 0.0, 5.0};
    TVector2Vec means;
    for (std::size_t i = 0; i < boost::size(means_); ++i) {
        means.push_back(TVector2(std::begin(means_[i]), std::end(means_[i])));
    }
    TMatrix2 covariance(std::begin(lowerTriangle), std::end(lowerTriangle));
    LOG_DEBUG(<< "means = " << core::CContainerPrinter::print(means));
    LOG_DEBUG(<< "covariance = " << covariance);

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "*** trial = " << t + 1 << " ***");

        TVector2VecVec points(means.size());
        TSphericalCluster2VecVec clusters;

        for (std::size_t i = 0; i < means.size(); ++i) {
            maths::common::CSampling::multivariateNormalSample(means[i], covariance,
                                                               1000, points[i]);
            TMeanVar2Accumulator moments;
            moments.add(points[i]);
            double n = maths::common::CBasicStatistics::count(moments);
            TVector2 m = maths::common::CBasicStatistics::mean(moments);
            TVector2 v = maths::common::CBasicStatistics::maximumLikelihoodVariance(moments);
            TSphericalCluster2::TAnnotation countAndVariance(n, (v(0) + v(1)) / 2.0);
            TSphericalCluster2 cluster(m, countAndVariance);
            clusters.push_back(TSphericalCluster2Vec(1, cluster));
        }

        maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_BIC> bicPoints;
        bicPoints.add(points);
        maths::common::CGaussianInfoCriterion<TSphericalCluster2, maths::common::E_BIC> bicClusters;
        bicClusters.add(clusters);
        LOG_DEBUG(<< "BIC points  = " << bicPoints.calculate());
        LOG_DEBUG(<< "BIC clusters  = " << bicClusters.calculate());

        BOOST_REQUIRE_CLOSE_ABSOLUTE(bicPoints.calculate(), bicClusters.calculate(),
                                     2e-3 * bicPoints.calculate());

        maths::common::CGaussianInfoCriterion<TVector2, maths::common::E_AICc> aicPoints;
        aicPoints.add(points);
        maths::common::CGaussianInfoCriterion<TSphericalCluster2, maths::common::E_AICc> aicClusters;
        aicClusters.add(clusters);
        LOG_DEBUG(<< "AICc points   = " << aicPoints.calculate());
        LOG_DEBUG(<< "AICc clusters = " << aicClusters.calculate());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(aicPoints.calculate(), aicClusters.calculate(),
                                     2e-3 * aicPoints.calculate());
    }
}

BOOST_AUTO_TEST_SUITE_END()
