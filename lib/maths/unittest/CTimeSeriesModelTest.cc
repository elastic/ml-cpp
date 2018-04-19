/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesModelTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CDecayRateController.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CMultivariateMultimodalPrior.h>
#include <maths/CMultivariateNormalConjugate.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesDecompositionStub.h>
#include <maths/CTimeSeriesModel.h>
#include <maths/CXMeansOnline.h>
#include <maths/CXMeansOnline1d.h>

#include <test/CRandomNumbers.h>

#include <cmath>
#include <memory>

using namespace ml;

namespace {
using TBool2Vec = core::CSmallVector<bool, 2>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDoubleWeightsAry1Vec = maths_t::TDoubleWeightsAry1Vec;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TDouble2VecWeightsAry = maths_t::TDouble2VecWeightsAry;
using TDouble2VecWeightsAryVec = std::vector<TDouble2VecWeightsAry>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
using TDouble10VecWeightsAry1Vec = maths_t::TDouble10VecWeightsAry1Vec;
using TSize1Vec = core::CSmallVector<std::size_t, 1>;
using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;
using TTail10Vec = core::CSmallVector<maths_t::ETail, 10>;
using TTimeDouble2VecSizeTrVec = maths::CModel::TTimeDouble2VecSizeTrVec;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulator2Vec = core::CSmallVector<TMeanAccumulator, 2>;
using TDecompositionPtr = std::shared_ptr<maths::CTimeSeriesDecompositionInterface>;
using TDecompositionPtr10Vec = core::CSmallVector<TDecompositionPtr, 10>;
using TDecayRateController2Ary = maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary;
using TSetWeightsFunc = void (*)(double, std::size_t, TDouble2VecWeightsAry&);

const double MINIMUM_SEASONAL_SCALE{0.25};
const double MINIMUM_SIGNIFICANT_CORRELATION{0.4};
const double DECAY_RATE{0.0005};
const std::size_t TAG{0u};

//! \brief Implements the allocator for new correlate priors.
class CTimeSeriesCorrelateModelAllocator : public maths::CTimeSeriesCorrelateModelAllocator {
public:
    //! Check if we can still allocate any correlations.
    virtual bool areAllocationsAllowed() const { return true; }

    //! Check if \p correlations exceeds the memory limit.
    virtual bool exceedsLimit(std::size_t /*correlations*/) const {
        return false;
    }

    //! Get the maximum number of correlations we should model.
    virtual std::size_t maxNumberCorrelations() const { return 5000; }

    //! Get the chunk size in which to allocate correlations.
    virtual std::size_t chunkSize() const { return 500; }

    //! Create a new prior for a correlation model.
    virtual TMultivariatePriorPtr newPrior() const {
        return TMultivariatePriorPtr(maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                                         maths_t::E_ContinuousData, DECAY_RATE)
                                         .clone());
    }
};

maths::CModelParams params(core_t::TTime bucketLength) {
    using TTimeDoubleMap = std::map<core_t::TTime, double>;
    static TTimeDoubleMap learnRates;
    learnRates[bucketLength] = static_cast<double>(bucketLength) / 1800.0;
    double minimumSeasonalVarianceScale{MINIMUM_SEASONAL_SCALE};
    return maths::CModelParams{bucketLength, learnRates[bucketLength],
                               DECAY_RATE, minimumSeasonalVarianceScale};
}

maths::CNormalMeanPrecConjugate univariateNormal() {
    return maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE);
}

maths::CLogNormalMeanPrecConjugate univariateLogNormal() {
    return maths::CLogNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 0.0, DECAY_RATE);
}

maths::CMultimodalPrior univariateMultimodal() {
    maths::CXMeansOnline1d clusterer{maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight, DECAY_RATE};
    return maths::CMultimodalPrior{maths_t::E_ContinuousData, clusterer,
                                   univariateNormal(), DECAY_RATE};
}

maths::CMultivariateNormalConjugate<3> multivariateNormal() {
    return maths::CMultivariateNormalConjugate<3>::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE);
}

maths::CMultivariateMultimodalPrior<3> multivariateMultimodal() {
    maths::CXMeansOnline<maths::CFloatStorage, 3> clusterer(
        maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight, DECAY_RATE);
    return maths::CMultivariateMultimodalPrior<3>(
        maths_t::E_ContinuousData, clusterer,
        maths::CMultivariateNormalConjugate<3>::nonInformativePrior(
            maths_t::E_ContinuousData, DECAY_RATE),
        DECAY_RATE);
}

maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary
decayRateControllers(std::size_t dimension) {
    return {{maths::CDecayRateController(maths::CDecayRateController::E_PredictionBias |
                                             maths::CDecayRateController::E_PredictionErrorIncrease,
                                         dimension),
             maths::CDecayRateController(maths::CDecayRateController::E_PredictionBias |
                                             maths::CDecayRateController::E_PredictionErrorIncrease |
                                             maths::CDecayRateController::E_PredictionErrorDecrease,
                                         dimension)}};
}

void reinitializePrior(double learnRate,
                       const maths::CMultivariateTimeSeriesModel& model,
                       TDecompositionPtr10Vec& trends,
                       maths::CMultivariatePrior& prior,
                       TDecayRateController2Ary* controllers = nullptr) {
    prior.setToNonInformative(0.0, prior.decayRate());
    TDouble10Vec1Vec detrended_{TDouble10Vec(3)};
    for (const auto& value : model.slidingWindow()) {
        for (std::size_t i = 0u; i < value.second.size(); ++i) {
            detrended_[0][i] = trends[i]->detrend(value.first, value.second[i], 0.0);
        }
        prior.addSamples(detrended_,
                         {maths_t::countWeight(learnRate, value.second.size())});
    }
    if (controllers) {
        for (auto& trend : trends) {
            trend->decayRate(trend->decayRate() / (*controllers)[0].multiplier());
        }
        prior.decayRate(prior.decayRate() / (*controllers)[1].multiplier());
        (*controllers)[0].reset();
        (*controllers)[1].reset();
    }
}
}

void CTimeSeriesModelTest::testClone() {
    // Test all the state is cloned.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    {
        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CTimeSeriesCorrelations correlations{MINIMUM_SIGNIFICANT_CORRELATION, DECAY_RATE};
        maths::CUnivariateTimeSeriesModel model(params(bucketLength), 1, trend,
                                                univariateNormal(), &controllers);
        model.modelCorrelations(correlations);

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        uint64_t checksum1 = model.checksum();
        std::unique_ptr<maths::CUnivariateTimeSeriesModel> clone1{model.clone(1)};
        uint64_t checksum2 = clone1->checksum();
        CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
        std::unique_ptr<maths::CUnivariateTimeSeriesModel> clone2{model.clone(2)};
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), clone2->identifier());
    }
    {
        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(3);
        maths::CMultivariateTimeSeriesModel model(
            params(bucketLength), trend, multivariateNormal(), &controllers);

        TDoubleVec mean{13.0, 9.0, 10.0};
        TDoubleVecVec covariance{{3.5, 2.9, 0.5}, {2.9, 3.6, 0.1}, {0.5, 0.1, 2.1}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (const auto& sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        uint64_t checksum1 = model.checksum();
        std::unique_ptr<maths::CMultivariateTimeSeriesModel> clone1{model.clone(1)};
        uint64_t checksum2 = clone1->checksum();
        CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
        std::unique_ptr<maths::CMultivariateTimeSeriesModel> clone2{model.clone(2)};
        uint64_t checksum3 = clone2->checksum();
        CPPUNIT_ASSERT_EQUAL(checksum1, checksum3);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), clone2->identifier());
    }
}

void CTimeSeriesModelTest::testMode() {
    // Test that we get the modes we expect based versus updating the trend(s)
    // and prior directly.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate no trend");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);

        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

        core_t::TTime time{0};
        for (auto sample : samples) {
            trend.addPoint(time, sample);
            TDouble1Vec sample_{trend.detrend(time, sample, 0.0)};
            prior.addSamples(sample_, maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);
            time += bucketLength;
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        time = 0;
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }
        double expectedMode{maths::CBasicStatistics::mean(trend.baseline(time)) +
                            prior.marginalLikelihoodMode()};
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(1)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode[0]);
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), mode.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode, mode[0], 1e-3 * expectedMode);
    }

    LOG_DEBUG(<< "Univariate trend");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

        core_t::TTime time{0};
        for (auto& sample : samples) {
            sample += 20.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                             static_cast<double>(time) /
                                             static_cast<double>(core::constants::DAY));
            time += bucketLength;
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        time = 0;
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            if (trend.addPoint(time, sample)) {
                prior.setToNonInformative(0.0, DECAY_RATE);
                for (const auto& value : model.slidingWindow()) {
                    prior.addSamples({trend.detrend(value.first, value.second, 0.0)},
                                     maths_t::CUnitWeights::SINGLE_UNIT);
                }
            }
            TDouble1Vec sample_{trend.detrend(time, sample, 0.0)};
            prior.addSamples(sample_, maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);
            time += bucketLength;
        }

        double expectedMode{maths::CBasicStatistics::mean(trend.baseline(time)) +
                            prior.marginalLikelihoodMode()};
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(1)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode[0]);
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), mode.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode, mode[0], 1e-3 * expectedMode);
    }

    LOG_DEBUG(<< "Multivariate no trend");
    {
        TDoubleVec mean{10.0, 15.0, 11.0};
        TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params(bucketLength), *trends[0], prior};

        core_t::TTime time{0};
        for (const auto& sample : samples) {
            TDouble10Vec1Vec detrended{TDouble10Vec(3)};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                trends[i]->addPoint(time, sample[i]);
                detrended[0][i] = trends[i]->detrend(time, sample[i], 0.0);
            }
            prior.addSamples(detrended,
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
            prior.propagateForwardsByTime(1.0);
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        time = 0;
        for (const auto& sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }
        TDouble2Vec expectedMode(prior.marginalLikelihoodMode(
            maths_t::CUnitWeights::unit<TDouble10Vec>(3)));
        for (std::size_t i = 0u; i < trends.size(); ++i) {
            expectedMode[i] += maths::CBasicStatistics::mean(trends[i]->baseline(time));
        }
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(3)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode);
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), mode.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode[0], mode[0], 0.02 * expectedMode[0]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode[1], mode[1], 0.02 * expectedMode[0]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode[2], mode[2], 0.02 * expectedMode[0]);
    }

    LOG_DEBUG(<< "Multivariate trend");
    {
        TDoubleVec mean{10.0, 15.0, 11.0};
        TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        double learnRate{params(bucketLength).learnRate()};
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params(bucketLength), *trends[0], prior};

        core_t::TTime time{0};
        for (auto& sample : samples) {
            double amplitude{10.0};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                sample[i] +=
                    30.0 + amplitude *
                               std::sin(boost::math::double_constants::two_pi *
                                        static_cast<double>(time) /
                                        static_cast<double>(core::constants::DAY));
                amplitude += 4.0;
            }
            time += bucketLength;
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        time = 0;
        for (const auto& sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});

            bool reinitialize{false};
            TDouble10Vec1Vec detrended{TDouble10Vec(3)};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                reinitialize |= trends[i]->addPoint(time, sample[i]);
                detrended[0][i] = trends[i]->detrend(time, sample[i], 0.0);
            }
            if (reinitialize) {
                reinitializePrior(learnRate, model, trends, prior);
            }
            prior.addSamples(detrended,
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
            prior.propagateForwardsByTime(1.0);

            time += bucketLength;
        }
        TDouble2Vec expectedMode(prior.marginalLikelihoodMode(
            maths_t::CUnitWeights::unit<TDouble10Vec>(3)));
        for (std::size_t i = 0u; i < trends.size(); ++i) {
            expectedMode[i] += maths::CBasicStatistics::mean(trends[i]->baseline(time));
        }
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(3)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode);
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), mode.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode[0], mode[0], 1e-3 * expectedMode[0]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode[1], mode[1], 1e-3 * expectedMode[1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMode[2], mode[2], 1e-3 * expectedMode[2]);
    }
}

void CTimeSeriesModelTest::testAddBucketValue() {
    // Test that the prior support is correctly updated to account
    // for negative bucket values.

    core_t::TTime bucketLength{600};
    maths::CTimeSeriesDecompositionStub trend;
    maths::CLogNormalMeanPrecConjugate prior{univariateLogNormal()};
    maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

    TTimeDouble2VecSizeTrVec samples{
        core::make_triple(core_t::TTime{20}, TDouble2Vec{3.5}, TAG),
        core::make_triple(core_t::TTime{12}, TDouble2Vec{3.9}, TAG),
        core::make_triple(core_t::TTime{18}, TDouble2Vec{2.1}, TAG),
        core::make_triple(core_t::TTime{12}, TDouble2Vec{1.2}, TAG)};
    TDoubleVec weights{1.0, 1.5, 0.9, 1.9};
    TDouble2VecWeightsAryVec modelWeights{
        maths_t::countWeight(TDouble2Vec{weights[0]}),
        maths_t::countWeight(TDouble2Vec{weights[1]}),
        maths_t::countWeight(TDouble2Vec{weights[2]}),
        maths_t::countWeight(TDouble2Vec{weights[3]})};

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        prior.addSamples({samples[i].second[0]}, {maths_t::countWeight(weights[i])});
    }
    prior.propagateForwardsByTime(1.0);
    prior.adjustOffset({-1.0}, maths_t::CUnitWeights::SINGLE_UNIT);

    maths::CModelAddSamplesParams params;
    params.integer(false).propagationInterval(1.0).trendWeights(modelWeights).priorWeights(modelWeights);
    model.addSamples(params, samples);
    model.addBucketValue({core::make_triple(core_t::TTime{20}, TDouble2Vec{-1.0}, TAG)});

    CPPUNIT_ASSERT_EQUAL(prior.checksum(), model.prior().checksum());
}

void CTimeSeriesModelTest::testAddSamples() {
    // Test: 1) Test multiple samples
    //       2) Test propagation interval
    //       3) Test decay rate control

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{1800};

    LOG_DEBUG(<< "Multiple samples univariate");
    {
        maths::CTimeSeriesDecompositionStub trend;
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

        TTimeDouble2VecSizeTrVec samples{
            core::make_triple(core_t::TTime{20}, TDouble2Vec{3.5}, TAG),
            core::make_triple(core_t::TTime{12}, TDouble2Vec{3.9}, TAG),
            core::make_triple(core_t::TTime{18}, TDouble2Vec{2.1}, TAG)};
        TDoubleVec weights{1.0, 1.5, 0.9};
        TDouble2VecWeightsAryVec modelWeights{
            maths_t::countWeight(TDouble2Vec{weights[0]}),
            maths_t::countWeight(TDouble2Vec{weights[1]}),
            maths_t::countWeight(TDouble2Vec{weights[2]})};

        maths::CModelAddSamplesParams params;
        params.integer(false)
            .propagationInterval(1.0)
            .trendWeights(modelWeights)
            .priorWeights(modelWeights);

        model.addSamples(params, samples);

        trend.addPoint(samples[1].first, samples[1].second[0],
                       maths_t::countWeight(weights[1]));
        trend.addPoint(samples[2].first, samples[2].second[0],
                       maths_t::countWeight(weights[2]));
        trend.addPoint(samples[0].first, samples[0].second[0],
                       maths_t::countWeight(weights[0]));
        prior.addSamples(
            {samples[2].second[0], samples[0].second[0], samples[1].second[0]},
            {maths_t::countWeight(weights[2]), maths_t::countWeight(weights[0]),
             maths_t::countWeight(weights[1])});
        prior.propagateForwardsByTime(1.0);

        uint64_t checksum1{trend.checksum()};
        uint64_t checksum2{model.trend().checksum()};
        LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
        CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
        checksum1 = prior.checksum();
        checksum2 = model.prior().checksum();
        LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
        CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
    }

    LOG_DEBUG(<< "Multiple samples multivariate");
    {
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params(bucketLength), *trends[0], prior};

        TTimeDouble2VecSizeTrVec samples{
            core::make_triple(core_t::TTime{20}, TDouble2Vec{3.5, 3.4, 3.3}, TAG),
            core::make_triple(core_t::TTime{12}, TDouble2Vec{3.9, 3.8, 3.7}, TAG),
            core::make_triple(core_t::TTime{18}, TDouble2Vec{2.1, 2.0, 1.9}, TAG)};
        double weights[][3]{{1.0, 1.1, 1.2}, {1.5, 1.6, 1.7}, {0.9, 1.0, 1.1}};
        TDouble2VecWeightsAryVec modelWeights{
            maths_t::countWeight(TDouble2Vec(weights[0], weights[0] + 3)),
            maths_t::countWeight(TDouble2Vec(weights[1], weights[1] + 3)),
            maths_t::countWeight(TDouble2Vec(weights[2], weights[2] + 3))};

        maths::CModelAddSamplesParams params;
        params.integer(false)
            .propagationInterval(1.0)
            .trendWeights(modelWeights)
            .priorWeights(modelWeights);

        model.addSamples(params, samples);

        for (std::size_t i = 0u; i < trends.size(); ++i) {
            trends[i]->addPoint(samples[1].first, samples[1].second[i],
                                maths_t::countWeight(weights[0][i]));
            trends[i]->addPoint(samples[2].first, samples[2].second[i],
                                maths_t::countWeight(weights[1][i]));
            trends[i]->addPoint(samples[0].first, samples[0].second[i],
                                maths_t::countWeight(weights[2][i]));
        }
        TDouble10Vec1Vec samples_{samples[2].second, samples[0].second,
                                  samples[1].second};
        TDouble10VecWeightsAry1Vec weights_{
            maths_t::countWeight(TDouble10Vec(weights[2], weights[2] + 3)),
            maths_t::countWeight(TDouble10Vec(weights[0], weights[0] + 3)),
            maths_t::countWeight(TDouble10Vec(weights[1], weights[1] + 3))};
        prior.addSamples(samples_, weights_);
        prior.propagateForwardsByTime(1.0);

        for (std::size_t i = 0u; i < trends.size(); ++i) {
            uint64_t checksum1{trends[i]->checksum()};
            uint64_t checksum2{model.trend()[i]->checksum()};
            LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
            CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
        }
        uint64_t checksum1{prior.checksum()};
        uint64_t checksum2{model.prior().checksum()};
        LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
        CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
    }

    LOG_DEBUG(<< "Propagation interval univariate");
    {
        maths::CTimeSeriesDecompositionStub trend;
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

        double interval[]{1.0, 1.1, 0.4};
        TDouble2Vec samples[]{{10.0}, {13.9}, {27.1}};
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        maths_t::setCount(TDouble2Vec{1.5}, weights[0]);
        maths_t::setWinsorisationWeight(TDouble2Vec{0.9}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{1.1}, weights[0]);

        core_t::TTime time{0};
        for (std::size_t i = 0u; i < 3; ++i) {
            TTimeDouble2VecSizeTrVec sample{core::make_triple(time, samples[i], TAG)};
            maths::CModelAddSamplesParams params;
            params.integer(false)
                .propagationInterval(interval[i])
                .trendWeights(weights)
                .priorWeights(weights);
            model.addSamples(params, sample);

            TDoubleWeightsAry1Vec weight{maths_t::CUnitWeights::UNIT};
            for (std::size_t j = 0u; j < weights[0].size(); ++j) {
                weight[0][j] = weights[0][j][0];
            }
            prior.addSamples(TDouble1Vec(samples[i]), weight);
            prior.propagateForwardsByTime(interval[i]);

            uint64_t checksum1{prior.checksum()};
            uint64_t checksum2{model.prior().checksum()};
            LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
            CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }

    LOG_DEBUG(<< "Propagation interval multivariate");
    {
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params(bucketLength), *trends[0], prior};

        double interval[]{1.0, 1.1, 0.4};
        TDouble2Vec samples[]{{13.5, 13.4, 13.3}, {13.9, 13.8, 13.7}, {20.1, 20.0, 10.9}};
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        maths_t::setCount(TDouble2Vec{1.0, 1.1, 1.2}, weights[0]);
        maths_t::setWinsorisationWeight(TDouble2Vec{0.1, 0.1, 0.2}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{2.0, 2.1, 2.2}, weights[0]);

        core_t::TTime time{0};
        for (std::size_t i = 0u; i < 3; ++i) {
            TTimeDouble2VecSizeTrVec sample{core::make_triple(time, samples[i], TAG)};
            maths::CModelAddSamplesParams params;
            params.integer(false)
                .propagationInterval(interval[i])
                .trendWeights(weights)
                .priorWeights(weights);
            model.addSamples(params, sample);

            TDouble10VecWeightsAry1Vec weight{maths_t::CUnitWeights::unit<TDouble10Vec>(3)};
            for (std::size_t j = 0u; j < weights[0].size(); ++j) {
                weight[0][j] = weights[0][j];
            }
            prior.addSamples({TDouble10Vec(samples[i])}, weight);
            prior.propagateForwardsByTime(interval[i]);

            uint64_t checksum1{prior.checksum()};
            uint64_t checksum2{model.prior().checksum()};
            LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
            CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }

    LOG_DEBUG(<< "Decay rate control univariate");
    {
        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel model(params(bucketLength), 1, trend,
                                                prior, &controllers);

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 2000, samples);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};

        core_t::TTime time{0};
        for (auto noise : samples) {
            double sample{20.0 +
                          4.0 * std::sin(boost::math::double_constants::two_pi *
                                         static_cast<double>(time) / 86400.0) +
                          (time / bucketLength > 1800 ? 10.0 : 0.0) + noise};

            TTimeDouble2VecSizeTrVec sample_{
                core::make_triple(time, TDouble2Vec{sample}, TAG)};
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, sample_);

            if (trend.addPoint(time, sample)) {
                trend.decayRate(trend.decayRate() / controllers[0].multiplier());
                prior.setToNonInformative(0.0, prior.decayRate());
                for (const auto& value : model.slidingWindow()) {
                    prior.addSamples({trend.detrend(value.first, value.second, 0.0)},
                                     maths_t::CUnitWeights::SINGLE_UNIT);
                }
                prior.decayRate(prior.decayRate() / controllers[1].multiplier());
                controllers[0].reset();
                controllers[1].reset();
            }
            double detrended{trend.detrend(time, sample, 0.0)};
            prior.addSamples({detrended}, maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);

            if (trend.initialized()) {
                double multiplier{controllers[0].multiplier(
                    {trend.mean(time)}, {{detrended}}, bucketLength,
                    model.params().learnRate(), DECAY_RATE)};
                trend.decayRate(multiplier * trend.decayRate());
            }
            if (prior.numberSamples() > 20.0) {
                double multiplier{controllers[1].multiplier(
                    {prior.marginalLikelihoodMean()},
                    {{detrended - prior.marginalLikelihoodMean()}},
                    bucketLength, model.params().learnRate(), DECAY_RATE)};
                prior.decayRate(multiplier * prior.decayRate());
            }

            uint64_t checksum1{trend.checksum()};
            uint64_t checksum2{model.trend().checksum()};
            CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
            checksum1 = prior.checksum();
            checksum2 = model.prior().checksum();
            CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }

    LOG_DEBUG(<< "Decay rate control multivariate");
    {
        double learnRate{params(bucketLength).learnRate()};
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        auto controllers = decayRateControllers(3);
        maths::CMultivariateTimeSeriesModel model{params(bucketLength),
                                                  *trends[0], prior, &controllers};

        TDoubleVecVec samples;
        {
            TDoubleVec mean{10.0, 15.0, 11.0};
            TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
            rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};

        core_t::TTime time{0};
        for (auto& sample : samples) {
            bool reinitialize{false};
            bool hasTrend{false};
            TDouble10Vec1Vec detrended{TDouble10Vec(3)};
            TDouble1Vec mean(3);

            double amplitude{10.0};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                sample[i] = 30.0 +
                            amplitude * std::sin(boost::math::double_constants::two_pi *
                                                 static_cast<double>(time) / 86400.0) +
                            (time / bucketLength > 1800 ? 10.0 : 0.0) + sample[i];
                reinitialize |= trends[i]->addPoint(time, sample[i]);
                detrended[0][i] = trends[i]->detrend(time, sample[i], 0.0);
                mean[i] = trends[i]->mean(time);
                hasTrend |= true;
                amplitude += 4.0;
            }

            TTimeDouble2VecSizeTrVec sample_{
                core::make_triple(time, TDouble2Vec(sample), TAG)};
            maths::CModelAddSamplesParams params_;
            params_.integer(false)
                .propagationInterval(1.0)
                .trendWeights(weights)
                .priorWeights(weights);
            model.addSamples(params_, sample_);

            if (reinitialize) {
                reinitializePrior(learnRate, model, trends, prior, &controllers);
            }
            prior.addSamples(detrended,
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
            prior.propagateForwardsByTime(1.0);

            if (hasTrend) {
                double multiplier{controllers[0].multiplier(
                    mean, {detrended[0]}, bucketLength, model.params().learnRate(), DECAY_RATE)};
                for (const auto& trend : trends) {
                    trend->decayRate(multiplier * trend->decayRate());
                }
            }
            if (prior.numberSamples() > 20.0) {
                TDouble1Vec prediction(prior.marginalLikelihoodMean());
                TDouble1Vec predictionError(3);
                for (std::size_t d = 0u; d < 3; ++d) {
                    predictionError[d] = detrended[0][d] - prediction[d];
                }
                double multiplier{controllers[1].multiplier(
                    prediction, {predictionError}, bucketLength,
                    model.params().learnRate(), DECAY_RATE)};
                prior.decayRate(multiplier * prior.decayRate());
            }

            for (std::size_t i = 0u; i < trends.size(); ++i) {
                uint64_t checksum1{trends[i]->checksum()};
                uint64_t checksum2{model.trend()[i]->checksum()};
                CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);
            }
            uint64_t checksum1{prior.checksum()};
            uint64_t checksum2{model.prior().checksum()};
            CPPUNIT_ASSERT_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }
}

void CTimeSeriesModelTest::testPredict() {
    // Test prediction with a trend and with multimodal data.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate seasonal");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend,
                                                prior, &controllers};

        TDoubleVec samples;
        rng.generateNormalSamples(0.0, 4.0, 1008, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            sample += 10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(time) / 86400.0);

            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});

            if (trend.addPoint(time, sample)) {
                prior.setToNonInformative(0.0, DECAY_RATE);
                for (const auto& value : model.slidingWindow()) {
                    prior.addSamples({trend.detrend(value.first, value.second, 0.0)},
                                     maths_t::CUnitWeights::SINGLE_UNIT);
                }
            }
            prior.addSamples({trend.detrend(time, sample, 0.0)},
                             maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);

            time += bucketLength;
        }

        TMeanAccumulator meanError;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            double trend_{10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(time_) / 86400.0)};
            double expected{maths::CBasicStatistics::mean(trend.baseline(time_)) +
                            maths::CBasicStatistics::mean(
                                prior.marginalLikelihoodConfidenceInterval(0.0))};
            double predicted{model.predict(time_)[0]};
            LOG_DEBUG(<< "expected = " << expected << " predicted = " << predicted
                      << " (trend = " << trend_ << ")");
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, predicted, 1e-3 * expected);
            CPPUNIT_ASSERT(std::fabs(trend_ - predicted) / trend_ < 0.3);
            meanError.add(std::fabs(trend_ - predicted) / trend_);
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.06);
    }

    LOG_DEBUG(<< "Univariate nearest mode");
    {
        maths::CTimeSeriesDecompositionStub trend;
        maths::CMultimodalPrior prior{univariateMultimodal()};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

        TMeanAccumulator modes[2];
        TDoubleVec samples, samples_;
        rng.generateNormalSamples(0.0, 4.0, 500, samples);
        rng.generateNormalSamples(10.0, 4.0, 500, samples_);
        modes[0].add(samples);
        modes[1].add(samples_);
        samples.insert(samples.end(), samples_.begin(), samples_.end());
        rng.random_shuffle(samples.begin(), samples.end());

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        maths::CModel::TSizeDoublePr1Vec empty;
        double predicted[]{model.predict(time, empty, {-2.0})[0],
                           model.predict(time, empty, {12.0})[0]};

        LOG_DEBUG(<< "expected(0) = " << maths::CBasicStatistics::mean(modes[0])
                  << " actual(0) = " << predicted[0]);
        LOG_DEBUG(<< "expected(1) = " << maths::CBasicStatistics::mean(modes[1])
                  << " actual(1) = " << predicted[1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(modes[0]),
                                     predicted[0],
                                     0.1 * maths::CBasicStatistics::mean(modes[0]));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(modes[1]),
                                     predicted[1],
                                     0.01 * maths::CBasicStatistics::mean(modes[1]));
    }

    LOG_DEBUG(<< "Multivariate Seasonal");
    {
        double learnRate{params(bucketLength).learnRate()};
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{maths::CMultivariateTimeSeriesModel{
            params(bucketLength), *trends[0], prior}};

        TDoubleVecVec samples;
        TDoubleVec mean{0.0, 2.0, 1.0};
        {
            TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
            rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            for (auto& coordinate : sample) {
                coordinate += 10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                    static_cast<double>(time) / 86400.0);
            }
            bool reinitialize{false};
            TDouble10Vec detrended;
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                reinitialize |= trends[i]->addPoint(time, sample[i]);
                detrended.push_back(trends[i]->detrend(time, sample[i], 0.0));
            }
            if (reinitialize) {
                reinitializePrior(learnRate, model, trends, prior);
            }
            prior.addSamples({detrended},
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
            prior.propagateForwardsByTime(1.0);

            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});

            time += bucketLength;
        }

        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            maths::CMultivariatePrior::TSize10Vec marginalize{1, 2};
            maths::CMultivariatePrior::TSizeDoublePr10Vec condition;
            for (std::size_t i = 0u; i < mean.size(); ++i) {
                double trend_{mean[i] + 10.0 +
                              5.0 * std::sin(boost::math::double_constants::two_pi *
                                             static_cast<double>(time_) / 86400.0)};
                maths::CMultivariatePrior::TUnivariatePriorPtr margin{
                    prior.univariate(marginalize, condition).first};
                double expected{maths::CBasicStatistics::mean(trends[i]->baseline(time_)) +
                                maths::CBasicStatistics::mean(
                                    margin->marginalLikelihoodConfidenceInterval(0.0))};
                double predicted{model.predict(time_)[i]};
                --marginalize[std::min(i, marginalize.size() - 1)];
                LOG_DEBUG(<< "expected = " << expected << " predicted = " << predicted
                          << " (trend = " << trend_ << ")");
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, predicted, 1e-3 * expected);
                CPPUNIT_ASSERT(std::fabs(trend_ - predicted) / trend_ < 0.3);
            }
        }
    }

    LOG_DEBUG(<< "Multivariate nearest mode");
    {
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}}};
        maths::CMultivariateMultimodalPrior<3> prior{multivariateMultimodal()};
        maths::CMultivariateTimeSeriesModel model{maths::CMultivariateTimeSeriesModel{
            params(bucketLength), *trends[0], prior}};

        TMeanAccumulator2Vec modes[2]{TMeanAccumulator2Vec(3), TMeanAccumulator2Vec(3)};
        TDoubleVecVec samples;
        {
            TDoubleVec means[]{{0.0, 2.0, 1.0}, {10.0, 15.0, 12.0}};
            TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
            rng.generateMultivariateNormalSamples(means[0], covariance, 500, samples);
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(means[1], covariance, 500, samples_);
            for (const auto& sample : samples) {
                for (std::size_t i = 0u; i < 3; ++i) {
                    modes[0][i].add(sample[i]);
                }
            }
            for (const auto& sample : samples_) {
                for (std::size_t i = 0u; i < 3; ++i) {
                    modes[1][i].add(sample[i]);
                }
            }
            samples.insert(samples.end(), samples_.begin(), samples_.end());
            rng.random_shuffle(samples.begin(), samples.end());
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (const auto& sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        maths::CModel::TSizeDoublePr1Vec empty;
        TDouble2Vec expected[]{maths::CBasicStatistics::mean(modes[0]),
                               maths::CBasicStatistics::mean(modes[1])};
        TDouble2Vec predicted[]{model.predict(time, empty, {0.0, 0.0, 0.0}),
                                model.predict(time, empty, {10.0, 10.0, 10.0})};
        for (std::size_t i = 0u; i < 3; ++i) {
            LOG_DEBUG(<< "expected(0) = " << expected[0][i]
                      << " actual(0) = " << predicted[0][i]);
            LOG_DEBUG(<< "expected(1) = " << expected[1][i]
                      << " actual(1) = " << predicted[1][i]);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[0][i], predicted[0][i],
                                         std::fabs(0.2 * expected[0][i]));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[1][i], predicted[1][i],
                                         std::fabs(0.01 * expected[1][i]));
        }
    }
}

void CTimeSeriesModelTest::testProbability() {
    // Test: 1) Calculation, seasonal confidence interval, weights, etc.
    //       2) Test with and without trend.
    //       3) Test with anomalies.

    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{600};

    LOG_DEBUG(<< "Univariate");
    {
        maths::CUnivariateTimeSeriesModel models[]{
            maths::CUnivariateTimeSeriesModel{params(bucketLength), 1,
                                              maths::CTimeSeriesDecompositionStub{},
                                              univariateNormal(), nullptr, false},
            maths::CUnivariateTimeSeriesModel{
                params(bucketLength), 1,
                maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength},
                univariateNormal(), nullptr, false}};

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 4.0, 1000, samples);

        core_t::TTime time{0};
        {
            const TDouble2VecWeightsAryVec weight{
                maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
            for (auto sample : samples) {
                maths::CModelAddSamplesParams params;
                params.integer(false)
                    .propagationInterval(1.0)
                    .trendWeights(weight)
                    .priorWeights(weight);

                double trend{5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                  static_cast<double>(time) / 86400.0)};

                models[0].addSamples(
                    params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
                models[1].addSamples(
                    params, {core::make_triple(time, TDouble2Vec{trend + sample}, TAG)});

                time += bucketLength;
            }
        }

        TTime2Vec1Vec time_{{time}};
        TDouble2Vec sample{15.0};

        maths_t::EProbabilityCalculation calculations[]{maths_t::E_TwoSided,
                                                        maths_t::E_OneSidedAbove};
        double confidences[]{0.0, 20.0, 50.0};
        bool empties[]{true, false};
        TDouble2VecWeightsAryVec weights(2, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        maths_t::setCountVarianceScale(TDouble2Vec{0.9}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{1.1}, weights[1]);
        maths_t::setSeasonalVarianceScale(TDouble2Vec{1.8}, weights[1]);

        for (auto calculation : calculations) {
            LOG_DEBUG(<< "calculation = " << calculation);
            for (auto confidence : confidences) {
                LOG_DEBUG(<< " confidence = " << confidence);
                for (auto empty : empties) {
                    LOG_DEBUG(<< "  empty = " << empty);
                    for (const auto& weight : weights) {
                        LOG_DEBUG(<< "   weights = "
                                  << core::CContainerPrinter::print(weight));
                        double expectedProbability[2];
                        maths_t::ETail expectedTail[2];
                        {
                            maths_t::TDoubleWeightsAry weight_(maths_t::CUnitWeights::UNIT);
                            for (std::size_t i = 0u; i < weight.size(); ++i) {
                                weight_[i] = weight[i][0];
                            }
                            double lb[2], ub[2];
                            models[0].prior().probabilityOfLessLikelySamples(
                                calculation, sample, {weight_}, lb[0], ub[0],
                                expectedTail[0]);
                            models[1].prior().probabilityOfLessLikelySamples(
                                calculation,
                                {models[1].trend().detrend(time, sample[0], confidence)},
                                {weight_}, lb[1], ub[1], expectedTail[1]);
                            expectedProbability[0] = (lb[0] + ub[0]) / 2.0;
                            expectedProbability[1] = (lb[1] + ub[1]) / 2.0;
                        }

                        double probability[2];
                        TTail2Vec tail[2];
                        {
                            maths::CModelProbabilityParams params;
                            params.addCalculation(calculation)
                                .seasonalConfidenceInterval(confidence)
                                .addBucketEmpty({empty})
                                .addWeights(weight);
                            bool conditional;
                            TSize1Vec mostAnomalousCorrelate;
                            models[0].probability(params, time_, {sample},
                                                  probability[0], tail[0], conditional,
                                                  mostAnomalousCorrelate);
                            models[1].probability(params, time_, {sample},
                                                  probability[1], tail[1], conditional,
                                                  mostAnomalousCorrelate);
                        }

                        CPPUNIT_ASSERT_EQUAL(expectedProbability[0], probability[0]);
                        CPPUNIT_ASSERT_EQUAL(expectedTail[0], tail[0][0]);
                        CPPUNIT_ASSERT_EQUAL(expectedProbability[1], probability[1]);
                        CPPUNIT_ASSERT_EQUAL(expectedTail[1], tail[1][0]);
                    }
                }
            }
        }
    }

    LOG_DEBUG(<< "Multivariate");
    {
        maths::CMultivariateTimeSeriesModel models[]{
            maths::CMultivariateTimeSeriesModel{
                params(bucketLength), maths::CTimeSeriesDecompositionStub{},
                multivariateNormal(), nullptr, false},
            maths::CMultivariateTimeSeriesModel{
                params(bucketLength),
                maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength},
                multivariateNormal(), nullptr, false}};

        TDoubleVecVec samples;
        {
            TDoubleVec mean{10.0, 15.0, 11.0};
            TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
            rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);
        }

        core_t::TTime time{0};
        {
            TDouble2VecWeightsAryVec weight{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
            for (auto& sample : samples) {
                maths::CModelAddSamplesParams params;
                params.integer(false)
                    .propagationInterval(1.0)
                    .trendWeights(weight)
                    .priorWeights(weight);

                TDouble2Vec sample_(sample);
                models[0].addSamples(params, {core::make_triple(time, sample_, TAG)});

                double trend{5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                  static_cast<double>(time) / 86400.0)};
                for (auto& component : sample_) {
                    component += trend;
                }

                models[1].addSamples(params, {core::make_triple(time, sample_, TAG)});

                time += bucketLength;
            }
        }

        TTime2Vec1Vec time_{{time}};
        TDouble2Vec sample{15.0, 14.0, 16.0};

        maths_t::EProbabilityCalculation calculations[]{maths_t::E_TwoSided,
                                                        maths_t::E_OneSidedAbove};
        double confidences[]{0.0, 20.0, 50.0};
        bool empties[]{true, false};
        TDouble2VecWeightsAryVec weights(2, maths_t::CUnitWeights::unit<TDouble2Vec>(3));
        maths_t::setCountVarianceScale(TDouble2Vec{0.9, 0.9, 0.8}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{1.1, 1.0, 1.2}, weights[1]);
        maths_t::setSeasonalVarianceScale(TDouble2Vec{1.8, 1.7, 1.6}, weights[1]);

        for (auto calculation : calculations) {
            LOG_DEBUG(<< "calculation = " << calculation);
            for (auto confidence : confidences) {
                LOG_DEBUG(<< " confidence = " << confidence);
                for (auto empty : empties) {
                    LOG_DEBUG(<< "  empty = " << empty);
                    for (const auto& weight : weights) {
                        LOG_DEBUG(<< "   weights = "
                                  << core::CContainerPrinter::print(weight));
                        double expectedProbability[2];
                        TTail10Vec expectedTail[2];
                        {
                            maths_t::TDouble10VecWeightsAry weight_(
                                maths_t::CUnitWeights::unit<TDouble10Vec>(3));
                            for (std::size_t i = 0u; i < weight.size(); ++i) {
                                weight_[i] = weight[i];
                            }
                            double lb[2], ub[2];
                            models[0].prior().probabilityOfLessLikelySamples(
                                calculation, {TDouble10Vec(sample)}, {weight_},
                                lb[0], ub[0], expectedTail[0]);
                            TDouble10Vec detrended;
                            for (std::size_t j = 0u; j < sample.size(); ++j) {
                                detrended.push_back(models[1].trend()[j]->detrend(
                                    time, sample[j], confidence));
                            }
                            models[1].prior().probabilityOfLessLikelySamples(
                                calculation, {detrended}, {weight_}, lb[1],
                                ub[1], expectedTail[1]);
                            expectedProbability[0] = (lb[0] + ub[0]) / 2.0;
                            expectedProbability[1] = (lb[1] + ub[1]) / 2.0;
                        }

                        double probability[2];
                        TTail2Vec tail[2];
                        {
                            maths::CModelProbabilityParams params;
                            params.addCalculation(calculation)
                                .seasonalConfidenceInterval(confidence)
                                .addBucketEmpty({empty})
                                .addWeights(weight);
                            bool conditional;
                            TSize1Vec mostAnomalousCorrelate;
                            models[0].probability(params, time_, {sample},
                                                  probability[0], tail[0], conditional,
                                                  mostAnomalousCorrelate);
                            models[1].probability(params, time_, {sample},
                                                  probability[1], tail[1], conditional,
                                                  mostAnomalousCorrelate);
                        }

                        CPPUNIT_ASSERT_EQUAL(expectedProbability[0], probability[0]);
                        CPPUNIT_ASSERT_EQUAL(expectedProbability[1], probability[1]);
                        for (std::size_t j = 0u; j < 3; ++j) {
                            CPPUNIT_ASSERT_EQUAL(expectedTail[0][j], tail[0][j]);
                            CPPUNIT_ASSERT_EQUAL(expectedTail[1][j], tail[1][j]);
                        }
                    }
                }
            }
        }
    }

    LOG_DEBUG(<< "Anomalies");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 1, trend,
                                                univariateNormal()};

        TSizeVec anomalies;
        rng.generateUniformSamples(100, 1000, 10, anomalies);
        std::sort(anomalies.begin(), anomalies.end());
        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 4.0, 1000, samples);

        maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> smallest(10);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        std::size_t bucket{0};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            if (std::binary_search(anomalies.begin(), anomalies.end(), bucket++)) {
                sample += 10.0;
            }
            {
                maths::CModelAddSamplesParams params;
                params.integer(false)
                    .propagationInterval(1.0)
                    .trendWeights(weights)
                    .priorWeights(weights);
                model.addSamples(
                    params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            }
            {
                maths::CModelProbabilityParams params;
                params.addCalculation(maths_t::E_TwoSided)
                    .seasonalConfidenceInterval(50.0)
                    .addBucketEmpty({false})
                    .addWeights(weights[0]);
                TTail2Vec tail;
                double probability;
                bool conditional;
                TSize1Vec mostAnomalousCorrelate;
                model.probability(params, {{time}}, {{sample}}, probability,
                                  tail, conditional, mostAnomalousCorrelate);
                smallest.add({probability, bucket - 1});
            }
            time += bucketLength;
        }

        TSizeVec anomalies_;
        std::transform(smallest.begin(), smallest.end(), std::back_inserter(anomalies_),
                       [](const TDoubleSizePr& value) { return value.second; });
        std::sort(anomalies_.begin(), anomalies_.end());

        LOG_DEBUG(<< "expected anomalies = " << core::CContainerPrinter::print(anomalies));
        LOG_DEBUG(<< "actual anomalies   = " << core::CContainerPrinter::print(anomalies_));
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(anomalies),
                             core::CContainerPrinter::print(anomalies_));
    }
}

void CTimeSeriesModelTest::testWeights() {
    core_t::TTime bucketLength{1800};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 0, trend, prior};

        TDoubleVec samples;
        rng.generateNormalSamples(0.0, 4.0, 1008, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            double scale{10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                               static_cast<double>(time) / 86400.0)};
            sample = scale * (1.0 + 0.1 * sample);

            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});

            if (trend.addPoint(time, sample)) {
                prior.setToNonInformative(0.0, DECAY_RATE);
                for (const auto& value : model.slidingWindow()) {
                    prior.addSamples({trend.detrend(value.first, value.second, 0.0)},
                                     maths_t::CUnitWeights::SINGLE_UNIT);
                }
            }
            prior.addSamples({trend.detrend(time, sample, 0.0)},
                             maths_t::CUnitWeights::SINGLE_UNIT);

            time += bucketLength;
        }

        LOG_DEBUG(<< "Seasonal");
        TMeanAccumulator error;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            double dataScale{std::pow(
                1.0 + 0.5 * std::sin(boost::math::double_constants::two_pi *
                                     static_cast<double>(time_) / 86400.0),
                2.0)};

            double expectedScale{
                trend.scale(time_, prior.marginalLikelihoodVariance(), 0.0).second};
            double scale{model.seasonalWeight(0.0, time_)[0]};

            LOG_DEBUG(<< "expected weight = " << expectedScale << ", weight = " << scale
                      << " (data weight = " << dataScale << ")");
            CPPUNIT_ASSERT_EQUAL(std::max(expectedScale, MINIMUM_SEASONAL_SCALE), scale);

            error.add(std::fabs(scale - dataScale) / dataScale);
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.2);

        LOG_DEBUG(<< "Winsorisation");
        TDouble2Vec prediction(model.predict(time));
        double lastWeight = 1.0;
        for (std::size_t i = 0u; i < 10; ++i) {
            double weight_{model.winsorisationWeight(1.0, time, prediction)[0]};
            LOG_DEBUG(<< "weight = " << weight_);
            CPPUNIT_ASSERT(weight_ <= lastWeight);
            lastWeight = weight_;
            prediction[0] *= 1.1;
        }
    }

    LOG_DEBUG(<< "Multivariate");
    {
        double learnRate{params(bucketLength).learnRate()};
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params(bucketLength), *trends[0], prior};

        TDoubleVecVec samples;
        {
            TDoubleVec mean{10.0, 15.0, 11.0};
            TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
            rng.generateMultivariateNormalSamples(mean, covariance, 1008, samples);
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            double scale{10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                               static_cast<double>(time) / 86400.0)};

            bool reinitialize{false};
            TDouble10Vec1Vec detrended{TDouble10Vec(3)};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                sample[i] = scale * (1.0 + 0.1 * sample[i]);
                reinitialize |= trends[i]->addPoint(time, sample[i]);
                detrended[0][i] = trends[i]->detrend(time, sample[i], 0.0);
            }
            if (reinitialize) {
                reinitializePrior(learnRate, model, trends, prior);
            }
            prior.addSamples(detrended,
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));

            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            model.addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});

            time += bucketLength;
        }

        LOG_DEBUG(<< "Seasonal");
        TMeanAccumulator error;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            double dataScale{std::pow(
                1.0 + 0.5 * std::sin(boost::math::double_constants::two_pi *
                                     static_cast<double>(time_) / 86400.0),
                2.0)};

            for (std::size_t i = 0u; i < 3; ++i) {
                double expectedScale{
                    trends[i]
                        ->scale(time_, prior.marginalLikelihoodVariances()[i], 0.0)
                        .second};
                double scale{model.seasonalWeight(0.0, time_)[i]};
                LOG_DEBUG(<< "expected weight = " << expectedScale << ", weight = " << scale
                          << " (data weight = " << dataScale << ")");
                CPPUNIT_ASSERT_EQUAL(std::max(expectedScale, MINIMUM_SEASONAL_SCALE), scale);
                error.add(std::fabs(scale - dataScale) / dataScale);
            }
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.21);

        LOG_DEBUG(<< "Winsorisation");
        TDouble2Vec prediction(model.predict(time));
        double lastWeight = 1.0;
        for (std::size_t i = 0u; i < 10; ++i) {
            double weight_{model.winsorisationWeight(1.0, time, prediction)[0]};
            LOG_DEBUG(<< "weight = " << weight_);
            CPPUNIT_ASSERT(weight_ <= lastWeight);
            lastWeight = weight_;
            prediction[0] *= 1.1;
        }
    }
}

void CTimeSeriesModelTest::testMemoryUsage() {
    // Test we account for the appropriate memory.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        std::unique_ptr<maths::CModel> model{new maths::CUnivariateTimeSeriesModel{
            params(bucketLength), 0, trend, univariateNormal(), &controllers}};

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            sample += 10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(time) / 86400.0);
            trend.addPoint(time, sample);
            model->addSamples(params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        std::size_t expectedSize{
            sizeof(maths::CTimeSeriesDecomposition) + trend.memoryUsage() +
            sizeof(maths::CNormalMeanPrecConjugate) +
            sizeof(maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary) +
            2 * controllers[0].memoryUsage()};
        std::size_t size = model->memoryUsage();
        LOG_DEBUG(<< "size " << size << " expected " << expectedSize);
        CPPUNIT_ASSERT(size < 1.1 * expectedSize);
    }

    LOG_DEBUG(<< "Multivariate");
    {
        TDoubleVec mean{11.0, 10.0, 12.0};
        TDoubleVecVec covariance{{4.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        auto controllers = decayRateControllers(3);
        std::unique_ptr<maths::CModel> model{new maths::CMultivariateTimeSeriesModel{
            params(bucketLength), trend, prior, &controllers}};

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            for (auto& coordinate : sample) {
                coordinate += 10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                    static_cast<double>(time) / 86400.0);
            }
            trend.addPoint(time, sample[0]);
            model->addSamples(params, {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        std::size_t expectedSize{
            3 * sizeof(maths::CTimeSeriesDecomposition) + 3 * trend.memoryUsage() +
            sizeof(maths::CMultivariateNormalConjugate<3>) +
            sizeof(maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary) +
            2 * controllers[0].memoryUsage()};
        std::size_t size = model->memoryUsage();
        LOG_DEBUG(<< "size " << size << " expected " << expectedSize);
        CPPUNIT_ASSERT(size < 1.1 * expectedSize);
    }

    // TODO LOG_DEBUG(<< "Correlates");
}

void CTimeSeriesModelTest::testPersist() {
    // Test persist then restore is idempotent.

    core_t::TTime bucketLength{600};
    maths::CModelParams params_{params(bucketLength)};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel origModel{
            params_, 1, trend, univariateNormal(), &controllers};

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            origModel.addSamples(
                params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter{"root"};
            origModel.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        //LOG_DEBUG(<< "model XML representation:\n" << origXml);

        // Restore the XML into a new filter
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength,
            maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE};
        maths::SDistributionRestoreParams distributionParams{
            maths_t::E_ContinuousData, DECAY_RATE, 0.5, 24.0, 12};
        maths::SModelRestoreParams restoreParams{params_, decompositionParams, distributionParams};
        maths::CUnivariateTimeSeriesModel restoredModel{restoreParams, traverser};

        CPPUNIT_ASSERT_EQUAL(origModel.checksum(), restoredModel.checksum());
    }

    LOG_DEBUG(<< "Multivariate");
    {
        TDoubleVec mean{11.0, 10.0, 12.0};
        TDoubleVecVec covariance{{4.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        auto controllers = decayRateControllers(3);
        maths::CMultivariateTimeSeriesModel origModel{params(bucketLength),
                                                      trend, prior, &controllers};

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (const auto& sample : samples) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            origModel.addSamples(
                params, {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter{"root"};
            origModel.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        //LOG_DEBUG(<< "model XML representation:\n" << origXml);

        // Restore the XML into a new filter
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength,
            maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE};
        maths::SDistributionRestoreParams distributionParams{
            maths_t::E_ContinuousData, DECAY_RATE, 0.5, 24.0, 12};
        maths::SModelRestoreParams restoreParams{params_, decompositionParams, distributionParams};
        maths::CMultivariateTimeSeriesModel restoredModel{restoreParams, traverser};

        CPPUNIT_ASSERT_EQUAL(origModel.checksum(), restoredModel.checksum());
    }

    // TODO LOG_DEBUG(<< "Correlates");
}

void CTimeSeriesModelTest::testUpgrade() {
    using TStrVec = std::vector<std::string>;
    auto load = [](const std::string& name, std::string& result) {
        std::ifstream file;
        file.open(name);
        std::stringbuf buf;
        file >> &buf;
        result = buf.str();
    };

    core_t::TTime bucketLength{600};
    core_t::TTime halfHour{1800};
    maths::CModelParams params_{params(bucketLength)};
    std::string empty;

    LOG_DEBUG(<< "Univariate");
    {
        std::string xml;
        load("testfiles/CUnivariateTimeSeriesModel.6.2.state.xml", xml);
        LOG_DEBUG(<< "Saved state size = " << xml.size());

        std::string intervals;
        load("testfiles/CUnivariateTimeSeriesModel.6.2.expected_intervals.txt", intervals);
        LOG_DEBUG(<< "Expected intervals size = " << intervals.size());
        TStrVec expectedIntervals;
        core::CStringUtils::tokenise(";", intervals, expectedIntervals, empty);

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength,
            maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE};
        maths::SDistributionRestoreParams distributionParams{
            maths_t::E_ContinuousData, DECAY_RATE, 0.5, 24.0, 12};
        maths::SModelRestoreParams restoreParams{params_, decompositionParams, distributionParams};
        maths::CUnivariateTimeSeriesModel restoredModel{restoreParams, traverser};

        TStrVec expectedInterval;
        TStrVec interval;
        for (core_t::TTime time = 600000, i = 0;
             i < static_cast<core_t::TTime>(expectedIntervals.size());
             time += halfHour, ++i) {
            expectedInterval.clear();
            interval.clear();

            core::CStringUtils::tokenise(",", expectedIntervals[i], expectedInterval, empty);
            std::string interval_{core::CContainerPrinter::print(restoredModel.confidenceInterval(
                time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1)))};
            core::CStringUtils::replace("[", "", interval_);
            core::CStringUtils::replace("]", "", interval_);
            core::CStringUtils::replace(" ", "", interval_);
            interval_ += ",";
            core::CStringUtils::tokenise(",", interval_, interval, empty);

            CPPUNIT_ASSERT_EQUAL(expectedInterval.size(), interval.size());
            for (std::size_t j = 0u; j < expectedInterval.size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                    boost::lexical_cast<double>(expectedInterval[j]),
                    boost::lexical_cast<double>(interval[j]), 0.0001);
            }
        }
    }

    LOG_DEBUG(<< "Multivariate");
    {
        std::string xml;
        load("testfiles/CMultivariateTimeSeriesModel.6.2.state.xml", xml);
        LOG_DEBUG(<< "Saved state size = " << xml.size());

        std::string intervals;
        load("testfiles/CMultivariateTimeSeriesModel.6.2.expected_intervals.txt", intervals);
        LOG_DEBUG(<< "Expected intervals size = " << intervals.size());
        TStrVec expectedIntervals;
        core::CStringUtils::tokenise(";", intervals, expectedIntervals, empty);

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength,
            maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE};
        maths::SDistributionRestoreParams distributionParams{
            maths_t::E_ContinuousData, DECAY_RATE, 0.5, 24.0, 12};
        maths::SModelRestoreParams restoreParams{params_, decompositionParams, distributionParams};
        maths::CMultivariateTimeSeriesModel restoredModel{restoreParams, traverser};

        TStrVec expectedInterval;
        TStrVec interval;
        for (core_t::TTime time = 600000, i = 0;
             i < static_cast<core_t::TTime>(expectedIntervals.size());
             time += halfHour, ++i) {
            expectedInterval.clear();
            interval.clear();

            core::CStringUtils::tokenise(",", expectedIntervals[i], expectedInterval, empty);
            std::string interval_{core::CContainerPrinter::print(restoredModel.confidenceInterval(
                time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(3)))};
            core::CStringUtils::replace("[", "", interval_);
            core::CStringUtils::replace("]", "", interval_);
            core::CStringUtils::replace(" ", "", interval_);
            interval_ += ",";
            core::CStringUtils::tokenise(",", interval_, interval, empty);

            CPPUNIT_ASSERT_EQUAL(expectedInterval.size(), interval.size());
            for (std::size_t j = 0u; j < expectedInterval.size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                    boost::lexical_cast<double>(expectedInterval[j]),
                    boost::lexical_cast<double>(interval[j]), 0.0001);
            }
        }
    }
}

void CTimeSeriesModelTest::testAddSamplesWithCorrelations() {
    LOG_DEBUG(<< "Correlations no trend");

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    {
        TDoubleVec mean{10.0, 15.0};
        TDoubleVecVec covariance{{3.0, 2.9}, {2.9, 2.6}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        maths::CTimeSeriesCorrelations correlations{MINIMUM_SIGNIFICANT_CORRELATION, DECAY_RATE};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel models[]{
            {params(bucketLength), 0, trend, prior, nullptr},
            {params(bucketLength), 1, trend, prior, nullptr}};
        models[0].modelCorrelations(correlations);
        models[1].modelCorrelations(correlations);
        CTimeSeriesCorrelateModelAllocator allocator;

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            correlations.refresh(allocator);
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            models[0].addSamples(
                params, {core::make_triple(time, TDouble2Vec{sample[0]}, TAG)});
            models[1].addSamples(
                params, {core::make_triple(time, TDouble2Vec{sample[1]}, TAG)});
            correlations.processSamples();
            time += bucketLength;
        }
    }

    // TODO LOG_DEBUG(<< "Correlations trend");
    // TODO LOG_DEBUG(<< "Correlations with tags (for population)");
}

void CTimeSeriesModelTest::testProbabilityWithCorrelations() {
}

void CTimeSeriesModelTest::testAnomalyModel() {
    using TSizeVec = std::vector<std::size_t>;
    using TDoubleSizePr = std::pair<double, std::size_t>;

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate") {
        std::size_t length = 2000;

        TSizeVec anomalies;
        rng.generateUniformSamples(0, length, 30, anomalies);
        std::sort(anomalies.begin(), anomalies.end());

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 2.0, length, samples);

        core_t::TTime bucketLength{600};
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CUnivariateTimeSeriesModel model{params(bucketLength), 1, trend,
                                                univariateNormal()};

        //std::ofstream file;
        //file.open("results.m");
        //TDoubleVec scores;

        maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> mostAnomalous(10);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        std::size_t bucket{0};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            if (std::binary_search(anomalies.begin(), anomalies.end(), bucket++)) {
                sample += 12.0;
            }
            if (bucket >= length - 100 && bucket < length - 92) {
                sample += 8.0;
            }
            {
                maths::CModelAddSamplesParams params;
                params.integer(false)
                    .propagationInterval(1.0)
                    .trendWeights(weights)
                    .priorWeights(weights);
                model.addSamples(
                    params, {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            }
            {
                maths::CModelProbabilityParams params;
                params.addCalculation(maths_t::E_TwoSided)
                    .seasonalConfidenceInterval(50.0)
                    .addBucketEmpty({false})
                    .addWeights(weights[0]);
                TTail2Vec tail;
                double probability;
                bool conditional;
                TSize1Vec mostAnomalousCorrelate;
                model.probability(params, {{time}}, {{sample}}, probability,
                                  tail, conditional, mostAnomalousCorrelate);
                mostAnomalous.add({::log(probability), bucket});
                //scores.push_back(maths::CTools::deviation(probability));
            }
            time += bucketLength;
        }

        mostAnomalous.sort();

        TSizeVec anomalyBuckets;
        TDoubleVec anomalyProbabilities;
        for (const auto& anomaly : mostAnomalous) {
            anomalyBuckets.push_back(anomaly.second);
            anomalyProbabilities.push_back(std::exp(anomaly.first));
        }
        LOG_DEBUG(<< "anomalies = " << core::CContainerPrinter::print(anomalyBuckets));
        LOG_DEBUG(<< "probabilities = "
                  << core::CContainerPrinter::print(anomalyProbabilities));
        CPPUNIT_ASSERT(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                 1905) != anomalyBuckets.end());
        CPPUNIT_ASSERT(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                 1906) != anomalyBuckets.end());
        CPPUNIT_ASSERT(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                 1907) != anomalyBuckets.end());

        //file << "v = " << core::CContainerPrinter::print(samples) << ";\n";
        //file << "s = " << core::CContainerPrinter::print(scores) << ";\n";
        //file << "hold on;";
        //file << "subplot(2,1,1);\n";
        //file << "plot([1:length(v)], v);\n";
        //file << "subplot(2,1,2);\n";
        //file << "plot([1:length(s)], s, 'r');\n";
    }

    LOG_DEBUG(<< "Multivariate") {
        std::size_t length = 2000;

        TSizeVec anomalies;
        rng.generateUniformSamples(0, length, 30, anomalies);
        std::sort(anomalies.begin(), anomalies.end());
        core_t::TTime bucketLength{600};
        TDoubleVec mean{10.0, 10.0, 10.0};
        TDoubleVecVec covariance{{4.0, 0.9, 0.5}, {0.9, 2.6, 0.1}, {0.5, 0.1, 3.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, length, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params(bucketLength), trend, prior};

        //std::ofstream file;
        //file.open("results.m");
        //TDoubleVec scores;

        maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> mostAnomalous(10);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        std::size_t bucket{0};
        for (auto& sample : samples) {
            for (auto& coordinate : sample) {
                if (std::binary_search(anomalies.begin(), anomalies.end(), bucket)) {
                    coordinate += 12.0;
                }
                if (bucket >= length - 100 && bucket < length - 92) {
                    coordinate += 8.0;
                }
            }
            ++bucket;
            {
                maths::CModelAddSamplesParams params;
                params.integer(false)
                    .propagationInterval(1.0)
                    .trendWeights(weights)
                    .priorWeights(weights);
                model.addSamples(
                    params, {core::make_triple(time, TDouble2Vec(sample), TAG)});
            }
            {
                maths::CModelProbabilityParams params;
                params.addCalculation(maths_t::E_TwoSided)
                    .seasonalConfidenceInterval(50.0)
                    .addBucketEmpty({false})
                    .addWeights(weights[0]);
                TTail2Vec tail;
                double probability;
                bool conditional;
                TSize1Vec mostAnomalousCorrelate;
                model.probability(params, {{time}}, {(sample)}, probability,
                                  tail, conditional, mostAnomalousCorrelate);
                mostAnomalous.add({::log(probability), bucket});
                //scores.push_back(maths::CTools::deviation(probability));
            }
            time += bucketLength;
        }

        mostAnomalous.sort();

        TSizeVec anomalyBuckets;
        TDoubleVec anomalyProbabilities;
        for (const auto& anomaly : mostAnomalous) {
            anomalyBuckets.push_back(anomaly.second);
            anomalyProbabilities.push_back(std::exp(anomaly.first));
        }
        LOG_DEBUG(<< "anomalies = " << core::CContainerPrinter::print(anomalyBuckets));
        LOG_DEBUG(<< "probabilities = "
                  << core::CContainerPrinter::print(anomalyProbabilities));
        CPPUNIT_ASSERT(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                 1908) != anomalyBuckets.end());

        //file << "v = [";
        //for (const auto &sample : samples)
        //{
        //    file << sample[0] << "," << sample[1] << "," << sample[2] << "\n";
        //}
        //file << "];\n";
        //file << "s = " << core::CContainerPrinter::print(scores) << ";\n";
        //file << "hold on;";
        //file << "subplot(4,1,1);\n";
        //file << "plot([1:rows(v)], v(:,1));\n";
        //file << "subplot(4,1,2);\n";
        //file << "plot([1:rows(v)], v(:,2));\n";
        //file << "subplot(4,1,3);\n";
        //file << "plot([1:rows(v)], v(:,3));\n";
        //file << "subplot(4,1,4);\n";
        //file << "plot([1:length(s)], s, 'r');\n";
    }
}

CppUnit::Test* CTimeSeriesModelTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CTimeSeriesModelTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testClone", &CTimeSeriesModelTest::testClone));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testMode", &CTimeSeriesModelTest::testMode));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testAddBucketValue", &CTimeSeriesModelTest::testAddBucketValue));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testAddSamples", &CTimeSeriesModelTest::testAddSamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testPredict", &CTimeSeriesModelTest::testPredict));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testProbability", &CTimeSeriesModelTest::testProbability));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testWeights", &CTimeSeriesModelTest::testWeights));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testMemoryUsage", &CTimeSeriesModelTest::testMemoryUsage));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testPersist", &CTimeSeriesModelTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testUpgrade", &CTimeSeriesModelTest::testUpgrade));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testAddSamplesWithCorrelations",
        &CTimeSeriesModelTest::testAddSamplesWithCorrelations));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testProbabilityWithCorrelations",
        &CTimeSeriesModelTest::testProbabilityWithCorrelations));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesModelTest>(
        "CTimeSeriesModelTest::testAnomalyModel", &CTimeSeriesModelTest::testAnomalyModel));

    return suiteOfTests;
}
