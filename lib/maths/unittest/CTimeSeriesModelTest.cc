/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
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
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CXMeansOnline.h>
#include <maths/CXMeansOnline1d.h>
#include <maths/Constants.h>
#include <maths/MathsTypes.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <fstream>
#include <memory>

using TSizeVec = std::vector<std::size_t>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TSizeVec::iterator)

BOOST_AUTO_TEST_SUITE(CTimeSeriesModelTest)

using namespace ml;

namespace {
using namespace handy_typedefs;
using TBool2Vec = core::CSmallVector<bool, 2>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TDouble2VecWeightsAry = maths_t::TDouble2VecWeightsAry;
using TDouble2VecWeightsAryVec = std::vector<TDouble2VecWeightsAry>;
using TDouble10VecWeightsAry1Vec = maths_t::TDouble10VecWeightsAry1Vec;
using TSize1Vec = core::CSmallVector<std::size_t, 1>;
using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;
using TTail10Vec = core::CSmallVector<maths_t::ETail, 10>;
using TDoubleWeightsAry1Vec = maths_t::TDoubleWeightsAry1Vec;
using TTimeDouble2VecSizeTrVec = maths::CModel::TTimeDouble2VecSizeTrVec;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulator2Vec = core::CSmallVector<TMeanAccumulator, 2>;
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
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

maths::CModelParams modelParams(core_t::TTime bucketLength) {
    using TTimeDoubleMap = std::map<core_t::TTime, double>;
    static TTimeDoubleMap learnRates;
    learnRates[bucketLength] = static_cast<double>(bucketLength) / 1800.0;
    double minimumSeasonalVarianceScale{MINIMUM_SEASONAL_SCALE};
    return maths::CModelParams{bucketLength,
                               learnRates[bucketLength],
                               DECAY_RATE,
                               minimumSeasonalVarianceScale,
                               12 * core::constants::HOUR,
                               core::constants::DAY};
}

maths::CModelAddSamplesParams addSampleParams(double interval,
                                              const TDouble2VecWeightsAryVec& trendWeights,
                                              const TDouble2VecWeightsAryVec& residualWeights) {
    maths::CModelAddSamplesParams params;
    params.integer(false)
        .propagationInterval(interval)
        .trendWeights(trendWeights)
        .priorWeights(residualWeights);
    return params;
}

maths::CModelAddSamplesParams
addSampleParams(double interval, const TDouble2VecWeightsAryVec& weights) {
    return addSampleParams(interval, weights, weights);
}

maths::CModelAddSamplesParams addSampleParams(const TDouble2VecWeightsAryVec& weights) {
    return addSampleParams(1.0, weights);
}

maths::CModelProbabilityParams computeProbabilityParams(const TDouble2VecWeightsAry& weight) {
    maths::CModelProbabilityParams params;
    params.addCalculation(maths_t::E_TwoSided).seasonalConfidenceInterval(50.0).addWeights(weight);
    return params;
}

maths::CNormalMeanPrecConjugate univariateNormal(double decayRate = DECAY_RATE) {
    return maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, decayRate);
}

maths::CLogNormalMeanPrecConjugate univariateLogNormal(double decayRate = DECAY_RATE) {
    return maths::CLogNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 0.0, decayRate);
}

maths::CMultimodalPrior univariateMultimodal(double decayRate = DECAY_RATE) {
    maths::CXMeansOnline1d clusterer{maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight, decayRate};
    return maths::CMultimodalPrior{maths_t::E_ContinuousData, clusterer,
                                   univariateNormal(), decayRate};
}

maths::CMultivariateNormalConjugate<3> multivariateNormal(double decayRate = DECAY_RATE) {
    return maths::CMultivariateNormalConjugate<3>::nonInformativePrior(
        maths_t::E_ContinuousData, decayRate);
}

maths::CMultivariateMultimodalPrior<3> multivariateMultimodal(double decayRate = DECAY_RATE) {
    maths::CXMeansOnline<maths::CFloatStorage, 3> clusterer(
        maths_t::E_ContinuousData, maths_t::E_ClustersFractionWeight, decayRate);
    return maths::CMultivariateMultimodalPrior<3>(
        maths_t::E_ContinuousData, clusterer,
        maths::CMultivariateNormalConjugate<3>::nonInformativePrior(
            maths_t::E_ContinuousData, decayRate),
        decayRate);
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

auto makeComponentDetectedCallback(double learnRate,
                                   maths::CPrior& residualModel,
                                   TDecayRateController2Ary* controllers = nullptr) {

    return [learnRate, &residualModel, controllers](TFloatMeanAccumulatorVec residuals) {

        if (controllers != nullptr) {
            residualModel.decayRate(residualModel.decayRate() /
                                    (*controllers)[1].multiplier());
            (*controllers)[0].reset();
            (*controllers)[1].reset();
        }

        residualModel.setToNonInformative(0.0, residualModel.decayRate());
        if (residuals.size() > 0) {
            maths_t::TDoubleWeightsAry1Vec weights(1);
            double buckets{std::accumulate(residuals.begin(), residuals.end(), 0.0,
                                           [](auto partialBuckets, const auto& residual) {
                                               return partialBuckets +
                                                      maths::CBasicStatistics::count(residual);
                                           }) /
                           learnRate};
            double time{buckets / static_cast<double>(residuals.size())};
            for (const auto& residual : residuals) {
                double weight(maths::CBasicStatistics::count(residual));
                if (weight > 0.0) {
                    weights[0] = maths_t::countWeight(weight);
                    residualModel.addSamples({maths::CBasicStatistics::mean(residual)}, weights);
                    residualModel.propagateForwardsByTime(time);
                }
            }
        }
    };
}

void reinitializeResidualModel(double learnRate,
                               TDecompositionPtr10Vec& trends,
                               maths::CMultivariatePrior& residualModel,
                               TDecayRateController2Ary* controllers = nullptr) {

    using TFloatMeanAccumulatorVecVec = std::vector<TFloatMeanAccumulatorVec>;

    if (controllers) {
        for (auto& trend : trends) {
            trend->decayRate(trend->decayRate() / (*controllers)[0].multiplier());
        }
        residualModel.decayRate(residualModel.decayRate() / (*controllers)[1].multiplier());
        (*controllers)[0].reset();
        (*controllers)[1].reset();
    }

    residualModel.setToNonInformative(0.0, residualModel.decayRate());

    std::size_t dimension{residualModel.dimension()};
    TFloatMeanAccumulatorVecVec residuals(dimension);
    for (std::size_t d = 0; d < dimension; ++d) {
        residuals[d] = trends[d]->residuals();
    }
    if (residuals.size() > 0) {
        TDouble10Vec1Vec samples;
        TDoubleVec weights;
        double time{0.0};
        for (std::size_t d = 0; d < dimension; ++d) {
            samples.resize(residuals[d].size(), TDouble10Vec(dimension));
            weights.resize(residuals[d].size(), std::numeric_limits<double>::max());
            double buckets{
                std::accumulate(residuals[d].begin(), residuals[d].end(), 0.0,
                                [](auto partialBuckets, const auto& residual) {
                                    return partialBuckets +
                                           maths::CBasicStatistics::count(residual);
                                }) /
                learnRate};
            time += buckets / static_cast<double>(residuals.size());
            for (std::size_t i = 0; i < residuals[d].size(); ++i) {
                samples[i][d] = maths::CBasicStatistics::mean(residuals[d][i]);
                weights[i] = std::min(
                    weights[i],
                    static_cast<double>(maths::CBasicStatistics::count(residuals[d][i])));
            }
        }
        time /= static_cast<double>(dimension);

        maths_t::TDouble10VecWeightsAry1Vec weight(1);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (weights[i] > 0.0) {
                weight[0] = maths_t::countWeight(weights[i], dimension);
                residualModel.addSamples({samples[i]}, weight);
                residualModel.propagateForwardsByTime(time);
            }
        }
    }
}

class CChangeDebug {
public:
    static const bool ENABLED{true};

public:
    CChangeDebug(std::string file = "results.py") : m_File(file) {
        if (ENABLED) {
            m_ModelBounds.resize(3);
            m_Forecast.resize(3);
        }
    }
    ~CChangeDebug() {
        if (ENABLED) {
            std::ofstream file;
            file.open(m_File);
            file << "import matplotlib.pyplot as plt;\n";
            file << "a = " << core::CContainerPrinter::print(m_Actual) << ";\n";
            file << "plt.plot(a, 'b');\n";
            for (std::size_t i = 0; i < 3; ++i) {
                file << "p" << i << " = "
                     << core::CContainerPrinter::print(m_ModelBounds[i]) << ";\n";
                file << "f" << i << " = "
                     << core::CContainerPrinter::print(m_Forecast[i]) << ";\n";
                file << "plt.plot(p" << i << ", 'g');\n";
                file << "plt.plot(range(len(a)-len(f" << i << "),len(a)),f" << i << ", 'r');\n";
            }
            file << "plt.show();\n";
        }
    }
    void addValue(double value) {
        if (ENABLED) {
            m_Actual.push_back(value);
        }
    }
    void addValueAndPrediction(core_t::TTime time,
                               double value,
                               const maths::CUnivariateTimeSeriesModel& model) {
        if (ENABLED) {
            m_Actual.push_back(value);
            auto x = model.confidenceInterval(
                time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
            if (x.size() == 3) {
                for (std::size_t i = 0; i < 3; ++i) {
                    m_ModelBounds[i].push_back(x[i][0]);
                }
            }
        }
    }
    void addForecast(const maths::SErrorBar& errorBar) {
        if (ENABLED) {
            m_Forecast[0].push_back(errorBar.s_LowerBound);
            m_Forecast[1].push_back(errorBar.s_Predicted);
            m_Forecast[2].push_back(errorBar.s_UpperBound);
        }
    }

private:
    std::string m_File;
    TDoubleVec m_Actual;
    TDoubleVecVec m_ModelBounds;
    TDoubleVecVec m_Forecast;
};
}

BOOST_AUTO_TEST_CASE(testClone) {
    // Test all the state is cloned.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    {
        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CTimeSeriesCorrelations correlations{MINIMUM_SIGNIFICANT_CORRELATION, DECAY_RATE};
        maths::CUnivariateTimeSeriesModel model(modelParams(bucketLength), 1, trend,
                                                univariateNormal(), &controllers);
        model.modelCorrelations(correlations);

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        uint64_t checksum1 = model.checksum();
        std::unique_ptr<maths::CUnivariateTimeSeriesModel> clone1{model.clone(1)};
        uint64_t checksum2 = clone1->checksum();
        BOOST_REQUIRE_EQUAL(checksum1, checksum2);
        std::unique_ptr<maths::CUnivariateTimeSeriesModel> clone2{model.clone(2)};
        BOOST_REQUIRE_EQUAL(std::size_t(2), clone2->identifier());
    }
    {
        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(3);
        maths::CMultivariateTimeSeriesModel model(
            modelParams(bucketLength), trend, multivariateNormal(), &controllers);

        TDoubleVec mean{13.0, 9.0, 10.0};
        TDoubleVecVec covariance{{3.5, 2.9, 0.5}, {2.9, 3.6, 0.1}, {0.5, 0.1, 2.1}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (const auto& sample : samples) {
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        uint64_t checksum1 = model.checksum();
        std::unique_ptr<maths::CMultivariateTimeSeriesModel> clone1{model.clone(1)};
        uint64_t checksum2 = clone1->checksum();
        BOOST_REQUIRE_EQUAL(checksum1, checksum2);
        std::unique_ptr<maths::CMultivariateTimeSeriesModel> clone2{model.clone(2)};
        uint64_t checksum3 = clone2->checksum();
        BOOST_REQUIRE_EQUAL(checksum1, checksum3);
        BOOST_REQUIRE_EQUAL(std::size_t(0), clone2->identifier());
    }
}

BOOST_AUTO_TEST_CASE(testMode) {
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
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend, prior};

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
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }
        double expectedMode{maths::CBasicStatistics::mean(trend.value(time)) +
                            prior.marginalLikelihoodMode()};
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(1)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode[0]);
        BOOST_REQUIRE_EQUAL(std::size_t(1), mode.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode, mode[0], 1e-3 * expectedMode);
    }

    LOG_DEBUG(<< "Univariate trend");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);

        auto params = modelParams(bucketLength);
        maths::CTimeSeriesDecomposition trend{48.0 * DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params, 0, trend, prior};

        core_t::TTime time{0};
        for (auto& sample : samples) {
            sample += 20.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                             static_cast<double>(time) / 86400.0);
            time += bucketLength;
        }

        TDouble2VecWeightsAryVec unit{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};

        time = 0;
        for (auto sample : samples) {
            model.addSamples(addSampleParams(unit),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            trend.addPoint(time, sample, maths_t::CUnitWeights::UNIT,
                           makeComponentDetectedCallback(params.learnRate(), prior));
            prior.addSamples({trend.detrend(time, sample, 0.0)},
                             maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);
            time += bucketLength;
        }

        double expectedMode{maths::CBasicStatistics::mean(trend.value(time)) +
                            prior.marginalLikelihoodMode()};
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(1)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode[0]);
        BOOST_REQUIRE_EQUAL(std::size_t(1), mode.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode, mode[0], 1e-3 * expectedMode);
    }

    LOG_DEBUG(<< "Multivariate no trend");
    {
        TDoubleVec mean{10.0, 15.0, 11.0};
        TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength),
                                                  *trends[0], prior};

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
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }
        TDouble2Vec expectedMode(prior.marginalLikelihoodMode(
            maths_t::CUnitWeights::unit<TDouble10Vec>(3)));
        for (std::size_t i = 0u; i < trends.size(); ++i) {
            expectedMode[i] += maths::CBasicStatistics::mean(trends[i]->value(time));
        }
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(3)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode);
        BOOST_REQUIRE_EQUAL(std::size_t(3), mode.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode[0], mode[0], 0.02 * expectedMode[0]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode[1], mode[1], 0.02 * expectedMode[0]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode[2], mode[2], 0.02 * expectedMode[0]);
    }

    LOG_DEBUG(<< "Multivariate trend");
    {
        TDoubleVec mean{10.0, 15.0, 11.0};
        TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        auto params = modelParams(bucketLength);
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{params, *trends[0], prior};

        core_t::TTime time{0};
        for (auto& sample : samples) {
            double amplitude{10.0};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                sample[i] += 30.0 + amplitude *
                                        std::sin(boost::math::double_constants::two_pi *
                                                 static_cast<double>(time) / 86400.0);
                amplitude += 4.0;
            }
            time += bucketLength;
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        time = 0;
        for (const auto& sample : samples) {
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});

            bool reinitialize{false};

            TDouble10Vec1Vec detrended{TDouble10Vec(3)};
            for (std::size_t i = 0; i < sample.size(); ++i) {
                trends[i]->addPoint(time, sample[i], maths_t::CUnitWeights::UNIT,
                                    [&reinitialize](TFloatMeanAccumulatorVec) {
                                        reinitialize = true;
                                    });
                detrended[0][i] = trends[i]->detrend(time, sample[i], 0.0);
            }
            if (reinitialize) {
                reinitializeResidualModel(params.learnRate(), trends, prior);
            }

            prior.addSamples(detrended,
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
            prior.propagateForwardsByTime(1.0);

            time += bucketLength;
        }
        TDouble2Vec expectedMode(prior.marginalLikelihoodMode(
            maths_t::CUnitWeights::unit<TDouble10Vec>(3)));
        for (std::size_t i = 0u; i < trends.size(); ++i) {
            expectedMode[i] += maths::CBasicStatistics::mean(trends[i]->value(time));
        }
        TDouble2Vec mode(model.mode(time, maths_t::CUnitWeights::unit<TDouble2Vec>(3)));

        LOG_DEBUG(<< "expected mode = " << expectedMode);
        LOG_DEBUG(<< "mode          = " << mode);
        BOOST_REQUIRE_EQUAL(std::size_t(3), mode.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode[0], mode[0], 1e-3 * expectedMode[0]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode[1], mode[1], 1e-3 * expectedMode[1]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMode[2], mode[2], 1e-3 * expectedMode[2]);
    }
}

BOOST_AUTO_TEST_CASE(testAddBucketValue) {
    // Test that the prior support is correctly updated to account
    // for negative bucket values.

    core_t::TTime bucketLength{600};
    maths::CTimeSeriesDecompositionStub trend;
    maths::CLogNormalMeanPrecConjugate prior{univariateLogNormal()};
    maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend, prior};

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

    model.addSamples(addSampleParams(modelWeights), samples);
    model.addBucketValue({core::make_triple(core_t::TTime{20}, TDouble2Vec{-1.0}, TAG)});

    BOOST_REQUIRE_EQUAL(prior.checksum(), model.residualModel().checksum());
}

BOOST_AUTO_TEST_CASE(testAddSamples) {
    // Test: 1) Test multiple samples
    //       2) Test propagation interval
    //       3) Test decay rate control

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{1800};

    LOG_DEBUG(<< "Multiple samples univariate");
    {
        maths::CTimeSeriesDecompositionStub trend;
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend, prior};

        TTimeDouble2VecSizeTrVec samples{
            core::make_triple(core_t::TTime{20}, TDouble2Vec{3.5}, TAG),
            core::make_triple(core_t::TTime{12}, TDouble2Vec{3.9}, TAG),
            core::make_triple(core_t::TTime{18}, TDouble2Vec{2.1}, TAG)};
        TDoubleVec weights{1.0, 1.5, 0.9};
        TDouble2VecWeightsAryVec modelWeights{
            maths_t::countWeight(TDouble2Vec{weights[0]}),
            maths_t::countWeight(TDouble2Vec{weights[1]}),
            maths_t::countWeight(TDouble2Vec{weights[2]})};

        model.addSamples(addSampleParams(modelWeights), samples);

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
        uint64_t checksum2{model.trendModel().checksum()};
        LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
        BOOST_REQUIRE_EQUAL(checksum1, checksum2);
        checksum1 = prior.checksum();
        checksum2 = model.residualModel().checksum();
        LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
        BOOST_REQUIRE_EQUAL(checksum1, checksum2);
    }

    LOG_DEBUG(<< "Multiple samples multivariate");
    {
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength),
                                                  *trends[0], prior};

        TTimeDouble2VecSizeTrVec samples{
            core::make_triple(core_t::TTime{20}, TDouble2Vec{3.5, 3.4, 3.3}, TAG),
            core::make_triple(core_t::TTime{12}, TDouble2Vec{3.9, 3.8, 3.7}, TAG),
            core::make_triple(core_t::TTime{18}, TDouble2Vec{2.1, 2.0, 1.9}, TAG)};
        double weights[][3]{{1.0, 1.1, 1.2}, {1.5, 1.6, 1.7}, {0.9, 1.0, 1.1}};
        TDouble2VecWeightsAryVec modelWeights{
            maths_t::countWeight(TDouble2Vec(weights[0], weights[0] + 3)),
            maths_t::countWeight(TDouble2Vec(weights[1], weights[1] + 3)),
            maths_t::countWeight(TDouble2Vec(weights[2], weights[2] + 3))};

        model.addSamples(addSampleParams(modelWeights), samples);

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
            uint64_t checksum2{model.trendModel()[i]->checksum()};
            LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
            BOOST_REQUIRE_EQUAL(checksum1, checksum2);
        }
        uint64_t checksum1{prior.checksum()};
        uint64_t checksum2{model.residualModel().checksum()};
        LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
        BOOST_REQUIRE_EQUAL(checksum1, checksum2);
    }

    LOG_DEBUG(<< "Propagation interval univariate");
    {
        maths::CTimeSeriesDecompositionStub trend;
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend, prior};

        double interval[]{1.0, 1.1, 0.4};
        TDouble2Vec samples[]{{10.0}, {13.9}, {27.1}};
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        maths_t::setCount(TDouble2Vec{1.5}, weights[0]);
        maths_t::setWinsorisationWeight(TDouble2Vec{0.9}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{1.1}, weights[0]);

        core_t::TTime time{0};
        for (std::size_t i = 0u; i < 3; ++i) {
            TTimeDouble2VecSizeTrVec sample{core::make_triple(time, samples[i], TAG)};
            model.addSamples(addSampleParams(interval[i], weights), sample);

            TDoubleWeightsAry1Vec weight{maths_t::CUnitWeights::UNIT};
            for (std::size_t j = 0u; j < weights[0].size(); ++j) {
                weight[0][j] = weights[0][j][0];
            }
            prior.addSamples(TDouble1Vec(samples[i]), weight);
            prior.propagateForwardsByTime(interval[i]);

            uint64_t checksum1{prior.checksum()};
            uint64_t checksum2{model.residualModel().checksum()};
            LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
            BOOST_REQUIRE_EQUAL(checksum1, checksum2);

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
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength),
                                                  *trends[0], prior};

        double interval[]{1.0, 1.1, 0.4};
        TDouble2Vec samples[]{{13.5, 13.4, 13.3}, {13.9, 13.8, 13.7}, {20.1, 20.0, 10.9}};
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        maths_t::setCount(TDouble2Vec{1.0, 1.1, 1.2}, weights[0]);
        maths_t::setWinsorisationWeight(TDouble2Vec{0.1, 0.1, 0.2}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{2.0, 2.1, 2.2}, weights[0]);

        core_t::TTime time{0};
        for (std::size_t i = 0u; i < 3; ++i) {
            TTimeDouble2VecSizeTrVec sample{core::make_triple(time, samples[i], TAG)};
            model.addSamples(addSampleParams(interval[i], weights), sample);

            TDouble10VecWeightsAry1Vec weight{maths_t::CUnitWeights::unit<TDouble10Vec>(3)};
            for (std::size_t j = 0u; j < weights[0].size(); ++j) {
                weight[0][j] = weights[0][j];
            }
            prior.addSamples({TDouble10Vec(samples[i])}, weight);
            prior.propagateForwardsByTime(interval[i]);

            uint64_t checksum1{prior.checksum()};
            uint64_t checksum2{model.residualModel().checksum()};
            LOG_DEBUG(<< "checksum1 = " << checksum1 << " checksum2 = " << checksum2);
            BOOST_REQUIRE_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }

    LOG_DEBUG(<< "Decay rate control univariate");
    {
        auto params = modelParams(bucketLength);
        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel model(params, 1, trend, prior, &controllers);

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

            model.addSamples(addSampleParams(weights), sample_);

            trend.addPoint(time, sample, maths_t::CUnitWeights::UNIT,
                           makeComponentDetectedCallback(params.learnRate(),
                                                         prior, &controllers));

            double detrended{trend.detrend(time, sample, 0.0)};
            prior.addSamples({detrended}, maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);

            if (trend.initialized()) {
                double multiplier{controllers[0].multiplier(
                    {trend.meanValue(time)}, {{detrended}}, bucketLength,
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
            uint64_t checksum2{model.trendModel().checksum()};
            BOOST_REQUIRE_EQUAL(checksum1, checksum2);
            checksum1 = prior.checksum();
            checksum2 = model.residualModel().checksum();
            BOOST_REQUIRE_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }

    LOG_DEBUG(<< "Decay rate control multivariate");
    {
        auto params = modelParams(bucketLength);
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        auto controllers = decayRateControllers(3);
        maths::CMultivariateTimeSeriesModel model{params, *trends[0], prior, &controllers};

        TDoubleVecVec samples;
        {
            TDoubleVec mean{10.0, 15.0, 11.0};
            TDoubleVecVec covariance{{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
            rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);
        }

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};

        core_t::TTime time{0};
        for (auto& sample : samples) {
            TDouble10Vec1Vec detrended{TDouble10Vec(3)};
            TDouble1Vec mean(3);

            double amplitude{10.0};
            for (std::size_t i = 0u; i < sample.size(); ++i) {
                sample[i] = 30.0 +
                            amplitude * std::sin(boost::math::double_constants::two_pi *
                                                 static_cast<double>(time) / 86400.0) +
                            (time / bucketLength > 1800 ? 10.0 : 0.0) + sample[i];
                amplitude += 4.0;
            }

            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});

            bool hasTrend{false};
            bool reinitialize{false};

            for (std::size_t i = 0u; i < sample.size(); ++i) {
                trends[i]->addPoint(time, sample[i], maths_t::CUnitWeights::UNIT,
                                    [&reinitialize](TFloatMeanAccumulatorVec) {
                                        reinitialize = true;
                                    });
                detrended[0][i] = trends[i]->detrend(time, sample[i], 0.0);
                mean[i] = trends[i]->meanValue(time);
                hasTrend |= true;
            }

            if (reinitialize) {
                reinitializeResidualModel(params.learnRate(), trends, prior, &controllers);
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
                uint64_t checksum2{model.trendModel()[i]->checksum()};
                BOOST_REQUIRE_EQUAL(checksum1, checksum2);
            }
            uint64_t checksum1{prior.checksum()};
            uint64_t checksum2{model.residualModel().checksum()};
            BOOST_REQUIRE_EQUAL(checksum1, checksum2);

            time += bucketLength;
        }
    }
}

BOOST_AUTO_TEST_CASE(testPredict) {
    // Test prediction with a trend and with multimodal data.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate seasonal");
    {
        auto params = modelParams(bucketLength);
        maths::CTimeSeriesDecomposition trend{48.0 * DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{params, 0, trend, prior};

        TDoubleVec samples;
        rng.generateNormalSamples(0.0, 4.0, 1008, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            sample += 10.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                             static_cast<double>(time) / 86400.0);

            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});

            trend.addPoint(time, sample, maths_t::CUnitWeights::UNIT,
                           makeComponentDetectedCallback(params.learnRate(), prior));

            prior.addSamples({trend.detrend(time, sample, 0.0)},
                             maths_t::CUnitWeights::SINGLE_UNIT);
            prior.propagateForwardsByTime(1.0);

            time += bucketLength;
        }

        TMeanAccumulator meanError;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            double trend_{10.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                                 static_cast<double>(time_) / 86400.0)};
            double expected{maths::CBasicStatistics::mean(trend.value(time_)) +
                            maths::CBasicStatistics::mean(
                                prior.marginalLikelihoodConfidenceInterval(0.0))};
            double predicted{model.predict(time_)[0]};
            LOG_DEBUG(<< "expected = " << expected << " predicted = " << predicted
                      << " (trend = " << trend_ << ")");
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, predicted, 1e-3 * std::fabs(expected));
            meanError.add(std::fabs(trend_ - predicted) / 10.0);
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.07);
    }

    LOG_DEBUG(<< "Univariate nearest mode");
    {
        maths::CTimeSeriesDecompositionStub trend;
        maths::CMultimodalPrior prior{univariateMultimodal()};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend, prior};

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
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        maths::CModel::TSizeDoublePr1Vec empty;
        double predicted[]{model.predict(time, empty, {-2.0})[0],
                           model.predict(time, empty, {12.0})[0]};

        LOG_DEBUG(<< "expected(0) = " << maths::CBasicStatistics::mean(modes[0])
                  << " actual(0) = " << predicted[0]);
        LOG_DEBUG(<< "expected(1) = " << maths::CBasicStatistics::mean(modes[1])
                  << " actual(1) = " << predicted[1]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(modes[0]),
                                     predicted[0],
                                     0.1 * maths::CBasicStatistics::mean(modes[0]));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(modes[1]),
                                     predicted[1],
                                     0.01 * maths::CBasicStatistics::mean(modes[1]));
    }

    LOG_DEBUG(<< "Multivariate Seasonal");
    {
        auto params = modelParams(bucketLength);
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{48.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{48.0 * DECAY_RATE, bucketLength}},
            TDecompositionPtr{new maths::CTimeSeriesDecomposition{48.0 * DECAY_RATE, bucketLength}}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{
            maths::CMultivariateTimeSeriesModel{params, *trends[0], prior}};

        TDoubleVecVec samples;
        TDoubleVec mean{0.0, 2.0, 1.0};
        rng.generateMultivariateNormalSamples(
            mean, {{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}}, 1000, samples);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            for (auto& coordinate : sample) {
                coordinate += 10.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                                     static_cast<double>(time) / 86400.0);
            }

            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});

            bool reinitialize{false};

            TDouble10Vec detrended;
            for (std::size_t i = 0; i < sample.size(); ++i) {
                trends[i]->addPoint(time, sample[i], maths_t::CUnitWeights::UNIT,
                                    [&reinitialize](TFloatMeanAccumulatorVec) {
                                        reinitialize = true;
                                    });
                detrended.push_back(trends[i]->detrend(time, sample[i], 0.0));
            }
            if (reinitialize) {
                reinitializeResidualModel(params.learnRate(), trends, prior);
            }

            prior.addSamples({detrended},
                             maths_t::CUnitWeights::singleUnit<TDouble10Vec>(3));
            prior.propagateForwardsByTime(1.0);

            time += bucketLength;
        }

        TMeanAccumulator meanError;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            maths::CMultivariatePrior::TSize10Vec marginalize{1, 2};
            maths::CMultivariatePrior::TSizeDoublePr10Vec condition;
            for (std::size_t i = 0u; i < mean.size(); ++i) {
                double trend_{mean[i] + 10.0 +
                              10.0 * std::sin(boost::math::double_constants::two_pi *
                                              static_cast<double>(time_) / 86400.0)};
                maths::CMultivariatePrior::TUnivariatePriorPtr margin{
                    prior.univariate(marginalize, condition).first};
                double expected{maths::CBasicStatistics::mean(trends[i]->value(time_)) +
                                maths::CBasicStatistics::mean(
                                    margin->marginalLikelihoodConfidenceInterval(0.0))};
                double predicted{model.predict(time_)[i]};
                --marginalize[std::min(i, marginalize.size() - 1)];
                LOG_DEBUG(<< "expected = " << expected << " predicted = " << predicted
                          << " (trend = " << trend_ << ")");
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expected, predicted, 1e-3 * expected);
                meanError.add(std::fabs(trend_ - predicted) / 10.0);
            }
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.07);
    }

    LOG_DEBUG(<< "Multivariate nearest mode");
    {
        TDecompositionPtr10Vec trends{
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}},
            TDecompositionPtr{new maths::CTimeSeriesDecompositionStub{}}};
        maths::CMultivariateMultimodalPrior<3> prior{multivariateMultimodal()};
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength),
                                                  *trends[0], prior};

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
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});
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
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected[0][i], predicted[0][i],
                                         std::fabs(0.2 * expected[0][i]));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected[1][i], predicted[1][i],
                                         std::fabs(0.01 * expected[1][i]));
        }
    }
}

BOOST_AUTO_TEST_CASE(testProbability) {
    // Test: 1) The different the calculation matches the expected values
    //          given the trend and decomposition for different calculations,
    //          seasonal confidence intervals, weights and so on.
    //       2) Test the calculation with and without trend.
    //       3) Test manually injected anomalies have low probabilities.

    using TDoubleSizePr = std::pair<double, std::size_t>;

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{600};

    LOG_DEBUG(<< "Univariate");
    {
        maths::CUnivariateTimeSeriesModel model0{modelParams(bucketLength),
                                                 1, // id
                                                 maths::CTimeSeriesDecompositionStub{},
                                                 univariateNormal(),
                                                 nullptr, // no decay rate control
                                                 nullptr, // no multi-bucket
                                                 false};
        maths::CUnivariateTimeSeriesModel model1{
            modelParams(bucketLength),
            1, // id
            maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength},
            univariateNormal(),
            nullptr, // no decay rate control
            nullptr, // no multi-bucket
            false};

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 4.0, 1000, samples);

        core_t::TTime time{0};
        {
            const TDouble2VecWeightsAryVec weight{
                maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
            for (auto sample : samples) {
                double trend{5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                  static_cast<double>(time) / 86400.0)};
                model0.addSamples(addSampleParams(weight),
                                  {core::make_triple(time, TDouble2Vec{sample}, TAG)});
                model1.addSamples(
                    addSampleParams(weight),
                    {core::make_triple(time, TDouble2Vec{trend + sample}, TAG)});
                time += bucketLength;
            }
        }

        TTime2Vec1Vec time_{{time}};
        TDouble2Vec sample{15.0};

        maths_t::EProbabilityCalculation calculations[]{maths_t::E_TwoSided,
                                                        maths_t::E_OneSidedAbove};
        double confidences[]{0.0, 20.0, 50.0};
        TDouble2VecWeightsAryVec weights(2, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        maths_t::setCountVarianceScale(TDouble2Vec{0.9}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{1.1}, weights[1]);
        maths_t::setSeasonalVarianceScale(TDouble2Vec{1.8}, weights[1]);

        for (auto calculation : calculations) {
            LOG_DEBUG(<< "calculation = " << calculation);
            for (auto confidence : confidences) {
                LOG_DEBUG(<< " confidence = " << confidence);
                for (const auto& weight : weights) {
                    LOG_DEBUG(<< "   weights = " << core::CContainerPrinter::print(weight));
                    double expectedProbability[2];
                    maths_t::ETail expectedTail[2];
                    {
                        maths_t::TDoubleWeightsAry weight_(maths_t::CUnitWeights::UNIT);
                        for (std::size_t i = 0u; i < weight.size(); ++i) {
                            weight_[i] = weight[i][0];
                        }
                        double lb[2], ub[2];
                        model0.residualModel().probabilityOfLessLikelySamples(
                            calculation, sample, {weight_}, lb[0], ub[0], expectedTail[0]);
                        model1.residualModel().probabilityOfLessLikelySamples(
                            calculation,
                            {model1.trendModel().detrend(time, sample[0], confidence)},
                            {weight_}, lb[1], ub[1], expectedTail[1]);
                        expectedProbability[0] = (lb[0] + ub[0]) / 2.0;
                        expectedProbability[1] = (lb[1] + ub[1]) / 2.0;
                    }

                    maths::SModelProbabilityResult results[2];
                    {
                        maths::CModelProbabilityParams params;
                        params.addCalculation(calculation)
                            .seasonalConfidenceInterval(confidence)
                            .addWeights(weight)
                            .useMultibucketFeatures(false);
                        model0.probability(params, time_, {sample}, results[0]);
                        model1.probability(params, time_, {sample}, results[1]);
                    }

                    BOOST_REQUIRE_EQUAL(expectedProbability[0], results[0].s_Probability);
                    BOOST_REQUIRE_EQUAL(expectedTail[0], results[0].s_Tail[0]);
                    BOOST_REQUIRE_EQUAL(expectedProbability[1], results[1].s_Probability);
                    BOOST_REQUIRE_EQUAL(expectedTail[1], results[1].s_Tail[0]);
                }
            }
        }
    }

    LOG_DEBUG(<< "Multivariate");
    {
        maths::CMultivariateTimeSeriesModel model0{modelParams(bucketLength),
                                                   maths::CTimeSeriesDecompositionStub{},
                                                   multivariateNormal(),
                                                   nullptr,
                                                   nullptr,
                                                   false};
        maths::CMultivariateTimeSeriesModel model1{
            modelParams(bucketLength),
            maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength},
            multivariateNormal(),
            nullptr,
            nullptr,
            false};

        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(
            {10.0, 15.0, 11.0},
            {{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}}, 1000, samples);

        core_t::TTime time{0};
        {
            TDouble2VecWeightsAryVec weight{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
            for (auto& sample : samples) {
                TDouble2Vec sample_(sample);
                model0.addSamples(addSampleParams(weight),
                                  {core::make_triple(time, sample_, TAG)});
                double trend{5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                  static_cast<double>(time) / 86400.0)};
                for (auto& component : sample_) {
                    component += trend;
                }
                model1.addSamples(addSampleParams(weight),
                                  {core::make_triple(time, sample_, TAG)});
                time += bucketLength;
            }
        }

        TTime2Vec1Vec time_{{time}};
        TDouble2Vec sample{15.0, 14.0, 16.0};

        maths_t::EProbabilityCalculation calculations[]{maths_t::E_TwoSided,
                                                        maths_t::E_OneSidedAbove};
        double confidences[]{0.0, 20.0, 50.0};
        TDouble2VecWeightsAryVec weights(2, maths_t::CUnitWeights::unit<TDouble2Vec>(3));
        maths_t::setCountVarianceScale(TDouble2Vec{0.9, 0.9, 0.8}, weights[0]);
        maths_t::setCountVarianceScale(TDouble2Vec{1.1, 1.0, 1.2}, weights[1]);
        maths_t::setSeasonalVarianceScale(TDouble2Vec{1.8, 1.7, 1.6}, weights[1]);

        for (auto calculation : calculations) {
            LOG_DEBUG(<< "calculation = " << calculation);
            for (auto confidence : confidences) {
                LOG_DEBUG(<< " confidence = " << confidence);
                for (const auto& weight : weights) {
                    LOG_DEBUG(<< "   weights = " << core::CContainerPrinter::print(weight));
                    double expectedProbability[2];
                    TTail10Vec expectedTail[2];
                    {
                        maths_t::TDouble10VecWeightsAry weight_(
                            maths_t::CUnitWeights::unit<TDouble10Vec>(3));
                        for (std::size_t i = 0u; i < weight.size(); ++i) {
                            weight_[i] = weight[i];
                        }
                        double lb[2], ub[2];
                        model0.residualModel().probabilityOfLessLikelySamples(
                            calculation, {TDouble10Vec(sample)}, {weight_},
                            lb[0], ub[0], expectedTail[0]);
                        TDouble10Vec detrended;
                        for (std::size_t j = 0u; j < sample.size(); ++j) {
                            detrended.push_back(model1.trendModel()[j]->detrend(
                                time, sample[j], confidence));
                        }
                        model1.residualModel().probabilityOfLessLikelySamples(
                            calculation, {detrended}, {weight_}, lb[1], ub[1],
                            expectedTail[1]);
                        expectedProbability[0] = (lb[0] + ub[0]) / 2.0;
                        expectedProbability[1] = (lb[1] + ub[1]) / 2.0;
                    }

                    maths::SModelProbabilityResult results[2];
                    {
                        maths::CModelProbabilityParams params;
                        params.addCalculation(calculation)
                            .seasonalConfidenceInterval(confidence)
                            .addWeights(weight)
                            .useMultibucketFeatures(false);
                        model0.probability(params, time_, {sample}, results[0]);
                        model1.probability(params, time_, {sample}, results[1]);
                    }

                    BOOST_REQUIRE_EQUAL(expectedProbability[0], results[0].s_Probability);
                    BOOST_REQUIRE_EQUAL(expectedProbability[1], results[1].s_Probability);
                    for (std::size_t j = 0u; j < 3; ++j) {
                        BOOST_REQUIRE_EQUAL(expectedTail[0][j], results[0].s_Tail[j]);
                        BOOST_REQUIRE_EQUAL(expectedTail[1][j], results[1].s_Tail[j]);
                    }
                }
            }
        }
    }

    LOG_DEBUG(<< "Anomalies");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 1,
                                                trend, univariateNormal()};

        TSizeVec anomalies;
        rng.generateUniformSamples(100, 1000, 10, anomalies);
        std::sort(anomalies.begin(), anomalies.end());
        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 4.0, 1000, samples);

        maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> smallest(10);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        std::size_t bucket{0};
        core_t::TTime time{0};
        for (auto sample : samples) {
            if (std::binary_search(anomalies.begin(), anomalies.end(), bucket++)) {
                sample += 18.0;
            }
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            maths::SModelProbabilityResult result;
            model.probability(computeProbabilityParams(weights[0]), {{time}},
                              {{sample}}, result);
            smallest.add({result.s_Probability, bucket - 1});
            time += bucketLength;
        }

        TSizeVec anomalies_;
        std::transform(smallest.begin(), smallest.end(), std::back_inserter(anomalies_),
                       [](const TDoubleSizePr& value) { return value.second; });
        std::sort(anomalies_.begin(), anomalies_.end());

        LOG_DEBUG(<< "expected anomalies = " << core::CContainerPrinter::print(anomalies));
        LOG_DEBUG(<< "actual anomalies   = " << core::CContainerPrinter::print(anomalies_));
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(anomalies),
                            core::CContainerPrinter::print(anomalies_));
    }
}

BOOST_AUTO_TEST_CASE(testWeights) {
    // Check that the seasonal weight matches the value we expect given
    //   1) the trend and residual model
    //   2) the variation in the input data
    //
    // And that the Winsorisation weight is monotonic decreasing with
    // increasing distance from the expected value.

    core_t::TTime bucketLength{1800};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend, prior};

        TDoubleVec samples;
        rng.generateNormalSamples(0.0, 4.0, 1008, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            double scale{10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                               static_cast<double>(time) / 86400.0)};
            sample = scale * (1.0 + 0.1 * sample);
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        LOG_DEBUG(<< "Seasonal");
        TMeanAccumulator error;
        TDouble2Vec scale;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            double dataScale{std::pow(
                1.0 + 0.5 * std::sin(boost::math::double_constants::two_pi *
                                     static_cast<double>(time_) / 86400.0),
                2.0)};

            double expectedScale{
                model.trendModel()
                    .varianceScaleWeight(
                        time_, model.residualModel().marginalLikelihoodVariance(), 0.0)
                    .second};
            model.seasonalWeight(0.0, time_, scale);

            LOG_DEBUG(<< "expected weight = " << expectedScale << ", scale = " << scale
                      << " (data weight = " << dataScale << ")");
            BOOST_REQUIRE_EQUAL(std::max(expectedScale, MINIMUM_SEASONAL_SCALE), scale[0]);

            error.add(std::fabs(scale[0] - dataScale) / dataScale);
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.26);

        LOG_DEBUG(<< "Winsorisation");
        TDouble2Vec prediction(model.predict(time));
        TDouble2VecWeightsAry trendWeights;
        TDouble2VecWeightsAry residualWeights;
        double lastTrendWinsorisationWeight{1.0};
        double lastResidualWinsorisationWeight{1.0};
        for (std::size_t i = 0; i < 10; ++i) {
            model.countWeights(time, prediction, 1.0, 1.0, 1.0, 1.0,
                               trendWeights, residualWeights);
            double trendWinsorisationWeight{maths_t::winsorisationWeight(trendWeights)[0]};
            double residualWinsorisationWeight{
                maths_t::winsorisationWeight(residualWeights)[0]};
            LOG_DEBUG(<< "trend weight = " << trendWinsorisationWeight
                      << ", residual weight = " << residualWinsorisationWeight);
            BOOST_TEST_REQUIRE(trendWinsorisationWeight <= lastTrendWinsorisationWeight);
            BOOST_TEST_REQUIRE(residualWinsorisationWeight <= lastResidualWinsorisationWeight);
            lastTrendWinsorisationWeight = trendWinsorisationWeight;
            lastResidualWinsorisationWeight = residualWinsorisationWeight;
            prediction[0] *= 1.1;
        }
    }

    LOG_DEBUG(<< "Multivariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength), trend, prior};

        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(
            {10.0, 15.0, 11.0},
            {{3.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}}, 1008, samples);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            double scale{10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                               static_cast<double>(time) / 86400.0)};
            for (auto& component : sample) {
                component = scale * (1.0 + 0.1 * component);
            }
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        LOG_DEBUG(<< "Seasonal");
        TMeanAccumulator error;
        TDouble2Vec scale;
        for (core_t::TTime time_ = time; time_ < time + 86400; time_ += 3600) {
            double dataScale{std::pow(
                1.0 + 0.5 * std::sin(boost::math::double_constants::two_pi *
                                     static_cast<double>(time_) / 86400.0),
                2.0)};

            model.seasonalWeight(0.0, time_, scale);
            for (std::size_t i = 0; i < 3; ++i) {
                double expectedScale{
                    model.trendModel()[i]
                        ->varianceScaleWeight(
                            time_, model.residualModel().marginalLikelihoodVariances()[i], 0.0)
                        .second};
                LOG_DEBUG(<< "expected weight = " << expectedScale << ", weight = " << scale
                          << " (data weight = " << dataScale << ")");
                BOOST_REQUIRE_EQUAL(std::max(expectedScale, MINIMUM_SEASONAL_SCALE),
                                    scale[i]);
                error.add(std::fabs(scale[i] - dataScale) / dataScale);
            }
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.26);

        LOG_DEBUG(<< "Winsorisation");
        TDouble2Vec prediction(model.predict(time));
        TDouble2VecWeightsAry trendWeights;
        TDouble2VecWeightsAry residualWeights;
        double lastTrendWinsorisationWeight{1.0};
        double lastResidualWinsorisationWeight{1.0};
        for (std::size_t i = 0; i < 10; ++i) {
            model.countWeights(time, prediction, 1.0, 1.0, 1.0, 1.0,
                               trendWeights, residualWeights);
            double trendWinsorisationWeight{maths_t::winsorisationWeight(trendWeights)[0]};
            double residualWinsorisationWeight{
                maths_t::winsorisationWeight(residualWeights)[0]};
            LOG_DEBUG(<< "trend weight = " << trendWinsorisationWeight
                      << ", residual weight = " << residualWinsorisationWeight);
            BOOST_TEST_REQUIRE(trendWinsorisationWeight <= lastTrendWinsorisationWeight);
            BOOST_TEST_REQUIRE(residualWinsorisationWeight <= lastResidualWinsorisationWeight);
            lastTrendWinsorisationWeight = trendWinsorisationWeight;
            lastResidualWinsorisationWeight = residualWinsorisationWeight;
            prediction[0] *= 1.1;
        }
    }
}

BOOST_AUTO_TEST_CASE(testMemoryUsage) {
    // Test we account for the appropriate memory.

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        std::unique_ptr<maths::CModel> model{new maths::CUnivariateTimeSeriesModel{
            modelParams(bucketLength), 0, trend, univariateNormal(), &controllers}};

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            sample += 10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(time) / 86400.0);
            trend.addPoint(time, sample);
            model->addSamples(addSampleParams(weights),
                              {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        std::size_t expectedSize{
            sizeof(maths::CTimeSeriesDecomposition) + trend.memoryUsage() +
            3 * sizeof(maths::CNormalMeanPrecConjugate) +
            sizeof(maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary) +
            2 * controllers[0].memoryUsage() + 16 * 12 /*Recent samples*/};
        std::size_t size{model->memoryUsage()};
        LOG_DEBUG(<< "size " << size << " expected " << expectedSize);
        BOOST_TEST_REQUIRE(static_cast<double>(size) < 1.1 * static_cast<double>(expectedSize));
    }

    LOG_DEBUG(<< "Multivariate");
    {
        TDoubleVec mean{11.0, 10.0, 12.0};
        TDoubleVecVec covariance{{4.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(mean, covariance, 1000, samples);

        maths::CTimeSeriesDecomposition trend[]{
            maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength},
            maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength},
            maths::CTimeSeriesDecomposition{24.0 * DECAY_RATE, bucketLength}};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        auto controllers = decayRateControllers(3);
        auto model = std::make_unique<maths::CMultivariateTimeSeriesModel>(
            modelParams(bucketLength), trend[0], prior, &controllers);

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (auto& sample : samples) {
            for (std::size_t i = 0; i < sample.size(); ++i) {
                sample[i] += 10.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                                   static_cast<double>(time) / 86400.0);
                trend[i].addPoint(time, sample[i]);
            }
            model->addSamples(addSampleParams(weights),
                              {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        std::size_t expectedSize{
            3 * sizeof(maths::CTimeSeriesDecomposition) + trend[0].memoryUsage() +
            trend[1].memoryUsage() + trend[2].memoryUsage() +
            2 * sizeof(maths::CMultivariateNormalConjugate<3>) +
            sizeof(maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary) +
            2 * controllers[0].memoryUsage() + 32 * 12 /*Recent samples*/};
        std::size_t size{model->memoryUsage()};
        LOG_DEBUG(<< "size " << size << " expected " << expectedSize);
        BOOST_TEST_REQUIRE(static_cast<double>(size) < 1.2 * static_cast<double>(expectedSize));
    }

    // TODO LOG_DEBUG(<< "Correlates");
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Test the restored model checksum matches the persisted model.

    core_t::TTime bucketLength{600};
    maths::CModelParams params{modelParams(bucketLength)};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate");
    {
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel origModel{
            params, 1, trend, univariateNormal(), &controllers};

        TDoubleVec samples;
        rng.generateNormalSamples(1.0, 4.0, 1000, samples);
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            origModel.addSamples(addSampleParams(weights),
                                 {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            time += bucketLength;
        }

        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter{"root"};
            origModel.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_TRACE(<< "model XML representation:\n" << origXml);
        LOG_DEBUG(<< "model XML size: " << origXml.size());

        // Restore the XML into a new time series model.
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::SDistributionRestoreParams distributionParams{maths_t::E_ContinuousData,
                                                             DECAY_RATE};
        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength, distributionParams};
        maths::SModelRestoreParams restoreParams{params, decompositionParams, distributionParams};
        maths::CUnivariateTimeSeriesModel restoredModel{restoreParams, traverser};

        BOOST_REQUIRE_EQUAL(origModel.checksum(), restoredModel.checksum());
    }

    LOG_DEBUG(<< "Multivariate");
    {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(
            {11.0, 10.0, 12.0},
            {{4.0, 2.9, 0.5}, {2.9, 2.6, 0.1}, {0.5, 0.1, 2.0}}, 1000, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        auto controllers = decayRateControllers(3);
        maths::CMultivariateTimeSeriesModel origModel{modelParams(bucketLength),
                                                      trend, prior, &controllers};

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (const auto& sample : samples) {
            origModel.addSamples(addSampleParams(weights),
                                 {core::make_triple(time, TDouble2Vec(sample), TAG)});
            time += bucketLength;
        }

        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter{"root"};
            origModel.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_TRACE(<< "model XML representation:\n" << origXml);
        LOG_DEBUG(<< "model XML size: " << origXml.size());

        // Restore the XML into a new time series model.
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::SDistributionRestoreParams distributionParams{maths_t::E_ContinuousData,
                                                             DECAY_RATE};
        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength, distributionParams};
        maths::SModelRestoreParams restoreParams{params, decompositionParams, distributionParams};
        maths::CMultivariateTimeSeriesModel restoredModel{restoreParams, traverser};

        BOOST_REQUIRE_EQUAL(origModel.checksum(), restoredModel.checksum());
    }

    // TODO LOG_DEBUG(<< "Correlates");
}

BOOST_AUTO_TEST_CASE(testUpgradeFrom6p2) {
    // Test upgrade is minimally disruptive. We test the upgraded model
    // predicted confidence intervals verses the values we obtain from
    // the previous model. Note the confidence interval depends on both
    // trend and residual model so this test is sensitive to problems
    // restoring either.

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
    maths::CModelParams params{modelParams(bucketLength)};
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
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::SDistributionRestoreParams distributionParams{maths_t::E_ContinuousData,
                                                             DECAY_RATE};
        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength, distributionParams};
        maths::SModelRestoreParams restoreParams{params, decompositionParams, distributionParams};
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

            BOOST_REQUIRE_EQUAL(expectedInterval.size(), interval.size());
            for (std::size_t j = 0u; j < expectedInterval.size(); ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    boost::lexical_cast<double>(expectedInterval[j]),
                    boost::lexical_cast<double>(interval[j]), 0.0001);
            }
        }
        BOOST_TEST_REQUIRE(restoredModel.decayRateControllers() != nullptr);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[0].checks() != 0);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[1].checks() != 0);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[0].dimension() == 1);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[1].dimension() == 1);
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
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::SDistributionRestoreParams distributionParams{maths_t::E_ContinuousData,
                                                             DECAY_RATE};
        maths::STimeSeriesDecompositionRestoreParams decompositionParams{
            24.0 * DECAY_RATE, bucketLength, distributionParams};
        maths::SModelRestoreParams restoreParams{params, decompositionParams, distributionParams};
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

            BOOST_REQUIRE_EQUAL(expectedInterval.size(), interval.size());
            for (std::size_t j = 0u; j < expectedInterval.size(); ++j) {
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    boost::lexical_cast<double>(expectedInterval[j]),
                    boost::lexical_cast<double>(interval[j]), 0.0001);
            }
        }
        BOOST_TEST_REQUIRE(restoredModel.decayRateControllers() != nullptr);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[0].checks() != 0);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[1].checks() != 0);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[0].dimension() == 3);
        BOOST_TEST_REQUIRE((*restoredModel.decayRateControllers())[1].dimension() == 3);
    }
}

BOOST_AUTO_TEST_CASE(testAddSamplesWithCorrelations) {
    LOG_DEBUG(<< "Correlations no trend");

    core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;

    {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples({10.0, 15.0}, {{3.0, 2.9}, {2.9, 2.6}},
                                              1000, samples);

        maths::CTimeSeriesDecomposition trend{DECAY_RATE, bucketLength};
        maths::CTimeSeriesCorrelations correlations{MINIMUM_SIGNIFICANT_CORRELATION, DECAY_RATE};
        maths::CNormalMeanPrecConjugate prior{univariateNormal()};
        maths::CUnivariateTimeSeriesModel model0{modelParams(bucketLength), 0,
                                                 trend, prior, nullptr};
        maths::CUnivariateTimeSeriesModel model1{modelParams(bucketLength), 1,
                                                 trend, prior, nullptr};
        model0.modelCorrelations(correlations);
        model1.modelCorrelations(correlations);
        CTimeSeriesCorrelateModelAllocator allocator;

        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (auto sample : samples) {
            correlations.refresh(allocator);
            model0.addSamples(addSampleParams(weights),
                              {core::make_triple(time, TDouble2Vec{sample[0]}, TAG)});
            model1.addSamples(addSampleParams(weights),
                              {core::make_triple(time, TDouble2Vec{sample[1]}, TAG)});
            correlations.processSamples();
            time += bucketLength;
        }

        BOOST_TEST_REQUIRE(correlations.correlationModels().size() > 0);
    }

    // TODO LOG_DEBUG(<< "Correlations trend");
    // TODO LOG_DEBUG(<< "Correlations with tags (for population)");
}

BOOST_AUTO_TEST_CASE(testAnomalyModel) {
    // We test we can find the "odd anomaly out".

    using TDoubleSizePr = std::pair<double, std::size_t>;

    test::CRandomNumbers rng;

    std::size_t length = 2000;

    LOG_DEBUG(<< "Univariate");
    {
        TSizeVec anomalies;
        rng.generateUniformSamples(0, length, 30, anomalies);
        std::sort(anomalies.begin(), anomalies.end());

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 2.0, length, samples);

        core_t::TTime bucketLength{600};
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 1,
                                                trend, univariateNormal()};

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
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec{sample}, TAG)});
            maths::SModelProbabilityResult result;
            model.probability(computeProbabilityParams(weights[0]), {{time}},
                              {{sample}}, result);
            mostAnomalous.add({std::log(result.s_Probability), bucket});
            //scores.push_back(maths::CTools::anomalyScore(probability));
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
        BOOST_TEST_REQUIRE(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                     1905) != anomalyBuckets.end());
        BOOST_TEST_REQUIRE(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                     1906) != anomalyBuckets.end());
        BOOST_TEST_REQUIRE(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                     1907) != anomalyBuckets.end());

        //file << "v = " << core::CContainerPrinter::print(samples) << ";\n";
        //file << "s = " << core::CContainerPrinter::print(scores) << ";\n";
        //file << "hold on;";
        //file << "subplot(2,1,1);\n";
        //file << "plot([1:length(v)], v);\n";
        //file << "subplot(2,1,2);\n";
        //file << "plot([1:length(s)], s, 'r');\n";
    }

    LOG_DEBUG(<< "Multivariate");
    {
        TSizeVec anomalies;
        rng.generateUniformSamples(0, length, 30, anomalies);
        std::sort(anomalies.begin(), anomalies.end());
        core_t::TTime bucketLength{600};
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(
            {10.0, 10.0, 10.0},
            {{4.0, 0.9, 0.5}, {0.9, 2.6, 0.1}, {0.5, 0.1, 3.0}}, length, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength), trend, prior};

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
            model.addSamples(addSampleParams(weights),
                             {core::make_triple(time, TDouble2Vec(sample), TAG)});
            maths::SModelProbabilityResult result;
            model.probability(computeProbabilityParams(weights[0]), {{time}},
                              {(sample)}, result);
            mostAnomalous.add({std::log(result.s_Probability), bucket});
            //scores.push_back(maths::CTools::anomalyScore(probability));
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
        BOOST_TEST_REQUIRE(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                     1906) != anomalyBuckets.end());
        BOOST_TEST_REQUIRE(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
                                     1907) != anomalyBuckets.end());
        BOOST_TEST_REQUIRE(std::find(anomalyBuckets.begin(), anomalyBuckets.end(),
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

BOOST_AUTO_TEST_CASE(testStepChangeDiscontinuities) {
    // Test reinitialization of the residual model after detecting a
    // step change.
    //
    // Check detection and modelling of step changes in data with
    //    1) Piecewise constant,
    //    2) Saw tooth.

    using TDouble3Vec = core::CSmallVector<double, 3>;
    using TDouble3VecVec = std::vector<TDouble3Vec>;

    TDouble2VecWeightsAryVec trendWeights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    TDouble2VecWeightsAryVec residualWeights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    auto updateModel = [&](core_t::TTime time, double value,
                           maths::CUnivariateTimeSeriesModel& model) {
        model.countWeights(time, {value}, 1.0, 1.0, 0.0, 1.0, trendWeights[0],
                           residualWeights[0]);
        model.addSamples(addSampleParams(1.0, trendWeights, residualWeights),
                         {core::make_triple(time, TDouble2Vec{value}, TAG)});
    };

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Univariate: Residual Model Reinitialization");
    {
        core_t::TTime bucketLength{600};
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend,
                                                univariateNormal(DECAY_RATE / 3.0),
                                                &controllers};
        CChangeDebug debug("prior_reinitialization.py");

        core_t::TTime time{0};
        TDoubleVec samples;
        for (auto level : {20.0, 40.0}) {
            for (std::size_t i = 0; i < 200; ++i) {
                updateModel(time, level, model);
                debug.addValueAndPrediction(time, level, model);
                time += bucketLength;
            }
        }

        auto confidenceInterval = model.confidenceInterval(
            time, 50.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        LOG_DEBUG(<< "confidence interval = " << confidenceInterval);
        BOOST_TEST_REQUIRE(std::fabs(confidenceInterval[1][0] - 40.0) < 1.0);
        BOOST_TEST_REQUIRE(confidenceInterval[2][0] - confidenceInterval[0][0] < 4.0);
    }
    LOG_DEBUG(<< "Univariate: Piecewise Constant");
    {
        core_t::TTime bucketLength{1800};
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend,
                                                univariateNormal(DECAY_RATE / 3.0),
                                                &controllers};
        CChangeDebug debug("piecewise_constant.py");

        // Add some data to the model.

        core_t::TTime time{0};
        TDoubleVec samples;
        double level{20.0};
        for (auto dl : {10.0, 20.0, 25.0, 50.0, 30.0, 40.0, 15.0, 40.0, 25.0}) {
            level += dl;
            rng.generateNormalSamples(
                level, 2.0, 300 + static_cast<std::size_t>(2.0 * dl), samples);
            for (auto sample : samples) {
                updateModel(time, sample, model);
                debug.addValueAndPrediction(time, sample, model);
                time += bucketLength;
            }
        }
        level += 30.0;
        rng.generateNormalSamples(level, 2.0, 100, samples);
        for (auto sample : samples) {
            updateModel(time, sample, model);
            debug.addValueAndPrediction(time, sample, model);
            time += bucketLength;
        }

        // Generate expected values from the same process.

        TDoubleVec expected;
        rng.generateNormalSamples(level, 2.0, 260, expected);
        for (auto dl : {25.0, 40.0}) {
            level += dl;
            rng.generateNormalSamples(
                level, 2.0, 300 + static_cast<std::size_t>(2.0 * dl), samples);
            expected.insert(expected.end(), samples.begin(), samples.end());
        }
        std::for_each(expected.begin(), expected.end(),
                      [&debug](double sample) { debug.addValue(sample); });

        // Test forecasting.

        TDouble3VecVec forecast;
        auto pushErrorBar = [&](const maths::SErrorBar& errorBar) {
            forecast.push_back({errorBar.s_LowerBound, errorBar.s_Predicted,
                                errorBar.s_UpperBound});
            debug.addForecast(errorBar);
        };

        std::string m;
        model.forecast(0, time, time, time + expected.size() * bucketLength,
                       90.0, {-1000.0}, {1000.0}, pushErrorBar, m);

        double outOfBounds{0.0};
        for (std::size_t i = 0; i < forecast.size(); ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expected[i], forecast[i][1], 0.1 * expected[i]);
            outOfBounds += static_cast<double>(expected[i] < forecast[i][0] ||
                                               expected[i] > forecast[i][2]);
        }
        double percentageOutOfBounds{100.0 * outOfBounds /
                                     static_cast<double>(forecast.size())};
        LOG_DEBUG(<< "% out-of-bounds = " << percentageOutOfBounds);
        BOOST_TEST_REQUIRE(percentageOutOfBounds < 2.0);
    }

    LOG_DEBUG(<< "Univariate: Saw Tooth");
    {
        core_t::TTime bucketLength{1800};
        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        auto controllers = decayRateControllers(1);
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend,
                                                univariateNormal(), &controllers};
        CChangeDebug debug("saw_tooth.py");

        // Add some data to the model.

        core_t::TTime time{0};
        double value{10.0};
        TDoubleVec noise;
        for (auto slope : {0.08, 0.056, 0.028, 0.044, 0.06, 0.03}) {
            value = 5.0;
            while (value < 95.0) {
                rng.generateNormalSamples(0.0, 2.0, 1, noise);
                updateModel(time, value + noise[0], model);
                debug.addValueAndPrediction(time, value + noise[0], model);
                time += bucketLength;
                value += slope;
            }
        }
        for (auto slope : {0.042}) {
            value = 5.0;
            for (std::size_t i = 0; i < 1500; ++i) {
                rng.generateNormalSamples(0.0, 2.0, 1, noise);
                updateModel(time, value + noise[0], model);
                debug.addValueAndPrediction(time, value + noise[0], model);
                time += bucketLength;
                value += slope;
            }
        }

        // Generate expected values from the same process.

        TDoubleVec expected;
        for (auto slope : {0.05, 0.04}) {
            while (expected.size() < 2000 && value < 95.0) {
                rng.generateNormalSamples(0.0, 2.0, 1, noise);
                expected.push_back(value + noise[0]);
                debug.addValue(value + noise[0]);
                value += slope;
            }
            value = 5.0;
        }

        // Test forecasting.

        TDouble3VecVec forecast;
        auto pushErrorBar = [&](const maths::SErrorBar& errorBar) {
            forecast.push_back({errorBar.s_LowerBound, errorBar.s_Predicted,
                                errorBar.s_UpperBound});
            debug.addForecast(errorBar);
        };

        std::string m;
        model.forecast(0, time, time, time + expected.size() * bucketLength,
                       90.0, {-1000.0}, {1000.0}, pushErrorBar, m);

        double outOfBounds{0.0};
        for (std::size_t i = 0; i < forecast.size(); ++i) {
            outOfBounds += static_cast<double>(expected[i] < forecast[i][0] ||
                                               expected[i] > forecast[i][2]);
        }
        double percentageOutOfBounds{100.0 * outOfBounds /
                                     static_cast<double>(forecast.size())};
        LOG_DEBUG(<< "% out-of-bounds = " << percentageOutOfBounds);
        BOOST_TEST_REQUIRE(percentageOutOfBounds < 6.0);
    }
}

BOOST_AUTO_TEST_CASE(testLinearScaling) {
    // We test that the predictions are good and the bounds do not
    // blow up after we:
    //   1) linearly scale down a periodic pattern,
    //   2) linearly scale up the same periodic pattern.

    TDouble2VecWeightsAryVec trendWeights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    TDouble2VecWeightsAryVec residualWeights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    auto updateModel = [&](core_t::TTime time, double value,
                           maths::CUnivariateTimeSeriesModel& model) {
        model.countWeights(time, {value}, 1.0, 1.0, 0.0, 1.0, trendWeights[0],
                           residualWeights[0]);
        model.addSamples(addSampleParams(1.0, trendWeights, residualWeights),
                         {core::make_triple(time, TDouble2Vec{value}, TAG)});
    };

    test::CRandomNumbers rng;

    double noiseVariance{3.0};

    core_t::TTime bucketLength{600};
    maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
    auto controllers = decayRateControllers(1);
    maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend,
                                            univariateNormal(DECAY_RATE / 3.0), &controllers};
    CChangeDebug debug;

    core_t::TTime time{0};
    TDoubleVec samples;
    rng.generateNormalSamples(0.0, noiseVariance, 1500, samples);
    for (auto sample : samples) {
        sample += 12.0 + 10.0 * smoothDaily(time);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        time += bucketLength;
    }

    LOG_DEBUG(<< "scale by 0.3");

    rng.generateNormalSamples(0.0, noiseVariance, 300, samples);
    for (auto sample : samples) {
        sample = 0.3 * (12.0 + 10.0 * smoothDaily(time) + sample);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        time += bucketLength;
    }
    rng.generateNormalSamples(0.0, noiseVariance, 1500, samples);
    for (auto sample : samples) {
        sample = 0.3 * (12.0 + 10.0 * smoothDaily(time) + sample);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        auto x = model.confidenceInterval(
            time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        BOOST_TEST_REQUIRE(std::fabs(sample - x[1][0]) < 1.6 * std::sqrt(noiseVariance));
        BOOST_TEST_REQUIRE(std::fabs(x[2][0] - x[0][0]) < 3.5 * std::sqrt(noiseVariance));
        time += bucketLength;
    }

    LOG_DEBUG(<< "scale by 6.6");

    rng.generateNormalSamples(0.0, noiseVariance, 300, samples);
    for (auto sample : samples) {
        sample = 2.0 * (12.0 + 10.0 * smoothDaily(time)) + sample;
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        time += bucketLength;
    }
    rng.generateNormalSamples(0.0, noiseVariance, 400, samples);
    for (auto sample : samples) {
        sample = 2.0 * (12.0 + 10.0 * smoothDaily(time)) + sample;
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        auto x = model.confidenceInterval(
            time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        BOOST_TEST_REQUIRE(std::fabs(sample - x[1][0]) < 3.6 * std::sqrt(noiseVariance));
        BOOST_TEST_REQUIRE(std::fabs(x[2][0] - x[0][0]) < 4.0 * std::sqrt(noiseVariance));
        time += bucketLength;
    }
}

BOOST_AUTO_TEST_CASE(testDaylightSaving) {
    // Test we detect daylight saving time shifts.

    TDouble2VecWeightsAryVec trendWeights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    TDouble2VecWeightsAryVec residualWeights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    auto updateModel = [&](core_t::TTime time, double value,
                           maths::CUnivariateTimeSeriesModel& model) {
        model.countWeights(time, {value}, 1.0, 1.0, 0.0, 1.0, trendWeights[0],
                           residualWeights[0]);
        model.addSamples(addSampleParams(1.0, trendWeights, residualWeights),
                         {core::make_triple(time, TDouble2Vec{value}, TAG)});
    };

    test::CRandomNumbers rng;

    core_t::TTime hour{core::constants::HOUR};
    double noiseVariance{0.36};

    core_t::TTime bucketLength{600};
    maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
    auto controllers = decayRateControllers(1);
    maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 0, trend,
                                            univariateNormal(DECAY_RATE / 3.0), &controllers};
    CChangeDebug debug;

    core_t::TTime time{0};
    TDoubleVec samples;
    rng.generateNormalSamples(0.0, noiseVariance, 2000, samples);
    for (auto sample : samples) {
        sample += 12.0 + 10.0 * smoothDaily(time);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        time += bucketLength;
    }

    LOG_DEBUG(<< "shift by 3600s");

    rng.generateNormalSamples(0.0, noiseVariance, 300, samples);
    for (auto sample : samples) {
        sample += 12.0 + 10.0 * smoothDaily(time + hour);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        time += bucketLength;
    }
    rng.generateNormalSamples(0.0, noiseVariance, 1500, samples);
    for (auto sample : samples) {
        sample += 12.0 + 10.0 * smoothDaily(time + hour);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        BOOST_REQUIRE_EQUAL(hour, model.trendModel().timeShift());
        auto x = model.confidenceInterval(
            time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        BOOST_TEST_REQUIRE(std::fabs(sample - x[1][0]) < 4.2 * std::sqrt(noiseVariance));
        BOOST_TEST_REQUIRE(std::fabs(x[2][0] - x[0][0]) < 3.6 * std::sqrt(noiseVariance));
        time += bucketLength;
    }

    LOG_DEBUG(<< "shift by -3600s");

    rng.generateNormalSamples(0.0, noiseVariance, 300, samples);
    for (auto sample : samples) {
        sample += 12.0 + 10.0 * smoothDaily(time);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        time += bucketLength;
    }
    rng.generateNormalSamples(0.0, noiseVariance, 400, samples);
    for (auto sample : samples) {
        sample += 12.0 + 10.0 * smoothDaily(time);
        updateModel(time, sample, model);
        debug.addValueAndPrediction(time, sample, model);
        BOOST_REQUIRE_EQUAL(core_t::TTime(0), model.trendModel().timeShift());
        auto x = model.confidenceInterval(
            time, 90.0, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
        BOOST_TEST_REQUIRE(std::fabs(sample - x[1][0]) < 3.5 * std::sqrt(noiseVariance));
        BOOST_TEST_REQUIRE(std::fabs(x[2][0] - x[0][0]) < 3.8 * std::sqrt(noiseVariance));
        time += bucketLength;
    }
}

BOOST_AUTO_TEST_CASE(testSkipAnomalyModelUpdate) {
    // We test that when parameters dictate to skip updating the anomaly model
    // probabilities become smaller despite seeing many anomalies.

    test::CRandomNumbers rng;

    std::size_t length = 2000;
    core_t::TTime bucketLength{600};

    LOG_DEBUG(<< "Univariate");
    {
        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 2.0, length, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CUnivariateTimeSeriesModel model{modelParams(bucketLength), 1,
                                                trend, univariateNormal()};

        TDoubleVec probabilities;
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        core_t::TTime time{0};
        for (std::size_t bucket = 0; bucket < samples.size(); ++bucket) {
            auto sample = samples[bucket];
            auto currentComputeProbabilityParams = computeProbabilityParams(weights[0]);
            maths::SModelProbabilityResult result;

            // We create anomalies in 10 consecutive buckets
            if (bucket >= 1700 && bucket < 1710) {
                sample = 100.0;
                currentComputeProbabilityParams.skipAnomalyModelUpdate(true);
                model.probability(currentComputeProbabilityParams, {{time}},
                                  {{sample}}, result);
                probabilities.push_back(result.s_Probability);
            } else {
                model.addSamples(addSampleParams(weights),
                                 {core::make_triple(time, TDouble2Vec{sample}, TAG)});
                model.probability(currentComputeProbabilityParams, {{time}},
                                  {{sample}}, result);
            }

            time += bucketLength;
        }

        LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));

        // Assert probs are decreasing
        BOOST_TEST_REQUIRE(probabilities[0] < 0.00001);
        BOOST_TEST_REQUIRE(std::is_sorted(probabilities.rbegin(), probabilities.rend()));
    }

    LOG_DEBUG(<< "Multivariate");
    {
        TDoubleVecVec samples;
        rng.generateMultivariateNormalSamples(
            {10.0, 10.0, 10.0},
            {{4.0, 0.9, 0.5}, {0.9, 2.6, 0.1}, {0.5, 0.1, 3.0}}, length, samples);

        maths::CTimeSeriesDecomposition trend{24.0 * DECAY_RATE, bucketLength};
        maths::CMultivariateNormalConjugate<3> prior{multivariateNormal()};
        maths::CMultivariateTimeSeriesModel model{modelParams(bucketLength), trend, prior};

        TDoubleVec probabilities;
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(3)};
        core_t::TTime time{0};
        for (std::size_t bucket = 0; bucket < samples.size(); ++bucket) {
            auto& sample = samples[bucket];
            auto currentComputeProbabilityParams = computeProbabilityParams(weights[0]);
            maths::SModelProbabilityResult result;

            if (bucket >= 1700 && bucket < 1710) {
                for (auto& coordinate : sample) {
                    coordinate += 100.0;
                }
                currentComputeProbabilityParams.skipAnomalyModelUpdate(true);
                model.probability(currentComputeProbabilityParams, {{time}},
                                  {(sample)}, result);
                probabilities.push_back(result.s_Probability);
            } else {
                model.addSamples(addSampleParams(weights),
                                 {core::make_triple(time, TDouble2Vec(sample), TAG)});
                model.probability(currentComputeProbabilityParams, {{time}},
                                  {(sample)}, result);
            }

            time += bucketLength;
        }

        LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));

        // Assert probs are decreasing
        BOOST_TEST_REQUIRE(probabilities[0] < 0.00001);
        BOOST_TEST_REQUIRE(std::is_sorted(probabilities.rbegin(), probabilities.rend()));
    }
}

BOOST_AUTO_TEST_SUITE_END()
