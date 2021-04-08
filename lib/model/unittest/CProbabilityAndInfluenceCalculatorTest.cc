/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CMultivariateNormalConjugate.h>
#include <maths/CMultivariateNormalConjugateFactory.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>

#include <model/CPartitioningFields.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/CStringStore.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CProbabilityAndInfluenceCalculatorTest)

using namespace ml;

namespace {

using TDoubleDoublePr = std::pair<double, double>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TTimeVec = std::vector<core_t::TTime>;
using TBool2Vec = core::CSmallVector<bool, 2>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;
using TSize1Vec = core::CSmallVector<std::size_t, 1>;
using TSize10Vec = core::CSmallVector<std::size_t, 10>;
using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;
using TTail10Vec = core::CSmallVector<maths_t::ETail, 10>;
using TTimeDouble2VecSizeTr = maths::CModel::TTimeDouble2VecSizeTr;
using TTimeDouble2VecSizeTrVec = maths::CModel::TTimeDouble2VecSizeTrVec;
using TStrCRef = model::CProbabilityAndInfluenceCalculator::TStrCRef;
using TTime2Vec = model::CProbabilityAndInfluenceCalculator::TTime2Vec;
using TTime2Vec1Vec = model::CProbabilityAndInfluenceCalculator::TTime2Vec1Vec;
using TDouble1VecDoublePr = model::CProbabilityAndInfluenceCalculator::TDouble1VecDoublePr;
using TDouble1VecDouble1VecPr = model::CProbabilityAndInfluenceCalculator::TDouble1VecDouble1VecPr;
using TStrCRefDouble1VecDoublePrPr =
    model::CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDoublePrPr;
using TStrCRefDouble1VecDoublePrPrVec =
    model::CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDoublePrPrVec;
using TStrCRefDouble1VecDoublePrPrVecVec = std::vector<TStrCRefDouble1VecDoublePrPrVec>;
using TStrCRefDouble1VecDouble1VecPrPr =
    model::CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDouble1VecPrPr;
using TStrCRefDouble1VecDouble1VecPrPrVec =
    model::CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDouble1VecPrPrVec;
using TStoredStringPtrStoredStringPtrPrDoublePrVec =
    model::CProbabilityAndInfluenceCalculator::TStoredStringPtrStoredStringPtrPrDoublePrVec;
using TInfluenceCalculatorCPtr = std::shared_ptr<const model::CInfluenceCalculator>;

TDouble1VecDoublePr make_pair(double first, double second) {
    return TDouble1VecDoublePr{{first}, second};
}

TDouble1VecDoublePr make_pair(double first1, double first2, double second) {
    return TDouble1VecDoublePr{{first1, first2}, second};
}

TDouble1VecDouble1VecPr make_pair(double first1, double first2, double second1, double second2) {
    return TDouble1VecDouble1VecPr{{first1, first2}, {second1, second2}};
}

maths::CModelParams params(core_t::TTime bucketLength) {
    double learnRate{static_cast<double>(bucketLength) / 1800.0};
    double minimumSeasonalVarianceScale{0.4};
    return maths::CModelParams{bucketLength,
                               learnRate,
                               0.0,
                               minimumSeasonalVarianceScale,
                               6 * core::constants::HOUR,
                               24 * core::constants::HOUR};
}

std::size_t dimension(double) {
    return 1;
}
std::size_t dimension(const TDoubleVec& sample) {
    return sample.size();
}

TTimeDouble2VecSizeTr sample(core_t::TTime time, double sample) {
    return core::make_triple(time, TDouble2Vec{sample}, std::size_t{0});
}
TTimeDouble2VecSizeTr sample(core_t::TTime time, const TDoubleVec& sample) {
    return core::make_triple(time, TDouble2Vec(sample), std::size_t{0});
}

template<typename SAMPLES>
core_t::TTime
addSamples(core_t::TTime bucketLength, const SAMPLES& samples, maths::CModel& model) {
    TDouble2VecWeightsAryVec weights{
        maths_t::CUnitWeights::unit<TDouble2Vec>(dimension(samples[0]))};
    maths::CModelAddSamplesParams params;
    params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
    core_t::TTime time{0};
    for (const auto& sample_ : samples) {
        model.addSamples(params, {sample(time, sample_)});
        time += bucketLength;
    }
    return time;
}

void computeProbability(core_t::TTime time,
                        maths_t::EProbabilityCalculation calculation,
                        const TDouble2Vec& sample,
                        const maths::CModel& model,
                        double& probability,
                        TTail2Vec& tail) {
    TDouble2Vec varianceScale;
    model.seasonalWeight(0.0, time, varianceScale);
    maths_t::TDouble2VecWeightsAry weight(
        maths_t::CUnitWeights::unit<TDouble2Vec>(sample.size()));
    maths_t::setSeasonalVarianceScale(varianceScale, weight);
    maths::CModelProbabilityParams params;
    params.addCalculation(calculation).addWeights(weight);
    maths::SModelProbabilityResult result;
    model.probability(params, {{time}}, {sample}, result);
    probability = result.s_Probability;
    tail = std::move(result.s_Tail);
}

const std::string I("I");
const std::string i1("i1");
const std::string i2("i2");
const std::string i3("i3");
const std::string EMPTY_STRING;

template<typename CALCULATOR>
void computeInfluences(CALCULATOR& calculator,
                       model_t::EFeature feature,
                       const maths::CModel& model,
                       core_t::TTime time,
                       double value,
                       double count,
                       double probability,
                       const TTail2Vec& tail,
                       const std::string& influencerName,
                       const TStrCRefDouble1VecDoublePrPrVec& influencerValues,
                       TStoredStringPtrStoredStringPtrPrDoublePrVec& result) {
    TDouble2Vec varianceScale;
    model.seasonalWeight(0.0, time, varianceScale);
    maths_t::TDouble2VecWeightsAry weight(maths_t::CUnitWeights::unit<TDouble2Vec>(1));
    maths_t::setSeasonalVarianceScale(varianceScale, weight);
    model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model::CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
    params.s_Feature = feature;
    params.s_Model = &model;
    params.s_Time = TTime2Vec1Vec{TTimeVec{time}};
    params.s_Value = TDouble2Vec1Vec{TDoubleVec{value}};
    params.s_Count = count;
    params.s_ComputeProbabilityParams.addWeights(weight);
    params.s_Probability = probability;
    params.s_Tail = tail;
    params.s_InfluencerName = model::CStringStore::influencers().get(influencerName);
    params.s_InfluencerValues = influencerValues;
    params.s_Cutoff = 0.5;
    calculator.computeInfluences(params);
    result.swap(params.s_Influences);
}

template<typename CALCULATOR>
void computeInfluences(CALCULATOR& calculator,
                       model_t::EFeature feature,
                       const maths::CModel& model,
                       const core_t::TTime (&times)[2],
                       const double (&values)[2],
                       const double (&counts)[2],
                       double probability,
                       const TTail2Vec& tail,
                       const std::string& influencerName,
                       const TStrCRefDouble1VecDouble1VecPrPrVec& influencerValues,
                       TStoredStringPtrStoredStringPtrPrDoublePrVec& result) {
    model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    TTime2Vec times_(&times[0], &times[2]);
    TDouble2Vec values_(&values[0], &values[2]);
    TDouble2Vec counts_(&counts[0], &counts[2]);
    model::CProbabilityAndInfluenceCalculator::SCorrelateParams params(partitioningFields);
    params.s_Feature = feature;
    params.s_Model = &model;
    params.s_ElapsedTime = 0;
    params.s_Times.push_back(times_);
    params.s_Values.push_back(values_);
    params.s_Counts.push_back(counts_);
    params.s_ComputeProbabilityParams.addWeights(
        maths_t::CUnitWeights::unit<TDouble2Vec>(2));
    params.s_Probability = probability;
    params.s_Tail = tail;
    params.s_MostAnomalousCorrelate.push_back(0);
    params.s_InfluencerName = model::CStringStore::influencers().get(influencerName);
    params.s_InfluencerValues = influencerValues;
    params.s_Cutoff = 0.5;
    calculator.computeInfluences(params);
    result.swap(params.s_Influences);
}

void testProbabilityAndGetInfluences(model_t::EFeature feature,
                                     const maths::CModel& model,
                                     core_t::TTime time_,
                                     const TDoubleVecVec& values,
                                     const TStrCRefDouble1VecDoublePrPrVecVec& influencerValues,
                                     TStoredStringPtrStoredStringPtrPrDoublePrVec& influences) {
    model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);

    model::CProbabilityAndInfluenceCalculator calculator(0.3);
    calculator.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
    calculator.addAggregator(maths::CProbabilityOfExtremeSample());
    TInfluenceCalculatorCPtr influenceCalculator(model_t::influenceCalculator(feature));
    calculator.plugin(*influenceCalculator);

    maths::CJointProbabilityOfLessLikelySamples pJoint;
    maths::CProbabilityOfExtremeSample pExtreme;

    for (std::size_t i = 0; i < values.size(); ++i) {
        std::size_t dimension{values[i].size() - 1};
        TTime2Vec1Vec time{TTime2Vec{time_}};
        TDouble2Vec1Vec value{TDouble2Vec(&values[i][0], &values[i][dimension])};
        maths_t::TDouble2VecWeightsAry weight(
            maths_t::CUnitWeights::unit<TDouble2Vec>(dimension));
        maths_t::setSeasonalVarianceScale(TDouble2Vec(dimension, values[i][dimension]), weight);
        maths_t::setCountVarianceScale(TDouble2Vec(dimension, values[i][dimension]), weight);
        double count{0.0};
        for (const auto& influence : influencerValues[i]) {
            count += influence.second.second;
        }

        maths::CModelProbabilityParams params_;
        params_.addCalculation(model_t::probabilityCalculation(feature))
            .seasonalConfidenceInterval(0.0)
            .addWeights(weight);

        double p = 0.0;
        TTail2Vec tail;
        model_t::CResultType type;
        TSize1Vec mostAnomalousCorrelate;
        calculator.addProbability(feature, 0, model, 0 /*elapsedTime*/, params_, time,
                                  value, p, tail, type, mostAnomalousCorrelate);
        LOG_DEBUG(<< "  p = " << p);

        pJoint.add(p);
        pExtreme.add(p);
        model::CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
        params.s_Feature = feature;
        params.s_Model = &model;
        params.s_ElapsedTime = 0;
        params.s_Time = time;
        params.s_Value = value;
        params.s_Count = count;
        params.s_ComputeProbabilityParams = params_;
        params.s_Probability = p;
        params.s_Tail = tail;
        calculator.addInfluences(I, influencerValues[i], params);
    }

    double probability;
    BOOST_TEST_REQUIRE(calculator.calculate(probability, influences));

    double pj, pe;
    BOOST_TEST_REQUIRE(pJoint.calculate(pj));
    BOOST_TEST_REQUIRE(pExtreme.calculate(pe));

    LOG_DEBUG(<< "  probability = " << probability
              << ", expected probability = " << std::min(pj, pe));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(std::min(pe, pj), probability, 1e-10);
}
}

BOOST_AUTO_TEST_CASE(testInfluenceUnavailableCalculator) {
    test::CRandomNumbers rng;

    core_t::TTime bucketLength{1800};

    {
        LOG_DEBUG(<< "Test univariate");

        model::CInfluenceUnavailableCalculator calculator;
        maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
        maths::CNormalMeanPrecConjugate prior =
            maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
        maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

        TDoubleVec samples;
        rng.generateNormalSamples(10.0, 1.0, 50, samples);
        addSamples(bucketLength, samples, model);

        TStrCRefDouble1VecDoublePrPrVec influencerValues{
            {TStrCRef(i1), make_pair(11.0, 1.0)},
            {TStrCRef(i2), make_pair(11.0, 1.0)},
            {TStrCRef(i3), make_pair(15.0, 1.0)}};

        TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        computeInfluences(calculator, model_t::E_IndividualLowCountsByBucketAndPerson,
                          model, 0 /*time*/, 15.0 /*value*/, 1.0 /*count*/,
                          0.001 /*probability*/, {maths_t::E_RightTail}, I,
                          influencerValues, influences);

        LOG_DEBUG(<< "influences = " << core::CContainerPrinter::print(influences));
        BOOST_TEST_REQUIRE(influences.empty());
    }
    {
        LOG_DEBUG(<< "Test correlated");

        model::CInfluenceUnavailableCalculator calculator;

        maths::CTimeSeriesDecomposition trend{0.0, 600};
        maths::CMultivariateNormalConjugate<2> prior{
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                maths_t::E_ContinuousData, 0.0)};
        maths::CMultivariateTimeSeriesModel model{params(600), trend, prior};

        TDoubleVec samples_;
        rng.generateNormalSamples(10.0, 1.0, 50, samples_);
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            prior.addSamples({TDouble10Vec(2, samples_[i])},
                             maths_t::CUnitWeights::singleUnit<maths_t::TDouble10Vec>(2));
        }

        core_t::TTime times[] = {0, 0};
        double values[]{15.0, 15.0};
        double counts[]{1.0, 1.0};
        TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
        influencerValues.emplace_back(TStrCRef(i1), make_pair(11.0, 11.0, 1.0, 1.0));
        influencerValues.emplace_back(TStrCRef(i2), make_pair(11.0, 11.0, 1.0, 1.0));
        influencerValues.emplace_back(TStrCRef(i3), make_pair(15.0, 15.0, 1.0, 1.0));

        TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        computeInfluences(calculator, model_t::E_IndividualLowCountsByBucketAndPerson,
                          model, times, values, counts, 0.1 /*probability*/,
                          TTail2Vec(2, maths_t::E_RightTail), I,
                          influencerValues, influences);

        LOG_DEBUG(<< "influences = " << core::CContainerPrinter::print(influences));
        BOOST_TEST_REQUIRE(influences.empty());
    }
}

BOOST_AUTO_TEST_CASE(testLogProbabilityComplementInfluenceCalculator) {
    test::CRandomNumbers rng;

    model::CLogProbabilityComplementInfluenceCalculator calculator;

    core_t::TTime bucketLength{600};

    {
        LOG_DEBUG(<< "Test univariate");
        {
            LOG_DEBUG(<< "One influencer value");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            rng.generateNormalSamples(10.0, 1.0, 50, samples);
            core_t::TTime now{addSamples(bucketLength, samples, model)};

            double p;
            TTail2Vec tail;
            computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{20.0}, model, p, tail);

            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(10.0, 1.0)}};

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualCountByBucketAndPerson,
                              model, 0 /*time*/, 20.0 /*value*/, 1.0 /*count*/,
                              p, tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        {
            LOG_DEBUG(<< "No trend");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            rng.generateNormalSamples(10.0, 1.0, 50, samples);
            core_t::TTime now{addSamples(bucketLength, samples, model)};

            double p;
            TTail2Vec tail;
            computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{20.0}, model, p, tail);

            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(1.0, 1.0)},
                {TStrCRef(i2), make_pair(1.0, 1.0)},
                {TStrCRef(i3), make_pair(18.0, 1.0)}};

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualCountByBucketAndPerson,
                              model, 0 /*time*/, 20.0 /*value*/, 1.0 /*count*/,
                              p, tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i3), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        {
            LOG_DEBUG(<< "Trend");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            {
                rng.generateNormalSamples(0.0, 100.0, 10 * 86400 / 600, samples);
                core_t::TTime time{0};
                for (auto& sample : samples) {
                    sample += 100.0 + 100.0 * std::sin(2.0 * 3.1416 *
                                                       static_cast<double>(time) / 86400.0);
                    time += bucketLength;
                }
            }
            addSamples(bucketLength, samples, model);

            TTimeVec testTimes{0, 86400 / 4, 86400 / 2, (3 * 86400) / 4};
            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(70.0, 1.0)},
                {TStrCRef(i2), make_pair(50.0, 1.0)}};
            std::string expectedInfluencerValues[]{"i1", "i2"};
            TDoubleVecVec expectedInfluences{{1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}, {0.8, 0.6}};

            for (std::size_t i = 0; i < testTimes.size(); ++i) {
                core_t::TTime time = testTimes[i];
                LOG_DEBUG(<< "  time = " << time);
                LOG_DEBUG(<< "  baseline = " << model.predict(time));

                double p;
                TTail2Vec tail;
                computeProbability(time, maths_t::E_TwoSided,
                                   TDouble2Vec{120.0}, model, p, tail);
                LOG_DEBUG(<< "  p = " << p << ", tail = " << tail);

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator, model_t::E_IndividualCountByBucketAndPerson,
                                  model, time, 120.0 /*value*/, 1.0 /*count*/,
                                  p, tail, I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                for (std::size_t j = 0; j < influences.size(); ++j) {
                    BOOST_REQUIRE_EQUAL(expectedInfluencerValues[j],
                                        *influences[j].first.second);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedInfluences[i][j],
                                                 influences[j].second, 0.06);
                }
            }
        }
    }
    {
        LOG_DEBUG(<< "Test correlated");

        double counts[]{1.0, 1.0};

        {
            LOG_DEBUG(<< "One influencer value");

            maths::CTimeSeriesDecomposition trend{0.0, 600};
            maths::CMultivariateNormalConjugate<2> prior{
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData, 0.0)};
            maths::CMultivariateTimeSeriesModel model{params(600), trend, prior};

            TDoubleVec mean(2, 10.0);
            TDoubleVecVec covariances(2, TDoubleVec(2));
            covariances[0][0] = covariances[1][1] = 5.0;
            covariances[0][1] = covariances[1][0] = 4.0;
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(mean, covariances, 50, samples_);
            for (std::size_t i = 0; i < samples_.size(); ++i) {
                prior.addSamples({TDouble10Vec(samples_[i])},
                                 maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
            }

            core_t::TTime times[]{0, 0};
            double values[]{15.0, 15.0};
            double lb, ub;
            TTail10Vec tail;
            prior.probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, {TDouble10Vec(&values[0], &values[2])},
                maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
            influencerValues.push_back({TStrCRef(i1), make_pair(15.0, 15.0, 1.0, 1.0)});

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualCountByBucketAndPerson,
                              model, times, values, counts, 0.5 * (lb + ub),
                              tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        /*{
            LOG_DEBUG(<< "No trend");

            maths::CTimeSeriesDecomposition trend{0.0, 600};
            maths::CMultivariateNormalConjugate<2> prior{
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData, 0.0)};
            maths::CMultivariateTimeSeriesModel model{params(600), trend, prior};

            TDoubleVec mean(2, 10.0);
            TDoubleVecVec covariances(2, TDoubleVec(2));
            covariances[0][0] = covariances[1][1] = 5.0;
            covariances[0][1] = covariances[1][0] = 4.0;
            TDoubleVecVec samples_;
            rng.generateMultivariateNormalSamples(mean, covariances, 50, samples_);
            for (std::size_t i = 0; i < samples_.size(); ++i) {
                prior.addSamples({TDouble10Vec(samples_[i])},
                                 maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
            }

            core_t::TTime times[]{0, 0};
            double values[]{20.0, 10.0};
            double lb, ub;
            TTail10Vec tail;
            prior.probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, {TDouble10Vec(&values[0], &values[2])},
                maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
            influencerValues.push_back({TStrCRef(i1), make_pair(1.0, 1.0, 1.0, 1.0)});
            influencerValues.push_back({TStrCRef(i2), make_pair(1.0, 1.0, 1.0, 1.0)});
            influencerValues.push_back({TStrCRef(i3), make_pair(18.0, 8.0, 1.0, 1.0)});

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualCountByBucketAndPerson,
                              model, times, values, counts, 0.5 * (lb + ub),
                              tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i3), 1)]"),
                                 core::CContainerPrinter::print(influences));
        }
        {
            LOG_DEBUG(<< "Trend");

            TDecompositionPtrVec trend;
            trend.push_back(TDecompositionPtr(new maths::CTimeSeriesDecomposition));
            trend.push_back(TDecompositionPtr(new maths::CTimeSeriesDecomposition));
            maths::CMultivariateNormalConjugateFactory::TPriorPtr prior =
                    maths::CMultivariateNormalConjugateFactory::nonInformative(2, maths_t::E_ContinuousData, 0.0);
            {
                TDoubleVec mean(2, 0.0);
                TDoubleVecVec covariances(2, TDoubleVec(2));
                covariances[0][0] = covariances[1][1] = 100.0;
                covariances[0][1] = covariances[1][0] =  80.0;
                TDoubleVecVec samples;
                rng.generateMultivariateNormalSamples(mean, covariances, 10 * 86400 / 600, samples);
                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
                for (core_t::TTime time = 0, i = 0; time < 10 * 86400; time += 600, ++i)
                {
                    double y = 100.0 + 100.0 * std::sin(2.0 * 3.1416 * static_cast<double>(time) / 86400.0);
                    trend[0]->addPoint(time, y + samples[i][0]);
                    trend[1]->addPoint(time, y + samples[i][0]);
                    prior->addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, TDouble10Vec(samples[i])), weight);
                }
            }

            core_t::TTime testTimes[] = {0, 86400 / 4, 86400 / 2, (3 * 86400) / 4};

            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(70.0, 70.0, 1.0, 1.0)));
            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair(50.0, 50.0, 1.0, 1.0)));

            std::string expectedInfluencerValues[] = {"i1", "i2" };
            double expectedInfluences[][2] =
                {
                    {1.0, 1.0},
                    {0.0, 0.0},
                    {1.0, 1.0},
                    {0.8, 0.65}
                };

            for (std::size_t i = 0; i < boost::size(testTimes); ++i)
            {
                core_t::TTime time = testTimes[i];
                LOG_DEBUG(<< "  time = " << time);
                LOG_DEBUG(<< "  baseline[0] = " << core::CContainerPrinter::print(trend[0]->baseline(time, 0.0)));
                LOG_DEBUG(<< "  baseline[1] = " << core::CContainerPrinter::print(trend[1]->baseline(time, 0.0)));

                core_t::TTime times[] = {time, time };
                double values[] = {120.0, 120.0};
                double detrended[] =
                    {
                        trend[0]->detrend(time, values[0], 0.0),
                        trend[1]->detrend(time, values[1], 0.0)
                    };
                double vs[] =
                    {
                        trend[0]->scale(time, prior->marginalLikelihoodVariances()[0], 0.0).second,
                        trend[1]->scale(time, prior->marginalLikelihoodVariances()[1], 0.0).second
                    };
                LOG_DEBUG(<< "  detrended = " << core::CContainerPrinter::print(detrended)
                          << ", vs = " << core::CContainerPrinter::print(vs));
                TSize10Vec coordinates(std::size_t(1), 0);
                TDouble10Vec2Vec lbs, ubs;
                TTail10Vec tail;
                TDouble10Vec1Vec sample(1, TDouble10Vec(&detrended[0], &detrended[2]));
                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(&vs[0], &vs[2])));
                prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyle,
                                                      sample,
                                                      weight,
                                                      coordinates,
                                                      lbs, ubs, tail);
                double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
                double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
                LOG_DEBUG(<< "  p = " << 0.5*(lb+ub) << ", tail = " << tail);

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                TDecompositionCPtr1Vec trends;
                for (TDecompositionPtrVecCItr itr = trend.begin(); itr != trend.end(); ++itr)
                {
                    trends.push_back(itr->get());
                }
                computeInfluences(calculator,
                                  model_t::E_IndividualCountByBucketAndPerson,
                                  trends, *prior,
                                  times, values, weight, counts,
                                  0.5*(lb+ub), tail, coordinates[0], 0.0confidence,
                                  I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                for (std::size_t j = 0; j < influences.size(); ++j)
                {
                    BOOST_REQUIRE_EQUAL(expectedInfluencerValues[j], *influences[j].first.second);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedInfluences[i][j], influences[j].second, 0.05);
                }
            }
        }*/
    }
}

BOOST_AUTO_TEST_CASE(testMeanInfluenceCalculator) {
    test::CRandomNumbers rng;

    model::CMeanInfluenceCalculator calculator;

    core_t::TTime bucketLength{600};

    {
        LOG_DEBUG(<< "Test univariate");
        {
            LOG_DEBUG(<< "One influencer value");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            rng.generateNormalSamples(10.0, 1.0, 50, samples);
            core_t::TTime now{addSamples(bucketLength, samples, model)};

            double p;
            TTail2Vec tail;
            computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{20.0}, model, p, tail);

            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(5.0, 1.0)}};

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualMeanByPerson,
                              model, 0 /*time*/, 5.0 /*value*/, 1.0 /*count*/,
                              p, tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        {
            LOG_DEBUG(<< "No trend");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            rng.generateNormalSamples(10.2, 1.0, 50, samples);
            core_t::TTime now{addSamples(bucketLength, samples, model)};

            {
                LOG_DEBUG(<< "Right tail, one clear influence");

                double p;
                TTail2Vec tail;
                computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{12.5},
                                   model, p, tail);

                TStrCRefDouble1VecDoublePrPrVec influencerValues{
                    {TStrCRef(i1), make_pair(20.0, 5.0)},
                    {TStrCRef(i2), make_pair(10.0, 7.0)},
                    {TStrCRef(i3), make_pair(10.0, 8.0)}};

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator, model_t::E_IndividualMeanByPerson,
                                  model, 0 /*time*/, 12.5 /*value*/, 20.0 /*count*/,
                                  p, tail, I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                    core::CContainerPrinter::print(influences));
            }
            {
                LOG_DEBUG(<< "Right tail, no clear influences");

                double p;
                TTail2Vec tail;
                computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{15.0},
                                   model, p, tail);

                TStrCRefDouble1VecDoublePrPrVec influencerValues{
                    {TStrCRef(i1), make_pair(15.0, 5.0)},
                    {TStrCRef(i2), make_pair(15.0, 6.0)}};

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator, model_t::E_IndividualMeanByPerson,
                                  model, 0 /*time*/, 15.0 /*value*/, 11.0 /*count*/,
                                  p, tail, I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_TEST_REQUIRE(influences.empty());
            }
            {
                LOG_DEBUG(<< "Left tail, no clear influences");

                double p;
                TTail2Vec tail;
                computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{5.0},
                                   model, p, tail);

                TStrCRefDouble1VecDoublePrPrVec influencerValues{
                    {TStrCRef(i1), make_pair(5.0, 5.0)},
                    {TStrCRef(i2), make_pair(5.0, 6.0)}};

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator, model_t::E_IndividualMeanByPerson,
                                  model, 0 /*time*/, 5.0 /*value*/, 11.0 /*count*/,
                                  p, tail, I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_TEST_REQUIRE(influences.empty());
            }
            {
                LOG_DEBUG(<< "Left tail, two influences");

                double p;
                TTail2Vec tail;
                computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{8.0},
                                   model, p, tail);

                TStrCRefDouble1VecDoublePrPrVec influencerValues{
                    {TStrCRef(i1), make_pair(5.0, 9.0)},
                    {TStrCRef(i2), make_pair(11.0, 20.0)},
                    {TStrCRef(i3), make_pair(5.0, 11.0)}};

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator, model_t::E_IndividualMeanByPerson,
                                  model, 0 /*time*/, 8.0 /*value*/, 40.0 /*count*/,
                                  p, tail, I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_REQUIRE_EQUAL(std::size_t(2), influences.size());
                BOOST_REQUIRE_EQUAL(i3, *influences[0].first.second);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(0.7, influences[0].second, 0.04);
                BOOST_REQUIRE_EQUAL(i1, *influences[1].first.second);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6, influences[1].second, 0.03);
            }
        }
    }
    {
        LOG_DEBUG(<< "Test correlated");

        core_t::TTime times[]{0, 0};

        {
            LOG_DEBUG(<< "One influencer value");

            maths::CTimeSeriesDecomposition trend{0.0, 600};
            maths::CMultivariateNormalConjugate<2> prior{
                maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                    maths_t::E_ContinuousData, 0.0)};
            maths::CMultivariateTimeSeriesModel model{params(600), trend, prior};

            {
                TDoubleVec mean(2, 10.0);
                TDoubleVecVec covariances(2, TDoubleVec(2));
                covariances[0][0] = covariances[1][1] = 5.0;
                covariances[0][1] = covariances[1][0] = 4.0;
                TDoubleVecVec samples_;
                rng.generateMultivariateNormalSamples(mean, covariances, 50, samples_);
                for (std::size_t i = 0; i < samples_.size(); ++i) {
                    prior.addSamples({TDouble10Vec(samples_[i])},
                                     maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2));
                }
            }

            double values[]{5.0, 5.0};
            double counts[]{1.0, 1.0};
            double lb, ub;
            TTail10Vec tail;
            prior.probabilityOfLessLikelySamples(
                maths_t::E_TwoSided, {TDouble10Vec(&values[0], &values[2])},
                maths_t::CUnitWeights::singleUnit<TDouble10Vec>(2), lb, ub, tail);
            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(
                TStrCRef(i1), make_pair(5.0, 5.0, 1.0, 1.0)));

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualMeanByPerson,
                              model, times, values, counts, 0.5 * (lb + ub),
                              tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        /*{
            LOG_DEBUG(<< "No trend");

            maths::CMultivariateNormalConjugateFactory::TPriorPtr prior =
                    maths::CMultivariateNormalConjugateFactory::nonInformative(2, maths_t::E_ContinuousData, 0.0);
            {
                TDoubleVec mean(2, 10.0);
                TDoubleVecVec covariances(2, TDoubleVec(2));
                covariances[0][0] = covariances[1][1] = 5.0;
                covariances[0][1] = covariances[1][0] = 4.0;
                TDoubleVecVec samples_;
                rng.generateMultivariateNormalSamples(mean, covariances, 50, samples_);
                TDouble10Vec1Vec samples;
                for (std::size_t i = 0; i < samples_.size(); ++i)
                {
                    samples.push_back(samples_[i]);
                }
                TDouble10Vec4Vec1Vec weights(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
                prior->addSamples(COUNT_WEIGHT, samples, weights);
            }
            {
                LOG_DEBUG(<< "Right tail, one clear influence");

                double values[] = {10.0, 12.15};
                double counts[] = {20.0, 10.0};
                TSize10Vec coordinates(std::size_t(1), 1);
                TDouble10Vec2Vec lbs, ubs;
                TTail10Vec tail;
                TDouble10Vec1Vec sample(1, TDouble10Vec(&values[0], &values[2]));
                TDouble10Vec4Vec1Vec weights(1, TDouble10Vec4Vec(2, TDouble10Vec(2, 1.0)));
                prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyles,
                                                      sample,
                                                      weights,
                                                      coordinates,
                                                      lbs, ubs, tail);
                double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
                double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
                TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(10.0, 20.0, 5.0, 2.5)));
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair( 9.0,  9.0, 7.0, 3.5)));
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i3), make_pair(10.0, 10.0, 8.0, 4.0)));

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator,
                                  model_t::E_IndividualMeanByPerson, TDecompositionCPtr1Vec(), *prior,
                                  times, values, weights, counts,
                                  0.5*(lb+ub), tail, coordinates[0],
                                  I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                     core::CContainerPrinter::print(influences));
            }
            {
                LOG_DEBUG(<< "Right tail, no clear influences");

                double values[] = {11.0, 15.0};
                double counts[] = { 4.0, 11.0};
                TSize10Vec coordinates(std::size_t(1), 1);
                TDouble10Vec2Vec lbs, ubs;
                TTail10Vec tail;
                TDouble10Vec1Vec sample(1, TDouble10Vec(&values[0], &values[2]));
                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(2, TDouble10Vec(2, 1.0)));
                prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyles,
                                                      sample,
                                                      weight,
                                                      coordinates,
                                                      lbs, ubs, tail);
                double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
                double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
                TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(10.0, 15.0, 2.0, 5.0)));
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair(12.0, 15.0, 2.0, 6.0)));

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator,
                                  model_t::E_IndividualMeanByPerson, TDecompositionCPtr1Vec(), *prior,
                                  times, values, weight, counts,
                                  0.5*(lb+ub), tail, coordinates[0],
                                  I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_TEST_REQUIRE(influences.empty());
            }
            {
                LOG_DEBUG(<< "Left tail, no clear influences");

                double values[] = { 5.0,  5.0};
                double counts[] = {11.0, 11.0};
                TSize10Vec coordinates(std::size_t(1), 0);
                TDouble10Vec2Vec lbs, ubs;
                TTail10Vec tail;
                TDouble10Vec1Vec sample(1, TDouble10Vec(&values[0], &values[2]));
                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(2, TDouble10Vec(2, 1.0)));
                prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyles,
                                                      sample,
                                                      weight,
                                                      coordinates,
                                                      lbs, ubs, tail);
                double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
                double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
                TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(5.0, 5.0, 5.0, 5.0)));
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair(5.0, 5.0, 6.0, 5.0)));

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator,
                                  model_t::E_IndividualMeanByPerson, TDecompositionCPtr1Vec(), *prior,
                                  times, values, weight, counts,
                                  0.5*(lb+ub), tail, coordinates[0],
                                  I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_TEST_REQUIRE(influences.empty());
            }
            {
                LOG_DEBUG(<< "Left tail, two influences");

                double values[] = { 8.0, 10.0};
                double counts[] = {40.0, 12.0};
                TSize10Vec coordinates(std::size_t(1), 0);
                TDouble10Vec2Vec lbs, ubs;
                TTail10Vec tail;
                TDouble10Vec1Vec sample(1, TDouble10Vec(&values[0], &values[2]));
                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(2, TDouble10Vec(2, 1.0)));
                prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
                                                      weightStyles,
                                                      sample,
                                                      weight,
                                                      coordinates,
                                                      lbs, ubs, tail);
                double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
                double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
                TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair( 4.5, 10.0,  9.0, 4.0)));
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair(11.5, 11.0, 20.0, 4.0)));
                influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i3), make_pair( 4.5,  9.0, 11.0, 4.0)));

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator,
                                  model_t::E_IndividualMeanByPerson, TDecompositionCPtr1Vec(), *prior,
                                  times, values, weight, counts,
                                  0.5*(lb+ub), tail, coordinates[0],
                                  I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                BOOST_REQUIRE_EQUAL(std::size_t(2), influences.size());
                BOOST_REQUIRE_EQUAL(i1, *influences[0].first.second);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6, influences[0].second, 0.04);
                BOOST_REQUIRE_EQUAL(i3, *influences[1].first.second);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6, influences[1].second, 0.08);
            }
        }*/
    }
}

BOOST_AUTO_TEST_CASE(testLogProbabilityInfluenceCalculator) {
    test::CRandomNumbers rng;

    model::CLogProbabilityInfluenceCalculator calculator;

    core_t::TTime bucketLength{600};

    {
        LOG_DEBUG(<< "Test univariate");
        {
            LOG_DEBUG(<< "One influencer value");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            rng.generateNormalSamples(10.0, 1.0, 50, samples);
            core_t::TTime now{addSamples(bucketLength, samples, model)};

            double p;
            TTail2Vec tail;
            computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{5.0}, model, p, tail);

            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(5.0, 1.0)}};

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualUniqueCountByBucketAndPerson,
                              model, now /*time*/, 5.0 /*value*/, 1.0 /*count*/,
                              p, tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        {
            LOG_DEBUG(<< "No trend");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            rng.generateNormalSamples(10.0, 1.0, 50, samples);
            core_t::TTime now{addSamples(bucketLength, samples, model)};

            double p;
            TTail2Vec tail;
            computeProbability(now, maths_t::E_TwoSided, TDouble1Vec{6.0}, model, p, tail);

            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(9.0, 1.0)},
                {TStrCRef(i2), make_pair(6.0, 1.0)},
                {TStrCRef(i3), make_pair(6.0, 1.0)}};

            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            computeInfluences(calculator, model_t::E_IndividualUniqueCountByBucketAndPerson,
                              model, now /*time*/, 6.0 /*value*/, 1.0 /*count*/,
                              p, tail, I, influencerValues, influences);

            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::string("[((I, i2), 1), ((I, i3), 1)]"),
                                core::CContainerPrinter::print(influences));
        }
        {
            LOG_DEBUG(<< "Trend");

            maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
            maths::CNormalMeanPrecConjugate prior =
                maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            maths::CUnivariateTimeSeriesModel model(params(bucketLength), 0, trend, prior);

            TDoubleVec samples;
            {
                rng.generateNormalSamples(0.0, 100.0, 10 * 86400 / 600, samples);
                core_t::TTime time{0};
                for (auto& sample : samples) {
                    sample += 100.0 + 100.0 * std::sin(2.0 * 3.1416 *
                                                       static_cast<double>(time) / 86400.0);
                    time += bucketLength;
                }
            }

            addSamples(bucketLength, samples, model);

            TTimeVec testTimes{0, 86400 / 4, 86400 / 2, (3 * 86400) / 4};
            TStrCRefDouble1VecDoublePrPrVec influencerValues{
                {TStrCRef(i1), make_pair(60.0, 1.0)},
                {TStrCRef(i2), make_pair(50.0, 1.0)}};
            std::string expectedInfluencerValues[] = {"i1", "i2"};
            TDoubleVecVec expectedInfluences{{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 0.7}};

            for (std::size_t i = 0; i < testTimes.size(); ++i) {
                core_t::TTime time = testTimes[i];
                LOG_DEBUG(<< "  time = " << time);

                double p;
                TTail2Vec tail;
                computeProbability(time, maths_t::E_TwoSided, TDouble2Vec{60.0},
                                   model, p, tail);
                LOG_DEBUG(<< "  p = " << p << ", tail = " << tail);

                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
                computeInfluences(calculator, model_t::E_IndividualHighUniqueCountByBucketAndPerson,
                                  model, time, 60.0 /*value*/, 1.0 /*count*/, p,
                                  tail, I, influencerValues, influences);

                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
                std::sort(influences.begin(), influences.end(),
                          maths::COrderings::SFirstLess());
                for (std::size_t j = 0; j < influences.size(); ++j) {
                    BOOST_REQUIRE_EQUAL(expectedInfluencerValues[j],
                                        *influences[j].first.second);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedInfluences[i][j],
                                                 influences[j].second, 0.03);
                }
            }
        }
    }
    //    {
    //        LOG_DEBUG(<< "Test correlated");
    //
    //        double counts[] = {1.0, 1.0};
    //
    //        {
    //            LOG_DEBUG(<< "One influencer value");
    //
    //            maths::CMultivariateNormalConjugateFactory::TPriorPtr prior =
    //                    maths::CMultivariateNormalConjugateFactory::nonInformative(2, maths_t::E_ContinuousData, 0.0);
    //            {
    //                TDoubleVec mean(2, 10.0);
    //                TDoubleVecVec covariances(2, TDoubleVec(2));
    //                covariances[0][0] = covariances[1][1] = 1.0;
    //                covariances[0][1] = covariances[1][0] = 0.9;
    //                TDoubleVecVec samples_;
    //                rng.generateMultivariateNormalSamples(mean, covariances, 50, samples_);
    //                TDouble10Vec1Vec samples;
    //                for (std::size_t i = 0; i < samples_.size(); ++i)
    //                {
    //                    samples.push_back(samples_[i]);
    //                }
    //                TDouble10Vec4Vec1Vec weights(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
    //                prior->addSamples(COUNT_WEIGHT, samples, weights);
    //            }
    //
    //            core_t::TTime times[] = {0, 0};
    //            double values[] = {5.0, 5.0};
    //            double lb, ub;
    //            TTail10Vec tail;
    //            TDouble10Vec1Vec sample(1, TDouble10Vec(&values[0], &values[2]));
    //            TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(2, TDouble10Vec(2, 1.0)));
    //            prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
    //                                                  weightStyle,
    //                                                  sample,
    //                                                  weight,
    //                                                  lb, ub, tail);
    //            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
    //            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(5.0, 10.0, 1.0, 1.0)));
    //
    //            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
    //            computeInfluences(calculator,
    //                              model_t::E_IndividualUniqueCountByBucketAndPerson, TDecompositionCPtr1Vec(), *prior,
    //                              times, values, weight, counts,
    //                              0.5*(lb+ub), tail, 0, 0.0/*confidence*/,
    //                              I, influencerValues, influences);
    //
    //            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
    //            BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1)]"),
    //                                 core::CContainerPrinter::print(influences));
    //        }
    //        {
    //            LOG_DEBUG(<< "No trend");
    //
    //            maths::CMultivariateNormalConjugateFactory::TPriorPtr prior =
    //                    maths::CMultivariateNormalConjugateFactory::nonInformative(2, maths_t::E_ContinuousData, 0.0);
    //
    //            {
    //                TDoubleVec mean(2, 10.0);
    //                TDoubleVecVec covariances(2, TDoubleVec(2));
    //                covariances[0][0] = covariances[1][1] = 1.0;
    //                covariances[0][1] = covariances[1][0] = -0.9;
    //                TDoubleVecVec samples_;
    //                rng.generateMultivariateNormalSamples(mean, covariances, 50, samples_);
    //                TDouble10Vec1Vec samples;
    //                for (std::size_t i = 0; i < samples_.size(); ++i)
    //                {
    //                    samples.push_back(samples_[i]);
    //                }
    //                TDouble10Vec4Vec1Vec weights(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
    //                prior->addSamples(COUNT_WEIGHT, samples, weights);
    //            }
    //
    //            core_t::TTime times[] = {0, 0};
    //            double values[] = {10.0, 6.0};
    //            TSize10Vec coordinates(std::size_t(1), 1);
    //            TDouble10Vec2Vec lbs, ubs;
    //            TTail10Vec tail;
    //            TDouble10Vec1Vec sample(1, TDouble10Vec(&values[0], &values[2]));
    //            TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(2, TDouble10Vec(2, 1.0)));
    //            prior->probabilityOfLessLikelySamples(maths_t::E_TwoSided,
    //                                                  weightStyle,
    //                                                  sample,
    //                                                  weight,
    //                                                  coordinates,
    //                                                  lbs, ubs, tail);
    //            double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
    //            double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
    //            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
    //            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(11.0, 9.0, 1.0, 1.0)));
    //            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair(10.0, 6.0, 1.0, 1.0)));
    //            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i3), make_pair( 9.0, 6.0, 1.0, 1.0)));
    //
    //            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
    //            computeInfluences(calculator,
    //                              model_t::E_IndividualUniqueCountByBucketAndPerson, TDecompositionCPtr1Vec(), *prior,
    //                              times, values, weight, counts,
    //                              0.5*(lb+ub), tail, coordinates[0], 0.0/*confidence*/,
    //                              I, influencerValues, influences);
    //
    //            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
    //            BOOST_REQUIRE_EQUAL(std::string("[((I, i2), 1), ((I, i3), 1)]"),
    //                                 core::CContainerPrinter::print(influences));
    //        }
    //        {
    //            LOG_DEBUG(<< "Trend");
    //
    //            TDecompositionPtrVec trend;
    //            trend.push_back(TDecompositionPtr(new maths::CTimeSeriesDecomposition));
    //            trend.push_back(TDecompositionPtr(new maths::CTimeSeriesDecomposition));
    //            maths::CMultivariateNormalConjugateFactory::TPriorPtr prior =
    //                    maths::CMultivariateNormalConjugateFactory::nonInformative(2, maths_t::E_ContinuousData, 0.0);
    //            {
    //                TDoubleVec mean(2, 0.0);
    //                TDoubleVecVec covariances(2, TDoubleVec(2));
    //                covariances[0][0] = covariances[1][1] = 100.0;
    //                covariances[0][1] = covariances[1][0] =  80.0;
    //                TDoubleVecVec samples;
    //                rng.generateMultivariateNormalSamples(mean, covariances, 10 * 86400 / 600, samples);
    //                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(2, 1.0)));
    //                for (core_t::TTime time = 0, i = 0; time < 10 * 86400; time += 600, ++i)
    //                {
    //                    double y[] =
    //                        {
    //                            200.0 + 200.0 * std::sin(2.0 * 3.1416 * static_cast<double>(time) / 86400.0),
    //                            100.0 + 100.0 * std::sin(2.0 * 3.1416 * static_cast<double>(time) / 86400.0)
    //                        };
    //                    trend[0]->addPoint(time, y[0] + samples[i][0]);
    //                    trend[1]->addPoint(time, y[1] + samples[i][1]);
    //                    prior->addSamples(COUNT_WEIGHT, TDouble10Vec1Vec(1, TDouble10Vec(samples[i])), weight);
    //                }
    //            }
    //
    //            core_t::TTime testTimes[] = {0, 86400 / 4, 86400 / 2, (3 * 86400) / 4};
    //
    //            TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
    //            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i1), make_pair(60.0, 60.0, 1.0, 1.0)));
    //            influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(TStrCRef(i2), make_pair(50.0, 50.0, 1.0, 1.0)));
    //
    //            std::string expectedInfluencerValues[] = {"i1", "i2" };
    //            double expectedInfluences[][2] =
    //                {
    //                    {1.0, 1.0},
    //                    {1.0, 1.0},
    //                    {1.0, 1.0},
    //                    {1.0, 0.85}
    //                };
    //
    //            for (std::size_t i = 0; i < boost::size(testTimes); ++i)
    //            {
    //                core_t::TTime time = testTimes[i];
    //                LOG_DEBUG(<< "  time = " << time);
    //                LOG_DEBUG(<< "  baseline[0] = " << core::CContainerPrinter::print(trend[0]->baseline(time, 0.0)));
    //                LOG_DEBUG(<< "  baseline[1] = " << core::CContainerPrinter::print(trend[1]->baseline(time, 0.0)));
    //
    //                core_t::TTime times[] = {time, time };
    //                double values[] = {120.0, 60.0};
    //                double detrended[] =
    //                    {
    //                        trend[0]->detrend(time, values[0], 0.0),
    //                        trend[1]->detrend(time, values[1], 0.0)
    //                    };
    //                double vs[] =
    //                    {
    //                        trend[0]->scale(time, prior->marginalLikelihoodVariances()[0], 0.0).second,
    //                        trend[1]->scale(time, prior->marginalLikelihoodVariances()[1], 0.0).second
    //                    };
    //                LOG_DEBUG(<< "  detrended = " << core::CContainerPrinter::print(detrended)
    //                          << ", vs = " << core::CContainerPrinter::print(vs));
    //                TSize10Vec coordinates(std::size_t(1), i % 2);
    //                TDouble10Vec2Vec lbs, ubs;
    //                TTail10Vec tail;
    //                TDouble10Vec1Vec sample(1, TDouble10Vec(&detrended[0], &detrended[2]));
    //                TDouble10Vec4Vec1Vec weight(1, TDouble10Vec4Vec(1, TDouble10Vec(&vs[0], &vs[2])));
    //                prior->probabilityOfLessLikelySamples(maths_t::E_OneSidedAbove,
    //                                                      weightStyle,
    //                                                      sample,
    //                                                      weight,
    //                                                      coordinates,
    //                                                      lbs, ubs, tail);
    //                double lb = std::sqrt(lbs[0][0] * lbs[1][0]);
    //                double ub = std::sqrt(ubs[0][0] * ubs[1][0]);
    //                LOG_DEBUG(<< "  p = " << 0.5*(lb+ub) << ", tail = " << tail);
    //
    //                TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
    //                TDecompositionCPtr1Vec trends;
    //                for (TDecompositionPtrVecCItr itr = trend.begin(); itr != trend.end(); ++itr)
    //                {
    //                    trends.push_back(itr->get());
    //                }
    //                computeInfluences(calculator,
    //                                  model_t::E_IndividualHighUniqueCountByBucketAndPerson,
    //                                  trends, *prior,
    //                                  times, values, weight, counts,
    //                                  0.5*(lb+ub), tail, coordinates[0], 0.0/*confidence*/,
    //                                  I, influencerValues, influences);
    //
    //                LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
    //                for (std::size_t j = 0; j < influences.size(); ++j)
    //                {
    //                    BOOST_REQUIRE_EQUAL(expectedInfluencerValues[j],
    //                                         *influences[j].first.second);
    //                    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedInfluences[i][j], influences[j].second, 0.04);
    //                }
    //            }
    //        }
    //    }
}

BOOST_AUTO_TEST_CASE(testIndicatorInfluenceCalculator) {
    {
        LOG_DEBUG(<< "Test univariate");

        model::CIndicatorInfluenceCalculator calculator;

        maths::CTimeSeriesDecomposition trend{0.0, 600};
        maths::CNormalMeanPrecConjugate prior =
            maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
        maths::CUnivariateTimeSeriesModel model(params(600), 0, trend, prior);

        TStrCRefDouble1VecDoublePrPrVec influencerValues{
            {TStrCRef(i1), make_pair(1.0, 1.0)},
            {TStrCRef(i2), make_pair(1.0, 1.0)},
            {TStrCRef(i3), make_pair(1.0, 1.0)}};

        TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        computeInfluences(calculator, model_t::E_IndividualIndicatorOfBucketPerson,
                          model, 0 /*time*/, 1.0 /*value*/, 1.0 /*count*/, 0.1 /*probability*/,
                          {maths_t::E_RightTail}, I, influencerValues, influences);

        LOG_DEBUG(<< "influences = " << core::CContainerPrinter::print(influences));
        BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1), ((I, i2), 1), ((I, i3), 1)]"),
                            core::CContainerPrinter::print(influences));
    }
    {
        LOG_DEBUG(<< "Test correlated");

        model::CIndicatorInfluenceCalculator calculator;

        maths::CTimeSeriesDecomposition trend{0.0, 600};
        maths::CMultivariateNormalConjugate<2> prior{
            maths::CMultivariateNormalConjugate<2>::nonInformativePrior(
                maths_t::E_ContinuousData, 0.0)};
        maths::CMultivariateTimeSeriesModel model{params(600), trend, prior};

        core_t::TTime times[]{0, 0};
        double values[]{1.0, 1.0};
        double counts[]{1.0, 1.0};
        TStrCRefDouble1VecDouble1VecPrPrVec influencerValues;
        influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(
            TStrCRef(i1), make_pair(1.0, 1.0, 1.0, 1.0)));
        influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(
            TStrCRef(i2), make_pair(1.0, 1.0, 1.0, 1.0)));
        influencerValues.push_back(TStrCRefDouble1VecDouble1VecPrPr(
            TStrCRef(i3), make_pair(1.0, 1.0, 1.0, 1.0)));

        TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        computeInfluences(calculator, model_t::E_IndividualIndicatorOfBucketPerson,
                          model, times, values, counts, 0.1 /*probability*/,
                          TTail2Vec(2, maths_t::E_RightTail), I,
                          influencerValues, influences);

        LOG_DEBUG(<< "influences = " << core::CContainerPrinter::print(influences));
        BOOST_REQUIRE_EQUAL(std::string("[((I, i1), 1), ((I, i2), 1), ((I, i3), 1)]"),
                            core::CContainerPrinter::print(influences));
    }
}

BOOST_AUTO_TEST_CASE(testProbabilityAndInfluenceCalculator) {
    test::CRandomNumbers rng;

    core_t::TTime bucketLength{600};

    maths::CTimeSeriesDecomposition trend{0.0, bucketLength};
    maths::CNormalMeanPrecConjugate prior =
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
    maths::CMultivariateNormalConjugate<2> multivariatePrior =
        maths::CMultivariateNormalConjugate<2>::nonInformativePrior(maths_t::E_ContinuousData);
    maths::CUnivariateTimeSeriesModel univariateModel(
        params(bucketLength), 0, trend, prior, nullptr, nullptr, false);
    maths::CMultivariateTimeSeriesModel multivariateModel(
        params(bucketLength), trend, multivariatePrior, nullptr, nullptr, false);

    TDoubleVec samples;
    rng.generateNormalSamples(10.0, 1.0, 50, samples);
    addSamples(bucketLength, samples, univariateModel);

    TDoubleVec mean{10.0, 15.0};
    TDoubleVecVec covariances{{1.0, 0.5}, {0.5, 2.0}};
    TDoubleVecVec multivariateSamples;
    rng.generateMultivariateNormalSamples(mean, covariances, 50, multivariateSamples);
    core_t::TTime now{addSamples(bucketLength, multivariateSamples, multivariateModel)};

    model_t::TFeatureVec features{model_t::E_IndividualSumByBucketAndPerson,
                                  model_t::E_IndividualMeanLatLongByPerson};
    const maths::CModel* models[]{&univariateModel, &multivariateModel};

    model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);

    {
        LOG_DEBUG(<< "error case");

        model::CProbabilityAndInfluenceCalculator calculator(0.5);
        calculator.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
        calculator.addAggregator(maths::CProbabilityOfExtremeSample());

        TDoubleVecVec values{{12.0, 1.0},       {15.0, 1.0},
                             {7.0, 1.5},        {9.0, 1.0},
                             {17.0, 2.0},       {12.0, 17.0, 1.0},
                             {15.0, 20.0, 1.0}, {7.0, 12.0, 1.5},
                             {15.0, 10.0, 1.0}, {17.0, 22.0, 2.0}};
        TStrCRefDouble1VecDoublePrPrVec influencerValues{
            {TStrCRef(i2), make_pair(12.0, 1.0)},
            {TStrCRef(i1), make_pair(15.0, 1.0)},
            {TStrCRef(i2), make_pair(7.0, 1.0)},
            {TStrCRef(i2), make_pair(9.0, 1.0)},
            {TStrCRef(i1), make_pair(17.0, 1.0)},
            {TStrCRef(i2), make_pair(12.0, 17.0, 1.0)},
            {TStrCRef(i1), make_pair(15.0, 20.0, 1.0)},
            {TStrCRef(i2), make_pair(7.0, 12.0, 1.0)},
            {TStrCRef(i2), make_pair(9.0, 14.0, 1.0)},
            {TStrCRef(i1), make_pair(17.0, 22.0, 1.0)}};

        maths::CJointProbabilityOfLessLikelySamples pJoint;
        maths::CProbabilityOfExtremeSample pExtreme;

        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < features.size(); ++j) {
                TDouble2Vec1Vec value{TDouble2Vec(&values[i + 5 * j][0],
                                                  &values[i + 5 * j][1 + j])};
                maths_t::TDouble2VecWeightsAry weights(
                    maths_t::CUnitWeights::unit<TDouble2Vec>(1 + j));
                maths_t::setSeasonalVarianceScale(
                    TDouble2Vec(1 + j, values[i + 5 * j][1 + j]), weights);
                maths::CModelProbabilityParams params_;
                params_.addCalculation(maths_t::E_TwoSided)
                    .seasonalConfidenceInterval(0.0)
                    .addWeights(weights);
                double p;
                TTail2Vec tail;
                model_t::CResultType type;
                TSize1Vec mostAnomalousCorrelate;
                calculator.addProbability(features[j], 0, *models[j], 0 /*elapsedTime*/,
                                          params_, TTime2Vec1Vec{TTime2Vec{now}}, value,
                                          p, tail, type, mostAnomalousCorrelate);
                pJoint.add(p);
                pExtreme.add(p);
                model::CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
                params.s_Feature = features[j];
                params.s_Model = models[j];
                params.s_ElapsedTime = 0;
                params.s_Time = TTime2Vec1Vec{TTime2Vec{now}};
                params.s_Value = value;
                params.s_Count = 1.0;
                params.s_ComputeProbabilityParams = params_;
                params.s_Probability = p;
                params.s_Tail = tail;
                calculator.addInfluences(
                    I, TStrCRefDouble1VecDoublePrPrVec{influencerValues[i]}, params);
            }
        }

        calculator.addProbability(0.02);
        pJoint.add(0.02);
        pExtreme.add(0.02);

        double probability;
        TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
        BOOST_TEST_REQUIRE(calculator.calculate(probability, influences));

        double pj, pe;
        BOOST_TEST_REQUIRE(pJoint.calculate(pj));
        BOOST_TEST_REQUIRE(pExtreme.calculate(pe));

        LOG_DEBUG(<< "  probability = " << probability
                  << ", expected probability = " << std::min(pj, pe));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(std::min(pe, pj), probability, 1e-10);
    }
    {
        LOG_DEBUG(<< "influencing joint probability");

        TDoubleVecVec values[]{
            {{12.0, 1.0}, {15.0, 1.0}, {7.0, 1.5}, {9.0, 1.0}, {17.0, 2.0}},
            {{12.0, 17.0, 1.0}, {15.0, 20.0, 1.0}, {7.0, 12.0, 1.5}, {9.0, 14.0, 1.0}, {17.0, 22.0, 2.0}}};
        TStrCRefDouble1VecDoublePrPrVecVec influencerValues[]{
            {{{TStrCRef(i2), make_pair(12.0, 1.0)}},
             {{TStrCRef(i1), make_pair(15.0, 1.0)}},
             {{TStrCRef(i2), make_pair(7.0, 1.5)}},
             {{TStrCRef(i2), make_pair(9.0, 1.0)}},
             {{TStrCRef(i1), make_pair(17.0, 2.0)}}},
            {{{TStrCRef(i2), make_pair(12.0, 17.0, 1.0)}},
             {{TStrCRef(i1), make_pair(15.0, 20.0, 1.0)}},
             {{TStrCRef(i2), make_pair(7.0, 12.0, 1.5)}},
             {{TStrCRef(i2), make_pair(9.0, 14.0, 1.0)}},
             {{TStrCRef(i1), make_pair(17.0, 22.0, 2.0)}}}};
        for (std::size_t i = 0; i < features.size(); ++i) {
            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            testProbabilityAndGetInfluences(features[i], *models[i], now, values[i],
                                            influencerValues[i], influences);
            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::size_t(1), influences.size());
            BOOST_REQUIRE_EQUAL(i1, *influences[0].first.second);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.95, influences[0].second, 0.06);
        }
    }
    {
        LOG_DEBUG(<< "influencing extreme probability");

        TDoubleVecVec values[]{
            {{11.0, 1.0}, {10.5, 1.0}, {8.5, 1.5}, {10.8, 1.5}, {19.0, 1.0}},
            {{11.0, 16.0, 1.0}, {10.5, 15.5, 1.0}, {8.5, 13.5, 1.5}, {10.8, 15.8, 1.5}, {19.0, 24.0, 1.0}}};
        TStrCRefDouble1VecDoublePrPrVecVec influencerValues[]{
            {{{TStrCRef(i1), make_pair(11.0, 1.0)}},
             {{TStrCRef(i1), make_pair(10.5, 1.0)}},
             {{TStrCRef(i1), make_pair(8.5, 1.0)}},
             {{TStrCRef(i1), make_pair(10.8, 1.0)}},
             {{TStrCRef(i2), make_pair(19.0, 1.0)}}},
            {{{TStrCRef(i1), make_pair(11.0, 16.0, 1.0)}},
             {{TStrCRef(i1), make_pair(10.5, 15.5, 1.0)}},
             {{TStrCRef(i1), make_pair(8.5, 13.5, 1.0)}},
             {{TStrCRef(i1), make_pair(10.8, 15.8, 1.0)}},
             {{TStrCRef(i2), make_pair(19.0, 24.0, 1.0)}}}};

        for (std::size_t i = 0; i < features.size(); ++i) {
            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            testProbabilityAndGetInfluences(features[i], *models[i], now, values[i],
                                            influencerValues[i], influences);
            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::size_t(1), influences.size());
            BOOST_REQUIRE_EQUAL(i2, *influences[0].first.second);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, influences[0].second, 0.03);
        }
    }
    {
        LOG_DEBUG(<< "marginal influence");

        TDoubleVecVec values[]{
            {{11.0, 1.0}, {10.5, 1.0}, {8.0, 1.0}, {10.8, 1.0}, {14.0, 1.0}},
            {{11.0, 16.0, 1.0}, {10.5, 15.5, 1.0}, {8.0, 13.0, 1.0}, {10.8, 15.8, 1.0}, {14.0, 19.0, 1.0}}};
        TStrCRefDouble1VecDoublePrPrVecVec influencerValues[]{
            {{{TStrCRef(i1), make_pair(12.0, 1.0)}, {TStrCRef(i2), make_pair(10.0, 1.0)}},
             {{TStrCRef(i1), make_pair(10.5, 1.0)}, {TStrCRef(i2), make_pair(10.5, 1.0)}},
             {{TStrCRef(i1), make_pair(9.0, 1.0)}, {TStrCRef(i2), make_pair(7.0, 1.0)}},
             {{TStrCRef(i1), make_pair(11.0, 1.0)}, {TStrCRef(i2), make_pair(10.6, 1.0)}},
             {{TStrCRef(i1), make_pair(16.0, 1.0)}, {TStrCRef(i2), make_pair(12.0, 1.0)}}},
            {{{TStrCRef(i1), make_pair(12.0, 17.0, 1.0)},
              {TStrCRef(i2), make_pair(10.0, 15.0, 1.0)}},
             {{TStrCRef(i1), make_pair(10.5, 15.5, 1.0)},
              {TStrCRef(i2), make_pair(10.5, 15.5, 1.0)}},
             {{TStrCRef(i1), make_pair(9.0, 14.0, 1.0)},
              {TStrCRef(i2), make_pair(7.0, 12.0, 1.0)}},
             {{TStrCRef(i1), make_pair(11.0, 16.0, 1.0)},
              {TStrCRef(i2), make_pair(10.6, 15.6, 1.0)}},
             {{TStrCRef(i1), make_pair(16.0, 21.0, 1.0)},
              {TStrCRef(i2), make_pair(12.0, 17.0, 1.0)}}}};

        {
            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            testProbabilityAndGetInfluences(model_t::E_IndividualMeanByPerson,
                                            univariateModel, now, values[0],
                                            influencerValues[0], influences);
            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::size_t(1), influences.size());
            BOOST_REQUIRE_EQUAL(i1, *influences[0].first.second);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.75, influences[0].second, 0.05);
        }
        {
            TStoredStringPtrStoredStringPtrPrDoublePrVec influences;
            testProbabilityAndGetInfluences(model_t::E_IndividualMeanLatLongByPerson,
                                            multivariateModel, now, values[1],
                                            influencerValues[1], influences);
            LOG_DEBUG(<< "  influences = " << core::CContainerPrinter::print(influences));
            BOOST_REQUIRE_EQUAL(std::size_t(2), influences.size());
            BOOST_REQUIRE_EQUAL(i2, *influences[0].first.second);
            BOOST_REQUIRE_EQUAL(i1, *influences[1].first.second);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, influences[0].second, 1e-3);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, influences[1].second, 1e-3);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
