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
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CPrior.h>
#include <maths/CSampling.h>

#include <model/CAnnotatedProbability.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CCountingModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CMetricModel.h>
#include <model/CMetricModelFactory.h>
#include <model/CModelDetailsView.h>
#include <model/CModelFactory.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>
#include <model/CRuleCondition.h>
#include <model/ModelTypes.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "CModelTestFixtureBase.h"

#include <boost/optional/optional_io.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CMetricModelTest)

using namespace ml;
using namespace model;

namespace {

using TMinAccumulator = maths::CBasicStatistics::SMin<double>::TAccumulator;
using TMaxAccumulator = maths::CBasicStatistics::SMax<double>::TAccumulator;

const CModelTestFixtureBase::TSizeDoublePr1Vec NO_CORRELATES;
}

class CTestFixture : public CModelTestFixtureBase {
public:
    TDouble1Vec featureData(const CMetricModel& model,
                            model_t::EFeature feature,
                            std::size_t pid,
                            core_t::TTime time) {
        const CMetricModel::TFeatureData* data = model.featureData(feature, pid, time);
        if (!data) {
            return TDouble1Vec();
        }
        return data->s_BucketValue ? data->s_BucketValue->value() : TDouble1Vec();
    }

    void makeModel(const SModelParams& params,
                   const model_t::TFeatureVec& features,
                   core_t::TTime startTime,
                   TOptionalUInt sampleCount = TOptionalUInt()) {
        this->makeModelT<CMetricModelFactory>(params, features, startTime,
                                              model_t::E_MetricOnline,
                                              m_Gatherer, m_Model, sampleCount);
    }
};

BOOST_FIXTURE_TEST_CASE(testSample, CTestFixture) {
    core_t::TTime startTime{45};
    core_t::TTime bucketLength{5};
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    params.s_MaximumUpdatesPerBucket = 0.0;

    // Check basic sampling.
    {
        TTimeDoublePrVec data{{49, 1.5},  {60, 1.3},  {61, 1.3},   {62, 1.6},
                              {65, 1.7},  {66, 1.33}, {68, 1.5},   {84, 1.58},
                              {87, 1.69}, {157, 1.6}, {164, 1.66}, {199, 1.28},
                              {202, 1.2}, {204, 1.5}};

        TUIntVec sampleCounts{2, 1};
        TUIntVec expectedSampleCounts{2, 1};
        std::size_t i{0};
        for (auto& sampleCount : sampleCounts) {
            model_t::TFeatureVec features{model_t::E_IndividualMeanByPerson,
                                          model_t::E_IndividualMinByPerson,
                                          model_t::E_IndividualMaxByPerson};

            this->makeModel(params, features, startTime, sampleCount);
            CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
            BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

            // Bucket values.
            uint64_t expectedCount{0};
            TMeanAccumulator baselineMeanError;
            TMeanAccumulator expectedMean;
            TMeanAccumulator expectedBaselineMean;
            TMinAccumulator expectedMin;
            TMaxAccumulator expectedMax;

            // Sampled values.
            TMeanAccumulator expectedSampleTime;
            TMeanAccumulator expectedMeanSample;
            TMinAccumulator expectedMinSample;
            TMaxAccumulator expectedMaxSample;
            TDouble1Vec expectedSampleTimes;
            TDouble1Vec expectedMeanSamples;
            TDouble1Vec expectedMinSamples;
            TDouble1Vec expectedMaxSamples;
            std::size_t numberSamples = 0;

            TMathsModelPtr expectedMeanModel = m_Factory->defaultFeatureModel(
                model_t::E_IndividualMeanByPerson, bucketLength, 0.4, true);
            TMathsModelPtr expectedMinModel = m_Factory->defaultFeatureModel(
                model_t::E_IndividualMinByPerson, bucketLength, 0.4, true);
            TMathsModelPtr expectedMaxModel = m_Factory->defaultFeatureModel(
                model_t::E_IndividualMaxByPerson, bucketLength, 0.4, true);

            std::size_t j{0};
            core_t::TTime time{startTime};
            for (;;) {
                if (j < data.size() && data[j].first < time + bucketLength) {
                    LOG_DEBUG(<< "Adding " << data[j].second << " at "
                              << data[j].first);

                    this->addArrival(SMessage(data[j].first, "p", data[j].second), m_Gatherer);

                    ++expectedCount;
                    expectedMean.add(data[j].second);
                    expectedMin.add(data[j].second);
                    expectedMax.add(data[j].second);

                    expectedSampleTime.add(static_cast<double>(data[j].first));
                    expectedMeanSample.add(data[j].second);
                    expectedMinSample.add(data[j].second);
                    expectedMaxSample.add(data[j].second);

                    ++j;

                    if (j % expectedSampleCounts[i] == 0) {
                        ++numberSamples;
                        expectedSampleTimes.push_back(
                            maths::CBasicStatistics::mean(expectedSampleTime));
                        expectedMeanSamples.push_back(
                            maths::CBasicStatistics::mean(expectedMeanSample));
                        expectedMinSamples.push_back(expectedMinSample[0]);
                        expectedMaxSamples.push_back(expectedMaxSample[0]);
                        expectedSampleTime = TMeanAccumulator();
                        expectedMeanSample = TMeanAccumulator();
                        expectedMinSample = TMinAccumulator();
                        expectedMaxSample = TMaxAccumulator();
                    }
                } else {
                    LOG_DEBUG(<< "Sampling [" << time << ", " << time + bucketLength << ")");

                    model.sample(time, time + bucketLength, m_ResourceMonitor);
                    if (maths::CBasicStatistics::count(expectedMean) > 0.0) {
                        expectedBaselineMean.add(maths::CBasicStatistics::mean(expectedMean));
                    }
                    if (numberSamples > 0) {
                        LOG_DEBUG(<< "Adding mean samples = "
                                  << core::CContainerPrinter::print(expectedMeanSamples) << ", min samples = "
                                  << core::CContainerPrinter::print(expectedMinSamples) << ", max samples = "
                                  << core::CContainerPrinter::print(expectedMaxSamples));

                        maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec weights(
                            numberSamples, maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                        maths::CModelAddSamplesParams params_;
                        params_.integer(false)
                            .nonNegative(true)
                            .propagationInterval(1.0)
                            .trendWeights(weights)
                            .priorWeights(weights);

                        maths::CModel::TTimeDouble2VecSizeTrVec expectedMeanSamples_;
                        maths::CModel::TTimeDouble2VecSizeTrVec expectedMinSamples_;
                        maths::CModel::TTimeDouble2VecSizeTrVec expectedMaxSamples_;
                        for (std::size_t k = 0; k < numberSamples; ++k) {
                            // We round to the nearest integer time (note this has to match
                            // the behaviour of CMetricPartialStatistic::time).
                            core_t::TTime sampleTime{static_cast<core_t::TTime>(
                                expectedSampleTimes[k] + 0.5)};
                            expectedMeanSamples_.emplace_back(
                                sampleTime, TDouble2Vec{expectedMeanSamples[k]}, 0);
                            expectedMinSamples_.emplace_back(
                                sampleTime, TDouble2Vec{expectedMinSamples[k]}, 0);
                            expectedMaxSamples_.emplace_back(
                                sampleTime, TDouble2Vec{expectedMaxSamples[k]}, 0);
                        }
                        expectedMeanModel->addSamples(params_, expectedMeanSamples_);
                        expectedMinModel->addSamples(params_, expectedMinSamples_);
                        expectedMaxModel->addSamples(params_, expectedMaxSamples_);
                        numberSamples = 0;
                        expectedSampleTimes.clear();
                        expectedMeanSamples.clear();
                        expectedMinSamples.clear();
                        expectedMaxSamples.clear();
                    }

                    model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                              model_t::CResultType::E_Final);
                    TOptionalUInt64 currentCount = model.currentBucketCount(0, time);
                    TDouble1Vec bucketMean = model.currentBucketValue(
                        model_t::E_IndividualMeanByPerson, 0, 0, time);
                    TDouble1Vec baselineMean = model.baselineBucketMean(
                        model_t::E_IndividualMeanByPerson, 0, 0, type, NO_CORRELATES, time);

                    LOG_DEBUG(<< "bucket count = "
                              << core::CContainerPrinter::print(currentCount));
                    LOG_DEBUG(<< "current bucket mean = "
                              << core::CContainerPrinter::print(bucketMean) << ", expected baseline bucket mean = "
                              << maths::CBasicStatistics::mean(expectedBaselineMean) << ", baseline bucket mean = "
                              << core::CContainerPrinter::print(baselineMean));

                    BOOST_TEST_REQUIRE(currentCount);
                    BOOST_REQUIRE_EQUAL(expectedCount, *currentCount);

                    TDouble1Vec mean =
                        maths::CBasicStatistics::count(expectedMean) > 0.0
                            ? TDouble1Vec(1, maths::CBasicStatistics::mean(expectedMean))
                            : TDouble1Vec();
                    TDouble1Vec min = expectedMin.count() > 0
                                          ? TDouble1Vec(1, expectedMin[0])
                                          : TDouble1Vec();
                    TDouble1Vec max = expectedMax.count() > 0
                                          ? TDouble1Vec(1, expectedMax[0])
                                          : TDouble1Vec();

                    BOOST_TEST_REQUIRE(mean == bucketMean);
                    if (!baselineMean.empty()) {
                        baselineMeanError.add(std::fabs(
                            baselineMean[0] - maths::CBasicStatistics::mean(expectedBaselineMean)));
                    }

                    BOOST_TEST_REQUIRE(mean == featureData(model, model_t::E_IndividualMeanByPerson,
                                                           0, time));
                    BOOST_TEST_REQUIRE(min == featureData(model, model_t::E_IndividualMinByPerson,
                                                          0, time));
                    BOOST_TEST_REQUIRE(max == featureData(model, model_t::E_IndividualMaxByPerson,
                                                          0, time));

                    BOOST_REQUIRE_EQUAL(expectedMeanModel->checksum(),
                                        model.details()
                                            ->model(model_t::E_IndividualMeanByPerson, 0)
                                            ->checksum());
                    BOOST_REQUIRE_EQUAL(expectedMinModel->checksum(),
                                        model.details()
                                            ->model(model_t::E_IndividualMinByPerson, 0)
                                            ->checksum());
                    BOOST_REQUIRE_EQUAL(expectedMaxModel->checksum(),
                                        model.details()
                                            ->model(model_t::E_IndividualMaxByPerson, 0)
                                            ->checksum());

                    // Test persistence. (We check for idempotency.)
                    std::string origXml;
                    {
                        core::CRapidXmlStatePersistInserter inserter("root");
                        model.acceptPersistInserter(inserter);
                        inserter.toXml(origXml);
                    }

                    // Restore the XML into a new filter
                    core::CRapidXmlParser parser;
                    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
                    core::CRapidXmlStateRestoreTraverser traverser(parser);
                    CModelFactory::TModelPtr restoredModel(
                        m_Factory->makeModel(m_Gatherer, traverser));

                    // The XML representation of the new filter should be the same as the original
                    std::string newXml;
                    {
                        ml::core::CRapidXmlStatePersistInserter inserter("root");
                        restoredModel->acceptPersistInserter(inserter);
                        inserter.toXml(newXml);
                    }

                    uint64_t origChecksum = model.checksum(false);
                    LOG_DEBUG(<< "original checksum = " << origChecksum);
                    uint64_t restoredChecksum = restoredModel->checksum(false);
                    LOG_DEBUG(<< "restored checksum = " << restoredChecksum);
                    BOOST_REQUIRE_EQUAL(origChecksum, restoredChecksum);
                    BOOST_REQUIRE_EQUAL(origXml, newXml);

                    expectedCount = 0;
                    expectedMean = TMeanAccumulator();
                    expectedMin = TMinAccumulator();
                    expectedMax = TMaxAccumulator();

                    if (j >= data.size()) {
                        break;
                    }

                    time += bucketLength;
                }
            }
            LOG_DEBUG(<< "baseline mean error = "
                      << maths::CBasicStatistics::mean(baselineMeanError));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(baselineMeanError) < 0.25);

            ++i;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testMultivariateSample, CTestFixture) {
    using TVector2 = maths::CVectorNx1<double, 2>;
    using TMean2Accumulator = maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator;
    using TTimeDouble2AryPr = std::pair<core_t::TTime, std::array<double, 2>>;
    using TTimeDouble2AryPrVec = std::vector<TTimeDouble2AryPr>;

    core_t::TTime startTime(45);
    core_t::TTime bucketLength(5);
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    params.s_MaximumUpdatesPerBucket = 0.0;
    auto interimBucketCorrector = std::make_shared<model::CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector);

    TTimeDouble2AryPrVec data{
        {49, {1.5, 1.1}},  {60, {1.3, 1.2}},    {61, {1.3, 2.1}},
        {62, {1.6, 1.5}},  {65, {1.7, 1.4}},    {66, {1.33, 1.6}},
        {68, {1.5, 1.37}}, {84, {1.58, 1.42}},  {87, {1.6, 1.6}},
        {157, {1.6, 1.6}}, {164, {1.66, 1.55}}, {199, {1.28, 1.4}},
        {202, {1.3, 1.1}}, {204, {1.5, 1.8}}};

    TUIntVec sampleCounts{2u, 1u};
    TUIntVec expectedSampleCounts{2u, 1u};

    std::size_t i{0};
    for (auto& sampleCount : sampleCounts) {
        LOG_DEBUG(<< "*** sample count = " << sampleCount << " ***");

        this->makeModel(params, {model_t::E_IndividualMeanLatLongByPerson},
                        startTime, sampleCount);
        CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

        // Bucket values.
        uint64_t expectedCount{0};
        TMean2Accumulator baselineLatLongError;
        TMean2Accumulator expectedLatLong;
        TMean2Accumulator expectedBaselineLatLong;

        // Sampled values.
        TMean2Accumulator expectedLatLongSample;
        std::size_t numberSamples{0u};
        TDoubleVecVec expectedLatLongSamples;
        TMultivariatePriorPtr expectedPrior =
            factory.defaultMultivariatePrior(model_t::E_IndividualMeanLatLongByPerson);

        std::size_t j{0u};
        core_t::TTime time{startTime};
        for (;;) {
            if (j < data.size() && data[j].first < time + bucketLength) {
                LOG_DEBUG(<< "Adding " << data[j].second[0] << ","
                          << data[j].second[1] << " at " << data[j].first);

                this->addArrival(
                    SMessage(data[j].first, "p", {},
                             TDoubleDoublePr(data[j].second[0], data[j].second[1])),
                    m_Gatherer);

                ++expectedCount;
                expectedLatLong.add(TVector2(data[j].second));
                expectedLatLongSample.add(TVector2(data[j].second));

                if (++j % expectedSampleCounts[i] == 0) {
                    ++numberSamples;
                    expectedLatLongSamples.push_back(TDoubleVec(
                        maths::CBasicStatistics::mean(expectedLatLongSample).begin(),
                        maths::CBasicStatistics::mean(expectedLatLongSample).end()));
                    expectedLatLongSample = TMean2Accumulator();
                }
            } else {
                LOG_DEBUG(<< "Sampling [" << time << ", " << time + bucketLength << ")");
                model.sample(time, time + bucketLength, m_ResourceMonitor);

                if (maths::CBasicStatistics::count(expectedLatLong) > 0.0) {
                    expectedBaselineLatLong.add(maths::CBasicStatistics::mean(expectedLatLong));
                }
                if (numberSamples > 0) {
                    std::sort(expectedLatLongSamples.begin(),
                              expectedLatLongSamples.end());
                    LOG_DEBUG(<< "Adding mean samples = "
                              << core::CContainerPrinter::print(expectedLatLongSamples));
                    expectedPrior->dataType(maths_t::E_ContinuousData);
                    expectedPrior->addSamples(
                        expectedLatLongSamples,
                        maths_t::TDouble10VecWeightsAry1Vec(
                            expectedLatLongSamples.size(),
                            maths_t::CUnitWeights::unit<maths_t::TDouble10Vec>(2)));
                    expectedPrior->propagateForwardsByTime(1.0);
                    numberSamples = 0;
                    expectedLatLongSamples.clear();
                }

                model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                          model_t::CResultType::E_Final);
                TOptionalUInt64 count = model.currentBucketCount(0, time);
                TDouble1Vec bucketLatLong = model.currentBucketValue(
                    model_t::E_IndividualMeanLatLongByPerson, 0, 0, time);
                TDouble1Vec baselineLatLong =
                    model.baselineBucketMean(model_t::E_IndividualMeanLatLongByPerson,
                                             0, 0, type, NO_CORRELATES, time);
                TDouble1Vec featureLatLong = featureData(
                    model, model_t::E_IndividualMeanLatLongByPerson, 0, time);
                const auto& prior =
                    dynamic_cast<const maths::CMultivariateTimeSeriesModel*>(
                        model.details()->model(model_t::E_IndividualMeanLatLongByPerson, 0))
                        ->residualModel();

                LOG_DEBUG(<< "bucket count = " << core::CContainerPrinter::print(count));
                LOG_DEBUG(<< "current = " << core::CContainerPrinter::print(bucketLatLong)
                          << ", expected baseline = "
                          << maths::CBasicStatistics::mean(expectedBaselineLatLong) << ", actual baseline = "
                          << core::CContainerPrinter::print(baselineLatLong));

                BOOST_TEST_REQUIRE(count);
                BOOST_REQUIRE_EQUAL(expectedCount, *count);

                TDouble1Vec latLong;
                if (maths::CBasicStatistics::count(expectedLatLong) > 0.0) {
                    latLong.push_back(maths::CBasicStatistics::mean(expectedLatLong)(0));
                    latLong.push_back(maths::CBasicStatistics::mean(expectedLatLong)(1));
                }
                BOOST_REQUIRE_EQUAL(latLong, bucketLatLong);
                if (!baselineLatLong.empty()) {
                    baselineLatLongError.add(maths::fabs(
                        TVector2(baselineLatLong) -
                        maths::CBasicStatistics::mean(expectedBaselineLatLong)));
                }

                BOOST_REQUIRE_EQUAL(latLong, featureLatLong);
                BOOST_REQUIRE_EQUAL(expectedPrior->checksum(), prior.checksum());

                // Test persistence. (We check for idempotency.)
                std::string origXml;
                {
                    core::CRapidXmlStatePersistInserter inserter("root");
                    model.acceptPersistInserter(inserter);
                    inserter.toXml(origXml);
                }

                // Restore the XML into a new filter
                core::CRapidXmlParser parser;
                BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);
                CModelFactory::TModelPtr restoredModel(factory.makeModel(m_Gatherer, traverser));

                // The XML representation of the new filter should be the same as the original
                std::string newXml;
                {
                    ml::core::CRapidXmlStatePersistInserter inserter("root");
                    restoredModel->acceptPersistInserter(inserter);
                    inserter.toXml(newXml);
                }

                uint64_t origChecksum = model.checksum(false);
                LOG_DEBUG(<< "original checksum = " << origChecksum);
                uint64_t restoredChecksum = restoredModel->checksum(false);
                LOG_DEBUG(<< "restored checksum = " << restoredChecksum);
                BOOST_REQUIRE_EQUAL(origChecksum, restoredChecksum);
                BOOST_REQUIRE_EQUAL(origXml, newXml);

                expectedCount = 0;
                expectedLatLong = TMean2Accumulator();

                if (j >= data.size()) {
                    break;
                }

                time += bucketLength;
            }
        }
        LOG_DEBUG(<< "baseline mean error = "
                  << maths::CBasicStatistics::mean(baselineLatLongError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(baselineLatLongError)(0) < 0.25);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(baselineLatLongError)(1) < 0.25);

        ++i;
    }
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForMetric, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{10};

    TSizeVec bucketCounts{5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6};

    double mean{5.0};
    double variance{2.0};
    std::size_t anomalousBucket{12u};
    double anomaly{5 * std::sqrt(variance)};

    SModelParams params(bucketLength);
    model_t::TFeatureVec features{model_t::E_IndividualMeanByPerson,
                                  model_t::E_IndividualMinByPerson,
                                  model_t::E_IndividualMaxByPerson};

    this->makeModel(params, features, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> minProbabilities(2u);
    test::CRandomNumbers rng;

    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < bucketCounts.size(); ++i) {
        TDoubleVec values;
        rng.generateNormalSamples(mean, variance, bucketCounts[i], values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));
        LOG_DEBUG(<< "i = " << i << ", anomalousBucket = " << anomalousBucket
                  << ", offset = " << (i == anomalousBucket ? anomaly : 0.0));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(SMessage(time + static_cast<core_t::TTime>(j), "p",
                                      values[j] + (i == anomalousBucket ? anomaly : 0.0)),
                             m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        if (model.computeProbability(0 /*pid*/, time, time + bucketLength, partitioningFields,
                                     1, annotatedProbability) == false) {
            continue;
        }
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        if (*model.currentBucketCount(0, time) > 0) {
            minProbabilities.add(TDoubleSizePr(annotatedProbability.s_Probability, i));
        }
        time += bucketLength;
    }

    minProbabilities.sort();
    LOG_DEBUG(<< "minProbabilities = "
              << core::CContainerPrinter::print(minProbabilities.begin(),
                                                minProbabilities.end()));
    BOOST_REQUIRE_EQUAL(anomalousBucket, minProbabilities[0].second);
    BOOST_TEST_REQUIRE(minProbabilities[0].first / minProbabilities[1].first < 0.1);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForMedian, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{10};
    TSizeVec bucketCounts{5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6};
    double mean{5.0};
    double variance{2.0};
    std::size_t anomalousBucket{12u};

    SModelParams params(bucketLength);
    this->makeModel(params, {model_t::E_IndividualMedianByPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> minProbabilities(2u);
    test::CRandomNumbers rng;

    core_t::TTime time{startTime};
    for (std::size_t i = 0; i < bucketCounts.size(); ++i) {
        LOG_DEBUG(<< "i = " << i << ", anomalousBucket = " << anomalousBucket);

        TDoubleVec values;
        if (i == anomalousBucket) {
            values.push_back(0.0);
            values.push_back(mean * 3.0);
            values.push_back(mean * 3.0);
        } else {
            rng.generateNormalSamples(mean, variance, bucketCounts[i], values);
        }

        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }

        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        if (model.computeProbability(0 /*pid*/, time, time + bucketLength, partitioningFields,
                                     1, annotatedProbability) == false) {
            continue;
        }

        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        if (*model.currentBucketCount(0, time) > 0) {
            minProbabilities.add(TDoubleSizePr(annotatedProbability.s_Probability, i));
        }
        time += bucketLength;
    }

    minProbabilities.sort();
    LOG_DEBUG(<< "minProbabilities = "
              << core::CContainerPrinter::print(minProbabilities.begin(),
                                                minProbabilities.end()));
    BOOST_REQUIRE_EQUAL(anomalousBucket, minProbabilities[0].second);
    BOOST_TEST_REQUIRE(minProbabilities[0].first / minProbabilities[1].first < 0.05);

    std::size_t pid{0};
    const CMetricModel::TFeatureData* fd = model.featureData(
        ml::model_t::E_IndividualMedianByPerson, pid, time - bucketLength);

    // assert there is only 1 value in the last bucket and its the median
    BOOST_REQUIRE_EQUAL(fd->s_BucketValue->value()[0], mean * 3.0);
    BOOST_REQUIRE_EQUAL(fd->s_BucketValue->value().size(), 1);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForLowMean, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{10};
    std::size_t numberOfBuckets{100u};
    std::size_t bucketCount{5u};
    std::size_t lowMeanBucket{60u};
    std::size_t highMeanBucket{80u};
    double mean{5.0};
    double variance{0.00001};
    double lowMean{2.0};
    double highMean{10.0};

    SModelParams params(bucketLength);
    this->makeModel(params, {model_t::E_IndividualLowMeanByPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time{startTime};
    for (std::size_t i = 0; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMeanBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMeanBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        BOOST_TEST_REQUIRE(model.computeProbability(
            0 /*pid*/, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = "
              << core::CContainerPrinter::print(probabilities.begin(),
                                                probabilities.end()));

    BOOST_TEST_REQUIRE(probabilities[lowMeanBucket] < 0.01);
    BOOST_TEST_REQUIRE(probabilities[highMeanBucket] > 0.1);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForHighMean, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{10};
    std::size_t numberOfBuckets{100u};
    std::size_t bucketCount{5u};
    std::size_t lowMeanBucket{60u};
    std::size_t highMeanBucket{80u};
    double mean{5.0};
    double variance{0.00001};
    double lowMean{2.0};
    double highMean{10.0};

    SModelParams params(bucketLength);
    this->makeModel(params, {model_t::E_IndividualHighMeanByPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time{startTime};
    for (std::size_t i = 0; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMeanBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMeanBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        BOOST_TEST_REQUIRE(model.computeProbability(
            0 /*pid*/, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));

    BOOST_TEST_REQUIRE(probabilities[lowMeanBucket] > 0.1);
    BOOST_TEST_REQUIRE(probabilities[highMeanBucket] < 0.01);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForLowSum, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{10};
    std::size_t numberOfBuckets{100u};
    std::size_t bucketCount{5u};
    std::size_t lowSumBucket{60u};
    std::size_t highSumBucket{80u};
    double mean{50.0};
    double variance{5.0};
    double lowMean{5.0};
    double highMean{95.0};

    SModelParams params(bucketLength);
    this->makeModel(params, {model_t::E_IndividualLowSumByBucketAndPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time{startTime};
    for (std::size_t i = 0; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowSumBucket) {
            meanForBucket = lowMean;
        }
        if (i == highSumBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        BOOST_TEST_REQUIRE(model.computeProbability(
            0 /*pid*/, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
    BOOST_TEST_REQUIRE(probabilities[lowSumBucket] < 0.01);
    BOOST_TEST_REQUIRE(probabilities[highSumBucket] > 0.1);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForHighSum, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{10};
    std::size_t numberOfBuckets{100};
    std::size_t bucketCount{5u};
    std::size_t lowSumBucket{60u};
    std::size_t highSumBucket{80u};
    double mean{50.0};
    double variance{5.0};
    double lowMean{5.0};
    double highMean{95.0};

    SModelParams params(bucketLength);
    this->makeModel(params, {model_t::E_IndividualHighSumByBucketAndPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time{startTime};
    for (std::size_t i = 0; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowSumBucket) {
            meanForBucket = lowMean;
        }
        if (i == highSumBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        BOOST_TEST_REQUIRE(model.computeProbability(
            0 /*pid*/, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
    BOOST_TEST_REQUIRE(probabilities[lowSumBucket] > 0.1);
    BOOST_TEST_REQUIRE(probabilities[highSumBucket] < 0.01);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForLatLong,
                        CTestFixture,
                        *boost::unit_test::disabled()) {
    // TODO
}

BOOST_FIXTURE_TEST_CASE(testInfluence, CTestFixture) {
    using TStrDoubleDoubleTr = core::CTriple<std::string, double, double>;
    using TStrDoubleDoubleTrVec = std::vector<TStrDoubleDoubleTr>;
    using TStrDoubleDoubleTrVecVec = std::vector<TStrDoubleDoubleTrVec>;

    LOG_DEBUG(<< "Test min and max influence");

    for (auto feature : {model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson}) {
        core_t::TTime startTime{0};
        core_t::TTime bucketLength{10};
        std::size_t numberOfBuckets{50u};
        std::size_t bucketCount{5u};
        double mean{5.0};
        double variance{1.0};
        std::string influencer{"I"};
        TStrVec influencerValues{"i1", "i2", "i3", "i4", "i5"};

        SModelParams params(bucketLength);
        auto interimBucketCorrector =
            std::make_shared<model::CInterimBucketCorrector>(bucketLength);
        CMetricModelFactory factory(params, interimBucketCorrector);
        factory.features({feature});
        factory.bucketLength(bucketLength);
        factory.fieldNames("", "", "P", "V", TStrVec{"I"});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer));
        CModelFactory::TModelPtr model_(factory.makeModel(gatherer));
        BOOST_TEST_REQUIRE(model_);
        BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, model_->category());
        CMetricModel& model = static_cast<CMetricModel&>(*model_.get());

        test::CRandomNumbers rng;
        core_t::TTime time{startTime};
        for (std::size_t i = 0; i < numberOfBuckets; ++i, time += bucketLength) {
            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, bucketCount, samples);

            maths::CBasicStatistics::SMin<TDoubleStrPr>::TAccumulator min;
            maths::CBasicStatistics::SMax<TDoubleStrPr>::TAccumulator max;
            for (std::size_t j = 0; j < samples.size(); ++j) {
                this->addArrival(SMessage(time, "p", samples[j], {},
                                          TOptionalStr(influencerValues[j])),
                                 gatherer);
                min.add(TDoubleStrPr(samples[j], influencerValues[j]));
                max.add(TDoubleStrPr(samples[j], influencerValues[j]));
            }

            model.sample(time, time + bucketLength, m_ResourceMonitor);

            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            SAnnotatedProbability annotatedProbability;
            model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                     partitioningFields, 1, annotatedProbability);

            LOG_DEBUG(<< "influences = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            if (!annotatedProbability.s_Influences.empty()) {
                std::size_t j = 0;
                for (/**/; j < annotatedProbability.s_Influences.size(); ++j) {
                    if (feature == model_t::E_IndividualMinByPerson &&
                        *annotatedProbability.s_Influences[j].first.second ==
                            min[0].second &&
                        std::fabs(annotatedProbability.s_Influences[j].second - 1.0) < 1e-10) {
                        break;
                    }
                    if (feature == model_t::E_IndividualMaxByPerson &&
                        *annotatedProbability.s_Influences[j].first.second ==
                            max[0].second &&
                        std::fabs(annotatedProbability.s_Influences[j].second - 1.0) < 1e-10) {
                        break;
                    }
                }
                BOOST_TEST_REQUIRE(j < annotatedProbability.s_Influences.size());
            }
        }
    }

    auto testFeature = [this](model_t::EFeature feature, const TDoubleVecVec& values,
                              const TStrVecVec& influencers,
                              const TStrDoubleDoubleTrVecVec& influences) {
        core_t::TTime startTime{0};
        core_t::TTime bucketLength{10};

        SModelParams params(bucketLength);
        auto interimBucketCorrector =
            std::make_shared<model::CInterimBucketCorrector>(bucketLength);
        CMetricModelFactory factory(params, interimBucketCorrector);
        factory.features({feature});
        factory.bucketLength(bucketLength);
        factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer));
        CModelFactory::TModelPtr model_(factory.makeModel(gatherer));
        BOOST_TEST_REQUIRE(model_);
        BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, model_->category());
        CMetricModel& model = static_cast<CMetricModel&>(*model_.get());

        SAnnotatedProbability annotatedProbability;

        core_t::TTime time{startTime};
        for (std::size_t i = 0; i < values.size(); ++i) {
            this->processBucket(time, bucketLength, values[i], influencers[i],
                                gatherer, model, annotatedProbability);
            BOOST_REQUIRE_EQUAL(influences[i].size(),
                                annotatedProbability.s_Influences.size());
            if (influences[i].size() > 0) {
                for (const auto& expected : influences[i]) {
                    bool found{false};
                    for (const auto& actual : annotatedProbability.s_Influences) {
                        if (expected.first == *actual.first.second) {
                            BOOST_TEST_REQUIRE(actual.second >= expected.second);
                            BOOST_TEST_REQUIRE(actual.second <= expected.third);
                            found = true;
                            break;
                        }
                    }
                    BOOST_TEST_REQUIRE(found);
                }
            }
            time += bucketLength;
        }
    };

    LOG_DEBUG(<< "Test mean");

    {
        TDoubleVecVec values{{1.0, 2.3, 2.1},
                             {8.0},
                             {4.3, 5.2, 3.4},
                             {3.2, 3.9},
                             {20.1, 2.8, 3.9},
                             {12.1, 4.2, 5.7, 3.2},
                             {0.1, 0.3, 5.4},
                             {40.5, 7.3},
                             {6.4, 7.0, 7.1, 6.6, 7.1, 6.7},
                             {0.3}};
        TStrVecVec influencers{{"i1", "i1", "i2"},
                               {"i1"},
                               {"i1", "i1", "i1"},
                               {"i3", "i3"},
                               {"i2", "i1", "i1"},
                               {"i1", "i2", "i2", "i2"},
                               {"i1", "i1", "i3"},
                               {"i1", "i2"},
                               {"i1", "i2", "i3", "i4", "i5", "i6"},
                               {"i2"}};
        TStrDoubleDoubleTrVecVec influences{
            {},
            {},
            {},
            {},
            {},
            {},
            {core::make_triple(std::string{"i1"}, 0.9, 1.0)},
            {core::make_triple(std::string{"i1"}, 0.8, 0.9)},
            {},
            {core::make_triple(std::string{"i2"}, 1.0, 1.0)}};
        testFeature(model_t::E_IndividualMeanByPerson, values, influencers, influences);
    }

    LOG_DEBUG(<< "Test sum");
    {
        TDoubleVecVec values{{1.0, 2.3, 2.1, 5.9},
                             {10.0},
                             {4.3, 5.2, 3.4, 6.2, 7.8},
                             {3.2, 3.9},
                             {20.1, 2.8, 3.9},
                             {12.1, 4.2, 5.7, 3.2},
                             {0.1, 0.3, 5.4},
                             {48.1, 10.1},
                             {6.8, 7.2, 7.3, 6.8, 7.3, 6.9},
                             {0.4}};
        TStrVecVec influencers{{"i1", "i1", "i2", "i2"},
                               {"i1"},
                               {"i1", "i1", "i1", "i1", "i3"},
                               {"i3", "i3"},
                               {"i2", "i1", "i1"},
                               {"i1", "i2", "i2", "i2"},
                               {"i1", "i1", "i3"},
                               {"i1", "i2"},
                               {"i1", "i2", "i3", "i4", "i5", "i6"},
                               {"i2"}};
        TStrDoubleDoubleTrVecVec influences{
            {},
            {},
            {},
            {},
            {core::make_triple(std::string{"i1"}, 0.5, 0.6),
             core::make_triple(std::string{"i2"}, 0.9, 1.0)},
            {core::make_triple(std::string{"i1"}, 0.9, 1.0),
             core::make_triple(std::string{"i2"}, 0.9, 1.0)},
            {},
            {core::make_triple(std::string{"i1"}, 0.9, 1.0)},
            {},
            {core::make_triple(std::string{"i2"}, 1.0, 1.0)}};
        testFeature(model_t::E_IndividualSumByBucketAndPerson, values, influencers, influences);
    }

    LOG_DEBUG(<< "Test varp");
    {
        TDoubleVecVec values{{1.0, 2.3, 2.1, 5.9},
                             {10.0},
                             {4.3, 5.2, 3.4, 6.2, 7.8},
                             {3.2, 4.9},
                             {3.3, 3.2, 2.4, 4.2, 6.8},
                             {3.2, 5.9},
                             {20.5, 12.3},
                             {12.1, 4.2, 5.7, 3.2},
                             {0.1, 0.3, 0.2},
                             {10.1, 12.8, 3.9},
                             {7.0, 7.0, 7.1, 6.8, 37.1, 6.7},
                             {0.3}};
        TStrVecVec influencers{{"i1", "i1", "i2", "i2"},
                               {"i1"},
                               {"i1", "i1", "i1", "i1", "i3"},
                               {"i3", "i3"},
                               {"i1", "i1", "i1", "i1", "i3"},
                               {"i3", "i3"},
                               {"i1", "i2"},
                               {"i1", "i2", "i2", "i2"},
                               {"i1", "i1", "i3"},
                               {"i2", "i1", "i1"},
                               {"i1", "i2", "i3", "i4", "i5", "i6"},
                               {"i2"}};
        TStrDoubleDoubleTrVecVec influences{
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {core::make_triple(std::string{"i1"}, 0.9, 1.0),
             core::make_triple(std::string{"i3"}, 0.9, 1.0)},
            {core::make_triple(std::string{"i1"}, 0.9, 1.0)},
            {core::make_triple(std::string{"i5"}, 0.9, 1.0)},
            {}};
        testFeature(model_t::E_IndividualVarianceByPerson, values, influencers, influences);
    }
}

BOOST_FIXTURE_TEST_CASE(testLatLongInfluence, CTestFixture, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_FIXTURE_TEST_CASE(testPrune, CTestFixture) {
    maths::CSampling::CScopeMockRandomNumberGenerator scopeMockRng;

    using TEventDataVec = std::vector<CEventData>;
    using TSizeSizeMap = std::map<std::size_t, std::size_t>;

    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};

    const TStrVec people{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"};

    TSizeVecVec eventCounts;
    eventCounts.push_back(TSizeVec(1000u, 0));
    eventCounts[0][0] = 4;
    eventCounts[0][1] = 3;
    eventCounts[0][2] = 5;
    eventCounts[0][4] = 2;
    eventCounts.push_back(TSizeVec(1000u, 1));
    eventCounts.push_back(TSizeVec(1000u, 0));
    eventCounts[2][1] = 10;
    eventCounts[2][2] = 13;
    eventCounts[2][8] = 5;
    eventCounts[2][15] = 2;
    eventCounts.push_back(TSizeVec(1000u, 0));
    eventCounts[3][2] = 13;
    eventCounts[3][8] = 9;
    eventCounts[3][15] = 12;
    eventCounts.push_back(TSizeVec(1000u, 2));
    eventCounts.push_back(TSizeVec(1000u, 1));
    eventCounts.push_back(TSizeVec(1000u, 0));
    eventCounts[6][0] = 4;
    eventCounts[6][1] = 3;
    eventCounts[6][2] = 5;
    eventCounts[6][4] = 2;
    eventCounts.push_back(TSizeVec(1000u, 0));
    eventCounts[7][2] = 13;
    eventCounts[7][8] = 9;
    eventCounts[7][15] = 12;

    const TSizeVec expectedPeople{1, 4, 5};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.01;
    model_t::TFeatureVec features{model_t::E_IndividualMeanByPerson,
                                  model_t::E_IndividualMinByPerson,
                                  model_t::E_IndividualMaxByPerson};

    CModelFactory::TDataGathererPtr gatherer;
    CModelFactory::TModelPtr model_;
    this->makeModelT<CMetricModelFactory>(params, features, startTime,
                                          model_t::E_MetricOnline, gatherer, model_);
    CMetricModel* model = dynamic_cast<CMetricModel*>(model_.get());
    BOOST_TEST_REQUIRE(model);
    CModelFactory::TDataGathererPtr expectedGatherer;
    CModelFactory::TModelPtr expectedModel_;
    this->makeModelT<CMetricModelFactory>(params, features, startTime, model_t::E_MetricOnline,
                                          expectedGatherer, expectedModel_);
    CMetricModel* expectedModel = dynamic_cast<CMetricModel*>(expectedModel_.get());
    BOOST_TEST_REQUIRE(expectedModel);

    test::CRandomNumbers rng;

    TEventDataVec events;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < eventCounts.size(); ++i, bucketStart = startTime) {
        for (std::size_t j = 0; j < eventCounts[i].size(); ++j, bucketStart += bucketLength) {
            core_t::TTime n = static_cast<core_t::TTime>(eventCounts[i][j]);
            if (n > 0) {
                TDoubleVec samples;
                rng.generateUniformSamples(0.0, 5.0, static_cast<size_t>(n), samples);

                for (core_t::TTime k = 0, time = bucketStart, dt = bucketLength / n;
                     k < n; ++k, time += dt) {
                    std::size_t pid = this->addPerson(people[i], gatherer);
                    events.push_back(
                        makeEventData(time, pid, {samples[static_cast<size_t>(k)]}));
                }
            }
        }
    }
    std::sort(events.begin(), events.end(), [](const CEventData& lhs, const CEventData& rhs) {
        return lhs.time() < rhs.time();
    });

    TEventDataVec expectedEvents;
    expectedEvents.reserve(events.size());
    TSizeSizeMap mapping;
    for (const auto& expectedPerson : expectedPeople) {
        std::size_t pid = this->addPerson(people[expectedPerson], expectedGatherer);
        mapping[expectedPerson] = pid;
    }
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (std::binary_search(std::begin(expectedPeople),
                               std::end(expectedPeople), events[i].personId())) {
            expectedEvents.push_back(makeEventData(events[i].time(),
                                                   mapping[*events[i].personId()],
                                                   {events[i].values()[0][0]}));
        }
    }

    bucketStart = startTime;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].time() >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        this->addArrival(SMessage(events[i].time(),
                                  gatherer->personName(events[i].personId().get()),
                                  events[i].values()[0][0]),
                         gatherer);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune(model->defaultPruneWindow());
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    BOOST_REQUIRE_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = startTime;
    for (std::size_t i = 0; i < expectedEvents.size(); ++i) {
        if (expectedEvents[i].time() >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }

        this->addArrival(
            SMessage(expectedEvents[i].time(),
                     expectedGatherer->personName(expectedEvents[i].personId().get()),
                     expectedEvents[i].values()[0][0]),
            expectedGatherer);
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person slots.

    bucketStart = gatherer->currentBucketStartTime() + bucketLength;
    TStrVec newPersons{"p9", "p10", "p11", "p12", "13"};
    for (const auto& newPerson : newPersons) {
        std::size_t newPid = this->addPerson(newPerson, gatherer);
        BOOST_TEST_REQUIRE(newPid < 8);

        std::size_t expectedNewPid = this->addPerson(newPerson, expectedGatherer);

        this->addArrival(SMessage(bucketStart + 1, gatherer->personName(newPid), 10.0), gatherer);
        this->addArrival(SMessage(bucketStart + 2000, gatherer->personName(newPid), 15.0),
                         gatherer);
        this->addArrival(SMessage(bucketStart + 1,
                                  expectedGatherer->personName(expectedNewPid), 10.0),
                         expectedGatherer);
        this->addArrival(SMessage(bucketStart + 2000,
                                  expectedGatherer->personName(expectedNewPid), 15.0),
                         expectedGatherer);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Test that calling prune on a cloned model which has seen no new data does nothing
    CModelFactory::TModelPtr clonedModelHolder(model->cloneForPersistence());
    std::size_t numberOfPeopleBeforePrune(
        clonedModelHolder->dataGatherer().numberActivePeople());
    BOOST_TEST_REQUIRE(numberOfPeopleBeforePrune > 0);
    clonedModelHolder->prune(clonedModelHolder->defaultPruneWindow());
    BOOST_REQUIRE_EQUAL(numberOfPeopleBeforePrune,
                        clonedModelHolder->dataGatherer().numberActivePeople());
}

BOOST_FIXTURE_TEST_CASE(testKey, CTestFixture) {
    function_t::TFunctionVec countFunctions{
        function_t::E_IndividualMetric, function_t::E_IndividualMetricMean,
        function_t::E_IndividualMetricMin, function_t::E_IndividualMetricMax,
        function_t::E_IndividualMetricSum};

    std::string fieldName{"value"};
    std::string overFieldName;

    generateAndCompareKey(countFunctions, fieldName, overFieldName,
                          [](CSearchKey expectedKey, CSearchKey actualKey) {
                              BOOST_TEST_REQUIRE(expectedKey == actualKey);
                          });
}

BOOST_FIXTURE_TEST_CASE(testSkipSampling, CTestFixture) {
    core_t::TTime startTime{100};
    core_t::TTime bucketLength{100};
    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<model::CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector);

    factory.features({model_t::E_IndividualSumByBucketAndPerson});
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::TDataGathererPtr gathererNoGap(factory.makeDataGatherer(startTime));
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gathererNoGap));
    CModelFactory::TModelPtr modelNoGapPtr(factory.makeModel(gathererNoGap));
    BOOST_TEST_REQUIRE(modelNoGapPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelNoGapPtr->category());
    CMetricModel& modelNoGap = static_cast<CMetricModel&>(*modelNoGapPtr.get());

    {
        TStrVec influencerValues1{"i1"};
        TDoubleVec bucket1{1.0};
        TDoubleVec bucket2{5.0};
        TDoubleVec bucket3{10.0};

        SAnnotatedProbability annotatedProbability;

        core_t::TTime time{startTime};
        this->processBucket(time, bucketLength, bucket1, influencerValues1,
                            gathererNoGap, modelNoGap, annotatedProbability);

        time += bucketLength;
        this->processBucket(time, bucketLength, bucket2, influencerValues1,
                            gathererNoGap, modelNoGap, annotatedProbability);

        time += bucketLength;
        this->processBucket(time, bucketLength, bucket3, influencerValues1,
                            gathererNoGap, modelNoGap, annotatedProbability);
    }

    CModelFactory::TDataGathererPtr gathererWithGap(factory.makeDataGatherer(startTime));
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gathererWithGap));
    CModelFactory::TModelPtr modelWithGapPtr(factory.makeModel(gathererWithGap));
    BOOST_TEST_REQUIRE(modelWithGapPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelWithGapPtr->category());
    CMetricModel& modelWithGap = static_cast<CMetricModel&>(*modelWithGapPtr.get());
    core_t::TTime gap(bucketLength * 10);

    {
        TStrVec influencerValues1{"i1"};
        TDoubleVec bucket1{1.0};
        TDoubleVec bucket2{5.0};
        TDoubleVec bucket3{10.0};

        SAnnotatedProbability annotatedProbability;

        core_t::TTime time{startTime};
        this->processBucket(time, bucketLength, bucket1, influencerValues1,
                            gathererWithGap, modelWithGap, annotatedProbability);

        time += gap;
        modelWithGap.skipSampling(time);
        LOG_DEBUG(<< "Calling sample over skipped interval should do nothing except print some ERRORs");
        modelWithGap.sample(startTime + bucketLength, time, m_ResourceMonitor);

        this->processBucket(time, bucketLength, bucket2, influencerValues1,
                            gathererWithGap, modelWithGap, annotatedProbability);

        time += bucketLength;
        this->processBucket(time, bucketLength, bucket3, influencerValues1,
                            gathererWithGap, modelWithGap, annotatedProbability);
    }

    BOOST_REQUIRE_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))
            ->residualModel()
            .checksum());
}

BOOST_FIXTURE_TEST_CASE(testExplicitNulls, CTestFixture) {
    core_t::TTime startTime{100};
    core_t::TTime bucketLength{100};
    SModelParams params(bucketLength);
    std::string summaryCountField{"count"};
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector,
                                model_t::E_Manual, summaryCountField);

    factory.features({model_t::E_IndividualSumByBucketAndPerson});
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::TDataGathererPtr gathererSkipGap(factory.makeDataGatherer(startTime));
    CModelFactory::TModelPtr modelSkipGapPtr(factory.makeModel(gathererSkipGap));
    BOOST_TEST_REQUIRE(modelSkipGapPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelSkipGapPtr->category());
    CMetricModel& modelSkipGap = static_cast<CMetricModel&>(*modelSkipGapPtr.get());

    // The idea here is to compare a model that has a gap skipped against a model
    // that has explicit nulls for the buckets that sampling was skipped.

    // p1: |(1, 42.0)|(1, 1.0)|(1, 1.0)|X|X|(1, 42.0)|
    // p2: |(1, 42.)|(0, 0.0)|(0, 0.0)|X|X|(0, 0.0)|
    this->addArrival(SMessage(100, "p1", 42.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererSkipGap);
    this->addArrival(SMessage(100, "p2", 42.0, {}, TOptionalStr("i2"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap.sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", 1.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap.sample(200, 300, m_ResourceMonitor);
    this->addArrival(SMessage(300, "p1", 1.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap.sample(300, 400, m_ResourceMonitor);
    modelSkipGap.skipSampling(600);
    this->addArrival(SMessage(600, "p1", 42.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap.sample(600, 700, m_ResourceMonitor);

    CModelFactory::TDataGathererPtr gathererExNull(factory.makeDataGatherer(startTime));
    CModelFactory::TModelPtr modelExNullPtr(factory.makeModel(gathererExNull));
    BOOST_TEST_REQUIRE(modelExNullPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelExNullPtr->category());
    CMetricModel& modelExNullGap = static_cast<CMetricModel&>(*modelExNullPtr.get());

    // p1: |(1, 42.0), ("", 42.0), (null, 42.0)|(1, 1.0)|(1, 1.0)|(null, 100.0)|(null, 100.0)|(1, 42.0)|
    // p2: |(1, 42.0), ("", 42.0)|(0, 0.0)|(0, 0.0)|(null, 100.0)|(null, 100.0)|(0, 0.0)|
    this->addArrival(SMessage(100, "p1", 42.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p1", 42.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p1", 42.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("null")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p2", 42.0, {}, TOptionalStr("i2"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p2", 42.0, {}, TOptionalStr("i2"),
                              TOptionalStr(), TOptionalStr("")),
                     gathererExNull);
    modelExNullGap.sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", 1.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererExNull);
    modelExNullGap.sample(200, 300, m_ResourceMonitor);
    this->addArrival(SMessage(300, "p1", 1.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererExNull);
    modelExNullGap.sample(300, 400, m_ResourceMonitor);
    this->addArrival(SMessage(400, "p1", 100.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("null")),
                     gathererExNull);
    this->addArrival(SMessage(400, "p2", 100.0, {}, TOptionalStr("i2"),
                              TOptionalStr(), TOptionalStr("null")),
                     gathererExNull);
    modelExNullGap.sample(400, 500, m_ResourceMonitor);
    this->addArrival(SMessage(500, "p1", 100.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("null")),
                     gathererExNull);
    this->addArrival(SMessage(500, "p2", 100.0, {}, TOptionalStr("i2"),
                              TOptionalStr(), TOptionalStr("null")),
                     gathererExNull);
    modelExNullGap.sample(500, 600, m_ResourceMonitor);
    this->addArrival(SMessage(600, "p1", 42.0, {}, TOptionalStr("i1"),
                              TOptionalStr(), TOptionalStr("1")),
                     gathererExNull);
    modelExNullGap.sample(600, 700, m_ResourceMonitor);

    BOOST_REQUIRE_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelSkipGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelExNullGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))
            ->residualModel()
            .checksum());
}

BOOST_FIXTURE_TEST_CASE(testVarp, CTestFixture) {
    core_t::TTime startTime{500000};
    core_t::TTime bucketLength{1000};
    SModelParams params(bucketLength);

    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_IndividualVarianceByPerson});
    factory.bucketLength(bucketLength);
    factory.fieldNames("", "", "P", "V", TStrVec());
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
    BOOST_TEST_REQUIRE(!gatherer->isPopulation());
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer));
    BOOST_REQUIRE_EQUAL(1, this->addPerson("q", gatherer));
    CModelFactory::TModelPtr model_(factory.makeModel(gatherer));
    BOOST_TEST_REQUIRE(model_);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, model_->category());
    CMetricModel& model = static_cast<CMetricModel&>(*model_.get());

    TDoubleVec bucket1{1.0, 1.1};
    TDoubleVec bucket2{10.0, 10.1};
    TDoubleVec bucket3{4.3, 4.45};
    TDoubleVec bucket4{3.2, 3.303};
    TDoubleVec bucket5{20.1, 20.8, 20.9, 20.8};
    TDoubleVec bucket6{4.1, 4.2};
    TDoubleVec bucket7{0.1, 0.3, 0.2, 0.4};
    TDoubleVec bucket8{12.5, 12.3};
    TDoubleVec bucket9{6.9, 7.0, 7.1, 6.6, 7.1, 6.7};
    TDoubleVec bucket10{0.3, 0.2};
    TDoubleVec bucket11{0.0};

    SAnnotatedProbability annotatedProbability;
    SAnnotatedProbability annotatedProbability2;

    core_t::TTime time{startTime};
    this->processBucket(time, bucketLength, bucket1, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket2, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket3, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket4, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket5, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket6, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket7, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket8, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.5);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.5);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket9, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.5);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.5);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket10, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.5);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.5);

    time += bucketLength;
    this->processBucket(time, bucketLength, bucket11, gatherer, model,
                        annotatedProbability, annotatedProbability2);
    LOG_DEBUG(<< "P1 " << annotatedProbability.s_Probability << ", P2 "
              << annotatedProbability2.s_Probability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.5);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.5);
}

BOOST_FIXTURE_TEST_CASE(testInterimCorrections, CTestFixture) {
    core_t::TTime startTime{3600};
    core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_IndividualSumByBucketAndPerson});
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer));
    CModelFactory::TModelPtr model_(factory.makeModel(gatherer));
    BOOST_TEST_REQUIRE(model_);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, model_->category());
    CMetricModel& model = static_cast<CMetricModel&>(*model_.get());
    CCountingModel countingModel(params, gatherer, interimBucketCorrector);

    std::size_t pid1 = this->addPerson("p1", gatherer);
    std::size_t pid2 = this->addPerson("p2", gatherer);
    std::size_t pid3 = this->addPerson("p3", gatherer);

    core_t::TTime now = startTime;
    core_t::TTime endTime(now + 2 * 24 * bucketLength);
    test::CRandomNumbers rng;
    TDoubleVec samples(3, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, 3, samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p1", 1.0, {}, TOptionalStr("i1")), gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[1] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p2", 1.0, {}, TOptionalStr("i2")), gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[2] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p3", 1.0, {}, TOptionalStr("i3")), gatherer);
        }
        countingModel.sample(now, now + bucketLength, m_ResourceMonitor);
        model.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        this->addArrival(SMessage(now, "p1", 1.0, {}, TOptionalStr("i1")), gatherer);
    }
    for (std::size_t i = 0; i < 1; ++i) {
        this->addArrival(SMessage(now, "p2", 1.0, {}, TOptionalStr("i2")), gatherer);
    }
    for (std::size_t i = 0; i < 100; ++i) {
        this->addArrival(SMessage(now, "p3", 1.0, {}, TOptionalStr("i3")), gatherer);
    }
    countingModel.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);
    model.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model_t::CResultType type(model_t::CResultType::E_Unconditional |
                              model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    BOOST_TEST_REQUIRE(model.computeProbability(
        pid1, now, now + bucketLength, partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    BOOST_TEST_REQUIRE(model.computeProbability(
        pid2, now, now + bucketLength, partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    BOOST_TEST_REQUIRE(model.computeProbability(
        pid3, now, now + bucketLength, partitioningFields, 1, annotatedProbability3));

    TDouble1Vec p1Baseline = model.baselineBucketMean(
        model_t::E_IndividualSumByBucketAndPerson, pid1, 0, type, NO_CORRELATES, now);
    TDouble1Vec p2Baseline = model.baselineBucketMean(
        model_t::E_IndividualSumByBucketAndPerson, pid2, 0, type, NO_CORRELATES, now);
    TDouble1Vec p3Baseline = model.baselineBucketMean(
        model_t::E_IndividualSumByBucketAndPerson, pid3, 0, type, NO_CORRELATES, now);

    LOG_DEBUG(<< "p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG(<< "p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG(<< "p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG(<< "p1 baseline = " << p1Baseline[0]);
    LOG_DEBUG(<< "p2 baseline = " << p2Baseline[0]);
    LOG_DEBUG(<< "p3 baseline = " << p3Baseline[0]);

    BOOST_TEST_REQUIRE(annotatedProbability1.s_Probability > 0.05);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability < 0.05);
    BOOST_TEST_REQUIRE(annotatedProbability3.s_Probability < 0.05);
    BOOST_TEST_REQUIRE(p1Baseline[0] > 44.0);
    BOOST_TEST_REQUIRE(p1Baseline[0] < 46.0);
    BOOST_TEST_REQUIRE(p2Baseline[0] > 45.0);
    BOOST_TEST_REQUIRE(p2Baseline[0] < 46.0);
    BOOST_TEST_REQUIRE(p3Baseline[0] > 59.0);
    BOOST_TEST_REQUIRE(p3Baseline[0] < 61.0);
}

BOOST_FIXTURE_TEST_CASE(testInterimCorrectionsWithCorrelations, CTestFixture) {
    core_t::TTime startTime{3600};
    core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);
    params.s_MultivariateByFields = true;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector);

    factory.features({model_t::E_IndividualSumByBucketAndPerson});
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
    CModelFactory::TModelPtr modelPtr(factory.makeModel(gatherer));
    BOOST_TEST_REQUIRE(modelPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelPtr->category());
    CMetricModel& model = static_cast<CMetricModel&>(*modelPtr.get());
    CCountingModel countingModel(params, gatherer, interimBucketCorrector);

    std::size_t pid1 = this->addPerson("p1", gatherer);
    std::size_t pid2 = this->addPerson("p2", gatherer);
    std::size_t pid3 = this->addPerson("p3", gatherer);

    core_t::TTime now = startTime;
    core_t::TTime endTime(now + 2 * 24 * bucketLength);
    test::CRandomNumbers rng;
    TDoubleVec samples(1, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(80.0, 100.0, 1, samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p1", 1.0, {}, TOptionalStr("i1")), gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 10.5); ++i) {
            this->addArrival(SMessage(now, "p2", 1.0, {}, TOptionalStr("i2")), gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] - 9.5); ++i) {
            this->addArrival(SMessage(now, "p3", 1.0, {}, TOptionalStr("i3")), gatherer);
        }
        countingModel.sample(now, now + bucketLength, m_ResourceMonitor);
        model.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 9; ++i) {
        this->addArrival(SMessage(now, "p1", 1.0, {}, TOptionalStr("i1")), gatherer);
    }
    for (std::size_t i = 0; i < 10; ++i) {
        this->addArrival(SMessage(now, "p2", 1.0, {}, TOptionalStr("i2")), gatherer);
    }
    for (std::size_t i = 0; i < 8; ++i) {
        this->addArrival(SMessage(now, "p3", 1.0, {}, TOptionalStr("i3")), gatherer);
    }
    countingModel.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);
    model.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model_t::CResultType type(model_t::CResultType::E_Conditional |
                              model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    BOOST_TEST_REQUIRE(model.computeProbability(
        pid1, now, now + bucketLength, partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    BOOST_TEST_REQUIRE(model.computeProbability(
        pid2, now, now + bucketLength, partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    BOOST_TEST_REQUIRE(model.computeProbability(
        pid3, now, now + bucketLength, partitioningFields, 1, annotatedProbability3));

    TDouble1Vec p1Baseline = model.baselineBucketMean(
        model_t::E_IndividualSumByBucketAndPerson, pid1, 0, type,
        annotatedProbability1.s_AttributeProbabilities[0].s_Correlated, now);
    TDouble1Vec p2Baseline = model.baselineBucketMean(
        model_t::E_IndividualSumByBucketAndPerson, pid2, 0, type,
        annotatedProbability2.s_AttributeProbabilities[0].s_Correlated, now);
    TDouble1Vec p3Baseline = model.baselineBucketMean(
        model_t::E_IndividualSumByBucketAndPerson, pid3, 0, type,
        annotatedProbability3.s_AttributeProbabilities[0].s_Correlated, now);

    LOG_DEBUG(<< "p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG(<< "p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG(<< "p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG(<< "p1 baseline = " << p1Baseline[0]);
    LOG_DEBUG(<< "p2 baseline = " << p2Baseline[0]);
    LOG_DEBUG(<< "p3 baseline = " << p3Baseline[0]);

    BOOST_TEST_REQUIRE(annotatedProbability1.s_Probability > 0.7);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.7);
    BOOST_TEST_REQUIRE(annotatedProbability3.s_Probability > 0.7);
    BOOST_TEST_REQUIRE(p1Baseline[0] > 8.4);
    BOOST_TEST_REQUIRE(p1Baseline[0] < 8.6);
    BOOST_TEST_REQUIRE(p2Baseline[0] > 9.4);
    BOOST_TEST_REQUIRE(p2Baseline[0] < 9.6);
    BOOST_TEST_REQUIRE(p3Baseline[0] > 7.4);
    BOOST_TEST_REQUIRE(p3Baseline[0] < 7.6);
}

BOOST_FIXTURE_TEST_CASE(testCorrelatePersist, CTestFixture) {
    using TVector2 = maths::CVectorNx1<double, 2>;
    using TMatrix2 = maths::CSymmetricMatrixNxN<double, 2>;

    const core_t::TTime startTime{0};
    const core_t::TTime bucketLength{600};
    TDoubleVec means{10.0, 20.0};
    TDoubleVec covariances{3.0, 2.0, 2.0};
    TVector2 mean(means.begin(), means.end());
    TMatrix2 covariance(covariances.begin(), covariances.end());

    test::CRandomNumbers rng;

    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(mean.toVector<TDoubleVec>(),
                                          covariance.toVectors<TDoubleVecVec>(),
                                          10000, samples);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MultivariateByFields = true;
    this->makeModel(params, {model_t::E_IndividualMeanByPerson}, startTime);
    this->addPerson("p1", m_Gatherer);
    this->addPerson("p2", m_Gatherer);

    core_t::TTime time{startTime};
    core_t::TTime bucket{time + bucketLength};
    for (std::size_t i = 0; i < samples.size(); ++i, time += 60) {
        if (time >= bucket) {
            m_Model->sample(bucket - bucketLength, bucket, m_ResourceMonitor);
            bucket += bucketLength;
        }
        this->addArrival(SMessage(time, "p1", samples[i][0]), m_Gatherer);
        this->addArrival(SMessage(time, "p2", samples[i][0]), m_Gatherer);

        if ((i + 1) % 1000 == 0) {
            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                m_Model->acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CModelFactory::TModelPtr restoredModel(m_Factory->makeModel(m_Gatherer, traverser));

            // The XML representation of the new filter should be the same as the original
            std::string newXml;
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                restoredModel->acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }

            uint64_t origChecksum = m_Model->checksum(false);
            LOG_DEBUG(<< "original checksum = " << origChecksum);
            uint64_t restoredChecksum = restoredModel->checksum(false);
            LOG_DEBUG(<< "restored checksum = " << restoredChecksum);
            BOOST_REQUIRE_EQUAL(origChecksum, restoredChecksum);
            BOOST_REQUIRE_EQUAL(origXml, newXml);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testSummaryCountZeroRecordsAreIgnored, CTestFixture) {
    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);
    SModelParams params(bucketLength);
    std::string summaryCountField("count");
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector,
                                model_t::E_Manual, summaryCountField);

    factory.features({model_t::E_IndividualSumByBucketAndPerson});
    factory.bucketLength(bucketLength);
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::TDataGathererPtr gathererWithZeros(factory.makeDataGatherer(startTime));
    CModelFactory::TModelPtr modelWithZerosPtr(factory.makeModel(gathererWithZeros));
    BOOST_TEST_REQUIRE(modelWithZerosPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelWithZerosPtr->category());
    CMetricModel& modelWithZeros = static_cast<CMetricModel&>(*modelWithZerosPtr.get());

    CModelFactory::SGathererInitializationData gathererNoZerosInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoZeros(
        factory.makeDataGatherer(gathererNoZerosInitData));
    CModelFactory::SModelInitializationData initDataNoZeros(gathererNoZeros);
    CModelFactory::TModelPtr modelNoZerosPtr(factory.makeModel(initDataNoZeros));
    BOOST_TEST_REQUIRE(modelNoZerosPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelNoZerosPtr->category());
    CMetricModel& modelNoZeros = static_cast<CMetricModel&>(*modelNoZerosPtr.get());

    // The idea here is to compare a model that has records with summary count of zero
    // against a model that has no records at all where the first model had the zero-count records.

    core_t::TTime now = 100;
    core_t::TTime end = now + 50 * bucketLength;
    test::CRandomNumbers rng;
    double mean = 5.0;
    double variance = 2.0;
    TDoubleVec values;
    std::string summaryCountZero("0");
    std::string summaryCountOne("1");
    while (now < end) {
        for (std::size_t i = 0; i < 10; ++i) {
            rng.generateNormalSamples(mean, variance, 1, values);
            double value = values[0];
            rng.generateUniformSamples(0.0, 1.0, 1, values);
            if (values[0] < 0.05) {
                this->addArrival(SMessage(now, "p1", value, {}, TOptionalStr("i1"),
                                          TOptionalStr(), TOptionalStr(summaryCountZero)),
                                 gathererWithZeros);
            } else {
                this->addArrival(SMessage(now, "p1", value, {}, TOptionalStr("i1"),
                                          TOptionalStr(), TOptionalStr(summaryCountOne)),
                                 gathererWithZeros);
                this->addArrival(SMessage(now, "p1", value, {}, TOptionalStr("i1"),
                                          TOptionalStr(), TOptionalStr(summaryCountOne)),
                                 gathererNoZeros);
            }
        }
        modelWithZeros.sample(now, now + bucketLength, m_ResourceMonitor);
        modelNoZeros.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }

    BOOST_REQUIRE_EQUAL(modelWithZeros.checksum(), modelNoZeros.checksum());
}

BOOST_FIXTURE_TEST_CASE(testDecayRateControl, CTestFixture) {
    core_t::TTime startTime = 0;
    core_t::TTime bucketLength = 1800;

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MinimumModeFraction = model::CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
    model_t::EFeature feature = model_t::E_IndividualMeanByPerson;
    model_t::TFeatureVec features{feature};

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "*** Test anomaly ***");
    {
        // Test we don't adapt the decay rate if there is a short-lived
        // anomaly. We should get essentially identical prediction errors
        // with and without decay control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CMetricModelFactory factory(params, interimBucketCorrector);
        factory.features(features);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr model(factory.makeModel(gatherer));

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.0001;
        CMetricModelFactory referenceFactory(params, interimBucketCorrector);
        referenceFactory.features(features);
        CModelFactory::TDataGathererPtr referenceGatherer(
            referenceFactory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr referenceModel(referenceFactory.makeModel(referenceGatherer));

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                  model_t::CResultType::E_Interim);
        for (core_t::TTime t = startTime;
             t < startTime + 4 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG(<< "week " << t / core::constants::WEEK + 1);
            }

            TDoubleVec value;
            rng.generateUniformSamples(0.0, 10.0, 1, value);
            value[0] += 20.0 * (t > 3 * core::constants::WEEK &&
                                        t < core::constants::WEEK + 4 * 3600
                                    ? 1.0
                                    : 0.0);
            this->addArrival(SMessage(t + bucketLength / 2, "p1", value[0]), gatherer);
            this->addArrival(SMessage(t + bucketLength / 2, "p1", value[0]), referenceGatherer);
            model->sample(t, t + bucketLength, m_ResourceMonitor);
            referenceModel->sample(t, t + bucketLength, m_ResourceMonitor);
            meanPredictionError.add(std::fabs(
                model->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0] -
                model->baselineBucketMean(feature, 0, 0, type, NO_CORRELATES,
                                          t + bucketLength / 2)[0]));
            meanReferencePredictionError.add(std::fabs(
                referenceModel->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0] -
                referenceModel->baselineBucketMean(feature, 0, 0, type, NO_CORRELATES,
                                                   t + bucketLength / 2)[0]));
        }
        LOG_DEBUG(<< "mean = " << maths::CBasicStatistics::mean(meanPredictionError));
        LOG_DEBUG(<< "reference = "
                  << maths::CBasicStatistics::mean(meanReferencePredictionError));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            maths::CBasicStatistics::mean(meanReferencePredictionError),
            maths::CBasicStatistics::mean(meanPredictionError), 0.05);
    }

    LOG_DEBUG(<< "*** Test step change ***");
    {
        // This change point is amongst those we explicitly detect so
        // check we get similar detection performance with and without
        // decay rate control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CMetricModelFactory factory(params, interimBucketCorrector);
        factory.features(features);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr model(factory.makeModel(gatherer));

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CMetricModelFactory referenceFactory(params, interimBucketCorrector);
        referenceFactory.features(features);
        CModelFactory::TDataGathererPtr referenceGatherer(
            referenceFactory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr referenceModel(referenceFactory.makeModel(referenceGatherer));

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                  model_t::CResultType::E_Interim);
        for (core_t::TTime t = startTime; t < 10 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG(<< "week " << t / core::constants::WEEK + 1);
            }

            double value = 10.0 *
                           (1.0 + std::sin(boost::math::double_constants::two_pi *
                                           static_cast<double>(t) /
                                           static_cast<double>(core::constants::DAY))) *
                           (t < 5 * core::constants::WEEK ? 1.0 : 2.0);
            TDoubleVec noise;
            rng.generateUniformSamples(0.0, 3.0, 1, noise);
            this->addArrival(SMessage(t + bucketLength / 2, "p1", value + noise[0]), gatherer);
            this->addArrival(SMessage(t + bucketLength / 2, "p1", value + noise[0]),
                             referenceGatherer);
            model->sample(t, t + bucketLength, m_ResourceMonitor);
            referenceModel->sample(t, t + bucketLength, m_ResourceMonitor);
            meanPredictionError.add(std::fabs(
                model->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0] -
                model->baselineBucketMean(feature, 0, 0, type, NO_CORRELATES,
                                          t + bucketLength / 2)[0]));
            meanReferencePredictionError.add(std::fabs(
                referenceModel->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0] -
                referenceModel->baselineBucketMean(feature, 0, 0, type, NO_CORRELATES,
                                                   t + bucketLength / 2)[0]));
        }
        LOG_DEBUG(<< "mean = " << maths::CBasicStatistics::mean(meanPredictionError));
        LOG_DEBUG(<< "reference = "
                  << maths::CBasicStatistics::mean(meanReferencePredictionError));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            maths::CBasicStatistics::mean(meanReferencePredictionError),
            maths::CBasicStatistics::mean(meanPredictionError), 0.05);
    }

    LOG_DEBUG(<< "*** Test unmodelled cyclic component ***");
    {
        // This modulates the event rate using a sine with period 10 weeks
        // effectively there are significant "manoeuvres" in the event rate
        // every 5 weeks at the function turning points. We check we get a
        // significant reduction in the prediction error with decay rate
        // control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.0005;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CMetricModelFactory factory(params, interimBucketCorrector);
        factory.features(features);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr model(factory.makeModel(gatherer));

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.0005;
        CMetricModelFactory referenceFactory(params, interimBucketCorrector);
        referenceFactory.features(features);
        CModelFactory::TDataGathererPtr referenceGatherer(
            referenceFactory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr referenceModel(referenceFactory.makeModel(referenceGatherer));

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                  model_t::CResultType::E_Interim);
        for (core_t::TTime t = startTime; t < 20 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG(<< "week " << t / core::constants::WEEK + 1);
            }

            double value =
                10.0 *
                (1.0 + std::sin(boost::math::double_constants::two_pi *
                                static_cast<double>(t) /
                                static_cast<double>(core::constants::DAY))) *
                (1.0 + std::sin(boost::math::double_constants::two_pi *
                                static_cast<double>(t) / 10.0 /
                                static_cast<double>(core::constants::WEEK)));
            TDoubleVec noise;
            rng.generateUniformSamples(0.0, 3.0, 1, noise);
            this->addArrival(SMessage(t + bucketLength / 2, "p1", value + noise[0]), gatherer);
            this->addArrival(SMessage(t + bucketLength / 2, "p1", value + noise[0]),
                             referenceGatherer);
            model->sample(t, t + bucketLength, m_ResourceMonitor);
            referenceModel->sample(t, t + bucketLength, m_ResourceMonitor);
            meanPredictionError.add(std::fabs(
                model->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0] -
                model->baselineBucketMean(feature, 0, 0, type, NO_CORRELATES,
                                          t + bucketLength / 2)[0]));
            meanReferencePredictionError.add(std::fabs(
                referenceModel->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0] -
                referenceModel->baselineBucketMean(feature, 0, 0, type, NO_CORRELATES,
                                                   t + bucketLength / 2)[0]));
        }
        LOG_DEBUG(<< "mean = " << maths::CBasicStatistics::mean(meanPredictionError));
        LOG_DEBUG(<< "reference = "
                  << maths::CBasicStatistics::mean(meanReferencePredictionError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanPredictionError) <
                           0.8 * maths::CBasicStatistics::mean(meanReferencePredictionError));
    }
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForLowMedian, CTestFixture) {
    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowMedianBucket = 60;
    std::size_t highMedianBucket = 80;
    double mean = 5.0;
    double variance = 0.00001;
    double lowMean = 2.0;
    double highMean = 10.0;

    SModelParams params(bucketLength);
    this->makeModel(params, {model_t::E_IndividualLowMedianByPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMedianBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMedianBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        BOOST_TEST_REQUIRE(model.computeProbability(
            0 /*pid*/, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));

    BOOST_TEST_REQUIRE(probabilities[lowMedianBucket] < 0.01);
    BOOST_TEST_REQUIRE(probabilities[highMedianBucket] > 0.1);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForHighMedian, CTestFixture) {
    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowMedianBucket = 60;
    std::size_t highMedianBucket = 80;
    double mean = 5.0;
    double variance = 0.00001;
    double lowMean = 2.0;
    double highMean = 10.0;

    SModelParams params(bucketLength);
    makeModel(params, {model_t::E_IndividualHighMeanByPerson}, startTime);
    CMetricModel& model = static_cast<CMetricModel&>(*m_Model);
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMedianBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMedianBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0; j < values.size(); ++j) {
            this->addArrival(
                SMessage(time + static_cast<core_t::TTime>(j), "p", values[j]), m_Gatherer);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        BOOST_TEST_REQUIRE(model.computeProbability(
            0 /*pid*/, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));

    BOOST_TEST_REQUIRE(probabilities[lowMedianBucket] > 0.1);
    BOOST_TEST_REQUIRE(probabilities[highMedianBucket] < 0.01);
}

BOOST_FIXTURE_TEST_CASE(testIgnoreSamplingGivenDetectionRules, CTestFixture) {
    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.

    // Create a rule to filter buckets where the actual value > 100
    CRuleCondition condition;
    condition.appliesTo(CRuleCondition::E_Actual);
    condition.op(CRuleCondition::E_GT);
    condition.value(100.0);
    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipModelUpdate);
    rule.addCondition(condition);

    std::size_t bucketLength(300);
    std::size_t startTime(300);

    // Model without the skip sampling rule
    SModelParams paramsNoRules(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(paramsNoRules, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_IndividualMeanByPerson};
    CModelFactory::TDataGathererPtr gathererNoSkip(factory.makeDataGatherer(startTime));
    CModelFactory::TModelPtr modelPtrNoSkip(factory.makeModel(gathererNoSkip));
    CMetricModel* modelNoSkip = dynamic_cast<CMetricModel*>(modelPtrNoSkip.get());

    // Model with the skip sampling rule
    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);
    CMetricModelFactory factoryWithSkip(paramsWithRules, interimBucketCorrector);
    CModelFactory::TDataGathererPtr gathererWithSkip(
        factoryWithSkip.makeDataGatherer(startTime));
    CModelFactory::TModelPtr modelPtrWithSkip(factoryWithSkip.makeModel(gathererWithSkip));
    CMetricModel* modelWithSkip = dynamic_cast<CMetricModel*>(modelPtrWithSkip.get());

    std::size_t endTime = startTime + bucketLength;

    // Add a bucket to both models
    for (std::size_t i = 0; i < 60; i++) {
        this->addArrival(SMessage(startTime + i, "p1", 1.0), gathererNoSkip);
        this->addArrival(SMessage(startTime + i, "p1", 1.0), gathererWithSkip);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // Add a bucket to both models
    for (std::size_t i = 0; i < 60; i++) {
        this->addArrival(SMessage(startTime + i, "p1", 1.0), gathererNoSkip);
        this->addArrival(SMessage(startTime + i, "p1", 1.0), gathererWithSkip);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // this sample will be skipped by the detection rule
    for (std::size_t i = 0; i < 60; i++) {
        this->addArrival(SMessage(startTime + i, "p1", 110.0), gathererWithSkip);
    }
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    startTime = endTime;
    endTime += bucketLength;

    // Wind the other model forward
    modelNoSkip->skipSampling(startTime);

    for (std::size_t i = 0; i < 60; i++) {
        this->addArrival(SMessage(startTime + i, "p1", 2.0), gathererNoSkip);
        this->addArrival(SMessage(startTime + i, "p1", 2.0), gathererWithSkip);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    // Checksums will be different due to the data gatherers
    BOOST_TEST_REQUIRE(modelWithSkip->checksum() != modelNoSkip->checksum());

    // but the underlying models should be the same
    CAnomalyDetectorModel::TModelDetailsViewUPtr modelWithSkipView =
        modelWithSkip->details();
    CAnomalyDetectorModel::TModelDetailsViewUPtr modelNoSkipView = modelNoSkip->details();

    // TODO this test fails due a different checksums for the decay rate and prior
    // uint64_t withSkipChecksum = modelWithSkipView->model(model_t::E_IndividualMeanByPerson, 0)->checksum();
    // uint64_t noSkipChecksum = modelNoSkipView->model(model_t::E_IndividualMeanByPerson, 0)->checksum();
    // BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    // TODO These checks fail see elastic/machine-learning-cpp/issues/485
    // Check the last value times of the underlying models are the same
    // const maths::CUnivariateTimeSeriesModel *timeSeriesModel =
    //     dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_IndividualMeanByPerson, 0));
    // BOOST_TEST_REQUIRE(timeSeriesModel != 0);

    // core_t::TTime time = timeSeriesModel->trend().lastValueTime();
    // BOOST_REQUIRE_EQUAL(model_t::sampleTime(model_t::E_IndividualMeanByPerson, startTime, bucketLength), time);

    // // The last times of model with a skip should be the same
    // timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_IndividualMeanByPerson, 0));
    // BOOST_REQUIRE_EQUAL(time, timeSeriesModel->trend().lastValueTime());
}

BOOST_AUTO_TEST_SUITE_END()
