/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CIntegerTools.h>
#include <maths/CModelWeight.h>
#include <maths/CPrior.h>

#include <model/CAnnotatedProbability.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CCountingModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CEventRateModel.h>
#include <model/CEventRateModelFactory.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>
#include <model/CModelFactory.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>
#include <model/CRuleCondition.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "CModelTestFixtureBase.h"

#include <boost/test/unit_test.hpp>

#include <memory>
#include <string>
#include <vector>

#include <stdint.h>

BOOST_TEST_DONT_PRINT_LOG_VALUE(CModelTestFixtureBase::TStrVec::iterator)

BOOST_AUTO_TEST_SUITE(CEventRateModelTest)

using namespace ml;
using namespace model;

namespace {
const std::string EMPTY_STRING;
const CModelTestFixtureBase::TSizeDoublePr1Vec NO_CORRELATES;
} // unnamed::

class CTestFixture : public CModelTestFixtureBase {
public:
    TUInt64Vec rawEventCounts(std::size_t copies = 1) {
        uint64_t counts[] = {54, 67, 39, 58, 46, 50, 42,
                             48, 53, 51, 50, 57, 53, 49};
        TUInt64Vec result;
        for (std::size_t i = 0u; i < copies; ++i) {
            result.insert(result.end(), std::begin(counts), std::end(counts));
        }
        return result;
    }

    void generateEvents(const core_t::TTime& startTime,
                        const core_t::TTime& bucketLength,
                        const TUInt64Vec& eventCountsPerBucket,
                        TTimeVec& eventArrivalTimes) {
        // Generate an ordered collection of event arrival times.
        test::CRandomNumbers rng;
        double bucketStartTime = static_cast<double>(startTime);
        for (auto count : eventCountsPerBucket) {
            double bucketEndTime = bucketStartTime + static_cast<double>(bucketLength);

            TDoubleVec bucketEventTimes;
            rng.generateUniformSamples(bucketStartTime, bucketEndTime - 1.0,
                                       static_cast<std::size_t>(count), bucketEventTimes);

            std::sort(bucketEventTimes.begin(), bucketEventTimes.end());

            for (auto time_ : bucketEventTimes) {
                core_t::TTime time = static_cast<core_t::TTime>(time_);
                time = std::min(static_cast<core_t::TTime>(bucketEndTime - 1.0),
                                std::max(static_cast<core_t::TTime>(bucketStartTime), time));
                eventArrivalTimes.push_back(time);
            }

            bucketStartTime = bucketEndTime;
        }
    }

    void generateSporadicEvents(const core_t::TTime& startTime,
                                const core_t::TTime& bucketLength,
                                const TUInt64Vec& nonZeroEventCountsPerBucket,
                                TTimeVec& eventArrivalTimes) {
        // Generate an ordered collection of event arrival times.
        test::CRandomNumbers rng;
        double bucketStartTime = static_cast<double>(startTime);
        for (auto count : nonZeroEventCountsPerBucket) {
            double bucketEndTime = bucketStartTime + static_cast<double>(bucketLength);

            TDoubleVec bucketEventTimes;
            rng.generateUniformSamples(bucketStartTime, bucketEndTime - 1.0,
                                       static_cast<std::size_t>(count), bucketEventTimes);

            std::sort(bucketEventTimes.begin(), bucketEventTimes.end());

            for (auto time_ : bucketEventTimes) {
                core_t::TTime time = static_cast<core_t::TTime>(time_);
                time = std::min(static_cast<core_t::TTime>(bucketEndTime - 1.0),
                                std::max(static_cast<core_t::TTime>(bucketStartTime), time));
                eventArrivalTimes.push_back(time);
            }

            TDoubleVec gap;
            rng.generateUniformSamples(0.0, 10.0, 1u, gap);
            bucketStartTime += static_cast<double>(bucketLength) * std::ceil(gap[0]);
        }
    }

    void handleEvent(const CDataGatherer::TStrCPtrVec& fields,
                     core_t::TTime time,
                     CModelFactory::TDataGathererPtr& gatherer,
                     CResourceMonitor& resourceMonitor) {
        CEventData eventResult;
        eventResult.time(time);
        gatherer->addArrival(fields, eventResult, resourceMonitor);
    }

    void testModelWithValueField(model_t::EFeature feature,
                                 TSizeVecVecVec& fields,
                                 TStrVec& strings,
                                 CResourceMonitor& resourceMonitor) {
        LOG_DEBUG(<< "  *** testing feature " << model_t::print(feature));

        const core_t::TTime startTime{1346968800};
        const core_t::TTime bucketLength{3600};
        SModelParams params(bucketLength);
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRatePopulationModelFactory factory(params, interimBucketCorrector);
        factory.features({feature});
        factory.fieldNames("", "", "P", "V", TStrVec());
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        CModelFactory::TModelPtr model(factory.makeModel(gatherer));
        BOOST_TEST_REQUIRE(model);

        std::size_t anomalousBucket{20u};
        std::size_t numberBuckets{30u};

        const core_t::TTime endTime = startTime + (numberBuckets * bucketLength);

        std::size_t i{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, i++) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            for (std::size_t j = 0; j < fields[i].size(); ++j) {
                CDataGatherer::TStrCPtrVec f{&strings[fields[i][j][0]],
                                             &strings[fields[i][j][1]],
                                             &strings[fields[i][j][2]]};
                handleEvent(f, bucketStartTime + j, gatherer, resourceMonitor);
            }

            model->sample(bucketStartTime, bucketEndTime, resourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            model->computeProbability(0 /*pid*/, bucketStartTime, bucketEndTime,
                                      partitioningFields, 1, annotatedProbability);
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            if (i == anomalousBucket) {
                BOOST_TEST_REQUIRE(annotatedProbability.s_Probability < 0.001);
            } else {
                BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.6);
            }
        }
    }

    void makeModel(const SModelParams& params,
                   const model_t::TFeatureVec& features,
                   core_t::TTime startTime,
                   std::size_t numberPeople,
                   const std::string& summaryCountField = EMPTY_STRING) {
        this->makeModelT<CEventRateModelFactory>(params, features, startTime,
                                                 model_t::E_EventRateOnline, m_Gatherer,
                                                 m_Model, {}, summaryCountField);

        for (std::size_t i = 0u; i < numberPeople; ++i) {
            BOOST_REQUIRE_EQUAL(
                i, this->addPerson("p" + core::CStringUtils::typeToString(i + 1), m_Gatherer));
        }
    }

protected:
    using TDoubleSizeStrTr = core::CTriple<double, std::size_t, std::string>;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizeStrTr>;
    using TMinAccumulatorVec = std::vector<TMinAccumulator>;
};

BOOST_FIXTURE_TEST_CASE(testCountSample, CTestFixture) {
    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    TMathsModelPtr timeseriesModel{m_Factory->defaultFeatureModel(
        model_t::E_IndividualCountByBucketAndPerson, bucketLength, 0.4, true)};
    maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec weights{
        maths_t::CUnitWeights::unit<TDouble2Vec>(1)};

    // Generate some events.
    TTimeVec eventTimes;
    TUInt64Vec expectedEventCounts(rawEventCounts());
    generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
    core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
    LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
              << ", # events = " << eventTimes.size());

    std::size_t i{0u};
    std::size_t j{0u};
    for (core_t::TTime bucketStartTime = startTime; bucketStartTime < endTime;
         bucketStartTime += bucketLength, ++j) {
        core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

        double count{0.0};
        for (/**/; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
            this->addArrival(SMessage(eventTimes[i], "p1", TOptionalDouble()), m_Gatherer);
            count += 1.0;
        }

        LOG_DEBUG(<< "Bucket count = " << count);

        model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

        maths::CModelAddSamplesParams params_;
        params_.integer(true)
            .nonNegative(true)
            .propagationInterval(1.0)
            .trendWeights(weights)
            .priorWeights(weights);
        double sample{static_cast<double>(expectedEventCounts[j])};
        maths::CModel::TTimeDouble2VecSizeTrVec expectedSamples{
            core::make_triple((bucketStartTime + bucketEndTime) / 2,
                              maths::CModel::TDouble2Vec{sample}, std::size_t{0})};
        timeseriesModel->addSamples(params_, expectedSamples);

        // Test we sample the data correctly.
        BOOST_REQUIRE_EQUAL(expectedEventCounts[j],
                            static_cast<uint64_t>(model->currentBucketValue(
                                model_t::E_IndividualCountByBucketAndPerson, 0,
                                0, bucketStartTime)[0]));
        BOOST_REQUIRE_EQUAL(timeseriesModel->checksum(),
                            model->details()
                                ->model(model_t::E_IndividualCountByBucketAndPerson, 0)
                                ->checksum());
    }

    // Test persistence. (We check for idempotency.)

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        model->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "origXml = " << origXml);
    LOG_DEBUG(<< "origXml size = " << origXml.size());
    BOOST_TEST_REQUIRE(origXml.size() < 41000);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    CModelFactory::TModelPtr restoredModelPtr(m_Factory->makeModel(m_Gatherer, traverser));

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModelPtr->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    LOG_DEBUG(<< "original checksum = " << model->checksum(false));
    LOG_DEBUG(<< "restored checksum = " << restoredModelPtr->checksum(false));
    BOOST_REQUIRE_EQUAL(model->checksum(false), restoredModelPtr->checksum(false));
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testNonZeroCountSample, CTestFixture) {
    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    this->makeModel(params, {model_t::E_IndividualNonZeroCountByBucketAndPerson},
                    startTime, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    TMathsModelPtr timeseriesModel{m_Factory->defaultFeatureModel(
        model_t::E_IndividualNonZeroCountByBucketAndPerson, bucketLength, 0.4, true)};
    maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec weights{
        maths_t::CUnitWeights::unit<TDouble2Vec>(1)};

    // Generate some events.
    TTimeVec eventTimes;
    TUInt64Vec expectedEventCounts = rawEventCounts();
    generateSporadicEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
    core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
    LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
              << ", # events = " << eventTimes.size());

    std::size_t i{0u};
    std::size_t j{0u};
    for (core_t::TTime bucketStartTime = startTime; bucketStartTime < endTime;
         bucketStartTime += bucketLength) {
        core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

        double count{0.0};
        for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
            this->addArrival(SMessage(eventTimes[i], "p1", TOptionalDouble()), m_Gatherer);
            count += 1.0;
        }

        LOG_DEBUG(<< "Bucket count = " << count);

        model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

        if (*model->currentBucketCount(0, bucketStartTime) > 0) {
            maths::CModelAddSamplesParams params_;
            params_.integer(true)
                .nonNegative(true)
                .propagationInterval(1.0)
                .trendWeights(weights)
                .priorWeights(weights);
            double sample{static_cast<double>(model_t::offsetCountToZero(
                model_t::E_IndividualNonZeroCountByBucketAndPerson,
                static_cast<double>(expectedEventCounts[j])))};
            maths::CModel::TTimeDouble2VecSizeTrVec expectedSamples{core::make_triple(
                (bucketStartTime + bucketEndTime) / 2,
                maths::CModel::TDouble2Vec{sample}, std::size_t{0})};
            timeseriesModel->addSamples(params_, expectedSamples);

            // Test we sample the data correctly.
            BOOST_REQUIRE_EQUAL(expectedEventCounts[j],
                                static_cast<uint64_t>(model->currentBucketValue(
                                    model_t::E_IndividualNonZeroCountByBucketAndPerson,
                                    0, 0, bucketStartTime)[0]));
            BOOST_REQUIRE_EQUAL(timeseriesModel->checksum(),
                                model->details()
                                    ->model(model_t::E_IndividualNonZeroCountByBucketAndPerson, 0)
                                    ->checksum());

            ++j;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testRare, CTestFixture) {
    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);
    this->makeModel(params,
                    {model_t::E_IndividualTotalBucketCountByPerson,
                     model_t::E_IndividualIndicatorOfBucketPerson},
                    startTime, 5);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());

    core_t::TTime time{startTime};
    for (/**/; time < startTime + 10 * bucketLength; time += bucketLength) {
        this->addArrival(SMessage(time + bucketLength / 2, "p1", TOptionalDouble()), m_Gatherer);
        this->addArrival(SMessage(time + bucketLength / 2, "p2", TOptionalDouble()), m_Gatherer);
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }
    for (/**/; time < startTime + 50 * bucketLength; time += bucketLength) {
        this->addArrival(SMessage(time + bucketLength / 2, "p1", TOptionalDouble()), m_Gatherer);
        this->addArrival(SMessage(time + bucketLength / 2, "p2", TOptionalDouble()), m_Gatherer);
        this->addArrival(SMessage(time + bucketLength / 2, "p3", TOptionalDouble()), m_Gatherer);
        this->addArrival(SMessage(time + bucketLength / 2, "p4", TOptionalDouble()), m_Gatherer);
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }

    this->addArrival(SMessage(time + bucketLength / 2, "p1", TOptionalDouble()), m_Gatherer);
    this->addArrival(SMessage(time + bucketLength / 2, "p2", TOptionalDouble()), m_Gatherer);
    this->addArrival(SMessage(time + bucketLength / 2, "p3", TOptionalDouble()), m_Gatherer);
    this->addArrival(SMessage(time + bucketLength / 2, "p4", TOptionalDouble()), m_Gatherer);
    this->addArrival(SMessage(time + bucketLength / 2, "p5", TOptionalDouble()), m_Gatherer);
    model->sample(time, time + bucketLength, m_ResourceMonitor);

    TDoubleVec probabilities;
    for (std::size_t pid = 0u; pid < 5; ++pid) {
        SAnnotatedProbability annotatedProbability;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        BOOST_TEST_REQUIRE(model->computeProbability(
            pid, time, time + bucketLength, partitioningFields, 0, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);
    }

    // We expect "p1 = p2 > p3 = p4 >> p5".
    BOOST_REQUIRE_EQUAL(5, probabilities.size());
    BOOST_REQUIRE_EQUAL(probabilities[0], probabilities[1]);
    BOOST_TEST_REQUIRE(probabilities[1] > probabilities[2]);
    BOOST_REQUIRE_EQUAL(probabilities[2], probabilities[3]);
    BOOST_TEST_REQUIRE(probabilities[3] > 50.0 * probabilities[4]);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        model->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "origXml = " << origXml);
    LOG_DEBUG(<< "origXml size = " << origXml.size());
    BOOST_TEST_REQUIRE(origXml.size() < 22000);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    CModelFactory::TModelPtr restoredModelPtr(m_Factory->makeModel(m_Gatherer, traverser));

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModelPtr->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculation, CTestFixture) {
    using TDoubleSizeAnotatedProbabilityTr =
        core::CTriple<double, std::size_t, SAnnotatedProbability>;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<
        TDoubleSizeAnotatedProbabilityTr,
        std::function<bool(const TDoubleSizeAnotatedProbabilityTr&, const TDoubleSizeAnotatedProbabilityTr&)>>;

    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};

    TSizeVecVec anomalousBuckets{{25}, {24, 25, 26, 27}};
    TDoubleVec anomalousBucketsRateMultipliers{3.0, 1.3};

    for (std::size_t t = 0; t < 2; ++t) {

        // Create the model.
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 1);
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts(2);
        for (auto i : anomalousBuckets[t]) {
            expectedEventCounts[i] =
                static_cast<std::size_t>(static_cast<double>(expectedEventCounts[i]) *
                                         anomalousBucketsRateMultipliers[t]);
        }
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        // Play the data through the model and get the lowest probability buckets.
        TMinAccumulator minProbabilities(2, TDoubleSizeAnotatedProbabilityTr{},
                                         [](const TDoubleSizeAnotatedProbabilityTr& lhs,
                                            const TDoubleSizeAnotatedProbabilityTr& rhs) {
                                             return lhs.first < rhs.first;
                                         });

        std::size_t i{0u};
        for (core_t::TTime j = 0, bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(SMessage(eventTimes[i], "p1", TOptionalDouble()), m_Gatherer);
                count += 1.0;
            }

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability p;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields, 1, p));
            LOG_DEBUG(<< "bucket count = " << count << ", probability = " << p.s_Probability);
            minProbabilities.add({p.s_Probability, static_cast<std::size_t>(j), p});
        }

        minProbabilities.sort();

        if (anomalousBuckets[t].size() == 1) {
            // Check the one anomalous bucket has the lowest probability by a significant margin.
            BOOST_REQUIRE_EQUAL(anomalousBuckets[0][0], minProbabilities[0].second);
            BOOST_TEST_REQUIRE(minProbabilities[0].first / minProbabilities[1].first < 0.1);
        } else {
            // Check the multi-bucket impact values are relatively high
            // (indicating a large contribution from multi-bucket analysis)
            double expectedMultiBucketImpactThresholds[2]{0.3, 2.5};
            for (int j = 0; j < 2; ++j) {
                double multiBucketImpact = minProbabilities[j].third.s_MultiBucketImpact;
                LOG_DEBUG(<< "multi_bucket_impact = " << multiBucketImpact);
                BOOST_TEST_REQUIRE(multiBucketImpact >=
                                   expectedMultiBucketImpactThresholds[j]);
                BOOST_TEST_REQUIRE(multiBucketImpact <=
                                   CAnomalyDetectorModelConfig::MAXIMUM_MULTI_BUCKET_IMPACT_MAGNITUDE);
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForLowNonZeroCount, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{100};
    std::size_t lowNonZeroCountBucket{6u};
    std::size_t highNonZeroCountBucket{8u};

    TSizeVec bucketCounts{50, 50, 50, 50, 50, 0, 0, 0, 50, 1, 50, 100, 50, 50};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    this->makeModel(params, {model_t::E_IndividualLowNonZeroCountByBucketAndPerson},
                    startTime, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());

    TDoubleVec probabilities;

    core_t::TTime time{startTime};
    for (auto count : bucketCounts) {
        LOG_DEBUG(<< "Writing " << count << " values");

        for (std::size_t i = 0u; i < count; ++i) {
            this->addArrival(SMessage(time + static_cast<core_t::TTime>(i),
                                      "p1", TOptionalDouble()),
                             m_Gatherer);
        }
        model->sample(time, time + bucketLength, m_ResourceMonitor);

        SAnnotatedProbability p;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        if (model->computeProbability(0 /*pid*/, time, time + bucketLength,
                                      partitioningFields, 0, p) == false) {
            continue;
        }
        LOG_DEBUG(<< "probability = " << p.s_Probability);
        if (*model->currentBucketCount(0, time) > 0) {
            probabilities.push_back(p.s_Probability);
        }
        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
    BOOST_REQUIRE_EQUAL(11, probabilities.size());
    BOOST_TEST_REQUIRE(probabilities[lowNonZeroCountBucket] < 0.06);
    BOOST_TEST_REQUIRE(probabilities[highNonZeroCountBucket] > 0.9);
}

BOOST_FIXTURE_TEST_CASE(testProbabilityCalculationForHighNonZeroCount, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{100};
    std::size_t lowNonZeroCountBucket{6u};
    std::size_t highNonZeroCountBucket{8u};

    TSizeVec bucketCounts{50, 50, 50, 50, 50, 0, 0, 0, 50, 100, 50, 1, 50, 50};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    this->makeModel(params, {model_t::E_IndividualHighNonZeroCountByBucketAndPerson},
                    startTime, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());

    TDoubleVec probabilities;

    core_t::TTime time{startTime};
    for (auto count : bucketCounts) {
        LOG_DEBUG(<< "Writing " << count << " values");

        for (std::size_t i = 0u; i < count; ++i) {
            this->addArrival(SMessage(time + static_cast<core_t::TTime>(i),
                                      "p1", TOptionalDouble()),
                             m_Gatherer);
        }
        model->sample(time, time + bucketLength, m_ResourceMonitor);

        SAnnotatedProbability p;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        if (model->computeProbability(0 /*pid*/, time, time + bucketLength,
                                      partitioningFields, 1, p) == false) {
            continue;
        }
        LOG_DEBUG(<< "probability = " << p.s_Probability);
        if (*model->currentBucketCount(0, time) > 0) {
            probabilities.push_back(p.s_Probability);
        }
        time += bucketLength;
    }

    LOG_DEBUG(<< "probabilities = " << core::CContainerPrinter::print(probabilities));
    BOOST_REQUIRE_EQUAL(11, probabilities.size());
    BOOST_TEST_REQUIRE(probabilities[lowNonZeroCountBucket] < 0.06);
    BOOST_TEST_REQUIRE(probabilities[highNonZeroCountBucket] > 0.9);
}

BOOST_FIXTURE_TEST_CASE(testCorrelatedNoTrend, CTestFixture) {
    // Check we find the correct correlated variables, and identify
    // correlate and marginal anomalies.

    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};

    test::CRandomNumbers rng;

    const std::size_t numberBuckets{200};
    const TDoubleVec means{20.0, 25.0, 100.0, 800.0};
    const TDoubleVecVec covariances{{3.0, 2.5, 0.0, 0.0},
                                    {2.5, 4.0, 0.0, 0.0},
                                    {0.0, 0.0, 100.0, -500.0},
                                    {0.0, 0.0, -500.0, 3000.0}};

    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(means, covariances, numberBuckets, samples);

    {
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        params.s_LearnRate = 1.0;
        params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
        params.s_MinimumModeCount = 24.0;
        params.s_MultivariateByFields = true;
        this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 4);
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());
        BOOST_TEST_REQUIRE(model);

        LOG_DEBUG(<< "Test correlation anomalies");

        TSizeVec anomalyBuckets{100, 160, 190, numberBuckets};
        TDoubleVecVec anomalies{{-5.73, 4.29, 0.0, 0.0},
                                {0.0, 0.0, 89.99, 15.38},
                                {-7.73, 5.59, 52.99, 9.03}};

        TMinAccumulatorVec probabilities{TMinAccumulator{2}, TMinAccumulator{2},
                                         TMinAccumulator{2}, TMinAccumulator{2}};

        core_t::TTime time{startTime};
        for (std::size_t i = 0u, anomaly = 0u; i < numberBuckets; ++i) {
            for (std::size_t j = 0u; j < samples[i].size(); ++j) {
                std::string person = std::string("p") +
                                     core::CStringUtils::typeToString(j + 1);
                double n = samples[i][j];
                if (i == anomalyBuckets[anomaly]) {
                    n += anomalies[anomaly][j];
                }
                for (std::size_t k = 0u; k < static_cast<std::size_t>(n); ++k) {
                    this->addArrival(SMessage(time + static_cast<core_t::TTime>(j),
                                              person, TOptionalDouble()),
                                     m_Gatherer);
                }
            }
            if (i == anomalyBuckets[anomaly]) {
                ++anomaly;
            }
            model->sample(time, time + bucketLength, m_ResourceMonitor);

            for (std::size_t pid = 0u; pid < samples[i].size(); ++pid) {
                SAnnotatedProbability p;
                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                BOOST_TEST_REQUIRE(model->computeProbability(
                    pid, time, time + bucketLength, partitioningFields, 1, p));
                std::string correlated;
                if (p.s_AttributeProbabilities[0].s_CorrelatedAttributes.size() > 0 &&
                    p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0] != nullptr &&
                    p.s_AttributeProbabilities[0].s_Type.isUnconditional() == false) {
                    correlated = *p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0];
                }
                probabilities[pid].add(TDoubleSizeStrTr(p.s_Probability, i, correlated));
            }
            time += bucketLength;
        }

        TStrVec expectedResults{"[(100,p2), (190,p2)]", "[(100,p1), (190,p1)]",
                                "[(160,p4), (190,p4)]", "[(160,p3), (190,p3)]"};
        for (std::size_t i = 0u; i < probabilities.size(); ++i) {
            LOG_DEBUG(<< "probabilities = " << probabilities[i].print());
            std::string results[2];
            for (std::size_t j = 0u; j < 2; ++j) {
                results[j] =
                    std::string("(") +
                    core::CStringUtils::typeToString(probabilities[i][j].second) +
                    "," + probabilities[i][j].third + ")";
            }
            std::sort(results, results + 2);
            BOOST_REQUIRE_EQUAL(expectedResults[i], core::CContainerPrinter::print(results));
        }

        // Test persist and restore with correlate models.
        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            model->acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_TRACE(<< "origXml = " << origXml);
        LOG_DEBUG(<< "origXml size = " << origXml.size());
        BOOST_TEST_REQUIRE(origXml.size() < 195000);

        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        CModelFactory::TModelPtr restoredModel(m_Factory->makeModel(m_Gatherer, traverser));
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredModel->acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }

        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }
    {
        LOG_DEBUG(<< "Test marginal anomalies");

        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        params.s_LearnRate = 1.0;
        params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
        params.s_MinimumModeCount = 24.0;
        params.s_MultivariateByFields = true;
        this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 4);
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());
        BOOST_TEST_REQUIRE(model);

        TSizeVec anomalyBuckets{100, 160, 190, numberBuckets};
        TDoubleVecVec anomalies{{11.07, 14.19, 0.0, 0.0},
                                {0.0, 0.0, -66.9, 399.95},
                                {11.07, 14.19, -48.15, 329.95}};

        TMinAccumulatorVec probabilities{TMinAccumulator{3}, TMinAccumulator{3},
                                         TMinAccumulator{3}, TMinAccumulator{3}};

        core_t::TTime time{startTime};
        for (std::size_t i = 0u, anomaly = 0u; i < numberBuckets; ++i) {
            for (std::size_t j = 0u; j < samples[i].size(); ++j) {
                std::string person = std::string("p") +
                                     core::CStringUtils::typeToString(j + 1);
                double n = samples[i][j];
                if (i == anomalyBuckets[anomaly]) {
                    n += anomalies[anomaly][j];
                }
                n = std::max(n, 0.0);
                for (std::size_t k = 0u; k < static_cast<std::size_t>(n); ++k) {
                    this->addArrival(SMessage(time + static_cast<core_t::TTime>(j),
                                              person, TOptionalDouble()),
                                     m_Gatherer);
                }
            }
            if (i == anomalyBuckets[anomaly]) {
                ++anomaly;
            }
            model->sample(time, time + bucketLength, m_ResourceMonitor);

            for (std::size_t pid = 0u; pid < samples[i].size(); ++pid) {
                SAnnotatedProbability p;
                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                BOOST_TEST_REQUIRE(model->computeProbability(
                    pid, time, time + bucketLength, partitioningFields, 1, p));
                std::string correlated;
                if (p.s_AttributeProbabilities[0].s_CorrelatedAttributes.size() > 0 &&
                    p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0] != nullptr &&
                    p.s_AttributeProbabilities[0].s_Type.isUnconditional() == false) {
                    correlated = *p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0];
                }
                probabilities[pid].add(TDoubleSizeStrTr(p.s_Probability, i, correlated));
            }
            time += bucketLength;
        }

        TStrVecVec expectedResults{
            {"100,", "190,"}, {"100,", "190,"}, {"160,", "190,"}, {"160,", "190,"}};
        for (std::size_t i = 0u; i < probabilities.size(); ++i) {
            LOG_DEBUG(<< "probabilities = " << probabilities[i].print());
            TStrVec results;
            for (const auto& result : probabilities[i]) {
                results.push_back(core::CStringUtils::typeToString(result.second) +
                                  "," + result.third);
            }
            for (const auto& expectedResult : expectedResults[i]) {
                BOOST_TEST_REQUIRE(std::find(results.begin(), results.end(),
                                             expectedResult) != results.end());
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testCorrelatedTrend, CTestFixture) {
    // Check we find the correct correlated variables, and identify
    // correlate and marginal anomalies.

    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{600};

    test::CRandomNumbers rng;
    rng.discard(200000);

    const std::size_t numberBuckets{2880};
    const TDoubleVec means{20.0, 25.0, 50.0, 100.0};
    const TDoubleVecVec covariances{{20.0, 10.0, 0.0, 0.0},
                                    {10.0, 30.0, 0.0, 0.0},
                                    {0.0, 0.0, 40.0, -30.0},
                                    {0.0, 0.0, -30.0, 40.0}};
    const TDoubleVecVec trends{
        {0.0, 0.0, 0.0,  1.0, 1.0, 2.0, 4.0, 10.0, 11.0, 10.0, 8.0, 8.0,
         7.0, 9.0, 12.0, 4.0, 3.0, 1.0, 1.0, 0.0,  0.0,  0.0,  0.0, 0.0},
        {0.0,  0.0,  0.0,  2.0, 2.0, 4.0, 8.0, 15.0, 18.0, 14.0, 12.0, 12.0,
         11.0, 10.0, 16.0, 7.0, 4.0, 2.0, 1.0, 0.0,  0.0,  0.0,  0.0,  0.0},
        {4.0,   3.0,   5.0,   20.0,  20.0,  40.0,  80.0,  150.0,
         180.0, 140.0, 120.0, 120.0, 110.0, 100.0, 160.0, 70.0,
         40.0,  20.0,  10.0,  3.0,   5.0,   2.0,   1.0,   3.0},
        {0.0,   0.0,   0.0,   20.0,  20.0,  40.0,  80.0,  150.0,
         180.0, 140.0, 120.0, 120.0, 110.0, 100.0, 160.0, 70.0,
         40.0,  40.0,  30.0,  20.0,  10.0,  0.0,   0.0,   0.0},
    };

    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(means, covariances, numberBuckets, samples);

    TSizeVec anomalyBuckets{1950, 2400, 2700, numberBuckets};
    TDoubleVecVec anomalies{
        {-23.9, 19.7, 0.0, 0.0}, {0.0, 0.0, 36.4, 36.4}, {-28.7, 30.4, 36.4, 36.4}};
    TMinAccumulatorVec probabilities{TMinAccumulator{4}, TMinAccumulator{4},
                                     TMinAccumulator{4}, TMinAccumulator{4}};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.0002;
    params.s_LearnRate = 1.0;
    params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
    params.s_MinimumModeCount = 24.0;
    params.s_MultivariateByFields = true;
    this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 4);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    core_t::TTime time{startTime};
    for (std::size_t i = 0, anomaly = 0; i < numberBuckets; ++i) {
        std::size_t hour1{static_cast<std::size_t>((time / 3600) % 24)};
        std::size_t hour2{(hour1 + 1) % 24};
        double dt{static_cast<double>(time % 3600) / 3600.0};

        for (std::size_t j = 0; j < samples[i].size(); ++j) {
            std::string person{std::string("p") + core::CStringUtils::typeToString(j + 1)};
            double n{std::max((1.0 - dt) * trends[j][hour1] +
                                  dt * trends[j][hour2] + samples[i][j] +
                                  (i == anomalyBuckets[anomaly] ? anomalies[anomaly][j] : 0.0),
                              0.0) /
                     3.0};
            for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) {
                this->addArrival(SMessage(time + static_cast<core_t::TTime>(j),
                                          person, TOptionalDouble()),
                                 m_Gatherer);
            }
        }
        if (i == anomalyBuckets[anomaly]) {
            ++anomaly;
        }
        model->sample(time, time + bucketLength, m_ResourceMonitor);

        for (std::size_t pid = 0; pid < samples[i].size(); ++pid) {
            SAnnotatedProbability p;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                pid, time, time + bucketLength, partitioningFields, 1, p));
            std::string correlated;
            if (p.s_AttributeProbabilities[0].s_CorrelatedAttributes.size() > 0 &&
                p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0] != nullptr &&
                p.s_AttributeProbabilities[0].s_Type.isUnconditional() == false) {
                correlated = *p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0];
            }
            probabilities[pid].add(TDoubleSizeStrTr(p.s_Probability, i, correlated));
        }
        time += bucketLength;
    }

    TStrVecVec expectedResults{{"1950,p2", "2700,p2"},
                               {"1950,p1", "2700,p1"},
                               {"2400,p4", "2700,p4"},
                               {"2400,p3", "2700,p3"}};
    for (std::size_t i = 0; i < 4; ++i) {
        LOG_DEBUG(<< "probabilities = " << probabilities[i].print());
        TStrVec results;
        for (const auto& result : probabilities[i]) {
            results.push_back(core::CStringUtils::typeToString(result.second) +
                              "," + result.third);
        }
        for (const auto& expectedResult : expectedResults[i]) {
            BOOST_TEST_REQUIRE(std::find(results.begin(), results.end(),
                                         expectedResult) != results.end());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testPrune, CTestFixture) {
    using TUInt64VecVec = std::vector<TUInt64Vec>;
    using TEventDataVec = std::vector<CEventData>;
    using TSizeSizeMap = std::map<std::size_t, std::size_t>;

    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};

    const TStrVec people{"p1", "p2", "p3", "p4", "p5", "p6"};

    TUInt64VecVec eventCounts;
    eventCounts.push_back(TUInt64Vec(1000u, 0));
    eventCounts[0][0] = 4;
    eventCounts[0][1] = 3;
    eventCounts[0][2] = 5;
    eventCounts[0][4] = 2;
    eventCounts.push_back(TUInt64Vec(1000u, 1));
    eventCounts.push_back(TUInt64Vec(1000u, 0));
    eventCounts[2][1] = 10;
    eventCounts[2][2] = 13;
    eventCounts[2][8] = 5;
    eventCounts[2][15] = 2;
    eventCounts.push_back(TUInt64Vec(1000u, 0));
    eventCounts[3][2] = 13;
    eventCounts[3][8] = 9;
    eventCounts[3][15] = 12;
    eventCounts.push_back(TUInt64Vec(1000u, 2));
    eventCounts.push_back(TUInt64Vec(1000u, 1));

    TSizeVec expectedPeople{1, 4, 5};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.01;
    model_t::TFeatureVec features;
    features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
    features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CModelFactory::TModelPtr model_;
    this->makeModelT<CEventRateModelFactory>(
        params, features, startTime, model_t::E_EventRateOnline, gatherer, model_);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
    BOOST_TEST_REQUIRE(model);
    CModelFactory::TDataGathererPtr expectedGatherer;
    CModelFactory::TModelPtr expectedModel_;
    this->makeModelT<CEventRateModelFactory>(params, features, startTime,
                                             model_t::E_EventRateOnline,
                                             expectedGatherer, expectedModel_);
    CEventRateModel* expectedModel =
        dynamic_cast<CEventRateModel*>(expectedModel_.get());
    BOOST_TEST_REQUIRE(expectedModel);

    TEventDataVec events;
    for (std::size_t i = 0u; i < eventCounts.size(); ++i) {
        TTimeVec eventTimes;
        generateEvents(startTime, bucketLength, eventCounts[i], eventTimes);
        if (eventTimes.size() > 0) {
            std::sort(eventTimes.begin(), eventTimes.end());
            std::size_t pid = this->addPerson(people[i], gatherer);
            for (auto time : eventTimes) {
                events.push_back(makeEventData(time, pid));
            }
        }
    }
    std::sort(events.begin(), events.end(), [](const CEventData& lhs, const CEventData& rhs) {
        return lhs.time() < rhs.time();
    });

    TEventDataVec expectedEvents;
    expectedEvents.reserve(events.size());
    TSizeSizeMap mapping;
    for (auto person : expectedPeople) {
        mapping[person] = this->addPerson(people[person], expectedGatherer);
    }
    for (const auto& event : events) {
        if (std::binary_search(expectedPeople.begin(), expectedPeople.end(),
                               event.personId())) {
            expectedEvents.push_back(
                makeEventData(event.time(), mapping[*event.personId()]));
        }
    }
    for (auto person : expectedPeople) {
        this->addPerson(people[person], expectedGatherer);
    }

    core_t::TTime bucketStart = startTime;
    for (const auto& event : events) {
        while (event.time() >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        this->addArrival(SMessage(event.time(),
                                  gatherer->personName(event.personId().get()),
                                  TOptionalDouble()),
                         gatherer);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune(model->defaultPruneWindow());
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    BOOST_REQUIRE_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = maths::CIntegerTools::floor(expectedEvents[0].time(), bucketLength);
    for (const auto& event : expectedEvents) {
        while (event.time() >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        this->addArrival(SMessage(event.time(),
                                  expectedGatherer->personName(event.personId().get()),
                                  TOptionalDouble()),
                         expectedGatherer);
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person slots.

    bucketStart = gatherer->currentBucketStartTime() + bucketLength;
    TStrVec newPeople{"p7", "p8", "p9"};
    for (const auto& person : newPeople) {
        std::size_t newPid = this->addPerson(person, gatherer);
        BOOST_TEST_REQUIRE(newPid < 6);
        std::size_t expectedNewPid = this->addPerson(person, expectedGatherer);

        this->addArrival(SMessage(bucketStart + 1, gatherer->personName(newPid),
                                  TOptionalDouble()),
                         gatherer);
        this->addArrival(SMessage(bucketStart + 2000,
                                  gatherer->personName(newPid), TOptionalDouble()),
                         gatherer);
        this->addArrival(SMessage(bucketStart + 1, expectedGatherer->personName(expectedNewPid),
                                  TOptionalDouble()),
                         expectedGatherer);
        this->addArrival(SMessage(bucketStart + 2000, expectedGatherer->personName(expectedNewPid),
                                  TOptionalDouble()),
                         expectedGatherer);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Test that calling prune on a cloned model which has seen no new data does nothing.
    CModelFactory::TModelPtr clonedModel(model->cloneForPersistence());
    std::size_t numberOfPeopleBeforePrune(clonedModel->dataGatherer().numberActivePeople());
    BOOST_TEST_REQUIRE(numberOfPeopleBeforePrune > 0);
    clonedModel->prune(clonedModel->defaultPruneWindow());
    BOOST_REQUIRE_EQUAL(numberOfPeopleBeforePrune,
                        clonedModel->dataGatherer().numberActivePeople());
}

BOOST_FIXTURE_TEST_CASE(testKey, CTestFixture) {
    function_t::TFunctionVec countFunctions{function_t::E_IndividualCount,
                                            function_t::E_IndividualNonZeroCount,
                                            function_t::E_IndividualRareCount,
                                            function_t::E_IndividualRareNonZeroCount,
                                            function_t::E_IndividualRare,
                                            function_t::E_IndividualLowCounts,
                                            function_t::E_IndividualHighCounts};

    std::string fieldName;
    std::string overFieldName;

    generateAndCompareKey(countFunctions, fieldName, overFieldName,
                          [](CSearchKey expectedKey, CSearchKey actualKey) {
                              BOOST_TEST_REQUIRE(expectedKey == actualKey);
                          });
}

BOOST_FIXTURE_TEST_CASE(testModelsWithValueFields, CTestFixture) {
    // Check that attributeConditional features are correctly
    // marked as such:
    // Create some models with attribute conditional data and
    // check that the values vary accordingly

    LOG_DEBUG(<< "*** testModelsValueFields ***");
    {
        // check E_PopulationUniqueCountByBucketPersonAndAttribute
        std::size_t anomalousBucket{20u};
        std::size_t numberBuckets{30u};

        TStrVec strings{"p1", "c1", "c2"};
        TSizeVecVecVec fieldsPerBucket;

        for (std::size_t i = 0u; i < numberBuckets; i++) {
            TSizeVecVec fields;
            std::size_t attribute1Strings = 10;
            std::size_t attribute2Strings = 10;
            if (i == anomalousBucket) {
                attribute1Strings = 5;
                attribute2Strings = 15;
            }

            for (std::size_t j = 0u;
                 j < std::max(attribute1Strings, attribute2Strings); j++) {
                std::ostringstream ss1;
                std::ostringstream ss2;
                ss1 << "one_plus_" << i << "_" << j;
                ss2 << "two_plus_" << j;
                strings.push_back(ss1.str());
                strings.push_back(ss2.str());

                if (j < attribute1Strings) {
                    TSizeVec f;
                    f.push_back(0);
                    f.push_back(1);
                    f.push_back(strings.size() - 2);
                    fields.push_back(f);
                }

                if (j < attribute2Strings) {
                    TSizeVec f;
                    f.push_back(0);
                    f.push_back(2);
                    f.push_back(strings.size() - 1);
                    fields.push_back(f);
                }
            }
            fieldsPerBucket.push_back(fields);
        }
        testModelWithValueField(model_t::E_PopulationUniqueCountByBucketPersonAndAttribute,
                                fieldsPerBucket, strings, m_ResourceMonitor);
    }
    {
        // Check E_PopulationInfoContentByBucketPersonAndAttribute
        std::size_t anomalousBucket{20u};
        std::size_t numberBuckets{30u};

        TStrVec strings{"p1",
                        "c1",
                        "c2",
                        "trwh5jks9djadkn453hgfadadfjhadhfkdhakj4hkahdlagl4iuygalshkdjbvlaus4hliu4WHGFLIUSDHLKAJ",
                        "2H4G55HALFMN569DNIVJ55B3BSJXU;4VBQ-LKDFNUE9HNV904U5QGA;DDFLVJKF95NSD,MMVASD.,A.4,A.SD4",
                        "a",
                        "b"};

        TSizeVecVecVec fieldsPerBucket;

        for (std::size_t i = 0u; i < numberBuckets; i++) {
            TSizeVecVec fields;

            TSizeVec fb;

            if (i == anomalousBucket) {
                // Load "c1" with "a" and "b"
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(1);
                fields.back().push_back(5);
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(1);
                fields.back().push_back(6);

                // Load "c2" with the random strings
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(2);
                fields.back().push_back(3);
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(2);
                fields.back().push_back(4);
            } else {
                // Load "c1" and "c2" with similarly random strings
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(1);
                fields.back().push_back(3);
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(1);
                fields.back().push_back(6);

                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(2);
                fields.back().push_back(4);
                fields.push_back(TSizeVec());
                fields.back().push_back(0);
                fields.back().push_back(2);
                fields.back().push_back(5);
            }

            fieldsPerBucket.push_back(fields);
        }
        testModelWithValueField(model_t::E_PopulationInfoContentByBucketPersonAndAttribute,
                                fieldsPerBucket, strings, m_ResourceMonitor);
    }
}

BOOST_FIXTURE_TEST_CASE(testCountProbabilityCalculationWithInfluence, CTestFixture) {
    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};

    {
        // Test single influence name, single influence value
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "", influenceFieldNames);
        factory.features({model_t::E_IndividualCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(
                    SMessage(eventTimes[i], "p", {}, {}, TOptionalStr("inf1")), gatherer);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // All the influence should be assigned to our one influencer
        BOOST_REQUIRE_EQUAL(std::string("[((IF1, inf1), 1)]"),
                            core::CContainerPrinter::print(lastInfluencersResult));
    }
    {
        // Test single influence name, two influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "", influenceFieldNames);
        factory.features({model_t::E_IndividualCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf" << (i % 2);
                const std::string inf(ss.str());
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, inf), gatherer);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // We expect equal influence since the influencers share the count.
        // Also the count would be fairly normal if either influencer were
        // removed so their influence is high.
        BOOST_REQUIRE_EQUAL(2, lastInfluencersResult.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(lastInfluencersResult[0].second,
                                     lastInfluencersResult[1].second, 0.05);
        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.75);
    }
    {
        // Test single influence name, two influence values, low influence
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "", influenceFieldNames);
        factory.features({model_t::E_IndividualCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 6;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf" << (i % 2);
                const std::string inf(ss.str());
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, inf), gatherer);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // We expect equal influence since the influencers share the count.
        // However, the bucket is still significantly anomalous omitting
        // the records from either influencer so their influence is smaller.
        BOOST_REQUIRE_EQUAL(2, lastInfluencersResult.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(lastInfluencersResult[0].second,
                                     lastInfluencersResult[1].second, 0.05);
        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.5);
        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second < 0.6);
    }
    {
        // Test single influence name, two asymmetric influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "", influenceFieldNames);
        factory.features({model_t::E_IndividualCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf";
                if (i % 10 == 0) {
                    ss << "_extra";
                }
                const std::string inf(ss.str());
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, inf), gatherer);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer, and the
        // _extra influencers should be dropped by the cutoff threshold
        BOOST_REQUIRE_EQUAL(1, lastInfluencersResult.size());
        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.99);
    }
    {
        // Test two influence names, two asymmetric influence values in each
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1", "IF2"};
        factory.fieldNames("", "", "", "", influenceFieldNames);
        factory.features({model_t::E_IndividualCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 2));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf";
                if (i % 10 == 0) {
                    ss << "_extra";
                }
                const std::string inf1(ss.str());
                ss << "_another";
                const std::string inf2(ss.str());
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, inf1, inf2), gatherer);

                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer for both fields,
        // and the _extra influencers should be dropped by the cutoff threshold
        BOOST_REQUIRE_EQUAL(2, lastInfluencersResult.size());
        BOOST_REQUIRE_EQUAL(std::string("IF1"), *lastInfluencersResult[0].first.first);
        BOOST_REQUIRE_EQUAL(std::string("inf"), *lastInfluencersResult[0].first.second);
        BOOST_REQUIRE_EQUAL(std::string("IF2"), *lastInfluencersResult[1].first.first);
        BOOST_REQUIRE_EQUAL(std::string("inf_another"),
                            *lastInfluencersResult[1].first.second);

        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.99);
        BOOST_TEST_REQUIRE(lastInfluencersResult[1].second > 0.99);
    }
    {
        // The influencer is one of the partitioning fields.
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        std::string byFieldName{"P"};
        factory.fieldNames("", "", byFieldName, "", {byFieldName});
        factory.features({model_t::E_IndividualCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        core_t::TTime bucketStartTime = startTime;
        core_t::TTime bucketEndTime = startTime + bucketLength;
        for (std::size_t i = 0u, j = 0u; bucketStartTime < endTime;
             bucketStartTime += bucketLength, bucketEndTime += bucketLength, ++j) {

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(
                    SMessage(eventTimes[i], "p", {}, {}, TOptionalStr("p")), gatherer);
                count += 1.0;
            }

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);
        }

        // Check we still have influences for an empty bucket.

        model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

        SAnnotatedProbability annotatedProbability;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        partitioningFields.add(byFieldName, EMPTY_STRING);
        BOOST_TEST_REQUIRE(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        LOG_DEBUG(<< "influencers = "
                  << core::CContainerPrinter::print(annotatedProbability.s_Influences));
        BOOST_REQUIRE_EQUAL(false, annotatedProbability.s_Influences.empty());
    }
}

BOOST_FIXTURE_TEST_CASE(testDistinctCountProbabilityCalculationWithInfluence, CTestFixture) {
    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};

    {
        // Test single influence name, single influence value
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        factory.features({model_t::E_IndividualUniqueCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1, TOptionalStr("v")));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, TOptionalStr("inf1"),
                                          TOptionalStr(uniqueValue)),
                                 gatherer);
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 0; k < 20; k++) {
                    std::stringstream ss;
                    ss << uniqueValue << "_" << k;
                    this->addArrival(SMessage(eventTimes[i - 1], "p", {}, {},
                                              TOptionalStr("inf1"),
                                              TOptionalStr(ss.str())),
                                     gatherer);
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // All the influence should be assigned to our one influencer
        BOOST_REQUIRE_EQUAL(std::string("[((IF1, inf1), 1)]"),
                            core::CContainerPrinter::print(lastInfluencersResult));
    }
    {
        // Test single influence name, two influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        factory.features({model_t::E_IndividualUniqueCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1, TOptionalStr("v")));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, TOptionalStr("inf1"),
                                          TOptionalStr(uniqueValue)),
                                 gatherer);
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 1; k < 20; k++) {
                    std::stringstream ss;
                    ss << uniqueValue << "_" << k;
                    CEventData d = makeEventData(eventTimes[i - 1], 0, {}, ss.str());
                    if (k % 2 == 0) {
                        this->addArrival(SMessage(eventTimes[i - 1], "p", {},
                                                  {}, TOptionalStr("inf1"),
                                                  TOptionalStr(ss.str())),
                                         gatherer);
                    } else {
                        this->addArrival(SMessage(eventTimes[i - 1], "p", {},
                                                  {}, TOptionalStr("inf2"),
                                                  TOptionalStr(ss.str())),
                                         gatherer);
                    }
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be shared by the two influencers, and as the anomaly
        // is about twice the regular count, each influencer contributes a lot to
        // the anomaly
        BOOST_REQUIRE_EQUAL(2, lastInfluencersResult.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(lastInfluencersResult[0].second,
                                     lastInfluencersResult[1].second, 0.05);
        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.6);
    }
    {
        // Test single influence name, two asymmetric influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1"};
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        factory.features({model_t::E_IndividualUniqueCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 1, TOptionalStr("v")));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(SMessage(eventTimes[i], "p", {}, {}, TOptionalStr("inf1"),
                                          TOptionalStr(uniqueValue)),
                                 gatherer);
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 1; k < 20; k++) {
                    std::stringstream ss;
                    ss << uniqueValue << "_" << k;
                    if (k == 1) {
                        this->addArrival(SMessage(eventTimes[i - 1], "p", {},
                                                  {}, TOptionalStr("inf2"),
                                                  TOptionalStr(ss.str())),
                                         gatherer);
                    } else {
                        this->addArrival(SMessage(eventTimes[i - 1], "p", {},
                                                  {}, TOptionalStr("inf1"),
                                                  TOptionalStr(ss.str())),
                                         gatherer);
                    }
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer, and the
        // _extra influencer should be dropped by the cutoff threshold
        BOOST_REQUIRE_EQUAL(1, lastInfluencersResult.size());
        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.8);
    }
    {
        // Test two influence names, two asymmetric influence values in each
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        TStrVec influenceFieldNames{"IF1", "IF2"};
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        factory.features({model_t::E_IndividualUniqueCountByBucketAndPerson});
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gatherer, 2, TOptionalStr("v")));
        CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        BOOST_TEST_REQUIRE(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i{0u};
        std::size_t j{0u};
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count{0.0};
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                this->addArrival(
                    SMessage(eventTimes[i], "p", {}, {}, TOptionalStr("inf1"),
                             TOptionalStr("inf1"), TOptionalStr(uniqueValue)),
                    gatherer);
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 1; k < 22; k++) {
                    std::stringstream ss1;
                    ss1 << uniqueValue << "_" << k;
                    std::stringstream ss2;
                    ss2 << "inf";
                    if (i % 10 == 0) {
                        ss2 << "_extra";
                    }
                    const std::string inf1(ss2.str());
                    ss2 << "_another";
                    const std::string inf2(ss2.str());
                    LOG_DEBUG(<< "Inf1 = " << inf1);
                    LOG_DEBUG(<< "Inf2 = " << inf2);
                    LOG_DEBUG(<< "Value = " << ss1.str());
                    this->addArrival(SMessage(eventTimes[i - 1], "p", {}, {},
                                              TOptionalStr(inf1), TOptionalStr(inf2),
                                              TOptionalStr(ss1.str())),
                                     gatherer);
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            BOOST_TEST_REQUIRE(model->computeProbability(
                0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields,
                1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            BOOST_TEST_REQUIRE(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer for both fields, and the
        // _extra influencers should be dropped by the cutoff threshold
        BOOST_REQUIRE_EQUAL(2, lastInfluencersResult.size());
        BOOST_REQUIRE_EQUAL(std::string("IF1"), *lastInfluencersResult[0].first.first);
        BOOST_REQUIRE_EQUAL(std::string("inf"), *lastInfluencersResult[0].first.second);
        BOOST_REQUIRE_EQUAL(std::string("IF2"), *lastInfluencersResult[1].first.first);
        BOOST_REQUIRE_EQUAL(std::string("inf_another"),
                            *lastInfluencersResult[1].first.second);

        BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.8);
        BOOST_TEST_REQUIRE(lastInfluencersResult[1].second > 0.8);
    }
}

BOOST_FIXTURE_TEST_CASE(testRareWithInfluence, CTestFixture) {
    const core_t::TTime startTime{1346968800};
    const core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CEventRateModelFactory factory(params, interimBucketCorrector);
    TStrVec influenceFieldNames{"IF1"};
    factory.fieldNames("", "", "", "", influenceFieldNames);
    factory.features(function_t::features(function_t::E_IndividualRare));
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
    BOOST_REQUIRE_EQUAL(0, this->addPerson("p1", gatherer, 1));
    BOOST_REQUIRE_EQUAL(1, this->addPerson("p2", gatherer, 1));
    BOOST_REQUIRE_EQUAL(2, this->addPerson("p3", gatherer, 1));
    BOOST_REQUIRE_EQUAL(3, this->addPerson("p4", gatherer, 1));
    BOOST_REQUIRE_EQUAL(4, this->addPerson("p5", gatherer, 1));
    CModelFactory::TModelPtr modelHolder(factory.makeModel(gatherer));
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
    BOOST_TEST_REQUIRE(model);

    SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;

    core_t::TTime time{startTime};

    for (/**/; time < startTime + 50 * bucketLength; time += bucketLength) {
        this->addArrival(
            SMessage(time + bucketLength / 2, "p1", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p2", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p3", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p4", {}, {}, TOptionalStr("inf1")), gatherer);
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }

    {
        this->addArrival(
            SMessage(time + bucketLength / 2, "p1", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p2", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p3", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p4", {}, {}, TOptionalStr("inf1")), gatherer);
        this->addArrival(
            SMessage(time + bucketLength / 2, "p5", {}, {}, TOptionalStr("inf2")), gatherer);
    }
    model->sample(time, time + bucketLength, m_ResourceMonitor);

    TDoubleVec probabilities;
    for (std::size_t pid = 0u; pid < 5; ++pid) {
        SAnnotatedProbability annotatedProbability;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        BOOST_TEST_REQUIRE(model->computeProbability(
            pid, time, time + bucketLength, partitioningFields, 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        LOG_DEBUG(<< "influencers = "
                  << core::CContainerPrinter::print(annotatedProbability.s_Influences));
        lastInfluencersResult = annotatedProbability.s_Influences;
        probabilities.push_back(annotatedProbability.s_Probability);
    }

    // We expect "p1 = p2 = p3 = p4 >> p5".
    BOOST_REQUIRE_EQUAL(5, probabilities.size());
    BOOST_REQUIRE_EQUAL(probabilities[0], probabilities[1]);
    BOOST_REQUIRE_EQUAL(probabilities[1], probabilities[2]);
    BOOST_REQUIRE_EQUAL(probabilities[2], probabilities[3]);
    BOOST_TEST_REQUIRE(probabilities[3] > 50.0 * probabilities[4]);

    // Expect the influence for this anomaly to be "INF1":"inf2"
    LOG_DEBUG(<< core::CContainerPrinter::print(lastInfluencersResult));
    BOOST_REQUIRE_EQUAL(1, lastInfluencersResult.size());
    BOOST_TEST_REQUIRE(lastInfluencersResult[0].second > 0.75);
    BOOST_REQUIRE_EQUAL(std::string("IF1"), *lastInfluencersResult[0].first.first);
    BOOST_REQUIRE_EQUAL(std::string("inf2"), *lastInfluencersResult[0].first.second);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        model->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    CModelFactory::TModelPtr restoredModelPtr(factory.makeModel(gatherer, traverser));

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModelPtr->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testSkipSampling, CTestFixture) {
    core_t::TTime startTime{100};
    std::size_t bucketLength{100};
    std::size_t maxAgeBuckets{5};

    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    model_t::EFeature feature{model_t::E_IndividualCountByBucketAndPerson};
    model_t::TFeatureVec features{feature};
    CModelFactory::TDataGathererPtr gathererNoGap;
    CModelFactory::TModelPtr modelNoGap_;
    this->makeModelT<CEventRateModelFactory>(params, features, startTime,
                                             model_t::E_EventRateOnline,
                                             gathererNoGap, modelNoGap_);
    CEventRateModel* modelNoGap = dynamic_cast<CEventRateModel*>(modelNoGap_.get());
    for (std::size_t i = 0u; i < 2; ++i) {
        BOOST_REQUIRE_EQUAL(
            i, this->addPerson("p" + core::CStringUtils::typeToString(i + 1), gathererNoGap));
    }

    // p1: |1|1|1|
    // p2: |1|0|0|
    this->addArrival(SMessage(100, "p1", TOptionalDouble()), gathererNoGap);
    this->addArrival(SMessage(100, "p2", TOptionalDouble()), gathererNoGap);
    modelNoGap->sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", TOptionalDouble()), gathererNoGap);
    modelNoGap->sample(200, 300, m_ResourceMonitor);
    this->addArrival(SMessage(300, "p1", TOptionalDouble()), gathererNoGap);
    modelNoGap->sample(300, 400, m_ResourceMonitor);

    CModelFactory::TDataGathererPtr gathererWithGap;
    CModelFactory::TModelPtr modelWithGap_;
    this->makeModelT<CEventRateModelFactory>(params, features, startTime,
                                             model_t::E_EventRateOnline,
                                             gathererWithGap, modelWithGap_);
    CEventRateModel* modelWithGap = dynamic_cast<CEventRateModel*>(modelWithGap_.get());
    for (std::size_t i = 0u; i < 2; ++i) {
        BOOST_REQUIRE_EQUAL(
            i, this->addPerson("p" + core::CStringUtils::typeToString(i + 1), gathererWithGap));
    }

    // p1: |1|1|0|0|0|0|0|0|0|0|1|1|
    // p1: |1|X|X|X|X|X|X|X|X|X|1|1| -> equal to |1|1|1|
    // p2: |1|1|0|0|0|0|0|0|0|0|0|0|
    // p2: |1|X|X|X|X|X|X|X|X|X|0|0| -> equal to |1|0|0|
    // where X means skipped bucket

    this->addArrival(SMessage(100, "p1", TOptionalDouble()), gathererWithGap);
    this->addArrival(SMessage(100, "p2", TOptionalDouble()), gathererWithGap);
    modelWithGap->sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", TOptionalDouble()), gathererWithGap);
    this->addArrival(SMessage(200, "p2", TOptionalDouble()), gathererWithGap);
    modelWithGap->skipSampling(1000);
    LOG_DEBUG(<< "Calling sample over skipped interval should do nothing except print some ERRORs");
    modelWithGap->sample(200, 1000, m_ResourceMonitor);

    // Check prune does not remove people because last seen times are updated by adding gap duration
    modelWithGap->prune(maxAgeBuckets);
    BOOST_REQUIRE_EQUAL(2, gathererWithGap->numberActivePeople());

    this->addArrival(SMessage(1000, "p1", TOptionalDouble()), gathererWithGap);
    modelWithGap->sample(1000, 1100, m_ResourceMonitor);
    this->addArrival(SMessage(1100, "p1", TOptionalDouble()), gathererWithGap);
    modelWithGap->sample(1100, 1200, m_ResourceMonitor);

    // Check priors are the same
    BOOST_REQUIRE_EQUAL(static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelWithGap->details()->model(feature, 0))
                            ->residualModel()
                            .checksum(),
                        static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelNoGap->details()->model(feature, 0))
                            ->residualModel()
                            .checksum());
    BOOST_REQUIRE_EQUAL(static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelWithGap->details()->model(feature, 1))
                            ->residualModel()
                            .checksum(),
                        static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelNoGap->details()->model(feature, 1))
                            ->residualModel()
                            .checksum());

    // Confirm last seen times are only updated by gap duration by forcing p2 to be pruned
    modelWithGap->sample(1200, 1500, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 500 and since it's equal to maxAge it should still be here
    BOOST_REQUIRE_EQUAL(2, gathererWithGap->numberActivePeople());
    modelWithGap->sample(1500, 1600, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 600 so it should get pruned
    BOOST_REQUIRE_EQUAL(1, gathererWithGap->numberActivePeople());
}

BOOST_FIXTURE_TEST_CASE(testExplicitNulls, CTestFixture) {
    core_t::TTime startTime{100};
    std::size_t bucketLength{100};
    std::string summaryCountField{"count"};

    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    model_t::EFeature feature{model_t::E_IndividualNonZeroCountByBucketAndPerson};
    model_t::TFeatureVec features{feature};
    CModelFactory::TDataGathererPtr gathererSkipGap;
    CModelFactory::TModelPtr modelSkipGap_;
    this->makeModelT<CEventRateModelFactory>(params, features, startTime,
                                             model_t::E_EventRateOnline, gathererSkipGap,
                                             modelSkipGap_, {}, summaryCountField);
    CEventRateModel* modelSkipGap = dynamic_cast<CEventRateModel*>(modelSkipGap_.get());

    // The idea here is to compare a model that has a gap skipped against a model
    // that has explicit nulls for the buckets that sampling was skipped.

    // p1: |1|1|1|X|X|1|
    // p2: |1|1|0|X|X|0|
    this->addArrival(SMessage(100, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererSkipGap);
    this->addArrival(SMessage(100, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap->sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererSkipGap);
    this->addArrival(SMessage(200, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap->sample(200, 300, m_ResourceMonitor);
    this->addArrival(SMessage(300, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap->sample(300, 400, m_ResourceMonitor);
    modelSkipGap->skipSampling(600);
    this->addArrival(SMessage(600, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererSkipGap);
    modelSkipGap->sample(600, 700, m_ResourceMonitor);

    CModelFactory::TDataGathererPtr gathererExNull;
    CModelFactory::TModelPtr modelExNullGap_;
    this->makeModelT<CEventRateModelFactory>(params, features, startTime,
                                             model_t::E_EventRateOnline, gathererExNull,
                                             modelExNullGap_, {}, summaryCountField);
    CEventRateModel* modelExNullGap =
        dynamic_cast<CEventRateModel*>(modelExNullGap_.get());

    // p1: |1,"",null|1|1|null|null|1|
    // p2: |1,""|1|0|null|null|0|
    this->addArrival(SMessage(100, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("null")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererExNull);
    this->addArrival(SMessage(100, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("")),
                     gathererExNull);
    modelExNullGap->sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererExNull);
    this->addArrival(SMessage(200, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererExNull);
    modelExNullGap->sample(200, 300, m_ResourceMonitor);
    this->addArrival(SMessage(300, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererExNull);
    modelExNullGap->sample(300, 400, m_ResourceMonitor);
    this->addArrival(SMessage(400, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("null")),
                     gathererExNull);
    this->addArrival(SMessage(400, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("null")),
                     gathererExNull);
    modelExNullGap->sample(400, 500, m_ResourceMonitor);
    this->addArrival(SMessage(500, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("null")),
                     gathererExNull);
    this->addArrival(SMessage(500, "p2", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("null")),
                     gathererExNull);
    modelExNullGap->sample(500, 600, m_ResourceMonitor);
    this->addArrival(SMessage(600, "p1", {}, {}, TOptionalStr(), TOptionalStr(),
                              TOptionalStr("1")),
                     gathererExNull);
    modelExNullGap->sample(600, 700, m_ResourceMonitor);

    // Check priors are the same
    BOOST_REQUIRE_EQUAL(static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelExNullGap->details()->model(feature, 0))
                            ->residualModel()
                            .checksum(),
                        static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelSkipGap->details()->model(feature, 0))
                            ->residualModel()
                            .checksum());
    BOOST_REQUIRE_EQUAL(static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelExNullGap->details()->model(feature, 1))
                            ->residualModel()
                            .checksum(),
                        static_cast<const maths::CUnivariateTimeSeriesModel*>(
                            modelSkipGap->details()->model(feature, 1))
                            ->residualModel()
                            .checksum());
}

BOOST_FIXTURE_TEST_CASE(testInterimCorrections, CTestFixture) {
    core_t::TTime startTime{3600};
    core_t::TTime bucketLength{3600};
    core_t::TTime endTime{2 * 24 * bucketLength};
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    params.s_MultibucketFeaturesWindowLength = 0;
    this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 3);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());
    CCountingModel countingModel(params, m_Gatherer, m_InterimBucketCorrector);

    test::CRandomNumbers rng;
    core_t::TTime now{startTime};
    TDoubleVec samples(3, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, 3, samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[1] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p2", TOptionalDouble()), m_Gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[2] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p3", TOptionalDouble()), m_Gatherer);
        }
        countingModel.sample(now, now + bucketLength, m_ResourceMonitor);
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
    }
    for (std::size_t i = 0; i < 1; ++i) {
        this->addArrival(SMessage(now, "p2", TOptionalDouble()), m_Gatherer);
    }
    for (std::size_t i = 0; i < 100; ++i) {
        this->addArrival(SMessage(now, "p3", TOptionalDouble()), m_Gatherer);
    }
    countingModel.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);

    model_t::CResultType type(model_t::CResultType::E_Unconditional |
                              model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    BOOST_TEST_REQUIRE(model->computeProbability(
        0 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    BOOST_TEST_REQUIRE(model->computeProbability(
        1 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    BOOST_TEST_REQUIRE(model->computeProbability(
        2 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability3));

    TDouble1Vec p1Baseline = model->baselineBucketMean(
        model_t::E_IndividualCountByBucketAndPerson, 0, 0, type, NO_CORRELATES, now);
    TDouble1Vec p2Baseline = model->baselineBucketMean(
        model_t::E_IndividualCountByBucketAndPerson, 1, 0, type, NO_CORRELATES, now);
    TDouble1Vec p3Baseline = model->baselineBucketMean(
        model_t::E_IndividualCountByBucketAndPerson, 2, 0, type, NO_CORRELATES, now);

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
    BOOST_TEST_REQUIRE(p2Baseline[0] > 43.0);
    BOOST_TEST_REQUIRE(p2Baseline[0] < 47.0);
    BOOST_TEST_REQUIRE(p3Baseline[0] > 57.0);
    BOOST_TEST_REQUIRE(p3Baseline[0] < 62.0);

    for (std::size_t i = 0; i < 25; ++i) {
        this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
    }
    for (std::size_t i = 0; i < 59; ++i) {
        this->addArrival(SMessage(now, "p2", TOptionalDouble()), m_Gatherer);
    }
    for (std::size_t i = 0; i < 100; ++i) {
        this->addArrival(SMessage(now, "p3", TOptionalDouble()), m_Gatherer);
    }
    countingModel.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    BOOST_TEST_REQUIRE(model->computeProbability(
        0 /*pid*/, now, now + bucketLength, partitioningFields, 0, annotatedProbability1));
    BOOST_TEST_REQUIRE(model->computeProbability(
        1 /*pid*/, now, now + bucketLength, partitioningFields, 0, annotatedProbability2));
    BOOST_TEST_REQUIRE(model->computeProbability(
        2 /*pid*/, now, now + bucketLength, partitioningFields, 0, annotatedProbability3));

    p1Baseline = model->baselineBucketMean(model_t::E_IndividualCountByBucketAndPerson,
                                           0, 0, type, NO_CORRELATES, now);
    p2Baseline = model->baselineBucketMean(model_t::E_IndividualCountByBucketAndPerson,
                                           1, 0, type, NO_CORRELATES, now);
    p3Baseline = model->baselineBucketMean(model_t::E_IndividualCountByBucketAndPerson,
                                           2, 0, type, NO_CORRELATES, now);

    LOG_DEBUG(<< "p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG(<< "p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG(<< "p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG(<< "p1 baseline = " << p1Baseline[0]);
    LOG_DEBUG(<< "p2 baseline = " << p2Baseline[0]);
    LOG_DEBUG(<< "p3 baseline = " << p3Baseline[0]);

    BOOST_TEST_REQUIRE(annotatedProbability1.s_Probability > 0.75);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability > 0.9);
    BOOST_TEST_REQUIRE(annotatedProbability3.s_Probability < 0.05);
    BOOST_TEST_REQUIRE(p1Baseline[0] > 58.0);
    BOOST_TEST_REQUIRE(p1Baseline[0] < 62.0);
    BOOST_TEST_REQUIRE(p2Baseline[0] > 58.0);
    BOOST_TEST_REQUIRE(p2Baseline[0] < 62.0);
    BOOST_TEST_REQUIRE(p3Baseline[0] > 58.0);
    BOOST_TEST_REQUIRE(p3Baseline[0] < 62.0);
}

BOOST_FIXTURE_TEST_CASE(testInterimCorrectionsWithCorrelations, CTestFixture) {
    core_t::TTime startTime{3600};
    core_t::TTime bucketLength{3600};

    SModelParams params(bucketLength);
    params.s_MultivariateByFields = true;
    this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 3);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());

    core_t::TTime now{startTime};
    core_t::TTime endTime{now + 2 * 24 * bucketLength};
    test::CRandomNumbers rng;
    TDoubleVec samples(1, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(80.0, 100.0, 1, samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 10.5); ++i) {
            this->addArrival(SMessage(now, "p2", TOptionalDouble()), m_Gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] - 9.5); ++i) {
            this->addArrival(SMessage(now, "p3", TOptionalDouble()), m_Gatherer);
        }
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 9; ++i) {
        this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
    }
    for (std::size_t i = 0; i < 10; ++i) {
        this->addArrival(SMessage(now, "p2", TOptionalDouble()), m_Gatherer);
    }
    for (std::size_t i = 0; i < 8; ++i) {
        this->addArrival(SMessage(now, "p3", TOptionalDouble()), m_Gatherer);
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model_t::CResultType type(model_t::CResultType::E_Conditional |
                              model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    BOOST_TEST_REQUIRE(model->computeProbability(
        0 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    BOOST_TEST_REQUIRE(model->computeProbability(
        1 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    BOOST_TEST_REQUIRE(model->computeProbability(
        2 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability3));

    TDouble1Vec p1Baseline = model->baselineBucketMean(
        model_t::E_IndividualCountByBucketAndPerson, 0, 0, type,
        annotatedProbability1.s_AttributeProbabilities[0].s_Correlated, now);
    TDouble1Vec p2Baseline = model->baselineBucketMean(
        model_t::E_IndividualCountByBucketAndPerson, 1, 0, type,
        annotatedProbability2.s_AttributeProbabilities[0].s_Correlated, now);
    TDouble1Vec p3Baseline = model->baselineBucketMean(
        model_t::E_IndividualCountByBucketAndPerson, 2, 0, type,
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

BOOST_FIXTURE_TEST_CASE(testSummaryCountZeroRecordsAreIgnored, CTestFixture) {
    core_t::TTime startTime{100};
    core_t::TTime bucketLength{100};

    SModelParams params(bucketLength);
    std::string summaryCountField{"count"};

    CModelFactory::TDataGathererPtr gathererWithZeros;
    CModelFactory::TModelPtr modelWithZerosPtr;
    this->makeModelT<CEventRateModelFactory>(
        params, {model_t::E_IndividualCountByBucketAndPerson}, startTime,
        model_t::E_EventRateOnline, gathererWithZeros, modelWithZerosPtr, {},
        summaryCountField);
    CEventRateModel& modelWithZeros = static_cast<CEventRateModel&>(*modelWithZerosPtr);

    CModelFactory::TDataGathererPtr gathererNoZeros;
    CModelFactory::TModelPtr modelNoZerosPtr;
    this->makeModelT<CEventRateModelFactory>(
        params, {model_t::E_IndividualCountByBucketAndPerson}, startTime,
        model_t::E_EventRateOnline, gathererNoZeros, modelNoZerosPtr, {}, summaryCountField);
    CEventRateModel& modelNoZeros = static_cast<CEventRateModel&>(*modelNoZerosPtr);

    // The idea here is to compare a model that has records with summary count of zero
    // against a model that has no records at all where the first model had the zero-count records.

    core_t::TTime now{100};
    core_t::TTime end{now + 50 * bucketLength};
    test::CRandomNumbers rng;
    TSizeVec samples;
    TDoubleVec zeroCountProbability;
    std::string summaryCountZero{"0"};
    std::string summaryCountOne{"1"};
    while (now < end) {
        rng.generateUniformSamples(1, 10, 1, samples);
        rng.generateUniformSamples(0.0, 1.0, 1, zeroCountProbability);
        for (std::size_t i = 0; i < samples[0]; ++i) {
            if (zeroCountProbability[0] < 0.2) {
                this->addArrival(SMessage(now, "p1", {}, {}, TOptionalStr(),
                                          TOptionalStr(), TOptionalStr(summaryCountZero)),
                                 gathererWithZeros);
            } else {
                this->addArrival(SMessage(now, "p1", {}, {}, TOptionalStr(),
                                          TOptionalStr(), TOptionalStr(summaryCountOne)),
                                 gathererWithZeros);
                this->addArrival(SMessage(now, "p1", {}, {}, TOptionalStr(),
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

BOOST_FIXTURE_TEST_CASE(testComputeProbabilityGivenDetectionRule, CTestFixture) {
    CRuleCondition condition;
    condition.appliesTo(CRuleCondition::E_Actual);
    condition.op(CRuleCondition::E_LT);
    condition.value(100.0);
    CDetectionRule rule;
    rule.addCondition(condition);

    core_t::TTime startTime{3600};
    core_t::TTime bucketLength{3600};
    core_t::TTime endTime{24 * bucketLength};

    SModelParams params(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    params.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);
    this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, startTime, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(m_Model.get());

    test::CRandomNumbers rng;
    core_t::TTime now = startTime;
    TDoubleVec samples(1, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, 1, samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
        }
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        this->addArrival(SMessage(now, "p1", TOptionalDouble()), m_Gatherer);
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);

    SAnnotatedProbability annotatedProbability;
    BOOST_TEST_REQUIRE(model->computeProbability(
        0 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(annotatedProbability.s_Probability, 1.0, 0.00001);
}

BOOST_FIXTURE_TEST_CASE(testDecayRateControl, CTestFixture) {
    core_t::TTime startTime{0};
    core_t::TTime bucketLength{1800};

    model_t::EFeature feature = model_t::E_IndividualCountByBucketAndPerson;
    model_t::TFeatureVec features{feature};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "*** Test anomaly ***");
    {
        // Test we don't adapt the decay rate if there is a short-lived
        // anomaly. We should get essentially identical prediction errors
        // with and without decay control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        factory.features(features);
        CModelFactory::TDataGathererPtr gatherer{factory.makeDataGatherer(startTime)};
        CModelFactory::TModelPtr model{factory.makeModel(gatherer)};
        this->addPerson("p1", gatherer);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CEventRateModelFactory referenceFactory(params, interimBucketCorrector);
        referenceFactory.features(features);
        CModelFactory::TDataGathererPtr referenceGatherer{
            referenceFactory.makeDataGatherer(startTime)};
        CModelFactory::TModelPtr referenceModel{referenceFactory.makeModel(referenceGatherer)};
        this->addPerson("p1", referenceGatherer);

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                  model_t::CResultType::E_Interim);
        for (core_t::TTime t = 0; t < 4 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG(<< "week " << t / core::constants::WEEK + 1);
            }

            TDoubleVec rate;
            rng.generateUniformSamples(0.0, 10.0, 1, rate);
            rate[0] += 20.0 * (t > 3 * core::constants::WEEK &&
                                       t < core::constants::WEEK + 4 * 3600
                                   ? 1.0
                                   : 0.0);
            for (std::size_t i = 0u; i < static_cast<std::size_t>(rate[0]); ++i) {
                this->addArrival(
                    SMessage(t + bucketLength / 2, "p1", TOptionalDouble()), gatherer);
                this->addArrival(SMessage(t + bucketLength / 2, "p1", TOptionalDouble()),
                                 referenceGatherer);
            }
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
            maths::CBasicStatistics::mean(meanPredictionError), 0.01);
    }

    LOG_DEBUG(<< "*** Test linear scaling ***");
    {
        // This change point is amongst those we explicitly detect so
        // check we get similar detection performance with and without
        // decay rate control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CEventRateModelFactory factory(params, interimBucketCorrector);
        factory.features(features);
        CModelFactory::TDataGathererPtr gatherer{factory.makeDataGatherer(startTime)};
        CModelFactory::TModelPtr model{factory.makeModel(gatherer)};
        this->addPerson("p1", gatherer);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CEventRateModelFactory referenceFactory(params, interimBucketCorrector);
        referenceFactory.features(features);
        CModelFactory::TDataGathererPtr referenceGatherer{factory.makeDataGatherer(startTime)};
        CModelFactory::TModelPtr referenceModel{factory.makeModel(gatherer)};
        this->addPerson("p1", referenceGatherer);

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                  model_t::CResultType::E_Interim);
        for (core_t::TTime t = 0; t < 10 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG(<< "week " << t / core::constants::WEEK + 1);
            }

            double rate = 10.0 *
                          (1.0 + std::sin(boost::math::double_constants::two_pi *
                                          static_cast<double>(t) /
                                          static_cast<double>(core::constants::DAY))) *
                          (t < 5 * core::constants::WEEK ? 1.0 : 2.0);
            TDoubleVec noise;
            rng.generateUniformSamples(0.0, 3.0, 1, noise);
            for (std::size_t i = 0u; i < static_cast<std::size_t>(rate + noise[0]); ++i) {
                this->addArrival(
                    SMessage(t + bucketLength / 2, "p1", TOptionalDouble()), gatherer);
                this->addArrival(SMessage(t + bucketLength / 2, "p1", TOptionalDouble()),
                                 referenceGatherer);
            }
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
        CEventRateModelFactory factory(params, interimBucketCorrector);
        factory.features(features);
        CModelFactory::TDataGathererPtr gatherer{factory.makeDataGatherer(startTime)};
        CModelFactory::TModelPtr model{factory.makeModel(gatherer)};
        this->addPerson("p1", gatherer);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.0005;
        CEventRateModelFactory referenceFactory(params, interimBucketCorrector);
        referenceFactory.features(features);
        CModelFactory::TDataGathererPtr referenceGatherer{
            referenceFactory.makeDataGatherer(startTime)};
        CModelFactory::TModelPtr referenceModel{referenceFactory.makeModel(gatherer)};
        this->addPerson("p1", referenceGatherer);

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional |
                                  model_t::CResultType::E_Interim);
        for (core_t::TTime t = 0; t < 20 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG(<< "week " << t / core::constants::WEEK + 1);
            }

            double rate = 10.0 *
                          (1.0 + std::sin(boost::math::double_constants::two_pi *
                                          static_cast<double>(t) /
                                          static_cast<double>(core::constants::DAY))) *
                          (1.0 + std::sin(boost::math::double_constants::two_pi *
                                          static_cast<double>(t) / 10.0 /
                                          static_cast<double>(core::constants::WEEK)));
            TDoubleVec noise;
            rng.generateUniformSamples(0.0, 3.0, 1, noise);
            for (std::size_t i = 0u; i < static_cast<std::size_t>(rate + noise[0]); ++i) {
                this->addArrival(
                    SMessage(t + bucketLength / 2, "p1", TOptionalDouble()), gatherer);
                this->addArrival(SMessage(t + bucketLength / 2, "p1", TOptionalDouble()),
                                 referenceGatherer);
            }
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

BOOST_FIXTURE_TEST_CASE(testIgnoreSamplingGivenDetectionRules, CTestFixture) {
    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.

    // Create a rule to filter buckets where the count > 100
    CRuleCondition condition;
    condition.appliesTo(CRuleCondition::E_Actual);
    condition.op(CRuleCondition::E_GT);
    condition.value(100.0);
    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipModelUpdate);
    rule.addCondition(condition);

    std::size_t bucketLength{100};
    std::size_t startTime{100};

    // Model without the skip sampling rule
    SModelParams paramsNoRules(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CEventRateModelFactory factory(paramsNoRules, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_IndividualCountByBucketAndPerson};
    factory.features(features);
    CModelFactory::TDataGathererPtr gathererNoSkip{factory.makeDataGatherer(startTime)};
    CModelFactory::TModelPtr modelPtrNoSkip{factory.makeModel(gathererNoSkip)};
    CEventRateModel* modelNoSkip = dynamic_cast<CEventRateModel*>(modelPtrNoSkip.get());
    this->addPerson("p1", gathererNoSkip);

    // Model with the skip sampling rule
    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);
    CEventRateModelFactory factoryWithSkip(paramsWithRules, interimBucketCorrector);
    factoryWithSkip.features(features);
    CModelFactory::TDataGathererPtr gathererWithSkip{
        factoryWithSkip.makeDataGatherer(startTime)};
    CModelFactory::TModelPtr modelPtrWithSkip{factoryWithSkip.makeModel(gathererWithSkip)};
    CEventRateModel* modelWithSkip =
        dynamic_cast<CEventRateModel*>(modelPtrWithSkip.get());
    this->addPerson("p1", gathererWithSkip);

    std::size_t endTime = startTime + bucketLength;

    // Add a bucket to both models
    for (int i = 0; i < 66; ++i) {
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererNoSkip);
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererWithSkip);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // Add a bucket to both models
    for (int i = 0; i < 55; ++i) {
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererNoSkip);
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererWithSkip);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // this sample will be skipped by the detection rule
    for (int i = 0; i < 110; ++i) {
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererWithSkip);
    }
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    startTime = endTime;
    endTime += bucketLength;

    // Wind the other model forward
    modelNoSkip->skipSampling(startTime);

    for (int i = 0; i < 55; ++i) {
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererNoSkip);
        this->addArrival(SMessage(startTime, "p1", TOptionalDouble()), gathererWithSkip);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    // Checksums will be different due to the data gatherers
    BOOST_TEST_REQUIRE(modelWithSkip->checksum() != modelNoSkip->checksum());

    // but the underlying models should be the same
    CAnomalyDetectorModel::TModelDetailsViewUPtr modelWithSkipView =
        modelWithSkip->details();
    CAnomalyDetectorModel::TModelDetailsViewUPtr modelNoSkipView = modelNoSkip->details();

    std::uint64_t withSkipChecksum{
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum()};
    std::uint64_t noSkipChecksum{
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum()};
    BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    // Check the last value times of the underlying models are the same
    const maths::CUnivariateTimeSeriesModel* timeSeriesModel =
        dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0));
    BOOST_TEST_REQUIRE(timeSeriesModel);
    const auto* trendModel = dynamic_cast<const maths::CTimeSeriesDecomposition*>(
        &timeSeriesModel->trendModel());
    BOOST_TEST_REQUIRE(trendModel);

    core_t::TTime time = trendModel->lastValueTime();
    BOOST_REQUIRE_EQUAL(model_t::sampleTime(model_t::E_IndividualCountByBucketAndPerson,
                                            startTime, bucketLength),
                        time);

    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0));
    BOOST_TEST_REQUIRE(timeSeriesModel);
    trendModel = dynamic_cast<const maths::CTimeSeriesDecomposition*>(
        &timeSeriesModel->trendModel());
    BOOST_TEST_REQUIRE(trendModel);

    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());
}

BOOST_AUTO_TEST_SUITE_END()
