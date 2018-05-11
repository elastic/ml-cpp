/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CEventRateModelTest.h"

#include <core/CContainerPrinter.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CIntegerTools.h>
#include <maths/CModelWeight.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPrior.h>
#include <maths/CTimeSeriesDecompositionInterface.h>

#include <model/CAnnotatedProbability.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CEventRateModel.h>
#include <model/CEventRateModelFactory.h>
#include <model/CEventRatePopulationModel.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CModelDetailsView.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>
#include <model/CRuleCondition.h>

#include <test/CRandomNumbers.h>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/range.hpp>

#include <memory>
#include <string>
#include <vector>

#include <stdint.h>

using namespace ml;
using namespace model;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TUInt64Vec = std::vector<uint64_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TSizeVecVecVec = std::vector<TSizeVecVec>;
using TStrVec = std::vector<std::string>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
using TOptionalStr = boost::optional<std::string>;
using TOptionalUInt64 = boost::optional<uint64_t>;
using TOptionalDouble = boost::optional<double>;
using TOptionalDoubleVec = std::vector<TOptionalDouble>;
using TMathsModelPtr = std::shared_ptr<maths::CModel>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

const std::string EMPTY_STRING;

TUInt64Vec rawEventCounts(std::size_t copies = 1) {
    uint64_t counts[] = {54, 67, 39, 58, 46, 50, 42,
                         48, 53, 51, 50, 57, 53, 49};
    TUInt64Vec result;
    for (std::size_t i = 0u; i < copies; ++i) {
        result.insert(result.end(), boost::begin(counts), boost::end(counts));
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

std::size_t addPerson(const std::string& p,
                      const CModelFactory::TDataGathererPtr& gatherer,
                      CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    CEventData result;
    gatherer->processFields(person, result, resourceMonitor);
    return *result.personId();
}

std::size_t addPersonWithInfluence(const std::string& p,
                                   const CModelFactory::TDataGathererPtr& gatherer,
                                   CResourceMonitor& resourceMonitor,
                                   std::size_t numInfluencers,
                                   TOptionalStr value = TOptionalStr()) {
    std::string i("i");
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    for (std::size_t j = 0; j < numInfluencers; ++j) {
        person.push_back(&i);
    }
    if (value) {
        person.push_back(&(value.get()));
    }
    CEventData result;
    gatherer->processFields(person, result, resourceMonitor);
    return *result.personId();
}

void makeModel(CEventRateModelFactory& factory,
               const CDataGatherer::TFeatureVec& features,
               CResourceMonitor& resourceMonitor,
               core_t::TTime startTime,
               core_t::TTime bucketLength,
               CModelFactory::TDataGathererPtr& gatherer,
               CAnomalyDetectorModel::TModelPtr& model,
               std::size_t numberPeople) {
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    gatherer.reset(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData initData(gatherer);
    model.reset(factory.makeModel(initData));
    CPPUNIT_ASSERT(model);
    CPPUNIT_ASSERT_EQUAL(bucketLength, model->bucketLength());
    for (std::size_t i = 0u; i < numberPeople; ++i) {
        CPPUNIT_ASSERT_EQUAL(std::size_t(i),
                             addPerson("p" + core::CStringUtils::typeToString(i + 1),
                                       gatherer, resourceMonitor));
    }
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                const TOptionalStr& inf1 = TOptionalStr(),
                const TOptionalStr& inf2 = TOptionalStr(),
                const TOptionalStr& value = TOptionalStr()) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    if (inf1) {
        fieldValues.push_back(&(inf1.get()));
    }
    if (inf2) {
        fieldValues.push_back(&(inf2.get()));
    }

    if (value) {
        fieldValues.push_back(&(value.get()));
    }

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

CEventData makeEventData(core_t::TTime time, std::size_t pid) {
    CEventData eventData;
    eventData.time(time);
    eventData.person(pid);
    eventData.addAttribute(std::size_t(0));
    eventData.addValue(TDoubleVec(1, 0.0));
    return eventData;
}

CEventData makeEventData(core_t::TTime time, std::size_t pid, const std::string value) {
    CEventData eventData;
    eventData.time(time);
    eventData.person(pid);
    eventData.addAttribute(std::size_t(0));
    eventData.addValue(TDoubleVec(1, 0.0));
    eventData.stringValue(value);
    return eventData;
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

    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    CEventRatePopulationModelFactory factory(params);
    model_t::TFeatureVec features(1u, feature);
    factory.features(features);
    factory.fieldNames("", "", "P", "V", TStrVec());
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));
    CPPUNIT_ASSERT(model);

    std::size_t anomalousBucket = 20;
    std::size_t numberBuckets = 30;

    const core_t::TTime endTime = startTime + (numberBuckets * bucketLength);

    std::size_t i = 0u;
    for (core_t::TTime bucketStartTime = startTime; bucketStartTime < endTime;
         bucketStartTime += bucketLength, i++) {
        core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

        for (std::size_t j = 0; j < fields[i].size(); ++j) {
            CDataGatherer::TStrCPtrVec f;
            f.push_back(&strings[fields[i][j][0]]);
            f.push_back(&strings[fields[i][j][1]]);
            f.push_back(&strings[fields[i][j][2]]);
            handleEvent(f, bucketStartTime + j, gatherer, resourceMonitor);
        }

        model->sample(bucketStartTime, bucketEndTime, resourceMonitor);

        SAnnotatedProbability annotatedProbability;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        model->computeProbability(0 /*pid*/, bucketStartTime, bucketEndTime,
                                  partitioningFields, 1, annotatedProbability);
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        if (i == anomalousBucket) {
            CPPUNIT_ASSERT(annotatedProbability.s_Probability < 0.001);
        } else {
            CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.6);
        }
    }
}

const TSizeDoublePr1Vec NO_CORRELATES;

} // unnamed::

void CEventRateModelTest::testOnlineCountSample() {
    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
    CPPUNIT_ASSERT(model);

    TMathsModelPtr timeseriesModel{factory.defaultFeatureModel(
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

    std::size_t i = 0u, j = 0u;
    for (core_t::TTime bucketStartTime = startTime; bucketStartTime < endTime;
         bucketStartTime += bucketLength, ++j) {
        core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

        double count = 0.0;
        for (/**/; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
            addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p1");
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
        CPPUNIT_ASSERT_EQUAL(expectedEventCounts[j],
                             static_cast<uint64_t>(model->currentBucketValue(
                                 model_t::E_IndividualCountByBucketAndPerson, 0,
                                 0, bucketStartTime)[0]));
        CPPUNIT_ASSERT_EQUAL(timeseriesModel->checksum(),
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
    CPPUNIT_ASSERT(origXml.size() < 41000);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr restoredModelPtr(factory.makeModel(initData, traverser));

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModelPtr->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    LOG_DEBUG(<< "original checksum = " << model->checksum(false));
    LOG_DEBUG(<< "restored checksum = " << restoredModelPtr->checksum(false));
    CPPUNIT_ASSERT_EQUAL(model->checksum(false), restoredModelPtr->checksum(false));
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CEventRateModelTest::testOnlineNonZeroCountSample() {
    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualNonZeroCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
    CPPUNIT_ASSERT(model);

    TMathsModelPtr timeseriesModel{factory.defaultFeatureModel(
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

    std::size_t i = 0u, j = 0u;
    for (core_t::TTime bucketStartTime = startTime; bucketStartTime < endTime;
         bucketStartTime += bucketLength) {
        core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

        double count = 0.0;
        for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
            addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p1");
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
            CPPUNIT_ASSERT_EQUAL(expectedEventCounts[j],
                                 static_cast<uint64_t>(model->currentBucketValue(
                                     model_t::E_IndividualNonZeroCountByBucketAndPerson,
                                     0, 0, bucketStartTime)[0]));
            CPPUNIT_ASSERT_EQUAL(timeseriesModel->checksum(),
                                 model->details()
                                     ->model(model_t::E_IndividualNonZeroCountByBucketAndPerson, 0)
                                     ->checksum());

            ++j;
        }
    }
}

void CEventRateModelTest::testOnlineRare() {
    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features;
    features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
    features.push_back(model_t::E_IndividualIndicatorOfBucketPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 5);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    core_t::TTime time = startTime;
    for (/**/; time < startTime + 10 * bucketLength; time += bucketLength) {
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p1");
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p2");
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }
    for (/**/; time < startTime + 50 * bucketLength; time += bucketLength) {
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p1");
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p2");
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p3");
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p4");
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }

    addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p1");
    addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p2");
    addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p3");
    addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p4");
    addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p5");
    model->sample(time, time + bucketLength, m_ResourceMonitor);

    TDoubleVec probabilities;
    for (std::size_t pid = 0u; pid < 5; ++pid) {
        SAnnotatedProbability annotatedProbability;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        CPPUNIT_ASSERT(model->computeProbability(pid, time, time + bucketLength, partitioningFields,
                                                 0, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);
    }

    // We expect "p1 = p2 > p3 = p4 >> p5".
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), probabilities.size());
    CPPUNIT_ASSERT_EQUAL(probabilities[0], probabilities[1]);
    CPPUNIT_ASSERT(probabilities[1] > probabilities[2]);
    CPPUNIT_ASSERT_EQUAL(probabilities[2], probabilities[3]);
    CPPUNIT_ASSERT(probabilities[3] > 50.0 * probabilities[4]);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        model->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "origXml = " << origXml);
    LOG_DEBUG(<< "size = " << origXml.size());
    CPPUNIT_ASSERT(origXml.size() < 22000);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr restoredModelPtr(factory.makeModel(initData, traverser));

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModelPtr->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CEventRateModelTest::testOnlineProbabilityCalculation() {
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr>;

    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;
    const std::size_t anomalousBucket = 25u;

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    TMinAccumulator minProbabilities(2u);

    // Generate some events.
    TTimeVec eventTimes;
    TUInt64Vec expectedEventCounts = rawEventCounts(2);
    expectedEventCounts[anomalousBucket] *= 3;
    generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
    core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
    LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
              << ", # events = " << eventTimes.size());

    std::size_t i = 0u, j = 0u;
    for (core_t::TTime bucketStartTime = startTime; bucketStartTime < endTime;
         bucketStartTime += bucketLength, ++j) {
        core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

        double count = 0.0;
        for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
            addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p1");
            count += 1.0;
        }

        LOG_DEBUG(<< "bucket count = " << count);

        model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

        SAnnotatedProbability p;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        CPPUNIT_ASSERT(model->computeProbability(
            0 /*pid*/, bucketStartTime, bucketEndTime, partitioningFields, 1, p));
        LOG_DEBUG(<< "probability = " << p.s_Probability);
        minProbabilities.add(TDoubleSizePr(p.s_Probability, j));
    }

    minProbabilities.sort();
    LOG_DEBUG(<< "minProbabilities = " << core::CContainerPrinter::print(minProbabilities));
    CPPUNIT_ASSERT_EQUAL(anomalousBucket, minProbabilities[0].second);
    CPPUNIT_ASSERT(minProbabilities[0].first / minProbabilities[1].first < 0.1);
}

void CEventRateModelTest::testOnlineProbabilityCalculationForLowNonZeroCount() {
    core_t::TTime startTime(0);
    core_t::TTime bucketLength(100);
    std::size_t lowNonZeroCountBucket = 6u;
    std::size_t highNonZeroCountBucket = 8u;

    std::size_t bucketCounts[] = {50, 50, 50, 50, 50,  0,  0,
                                  0,  50, 1,  50, 100, 50, 50};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualLowNonZeroCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    TDoubleVec probabilities;

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        LOG_DEBUG(<< "Writing " << bucketCounts[i] << " values");

        for (std::size_t j = 0u; j < bucketCounts[i]; ++j) {
            addArrival(*gatherer, m_ResourceMonitor,
                       time + static_cast<core_t::TTime>(j), "p1");
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
    CPPUNIT_ASSERT_EQUAL(std::size_t(11), probabilities.size());
    CPPUNIT_ASSERT(probabilities[lowNonZeroCountBucket] < 0.06);
    CPPUNIT_ASSERT(probabilities[highNonZeroCountBucket] > 0.9);
}

void CEventRateModelTest::testOnlineProbabilityCalculationForHighNonZeroCount() {
    core_t::TTime startTime(0);
    core_t::TTime bucketLength(100);
    std::size_t lowNonZeroCountBucket = 6u;
    std::size_t highNonZeroCountBucket = 8u;

    std::size_t bucketCounts[] = {50, 50, 50,  50, 50, 0,  0,
                                  0,  50, 100, 50, 1,  50, 50};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualHighNonZeroCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    TDoubleVec probabilities;

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        LOG_DEBUG(<< "Writing " << bucketCounts[i] << " values");

        for (std::size_t j = 0u; j < bucketCounts[i]; ++j) {
            addArrival(*gatherer, m_ResourceMonitor,
                       time + static_cast<core_t::TTime>(j), "p1");
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
    CPPUNIT_ASSERT_EQUAL(std::size_t(11), probabilities.size());
    CPPUNIT_ASSERT(probabilities[lowNonZeroCountBucket] < 0.06);
    CPPUNIT_ASSERT(probabilities[highNonZeroCountBucket] > 0.9);
}

void CEventRateModelTest::testOnlineCorrelatedNoTrend() {
    // Check we find the correct correlated variables, and identify
    // correlate and marginal anomalies.

    using TDoubleSizeStrTr = core::CTriple<double, std::size_t, std::string>;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizeStrTr>;

    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;

    test::CRandomNumbers rng;

    const std::size_t b = 200;
    const double means_[] = {20.0, 25.0, 100.0, 800.0};
    const double covariances_[][4] = {{3.0, 2.5, 0.0, 0.0},
                                      {2.5, 4.0, 0.0, 0.0},
                                      {0.0, 0.0, 100.0, -500.0},
                                      {0.0, 0.0, -500.0, 3000.0}};

    TDoubleVec means(&means_[0], &means_[4]);
    TDoubleVecVec covariances;
    for (std::size_t i = 0u; i < 4; ++i) {
        covariances.push_back(TDoubleVec(&covariances_[i][0], &covariances_[i][4]));
    }
    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(means, covariances, b, samples);

    {
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        params.s_LearnRate = 1.0;
        params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
        params.s_MinimumModeCount = 24.0;
        params.s_MultivariateByFields = true;
        CEventRateModelFactory factory(params);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model_;
        makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
                  gatherer, model_, 4);
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
        CPPUNIT_ASSERT(model);

        LOG_DEBUG(<< "Test correlation anomalies");

        std::size_t anomalyBuckets[] = {100, 160, 190, b};
        double anomalies[][4] = {{-5.73, 4.29, 0.0, 0.0},
                                 {0.0, 0.0, 89.99, 15.38},
                                 {-7.73, 5.59, 52.99, 9.03}};

        TMinAccumulator probabilities[4] = {TMinAccumulator(2), TMinAccumulator(2),
                                            TMinAccumulator(2), TMinAccumulator(2)};

        core_t::TTime time = startTime;
        for (std::size_t i = 0u, anomaly = 0u; i < b; ++i) {
            for (std::size_t j = 0u; j < samples[i].size(); ++j) {
                std::string person = std::string("p") +
                                     core::CStringUtils::typeToString(j + 1);
                double n = samples[i][j];
                if (i == anomalyBuckets[anomaly]) {
                    n += anomalies[anomaly][j];
                }
                for (std::size_t k = 0u; k < static_cast<std::size_t>(n); ++k) {
                    addArrival(*gatherer, m_ResourceMonitor,
                               time + static_cast<core_t::TTime>(j), person);
                }
            }
            if (i == anomalyBuckets[anomaly]) {
                ++anomaly;
            }
            model->sample(time, time + bucketLength, m_ResourceMonitor);

            for (std::size_t pid = 0u; pid < samples[i].size(); ++pid) {
                SAnnotatedProbability p;
                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                CPPUNIT_ASSERT(model->computeProbability(
                    pid, time, time + bucketLength, partitioningFields, 1, p));
                std::string correlated;
                if (p.s_AttributeProbabilities[0].s_CorrelatedAttributes.size() > 0 &&
                    p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0] != nullptr &&
                    !p.s_AttributeProbabilities[0].s_Type.isUnconditional()) {
                    correlated = *p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0];
                }
                probabilities[pid].add(TDoubleSizeStrTr(p.s_Probability, i, correlated));
            }
            time += bucketLength;
        }

        std::string expected[] = {"[(100,p2), (190,p2)]", "[(100,p1), (190,p1)]",
                                  "[(160,p4), (190,p4)]", "[(160,p3), (190,p3)]"};
        for (std::size_t i = 0u; i < boost::size(probabilities); ++i) {
            std::string actual[2];
            for (std::size_t j = 0u; j < 2; ++j) {
                actual[j] = std::string("(") +
                            core::CStringUtils::typeToString(probabilities[i][j].second) +
                            "," + probabilities[i][j].third + ")";
            }
            std::sort(actual, actual + 2);
            CPPUNIT_ASSERT_EQUAL(expected[i], core::CContainerPrinter::print(actual));
        }

        // Test persist and restore with correlate models.
        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            model->acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_TRACE(<< "origXml = " << origXml);
        LOG_DEBUG(<< "size = " << origXml.size());
        CPPUNIT_ASSERT(origXml.size() < 151000);

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        CModelFactory::SModelInitializationData initData(gatherer);
        CAnomalyDetectorModel::TModelPtr restoredModel(factory.makeModel(initData, traverser));
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredModel->acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }

        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
    {
        LOG_DEBUG(<< "Test marginal anomalies");

        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        params.s_LearnRate = 1.0;
        params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
        params.s_MinimumModeCount = 24.0;
        params.s_MultivariateByFields = true;
        CEventRateModelFactory factory(params);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model_;
        makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
                  gatherer, model_, 4);
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
        CPPUNIT_ASSERT(model);

        std::size_t anomalyBuckets[] = {100, 160, 190, b};
        double anomalies[][4] = {{11.07, 14.19, 0.0, 0.0},
                                 {0.0, 0.0, -66.9, 399.95},
                                 {11.07, 14.19, -48.15, 329.95}};

        TMinAccumulator probabilities[4] = {TMinAccumulator(2), TMinAccumulator(2),
                                            TMinAccumulator(2), TMinAccumulator(2)};

        core_t::TTime time = startTime;
        for (std::size_t i = 0u, anomaly = 0u; i < b; ++i) {
            for (std::size_t j = 0u; j < samples[i].size(); ++j) {
                std::string person = std::string("p") +
                                     core::CStringUtils::typeToString(j + 1);
                double n = samples[i][j];
                if (i == anomalyBuckets[anomaly]) {
                    n += anomalies[anomaly][j];
                }
                n = std::max(n, 0.0);
                for (std::size_t k = 0u; k < static_cast<std::size_t>(n); ++k) {
                    addArrival(*gatherer, m_ResourceMonitor,
                               time + static_cast<core_t::TTime>(j), person);
                }
            }
            if (i == anomalyBuckets[anomaly]) {
                ++anomaly;
            }
            model->sample(time, time + bucketLength, m_ResourceMonitor);

            for (std::size_t pid = 0u; pid < samples[i].size(); ++pid) {
                SAnnotatedProbability p;
                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                CPPUNIT_ASSERT(model->computeProbability(
                    pid, time, time + bucketLength, partitioningFields, 1, p));
                std::string correlated;
                if (p.s_AttributeProbabilities[0].s_CorrelatedAttributes.size() > 0 &&
                    p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0] != nullptr &&
                    !p.s_AttributeProbabilities[0].s_Type.isUnconditional()) {
                    correlated = *p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0];
                }
                probabilities[pid].add(TDoubleSizeStrTr(p.s_Probability, i, correlated));
            }
            time += bucketLength;
        }

        std::string expected[] = {"[(100,), (190,)]", "[(100,), (190,)]",
                                  "[(160,), (190,)]", "[(160,), (190,)]"};
        for (std::size_t i = 0u; i < boost::size(probabilities); ++i) {
            std::string actual[2];
            for (std::size_t j = 0u; j < 2; ++j) {
                actual[j] = std::string("(") +
                            core::CStringUtils::typeToString(probabilities[i][j].second) +
                            "," + probabilities[i][j].third + ")";
            }
            std::sort(actual, actual + 2);
            CPPUNIT_ASSERT_EQUAL(expected[i], core::CContainerPrinter::print(actual));
        }
    }
}

void CEventRateModelTest::testOnlineCorrelatedTrend() {
    // FIXME
    return;

    // Check we find the correct correlated variables, and identify
    // correlate and marginal anomalies.

    using TDoubleSizeStrTr = core::CTriple<double, std::size_t, std::string>;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizeStrTr>;

    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 600;

    test::CRandomNumbers rng;
    rng.discard(200000);

    const std::size_t b = 2880;
    const double means_[] = {20.0, 25.0, 50.0, 100.0};
    const double covariances_[][4] = {{30.0, 20.0, 0.0, 0.0},
                                      {20.0, 40.0, 0.0, 0.0},
                                      {0.0, 0.0, 60.0, -50.0},
                                      {0.0, 0.0, -50.0, 60.0}};
    double trends[][24] = {
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

    TDoubleVec means(&means_[0], &means_[4]);
    TDoubleVecVec covariances;
    for (std::size_t i = 0u; i < 4; ++i) {
        covariances.push_back(TDoubleVec(&covariances_[i][0], &covariances_[i][4]));
    }
    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(means, covariances, b, samples);

    std::size_t anomalyBuckets[] = {1950, 2400, 2700, b};
    double anomalies[][4] = {
        {-23.9, 19.7, 0.0, 0.0}, {0.0, 0.0, 36.4, 36.4}, {-28.7, 30.4, 36.4, 36.4}};
    TMinAccumulator probabilities[4] = {TMinAccumulator(2), TMinAccumulator(2),
                                        TMinAccumulator(2), TMinAccumulator(2)};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.0002;
    params.s_LearnRate = 1.0;
    params.s_MinimumModeFraction = CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
    params.s_MinimumModeCount = 24.0;
    params.s_MultivariateByFields = true;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 4);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
    CPPUNIT_ASSERT(model);

    core_t::TTime time = startTime;
    for (std::size_t i = 0u, anomaly = 0u; i < b; ++i) {
        LOG_DEBUG(<< i << ") processing bucket [" << time << ", "
                  << time + bucketLength << ")");

        std::size_t hour1 = static_cast<std::size_t>((time / 3600) % 24);
        std::size_t hour2 = (hour1 + 1) % 24;
        double dt = static_cast<double>(time % 3600) / 3600.0;

        for (std::size_t j = 0u; j < samples[i].size(); ++j) {
            std::string person = std::string("p") +
                                 core::CStringUtils::typeToString(j + 1);

            double n = (1.0 - dt) * trends[j][hour1] + dt * trends[j][hour2] +
                       samples[i][j];
            if (i == anomalyBuckets[anomaly]) {
                n += anomalies[anomaly][j];
            }
            n = std::max(n / 3.0, 0.0);
            for (std::size_t k = 0u; k < static_cast<std::size_t>(n); ++k) {
                addArrival(*gatherer, m_ResourceMonitor,
                           time + static_cast<core_t::TTime>(j), person);
            }
        }
        if (i == anomalyBuckets[anomaly]) {
            ++anomaly;
        }
        model->sample(time, time + bucketLength, m_ResourceMonitor);

        for (std::size_t pid = 0u; pid < samples[i].size(); ++pid) {
            SAnnotatedProbability p;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(pid, time, time + bucketLength,
                                                     partitioningFields, 1, p));
            std::string correlated;
            if (p.s_AttributeProbabilities[0].s_CorrelatedAttributes.size() > 0 &&
                p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0] != nullptr &&
                !p.s_AttributeProbabilities[0].s_Type.isUnconditional()) {
                correlated = *p.s_AttributeProbabilities[0].s_CorrelatedAttributes[0];
            }
            probabilities[pid].add(TDoubleSizeStrTr(p.s_Probability, i, correlated));
        }
        time += bucketLength;
    }

    std::string expected[] = {"[(1950,p2), (2700,p2)]", "[(1950,p1), (2700,p1)]",
                              "[(2400,p4), (2700,p4)]", "[(2400,p3), (2700,p3)]"};
    for (std::size_t i = 0u; i < boost::size(probabilities); ++i) {
        LOG_DEBUG(<< probabilities[i].print());
        std::string actual[2];
        for (std::size_t j = 0u; j < 2; ++j) {
            actual[j] = std::string("(") +
                        core::CStringUtils::typeToString(probabilities[i][j].second) +
                        "," + probabilities[i][j].third + ")";
        }
        std::sort(actual, actual + 2);
        CPPUNIT_ASSERT_EQUAL(expected[i], core::CContainerPrinter::print(actual));
    }
}

void CEventRateModelTest::testPrune() {
    using TUInt64VecVec = std::vector<TUInt64Vec>;
    using TEventDataVec = std::vector<CEventData>;
    using TSizeSizeMap = std::map<std::size_t, std::size_t>;

    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;

    const std::string people[] = {std::string("p1"), std::string("p2"),
                                  std::string("p3"), std::string("p4"),
                                  std::string("p5"), std::string("p6")};

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
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features;
    features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
    features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 0);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());
    CPPUNIT_ASSERT(model);
    CModelFactory::TDataGathererPtr expectedGatherer;
    CAnomalyDetectorModel::TModelPtr expectedModel_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              expectedGatherer, expectedModel_, 0);
    CEventRateModel* expectedModel =
        dynamic_cast<CEventRateModel*>(expectedModel_.get());
    CPPUNIT_ASSERT(expectedModel);

    TEventDataVec events;
    for (std::size_t i = 0u; i < eventCounts.size(); ++i) {
        TTimeVec eventTimes;
        generateEvents(startTime, bucketLength, eventCounts[i], eventTimes);
        if (eventTimes.size() > 0) {
            std::sort(eventTimes.begin(), eventTimes.end());
            std::size_t pid = addPerson(people[i], gatherer, m_ResourceMonitor);
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
        mapping[person] = addPerson(people[person], expectedGatherer, m_ResourceMonitor);
    }
    for (const auto& event : events) {
        if (std::binary_search(expectedPeople.begin(), expectedPeople.end(),
                               event.personId())) {
            expectedEvents.push_back(
                makeEventData(event.time(), mapping[*event.personId()]));
        }
    }
    for (auto person : expectedPeople) {
        addPerson(people[person], expectedGatherer, m_ResourceMonitor);
    }

    core_t::TTime bucketStart = startTime;
    for (const auto& event : events) {
        while (event.time() >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(*gatherer, m_ResourceMonitor, event.time(),
                   gatherer->personName(event.personId().get()));
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune(model->defaultPruneWindow());
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    CPPUNIT_ASSERT_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = maths::CIntegerTools::floor(expectedEvents[0].time(), bucketLength);
    for (const auto& event : expectedEvents) {
        while (event.time() >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(*expectedGatherer, m_ResourceMonitor, event.time(),
                   expectedGatherer->personName(event.personId().get()));
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    CPPUNIT_ASSERT_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person slots.

    bucketStart = gatherer->currentBucketStartTime() + bucketLength;
    TStrVec newPeople{"p7", "p8", "p9"};
    for (const auto& person : newPeople) {
        std::size_t newPid = addPerson(person, gatherer, m_ResourceMonitor);
        CPPUNIT_ASSERT(newPid < 6);
        std::size_t expectedNewPid = addPerson(person, expectedGatherer, m_ResourceMonitor);

        addArrival(*gatherer, m_ResourceMonitor, bucketStart + 1,
                   gatherer->personName(newPid));
        addArrival(*gatherer, m_ResourceMonitor, bucketStart + 2000,
                   gatherer->personName(newPid));
        addArrival(*expectedGatherer, m_ResourceMonitor, bucketStart + 1,
                   expectedGatherer->personName(expectedNewPid));
        addArrival(*expectedGatherer, m_ResourceMonitor, bucketStart + 2000,
                   expectedGatherer->personName(expectedNewPid));
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    CPPUNIT_ASSERT_EQUAL(expectedModel->checksum(), model->checksum());

    // Test that calling prune on a cloned model which has seen no new data does nothing.
    CAnomalyDetectorModel::TModelPtr clonedModel(model->cloneForPersistence());
    std::size_t numberOfPeopleBeforePrune(clonedModel->dataGatherer().numberActivePeople());
    CPPUNIT_ASSERT(numberOfPeopleBeforePrune > 0);
    clonedModel->prune(clonedModel->defaultPruneWindow());
    CPPUNIT_ASSERT_EQUAL(numberOfPeopleBeforePrune,
                         clonedModel->dataGatherer().numberActivePeople());
}

void CEventRateModelTest::testKey() {
    function_t::EFunction countFunctions[] = {function_t::E_IndividualCount,
                                              function_t::E_IndividualNonZeroCount,
                                              function_t::E_IndividualRareCount,
                                              function_t::E_IndividualRareNonZeroCount,
                                              function_t::E_IndividualRare,
                                              function_t::E_IndividualLowCounts,
                                              function_t::E_IndividualHighCounts};
    bool useNull[] = {true, false};
    std::string byField[] = {"", "by"};
    std::string partitionField[] = {"", "partition"};

    CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();

    int identifier = 0;
    for (std::size_t i = 0u; i < boost::size(countFunctions); ++i) {
        for (std::size_t j = 0u; j < boost::size(useNull); ++j) {
            for (std::size_t k = 0u; k < boost::size(byField); ++k) {
                for (std::size_t l = 0u; l < boost::size(partitionField); ++l) {
                    CSearchKey key(++identifier, countFunctions[i], useNull[j],
                                   model_t::E_XF_None, "", byField[k], "",
                                   partitionField[l]);

                    CAnomalyDetectorModelConfig::TModelFactoryCPtr factory =
                        config.factory(key);

                    LOG_DEBUG(<< "expected key = " << key);
                    LOG_DEBUG(<< "actual key   = " << factory->searchKey());
                    CPPUNIT_ASSERT(key == factory->searchKey());
                }
            }
        }
    }
}

void CEventRateModelTest::testModelsWithValueFields() {
    // Check that attributeConditional features are correctly
    // marked as such:
    // Create some models with attribute conditional data and
    // check that the values vary accordingly

    LOG_DEBUG(<< "*** testModelsValueFields ***");
    {
        // check E_PopulationUniqueCountByBucketPersonAndAttribute
        std::size_t anomalousBucket = 20;
        std::size_t numberBuckets = 30;

        TStrVec strings;
        strings.push_back("p1");
        strings.push_back("c1");
        strings.push_back("c2");
        TSizeVecVecVec fieldsPerBucket;

        for (std::size_t i = 0; i < numberBuckets; i++) {
            TSizeVecVec fields;
            std::size_t attribute1Strings = 10;
            std::size_t attribute2Strings = 10;
            if (i == anomalousBucket) {
                attribute1Strings = 5;
                attribute2Strings = 15;
            }

            for (std::size_t j = 0;
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
        std::size_t anomalousBucket = 20;
        std::size_t numberBuckets = 30;

        TStrVec strings;
        strings.push_back("p1");
        strings.push_back("c1");
        strings.push_back("c2");
        strings.push_back("trwh5jks9djadkn453hgfadadfjhadhfkdhakj4hkahdlagl4iuygalshkdjbvlaus4hliu4WHGFLIUSDHLKAJ");
        strings.push_back("2H4G55HALFMN569DNIVJ55B3BSJXU;4VBQ-LKDFNUE9HNV904U5QGA;DDFLVJKF95NSD,MMVASD.,A.4,A.SD4");
        strings.push_back("a");
        strings.push_back("b");

        TSizeVecVecVec fieldsPerBucket;

        for (std::size_t i = 0; i < numberBuckets; i++) {
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

void CEventRateModelTest::testCountProbabilityCalculationWithInfluence() {
    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;

    {
        // Test single influence name, single influence value
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor, 1));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p",
                           TOptionalStr("inf1"));
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // All the influence should be assigned to our one influencer
        CPPUNIT_ASSERT_EQUAL(std::string("[((IF1, inf1), 1)]"),
                             core::CContainerPrinter::print(lastInfluencersResult));
    }
    {
        // Test single influence name, two influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor, 1));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf" << (i % 2);
                const std::string inf(ss.str());
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p", inf);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be shared by the two influencers, and as the anomaly
        // is about twice the regular count, each influencer contributes a lot to
        // the anomaly
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), lastInfluencersResult.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(lastInfluencersResult[0].second,
                                     lastInfluencersResult[1].second, 0.05);
        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.8);
    }
    {
        // Test single influence name, two influence values, less anomalousness for each
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor, 1));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 6;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf" << (i % 2);
                const std::string inf(ss.str());
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p", inf);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be shared by the two influencers, and as the anomaly
        // is about twice the regular count, each influencer contributes to
        // the anomaly, but less than the previous test as each the results would
        // be anomalous even without the contribution from the influencer
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), lastInfluencersResult.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(lastInfluencersResult[0].second,
                                     lastInfluencersResult[1].second, 0.05);
        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.5);
        CPPUNIT_ASSERT(lastInfluencersResult[0].second < 0.6);
    }
    {
        // Test single influence name, two asymmetric influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor, 1));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf";
                if (i % 10 == 0) {
                    ss << "_extra";
                }
                const std::string inf(ss.str());
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p", inf);
                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer, and the
        // _extra influencer should be dropped by the cutoff threshold
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), lastInfluencersResult.size());
        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.99);
    }
    {
        // Test two influence names, two asymmetric influence values in each
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        influenceFieldNames.push_back("IF2");
        factory.fieldNames("", "", "", "", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor, 2));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        expectedEventCounts.back() *= 3;
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::stringstream ss;
                ss << "inf";
                if (i % 10 == 0) {
                    ss << "_extra";
                }
                const std::string inf1(ss.str());
                ss << "_another";
                const std::string inf2(ss.str());
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p", inf1, inf2);

                count += 1.0;
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer for both fields, and the
        // _extra influencers should be dropped by the cutoff threshold
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), lastInfluencersResult.size());
        CPPUNIT_ASSERT_EQUAL(std::string("IF1"), *lastInfluencersResult[0].first.first);
        CPPUNIT_ASSERT_EQUAL(std::string("inf"), *lastInfluencersResult[0].first.second);
        CPPUNIT_ASSERT_EQUAL(std::string("IF2"), *lastInfluencersResult[1].first.first);
        CPPUNIT_ASSERT_EQUAL(std::string("inf_another"),
                             *lastInfluencersResult[1].first.second);

        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.99);
        CPPUNIT_ASSERT(lastInfluencersResult[1].second > 0.99);
    }
}

void CEventRateModelTest::testDistinctCountProbabilityCalculationWithInfluence() {
    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;

    {
        // Test single influence name, single influence value
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualUniqueCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor,
                                                    1, TOptionalStr("v")));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p",
                           TOptionalStr("inf1"), TOptionalStr(uniqueValue));
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 0; k < 20; k++) {
                    std::stringstream ss;
                    ss << uniqueValue << "_" << k;
                    addArrival(*gatherer, m_ResourceMonitor, eventTimes[i - 1],
                               "p", TOptionalStr("inf1"), TOptionalStr(ss.str()));
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // All the influence should be assigned to our one influencer
        CPPUNIT_ASSERT_EQUAL(std::string("[((IF1, inf1), 1)]"),
                             core::CContainerPrinter::print(lastInfluencersResult));
    }
    {
        // Test single influence name, two influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualUniqueCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor,
                                                    1, TOptionalStr("v")));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p",
                           TOptionalStr("inf1"), TOptionalStr(uniqueValue));
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 1; k < 20; k++) {
                    std::stringstream ss;
                    ss << uniqueValue << "_" << k;
                    CEventData d = makeEventData(eventTimes[i - 1], 0, ss.str());
                    if (k % 2 == 0) {
                        addArrival(*gatherer, m_ResourceMonitor, eventTimes[i - 1], "p",
                                   TOptionalStr("inf1"), TOptionalStr(ss.str()));
                    } else {
                        addArrival(*gatherer, m_ResourceMonitor, eventTimes[i - 1], "p",
                                   TOptionalStr("inf2"), TOptionalStr(ss.str()));
                    }
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be shared by the two influencers, and as the anomaly
        // is about twice the regular count, each influencer contributes a lot to
        // the anomaly
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), lastInfluencersResult.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(lastInfluencersResult[0].second,
                                     lastInfluencersResult[1].second, 0.05);
        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.6);
    }
    {
        // Test single influence name, two asymmetric influence values
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualUniqueCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor,
                                                    1, TOptionalStr("v")));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p",
                           TOptionalStr("inf1"), TOptionalStr(uniqueValue));
                count += 1.0;
            }
            if (i == eventTimes.size()) {
                // Generate anomaly
                LOG_DEBUG(<< "Generating anomaly");
                for (std::size_t k = 1; k < 20; k++) {
                    std::stringstream ss;
                    ss << uniqueValue << "_" << k;
                    if (k == 1) {
                        addArrival(*gatherer, m_ResourceMonitor, eventTimes[i - 1], "p",
                                   TOptionalStr("inf2"), TOptionalStr(ss.str()));
                    } else {
                        addArrival(*gatherer, m_ResourceMonitor, eventTimes[i - 1], "p",
                                   TOptionalStr("inf1"), TOptionalStr(ss.str()));
                    }
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer, and the
        // _extra influencer should be dropped by the cutoff threshold
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), lastInfluencersResult.size());
        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.8);
    }
    {
        // Test two influence names, two asymmetric influence values in each
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        TStrVec influenceFieldNames;
        influenceFieldNames.push_back("IF1");
        influenceFieldNames.push_back("IF2");
        factory.fieldNames("", "", "", "foo", influenceFieldNames);
        model_t::TFeatureVec features(1u, model_t::E_IndividualUniqueCountByBucketAndPerson);
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPersonWithInfluence("p", gatherer, m_ResourceMonitor,
                                                    2, TOptionalStr("v")));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
        CPPUNIT_ASSERT(model);

        const std::string uniqueValue("str_value");
        // Generate some events.
        TTimeVec eventTimes;
        TUInt64Vec expectedEventCounts = rawEventCounts();
        generateEvents(startTime, bucketLength, expectedEventCounts, eventTimes);
        core_t::TTime endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
        LOG_DEBUG(<< "startTime = " << startTime << ", endTime = " << endTime
                  << ", # events = " << eventTimes.size());

        SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;
        std::size_t i = 0u, j = 0u;
        for (core_t::TTime bucketStartTime = startTime;
             bucketStartTime < endTime; bucketStartTime += bucketLength, ++j) {
            core_t::TTime bucketEndTime = bucketStartTime + bucketLength;

            double count = 0.0;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                addArrival(*gatherer, m_ResourceMonitor, eventTimes[i], "p",
                           TOptionalStr("inf1"), TOptionalStr("inf1"),
                           TOptionalStr(uniqueValue));
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
                    addArrival(*gatherer, m_ResourceMonitor, eventTimes[i - 1],
                               "p", TOptionalStr(inf1), TOptionalStr(inf2),
                               TOptionalStr(ss1.str()));
                }
            }

            LOG_DEBUG(<< "Bucket count = " << count);

            model->sample(bucketStartTime, bucketEndTime, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, bucketStartTime,
                                                     bucketEndTime, partitioningFields,
                                                     1, annotatedProbability));
            LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
            LOG_DEBUG(<< "influencers = "
                      << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            CPPUNIT_ASSERT(annotatedProbability.s_Probability);
            lastInfluencersResult = annotatedProbability.s_Influences;
        }
        // The influence should be dominated by the first influencer for both fields, and the
        // _extra influencers should be dropped by the cutoff threshold
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), lastInfluencersResult.size());
        CPPUNIT_ASSERT_EQUAL(std::string("IF1"), *lastInfluencersResult[0].first.first);
        CPPUNIT_ASSERT_EQUAL(std::string("inf"), *lastInfluencersResult[0].first.second);
        CPPUNIT_ASSERT_EQUAL(std::string("IF2"), *lastInfluencersResult[1].first.first);
        CPPUNIT_ASSERT_EQUAL(std::string("inf_another"),
                             *lastInfluencersResult[1].first.second);

        CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.8);
        CPPUNIT_ASSERT(lastInfluencersResult[1].second > 0.8);
    }
}

void CEventRateModelTest::testOnlineRareWithInfluence() {
    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    CEventRateModelFactory factory(params);
    TStrVec influenceFieldNames;
    influenceFieldNames.push_back("IF1");
    factory.fieldNames("", "", "", "", influenceFieldNames);
    model_t::TFeatureVec features = function_t::features(function_t::E_IndividualRare);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                         addPersonWithInfluence("p1", gatherer, m_ResourceMonitor, 1));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                         addPersonWithInfluence("p2", gatherer, m_ResourceMonitor, 1));
    CPPUNIT_ASSERT_EQUAL(std::size_t(2),
                         addPersonWithInfluence("p3", gatherer, m_ResourceMonitor, 1));
    CPPUNIT_ASSERT_EQUAL(std::size_t(3),
                         addPersonWithInfluence("p4", gatherer, m_ResourceMonitor, 1));
    CPPUNIT_ASSERT_EQUAL(std::size_t(4),
                         addPersonWithInfluence("p5", gatherer, m_ResourceMonitor, 1));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(modelHolder.get());
    CPPUNIT_ASSERT(model);

    SAnnotatedProbability::TStoredStringPtrStoredStringPtrPrDoublePrVec lastInfluencersResult;

    core_t::TTime time = startTime;

    for (/**/; time < startTime + 50 * bucketLength; time += bucketLength) {
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p1",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p2",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p3",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p4",
                   TOptionalStr("inf1"));
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }

    {
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p1",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p2",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p3",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p4",
                   TOptionalStr("inf1"));
        addArrival(*gatherer, m_ResourceMonitor, time + bucketLength / 2, "p5",
                   TOptionalStr("inf2"));
    }
    model->sample(time, time + bucketLength, m_ResourceMonitor);

    TDoubleVec probabilities;
    for (std::size_t pid = 0u; pid < 5; ++pid) {
        SAnnotatedProbability annotatedProbability;
        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        CPPUNIT_ASSERT(model->computeProbability(pid, time, time + bucketLength, partitioningFields,
                                                 1, annotatedProbability));
        LOG_DEBUG(<< "probability = " << annotatedProbability.s_Probability);
        LOG_DEBUG(<< "influencers = "
                  << core::CContainerPrinter::print(annotatedProbability.s_Influences));
        lastInfluencersResult = annotatedProbability.s_Influences;
        probabilities.push_back(annotatedProbability.s_Probability);
    }

    // We expect "p1 = p2 = p3 = p4 >> p5".
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), probabilities.size());
    CPPUNIT_ASSERT_EQUAL(probabilities[0], probabilities[1]);
    CPPUNIT_ASSERT_EQUAL(probabilities[1], probabilities[2]);
    CPPUNIT_ASSERT_EQUAL(probabilities[2], probabilities[3]);
    CPPUNIT_ASSERT(probabilities[3] > 50.0 * probabilities[4]);

    // Expect the influence for this anomaly to be "INF1":"inf2"
    LOG_DEBUG(<< core::CContainerPrinter::print(lastInfluencersResult));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), lastInfluencersResult.size());
    CPPUNIT_ASSERT(lastInfluencersResult[0].second > 0.75);
    CPPUNIT_ASSERT_EQUAL(std::string("IF1"), *lastInfluencersResult[0].first.first);
    CPPUNIT_ASSERT_EQUAL(std::string("inf2"), *lastInfluencersResult[0].first.second);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        model->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CAnomalyDetectorModel::TModelPtr restoredModelPtr(
        factory.makeModel(modelInitData, traverser));

    // The XML representation of the new filter should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModelPtr->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CEventRateModelTest::testSkipSampling() {
    core_t::TTime startTime(100);
    std::size_t bucketLength(100);
    std::size_t maxAgeBuckets(5);

    SModelParams params(bucketLength);
    CEventRateModelFactory factory(params);
    params.s_InitialDecayRateMultiplier = 1.0;
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gathererNoGap;
    CAnomalyDetectorModel::TModelPtr modelNoGap_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gathererNoGap, modelNoGap_, 2);
    CEventRateModel* modelNoGap = dynamic_cast<CEventRateModel*>(modelNoGap_.get());

    // p1: |1|1|1|
    // p2: |1|0|0|
    addArrival(*gathererNoGap, m_ResourceMonitor, 100, "p1");
    addArrival(*gathererNoGap, m_ResourceMonitor, 100, "p2");
    modelNoGap->sample(100, 200, m_ResourceMonitor);
    addArrival(*gathererNoGap, m_ResourceMonitor, 200, "p1");
    modelNoGap->sample(200, 300, m_ResourceMonitor);
    addArrival(*gathererNoGap, m_ResourceMonitor, 300, "p1");
    modelNoGap->sample(300, 400, m_ResourceMonitor);

    CAnomalyDetectorModel::TModelPtr modelWithGap_;
    CModelFactory::TDataGathererPtr gathererWithGap;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gathererWithGap, modelWithGap_, 2);
    CEventRateModel* modelWithGap = dynamic_cast<CEventRateModel*>(modelWithGap_.get());

    // p1: |1|1|0|0|0|0|0|0|0|0|1|1|
    // p1: |1|X|X|X|X|X|X|X|X|X|1|1| -> equal to |1|1|1|
    // p2: |1|1|0|0|0|0|0|0|0|0|0|0|
    // p2: |1|X|X|X|X|X|X|X|X|X|0|0| -> equal to |1|0|0|
    // where X means skipped bucket

    addArrival(*gathererWithGap, m_ResourceMonitor, 100, "p1");
    addArrival(*gathererWithGap, m_ResourceMonitor, 100, "p2");
    modelWithGap->sample(100, 200, m_ResourceMonitor);
    addArrival(*gathererWithGap, m_ResourceMonitor, 200, "p1");
    addArrival(*gathererWithGap, m_ResourceMonitor, 200, "p2");
    modelWithGap->skipSampling(1000);
    LOG_DEBUG(<< "Calling sample over skipped interval should do nothing except print some ERRORs");
    modelWithGap->sample(200, 1000, m_ResourceMonitor);

    // Check prune does not remove people because last seen times are updated by adding gap duration
    modelWithGap->prune(maxAgeBuckets);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), gathererWithGap->numberActivePeople());

    addArrival(*gathererWithGap, m_ResourceMonitor, 1000, "p1");
    modelWithGap->sample(1000, 1100, m_ResourceMonitor);
    addArrival(*gathererWithGap, m_ResourceMonitor, 1100, "p1");
    modelWithGap->sample(1100, 1200, m_ResourceMonitor);

    // Check priors are the same
    CPPUNIT_ASSERT_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum());
    CPPUNIT_ASSERT_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 1))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 1))
            ->residualModel()
            .checksum());

    // Confirm last seen times are only updated by gap duration by forcing p2 to be pruned
    modelWithGap->sample(1200, 1500, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 500 and since it's equal to maxAge it should still be here
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), gathererWithGap->numberActivePeople());
    modelWithGap->sample(1500, 1600, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 600 so it should get pruned
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gathererWithGap->numberActivePeople());
}

void CEventRateModelTest::testExplicitNulls() {
    core_t::TTime startTime(100);
    std::size_t bucketLength(100);
    std::string summaryCountField("count");

    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    CEventRateModelFactory factory(params, model_t::E_Manual, summaryCountField);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gathererSkipGap;
    CAnomalyDetectorModel::TModelPtr modelSkipGap_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gathererSkipGap, modelSkipGap_, 0);
    CEventRateModel* modelSkipGap = dynamic_cast<CEventRateModel*>(modelSkipGap_.get());

    // The idea here is to compare a model that has a gap skipped against a model
    // that has explicit nulls for the buckets that sampling was skipped.

    // p1: |1|1|1|X|X|1|
    // p2: |1|1|0|X|X|0|
    addArrival(*gathererSkipGap, m_ResourceMonitor, 100, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererSkipGap, m_ResourceMonitor, 100, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelSkipGap->sample(100, 200, m_ResourceMonitor);
    addArrival(*gathererSkipGap, m_ResourceMonitor, 200, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererSkipGap, m_ResourceMonitor, 200, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelSkipGap->sample(200, 300, m_ResourceMonitor);
    addArrival(*gathererSkipGap, m_ResourceMonitor, 300, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelSkipGap->sample(300, 400, m_ResourceMonitor);
    modelSkipGap->skipSampling(600);
    addArrival(*gathererSkipGap, m_ResourceMonitor, 600, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelSkipGap->sample(600, 700, m_ResourceMonitor);

    CModelFactory::TDataGathererPtr gathererExNull;
    CAnomalyDetectorModel::TModelPtr modelExNullGap_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gathererExNull, modelExNullGap_, 0);
    CEventRateModel* modelExNullGap =
        dynamic_cast<CEventRateModel*>(modelExNullGap_.get());

    // p1: |1,"",null|1|1|null|null|1|
    // p2: |1,""|1|0|null|null|0|
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr(""));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("null"));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr(""));
    modelExNullGap->sample(100, 200, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 200, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererExNull, m_ResourceMonitor, 200, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelExNullGap->sample(200, 300, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 300, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelExNullGap->sample(300, 400, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 400, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("null"));
    addArrival(*gathererExNull, m_ResourceMonitor, 400, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr("null"));
    modelExNullGap->sample(400, 500, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 500, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("null"));
    addArrival(*gathererExNull, m_ResourceMonitor, 500, "p2", TOptionalStr(),
               TOptionalStr(), TOptionalStr("null"));
    modelExNullGap->sample(500, 600, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 600, "p1", TOptionalStr(),
               TOptionalStr(), TOptionalStr("1"));
    modelExNullGap->sample(600, 700, m_ResourceMonitor);

    // Check priors are the same
    CPPUNIT_ASSERT_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelExNullGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelSkipGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum());
    CPPUNIT_ASSERT_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelExNullGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 1))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelSkipGap->details()->model(model_t::E_IndividualCountByBucketAndPerson, 1))
            ->residualModel()
            .checksum());
}

void CEventRateModelTest::testInterimCorrections() {
    core_t::TTime startTime(3600);
    core_t::TTime bucketLength(3600);
    core_t::TTime endTime(2 * 24 * bucketLength);
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 3);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    test::CRandomNumbers rng;
    core_t::TTime now = startTime;
    TDoubleVec samples(3, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, std::size_t(3), samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p1");
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[1] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p2");
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[2] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p3");
        }
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p1");
    }
    for (std::size_t i = 0; i < 1; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p2");
    }
    for (std::size_t i = 0; i < 100; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p3");
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);

    model_t::CResultType type(model_t::CResultType::E_Unconditional |
                              model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    CPPUNIT_ASSERT(model->computeProbability(1 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    CPPUNIT_ASSERT(model->computeProbability(2 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability3));

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

    CPPUNIT_ASSERT(annotatedProbability1.s_Probability > 0.05);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability < 0.05);
    CPPUNIT_ASSERT(annotatedProbability3.s_Probability < 0.05);
    CPPUNIT_ASSERT(p1Baseline[0] > 44.0 && p1Baseline[0] < 46.0);
    CPPUNIT_ASSERT(p2Baseline[0] > 43.0 && p2Baseline[0] < 47.0);
    CPPUNIT_ASSERT(p3Baseline[0] > 57.0 && p3Baseline[0] < 62.0);

    for (std::size_t i = 0; i < 25; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p1");
    }
    for (std::size_t i = 0; i < 59; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p2");
    }
    for (std::size_t i = 0; i < 100; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p3");
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, now, now + bucketLength, partitioningFields,
                                             0, annotatedProbability1));
    CPPUNIT_ASSERT(model->computeProbability(1 /*pid*/, now, now + bucketLength, partitioningFields,
                                             0, annotatedProbability2));
    CPPUNIT_ASSERT(model->computeProbability(2 /*pid*/, now, now + bucketLength, partitioningFields,
                                             0, annotatedProbability3));

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

    CPPUNIT_ASSERT(annotatedProbability1.s_Probability > 0.75);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.9);
    CPPUNIT_ASSERT(annotatedProbability3.s_Probability < 0.05);
    CPPUNIT_ASSERT(p1Baseline[0] > 58.0 && p1Baseline[0] < 62.0);
    CPPUNIT_ASSERT(p2Baseline[0] > 58.0 && p2Baseline[0] < 62.0);
    CPPUNIT_ASSERT(p3Baseline[0] > 58.0 && p3Baseline[0] < 62.0);
}

void CEventRateModelTest::testInterimCorrectionsWithCorrelations() {
    core_t::TTime startTime(3600);
    core_t::TTime bucketLength(3600);

    SModelParams params(bucketLength);
    params.s_MultivariateByFields = true;
    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 3);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    core_t::TTime now = startTime;
    core_t::TTime endTime(now + 2 * 24 * bucketLength);
    test::CRandomNumbers rng;
    TDoubleVec samples(1, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(80.0, 100.0, std::size_t(1), samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p1");
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 10.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p2");
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] - 9.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p3");
        }
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 9; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p1");
    }
    for (std::size_t i = 0; i < 10; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p2");
    }
    for (std::size_t i = 0; i < 8; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p3");
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model_t::CResultType type(model_t::CResultType::E_Conditional |
                              model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    CPPUNIT_ASSERT(model->computeProbability(1 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    CPPUNIT_ASSERT(model->computeProbability(2 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability3));

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

    CPPUNIT_ASSERT(annotatedProbability1.s_Probability > 0.7);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.7);
    CPPUNIT_ASSERT(annotatedProbability3.s_Probability > 0.7);
    CPPUNIT_ASSERT(p1Baseline[0] > 8.4 && p1Baseline[0] < 8.6);
    CPPUNIT_ASSERT(p2Baseline[0] > 9.4 && p2Baseline[0] < 9.6);
    CPPUNIT_ASSERT(p3Baseline[0] > 7.4 && p3Baseline[0] < 7.6);
}

void CEventRateModelTest::testSummaryCountZeroRecordsAreIgnored() {
    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);
    SModelParams params(bucketLength);
    std::string summaryCountField("count");
    CEventRateModelFactory factory(params, model_t::E_Manual, summaryCountField);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererWithZerosInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithZeros(
        factory.makeDataGatherer(gathererWithZerosInitData));
    CModelFactory::SModelInitializationData initDataWithZeros(gathererWithZeros);
    CAnomalyDetectorModel::TModelPtr modelWithZerosPtr(factory.makeModel(initDataWithZeros));
    CPPUNIT_ASSERT(modelWithZerosPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_EventRateOnline, modelWithZerosPtr->category());
    CEventRateModel& modelWithZeros =
        static_cast<CEventRateModel&>(*modelWithZerosPtr.get());

    CModelFactory::SGathererInitializationData gathererNoZerosInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoZeros(
        factory.makeDataGatherer(gathererNoZerosInitData));
    CModelFactory::SModelInitializationData initDataNoZeros(gathererNoZeros);
    CAnomalyDetectorModel::TModelPtr modelNoZerosPtr(factory.makeModel(initDataNoZeros));
    CPPUNIT_ASSERT(modelNoZerosPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_EventRateOnline, modelNoZerosPtr->category());
    CEventRateModel& modelNoZeros =
        static_cast<CEventRateModel&>(*modelNoZerosPtr.get());

    // The idea here is to compare a model that has records with summary count of zero
    // against a model that has no records at all where the first model had the zero-count records.

    core_t::TTime now = 100;
    core_t::TTime end = now + 50 * bucketLength;
    test::CRandomNumbers rng;
    TSizeVec samples;
    TDoubleVec zeroCountProbability;
    std::string summaryCountZero("0");
    std::string summaryCountOne("1");
    while (now < end) {
        rng.generateUniformSamples(1, 10, 1, samples);
        rng.generateUniformSamples(0.0, 1.0, 1, zeroCountProbability);
        for (std::size_t i = 0; i < samples[0]; ++i) {
            if (zeroCountProbability[0] < 0.2) {
                addArrival(*gathererWithZeros, m_ResourceMonitor, now, "p1",
                           TOptionalStr(), TOptionalStr(), TOptionalStr(summaryCountZero));
            } else {
                addArrival(*gathererWithZeros, m_ResourceMonitor, now, "p1",
                           TOptionalStr(), TOptionalStr(), TOptionalStr(summaryCountOne));
                addArrival(*gathererNoZeros, m_ResourceMonitor, now, "p1",
                           TOptionalStr(), TOptionalStr(), TOptionalStr(summaryCountOne));
            }
        }
        modelWithZeros.sample(now, now + bucketLength, m_ResourceMonitor);
        modelNoZeros.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }

    CPPUNIT_ASSERT_EQUAL(modelWithZeros.checksum(), modelNoZeros.checksum());
}

void CEventRateModelTest::testComputeProbabilityGivenDetectionRule() {
    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 100.0;
    CDetectionRule rule;
    rule.addCondition(condition);

    core_t::TTime startTime(3600);
    core_t::TTime bucketLength(3600);
    core_t::TTime endTime(24 * bucketLength);

    SModelParams params(bucketLength);
    SModelParams::TDetectionRuleVec rules(1, rule);
    params.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);

    CEventRateModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gatherer, model_, 1);
    CEventRateModel* model = dynamic_cast<CEventRateModel*>(model_.get());

    test::CRandomNumbers rng;
    core_t::TTime now = startTime;
    TDoubleVec samples(1, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, std::size_t(1), samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p1");
        }
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p1");
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);

    SAnnotatedProbability annotatedProbability;
    CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, now, now + bucketLength, partitioningFields,
                                             1, annotatedProbability) == false);
}

void CEventRateModelTest::testDecayRateControl() {
    core_t::TTime startTime = 0;
    core_t::TTime bucketLength = 1800;

    model_t::EFeature feature = model_t::E_IndividualCountByBucketAndPerson;
    model_t::TFeatureVec features(1, feature);

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
        CEventRateModelFactory factory(params);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model;
        makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
                  gatherer, model, 1);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.0001;
        CEventRateModelFactory referenceFactory(params);
        CModelFactory::TDataGathererPtr referenceGatherer;
        CAnomalyDetectorModel::TModelPtr referenceModel;
        makeModel(referenceFactory, features, m_ResourceMonitor, startTime,
                  bucketLength, referenceGatherer, referenceModel, 1);

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
                addArrival(*gatherer, m_ResourceMonitor, t + bucketLength / 2, "p1");
                addArrival(*referenceGatherer, m_ResourceMonitor,
                           t + bucketLength / 2, "p1");
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
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            maths::CBasicStatistics::mean(meanReferencePredictionError),
            maths::CBasicStatistics::mean(meanPredictionError), 0.05);
    }

    LOG_DEBUG(<< "*** Test linear scaling ***");
    {
        // This change point is amongst those we explicitly detect so
        // check we get similar detection performance with and without
        // decay rate control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model;
        makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
                  gatherer, model, 1);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CEventRateModelFactory referenceFactory(params);
        CModelFactory::TDataGathererPtr referenceGatherer;
        CAnomalyDetectorModel::TModelPtr referenceModel;
        makeModel(referenceFactory, features, m_ResourceMonitor, startTime,
                  bucketLength, referenceGatherer, referenceModel, 1);

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
                addArrival(*gatherer, m_ResourceMonitor, t + bucketLength / 2, "p1");
                addArrival(*referenceGatherer, m_ResourceMonitor,
                           t + bucketLength / 2, "p1");
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
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
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
        params.s_DecayRate = 0.001;
        CEventRateModelFactory factory(params);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model;
        makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
                  gatherer, model, 1);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CEventRateModelFactory referenceFactory(params);
        CModelFactory::TDataGathererPtr referenceGatherer;
        CAnomalyDetectorModel::TModelPtr referenceModel;
        makeModel(referenceFactory, features, m_ResourceMonitor, startTime,
                  bucketLength, referenceGatherer, referenceModel, 1);

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
                addArrival(*gatherer, m_ResourceMonitor, t + bucketLength / 2, "p1");
                addArrival(*referenceGatherer, m_ResourceMonitor,
                           t + bucketLength / 2, "p1");
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
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanPredictionError) <
                       0.7 * maths::CBasicStatistics::mean(meanReferencePredictionError));
    }
}

void CEventRateModelTest::testIgnoreSamplingGivenDetectionRules() {
    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.

    // Create a rule to filter buckets where the count > 100
    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    condition.condition().s_Op = CRuleCondition::E_GT;
    condition.condition().s_Threshold = 100.0;
    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipSampling);
    rule.addCondition(condition);

    std::size_t bucketLength(100);
    std::size_t startTime(100);
    SModelParams paramsNoRules(bucketLength);

    // Model without the skip sampling rule
    CEventRateModelFactory factory(paramsNoRules);
    model_t::TFeatureVec features{model_t::E_IndividualCountByBucketAndPerson};
    CModelFactory::TDataGathererPtr gathererNoSkip;
    CAnomalyDetectorModel::TModelPtr modelPtrNoSkip;
    makeModel(factory, features, m_ResourceMonitor, startTime, bucketLength,
              gathererNoSkip, modelPtrNoSkip, 1);
    CEventRateModel* modelNoSkip = dynamic_cast<CEventRateModel*>(modelPtrNoSkip.get());

    // Model with the skip sampling rule
    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);

    CEventRateModelFactory factoryWithSkip(paramsWithRules);
    CModelFactory::TDataGathererPtr gathererWithSkip;
    CAnomalyDetectorModel::TModelPtr modelPtrWithSkip;
    makeModel(factoryWithSkip, features, m_ResourceMonitor, startTime,
              bucketLength, gathererWithSkip, modelPtrWithSkip, 1);
    CEventRateModel* modelWithSkip =
        dynamic_cast<CEventRateModel*>(modelPtrWithSkip.get());

    std::size_t endTime = startTime + bucketLength;

    // Add a bucket to both models
    for (int i = 0; i < 66; ++i) {
        addArrival(*gathererNoSkip, m_ResourceMonitor, startTime, "p1");
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime, "p1");
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    CPPUNIT_ASSERT_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // Add a bucket to both models
    for (int i = 0; i < 55; ++i) {
        addArrival(*gathererNoSkip, m_ResourceMonitor, startTime, "p1");
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime, "p1");
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    CPPUNIT_ASSERT_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // this sample will be skipped by the detection rule
    for (int i = 0; i < 110; ++i) {
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime, "p1");
    }
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    startTime = endTime;
    endTime += bucketLength;

    // Wind the other model forward
    modelNoSkip->skipSampling(startTime);

    for (int i = 0; i < 55; ++i) {
        addArrival(*gathererNoSkip, m_ResourceMonitor, startTime, "p1");
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime, "p1");
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    // Checksums will be different due to the data gatherers
    CPPUNIT_ASSERT(modelWithSkip->checksum() != modelNoSkip->checksum());

    // but the underlying models should be the same
    CAnomalyDetectorModel::CModelDetailsViewPtr modelWithSkipView =
        modelWithSkip->details();
    CAnomalyDetectorModel::CModelDetailsViewPtr modelNoSkipView = modelNoSkip->details();

    uint64_t withSkipChecksum =
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum();
    uint64_t noSkipChecksum =
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0))
            ->residualModel()
            .checksum();
    CPPUNIT_ASSERT_EQUAL(withSkipChecksum, noSkipChecksum);

    // Check the last value times of the underlying models are the same
    const maths::CUnivariateTimeSeriesModel* timeSeriesModel =
        dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0));
    CPPUNIT_ASSERT(timeSeriesModel);

    core_t::TTime time = timeSeriesModel->trendModel().lastValueTime();
    CPPUNIT_ASSERT_EQUAL(model_t::sampleTime(model_t::E_IndividualCountByBucketAndPerson,
                                             startTime, bucketLength),
                         time);

    // The last times of model with a skip should be the same
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_IndividualCountByBucketAndPerson, 0));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trendModel().lastValueTime());
}

CppUnit::Test* CEventRateModelTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CEventRateModelTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineCountSample", &CEventRateModelTest::testOnlineCountSample));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineNonZeroCountSample",
        &CEventRateModelTest::testOnlineNonZeroCountSample));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineRare", &CEventRateModelTest::testOnlineRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineProbabilityCalculation",
        &CEventRateModelTest::testOnlineProbabilityCalculation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::"
        "testOnlineProbabilityCalculationForLowNonZeroCount",
        &CEventRateModelTest::testOnlineProbabilityCalculationForLowNonZeroCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::"
        "testOnlineProbabilityCalculationForHighNonZeroCount",
        &CEventRateModelTest::testOnlineProbabilityCalculationForHighNonZeroCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineCorrelatedNoTrend",
        &CEventRateModelTest::testOnlineCorrelatedNoTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineCorrelatedTrend",
        &CEventRateModelTest::testOnlineCorrelatedTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testPrune", &CEventRateModelTest::testPrune));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testKey", &CEventRateModelTest::testKey));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testModelsWithValueFields",
        &CEventRateModelTest::testModelsWithValueFields));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testCountProbabilityCalculationWithInfluence",
        &CEventRateModelTest::testCountProbabilityCalculationWithInfluence));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::"
        "testDistinctCountProbabilityCalculationWithInfluence",
        &CEventRateModelTest::testDistinctCountProbabilityCalculationWithInfluence));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testOnlineRareWithInfluence",
        &CEventRateModelTest::testOnlineRareWithInfluence));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testSkipSampling", &CEventRateModelTest::testSkipSampling));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testExplicitNulls", &CEventRateModelTest::testExplicitNulls));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testInterimCorrections",
        &CEventRateModelTest::testInterimCorrections));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testInterimCorrectionsWithCorrelations",
        &CEventRateModelTest::testInterimCorrectionsWithCorrelations));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testSummaryCountZeroRecordsAreIgnored",
        &CEventRateModelTest::testSummaryCountZeroRecordsAreIgnored));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testComputeProbabilityGivenDetectionRule",
        &CEventRateModelTest::testComputeProbabilityGivenDetectionRule));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testDecayRateControl", &CEventRateModelTest::testDecayRateControl));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateModelTest>(
        "CEventRateModelTest::testIgnoreSamplingGivenDetectionRules",
        &CEventRateModelTest::testIgnoreSamplingGivenDetectionRules));
    return suiteOfTests;
}
