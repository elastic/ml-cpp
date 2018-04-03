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

#include "CMetricModelTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CModelWeight.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CPrior.h>
#include <maths/CSampling.h>
#include <maths/CTimeSeriesDecompositionInterface.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CAnnotatedProbability.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CMetricModel.h>
#include <model/CMetricModelFactory.h>
#include <model/CMetricPopulationModel.h>
#include <model/CMetricPopulationModelFactory.h>
#include <model/CModelDetailsView.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>
#include <model/CRuleCondition.h>
#include <model/ModelTypes.h>

#include <test/CRandomNumbers.h>

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <cmath>
#include <utility>
#include <vector>

using namespace ml;
using namespace model;

namespace {

typedef std::pair<double, double>                                                         TDoubleDoublePr;
typedef std::pair<std::size_t, double>                                                    TSizeDoublePr;
typedef std::pair<double, std::size_t>                                                    TDoubleSizePr;
typedef std::vector<double>                                                               TDoubleVec;
typedef std::vector<TDoubleVec>                                                           TDoubleVecVec;
typedef std::vector<TDoubleDoublePr>                                                      TDoubleDoublePrVec;
typedef std::vector<std::string>                                                          TStrVec;
typedef std::vector<TStrVec>                                                              TStrVecVec;
typedef boost::optional<uint64_t>                                                         TOptionalUInt64;
typedef boost::optional<double>                                                           TOptionalDouble;
typedef std::vector<TOptionalDouble>                                                      TOptionalDoubleVec;
typedef boost::optional<std::string>                                                      TOptionalStr;
typedef std::pair<core_t::TTime, double>                                                  TTimeDoublePr;
typedef boost::optional<TTimeDoublePr>                                                    TOptionalTimeDoublePr;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator                        TMeanAccumulator;
typedef maths::CBasicStatistics::COrderStatisticsStack<double, 1u>                        TMinAccumulator;
typedef maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double> > TMaxAccumulator;
typedef boost::shared_ptr<maths::CModel>                                                  TMathsModelPtr;
typedef boost::shared_ptr<maths::CPrior>                                                  TPriorPtr;
typedef boost::shared_ptr<maths::CMultivariatePrior>                                      TMultivariatePriorPtr;
typedef std::pair<double, std::string>                                                    TDoubleStrPr;
typedef core::CSmallVector<double, 1>                                                     TDouble1Vec;
typedef core::CSmallVector<double, 2>                                                     TDouble2Vec;
typedef core::CSmallVector<double, 4>                                                     TDouble4Vec;
typedef core::CSmallVector<TDouble4Vec, 1>                                                TDouble4Vec1Vec;
typedef std::pair<std::size_t, double>                                                    TSizeDoublePr;
typedef core::CSmallVector<TSizeDoublePr, 1>                                              TSizeDoublePr1Vec;
typedef std::vector<std::string>                                                          TStrVec;
typedef std::pair<core_t::TTime, TStrVec>                                                 TTimeStrVecPr;
typedef std::vector<TTimeStrVecPr>                                                        TTimeStrVecPrVec;

const std::string EMPTY_STRING;

class CTimeLess {
    public:
        bool operator()(const CEventData &lhs,
                        const CEventData &rhs) const {
            return lhs.time() < rhs.time();
        }
};

void makeModel(CMetricModelFactory &factory,
               const CDataGatherer::TFeatureVec &features,
               core_t::TTime startTime,
               core_t::TTime bucketLength,
               CModelFactory::TDataGathererPtr &gatherer,
               CAnomalyDetectorModel::TModelPtr &model,
               unsigned int *sampleCount = 0) {
    factory.features(features);
    factory.bucketLength(bucketLength);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    if (sampleCount) {
        gathererInitData.s_SampleOverrideCount = *sampleCount;
    }
    gatherer.reset(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData initData(gatherer);
    model.reset(factory.makeModel(initData));
    CPPUNIT_ASSERT(model);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, model->category());
    CPPUNIT_ASSERT_EQUAL(bucketLength, model->bucketLength());
}

std::size_t addPerson(const std::string &p,
                      const CModelFactory::TDataGathererPtr &gatherer,
                      CResourceMonitor &resourceMonitor) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    person.resize(gatherer->fieldsOfInterest().size(), 0);
    CEventData result;
    gatherer->processFields(person, result, resourceMonitor);
    return *result.personId();
}

void addArrival(CDataGatherer &gatherer,
                CResourceMonitor &resourceMonitor,
                core_t::TTime time,
                const std::string &person,
                double value,
                const TOptionalStr &inf1 = TOptionalStr(),
                const TOptionalStr &inf2 = TOptionalStr(),
                const TOptionalStr &count = TOptionalStr()) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    if (inf1) {
        fieldValues.push_back(&(inf1.get()));
    }
    if (inf2) {
        fieldValues.push_back(&(inf2.get()));
    }
    if (count) {
        fieldValues.push_back(&(count.get()));
    }
    std::string valueAsString(core::CStringUtils::typeToStringPrecise(
                                  value,
                                  core::CIEEE754::E_DoublePrecision));
    fieldValues.push_back(&valueAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer &gatherer,
                CResourceMonitor &resourceMonitor,
                core_t::TTime time,
                const std::string &person,
                double lat, double lng,
                const TOptionalStr &inf1 = TOptionalStr(),
                const TOptionalStr &inf2 = TOptionalStr()) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    if (inf1) {
        fieldValues.push_back(&(inf1.get()));
    }
    if (inf2) {
        fieldValues.push_back(&(inf2.get()));
    }
    std::string valueAsString;
    valueAsString += core::CStringUtils::typeToStringPrecise(
        lat,
        core::CIEEE754::E_DoublePrecision);
    valueAsString += model::CAnomalyDetectorModelConfig::DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER;
    valueAsString += core::CStringUtils::typeToStringPrecise(
        lng,
        core::CIEEE754::E_DoublePrecision);
    fieldValues.push_back(&valueAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

CEventData makeEventData(core_t::TTime time,
                         std::size_t pid,
                         double value,
                         const TOptionalStr &influence = TOptionalStr()) {
    CEventData result;
    result.time(time);
    result.person(pid);
    result.addAttribute(std::size_t(0));
    result.addValue(TDouble1Vec(1, value));
    result.addInfluence(influence);
    return result;
}

TDouble1Vec featureData(const CMetricModel &model,
                        model_t::EFeature feature,
                        std::size_t pid,
                        core_t::TTime time) {
    const CMetricModel::TFeatureData *data = model.featureData(feature, pid, time);
    if (!data) {
        return TDouble1Vec();
    }
    return data->s_BucketValue ? data->s_BucketValue->value() : TDouble1Vec();
}

TDouble1Vec multivariateFeatureData(const CMetricModel &model,
                                    model_t::EFeature feature,
                                    std::size_t pid,
                                    core_t::TTime time) {
    const CMetricModel::TFeatureData *data = model.featureData(feature, pid, time);
    if (!data) {
        return TDouble1Vec();
    }
    return data->s_BucketValue ? data->s_BucketValue->value() : TDouble1Vec();
}

void processBucket(core_t::TTime time,
                   core_t::TTime bucketLength,
                   const TDoubleVec &bucket,
                   const TStrVec &influencerValues,
                   CDataGatherer &gatherer,
                   CResourceMonitor &resourceMonitor,
                   CMetricModel &model,
                   SAnnotatedProbability &probability) {
    for (std::size_t i = 0u; i < bucket.size(); ++i) {
        addArrival(gatherer, resourceMonitor, time, "p", bucket[i], TOptionalStr(influencerValues[i]));
    }
    model.sample(time, time + bucketLength, resourceMonitor);
    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model.computeProbability(0 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability);
    LOG_DEBUG("influences = " << core::CContainerPrinter::print(probability.s_Influences));
}

void processBucket(core_t::TTime time,
                   core_t::TTime bucketLength,
                   const TDoubleVec &bucket,
                   CDataGatherer &gatherer,
                   CResourceMonitor &resourceMonitor,
                   CMetricModel &model,
                   SAnnotatedProbability &probability,
                   SAnnotatedProbability &probability2) {
    const std::string person("p");
    const std::string person2("q");
    for (std::size_t i = 0u; i < bucket.size(); ++i) {
        CDataGatherer::TStrCPtrVec fieldValues;
        if (i % 2 == 0) {
            fieldValues.push_back(&person);
        } else   {
            fieldValues.push_back(&person2);
        }

        std::string valueAsString(core::CStringUtils::typeToStringPrecise(
                                      bucket[i],
                                      core::CIEEE754::E_DoublePrecision));
        fieldValues.push_back(&valueAsString);

        CEventData eventData;
        eventData.time(time);

        gatherer.addArrival(fieldValues, eventData, resourceMonitor);
    }
    model.sample(time, time + bucketLength, resourceMonitor);
    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model.computeProbability(0 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability);
    model.computeProbability(1 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability2);
}

const maths_t::TWeightStyleVec COUNT_WEIGHT(1, maths_t::E_SampleCountWeight);
const TDouble4Vec1Vec UNIT_WEIGHT(1, TDouble4Vec(1, 1.0));
const TSizeDoublePr1Vec NO_CORRELATES;

}

void CMetricModelTest::testSample(void) {
    LOG_DEBUG("*** testSample ***");

    core_t::TTime startTime(45);
    core_t::TTime bucketLength(5);
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    params.s_MaximumUpdatesPerBucket = 0.0;
    CMetricModelFactory factory(params);

    // Check basic sampling.
    {
        TTimeDoublePr data[] =
        {
            TTimeDoublePr(49, 1.5),
            TTimeDoublePr(60, 1.3),
            TTimeDoublePr(61, 1.3),
            TTimeDoublePr(62, 1.6),
            TTimeDoublePr(65, 1.7),
            TTimeDoublePr(66, 1.33),
            TTimeDoublePr(68, 1.5),
            TTimeDoublePr(84, 1.58),
            TTimeDoublePr(87, 1.99),
            TTimeDoublePr(157, 1.6),
            TTimeDoublePr(164, 1.66),
            TTimeDoublePr(199, 1.28),
            TTimeDoublePr(202, 1.0),
            TTimeDoublePr(204, 1.5)
        };

        unsigned int sampleCounts[] = { 2, 1 };
        unsigned int expectedSampleCounts[] = { 2, 1 };

        for (std::size_t i = 0; i < boost::size(sampleCounts); ++i) {
            CDataGatherer::TFeatureVec features;
            features.push_back(model_t::E_IndividualMeanByPerson);
            features.push_back(model_t::E_IndividualMinByPerson);
            features.push_back(model_t::E_IndividualMaxByPerson);
            CModelFactory::TDataGathererPtr gatherer;
            CAnomalyDetectorModel::TModelPtr model_;
            makeModel(factory, features, startTime, bucketLength, gatherer, model_, &sampleCounts[i]);
            CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
            CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

            // Bucket values.
            uint64_t expectedCount = 0;
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

            TMathsModelPtr expectedMeanModel = factory.defaultFeatureModel(model_t::E_IndividualMeanByPerson,
                                                                           bucketLength, 0.4, true);
            TMathsModelPtr expectedMinModel  = factory.defaultFeatureModel(model_t::E_IndividualMinByPerson,
                                                                           bucketLength, 0.4, true);
            TMathsModelPtr expectedMaxModel  = factory.defaultFeatureModel(model_t::E_IndividualMaxByPerson,
                                                                           bucketLength, 0.4, true);

            std::size_t j = 0;
            core_t::TTime time = startTime;
            for (;;) {
                if (j < boost::size(data) && data[j].first < time + bucketLength) {
                    LOG_DEBUG("Adding " << data[j].second << " at " << data[j].first);

                    addArrival(*gatherer, m_ResourceMonitor, data[j].first, "p", data[j].second);

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
                        expectedSampleTimes.push_back(maths::CBasicStatistics::mean(expectedSampleTime));
                        expectedMeanSamples.push_back(maths::CBasicStatistics::mean(expectedMeanSample));
                        expectedMinSamples.push_back(expectedMinSample[0]);
                        expectedMaxSamples.push_back(expectedMaxSample[0]);
                        expectedSampleTime = TMeanAccumulator();
                        expectedMeanSample = TMeanAccumulator();
                        expectedMinSample  = TMinAccumulator();
                        expectedMaxSample  = TMaxAccumulator();
                    }
                } else   {
                    LOG_DEBUG("Sampling [" << time << ", " << time + bucketLength << ")");

                    model.sample(time, time + bucketLength, m_ResourceMonitor);
                    if (maths::CBasicStatistics::count(expectedMean) > 0.0) {
                        expectedBaselineMean.add(maths::CBasicStatistics::mean(expectedMean));
                    }
                    if (numberSamples > 0) {
                        LOG_DEBUG("Adding mean samples = " << core::CContainerPrinter::print(expectedMeanSamples)
                                  << ", min samples = " << core::CContainerPrinter::print(expectedMinSamples)
                                  << ", max samples = " << core::CContainerPrinter::print(expectedMaxSamples));

                        maths::CModelAddSamplesParams::TDouble2Vec4VecVec weights(numberSamples,
                                                                                  maths::CConstantWeights::unit<TDouble2Vec>(1));
                        maths::CModelAddSamplesParams params_;
                        params_.integer(false)
                        .nonNegative(true)
                        .propagationInterval(1.0)
                        .weightStyles(COUNT_WEIGHT)
                        .trendWeights(weights)
                        .priorWeights(weights);

                        maths::CModel::TTimeDouble2VecSizeTrVec expectedMeanSamples_;
                        maths::CModel::TTimeDouble2VecSizeTrVec expectedMinSamples_;
                        maths::CModel::TTimeDouble2VecSizeTrVec expectedMaxSamples_;
                        for (std::size_t k = 0u; k < numberSamples; ++k) {
                            // We round to the nearest integer time (note this has to match
                            // the behaviour of CMetricPartialStatistic::time).
                            core_t::TTime sampleTime{static_cast<core_t::TTime>(expectedSampleTimes[k] + 0.5)};
                            expectedMeanSamples_.emplace_back(sampleTime,
                                                              TDouble2Vec{expectedMeanSamples[k]},
                                                              std::size_t(0));
                            expectedMinSamples_.emplace_back(sampleTime,
                                                             TDouble2Vec{expectedMinSamples[k]},
                                                             std::size_t(0));
                            expectedMaxSamples_.emplace_back(sampleTime,
                                                             TDouble2Vec{expectedMaxSamples[k]},
                                                             std::size_t(0));
                        }
                        expectedMeanModel->addSamples(params_, expectedMeanSamples_);
                        expectedMinModel->addSamples(params_, expectedMinSamples_);
                        expectedMaxModel->addSamples(params_, expectedMaxSamples_);
                        numberSamples = 0u;
                        expectedSampleTimes.clear();
                        expectedMeanSamples.clear();
                        expectedMinSamples.clear();
                        expectedMaxSamples.clear();
                    }

                    model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Final);
                    TOptionalUInt64 currentCount = model.currentBucketCount(0, time);
                    TDouble1Vec bucketMean   = model.currentBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, time);
                    TDouble1Vec baselineMean = model.baselineBucketMean(model_t::E_IndividualMeanByPerson, 0, 0,
                                                                        type, NO_CORRELATES, time);

                    LOG_DEBUG("bucket count = " << core::CContainerPrinter::print(currentCount));
                    LOG_DEBUG("current bucket mean = " << core::CContainerPrinter::print(bucketMean)
                              << ", expected baseline bucket mean = " << maths::CBasicStatistics::mean(expectedBaselineMean)
                              << ", baseline bucket mean = " << core::CContainerPrinter::print(baselineMean));

                    CPPUNIT_ASSERT(currentCount);
                    CPPUNIT_ASSERT_EQUAL(expectedCount, *currentCount);

                    TDouble1Vec mean = maths::CBasicStatistics::count(expectedMean) > 0.0 ?
                                       TDouble1Vec(1, maths::CBasicStatistics::mean(expectedMean)) :
                                       TDouble1Vec();
                    TDouble1Vec min  = expectedMin.count() > 0 ?
                                       TDouble1Vec(1, expectedMin[0]) : TDouble1Vec();
                    TDouble1Vec max  = expectedMax.count() > 0 ?
                                       TDouble1Vec(1, expectedMax[0]) : TDouble1Vec();

                    CPPUNIT_ASSERT(mean == bucketMean);
                    if (!baselineMean.empty()) {
                        baselineMeanError.add(std::fabs(baselineMean[0] - maths::CBasicStatistics::mean(expectedBaselineMean)));
                    }

                    CPPUNIT_ASSERT(mean == featureData(model, model_t::E_IndividualMeanByPerson, 0, time));
                    CPPUNIT_ASSERT(min  == featureData(model, model_t::E_IndividualMinByPerson, 0, time));
                    CPPUNIT_ASSERT(max  == featureData(model, model_t::E_IndividualMaxByPerson, 0, time));

                    CPPUNIT_ASSERT_EQUAL(expectedMeanModel->checksum(),
                                         model.details()->model(model_t::E_IndividualMeanByPerson, 0)->checksum());
                    CPPUNIT_ASSERT_EQUAL(expectedMinModel->checksum(),
                                         model.details()->model(model_t::E_IndividualMinByPerson, 0)->checksum());
                    CPPUNIT_ASSERT_EQUAL(expectedMaxModel->checksum(),
                                         model.details()->model(model_t::E_IndividualMaxByPerson, 0)->checksum());

                    // Test persistence. (We check for idempotency.)
                    std::string origXml;
                    {
                        core::CRapidXmlStatePersistInserter inserter("root");
                        model.acceptPersistInserter(inserter);
                        inserter.toXml(origXml);
                    }

                    // Restore the XML into a new filter
                    core::CRapidXmlParser parser;
                    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
                    core::CRapidXmlStateRestoreTraverser traverser(parser);

                    CModelFactory::SModelInitializationData initData(gatherer);
                    CAnomalyDetectorModel::TModelPtr restoredModel(factory.makeModel(initData, traverser));

                    // The XML representation of the new filter should be the same as the original
                    std::string newXml;
                    {
                        ml::core::CRapidXmlStatePersistInserter inserter("root");
                        restoredModel->acceptPersistInserter(inserter);
                        inserter.toXml(newXml);
                    }

                    uint64_t origChecksum = model.checksum(false);
                    LOG_DEBUG("original checksum = " << origChecksum);
                    uint64_t restoredChecksum = restoredModel->checksum(false);
                    LOG_DEBUG("restored checksum = " << restoredChecksum);
                    CPPUNIT_ASSERT_EQUAL(origChecksum, restoredChecksum);
                    CPPUNIT_ASSERT_EQUAL(origXml, newXml);

                    expectedCount = 0;
                    expectedMean  = TMeanAccumulator();
                    expectedMin   = TMinAccumulator();
                    expectedMax   = TMaxAccumulator();

                    if (j >= boost::size(data)) {
                        break;
                    }

                    time += bucketLength;
                }
            }
            LOG_DEBUG("baseline mean error = " << maths::CBasicStatistics::mean(baselineMeanError));
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(baselineMeanError) < 0.25);
        }
    }
}

void CMetricModelTest::testMultivariateSample(void) {
    LOG_DEBUG("*** testMultivariateSample ***");

    typedef std::vector<TDoubleVecVec>                                   TDoubleVecVecVec;
    typedef maths::CVectorNx1<double, 2>                                 TVector2;
    typedef maths::CBasicStatistics::SSampleMean<TVector2>::TAccumulator TMean2Accumulator;
    typedef std::pair<core_t::TTime, boost::array<double, 2> >           TTimeDouble2AryPr;
    typedef std::vector<TTimeDouble2AryPr>                               TTimeDouble2AryPrVec;

    core_t::TTime startTime(45);
    core_t::TTime bucketLength(5);
    SModelParams params(bucketLength);
    params.s_MaximumUpdatesPerBucket = 0.0;
    CMetricModelFactory factory(params);

    double data_[][3] =
    {
        { 49,  1.5,  1.1 },
        { 60,  1.3,  1.2 },
        { 61,  1.3,  2.1 },
        { 62,  1.6,  1.5 },
        { 65,  1.7,  1.4 },
        { 66,  1.33, 1.6 },
        { 68,  1.5,  1.37},
        { 84,  1.58, 1.42},
        { 87,  1.99, 2.2 },
        { 157, 1.6,  1.6 },
        { 164, 1.66, 1.55},
        { 199, 1.28, 1.4 },
        { 202, 1.0,  0.7 },
        { 204, 1.5,  1.8 }
    };
    TTimeDouble2AryPrVec data;
    for (std::size_t i = 0u; i < boost::size(data_); ++i) {
        boost::array<double, 2> values = { { data_[i][1], data_[i][2] } };
        data.push_back(TTimeDouble2AryPr(static_cast<core_t::TTime>(data_[i][0]), values));
    }

    unsigned int sampleCounts[] = { 2u, 1u };
    unsigned int expectedSampleCounts[] = { 2u, 1u };

    for (std::size_t i = 0; i < boost::size(sampleCounts); ++i) {
        LOG_DEBUG("*** sample count = " << sampleCounts[i] << " ***");

        CDataGatherer::TFeatureVec features(1, model_t::E_IndividualMeanLatLongByPerson);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model_;
        makeModel(factory, features, startTime, bucketLength, gatherer, model_, &sampleCounts[i]);
        CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

        // Bucket values.
        uint64_t expectedCount = 0;
        TMean2Accumulator baselineLatLongError;
        TMean2Accumulator expectedLatLong;
        TMean2Accumulator expectedBaselineLatLong;

        // Sampled values.
        TMean2Accumulator expectedLatLongSample;
        std::size_t numberSamples = 0;
        TDoubleVecVec expectedLatLongSamples;
        TMultivariatePriorPtr expectedMeanPrior = factory.defaultMultivariatePrior(model_t::E_IndividualMeanLatLongByPerson);

        std::size_t j = 0;
        core_t::TTime time = startTime;
        for (;;) {
            if (j < data.size() && data[j].first < time + bucketLength) {
                LOG_DEBUG("Adding " << data[j].second[0] << "," << data[j].second[1] << " at " << data[j].first);

                addArrival(*gatherer, m_ResourceMonitor, data[j].first, "p", data[j].second[0], data[j].second[1]);

                ++expectedCount;
                expectedLatLong.add(TVector2(data[j].second));
                expectedLatLongSample.add(TVector2(data[j].second));

                if (++j % expectedSampleCounts[i] == 0) {
                    ++numberSamples;
                    expectedLatLongSamples.push_back(
                        TDoubleVec(maths::CBasicStatistics::mean(expectedLatLongSample).begin(),
                                   maths::CBasicStatistics::mean(expectedLatLongSample).end()));
                    expectedLatLongSample = TMean2Accumulator();
                }
            } else   {
                LOG_DEBUG("Sampling [" << time << ", " << time + bucketLength << ")");
                model.sample(time, time + bucketLength, m_ResourceMonitor);

                if (maths::CBasicStatistics::count(expectedLatLong) > 0.0) {
                    expectedBaselineLatLong.add(maths::CBasicStatistics::mean(expectedLatLong));
                }
                if (numberSamples > 0) {
                    std::sort(expectedLatLongSamples.begin(), expectedLatLongSamples.end());
                    LOG_DEBUG("Adding mean samples = " << core::CContainerPrinter::print(expectedLatLongSamples));
                    expectedMeanPrior->dataType(maths_t::E_ContinuousData);
                    expectedMeanPrior->addSamples(COUNT_WEIGHT,
                                                  expectedLatLongSamples,
                                                  TDoubleVecVecVec(expectedLatLongSamples.size(),
                                                                   TDoubleVecVec(1, TDoubleVec(2, 1.0))));
                    numberSamples = 0u;
                    expectedLatLongSamples.clear();
                }

                model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Final);
                TOptionalUInt64 currentCount = model.currentBucketCount(0, time);
                TDouble1Vec bucketLatLong    = model.currentBucketValue(model_t::E_IndividualMeanLatLongByPerson, 0, 0, time);
                TDouble1Vec baselineLatLong  = model.baselineBucketMean(model_t::E_IndividualMeanLatLongByPerson, 0, 0,
                                                                        type, NO_CORRELATES, time);

                LOG_DEBUG("bucket count = " << core::CContainerPrinter::print(currentCount));
                LOG_DEBUG("current bucket mean = " << core::CContainerPrinter::print(bucketLatLong)
                          << ", expected baseline bucket mean = " << maths::CBasicStatistics::mean(expectedBaselineLatLong)
                          << ", baseline bucket mean = " << core::CContainerPrinter::print(baselineLatLong));

                CPPUNIT_ASSERT(currentCount);
                CPPUNIT_ASSERT_EQUAL(expectedCount, *currentCount);

                TDouble1Vec latLong;
                if (maths::CBasicStatistics::count(expectedLatLong) > 0.0) {
                    latLong.push_back(maths::CBasicStatistics::mean(expectedLatLong)(0));
                    latLong.push_back(maths::CBasicStatistics::mean(expectedLatLong)(1));
                }
                CPPUNIT_ASSERT(latLong == bucketLatLong);
                if (!baselineLatLong.empty()) {
                    baselineLatLongError.add(maths::fabs(  TVector2(baselineLatLong)
                                                           - maths::CBasicStatistics::mean(expectedBaselineLatLong)));
                }
                CPPUNIT_ASSERT(latLong == multivariateFeatureData(model, model_t::E_IndividualMeanLatLongByPerson, 0, time));
                CPPUNIT_ASSERT_EQUAL(expectedMeanPrior->checksum(),
                                     dynamic_cast<const maths::CMultivariateTimeSeriesModel*>(
                                         model.details()->model(model_t::E_IndividualMeanLatLongByPerson, 0))->prior().checksum());

                // Test persistence. (We check for idempotency.)
                std::string origXml;
                {
                    core::CRapidXmlStatePersistInserter inserter("root");
                    model.acceptPersistInserter(inserter);
                    inserter.toXml(origXml);
                }

                // Restore the XML into a new filter
                core::CRapidXmlParser parser;
                CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
                core::CRapidXmlStateRestoreTraverser traverser(parser);

                CModelFactory::SModelInitializationData initData(gatherer);
                CAnomalyDetectorModel::TModelPtr restoredModel(factory.makeModel(initData, traverser));

                // The XML representation of the new filter should be the same as the original
                std::string newXml;
                {
                    ml::core::CRapidXmlStatePersistInserter inserter("root");
                    restoredModel->acceptPersistInserter(inserter);
                    inserter.toXml(newXml);
                }

                uint64_t origChecksum = model.checksum(false);
                LOG_DEBUG("original checksum = " << origChecksum);
                uint64_t restoredChecksum = restoredModel->checksum(false);
                LOG_DEBUG("restored checksum = " << restoredChecksum);
                CPPUNIT_ASSERT_EQUAL(origChecksum, restoredChecksum);
                CPPUNIT_ASSERT_EQUAL(origXml, newXml);

                expectedCount   = 0;
                expectedLatLong = TMean2Accumulator();

                if (j >= boost::size(data)) {
                    break;
                }

                time += bucketLength;
            }
        }
        LOG_DEBUG("baseline mean error = " << maths::CBasicStatistics::mean(baselineLatLongError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(baselineLatLongError)(0) < 0.25);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(baselineLatLongError)(1) < 0.25);
    }
}

void CMetricModelTest::testProbabilityCalculationForMetric(void) {
    LOG_DEBUG("*** testProbabilityCalculationForMetric ***");

    typedef maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> TMinAccumulator;

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t bucketCounts[] = { 5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6 };

    double mean = 5.0;
    double variance = 2.0;
    std::size_t anomalousBucket = 12u;
    double anomaly = 5 * std::sqrt(variance);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TMinAccumulator minProbabilities(2u);

    test::CRandomNumbers rng;

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        TDoubleVec values;
        rng.generateNormalSamples(mean, variance, bucketCounts[i], values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));
        LOG_DEBUG("i = " << i << ", anomalousBucket = " << anomalousBucket
                  << ", offset = " << (i == anomalousBucket ? anomaly : 0.0));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p",
                       values[j] + (i == anomalousBucket ? anomaly : 0.0));
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        if (model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                     partitioningFields, 1, annotatedProbability) == false) {
            continue;
        }
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        if (*model.currentBucketCount(0, time) > 0) {
            minProbabilities.add(TDoubleSizePr(annotatedProbability.s_Probability, i));
        }
        time += bucketLength;
    }

    minProbabilities.sort();
    LOG_DEBUG("minProbabilities = "
              << core::CContainerPrinter::print(minProbabilities.begin(),
                                                minProbabilities.end()));
    CPPUNIT_ASSERT_EQUAL(anomalousBucket, minProbabilities[0].second);
    CPPUNIT_ASSERT(minProbabilities[0].first / minProbabilities[1].first < 0.05);
}

void CMetricModelTest::testProbabilityCalculationForMedian(void) {
    LOG_DEBUG("*** testProbabilityCalculationForMedian ***");

    typedef maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr> TMinAccumulator;

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t bucketCounts[] = { 5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6 };

    double mean = 5.0;
    double variance = 2.0;
    std::size_t anomalousBucket = 12u;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualMedianByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TMinAccumulator minProbabilities(2u);

    test::CRandomNumbers rng;

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        LOG_DEBUG("i = " << i << ", anomalousBucket = " << anomalousBucket);

        TDoubleVec values;
        if (i == anomalousBucket) {
            values.push_back(0.0);
            values.push_back(mean * 3.0);
            values.push_back(mean * 3.0);
        } else   {
            rng.generateNormalSamples(mean, variance, bucketCounts[i], values);
        }

        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }


        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        if (model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                     partitioningFields, 1, annotatedProbability) == false) {
            continue;
        }

        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        if (*model.currentBucketCount(0, time) > 0) {
            minProbabilities.add(TDoubleSizePr(annotatedProbability.s_Probability, i));
        }
        time += bucketLength;
    }

    minProbabilities.sort();
    LOG_DEBUG("minProbabilities = "
              << core::CContainerPrinter::print(minProbabilities.begin(),
                                                minProbabilities.end()));
    CPPUNIT_ASSERT_EQUAL(anomalousBucket, minProbabilities[0].second);
    CPPUNIT_ASSERT(minProbabilities[0].first / minProbabilities[1].first < 0.05);



    std::size_t pid(0);
    const CMetricModel::TFeatureData *fd = model.featureData(
        ml::model_t::E_IndividualMedianByPerson, pid,
        time - bucketLength);

    // assert there is only 1 value in the last bucket and its the median
    CPPUNIT_ASSERT_EQUAL(fd->s_BucketValue->value()[0], mean * 3.0);
    CPPUNIT_ASSERT_EQUAL(fd->s_BucketValue->value().size(), std::size_t(1));
}

void CMetricModelTest::testProbabilityCalculationForLowMean(void) {
    LOG_DEBUG("*** testProbabilityCalculationForLowMean ***");

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowMeanBucket = 60u;
    std::size_t highMeanBucket = 80u;

    double mean = 5.0;
    double variance = 0.00001;
    double lowMean = 2.0;
    double highMean = 10.0;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualLowMeanByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMeanBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMeanBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        CPPUNIT_ASSERT(model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                                partitioningFields, 1, annotatedProbability));
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG("probabilities = " << core::CContainerPrinter::print(probabilities.begin(),
                                                                   probabilities.end()));

    CPPUNIT_ASSERT(probabilities[lowMeanBucket] < 0.01);
    CPPUNIT_ASSERT(probabilities[highMeanBucket] > 0.1);
}

void CMetricModelTest::testProbabilityCalculationForHighMean(void) {
    LOG_DEBUG("*** testProbabilityCalculationForHighMean ***");

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowMeanBucket = 60;
    std::size_t highMeanBucket = 80;

    double mean = 5.0;
    double variance = 0.00001;
    double lowMean = 2.0;
    double highMean = 10.0;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualHighMeanByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMeanBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMeanBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        CPPUNIT_ASSERT(model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                                partitioningFields, 1, annotatedProbability));
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG("probabilities = " << core::CContainerPrinter::print(probabilities));

    CPPUNIT_ASSERT(probabilities[lowMeanBucket] > 0.1);
    CPPUNIT_ASSERT(probabilities[highMeanBucket] < 0.01);
}

void CMetricModelTest::testProbabilityCalculationForLowSum(void) {
    LOG_DEBUG("*** testProbabilityCalculationForLowSum ***");

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowSumBucket = 60u;
    std::size_t highSumBucket = 80u;

    double mean = 50.0;
    double variance = 5.0;
    double lowMean = 5.0;
    double highMean = 95.0;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualLowSumByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowSumBucket) {
            meanForBucket = lowMean;
        }
        if (i == highSumBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        CPPUNIT_ASSERT(model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                                partitioningFields, 1, annotatedProbability));
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG("probabilities = " << core::CContainerPrinter::print(probabilities));
    CPPUNIT_ASSERT(probabilities[lowSumBucket] < 0.01);
    CPPUNIT_ASSERT(probabilities[highSumBucket] > 0.1);
}

void CMetricModelTest::testProbabilityCalculationForHighSum(void) {
    LOG_DEBUG("*** testProbabilityCalculationForLowSum ***");

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowSumBucket = 60u;
    std::size_t highSumBucket = 80u;

    double mean = 50.0;
    double variance = 5.0;
    double lowMean = 5.0;
    double highMean = 95.0;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualHighSumByBucketAndPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowSumBucket) {
            meanForBucket = lowMean;
        }
        if (i == highSumBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        CPPUNIT_ASSERT(model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                                partitioningFields, 1, annotatedProbability));
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG("probabilities = " << core::CContainerPrinter::print(probabilities));
    CPPUNIT_ASSERT(probabilities[lowSumBucket] > 0.1);
    CPPUNIT_ASSERT(probabilities[highSumBucket] < 0.01);
}

void CMetricModelTest::testProbabilityCalculationForLatLong(void) {
    LOG_DEBUG("*** testProbabilityCalculationForLatLong ***");

    // TODO
}

void CMetricModelTest::testInfluence(void) {
    LOG_DEBUG("*** testInfluence ***");

    using TMinAccumulator = maths::CBasicStatistics::SMin<TDoubleStrPr>::TAccumulator;
    using TMaxAccumulator = maths::CBasicStatistics::SMax<TDoubleStrPr>::TAccumulator;
    using TStrDoubleDoubleTr = core::CTriple<std::string, double, double>;
    using TStrDoubleDoubleTrVec = std::vector<TStrDoubleDoubleTr>;
    using TStrDoubleDoubleTrVecVec = std::vector<TStrDoubleDoubleTrVec>;

    LOG_DEBUG("Test min and max influence");

    for (auto feature : {model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson}) {
        core_t::TTime startTime{0};
        core_t::TTime bucketLength{10};
        std::size_t numberOfBuckets{50};
        std::size_t bucketCount{5};
        double mean{5.0};
        double variance{1.0};
        std::string influencer{"I"};
        TStrVec influencerValues{ "i1", "i2", "i3", "i4", "i5" };

        SModelParams params(bucketLength);
        CMetricModelFactory factory(params);
        CDataGatherer::TFeatureVec features{feature};
        factory.features(features);
        factory.bucketLength(bucketLength);
        factory.fieldNames("", "", "P", "V", TStrVec{"I"});
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));
        CModelFactory::SModelInitializationData initData(gatherer);
        CAnomalyDetectorModel::TModelPtr model_(factory.makeModel(initData));
        CPPUNIT_ASSERT(model_);
        CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, model_->category());
        CMetricModel &model = static_cast<CMetricModel&>(*model_.get());

        test::CRandomNumbers rng;
        core_t::TTime time = startTime;
        for (std::size_t i = 0u; i < numberOfBuckets; ++i, time += bucketLength) {
            TDoubleVec samples;
            rng.generateNormalSamples(mean, variance, bucketCount, samples);

            TMinAccumulator min;
            TMaxAccumulator max;
            for (std::size_t j = 0u; j < samples.size(); ++j) {
                addArrival(*gatherer, m_ResourceMonitor, time, "p", samples[j], TOptionalStr(influencerValues[j]));
                min.add(TDoubleStrPr(samples[j], influencerValues[j]));
                max.add(TDoubleStrPr(samples[j], influencerValues[j]));
            }

            model.sample(time, time + bucketLength, m_ResourceMonitor);

            CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            SAnnotatedProbability annotatedProbability;
            model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                     partitioningFields, 1, annotatedProbability);

            LOG_DEBUG("influences = " << core::CContainerPrinter::print(annotatedProbability.s_Influences));
            if (!annotatedProbability.s_Influences.empty()) {
                std::size_t j = 0u;
                for (/**/; j < annotatedProbability.s_Influences.size(); ++j) {
                    if (   feature == model_t::E_IndividualMinByPerson &&
                           *annotatedProbability.s_Influences[j].first.second == min[0].second &&
                           std::fabs(annotatedProbability.s_Influences[j].second - 1.0) < 1e-10) {
                        break;
                    }
                    if (   feature == model_t::E_IndividualMaxByPerson &&
                           *annotatedProbability.s_Influences[j].first.second == max[0].second &&
                           std::fabs(annotatedProbability.s_Influences[j].second - 1.0) < 1e-10) {
                        break;
                    }
                }
                CPPUNIT_ASSERT(j < annotatedProbability.s_Influences.size());
            }
        }
    }

    auto testFeature = [this] (model_t::EFeature feature,
                               const TDoubleVecVec &values,
                               const TStrVecVec &influencers,
                               const TStrDoubleDoubleTrVecVec &influences) {
                           core_t::TTime startTime{0};
                           core_t::TTime bucketLength{10};

                           SModelParams params(bucketLength);
                           CMetricModelFactory factory(params);
                           CDataGatherer::TFeatureVec features{feature};
                           factory.features(features);
                           factory.bucketLength(bucketLength);
                           factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));
                           CModelFactory::SGathererInitializationData gathererInitData(startTime);
                           CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
                           CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));
                           CModelFactory::SModelInitializationData initData(gatherer);
                           CAnomalyDetectorModel::TModelPtr model_(factory.makeModel(initData));
                           CPPUNIT_ASSERT(model_);
                           CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, model_->category());
                           CMetricModel &model = static_cast<CMetricModel&>(*model_.get());

                           SAnnotatedProbability annotatedProbability;

                           core_t::TTime time{startTime};
                           for (std::size_t i = 0u; i < values.size(); ++i) {
                               processBucket(time, bucketLength, values[i], influencers[i],
                                             *gatherer, m_ResourceMonitor, model, annotatedProbability);
                               CPPUNIT_ASSERT_EQUAL(influences[i].size(), annotatedProbability.s_Influences.size());
                               if (influences[i].size() > 0) {
                                   for (const auto &expected : influences[i]) {
                                       bool found{false};
                                       for (const auto &actual : annotatedProbability.s_Influences) {
                                           if (expected.first == *actual.first.second) {
                                               CPPUNIT_ASSERT(   actual.second >= expected.second &&
                                                                 actual.second <= expected.third);
                                               found = true;
                                               break;
                                           }
                                       }
                                       CPPUNIT_ASSERT(found);
                                   }
                               }
                               time += bucketLength;
                           }
                       };

    LOG_DEBUG("Test mean");

    {
        TDoubleVecVec values{ {  1.0, 2.3, 2.1 },
                              {  8.0 },
                              {  4.3, 5.2, 3.4 },
                              {  3.2, 3.9 },
                              { 20.1, 2.8, 3.9 },
                              { 12.1, 4.2, 5.7, 3.2 },
                              {  0.1, 0.3, 5.4 },
                              { 40.5, 7.3 },
                              {  6.4, 7.0, 7.1, 6.6, 7.1, 6.7 },
                              {  0.3 } };
        TStrVecVec influencers{ { "i1", "i1", "i2" },
                                { "i1" },
                                { "i1", "i1", "i1" },
                                { "i3", "i3" },
                                { "i2", "i1", "i1" },
                                { "i1", "i2", "i2", "i2" },
                                { "i1", "i1", "i3" },
                                { "i1", "i2" },
                                { "i1", "i2", "i3", "i4", "i5", "i6" },
                                { "i2" } };
        TStrDoubleDoubleTrVecVec influences{ { },
                                             { },
                                             { },
                                             { },
                                             { },
                                             { },
                                             { core::make_triple(std::string{"i1"}, 0.9, 1.0) },
                                             { core::make_triple(std::string{"i1"}, 0.8, 0.9) },
                                             { },
                                             { core::make_triple(std::string{"i2"}, 1.0, 1.0) } };
        testFeature(model_t::E_IndividualMeanByPerson, values, influencers, influences);
    }

    LOG_DEBUG("Test sum");
    {
        TDoubleVecVec values{ { 1.0, 2.3, 2.1, 5.9 },
                              { 10.0 },
                              {  4.3, 5.2, 3.4, 6.2, 7.8 },
                              {  3.2, 3.9 },
                              { 20.1, 2.8, 3.9 },
                              { 12.1, 4.2, 5.7,  3.2 },
                              {  0.1, 0.3, 5.4 },
                              { 48.1, 10.1 },
                              {  6.8, 7.2, 7.3, 6.8, 7.3, 6.9 },
                              { 0.4 } };
        TStrVecVec influencers{ { "i1", "i1", "i2", "i2" },
                                { "i1" },
                                { "i1", "i1", "i1", "i1", "i3" },
                                { "i3", "i3" },
                                { "i2", "i1", "i1" },
                                { "i1", "i2", "i2", "i2" },
                                { "i1", "i1", "i3" },
                                { "i1", "i2" },
                                { "i1", "i2", "i3", "i4", "i5", "i6" },
                                { "i2" } };
        TStrDoubleDoubleTrVecVec influences{ { },
                                             { },
                                             { },
                                             { },
                                             { core::make_triple(std::string{"i1"}, 0.6, 0.7),
                                               core::make_triple(std::string{"i2"}, 0.9, 1.0) },
                                             { core::make_triple(std::string{"i1"}, 0.9, 1.0),
                                               core::make_triple(std::string{"i2"}, 0.9, 1.0) },
                                             { },
                                             { core::make_triple(std::string{"i1"}, 1.0, 1.0) },
                                             { },
                                             { core::make_triple(std::string{"i2"}, 1.0, 1.0) } };
        testFeature(model_t::E_IndividualSumByBucketAndPerson, values, influencers, influences);
    }

    LOG_DEBUG("Test varp");
    {
        TDoubleVecVec values{ { 1.0, 2.3, 2.1, 5.9 },
                              { 10.0 },
                              {  4.3, 5.2, 3.4, 6.2, 7.8 },
                              {  3.2, 4.9 },
                              {  3.3, 3.2, 2.4, 4.2, 6.8 },
                              {  3.2, 5.9 },
                              { 20.5, 12.3 },
                              { 12.1, 4.2, 5.7, 3.2 },
                              {  0.1, 0.3, 0.2 },
                              { 10.1, 12.8, 3.9 },
                              {  7.0,  7.0,  7.1,  6.8,  37.1,  6.7 },
                              {  0.3 } };
        TStrVecVec influencers{ { "i1", "i1", "i2", "i2" },
                                { "i1" },
                                { "i1", "i1", "i1", "i1", "i3" },
                                { "i3", "i3" },
                                { "i1", "i1", "i1", "i1", "i3" },
                                { "i3", "i3" },
                                { "i1", "i2" },
                                { "i1", "i2", "i2", "i2" },
                                { "i1", "i1", "i3" },
                                { "i2", "i1", "i1" },
                                { "i1", "i2", "i3", "i4", "i5", "i6" },
                                { "i2" } };
        TStrDoubleDoubleTrVecVec influences{ { },
                                             { },
                                             { },
                                             { },
                                             { },
                                             { },
                                             { },
                                             { },
                                             { core::make_triple(std::string{"i1"}, 0.9, 1.0),
                                               core::make_triple(std::string{"i3"}, 0.9, 1.0) },
                                             { core::make_triple(std::string{"i1"}, 0.9, 1.0) },
                                             { core::make_triple(std::string{"i5"}, 0.9, 1.0) },
                                             { } };
        testFeature(model_t::E_IndividualVarianceByPerson, values, influencers, influences);
    }
}

void CMetricModelTest::testLatLongInfluence(void) {
    LOG_DEBUG("*** testLatLongInfluence ***");

    // TODO
}

void CMetricModelTest::testPrune(void) {
    LOG_DEBUG("*** testPrune ***");

    maths::CSampling::CScopeMockRandomNumberGenerator scopeMockRng;

    typedef std::vector<std::size_t>           TSizeVec;
    typedef std::vector<TSizeVec>              TSizeVecVec;
    typedef std::vector<CEventData>            TEventDataVec;
    typedef std::map<std::size_t, std::size_t> TSizeSizeMap;

    const core_t::TTime startTime = 1346968800;
    const core_t::TTime bucketLength = 3600;

    const std::string people[] =
    {
        std::string("p1"),
        std::string("p2"),
        std::string("p3"),
        std::string("p4"),
        std::string("p5"),
        std::string("p6"),
        std::string("p7"),
        std::string("p8")
    };

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

    const std::size_t expectedPeople[] = { 1, 4, 5 };

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.01;
    CMetricModelFactory factory(params);
    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model_(factory.makeModel(modelInitData));
    CMetricModel *model = dynamic_cast<CMetricModel*>(model_.get());
    CPPUNIT_ASSERT(model);
    CModelFactory::TDataGathererPtr expectedGatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData expectedModelInitData(expectedGatherer);
    CAnomalyDetectorModel::TModelPtr expectedModelHolder(factory.makeModel(expectedModelInitData));
    CMetricModel *expectedModel = dynamic_cast<CMetricModel*>(expectedModelHolder.get());
    CPPUNIT_ASSERT(expectedModel);

    test::CRandomNumbers rng;

    TEventDataVec events;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < eventCounts.size(); ++i, bucketStart = startTime) {
        for (std::size_t j = 0u; j < eventCounts[i].size(); ++j, bucketStart += bucketLength) {
            core_t::TTime n = static_cast<core_t::TTime>(eventCounts[i][j]);
            if (n > 0) {
                TDoubleVec samples;
                rng.generateUniformSamples(0.0, 5.0, static_cast<size_t>(n), samples);

                for (core_t::TTime k = 0, time = bucketStart, dt = bucketLength / n;
                     k < n;
                     ++k, time += dt) {
                    std::size_t pid = addPerson(people[i], gatherer, m_ResourceMonitor);
                    events.push_back(makeEventData(time, pid, samples[static_cast<size_t>(k)]));
                }
            }
        }
    }
    std::sort(events.begin(), events.end(), CTimeLess());

    TEventDataVec expectedEvents;
    expectedEvents.reserve(events.size());
    TSizeSizeMap mapping;
    for (std::size_t i = 0u; i < boost::size(expectedPeople); ++i) {
        std::size_t pid = addPerson(people[expectedPeople[i]], expectedGatherer, m_ResourceMonitor);
        mapping[expectedPeople[i]] = pid;
    }
    for (std::size_t i = 0u; i < events.size(); ++i) {
        if (std::binary_search(boost::begin(expectedPeople),
                               boost::end(expectedPeople),
                               events[i].personId())) {
            expectedEvents.push_back(makeEventData(events[i].time(),
                                                   mapping[*events[i].personId()],
                                                   events[i].values()[0][0]));
        }
    }

    bucketStart = startTime;
    for (std::size_t i = 0u; i < events.size(); ++i) {
        if (events[i].time() >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(*gatherer,
                   m_ResourceMonitor,
                   events[i].time(),
                   gatherer->personName(events[i].personId().get()),
                   events[i].values()[0][0]);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune(model->defaultPruneWindow());
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    CPPUNIT_ASSERT_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = startTime;
    for (std::size_t i = 0u; i < expectedEvents.size(); ++i) {
        if (expectedEvents[i].time() >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }

        addArrival(*expectedGatherer,
                   m_ResourceMonitor,
                   expectedEvents[i].time(),
                   expectedGatherer->personName(expectedEvents[i].personId().get()),
                   expectedEvents[i].values()[0][0]);
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG("checksum          = " << model->checksum());
    LOG_DEBUG("expected checksum = " << expectedModel->checksum());
    CPPUNIT_ASSERT_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person slots.

    bucketStart = gatherer->currentBucketStartTime() + bucketLength;
    std::string newPersons[] = {"p9", "p10", "p11", "p12", "13"};
    for (std::size_t i = 0u; i < boost::size(newPersons); ++i) {
        std::size_t newPid = addPerson(newPersons[i], gatherer, m_ResourceMonitor);
        CPPUNIT_ASSERT(newPid < 8);

        std::size_t expectedNewPid = addPerson(newPersons[i], expectedGatherer, m_ResourceMonitor);

        addArrival(*gatherer, m_ResourceMonitor, bucketStart + 1, gatherer->personName(newPid), 10.0);
        addArrival(*gatherer, m_ResourceMonitor, bucketStart + 2000, gatherer->personName(newPid), 15.0);
        addArrival(*expectedGatherer, m_ResourceMonitor, bucketStart + 1, expectedGatherer->personName(expectedNewPid), 10.0);
        addArrival(*expectedGatherer, m_ResourceMonitor, bucketStart + 2000, expectedGatherer->personName(expectedNewPid), 15.0);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG("checksum          = " << model->checksum());
    LOG_DEBUG("expected checksum = " << expectedModel->checksum());
    CPPUNIT_ASSERT_EQUAL(expectedModel->checksum(), model->checksum());

    // Test that calling prune on a cloned model which has seen no new data does nothing
    CAnomalyDetectorModel::TModelPtr clonedModelHolder(model->cloneForPersistence());
    std::size_t numberOfPeopleBeforePrune(clonedModelHolder->dataGatherer().numberActivePeople());
    CPPUNIT_ASSERT(numberOfPeopleBeforePrune > 0);
    clonedModelHolder->prune(clonedModelHolder->defaultPruneWindow());
    CPPUNIT_ASSERT_EQUAL(numberOfPeopleBeforePrune, clonedModelHolder->dataGatherer().numberActivePeople());
}

void CMetricModelTest::testKey(void) {
    function_t::EFunction countFunctions[] =
    {
        function_t::E_IndividualMetric,
        function_t::E_IndividualMetricMean,
        function_t::E_IndividualMetricMin,
        function_t::E_IndividualMetricMax,
        function_t::E_IndividualMetricSum
    };
    bool useNull[] = { true, false };
    std::string byField[] = { "", "by" };
    std::string partitionField[] = { "", "partition" };

    CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();

    int identifier = 0;
    for (std::size_t i = 0u; i < boost::size(countFunctions); ++i) {
        for (std::size_t j = 0u; j < boost::size(useNull); ++j) {
            for (std::size_t k = 0u; k < boost::size(byField); ++k) {
                for (std::size_t l = 0u; l < boost::size(partitionField); ++l) {
                    CSearchKey key(++identifier,
                                   countFunctions[i],
                                   useNull[j],
                                   model_t::E_XF_None,
                                   "value",
                                   byField[k],
                                   "",
                                   partitionField[l]);

                    CAnomalyDetectorModelConfig::TModelFactoryCPtr factory = config.factory(key);

                    LOG_DEBUG("expected key = " << key);
                    LOG_DEBUG("actual key   = " << factory->searchKey());
                    CPPUNIT_ASSERT(key == factory->searchKey());
                }
            }
        }
    }
}

void CMetricModelTest::testSkipSampling(void) {
    LOG_DEBUG("*** testSkipSampling ***");

    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    factory.features(features);
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::SGathererInitializationData gathererNoGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoGap(factory.makeDataGatherer(gathererNoGapInitData));
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gathererNoGap, m_ResourceMonitor));
    CModelFactory::SModelInitializationData initDataNoGap(gathererNoGap);
    CAnomalyDetectorModel::TModelPtr modelNoGapPtr(factory.makeModel(initDataNoGap));
    CPPUNIT_ASSERT(modelNoGapPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelNoGapPtr->category());
    CMetricModel &modelNoGap = static_cast<CMetricModel&>(*modelNoGapPtr.get());

    {
        TStrVec influencerValues1{ "i1" };
        TDoubleVec bucket1{ 1.0 };
        TDoubleVec bucket2{ 5.0 };
        TDoubleVec bucket3{ 10.0 };

        SAnnotatedProbability annotatedProbability;

        core_t::TTime time = startTime;
        processBucket(time, bucketLength, bucket1, influencerValues1,
                      *gathererNoGap, m_ResourceMonitor, modelNoGap, annotatedProbability);

        time += bucketLength;
        processBucket(time, bucketLength, bucket2, influencerValues1,
                      *gathererNoGap, m_ResourceMonitor, modelNoGap, annotatedProbability);

        time += bucketLength;
        processBucket(time, bucketLength, bucket3, influencerValues1,
                      *gathererNoGap, m_ResourceMonitor, modelNoGap, annotatedProbability);
    }

    CModelFactory::SGathererInitializationData gathererWithGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithGap(factory.makeDataGatherer(gathererWithGapInitData));
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gathererWithGap, m_ResourceMonitor));
    CModelFactory::SModelInitializationData initDataWithGap(gathererWithGap);
    CAnomalyDetectorModel::TModelPtr modelWithGapPtr(factory.makeModel(initDataWithGap));
    CPPUNIT_ASSERT(modelWithGapPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelWithGapPtr->category());
    CMetricModel &modelWithGap = static_cast<CMetricModel&>(*modelWithGapPtr.get());
    core_t::TTime gap(bucketLength * 10);

    {
        TStrVec influencerValues1{ "i1" };
        TDoubleVec bucket1{ 1.0 };
        TDoubleVec bucket2{ 5.0 };
        TDoubleVec bucket3{ 10.0 };

        SAnnotatedProbability annotatedProbability;

        core_t::TTime time = startTime;
        processBucket(time, bucketLength, bucket1, influencerValues1,
                      *gathererWithGap, m_ResourceMonitor, modelWithGap, annotatedProbability);

        time += gap;
        modelWithGap.skipSampling(time);
        LOG_DEBUG("Calling sample over skipped interval should do nothing except print some ERRORs");
        modelWithGap.sample(startTime + bucketLength, time, m_ResourceMonitor);

        processBucket(time, bucketLength, bucket2, influencerValues1,
                      *gathererWithGap, m_ResourceMonitor, modelWithGap, annotatedProbability);

        time += bucketLength;
        processBucket(time, bucketLength, bucket3, influencerValues1,
                      *gathererWithGap, m_ResourceMonitor, modelWithGap, annotatedProbability);
    }

    CPPUNIT_ASSERT_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))->prior().checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))->prior().checksum());
}

void CMetricModelTest::testExplicitNulls(void) {
    LOG_DEBUG("*** testExplicitNulls ***");

    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);
    SModelParams params(bucketLength);
    std::string summaryCountField("count");
    CMetricModelFactory factory(params, model_t::E_Manual, summaryCountField);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    factory.features(features);
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::SGathererInitializationData gathererSkipGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererSkipGap(factory.makeDataGatherer(gathererSkipGapInitData));
    CModelFactory::SModelInitializationData initDataSkipGap(gathererSkipGap);
    CAnomalyDetectorModel::TModelPtr modelSkipGapPtr(factory.makeModel(initDataSkipGap));
    CPPUNIT_ASSERT(modelSkipGapPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelSkipGapPtr->category());
    CMetricModel &modelSkipGap = static_cast<CMetricModel&>(*modelSkipGapPtr.get());

    // The idea here is to compare a model that has a gap skipped against a model
    // that has explicit nulls for the buckets that sampling was skipped.

    // p1: |(1, 42.0)|(1, 1.0)|(1, 1.0)|X|X|(1, 42.0)|
    // p2: |(1, 42.)|(0, 0.0)|(0, 0.0)|X|X|(0, 0.0)|
    addArrival(*gathererSkipGap, m_ResourceMonitor, 100, "p1", 42.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererSkipGap, m_ResourceMonitor, 100, "p2", 42.0, TOptionalStr("i2"), TOptionalStr(), TOptionalStr("1"));
    modelSkipGap.sample(100, 200, m_ResourceMonitor);
    addArrival(*gathererSkipGap, m_ResourceMonitor, 200, "p1", 1.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    modelSkipGap.sample(200, 300, m_ResourceMonitor);
    addArrival(*gathererSkipGap, m_ResourceMonitor, 300, "p1", 1.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    modelSkipGap.sample(300, 400, m_ResourceMonitor);
    modelSkipGap.skipSampling(600);
    addArrival(*gathererSkipGap, m_ResourceMonitor, 600, "p1", 42.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    modelSkipGap.sample(600, 700, m_ResourceMonitor);

    CModelFactory::SGathererInitializationData gathererExNullInitData(startTime);
    CModelFactory::TDataGathererPtr gathererExNull(factory.makeDataGatherer(gathererExNullInitData));
    CModelFactory::SModelInitializationData initDataExNull(gathererExNull);
    CAnomalyDetectorModel::TModelPtr modelExNullPtr(factory.makeModel(initDataExNull));
    CPPUNIT_ASSERT(modelExNullPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelExNullPtr->category());
    CMetricModel &modelExNullGap = static_cast<CMetricModel&>(*modelExNullPtr.get());

    // p1: |(1, 42.0), ("", 42.0), (null, 42.0)|(1, 1.0)|(1, 1.0)|(null, 100.0)|(null, 100.0)|(1, 42.0)|
    // p2: |(1, 42.0), ("", 42.0)|(0, 0.0)|(0, 0.0)|(null, 100.0)|(null, 100.0)|(0, 0.0)|
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p1", 42.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p1", 42.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr(""));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p1", 42.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("null"));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p2", 42.0, TOptionalStr("i2"), TOptionalStr(), TOptionalStr("1"));
    addArrival(*gathererExNull, m_ResourceMonitor, 100, "p2", 42.0, TOptionalStr("i2"), TOptionalStr(), TOptionalStr(""));
    modelExNullGap.sample(100, 200, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 200, "p1", 1.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    modelExNullGap.sample(200, 300, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 300, "p1", 1.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    modelExNullGap.sample(300, 400, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 400, "p1", 100.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("null"));
    addArrival(*gathererExNull, m_ResourceMonitor, 400, "p2", 100.0, TOptionalStr("i2"), TOptionalStr(), TOptionalStr("null"));
    modelExNullGap.sample(400, 500, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 500, "p1", 100.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("null"));
    addArrival(*gathererExNull, m_ResourceMonitor, 500, "p2", 100.0, TOptionalStr("i2"), TOptionalStr(), TOptionalStr("null"));
    modelExNullGap.sample(500, 600, m_ResourceMonitor);
    addArrival(*gathererExNull, m_ResourceMonitor, 600, "p1", 42.0, TOptionalStr("i1"), TOptionalStr(), TOptionalStr("1"));
    modelExNullGap.sample(600, 700, m_ResourceMonitor);

    CPPUNIT_ASSERT_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelSkipGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))->prior().checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelExNullGap.details()->model(model_t::E_IndividualSumByBucketAndPerson, 0))->prior().checksum());
}

void CMetricModelTest::testVarp(void) {
    LOG_DEBUG("*** testVarp ***");

    core_t::TTime startTime(500000);
    core_t::TTime bucketLength(1000);
    SModelParams params(bucketLength);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualVarianceByPerson);
    CMetricModelFactory factory(params);
    factory.features(features);
    factory.bucketLength(bucketLength);
    factory.fieldNames("", "", "P", "V", TStrVec());
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CPPUNIT_ASSERT(!gatherer->isPopulation());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson("q", gatherer, m_ResourceMonitor));
    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr model_(factory.makeModel(initData));
    CPPUNIT_ASSERT(model_);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, model_->category());
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());

    TDoubleVec bucket1{ 1.0, 1.1 };
    TDoubleVec bucket2{ 10.0, 10.1 };
    TDoubleVec bucket3{ 4.3, 4.45 };
    TDoubleVec bucket4{ 3.2, 3.303 };
    TDoubleVec bucket5{ 20.1, 20.8, 20.9, 20.8 };
    TDoubleVec bucket6{ 4.1, 4.2 };
    TDoubleVec bucket7{ 0.1, 0.3, 0.2, 0.4 };
    TDoubleVec bucket8{ 12.5, 12.3 };
    TDoubleVec bucket9{ 6.9, 7.0, 7.1, 6.6, 7.1, 6.7 };
    TDoubleVec bucket10{ 0.3, 0.2 };
    TDoubleVec bucket11{ 0.0 };

    SAnnotatedProbability annotatedProbability;
    SAnnotatedProbability annotatedProbability2;

    core_t::TTime time = startTime;
    processBucket(time, bucketLength, bucket1,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket2,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket3,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket4,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket5,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket6,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket7,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.8);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, bucket8,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.5);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.5);

    time += bucketLength;
    processBucket(time, bucketLength, bucket9,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.5);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.5);

    time += bucketLength;
    processBucket(time, bucketLength, bucket10,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.5);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.5);

    time += bucketLength;
    processBucket(time, bucketLength, bucket11,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability, annotatedProbability2);
    LOG_DEBUG("P1 " << annotatedProbability.s_Probability << ", P2 " << annotatedProbability2.s_Probability);
    CPPUNIT_ASSERT(annotatedProbability.s_Probability > 0.5);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.5);
}

void CMetricModelTest::testInterimCorrections(void) {
    LOG_DEBUG("*** testInterimCorrections ***");

    core_t::TTime startTime(3600);
    core_t::TTime bucketLength(3600);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);
    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    factory.features(features);
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));
    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr model_(factory.makeModel(initData));
    CPPUNIT_ASSERT(model_);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, model_->category());
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());

    std::size_t pid1 = addPerson("p1", gatherer, m_ResourceMonitor);
    std::size_t pid2 = addPerson("p2", gatherer, m_ResourceMonitor);
    std::size_t pid3 = addPerson("p3", gatherer, m_ResourceMonitor);

    core_t::TTime now = startTime;
    core_t::TTime endTime(now + 2 * 24 * bucketLength);
    test::CRandomNumbers rng;
    TDoubleVec samples(3, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, std::size_t(3), samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p1", 1.0, TOptionalStr("i1"));
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[1] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p2", 1.0, TOptionalStr("i2"));
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[2] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p3", 1.0, TOptionalStr("i3"));
        }
        model.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p1", 1.0, TOptionalStr("i1"));
    }
    for (std::size_t i = 0; i < 1; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p2", 1.0, TOptionalStr("i2"));
    }
    for (std::size_t i = 0; i < 100; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p3", 1.0, TOptionalStr("i3"));
    }
    model.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    CPPUNIT_ASSERT(model.computeProbability(pid1, now, now + bucketLength,
                                            partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    CPPUNIT_ASSERT(model.computeProbability(pid2, now, now + bucketLength,
                                            partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    CPPUNIT_ASSERT(model.computeProbability(pid3, now, now + bucketLength,
                                            partitioningFields, 1, annotatedProbability3));

    TDouble1Vec p1Baseline = model.baselineBucketMean(model_t::E_IndividualSumByBucketAndPerson,
                                                      pid1, 0, type, NO_CORRELATES, now);
    TDouble1Vec p2Baseline = model.baselineBucketMean(model_t::E_IndividualSumByBucketAndPerson,
                                                      pid2, 0, type, NO_CORRELATES, now);
    TDouble1Vec p3Baseline = model.baselineBucketMean(model_t::E_IndividualSumByBucketAndPerson,
                                                      pid3, 0, type, NO_CORRELATES, now);

    LOG_DEBUG("p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG("p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG("p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG("p1 baseline = " << p1Baseline[0]);
    LOG_DEBUG("p2 baseline = " << p2Baseline[0]);
    LOG_DEBUG("p3 baseline = " << p3Baseline[0]);

    CPPUNIT_ASSERT(annotatedProbability1.s_Probability > 0.05);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability < 0.05);
    CPPUNIT_ASSERT(annotatedProbability3.s_Probability < 0.05);
    CPPUNIT_ASSERT(p1Baseline[0] > 44.0 && p1Baseline[0] < 46.0);
    CPPUNIT_ASSERT(p2Baseline[0] > 45.0 && p2Baseline[0] < 46.0);
    CPPUNIT_ASSERT(p3Baseline[0] > 59.0 && p3Baseline[0] < 61.0);
}

void CMetricModelTest::testInterimCorrectionsWithCorrelations(void) {
    LOG_DEBUG("*** testInterimCorrectionsWithCorrelations ***");

    core_t::TTime startTime(3600);
    core_t::TTime bucketLength(3600);
    SModelParams params(bucketLength);
    params.s_MultivariateByFields = true;
    CMetricModelFactory factory(params);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    factory.features(features);
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelPtr(factory.makeModel(initData));
    CPPUNIT_ASSERT(modelPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelPtr->category());
    CMetricModel &model = static_cast<CMetricModel&>(*modelPtr.get());

    std::size_t pid1 = addPerson("p1", gatherer, m_ResourceMonitor);
    std::size_t pid2 = addPerson("p2", gatherer, m_ResourceMonitor);
    std::size_t pid3 = addPerson("p3", gatherer, m_ResourceMonitor);

    core_t::TTime now = startTime;
    core_t::TTime endTime(now + 2 * 24 * bucketLength);
    test::CRandomNumbers rng;
    TDoubleVec samples(1, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(80.0, 100.0, std::size_t(1), samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p1", 1.0, TOptionalStr("i1"));
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 10.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p2", 1.0, TOptionalStr("i2"));
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] - 9.5); ++i) {
            addArrival(*gatherer, m_ResourceMonitor, now, "p3", 1.0, TOptionalStr("i3"));
        }
        model.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 9; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p1", 1.0, TOptionalStr("i1"));
    }
    for (std::size_t i = 0; i < 10; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p2", 1.0, TOptionalStr("i2"));
    }
    for (std::size_t i = 0; i < 8; ++i) {
        addArrival(*gatherer, m_ResourceMonitor, now, "p3", 1.0, TOptionalStr("i3"));
    }
    model.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model_t::CResultType type(model_t::CResultType::E_Conditional | model_t::CResultType::E_Interim);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType = type;
    CPPUNIT_ASSERT(model.computeProbability(pid1, now, now + bucketLength,
                                            partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType = type;
    CPPUNIT_ASSERT(model.computeProbability(pid2, now, now + bucketLength,
                                            partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType = type;
    CPPUNIT_ASSERT(model.computeProbability(pid3, now, now + bucketLength,
                                            partitioningFields, 1, annotatedProbability3));

    TDouble1Vec p1Baseline = model.baselineBucketMean(model_t::E_IndividualSumByBucketAndPerson,
                                                      pid1, 0, type,
                                                      annotatedProbability1.s_AttributeProbabilities[0].s_Correlated,
                                                      now);
    TDouble1Vec p2Baseline = model.baselineBucketMean(model_t::E_IndividualSumByBucketAndPerson,
                                                      pid2, 0, type,
                                                      annotatedProbability2.s_AttributeProbabilities[0].s_Correlated,
                                                      now);
    TDouble1Vec p3Baseline = model.baselineBucketMean(model_t::E_IndividualSumByBucketAndPerson,
                                                      pid3, 0, type,
                                                      annotatedProbability3.s_AttributeProbabilities[0].s_Correlated,
                                                      now);

    LOG_DEBUG("p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG("p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG("p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG("p1 baseline = " << p1Baseline[0]);
    LOG_DEBUG("p2 baseline = " << p2Baseline[0]);
    LOG_DEBUG("p3 baseline = " << p3Baseline[0]);

    CPPUNIT_ASSERT(annotatedProbability1.s_Probability > 0.7);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability > 0.7);
    CPPUNIT_ASSERT(annotatedProbability3.s_Probability > 0.7);
    CPPUNIT_ASSERT(p1Baseline[0] > 8.4 && p1Baseline[0] < 8.6);
    CPPUNIT_ASSERT(p2Baseline[0] > 9.4 && p2Baseline[0] < 9.6);
    CPPUNIT_ASSERT(p3Baseline[0] > 7.4 && p3Baseline[0] < 7.6);
}

void CMetricModelTest::testCorrelatePersist(void) {
    LOG_DEBUG("*** testCorrelatePersist ***");

    typedef maths::CVectorNx1<double, 2>          TVector2;
    typedef maths::CSymmetricMatrixNxN<double, 2> TMatrix2;

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    const double means[] = { 10.0, 20.0 };
    const double covariances[] = { 3.0, 2.0, 2.0 };
    TVector2 mean(means, means + 2);
    TMatrix2 covariance(covariances, covariances + 3);

    test::CRandomNumbers rng;

    TDoubleVecVec samples;
    rng.generateMultivariateNormalSamples(mean.toVector<TDoubleVec>(),
                                          covariance.toVectors<TDoubleVecVec>(),
                                          10000,
                                          samples);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MultivariateByFields = true;
    CMetricModelFactory factory(params);
    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualMeanByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model;
    makeModel(factory, features, startTime, bucketLength, gatherer, model);

    addPerson("p1", gatherer, m_ResourceMonitor);
    addPerson("p2", gatherer, m_ResourceMonitor);

    core_t::TTime time   = startTime;
    core_t::TTime bucket = time + bucketLength;
    for (std::size_t i = 0u; i < samples.size(); ++i, time += 60) {
        if (time >= bucket) {
            model->sample(bucket - bucketLength, bucket, m_ResourceMonitor);
            bucket += bucketLength;
        }
        addArrival(*gatherer, m_ResourceMonitor, time, "p1", samples[i][0]);
        addArrival(*gatherer, m_ResourceMonitor, time, "p2", samples[i][0]);

        if ((i + 1) % 1000 == 0) {
            // Test persistence. (We check for idempotency.)
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

            CModelFactory::SModelInitializationData initData(gatherer);
            CAnomalyDetectorModel::TModelPtr restoredModel(factory.makeModel(initData, traverser));

            // The XML representation of the new filter should be the same as the original
            std::string newXml;
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                restoredModel->acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }

            uint64_t origChecksum = model->checksum(false);
            LOG_DEBUG("original checksum = " << origChecksum);
            uint64_t restoredChecksum = restoredModel->checksum(false);
            LOG_DEBUG("restored checksum = " << restoredChecksum);
            CPPUNIT_ASSERT_EQUAL(origChecksum, restoredChecksum);
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }
    }
}

void CMetricModelTest::testSummaryCountZeroRecordsAreIgnored(void) {
    LOG_DEBUG("*** testSummaryCountZeroRecordsAreIgnored ***");

    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);
    SModelParams params(bucketLength);
    std::string summaryCountField("count");
    CMetricModelFactory factory(params, model_t::E_Manual, summaryCountField);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    factory.features(features);
    factory.bucketLength(bucketLength);
    factory.fieldNames("", "", "P", "V", TStrVec(1, "I"));

    CModelFactory::SGathererInitializationData gathererWithZerosInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithZeros(factory.makeDataGatherer(gathererWithZerosInitData));
    CModelFactory::SModelInitializationData initDataWithZeros(gathererWithZeros);
    CAnomalyDetectorModel::TModelPtr modelWithZerosPtr(factory.makeModel(initDataWithZeros));
    CPPUNIT_ASSERT(modelWithZerosPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelWithZerosPtr->category());
    CMetricModel &modelWithZeros = static_cast<CMetricModel&>(*modelWithZerosPtr.get());

    CModelFactory::SGathererInitializationData gathererNoZerosInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoZeros(factory.makeDataGatherer(gathererNoZerosInitData));
    CModelFactory::SModelInitializationData initDataNoZeros(gathererNoZeros);
    CAnomalyDetectorModel::TModelPtr modelNoZerosPtr(factory.makeModel(initDataNoZeros));
    CPPUNIT_ASSERT(modelNoZerosPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelNoZerosPtr->category());
    CMetricModel &modelNoZeros = static_cast<CMetricModel&>(*modelNoZerosPtr.get());

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
                addArrival(*gathererWithZeros, m_ResourceMonitor, now, "p1", value,
                           TOptionalStr("i1"), TOptionalStr(), TOptionalStr(summaryCountZero));
            } else   {
                addArrival(*gathererWithZeros, m_ResourceMonitor, now, "p1", value,
                           TOptionalStr("i1"), TOptionalStr(), TOptionalStr(summaryCountOne));
                addArrival(*gathererNoZeros, m_ResourceMonitor, now, "p1", value,
                           TOptionalStr("i1"), TOptionalStr(), TOptionalStr(summaryCountOne));
            }
        }
        modelWithZeros.sample(now, now + bucketLength, m_ResourceMonitor);
        modelNoZeros.sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }

    CPPUNIT_ASSERT_EQUAL(modelWithZeros.checksum(), modelNoZeros.checksum());
}

void CMetricModelTest::testDecayRateControl(void) {
    LOG_DEBUG("*** testDecayRateControl ***");

    core_t::TTime startTime = 0;
    core_t::TTime bucketLength = 1800;

    model_t::EFeature feature = model_t::E_IndividualMeanByPerson;
    model_t::TFeatureVec features(1, feature);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MinimumModeFraction = model::CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;

    test::CRandomNumbers rng;

    LOG_DEBUG("*** Test anomaly ***");
    {
        // Test we don't adapt the decay rate if there is a short-lived
        // anomaly. We should get essentially identical prediction errors
        // with and without decay control.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        CMetricModelFactory factory(params);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model;
        makeModel(factory, features, startTime, bucketLength, gatherer, model);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.0001;
        CMetricModelFactory referenceFactory(params);
        CModelFactory::TDataGathererPtr referenceGatherer;
        CAnomalyDetectorModel::TModelPtr referenceModel;
        makeModel(referenceFactory, features, startTime, bucketLength, referenceGatherer, referenceModel);

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Interim);
        for (core_t::TTime t = 0; t < 4 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG("week " << t / core::constants::WEEK + 1);
            }

            TDoubleVec value;
            rng.generateUniformSamples(0.0, 10.0, 1, value);
            value[0] += 20.0 * (t > 3 * core::constants::WEEK && t < core::constants::WEEK + 4 * 3600 ? 1.0 : 0.0);
            addArrival(*gatherer, m_ResourceMonitor, t + bucketLength / 2, "p1", value[0]);
            addArrival(*referenceGatherer, m_ResourceMonitor, t + bucketLength / 2, "p1", value[0]);
            model->sample(t, t + bucketLength, m_ResourceMonitor);
            referenceModel->sample(t, t + bucketLength, m_ResourceMonitor);
            meanPredictionError.add(std::fabs(
                                        model->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0]
                                        - model->baselineBucketMean(feature, 0, 0, type,
                                                                    NO_CORRELATES, t + bucketLength / 2)[0]));
            meanReferencePredictionError.add(std::fabs(
                                                 referenceModel->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0]
                                                 - referenceModel->baselineBucketMean(feature, 0, 0, type,
                                                                                      NO_CORRELATES, t + bucketLength / 2)[0]));
        }
        LOG_DEBUG("mean = " << maths::CBasicStatistics::mean(meanPredictionError));
        LOG_DEBUG("reference = " << maths::CBasicStatistics::mean(meanReferencePredictionError));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(maths::CBasicStatistics::mean(meanReferencePredictionError),
                                     maths::CBasicStatistics::mean(meanPredictionError),
                                     0.06);
    }

    LOG_DEBUG("*** Test step change ***");
    {
        // Test a step change in a stable signal is detected and we get a
        // significant reduction in the prediction error.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        CMetricModelFactory factory(params);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model;
        makeModel(factory, features, startTime, bucketLength, gatherer, model);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CMetricModelFactory referenceFactory(params);
        CModelFactory::TDataGathererPtr referenceGatherer;
        CAnomalyDetectorModel::TModelPtr referenceModel;
        makeModel(referenceFactory, features, startTime, bucketLength, referenceGatherer, referenceModel);

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Interim);
        for (core_t::TTime t = 0; t < 10 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG("week " << t / core::constants::WEEK + 1);
            }

            double value = 10.0 * (1.0 + ::sin(  boost::math::double_constants::two_pi
                                                 * static_cast<double>(t)
                                                 / static_cast<double>(core::constants::DAY)))
                           * (t < 5 * core::constants::WEEK ? 1.0 : 2.0);
            TDoubleVec noise;
            rng.generateUniformSamples(0.0, 3.0, 1, noise);
            addArrival(*gatherer, m_ResourceMonitor, t + bucketLength / 2, "p1", value + noise[0]);
            addArrival(*referenceGatherer, m_ResourceMonitor, t + bucketLength / 2, "p1", value + noise[0]);
            model->sample(t, t + bucketLength, m_ResourceMonitor);
            referenceModel->sample(t, t + bucketLength, m_ResourceMonitor);
            meanPredictionError.add(std::fabs(
                                        model->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0]
                                        - model->baselineBucketMean(feature, 0, 0, type,
                                                                    NO_CORRELATES, t + bucketLength / 2)[0]));
            meanReferencePredictionError.add(std::fabs(
                                                 referenceModel->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0]
                                                 - referenceModel->baselineBucketMean(feature, 0, 0, type,
                                                                                      NO_CORRELATES, t + bucketLength / 2)[0]));
        }
        LOG_DEBUG("mean = " << maths::CBasicStatistics::mean(meanPredictionError));
        LOG_DEBUG("reference = " << maths::CBasicStatistics::mean(meanReferencePredictionError));
        CPPUNIT_ASSERT(         maths::CBasicStatistics::mean(meanPredictionError)
                                < 0.94 * maths::CBasicStatistics::mean(meanReferencePredictionError));
    }

    LOG_DEBUG("*** Test unmodelled cyclic component ***");
    {
        // This modulates the event rate using a sine with period 10 weeks
        // effectively there are significant "manoeuvres" in the event rate
        // every 5 weeks at the function turning points. We check we get a
        // significant reduction in the prediction error.

        params.s_ControlDecayRate = true;
        params.s_DecayRate = 0.001;
        CMetricModelFactory factory(params);
        CModelFactory::TDataGathererPtr gatherer;
        CAnomalyDetectorModel::TModelPtr model;
        makeModel(factory, features, startTime, bucketLength, gatherer, model);

        params.s_ControlDecayRate = false;
        params.s_DecayRate = 0.001;
        CMetricModelFactory referenceFactory(params);
        CModelFactory::TDataGathererPtr referenceGatherer;
        CAnomalyDetectorModel::TModelPtr referenceModel;
        makeModel(referenceFactory, features, startTime, bucketLength, referenceGatherer, referenceModel);

        TMeanAccumulator meanPredictionError;
        TMeanAccumulator meanReferencePredictionError;
        model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Interim);
        for (core_t::TTime t = 0; t < 20 * core::constants::WEEK; t += bucketLength) {
            if (t % core::constants::WEEK == 0) {
                LOG_DEBUG("week " << t / core::constants::WEEK + 1);
            }

            double value = 10.0 * (1.0 + ::sin(  boost::math::double_constants::two_pi
                                                 * static_cast<double>(t)
                                                 / static_cast<double>(core::constants::DAY)))
                           * (1.0 + ::sin(  boost::math::double_constants::two_pi
                                            * static_cast<double>(t)
                                            / 10.0 / static_cast<double>(core::constants::WEEK)));
            TDoubleVec noise;
            rng.generateUniformSamples(0.0, 3.0, 1, noise);
            addArrival(*gatherer, m_ResourceMonitor, t + bucketLength / 2, "p1", value + noise[0]);
            addArrival(*referenceGatherer, m_ResourceMonitor, t + bucketLength / 2, "p1", value + noise[0]);
            model->sample(t, t + bucketLength, m_ResourceMonitor);
            referenceModel->sample(t, t + bucketLength, m_ResourceMonitor);
            meanPredictionError.add(std::fabs(
                                        model->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0]
                                        - model->baselineBucketMean(feature, 0, 0, type,
                                                                    NO_CORRELATES, t + bucketLength / 2)[0]));
            meanReferencePredictionError.add(std::fabs(
                                                 referenceModel->currentBucketValue(feature, 0, 0, t + bucketLength / 2)[0]
                                                 - referenceModel->baselineBucketMean(feature, 0, 0, type,
                                                                                      NO_CORRELATES, t + bucketLength / 2)[0]));
        }
        LOG_DEBUG("mean = " << maths::CBasicStatistics::mean(meanPredictionError));
        LOG_DEBUG("reference = " << maths::CBasicStatistics::mean(meanReferencePredictionError));
        CPPUNIT_ASSERT(        maths::CBasicStatistics::mean(meanPredictionError)
                               < 0.7 * maths::CBasicStatistics::mean(meanReferencePredictionError));
    }
}

void CMetricModelTest::testProbabilityCalculationForLowMedian(void) {
    LOG_DEBUG("*** testProbabilityCalculationForLowMedian ***");

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowMedianBucket = 60u;
    std::size_t highMedianBucket = 80u;

    double mean = 5.0;
    double variance = 0.00001;
    double lowMean = 2.0;
    double highMean = 10.0;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualLowMedianByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMedianBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMedianBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        CPPUNIT_ASSERT(model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                                partitioningFields, 1, annotatedProbability));
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG("probabilities = " << core::CContainerPrinter::print(probabilities.begin(),
                                                                   probabilities.end()));

    CPPUNIT_ASSERT(probabilities[lowMedianBucket] < 0.01);
    CPPUNIT_ASSERT(probabilities[highMedianBucket] > 0.1);
}

void CMetricModelTest::testProbabilityCalculationForHighMedian(void) {
    LOG_DEBUG("*** testProbabilityCalculationForHighMedian ***");

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t numberOfBuckets = 100;
    std::size_t bucketCount = 5;
    std::size_t lowMedianBucket = 60;
    std::size_t highMedianBucket = 80;

    double mean = 5.0;
    double variance = 0.00001;
    double lowMean = 2.0;
    double highMean = 10.0;

    CDataGatherer::TFeatureVec features(1, model_t::E_IndividualHighMeanByPerson);
    CModelFactory::TDataGathererPtr gatherer;
    CAnomalyDetectorModel::TModelPtr model_;
    makeModel(factory, features, startTime, bucketLength, gatherer, model_);
    CMetricModel &model = static_cast<CMetricModel&>(*model_.get());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

    TOptionalDoubleVec probabilities;
    test::CRandomNumbers rng;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberOfBuckets; ++i) {
        double meanForBucket = mean;
        if (i == lowMedianBucket) {
            meanForBucket = lowMean;
        }
        if (i == highMedianBucket) {
            meanForBucket = highMean;
        }
        TDoubleVec values;
        rng.generateNormalSamples(meanForBucket, variance, bucketCount, values);
        LOG_DEBUG("values = " << core::CContainerPrinter::print(values));

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer, m_ResourceMonitor, time + static_cast<core_t::TTime>(j), "p", values[j]);
        }
        model.sample(time, time + bucketLength, m_ResourceMonitor);

        CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
        SAnnotatedProbability annotatedProbability;
        CPPUNIT_ASSERT(model.computeProbability(0 /*pid*/, time, time + bucketLength,
                                                partitioningFields, 1, annotatedProbability));
        LOG_DEBUG("probability = " << annotatedProbability.s_Probability);
        probabilities.push_back(annotatedProbability.s_Probability);

        time += bucketLength;
    }

    LOG_DEBUG("probabilities = " << core::CContainerPrinter::print(probabilities));

    CPPUNIT_ASSERT(probabilities[lowMedianBucket] > 0.1);
    CPPUNIT_ASSERT(probabilities[highMedianBucket] < 0.01);
}

void CMetricModelTest::testIgnoreSamplingGivenDetectionRules(void) {
    LOG_DEBUG("*** testIgnoreSamplingGivenDetectionRules ***");

    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.


    // Create a rule to filter buckets where the actual value > 100
    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    condition.condition().s_Op = CRuleCondition::E_GT;
    condition.condition().s_Threshold = 100.0;
    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipSampling);
    rule.addCondition(condition);

    std::size_t bucketLength(300);
    std::size_t startTime(300);
    SModelParams paramsNoRules(bucketLength);


    // Model without the skip sampling rule
    CMetricModelFactory factory(paramsNoRules);
    model_t::TFeatureVec features{model_t::E_IndividualMeanByPerson};
    CModelFactory::TDataGathererPtr gathererNoSkip;
    CAnomalyDetectorModel::TModelPtr modelPtrNoSkip;
    makeModel(factory, features, startTime, bucketLength, gathererNoSkip, modelPtrNoSkip);
    CMetricModel *modelNoSkip = dynamic_cast<CMetricModel*>(modelPtrNoSkip.get());

    // Model with the skip sampling rule
    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);

    CMetricModelFactory factoryWithSkip(paramsWithRules);
    CModelFactory::TDataGathererPtr gathererWithSkip;
    CAnomalyDetectorModel::TModelPtr modelPtrWithSkip;
    makeModel(factoryWithSkip, features, startTime, bucketLength, gathererWithSkip, modelPtrWithSkip);
    CMetricModel *modelWithSkip = dynamic_cast<CMetricModel*>(modelPtrWithSkip.get());

    std::size_t endTime = startTime + bucketLength;

    // Add a bucket to both models
    for (std::size_t i=0; i<60; i++) {
        addArrival(*gathererNoSkip, m_ResourceMonitor, startTime + i, "p1", 1.0);
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime + i, "p1", 1.0);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    CPPUNIT_ASSERT_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // Add a bucket to both models
    for (std::size_t i=0; i<60; i++) {
        addArrival(*gathererNoSkip, m_ResourceMonitor, startTime + i, "p1", 1.0);
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime + i, "p1", 1.0);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    CPPUNIT_ASSERT_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    // this sample will be skipped by the detection rule
    for (std::size_t i=0; i<60; i++) {
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime + i, "p1", 110.0);
    }
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    startTime = endTime;
    endTime += bucketLength;

    // Wind the other model forward
    modelNoSkip->skipSampling(startTime);

    for (std::size_t i=0; i<60; i++) {
        addArrival(*gathererNoSkip, m_ResourceMonitor, startTime + i, "p1", 2.0);
        addArrival(*gathererWithSkip, m_ResourceMonitor, startTime + i, "p1", 2.0);
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    // Checksums will be different due to the data gatherers
    CPPUNIT_ASSERT(modelWithSkip->checksum() != modelNoSkip->checksum());

    // but the underlying models should be the same
    CAnomalyDetectorModel::CModelDetailsViewPtr modelWithSkipView = modelWithSkip->details();
    CAnomalyDetectorModel::CModelDetailsViewPtr modelNoSkipView = modelNoSkip->details();

    // TODO this test fails due a different checksums for the decay rate and prior
    // uint64_t withSkipChecksum = modelWithSkipView->model(model_t::E_IndividualMeanByPerson, 0)->checksum();
    // uint64_t noSkipChecksum = modelNoSkipView->model(model_t::E_IndividualMeanByPerson, 0)->checksum();
    // CPPUNIT_ASSERT_EQUAL(withSkipChecksum, noSkipChecksum);


    // TODO These checks fail see elastic/machine-learning-cpp/issues/485
    // Check the last value times of the underlying models are the same
    // const maths::CUnivariateTimeSeriesModel *timeSeriesModel =
    //     dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_IndividualMeanByPerson, 0));
    // CPPUNIT_ASSERT(timeSeriesModel != 0);

    // core_t::TTime time = timeSeriesModel->trend().lastValueTime();
    // CPPUNIT_ASSERT_EQUAL(model_t::sampleTime(model_t::E_IndividualMeanByPerson, startTime, bucketLength), time);

    // // The last times of model with a skip should be the same
    // timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_IndividualMeanByPerson, 0));
    // CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());
}


CppUnit::Test *CMetricModelTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMetricModelTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testSample",
                               &CMetricModelTest::testSample) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testMultivariateSample",
                               &CMetricModelTest::testMultivariateSample) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForMetric",
                               &CMetricModelTest::testProbabilityCalculationForMetric) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForMedian",
                               &CMetricModelTest::testProbabilityCalculationForMedian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForLowMean",
                               &CMetricModelTest::testProbabilityCalculationForLowMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForHighMean",
                               &CMetricModelTest::testProbabilityCalculationForHighMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForLowSum",
                               &CMetricModelTest::testProbabilityCalculationForLowSum) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForHighSum",
                               &CMetricModelTest::testProbabilityCalculationForHighSum) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForLatLong",
                               &CMetricModelTest::testProbabilityCalculationForLatLong) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testInfluence",
                               &CMetricModelTest::testInfluence) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testLatLongInfluence",
                               &CMetricModelTest::testLatLongInfluence) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testPrune",
                               &CMetricModelTest::testPrune) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testKey",
                               &CMetricModelTest::testKey) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testSkipSampling",
                               &CMetricModelTest::testSkipSampling) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testExplicitNulls",
                               &CMetricModelTest::testExplicitNulls) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testVarp",
                               &CMetricModelTest::testVarp) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testInterimCorrections",
                               &CMetricModelTest::testInterimCorrections) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testInterimCorrectionsWithCorrelations",
                               &CMetricModelTest::testInterimCorrectionsWithCorrelations) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testCorrelatePersist",
                               &CMetricModelTest::testCorrelatePersist) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testSummaryCountZeroRecordsAreIgnored",
                               &CMetricModelTest::testSummaryCountZeroRecordsAreIgnored) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testSummaryCountZeroRecordsAreIgnored",
                               &CMetricModelTest::testSummaryCountZeroRecordsAreIgnored) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testDecayRateControl",
                               &CMetricModelTest::testDecayRateControl) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForLowMedian",
                               &CMetricModelTest::testProbabilityCalculationForLowMedian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testProbabilityCalculationForHighMedian",
                               &CMetricModelTest::testProbabilityCalculationForHighMedian) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricModelTest>(
                               "CMetricModelTest::testIgnoreSamplingGivenDetectionRules",
                               &CMetricModelTest::testIgnoreSamplingGivenDetectionRules) );

    return suiteOfTests;
}
