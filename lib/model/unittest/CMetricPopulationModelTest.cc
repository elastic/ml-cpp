/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CIEEE754.h>
#include <core/CLogger.h>
#include <core/CPatternSet.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CModelWeight.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CSampling.h>
#include <maths/CTimeSeriesDecompositionInterface.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CMetricPopulationModel.h>
#include <model/CMetricPopulationModelFactory.h>
#include <model/CModelDetailsView.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/BoostTestPointerOutput.h>
#include <test/CRandomNumbers.h>

#include <boost/optional/optional_io.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CMetricPopulationModelTest)

using namespace ml;
using namespace model;

namespace {

using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrVec = std::vector<TSizeSizePr>;
using TSizeSizePrVecVec = std::vector<TSizeSizePrVec>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TDoubleStrPr = std::pair<double, std::string>;
using TDoubleStrPrVec = std::vector<TDoubleStrPr>;
using TStrVec = std::vector<std::string>;
using TUIntVec = std::vector<unsigned int>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u>;
using TMaxAccumulator =
    maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;

const std::string EMPTY_STRING;

struct SAnomaly {
    SAnomaly() : s_Bucket(0u), s_Person(), s_Attributes() {}
    SAnomaly(std::size_t bucket, const std::string& person, const TDoubleStrPrVec& attributes)
        : s_Bucket(bucket), s_Person(person), s_Attributes(attributes) {}

    std::size_t s_Bucket;
    std::string s_Person;
    TDoubleStrPrVec s_Attributes;

    bool operator<(const SAnomaly& other) const {
        return s_Bucket < other.s_Bucket;
    }

    std::string print() const {
        std::ostringstream result;
        result << "[" << s_Bucket << ", " + s_Person << ",";
        for (std::size_t i = 0u; i < s_Attributes.size(); ++i) {
            if (s_Attributes[i].first < 0.01) {
                result << " " << s_Attributes[i].second;
            }
        }
        result << "]";
        return result.str();
    }
};

struct SMessage {
    SMessage(core_t::TTime time,
             const std::string& person,
             const std::string& attribute,
             const TDouble1Vec& value)
        : s_Time(time), s_Person(person), s_Attribute(attribute), s_Value(value) {}

    bool operator<(const SMessage& other) const {
        return maths::COrderings::lexicographical_compare(
            s_Time, s_Person, s_Attribute, other.s_Time, other.s_Person, other.s_Attribute);
    }

    core_t::TTime s_Time;
    std::string s_Person;
    std::string s_Attribute;
    TDouble1Vec s_Value;
};

using TMessageVec = std::vector<SMessage>;

const std::size_t numberAttributes = 5u;
const std::size_t numberPeople = 10u;

double roundToNearestPersisted(double value) {
    std::string valueAsString(core::CStringUtils::typeToStringPrecise(
        value, core::CIEEE754::E_DoublePrecision));
    double result = 0.0;
    core::CStringUtils::stringToType(valueAsString, result);
    return result;
}

void generateTestMessages(std::size_t dimension,
                          core_t::TTime startTime,
                          core_t::TTime bucketLength,
                          TMessageVec& messages) {
    // The test case is as follows:
    //
    //      attribute    |    0    |    1    |    2    |    3    |    4
    // ------------------+---------+---------+---------+---------+--------
    //        rate       |   10    |    2    |   15    |    2    |    1
    // ------------------+---------+---------+---------+---------+--------
    //        mean       |    5    |   10    |    7    |    3    |   15
    // ------------------+---------+---------+---------+---------+--------
    //     variance      |    1    |   0.5   |    2    |   0.1   |    4
    // ------------------+---------+---------+---------+---------+--------
    //  metric anomaly   | (12,2), |    -    | (30,5), | (12,2), | (60,2)
    //  (bucket, people) | (15,3), |         | (44,9)  | (80,1)  |
    //                   | (40,6)  |         |         |         |
    //
    // There are 10 people, 4 attributes and 100 buckets.

    const std::size_t numberBuckets = 100u;

    TStrVec people;
    for (std::size_t i = 0u; i < numberPeople; ++i) {
        people.push_back("p" + core::CStringUtils::typeToString(i));
    }
    LOG_DEBUG(<< "people = " << core::CContainerPrinter::print(people));

    TStrVec attributes;
    for (std::size_t i = 0u; i < numberAttributes; ++i) {
        attributes.push_back("c" + core::CStringUtils::typeToString(i));
    }
    LOG_DEBUG(<< "attributes = " << core::CContainerPrinter::print(attributes));

    double attributeRates[] = {10.0, 2.0, 15.0, 2.0, 1.0};
    double means[] = {5.0, 10.0, 7.0, 3.0, 15.0};
    double variances[] = {1.0, 0.5, 2.0, 0.1, 4.0};

    TSizeSizePr attribute0AnomalyBucketPerson[] = {
        TSizeSizePr(40u, 6u), TSizeSizePr(15u, 3u), TSizeSizePr(12u, 2u)};
    TSizeSizePr attribute2AnomalyBucketPerson[] = {TSizeSizePr(44u, 9u),
                                                   TSizeSizePr(30u, 5u)};
    TSizeSizePr attribute3AnomalyBucketPerson[] = {TSizeSizePr(80u, 1u),
                                                   TSizeSizePr(12u, 2u)};
    TSizeSizePr attribute4AnomalyBucketPerson[] = {TSizeSizePr(60u, 2u)};

    TSizeSizePrVecVec anomalies;
    anomalies.push_back(TSizeSizePrVec(std::begin(attribute0AnomalyBucketPerson),
                                       std::end(attribute0AnomalyBucketPerson)));
    anomalies.push_back(TSizeSizePrVec());
    anomalies.push_back(TSizeSizePrVec(std::begin(attribute2AnomalyBucketPerson),
                                       std::end(attribute2AnomalyBucketPerson)));
    anomalies.push_back(TSizeSizePrVec(std::begin(attribute3AnomalyBucketPerson),
                                       std::end(attribute3AnomalyBucketPerson)));
    anomalies.push_back(TSizeSizePrVec(std::begin(attribute4AnomalyBucketPerson),
                                       std::end(attribute4AnomalyBucketPerson)));

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberBuckets; ++i, startTime += bucketLength) {
        for (std::size_t j = 0u; j < numberAttributes; ++j) {
            TUIntVec samples;
            rng.generatePoissonSamples(attributeRates[j], numberPeople, samples);

            for (std::size_t k = 0u; k < numberPeople; ++k) {
                bool anomaly = !anomalies[j].empty() && anomalies[j].back().first == i &&
                               anomalies[j].back().second == k;
                if (anomaly) {
                    samples[k] += 4;
                    anomalies[j].pop_back();
                }

                if (samples[k] == 0) {
                    continue;
                }

                TDoubleVec values;
                rng.generateNormalSamples(means[j], variances[j],
                                          dimension * samples[k], values);

                for (std::size_t l = 0u; l < values.size(); l += dimension) {
                    TDouble1Vec value(dimension);
                    for (std::size_t d = 0u; d < dimension; ++d) {
                        double vd = values[l + d];
                        if (anomaly && (l % (2 * dimension)) == 0) {
                            vd += 6.0 * std::sqrt(variances[j]);
                        }
                        value[d] = roundToNearestPersisted(vd);
                    }
                    core_t::TTime dt = (static_cast<core_t::TTime>(l) * bucketLength) /
                                       static_cast<core_t::TTime>(values.size());
                    messages.push_back(SMessage(startTime + dt, people[k],
                                                attributes[j], value));
                }
            }
        }
    }

    LOG_DEBUG(<< "# messages = " << messages.size());
    std::sort(messages.begin(), messages.end());
}

std::string valueAsString(const TDouble1Vec& value) {
    std::string result = core::CStringUtils::typeToStringPrecise(
        value[0], core::CIEEE754::E_DoublePrecision);
    for (std::size_t i = 1u; i < value.size(); ++i) {
        result += CAnomalyDetectorModelConfig::DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER +
                  core::CStringUtils::typeToStringPrecise(
                      value[i], core::CIEEE754::E_DoublePrecision);
    }
    return result;
}

CEventData addArrival(const SMessage& message,
                      const CModelFactory::TDataGathererPtr& gatherer,
                      CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&message.s_Person);
    fields.push_back(&message.s_Attribute);
    std::string value = valueAsString(message.s_Value);
    fields.push_back(&value);
    CEventData result;
    result.time(message.s_Time);
    gatherer->addArrival(fields, result, resourceMonitor);
    return result;
}

void processBucket(core_t::TTime time,
                   core_t::TTime bucketLength,
                   std::size_t n,
                   const double* bucket,
                   const std::string* influencerValues,
                   CDataGatherer& gatherer,
                   CResourceMonitor& resourceMonitor,
                   CMetricPopulationModel& model,
                   SAnnotatedProbability& probability) {
    const std::string person("p");
    const std::string attribute("a");
    for (std::size_t i = 0u; i < n; ++i) {
        CDataGatherer::TStrCPtrVec fieldValues;
        fieldValues.push_back(&person);
        fieldValues.push_back(&attribute);
        fieldValues.push_back(&influencerValues[i]);
        std::string valueAsString(core::CStringUtils::typeToStringPrecise(
            bucket[i], core::CIEEE754::E_DoublePrecision));
        fieldValues.push_back(&valueAsString);

        CEventData eventData;
        eventData.time(time);

        gatherer.addArrival(fieldValues, eventData, resourceMonitor);
    }
    model.sample(time, time + bucketLength, resourceMonitor);
    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model.computeProbability(0 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability);
    LOG_DEBUG(<< "influences = " << core::CContainerPrinter::print(probability.s_Influences));
}
}

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testBasicAccessors, CTestFixture) {
    // Check that the correct data is read retrieved by the
    // basic model accessors.

    using TOptionalUInt64 = boost::optional<uint64_t>;
    using TStrUInt64Map = std::map<std::string, uint64_t>;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMinAccumulatorVec = std::vector<TMinAccumulator>;
    using TMaxAccumulatorVec = std::vector<TMaxAccumulator>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(1, startTime, bucketLength, messages);
    LOG_DEBUG(<< "# messages = " << messages.size());

    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(
        dynamic_cast<CDataGatherer*>(factory.makeDataGatherer(gathererInitData)));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));

    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, model->category());

    TStrUInt64Map expectedBucketPersonCounts;
    TMeanAccumulatorVec expectedBucketMeans(numberPeople * numberAttributes);
    TMinAccumulatorVec expectedBucketMins(numberPeople * numberAttributes);
    TMaxAccumulatorVec expectedBucketMaxs(numberPeople * numberAttributes);

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            LOG_DEBUG(<< "Testing bucket = [" << startTime << ","
                      << startTime + bucketLength << ")");

            BOOST_REQUIRE_EQUAL(numberPeople, gatherer->numberActivePeople());
            BOOST_REQUIRE_EQUAL(numberAttributes, gatherer->numberActiveAttributes());

            // Test the person and attribute invariants.
            for (std::size_t j = 0u; j < gatherer->numberActivePeople(); ++j) {
                const std::string& name = model->personName(j);
                std::size_t pid;
                BOOST_TEST_REQUIRE(gatherer->personId(name, pid));
                BOOST_REQUIRE_EQUAL(j, pid);
            }
            for (std::size_t j = 0u; j < gatherer->numberActiveAttributes(); ++j) {
                const std::string& name = model->attributeName(j);
                std::size_t cid;
                BOOST_TEST_REQUIRE(gatherer->attributeId(name, cid));
                BOOST_REQUIRE_EQUAL(j, cid);
            }

            LOG_DEBUG(<< "expected counts = "
                      << core::CContainerPrinter::print(expectedBucketPersonCounts));

            TSizeVec expectedCurrentBucketPersonIds;

            // Test the person counts.
            for (const auto& count_ : expectedBucketPersonCounts) {
                std::size_t pid;
                BOOST_TEST_REQUIRE(gatherer->personId(count_.first, pid));
                expectedCurrentBucketPersonIds.push_back(pid);
                TOptionalUInt64 count = model->currentBucketCount(pid, startTime);
                BOOST_TEST_REQUIRE(count);
                BOOST_REQUIRE_EQUAL(count_.second, *count);
            }

            std::sort(expectedCurrentBucketPersonIds.begin(),
                      expectedCurrentBucketPersonIds.end());

            TSizeVec bucketPersonIds;
            model->currentBucketPersonIds(startTime, bucketPersonIds);

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedCurrentBucketPersonIds),
                                core::CContainerPrinter::print(bucketPersonIds));

            if ((startTime / bucketLength) % 10 == 0) {
                LOG_DEBUG(<< "expected means = "
                          << core::CContainerPrinter::print(expectedBucketMeans));
                LOG_DEBUG(<< "expected mins = "
                          << core::CContainerPrinter::print(expectedBucketMins));
                LOG_DEBUG(<< "expected maxs = "
                          << core::CContainerPrinter::print(expectedBucketMaxs));
            }
            for (std::size_t cid = 0u; cid < numberAttributes; ++cid) {
                for (std::size_t pid = 0u; pid < numberPeople; ++pid) {
                    const TMeanAccumulator& expectedMean =
                        expectedBucketMeans[pid * numberAttributes + cid];
                    const TMinAccumulator& expectedMin =
                        expectedBucketMins[pid * numberAttributes + cid];
                    const TMaxAccumulator& expectedMax =
                        expectedBucketMaxs[pid * numberAttributes + cid];

                    TDouble1Vec mean = model->currentBucketValue(
                        model_t::E_PopulationMeanByPersonAndAttribute, pid, cid, startTime);
                    TDouble1Vec min = model->currentBucketValue(
                        model_t::E_PopulationMinByPersonAndAttribute, pid, cid, startTime);
                    TDouble1Vec max = model->currentBucketValue(
                        model_t::E_PopulationMaxByPersonAndAttribute, pid, cid, startTime);

                    if (mean.empty()) {
                        BOOST_TEST_REQUIRE(maths::CBasicStatistics::count(expectedMean) == 0.0);
                    } else {
                        BOOST_TEST_REQUIRE(maths::CBasicStatistics::count(expectedMean) > 0.0);
                        BOOST_REQUIRE_EQUAL(
                            maths::CBasicStatistics::mean(expectedMean), mean[0]);
                    }
                    if (min.empty()) {
                        BOOST_TEST_REQUIRE(expectedMin.count() == 0u);
                    } else {
                        BOOST_TEST_REQUIRE(expectedMin.count() > 0u);
                        BOOST_REQUIRE_EQUAL(expectedMin[0], min[0]);
                    }
                    if (max.empty()) {
                        BOOST_TEST_REQUIRE(expectedMax.count() == 0u);
                    } else {
                        BOOST_TEST_REQUIRE(expectedMax.count() > 0u);
                        BOOST_REQUIRE_EQUAL(expectedMax[0], max[0]);
                    }
                }
            }

            expectedBucketMeans = TMeanAccumulatorVec(numberPeople * numberAttributes);
            expectedBucketMins = TMinAccumulatorVec(numberPeople * numberAttributes);
            expectedBucketMaxs = TMaxAccumulatorVec(numberPeople * numberAttributes);
            expectedBucketPersonCounts.clear();
            startTime += bucketLength;
        }

        CEventData eventData = addArrival(message, gatherer, m_ResourceMonitor);
        std::size_t pid = *eventData.personId();
        std::size_t cid = *eventData.attributeId();
        ++expectedBucketPersonCounts[message.s_Person];
        expectedBucketMeans[pid * numberAttributes + cid].add(message.s_Value[0]);
        expectedBucketMins[pid * numberAttributes + cid].add(message.s_Value[0]);
        expectedBucketMaxs[pid * numberAttributes + cid].add(message.s_Value[0]);
    }
}

BOOST_FIXTURE_TEST_CASE(testMinMaxAndMean, CTestFixture) {
    // We check that the correct data is read from the gatherer
    // into the model on sample.

    using TTimeDouble2VecSizeTr = core::CTriple<core_t::TTime, TDouble2Vec, std::size_t>;
    using TTimeDouble2VecSizeTrVec = std::vector<TTimeDouble2VecSizeTr>;
    using TDouble2VecWeightsAry = maths_t::TDouble2VecWeightsAry;
    using TDouble2VecWeightsAryVec = std::vector<TDouble2VecWeightsAry>;
    using TSizeSizePrDoubleVecMap = std::map<TSizeSizePr, TDoubleVec>;
    using TSizeSizePrMeanAccumulatorUMap = std::map<TSizeSizePr, TMeanAccumulator>;
    using TSizeSizePrMinAccumulatorMap = std::map<TSizeSizePr, TMinAccumulator>;
    using TSizeSizePrMaxAccumulatorMap = std::map<TSizeSizePr, TMaxAccumulator>;
    using TMathsModelPtr = std::shared_ptr<maths::CModel>;
    using TSizeMathsModelPtrMap = std::map<std::size_t, TMathsModelPtr>;
    using TTimeDouble2VecSizeTrVecDouble2VecWeightsAryVecPr =
        std::pair<TTimeDouble2VecSizeTrVec, TDouble2VecWeightsAryVec>;
    using TSizeTimeDouble2VecSizeTrVecDouble2VecWeightsAryVecPrMap =
        std::map<std::size_t, TTimeDouble2VecSizeTrVecDouble2VecWeightsAryVecPr>;
    using TSizeSizeTimeDouble2VecSizeTrVecDouble2VecWeightAryVecPrMapMap =
        std::map<std::size_t, TSizeTimeDouble2VecSizeTrVecDouble2VecWeightsAryVecPrMap>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(1, startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    params.s_MaximumUpdatesPerBucket = 0.0;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute,
                                  model_t::E_PopulationMinByPersonAndAttribute,
                                  model_t::E_PopulationMaxByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(
        dynamic_cast<CDataGatherer*>(factory.makeDataGatherer(gathererInitData)));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
    CMetricPopulationModel* model =
        dynamic_cast<CMetricPopulationModel*>(modelHolder.get());

    CModelFactory::TFeatureMathsModelPtrPrVec models{
        factory.defaultFeatureModels(features, bucketLength, 1.0, false)};
    BOOST_REQUIRE_EQUAL(features.size(), models.size());
    BOOST_REQUIRE_EQUAL(features[0], models[0].first);
    BOOST_REQUIRE_EQUAL(features[1], models[1].first);
    BOOST_REQUIRE_EQUAL(features[2], models[2].first);

    TSizeSizePrMeanAccumulatorUMap sampleTimes;
    TSizeSizePrMeanAccumulatorUMap sampleMeans;
    TSizeSizePrMinAccumulatorMap sampleMins;
    TSizeSizePrMaxAccumulatorMap sampleMaxs;
    TSizeSizePrDoubleVecMap expectedSampleTimes;
    TSizeSizePrDoubleVecMap expectedSamples[3];
    TSizeMathsModelPtrMap expectedPopulationModels[3];
    bool nonNegative = true;

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            TSizeSizeTimeDouble2VecSizeTrVecDouble2VecWeightAryVecPrMapMap populationWeightedSamples;
            for (std::size_t feature = 0u; feature < features.size(); ++feature) {
                for (const auto& samples_ : expectedSamples[feature]) {
                    std::size_t pid = samples_.first.first;
                    std::size_t cid = samples_.first.second;
                    TTimeDouble2VecSizeTrVec& samples =
                        populationWeightedSamples[feature][cid].first;
                    TDouble2VecWeightsAryVec& weights =
                        populationWeightedSamples[feature][cid].second;
                    TMathsModelPtr& model_ = expectedPopulationModels[feature][cid];
                    if (!model_) {
                        model_ = factory.defaultFeatureModel(
                            features[feature], bucketLength, 1.0, false);
                    }
                    for (std::size_t j = 0u; j < samples_.second.size(); ++j) {
                        // We round to the nearest integer time (note this has to match
                        // the behaviour of CMetricPartialStatistic::time).
                        core_t::TTime time_ = static_cast<core_t::TTime>(
                            expectedSampleTimes[{pid, cid}][j] + 0.5);
                        TDouble2Vec sample{samples_.second[j]};
                        samples.emplace_back(time_, sample, pid);
                        weights.push_back(maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                        auto& weight = weights.back();
                        maths_t::setCount(
                            TDouble2Vec{model->sampleRateWeight(pid, cid)}, weight);
                        maths_t::setWinsorisationWeight(
                            model_->winsorisationWeight(1.0, time_, sample), weight);
                    }
                }
            }
            for (auto& feature : populationWeightedSamples) {
                for (auto& attribute : feature.second) {
                    std::size_t cid = attribute.first;
                    TTimeDouble2VecSizeTrVec& samples = attribute.second.first;
                    TDouble2VecWeightsAryVec& weights = attribute.second.second;
                    maths::COrderings::simultaneousSort(samples, weights);
                    maths::CModelAddSamplesParams params_;
                    params_.integer(false)
                        .nonNegative(nonNegative)
                        .propagationInterval(1.0)
                        .trendWeights(weights)
                        .priorWeights(weights);
                    expectedPopulationModels[feature.first][cid]->addSamples(params_, samples);
                }
            }

            for (std::size_t feature = 0u; feature < features.size(); ++feature) {
                if ((startTime / bucketLength) % 10 == 0) {
                    LOG_DEBUG(<< "Testing priors for feature "
                              << model_t::print(features[feature]));
                }
                for (std::size_t cid = 0u; cid < numberAttributes; ++cid) {
                    if (expectedPopulationModels[feature].count(cid) > 0) {
                        BOOST_REQUIRE_EQUAL(
                            expectedPopulationModels[feature][cid]->checksum(),
                            model->details()->model(features[feature], cid)->checksum());
                    }
                }
            }

            expectedSampleTimes.clear();
            expectedSamples[0].clear();
            expectedSamples[1].clear();
            expectedSamples[2].clear();
            startTime += bucketLength;
        }

        CEventData eventData = addArrival(message, gatherer, m_ResourceMonitor);
        std::size_t pid = *eventData.personId();
        std::size_t cid = *eventData.attributeId();
        nonNegative &= message.s_Value[0] < 0.0;

        double sampleCount = gatherer->sampleCount(cid);
        if (sampleCount > 0.0) {
            TSizeSizePr key{pid, cid};
            sampleTimes[key].add(static_cast<double>(message.s_Time));
            sampleMeans[key].add(message.s_Value[0]);
            sampleMins[key].add(message.s_Value[0]);
            sampleMaxs[key].add(message.s_Value[0]);
            if (maths::CBasicStatistics::count(sampleTimes[key]) == sampleCount) {
                expectedSampleTimes[key].push_back(
                    maths::CBasicStatistics::mean(sampleTimes[key]));
                expectedSamples[0][key].push_back(
                    maths::CBasicStatistics::mean(sampleMeans[key]));
                expectedSamples[1][key].push_back(sampleMins[key][0]);
                expectedSamples[2][key].push_back(sampleMaxs[key][0]);
                sampleTimes[key] = TMeanAccumulator();
                sampleMeans[key] = TMeanAccumulator();
                sampleMins[key] = TMinAccumulator();
                sampleMaxs[key] = TMaxAccumulator();
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testVarp, CTestFixture) {
    core_t::TTime startTime(3600);
    core_t::TTime bucketLength(3600);
    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationVarianceByPersonAndAttribute});
    factory.fieldNames("", "P", "", "V", TStrVec(1, "I"));
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    BOOST_TEST_REQUIRE(gatherer->isPopulation());
    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr model_(factory.makeModel(initData));
    BOOST_TEST_REQUIRE(model_);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, model_->category());
    CMetricPopulationModel& model = static_cast<CMetricPopulationModel&>(*model_.get());

    double bucket1[] = {1.0, 1.1, 1.01, 1.02};
    std::string influencerValues1[] = {"i1", "i1", "i2", "i2"};
    double bucket2[] = {10.0};
    std::string influencerValues2[] = {"i1"};
    double bucket3[] = {4.3, 4.4, 4.6, 4.2, 4.8};
    std::string influencerValues3[] = {"i1", "i1", "i1", "i1", "i3"};
    double bucket4[] = {3.2, 3.3};
    std::string influencerValues4[] = {"i3", "i3"};
    double bucket5[] = {20.1, 20.8, 20.9};
    std::string influencerValues5[] = {"i2", "i1", "i1"};
    double bucket6[] = {4.1, 4.2, 3.9, 4.2};
    std::string influencerValues6[] = {"i1", "i2", "i2", "i2"};
    double bucket7[] = {0.1, 0.3, 0.2};
    std::string influencerValues7[] = {"i1", "i1", "i3"};
    double bucket8[] = {12.5, 12.3};
    std::string influencerValues8[] = {"i1", "i2"};
    double bucket9[] = {6.9, 7.0, 7.1, 6.6, 7.1, 6.7};
    std::string influencerValues9[] = {"i1", "i2", "i3", "i4", "i5", "i6"};
    // This last bucket is much more improbable, with influencer i2 being responsible
    double bucket10[] = {0.3, 15.4, 77.62, 112.999, 5.1, 5.1, 5.1, 5.1, 5.1};
    std::string influencerValues10[] = {"i2", "i2", "i2", "i2", "i1",
                                        "i1", "i1", "i1", "i1"};

    SAnnotatedProbability annotatedProbability;

    core_t::TTime time = startTime;
    processBucket(time, bucketLength, boost::size(bucket1), bucket1, influencerValues1,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket2), bucket2, influencerValues2,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket3), bucket3, influencerValues3,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket4), bucket4, influencerValues4,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket5), bucket5, influencerValues5,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket6), bucket6, influencerValues6,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket7), bucket7, influencerValues7,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket8), bucket8, influencerValues8,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket9), bucket9, influencerValues9,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability < 0.85);

    time += bucketLength;
    processBucket(time, bucketLength, boost::size(bucket10), bucket10, influencerValues10,
                  *gatherer, m_ResourceMonitor, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability < 0.1);
    BOOST_REQUIRE_EQUAL(std::size_t(1), annotatedProbability.s_Influences.size());
    BOOST_REQUIRE_EQUAL(std::string("I"),
                        *annotatedProbability.s_Influences[0].first.first);
    BOOST_REQUIRE_EQUAL(std::string("i2"),
                        *annotatedProbability.s_Influences[0].first.second);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, annotatedProbability.s_Influences[0].second, 0.00001);
}

BOOST_FIXTURE_TEST_CASE(testComputeProbability, CTestFixture) {
    maths::CSampling::CScopeMockRandomNumberGenerator scopeMockRng;

    // Test that we correctly pick out synthetic the anomalies,
    // their people and attributes.

    using TAnomalyVec = std::vector<SAnomaly>;
    using TDoubleAnomalyPr = std::pair<double, SAnomaly>;
    using TAnomalyAccumulator =
        maths::CBasicStatistics::COrderStatisticsHeap<TDoubleAnomalyPr, maths::COrderings::SFirstLess>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    model_t::EFeature features_[] = {model_t::E_PopulationMaxByPersonAndAttribute,
                                     model_t::E_PopulationMeanLatLongByPersonAndAttribute};

    for (std::size_t i = 0u; i < boost::size(features_); ++i) {
        LOG_DEBUG(<< "Testing " << model_t::print(features_[i]));

        TMessageVec messages;
        generateTestMessages(model_t::dimension(features_[i]), startTime,
                             bucketLength, messages);

        SModelParams params(bucketLength);
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        CMetricPopulationModelFactory factory(params, interimBucketCorrector);
        factory.features({features_[i]});
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CMetricPopulationModel* model =
            dynamic_cast<CMetricPopulationModel*>(modelHolder.get());

        TAnomalyAccumulator anomalies(7);

        std::size_t bucket = 0u;
        for (const auto& message : messages) {
            if (message.s_Time >= startTime + bucketLength) {
                model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

                LOG_DEBUG(<< "Testing bucket " << bucket << " = [" << startTime
                          << "," << startTime + bucketLength << ")");

                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                SAnnotatedProbability annotatedProbability;
                for (std::size_t pid = 0u; pid < numberPeople; ++pid) {
                    model->computeProbability(pid, startTime, startTime + bucketLength,
                                              partitioningFields, 2, annotatedProbability);

                    if ((startTime / bucketLength) % 10 == 0) {
                        LOG_DEBUG(<< "person = " << model->personName(pid) << ", probability = "
                                  << annotatedProbability.s_Probability);
                    }
                    std::string person = model->personName(pid);
                    TDoubleStrPrVec attributes;
                    for (const auto& probability : annotatedProbability.s_AttributeProbabilities) {
                        attributes.emplace_back(probability.s_Probability,
                                                *probability.s_Attribute);
                    }
                    anomalies.add({annotatedProbability.s_Probability,
                                   SAnomaly(bucket, person, attributes)});
                }

                startTime += bucketLength;
                ++bucket;
            }

            addArrival(message, gatherer, m_ResourceMonitor);
        }

        anomalies.sort();
        LOG_DEBUG(<< "Anomalies = " << anomalies.print());

        TAnomalyVec orderedAnomalies;
        for (std::size_t j = 0u; j < anomalies.count(); ++j) {
            orderedAnomalies.push_back(anomalies[j].second);
        }
        std::sort(orderedAnomalies.begin(), orderedAnomalies.end());

        LOG_DEBUG(<< "orderedAnomalies = "
                  << core::CContainerPrinter::print(orderedAnomalies));

        std::string expectedAnomalies[] = {
            std::string("[12, p2, c0 c3]"), std::string("[15, p3, c0]"),
            std::string("[30, p5, c2]"),    std::string("[40, p6, c0]"),
            std::string("[44, p9, c2]"),    std::string("[60, p2, c4]"),
            std::string("[80, p1, c3]")};

        BOOST_REQUIRE_EQUAL(boost::size(expectedAnomalies), orderedAnomalies.size());
        for (std::size_t j = 0u; j < orderedAnomalies.size(); ++j) {
            BOOST_REQUIRE_EQUAL(expectedAnomalies[j], orderedAnomalies[j].print());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testPrune, CTestFixture) {
    // This test has four people and five attributes. We expect
    // person 2 and attributes 1, 2 and 5 to be deleted.

    using TStrSizePr = std::pair<std::string, std::size_t>;
    using TStrSizePrVec = std::vector<TStrSizePr>;
    using TStrSizePrVecVec = std::vector<TStrSizePrVec>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 1000u;

    std::string people[] = {std::string("p1"), std::string("p2"),
                            std::string("p3"), std::string("p4")};
    std::string attributes[] = {std::string("c1"), std::string("c2"), std::string("c3"),
                                std::string("c4"), std::string("c5")};

    TStrSizePrVecVec eventCounts[] = {TStrSizePrVecVec(), TStrSizePrVecVec(),
                                      TStrSizePrVecVec(), TStrSizePrVecVec()};
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.push_back(TStrSizePr(attributes[0], 0));
        attributeCounts.push_back(TStrSizePr(attributes[4], 0));
        eventCounts[0].resize(numberBuckets, attributeCounts);
        eventCounts[0][0][0].second = 2; // p1, bucket 1,  c1
        eventCounts[0][2][0].second = 3; // p1, bucket 3,  c1
        eventCounts[0][4][0].second = 4; // p1, bucket 5,  c1
        eventCounts[0][8][0].second = 5; // p1, bucket 9,  c1
        eventCounts[0][9][0].second = 3; // p1, bucket 10, c1
        eventCounts[0][0][1].second = 4; // p1, bucket 1,  c5
        eventCounts[0][3][1].second = 1; // p1, bucket 4,  c5
        eventCounts[0][7][1].second = 1; // p1, bucket 8,  c5
        eventCounts[0][8][1].second = 3; // p1, bucket 9,  c5
    }
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.push_back(TStrSizePr(attributes[0], 0));
        attributeCounts.push_back(TStrSizePr(attributes[1], 0));
        attributeCounts.push_back(TStrSizePr(attributes[2], 1));
        attributeCounts.push_back(TStrSizePr(attributes[4], 0));
        eventCounts[1].resize(numberBuckets, attributeCounts);
        eventCounts[1][1][0].second = 2; // p2, bucket 2, c1
        eventCounts[1][3][0].second = 4; // p2, bucket 3, c1
        eventCounts[1][4][0].second = 4; // p2, bucket 5, c1
        eventCounts[1][7][0].second = 3; // p2, bucket 8, c1
        eventCounts[1][5][1].second = 4; // p2, bucket 6, c2
        eventCounts[1][6][1].second = 4; // p2, bucket 7, c2
        eventCounts[1][7][1].second = 3; // p2, bucket 8, c2
        eventCounts[1][8][1].second = 3; // p2, bucket 9, c2
        eventCounts[1][0][3].second = 3; // p2, bucket 1, c5
        eventCounts[1][1][3].second = 3; // p2, bucket 2, c5
        eventCounts[1][2][3].second = 3; // p2, bucket 3, c5
        eventCounts[1][3][3].second = 3; // p2, bucket 4, c5
        eventCounts[1][4][3].second = 2; // p2, bucket 5, c5
        eventCounts[1][5][3].second = 2; // p2, bucket 6, c5
    }
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.push_back(TStrSizePr(attributes[2], 0));
        attributeCounts.push_back(TStrSizePr(attributes[3], 2));
        eventCounts[2].resize(numberBuckets, attributeCounts);
        eventCounts[2][0][0].second = 1;   // p3, bucket 1,   c3
        eventCounts[2][20][0].second = 4;  // p3, bucket 21,  c3
        eventCounts[2][25][0].second = 6;  // p3, bucket 26,  c3
        eventCounts[2][80][0].second = 3;  // p3, bucket 81,  c3
        eventCounts[2][180][0].second = 7; // p3, bucket 181, c3
        eventCounts[2][200][0].second = 9; // p3, bucket 201, c3
        eventCounts[2][800][0].second = 2; // p3, bucket 801, c3
    }
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.push_back(TStrSizePr(attributes[1], 0));
        attributeCounts.push_back(TStrSizePr(attributes[3], 3));
        eventCounts[3].resize(numberBuckets, attributeCounts);
        eventCounts[3][0][0].second = 2;  // p4, bucket 1,  c2
        eventCounts[3][1][0].second = 1;  // p4, bucket 2,  c2
        eventCounts[3][2][0].second = 5;  // p4, bucket 3,  c2
        eventCounts[3][15][0].second = 3; // p4, bucket 16, c2
        eventCounts[3][26][0].second = 1; // p4, bucket 27, c2
        eventCounts[3][70][0].second = 4; // p4, bucket 70, c2
    }

    const std::string expectedPeople[] = {people[1], people[2], people[3]};
    const std::string expectedAttributes[] = {attributes[2], attributes[3]};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.01;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));
    BOOST_TEST_REQUIRE(model);
    CModelFactory::TDataGathererPtr expectedGatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData expectedModelInitData(expectedGatherer);
    CAnomalyDetectorModel::TModelPtr expectedModel(factory.makeModel(expectedModelInitData));
    BOOST_TEST_REQUIRE(expectedModel);

    test::CRandomNumbers rng;

    TMessageVec messages;
    for (std::size_t i = 0u; i < boost::size(people); ++i) {
        core_t::TTime bucketStart = startTime;
        for (std::size_t j = 0u; j < numberBuckets; ++j, bucketStart += bucketLength) {
            const TStrSizePrVec& attributeEventCounts = eventCounts[i][j];
            for (std::size_t k = 0u; k < attributeEventCounts.size(); ++k) {
                if (attributeEventCounts[k].second == 0) {
                    continue;
                }

                std::size_t n = attributeEventCounts[k].second;

                TDoubleVec samples;
                rng.generateUniformSamples(0.0, 8.0, n, samples);

                core_t::TTime time = bucketStart;
                core_t::TTime dt = bucketLength / static_cast<core_t::TTime>(n);

                for (std::size_t l = 0u; l < n; ++l, time += dt) {
                    messages.push_back(SMessage(time, people[i],
                                                attributeEventCounts[k].first,
                                                TDouble1Vec(1, samples[l])));
                }
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    TMessageVec expectedMessages;
    expectedMessages.reserve(messages.size());
    for (std::size_t i = 0u; i < messages.size(); ++i) {
        if (std::binary_search(std::begin(expectedPeople),
                               std::end(expectedPeople), messages[i].s_Person) &&
            std::binary_search(std::begin(expectedAttributes),
                               std::end(expectedAttributes), messages[i].s_Attribute)) {
            expectedMessages.push_back(messages[i]);
        }
    }

    core_t::TTime bucketStart = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(message, gatherer, m_ResourceMonitor);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune();
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    BOOST_REQUIRE_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = startTime;
    for (std::size_t i = 0u; i < expectedMessages.size(); ++i) {
        if (expectedMessages[i].s_Time >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(expectedMessages[i], expectedGatherer, m_ResourceMonitor);
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person and attribute slots.

    bucketStart = gatherer->currentBucketStartTime() + bucketLength;

    SMessage newMessages[] = {
        SMessage(bucketStart + 10, "p1", "c2", TDouble1Vec(1, 20.0)),
        SMessage(bucketStart + 200, "p5", "c6", TDouble1Vec(1, 10.0)),
        SMessage(bucketStart + 2100, "p5", "c6", TDouble1Vec(1, 15.0))};

    for (std::size_t i = 0u; i < boost::size(newMessages); ++i) {
        addArrival(newMessages[i], gatherer, m_ResourceMonitor);
        addArrival(newMessages[i], expectedGatherer, m_ResourceMonitor);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Test that calling prune on a cloned model which has seen no new data does nothing
    CAnomalyDetectorModel::TModelPtr clonedModelHolder(model->cloneForPersistence());
    std::size_t numberOfPeopleBeforePrune(
        clonedModelHolder->dataGatherer().numberActivePeople());
    BOOST_TEST_REQUIRE(numberOfPeopleBeforePrune > 0);
    clonedModelHolder->prune(clonedModelHolder->defaultPruneWindow());
    BOOST_REQUIRE_EQUAL(numberOfPeopleBeforePrune,
                        clonedModelHolder->dataGatherer().numberActivePeople());
}

BOOST_FIXTURE_TEST_CASE(testKey, CTestFixture) {
    function_t::EFunction countFunctions[] = {
        function_t::E_PopulationMetric, function_t::E_PopulationMetricMean,
        function_t::E_PopulationMetricMin, function_t::E_PopulationMetricMax,
        function_t::E_PopulationMetricSum};
    bool useNull[] = {true, false};
    std::string byField[] = {"", "by"};
    std::string partitionField[] = {"", "partition"};

    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();

        int identifier = 0;
        for (std::size_t i = 0u; i < boost::size(countFunctions); ++i) {
            for (std::size_t j = 0u; j < boost::size(useNull); ++j) {
                for (std::size_t k = 0u; k < boost::size(byField); ++k) {
                    for (std::size_t l = 0u; l < boost::size(partitionField); ++l) {
                        CSearchKey key(++identifier, countFunctions[i],
                                       useNull[j], model_t::E_XF_None, "value",
                                       byField[k], "over", partitionField[l]);

                        CAnomalyDetectorModelConfig::TModelFactoryCPtr factory =
                            config.factory(key);

                        LOG_DEBUG(<< "expected key = " << key);
                        LOG_DEBUG(<< "actual key   = " << factory->searchKey());
                        BOOST_TEST_REQUIRE(key == factory->searchKey());
                    }
                }
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testFrequency, CTestFixture) {
    // Test we correctly compute frequencies for people and attributes.

    const core_t::TTime bucketLength = 600;
    const std::string attributes[] = {"a1", "a2", "a3", "a4", "a5",
                                      "a6", "a7", "a8", "a9", "a10"};
    const std::string people[] = {"p1", "p2", "p3", "p4", "p5",
                                  "p6", "p7", "p8", "p9", "p10"};
    std::size_t period[] = {1u, 1u, 10u, 3u, 4u, 5u, 2u, 1u, 3u, 7u};

    core_t::TTime startTime = 0;

    TMessageVec messages;
    std::size_t bucket = 0u;
    for (core_t::TTime bucketStart = startTime; bucketStart < 100 * bucketLength;
         bucketStart += bucketLength, ++bucket) {
        for (std::size_t i = 0u; i < boost::size(people); ++i) {
            if (bucket % period[i] == 0) {
                for (std::size_t j = 0u; j < i + 1; ++j) {
                    messages.push_back(SMessage(bucketStart + bucketLength / 2, people[i],
                                                attributes[j], TDouble1Vec(1, 0.0)));
                }
            }
        }
    }

    std::sort(messages.begin(), messages.end());
    LOG_DEBUG(<< "# messages = " << messages.size());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationMeanByPersonAndAttribute});
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    const model::CDataGatherer& populationGatherer(
        dynamic_cast<const model::CDataGatherer&>(*gatherer));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));

    CMetricPopulationModel* populationModel =
        dynamic_cast<CMetricPopulationModel*>(model.get());
    BOOST_TEST_REQUIRE(populationModel);

    core_t::TTime time = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            populationModel->sample(time, time + bucketLength, m_ResourceMonitor);
            time += bucketLength;
        }
        addArrival(message, gatherer, m_ResourceMonitor);
    }

    {
        TMeanAccumulator meanError;
        for (std::size_t i = 0u; i < boost::size(people); ++i) {
            LOG_DEBUG(<< "*** person = " << people[i] << " ***");
            std::size_t pid;
            BOOST_TEST_REQUIRE(gatherer->personId(people[i], pid));
            LOG_DEBUG(<< "frequency = " << populationModel->personFrequency(pid));
            LOG_DEBUG(<< "expected frequency = " << 1.0 / static_cast<double>(period[i]));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0 / static_cast<double>(period[i]),
                                         populationModel->personFrequency(pid),
                                         0.1 / static_cast<double>(period[i]));
            meanError.add(std::fabs(populationModel->personFrequency(pid) -
                                    1.0 / static_cast<double>(period[i])));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(meanError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.002);
    }
    {
        for (std::size_t i = 0u; i < boost::size(attributes); ++i) {
            LOG_DEBUG(<< "*** attributes = " << attributes[i] << " ***");
            std::size_t cid;
            BOOST_TEST_REQUIRE(populationGatherer.attributeId(attributes[i], cid));
            LOG_DEBUG(<< "frequency = " << populationModel->attributeFrequency(cid));
            LOG_DEBUG(<< "expected frequency = " << (10.0 - static_cast<double>(i)) / 10.0);
            BOOST_REQUIRE_EQUAL((10.0 - static_cast<double>(i)) / 10.0,
                                populationModel->attributeFrequency(cid));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testSampleRateWeight, CTestFixture) {
    // Test that we correctly compensate for heavy hitters.

    // There are 10 attributes.
    //
    // People p1 and p5 generate messages for every attribute every bucket.
    // The remaining 18 people only generate one message per bucket, i.e.
    // one message per attribute per 10 buckets.

    const core_t::TTime bucketLength = 600;
    const std::string attributes[] = {"a1", "a2", "a3", "a4", "a5",
                                      "a6", "a7", "a8", "a9", "a10"};
    const std::string people[] = {
        "p1",  "p2",  "p3",  "p4",  "p5",  "p6",  "p7",  "p8",  "p9",  "p10",
        "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"};
    std::size_t heavyHitters[] = {0u, 4u};
    std::size_t normal[] = {1u,  2u,  3u,  5u,  6u,  7u,  8u,  9u,  10u,
                            11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u};

    std::size_t messagesPerBucket =
        boost::size(heavyHitters) * boost::size(attributes) + boost::size(normal);

    test::CRandomNumbers rng;

    core_t::TTime startTime = 0;

    TMessageVec messages;
    for (core_t::TTime bucketStart = startTime;
         bucketStart < 100 * bucketLength; bucketStart += bucketLength) {
        TSizeVec times;
        rng.generateUniformSamples(static_cast<std::size_t>(bucketStart),
                                   static_cast<std::size_t>(bucketStart + bucketLength),
                                   messagesPerBucket, times);

        std::size_t m = 0u;
        for (std::size_t i = 0u; i < boost::size(attributes); ++i) {
            for (std::size_t j = 0u; j < boost::size(heavyHitters); ++j) {
                messages.push_back(SMessage(static_cast<core_t::TTime>(times[m++]),
                                            people[heavyHitters[j]],
                                            attributes[i], TDouble1Vec(1, 0.0)));
            }
        }

        TSizeVec attributeIndexes;
        rng.generateUniformSamples(0, boost::size(attributes),
                                   boost::size(normal), attributeIndexes);
        for (std::size_t i = 0u; i < boost::size(normal); ++i) {
            messages.push_back(
                SMessage(static_cast<core_t::TTime>(times[m++]), people[normal[i]],
                         attributes[attributeIndexes[i]], TDouble1Vec(1, 0.0)));
        }
    }

    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationSumByBucketPersonAndAttribute});
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));

    CMetricPopulationModel* populationModel =
        dynamic_cast<CMetricPopulationModel*>(model.get());
    BOOST_TEST_REQUIRE(populationModel);

    core_t::TTime time = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            populationModel->sample(time, time + bucketLength, m_ResourceMonitor);
            time += bucketLength;
        }
        addArrival(message, gatherer, m_ResourceMonitor);
    }

    // The heavy hitters generate one value per attribute per bucket.
    // The rest generate one value per bucket. Therefore, we expect
    // the mean person rate per bucket to be:
    //     (  ("# people" - "# heavy hitters" / "# attributes")
    //      + ("# heavy hitters"))
    //   / "# people"

    double expectedRateWeight = (static_cast<double>(boost::size(normal)) /
                                     static_cast<double>(boost::size(attributes)) +
                                 static_cast<double>(boost::size(heavyHitters))) /
                                static_cast<double>(boost::size(people));
    LOG_DEBUG(<< "expectedRateWeight = " << expectedRateWeight);

    for (std::size_t i = 0u; i < boost::size(heavyHitters); ++i) {
        LOG_DEBUG(<< "*** person = " << people[heavyHitters[i]] << " ***");
        std::size_t pid;
        BOOST_TEST_REQUIRE(gatherer->personId(people[heavyHitters[i]], pid));
        for (std::size_t cid = 0u; cid < boost::size(attributes); ++cid) {
            double sampleRateWeight = populationModel->sampleRateWeight(pid, cid);
            LOG_DEBUG(<< "attribute = " << populationModel->attributeName(cid)
                      << ", sampleRateWeight = " << sampleRateWeight);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedRateWeight, sampleRateWeight,
                                         0.15 * expectedRateWeight);
        }
    }

    for (std::size_t i = 0u; i < boost::size(normal); ++i) {
        LOG_DEBUG(<< "*** person = " << people[normal[i]] << " ***");
        std::size_t pid;
        BOOST_TEST_REQUIRE(gatherer->personId(people[normal[i]], pid));
        for (std::size_t cid = 0u; cid < boost::size(attributes); ++cid) {
            double sampleRateWeight = populationModel->sampleRateWeight(pid, cid);
            LOG_DEBUG(<< "attribute = " << populationModel->attributeName(cid)
                      << ", sampleRateWeight = " << sampleRateWeight);
            BOOST_REQUIRE_EQUAL(1.0, sampleRateWeight);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testPeriodicity, CTestFixture) {
    // Create a daily periodic population and check that the
    // periodicity is learned and compensated (approximately).

    using TStrDoubleMap = std::map<std::string, double>;

    static const core_t::TTime HOUR = 3600;
    static const core_t::TTime DAY = 86400;

    const core_t::TTime bucketLength = 3600;
    double baseline[] = {1, 1, 2, 2,  3, 5, 6, 6, 20, 21, 4, 3,
                         4, 4, 8, 25, 7, 6, 5, 1, 1,  4,  1, 1};
    const std::string attributes[] = {"a1", "a2"};
    double scales[] = {2.0, 3.0};
    const std::string people[] = {"p1", "p2", "p3", "p4", "p5",
                                  "p6", "p7", "p8", "p9", "p10"};

    test::CRandomNumbers rng;

    core_t::TTime startTime = 0;
    core_t::TTime endTime = 604800;

    TMessageVec messages;
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        for (std::size_t i = 0u; i < boost::size(attributes); ++i) {
            TDoubleVec values;
            rng.generateNormalSamples(baseline[(time % DAY) / HOUR],
                                      scales[i] * scales[i], boost::size(people), values);

            for (std::size_t j = 0u; j < values.size(); ++j) {
                for (unsigned int t = 0; t < 4; ++t) {
                    messages.push_back(SMessage(time + (t * bucketLength) / 4,
                                                people[j], attributes[i],
                                                TDouble1Vec(1, values[j])));
                }
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationMeanByPersonAndAttribute});

    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));
    CMetricPopulationModel* populationModel =
        dynamic_cast<CMetricPopulationModel*>(model.get());
    BOOST_TEST_REQUIRE(populationModel);

    TStrDoubleMap personProbabilitiesWithoutPeriodicity;
    TStrDoubleMap personProbabilitiesWithPeriodicity;

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i) {
        if (messages[i].s_Time >= time + bucketLength) {
            populationModel->sample(time, time + bucketLength, m_ResourceMonitor);

            for (std::size_t j = 0u; j < boost::size(people); ++j) {
                std::size_t pid;
                if (!gatherer->personId(people[j], pid)) {
                    continue;
                }

                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                SAnnotatedProbability annotatedProbability;
                if (populationModel->computeProbability(
                        pid, time, time + bucketLength, partitioningFields, 1,
                        annotatedProbability) == false) {
                    continue;
                }

                if (time < startTime + 3 * DAY) {
                    double& minimumProbability = personProbabilitiesWithoutPeriodicity
                                                     .insert({people[j], 1.0})
                                                     .first->second;
                    minimumProbability = std::min(
                        minimumProbability, annotatedProbability.s_Probability);
                } else if (time > startTime + 5 * DAY) {
                    double& minimumProbability = personProbabilitiesWithPeriodicity
                                                     .insert({people[j], 1.0})
                                                     .first->second;
                    minimumProbability = std::min(
                        minimumProbability, annotatedProbability.s_Probability);
                }
            }
            time += bucketLength;
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    double totalw = 0.0;
    double totalwo = 0.0;

    for (std::size_t i = 0u; i < boost::size(people); ++i) {
        auto wo = personProbabilitiesWithoutPeriodicity.find(people[i]);
        auto w = personProbabilitiesWithPeriodicity.find(people[i]);
        LOG_DEBUG(<< "person = " << people[i]);
        LOG_DEBUG(<< "minimum probability with periodicity    = " << w->second);
        LOG_DEBUG(<< "minimum probability without periodicity = " << wo->second);
        totalwo += wo->second;
        totalw += w->second;
    }

    LOG_DEBUG(<< "total minimum probability with periodicity    = " << totalw);
    LOG_DEBUG(<< "total minimum probability without periodicity = " << totalwo);
    BOOST_TEST_REQUIRE(totalw > 3.0 * totalwo);
}

BOOST_FIXTURE_TEST_CASE(testPersistence, CTestFixture) {
    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(1, startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr origModel(factory.makeModel(modelInitData));

    CMetricPopulationModel* populationModel =
        dynamic_cast<CMetricPopulationModel*>(origModel.get());
    BOOST_TEST_REQUIRE(populationModel);

    for (std::size_t i = 0u; i < messages.size(); ++i) {
        if (messages[i].s_Time >= startTime + bucketLength) {
            origModel->sample(startTime, startTime + bucketLength, m_ResourceMonitor);
            startTime += bucketLength;
        }
        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origModel->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "origXml = " << origXml);

    // Restore the XML into a new data gatherer
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CAnomalyDetectorModel::TModelPtr restoredModel(factory.makeModel(modelInitData, traverser));

    populationModel = dynamic_cast<CMetricPopulationModel*>(restoredModel.get());
    BOOST_TEST_REQUIRE(populationModel);

    // The XML representation of the new data gatherer should be the same as the
    // original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModel->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    LOG_DEBUG(<< "original checksum = " << origModel->checksum(false));
    LOG_DEBUG(<< "restored checksum = " << restoredModel->checksum(false));
    BOOST_REQUIRE_EQUAL(origModel->checksum(false), restoredModel->checksum(false));
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testIgnoreSamplingGivenDetectionRules, CTestFixture) {
    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.

    core_t::TTime startTime(100);
    std::size_t bucketLength(100);
    core_t::TTime endTime = startTime + bucketLength;

    // Create a categorical rule to filter out attribute a3
    std::string filterJson("[\"c3\"]");
    core::CPatternSet valueFilter;
    valueFilter.initFromJson(filterJson);

    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipModelUpdate);
    rule.includeScope("", valueFilter);

    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute};

    SModelParams paramsNoRules(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factoryNoSkip(paramsNoRules, interimBucketCorrector);
    factoryNoSkip.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoSkip(
        factoryNoSkip.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelNoSkipInitData(gathererNoSkip);
    CAnomalyDetectorModel::TModelPtr modelNoSkip(factoryNoSkip.makeModel(modelNoSkipInitData));

    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);
    auto interimBucketCorrectorWithRules =
        std::make_shared<CInterimBucketCorrector>(bucketLength);

    CMetricPopulationModelFactory factoryWithSkip(paramsWithRules, interimBucketCorrectorWithRules);
    factoryWithSkip.features(features);
    CModelFactory::TDataGathererPtr gathererWithSkip(
        factoryWithSkip.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelWithSkipInitData(gathererWithSkip);
    CAnomalyDetectorModel::TModelPtr modelWithSkip(
        factoryWithSkip.makeModel(modelWithSkipInitData));

    std::vector<SMessage> messages;
    messages.push_back(SMessage(startTime + 10, "p1", "c1", TDouble1Vec(1, 20.0)));
    messages.push_back(SMessage(startTime + 10, "p1", "c2", TDouble1Vec(1, 22.0)));
    messages.push_back(SMessage(startTime + 10, "p2", "c1", TDouble1Vec(1, 20.0)));
    messages.push_back(SMessage(startTime + 10, "p2", "c2", TDouble1Vec(1, 22.0)));

    std::vector<CModelFactory::TDataGathererPtr> gatherers{gathererNoSkip, gathererWithSkip};
    for (auto& gatherer : gatherers) {
        for (auto& message : messages) {
            addArrival(message, gatherer, m_ResourceMonitor);
        }
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    messages.clear();
    messages.push_back(SMessage(startTime + 10, "p1", "c1", TDouble1Vec(1, 21.0)));
    messages.push_back(SMessage(startTime + 10, "p1", "c2", TDouble1Vec(1, 21.0)));
    messages.push_back(SMessage(startTime + 10, "p2", "c1", TDouble1Vec(1, 21.0)));
    messages.push_back(SMessage(startTime + 10, "p2", "c2", TDouble1Vec(1, 21.0)));
    for (auto& gatherer : gatherers) {
        for (auto& message : messages) {
            addArrival(message, gatherer, m_ResourceMonitor);
        }
    }

    // This should be filtered out
    addArrival(SMessage(startTime + 10, "p1", "c3", TDouble1Vec(1, 21.0)),
               gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(startTime + 10, "p2", "c3", TDouble1Vec(1, 21.0)),
               gathererWithSkip, m_ResourceMonitor);

    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);

    // Checksums will be different because a 3rd model is created for attribute c3
    BOOST_TEST_REQUIRE(modelWithSkip->checksum() != modelNoSkip->checksum());

    CAnomalyDetectorModel::TModelDetailsViewUPtr modelWithSkipView =
        modelWithSkip->details();
    CAnomalyDetectorModel::TModelDetailsViewUPtr modelNoSkipView = modelNoSkip->details();

    // but the underlying models for people p1 and p2 are the same
    uint64_t withSkipChecksum = modelWithSkipView
                                    ->model(model_t::E_PopulationMeanByPersonAndAttribute, 0)
                                    ->checksum();
    uint64_t noSkipChecksum =
        modelNoSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 0)->checksum();
    BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    withSkipChecksum = modelWithSkipView
                           ->model(model_t::E_PopulationMeanByPersonAndAttribute, 1)
                           ->checksum();
    noSkipChecksum =
        modelNoSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 1)->checksum();
    BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    // TODO These checks fail see elastic/machine-learning-cpp/issues/485
    // Check the last value times of all the underlying models are the same
    //     const maths::CUnivariateTimeSeriesModel *timeSeriesModel =
    //         dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 1));
    //     BOOST_TEST_REQUIRE(timeSeriesModel != 0);

    //     core_t::TTime time = timeSeriesModel->trend().lastValueTime();
    //     BOOST_REQUIRE_EQUAL(model_t::sampleTime(model_t::E_PopulationMeanByPersonAndAttribute, startTime, bucketLength), time);

    //     // The last times of the underlying time series models should all be the same
    //     timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelNoSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 1));
    //     BOOST_REQUIRE_EQUAL(time, timeSeriesModel->trend().lastValueTime());

    //     timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 0));
    //     BOOST_REQUIRE_EQUAL(time, timeSeriesModel->trend().lastValueTime());
    //     timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 1));
    //     BOOST_REQUIRE_EQUAL(time, timeSeriesModel->trend().lastValueTime());
    //     timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelWithSkipView->model(model_t::E_PopulationMeanByPersonAndAttribute, 2));
    //     BOOST_REQUIRE_EQUAL(time, timeSeriesModel->trend().lastValueTime());
}

BOOST_AUTO_TEST_SUITE_END()
