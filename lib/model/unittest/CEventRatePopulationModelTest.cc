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

#include "CEventRatePopulationModelTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPatternSet.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>

#include <maths/CModelWeight.h>
#include <maths/COrderings.h>
#include <maths/CTimeSeriesDecompositionInterface.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CEventRatePopulationModel.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CModelDetailsView.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>
#include <model/CRuleCondition.h>

#include <test/CRandomNumbers.h>

#include <boost/lexical_cast.hpp>
#include <boost/range.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace ml;
using namespace model;

namespace {

using TOptionalDouble = boost::optional<double>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TStrVec = std::vector<std::string>;
using TUIntVec = std::vector<unsigned int>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TDoubleStrPr = std::pair<double, std::string>;
using TDoubleStrPrVec = std::vector<TDoubleStrPr>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;

const std::string EMPTY_STRING;

struct SMessage {
    SMessage(core_t::TTime time, const std::string& person, const std::string& attribute)
        : s_Time(time), s_Person(person), s_Attribute(attribute) {}

    bool operator<(const SMessage& other) const {
        return maths::COrderings::lexicographical_compare(s_Time, s_Person, s_Attribute, other.s_Time, other.s_Person, other.s_Attribute);
    }

    core_t::TTime s_Time;
    std::string s_Person;
    std::string s_Attribute;
};

struct SAnomaly {
    SAnomaly() : s_Bucket(0u), s_Person(), s_Attributes() {}
    SAnomaly(std::size_t bucket, const std::string& person, const TDoubleStrPrVec& attributes)
        : s_Bucket(bucket), s_Person(person), s_Attributes(attributes) {}

    std::size_t s_Bucket;
    std::string s_Person;
    TDoubleStrPrVec s_Attributes;

    bool operator<(const SAnomaly& other) const { return s_Bucket < other.s_Bucket; }

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

using TMessageVec = std::vector<SMessage>;

void generateTestMessages(core_t::TTime startTime, core_t::TTime bucketLength, TMessageVec& messages) {
    // The test case is as follows:
    //
    //   attribute   |    0    |    1    |    2    |    3    |   4
    // --------------+---------+---------+---------+---------+--------
    //    people     |  [0-19] |  [0-2], |  [0,2], |   3,4   |   3
    //               |         |  [4-19] |  [5,19] |         |
    // --------------+---------+---------+---------+---------+--------
    //    rate       |   10    |   0.02  |   15    |    2    |   1
    // --------------+---------+---------+---------+---------+--------
    //  rate anomaly |   1,11  |    -    |   4,5   |   n/a   |  n/a
    //     people    |         |         |         |         |
    //
    // There are 100 buckets.

    using TSizeSizeSizeTr = boost::tuple<std::size_t, std::size_t, size_t>;

    const std::size_t numberBuckets = 100u;
    const std::size_t numberAttributes = 5u;
    const std::size_t numberPeople = 20u;

    TStrVec attributes;
    for (std::size_t i = 0u; i < numberAttributes; ++i) {
        attributes.push_back("c" + boost::lexical_cast<std::string>(i));
    }

    TStrVec people;
    for (std::size_t i = 0u; i < numberPeople; ++i) {
        people.push_back("p" + boost::lexical_cast<std::string>(i));
    }

    std::size_t c0People[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::size_t c1People[] = {0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::size_t c2People[] = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::size_t c3People[] = {3, 4};
    std::size_t c4People[] = {3};

    TSizeVecVec attributePeople;
    attributePeople.push_back(TSizeVec(boost::begin(c0People), boost::end(c0People)));
    attributePeople.push_back(TSizeVec(boost::begin(c1People), boost::end(c1People)));
    attributePeople.push_back(TSizeVec(boost::begin(c2People), boost::end(c2People)));
    attributePeople.push_back(TSizeVec(boost::begin(c3People), boost::end(c3People)));
    attributePeople.push_back(TSizeVec(boost::begin(c4People), boost::end(c4People)));

    double attributeRates[] = {10.0, 0.02, 15.0, 2.0, 1.0};

    TSizeSizeSizeTr anomaliesAttributePerson[] = {TSizeSizeSizeTr(10u, 0u, 1u),
                                                  TSizeSizeSizeTr(15u, 0u, 11u),
                                                  TSizeSizeSizeTr(30u, 2u, 4u),
                                                  TSizeSizeSizeTr(35u, 2u, 5u),
                                                  TSizeSizeSizeTr(50u, 0u, 11u),
                                                  TSizeSizeSizeTr(75u, 2u, 5u)};

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberBuckets; ++i, startTime += bucketLength) {
        for (std::size_t j = 0u; j < numberAttributes; ++j) {
            TUIntVec samples;
            rng.generatePoissonSamples(attributeRates[j], attributePeople[j].size(), samples);

            for (std::size_t k = 0u; k < samples.size(); ++k) {
                unsigned int n = samples[k];
                if (std::binary_search(boost::begin(anomaliesAttributePerson),
                                       boost::end(anomaliesAttributePerson),
                                       TSizeSizeSizeTr(i, j, attributePeople[j][k]))) {
                    n += static_cast<unsigned int>(2.5 * attributeRates[j]);
                    LOG_DEBUG(i << " " << attributes[j] << " generating anomaly " << n);
                }

                TDoubleVec times;
                rng.generateUniformSamples(0.0, static_cast<double>(bucketLength - 1), n, times);

                for (std::size_t l = 0u; l < times.size(); ++l) {
                    core_t::TTime time = startTime + static_cast<core_t::TTime>(times[l]);
                    messages.push_back(SMessage(time, people[attributePeople[j][k]], attributes[j]));
                }
            }
        }
    }

    std::sort(messages.begin(), messages.end());
}

void addArrival(const SMessage& message, const CModelFactory::TDataGathererPtr& gatherer, CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&message.s_Person);
    fields.push_back(&message.s_Attribute);
    CEventData result;
    result.time(message.s_Time);
    gatherer->addArrival(fields, result, resourceMonitor);
}

const TSizeDoublePr1Vec NO_CORRELATES;
}

void CEventRatePopulationModelTest::testBasicAccessors() {
    LOG_DEBUG("*** testBasicAccessors ***");

    // Check that the correct data is read retrieved by the
    // basic model accessors.

    using TOptionalUInt64 = boost::optional<uint64_t>;
    using TStrUInt64Map = std::map<std::string, uint64_t>;
    using TStrUInt64MapCItr = TStrUInt64Map::const_iterator;
    using TStrStrPr = std::pair<std::string, std::string>;
    using TStrStrPrDoubleMap = std::map<TStrStrPr, double>;
    using TStrStrPrDoubleMapCItr = TStrStrPrDoubleMap::const_iterator;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);
    LOG_DEBUG("# messages = " << messages.size());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRatePopulationModelFactory factory(params);
    CModelFactory::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(dynamic_cast<CDataGatherer*>(factory.makeDataGatherer(gathererInitData)));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));

    CPPUNIT_ASSERT_EQUAL(model_t::E_EventRateOnline, model->category());

    TStrUInt64Map expectedBucketPersonCounts;
    TStrStrPrDoubleMap expectedBucketPersonAttributeCounts;

    for (std::size_t i = 0u; i < messages.size(); ++i) {
        if (messages[i].s_Time >= startTime + bucketLength) {
            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            LOG_DEBUG("Testing bucket = [" << startTime << "," << startTime + bucketLength << ")");

            // Test the person and attribute invariants.
            for (std::size_t j = 0u; j < gatherer->numberActivePeople(); ++j) {
                const std::string& name = model->personName(j);
                std::size_t pid;
                CPPUNIT_ASSERT(gatherer->personId(name, pid));
                CPPUNIT_ASSERT_EQUAL(j, pid);
            }
            for (std::size_t j = 0u; j < gatherer->numberActiveAttributes(); ++j) {
                const std::string& name = model->attributeName(j);
                std::size_t cid;
                CPPUNIT_ASSERT(gatherer->attributeId(name, cid));
                CPPUNIT_ASSERT_EQUAL(j, cid);
            }

            TSizeVec expectedCurrentBucketPersonIds;

            // Test the person counts.
            for (TStrUInt64MapCItr j = expectedBucketPersonCounts.begin(); j != expectedBucketPersonCounts.end(); ++j) {
                std::size_t pid;
                CPPUNIT_ASSERT(gatherer->personId(j->first, pid));

                expectedCurrentBucketPersonIds.push_back(pid);

                TOptionalUInt64 count = model->currentBucketCount(pid, startTime);
                CPPUNIT_ASSERT(count);
                CPPUNIT_ASSERT_EQUAL(j->second, *count);
            }

            // Test the person attribute counts.
            for (TStrStrPrDoubleMapCItr j = expectedBucketPersonAttributeCounts.begin(); j != expectedBucketPersonAttributeCounts.end();
                 ++j) {
                std::size_t pid;
                CPPUNIT_ASSERT(gatherer->personId(j->first.first, pid));
                std::size_t cid;
                CPPUNIT_ASSERT(gatherer->attributeId(j->first.second, cid));

                TDouble1Vec count = model->currentBucketValue(model_t::E_PopulationCountByBucketPersonAndAttribute, pid, cid, startTime);
                CPPUNIT_ASSERT(!count.empty());
                CPPUNIT_ASSERT_EQUAL(j->second, count[0]);
            }

            // Test the current bucket people.
            std::sort(expectedCurrentBucketPersonIds.begin(), expectedCurrentBucketPersonIds.end());
            TSizeVec bucketPersonIds;
            model->currentBucketPersonIds(startTime, bucketPersonIds);
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedCurrentBucketPersonIds),
                                 core::CContainerPrinter::print(bucketPersonIds));

            expectedBucketPersonCounts.clear();
            expectedBucketPersonAttributeCounts.clear();
            startTime += bucketLength;
        }

        const SMessage& m = messages[i];
        addArrival(m, gatherer, m_ResourceMonitor);
        ++expectedBucketPersonCounts[m.s_Person];
        expectedBucketPersonAttributeCounts[TStrStrPr(m.s_Person, m.s_Attribute)] += 1.0;
    }
}

void CEventRatePopulationModelTest::testFeatures() {
    LOG_DEBUG("*** testFeatures ***");

    // We check that the correct data is read from the gatherer
    // into the model on sample.

    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2VecVec = std::vector<TDouble2Vec>;
    using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
    using TDouble2Vec4VecVec = std::vector<TDouble2Vec4Vec>;
    using TSizeSet = std::set<std::size_t>;
    using TSizeSizeSetMap = std::map<std::size_t, TSizeSet>;
    using TStrStrPr = std::pair<std::string, std::string>;
    using TStrStrPrUint64Map = std::map<TStrStrPr, uint64_t>;
    using TFeatureData = SEventRateFeatureData;
    using TSizeSizePrFeatureDataPr = CEventRatePopulationModel::TSizeSizePrFeatureDataPr;
    using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
    using TSizeSizePrUInt64Map = std::map<TSizeSizePr, uint64_t>;
    using TMathsModelPtr = boost::shared_ptr<maths::CModel>;
    using TSizeMathsModelPtrMap = std::map<std::size_t, TMathsModelPtr>;
    using TDouble2VecVecDouble2Vec4VecVecPr = std::pair<TDouble2VecVec, TDouble2Vec4VecVec>;
    using TSizeDouble2VecVecDouble2Vec4VecVecPrMap = std::map<std::size_t, TDouble2VecVecDouble2Vec4VecVecPr>;

    static const maths_t::TWeightStyleVec WEIGHT_STYLES{maths_t::E_SampleCountWeight, maths_t::E_SampleWinsorisationWeight};

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);
    LOG_DEBUG("# messages = " << messages.size());

    // Bucket non-zero count unique person count.
    {
        SModelParams params(bucketLength);
        params.s_InitialDecayRateMultiplier = 1.0;
        CEventRatePopulationModelFactory factory(params);
        CModelFactory::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                            model_t::E_PopulationUniquePersonCountByAttribute};
        factory.features(features);
        CModelFactory::SGathererInitializationData gathererInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(dynamic_cast<CDataGatherer*>(factory.makeDataGatherer(gathererInitData)));
        CModelFactory::SModelInitializationData modelInitData(gatherer);
        CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
        CEventRatePopulationModel* model = dynamic_cast<CEventRatePopulationModel*>(modelHolder.get());

        model::CModelFactory::TFeatureMathsModelPtrPrVec models{factory.defaultFeatureModels(features, bucketLength, 1.0, false)};
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), models.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute, models[0].first);

        std::size_t numberAttributes = 0u;
        std::size_t numberPeople = 0u;
        TSizeSizeSetMap attributePeople;
        TStrStrPrUint64Map expectedCounts;

        TSizeMathsModelPtrMap expectedPopulationModels;

        for (const auto& message : messages) {
            if (message.s_Time >= startTime + bucketLength) {
                model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);
                TSizeSizePrUInt64Map expectedNonZeroCounts;

                for (const auto& count : expectedCounts) {
                    std::size_t pid, cid;
                    CPPUNIT_ASSERT(gatherer->personId(count.first.first, pid));
                    CPPUNIT_ASSERT(gatherer->attributeId(count.first.second, cid));

                    numberAttributes = std::max(numberAttributes, cid + 1);
                    numberPeople = std::max(numberPeople, pid + 1);
                    attributePeople[cid].insert(pid);
                    expectedNonZeroCounts[{pid, cid}] = count.second;
                }

                TSizeDouble2VecVecDouble2Vec4VecVecPrMap populationSamples;
                for (const auto& count_ : expectedNonZeroCounts) {
                    std::size_t pid = count_.first.first;
                    std::size_t cid = count_.first.second;
                    core_t::TTime time = startTime + bucketLength / 2;
                    double count = model_t::offsetCountToZero(model_t::E_PopulationCountByBucketPersonAndAttribute,
                                                              static_cast<double>(count_.second));
                    TMathsModelPtr& model_ = expectedPopulationModels[cid];
                    if (model_ == nullptr) {
                        model_.reset(models[0].second->clone(cid));
                    }

                    TDoubleVec sample(1, count);
                    TDouble2Vec4Vec weight{{model->sampleRateWeight(pid, cid)}, model_->winsorisationWeight(1.0, time, sample)};
                    populationSamples[cid].first.push_back({sample[0]});
                    populationSamples[cid].second.push_back(weight);
                }
                for (auto& samples_ : populationSamples) {
                    std::size_t cid = samples_.first;
                    TDouble2Vec4VecVec& weights = samples_.second.second;
                    maths::COrderings::simultaneousSort(samples_.second.first, weights);
                    maths::CModel::TTimeDouble2VecSizeTrVec samples;
                    for (const auto& sample : samples_.second.first) {
                        samples.emplace_back(startTime + bucketLength / 2, sample, 0);
                    }
                    maths::CModelAddSamplesParams params_;
                    params_.integer(true)
                        .nonNegative(true)
                        .propagationInterval(1.0)
                        .weightStyles(WEIGHT_STYLES)
                        .trendWeights(weights)
                        .priorWeights(weights);
                    expectedPopulationModels[cid]->addSamples(params_, samples);
                }

                TSizeSizePrFeatureDataPrVec expectedPeoplePerAttribute;
                expectedPeoplePerAttribute.reserve(numberAttributes);
                for (std::size_t j = 0u; j < numberAttributes; ++j) {
                    expectedPeoplePerAttribute.emplace_back(std::make_pair(size_t(0), j), TFeatureData(j));
                }
                for (const auto& attribute : attributePeople) {
                    expectedPeoplePerAttribute[attribute.first].second = attribute.second.size();
                }

                // Check the number of people per attribute.
                const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
                    model->featureData(model_t::E_PopulationUniquePersonCountByAttribute, startTime);
                CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedPeoplePerAttribute),
                                     core::CContainerPrinter::print(peoplePerAttribute));

                // Check the non-zero (person, attribute) counts.
                const TSizeSizePrFeatureDataPrVec& nonZeroCounts =
                    model->featureData(model_t::E_PopulationCountByBucketPersonAndAttribute, startTime);
                CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedNonZeroCounts), core::CContainerPrinter::print(nonZeroCounts));

                for (std::size_t cid = 0u; cid < numberAttributes; ++cid) {
                    const maths::CModel* populationModel =
                        model->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, cid);
                    CPPUNIT_ASSERT(populationModel);
                    CPPUNIT_ASSERT_EQUAL(expectedPopulationModels[cid]->checksum(), populationModel->checksum());
                }

                startTime += bucketLength;
                expectedCounts = TStrStrPrUint64Map();
            }

            addArrival(message, gatherer, m_ResourceMonitor);
            ++expectedCounts[{message.s_Person, message.s_Attribute}];
        }
    }
}

void CEventRatePopulationModelTest::testComputeProbability() {
    LOG_DEBUG("*** testComputeProbability ***");

    // Check that we get the probabilities we expect.

    using TAnomalyVec = std::vector<SAnomaly>;
    using TDoubleAnomalyPr = std::pair<double, SAnomaly>;
    using TAnomalyAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<TDoubleAnomalyPr, maths::COrderings::SFirstLess>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRatePopulationModelFactory factory(params);
    CModelFactory::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
    CEventRatePopulationModel* model = dynamic_cast<CEventRatePopulationModel*>(modelHolder.get());

    TAnomalyAccumulator anomalies(6u);

    for (std::size_t i = 0u, bucket = 0u; i < messages.size(); ++i) {
        if (messages[i].s_Time >= startTime + bucketLength) {
            LOG_DEBUG("Updating and testing bucket = [" << startTime << "," << startTime + bucketLength << ")");

            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            SAnnotatedProbability annotatedProbability;
            for (std::size_t pid = 0u; pid < gatherer->numberActivePeople(); ++pid) {
                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                model->computeProbability(pid, startTime, startTime + bucketLength, partitioningFields, 2, annotatedProbability);

                std::string person = model->personName(pid);
                TDoubleStrPrVec attributes;
                for (std::size_t j = 0u; j < annotatedProbability.s_AttributeProbabilities.size(); ++j) {
                    attributes.emplace_back(annotatedProbability.s_AttributeProbabilities[j].s_Probability,
                                            *annotatedProbability.s_AttributeProbabilities[j].s_Attribute);
                }
                anomalies.add({annotatedProbability.s_Probability, SAnomaly(bucket, person, attributes)});
            }

            startTime += bucketLength;
            ++bucket;
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    anomalies.sort();
    LOG_DEBUG("Anomalies = " << anomalies.print());

    TAnomalyVec orderedAnomalies;
    for (std::size_t i = 0u; i < anomalies.count(); ++i) {
        orderedAnomalies.push_back(anomalies[i].second);
    }

    std::sort(orderedAnomalies.begin(), orderedAnomalies.end());

    LOG_DEBUG("orderedAnomalies = " << core::CContainerPrinter::print(orderedAnomalies));

    std::string expectedAnomalies[] = {std::string("[10, p1, c0]"),
                                       std::string("[15, p11, c0]"),
                                       std::string("[30, p4, c2]"),
                                       std::string("[35, p5, c2]"),
                                       std::string("[50, p11, c0]"),
                                       std::string("[75, p5, c2]")};

    CPPUNIT_ASSERT_EQUAL(boost::size(expectedAnomalies), orderedAnomalies.size());
    for (std::size_t i = 0u; i < orderedAnomalies.size(); ++i) {
        CPPUNIT_ASSERT_EQUAL(expectedAnomalies[i], orderedAnomalies[i].print());
    }
}

void CEventRatePopulationModelTest::testPrune() {
    LOG_DEBUG("*** testPrune ***");

    // This test has four people and five attributes. We expect
    // person 2 and attributes 1, 2 and 5 to be deleted.

    using TStrSizePr = std::pair<std::string, std::size_t>;
    using TStrSizePrVec = std::vector<TStrSizePr>;
    using TStrSizePrVecVec = std::vector<TStrSizePrVec>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 1000u;

    std::string people[] = {std::string("p1"), std::string("p2"), std::string("p3"), std::string("p4")};
    std::string attributes[] = {std::string("c1"), std::string("c2"), std::string("c3"), std::string("c4"), std::string("c5")};

    TStrSizePrVecVec eventCounts[] = {TStrSizePrVecVec(), TStrSizePrVecVec(), TStrSizePrVecVec(), TStrSizePrVecVec()};
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.push_back(TStrSizePr(attributes[0], 0));
        attributeCounts.push_back(TStrSizePr(attributes[1], 0));
        attributeCounts.push_back(TStrSizePr(attributes[2], 1));
        attributeCounts.push_back(TStrSizePr(attributes[4], 0));
        eventCounts[0].resize(numberBuckets, attributeCounts);
        eventCounts[0][1][0].second = 2; // p1, bucket 2, c1
        eventCounts[0][3][0].second = 4; // p1, bucket 3, c1
        eventCounts[0][5][1].second = 4; // p1, bucket 5, c2
        eventCounts[0][5][3].second = 3; // p1, bucket 5, c5
    }
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.push_back(TStrSizePr(attributes[0], 0));
        attributeCounts.push_back(TStrSizePr(attributes[4], 0));
        eventCounts[1].resize(numberBuckets, attributeCounts);
        eventCounts[1][0][0].second = 2; // p2, bucket 1, c1
        eventCounts[1][2][0].second = 3; // p2, bucket 3, c1
        eventCounts[1][0][1].second = 4; // p2, bucket 1, c5
        eventCounts[1][3][1].second = 1; // p2, bucket 4, c5
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
        eventCounts[3][15][0].second = 3; // p4, bucket 16, c2
        eventCounts[3][26][0].second = 1; // p4, bucket 27, c2
        eventCounts[3][70][0].second = 4; // p4, bucket 70, c2
    }

    const std::string expectedPeople[] = {people[0], people[2], people[3]};
    const std::string expectedAttributes[] = {attributes[2], attributes[3]};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.01;
    CEventRatePopulationModelFactory factory(params);
    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));
    CPPUNIT_ASSERT(model);
    CModelFactory::TDataGathererPtr expectedGatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData expectedModelInitData(expectedGatherer);
    CAnomalyDetectorModel::TModelPtr expectedModel(factory.makeModel(expectedModelInitData));
    CPPUNIT_ASSERT(expectedModel);

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
                core_t::TTime time = bucketStart;
                core_t::TTime dt = bucketLength / static_cast<core_t::TTime>(n);
                for (std::size_t l = 0u; l < n; ++l, time += dt) {
                    messages.push_back(SMessage(time, people[i], attributeEventCounts[k].first));
                }
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    TMessageVec expectedMessages;
    expectedMessages.reserve(messages.size());
    for (std::size_t i = 0u; i < messages.size(); ++i) {
        if (std::binary_search(boost::begin(expectedPeople), boost::end(expectedPeople), messages[i].s_Person) &&
            std::binary_search(boost::begin(expectedAttributes), boost::end(expectedAttributes), messages[i].s_Attribute)) {
            expectedMessages.push_back(messages[i]);
        }
    }

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune();
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    CPPUNIT_ASSERT_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = startTime;
    for (std::size_t i = 0u; i < expectedMessages.size(); ++i) {
        if (expectedMessages[i].s_Time >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        addArrival(expectedMessages[i], expectedGatherer, m_ResourceMonitor);
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG("checksum          = " << model->checksum());
    LOG_DEBUG("expected checksum = " << expectedModel->checksum());
    CPPUNIT_ASSERT_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person and attribute slots.

    bucketStart = gatherer->currentBucketStartTime() + bucketLength;

    SMessage newMessages[] = {
        SMessage(bucketStart + 10, "p1", "c2"), SMessage(bucketStart + 200, "p5", "c6"), SMessage(bucketStart + 2100, "p5", "c6")};

    for (std::size_t i = 0u; i < boost::size(newMessages); ++i) {
        addArrival(newMessages[i], gatherer, m_ResourceMonitor);
        addArrival(newMessages[i], expectedGatherer, m_ResourceMonitor);
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

void CEventRatePopulationModelTest::testKey() {
    LOG_DEBUG("*** testKey ***");

    function_t::EFunction countFunctions[] = {function_t::E_PopulationCount,
                                              function_t::E_PopulationDistinctCount,
                                              function_t::E_PopulationRare,
                                              function_t::E_PopulationRareCount,
                                              function_t::E_PopulationFreqRare,
                                              function_t::E_PopulationFreqRareCount,
                                              function_t::E_PopulationLowCounts,
                                              function_t::E_PopulationHighCounts};
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
                        CSearchKey key(
                            ++identifier, countFunctions[i], useNull[j], model_t::E_XF_None, "", byField[k], "over", partitionField[l]);

                        CAnomalyDetectorModelConfig::TModelFactoryCPtr factory = config.factory(key);

                        LOG_DEBUG("expected key = " << key);
                        LOG_DEBUG("actual key   = " << factory->searchKey());
                        CPPUNIT_ASSERT(key == factory->searchKey());
                    }
                }
            }
        }
    }
}

void CEventRatePopulationModelTest::testFrequency() {
    LOG_DEBUG("*** CEventRatePopulationModelTest::testFrequency ***");

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    // Test we correctly compute frequencies for people and attributes.

    const core_t::TTime bucketLength = 600;
    const std::string attributes[] = {"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"};
    const std::string people[] = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"};
    std::size_t period[] = {1u, 1u, 10u, 3u, 4u, 5u, 2u, 1u, 3u, 7u};

    core_t::TTime startTime = 0;

    TMessageVec messages;
    std::size_t bucket = 0u;
    for (core_t::TTime bucketStart = startTime; bucketStart < 100 * bucketLength; bucketStart += bucketLength, ++bucket) {
        for (std::size_t i = 0u; i < boost::size(people); ++i) {
            if (bucket % period[i] == 0) {
                for (std::size_t j = 0u; j < i + 1; ++j) {
                    messages.push_back(SMessage(bucketStart + bucketLength / 2, people[i], attributes[j]));
                }
            }
        }
    }

    std::sort(messages.begin(), messages.end());
    LOG_DEBUG("# messages = " << messages.size());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRatePopulationModelFactory factory(params);
    CModelFactory::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    const model::CDataGatherer& populationGatherer(dynamic_cast<const model::CDataGatherer&>(*gatherer));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));

    CEventRatePopulationModel* populationModel = dynamic_cast<CEventRatePopulationModel*>(model.get());
    CPPUNIT_ASSERT(populationModel);

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
            LOG_DEBUG("*** person = " << people[i] << " ***");
            std::size_t pid;
            CPPUNIT_ASSERT(gatherer->personId(people[i], pid));
            LOG_DEBUG("frequency = " << populationModel->personFrequency(pid));
            LOG_DEBUG("expected frequency = " << 1.0 / static_cast<double>(period[i]));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                1.0 / static_cast<double>(period[i]), populationModel->personFrequency(pid), 0.1 / static_cast<double>(period[i]));
            meanError.add(std::fabs(populationModel->personFrequency(pid) - 1.0 / static_cast<double>(period[i])));
        }
        LOG_DEBUG("error = " << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.002);
    }
    {
        for (std::size_t i = 0u; i < boost::size(attributes); ++i) {
            LOG_DEBUG("*** attribute = " << attributes[i] << " ***");
            std::size_t cid;
            CPPUNIT_ASSERT(populationGatherer.attributeId(attributes[i], cid));
            LOG_DEBUG("frequency = " << populationModel->attributeFrequency(cid));
            LOG_DEBUG("expected frequency = " << (10.0 - static_cast<double>(i)) / 10.0);
            CPPUNIT_ASSERT_EQUAL((10.0 - static_cast<double>(i)) / 10.0, populationModel->attributeFrequency(cid));
        }
    }
}

void CEventRatePopulationModelTest::testSampleRateWeight() {
    LOG_DEBUG("*** CEventRatePopulationModelTest::testSampleRateWeight ***");

    // Test that we correctly compensate for heavy hitters.

    // There are 10 attributes.
    //
    // People p1 and p5 generate messages for every attribute every bucket.
    // The remaining 18 people only generate one message per bucket, i.e.
    // one message per attribute per 10 buckets.

    const core_t::TTime bucketLength = 600;
    const std::string attributes[] = {"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"};
    const std::string people[] = {"p1",  "p2",  "p3",  "p4",  "p5",  "p6",  "p7",  "p8",  "p9",  "p10",
                                  "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"};
    std::size_t heavyHitters[] = {0u, 4u};
    std::size_t normal[] = {1u, 2u, 3u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u};

    std::size_t messagesPerBucket = boost::size(heavyHitters) * boost::size(attributes) + boost::size(normal);

    test::CRandomNumbers rng;

    core_t::TTime startTime = 0;

    TMessageVec messages;
    for (core_t::TTime bucketStart = startTime; bucketStart < 100 * bucketLength; bucketStart += bucketLength) {
        TSizeVec times;
        rng.generateUniformSamples(
            static_cast<std::size_t>(bucketStart), static_cast<std::size_t>(bucketStart + bucketLength), messagesPerBucket, times);

        std::size_t m = 0u;
        for (std::size_t i = 0u; i < boost::size(attributes); ++i) {
            for (std::size_t j = 0u; j < boost::size(heavyHitters); ++j) {
                messages.push_back(SMessage(static_cast<core_t::TTime>(times[m++]), people[heavyHitters[j]], attributes[i]));
            }
        }

        TSizeVec attributeIndexes;
        rng.generateUniformSamples(0, boost::size(attributes), boost::size(normal), attributeIndexes);
        for (std::size_t i = 0u; i < boost::size(normal); ++i) {
            messages.push_back(SMessage(static_cast<core_t::TTime>(times[m++]), people[normal[i]], attributes[attributeIndexes[i]]));
        }
    }

    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRatePopulationModelFactory factory(params);
    CModelFactory::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));
    CEventRatePopulationModel* populationModel = dynamic_cast<CEventRatePopulationModel*>(model.get());
    CPPUNIT_ASSERT(populationModel);

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

    double expectedRateWeight = (static_cast<double>(boost::size(normal)) / static_cast<double>(boost::size(attributes)) +
                                 static_cast<double>(boost::size(heavyHitters))) /
                                static_cast<double>(boost::size(people));
    LOG_DEBUG("expectedRateWeight = " << expectedRateWeight);

    for (std::size_t i = 0u; i < boost::size(heavyHitters); ++i) {
        LOG_DEBUG("*** person = " << people[heavyHitters[i]] << " ***");
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer->personId(people[heavyHitters[i]], pid));
        for (std::size_t cid = 0u; cid < boost::size(attributes); ++cid) {
            double sampleRateWeight = populationModel->sampleRateWeight(pid, cid);
            LOG_DEBUG("attribute = " << populationModel->attributeName(cid) << ", sampleRateWeight = " << sampleRateWeight);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedRateWeight, sampleRateWeight, 0.15 * expectedRateWeight);
        }
    }

    for (std::size_t i = 0u; i < boost::size(normal); ++i) {
        LOG_DEBUG("*** person = " << people[normal[i]] << " ***");
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer->personId(people[normal[i]], pid));
        for (std::size_t cid = 0u; cid < boost::size(attributes); ++cid) {
            double sampleRateWeight = populationModel->sampleRateWeight(pid, cid);
            LOG_DEBUG("attribute = " << populationModel->attributeName(cid) << ", sampleRateWeight = " << sampleRateWeight);
            CPPUNIT_ASSERT_EQUAL(1.0, sampleRateWeight);
        }
    }
}

void CEventRatePopulationModelTest::testPeriodicity() {
    LOG_DEBUG("*** testPeriodicity ***");

    // Create a daily periodic population and check that the
    // periodicity is learned and compensated (approximately).

    using TStrDoubleMap = std::map<std::string, double>;
    using TStrDoubleMapCItr = TStrDoubleMap::const_iterator;

    static const core_t::TTime HOUR = 3600;
    static const core_t::TTime DAY = 86400;

    const core_t::TTime bucketLength = 3600;
    double rate[] = {1, 1, 2, 2, 3, 5, 6, 6, 20, 21, 4, 3, 4, 4, 8, 25, 7, 6, 5, 1, 1, 4, 1, 1};
    const std::string attributes[] = {"a1", "a2"};
    double scales[] = {1.0, 1.5};
    const std::string people[] = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"};

    test::CRandomNumbers rng;

    core_t::TTime startTime = 0;
    core_t::TTime endTime = 604800;

    TMessageVec messages;
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        for (std::size_t i = 0u; i < boost::size(attributes); ++i) {
            TUIntVec rates;
            rng.generatePoissonSamples(scales[i] * rate[(time % DAY) / HOUR], boost::size(people), rates);

            for (std::size_t j = 0u; j < rates.size(); ++j) {
                for (unsigned int t = 0; t < rates[j]; ++t) {
                    messages.push_back(SMessage(time + (t * bucketLength) / (rates[j] + 1), people[j], attributes[i]));
                }
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MinimumModeCount = 24.0;
    CEventRatePopulationModelFactory factory(params);
    CModelFactory::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr model(factory.makeModel(modelInitData));
    CEventRatePopulationModel* populationModel = dynamic_cast<CEventRatePopulationModel*>(model.get());
    CPPUNIT_ASSERT(populationModel);

    TStrDoubleMap personProbabilitiesWithoutPeriodicity;
    TStrDoubleMap personProbabilitiesWithPeriodicity;

    core_t::TTime time = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            populationModel->sample(time, time + bucketLength, m_ResourceMonitor);

            for (std::size_t j = 0u; j < boost::size(people); ++j) {
                std::size_t pid;
                if (!gatherer->personId(people[j], pid)) {
                    continue;
                }

                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                SAnnotatedProbability annotatedProbability;
                if (populationModel->computeProbability(pid, time, time + bucketLength, partitioningFields, 1, annotatedProbability) ==
                    false) {
                    continue;
                }

                if (time < startTime + 3 * DAY) {
                    double& minimumProbability =
                        personProbabilitiesWithoutPeriodicity.insert(TStrDoubleMap::value_type(people[j], 1.0)).first->second;
                    minimumProbability = std::min(minimumProbability, annotatedProbability.s_Probability);
                } else if (time > startTime + 5 * DAY) {
                    double& minimumProbability =
                        personProbabilitiesWithPeriodicity.insert(TStrDoubleMap::value_type(people[j], 1.0)).first->second;
                    minimumProbability = std::min(minimumProbability, annotatedProbability.s_Probability);
                }
            }
            time += bucketLength;
        }

        addArrival(message, gatherer, m_ResourceMonitor);
    }

    double totalw = 0.0;
    double totalwo = 0.0;

    for (std::size_t i = 0u; i < boost::size(people); ++i) {
        TStrDoubleMapCItr wo = personProbabilitiesWithoutPeriodicity.find(people[i]);
        TStrDoubleMapCItr w = personProbabilitiesWithPeriodicity.find(people[i]);
        LOG_DEBUG("person = " << people[i]);
        LOG_DEBUG("minimum probability with periodicity    = " << w->second);
        LOG_DEBUG("minimum probability without periodicity = " << wo->second);
        totalwo += wo->second;
        totalw += w->second;
    }

    LOG_DEBUG("total minimum probability with periodicity    = " << totalw);
    LOG_DEBUG("total minimum probability without periodicity = " << totalwo);
    CPPUNIT_ASSERT(totalw > 3.0 * totalwo);
}

void CEventRatePopulationModelTest::testSkipSampling() {
    LOG_DEBUG("*** testSkipSampling ***");

    core_t::TTime startTime(100);
    std::size_t bucketLength(100);
    std::size_t maxAgeBuckets(5);
    SModelParams params(bucketLength);

    CEventRatePopulationModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_PopulationCountByBucketPersonAndAttribute);
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererNoGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoGap(factory.makeDataGatherer(gathererNoGapInitData));
    CModelFactory::SModelInitializationData modelNoGapInitData(gathererNoGap);
    CAnomalyDetectorModel::TModelPtr modelNoGapHolder(factory.makeModel(modelNoGapInitData));
    CEventRatePopulationModel* modelNoGap = dynamic_cast<CEventRatePopulationModel*>(modelNoGapHolder.get());

    addArrival(SMessage(100, "p1", "a1"), gathererNoGap, m_ResourceMonitor);
    addArrival(SMessage(100, "p1", "a2"), gathererNoGap, m_ResourceMonitor);
    addArrival(SMessage(100, "p2", "a1"), gathererNoGap, m_ResourceMonitor);
    modelNoGap->sample(100, 200, m_ResourceMonitor);
    addArrival(SMessage(200, "p1", "a1"), gathererNoGap, m_ResourceMonitor);
    modelNoGap->sample(200, 300, m_ResourceMonitor);
    addArrival(SMessage(300, "p1", "a1"), gathererNoGap, m_ResourceMonitor);
    modelNoGap->sample(300, 400, m_ResourceMonitor);

    CModelFactory::SGathererInitializationData gathererWithGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithGap(factory.makeDataGatherer(gathererWithGapInitData));
    CModelFactory::SModelInitializationData modelWithGapInitData(gathererWithGap);
    CAnomalyDetectorModel::TModelPtr modelWithGapHolder(factory.makeModel(modelWithGapInitData));
    CEventRatePopulationModel* modelWithGap = dynamic_cast<CEventRatePopulationModel*>(modelWithGapHolder.get());

    addArrival(SMessage(100, "p1", "a1"), gathererWithGap, m_ResourceMonitor);
    addArrival(SMessage(100, "p1", "a2"), gathererWithGap, m_ResourceMonitor);
    addArrival(SMessage(100, "p2", "a1"), gathererWithGap, m_ResourceMonitor);
    modelWithGap->sample(100, 200, m_ResourceMonitor);
    addArrival(SMessage(200, "p1", "a1"), gathererWithGap, m_ResourceMonitor);
    modelWithGap->skipSampling(1000);
    LOG_DEBUG("Calling sample over skipped interval should do nothing except print some ERRORs");
    modelWithGap->sample(200, 1000, m_ResourceMonitor);

    // Check prune does not remove people because last seen times are updated by adding gap duration
    modelWithGap->prune(maxAgeBuckets);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), gathererWithGap->numberActivePeople());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), gathererWithGap->numberActiveAttributes());

    addArrival(SMessage(1000, "p1", "a1"), gathererWithGap, m_ResourceMonitor);
    modelWithGap->sample(1000, 1100, m_ResourceMonitor);
    addArrival(SMessage(1100, "p1", "a1"), gathererWithGap, m_ResourceMonitor);
    modelWithGap->sample(1100, 1200, m_ResourceMonitor);

    // Check priors are the same
    CPPUNIT_ASSERT_EQUAL(static_cast<const maths::CUnivariateTimeSeriesModel*>(
                             modelWithGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0))
                             ->prior()
                             .checksum(),
                         static_cast<const maths::CUnivariateTimeSeriesModel*>(
                             modelNoGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0))
                             ->prior()
                             .checksum());
    CPPUNIT_ASSERT_EQUAL(static_cast<const maths::CUnivariateTimeSeriesModel*>(
                             modelWithGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1))
                             ->prior()
                             .checksum(),
                         static_cast<const maths::CUnivariateTimeSeriesModel*>(
                             modelNoGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1))
                             ->prior()
                             .checksum());

    // Confirm last seen times are only updated by gap duration by forcing p2 and a2 to be pruned
    modelWithGap->sample(1200, 1500, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 500 and since it's equal to maxAge it should still be here
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), gathererWithGap->numberActiveAttributes());
    modelWithGap->sample(1500, 1600, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 600 so it should get pruned
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gathererWithGap->numberActivePeople());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gathererWithGap->numberActiveAttributes());
}

void CEventRatePopulationModelTest::testInterimCorrections() {
    LOG_DEBUG("*** testInterimCorrections ***");

    core_t::TTime startTime(3600);
    std::size_t bucketLength(3600);
    SModelParams params(bucketLength);
    CEventRatePopulationModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_PopulationCountByBucketPersonAndAttribute);
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelHolder(factory.makeModel(modelInitData));
    CEventRatePopulationModel* model = dynamic_cast<CEventRatePopulationModel*>(modelHolder.get());

    test::CRandomNumbers rng;
    core_t::TTime now = startTime;
    core_t::TTime endTime = now + 2 * 24 * bucketLength;
    TDoubleVec samples(3, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, std::size_t(3), samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            addArrival(SMessage(now, "p1", "a1"), gatherer, m_ResourceMonitor);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[1] + 0.5); ++i) {
            addArrival(SMessage(now, "p2", "a1"), gatherer, m_ResourceMonitor);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[2] + 0.5); ++i) {
            addArrival(SMessage(now, "p3", "a2"), gatherer, m_ResourceMonitor);
        }
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        addArrival(SMessage(now, "p1", "a1"), gatherer, m_ResourceMonitor);
    }
    for (std::size_t i = 0; i < 1; ++i) {
        addArrival(SMessage(now, "p2", "a1"), gatherer, m_ResourceMonitor);
    }
    for (std::size_t i = 0; i < 100; ++i) {
        addArrival(SMessage(now, "p3", "a2"), gatherer, m_ResourceMonitor);
    }
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType.set(model_t::CResultType::E_Interim);
    CPPUNIT_ASSERT(model->computeProbability(0 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType.set(model_t::CResultType::E_Interim);
    CPPUNIT_ASSERT(model->computeProbability(1 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType.set(model_t::CResultType::E_Interim);
    CPPUNIT_ASSERT(model->computeProbability(2 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability3));

    model_t::CResultType type(model_t::CResultType::E_Unconditional | model_t::CResultType::E_Interim);
    TDouble1Vec p1a1Baseline =
        model->baselineBucketMean(model_t::E_PopulationCountByBucketPersonAndAttribute, 0, 0, type, NO_CORRELATES, now);
    TDouble1Vec p2a1Baseline =
        model->baselineBucketMean(model_t::E_PopulationCountByBucketPersonAndAttribute, 0, 0, type, NO_CORRELATES, now);
    TDouble1Vec p3a2Baseline =
        model->baselineBucketMean(model_t::E_PopulationCountByBucketPersonAndAttribute, 2, 1, type, NO_CORRELATES, now);

    LOG_DEBUG("p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG("p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG("p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG("p1a1 baseline = " << p1a1Baseline[0]);
    LOG_DEBUG("p2a1 baseline = " << p2a1Baseline[0]);
    LOG_DEBUG("p3a2 baseline = " << p3a2Baseline[0]);

    CPPUNIT_ASSERT(annotatedProbability1.s_Probability > 0.05);
    CPPUNIT_ASSERT(annotatedProbability2.s_Probability < 0.05);
    CPPUNIT_ASSERT(annotatedProbability3.s_Probability < 0.05);
    CPPUNIT_ASSERT(p1a1Baseline[0] > 45.0 && p1a1Baseline[0] < 46.0);
    CPPUNIT_ASSERT(p2a1Baseline[0] > 45.0 && p2a1Baseline[0] < 46.0);
    CPPUNIT_ASSERT(p3a2Baseline[0] > 59.0 && p3a2Baseline[0] < 61.0);
}

void CEventRatePopulationModelTest::testPersistence() {
    LOG_DEBUG("*** testPersistence ***");

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CEventRatePopulationModelFactory factory(params);
    CModelFactory::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));

    CModelFactory::SModelInitializationData modelInitData(gatherer);
    CAnomalyDetectorModel::TModelPtr origModel(factory.makeModel(modelInitData));

    CEventRatePopulationModel* populationModel = dynamic_cast<CEventRatePopulationModel*>(origModel.get());
    CPPUNIT_ASSERT(populationModel);

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            origModel->sample(startTime, startTime + bucketLength, m_ResourceMonitor);
            startTime += bucketLength;
        }
        addArrival(message, gatherer, m_ResourceMonitor);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origModel->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE("origXml = " << origXml);
    LOG_DEBUG("origXml size = " << origXml.size());

    // Restore the XML into a new data gatherer
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CAnomalyDetectorModel::TModelPtr restoredModel(factory.makeModel(modelInitData, traverser));

    populationModel = dynamic_cast<CEventRatePopulationModel*>(restoredModel.get());
    CPPUNIT_ASSERT(populationModel);

    // The XML representation of the new data gatherer should be the same as the
    // original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModel->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    LOG_DEBUG("original checksum = " << origModel->checksum(false));
    LOG_DEBUG("restored checksum = " << restoredModel->checksum(false));
    CPPUNIT_ASSERT_EQUAL(origModel->checksum(false), restoredModel->checksum(false));
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CEventRatePopulationModelTest::testIgnoreSamplingGivenDetectionRules() {
    LOG_DEBUG("*** testIgnoreSamplingGivenDetectionRules ***");

    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.

    core_t::TTime startTime(100);
    std::size_t bucketLength(100);

    // Create a categorical rule to filter out attribute a3
    std::string filterJson("[\"a3\"]");
    core::CPatternSet valueFilter;
    valueFilter.initFromJson(filterJson);

    CRuleCondition condition;
    condition.type(CRuleCondition::E_Categorical);
    condition.valueFilter(valueFilter);
    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipSampling);
    rule.addCondition(condition);

    SModelParams paramsNoRules(bucketLength);
    CEventRatePopulationModelFactory factory(paramsNoRules);
    model_t::TFeatureVec features(1u, model_t::E_PopulationCountByBucketPersonAndAttribute);
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererNoSkipInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoSkip(factory.makeDataGatherer(gathererNoSkipInitData));
    CModelFactory::SModelInitializationData modelNoSkipInitData(gathererNoSkip);
    CAnomalyDetectorModel::TModelPtr modelNoSkip(factory.makeModel(modelNoSkipInitData));

    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);

    CEventRatePopulationModelFactory factoryWithSkipRule(paramsWithRules);
    factoryWithSkipRule.features(features);

    CModelFactory::SGathererInitializationData gathererWithSkipInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithSkip(factoryWithSkipRule.makeDataGatherer(gathererWithSkipInitData));
    CModelFactory::SModelInitializationData modelWithSkipInitData(gathererWithSkip);
    CAnomalyDetectorModel::TModelPtr modelWithSkip(factoryWithSkipRule.makeModel(modelWithSkipInitData));

    addArrival(SMessage(100, "p1", "a1"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(100, "p1", "a1"), gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(100, "p1", "a2"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(100, "p1", "a2"), gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(100, "p2", "a1"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(100, "p2", "a1"), gathererWithSkip, m_ResourceMonitor);
    modelNoSkip->sample(100, 200, m_ResourceMonitor);
    modelWithSkip->sample(100, 200, m_ResourceMonitor);

    CPPUNIT_ASSERT_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    addArrival(SMessage(200, "p1", "a1"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p1", "a1"), gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p1", "a2"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p1", "a2"), gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p2", "a1"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p2", "a1"), gathererWithSkip, m_ResourceMonitor);

    // This should be filtered out
    addArrival(SMessage(200, "p1", "a3"), gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p2", "a3"), gathererWithSkip, m_ResourceMonitor);

    // Add another attribute that must not be skipped
    addArrival(SMessage(200, "p1", "a4"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p1", "a4"), gathererWithSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p2", "a4"), gathererNoSkip, m_ResourceMonitor);
    addArrival(SMessage(200, "p2", "a4"), gathererWithSkip, m_ResourceMonitor);

    modelNoSkip->sample(200, 300, m_ResourceMonitor);
    modelWithSkip->sample(200, 300, m_ResourceMonitor);

    // Checksums will be different because a model is created for attribute a3
    CPPUNIT_ASSERT(modelWithSkip->checksum() != modelNoSkip->checksum());

    CAnomalyDetectorModel::CModelDetailsViewPtr modelWithSkipView = modelWithSkip->details();
    CAnomalyDetectorModel::CModelDetailsViewPtr modelNoSkipView = modelNoSkip->details();

    // but the underlying models for attributes a1 and a2 are the same
    uint64_t withSkipChecksum = modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0)->checksum();
    uint64_t noSkipChecksum = modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0)->checksum();
    CPPUNIT_ASSERT_EQUAL(withSkipChecksum, noSkipChecksum);

    withSkipChecksum = modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1)->checksum();
    noSkipChecksum = modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1)->checksum();
    CPPUNIT_ASSERT_EQUAL(withSkipChecksum, noSkipChecksum);

    // The no skip model didn't see the a3 attribute only a1, a2 and a4.
    // The a4 models should be the same.
    withSkipChecksum = modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 3)->checksum();
    noSkipChecksum = modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 2)->checksum();
    CPPUNIT_ASSERT_EQUAL(withSkipChecksum, noSkipChecksum);

    // Check the last value times of all the underlying models are the same
    const maths::CUnivariateTimeSeriesModel* timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0));
    CPPUNIT_ASSERT(timeSeriesModel);

    core_t::TTime time = timeSeriesModel->trend().lastValueTime();
    CPPUNIT_ASSERT_EQUAL(model_t::sampleTime(model_t::E_PopulationCountByBucketPersonAndAttribute, 200, bucketLength), time);

    // The last times of the underlying time series models should all be the same
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 2));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());

    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 2));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 3));
    CPPUNIT_ASSERT_EQUAL(time, timeSeriesModel->trend().lastValueTime());
}

CppUnit::Test* CEventRatePopulationModelTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CEventRatePopulationModelTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testBasicAccessors",
                                                                                 &CEventRatePopulationModelTest::testBasicAccessors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testFeatures",
                                                                                 &CEventRatePopulationModelTest::testFeatures));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testComputeProbability",
                                                                                 &CEventRatePopulationModelTest::testComputeProbability));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testPrune",
                                                                                 &CEventRatePopulationModelTest::testPrune));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testKey",
                                                                                 &CEventRatePopulationModelTest::testKey));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testFrequency",
                                                                                 &CEventRatePopulationModelTest::testFrequency));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testSampleRateWeight",
                                                                                 &CEventRatePopulationModelTest::testSampleRateWeight));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testSkipSampling",
                                                                                 &CEventRatePopulationModelTest::testSkipSampling));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testInterimCorrections",
                                                                                 &CEventRatePopulationModelTest::testInterimCorrections));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testPeriodicity",
                                                                                 &CEventRatePopulationModelTest::testPeriodicity));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testPersistence",
                                                                                 &CEventRatePopulationModelTest::testPersistence));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CEventRatePopulationModelTest>("CEventRatePopulationModelTest::testIgnoreSamplingGivenDetectionRules",
                                                               &CEventRatePopulationModelTest::testIgnoreSamplingGivenDetectionRules));

    return suiteOfTests;
}
