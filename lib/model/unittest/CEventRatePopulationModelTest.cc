/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPatternSet.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>

#include <maths/CModelWeight.h>
#include <maths/COrderings.h>
#include <maths/CTimeSeriesDecomposition.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CCountingModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CEventData.h>
#include <model/CEventRatePopulationModel.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>
#include <model/CPartitioningFields.h>
#include <model/CResourceMonitor.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/BoostTestPointerOutput.h>
#include <test/CRandomNumbers.h>

#include "CModelTestFixtureBase.h"

#include <boost/optional/optional_io.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/unordered_map.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CEventRatePopulationModelTest)

using namespace ml;
using namespace model;

namespace {
const CModelTestFixtureBase::TSizeDoublePr1Vec NO_CORRELATES;
}

class CTestFixture : public CModelTestFixtureBase {
public:
    void generateTestMessages(core_t::TTime startTime,
                              core_t::TTime bucketLength,
                              TMessageVec& messages) {
        // The test case is as follows:
        //
        //   attribute   |    0    |    1    |    2    |    3    |   4
        // --------------+---------+---------+---------+---------+--------
        //    people     |  [0-19] |  [0-2], |  [0,2], |   3,4   |   3
        //               |         |  [5-19] |  [4,19] |         |
        // --------------+---------+---------+---------+---------+--------
        //    rate       |   10    |   0.02  |   15    |    2    |   1
        // --------------+---------+---------+---------+---------+--------
        //  rate anomaly |   1,11  |    -    |   4,5   |   n/a   |  n/a
        //     people    |         |         |         |         |
        //
        // There are 100 buckets.

        using TStrVec = std::vector<std::string>;
        using TSizeSizeSizeTr = boost::tuple<std::size_t, std::size_t, size_t>;

        const std::size_t numberBuckets = 100;
        const std::size_t numberAttributes = 5;
        const std::size_t numberPeople = 20;

        TStrVec attributes;
        for (std::size_t i = 0; i < numberAttributes; ++i) {
            attributes.push_back("c" + std::to_string(i));
        }

        TStrVec people;
        for (std::size_t i = 0; i < numberPeople; ++i) {
            people.push_back("p" + std::to_string(i));
        }

        TSizeVecVec attributePeople{
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
            {0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
            {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
            {3, 4},
            {3}};

        double attributeRates[] = {10.0, 0.02, 15.0, 2.0, 1.0};

        TSizeSizeSizeTr anomaliesAttributePerson[] = {
            {10u, 0u, 1u}, {15u, 0u, 11u}, {30u, 2u, 4u},
            {35u, 2u, 5u}, {50u, 0u, 11u}, {75u, 2u, 5u}};

        test::CRandomNumbers rng;

        for (std::size_t i = 0; i < numberBuckets; ++i, startTime += bucketLength) {
            for (std::size_t j = 0; j < numberAttributes; ++j) {
                TUIntVec samples;
                rng.generatePoissonSamples(attributeRates[j],
                                           attributePeople[j].size(), samples);

                for (std::size_t k = 0; k < samples.size(); ++k) {
                    unsigned int n = samples[k];
                    if (std::binary_search(std::begin(anomaliesAttributePerson),
                                           std::end(anomaliesAttributePerson),
                                           TSizeSizeSizeTr(i, j, attributePeople[j][k]))) {
                        n += static_cast<unsigned int>(2.5 * attributeRates[j]);
                        LOG_DEBUG(<< i << " " << attributes[j]
                                  << " generating anomaly " << n);
                    }

                    TDoubleVec times;
                    rng.generateUniformSamples(
                        0.0, static_cast<double>(bucketLength - 1), n, times);

                    for (std::size_t l = 0; l < times.size(); ++l) {
                        core_t::TTime time = startTime +
                                             static_cast<core_t::TTime>(times[l]);
                        messages.emplace_back(time, people[attributePeople[j][k]],
                                              attributes[j]);
                    }
                }
            }
        }

        std::sort(messages.begin(), messages.end());
    }

    void makeModel(const SModelParams& params,
                   const model_t::TFeatureVec& features,
                   core_t::TTime startTime) {
        this->makeModelT<CEventRatePopulationModelFactory>(
            params, features, startTime, model_t::E_EventRateOnline, m_Gatherer, m_Model);
    }
};

BOOST_FIXTURE_TEST_CASE(testBasicAccessors, CTestFixture) {
    // Check that the correct data is read retrieved by the
    // basic model accessors.

    using TSizeUInt64Map = std::map<std::size_t, uint64_t>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);
    LOG_DEBUG(<< "# messages = " << messages.size());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    this->makeModel(params, {model_t::E_PopulationCountByBucketPersonAndAttribute}, startTime);
    CEventRatePopulationModel* model =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    TSizeUInt64Map expectedBucketPersonCounts;
    TSizeSizePrUInt64Map expectedBucketPersonAttributeCounts;

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            LOG_DEBUG(<< "Testing bucket = [" << startTime << ","
                      << startTime + bucketLength << ")");

            // Test the person and attribute invariants.
            for (std::size_t j = 0; j < m_Gatherer->numberActivePeople(); ++j) {
                const std::string& name = model->personName(j);
                std::size_t pid;
                BOOST_TEST_REQUIRE(m_Gatherer->personId(name, pid));
                BOOST_REQUIRE_EQUAL(j, pid);
            }
            for (std::size_t j = 0; j < m_Gatherer->numberActiveAttributes(); ++j) {
                const std::string& name = model->attributeName(j);
                std::size_t cid;
                BOOST_TEST_REQUIRE(m_Gatherer->attributeId(name, cid));
                BOOST_REQUIRE_EQUAL(j, cid);
            }

            TSizeVec expectedCurrentBucketPersonIds;

            // Test the person counts.
            for (const auto& expectedCount : expectedBucketPersonCounts) {
                std::size_t pid = expectedCount.first;
                expectedCurrentBucketPersonIds.push_back(pid);
                auto count = model->currentBucketCount(pid, startTime);
                BOOST_TEST_REQUIRE(count);
                BOOST_REQUIRE_EQUAL(expectedCount.second, *count);
            }

            // Test the person attribute counts.
            for (const auto& expectedCount : expectedBucketPersonAttributeCounts) {
                std::size_t pid = expectedCount.first.first;
                std::size_t cid = expectedCount.first.second;
                TDouble1Vec count = model->currentBucketValue(
                    model_t::E_PopulationCountByBucketPersonAndAttribute, pid, cid, startTime);
                BOOST_TEST_REQUIRE(!count.empty());
                BOOST_REQUIRE_EQUAL(static_cast<double>(expectedCount.second), count[0]);
            }

            // Test the current bucket people.
            std::sort(expectedCurrentBucketPersonIds.begin(),
                      expectedCurrentBucketPersonIds.end());
            TSizeVec bucketPersonIds;
            model->currentBucketPersonIds(startTime, bucketPersonIds);
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedCurrentBucketPersonIds),
                                core::CContainerPrinter::print(bucketPersonIds));

            expectedBucketPersonCounts.clear();
            expectedBucketPersonAttributeCounts.clear();
            startTime += bucketLength;
        }

        this->addArrival(message, m_Gatherer);

        std::size_t pid, cid;
        BOOST_TEST_REQUIRE(m_Gatherer->personId(message.s_Person, pid));
        BOOST_TEST_REQUIRE(m_Gatherer->attributeId(message.s_Attribute.get(), cid));
        ++expectedBucketPersonCounts[pid];
        ++expectedBucketPersonAttributeCounts[{pid, cid}];
    }
}

BOOST_FIXTURE_TEST_CASE(testFeatures, CTestFixture) {
    // We check that the correct data is read from the gatherer
    // into the model on sample.

    using TSizeSet = std::set<std::size_t>;
    using TSizeSizeSetMap = std::map<std::size_t, TSizeSet>;
    using TFeatureData = SEventRateFeatureData;
    using TSizeSizePrFeatureDataPr = CEventRatePopulationModel::TSizeSizePrFeatureDataPr;
    using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2VecVec = std::vector<TDouble2Vec>;
    using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;
    using TMathsModelPtr = std::shared_ptr<maths::CModel>;
    using TSizeMathsModelPtrMap = std::map<std::size_t, TMathsModelPtr>;

    // Manages de-duplication of values.
    class CUniqueValues {
    public:
        void add(double value,
                 const maths_t::TDouble2VecWeightsAry& trendWeight,
                 const maths_t::TDouble2VecWeightsAry& residualWeight) {
            std::size_t duplicate =
                m_Uniques.emplace(value, m_Uniques.size()).first->second;
            if (duplicate < m_Values.size()) {
                maths_t::addCount(maths_t::count(trendWeight), m_TrendWeights[duplicate]);
                maths_t::addCount(maths_t::count(residualWeight),
                                  m_ResidualWeights[duplicate]);
            } else {
                m_Values.push_back({value});
                m_TrendWeights.push_back(trendWeight);
                m_ResidualWeights.push_back(residualWeight);
            }
        }
        TDouble2VecVec& values() { return m_Values; }
        TDouble2VecWeightsAryVec& trendWeights() { return m_TrendWeights; }
        TDouble2VecWeightsAryVec& residualWeights() {
            return m_ResidualWeights;
        }

    private:
        using TDoubleSizeUMap = boost::unordered_map<double, std::size_t>;

    private:
        TDoubleSizeUMap m_Uniques;
        TDouble2VecVec m_Values;
        TDouble2VecWeightsAryVec m_TrendWeights;
        TDouble2VecWeightsAryVec m_ResidualWeights;
    };
    using TSizeUniqueValuesUMap = boost::unordered_map<std::size_t, CUniqueValues>;

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);
    LOG_DEBUG(<< "# messages = " << messages.size());

    // Bucket non-zero count unique person count.
    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);
    CEventRatePopulationModel* model =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    model::CModelFactory::TFeatureMathsModelPtrPrVec models{
        m_Factory->defaultFeatureModels(features, bucketLength, 1.0, false)};
    BOOST_REQUIRE_EQUAL(1, models.size());
    BOOST_REQUIRE_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                        models[0].first);

    std::size_t numberAttributes = 0;
    std::size_t numberPeople = 0;
    TSizeSizeSetMap attributePeople;
    TSizeSizePrUInt64Map expectedCounts;
    TSizeMathsModelPtrMap expectedPopulationModels;

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            for (const auto& expectedCount : expectedCounts) {
                std::size_t pid = expectedCount.first.first;
                std::size_t cid = expectedCount.first.second;
                numberAttributes = std::max(numberAttributes, cid + 1);
                numberPeople = std::max(numberPeople, pid + 1);
                attributePeople[cid].insert(pid);
            }

            TSizeUniqueValuesUMap expectedValuesAndWeights;
            for (const auto& expectedCount : expectedCounts) {
                std::size_t pid = expectedCount.first.first;
                std::size_t cid = expectedCount.first.second;
                core_t::TTime time = startTime + bucketLength / 2;
                double count = model_t::offsetCountToZero(
                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                    static_cast<double>(expectedCount.second));
                TMathsModelPtr& attributeModel = expectedPopulationModels[cid];
                if (attributeModel == nullptr) {
                    attributeModel.reset(models[0].second->clone(cid));
                }
                double countWeight{model->sampleRateWeight(pid, cid)};

                maths_t::TDouble2VecWeightsAry trendWeight(
                    maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                maths_t::TDouble2VecWeightsAry residualWeight(
                    maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                attributeModel->countWeights(time, {count}, countWeight, countWeight,
                                             1.0, 1.0, trendWeight, residualWeight);
                expectedValuesAndWeights[cid].add(count, trendWeight, residualWeight);
            }
            for (auto& attributeExpectedValues : expectedValuesAndWeights) {
                std::size_t cid = attributeExpectedValues.first;
                TDouble2VecVec& values = attributeExpectedValues.second.values();
                TDouble2VecWeightsAryVec& trendWeights =
                    attributeExpectedValues.second.trendWeights();
                TDouble2VecWeightsAryVec& residualWeights =
                    attributeExpectedValues.second.residualWeights();
                maths::COrderings::simultaneousSort(values, trendWeights, residualWeights);
                maths::CModel::TTimeDouble2VecSizeTrVec samples;
                for (const auto& sample : values) {
                    samples.emplace_back(startTime + bucketLength / 2, sample, 0);
                }
                maths::CModelAddSamplesParams params_;
                params_.integer(true)
                    .nonNegative(true)
                    .propagationInterval(1.0)
                    .trendWeights(trendWeights)
                    .priorWeights(residualWeights);
                expectedPopulationModels[cid]->addSamples(params_, samples);
            }

            TSizeSizePrFeatureDataPrVec expectedPeoplePerAttribute;
            expectedPeoplePerAttribute.reserve(numberAttributes);
            for (std::size_t j = 0; j < numberAttributes; ++j) {
                expectedPeoplePerAttribute.emplace_back(std::make_pair(size_t(0), j),
                                                        TFeatureData(j));
            }
            for (const auto& attribute : attributePeople) {
                expectedPeoplePerAttribute[attribute.first].second =
                    attribute.second.size();
            }

            // Check the number of people per attribute.
            const TSizeSizePrFeatureDataPrVec& peoplePerAttribute = model->featureData(
                model_t::E_PopulationUniquePersonCountByAttribute, startTime);
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedPeoplePerAttribute),
                                core::CContainerPrinter::print(peoplePerAttribute));

            // Check the non-zero (person, attribute) counts.
            const TSizeSizePrFeatureDataPrVec& nonZeroCounts = model->featureData(
                model_t::E_PopulationCountByBucketPersonAndAttribute, startTime);
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedCounts),
                                core::CContainerPrinter::print(nonZeroCounts));

            for (std::size_t cid = 0; cid < numberAttributes; ++cid) {
                const maths::CModel* populationModel = model->details()->model(
                    model_t::E_PopulationCountByBucketPersonAndAttribute, cid);
                BOOST_TEST_REQUIRE(populationModel);
                BOOST_REQUIRE_EQUAL(expectedPopulationModels[cid]->checksum(),
                                    populationModel->checksum());
            }

            startTime += bucketLength;
            expectedCounts.clear();
        }

        this->addArrival(message, m_Gatherer);

        std::size_t pid, cid;
        BOOST_TEST_REQUIRE(m_Gatherer->personId(message.s_Person, pid));
        BOOST_TEST_REQUIRE(m_Gatherer->attributeId(message.s_Attribute.get(), cid));
        ++expectedCounts[{pid, cid}];
    }
}

BOOST_FIXTURE_TEST_CASE(testComputeProbability, CTestFixture) {
    // Check that we get the probabilities we expect.
    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);
    CEventRatePopulationModel* model =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    TStrVec expectedAnomalies{"[10, p1, c0]", "[15, p11, c0]", "[30, p4, c2]",
                              "[35, p5, c2]", "[50, p11, c0]", "[75, p5, c2]"};

    TAnomalyVec orderedAnomalies;

    this->generateOrderedAnomalies(6u, startTime, bucketLength, messages,
                                   m_Gatherer, *model, orderedAnomalies);

    BOOST_REQUIRE_EQUAL(expectedAnomalies.size(), orderedAnomalies.size());
    for (std::size_t i = 0; i < orderedAnomalies.size(); ++i) {
        BOOST_REQUIRE_EQUAL(expectedAnomalies[i], orderedAnomalies[i].print());
    }
}

BOOST_FIXTURE_TEST_CASE(testPrune, CTestFixture) {
    // This test has four people and five attributes. We expect
    // person 2 and attributes 1, 2 and 5 to be deleted.

    core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 1000;

    TStrVec people{"p1", "p2", "p3", "p4"};
    TStrVec attributes{"c1", "c2", "c3", "c4", "c5"};
    TStrSizePrVecVecVec eventCounts{{}, {}, {}, {}};

    {
        TStrSizePrVec attributeCounts;
        attributeCounts.emplace_back(attributes[0], 0);
        attributeCounts.emplace_back(attributes[1], 0);
        attributeCounts.emplace_back(attributes[2], 1);
        attributeCounts.emplace_back(attributes[4], 0);
        eventCounts[0].resize(numberBuckets, attributeCounts);
        eventCounts[0][1][0].second = 2; // p1, bucket 2, c1
        eventCounts[0][3][0].second = 4; // p1, bucket 3, c1
        eventCounts[0][5][1].second = 4; // p1, bucket 5, c2
        eventCounts[0][5][3].second = 3; // p1, bucket 5, c5
    }
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.emplace_back(attributes[0], 0);
        attributeCounts.emplace_back(attributes[4], 0);
        eventCounts[1].resize(numberBuckets, attributeCounts);
        eventCounts[1][0][0].second = 2; // p2, bucket 1, c1
        eventCounts[1][2][0].second = 3; // p2, bucket 3, c1
        eventCounts[1][0][1].second = 4; // p2, bucket 1, c5
        eventCounts[1][3][1].second = 1; // p2, bucket 4, c5
    }
    {
        TStrSizePrVec attributeCounts;
        attributeCounts.emplace_back(attributes[2], 0);
        attributeCounts.emplace_back(attributes[3], 2);
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
        attributeCounts.emplace_back(attributes[1], 0);
        attributeCounts.emplace_back(attributes[3], 3);
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
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);
    CEventRatePopulationModel* model =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    CModelFactory::TDataGathererPtr expectedGatherer(m_Factory->makeDataGatherer({startTime}));
    CAnomalyDetectorModel::TModelPtr expectedModel(m_Factory->makeModel({expectedGatherer}));
    BOOST_TEST_REQUIRE(expectedModel);

    TMessageVec messages;
    for (std::size_t i = 0; i < people.size(); ++i) {
        core_t::TTime bucketStart = startTime;
        for (std::size_t j = 0; j < numberBuckets; ++j, bucketStart += bucketLength) {
            const TStrSizePrVec& attributeEventCounts = eventCounts[i][j];
            for (std::size_t k = 0; k < attributeEventCounts.size(); ++k) {
                if (attributeEventCounts[k].second == 0) {
                    continue;
                }
                std::size_t n = attributeEventCounts[k].second;
                core_t::TTime time = bucketStart;
                core_t::TTime dt = bucketLength / static_cast<core_t::TTime>(n);
                for (std::size_t l = 0; l < n; ++l, time += dt) {
                    messages.emplace_back(time, people[i],
                                          attributeEventCounts[k].first);
                }
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    TMessageVec expectedMessages;
    expectedMessages.reserve(messages.size());
    for (const auto& message : messages) {
        if (std::binary_search(std::begin(expectedPeople),
                               std::end(expectedPeople), message.s_Person) &&
            std::binary_search(std::begin(expectedAttributes), std::end(expectedAttributes),
                               message.s_Attribute.get())) {
            expectedMessages.push_back(message);
        }
    }

    core_t::TTime bucketStart = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        this->addArrival(message, m_Gatherer);
    }
    model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(model->dataGatherer().maxDimension());
    model->prune(model->defaultPruneWindow());
    size_t maxDimensionAfterPrune(model->dataGatherer().maxDimension());
    BOOST_REQUIRE_EQUAL(maxDimensionBeforePrune, maxDimensionAfterPrune);

    bucketStart = startTime;
    for (const auto& expectedMessage : expectedMessages) {
        if (expectedMessage.s_Time >= bucketStart + bucketLength) {
            expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        this->addArrival(expectedMessage, expectedGatherer);
    }
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), model->checksum());

    // Now check that we recycle the person and attribute slots.

    bucketStart = m_Gatherer->currentBucketStartTime() + bucketLength;

    TMessageVec newMessages{{bucketStart + 10, "p1", TOptionalStr("c2")},
                            {bucketStart + 200, "p5", TOptionalStr("c6")},
                            {bucketStart + 2100, "p5", TOptionalStr("c6")}};

    for (const auto& newMessage : newMessages) {
        this->addArrival(newMessage, m_Gatherer);
        this->addArrival(newMessage, expectedGatherer);
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
    function_t::TFunctionVec countFunctions{function_t::E_PopulationCount,
                                            function_t::E_PopulationDistinctCount,
                                            function_t::E_PopulationRare,
                                            function_t::E_PopulationRareCount,
                                            function_t::E_PopulationFreqRare,
                                            function_t::E_PopulationFreqRareCount,
                                            function_t::E_PopulationLowCounts,
                                            function_t::E_PopulationHighCounts};

    std::string fieldName;
    std::string overFieldName{"over"};

    generateAndCompareKey(countFunctions, fieldName, overFieldName,
                          [](CSearchKey expectedKey, CSearchKey actualKey) {
                              BOOST_TEST_REQUIRE(expectedKey == actualKey);
                          });
}

BOOST_FIXTURE_TEST_CASE(testFrequency, CTestFixture) {
    // Test we correctly compute frequencies for people and attributes.

    struct SDatum {
        std::string s_Attribute;
        std::string s_Person;
        std::size_t s_Period{0};
    };

    using TDataVec = std::vector<SDatum>;
    TDataVec data{{"a1", "p1", 1u},  {"a2", "p2", 1u}, {"a3", "p3", 10u},
                  {"a4", "p4", 3u},  {"a5", "p5", 4u}, {"a6", "p6", 5u},
                  {"a7", "p7", 2u},  {"a8", "p8", 1u}, {"a9", "p9", 3u},
                  {"a10", "p10", 7u}};

    const core_t::TTime bucketLength{600};

    core_t::TTime startTime{0};

    TMessageVec messages;
    std::size_t bucket{0u};
    for (core_t::TTime bucketStart = startTime; bucketStart < 100 * bucketLength;
         bucketStart += bucketLength, ++bucket) {
        std::size_t i{0u};
        for (auto& datum : data) {
            if (bucket % datum.s_Period == 0) {
                for (std::size_t j = 0; j < i + 1; ++j) {
                    messages.emplace_back(bucketStart + bucketLength / 2,
                                          datum.s_Person, data[j].s_Attribute);
                }
            }
            ++i;
        }
    }

    std::sort(messages.begin(), messages.end());
    LOG_DEBUG(<< "# messages = " << messages.size());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);
    CEventRatePopulationModel* populationModel =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(populationModel);

    core_t::TTime time{startTime};
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            populationModel->sample(time, time + bucketLength, m_ResourceMonitor);
            time += bucketLength;
        }
        this->addArrival(message, m_Gatherer);
    }

    {
        TMeanAccumulator meanError;
        for (auto& datum : data) {
            LOG_DEBUG(<< "*** person = " << datum.s_Person << " ***");
            std::size_t pid;
            BOOST_TEST_REQUIRE(m_Gatherer->personId(datum.s_Person, pid));
            LOG_DEBUG(<< "frequency = " << populationModel->personFrequency(pid));
            LOG_DEBUG(<< "expected frequency = "
                      << 1.0 / static_cast<double>(datum.s_Period));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0 / static_cast<double>(datum.s_Period),
                                         populationModel->personFrequency(pid),
                                         0.1 / static_cast<double>(datum.s_Period));
            meanError.add(std::fabs(populationModel->personFrequency(pid) -
                                    1.0 / static_cast<double>(datum.s_Period)));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(meanError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.002);
    }
    {
        std::size_t i{0};
        for (auto& datum : data) {
            LOG_DEBUG(<< "*** attribute = " << datum.s_Attribute << " ***");
            std::size_t cid;
            BOOST_TEST_REQUIRE(m_Gatherer->attributeId(datum.s_Attribute, cid));
            LOG_DEBUG(<< "frequency = " << populationModel->attributeFrequency(cid));
            LOG_DEBUG(<< "expected frequency = " << (10.0 - static_cast<double>(i)) / 10.0);
            BOOST_REQUIRE_EQUAL((10.0 - static_cast<double>(i)) / 10.0,
                                populationModel->attributeFrequency(cid));
            ++i;
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
    const TStrVec attributes{"a1", "a2", "a3", "a4", "a5",
                             "a6", "a7", "a8", "a9", "a10"};
    const TStrVec people{"p1",  "p2",  "p3",  "p4",  "p5",  "p6",  "p7",
                         "p8",  "p9",  "p10", "p11", "p12", "p13", "p14",
                         "p15", "p16", "p17", "p18", "p19", "p20"};
    TSizeVec heavyHitters{0u, 4u};
    TSizeVec normal{1u,  2u,  3u,  5u,  6u,  7u,  8u,  9u,  10u,
                    11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u};

    std::size_t messagesPerBucket = heavyHitters.size() * attributes.size() +
                                    normal.size();

    test::CRandomNumbers rng;

    core_t::TTime startTime{0};

    TMessageVec messages;
    for (core_t::TTime bucketStart = startTime;
         bucketStart < 100 * bucketLength; bucketStart += bucketLength) {
        TSizeVec times;
        rng.generateUniformSamples(static_cast<std::size_t>(bucketStart),
                                   static_cast<std::size_t>(bucketStart + bucketLength),
                                   messagesPerBucket, times);

        std::size_t m{0u};
        for (auto& attribute : attributes) {
            for (auto& heavyHitter : heavyHitters) {
                messages.emplace_back(static_cast<core_t::TTime>(times[m++]),
                                      people[heavyHitter], attribute);
            }
        }

        TSizeVec attributeIndexes;
        rng.generateUniformSamples(0, attributes.size(), normal.size(), attributeIndexes);
        std::size_t i{0};
        for (auto& norm : normal) {
            messages.emplace_back(static_cast<core_t::TTime>(times[m++]),
                                  people[norm], attributes[attributeIndexes[i++]]);
        }
    }

    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);

    CEventRatePopulationModel* populationModel =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(populationModel);

    core_t::TTime time{startTime};
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            populationModel->sample(time, time + bucketLength, m_ResourceMonitor);
            time += bucketLength;
        }
        this->addArrival(message, m_Gatherer);
    }

    // The heavy hitters generate one value per attribute per bucket.
    // The rest generate one value per bucket. Therefore, we expect
    // the mean person rate per bucket to be:
    //     (  ("# people" - "# heavy hitters" / "# attributes")
    //      + ("# heavy hitters"))
    //   / "# people"

    double expectedRateWeight = (static_cast<double>(normal.size()) /
                                     static_cast<double>(attributes.size()) +
                                 static_cast<double>(heavyHitters.size())) /
                                static_cast<double>(people.size());
    LOG_DEBUG(<< "expectedRateWeight = " << expectedRateWeight);

    for (auto& heavyHitter : heavyHitters) {
        LOG_DEBUG(<< "*** person = " << people[heavyHitter] << " ***");
        std::size_t pid;
        BOOST_TEST_REQUIRE(m_Gatherer->personId(people[heavyHitter], pid));
        for (std::size_t cid = 0; cid < attributes.size(); ++cid) {
            double sampleRateWeight = populationModel->sampleRateWeight(pid, cid);
            LOG_DEBUG(<< "attribute = " << populationModel->attributeName(cid)
                      << ", sampleRateWeight = " << sampleRateWeight);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedRateWeight, sampleRateWeight,
                                         0.15 * expectedRateWeight);
        }
    }

    for (auto& norm : normal) {
        LOG_DEBUG(<< "*** person = " << people[norm] << " ***");
        std::size_t pid;
        BOOST_TEST_REQUIRE(m_Gatherer->personId(people[norm], pid));
        for (std::size_t cid = 0; cid < attributes.size(); ++cid) {
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

    static const core_t::TTime HOUR{3600};
    static const core_t::TTime DAY{86400};

    const core_t::TTime bucketLength{3600};
    TDoubleVec rate{1, 1, 2, 2,  3, 5, 6, 6, 20, 21, 4, 3,
                    4, 4, 8, 25, 7, 6, 5, 1, 1,  4,  1, 1};

    TDoubleStrPrVec attribs{{1.0, "a1"}, {1.5, "a2"}};

    const TStrVec people{"p1", "p2", "p3", "p4", "p5",
                         "p6", "p7", "p8", "p9", "p10"};

    test::CRandomNumbers rng;

    core_t::TTime startTime{0};
    core_t::TTime endTime{604800};

    TMessageVec messages;
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        for (const auto& attrib : attribs) {
            TUIntVec rates;
            rng.generatePoissonSamples(attrib.first * rate[(time % DAY) / HOUR],
                                       people.size(), rates);

            std::size_t j{0};
            for (const auto& rate_ : rates) {
                for (unsigned int t = 0; t < rate_; ++t) {
                    messages.emplace_back(time + (t * bucketLength) / (rate_ + 1),
                                          people[j], attrib.second);
                }
                ++j;
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    params.s_MinimumModeCount = 24.0;
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);

    TStrDoubleMap personProbabilitiesWithoutPeriodicity;
    TStrDoubleMap personProbabilitiesWithPeriodicity;

    core_t::TTime time{startTime};
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            m_Model->sample(time, time + bucketLength, m_ResourceMonitor);

            for (const auto& person : people) {
                std::size_t pid;
                if (!m_Gatherer->personId(person, pid)) {
                    continue;
                }

                CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
                SAnnotatedProbability annotatedProbability;
                if (m_Model->computeProbability(pid, time, time + bucketLength, partitioningFields,
                                                1, annotatedProbability) == false) {
                    continue;
                }

                if (time < startTime + 3 * DAY) {
                    double& minimumProbability =
                        personProbabilitiesWithoutPeriodicity
                            .insert(TStrDoubleMap::value_type(person, 1.0))
                            .first->second;
                    minimumProbability = std::min(
                        minimumProbability, annotatedProbability.s_Probability);
                } else if (time > startTime + 5 * DAY) {
                    double& minimumProbability =
                        personProbabilitiesWithPeriodicity
                            .insert(TStrDoubleMap::value_type(person, 1.0))
                            .first->second;
                    minimumProbability = std::min(
                        minimumProbability, annotatedProbability.s_Probability);
                }
            }
            time += bucketLength;
        }

        this->addArrival(message, m_Gatherer);
    }

    double totalw{0.0};
    double totalwo{0.0};

    for (const auto& person : people) {
        auto wo = personProbabilitiesWithoutPeriodicity.find(person);
        auto w = personProbabilitiesWithPeriodicity.find(person);
        LOG_DEBUG(<< "person = " << person);
        LOG_DEBUG(<< "minimum probability with periodicity    = " << w->second);
        LOG_DEBUG(<< "minimum probability without periodicity = " << wo->second);
        totalwo += wo->second;
        totalw += w->second;
    }

    LOG_DEBUG(<< "total minimum probability with periodicity    = " << totalw);
    LOG_DEBUG(<< "total minimum probability without periodicity = " << totalwo);
    BOOST_TEST_REQUIRE(totalw > 3.0 * totalwo);
}

BOOST_FIXTURE_TEST_CASE(testSkipSampling, CTestFixture) {
    core_t::TTime startTime{100};
    std::size_t bucketLength{100};
    std::size_t maxAgeBuckets{5};

    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CEventRatePopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationCountByBucketPersonAndAttribute});

    CModelFactory::SGathererInitializationData gathererNoGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoGap(factory.makeDataGatherer(gathererNoGapInitData));
    CModelFactory::SModelInitializationData modelNoGapInitData(gathererNoGap);
    CAnomalyDetectorModel::TModelPtr modelNoGapHolder(factory.makeModel(modelNoGapInitData));
    CEventRatePopulationModel* modelNoGap =
        dynamic_cast<CEventRatePopulationModel*>(modelNoGapHolder.get());

    this->addArrival(SMessage(100, "p1", TOptionalStr("a1")), gathererNoGap);
    this->addArrival(SMessage(100, "p1", TOptionalStr("a2")), gathererNoGap);
    this->addArrival(SMessage(100, "p2", TOptionalStr("a1")), gathererNoGap);
    modelNoGap->sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", TOptionalStr("a1")), gathererNoGap);
    modelNoGap->sample(200, 300, m_ResourceMonitor);
    this->addArrival(SMessage(300, "p1", TOptionalStr("a1")), gathererNoGap);
    modelNoGap->sample(300, 400, m_ResourceMonitor);

    CModelFactory::SGathererInitializationData gathererWithGapInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithGap(
        factory.makeDataGatherer(gathererWithGapInitData));
    CModelFactory::SModelInitializationData modelWithGapInitData(gathererWithGap);
    CAnomalyDetectorModel::TModelPtr modelWithGapHolder(factory.makeModel(modelWithGapInitData));
    CEventRatePopulationModel* modelWithGap =
        dynamic_cast<CEventRatePopulationModel*>(modelWithGapHolder.get());

    this->addArrival(SMessage(100, "p1", TOptionalStr("a1")), gathererWithGap);
    this->addArrival(SMessage(100, "p1", TOptionalStr("a2")), gathererWithGap);
    this->addArrival(SMessage(100, "p2", TOptionalStr("a1")), gathererWithGap);
    modelWithGap->sample(100, 200, m_ResourceMonitor);
    this->addArrival(SMessage(200, "p1", TOptionalStr("a1")), gathererWithGap);
    modelWithGap->skipSampling(1000);
    LOG_DEBUG(<< "Calling sample over skipped interval should do nothing except print some ERRORs");
    modelWithGap->sample(200, 1000, m_ResourceMonitor);

    // Check prune does not remove people because last seen times are updated by adding gap duration
    modelWithGap->prune(maxAgeBuckets);
    BOOST_REQUIRE_EQUAL(2, gathererWithGap->numberActivePeople());
    BOOST_REQUIRE_EQUAL(2, gathererWithGap->numberActiveAttributes());

    this->addArrival(SMessage(1000, "p1", TOptionalStr("a1")), gathererWithGap);
    modelWithGap->sample(1000, 1100, m_ResourceMonitor);
    this->addArrival(SMessage(1100, "p1", TOptionalStr("a1")), gathererWithGap);
    modelWithGap->sample(1100, 1200, m_ResourceMonitor);

    // Check priors are the same
    BOOST_REQUIRE_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0))
            ->residualModel()
            .checksum());
    BOOST_REQUIRE_EQUAL(
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelWithGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1))
            ->residualModel()
            .checksum(),
        static_cast<const maths::CUnivariateTimeSeriesModel*>(
            modelNoGap->details()->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1))
            ->residualModel()
            .checksum());

    // Confirm last seen times are only updated by gap duration by forcing p2 and a2 to be pruned
    modelWithGap->sample(1200, 1500, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 500 and since it's equal to maxAge it should still be here
    BOOST_REQUIRE_EQUAL(2, gathererWithGap->numberActiveAttributes());
    modelWithGap->sample(1500, 1600, m_ResourceMonitor);
    modelWithGap->prune(maxAgeBuckets);
    // Age at this point will be 600 so it should get pruned
    BOOST_REQUIRE_EQUAL(1, gathererWithGap->numberActivePeople());
    BOOST_REQUIRE_EQUAL(1, gathererWithGap->numberActiveAttributes());
}

BOOST_FIXTURE_TEST_CASE(testInterimCorrections, CTestFixture) {
    core_t::TTime startTime{3600};
    std::size_t bucketLength{3600};

    SModelParams params(bucketLength);
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute};
    this->makeModel(params, features, startTime);
    CEventRatePopulationModel* model =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    CCountingModel countingModel(params, m_Gatherer, m_InterimBucketCorrector);

    test::CRandomNumbers rng;
    core_t::TTime now{startTime};
    core_t::TTime endTime = now + 2 * 24 * bucketLength;
    TDoubleVec samples(3, 0.0);
    while (now < endTime) {
        rng.generateUniformSamples(50.0, 70.0, 3, samples);
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[0] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p1", TOptionalStr("a1")), m_Gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[1] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p2", TOptionalStr("a1")), m_Gatherer);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(samples[2] + 0.5); ++i) {
            this->addArrival(SMessage(now, "p3", TOptionalStr("a2")), m_Gatherer);
        }
        countingModel.sample(now, now + bucketLength, m_ResourceMonitor);
        model->sample(now, now + bucketLength, m_ResourceMonitor);
        now += bucketLength;
    }
    for (std::size_t i = 0; i < 35; ++i) {
        this->addArrival(SMessage(now, "p1", TOptionalStr("a1")), m_Gatherer);
    }
    for (std::size_t i = 0; i < 1; ++i) {
        this->addArrival(SMessage(now, "p2", TOptionalStr("a1")), m_Gatherer);
    }
    for (std::size_t i = 0; i < 100; ++i) {
        this->addArrival(SMessage(now, "p3", TOptionalStr("a2")), m_Gatherer);
    }
    countingModel.sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);
    model->sampleBucketStatistics(now, now + bucketLength, m_ResourceMonitor);

    CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    SAnnotatedProbability annotatedProbability1;
    annotatedProbability1.s_ResultType.set(model_t::CResultType::E_Interim);
    BOOST_TEST_REQUIRE(model->computeProbability(
        0 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability1));
    SAnnotatedProbability annotatedProbability2;
    annotatedProbability2.s_ResultType.set(model_t::CResultType::E_Interim);
    BOOST_TEST_REQUIRE(model->computeProbability(
        1 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability2));
    SAnnotatedProbability annotatedProbability3;
    annotatedProbability3.s_ResultType.set(model_t::CResultType::E_Interim);
    BOOST_TEST_REQUIRE(model->computeProbability(
        2 /*pid*/, now, now + bucketLength, partitioningFields, 1, annotatedProbability3));

    model_t::CResultType type(model_t::CResultType::E_Unconditional |
                              model_t::CResultType::E_Interim);
    TDouble1Vec p1a1Baseline = model->baselineBucketMean(
        model_t::E_PopulationCountByBucketPersonAndAttribute, 0, 0, type, NO_CORRELATES, now);
    TDouble1Vec p2a1Baseline = model->baselineBucketMean(
        model_t::E_PopulationCountByBucketPersonAndAttribute, 0, 0, type, NO_CORRELATES, now);
    TDouble1Vec p3a2Baseline = model->baselineBucketMean(
        model_t::E_PopulationCountByBucketPersonAndAttribute, 2, 1, type, NO_CORRELATES, now);

    LOG_DEBUG(<< "p1 probability = " << annotatedProbability1.s_Probability);
    LOG_DEBUG(<< "p2 probability = " << annotatedProbability2.s_Probability);
    LOG_DEBUG(<< "p3 probability = " << annotatedProbability3.s_Probability);
    LOG_DEBUG(<< "p1a1 baseline = " << p1a1Baseline[0]);
    LOG_DEBUG(<< "p2a1 baseline = " << p2a1Baseline[0]);
    LOG_DEBUG(<< "p3a2 baseline = " << p3a2Baseline[0]);

    BOOST_TEST_REQUIRE(annotatedProbability1.s_Probability > 0.05);
    BOOST_TEST_REQUIRE(annotatedProbability2.s_Probability < 0.05);
    BOOST_TEST_REQUIRE(annotatedProbability3.s_Probability < 0.05);
    BOOST_TEST_REQUIRE(p1a1Baseline[0] > 45.0);
    BOOST_TEST_REQUIRE(p1a1Baseline[0] < 46.0);
    BOOST_TEST_REQUIRE(p2a1Baseline[0] > 45.0);
    BOOST_TEST_REQUIRE(p2a1Baseline[0] < 46.0);
    BOOST_TEST_REQUIRE(p3a2Baseline[0] > 59.0);
    BOOST_TEST_REQUIRE(p3a2Baseline[0] < 61.0);
}

BOOST_FIXTURE_TEST_CASE(testPersistence, CTestFixture) {
    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};

    TMessageVec messages;
    generateTestMessages(startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute,
                                  model_t::E_PopulationUniquePersonCountByAttribute};
    this->makeModel(params, features, startTime);
    CEventRatePopulationModel* populationModel =
        dynamic_cast<CEventRatePopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(populationModel);

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            m_Model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);
            startTime += bucketLength;
        }
        this->addArrival(message, m_Gatherer);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        m_Model->acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "origXml = " << origXml);
    LOG_DEBUG(<< "origXml size = " << origXml.size());

    // Restore the XML into a new data gatherer
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CAnomalyDetectorModel::TModelPtr restoredModel(
        m_Factory->makeModel({m_Gatherer}, traverser));

    populationModel = dynamic_cast<CEventRatePopulationModel*>(restoredModel.get());
    BOOST_TEST_REQUIRE(populationModel);

    // The XML representation of the new data gatherer should be the same as the
    // original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredModel->acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }

    LOG_DEBUG(<< "original checksum = " << m_Model->checksum(false));
    LOG_DEBUG(<< "restored checksum = " << restoredModel->checksum(false));
    BOOST_REQUIRE_EQUAL(m_Model->checksum(false), restoredModel->checksum(false));
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testIgnoreSamplingGivenDetectionRules, CTestFixture) {
    // Create 2 models, one of which has a skip sampling rule.
    // Feed the same data into both models then add extra data
    // into the first model we know will be filtered out.
    // At the end the checksums for the underlying models should
    // be the same.

    core_t::TTime startTime{100};
    std::size_t bucketLength{100};

    // Create a categorical rule to filter out attribute a3
    std::string filterJson("[\"a3\"]");
    core::CPatternSet valueFilter;
    valueFilter.initFromJson(filterJson);

    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipModelUpdate);
    rule.includeScope("", valueFilter);

    SModelParams paramsNoRules(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CEventRatePopulationModelFactory factory(paramsNoRules, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationCountByBucketPersonAndAttribute};
    factory.features(features);

    CModelFactory::SGathererInitializationData gathererNoSkipInitData(startTime);
    CModelFactory::TDataGathererPtr gathererNoSkip(
        factory.makeDataGatherer(gathererNoSkipInitData));
    CModelFactory::SModelInitializationData modelNoSkipInitData(gathererNoSkip);
    CAnomalyDetectorModel::TModelPtr modelNoSkip(factory.makeModel(modelNoSkipInitData));

    SModelParams paramsWithRules(bucketLength);
    SModelParams::TDetectionRuleVec rules{rule};
    paramsWithRules.s_DetectionRules = SModelParams::TDetectionRuleVecCRef(rules);
    auto interimBucketCorrectorWithRules =
        std::make_shared<CInterimBucketCorrector>(bucketLength);

    CEventRatePopulationModelFactory factoryWithSkipRule(
        paramsWithRules, interimBucketCorrectorWithRules);
    factoryWithSkipRule.features(features);

    CModelFactory::SGathererInitializationData gathererWithSkipInitData(startTime);
    CModelFactory::TDataGathererPtr gathererWithSkip(
        factoryWithSkipRule.makeDataGatherer(gathererWithSkipInitData));
    CModelFactory::SModelInitializationData modelWithSkipInitData(gathererWithSkip);
    CAnomalyDetectorModel::TModelPtr modelWithSkip(
        factoryWithSkipRule.makeModel(modelWithSkipInitData));

    this->addArrival(SMessage(100, "p1", TOptionalStr("a1")), gathererNoSkip);
    this->addArrival(SMessage(100, "p1", TOptionalStr("a1")), gathererWithSkip);
    this->addArrival(SMessage(100, "p1", TOptionalStr("a2")), gathererNoSkip);
    this->addArrival(SMessage(100, "p1", TOptionalStr("a2")), gathererWithSkip);
    this->addArrival(SMessage(100, "p2", TOptionalStr("a1")), gathererNoSkip);
    this->addArrival(SMessage(100, "p2", TOptionalStr("a1")), gathererWithSkip);
    modelNoSkip->sample(100, 200, m_ResourceMonitor);
    modelWithSkip->sample(100, 200, m_ResourceMonitor);

    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    this->addArrival(SMessage(200, "p1", TOptionalStr("a1")), gathererNoSkip);
    this->addArrival(SMessage(200, "p1", TOptionalStr("a1")), gathererWithSkip);
    this->addArrival(SMessage(200, "p1", TOptionalStr("a2")), gathererNoSkip);
    this->addArrival(SMessage(200, "p1", TOptionalStr("a2")), gathererWithSkip);
    this->addArrival(SMessage(200, "p2", TOptionalStr("a1")), gathererNoSkip);
    this->addArrival(SMessage(200, "p2", TOptionalStr("a1")), gathererWithSkip);

    // This should be filtered out
    this->addArrival(SMessage(200, "p1", TOptionalStr("a3")), gathererWithSkip);
    this->addArrival(SMessage(200, "p2", TOptionalStr("a3")), gathererWithSkip);

    // Add another attribute that must not be skipped
    this->addArrival(SMessage(200, "p1", TOptionalStr("a4")), gathererNoSkip);
    this->addArrival(SMessage(200, "p1", TOptionalStr("a4")), gathererWithSkip);
    this->addArrival(SMessage(200, "p2", TOptionalStr("a4")), gathererNoSkip);
    this->addArrival(SMessage(200, "p2", TOptionalStr("a4")), gathererWithSkip);

    modelNoSkip->sample(200, 300, m_ResourceMonitor);
    modelWithSkip->sample(200, 300, m_ResourceMonitor);

    // Checksums will be different because a model is created for attribute a3
    BOOST_TEST_REQUIRE(modelWithSkip->checksum() != modelNoSkip->checksum());

    auto modelWithSkipView = modelWithSkip->details();
    auto modelNoSkipView = modelNoSkip->details();

    // but the underlying models for attributes a1 and a2 are the same
    std::uint64_t withSkipChecksum{
        modelWithSkipView
            ->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0)
            ->checksum()};
    std::uint64_t noSkipChecksum{
        modelNoSkipView
            ->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0)
            ->checksum()};
    BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    withSkipChecksum = modelWithSkipView
                           ->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1)
                           ->checksum();
    noSkipChecksum = modelNoSkipView
                         ->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1)
                         ->checksum();
    BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    // The no skip model didn't see the a3 attribute only a1, a2 and a4.
    // The a4 models should be the same.
    withSkipChecksum = modelWithSkipView
                           ->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 3)
                           ->checksum();
    noSkipChecksum = modelNoSkipView
                         ->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 2)
                         ->checksum();
    BOOST_REQUIRE_EQUAL(withSkipChecksum, noSkipChecksum);

    // Check the last value times of all the underlying models are the same
    const maths::CUnivariateTimeSeriesModel* timeSeriesModel{
        dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(modelNoSkipView->model(
            model_t::E_PopulationCountByBucketPersonAndAttribute, 0))};
    BOOST_TEST_REQUIRE(timeSeriesModel);
    const auto* trendModel = dynamic_cast<const maths::CTimeSeriesDecomposition*>(
        &timeSeriesModel->trendModel());
    BOOST_TEST_REQUIRE(trendModel);

    core_t::TTime time = trendModel->lastValueTime();
    BOOST_REQUIRE_EQUAL(model_t::sampleTime(model_t::E_PopulationCountByBucketPersonAndAttribute,
                                            200, bucketLength),
                        time);

    // The last times of the underlying time series models should all be the same
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1));
    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelNoSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 2));
    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());

    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 0));
    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 1));
    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 2));
    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());
    timeSeriesModel = dynamic_cast<const maths::CUnivariateTimeSeriesModel*>(
        modelWithSkipView->model(model_t::E_PopulationCountByBucketPersonAndAttribute, 3));
    BOOST_REQUIRE_EQUAL(time, trendModel->lastValueTime());
}

BOOST_AUTO_TEST_SUITE_END()
