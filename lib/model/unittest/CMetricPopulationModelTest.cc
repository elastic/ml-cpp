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
#include <maths/COrderings.h>
#include <maths/CSampling.h>

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

#include "CModelTestFixtureBase.h"

#include <boost/optional/optional_io.hpp>
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

using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u>;
using TMaxAccumulator =
    maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>>;
struct SValuesAndWeights {
    maths::CModel::TTimeDouble2VecSizeTrVec s_Values;
    maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec s_TrendWeights;
    maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec s_ResidualWeights;
};

const std::size_t numberAttributes{5};
const std::size_t numberPeople{10};
}

class CTestFixture : public CModelTestFixtureBase {
public:
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

        const std::size_t numberBuckets{100u};

        TStrVec people;
        for (std::size_t i = 0; i < numberPeople; ++i) {
            people.push_back("p" + core::CStringUtils::typeToString(i));
        }
        LOG_DEBUG(<< "people = " << core::CContainerPrinter::print(people));

        TStrVec attributes;
        for (std::size_t i = 0; i < numberAttributes; ++i) {
            attributes.push_back("c" + core::CStringUtils::typeToString(i));
        }
        LOG_DEBUG(<< "attributes = " << core::CContainerPrinter::print(attributes));

        const TDoubleVec attributeRates{10.0, 2.0, 15.0, 2.0, 1.0};
        const TDoubleVec means{5.0, 10.0, 7.0, 3.0, 15.0};
        const TDoubleVec variances{1.0, 0.5, 2.0, 0.1, 4.0};

        TSizeSizePrVecVec anomalies{{{40u, 6u}, {15u, 3u}, {12u, 2u}},
                                    {},
                                    {{44u, 9u}, {30u, 5u}},
                                    {{80u, 1u}, {12u, 2u}},
                                    {{60u, 2u}}};

        test::CRandomNumbers rng;

        for (std::size_t i = 0; i < numberBuckets; ++i, startTime += bucketLength) {
            for (std::size_t j = 0; j < numberAttributes; ++j) {
                TUIntVec samples;
                rng.generatePoissonSamples(attributeRates[j], numberPeople, samples);

                for (std::size_t k = 0; k < numberPeople; ++k) {
                    bool anomaly = !anomalies[j].empty() &&
                                   anomalies[j].back().first == i &&
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

                    for (std::size_t l = 0; l < values.size(); l += dimension) {
                        TDouble1Vec value(dimension);
                        for (std::size_t d = 0; d < dimension; ++d) {
                            double vd = values[l + d];
                            if (anomaly && (l % (2 * dimension)) == 0) {
                                vd += 6.0 * std::sqrt(variances[j]);
                            }
                            value[d] = this->roundToNearestPersisted(vd);
                        }
                        core_t::TTime dt = (static_cast<core_t::TTime>(l) * bucketLength) /
                                           static_cast<core_t::TTime>(values.size());
                        SMessage message(startTime + dt, people[k], attributes[j], value);
                        messages.push_back(message);
                    }
                }
            }
        }

        LOG_DEBUG(<< "# messages = " << messages.size());
        std::sort(messages.begin(), messages.end());
    }

    void makeModel(const SModelParams& params,
                   const model_t::TFeatureVec& features,
                   core_t::TTime startTime) {
        this->makeModelT<CMetricPopulationModelFactory>(
            params, features, startTime, model_t::E_MetricOnline, m_Gatherer, m_Model);
    }

private:
    double roundToNearestPersisted(double value) {
        std::string valueAsString{core::CStringUtils::typeToStringPrecise(
            value, core::CIEEE754::E_DoublePrecision)};
        double result{0.0};
        core::CStringUtils::stringToType(valueAsString, result);
        return result;
    }
};

BOOST_FIXTURE_TEST_CASE(testBasicAccessors, CTestFixture) {
    // Check the correct data is retrieved by the basic model accessors.

    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMinAccumulatorVec = std::vector<TMinAccumulator>;
    using TMaxAccumulatorVec = std::vector<TMaxAccumulator>;

    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};

    TMessageVec messages;
    generateTestMessages(1, startTime, bucketLength, messages);
    LOG_DEBUG(<< "# messages = " << messages.size());

    SModelParams params(bucketLength);

    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute,
                                  model_t::E_PopulationMinByPersonAndAttribute,
                                  model_t::E_PopulationMaxByPersonAndAttribute};

    this->makeModel(params, features, startTime);
    CMetricPopulationModel* model =
        dynamic_cast<CMetricPopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    TStrUInt64Map expectedBucketPersonCounts;
    TMeanAccumulatorVec expectedBucketMeans(numberPeople * numberAttributes);
    TMinAccumulatorVec expectedBucketMins(numberPeople * numberAttributes);
    TMaxAccumulatorVec expectedBucketMaxs(numberPeople * numberAttributes);

    for (const auto& message : messages) {
        if (message.s_Time >= startTime + bucketLength) {
            model->sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            LOG_DEBUG(<< "Testing bucket = [" << startTime << ","
                      << startTime + bucketLength << ")");

            BOOST_REQUIRE_EQUAL(numberPeople, m_Gatherer->numberActivePeople());
            BOOST_REQUIRE_EQUAL(numberAttributes, m_Gatherer->numberActiveAttributes());

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

            LOG_DEBUG(<< "expected counts = "
                      << core::CContainerPrinter::print(expectedBucketPersonCounts));

            TSizeVec expectedCurrentBucketPersonIds;

            // Test the person counts.
            for (const auto& count_ : expectedBucketPersonCounts) {
                std::size_t pid;
                BOOST_TEST_REQUIRE(m_Gatherer->personId(count_.first, pid));
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
            for (std::size_t cid = 0; cid < numberAttributes; ++cid) {
                for (std::size_t pid = 0; pid < numberPeople; ++pid) {
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

        CEventData eventData = this->addArrival(message, m_Gatherer);
        std::size_t pid = *eventData.personId();
        std::size_t cid = *eventData.attributeId();
        ++expectedBucketPersonCounts[message.s_Person];
        expectedBucketMeans[pid * numberAttributes + cid].add(message.s_Dbl1Vec.get()[0]);
        expectedBucketMins[pid * numberAttributes + cid].add(message.s_Dbl1Vec.get()[0]);
        expectedBucketMaxs[pid * numberAttributes + cid].add(message.s_Dbl1Vec.get()[0]);
    }
}

BOOST_FIXTURE_TEST_CASE(testMinMaxAndMean, CTestFixture) {
    // Check the correct data is read from the gatherer into the model on sample.

    using TSizeValueAndWeightsMap = std::map<std::size_t, SValuesAndWeights>;
    using TSizeSizeValueAndWeightsMapMap = std::map<std::size_t, TSizeValueAndWeightsMap>;
    using TSizeSizePrDoubleVecMap = std::map<TSizeSizePr, TDoubleVec>;
    using TSizeSizePrMeanAccumulatorUMap = std::map<TSizeSizePr, TMeanAccumulator>;
    using TSizeSizePrMinAccumulatorMap = std::map<TSizeSizePr, TMinAccumulator>;
    using TSizeSizePrMaxAccumulatorMap = std::map<TSizeSizePr, TMaxAccumulator>;
    using TMathsModelPtr = std::shared_ptr<maths::CModel>;
    using TSizeMathsModelPtrMap = std::map<std::size_t, TMathsModelPtr>;

    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};

    TMessageVec messages;
    generateTestMessages(1, startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_InitialDecayRateMultiplier = 1.0;
    params.s_MaximumUpdatesPerBucket = 0.0;

    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute,
                                  model_t::E_PopulationMinByPersonAndAttribute,
                                  model_t::E_PopulationMaxByPersonAndAttribute};

    this->makeModel(params, features, startTime);
    auto* model = dynamic_cast<CMetricPopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    CModelFactory::TFeatureMathsModelPtrPrVec models{
        m_Factory->defaultFeatureModels(features, bucketLength, 1.0, false)};
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

            TSizeSizeValueAndWeightsMapMap populationWeightedSamples;
            for (std::size_t feature = 0; feature < features.size(); ++feature) {
                for (const auto& samples_ : expectedSamples[feature]) {
                    std::size_t pid = samples_.first.first;
                    std::size_t cid = samples_.first.second;
                    auto& attribute = populationWeightedSamples[feature][cid];
                    TMathsModelPtr& attributeModel = expectedPopulationModels[feature][cid];
                    if (attributeModel == nullptr) {
                        attributeModel = m_Factory->defaultFeatureModel(
                            features[feature], bucketLength, 1.0, false);
                    }
                    for (std::size_t j = 0; j < samples_.second.size(); ++j) {
                        // We round to the nearest integer time (note this has to
                        // match the behaviour of CMetricPartialStatistic::time).
                        core_t::TTime time_ = static_cast<core_t::TTime>(
                            expectedSampleTimes[{pid, cid}][j] + 0.5);
                        TDouble2Vec sample{samples_.second[j]};
                        attribute.s_Values.emplace_back(time_, sample, pid);
                        attribute.s_TrendWeights.push_back(
                            maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                        attribute.s_ResidualWeights.push_back(
                            maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                        double countWeight{model->sampleRateWeight(pid, cid)};
                        attributeModel->countWeights(
                            time_, sample, countWeight, countWeight, 1.0, 1.0,
                            attribute.s_TrendWeights.back(),
                            attribute.s_ResidualWeights.back());
                    }
                }
            }

            for (auto& feature : populationWeightedSamples) {
                for (auto& attribute : feature.second) {
                    maths::COrderings::simultaneousSort(
                        attribute.second.s_Values, attribute.second.s_TrendWeights,
                        attribute.second.s_ResidualWeights);
                    maths::CModelAddSamplesParams params_;
                    params_.integer(false)
                        .nonNegative(nonNegative)
                        .propagationInterval(1.0)
                        .trendWeights(attribute.second.s_TrendWeights)
                        .priorWeights(attribute.second.s_ResidualWeights);
                    expectedPopulationModels[feature.first][attribute.first]->addSamples(
                        params_, attribute.second.s_Values);
                }
            }

            for (std::size_t feature = 0; feature < features.size(); ++feature) {
                if ((startTime / bucketLength) % 10 == 0) {
                    LOG_DEBUG(<< "Testing priors for feature "
                              << model_t::print(features[feature]));
                }
                for (std::size_t cid = 0; cid < numberAttributes; ++cid) {
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

        CEventData eventData = this->addArrival(message, m_Gatherer);
        std::size_t pid = *eventData.personId();
        std::size_t cid = *eventData.attributeId();
        nonNegative &= message.s_Dbl1Vec.get()[0] < 0.0;

        double sampleCount = m_Gatherer->sampleCount(cid);
        if (sampleCount > 0.0) {
            TSizeSizePr key{pid, cid};
            sampleTimes[key].add(static_cast<double>(message.s_Time));
            sampleMeans[key].add(message.s_Dbl1Vec.get()[0]);
            sampleMins[key].add(message.s_Dbl1Vec.get()[0]);
            sampleMaxs[key].add(message.s_Dbl1Vec.get()[0]);
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
    core_t::TTime startTime{3600};
    core_t::TTime bucketLength{3600};
    SModelParams params(bucketLength);

    model_t::TFeatureVec features{model_t::E_PopulationVarianceByPersonAndAttribute};

    m_InterimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    m_Factory.reset(new CMetricPopulationModelFactory(params, m_InterimBucketCorrector));
    m_Factory->features({model_t::E_PopulationVarianceByPersonAndAttribute});
    m_Factory->fieldNames("", "P", "", "V", TStrVec{1, "I"});

    this->makeModel(params, features, startTime);

    CMetricPopulationModel& model =
        static_cast<CMetricPopulationModel&>(*m_Model.get());

    TDoubleStrPrVec b1{{1.0, "i1"}, {1.1, "i1"}, {1.01, "i2"}, {1.02, "i2"}};
    TDoubleStrPrVec b2{{10.0, "i1"}};
    TDoubleStrPrVec b3{{4.3, "i1"}, {4.4, "i1"}, {4.6, "i1"}, {4.2, "i1"}, {4.8, "i3"}};
    TDoubleStrPrVec b4{{3.2, "i3"}, {3.3, "i3"}};
    TDoubleStrPrVec b5{{20.1, "i2"}, {20.8, "i1"}, {20.9, "i1"}};
    TDoubleStrPrVec b6{{4.1, "i1"}, {4.2, "i2"}, {3.9, "i2"}, {4.2, "i2"}};
    TDoubleStrPrVec b7{{0.1, "i1"}, {0.3, "i1"}, {0.2, "i3"}};
    TDoubleStrPrVec b8{{12.5, "i1"}, {12.3, "i2"}};
    TDoubleStrPrVec b9{{6.9, "i1"}, {7.0, "i2"}, {7.1, "i3"},
                       {6.6, "i4"}, {7.1, "i5"}, {6.7, "i6"}};
    // This last bucket is much more improbable, with influencer i2 being responsible
    TDoubleStrPrVec b10{{0.3, "i2"},     {15.4, "i2"}, {77.62, "i2"},
                        {112.999, "i2"}, {5.1, "i1"},  {5.1, "i1"},
                        {5.1, "i1"},     {5.1, "i1"},  {5.1, "i1"}};

    SAnnotatedProbability annotatedProbability;

    core_t::TTime time = startTime;
    processBucket(time, bucketLength, b1, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b2, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b3, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b4, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b5, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b6, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b7, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b8, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability > 0.8);

    time += bucketLength;
    processBucket(time, bucketLength, b9, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability < 0.85);

    time += bucketLength;
    processBucket(time, bucketLength, b10, m_Gatherer, model, annotatedProbability);
    BOOST_TEST_REQUIRE(annotatedProbability.s_Probability < 0.1);
    BOOST_REQUIRE_EQUAL(1, annotatedProbability.s_Influences.size());
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

    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};

    model_t::TFeatureVec features{model_t::E_PopulationMaxByPersonAndAttribute,
                                  model_t::E_PopulationMeanLatLongByPersonAndAttribute};

    for (auto& feature : features) {
        LOG_DEBUG(<< "Testing " << model_t::print(feature));

        TMessageVec messages;
        generateTestMessages(model_t::dimension(feature), startTime, bucketLength, messages);

        SModelParams params(bucketLength);
        this->makeModel(params, {feature}, startTime);
        CMetricPopulationModel* model =
            static_cast<CMetricPopulationModel*>(m_Model.get());
        BOOST_TEST_REQUIRE(model);

        TStrVec expectedAnomalies{
            "[12, p2, c0 c3]", "[15, p3, c0]", "[30, p5, c2]", "[40, p6, c0]",
            "[44, p9, c2]",    "[60, p2, c4]", "[80, p1, c3]"};

        TAnomalyVec orderedAnomalies;

        this->generateOrderedAnomalies(7u, startTime, bucketLength, messages,
                                       m_Gatherer, *model, orderedAnomalies);

        BOOST_REQUIRE_EQUAL(expectedAnomalies.size(), orderedAnomalies.size());
        for (std::size_t j = 0; j < orderedAnomalies.size(); ++j) {
            BOOST_REQUIRE_EQUAL(expectedAnomalies[j], orderedAnomalies[j].print());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testPrune, CTestFixture) {
    // This test has four people and five attributes. We expect
    // person 2 and attributes 1, 2 and 5 to be deleted.

    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};
    const std::size_t numberBuckets{1000u};

    TStrVec people{"p1", "p2", "p3", "p4"};
    TStrVec attributes{"c1", "c2", "c3", "c4", "c5"};
    TStrSizePrVecVecVec eventCounts{{}, {}, {}, {}};

    {
        TStrSizePrVec attributeCounts{{attributes[0], 0}, {attributes[4], 0}};
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
        TStrSizePrVec attributeCounts{
            {attributes[0], 0}, {attributes[1], 0}, {attributes[2], 1}, {attributes[4], 0}};
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
        TStrSizePrVec attributeCounts{{attributes[2], 0}, {attributes[3], 2}};
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
        TStrSizePrVec attributeCounts{{attributes[1], 0}, {attributes[3], 3}};
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
    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute,
                                  model_t::E_PopulationMinByPersonAndAttribute,
                                  model_t::E_PopulationMaxByPersonAndAttribute};

    this->makeModel(params, features, startTime);

    CModelFactory::TDataGathererPtr expectedGatherer(m_Factory->makeDataGatherer({startTime}));
    CModelFactory::SModelInitializationData expectedModelInitData(expectedGatherer);
    CAnomalyDetectorModel::TModelPtr expectedModel(m_Factory->makeModel(expectedModelInitData));
    BOOST_TEST_REQUIRE(expectedModel);

    test::CRandomNumbers rng;

    TMessageVec messages;
    for (std::size_t i = 0; i < people.size(); ++i) {
        core_t::TTime bucketStart{startTime};
        for (std::size_t j = 0; j < numberBuckets; ++j, bucketStart += bucketLength) {
            const TStrSizePrVec& attributeEventCounts = eventCounts[i][j];
            for (auto& attributeEventCount : attributeEventCounts) {
                if (attributeEventCount.second == 0) {
                    continue;
                }

                std::size_t n{attributeEventCount.second};

                TDoubleVec samples;
                rng.generateUniformSamples(0.0, 8.0, n, samples);

                core_t::TTime time{bucketStart};
                core_t::TTime dt{bucketLength / static_cast<core_t::TTime>(n)};

                for (std::size_t l = 0; l < n; ++l, time += dt) {
                    messages.emplace_back(time, people[i], attributeEventCount.first,
                                          TDouble1Vec{1, samples[l]});
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
            std::binary_search(std::begin(expectedAttributes),
                               std::end(expectedAttributes), message.s_Attribute)) {
            expectedMessages.push_back(message);
        }
    }

    core_t::TTime bucketStart{startTime};
    for (const auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            m_Model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
            bucketStart += bucketLength;
        }
        this->addArrival(message, m_Gatherer);
    }
    m_Model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    size_t maxDimensionBeforePrune(m_Model->dataGatherer().maxDimension());
    m_Model->prune();
    size_t maxDimensionAfterPrune(m_Model->dataGatherer().maxDimension());
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

    LOG_DEBUG(<< "checksum          = " << m_Model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), m_Model->checksum());

    // Now check that we recycle the person and attribute slots.

    bucketStart = m_Gatherer->currentBucketStartTime() + bucketLength;

    TMessageVec newMessages{
        {bucketStart + 10, "p1", TOptionalStr{"c2"}, TDouble1Vec(1, 20.0)},
        {bucketStart + 200, "p5", TOptionalStr{"c6"}, TDouble1Vec(1, 10.0)},
        {bucketStart + 2100, "p5", TOptionalStr{"c6"}, TDouble1Vec(1, 15.0)}};

    for (auto& newMessage : newMessages) {
        this->addArrival(newMessage, m_Gatherer);
        this->addArrival(newMessage, expectedGatherer);
    }
    m_Model->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);
    expectedModel->sample(bucketStart, bucketStart + bucketLength, m_ResourceMonitor);

    LOG_DEBUG(<< "checksum          = " << m_Model->checksum());
    LOG_DEBUG(<< "expected checksum = " << expectedModel->checksum());
    BOOST_REQUIRE_EQUAL(expectedModel->checksum(), m_Model->checksum());

    // Test that calling prune on a cloned model which has seen no new data does nothing
    CAnomalyDetectorModel::TModelPtr clonedModelHolder(m_Model->cloneForPersistence());
    std::size_t numberOfPeopleBeforePrune(
        clonedModelHolder->dataGatherer().numberActivePeople());
    BOOST_TEST_REQUIRE(numberOfPeopleBeforePrune > 0);
    clonedModelHolder->prune(clonedModelHolder->defaultPruneWindow());
    BOOST_REQUIRE_EQUAL(numberOfPeopleBeforePrune,
                        clonedModelHolder->dataGatherer().numberActivePeople());
}

BOOST_FIXTURE_TEST_CASE(testKey, CTestFixture) {
    function_t::TFunctionVec countFunctions{
        function_t::E_PopulationMetric, function_t::E_PopulationMetricMean,
        function_t::E_PopulationMetricMin, function_t::E_PopulationMetricMax,
        function_t::E_PopulationMetricSum};

    std::string fieldName{"value"};
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
                    messages.emplace_back(bucketStart + bucketLength / 2, datum.s_Person,
                                          data[j].s_Attribute, TDouble1Vec{1, 0.0});
                }
            }
            ++i;
        }
    }

    std::sort(messages.begin(), messages.end());
    LOG_DEBUG(<< "# messages = " << messages.size());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    this->makeModel(params, {model_t::E_PopulationMeanByPersonAndAttribute}, startTime);

    core_t::TTime time{startTime};
    for (const auto& message : messages) {
        if (message.s_Time >= time + bucketLength) {
            m_Model->sample(time, time + bucketLength, m_ResourceMonitor);
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
            LOG_DEBUG(<< "frequency = " << m_Model->personFrequency(pid));
            LOG_DEBUG(<< "expected frequency = "
                      << 1.0 / static_cast<double>(datum.s_Period));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0 / static_cast<double>(datum.s_Period),
                                         m_Model->personFrequency(pid),
                                         0.1 / static_cast<double>(datum.s_Period));
            meanError.add(std::fabs(m_Model->personFrequency(pid) -
                                    1.0 / static_cast<double>(datum.s_Period)));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(meanError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.002);
    }
    {
        std::size_t i{0};
        for (auto& datum : data) {
            LOG_DEBUG(<< "*** attributes = " << datum.s_Attribute << " ***");
            std::size_t cid;
            BOOST_TEST_REQUIRE(m_Gatherer->attributeId(datum.s_Attribute, cid));
            LOG_DEBUG(<< "frequency = " << m_Model->attributeFrequency(cid));
            LOG_DEBUG(<< "expected frequency = " << (10.0 - static_cast<double>(i)) / 10.0);
            BOOST_REQUIRE_EQUAL((10.0 - static_cast<double>(i)) / 10.0,
                                m_Model->attributeFrequency(cid));
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

    const core_t::TTime bucketLength{600};
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
                                      people[heavyHitter], attribute,
                                      TDouble1Vec{1, 0.0});
            }
        }

        TSizeVec attributeIndexes;
        rng.generateUniformSamples(0, attributes.size(), normal.size(), attributeIndexes);
        std::size_t i{0};
        for (auto& norm : normal) {
            messages.emplace_back(static_cast<core_t::TTime>(times[m++]),
                                  people[norm], attributes[attributeIndexes[i++]],
                                  TDouble1Vec{1, 0.0});
        }
    }

    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    this->makeModel(params, {model_t::E_PopulationSumByBucketPersonAndAttribute}, startTime);

    CMetricPopulationModel* populationModel =
        dynamic_cast<CMetricPopulationModel*>(m_Model.get());
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
    TDoubleVec baseline{1, 1, 2, 2,  3, 5, 6, 6, 20, 21, 4, 3,
                        4, 4, 8, 25, 7, 6, 5, 1, 1,  4,  1, 1};

    TDoubleStrPrVec attribs{{2.0, "a1"}, {3.0, "a2"}};

    const TStrVec people{"p1", "p2", "p3", "p4", "p5",
                         "p6", "p7", "p8", "p9", "p10"};

    test::CRandomNumbers rng;

    core_t::TTime startTime{0};
    core_t::TTime endTime{604800};

    TMessageVec messages;
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        for (const auto& attrib : attribs) {
            TDoubleVec values;
            rng.generateNormalSamples(baseline[(time % DAY) / HOUR],
                                      attrib.first * attrib.first, people.size(), values);

            std::size_t j{0};
            for (const auto& value : values) {
                for (unsigned int t = 0; t < 4; ++t) {
                    messages.emplace_back(time + (t * bucketLength) / 4, people[j],
                                          attrib.second, TDouble1Vec(1, value));
                }
                ++j;
            }
        }
    }
    std::sort(messages.begin(), messages.end());

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    this->makeModel(params, {model_t::E_PopulationMeanByPersonAndAttribute}, startTime);

    TStrDoubleMap personProbabilitiesWithoutPeriodicity;
    TStrDoubleMap personProbabilitiesWithPeriodicity;

    core_t::TTime time = startTime;
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
                    double& minimumProbability = personProbabilitiesWithoutPeriodicity
                                                     .insert({person, 1.0})
                                                     .first->second;
                    minimumProbability = std::min(
                        minimumProbability, annotatedProbability.s_Probability);
                } else if (time > startTime + 5 * DAY) {
                    double& minimumProbability =
                        personProbabilitiesWithPeriodicity.insert({person, 1.0})
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

BOOST_FIXTURE_TEST_CASE(testPersistence, CTestFixture) {
    core_t::TTime startTime{1367280000};
    const core_t::TTime bucketLength{3600};

    TMessageVec messages;
    generateTestMessages(1, startTime, bucketLength, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute,
                                  model_t::E_PopulationMinByPersonAndAttribute,
                                  model_t::E_PopulationMaxByPersonAndAttribute};

    this->makeModel(params, features, startTime);

    CMetricPopulationModel* populationModel =
        dynamic_cast<CMetricPopulationModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(populationModel);

    for (auto& message : messages) {
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

    // Restore the XML into a new data gatherer
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CAnomalyDetectorModel::TModelPtr restoredModel(
        m_Factory->makeModel({m_Gatherer}, traverser));

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
    const std::size_t bucketLength{100};
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

    TMessageVec messages{
        {startTime + 10, "p1", TOptionalStr{"c1"}, TDouble1Vec{1, 20.0}},
        {startTime + 10, "p1", TOptionalStr{"c2"}, TDouble1Vec{1, 22.0}},
        {startTime + 10, "p2", TOptionalStr{"c1"}, TDouble1Vec{1, 20.0}},
        {startTime + 10, "p2", TOptionalStr{"c2"}, TDouble1Vec{1, 22.0}}};

    std::vector<CModelFactory::TDataGathererPtr> gatherers{gathererNoSkip, gathererWithSkip};
    for (auto& gatherer : gatherers) {
        for (auto& message : messages) {
            this->addArrival(message, gatherer);
        }
    }
    modelNoSkip->sample(startTime, endTime, m_ResourceMonitor);
    modelWithSkip->sample(startTime, endTime, m_ResourceMonitor);
    startTime = endTime;
    endTime += bucketLength;
    BOOST_REQUIRE_EQUAL(modelWithSkip->checksum(), modelNoSkip->checksum());

    messages.clear();
    messages.emplace_back(startTime + 10, "p1", TOptionalStr{"c1"}, TDouble1Vec{1, 21.0});
    messages.emplace_back(startTime + 10, "p1", TOptionalStr{"c2"}, TDouble1Vec{1, 21.0});
    messages.emplace_back(startTime + 10, "p2", TOptionalStr{"c1"}, TDouble1Vec{1, 21.0});
    messages.emplace_back(startTime + 10, "p2", TOptionalStr{"c2"}, TDouble1Vec{1, 21.0});
    for (auto& gatherer : gatherers) {
        for (auto& message : messages) {
            this->addArrival(message, gatherer);
        }
    }

    // This should be filtered out
    this->addArrival(SMessage(startTime + 10, "p1", TOptionalStr{"c3"}, TDouble1Vec{1, 21.0}),
                     gathererWithSkip);
    this->addArrival(SMessage(startTime + 10, "p2", TOptionalStr{"c3"}, TDouble1Vec{1, 21.0}),
                     gathererWithSkip);

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
