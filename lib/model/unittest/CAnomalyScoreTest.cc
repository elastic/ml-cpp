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

#include "CAnomalyScoreTest.h"

#include <core/CLogger.h>
#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CAnomalyScore.h>

#include <test/CRandomNumbers.h>

#include <boost/bind.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <algorithm>
#include <sstream>
#include <stdio.h>

using namespace ml;

namespace
{
typedef std::vector<double> TDoubleVec;
typedef std::vector<std::size_t> TSizeVec;
}

void CAnomalyScoreTest::testComputeScores(void)
{
    typedef model::CAnomalyScore TScores;
    typedef maths::CJointProbabilityOfLessLikelySamples TJointProbabilityCalculator;
    typedef maths::CLogProbabilityOfMFromNExtremeSamples TLogExtremeProbabilityCalculator;

    const double jointProbabilityWeight = 0.5;
    const double extremeProbabilityWeight = 0.5;
    const std::size_t minExtremeSamples = 1u;
    const std::size_t maxExtremeSamples = 5u;
    const double maximumAnomalousProbability = 0.05;
    double overallScore = 0.0;
    double overallProbability = 1.0;

    // Test 1: single value.
    // Expect deviation of the probability.
    {
        TDoubleVec p1(1u, 0.001);
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         p1,
                         overallScore,
                         overallProbability);
        LOG_DEBUG("1) score 1 = " << overallScore);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.078557, overallScore, 5e-7);

        TDoubleVec p2(1u, 0.02);
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         p2,
                         overallScore,
                         overallProbability);
        LOG_DEBUG("1) score 2 = " << overallScore);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002405, overallScore, 5e-7);

        TDoubleVec p3(1u, 0.1);
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         p3,
                         overallScore,
                         overallProbability);
        LOG_DEBUG("1) score 3 = " << overallScore);
        CPPUNIT_ASSERT_EQUAL(0.0, overallScore);
    }

    // Test 2: low anomalousness.
    // Expect scores of zero.
    {
        double p[] =
            {
                0.21, 0.52, 0.13, 0.67, 0.89, 0.32, 0.46, 0.222, 0.35, 0.93
            };
        TDoubleVec probabilities(boost::begin(p), boost::end(p));
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         probabilities,
                         overallScore,
                         overallProbability);
        LOG_DEBUG("2) score = " << overallScore);
        CPPUNIT_ASSERT_EQUAL(0.0, overallScore);
    }

    // Test 3: a lot of slightly anomalous values.
    // Expect a high anomaly score which is generated by the
    // joint probability of less likely samples.
    {
        double p[] =
            {
                0.11, 0.13, 0.12, 0.22, 0.14, 0.09, 0.01, 0.13, 0.15, 0.14, 0.11, 0.13, 0.12, 0.22, 0.09, 0.01
            };

        TDoubleVec probabilities(boost::begin(p), boost::end(p));
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         probabilities,
                         overallScore,
                         overallProbability);

        TJointProbabilityCalculator jointProbabilityCalculator;
        TLogExtremeProbabilityCalculator extremeProbabilityCalculator(2);
        for (size_t i = 0; i < boost::size(p); ++i)
        {
            jointProbabilityCalculator.add(p[i]);
            extremeProbabilityCalculator.add(p[i]);
        }

        double jointProbability;
        CPPUNIT_ASSERT(jointProbabilityCalculator.calculate(jointProbability));
        double extremeProbability;
        CPPUNIT_ASSERT(extremeProbabilityCalculator.calculate(extremeProbability));
        extremeProbability = ::exp(extremeProbability);

        LOG_DEBUG("3) probabilities = " << core::CContainerPrinter::print(p));
        LOG_DEBUG("   joint probability = " << jointProbability
                  << ", extreme probability = " << extremeProbability
                  << ", overallScore = " << overallScore);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.318231, overallScore, 5e-7);
    }

    // Test 4: one very anomalous value.
    // Expect a high anomaly score which is generated by the
    // extreme samples probability.
    {
        double p[] =
            {
                0.21, 0.52, 0.13, 0.67, 0.89, 0.32, 0.46, 0.222, 0.35, 0.93, 0.89, 0.32, 0.46, 0.000021
            };

        TDoubleVec probabilities(boost::begin(p), boost::end(p));
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         probabilities,
                         overallScore,
                         overallProbability);

        TJointProbabilityCalculator jointProbabilityCalculator;
        TLogExtremeProbabilityCalculator extremeProbabilityCalculator(1);
        for (size_t i = 0; i < boost::size(p); ++i)
        {
            jointProbabilityCalculator.add(p[i]);
            extremeProbabilityCalculator.add(p[i]);
        }

        double jointProbability;
        CPPUNIT_ASSERT(jointProbabilityCalculator.calculate(jointProbability));

        double extremeProbability;
        CPPUNIT_ASSERT(extremeProbabilityCalculator.calculate(extremeProbability));
        extremeProbability = ::exp(extremeProbability);

        LOG_DEBUG("4) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG("   joint probability = " << jointProbability
                  << ", extreme probability = " << extremeProbability
                  << ", overallScore = " << overallScore);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.137591, overallScore, 5e-7);
    }

    // Test 5: a small number of "mid" anomalous values.
    // Expect a high anomaly score which is generated by the
    // extreme samples probability.
    {
        double p[] =
            {
                0.21, 0.52, 0.0058, 0.13, 0.67, 0.89, 0.32, 0.03, 0.46, 0.222, 0.35, 0.93, 0.01, 0.89, 0.32, 0.46, 0.0021
            };

        TDoubleVec probabilities(boost::begin(p), boost::end(p));
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         probabilities,
                         overallScore,
                         overallProbability);

        TJointProbabilityCalculator jointProbabilityCalculator;
        TLogExtremeProbabilityCalculator extremeProbabilityCalculator(4);
        for (size_t i = 0; i < boost::size(probabilities); ++i)
        {
            jointProbabilityCalculator.add(p[i]);
            extremeProbabilityCalculator.add(p[i]);
        }

        double jointProbability;
        CPPUNIT_ASSERT(jointProbabilityCalculator.calculate(jointProbability));

        double extremeProbability;
        CPPUNIT_ASSERT(extremeProbabilityCalculator.calculate(extremeProbability));
        extremeProbability = ::exp(extremeProbability);

        LOG_DEBUG("5) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG("   joint probability = " << jointProbability
                  << ", extreme probability = " << extremeProbability
                  << ", overallScore = " << overallScore);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.029413, overallScore, 5e-7);
    }

    {
        // Test underflow.

        double p[] =
            {
               1e-100, 1.7e-20, 1.6e-150, 2.2e-150, 1.3e-180, 1.35e-95, 1.7e-180, 1.21e-300
            };

        TDoubleVec probabilities(boost::begin(p), boost::end(p));
        TScores::compute(jointProbabilityWeight,
                         extremeProbabilityWeight,
                         minExtremeSamples,
                         maxExtremeSamples,
                         maximumAnomalousProbability,
                         probabilities,
                         overallScore,
                         overallProbability);

        LOG_DEBUG("6) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG("   overallScore = " << overallScore);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(86.995953, overallScore, 5e-7);
    }
}

void CAnomalyScoreTest::testNormalizeScoresQuantiles(void)
{
    typedef std::multiset<double> TDoubleMSet;
    typedef TDoubleMSet::iterator TDoubleMSetItr;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 2.0, 20000, samples);
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        if (samples[i] < 0.5)
        {
            samples[i] = 0.0;
        }
    }

    model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);

    double totalError = 0.0;
    double numberSamples = 0.0;

    TDoubleMSet scores;
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        scores.insert(samples[i]);
        TDoubleVec sample(1u, samples[i]);
        normalizer.updateQuantiles(sample);

        TDoubleMSetItr itr = scores.upper_bound(samples[i]);
        double trueQuantile = static_cast<double>(std::distance(scores.begin(), itr))
                              / static_cast<double>(scores.size());

        if (trueQuantile > 0.9)
        {
            double lowerBound;
            double upperBound;
            normalizer.quantile(samples[i], 0.0, lowerBound, upperBound);

            double quantile = (lowerBound + upperBound) / 2.0;
            double error = ::fabs(quantile - trueQuantile);

            totalError += error;
            numberSamples += 1.0;

            LOG_DEBUG("trueQuantile = " << trueQuantile
                      << ", lowerBound = " << lowerBound
                      << ", upperBound = " << upperBound);
            CPPUNIT_ASSERT(error < 0.02);
        }
    }

    LOG_DEBUG("meanError = " << totalError / numberSamples);
    CPPUNIT_ASSERT(totalError / numberSamples < 0.0043);
}

void CAnomalyScoreTest::testNormalizeScoresNoisy(void)
{
    typedef std::multimap<double, std::size_t> TDoubleSizeMap;
    typedef TDoubleSizeMap::const_iterator TDoubleSizeMapCItr;

    test::CRandomNumbers rng;

    // Exponential, i.e. shape = 1
    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 2.0, 2000, samples);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        if (samples[i] < 0.5)
        {
            samples[i] = 0.0;
        }
    }

    std::size_t largeAnomalyTimes[] =
        {
            50, 110, 190, 220, 290, 310, 600, 620, 790, 900, 1100, 1400, 1600, 1900
        };

    double largeAnomalies[] =
        {
            50.0, 350.0, 30.0, 100.0, 30.0, 45.0, 100.0, 120.0, 60.0, 130.0, 100.0, 90.0, 45.0, 30.0
        };

    // Add in the big anomalies.
    for (size_t i = 0; i < boost::size(largeAnomalyTimes); ++i)
    {
        samples[largeAnomalyTimes[i]] += largeAnomalies[i];
    }

    model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);

    //std::ostringstream raw;
    //std::ostringstream normalized;
    //raw << "raw = [";
    //normalized << "normalized = [";

    TDoubleSizeMap maxScores;

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        normalizer.updateQuantiles(samples[i]);
    }
    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        double sample = samples[i];
        normalizer.normalize(sample);
        LOG_DEBUG(i << ") raw = " << samples[i] << ", normalized = " << sample);

        //raw << " " << samples[i];
        //normalized << " " << sample[0];

        if (maxScores.size() < boost::size(largeAnomalyTimes))
        {
            maxScores.insert(TDoubleSizeMap::value_type(sample, i));
        }
        else if (sample > maxScores.begin()->first)
        {
            LOG_DEBUG("normalized = " << sample << " removing " << maxScores.begin()->first);
            maxScores.erase(maxScores.begin());
            maxScores.insert(TDoubleSizeMap::value_type(sample, i));
        }
    }

    //raw << " ];";
    //normalized << " ];";
    //std::ofstream f;
    //f.open("scores.m");
    //f << raw.str() << "\n" << normalized.str();

    LOG_DEBUG("maxScores = " << core::CContainerPrinter::print(maxScores));

    TSizeVec times;

    for (TDoubleSizeMapCItr itr = maxScores.begin();
         itr != maxScores.end();
         ++itr)
    {
        times.push_back(itr->second);
    }
    std::sort(times.begin(), times.end());

    LOG_DEBUG("times = " << core::CContainerPrinter::print(times));

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(largeAnomalyTimes),
                         core::CContainerPrinter::print(times));
}

void CAnomalyScoreTest::testNormalizeScoresLargeScore(void)
{
    // Test a large score isn't too dominant.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 0.1, 500, samples);

    std::size_t anomalyTimes[] = { 50, 110, 190, 220, 290 };
    double anomalies[] = { 2.0, 4.0, 4.0, 5.0, 20.0 };

    // Add in the anomalies.
    for (size_t i = 0; i < boost::size(anomalyTimes); ++i)
    {
        samples[anomalyTimes[i]] += anomalies[i];
    }

    model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);

    for (std::size_t i = 0u; i < samples.size(); ++i)
    {
        normalizer.updateQuantiles(samples[i]);
    }

    TDoubleVec scores;
    for (std::size_t i = 0u; i < boost::size(anomalyTimes); ++i)
    {
        double sample = samples[anomalyTimes[i]];
        normalizer.normalize(sample);
        scores.push_back(sample);
    }
    std::sort(scores.begin(), scores.end());
    LOG_DEBUG("scores = " << core::CContainerPrinter::print(scores));

    for (std::size_t i = 0u; i+1 < boost::size(anomalies); ++i)
    {
        double uplift = scores[i] - 100.0 * anomalies[i]/anomalies[boost::size(anomalies) - 1];
        LOG_DEBUG("uplift = " << uplift);
        CPPUNIT_ASSERT(uplift > 5.0);
        CPPUNIT_ASSERT(uplift < 13.0);
    }
}

void CAnomalyScoreTest::testNormalizeScoresNearZero(void)
{
    // Test the behaviour for scores near zero.

    std::size_t anomalyTimes[] = { 50, 110, 190, 220, 290 };
    double anomalies[] = { 0.02, 0.01, 0.006, 0.01, 0.015 };

    std::size_t nonZeroCounts[] = { 0, 100, 200, 249, 251, 300, 400, 450 };

    std::string expectedScores[] =
        {
            std::string("[41.62776, 32.36435, 26.16873, 32.36435, 37.68726]"),
            std::string("[41.62776, 32.36435, 26.16873, 32.36435, 37.68726]"),
            std::string("[41.62776, 32.36435, 17.74216, 32.36435, 37.68726]"),
            std::string("[41.62776, 32.36435, 11.1645, 32.36435, 37.68726]"),
            std::string("[41.62776, 32.36435, 11.05937, 32.36435, 37.68726]"),
            std::string("[41.62776, 32.36435, 8.523397, 32.36435, 37.68726]"),
            std::string("[1.14, 1.04, 1, 1.04, 1.09]"),
            std::string("[1.14, 1.04, 1, 1.04, 1.09]")
        };

    for (std::size_t i = 0u; i < boost::size(nonZeroCounts); ++i)
    {
        LOG_DEBUG("non-zero count = " << nonZeroCounts[i]);

        TDoubleVec samples(500u, 0.0);
        for (std::size_t j = 0u; j < nonZeroCounts[i]; ++j)
        {
            if (std::find(boost::begin(anomalyTimes),
                          boost::end(anomalyTimes), j) == boost::end(anomalyTimes))
            {
                samples[j] += 0.0055;
            }
        }
        for (std::size_t j = 0u; j < boost::size(anomalyTimes); ++j)
        {
            samples[anomalyTimes[j]] += anomalies[j];
        }

        model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig(1800);
        model::CAnomalyScore::CNormalizer normalizer(config);

        for (std::size_t j = 0u; j < samples.size(); ++j)
        {
            normalizer.updateQuantiles(samples[j]);
        }

        TDoubleVec maxScores;
        for (std::size_t j = 0u; j < boost::size(anomalyTimes); ++j)
        {
            double sample = samples[anomalyTimes[j]];
            normalizer.normalize(sample);
            maxScores.push_back(sample);
        }
        LOG_DEBUG("maxScores = " << core::CContainerPrinter::print(maxScores));

        CPPUNIT_ASSERT_EQUAL(expectedScores[i], core::CContainerPrinter::print(maxScores));
    }
}

void CAnomalyScoreTest::testNormalizeScoresOrdering(void)
{
    // Test that the normalized scores ordering matches the
    // -log(probability) ordering.

    test::CRandomNumbers rng;

    const std::size_t n = 5000;

    TDoubleVec allScores;
    rng.generateUniformSamples(0.0, 50.0, n, allScores);

    for (std::size_t i = 200u; i <= n; i += 200)
    {
        LOG_DEBUG("*** " << i << " ***");

        TDoubleVec scores(&allScores[0], &allScores[i]);

        model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig(300);
        model::CAnomalyScore::CNormalizer normalizer(config);
        for (std::size_t j = 0u; j < i; ++j)
        {
            normalizer.updateQuantiles(scores[j]);
        }

        TDoubleVec normalizedScores(scores);
        for (std::size_t j = 0u; j < i; ++j)
        {
            CPPUNIT_ASSERT(normalizer.normalize(normalizedScores[j]));
        }

        maths::COrderings::simultaneousSort(scores, normalizedScores);
        for (std::size_t j = 1u; j < normalizedScores.size(); ++j)
        {
            if (normalizedScores[j] - normalizedScores[j - 1] < -0.01)
            {
                LOG_DEBUG(normalizedScores[j] << " " << normalizedScores[j - 1]);
            }
            CPPUNIT_ASSERT(normalizedScores[j] - normalizedScores[j - 1] > -0.01);
        }
    }
}

void CAnomalyScoreTest::testJsonConversion(void)
{
    test::CRandomNumbers rng;

    model::CAnomalyScore::TDoubleVec samples;
    // Generate lots of low anomaly scores with some variation
    rng.generateGammaSamples(1.0, 4.0, 500, samples);
    // Add two high anomaly scores
    samples.push_back(222.2);
    samples.push_back(77.7);

    model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig();

    model::CAnomalyScore::CNormalizer origNormalizer(config);
    origNormalizer.updateQuantiles(samples);
    origNormalizer.normalize(samples);

    std::ostringstream ss;
    {
        core::CJsonStatePersistInserter inserter(ss);
        origNormalizer.acceptPersistInserter(inserter);
    }
    std::string origJson =  ss.str();

    // The traverser expects the state json in a embedded document
    std::string wrappedJson = "{\"topLevel\" : " + origJson + "}";

    // Restore the JSON into a new filter
    std::istringstream iss(wrappedJson);
    model::CAnomalyScore::CNormalizer restoredNormalizer(config);
    {
        core::CJsonStateRestoreTraverser traverser(iss);
        traverser.traverseSubLevel(boost::bind(&model::CAnomalyScore::CNormalizer::acceptRestoreTraverser,
                                                              &restoredNormalizer,
                                                              _1));
    }

    // The new JSON representation of the new filter should be the same as the original
    std::ostringstream restoredSs;
    {
        core::CJsonStatePersistInserter inserter(restoredSs);
        restoredNormalizer.acceptPersistInserter(inserter);
    }
    std::string newJson = restoredSs.str();
    CPPUNIT_ASSERT_EQUAL(ss.str(), newJson);

    // The persisted representation should include most the partial
    // representation and extra fields that are used for indexing
    // in a database
    std::string toJson;
    model::CAnomalyScore::normalizerToJson(origNormalizer,
                                           "dummy",
                                           "sysChange",
                                           "my normalizer",
                                           1234567890,
                                           toJson);


    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(toJson.c_str());

    CPPUNIT_ASSERT(doc.HasMember(model::CAnomalyScore::MLCUE_ATTRIBUTE.c_str()));
    CPPUNIT_ASSERT(doc.HasMember(model::CAnomalyScore::MLKEY_ATTRIBUTE.c_str()));
    CPPUNIT_ASSERT(doc.HasMember(model::CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE.c_str()));
    CPPUNIT_ASSERT(doc.HasMember(model::CAnomalyScore::MLVERSION_ATTRIBUTE.c_str()));
    CPPUNIT_ASSERT(doc.HasMember(model::CAnomalyScore::TIME_ATTRIBUTE.c_str()));
    CPPUNIT_ASSERT(doc.HasMember("a"));

    rapidjson::Value &stateDoc = doc["a"];

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    stateDoc.Accept(writer);

    // strip out the newlines before comparing
    std::string state(buffer.GetString());
    state.erase(std::remove(state.begin(), state.end(), '\n'), state.end());

    origJson.erase(std::remove(origJson.begin(), origJson.end(), '\n'), origJson.end());

    CPPUNIT_ASSERT_EQUAL(origJson, state);


    // restore from the JSON state with extra fields used for
    // indexing in the database
    model::CAnomalyScore::CNormalizer fromJsonNormalizer(config);
    CPPUNIT_ASSERT(model::CAnomalyScore::normalizerFromJson(toJson, fromJsonNormalizer));

    std::string restoredJson;
    model::CAnomalyScore::normalizerToJson(fromJsonNormalizer,
                                           "dummy",
                                           "sysChange",
                                           "my normalizer",
                                           1234567890,
                                           restoredJson);

    CPPUNIT_ASSERT_EQUAL(toJson, restoredJson);
}

void CAnomalyScoreTest::testPersistEmpty(void)
{
    // This tests what happens when we persist and restore quantiles that have
    // never had any data added - see bug 761 in Bugzilla

    model::CAnomalyDetectorModelConfig config = model::CAnomalyDetectorModelConfig::defaultConfig();

    model::CAnomalyScore::CNormalizer origNormalizer(config);

    CPPUNIT_ASSERT(!origNormalizer.canNormalize());

    std::string origJson;
    model::CAnomalyScore::normalizerToJson(origNormalizer,
                                           "test",
                                           "test",
                                           "test",
                                           1234567890,
                                           origJson);

    model::CAnomalyScore::CNormalizer newNormalizer(config);

    CPPUNIT_ASSERT(model::CAnomalyScore::normalizerFromJson(origJson,
                                                            newNormalizer));

    CPPUNIT_ASSERT(!newNormalizer.canNormalize());

    std::string newJson;
    model::CAnomalyScore::normalizerToJson(newNormalizer,
                                           "test",
                                           "test",
                                           "test",
                                           1234567890,
                                           newJson);

    CPPUNIT_ASSERT_EQUAL(origJson, newJson);
}

CppUnit::Test *CAnomalyScoreTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CAnomalyScoreTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testComputeScores",
                                   &CAnomalyScoreTest::testComputeScores) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testNormalizeScoresQuantiles",
                                   &CAnomalyScoreTest::testNormalizeScoresQuantiles) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testNormalizeScoresNoisy",
                                   &CAnomalyScoreTest::testNormalizeScoresNoisy) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testNormalizeScoresLargeScore",
                                   &CAnomalyScoreTest::testNormalizeScoresLargeScore) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testNormalizeScoresNearZero",
                                   &CAnomalyScoreTest::testNormalizeScoresNearZero) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testNormalizeScoresOrdering",
                                   &CAnomalyScoreTest::testNormalizeScoresOrdering) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testJsonConversion",
                                   &CAnomalyScoreTest::testJsonConversion) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyScoreTest>(
                                   "CAnomalyScoreTest::testPersistEmpty",
                                   &CAnomalyScoreTest::testPersistEmpty) );

    return suiteOfTests;
}
