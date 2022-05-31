/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CTools.h>
#include <maths/common/ProbabilityAggregators.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CAnomalyScore.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <sstream>
#include <stdio.h>

BOOST_AUTO_TEST_SUITE(CAnomalyScoreTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
}

BOOST_AUTO_TEST_CASE(testComputeScores) {
    using TScores = model::CAnomalyScore;
    using TJointProbabilityCalculator = maths::common::CJointProbabilityOfLessLikelySamples;
    using TLogExtremeProbabilityCalculator = maths::common::CLogProbabilityOfMFromNExtremeSamples;

    const double jointProbabilityWeight = 0.5;
    const double extremeProbabilityWeight = 0.5;
    const std::size_t minExtremeSamples = 1;
    const std::size_t maxExtremeSamples = 5;
    const double maximumAnomalousProbability = 0.05;
    double overallScore = 0.0;
    double overallProbability = 1.0;

    // Test 1: single value.
    // Expect deviation of the probability.
    {
        TDoubleVec p1(1, 0.001);
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         p1, overallScore, overallProbability);
        LOG_DEBUG(<< "1) score 1 = " << overallScore);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.078557, overallScore, 5e-7);

        TDoubleVec p2(1, 0.02);
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         p2, overallScore, overallProbability);
        LOG_DEBUG(<< "1) score 2 = " << overallScore);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.002405, overallScore, 5e-7);

        TDoubleVec p3(1, 0.1);
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         p3, overallScore, overallProbability);
        LOG_DEBUG(<< "1) score 3 = " << overallScore);
        BOOST_REQUIRE_EQUAL(0.0, overallScore);
    }

    // Test 2: low anomalousness.
    // Expect scores of zero.
    {
        TDoubleVec probabilities{0.21, 0.52, 0.13,  0.67, 0.89,
                                 0.32, 0.46, 0.222, 0.35, 0.93};
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         probabilities, overallScore, overallProbability);
        LOG_DEBUG(<< "2) score = " << overallScore);
        BOOST_REQUIRE_EQUAL(0.0, overallScore);
    }

    // Test 3: a lot of slightly anomalous values.
    // Expect a high anomaly score which is generated by the
    // joint probability of less likely samples.
    {
        TDoubleVec probabilities{0.11, 0.13, 0.12, 0.22, 0.14, 0.09,
                                 0.01, 0.13, 0.15, 0.14, 0.11, 0.13,
                                 0.12, 0.22, 0.09, 0.01};
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         probabilities, overallScore, overallProbability);

        TJointProbabilityCalculator jointProbabilityCalculator;
        TLogExtremeProbabilityCalculator extremeProbabilityCalculator(2);
        for (double p : probabilities) {
            jointProbabilityCalculator.add(p);
            extremeProbabilityCalculator.add(p);
        }

        double jointProbability;
        BOOST_TEST_REQUIRE(jointProbabilityCalculator.calculate(jointProbability));
        double extremeProbability;
        BOOST_TEST_REQUIRE(extremeProbabilityCalculator.calculate(extremeProbability));
        extremeProbability = std::exp(extremeProbability);

        LOG_DEBUG(<< "3) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG(<< "   joint probability = " << jointProbability << ", extreme probability = "
                  << extremeProbability << ", overallScore = " << overallScore);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.318231, overallScore, 5e-7);
    }

    // Test 4: one very anomalous value.
    // Expect a high anomaly score which is generated by the
    // extreme samples probability.
    {
        TDoubleVec probabilities{0.21,  0.52, 0.13, 0.67, 0.89, 0.32, 0.46,
                                 0.222, 0.35, 0.93, 0.89, 0.32, 0.46, 0.000021};
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         probabilities, overallScore, overallProbability);

        TJointProbabilityCalculator jointProbabilityCalculator;
        TLogExtremeProbabilityCalculator extremeProbabilityCalculator(1);
        for (double p : probabilities) {
            jointProbabilityCalculator.add(p);
            extremeProbabilityCalculator.add(p);
        }

        double jointProbability;
        BOOST_TEST_REQUIRE(jointProbabilityCalculator.calculate(jointProbability));

        double extremeProbability;
        BOOST_TEST_REQUIRE(extremeProbabilityCalculator.calculate(extremeProbability));
        extremeProbability = std::exp(extremeProbability);

        LOG_DEBUG(<< "4) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG(<< "   joint probability = " << jointProbability << ", extreme probability = "
                  << extremeProbability << ", overallScore = " << overallScore);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.137591, overallScore, 5e-7);
    }

    // Test 5: a small number of "mid" anomalous values.
    // Expect a high anomaly score which is generated by the
    // extreme samples probability.
    {
        TDoubleVec probabilities{0.21, 0.52, 0.0058, 0.13,  0.67,  0.89,
                                 0.32, 0.03, 0.46,   0.222, 0.35,  0.93,
                                 0.01, 0.89, 0.32,   0.46,  0.0021};
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         probabilities, overallScore, overallProbability);

        TJointProbabilityCalculator jointProbabilityCalculator;
        TLogExtremeProbabilityCalculator extremeProbabilityCalculator(4);
        for (double p : probabilities) {
            jointProbabilityCalculator.add(p);
            extremeProbabilityCalculator.add(p);
        }

        double jointProbability;
        BOOST_TEST_REQUIRE(jointProbabilityCalculator.calculate(jointProbability));

        double extremeProbability;
        BOOST_TEST_REQUIRE(extremeProbabilityCalculator.calculate(extremeProbability));
        extremeProbability = std::exp(extremeProbability);

        LOG_DEBUG(<< "5) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG(<< "   joint probability = " << jointProbability << ", extreme probability = "
                  << extremeProbability << ", overallScore = " << overallScore);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.029413, overallScore, 5e-7);
    }

    // Test 6: underflow.
    {
        TDoubleVec probabilities{1e-100,   1.7e-20,  1.6e-150, 2.2e-150,
                                 1.3e-180, 1.35e-95, 1.7e-180, 1.21e-300};
        TScores::compute(jointProbabilityWeight, extremeProbabilityWeight,
                         minExtremeSamples, maxExtremeSamples, maximumAnomalousProbability,
                         probabilities, overallScore, overallProbability);

        LOG_DEBUG(<< "6) probabilities = " << core::CContainerPrinter::print(probabilities));
        LOG_DEBUG(<< "   overallScore = " << overallScore);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(86.995953, overallScore, 5e-7);
    }
}

BOOST_AUTO_TEST_CASE(testNormalizeScoresQuantiles) {
    using TDoubleMSet = std::multiset<double>;

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 2.0, 20000, samples);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] < 0.5) {
            samples[i] = 0.0;
        }
    }

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);
    normalizer.isForMembersOfPopulation(false);

    double totalError = 0.0;
    double numberSamples = 0.0;

    TDoubleMSet scores;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        scores.insert(samples[i]);
        normalizer.updateQuantiles({"", "", "bucket_time", ""}, samples[i]);

        auto itr = scores.upper_bound(samples[i]);
        double trueQuantile = static_cast<double>(std::distance(scores.begin(), itr)) /
                              static_cast<double>(scores.size());

        if (trueQuantile > 0.9) {
            double lowerBound;
            double upperBound;
            normalizer.quantile(samples[i], 0.0, lowerBound, upperBound);

            double quantile = (lowerBound + upperBound) / 2.0;
            double error = std::fabs(quantile - trueQuantile);

            totalError += error;
            numberSamples += 1.0;

            LOG_TRACE(<< "trueQuantile = " << trueQuantile << ", lowerBound = " << lowerBound
                      << ", upperBound = " << upperBound);
            BOOST_TEST_REQUIRE(error < 0.02);
        }
    }

    LOG_DEBUG(<< "meanError = " << totalError / numberSamples);
    BOOST_TEST_REQUIRE(totalError / numberSamples < 0.0043);
}

// Test as with testNormalizeScores but with more than one partition
// The quantile state will differ but still should be within reasonable bounds
BOOST_AUTO_TEST_CASE(testNormalizeScoresQuantilesMultiplePartitions) {
    using TDoubleMSet = std::multiset<double>;

    test::CRandomNumbers rng;

    // generate two similar base samples
    TDoubleVec samplesAAL;
    rng.generateGammaSamples(1.0, 2.0, 20000, samplesAAL);

    TDoubleVec samplesKLM;
    rng.generateGammaSamples(1.0, 2.0, 20000, samplesKLM);

    TDoubleVec samplesJAL;
    rng.generateGammaSamples(1.0, 2.0, 20000, samplesJAL);

    BOOST_REQUIRE_EQUAL(samplesAAL.size(), samplesKLM.size());
    for (std::size_t i = 0; i < samplesAAL.size(); ++i) {
        if (samplesAAL[i] < 0.5) {
            samplesAAL[i] = 0.0;
        }
        if (samplesKLM[i] < 0.5) {
            samplesKLM[i] = 0.0;
        }
        if (samplesJAL[i] < 0.5) {
            samplesJAL[i] = 0.0;
        }
    }

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);
    normalizer.isForMembersOfPopulation(false);

    double totalError = 0.0;
    double numberSamples = 0.0;

    TDoubleMSet scores;
    for (std::size_t i = 0; i < samplesAAL.size(); ++i) {
        scores.insert(samplesAAL[i]);
        normalizer.updateQuantiles({"airline", "AAL", "", ""}, samplesAAL[i]);
        normalizer.updateQuantiles({"airline", "KLM", "", ""}, samplesAAL[i]);
        normalizer.updateQuantiles({"airline", "JAL", "", ""}, samplesAAL[i]);

        auto itr = scores.upper_bound(samplesAAL[i]);
        double trueQuantile = static_cast<double>(std::distance(scores.begin(), itr)) /
                              static_cast<double>(scores.size());

        if (trueQuantile > 0.9) {
            double lowerBound;
            double upperBound;
            normalizer.quantile(samplesAAL[i], 0.0, lowerBound, upperBound);

            double quantile = (lowerBound + upperBound) / 2.0;
            double error = std::fabs(quantile - trueQuantile);

            totalError += error;
            numberSamples += 1.0;

            LOG_TRACE(<< "trueQuantile = " << trueQuantile << ", lowerBound = " << lowerBound
                      << ", upperBound = " << upperBound);
            BOOST_TEST_REQUIRE(error < 0.02);
        }
    }

    LOG_DEBUG(<< "meanError = " << totalError / numberSamples);
    BOOST_TEST_REQUIRE(totalError / numberSamples < 0.0047);
}

BOOST_AUTO_TEST_CASE(testNormalizeScoresNoisy) {
    using TDoubleSizeMap = std::multimap<double, std::size_t>;

    test::CRandomNumbers rng;

    // Exponential, i.e. shape = 1
    TDoubleVec samples;
    rng.generateGammaSamples(1.0, 2.0, 2000, samples);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] < 0.5) {
            samples[i] = 0.0;
        }
    }

    TSizeVec largeAnomalyTimes{50,  110, 190, 220,  290,  310,  600,
                               620, 790, 900, 1100, 1400, 1600, 1900};

    TDoubleVec largeAnomalies{50.0,  350.0, 30.0,  100.0, 30.0, 45.0, 100.0,
                              120.0, 60.0,  130.0, 100.0, 90.0, 45.0, 30.0};

    // Add in the big anomalies.
    for (size_t i = 0; i < largeAnomalyTimes.size(); ++i) {
        samples[largeAnomalyTimes[i]] += largeAnomalies[i];
    }

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);
    normalizer.isForMembersOfPopulation(false);

    //std::ostringstream raw;
    //std::ostringstream normalized;
    //raw << "raw = [";
    //normalized << "normalized = [";

    TDoubleSizeMap maxScores;

    for (std::size_t i = 0; i < samples.size(); ++i) {
        normalizer.updateQuantiles({"", "", "", ""}, samples[i]);
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
        double sample = samples[i];
        normalizer.normalize({"", "", "bucket_time", ""}, sample);
        LOG_TRACE(<< i << ") raw = " << samples[i] << ", normalized = " << sample);

        //raw << " " << samples[i];
        //normalized << " " << sample;

        if (maxScores.size() < boost::size(largeAnomalyTimes)) {
            maxScores.insert(TDoubleSizeMap::value_type(sample, i));
        } else if (sample > maxScores.begin()->first) {
            LOG_TRACE(<< "normalized = " << sample << " removing "
                      << maxScores.begin()->first);
            maxScores.erase(maxScores.begin());
            maxScores.insert(TDoubleSizeMap::value_type(sample, i));
        }
    }

    //raw << " ];";
    //normalized << " ];";
    //std::ofstream f;
    //f.open("scores.m");
    //f << raw.str() << "\n" << normalized.str();

    LOG_DEBUG(<< "maxScores = " << core::CContainerPrinter::print(maxScores));

    TSizeVec times;

    for (const auto& maxScore : maxScores) {
        times.push_back(maxScore.second);
    }
    std::sort(times.begin(), times.end());

    LOG_DEBUG(<< "times = " << core::CContainerPrinter::print(times));

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(largeAnomalyTimes),
                        core::CContainerPrinter::print(times));
}

BOOST_AUTO_TEST_CASE(testNormalizeScoresPerPartitionMaxScore) {
    // Test that another partition in the same normalizer does noes not unduly affect the scores
    // (partitions are not totally independent and interact via the shared quantile state)

    // scores from a single partition
    TDoubleVec expectedAALScores{10.31767, 20.24881, 20.35545, 25.07737, 94.09136};

    test::CRandomNumbers rng;

    // generate two similar base samples
    TDoubleVec samplesAAL;
    rng.generateUniformSamples(0.0, 0.1, 500, samplesAAL);

    TDoubleVec samplesKLM;
    rng.generateUniformSamples(0.0, 0.1, 500, samplesKLM);

    TSizeVec anomalyTimes{50, 110, 190, 220, 290};
    TDoubleVec anomalies{2.0, 4.0, 4.0, 5.0, 30.0};

    // Add in the anomalies.
    for (size_t i = 0; i < anomalyTimes.size(); ++i) {
        samplesAAL[anomalyTimes[i]] += anomalies[i];
        samplesKLM[anomalyTimes[i]] += anomalies[i];
    }

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);
    normalizer.isForMembersOfPopulation(false);

    for (std::size_t i = 0; i < samplesAAL.size(); ++i) {
        normalizer.updateQuantiles({"", "", "airline", "AAL"}, samplesAAL[i]);
        normalizer.updateQuantiles({"", "", "airline", "KLM"}, samplesKLM[i]);
    }

    TDoubleVec actualAALScores;
    for (std::size_t i = 0; i < boost::size(anomalyTimes); ++i) {
        double sampleAAL = samplesAAL[anomalyTimes[i]];
        double sampleKLM = samplesKLM[anomalyTimes[i]];
        LOG_DEBUG(<< "sampleAAL = " << sampleAAL);
        LOG_DEBUG(<< "sampleKLM = " << sampleKLM);
        normalizer.normalize({"", "", "airline", "AAL"}, sampleAAL);
        normalizer.normalize({"", "", "airline", "KLM"}, sampleKLM);
        actualAALScores.push_back(sampleAAL);
    }
    std::sort(actualAALScores.begin(), actualAALScores.end());
    LOG_DEBUG(<< "actualAALScores = " << core::CContainerPrinter::print(actualAALScores));

    for (std::size_t i = 0; i < boost::size(expectedAALScores); ++i) {
        double pctChange = 100.0 *
                           (std::fabs(actualAALScores[i] - expectedAALScores[i])) /
                           expectedAALScores[i];
        LOG_DEBUG(<< "pctChange = " << pctChange << " actualAALScores[" << i
                  << "] = " << actualAALScores[i] << " expectedAALScores[" << i
                  << "] = " << expectedAALScores[i]);
        BOOST_TEST_REQUIRE(pctChange < 5.0);
    }
}

BOOST_AUTO_TEST_CASE(testNormalizeScoresLargeScore) {
    // Test a large score isn't too dominant.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 0.1, 500, samples);

    TSizeVec anomalyTimes{50, 110, 190, 220, 290};
    TDoubleVec anomalies{2.0, 4.0, 4.0, 5.0, 20.0};

    // Add in the anomalies.
    for (std::size_t i = 0; i < anomalyTimes.size(); ++i) {
        samples[anomalyTimes[i]] += anomalies[i];
    }

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);
    normalizer.isForMembersOfPopulation(false);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        normalizer.updateQuantiles({"", "", "", ""}, samples[i]);
    }

    TDoubleVec scores;
    for (std::size_t i = 0; i < boost::size(anomalyTimes); ++i) {
        double sample = samples[anomalyTimes[i]];
        normalizer.normalize({"", "", "bucket_time", ""}, sample);
        scores.push_back(sample);
    }
    std::sort(scores.begin(), scores.end());
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    for (std::size_t i = 0; i + 1 < anomalies.size(); ++i) {
        double change = 100.0 * anomalies[i] / anomalies[anomalies.size() - 1];
        double uplift = scores[i] - change;
        LOG_DEBUG(<< "uplift = " << uplift << " scores[" << i << "] = " << scores[i] << " anomalies["
                  << i << "] = " << anomalies[i] << " change = " << change);
        BOOST_TEST_REQUIRE(uplift > 5.0);
        BOOST_TEST_REQUIRE(uplift < 13.0);
    }
}

BOOST_AUTO_TEST_CASE(testNormalizeScoresNearZero) {
    // Test the behaviour for scores near zero.

    std::size_t anomalyTimes[] = {50, 110, 190, 220, 290};
    double anomalies[] = {0.02, 0.01, 0.006, 0.01, 0.015};

    std::size_t nonZeroCounts[] = {0, 100, 200, 249, 251, 300, 400, 450};

    std::string expectedScores[] = {
        std::string("[41.62776, 32.36435, 26.16873, 32.36435, 37.68726]"),
        std::string("[41.62776, 32.36435, 26.16873, 32.36435, 37.68726]"),
        std::string("[41.62776, 32.36435, 17.74216, 32.36435, 37.68726]"),
        std::string("[41.62776, 32.36435, 11.1645, 32.36435, 37.68726]"),
        std::string("[41.62776, 32.36435, 11.05937, 32.36435, 37.68726]"),
        std::string("[41.62776, 32.36435, 8.523397, 32.36435, 37.68726]"),
        std::string("[1.14, 1.04, 1, 1.04, 1.09]"),
        std::string("[1.14, 1.04, 1, 1.04, 1.09]")};

    for (std::size_t i = 0; i < std::size(nonZeroCounts); ++i) {
        LOG_DEBUG(<< "non-zero count = " << nonZeroCounts[i]);

        TDoubleVec samples(500, 0.0);
        for (std::size_t j = 0; j < nonZeroCounts[i]; ++j) {
            if (std::find(std::begin(anomalyTimes), std::end(anomalyTimes), j) ==
                std::end(anomalyTimes)) {
                samples[j] += 0.0055;
            }
        }
        for (std::size_t j = 0; j < boost::size(anomalyTimes); ++j) {
            samples[anomalyTimes[j]] += anomalies[j];
        }

        model::CAnomalyDetectorModelConfig config =
            model::CAnomalyDetectorModelConfig::defaultConfig(1800);
        model::CAnomalyScore::CNormalizer normalizer(config);
        normalizer.isForMembersOfPopulation(false);

        for (std::size_t j = 0; j < samples.size(); ++j) {
            normalizer.updateQuantiles({"", "", "", ""}, samples[j]);
        }

        TDoubleVec maxScores;
        for (std::size_t j = 0; j < boost::size(anomalyTimes); ++j) {
            double sample = samples[anomalyTimes[j]];
            normalizer.normalize({"", "", "bucket_time", ""}, sample);
            maxScores.push_back(sample);
        }
        LOG_DEBUG(<< "maxScores = " << core::CContainerPrinter::print(maxScores));

        BOOST_REQUIRE_EQUAL(expectedScores[i], core::CContainerPrinter::print(maxScores));
    }
}

BOOST_AUTO_TEST_CASE(testNormalizeScoresOrdering) {
    // Test that the normalized scores ordering matches the
    // -log(probability) ordering.

    test::CRandomNumbers rng;

    const std::size_t n = 5000;

    TDoubleVec allScores;
    rng.generateUniformSamples(0.0, 50.0, n, allScores);

    for (std::size_t i = 200; i <= n; i += 200) {
        LOG_DEBUG(<< "*** " << i << " ***");

        TDoubleVec scores(&allScores[0], &allScores[i]);

        model::CAnomalyDetectorModelConfig config =
            model::CAnomalyDetectorModelConfig::defaultConfig(300);
        model::CAnomalyScore::CNormalizer normalizer(config);
        normalizer.isForMembersOfPopulation(false);

        for (std::size_t j = 0; j < i; ++j) {
            normalizer.updateQuantiles({"", "", "", ""}, scores[j]);
        }

        TDoubleVec normalizedScores(scores);
        for (std::size_t j = 0; j < i; ++j) {
            BOOST_TEST_REQUIRE(normalizer.normalize({"", "", "bucket_time", ""},
                                                    normalizedScores[j]));
        }

        maths::common::COrderings::simultaneousSort(scores, normalizedScores);
        for (std::size_t j = 1; j < normalizedScores.size(); ++j) {
            if (normalizedScores[j] - normalizedScores[j - 1] < -0.01) {
                LOG_DEBUG(<< normalizedScores[j] << " " << normalizedScores[j - 1]);
            }
            BOOST_TEST_REQUIRE(normalizedScores[j] - normalizedScores[j - 1] > -0.01);
        }
    }
}

BOOST_AUTO_TEST_CASE(testNormalizerGetMaxScore) {
    test::CRandomNumbers rng;

    // generate two similar base samples
    TDoubleVec samplesAAL;
    rng.generateUniformSamples(0.0, 0.1, 500, samplesAAL);

    TDoubleVec samplesKLM;
    rng.generateUniformSamples(0.0, 0.1, 500, samplesKLM);

    TSizeVec anomalyTimesAAL{50, 110, 190, 220, 290};
    TDoubleVec anomaliesAAL{1.0, 3.0, 4.0, 5.0, 40.0};

    TSizeVec anomalyTimesKLM{45, 120, 210, 220, 300};
    TDoubleVec anomaliesKLM{2.0, 3.0, 4.0, 6.0, 30.0};

    // Add in the anomalies.
    for (size_t i = 0; i < anomalyTimesAAL.size(); ++i) {
        samplesAAL[anomalyTimesAAL[i]] += anomaliesAAL[i];
    }
    for (size_t i = 0; i < anomalyTimesKLM.size(); ++i) {
        samplesAAL[anomalyTimesKLM[i]] += anomaliesKLM[i];
    }

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig(1800);
    model::CAnomalyScore::CNormalizer normalizer(config);
    normalizer.isForMembersOfPopulation(false);

    for (std::size_t i = 0; i < samplesAAL.size(); ++i) {
        normalizer.updateQuantiles({"", "", "airline", "AAL"}, samplesAAL[i]);
        normalizer.updateQuantiles({"", "", "airline", "KLM"}, samplesKLM[i]);
    }

    double maxScoreAAL;
    BOOST_TEST_REQUIRE(normalizer.maxScore({"", "", "airline", "AAL"}, maxScoreAAL));

    double maxScoreKLM;
    BOOST_TEST_REQUIRE(normalizer.maxScore({"", "", "airline", "KLM"}, maxScoreKLM));

    LOG_DEBUG(<< "maxScoreAAL = " << maxScoreAAL);
    LOG_DEBUG(<< "maxScoreKLM = " << maxScoreKLM);

    const double expecteMaxScoreAAL = 40.0583;
    const double expecteMaxScoreKLM = 0.0999999;

    // Check that the partition wise max scores are as expected
    // (and hence different from one another)
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expecteMaxScoreAAL, maxScoreAAL, 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expecteMaxScoreKLM, maxScoreKLM, 5e-5);
}

BOOST_AUTO_TEST_CASE(testNoiseForDifferentBucketLengths) {
    // Simulate noise in different bucket lengths and check that distribution
    // alert severities. Because there are multiple factors which affect the
    // score we don't necessarily get the same distribution for all bucket
    // lengths. However, we do have an approximate upper bound on the count
    // by severity based on the rate limiting we perform.

    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

    test::CRandomNumbers rng;

    TMeanAccumulatorVecVec averageCounts;
    TDoubleVec probabilities;
    for (std::size_t samplesPerInterval : {1, 2, 6, 9, 12}) {
        averageCounts.emplace_back(5);
        for (std::size_t trial = 0; trial < 10; ++trial) {
            model::CAnomalyDetectorModelConfig config{
                model::CAnomalyDetectorModelConfig::defaultConfig(1800 / samplesPerInterval)};
            model::CAnomalyScore::CNormalizer normalizer(config);
            normalizer.isForMembersOfPopulation(false);

            TDoubleVec scoreThresholds{50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
            TSizeVec scoreCounts{0, 0, 0, 0, 0};
            for (std::size_t i = 0; i < 10000; ++i) {

                rng.generateUniformSamples(0.0, 1.0, samplesPerInterval, probabilities);

                double maxScore{0.0};
                for (auto p : probabilities) {
                    double score{maths::common::CTools::anomalyScore(p)};
                    normalizer.updateQuantiles({"", "", "", ""}, score);
                    normalizer.normalize({"", "", "", ""}, score);
                    normalizer.propagateForwardByTime(1.0);
                    maxScore = std::max(maxScore, score);
                }
                if (i < 1000) {
                    continue;
                }

                auto j = std::upper_bound(scoreThresholds.begin(),
                                          scoreThresholds.end(), maxScore) -
                         scoreThresholds.begin();
                if (j > 0) {
                    ++scoreCounts[j - 1];
                }
            }

            for (std::size_t i = 0; i < scoreCounts.size(); ++i) {
                averageCounts.back()[i].add(scoreCounts[i]);
            }
        }
    }

    TDoubleVec rateLimits{9000 * 0.0025, 9000 * 0.0025, 9000 * 0.0025,
                          9000 * 0.0025, 9000 * 0.001};
    for (const auto& averageCountsForBucketLength : averageCounts) {
        LOG_DEBUG(<< "averageCountsForBucketLength = "
                  << core::CContainerPrinter::print(averageCountsForBucketLength));
        for (std::size_t i = 0; i < rateLimits.size(); ++i) {
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(
                                   averageCountsForBucketLength[i]) < 1.1 * rateLimits[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testJsonConversion) {
    test::CRandomNumbers rng;

    const double maxScore{222.2};

    model::CAnomalyScore::TDoubleVec samples;
    // Generate lots of low anomaly scores with some variation
    rng.generateGammaSamples(1.0, 4.0, 500, samples);
    // Add two high anomaly scores
    samples.push_back(maxScore);
    samples.push_back(77.7);

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    model::CAnomalyScore::CNormalizer origNormalizer(config);
    origNormalizer.isForMembersOfPopulation(true);

    for (auto& sample : samples) {
        origNormalizer.updateQuantiles({"", "", "", ""}, sample);
        origNormalizer.normalize({"", "", "bucket_time", ""}, sample);
    }
    std::ostringstream ss;
    {
        core::CJsonStatePersistInserter inserter(ss);
        origNormalizer.acceptPersistInserter(inserter);
    }
    std::string origJson = ss.str();

    // The traverser expects the state json in a embedded document
    std::string wrappedJson = "{\"topLevel\" : " + origJson + "}";

    // Restore the JSON into a new filter
    std::istringstream iss(wrappedJson);
    model::CAnomalyScore::CNormalizer restoredNormalizer(config);
    {
        core::CJsonStateRestoreTraverser traverser(iss);
        traverser.traverseSubLevel(
            std::bind(&model::CAnomalyScore::CNormalizer::acceptRestoreTraverser,
                      &restoredNormalizer, std::placeholders::_1));
    }

    // The new JSON representation of the new filter should be the same as the original
    std::ostringstream restoredSs;
    {
        core::CJsonStatePersistInserter inserter(restoredSs);
        restoredNormalizer.acceptPersistInserter(inserter);
    }
    std::string newJson = restoredSs.str();
    BOOST_REQUIRE_EQUAL(ss.str(), newJson);

    // The persisted representation should include most the partial
    // representation and extra fields that are used for indexing
    // in a database
    std::string toJson;
    model::CAnomalyScore::normalizerToJson(origNormalizer, "dummy", "sysChange",
                                           "my normalizer", 1234567890, toJson);

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(toJson.c_str());

    BOOST_TEST_REQUIRE(doc.HasMember(model::CAnomalyScore::MLCUE_ATTRIBUTE.c_str()));
    BOOST_TEST_REQUIRE(doc.HasMember(model::CAnomalyScore::MLKEY_ATTRIBUTE.c_str()));
    BOOST_TEST_REQUIRE(doc.HasMember(
        model::CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE.c_str()));
    BOOST_TEST_REQUIRE(doc.HasMember(model::CAnomalyScore::MLVERSION_ATTRIBUTE.c_str()));
    BOOST_TEST_REQUIRE(doc.HasMember(model::CAnomalyScore::TIME_ATTRIBUTE.c_str()));
    BOOST_TEST_REQUIRE(doc.HasMember("a"));

    rapidjson::Value& stateDoc = doc["a"];

    {
        // Check that all required fields are present in the persisted state
        BOOST_TEST_REQUIRE(stateDoc.HasMember("a"));
        BOOST_TEST_REQUIRE(stateDoc.HasMember("b"));
        // Field 'c' - m_MaxScore - removed in version > 6.5
        BOOST_TEST_REQUIRE(stateDoc.HasMember("d"));
        BOOST_TEST_REQUIRE(stateDoc.HasMember("e"));
        BOOST_TEST_REQUIRE(stateDoc.HasMember("f"));
        BOOST_TEST_REQUIRE(stateDoc.HasMember("g"));
        BOOST_TEST_REQUIRE(stateDoc["g"].HasMember("d"));
        BOOST_TEST_REQUIRE(stateDoc["g"].HasMember("a"));
        BOOST_TEST_REQUIRE(stateDoc["g"]["a"].HasMember("a"));
        BOOST_TEST_REQUIRE(stateDoc["g"]["a"].HasMember("b"));

        rapidjson::Value& partitionMaxScoreDoc = stateDoc["g"]["a"]["b"]["a"];

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        partitionMaxScoreDoc.Accept(writer);

        std::string partitionMaxScoreStr(buffer.GetString());
        partitionMaxScoreStr.erase(std::remove(partitionMaxScoreStr.begin(),
                                               partitionMaxScoreStr.end(), '\n'),
                                   partitionMaxScoreStr.end());
        BOOST_REQUIRE_EQUAL(
            "\"" + core::CStringUtils::typeToStringPrecise(maxScore, core::CIEEE754::E_SinglePrecision) + "\"",
            partitionMaxScoreStr);
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    stateDoc.Accept(writer);

    // strip out the newlines before comparing
    std::string state(buffer.GetString());
    state.erase(std::remove(state.begin(), state.end(), '\n'), state.end());

    origJson.erase(std::remove(origJson.begin(), origJson.end(), '\n'), origJson.end());

    BOOST_REQUIRE_EQUAL(origJson, state);

    // restore from the JSON state with extra fields used for
    // indexing in the database
    model::CAnomalyScore::CNormalizer fromJsonNormalizer(config);
    BOOST_TEST_REQUIRE(model::CAnomalyScore::normalizerFromJson(toJson, fromJsonNormalizer));

    std::string restoredJson;
    model::CAnomalyScore::normalizerToJson(fromJsonNormalizer, "dummy", "sysChange",
                                           "my normalizer", 1234567890, restoredJson);

    BOOST_REQUIRE_EQUAL(toJson, restoredJson);
}

BOOST_AUTO_TEST_CASE(testPersistEmpty) {
    // This tests what happens when we persist and restore quantiles that have
    // never had any data added - see bug 761 in Bugzilla

    model::CAnomalyDetectorModelConfig config =
        model::CAnomalyDetectorModelConfig::defaultConfig();

    model::CAnomalyScore::CNormalizer origNormalizer(config);

    BOOST_TEST_REQUIRE(!origNormalizer.canNormalize());

    std::string origJson;
    model::CAnomalyScore::normalizerToJson(origNormalizer, "test", "test",
                                           "test", 1234567890, origJson);

    model::CAnomalyScore::CNormalizer newNormalizer(config);

    BOOST_TEST_REQUIRE(model::CAnomalyScore::normalizerFromJson(origJson, newNormalizer));

    BOOST_TEST_REQUIRE(!newNormalizer.canNormalize());

    std::string newJson;
    model::CAnomalyScore::normalizerToJson(newNormalizer, "test", "test",
                                           "test", 1234567890, newJson);

    BOOST_REQUIRE_EQUAL(origJson, newJson);
}

BOOST_AUTO_TEST_SUITE_END()
