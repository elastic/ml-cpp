/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CStringSimilarityTester.h>
#include <core/CTimeUtils.h>

#include <boost/test/unit_test.hpp>

#include <string>
#include <utility>
#include <vector>

#include <ctype.h>
#include <stdlib.h>

BOOST_AUTO_TEST_SUITE(CStringSimilarityTesterTest)

BOOST_AUTO_TEST_CASE(testStringSimilarity) {
    std::string str1("This is identical");
    std::string str2("This is identical");

    std::string str3("Identical bar some numbers 12345");
    std::string str4("Identical bar some numbers 67890");

    // Completely different
    std::string str5("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    std::string str6("abcdefghijklmnopqrstuvwxyz");

    // Empty strings
    std::string str7;
    std::string str8;

    ml::core::CStringSimilarityTester sst;

    double similarity1(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str1, str2, similarity1));
    LOG_DEBUG(<< "similarity1 = " << similarity1);

    double similarity2(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str3, str4, similarity2));
    LOG_DEBUG(<< "similarity2 = " << similarity2);

    double similarity3(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str5, str6, similarity3));
    LOG_DEBUG(<< "similarity3 = " << similarity3);

    // If the method is going to be of any use, these conditions
    // must hold
    BOOST_TEST_REQUIRE(similarity1 > similarity2);
    BOOST_TEST_REQUIRE(similarity2 > similarity3);

    double similarity4(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str7, str8, similarity4));
    LOG_DEBUG(<< "similarity4 = " << similarity4);

    // This is a boundary case that could cause division by 0
    BOOST_REQUIRE_EQUAL(1.0, similarity4);

    double similarity5(0.0);
    BOOST_TEST_REQUIRE(sst.similarityEx(str3, str4, &::isdigit, similarity5));
    LOG_DEBUG(<< "similarity5 = " << similarity5);

    std::string str3Stripped(sst.strippedString(str3, &::isdigit));
    std::string str4Stripped(sst.strippedString(str4, &::isdigit));

    double similarity6(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str3Stripped, str4Stripped, similarity6));
    LOG_DEBUG(<< "similarity6 = " << similarity6);

    // Stripping the strings within the similarityEx method or separately should
    // give the same results
    BOOST_REQUIRE_EQUAL(similarity5, similarity6);

    size_t str5CompLen(0);
    size_t str6CompLen(0);

    BOOST_TEST_REQUIRE(sst.compressedLengthOf(str5, str5CompLen));
    BOOST_TEST_REQUIRE(sst.compressedLengthOf(str6, str6CompLen));

    double similarity7(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str5, str5CompLen, str6, str6CompLen, similarity7));
    LOG_DEBUG(<< "similarity7 = " << similarity7);

    // Passing in pre-calculated compressed lengths of the individual string
    // should give the same results as letting the similarity method calculate
    // them
    BOOST_REQUIRE_EQUAL(similarity3, similarity7);

    double similarity8(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str6, str6CompLen, str5, str5CompLen, similarity8));
    LOG_DEBUG(<< "similarity8 = " << similarity8);

    // Results should be symmetrical when passing in pre-calculated compressed
    // lengths
    BOOST_REQUIRE_EQUAL(similarity7, similarity8);

    double similarity9(0.0);
    BOOST_TEST_REQUIRE(sst.similarity(str6, str5, similarity9));
    LOG_DEBUG(<< "similarity9 = " << similarity9);

    // Results should be symmetrical when letting the similarity method calculate
    // everything
    BOOST_REQUIRE_EQUAL(similarity3, similarity9);
}

BOOST_AUTO_TEST_CASE(testLevensteinDistance) {
    ml::core::CStringSimilarityTester sst;

    std::string cat("cat");
    std::string dog("dog");
    std::string mouse("mouse");
    std::string nothing;

    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistance(cat, cat));
    BOOST_REQUIRE_EQUAL(size_t(3), sst.levenshteinDistance(cat, dog));
    BOOST_REQUIRE_EQUAL(size_t(5), sst.levenshteinDistance(cat, mouse));
    BOOST_REQUIRE_EQUAL(size_t(3), sst.levenshteinDistance(cat, nothing));

    BOOST_REQUIRE_EQUAL(size_t(3), sst.levenshteinDistance(dog, cat));
    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistance(dog, dog));
    BOOST_REQUIRE_EQUAL(size_t(4), sst.levenshteinDistance(dog, mouse));
    BOOST_REQUIRE_EQUAL(size_t(3), sst.levenshteinDistance(dog, nothing));

    BOOST_REQUIRE_EQUAL(size_t(5), sst.levenshteinDistance(mouse, cat));
    BOOST_REQUIRE_EQUAL(size_t(4), sst.levenshteinDistance(mouse, dog));
    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistance(mouse, mouse));
    BOOST_REQUIRE_EQUAL(size_t(5), sst.levenshteinDistance(mouse, nothing));

    BOOST_REQUIRE_EQUAL(size_t(3), sst.levenshteinDistance(nothing, cat));
    BOOST_REQUIRE_EQUAL(size_t(3), sst.levenshteinDistance(nothing, dog));
    BOOST_REQUIRE_EQUAL(size_t(5), sst.levenshteinDistance(nothing, mouse));
    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistance(nothing, nothing));

    std::string str1("Monday 12345");
    std::string str2("Monday 67890");
    std::string str3("Sunday 12345");
    std::string str4;

    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistance(str1, str1));
    BOOST_REQUIRE_EQUAL(size_t(5), sst.levenshteinDistance(str1, str2));
    BOOST_REQUIRE_EQUAL(size_t(2), sst.levenshteinDistance(str1, str3));
    BOOST_REQUIRE_EQUAL(size_t(12), sst.levenshteinDistance(str1, str4));

    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistanceEx(str1, str1, &::isdigit));
    BOOST_REQUIRE_EQUAL(size_t(0), sst.levenshteinDistanceEx(str1, str2, &::isdigit));
    BOOST_REQUIRE_EQUAL(size_t(2), sst.levenshteinDistanceEx(str1, str3, &::isdigit));
    BOOST_REQUIRE_EQUAL(size_t(7), sst.levenshteinDistanceEx(str1, str4, &::isdigit));
}

BOOST_AUTO_TEST_CASE(testLevensteinDistance2) {
    ml::core::CStringSimilarityTester sst;

    using TStrVec = std::vector<std::string>;

    TStrVec sourceShutDown1;
    sourceShutDown1.push_back("ml13");
    sourceShutDown1.push_back("4608.1.p2ps");
    sourceShutDown1.push_back("Info");
    sourceShutDown1.push_back("Source");
    sourceShutDown1.push_back("ML_SERVICE2");
    sourceShutDown1.push_back("on");
    sourceShutDown1.push_back("has");
    sourceShutDown1.push_back("shut");
    sourceShutDown1.push_back("down");

    TStrVec sourceShutDown2;
    sourceShutDown2.push_back("ml13");
    sourceShutDown2.push_back("4606.1.p2ps");
    sourceShutDown2.push_back("Info");
    sourceShutDown2.push_back("Source");
    sourceShutDown2.push_back("MONEYBROKER");
    sourceShutDown2.push_back("on");
    sourceShutDown2.push_back("has");
    sourceShutDown2.push_back("shut");
    sourceShutDown2.push_back("down");

    TStrVec serviceStart;
    serviceStart.push_back("ml13");
    serviceStart.push_back("4606.1.p2ps");
    serviceStart.push_back("Info");
    serviceStart.push_back("Service");
    serviceStart.push_back("ML_FEED");
    serviceStart.push_back("id");
    serviceStart.push_back("of");
    serviceStart.push_back("has");
    serviceStart.push_back("started");

    TStrVec noImageData;
    noImageData.push_back("ml13");
    noImageData.push_back("4607.1.p2ps");
    noImageData.push_back("Debug");
    noImageData.push_back("Did");
    noImageData.push_back("not");
    noImageData.push_back("receive");
    noImageData.push_back("an");
    noImageData.push_back("image");
    noImageData.push_back("data");
    noImageData.push_back("for");
    noImageData.push_back("ML_FEED");
    noImageData.push_back("4205.T");
    noImageData.push_back("on");
    noImageData.push_back("Recalling");
    noImageData.push_back("item");

    TStrVec empty;

    BOOST_REQUIRE_EQUAL(size_t(2), sst.levenshteinDistance(sourceShutDown1, sourceShutDown2));
    BOOST_REQUIRE_EQUAL(size_t(2), sst.levenshteinDistance(sourceShutDown2, sourceShutDown1));

    BOOST_REQUIRE_EQUAL(size_t(7), sst.levenshteinDistance(sourceShutDown1, serviceStart));
    BOOST_REQUIRE_EQUAL(size_t(7), sst.levenshteinDistance(serviceStart, sourceShutDown1));

    BOOST_REQUIRE_EQUAL(size_t(6), sst.levenshteinDistance(sourceShutDown2, serviceStart));
    BOOST_REQUIRE_EQUAL(size_t(6), sst.levenshteinDistance(serviceStart, sourceShutDown2));

    BOOST_REQUIRE_EQUAL(size_t(13), sst.levenshteinDistance(noImageData, serviceStart));
    BOOST_REQUIRE_EQUAL(size_t(13), sst.levenshteinDistance(serviceStart, noImageData));

    BOOST_REQUIRE_EQUAL(size_t(14), sst.levenshteinDistance(noImageData, sourceShutDown1));
    BOOST_REQUIRE_EQUAL(size_t(14), sst.levenshteinDistance(sourceShutDown1, noImageData));

    BOOST_REQUIRE_EQUAL(size_t(9), sst.levenshteinDistance(serviceStart, empty));
    BOOST_REQUIRE_EQUAL(size_t(9), sst.levenshteinDistance(empty, serviceStart));
}

BOOST_AUTO_TEST_CASE(testLevensteinDistanceThroughputDifferent) {
    ml::core::CStringSimilarityTester sst;

    using TStrVec = std::vector<std::string>;

    static const size_t TEST_SIZE(700);
    static const int MAX_LEN(40);

    TStrVec input(TEST_SIZE);

    for (size_t index = 0; index < TEST_SIZE; ++index) {
        // Construct the strings from a random number of random lower case
        // letters - empty strings are possible
        for (int len = (::rand() % MAX_LEN); len > 0; --len) {
            input[index] += char('a' + (::rand() % 26));
        }
    }

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting Levenstein distance throughput test for low commonality strings at "
             << ml::core::CTimeUtils::toTimeString(start));

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        for (size_t j = 0; j < TEST_SIZE; ++j) {
            size_t result(sst.levenshteinDistance(input[i], input[j]));
            if (i == j) {
                BOOST_REQUIRE_EQUAL(size_t(0), result);
            }
        }
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished Levenstein distance throughput test for low commonality strings at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Levenstein distance throughput test for low commonality strings with size "
             << TEST_SIZE << " and maximum string length " << MAX_LEN
             << " took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testLevensteinDistanceThroughputSimilar) {
    ml::core::CStringSimilarityTester sst;

    using TStrVec = std::vector<std::string>;

    static const size_t TEST_SIZE(700);
    static const int EXTRA_CHARS(4);

    TStrVec input(TEST_SIZE);

    for (size_t index = 0; index < TEST_SIZE; ++index) {
        // Construct the strings with a large amount of commonality
        for (int count = 0; count < EXTRA_CHARS; ++count) {
            if (index % 2 == 0) {
                input[index] += "common";
            }
            input[index] += char('a' + (::rand() % 26));
            if (index % 2 != 0) {
                input[index] += "common";
            }
        }
    }

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting Levenstein distance throughput test for similar strings at "
             << ml::core::CTimeUtils::toTimeString(start));

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        for (size_t j = 0; j < TEST_SIZE; ++j) {
            size_t result(sst.levenshteinDistance(input[i], input[j]));
            if (i == j) {
                BOOST_REQUIRE_EQUAL(size_t(0), result);
            }
        }
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished Levenstein distance throughput test for similar strings at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Levenstein distance throughput test for similar strings with size "
             << TEST_SIZE << " and " << EXTRA_CHARS << " extra characters took "
             << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testLevensteinDistanceAlgorithmEquivalence) {
    // The intention here is to ensure that the Berghel-Roach algorithm delivers
    // the same results as the simple algorithm.  We take advantage of
    // friendship to call the private implementation methods directly.

    ml::core::CStringSimilarityTester sst;

    std::string cat("cat");
    std::string dog("dog");
    std::string mouse("mouse");
    std::string elephant("elephant");

    // Remember we're calling private implementation methods here that require:
    // 1) Neither input sequence is empty
    // 2) The first input sequence is no longer than the second input sequence
    BOOST_REQUIRE_EQUAL(sst.levenshteinDistanceSimple(cat, cat),
                        sst.berghelRoachEditDistance(cat, cat));
    BOOST_REQUIRE_EQUAL(sst.levenshteinDistanceSimple(cat, dog),
                        sst.berghelRoachEditDistance(cat, dog));
    BOOST_REQUIRE_EQUAL(sst.levenshteinDistanceSimple(cat, mouse),
                        sst.berghelRoachEditDistance(cat, mouse));
    BOOST_REQUIRE_EQUAL(sst.levenshteinDistanceSimple(cat, elephant),
                        sst.berghelRoachEditDistance(cat, elephant));
    BOOST_REQUIRE_EQUAL(sst.levenshteinDistanceSimple(mouse, elephant),
                        sst.berghelRoachEditDistance(mouse, elephant));
}

BOOST_AUTO_TEST_CASE(testWeightedEditDistance) {
    ml::core::CStringSimilarityTester sst;

    using TStrSizePr = std::pair<std::string, size_t>;
    using TStrSizePrVec = std::vector<TStrSizePr>;

    // These tests give a weight of 3 to dictionary words and 1 to other tokens

    TStrSizePrVec sourceShutDown1;
    sourceShutDown1.push_back(TStrSizePr("ml13", 1));
    sourceShutDown1.push_back(TStrSizePr("4608.1.p2ps", 1));
    sourceShutDown1.push_back(TStrSizePr("Info", 3));
    sourceShutDown1.push_back(TStrSizePr("Source", 3));
    sourceShutDown1.push_back(TStrSizePr("ML_SERVICE2", 1));
    sourceShutDown1.push_back(TStrSizePr("on", 3));
    sourceShutDown1.push_back(TStrSizePr("has", 3));
    sourceShutDown1.push_back(TStrSizePr("shut", 3));
    sourceShutDown1.push_back(TStrSizePr("down", 3));

    TStrSizePrVec sourceShutDown2;
    sourceShutDown2.push_back(TStrSizePr("ml13", 1));
    sourceShutDown2.push_back(TStrSizePr("4606.1.p2ps", 1));
    sourceShutDown2.push_back(TStrSizePr("Info", 3));
    sourceShutDown2.push_back(TStrSizePr("Source", 3));
    sourceShutDown2.push_back(TStrSizePr("MONEYBROKER", 1));
    sourceShutDown2.push_back(TStrSizePr("on", 3));
    sourceShutDown2.push_back(TStrSizePr("has", 3));
    sourceShutDown2.push_back(TStrSizePr("shut", 3));
    sourceShutDown2.push_back(TStrSizePr("down", 3));

    TStrSizePrVec serviceStart;
    serviceStart.push_back(TStrSizePr("ml13", 1));
    serviceStart.push_back(TStrSizePr("4606.1.p2ps", 1));
    serviceStart.push_back(TStrSizePr("Info", 3));
    serviceStart.push_back(TStrSizePr("Service", 3));
    serviceStart.push_back(TStrSizePr("ML_FEED", 1));
    serviceStart.push_back(TStrSizePr("id", 3));
    serviceStart.push_back(TStrSizePr("of", 3));
    serviceStart.push_back(TStrSizePr("has", 3));
    serviceStart.push_back(TStrSizePr("started", 3));

    TStrSizePrVec noImageData;
    noImageData.push_back(TStrSizePr("ml13", 1));
    noImageData.push_back(TStrSizePr("4607.1.p2ps", 1));
    noImageData.push_back(TStrSizePr("Debug", 3));
    noImageData.push_back(TStrSizePr("Did", 3));
    noImageData.push_back(TStrSizePr("not", 3));
    noImageData.push_back(TStrSizePr("receive", 3));
    noImageData.push_back(TStrSizePr("an", 3));
    noImageData.push_back(TStrSizePr("image", 3));
    noImageData.push_back(TStrSizePr("data", 3));
    noImageData.push_back(TStrSizePr("for", 3));
    noImageData.push_back(TStrSizePr("ML_FEED", 1));
    noImageData.push_back(TStrSizePr("4205.T", 1));
    noImageData.push_back(TStrSizePr("on", 3));
    noImageData.push_back(TStrSizePr("Recalling", 3));
    noImageData.push_back(TStrSizePr("item", 3));

    TStrSizePrVec empty;

    BOOST_REQUIRE_EQUAL(size_t(2), sst.weightedEditDistance(sourceShutDown1, sourceShutDown2));
    BOOST_REQUIRE_EQUAL(size_t(2), sst.weightedEditDistance(sourceShutDown2, sourceShutDown1));

    BOOST_REQUIRE_EQUAL(size_t(17), sst.weightedEditDistance(sourceShutDown1, serviceStart));
    BOOST_REQUIRE_EQUAL(size_t(17), sst.weightedEditDistance(serviceStart, sourceShutDown1));

    BOOST_REQUIRE_EQUAL(size_t(16), sst.weightedEditDistance(sourceShutDown2, serviceStart));
    BOOST_REQUIRE_EQUAL(size_t(16), sst.weightedEditDistance(serviceStart, sourceShutDown2));

    BOOST_REQUIRE_EQUAL(size_t(36), sst.weightedEditDistance(noImageData, serviceStart));
    BOOST_REQUIRE_EQUAL(size_t(36), sst.weightedEditDistance(serviceStart, noImageData));

    BOOST_REQUIRE_EQUAL(size_t(36), sst.weightedEditDistance(noImageData, sourceShutDown1));
    BOOST_REQUIRE_EQUAL(size_t(36), sst.weightedEditDistance(sourceShutDown1, noImageData));

    BOOST_REQUIRE_EQUAL(size_t(21), sst.weightedEditDistance(serviceStart, empty));
    BOOST_REQUIRE_EQUAL(size_t(21), sst.weightedEditDistance(empty, serviceStart));
}

BOOST_AUTO_TEST_SUITE_END()
