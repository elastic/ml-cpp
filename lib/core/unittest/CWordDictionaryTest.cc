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

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CWordDictionary.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CWordDictionaryTest)

BOOST_AUTO_TEST_CASE(testLookups) {
    const ml::core::CWordDictionary& dict = ml::core::CWordDictionary::instance();

    BOOST_TEST_REQUIRE(dict.isInDictionary("hello"));
    BOOST_TEST_REQUIRE(dict.isInDictionary("Hello"));
    BOOST_TEST_REQUIRE(dict.isInDictionary("HELLO"));
    BOOST_TEST_REQUIRE(dict.isInDictionary("service"));
    BOOST_TEST_REQUIRE(dict.isInDictionary("has"));
    BOOST_TEST_REQUIRE(dict.isInDictionary("started"));

    BOOST_TEST_REQUIRE(!dict.isInDictionary(""));
    BOOST_TEST_REQUIRE(!dict.isInDictionary("r"));
    BOOST_TEST_REQUIRE(!dict.isInDictionary("hkjsdfg"));
    BOOST_TEST_REQUIRE(!dict.isInDictionary("hello2"));
    BOOST_TEST_REQUIRE(!dict.isInDictionary("HELLO2"));
}

BOOST_AUTO_TEST_CASE(testPartOfSpeech) {
    const ml::core::CWordDictionary& dict = ml::core::CWordDictionary::instance();

    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_NotInDictionary,
                        dict.partOfSpeech("ajksdf"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_UnknownPart,
                        dict.partOfSpeech("callback"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Noun, dict.partOfSpeech("House"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Plural, dict.partOfSpeech("Houses"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Verb, dict.partOfSpeech("COMPLETED"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Adjective, dict.partOfSpeech("heavy"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Adverb, dict.partOfSpeech("slowly"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Conjunction, dict.partOfSpeech("AND"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Preposition,
                        dict.partOfSpeech("without"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Interjection,
                        dict.partOfSpeech("gosh"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_Pronoun, dict.partOfSpeech("hers"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_DefiniteArticle,
                        dict.partOfSpeech("the"));
    BOOST_REQUIRE_EQUAL(ml::core::CWordDictionary::E_IndefiniteArticle,
                        dict.partOfSpeech("a"));
}

BOOST_AUTO_TEST_CASE(testSimpleWeightingFunctors) {
    {
        ml::core::CWordDictionary::TWeightAll2 weighter;

        BOOST_REQUIRE_EQUAL(0, weighter(ml::core::CWordDictionary::E_NotInDictionary));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_UnknownPart));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Noun));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Plural));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Verb));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Adjective));
        weighter.reset(); // should make no difference
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Adverb));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Conjunction));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Preposition));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Interjection));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Pronoun));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_DefiniteArticle));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_IndefiniteArticle));
        // Any given token always gives the same weight, so min/max matching
        // should always be the same as the original
        for (std::size_t weight = 1; weight < 10; ++weight) {
            BOOST_REQUIRE_EQUAL(weight, weighter.minMatchingWeight(weight));
            BOOST_REQUIRE_EQUAL(weight, weighter.maxMatchingWeight(weight));
        }
    }
    {
        ml::core::CWordDictionary::TWeightVerbs5Other2 weighter;

        BOOST_REQUIRE_EQUAL(0, weighter(ml::core::CWordDictionary::E_NotInDictionary));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_UnknownPart));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Noun));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Plural));
        weighter.reset(); // should make no difference
        BOOST_REQUIRE_EQUAL(5, weighter(ml::core::CWordDictionary::E_Verb));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Adjective));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Adverb));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Conjunction));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Preposition));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Interjection));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Pronoun));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_DefiniteArticle));
        BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_IndefiniteArticle));
        // Any given token always gives the same weight, so min/max matching
        // should always be the same as the original
        for (std::size_t weight = 1; weight < 10; ++weight) {
            BOOST_REQUIRE_EQUAL(weight, weighter.minMatchingWeight(weight));
            BOOST_REQUIRE_EQUAL(weight, weighter.maxMatchingWeight(weight));
        }
    }
}

BOOST_AUTO_TEST_CASE(testAdjacencyDependentWeightingFunctor) {
    ml::core::CWordDictionary::TWeightVerbs5Other2AdjacentBoost6 weighter;

    BOOST_REQUIRE_EQUAL(0, weighter(ml::core::CWordDictionary::E_NotInDictionary));
    BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_UnknownPart));
    BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Noun));
    BOOST_REQUIRE_EQUAL(12, weighter(ml::core::CWordDictionary::E_Plural));
    BOOST_REQUIRE_EQUAL(30, weighter(ml::core::CWordDictionary::E_Verb));
    weighter.reset();
    // Explicit reset stops adjacency multiplier
    BOOST_REQUIRE_EQUAL(5, weighter(ml::core::CWordDictionary::E_Verb));
    BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Adjective));
    BOOST_REQUIRE_EQUAL(12, weighter(ml::core::CWordDictionary::E_Adverb));
    BOOST_REQUIRE_EQUAL(12, weighter(ml::core::CWordDictionary::E_Conjunction));
    BOOST_REQUIRE_EQUAL(0, weighter(ml::core::CWordDictionary::E_NotInDictionary));
    // Non-dictionary word stops adjacency multiplier
    BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Noun));
    BOOST_REQUIRE_EQUAL(5, weighter(ml::core::CWordDictionary::E_Verb));
    weighter.reset();
    // Explicit reset stops adjacency multiplier
    BOOST_REQUIRE_EQUAL(2, weighter(ml::core::CWordDictionary::E_Adjective));

    // Of the possible weights, 3 could map to 13 and 6 to 31 depending on
    // whether adjacency weighting takes place
    BOOST_REQUIRE_EQUAL(1, weighter.minMatchingWeight(1));
    BOOST_REQUIRE_EQUAL(1, weighter.maxMatchingWeight(1));
    BOOST_REQUIRE_EQUAL(3, weighter.minMatchingWeight(3));
    BOOST_REQUIRE_EQUAL(13, weighter.maxMatchingWeight(3));
    BOOST_REQUIRE_EQUAL(6, weighter.minMatchingWeight(6));
    BOOST_REQUIRE_EQUAL(31, weighter.maxMatchingWeight(6));
    BOOST_REQUIRE_EQUAL(3, weighter.minMatchingWeight(13));
    BOOST_REQUIRE_EQUAL(13, weighter.maxMatchingWeight(13));
    BOOST_REQUIRE_EQUAL(6, weighter.minMatchingWeight(31));
    BOOST_REQUIRE_EQUAL(31, weighter.maxMatchingWeight(31));
}

// Disabled because it doesn't assert anything
// Can be run on an ad hoc basis if performance is of interest
BOOST_AUTO_TEST_CASE(testPerformance, *boost::unit_test::disabled()) {
    const ml::core::CWordDictionary& dict = ml::core::CWordDictionary::instance();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting word dictionary throughput test at "
             << ml::core::CTimeUtils::toTimeString(start));

    static const std::size_t TEST_SIZE(100000);
    for (std::size_t count = 0; count < TEST_SIZE; ++count) {
        dict.isInDictionary("hello");
        dict.isInDictionary("Hello");
        dict.isInDictionary("HELLO");
        dict.isInDictionary("service");
        dict.isInDictionary("has");
        dict.isInDictionary("started");

        dict.isInDictionary("");
        dict.isInDictionary("r");
        dict.isInDictionary("hkjsdfg");
        dict.isInDictionary("hello2");
        dict.isInDictionary("HELLO2");
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished word dictionary throughput test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Word dictionary throughput test took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_SUITE_END()
