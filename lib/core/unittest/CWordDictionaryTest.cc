/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CWordDictionary.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CWordDictionaryTest)

BOOST_AUTO_TEST_CASE(testLookups) {
    const ml::core::CWordDictionary& dict = ml::core::CWordDictionary::instance();

    BOOST_TEST(dict.isInDictionary("hello"));
    BOOST_TEST(dict.isInDictionary("Hello"));
    BOOST_TEST(dict.isInDictionary("HELLO"));
    BOOST_TEST(dict.isInDictionary("service"));
    BOOST_TEST(dict.isInDictionary("has"));
    BOOST_TEST(dict.isInDictionary("started"));

    BOOST_TEST(!dict.isInDictionary(""));
    BOOST_TEST(!dict.isInDictionary("r"));
    BOOST_TEST(!dict.isInDictionary("hkjsdfg"));
    BOOST_TEST(!dict.isInDictionary("hello2"));
    BOOST_TEST(!dict.isInDictionary("HELLO2"));
}

BOOST_AUTO_TEST_CASE(testPartOfSpeech) {
    const ml::core::CWordDictionary& dict = ml::core::CWordDictionary::instance();

    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_NotInDictionary,
                      dict.partOfSpeech("ajksdf"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_UnknownPart,
                      dict.partOfSpeech("callback"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Noun, dict.partOfSpeech("House"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Plural, dict.partOfSpeech("Houses"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Verb, dict.partOfSpeech("COMPLETED"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Adjective, dict.partOfSpeech("heavy"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Adverb, dict.partOfSpeech("slowly"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Conjunction, dict.partOfSpeech("AND"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Preposition, dict.partOfSpeech("without"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Interjection, dict.partOfSpeech("gosh"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_Pronoun, dict.partOfSpeech("hers"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_DefiniteArticle,
                      dict.partOfSpeech("the"));
    BOOST_CHECK_EQUAL(ml::core::CWordDictionary::E_IndefiniteArticle,
                      dict.partOfSpeech("a"));
}

BOOST_AUTO_TEST_CASE(testWeightingFunctors) {
    {
        ml::core::CWordDictionary::TWeightAll2 weighter;

        BOOST_CHECK_EQUAL(size_t(0), weighter(ml::core::CWordDictionary::E_NotInDictionary));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_UnknownPart));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Noun));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Plural));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Verb));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Adjective));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Adverb));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Conjunction));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Preposition));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Interjection));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Pronoun));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_DefiniteArticle));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_IndefiniteArticle));
    }
    {
        ml::core::CWordDictionary::TWeightVerbs5Other2 weighter;

        BOOST_CHECK_EQUAL(size_t(0), weighter(ml::core::CWordDictionary::E_NotInDictionary));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_UnknownPart));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Noun));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Plural));
        BOOST_CHECK_EQUAL(size_t(5), weighter(ml::core::CWordDictionary::E_Verb));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Adjective));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Adverb));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Conjunction));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Preposition));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Interjection));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_Pronoun));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_DefiniteArticle));
        BOOST_CHECK_EQUAL(size_t(2), weighter(ml::core::CWordDictionary::E_IndefiniteArticle));
    }
}

BOOST_AUTO_TEST_CASE(testPerformance) {
    const ml::core::CWordDictionary& dict = ml::core::CWordDictionary::instance();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting word dictionary throughput test at "
             << ml::core::CTimeUtils::toTimeString(start));

    static const size_t TEST_SIZE(100000);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
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
