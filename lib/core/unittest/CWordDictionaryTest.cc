/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CWordDictionaryTest.h"

#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CWordDictionary.h>


CppUnit::Test *CWordDictionaryTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CWordDictionaryTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CWordDictionaryTest>(
                                   "CWordDictionaryTest::testLookups",
                                   &CWordDictionaryTest::testLookups) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CWordDictionaryTest>(
                                   "CWordDictionaryTest::testPartOfSpeech",
                                   &CWordDictionaryTest::testPartOfSpeech) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CWordDictionaryTest>(
                                   "CWordDictionaryTest::testWeightingFunctors",
                                   &CWordDictionaryTest::testWeightingFunctors) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CWordDictionaryTest>(
                                   "CWordDictionaryTest::testPerformance",
                                   &CWordDictionaryTest::testPerformance) );

    return suiteOfTests;
}

void CWordDictionaryTest::testLookups()
{
    const ml::core::CWordDictionary &dict = ml::core::CWordDictionary::instance();

    CPPUNIT_ASSERT(dict.isInDictionary("hello"));
    CPPUNIT_ASSERT(dict.isInDictionary("Hello"));
    CPPUNIT_ASSERT(dict.isInDictionary("HELLO"));
    CPPUNIT_ASSERT(dict.isInDictionary("service"));
    CPPUNIT_ASSERT(dict.isInDictionary("has"));
    CPPUNIT_ASSERT(dict.isInDictionary("started"));

    CPPUNIT_ASSERT(!dict.isInDictionary(""));
    CPPUNIT_ASSERT(!dict.isInDictionary("r"));
    CPPUNIT_ASSERT(!dict.isInDictionary("hkjsdfg"));
    CPPUNIT_ASSERT(!dict.isInDictionary("hello2"));
    CPPUNIT_ASSERT(!dict.isInDictionary("HELLO2"));
}

void CWordDictionaryTest::testPartOfSpeech()
{
    const ml::core::CWordDictionary &dict = ml::core::CWordDictionary::instance();

    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_NotInDictionary,
                         dict.partOfSpeech("ajksdf"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_UnknownPart,
                         dict.partOfSpeech("callback"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Noun,
                         dict.partOfSpeech("House"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Plural,
                         dict.partOfSpeech("Houses"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Verb,
                         dict.partOfSpeech("COMPLETED"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Adjective,
                         dict.partOfSpeech("heavy"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Adverb,
                         dict.partOfSpeech("slowly"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Conjunction,
                         dict.partOfSpeech("AND"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Preposition,
                         dict.partOfSpeech("without"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Interjection,
                         dict.partOfSpeech("gosh"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_Pronoun,
                         dict.partOfSpeech("hers"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_DefiniteArticle,
                         dict.partOfSpeech("the"));
    CPPUNIT_ASSERT_EQUAL(ml::core::CWordDictionary::E_IndefiniteArticle,
                         dict.partOfSpeech("a"));
}

void CWordDictionaryTest::testWeightingFunctors()
{
    {
        ml::core::CWordDictionary::TWeightAll2 weighter;

        CPPUNIT_ASSERT_EQUAL(size_t(0),
                             weighter(ml::core::CWordDictionary::E_NotInDictionary));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_UnknownPart));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Noun));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Plural));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Verb));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Adjective));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Adverb));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Conjunction));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Preposition));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Interjection));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Pronoun));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_DefiniteArticle));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_IndefiniteArticle));
    }
    {
        ml::core::CWordDictionary::TWeightVerbs5Other2 weighter;

        CPPUNIT_ASSERT_EQUAL(size_t(0),
                             weighter(ml::core::CWordDictionary::E_NotInDictionary));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_UnknownPart));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Noun));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Plural));
        CPPUNIT_ASSERT_EQUAL(size_t(5),
                             weighter(ml::core::CWordDictionary::E_Verb));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Adjective));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Adverb));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Conjunction));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Preposition));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Interjection));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_Pronoun));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_DefiniteArticle));
        CPPUNIT_ASSERT_EQUAL(size_t(2),
                             weighter(ml::core::CWordDictionary::E_IndefiniteArticle));
    }
}

void CWordDictionaryTest::testPerformance()
{
    const ml::core::CWordDictionary &dict = ml::core::CWordDictionary::instance();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting word dictionary throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    static const size_t TEST_SIZE(100000);
    for (size_t count = 0; count < TEST_SIZE; ++count)
    {
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
    LOG_INFO("Finished word dictionary throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO("Word dictionary throughput test took " << (end - start) <<
             " seconds");
}

