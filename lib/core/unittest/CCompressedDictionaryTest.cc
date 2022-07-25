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

#include <core/CCompressedDictionary.h>
#include <core/CLogger.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_set.hpp>

using namespace ml;
using namespace core;
using namespace test;

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TDictionary1 = CCompressedDictionary<1>;
using TDictionary2 = CCompressedDictionary<2>;
using TDictionary3 = CCompressedDictionary<3>;
using TDictionary4 = CCompressedDictionary<4>;
using TWordUSet = TDictionary2::TWordUSet;

BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary1::CWord)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary2::CWord)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary3::CWord)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary4::CWord)

BOOST_AUTO_TEST_SUITE(CCompressedDictionaryTest)

BOOST_AUTO_TEST_CASE(testCollisions) {

    TDictionary2 dictionary;

    // Test the same string maps to the same word.

    auto expected = dictionary.word("word");
    auto actual = dictionary.word("word");
    BOOST_REQUIRE_EQUAL(expected, actual);

    // Don't set this too high as it slows down every build - it can be
    // temporarily set high in uncommitted code for a thorough soak test
    // following changes to the class being tested
    const std::size_t numberTests{10};
    const std::size_t wordLength{16};
    const std::size_t numberWords{500000};

    // Test we don't get any collisions in 10 runs with 500000 distinct words.

    CRandomNumbers rng;
    TStrVec words;

    std::string word2("word2");
    std::string word3("word3");

    for (std::size_t i = 0; i < numberTests; ++i) {
        LOG_DEBUG(<< "Collision test = " << i);

        rng.generateWords(wordLength, numberWords, words);

        TWordUSet uniqueWords;
        for (const auto& word1 : words) {
            BOOST_TEST_REQUIRE(uniqueWords.insert(dictionary.word(word1)).second);
            BOOST_TEST_REQUIRE(uniqueWords.insert(dictionary.word(word1, word2)).second);
            BOOST_TEST_REQUIRE(
                uniqueWords.insert(dictionary.word(word1, word2, word3)).second);
        }
    }
}

BOOST_AUTO_TEST_CASE(testTranslate) {

    TDictionary2 dictionary;

    CRandomNumbers rng;

    TDoubleVec values;
    rng.generateNormalSamples(0.0, 1.0, 5, values);

    // Check the same vector maps to the same word.

    auto translator1 = dictionary.translator();
    translator1.add(values);
    auto word1 = translator1.word();

    auto translator2 = dictionary.translator();
    translator2.add(values);
    auto word2 = translator2.word();

    BOOST_REQUIRE_EQUAL(word1, word2);

    // Test we don't get any collisions with 500000 distinct vectors.

    TWordUSet uniqueWords;
    for (std::size_t i = 0; i < 500000; ++i) {
        auto translator = dictionary.translator();
        rng.generateNormalSamples(0.0, 1.0, 5, values);
        translator.add(values);
        BOOST_TEST_REQUIRE(uniqueWords.insert(translator.word()).second);
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    {
        TDictionary1 dictionary;
        TDictionary1::CWord word = dictionary.word("hello");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST_REQUIRE(dictionary.word("special") != word);
        BOOST_TEST_REQUIRE(word.fromDelimited(representation));
        BOOST_TEST_REQUIRE(dictionary.word("hello") == word);
    }
    {
        TDictionary2 dictionary;
        TDictionary2::CWord word = dictionary.word("world");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST_REQUIRE(dictionary.word("special") != word);
        BOOST_TEST_REQUIRE(word.fromDelimited(representation));
        BOOST_TEST_REQUIRE(dictionary.word("world") == word);
    }
    {
        TDictionary3 dictionary;
        TDictionary3::CWord word = dictionary.word("special");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST_REQUIRE(dictionary.word("special") != word);
        BOOST_TEST_REQUIRE(word.fromDelimited(representation));
        BOOST_TEST_REQUIRE(dictionary.word("special") == word);
    }
    {
        TDictionary4 dictionary;
        TDictionary4::CWord word = dictionary.word("TEST");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST_REQUIRE(dictionary.word("special") != word);
        BOOST_TEST_REQUIRE(word.fromDelimited(representation));
        BOOST_TEST_REQUIRE(dictionary.word("TEST") == word);
    }
}

BOOST_AUTO_TEST_SUITE_END()
