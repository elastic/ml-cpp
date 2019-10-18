/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CCompressedDictionary.h>
#include <core/CLogger.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

using namespace ml;
using namespace core;
using namespace test;

using TDictionary1 = CCompressedDictionary<1>;
using TDictionary2 = CCompressedDictionary<2>;
using TDictionary3 = CCompressedDictionary<3>;
using TDictionary4 = CCompressedDictionary<4>;

BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary1::CWord)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary2::CWord)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary3::CWord)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDictionary4::CWord)

BOOST_AUTO_TEST_SUITE(CCompressedDictionaryTest)

BOOST_AUTO_TEST_CASE(testAll) {
    using TStrVec = std::vector<std::string>;
    using TWordUSet = TDictionary2::TWordUSet;

    // Don't set this too high as it slows down every build - it can be
    // temporarily set high in uncommitted code for a thorough soak test
    // following changes to the class being tested
    const std::size_t numberTests = 10u;
    const std::size_t wordLength = 16u;
    const std::size_t numberWords = 500000u;

    CRandomNumbers rng;
    TStrVec words;

    std::string word2("word2");
    std::string word3("word3");

    for (std::size_t i = 0u; i < numberTests; ++i) {
        LOG_DEBUG(<< "Collision test = " << i);

        rng.generateWords(wordLength, numberWords, words);

        TDictionary2 dictionary;

        TWordUSet uniqueWords;
        for (std::size_t j = 0u; j < words.size(); ++j) {
            BOOST_TEST(uniqueWords.insert(dictionary.word(words[j])).second);
            BOOST_TEST(uniqueWords.insert(dictionary.word(words[j], word2)).second);
            BOOST_TEST(
                uniqueWords.insert(dictionary.word(words[j], word2, word3)).second);
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    {
        TDictionary1 dictionary;
        TDictionary1::CWord word = dictionary.word("hello");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST(dictionary.word("special") != word);
        BOOST_TEST(word.fromDelimited(representation));
        BOOST_TEST(dictionary.word("hello") == word);
    }
    {
        TDictionary2 dictionary;
        TDictionary2::CWord word = dictionary.word("world");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST(dictionary.word("special") != word);
        BOOST_TEST(word.fromDelimited(representation));
        BOOST_TEST(dictionary.word("world") == word);
    }
    {
        TDictionary3 dictionary;
        TDictionary3::CWord word = dictionary.word("special");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST(dictionary.word("special") != word);
        BOOST_TEST(word.fromDelimited(representation));
        BOOST_TEST(dictionary.word("special") == word);
    }
    {
        TDictionary4 dictionary;
        TDictionary4::CWord word = dictionary.word("TEST");
        const std::string representation = word.toDelimited();
        word = dictionary.word("blank");
        BOOST_TEST(dictionary.word("special") != word);
        BOOST_TEST(word.fromDelimited(representation));
        BOOST_TEST(dictionary.word("TEST") == word);
    }
}

BOOST_AUTO_TEST_SUITE_END()
