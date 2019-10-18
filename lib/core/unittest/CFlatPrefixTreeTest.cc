/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CFlatPrefixTree.h>
#include <core/CLogger.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_set.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <stdint.h>

BOOST_AUTO_TEST_SUITE(CFlatPrefixTreeTest)

using namespace ml;
using namespace core;

BOOST_AUTO_TEST_CASE(testBuildGivenUnsortedInput) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("b");
    prefixes.push_back("a");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes) == false);
}

BOOST_AUTO_TEST_CASE(testBuildGivenSortedInputWithDuplicates) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("a");
    prefixes.push_back("bb");
    prefixes.push_back("bb");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes) == false);
}

BOOST_AUTO_TEST_CASE(testEmptyString) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes));

    BOOST_TEST(prefixTree.matchesFully("") == false);
    BOOST_TEST(prefixTree.matchesStart("") == false);
}

BOOST_AUTO_TEST_CASE(testSimple) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("abc");
    prefixes.push_back("acd");
    prefixes.push_back("aqrs");
    prefixes.push_back("aqrt");
    prefixes.push_back("bdf");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes));

    LOG_TRACE(<< "Tree: " << prefixTree.print());

    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        BOOST_TEST(prefixTree.matchesStart(prefixes[i]));
        BOOST_TEST(prefixTree.matchesFully(prefixes[i]));
    }

    BOOST_TEST(prefixTree.matchesStart("abcz"));
    BOOST_TEST(prefixTree.matchesStart("acdz"));
    BOOST_TEST(prefixTree.matchesStart("aqrsz"));
    BOOST_TEST(prefixTree.matchesStart("aqrtz"));
    BOOST_TEST(prefixTree.matchesStart("bdfz"));
    BOOST_TEST(prefixTree.matchesFully("abcd") == false);
    BOOST_TEST(prefixTree.matchesStart("ab") == false);
    BOOST_TEST(prefixTree.matchesStart("c") == false);
    BOOST_TEST(prefixTree.matchesStart("") == false);
    BOOST_TEST(prefixTree.matchesFully("") == false);
}

BOOST_AUTO_TEST_CASE(testLeafAndBranch) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back(" oqt4g");
    prefixes.push_back(" oqt4glz-");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes));

    LOG_TRACE(<< "Tree: " << prefixTree.print());

    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        BOOST_TEST(prefixTree.matchesStart(prefixes[i]));
        BOOST_TEST(prefixTree.matchesFully(prefixes[i]));
    }
}

BOOST_AUTO_TEST_CASE(testMatchesStartGivenStringThatMatchesMoreThanAGivenPrefix) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("h2 vh5");
    prefixes.push_back("h2 vh55dm");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes));

    LOG_TRACE(<< "Tree: " << prefixTree.print());

    BOOST_TEST(prefixTree.matchesStart("h2 vh5"));
    BOOST_TEST(prefixTree.matchesStart("h2 vh55daetrqt4"));
}

BOOST_AUTO_TEST_CASE(testMatchesFullyGivenStringThatIsSubstringOfPrefix) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("foo");

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes));

    LOG_TRACE(<< "Tree: " << prefixTree.print());

    BOOST_TEST(prefixTree.matchesFully("fo") == false);
}

BOOST_AUTO_TEST_CASE(testRandom) {
    test::CRandomNumbers rng;
    test::CRandomNumbers::CUniform0nGenerator uniformGen = rng.uniformGenerator();
    CFlatPrefixTree::TStrVec grams;
    rng.generateWords(3, 200, grams);
    CFlatPrefixTree::TStrVec prefixes;
    for (std::size_t i = 0; i < grams.size(); ++i) {
        for (std::size_t j = 0; j < grams.size(); ++j) {
            prefixes.push_back(grams[i] + grams[j]);
            std::size_t n = uniformGen(5);
            for (std::size_t k = 0; k < n; ++k) {
                prefixes.back() += grams[uniformGen(grams.size())];
            }
        }
    }

    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());

    CFlatPrefixTree prefixTree;
    BOOST_TEST(prefixTree.build(prefixes));

    // Assert full lookups
    {
        boost::unordered_set<std::string> set(prefixes.begin(), prefixes.end());

        CFlatPrefixTree::TStrVec lookups;
        rng.generateWords(10, 200000, lookups);
        lookups.insert(lookups.end(), prefixes.begin(), prefixes.end());

        for (std::size_t i = 0; i < lookups.size(); ++i) {
            BOOST_TEST(prefixTree.matchesFully(lookups[i]) == (set.count(lookups[i]) > 0));
        }
    }

    // Assert startsWith lookups
    {
        CFlatPrefixTree::TStrVec suffixes;
        rng.generateWords(10, 1000, suffixes);
        for (std::size_t i = 0; i < 100000; i++) {
            std::string key = prefixes[uniformGen(prefixes.size())] +
                              suffixes[uniformGen(suffixes.size())];
            BOOST_TEST(prefixTree.matchesStart(key));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
