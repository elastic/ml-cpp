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
#include "CFlatPrefixTreeTest.h"

#include <core/CFlatPrefixTree.h>
#include <core/CLogger.h>

#include <test/CRandomNumbers.h>

#include <boost/unordered_set.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <stdint.h>

using namespace ml;
using namespace core;

CppUnit::Test* CFlatPrefixTreeTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CFlatPrefixTreeTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testBuildGivenUnsortedInput",
                                                                       &CFlatPrefixTreeTest::testBuildGivenUnsortedInput));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testBuildGivenSortedInputWithDuplicates",
                                                                       &CFlatPrefixTreeTest::testBuildGivenSortedInputWithDuplicates));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testEmptyString", &CFlatPrefixTreeTest::testEmptyString));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testSimple", &CFlatPrefixTreeTest::testSimple));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testLeafAndBranch", &CFlatPrefixTreeTest::testLeafAndBranch));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testMatchesStartGivenStringThatMatchesMoreThanAGivenPrefix",
                                                     &CFlatPrefixTreeTest::testMatchesStartGivenStringThatMatchesMoreThanAGivenPrefix));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testMatchesFullyGivenStringThatIsSubstringOfPrefix",
                                                     &CFlatPrefixTreeTest::testMatchesFullyGivenStringThatIsSubstringOfPrefix));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFlatPrefixTreeTest>("CFlatPrefixTreeTest::testRandom", &CFlatPrefixTreeTest::testRandom));

    return suiteOfTests;
}

void CFlatPrefixTreeTest::testBuildGivenUnsortedInput(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("b");
    prefixes.push_back("a");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes) == false);
}

void CFlatPrefixTreeTest::testBuildGivenSortedInputWithDuplicates(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("a");
    prefixes.push_back("bb");
    prefixes.push_back("bb");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes) == false);
}

void CFlatPrefixTreeTest::testEmptyString(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes));

    CPPUNIT_ASSERT(prefixTree.matchesFully("") == false);
    CPPUNIT_ASSERT(prefixTree.matchesStart("") == false);
}

void CFlatPrefixTreeTest::testSimple(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("abc");
    prefixes.push_back("acd");
    prefixes.push_back("aqrs");
    prefixes.push_back("aqrt");
    prefixes.push_back("bdf");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes));

    LOG_TRACE("Tree: " << prefixTree.print());

    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        CPPUNIT_ASSERT(prefixTree.matchesStart(prefixes[i]));
        CPPUNIT_ASSERT(prefixTree.matchesFully(prefixes[i]));
    }

    CPPUNIT_ASSERT(prefixTree.matchesStart("abcz"));
    CPPUNIT_ASSERT(prefixTree.matchesStart("acdz"));
    CPPUNIT_ASSERT(prefixTree.matchesStart("aqrsz"));
    CPPUNIT_ASSERT(prefixTree.matchesStart("aqrtz"));
    CPPUNIT_ASSERT(prefixTree.matchesStart("bdfz"));
    CPPUNIT_ASSERT(prefixTree.matchesFully("abcd") == false);
    CPPUNIT_ASSERT(prefixTree.matchesStart("ab") == false);
    CPPUNIT_ASSERT(prefixTree.matchesStart("c") == false);
    CPPUNIT_ASSERT(prefixTree.matchesStart("") == false);
    CPPUNIT_ASSERT(prefixTree.matchesFully("") == false);
}

void CFlatPrefixTreeTest::testLeafAndBranch(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back(" oqt4g");
    prefixes.push_back(" oqt4glz-");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes));

    LOG_TRACE("Tree: " << prefixTree.print());

    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        CPPUNIT_ASSERT(prefixTree.matchesStart(prefixes[i]));
        CPPUNIT_ASSERT(prefixTree.matchesFully(prefixes[i]));
    }
}

void CFlatPrefixTreeTest::testMatchesStartGivenStringThatMatchesMoreThanAGivenPrefix(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("h2 vh5");
    prefixes.push_back("h2 vh55dm");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes));

    LOG_TRACE("Tree: " << prefixTree.print());

    CPPUNIT_ASSERT(prefixTree.matchesStart("h2 vh5"));
    CPPUNIT_ASSERT(prefixTree.matchesStart("h2 vh55daetrqt4"));
}

void CFlatPrefixTreeTest::testMatchesFullyGivenStringThatIsSubstringOfPrefix(void) {
    CFlatPrefixTree::TStrVec prefixes;
    prefixes.push_back("foo");

    CFlatPrefixTree prefixTree;
    CPPUNIT_ASSERT(prefixTree.build(prefixes));

    LOG_TRACE("Tree: " << prefixTree.print());

    CPPUNIT_ASSERT(prefixTree.matchesFully("fo") == false);
}

void CFlatPrefixTreeTest::testRandom(void) {
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
    CPPUNIT_ASSERT(prefixTree.build(prefixes));

    // Assert full lookups
    {
        boost::unordered_set<std::string> set(prefixes.begin(), prefixes.end());

        CFlatPrefixTree::TStrVec lookups;
        rng.generateWords(10, 200000, lookups);
        lookups.insert(lookups.end(), prefixes.begin(), prefixes.end());

        for (std::size_t i = 0; i < lookups.size(); ++i) {
            CPPUNIT_ASSERT(prefixTree.matchesFully(lookups[i]) == set.count(lookups[i]) > 0);
        }
    }

    // Assert startsWith lookups
    {
        CFlatPrefixTree::TStrVec suffixes;
        rng.generateWords(10, 1000, suffixes);
        for (std::size_t i = 0; i < 100000; i++) {
            std::string key = prefixes[uniformGen(prefixes.size())] + suffixes[uniformGen(suffixes.size())];
            CPPUNIT_ASSERT(prefixTree.matchesStart(key));
        }
    }
}
