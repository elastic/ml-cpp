/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CCategoryExamplesCollector.h>

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CCategoryExamplesCollectorTest)

using namespace ml;
using namespace api;

BOOST_AUTO_TEST_CASE(testAddGivenMaxExamplesIsZero) {
    CCategoryExamplesCollector examplesCollector(0);
    BOOST_TEST(examplesCollector.add(1, "foo") == false);
    BOOST_TEST(examplesCollector.add(2, "foo") == false);
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(0));
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(2), std::size_t(0));
}

BOOST_AUTO_TEST_CASE(testAddGivenSameCategoryExamplePairAddedTwice) {
    CCategoryExamplesCollector examplesCollector(4);
    BOOST_TEST(examplesCollector.add(1, "foo") == true);
    BOOST_TEST(examplesCollector.add(1, "foo") == false);
}

BOOST_AUTO_TEST_CASE(testAddGivenMoreThanMaxExamplesAreAddedForSameCategory) {
    CCategoryExamplesCollector examplesCollector(3);
    BOOST_TEST(examplesCollector.add(1, "foo1") == true);
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(1));
    BOOST_TEST(examplesCollector.add(1, "foo2") == true);
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(2));
    BOOST_TEST(examplesCollector.add(1, "foo3") == true);
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(3));
    BOOST_TEST(examplesCollector.add(1, "foo4") == false);
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(3));
}

BOOST_AUTO_TEST_CASE(testAddGivenCategoryAddedIsNotSubsequent) {
    CCategoryExamplesCollector examplesCollector(2);
    BOOST_TEST(examplesCollector.add(1, "foo") == true);
    BOOST_TEST(examplesCollector.add(3, "bar") == true);
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(1));
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(2), std::size_t(0));
    BOOST_CHECK_EQUAL(examplesCollector.numberOfExamplesForCategory(3), std::size_t(1));
}

BOOST_AUTO_TEST_CASE(testExamples) {
    CCategoryExamplesCollector examplesCollector(3);
    examplesCollector.add(1, "foo");
    examplesCollector.add(1, "bar");
    examplesCollector.add(2, "foo");

    CCategoryExamplesCollector::TStrSet examples1 = examplesCollector.examples(1);
    BOOST_TEST(examples1.find("foo") != examples1.end());
    BOOST_TEST(examples1.find("bar") != examples1.end());
    BOOST_TEST(examples1.find("invalid") == examples1.end());

    CCategoryExamplesCollector::TStrSet examples2 = examplesCollector.examples(2);
    BOOST_TEST(examples2.find("foo") != examples2.end());
    BOOST_TEST(examples2.find("invalid") == examples2.end());
}

BOOST_AUTO_TEST_CASE(testPersist) {
    CCategoryExamplesCollector examplesCollector(3);
    examplesCollector.add(1, "foo");
    examplesCollector.add(1, "bar");
    examplesCollector.add(1, "foobar");
    examplesCollector.add(2, "baz");
    examplesCollector.add(2, "qux");
    examplesCollector.add(3, "quux");

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        examplesCollector.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE(<< "XML:\n" << origXml);

    core::CRapidXmlParser parser;
    BOOST_TEST(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CCategoryExamplesCollector restoredExamplesCollector(3, traverser);

    BOOST_TEST(restoredExamplesCollector.numberOfExamplesForCategory(1) == 3);

    BOOST_TEST(restoredExamplesCollector.add(2, "baz") == false);
    BOOST_TEST(restoredExamplesCollector.add(2, "qux") == false);
    BOOST_TEST(restoredExamplesCollector.numberOfExamplesForCategory(2) == 2);

    BOOST_TEST(restoredExamplesCollector.add(3, "quux") == false);
    BOOST_TEST(restoredExamplesCollector.numberOfExamplesForCategory(3) == 1);
}

BOOST_AUTO_TEST_CASE(testTruncation) {
    BOOST_TEST(CCategoryExamplesCollector::MAX_EXAMPLE_LENGTH > 5);
    const std::string baseExample(CCategoryExamplesCollector::MAX_EXAMPLE_LENGTH - 5, 'a');
    const std::string ellipsis(3, '.');

    CCategoryExamplesCollector examplesCollector(1);

    {
        // All single byte characters
        std::string example = baseExample + "bbbbbb";
        examplesCollector.add(1, example);
        BOOST_CHECK_EQUAL(baseExample + "bb" + ellipsis,
                          *examplesCollector.examples(1).begin());
    }
    {
        // Two byte character crosses truncation boundary
        std::string example = baseExample + "bébbb";
        examplesCollector.add(2, example);
        BOOST_CHECK_EQUAL(baseExample + "b" + ellipsis,
                          *examplesCollector.examples(2).begin());
    }
    {
        // Two byte characters either side of truncation boundary
        std::string example = baseExample + "éébbb";
        examplesCollector.add(3, example);
        BOOST_CHECK_EQUAL(baseExample + "é" + ellipsis,
                          *examplesCollector.examples(3).begin());
    }
    {
        // Two byte character before truncation boundary, single byte immediately after
        std::string example = baseExample + "ébbbb";
        examplesCollector.add(4, example);
        BOOST_CHECK_EQUAL(baseExample + "é" + ellipsis,
                          *examplesCollector.examples(4).begin());
    }
    {
        // Three byte character crosses truncation boundary with start character before
        std::string example = baseExample + "b中bbb";
        examplesCollector.add(5, example);
        BOOST_CHECK_EQUAL(baseExample + "b" + ellipsis,
                          *examplesCollector.examples(5).begin());
    }
    {
        // Three byte character crosses truncation boundary with continuation character before
        std::string example = baseExample + "中bbb";
        examplesCollector.add(6, example);
        BOOST_CHECK_EQUAL(baseExample + ellipsis, *examplesCollector.examples(6).begin());
    }
}

BOOST_AUTO_TEST_SUITE_END()
