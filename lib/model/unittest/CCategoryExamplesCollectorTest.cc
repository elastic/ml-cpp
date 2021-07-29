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

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <model/CCategoryExamplesCollector.h>

#include <boost/test/unit_test.hpp>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::model::CCategoryExamplesCollector::TStrFSet::iterator)

BOOST_AUTO_TEST_SUITE(CCategoryExamplesCollectorTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testAddGivenMaxExamplesIsZero) {
    CCategoryExamplesCollector examplesCollector(0);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo") == false);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{2}, "foo") == false);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{1}) == 0);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{2}) == 0);
}

BOOST_AUTO_TEST_CASE(testAddGivenSameCategoryExamplePairAddedTwice) {
    CCategoryExamplesCollector examplesCollector(4);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo") == true);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo") == false);
}

BOOST_AUTO_TEST_CASE(testAddGivenMoreThanMaxExamplesAreAddedForSameCategory) {
    CCategoryExamplesCollector examplesCollector(3);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo1") == true);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{1}) == 1);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo2") == true);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{1}) == 2);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo3") == true);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{1}) == 3);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo4") == false);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{1}) == 3);
}

BOOST_AUTO_TEST_CASE(testAddGivenCategoryAddedIsNotSubsequent) {
    CCategoryExamplesCollector examplesCollector(2);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{1}, "foo") == true);
    BOOST_TEST_REQUIRE(examplesCollector.add(CLocalCategoryId{3}, "bar") == true);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{1}) == 1);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{2}) == 0);
    BOOST_TEST_REQUIRE(
        examplesCollector.numberOfExamplesForCategory(CLocalCategoryId{3}) == 1);
}

BOOST_AUTO_TEST_CASE(testExamples) {
    CCategoryExamplesCollector examplesCollector(3);
    examplesCollector.add(CLocalCategoryId{1}, "foo");
    examplesCollector.add(CLocalCategoryId{1}, "bar");
    examplesCollector.add(CLocalCategoryId{2}, "foo");

    CCategoryExamplesCollector::TStrFSet examples1 =
        examplesCollector.examples(CLocalCategoryId{1});
    BOOST_TEST_REQUIRE(examples1.find("foo") != examples1.end());
    BOOST_TEST_REQUIRE(examples1.find("bar") != examples1.end());
    BOOST_TEST_REQUIRE(examples1.find("invalid") == examples1.end());

    CCategoryExamplesCollector::TStrFSet examples2 =
        examplesCollector.examples(CLocalCategoryId{2});
    BOOST_TEST_REQUIRE(examples2.find("foo") != examples2.end());
    BOOST_TEST_REQUIRE(examples2.find("invalid") == examples2.end());
}

BOOST_AUTO_TEST_CASE(testPersist) {
    CCategoryExamplesCollector examplesCollector(3);
    examplesCollector.add(CLocalCategoryId{1}, "foo");
    examplesCollector.add(CLocalCategoryId{1}, "bar");
    examplesCollector.add(CLocalCategoryId{1}, "foobar");
    examplesCollector.add(CLocalCategoryId{2}, "baz");
    examplesCollector.add(CLocalCategoryId{2}, "qux");
    examplesCollector.add(CLocalCategoryId{3}, "quux");

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        examplesCollector.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE(<< "XML:\n" << origXml);

    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CCategoryExamplesCollector restoredExamplesCollector(3, traverser);

    BOOST_TEST_REQUIRE(restoredExamplesCollector.numberOfExamplesForCategory(
                           CLocalCategoryId{1}) == 3);

    BOOST_TEST_REQUIRE(restoredExamplesCollector.add(CLocalCategoryId{2}, "baz") == false);
    BOOST_TEST_REQUIRE(restoredExamplesCollector.add(CLocalCategoryId{2}, "qux") == false);
    BOOST_TEST_REQUIRE(restoredExamplesCollector.numberOfExamplesForCategory(
                           CLocalCategoryId{2}) == 2);

    BOOST_TEST_REQUIRE(restoredExamplesCollector.add(CLocalCategoryId{3}, "quux") == false);
    BOOST_TEST_REQUIRE(restoredExamplesCollector.numberOfExamplesForCategory(
                           CLocalCategoryId{3}) == 1);
}

BOOST_AUTO_TEST_CASE(testTruncation) {
    BOOST_TEST_REQUIRE(CCategoryExamplesCollector::MAX_EXAMPLE_LENGTH > 5);
    const std::string baseExample(CCategoryExamplesCollector::MAX_EXAMPLE_LENGTH - 5, 'a');
    const std::string ellipsis(3, '.');

    CCategoryExamplesCollector examplesCollector(1);

    {
        // All single byte characters
        std::string example = baseExample + "bbbbbb";
        examplesCollector.add(CLocalCategoryId{1}, example);
        BOOST_REQUIRE_EQUAL(baseExample + "bb" + ellipsis,
                            *examplesCollector.examples(CLocalCategoryId{1}).begin());
    }
    {
        // Two byte character crosses truncation boundary
        std::string example = baseExample + "bébbb";
        examplesCollector.add(CLocalCategoryId{2}, example);
        BOOST_REQUIRE_EQUAL(baseExample + "b" + ellipsis,
                            *examplesCollector.examples(CLocalCategoryId{2}).begin());
    }
    {
        // Two byte characters either side of truncation boundary
        std::string example = baseExample + "éébbb";
        examplesCollector.add(CLocalCategoryId{3}, example);
        BOOST_REQUIRE_EQUAL(baseExample + "é" + ellipsis,
                            *examplesCollector.examples(CLocalCategoryId{3}).begin());
    }
    {
        // Two byte character before truncation boundary, single byte immediately after
        std::string example = baseExample + "ébbbb";
        examplesCollector.add(CLocalCategoryId{4}, example);
        BOOST_REQUIRE_EQUAL(baseExample + "é" + ellipsis,
                            *examplesCollector.examples(CLocalCategoryId{4}).begin());
    }
    {
        // Three byte character crosses truncation boundary with start character before
        std::string example = baseExample + "b中bbb";
        examplesCollector.add(CLocalCategoryId{5}, example);
        BOOST_REQUIRE_EQUAL(baseExample + "b" + ellipsis,
                            *examplesCollector.examples(CLocalCategoryId{5}).begin());
    }
    {
        // Three byte character crosses truncation boundary with continuation character before
        std::string example = baseExample + "中bbb";
        examplesCollector.add(CLocalCategoryId{6}, example);
        BOOST_REQUIRE_EQUAL(baseExample + ellipsis,
                            *examplesCollector.examples(CLocalCategoryId{6}).begin());
    }
}

BOOST_AUTO_TEST_SUITE_END()
