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
#include "CCategoryExamplesCollectorTest.h"

#include <api/CCategoryExamplesCollector.h>

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

using namespace ml;
using namespace api;

void CCategoryExamplesCollectorTest::testAddGivenMaxExamplesIsZero() {
    CCategoryExamplesCollector examplesCollector(0);
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo") == false);
    CPPUNIT_ASSERT(examplesCollector.add(2, "foo") == false);
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(0));
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(2), std::size_t(0));
}

void CCategoryExamplesCollectorTest::testAddGivenSameCategoryExamplePairAddedTwice() {
    CCategoryExamplesCollector examplesCollector(4);
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo") == true);
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo") == false);
}

void CCategoryExamplesCollectorTest::testAddGivenMoreThanMaxExamplesAreAddedForSameCategory() {
    CCategoryExamplesCollector examplesCollector(3);
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo1") == true);
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(1));
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo2") == true);
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(2));
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo3") == true);
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(3));
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo4") == false);
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(3));
}

void CCategoryExamplesCollectorTest::testAddGivenCategoryAddedIsNotSubsequent() {
    CCategoryExamplesCollector examplesCollector(2);
    CPPUNIT_ASSERT(examplesCollector.add(1, "foo") == true);
    CPPUNIT_ASSERT(examplesCollector.add(3, "bar") == true);
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(1), std::size_t(1));
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(2), std::size_t(0));
    CPPUNIT_ASSERT_EQUAL(examplesCollector.numberOfExamplesForCategory(3), std::size_t(1));
}

void CCategoryExamplesCollectorTest::testExamples() {
    CCategoryExamplesCollector examplesCollector(3);
    examplesCollector.add(1, "foo");
    examplesCollector.add(1, "bar");
    examplesCollector.add(2, "foo");

    CCategoryExamplesCollector::TStrSet examples1 = examplesCollector.examples(1);
    CPPUNIT_ASSERT(examples1.find("foo") != examples1.end());
    CPPUNIT_ASSERT(examples1.find("bar") != examples1.end());
    CPPUNIT_ASSERT(examples1.find("invalid") == examples1.end());

    CCategoryExamplesCollector::TStrSet examples2 = examplesCollector.examples(2);
    CPPUNIT_ASSERT(examples2.find("foo") != examples2.end());
    CPPUNIT_ASSERT(examples2.find("invalid") == examples2.end());
}

void CCategoryExamplesCollectorTest::testPersist() {
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
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CCategoryExamplesCollector restoredExamplesCollector(3, traverser);

    CPPUNIT_ASSERT(restoredExamplesCollector.numberOfExamplesForCategory(1) == 3);

    CPPUNIT_ASSERT(restoredExamplesCollector.add(2, "baz") == false);
    CPPUNIT_ASSERT(restoredExamplesCollector.add(2, "qux") == false);
    CPPUNIT_ASSERT(restoredExamplesCollector.numberOfExamplesForCategory(2) == 2);

    CPPUNIT_ASSERT(restoredExamplesCollector.add(3, "quux") == false);
    CPPUNIT_ASSERT(restoredExamplesCollector.numberOfExamplesForCategory(3) == 1);
}

void CCategoryExamplesCollectorTest::testTruncation() {
    CPPUNIT_ASSERT(CCategoryExamplesCollector::MAX_EXAMPLE_LENGTH > 5);
    const std::string baseExample(CCategoryExamplesCollector::MAX_EXAMPLE_LENGTH - 5, 'a');
    const std::string ellipsis(3, '.');

    CCategoryExamplesCollector examplesCollector(1);

    {
        // All single byte characters
        std::string example = baseExample + "bbbbbb";
        examplesCollector.add(1, example);
        CPPUNIT_ASSERT_EQUAL(baseExample + "bb" + ellipsis,
                             *examplesCollector.examples(1).begin());
    }
    {
        // Two byte character crosses truncation boundary
        std::string example = baseExample + "bébbb";
        examplesCollector.add(2, example);
        CPPUNIT_ASSERT_EQUAL(baseExample + "b" + ellipsis,
                             *examplesCollector.examples(2).begin());
    }
    {
        // Two byte characters either side of truncation boundary
        std::string example = baseExample + "éébbb";
        examplesCollector.add(3, example);
        CPPUNIT_ASSERT_EQUAL(baseExample + "é" + ellipsis,
                             *examplesCollector.examples(3).begin());
    }
    {
        // Two byte character before truncation boundary, single byte immediately after
        std::string example = baseExample + "ébbbb";
        examplesCollector.add(4, example);
        CPPUNIT_ASSERT_EQUAL(baseExample + "é" + ellipsis,
                             *examplesCollector.examples(4).begin());
    }
    {
        // Three byte character crosses truncation boundary with start character before
        std::string example = baseExample + "b中bbb";
        examplesCollector.add(5, example);
        CPPUNIT_ASSERT_EQUAL(baseExample + "b" + ellipsis,
                             *examplesCollector.examples(5).begin());
    }
    {
        // Three byte character crosses truncation boundary with continuation character before
        std::string example = baseExample + "中bbb";
        examplesCollector.add(6, example);
        CPPUNIT_ASSERT_EQUAL(baseExample + ellipsis,
                             *examplesCollector.examples(6).begin());
    }
}

CppUnit::Test* CCategoryExamplesCollectorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CCategoryExamplesCollectorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testAddGivenMaxExamplesIsZero",
        &CCategoryExamplesCollectorTest::testAddGivenMaxExamplesIsZero));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testAddGivenSameCategoryExamplePairAddedTwice",
        &CCategoryExamplesCollectorTest::testAddGivenSameCategoryExamplePairAddedTwice));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testAddGivenMoreThanMaxExamplesAreAddedForSameCategory",
        &CCategoryExamplesCollectorTest::testAddGivenMoreThanMaxExamplesAreAddedForSameCategory));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testAddGivenCategoryAddedIsNotSubsequent",
        &CCategoryExamplesCollectorTest::testAddGivenCategoryAddedIsNotSubsequent));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testExamples",
        &CCategoryExamplesCollectorTest::testExamples));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testPersist",
        &CCategoryExamplesCollectorTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCategoryExamplesCollectorTest>(
        "CCategoryExamplesCollectorTest::testTruncation",
        &CCategoryExamplesCollectorTest::testTruncation));

    return suiteOfTests;
}
