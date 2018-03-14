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
#include "CDynamicStringIdRegistryTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>

#include <model/CDynamicStringIdRegistry.h>
#include <model/CResourceMonitor.h>

#include <boost/bind.hpp>

#include <vector>

using namespace ml;
using namespace model;

CppUnit::Test* CDynamicStringIdRegistryTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDynamicStringIdRegistryTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CDynamicStringIdRegistryTest>("CDynamicStringIdRegistryTest::testAddName",
                                          &CDynamicStringIdRegistryTest::testAddName));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CDynamicStringIdRegistryTest>("CDynamicStringIdRegistryTest::testPersist",
                                          &CDynamicStringIdRegistryTest::testPersist));

    return suiteOfTests;
}

void CDynamicStringIdRegistryTest::testAddName(void) {
    LOG_DEBUG("*** testAddName ***");

    CResourceMonitor resourceMonitor;
    CDynamicStringIdRegistry registry("person",
                                      stat_t::E_NumberNewPeople,
                                      stat_t::E_NumberNewPeopleNotAllowed,
                                      stat_t::E_NumberNewPeopleRecycled);

    bool personAdded = false;
    std::string person1("foo");
    std::string person2("bar");
    CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                         registry.addName(person1, 100, resourceMonitor, personAdded));
    CPPUNIT_ASSERT(personAdded);

    personAdded = false;
    CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                         registry.addName(person2, 200, resourceMonitor, personAdded));
    CPPUNIT_ASSERT(personAdded);
    personAdded = false;

    CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                         registry.addName(person1, 300, resourceMonitor, personAdded));
    CPPUNIT_ASSERT(personAdded == false);

    std::string person3("noot");
    CPPUNIT_ASSERT_EQUAL(std::size_t(2),
                         registry.addName(person3, 400, resourceMonitor, personAdded));
    CPPUNIT_ASSERT(personAdded);
    personAdded = false;

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), registry.numberNames());
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), registry.numberActiveNames());

    std::string defaultName("-");
    CDynamicStringIdRegistry::TSizeVec toRecycle;
    toRecycle.push_back(std::size_t(1));
    registry.recycleNames(toRecycle, defaultName);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), registry.numberNames());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), registry.numberActiveNames());
    CPPUNIT_ASSERT(registry.isIdActive(0));
    CPPUNIT_ASSERT(registry.isIdActive(1) == false);
    CPPUNIT_ASSERT(registry.isIdActive(2));

    std::string person4("recycled");
    CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                         registry.addName(person4, 500, resourceMonitor, personAdded));
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), registry.numberNames());
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), registry.numberActiveNames());
    CPPUNIT_ASSERT(registry.isIdActive(0));
    CPPUNIT_ASSERT(registry.isIdActive(1));
    CPPUNIT_ASSERT(registry.isIdActive(2));
}

void CDynamicStringIdRegistryTest::testPersist(void) {
    LOG_DEBUG("*** testPersist ***");

    CResourceMonitor resourceMonitor;
    CDynamicStringIdRegistry registry("person",
                                      stat_t::E_NumberNewPeople,
                                      stat_t::E_NumberNewPeopleNotAllowed,
                                      stat_t::E_NumberNewPeopleRecycled);

    bool addedPerson = false;
    std::string person1("foo");
    std::string person2("bar");
    registry.addName(person1, 0, resourceMonitor, addedPerson);
    registry.addName(person2, 0, resourceMonitor, addedPerson);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        registry.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE("Original XML:\n" << origXml);

    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    CDynamicStringIdRegistry restoredRegistry("person",
                                              stat_t::E_NumberNewPeople,
                                              stat_t::E_NumberNewPeopleNotAllowed,
                                              stat_t::E_NumberNewPeopleRecycled);
    traverser.traverseSubLevel(
        boost::bind(&CDynamicStringIdRegistry::acceptRestoreTraverser, &restoredRegistry, _1));

    std::string restoredXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredRegistry.acceptPersistInserter(inserter);
        inserter.toXml(restoredXml);
    }
    LOG_TRACE("Restored XML:\n" << restoredXml);

    CPPUNIT_ASSERT_EQUAL(restoredXml, origXml);
}
