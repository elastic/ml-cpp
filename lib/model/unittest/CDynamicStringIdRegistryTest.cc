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

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CSmallVector.h>

#include <model/CDynamicStringIdRegistry.h>
#include <model/CResourceMonitor.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CDynamicStringIdRegistryTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testAddName) {
    CResourceMonitor resourceMonitor;
    CDynamicStringIdRegistry registry("person", counter_t::E_TSADNumberNewPeople,
                                      counter_t::E_TSADNumberNewPeopleNotAllowed,
                                      counter_t::E_TSADNumberNewPeopleRecycled);

    bool personAdded = false;
    std::string person1("foo");
    std::string person2("bar");
    BOOST_REQUIRE_EQUAL(0, registry.addName(person1, 100, resourceMonitor, personAdded));
    BOOST_TEST_REQUIRE(personAdded);

    personAdded = false;
    BOOST_REQUIRE_EQUAL(1, registry.addName(person2, 200, resourceMonitor, personAdded));
    BOOST_TEST_REQUIRE(personAdded);
    personAdded = false;

    BOOST_REQUIRE_EQUAL(0, registry.addName(person1, 300, resourceMonitor, personAdded));
    BOOST_TEST_REQUIRE(personAdded == false);

    std::string person3("noot");
    BOOST_REQUIRE_EQUAL(2, registry.addName(person3, 400, resourceMonitor, personAdded));
    BOOST_TEST_REQUIRE(personAdded);
    personAdded = false;

    BOOST_REQUIRE_EQUAL(3, registry.numberNames());
    BOOST_REQUIRE_EQUAL(3, registry.numberActiveNames());

    std::string defaultName("-");
    CDynamicStringIdRegistry::TSizeVec toRecycle;
    toRecycle.push_back(std::size_t(1));
    registry.recycleNames(toRecycle, defaultName);

    BOOST_REQUIRE_EQUAL(3, registry.numberNames());
    BOOST_REQUIRE_EQUAL(2, registry.numberActiveNames());
    BOOST_TEST_REQUIRE(registry.isIdActive(0));
    BOOST_TEST_REQUIRE(registry.isIdActive(1) == false);
    BOOST_TEST_REQUIRE(registry.isIdActive(2));

    std::string person4("recycled");
    BOOST_REQUIRE_EQUAL(1, registry.addName(person4, 500, resourceMonitor, personAdded));
    BOOST_REQUIRE_EQUAL(3, registry.numberNames());
    BOOST_REQUIRE_EQUAL(3, registry.numberActiveNames());
    BOOST_TEST_REQUIRE(registry.isIdActive(0));
    BOOST_TEST_REQUIRE(registry.isIdActive(1));
    BOOST_TEST_REQUIRE(registry.isIdActive(2));
}

BOOST_AUTO_TEST_CASE(testPersist) {
    CResourceMonitor resourceMonitor;
    CDynamicStringIdRegistry registry("person", counter_t::E_TSADNumberNewPeople,
                                      counter_t::E_TSADNumberNewPeopleNotAllowed,
                                      counter_t::E_TSADNumberNewPeopleRecycled);

    bool addedPerson = false;
    std::string person1("foo");
    std::string person2("bar");
    registry.addName(person1, 0, resourceMonitor, addedPerson);
    registry.addName(person2, 0, resourceMonitor, addedPerson);

    std::ostringstream origJson;
    {
        core::CJsonStatePersistInserter inserter(origJson);
        registry.acceptPersistInserter(inserter);
    }
    LOG_TRACE(<< "Original JSON:\n" << origJson.str());

    // The traverser expects the state json in a embedded document
    std::istringstream is("{\"topLevel\" : " + origJson.str() + "}");
    core::CJsonStateRestoreTraverser traverser(is);
    CDynamicStringIdRegistry restoredRegistry("person", counter_t::E_TSADNumberNewPeople,
                                              counter_t::E_TSADNumberNewPeopleNotAllowed,
                                              counter_t::E_TSADNumberNewPeopleRecycled);
    traverser.traverseSubLevel(std::bind(&CDynamicStringIdRegistry::acceptRestoreTraverser,
                                         &restoredRegistry, std::placeholders::_1));

    std::ostringstream restoredJson;
    {
        core::CJsonStatePersistInserter inserter(restoredJson);
        restoredRegistry.acceptPersistInserter(inserter);
    }
    LOG_TRACE(<< "Restored JSON:\n" << restoredJson.str());

    BOOST_REQUIRE_EQUAL(restoredJson.str(), origJson.str());
}

BOOST_AUTO_TEST_SUITE_END()
