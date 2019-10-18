/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CRapidXmlStateRestoreTraverserTest)


namespace {

bool traverse2ndLevel(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level2A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("3.14"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level2B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("z"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(!traverser.next());

    return true;
}

bool traverse1stLevel(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level1A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse2ndLevel));
    BOOST_TEST(!traverser.next());

    return true;
}
}

BOOST_AUTO_TEST_CASE(testRestore) {
    std::string xml("<root attr1=\"attrVal1\" "
                    "attr2=\"attrVal2\"><level1A>a</level1A><level1B>25</level1B><level1C><level2A>3.14</level2A><level2B>z</level2B></"
                    "level1C></root>");

    ml::core::CRapidXmlParser parser;
    BOOST_TEST(parser.parseStringIgnoreCdata(xml));

    ml::core::CRapidXmlStateRestoreTraverser traverser(parser);

    BOOST_CHECK_EQUAL(std::string("root"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_SUITE_END()
