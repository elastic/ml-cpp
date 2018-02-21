/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CRapidXmlStatePersistInserterTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>


CppUnit::Test *CRapidXmlStatePersistInserterTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CRapidXmlStatePersistInserterTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CRapidXmlStatePersistInserterTest>(
                                   "CRapidXmlStatePersistInserterTest::testPersist",
                                   &CRapidXmlStatePersistInserterTest::testPersist) );

    return suiteOfTests;
}

namespace
{

void insert2ndLevel(ml::core::CStatePersistInserter &inserter)
{
    inserter.insertValue("level2A", 3.14, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("level2B", 'z');
}

}

void CRapidXmlStatePersistInserterTest::testPersist(void)
{
    ml::core::CRapidXmlStatePersistInserter::TStrStrMap rootAttributes;
    rootAttributes["attr1"] = "attrVal1";
    rootAttributes["attr2"] = "attrVal2";

    ml::core::CRapidXmlStatePersistInserter inserter("root", rootAttributes);

    inserter.insertValue("level1A", "a");
    inserter.insertValue("level1B", 25);
    inserter.insertLevel("level1C", &insert2ndLevel);

    std::string xml;
    inserter.toXml(xml);

    LOG_DEBUG("XML is: " << xml);

    inserter.toXml(false, xml);
    CPPUNIT_ASSERT_EQUAL(std::string("<root attr1=\"attrVal1\" attr2=\"attrVal2\"><level1A>a</level1A><level1B>25</level1B><level1C><level2A>3.14</level2A><level2B>z</level2B></level1C></root>"),
                         xml);
}

