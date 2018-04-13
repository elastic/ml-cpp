/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CJsonStatePersistInserterTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <sstream>

CppUnit::Test* CJsonStatePersistInserterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CJsonStatePersistInserterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStatePersistInserterTest>(
        "CJsonStatePersistInserterTest::testPersist",
        &CJsonStatePersistInserterTest::testPersist));

    return suiteOfTests;
}

namespace {

void insert2ndLevel(ml::core::CStatePersistInserter& inserter) {
    inserter.insertValue("level2A", 3.14, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("level2B", 'z');
}
}

void CJsonStatePersistInserterTest::testPersist() {
    std::ostringstream strm;

    {
        ml::core::CJsonStatePersistInserter inserter(strm);

        inserter.insertValue("level1A", "a");
        inserter.insertValue("level1B", 25);
        inserter.insertLevel("level1C", &insert2ndLevel);
    }

    std::string json(strm.str());
    ml::core::CStringUtils::trimWhitespace(json);

    LOG_DEBUG(<< "JSON is: " << json);

    CPPUNIT_ASSERT_EQUAL(std::string("{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}"),
                         json);
}
