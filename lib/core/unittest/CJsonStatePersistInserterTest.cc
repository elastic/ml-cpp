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
#include "CJsonStatePersistInserterTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <sstream>

CppUnit::Test *CJsonStatePersistInserterTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CJsonStatePersistInserterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonStatePersistInserterTest>(
        "CJsonStatePersistInserterTest::testPersist", &CJsonStatePersistInserterTest::testPersist));

    return suiteOfTests;
}

namespace {

void insert2ndLevel(ml::core::CStatePersistInserter &inserter) {
    inserter.insertValue("level2A", 3.14, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("level2B", 'z');
}
}

void CJsonStatePersistInserterTest::testPersist(void) {
    std::ostringstream strm;

    {
        ml::core::CJsonStatePersistInserter inserter(strm);

        inserter.insertValue("level1A", "a");
        inserter.insertValue("level1B", 25);
        inserter.insertLevel("level1C", &insert2ndLevel);
    }

    std::string json(strm.str());
    ml::core::CStringUtils::trimWhitespace(json);

    LOG_DEBUG("JSON is: " << json);

    CPPUNIT_ASSERT_EQUAL(std::string("{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{"
                                     "\"level2A\":\"3.14\",\"level2B\":\"z\"}}"),
                         json);
}
