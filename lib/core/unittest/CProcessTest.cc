/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CProcessTest.h"

#include <core/CLogger.h>
#include <core/CProcess.h>

CppUnit::Test* CProcessTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CProcessTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CProcessTest>("CProcessTest::testPids", &CProcessTest::testPids));

    return suiteOfTests;
}

void CProcessTest::testPids() {
    ml::core::CProcess& process = ml::core::CProcess::instance();
    ml::core::CProcess::TPid pid = process.id();
    ml::core::CProcess::TPid ppid = process.parentId();

    LOG_DEBUG(<< "PID = " << pid << " and parent PID = " << ppid);

    CPPUNIT_ASSERT(pid != 0);
    CPPUNIT_ASSERT(ppid != 0);
    CPPUNIT_ASSERT(pid != ppid);
}
