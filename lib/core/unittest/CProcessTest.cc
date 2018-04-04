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

    LOG_DEBUG("PID = " << pid << " and parent PID = " << ppid);

    CPPUNIT_ASSERT(pid != 0);
    CPPUNIT_ASSERT(ppid != 0);
    CPPUNIT_ASSERT(pid != ppid);
}
