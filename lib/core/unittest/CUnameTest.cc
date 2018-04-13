/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CUnameTest.h"

#include <core/CLogger.h>
#include <core/CUname.h>

CppUnit::Test* CUnameTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CUnameTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CUnameTest>("CUnameTest::testUname", &CUnameTest::testUname));

    return suiteOfTests;
}

void CUnameTest::testUname() {
    LOG_DEBUG(<< ml::core::CUname::sysName());
    LOG_DEBUG(<< ml::core::CUname::nodeName());
    LOG_DEBUG(<< ml::core::CUname::release());
    LOG_DEBUG(<< ml::core::CUname::version());
    LOG_DEBUG(<< ml::core::CUname::machine());
    LOG_DEBUG(<< ml::core::CUname::all());
    LOG_DEBUG(<< ml::core::CUname::mlPlatform());
    LOG_DEBUG(<< ml::core::CUname::mlOsVer());
}
