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
#include "CUnameTest.h"

#include <core/CLogger.h>
#include <core/CUname.h>

CppUnit::Test *CUnameTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CUnameTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CUnameTest>("CUnameTest::testUname", &CUnameTest::testUname));

    return suiteOfTests;
}

void CUnameTest::testUname(void) {
    LOG_DEBUG(ml::core::CUname::sysName());
    LOG_DEBUG(ml::core::CUname::nodeName());
    LOG_DEBUG(ml::core::CUname::release());
    LOG_DEBUG(ml::core::CUname::version());
    LOG_DEBUG(ml::core::CUname::machine());
    LOG_DEBUG(ml::core::CUname::all());
    LOG_DEBUG(ml::core::CUname::mlPlatform());
    LOG_DEBUG(ml::core::CUname::mlOsVer());
}
