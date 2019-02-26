/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CSearchKeyTest.h"

#include <model/CSearchKey.h>

using namespace ml;
using namespace model;

void CSearchKeyTest::testSimpleCountComesFirst() {
    CSearchKey key1{1};
    CSearchKey key2{2};
    CPPUNIT_ASSERT(CSearchKey::simpleCountKey() < key1);
    CPPUNIT_ASSERT(CSearchKey::simpleCountKey() < key2);
}

CppUnit::Test* CSearchKeyTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSearchKeyTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSearchKeyTest>(
        "CSearchKeyTest::testSimpleCountComesFirst", &CSearchKeyTest::testSimpleCountComesFirst));

    return suiteOfTests;
}
