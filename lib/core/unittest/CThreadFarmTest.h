/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CThreadFarmTest_h
#define INCLUDED_CThreadFarmTest_h

#include <cppunit/extensions/HelperMacros.h>

class CThreadFarmTest : public CppUnit::TestFixture {
public:
    void testNumCpus();
    void testSendReceive();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CThreadFarmTest_h
