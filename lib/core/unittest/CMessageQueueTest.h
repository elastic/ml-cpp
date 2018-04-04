/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CMessageQueueTest_h
#define INCLUDED_CMessageQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMessageQueueTest : public CppUnit::TestFixture {
public:
    void testSendReceive();
    void testTiming();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMessageQueueTest_h
