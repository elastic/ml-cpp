/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBlockingMessageQueueTest_h
#define INCLUDED_CBlockingMessageQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBlockingMessageQueueTest : public CppUnit::TestFixture {
public:
    void testSendReceive();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBlockingMessageQueueTest_h
