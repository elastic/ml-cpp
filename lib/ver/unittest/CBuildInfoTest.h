/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBuildInfoTest_h
#define INCLUDED_CBuildInfoTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBuildInfoTest : public CppUnit::TestFixture {
public:
    void testFullInfo(void);

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBuildInfoTest_h
