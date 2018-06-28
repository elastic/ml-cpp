/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CJsonLogLayoutTest_h
#define INCLUDED_CJsonLogLayoutTest_h

#include <cppunit/extensions/HelperMacros.h>

class CJsonLogLayoutTest : public CppUnit::TestFixture {
public:
    void testPathCropping();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CJsonLogLayoutTest_h
