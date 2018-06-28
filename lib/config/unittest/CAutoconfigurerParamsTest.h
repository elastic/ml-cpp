/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CAutoconfigurerParamsTest_h
#define INCLUDED_CAutoconfigurerParamsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CAutoconfigurerParamsTest : public CppUnit::TestFixture {
public:
    void testDefaults();
    void testInit();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CAutoconfigurerParamsTest_h
