/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CAnomalyJobLimitTest_h
#define INCLUDED_CAnomalyJobLimitTest_h

#include <cppunit/extensions/HelperMacros.h>

class CAnomalyJobLimitTest : public CppUnit::TestFixture {
public:
    void testLimit();
    void testAccuracy();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CAnomalyJobLimitTest_h
