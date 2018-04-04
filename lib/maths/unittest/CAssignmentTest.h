/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CAssignmentTest_h
#define INCLUDED_CAssignmentTest_h

#include <cppunit/extensions/HelperMacros.h>

class CAssignmentTest : public CppUnit::TestFixture {
public:
    void testKuhnMunkres();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CAssignmentTest_h
