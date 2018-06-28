/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CModelToolsTest_h
#define INCLUDED_CModelToolsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CModelToolsTest : public CppUnit::TestFixture {
public:
    void testFuzzyDeduplicate();
    void testProbabilityCache();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CModelToolsTest_h
