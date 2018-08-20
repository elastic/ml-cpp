/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CHierarchicalResultsLevelSetTest_h
#define INCLUDED_CHierarchicalResultsLevelSetTest_h

#include <cppunit/extensions/HelperMacros.h>

class CHierarchicalResultsLevelSetTest : public CppUnit::TestFixture {
public:
    void testElements();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CHierarchicalResultsLevelSetTest_h
