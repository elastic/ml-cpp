/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CForecastModelPersistTest_h
#define INCLUDED_CForecastModelPersistTest_h

#include <cppunit/extensions/HelperMacros.h>

class CForecastModelPersistTest : public CppUnit::TestFixture {
public:
    void testPersistAndRestore();
    void testPersistAndRestoreEmpty();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CForecastModelPersistTest_h
