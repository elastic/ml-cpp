/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSeasonalComponentTest_h
#define INCLUDED_CSeasonalComponentTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSeasonalComponentTest : public CppUnit::TestFixture {
public:
    void testNoPeriodicity();
    void testConstantPeriodic();
    void testTimeVaryingPeriodic();
    void testVeryLowVariation();
    void testVariance();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CSeasonalComponentTest_h
