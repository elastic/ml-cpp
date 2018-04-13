/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSplineTest_h
#define INCLUDED_CSplineTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSplineTest : public CppUnit::TestFixture
{
    public:
        void testNatural();
        void testParabolicRunout();
        void testPeriodic();
        void testMean();
        void testIllposed();
        void testSlope();
        void testSplineReference();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CSpline_h
