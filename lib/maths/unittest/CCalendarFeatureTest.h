/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCalendarFeatureTest_h
#define INCLUDED_CCalendarFeatureTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCalendarFeatureTest : public CppUnit::TestFixture
{
    public:
        void testInitialize();
        void testComparison();
        void testOffset();
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CCalendarFeatureTest_h
