/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CBjkstUniqueValuesTest_h
#define INCLUDED_CBjkstUniqueValuesTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBjkstUniqueValuesTest : public CppUnit::TestFixture
{
    public:
        void testTrailingZeros();
        void testNumber();
        void testRemove();
        void testSwap();
        void testSmall();
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBjkstUniqueValuesTest_h
