/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_COrderingsTest_h
#define INCLUDED_COrderingsTest_h

#include <cppunit/extensions/HelperMacros.h>

class COrderingsTest : public CppUnit::TestFixture
{
    public:
        void testOptionalOrdering(void);
        void testPtrOrdering(void);
        void testLess(void);
        void testFirstLess(void);
        void testFirstGreater(void);
        void testSecondLess(void);
        void testSecondGreater(void);
        void testDereference(void);
        void testLexicographicalCompare(void);
        void testSimultaneousSort(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_COrderingsTest_h
