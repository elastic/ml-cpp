/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_COrdinalTest_h
#define INCLUDED_COrdinalTest_h

#include <cppunit/extensions/HelperMacros.h>

class COrdinalTest : public CppUnit::TestFixture
{
    public:
        void testEqual(void);
        void testLess(void);
        void testIsNan(void);
        void testAsDouble(void);
        void testHash(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_COrderingsTest_h
