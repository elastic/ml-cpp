/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPRNGTest_h
#define INCLUDED_CPRNGTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPRNGTest : public CppUnit::TestFixture
{
    public:
        void testSplitMix64();
        void testXorOShiro128Plus();
        void testXorShift1024Mult();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CPRNGTest_h
