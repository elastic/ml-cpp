/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CAgglomerativeClustererTest_h
#define INCLUDED_CAgglomerativeClustererTest_h

#include <cppunit/extensions/HelperMacros.h>

class CAgglomerativeClustererTest : public CppUnit::TestFixture
{
    public:
        void testNode(void);
        void testSimplePermutations(void);
        void testDegenerate(void);
        void testRandom(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CAgglomerativeClustererTest_h
