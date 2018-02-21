/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDetachedProcessSpawnerTest_h
#define INCLUDED_CDetachedProcessSpawnerTest_h

#include <cppunit/extensions/HelperMacros.h>


class CDetachedProcessSpawnerTest : public CppUnit::TestFixture
{
    public:
        void testSpawn(void);
        void testKill(void);
        void testPermitted(void);
        void testNonExistent(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CDetachedProcessSpawnerTest_h

