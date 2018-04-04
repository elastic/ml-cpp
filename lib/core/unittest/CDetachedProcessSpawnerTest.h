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
        void testSpawn();
        void testKill();
        void testPermitted();
        void testNonExistent();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CDetachedProcessSpawnerTest_h

