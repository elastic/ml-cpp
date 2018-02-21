/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPersistUtilsTest_h
#define INCLUDED_CPersistUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPersistUtilsTest : public CppUnit::TestFixture
{
    public:
        void testPersistContainers(void);
        void testPersistIterators(void);
        void testAppend(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CPersistUtilsTest_h
