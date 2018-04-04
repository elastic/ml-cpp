/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CResourceLocatorTest_h
#define INCLUDED_CResourceLocatorTest_h

#include <cppunit/extensions/HelperMacros.h>


class CResourceLocatorTest : public CppUnit::TestFixture
{
    public:
        void testResourceDir();
        void testLogDir();
        void testSrcRootDir();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CResourceLocatorTest_h

