/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CFileDeleterTest_h
#define INCLUDED_CFileDeleterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CFileDeleterTest : public CppUnit::TestFixture
{
    public:
        void testDelete();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CFileDeleterTest_h

