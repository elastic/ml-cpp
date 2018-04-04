/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CUnameTest_h
#define INCLUDED_CUnameTest_h

#include <cppunit/extensions/HelperMacros.h>


class CUnameTest : public CppUnit::TestFixture
{
    public:
        void testUname();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CUnameTest_h

