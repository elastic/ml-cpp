/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBase64FilterTest_h
#define INCLUDED_CBase64FilterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CBase64FilterTest : public CppUnit::TestFixture
{
    public:
        void testDecode();
        void testEncode();
        void testBoth();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBase64FilterTest_h
