/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStoredStringPtrTest_h
#define INCLUDED_CStoredStringPtrTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStoredStringPtrTest : public CppUnit::TestFixture
{
    public:
        void testPointerSemantics();
        void testMemoryUsage();
        void testHash();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CStoredStringPtrTest_h

