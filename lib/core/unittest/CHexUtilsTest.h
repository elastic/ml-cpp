/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CHexUtilsTest_h
#define INCLUDED_CHexUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CHexUtilsTest : public CppUnit::TestFixture
{
    public:
        void testHexOutput();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CHexUtilsTest_h

