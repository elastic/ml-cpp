/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CJsonOutputStreamWrapperTest_h
#define INCLUDED_CJsonOutputStreamWrapperTest_h

#include <cppunit/extensions/HelperMacros.h>

class CJsonOutputStreamWrapperTest : public CppUnit::TestFixture
{
    public:
        void testConcurrentWrites();
        void testShrink();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CJsonOutputStreamWrapperTest_h

