/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CLoggerTest_h
#define INCLUDED_CLoggerTest_h

#include <cppunit/extensions/HelperMacros.h>


class CLoggerTest : public CppUnit::TestFixture
{
    public:
        void testLogging(void);
        void testReconfiguration(void);
        void testSetLevel(void);
        void testLogEnvironment(void);
        void testNonAsciiJsonLogging(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CLoggerTest_h

