/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCommandProcessorTest_h
#define INCLUDED_CCommandProcessorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCommandProcessorTest : public CppUnit::TestFixture
{
    public:
        void testStartPermitted(void);
        void testStartNonPermitted(void);
        void testStartNonExistent(void);
        void testKillDisallowed(void);
        void testInvalidVerb(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CCommandProcessorTest_h

