/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CShellArgQuoterTest_h
#define INCLUDED_CShellArgQuoterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CShellArgQuoterTest : public CppUnit::TestFixture
{
    public:
        void testQuote(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CShellArgQuoterTest_h

