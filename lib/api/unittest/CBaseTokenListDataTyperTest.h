/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBaseTokenListDataTyperTest_h
#define INCLUDED_CBaseTokenListDataTyperTest_h

#include <cppunit/extensions/HelperMacros.h>


class CBaseTokenListDataTyperTest : public CppUnit::TestFixture
{
    public:
        void testMinMatchingWeights(void);
        void testMaxMatchingWeights(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBaseTokenListDataTyperTest_h

