/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CGathererToolsTest_h
#define INCLUDED_CGathererToolsTest_h

#include <cppunit/extensions/HelperMacros.h>


class CGathererToolsTest : public CppUnit::TestFixture
{
    public:
        void testSumGathererIsRedundant();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CGathererToolsTest_h

