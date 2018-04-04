/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEqualWithToleranceTest_h
#define INCLUDED_CEqualWithToleranceTest_h

#include <cppunit/extensions/HelperMacros.h>

class CEqualWithToleranceTest : public CppUnit::TestFixture
{
    public:
        void testScalar();
        void testVector();
        void testMatrix();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CEqualWithToleranceTest_h
