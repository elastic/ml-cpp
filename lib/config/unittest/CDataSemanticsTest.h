/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataSemanticsTest_h
#define INCLUDED_CDataSemanticsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataSemanticsTest : public CppUnit::TestFixture
{
    public:
        void testBinary(void);
        void testNonNumericCategorical(void);
        void testNumericCategorical(void);
        void testInteger(void);
        void testReal(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CDataSemanticsTest_h
