/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLocalOutlierFactorsTest_h
#define INCLUDED_CLocalOutlierFactorsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLocalOutlierFactorsTest : public CppUnit::TestFixture
{
    public:
        void testLof(void);
        void testDlof(void);
        void testDistancekNN(void);
        void testTotalDistancekNN(void);
        void testEnsemble(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CLocalOutlierFactorsTest_h
