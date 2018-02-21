/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CToolsTest_h
#define INCLUDED_CToolsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CToolsTest : public CppUnit::TestFixture
{
    public:
        void testDataGatherers(void);
        void testProbabilityAggregator(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CToolsTest_h
