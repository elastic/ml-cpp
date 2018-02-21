/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CModelMemoryTest_h
#define INCLUDED_CModelMemoryTest_h

#include <cppunit/extensions/HelperMacros.h>

class CModelMemoryTest : public CppUnit::TestFixture
{
    public:
        void testOnlineEventRateModel(void);
        void testOnlineMetricModel(void);
    
        static CppUnit::Test *suite(void);
};


#endif // INCLUDED_CModelMemoryTest_h
