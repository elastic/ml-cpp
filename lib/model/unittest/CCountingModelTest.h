/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CCountingModelTest_h
#define INCLUDED_CCountingModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CCountingModelTest : public CppUnit::TestFixture
{
    public:
        void testSkipSampling(void);
        void testCheckScheduledEvents(void);
        static CppUnit::Test *suite(void);
    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CCountingModelTest_h

