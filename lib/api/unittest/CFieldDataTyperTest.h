/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CFieldDataTyperTest_h
#define INCLUDED_CFieldDataTyperTest_h

#include <core/CoreTypes.h>

#include <cppunit/extensions/HelperMacros.h>

class CFieldDataTyperTest : public CppUnit::TestFixture
{
    public:
        void testAll(void);
        void testNodeReverseSearch(void);
        void testPassOnControlMessages(void);
        void testHandleControlMessages(void);
        void testRestoreStateFailsWithEmptyState(void);

        static CppUnit::Test *suite();

};

#endif // INCLUDED_CFieldDataTyperTest_h
