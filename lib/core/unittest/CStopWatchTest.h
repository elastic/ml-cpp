/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStopWatchTest_h
#define INCLUDED_CStopWatchTest_h

#include <cppunit/extensions/HelperMacros.h>


class CStopWatchTest : public CppUnit::TestFixture
{
    public:
        void testStopWatch(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CStopWatchTest_h

