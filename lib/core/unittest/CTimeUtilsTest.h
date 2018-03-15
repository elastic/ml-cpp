/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTimeUtilsTest_h
#define INCLUDED_CTimeUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>


class CTimeUtilsTest : public CppUnit::TestFixture
{
    public:
        void testNow(void);
        void testToIso8601(void);
        void testToLocal(void);
        void testToEpochMs(void);
        void testStrptime(void);
        void testTimezone(void);
        void testDateWords(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CTimeUtilsTest_h

