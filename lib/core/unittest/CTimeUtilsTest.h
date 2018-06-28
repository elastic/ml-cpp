/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTimeUtilsTest_h
#define INCLUDED_CTimeUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTimeUtilsTest : public CppUnit::TestFixture {
public:
    void testNow();
    void testToIso8601();
    void testToLocal();
    void testToEpochMs();
    void testStrptime();
    void testTimezone();
    void testDateWords();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTimeUtilsTest_h
