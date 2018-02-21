/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CPatternSetTest_h
#define INCLUDED_CPatternSetTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPatternSetTest : public CppUnit::TestFixture
{
    public:
        void testInitFromJson_GivenInvalidJson(void);
        void testInitFromJson_GivenNonArray(void);
        void testInitFromJson_GivenArrayWithNonStringItem(void);
        void testInitFromJson_GivenArrayWithDuplicates(void);
        void testContains_GivenFullMatchKeys(void);
        void testContains_GivenPrefixKeys(void);
        void testContains_GivenSuffixKeys(void);
        void testContains_GivenContainsKeys(void);
        void testContains_GivenMixedKeys(void);
        void testClear(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CPatternSetTest_h
