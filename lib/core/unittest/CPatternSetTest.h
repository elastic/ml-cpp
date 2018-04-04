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
        void testInitFromJson_GivenInvalidJson();
        void testInitFromJson_GivenNonArray();
        void testInitFromJson_GivenArrayWithNonStringItem();
        void testInitFromJson_GivenArrayWithDuplicates();
        void testContains_GivenFullMatchKeys();
        void testContains_GivenPrefixKeys();
        void testContains_GivenSuffixKeys();
        void testContains_GivenContainsKeys();
        void testContains_GivenMixedKeys();
        void testClear();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CPatternSetTest_h
