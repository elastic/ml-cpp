/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_CPatternSetTest_h
#define INCLUDED_CPatternSetTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPatternSetTest : public CppUnit::TestFixture {
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
