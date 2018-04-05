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

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CPatternSetTest_h
