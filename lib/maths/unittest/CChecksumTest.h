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

#ifndef INCLUDED_CChecksumTest_h
#define INCLUDED_CChecksumTest_h

#include <cppunit/extensions/HelperMacros.h>

class CChecksumTest : public CppUnit::TestFixture {
public:
    void testMemberChecksum(void);
    void testContainers(void);
    void testNullable(void);
    void testAccumulators(void);
    void testPair(void);
    void testArray(void);
    void testCombinations(void);

    static CppUnit::Test* suite(void);
};

#endif // INCLUDED_CChecksumTest_h
