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

#ifndef INCLUDED_CMemoryUsageTest_h
#define INCLUDED_CMemoryUsageTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMemoryUsageTest : public CppUnit::TestFixture {
public:
    void testUsage(void);
    void testDebug(void);
    void testDynamicSizeAlwaysZero(void);
    void testCompress(void);
    void testStringBehaviour(void);
    void testStringMemory(void);
    void testStringClear(void);
    void testSharedPointer(void);
    void testRawPointer(void);
    void testSmallVector(void);

    static CppUnit::Test *suite(void);
};

#endif// INCLUDED_CMemoryUsageTest_h
