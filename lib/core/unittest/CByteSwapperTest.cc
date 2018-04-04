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
#include "CByteSwapperTest.h"

#include <core/CByteSwapper.h>
#include <core/CLogger.h>

#include <stdint.h>


CppUnit::Test *CByteSwapperTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CByteSwapperTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CByteSwapperTest>(
                                   "CByteSwapperTest::testByteSwaps",
                                   &CByteSwapperTest::testByteSwaps) );

    return suiteOfTests;
}

void CByteSwapperTest::testByteSwaps()
{
    uint8_t type1(0x12);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type1) == 0x12);

    int8_t type2(0x21);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type2) == 0x21);

    uint16_t type3(0x1234);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type3) == 0x3412);

    int16_t type4(0x4321);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type4) == 0x2143);

    uint32_t type5(0x12345678);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type5) == 0x78563412);

    int32_t type6(0x87654321);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type6) == 0x21436587);

    // Need to explicitly specify the types of the constants for these two, as
    // they would overflow an int
    uint64_t type7(0x123456789ABCDEF0ULL);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type7) == 0xF0DEBC9A78563412ULL);

    int64_t type8(0x0FEDCBA987654321LL);
    CPPUNIT_ASSERT(ml::core::CByteSwapper::swapBytes(type8) == 0x21436587A9CBED0FLL);
}

