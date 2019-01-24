/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameTest_h
#define INCLUDED_CDataFrameTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameTest : public CppUnit::TestFixture {
public:
    void setUp();
    void tearDown();
    void testInMainMemoryBasicReadWrite();
    void testInMainMemoryParallelRead();
    void testOnDiskBasicReadWrite();
    void testOnDiskParallelRead();
    void testMemoryUsage();
    void testReserve();
    void testResizeColumns();
    void testWriteColumns();
    void testDocHashes();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameTest_h
