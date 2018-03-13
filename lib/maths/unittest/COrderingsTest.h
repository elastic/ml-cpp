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

#ifndef INCLUDED_COrderingsTest_h
#define INCLUDED_COrderingsTest_h

#include <cppunit/extensions/HelperMacros.h>

class COrderingsTest : public CppUnit::TestFixture {
public:
    void testOptionalOrdering(void);
    void testPtrOrdering(void);
    void testLess(void);
    void testFirstLess(void);
    void testFirstGreater(void);
    void testSecondLess(void);
    void testSecondGreater(void);
    void testDereference(void);
    void testLexicographicalCompare(void);
    void testSimultaneousSort(void);

    static CppUnit::Test *suite(void);
};

#endif// INCLUDED_COrderingsTest_h
