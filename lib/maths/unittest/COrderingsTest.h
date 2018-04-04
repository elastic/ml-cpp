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
    void testOptionalOrdering();
    void testPtrOrdering();
    void testLess();
    void testFirstLess();
    void testFirstGreater();
    void testSecondLess();
    void testSecondGreater();
    void testDereference();
    void testLexicographicalCompare();
    void testSimultaneousSort();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_COrderingsTest_h
