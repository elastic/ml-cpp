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

#ifndef INCLUDED_CPriorTest_h
#define INCLUDED_CPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPriorTest : public CppUnit::TestFixture {
public:
    void testExpectation(void);

    static CppUnit::Test *suite(void);
};

#endif// INCLUDED_CPriorTest_h
