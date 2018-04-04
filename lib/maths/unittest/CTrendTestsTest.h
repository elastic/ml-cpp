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

#ifndef INCLUDED_CTrendTestsTest_h
#define INCLUDED_CTrendTestsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTrendTestsTest : public CppUnit::TestFixture {
public:
    void testRandomizedPeriodicity();
    void testCalendarCyclic();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTrendTestsTest_h
