/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

class CVectorRangeTest : public CppUnit::TestFixture {
    public:
        void testCreation(void);
        void testAccessors(void);
        void testIterators(void);
        void testSizing(void);
        void testModifiers(void);
        void testComparisons(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CTrendTestsTest_h
