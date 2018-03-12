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

#ifndef INCLUDED_CGramSchmidtTest_h
#define INCLUDED_CGramSchmidtTest_h

#include <cppunit/extensions/HelperMacros.h>

class CGramSchmidtTest : public CppUnit::TestFixture {
    public:
        void testOrthogonality(void);
        void testNormalisation(void);
        void testSpan(void);
        void testEdgeCases(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CGramSchmidtTest_h
