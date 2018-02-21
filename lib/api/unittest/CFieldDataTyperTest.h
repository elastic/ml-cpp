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
#ifndef INCLUDED_CFieldDataTyperTest_h
#define INCLUDED_CFieldDataTyperTest_h

#include <core/CoreTypes.h>

#include <cppunit/extensions/HelperMacros.h>

class CFieldDataTyperTest : public CppUnit::TestFixture
{
    public:
        void testAll(void);
        void testNodeReverseSearch(void);
        void testPassOnControlMessages(void);
        void testHandleControlMessages(void);

        static CppUnit::Test *suite();

};

#endif // INCLUDED_CFieldDataTyperTest_h
