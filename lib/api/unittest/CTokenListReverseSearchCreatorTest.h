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
#ifndef INCLUDED_CTokenListReverseSearchCreatorTest_h
#define INCLUDED_CTokenListReverseSearchCreatorTest_h

#include <cppunit/extensions/HelperMacros.h>


class CTokenListReverseSearchCreatorTest : public CppUnit::TestFixture {
    public:
        void testCostOfToken(void);
        void testCreateNullSearch(void);
        void testCreateNoUniqueTokenSearch(void);
        void testInitStandardSearch(void);
        void testAddCommonUniqueToken(void);
        void testAddInOrderCommonToken(void);
        void testCloseStandardSearch(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CTokenListReverseSearchCreatorTest_h

