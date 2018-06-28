/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTokenListReverseSearchCreatorTest_h
#define INCLUDED_CTokenListReverseSearchCreatorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTokenListReverseSearchCreatorTest : public CppUnit::TestFixture {
public:
    void testCostOfToken();
    void testCreateNullSearch();
    void testCreateNoUniqueTokenSearch();
    void testInitStandardSearch();
    void testAddCommonUniqueToken();
    void testAddInOrderCommonToken();
    void testCloseStandardSearch();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTokenListReverseSearchCreatorTest_h
