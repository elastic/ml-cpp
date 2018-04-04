/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CRapidJsonWriterBaseTest_h
#define INCLUDED_CRapidJsonWriterBaseTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRapidJsonWriterBaseTest : public CppUnit::TestFixture {
public:
    void testAddFields();
    void testRemoveMemberIfPresent();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CRapidJsonWriterBaseTest_h
