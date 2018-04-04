/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStringStoreTest_h
#define INCLUDED_CStringStoreTest_h

#include <core/CoreTypes.h>

#include <cppunit/extensions/HelperMacros.h>

class CStringStoreTest : public CppUnit::TestFixture {
public:
    void testPersonStringPruning();
    void testAttributeStringPruning();
    void testInfluencerStringPruning();

    static CppUnit::Test* suite();

private:
    bool nameExists(const std::string& string);
    bool influencerExists(const std::string& string);
};

#endif // INCLUDED_CStringStoreTest_h
