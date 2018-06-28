/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPcaTest_h
#define INCLUDED_CPcaTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPcaTest : public CppUnit::TestFixture {
public:
    void testProjectOntoPrincipleComponents(void);
    void testSparseProjectOntoPrincipleComponents(void);
    void testNumericRank(void);

    static CppUnit::Test* suite(void);
};

#endif // INCLUDED_CPcaTest_h
