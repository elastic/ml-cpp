/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CImputerTest_h
#define INCLUDED_CImputerTest_h

#include <cppunit/extensions/HelperMacros.h>

class CImputerTest : public CppUnit::TestFixture {
public:
    void testRandom();
    void testNearestNeighbourPlain();
    void testNearestNeighbourBaggedSamples();
    void testNearestNeighbourBaggedAttributes();
    void testNearestNeighbourRandom();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CImputerTest_h
