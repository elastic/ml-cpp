/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CRadialBasisFunctionTest_h
#define INCLUDED_CRadialBasisFunctionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRadialBasisFunctionTest : public CppUnit::TestFixture {
public:
    void testDerivative();
    void testMean();
    void testMeanSquareDerivative();
    void testProduct();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CRadialBasisFunctionTest_h
