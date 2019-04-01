/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLbfgsTest_h
#define INCLUDED_CLbfgsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLbfgsTest : public CppUnit::TestFixture {
public:
    void testQuadtratic();
    void testSingularHessian();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLbfgsTest_h
