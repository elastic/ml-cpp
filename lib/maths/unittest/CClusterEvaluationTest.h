/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CClusterEvaluationTest_h
#define INCLUDED_CClusterEvaluationTest_h

#include <cppunit/extensions/HelperMacros.h>

class CClusterEvaluationTest : public CppUnit::TestFixture {
public:
    void testSilhouetteExact();
    void testSilhouetteApprox();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CClusterEvaluationTest_h
