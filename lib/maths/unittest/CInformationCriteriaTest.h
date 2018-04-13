/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CInformationCriteriaTest_h
#define INCLUDED_CInformationCriteriaTest_h

#include <cppunit/extensions/HelperMacros.h>

class CInformationCriteriaTest : public CppUnit::TestFixture
{
    public:
        void testSphericalGaussian();
        void testSphericalGaussianWithSphericalCluster();
        void testGaussian();
        void testGaussianWithSphericalCluster();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CInformationCriteriaTest_h
