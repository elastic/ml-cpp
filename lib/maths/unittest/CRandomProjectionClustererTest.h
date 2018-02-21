/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CRandomProjectionClustererTest_h
#define INCLUDED_CRandomProjectionClustererTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRandomProjectionClustererTest : public CppUnit::TestFixture
{
    public:
        void testGenerateProjections(void);
        void testClusterProjections(void);
        void testNeighbourhoods(void);
        void testSimilarities(void);
        void testClusterNeighbourhoods(void);
        void testAccuracy(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CRandomProjectionClustererTest_h
