/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStringSimilarityTesterTest_h
#define INCLUDED_CStringSimilarityTesterTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStringSimilarityTesterTest : public CppUnit::TestFixture
{
    public:
        void testStringSimilarity();
        void testLevensteinDistance();
        void testLevensteinDistance2();
        void testLevensteinDistanceThroughputDifferent();
        void testLevensteinDistanceThroughputSimilar();
        void testLevensteinDistanceAlgorithmEquivalence();
        void testWeightedEditDistance();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CStringSimilarityTesterTest_h

