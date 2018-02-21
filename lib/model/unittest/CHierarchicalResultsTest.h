/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CHierarchicalResultsTest_h
#define INCLUDED_CHierarchicalResultsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CHierarchicalResultsTest : public CppUnit::TestFixture
{
    public:
        void testBreadthFirstVisit(void);
        void testDepthFirstVisit(void);
        void testBuildHierarchy(void);
        void testBuildHierarchyGivenPartitionsWithSinglePersonFieldValue(void);
        void testBasicVisitor(void);
        void testAggregator(void);
        void testInfluence(void);
        void testScores(void);
        void testWriter(void);
        void testNormalizer(void);
        void testDetectorEqualizing(void);
        void testShouldWritePartition(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CHierarchicalResultsTest_h
