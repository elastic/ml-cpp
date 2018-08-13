/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CHierarchicalResultsTest_h
#define INCLUDED_CHierarchicalResultsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CHierarchicalResultsTest : public CppUnit::TestFixture {
public:
    void testBreadthFirstVisit();
    void testDepthFirstVisit();
    void testBuildHierarchy();
    void testBuildHierarchyGivenPartitionsWithSinglePersonFieldValue();
    void testBasicVisitor();
    void testAggregator();
    void testInfluence();
    void testScores();
    void testWriter();
    void testNormalizer();
    void testDetectorEqualizing();
    void testShouldWritePartition();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CHierarchicalResultsTest_h
