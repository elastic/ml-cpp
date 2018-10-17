/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CAnomalyScoreTest_h
#define INCLUDED_CAnomalyScoreTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <string>
#include <vector>

class CAnomalyScoreTest : public CppUnit::TestFixture {
public:
    using TDoubleVec = std::vector<double>;

public:
    void testComputeScores();
    void testNormalizeScoresQuantiles();
    void testNormalizeScoresQuantilesMultiplePartitions();
    void testNormalizeScoresNoisy();
    void testNormalizeScoresLargeScore();
    void testNormalizeScoresPerPartitionMaxScore();
    void testNormalizeScoresNearZero();
    void testNormalizeScoresOrdering();
    void testNormalizerGetMaxScore();
    void testJsonConversion();
    void testPersistEmpty();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CAnomalyScoreCalculatorTest_h
