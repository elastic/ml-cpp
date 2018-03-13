/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_CAnomalyScoreTest_h
#define INCLUDED_CAnomalyScoreTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <string>
#include <vector>

class CAnomalyScoreTest : public CppUnit::TestFixture {
public:
    typedef std::vector<double> TDoubleVec;

public:
    void testComputeScores(void);
    void testNormalizeScoresQuantiles(void);
    void testNormalizeScoresNoisy(void);
    void testNormalizeScoresLargeScore(void);
    void testNormalizeScoresNearZero(void);
    void testNormalizeScoresOrdering(void);
    void testJsonConversion(void);
    void testPersistEmpty(void);

    static CppUnit::Test *suite(void);
};

#endif// INCLUDED_CAnomalyScoreCalculatorTest_h
