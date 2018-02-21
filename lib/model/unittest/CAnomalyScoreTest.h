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


class CAnomalyScoreTest : public CppUnit::TestFixture
{
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

#endif // INCLUDED_CAnomalyScoreCalculatorTest_h

