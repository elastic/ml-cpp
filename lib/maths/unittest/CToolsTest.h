/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CToolsTest_h
#define INCLUDED_CToolsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CToolsTest : public CppUnit::TestFixture
{
    public:
        void testProbabilityOfLessLikelySample(void);
        void testIntervalExpectation(void);
        void testMixtureProbabilityOfLessLikelySample(void);
        void testAnomalyScore(void);
        void testSpread(void);
        void testFastLog(void);
        void testMiscellaneous(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CToolsTest_h
