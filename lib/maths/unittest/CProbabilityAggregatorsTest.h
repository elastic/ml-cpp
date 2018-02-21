/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CProbabilityAggregatorsTest_h
#define INCLUDED_CProbabilityAggregatorsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CProbabilityAggregatorsTest : public CppUnit::TestFixture
{
    public:
        void testJointProbabilityOfLessLikelySamples(void);
        void testLogJointProbabilityOfLessLikelySamples(void);
        void testProbabilityOfExtremeSample(void);
        void testProbabilityOfMFromNExtremeSamples(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CProbabilityAggregatorsTest_h
