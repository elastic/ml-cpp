/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CCategoricalToolsTest_h
#define INCLUDED_CCategoricalToolsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCategoricalToolsTest : public CppUnit::TestFixture
{
    public:
        void testProbabilityOfLessLikelyMultinomialSample(void);
        void testProbabilityOfLessLikelyCategoryCount(void);
        void testExpectedDistinctCategories(void);
        void testLogBinomialProbability(void);
        void testLogMultinomialProbability(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CCategoricalToolsTest_h
