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
        void testProbabilityOfLessLikelyMultinomialSample();
        void testProbabilityOfLessLikelyCategoryCount();
        void testExpectedDistinctCategories();
        void testLogBinomialProbability();
        void testLogMultinomialProbability();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CCategoricalToolsTest_h
