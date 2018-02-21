/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCategoryExamplesCollectorTest_h
#define INCLUDED_CCategoryExamplesCollectorTest_h

#include <cppunit/extensions/HelperMacros.h>


class CCategoryExamplesCollectorTest : public CppUnit::TestFixture
{
    public:
        void testAddGivenMaxExamplesIsZero(void);
        void testAddGivenSameCategoryExamplePairAddedTwice(void);
        void testAddGivenMoreThanMaxExamplesAreAddedForSameCategory(void);
        void testAddGivenCategoryAddedIsNotSubsequent(void);
        void testExamples(void);
        void testPersist(void);
        void testTruncation(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CCategoryExamplesCollectorTest_h

