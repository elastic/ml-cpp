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

