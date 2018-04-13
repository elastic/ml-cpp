/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CInterimBucketCorrectorTest_h
#define INCLUDED_CInterimBucketCorrectorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CInterimBucketCorrectorTest : public CppUnit::TestFixture {
public:
    void testCorrectionsGivenSingleValue();
    void testCorrectionsGivenSingleValueAndNoBaseline();
    void testCorrectionsGivenMultiValueAndMultiMode();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CInterimBucketCorrectorTest_h
