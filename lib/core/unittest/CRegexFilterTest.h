/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CRegexFilterTest_h
#define INCLUDED_CRegexFilterTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRegexFilterTest : public CppUnit::TestFixture {
public:
    void testConfigure_GivenInvalidRegex();
    void testApply_GivenEmptyFilter();
    void testApply_GivenSingleMatchAllRegex();
    void testApply_GivenSingleRegex();
    void testApply_GivenMultipleRegex();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CRegexFilterTest_h
