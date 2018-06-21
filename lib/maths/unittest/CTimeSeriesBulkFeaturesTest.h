/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTimeSeriesBulkFeaturesTest_h
#define INCLUDED_CTimeSeriesBulkFeaturesTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTimeSeriesBulkFeaturesTest : public CppUnit::TestFixture {
public:
    void testMean();
    void testContrast();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTimeSeriesBulkFeaturesTest_h
