/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTimeSeriesSegmentationTest_h
#define INCLUDED_CTimeSeriesSegmentationTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTimeSeriesSegmentationTest : public CppUnit::TestFixture {
public:
    void testTopDownPiecewiseLinear();
    void testTopDownPeriodicPiecewiseLinearScaling();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTimeSeriesSegmentationTest_h
