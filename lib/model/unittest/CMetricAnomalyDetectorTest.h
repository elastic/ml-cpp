/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMetricAnomalyDetectorTest_h
#define INCLUDED_CMetricAnomalyDetectorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMetricAnomalyDetectorTest : public CppUnit::TestFixture
{
    public:
        void testAnomalies(void);
        void testPersist(void);
        void testExcludeFrequent(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMetricAnomalyDetectorTest_h

