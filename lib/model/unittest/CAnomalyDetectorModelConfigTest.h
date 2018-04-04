/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CAnomalyDetectorModelConfigTest_h
#define INCLUDED_CAnomalyDetectorModelConfigTest_h

#include <cppunit/extensions/HelperMacros.h>


class CAnomalyDetectorModelConfigTest : public CppUnit::TestFixture
{
    public:
        void testNormal();
        void testErrors();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CAnomalyDetectorModelConfigTest_h
