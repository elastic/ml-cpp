/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTrendComponentTest_h
#define INCLUDED_CTrendComponentTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTrendComponentTest : public CppUnit::TestFixture
{
    public:
        void testValueAndVariance();
        void testDecayRate();
        void testForecast();
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CTrendComponentTest_h
