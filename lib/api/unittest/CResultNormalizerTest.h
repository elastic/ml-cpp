/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CResultNormalizerTest_h
#define INCLUDED_CResultNormalizerTest_h

#include <cppunit/extensions/HelperMacros.h>


class CResultNormalizerTest : public CppUnit::TestFixture
{
    public:
        void testInitNormalizer(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CResultNormalizerTest_h

