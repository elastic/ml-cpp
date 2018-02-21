/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CLineifiedJsonOutputWriterTest_h
#define INCLUDED_CLineifiedJsonOutputWriterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CLineifiedJsonOutputWriterTest : public CppUnit::TestFixture
{
    public:
        void testStringOutput(void);
        void testNumericOutput(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CLineifiedJsonOutputWriterTest_h

