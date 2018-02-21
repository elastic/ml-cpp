/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CLengthEncodedInputParserTest_h
#define INCLUDED_CLengthEncodedInputParserTest_h

#include <cppunit/extensions/HelperMacros.h>


class CLengthEncodedInputParserTest : public CppUnit::TestFixture
{
    public:
        void testCsvEquivalence(void);
        void testThroughput(void);
        void testCorruptStreamDetection(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CLengthEncodedInputParserTest_h

