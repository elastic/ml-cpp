/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCsvInputParserTest_h
#define INCLUDED_CCsvInputParserTest_h

#include <cppunit/extensions/HelperMacros.h>


class CCsvInputParserTest : public CppUnit::TestFixture
{
    public:
        void testSimpleDelims(void);
        void testComplexDelims(void);
        void testThroughput(void);
        void testDateParse(void);
        void testQuoteParsing(void);
        void testLineParser(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CCsvInputParserTest_h

