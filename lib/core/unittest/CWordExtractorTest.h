/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CWordExtractorTest_h
#define INCLUDED_CWordExtractorTest_h

#include <cppunit/extensions/HelperMacros.h>


class CWordExtractorTest : public CppUnit::TestFixture
{
    public:
        void testWordExtract();
        void testMinConsecutive();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CWordExtractorTest_h

