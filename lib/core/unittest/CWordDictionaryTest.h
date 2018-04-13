/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CWordDictionaryTest_h
#define INCLUDED_CWordDictionaryTest_h

#include <cppunit/extensions/HelperMacros.h>


class CWordDictionaryTest : public CppUnit::TestFixture
{
    public:
        void testLookups();
        void testPartOfSpeech();
        void testWeightingFunctors();
        void testPerformance();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CWordDictionaryTest_h

