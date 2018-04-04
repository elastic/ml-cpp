/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CLineifiedXmlInputParserTest_h
#define INCLUDED_CLineifiedXmlInputParserTest_h

#include <cppunit/extensions/HelperMacros.h>


class CLineifiedXmlInputParserTest : public CppUnit::TestFixture
{
    public:
        void testThroughputArbitraryConformant();
        void testThroughputCommonConformant();
        void testThroughputArbitraryRapid();
        void testThroughputCommonRapid();

        static CppUnit::Test *suite();

    private:
        template <typename PARSER>
        void runTest(bool allDocsSameStructure);
};

#endif // INCLUDED_CLineifiedXmlInputParserTest_h

