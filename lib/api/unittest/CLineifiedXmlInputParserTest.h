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
        void testThroughputArbitraryConformant(void);
        void testThroughputCommonConformant(void);
        void testThroughputArbitraryRapid(void);
        void testThroughputCommonRapid(void);

        static CppUnit::Test *suite();

    private:
        template <typename PARSER>
        void runTest(bool allDocsSameStructure);
};

#endif // INCLUDED_CLineifiedXmlInputParserTest_h

