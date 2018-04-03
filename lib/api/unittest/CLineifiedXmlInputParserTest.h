/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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

