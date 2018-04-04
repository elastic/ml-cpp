/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDelimiterTest_h
#define INCLUDED_CDelimiterTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <iterator>
#include <string>

class CDelimiterTest : public CppUnit::TestFixture {
public:
    void testSimpleTokenise();
    void testRegexTokenise();
    void testQuotedTokenise();
    void testQuotedEscapedTokenise();
    void testInvalidQuotedTokenise();
    void testQuoteEqualsEscapeTokenise();

    static CppUnit::Test* suite();

private:
    typedef std::ostream_iterator<std::string> TStrOStreamItr;
};

#endif // INCLUDED_CDelimiterTest_h
