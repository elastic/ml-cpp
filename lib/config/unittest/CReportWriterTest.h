/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CReportWriterTest_h
#define INCLUDED_CReportWriterTest_h

#include <cppunit/extensions/HelperMacros.h>

class CReportWriterTest : public CppUnit::TestFixture {
public:
    void testPretty();
    void testJSON();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CReportWriterTest_h
