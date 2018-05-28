/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CSystemCallFilterTest_h
#define INCLUDED_CSystemCallFilterTest_h

#include <core/CNamedPipeFactory.h>

#include <cppunit/extensions/HelperMacros.h>

class CSystemCallFilterTest : public CppUnit::TestFixture {
public:
    void testSystemCallFilter();

    static CppUnit::Test* suite();

private:
    void openPipeAndRead(const std::string& filename);
    void openPipeAndWrite(const std::string& filename);
};

#endif // INCLUDED_CSystemCallFilterTest_h
