/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CNdJsonOutputWriterTest_h
#define INCLUDED_CNdJsonOutputWriterTest_h

#include <cppunit/extensions/HelperMacros.h>

class CNdJsonOutputWriterTest : public CppUnit::TestFixture {
public:
    void testStringOutput();
    void testNumericOutput();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CNdJsonOutputWriterTest_h
