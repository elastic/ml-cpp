/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSignalTest_h
#define INCLUDED_CSignalTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSignalTest : public CppUnit::TestFixture {
public:
    void testFFTVersusOctave();
    void testIFFTVersusOctave();
    void testFFTRandomized();
    void testIFFTRandomized();
    void testFFTIFFTIdempotency();
    void testAutocorrelations();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CSignalTest_h
