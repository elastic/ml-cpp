/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CBoundingBostTest_h
#define INCLUDED_CBoundingBostTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBoundingBoxTest : public CppUnit::TestFixture {
public:
    void testAdd();
    void testCloserTo();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBoundingBostTest_h
