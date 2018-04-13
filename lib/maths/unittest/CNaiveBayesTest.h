/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CNaiveBayesTest_h
#define INCLUDED_CNaiveBayesTest_h

#include <cppunit/extensions/HelperMacros.h>

class CNaiveBayesTest : public CppUnit::TestFixture {
public:
    void testClassification();
    void testPropagationByTime();
    void testMemoryUsage();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CNaiveBayesTest_h
