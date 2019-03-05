/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CForecastRunnerTest_h
#define INCLUDED_CForecastRunnerTest_h

#include <cppunit/extensions/HelperMacros.h>

//! \brief
//! Module tests for forecasting functionality
//!
//! DESCRIPTION:\n
//! A couple of module tests of forecast including regression tests
class CForecastRunnerTest : public CppUnit::TestFixture {
public:
    void testSummaryCount();
    void testPopulation();
    void testRare();
    void testInsufficientData();
    void testValidateDefaultExpiry();
    void testValidateNoExpiry();
    void testValidateInvalidExpiry();
    void testValidateBrokenMessage();
    void testValidateMissingId();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CForecastRunnerTest_h
