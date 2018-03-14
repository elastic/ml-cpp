/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
        void testValidateDuration();
        void testValidateDefaultExpiry();
        void testValidateNoExpiry();
        void testValidateInvalidExpiry();
        void testValidateBrokenMessage();
        void testValidateMissingId();

        static CppUnit::Test *suite();
};


#endif // INCLUDED_CForecastRunnerTest_h
