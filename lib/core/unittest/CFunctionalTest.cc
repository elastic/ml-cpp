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

#include "CFunctionalTest.h"

#include <core/CFunctional.h>

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/shared_ptr.hpp>

using namespace ml;

void CFunctionalTest::testIsNull() {
    core::CFunctional::SIsNull isNull;

    {
        double five = 5.0;
        double* null = nullptr;
        const double* notNull = &five;
        CPPUNIT_ASSERT(isNull(null));
        CPPUNIT_ASSERT(!isNull(notNull));
    }
    {
        boost::optional<double> null;
        boost::optional<double> notNull(5.0);
        CPPUNIT_ASSERT(isNull(null));
        CPPUNIT_ASSERT(!isNull(notNull));
    }
    {
        boost::shared_ptr<double> null;
        boost::shared_ptr<double> notNull(new double(5.0));
        CPPUNIT_ASSERT(isNull(null));
        CPPUNIT_ASSERT(!isNull(notNull));
    }
}

void CFunctionalTest::testDereference() {
    double one(1.0);
    double two(2.0);
    double three(3.0);
    const double* null_ = nullptr;

    core::CFunctional::SDereference<core::CFunctional::SIsNull> derefIsNull;
    boost::optional<const double*> null(null_);
    boost::optional<const double*> notNull(&one);
    CPPUNIT_ASSERT(derefIsNull(null));
    CPPUNIT_ASSERT(!derefIsNull(notNull));

    std::less<double> less;
    core::CFunctional::SDereference<std::less<double>> derefLess;
    const double* values[] = {&one, &two, &three};
    for (std::size_t i = 0u; i < boost::size(values); ++i) {
        for (std::size_t j = 0u; j < boost::size(values); ++j) {
            CPPUNIT_ASSERT_EQUAL(less(*values[i], *values[j]),
                                 derefLess(values[i], values[j]));
        }
    }
}

CppUnit::Test* CFunctionalTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CFunctionalTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CFunctionalTest>(
        "CFunctionalTest::testIsNull", &CFunctionalTest::testIsNull));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFunctionalTest>(
        "CFunctionalTest::testDereference", &CFunctionalTest::testDereference));

    return suiteOfTests;
}
