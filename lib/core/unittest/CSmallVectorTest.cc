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

#include "CSmallVectorTest.h"

#include <core/CContainerPrinter.h>
#include <core/CSmallVector.h>

#include <vector>

using namespace ml;

void CSmallVectorTest::testNonStandard() {
    using TDoubleVec = std::vector<double>;
    using TDouble5Vec = core::CSmallVector<double, 5>;

    // Test construction and conversion to a std::vector.
    {
        TDoubleVec vec{0.1, 1.4, 7.4};
        TDouble5Vec svec(vec);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(vec), core::CContainerPrinter::print(svec));

        TDoubleVec cvec(svec);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(vec), core::CContainerPrinter::print(cvec));
    }

    // Test addition and subtraction.
    {
        TDouble5Vec vec1{1.0, 3.2, 1.4, 7.3};
        TDouble5Vec vec2{1.3, 1.6, 2.2, 1.6};

        vec1 -= vec2;
        CPPUNIT_ASSERT_EQUAL(std::string("[-0.3, 1.6, -0.8, 5.7]"), core::CContainerPrinter::print(vec1));

        vec1 += vec2;
        vec1 += vec2;
        CPPUNIT_ASSERT_EQUAL(std::string("[2.3, 4.8, 3.6, 8.9]"), core::CContainerPrinter::print(vec1));
    }
}

CppUnit::Test* CSmallVectorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSmallVectorTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CSmallVectorTest>("CSmallVectorTest::testNonStandard", &CSmallVectorTest::testNonStandard));

    return suiteOfTests;
}
