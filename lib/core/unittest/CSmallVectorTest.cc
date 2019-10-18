/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CSmallVector.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CSmallVectorTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testNonStandard) {
    using TDoubleVec = std::vector<double>;
    using TDouble5Vec = core::CSmallVector<double, 5>;

    // Test construction and conversion to a std::vector.
    {
        TDoubleVec vec{0.1, 1.4, 7.4};
        TDouble5Vec svec(vec);
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(vec),
                          core::CContainerPrinter::print(svec));

        TDoubleVec cvec(svec);
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(vec),
                          core::CContainerPrinter::print(cvec));
    }

    // Test addition and subtraction.
    {
        TDouble5Vec vec1{1.0, 3.2, 1.4, 7.3};
        TDouble5Vec vec2{1.3, 1.6, 2.2, 1.6};

        vec1 -= vec2;
        BOOST_CHECK_EQUAL(std::string("[-0.3, 1.6, -0.8, 5.7]"),
                          core::CContainerPrinter::print(vec1));

        vec1 += vec2;
        vec1 += vec2;
        BOOST_CHECK_EQUAL(std::string("[2.3, 4.8, 3.6, 8.9]"),
                          core::CContainerPrinter::print(vec1));
    }
}

BOOST_AUTO_TEST_SUITE_END()
