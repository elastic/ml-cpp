/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_BoostTestCloseAbsolute_h
#define INCLUDED_ml_test_BoostTestCloseAbsolute_h

#include <boost/assert.hpp>
#include <boost/test/test_tools.hpp>

#include <ostream>

namespace ml {
namespace test {

template<typename T>
struct SAbsoluteTolerance {

    explicit SAbsoluteTolerance(T tolerance) : s_Tolerance(tolerance) {}
    T s_Tolerance;
};

template<typename T>
SAbsoluteTolerance<T> absoluteTolerance(T tolerance) {
    // no reason for tolerance to be negative
    BOOST_ASSERT_MSG(tolerance >= T(0), "tolerance must not be negative!");
    return SAbsoluteTolerance<T>(tolerance);
}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const SAbsoluteTolerance<T>& tol) {
    return strm << tol.s_Tolerance;
}

class CIsCloseEnough {

public:
    using result_type = boost::test_tools::assertion_result;

public:
    template<typename T>
    result_type operator()(const T& lhs, const T& rhs, const SAbsoluteTolerance<T>& tol) const {

        T diff{(lhs < rhs) ? (rhs - lhs) : (lhs - rhs)};
        result_type result{diff <= tol.s_Tolerance};
        if (!result) {
            result.message() << diff;
        }
        return result;
    }
};
}
}

//! \brief
//! Extra Boost.Test macros for close absolute value.
//!
//! DESCRIPTION:\n
//! Boost.Test provides BOOST_REQUIRE_CLOSE_FRACTION, which
//! looks at the relative difference between two values.  If
//! we were starting from scratch this would be fine but we
//! have hundreds of assertions that were using
//! CPPUNIT_ASSERT_DOUBLES_EQUAL, which looks at the absolute
//! difference between two values.  These macros written in the
//! Boost.Test style are a drop-in replacement for
//! CPPUNIT_ASSERT_DOUBLES_EQUAL, asserting on the absolute
//! difference.
//!
//! IMPLEMENTATION DECISIONS:\n
//! They could have been implemented using BOOST_REQUIRE_PREDICATE,
//! but use a lower level integration into Boost.Test so that
//! the failure messages they generate can be in the same
//! format that BOOST_REQUIRE_CLOSE_FRACTION generates.  As time
//! goes by and we have a mix of BOOST_REQUIRE_CLOSE_FRACTION and
//! BOOST_REQUIRE_CLOSE_ABSOLUTE it will be better to have a
//! consistent failure message format.
//!

#define BOOST_WARN_CLOSE_ABSOLUTE(L, R, T)                                              \
    BOOST_TEST_TOOL_IMPL(0, ml::test::CIsCloseEnough(), "", WARN, CHECK_CLOSE_FRACTION, \
                         (L)(R)(ml::test::absoluteTolerance(T)))
#define BOOST_CHECK_CLOSE_ABSOLUTE(L, R, T)                                              \
    BOOST_TEST_TOOL_IMPL(0, ml::test::CIsCloseEnough(), "", CHECK, CHECK_CLOSE_FRACTION, \
                         (L)(R)(ml::test::absoluteTolerance(T)))
#define BOOST_REQUIRE_CLOSE_ABSOLUTE(L, R, T)                                              \
    BOOST_TEST_TOOL_IMPL(0, ml::test::CIsCloseEnough(), "", REQUIRE, CHECK_CLOSE_FRACTION, \
                         (L)(R)(ml::test::absoluteTolerance(T)))

#endif // INCLUDED_ml_test_BoostTestCloseAbsolute_h
