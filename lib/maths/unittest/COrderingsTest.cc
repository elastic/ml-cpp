/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CSmallVector.h>

#include <maths/COrderings.h>

#include <test/CRandomNumbers.h>

#include <boost/optional.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>

#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(COrderingsTest)

using namespace ml;

namespace {

using TOptionalDouble = boost::optional<double>;

class CDictionary {
public:
    using TStrVec = std::vector<std::string>;
    static std::size_t ms_Copies;

public:
    CDictionary(const TStrVec& words) : m_Words(words) {}

    CDictionary& operator=(const CDictionary& other) {
        ++ms_Copies;
        m_Words = other.m_Words;
        return *this;
    }

    void swap(CDictionary& other) { m_Words.swap(other.m_Words); }

    std::string print() const {
        return core::CContainerPrinter::print(m_Words);
    }

private:
    TStrVec m_Words;
};

std::size_t CDictionary::ms_Copies(0u);

void swap(CDictionary& lhs, CDictionary& rhs) {
    lhs.swap(rhs);
}
}

BOOST_AUTO_TEST_CASE(testOptionalOrdering) {
    TOptionalDouble null;
    TOptionalDouble one(1.0);
    TOptionalDouble two(2.0);
    TOptionalDouble big(std::numeric_limits<double>::max());

    maths::COrderings::SOptionalLess less;

    BOOST_TEST_REQUIRE(less(big, null));
    BOOST_TEST_REQUIRE(!less(null, big));
    BOOST_TEST_REQUIRE(!less(null, null));
    BOOST_TEST_REQUIRE(less(100.0, null));
    BOOST_TEST_REQUIRE(!less(null, 100.0));
    BOOST_TEST_REQUIRE(less(one, two));
    BOOST_TEST_REQUIRE(less(one, 2.0));
    BOOST_TEST_REQUIRE(!less(two, one));
    BOOST_TEST_REQUIRE(!less(2.0, one));
    BOOST_TEST_REQUIRE(!less(one, one));
    BOOST_TEST_REQUIRE(less(one, big));

    maths::COrderings::SOptionalGreater greater;

    BOOST_TEST_REQUIRE(!greater(big, null));
    BOOST_TEST_REQUIRE(greater(null, big));
    BOOST_TEST_REQUIRE(!greater(null, null));
    BOOST_TEST_REQUIRE(!greater(100.0, null));
    BOOST_TEST_REQUIRE(greater(null, 100.0));
    BOOST_TEST_REQUIRE(!greater(one, two));
    BOOST_TEST_REQUIRE(!greater(one, 2.0));
    BOOST_TEST_REQUIRE(greater(two, one));
    BOOST_TEST_REQUIRE(greater(2.0, one));
    BOOST_TEST_REQUIRE(!greater(one, one));
    BOOST_TEST_REQUIRE(!greater(one, big));
}

BOOST_AUTO_TEST_CASE(testPtrOrdering) {
    const double* null = nullptr;
    double one_(1.0);
    double two_(2.0);
    double hundred_(100.0);
    double big_(std::numeric_limits<double>::max());
    const double* one(&one_);
    const double* two(&two_);
    const double* hundred(&hundred_);
    const double* big(&big_);

    maths::COrderings::SPtrLess less;

    BOOST_TEST_REQUIRE(less(big, null));
    BOOST_TEST_REQUIRE(!less(null, big));
    BOOST_TEST_REQUIRE(!less(null, null));
    BOOST_TEST_REQUIRE(less(hundred, null));
    BOOST_TEST_REQUIRE(!less(null, hundred));
    BOOST_TEST_REQUIRE(less(one, two));
    BOOST_TEST_REQUIRE(!less(two, one));
    BOOST_TEST_REQUIRE(!less(one, one));
    BOOST_TEST_REQUIRE(less(one, big));

    maths::COrderings::SPtrGreater greater;

    BOOST_TEST_REQUIRE(!greater(big, null));
    BOOST_TEST_REQUIRE(greater(null, big));
    BOOST_TEST_REQUIRE(!greater(null, null));
    BOOST_TEST_REQUIRE(!greater(hundred, null));
    BOOST_TEST_REQUIRE(greater(null, hundred));
    BOOST_TEST_REQUIRE(!greater(one, two));
    BOOST_TEST_REQUIRE(greater(two, one));
    BOOST_TEST_REQUIRE(!greater(one, one));
    BOOST_TEST_REQUIRE(!greater(one, big));
}

BOOST_AUTO_TEST_CASE(testLess) {
    maths::COrderings::SLess less;

    {
        TOptionalDouble null;
        TOptionalDouble one(1.0);
        TOptionalDouble two(2.0);
        TOptionalDouble big(std::numeric_limits<double>::max());

        BOOST_TEST_REQUIRE(less(big, null));
        BOOST_TEST_REQUIRE(!less(null, big));
        BOOST_TEST_REQUIRE(!less(null, null));
        BOOST_TEST_REQUIRE(less(one, two));
        BOOST_TEST_REQUIRE(!less(two, one));
        BOOST_TEST_REQUIRE(!less(one, one));
        BOOST_TEST_REQUIRE(less(one, big));
    }
    {
        const double* null = nullptr;
        double one_(1.0);
        double two_(2.0);
        double hundred_(100.0);
        double big_(std::numeric_limits<double>::max());
        const double* one(&one_);
        const double* two(&two_);
        const double* hundred(&hundred_);
        const double* big(&big_);

        BOOST_TEST_REQUIRE(less(big, null));
        BOOST_TEST_REQUIRE(!less(null, big));
        BOOST_TEST_REQUIRE(!less(null, null));
        BOOST_TEST_REQUIRE(less(hundred, null));
        BOOST_TEST_REQUIRE(!less(null, hundred));
        BOOST_TEST_REQUIRE(less(one, two));
        BOOST_TEST_REQUIRE(!less(two, one));
        BOOST_TEST_REQUIRE(!less(one, one));
        BOOST_TEST_REQUIRE(less(one, big));
    }

    double one(1.0);
    double two(2.0);
    double three(3.0);

    {
        BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(one, three), three),
                                std::make_pair(std::make_pair(two, two), two)));
        BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(one, two), three),
                                std::make_pair(std::make_pair(one, three), two)));
        BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(one, two), two),
                                std::make_pair(std::make_pair(one, two), three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(one, two), three),
                                 std::make_pair(std::make_pair(one, two), three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(two, two), two),
                                 std::make_pair(std::make_pair(one, three), three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(one, three), two),
                                 std::make_pair(std::make_pair(one, two), three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(one, two), three),
                                 std::make_pair(std::make_pair(one, two), two)));
    }
    {
        BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(&one, three), three),
                                std::make_pair(std::make_pair(&two, two), two)));
        BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(&one, two), three),
                                std::make_pair(std::make_pair(&one, three), two)));
        BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(one, &two), two),
                                std::make_pair(std::make_pair(one, &two), three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(one, &two), three),
                                 std::make_pair(std::make_pair(one, &two), three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(two, two), &two),
                                 std::make_pair(std::make_pair(one, three), &three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(&one, &three), &two),
                                 std::make_pair(std::make_pair(&one, &two), &three)));
        BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(one, two), three),
                                 std::make_pair(std::make_pair(one, two), two)));
    }
}

BOOST_AUTO_TEST_CASE(testFirstLess) {
    maths::COrderings::SFirstLess less;

    BOOST_TEST_REQUIRE(less(std::make_pair(1.0, 1.0), std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(less(1.0, std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(less(std::make_pair(1.0, 2.0), 2.0));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, 1.0), std::make_pair(1.0, 2.0)));
    BOOST_TEST_REQUIRE(!less(1.0, std::make_pair(1.0, 2.0)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, 1.0), 1.0));
    BOOST_TEST_REQUIRE(!less(std::make_pair(2.0, 2.0), std::make_pair(1.0, 3.0)));
    BOOST_TEST_REQUIRE(!less(2.0, std::make_pair(1.0, 1.0)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(2.0, 2.0), 1.0));

    BOOST_TEST_REQUIRE(less(std::make_pair(std::make_pair(1.0, 1.0), 1.0),
                            std::make_pair(std::make_pair(1.0, 2.0), 1.0)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(1.0, 1.0), 1.0),
                             std::make_pair(std::make_pair(1.0, 1.0), 1.0)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(std::make_pair(1.0, 2.0), 1.0),
                             std::make_pair(std::make_pair(1.0, 1.0), 1.0)));

    double one(1.0);
    double two(2.0);
    double three(3.0);

    BOOST_TEST_REQUIRE(less(std::make_pair(&one, &one), std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(less(&one, std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(less(std::make_pair(&one, &two), &two));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&one, &one), std::make_pair(&one, &two)));
    BOOST_TEST_REQUIRE(!less(&one, std::make_pair(&one, &two)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&one, &one), &one));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&two, &two), std::make_pair(&one, &three)));
    BOOST_TEST_REQUIRE(!less(&two, std::make_pair(&one, &one)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&two, &two), &one));
}

BOOST_AUTO_TEST_CASE(testFirstGreater) {
    maths::COrderings::SFirstGreater greater;

    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, 2.0), std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!greater(2.0, std::make_pair(3.0, 1.0)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, 3.0), 2.0));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, 2.0), std::make_pair(1.0, 1.0)));
    BOOST_TEST_REQUIRE(!greater(2.0, std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, 2.0), 1.0));
    BOOST_TEST_REQUIRE(greater(std::make_pair(2.0, 2.0), std::make_pair(1.0, 3.0)));
    BOOST_TEST_REQUIRE(greater(2.0, std::make_pair(1.0, 1.0)));
    BOOST_TEST_REQUIRE(greater(std::make_pair(2.0, 2.0), 1.0));

    BOOST_TEST_REQUIRE(!greater(std::make_pair(std::make_pair(1.0, 1.0), 1.0),
                                std::make_pair(std::make_pair(1.0, 2.0), 1.0)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(std::make_pair(1.0, 1.0), 1.0),
                                std::make_pair(std::make_pair(1.0, 1.0), 1.0)));
    BOOST_TEST_REQUIRE(greater(std::make_pair(std::make_pair(1.0, 2.0), 1.0),
                               std::make_pair(std::make_pair(1.0, 1.0), 1.0)));

    double one(1.0);
    double two(2.0);
    double three(3.0);

    BOOST_TEST_REQUIRE(!greater(std::make_pair(&one, &two), std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!greater(&two, std::make_pair(&three, &one)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&one, &three), &two));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&one, &two), std::make_pair(&one, &one)));
    BOOST_TEST_REQUIRE(!greater(&two, std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&one, &two), &one));
    BOOST_TEST_REQUIRE(greater(std::make_pair(&two, &two), std::make_pair(&one, &three)));
    BOOST_TEST_REQUIRE(greater(&two, std::make_pair(&one, &two)));
    BOOST_TEST_REQUIRE(greater(std::make_pair(&two, &one), &one));
}

BOOST_AUTO_TEST_CASE(testSecondLess) {
    maths::COrderings::SSecondLess less;

    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, 2.0), std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!less(2.0, std::make_pair(3.0, 1.0)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, 3.0), 2.0));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, 1.0), std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!less(1.0, std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, 2.0), 2.0));
    BOOST_TEST_REQUIRE(less(std::make_pair(2.0, 2.0), std::make_pair(1.0, 3.0)));
    BOOST_TEST_REQUIRE(less(2.0, std::make_pair(1.0, 3.0)));
    BOOST_TEST_REQUIRE(less(std::make_pair(2.0, 1.0), 2.0));

    BOOST_TEST_REQUIRE(less(std::make_pair(1.0, std::make_pair(1.0, 2.0)),
                            std::make_pair(2.0, std::make_pair(2.0, 1.0))));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, std::make_pair(1.0, 2.0)),
                             std::make_pair(2.0, std::make_pair(1.0, 2.0))));
    BOOST_TEST_REQUIRE(!less(std::make_pair(1.0, std::make_pair(2.0, 2.0)),
                             std::make_pair(2.0, std::make_pair(2.0, 1.0))));
    BOOST_TEST_REQUIRE(less(std::make_pair(1.0, 1.0),
                            std::make_pair(3.0, std::make_pair(1.0, 2.0))));
    BOOST_TEST_REQUIRE(less(std::make_pair(1.0, std::make_pair(3.0, 1.0)),
                            std::make_pair(3.0, 2.0)));

    double one(1.0);
    double two(2.0);
    double three(3.0);

    BOOST_TEST_REQUIRE(!less(std::make_pair(&one, &two), std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!less(&two, std::make_pair(&three, &one)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&one, &three), &two));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&one, &one), std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!less(&one, std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!less(std::make_pair(&one, &two), &two));
    BOOST_TEST_REQUIRE(less(std::make_pair(&two, &two), std::make_pair(&one, &three)));
    BOOST_TEST_REQUIRE(less(&two, std::make_pair(&one, &three)));
    BOOST_TEST_REQUIRE(less(std::make_pair(&two, &one), &two));
}

BOOST_AUTO_TEST_CASE(testSecondGreater) {
    maths::COrderings::SSecondGreater greater;

    BOOST_TEST_REQUIRE(greater(std::make_pair(1.0, 2.0), std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(greater(2.0, std::make_pair(3.0, 1.0)));
    BOOST_TEST_REQUIRE(greater(std::make_pair(1.0, 3.0), 2.0));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, 1.0), std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!greater(1.0, std::make_pair(2.0, 1.0)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, 2.0), 2.0));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(2.0, 2.0), std::make_pair(1.0, 3.0)));
    BOOST_TEST_REQUIRE(!greater(2.0, std::make_pair(1.0, 3.0)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(2.0, 1.0), 2.0));

    BOOST_TEST_REQUIRE(greater(std::make_pair(1.0, std::make_pair(2.0, 2.0)),
                               std::make_pair(2.0, std::make_pair(2.0, 1.0))));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, std::make_pair(2.0, 2.0)),
                                std::make_pair(2.0, std::make_pair(2.0, 2.0))));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(1.0, std::make_pair(2.0, 2.0)),
                                std::make_pair(2.0, std::make_pair(2.0, 3.0))));
    BOOST_TEST_REQUIRE(greater(std::make_pair(2.0, 2.0),
                               std::make_pair(3.0, std::make_pair(1.0, 2.0))));
    BOOST_TEST_REQUIRE(greater(std::make_pair(1.0, std::make_pair(3.0, 3.0)),
                               std::make_pair(3.0, 2.0)));

    double one(1.0);
    double two(2.0);
    double three(3.0);

    BOOST_TEST_REQUIRE(greater(std::make_pair(&one, &two), std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(greater(&two, std::make_pair(&three, &one)));
    BOOST_TEST_REQUIRE(greater(std::make_pair(&one, &three), &two));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&one, &one), std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!greater(&one, std::make_pair(&two, &one)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&one, &two), &two));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&two, &two), std::make_pair(&one, &three)));
    BOOST_TEST_REQUIRE(!greater(&two, std::make_pair(&one, &three)));
    BOOST_TEST_REQUIRE(!greater(std::make_pair(&two, &two), &two));
}

BOOST_AUTO_TEST_CASE(testDereference) {
    using TDoubleVec = std::vector<double>;
    using TDoubleVecCItr = std::vector<double>::const_iterator;
    using TDoubleVecCItrVec = std::vector<TDoubleVecCItr>;

    double values_[]{10.0, 1.0, 5.0, 3.0, 1.0};
    TDoubleVec values(std::begin(values_), std::end(values_));
    TDoubleVecCItrVec iterators;
    for (TDoubleVecCItr i = values.begin(); i != values.end(); ++i) {
        iterators.push_back(i);
    }

    std::sort(iterators.begin(), iterators.end(),
              core::CFunctional::SDereference<maths::COrderings::SLess>());
    std::sort(std::begin(values_), std::end(values_));
    for (std::size_t i = 0; i < values.size(); ++i) {
        LOG_DEBUG(<< "expected " << values_[i] << ", got " << *iterators[i]);
        BOOST_REQUIRE_EQUAL(values_[i], *iterators[i]);
    }
}

BOOST_AUTO_TEST_CASE(testLexicographicalCompare) {
    using TDoubleVec = std::vector<double>;
    using TDoubleDoublePr = std::pair<double, double>;

    maths::COrderings::SReferenceGreater greater;

    int i1 = 1;
    int i2 = 3;

    double d1 = 0.1;
    double d2 = 1.3;

    TDoubleDoublePr p1(1.0, 2.0);
    TDoubleDoublePr p2(1.2, 1.5);

    TDoubleVec v1, v2;
    double a1[] = {1.2, 1.3, 2.1};
    double a2[] = {1.2, 2.2, 2.0};
    v1.assign(std::begin(a1), std::end(a1));
    v2.assign(std::begin(a2), std::end(a2));

    std::string s1("a");
    std::string s2("b");

    BOOST_TEST_REQUIRE(i1 < i2);
    BOOST_TEST_REQUIRE(d1 < d2);
    BOOST_TEST_REQUIRE(p1 < p2);
    BOOST_TEST_REQUIRE(v1 < v2);
    BOOST_TEST_REQUIRE(s1 < s2);

    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, i2));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, i1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i2, i1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, i2, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, i1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i2, i1, greater));

    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, i2, p1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, i1, p2));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p1, i1, p1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i2, p1, i1, p1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p2, i1, p1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p1, i2, p1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p1, i1, p2, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p1, i1, p1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i2, p1, i1, p1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p2, i1, p1, greater));

    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, i2, p1, d1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, i1, p2, d1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, i1, p1, d2));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p1, d1, i1, p1, d1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i2, p1, d1, i1, p1, d1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p2, d1, i1, p1, d1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(i1, p1, d2, i1, p1, d1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, i2, p1, d1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, i1, p2, d1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, i1, p1, d2, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, i1, p1, d1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i2, p1, d1, i1, p1,
                                                                  d1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p2, d1, i1, p1,
                                                                  d1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d2, i1, p1,
                                                                  d1, greater));

    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, v1,
                                                                  i2, p1, d1, v1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, v1,
                                                                  i1, p2, d1, v1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, v1,
                                                                  i1, p1, d2, v1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(i1, p1, d1, v1,
                                                                  i1, p1, d1, v2));
    BOOST_TEST_REQUIRE(
        !maths::COrderings::lexicographical_compare(i1, p1, d1, v1, i1, p1, d1, v1));
    BOOST_TEST_REQUIRE(
        !maths::COrderings::lexicographical_compare(i2, p1, d1, v1, i1, p1, d1, v1));
    BOOST_TEST_REQUIRE(
        !maths::COrderings::lexicographical_compare(i1, p2, d1, v1, i1, p1, d1, v1));
    BOOST_TEST_REQUIRE(
        !maths::COrderings::lexicographical_compare(i1, p1, d2, v1, i1, p1, d1, v1));
    BOOST_TEST_REQUIRE(
        !maths::COrderings::lexicographical_compare(i1, p1, d1, v2, i1, p1, d1, v1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, i2, p1, d1, v1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, i1, p2, d1, v1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, i1, p1, d2, v1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, i1, p1, d1, v2, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, i1, p1, d1, v1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i2, p1, d1, v1, i1, p1, d1, v1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p2, d1, v1, i1, p1, d1, v1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d2, v1, i1, p1, d1, v1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v2, i1, p1, d1, v1, greater));

    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i2, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p2, d1, v1, s1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d2, v1, s1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d1, v2, s1));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d1, v1, s2));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i2, p1, d1, v1, s1, i1, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p2, d1, v1, s1, i1, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d2, v1, s1, i1, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v2, s1, i1, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s2, i1, p1, d1, v1, s1));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i2, p1, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p2, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d2, v1, s1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d1, v2, s1, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d1, v1, s2, greater));
    BOOST_TEST_REQUIRE(!maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s1, i1, p1, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i2, p1, d1, v1, s1, i1, p1, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p2, d1, v1, s1, i1, p1, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d2, v1, s1, i1, p1, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v2, s1, i1, p1, d1, v1, s1, greater));
    BOOST_TEST_REQUIRE(maths::COrderings::lexicographical_compare(
        i1, p1, d1, v1, s2, i1, p1, d1, v1, s1, greater));
}

BOOST_AUTO_TEST_CASE(testSimultaneousSort) {
    using TDoubleVec = std::vector<double>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TStrVec = std::vector<std::string>;
    using TDictionaryVec = std::vector<CDictionary>;
    using TDoubleRangeVec = core::CVectorRange<TDoubleVec>;
    using TDoubleTuple = boost::tuple<double, double, double, double>;
    using TDoubleDoubleTupleMap = std::map<double, TDoubleTuple>;

    {
        TDoubleVec keys{0.0, 1.0, 0.2, 1.1, 0.7, 5.0};
        TStrVec values{std::string(1, 'c'), std::string(1, 'q'),
                       std::string(1, '!'), std::string(1, 'a'),
                       std::string(1, 'z'), std::string(1, 'p')};

        std::string expectedKeys("[0, 0.2, 0.7, 1, 1.1, 5]");
        std::string expectedValues("[c, !, z, q, a, p]");

        maths::COrderings::simultaneousSort(keys, values);
        LOG_DEBUG(<< "keys = " << core::CContainerPrinter::print(keys));
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));
        BOOST_REQUIRE_EQUAL(expectedKeys, core::CContainerPrinter::print(keys));
        BOOST_REQUIRE_EQUAL(expectedValues, core::CContainerPrinter::print(values));

        maths::COrderings::simultaneousSort(keys, values);
        BOOST_REQUIRE_EQUAL(expectedKeys, core::CContainerPrinter::print(keys));
        BOOST_REQUIRE_EQUAL(expectedValues, core::CContainerPrinter::print(values));
    }
    {
        TDouble1Vec keys{7.0, 1.0, 0.2, 1.1, 0.7, 5.0};
        TStrVec values1{std::string(1, 'w'), std::string(1, 'q'),
                        std::string(1, '~'), std::string(1, 'e'),
                        std::string(1, ';'), std::string(1, 'y')};
        TDoubleDoublePrVec values2{{2.0, 1.0},  {2.1, 1.1},  {1.3, 1.9},
                                   {3.2, 12.9}, {1.2, 10.1}, {1.3, 6.2}};

        std::string expectedKeys("[0.2, 0.7, 1, 1.1, 5, 7]");
        std::string expectedValues1("[~, ;, q, e, y, w]");
        std::string expectedValues2("[(1.3, 1.9), (1.2, 10.1), (2.1, 1.1), (3.2, 12.9), (1.3, 6.2), (2, 1)]");

        maths::COrderings::simultaneousSort(keys, values1, values2);
        LOG_DEBUG(<< "keys = " << core::CContainerPrinter::print(keys));
        LOG_DEBUG(<< "values1 = " << core::CContainerPrinter::print(values1));
        LOG_DEBUG(<< "values2 = " << core::CContainerPrinter::print(values2));
        BOOST_REQUIRE_EQUAL(expectedKeys, core::CContainerPrinter::print(keys));
        BOOST_REQUIRE_EQUAL(expectedValues1, core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(expectedValues2, core::CContainerPrinter::print(values2));
    }
    test::CRandomNumbers rng;
    {
        TDoubleVec keys{7.1, 0.1, 0.9, 1.4, 0.7, 5.1, 80.0, 4.0};
        TStrVec values1{std::string("a1"),  std::string("23"),
                        std::string("~1"),  std::string("b4"),
                        std::string(";;"),  std::string("zz"),
                        std::string("sss"), std::string("pq")};
        TDoubleDoublePrVec values2{{1.0, 1.0},   {4.1, 1.1},  {5.3, 3.9},
                                   {7.2, 22.9},  {2.2, 1.1},  {0.3, 16.2},
                                   {21.2, 11.1}, {10.3, 13.2}};
        TStrVec rawWords;
        rng.generateWords(5, keys.size() * 5, rawWords);
        TDictionaryVec values3;
        for (std::size_t i = 0u; i < rawWords.size(); i += 5) {
            TStrVec words(rawWords.begin() + i, rawWords.begin() + i + 5);
            values3.push_back(CDictionary(words));
        }
        LOG_DEBUG(<< "values3 = " << core::CContainerPrinter::print(values3));
        std::string expectedKeys("[0.1, 0.7, 0.9, 1.4, 4, 5.1, 7.1, 80]");
        std::string expectedValues1("[23, ;;, ~1, b4, pq, zz, a1, sss]");
        std::string expectedValues2("[(4.1, 1.1), (2.2, 1.1), (5.3, 3.9), (7.2, 22.9), (10.3, 13.2), (0.3, 16.2), (1, 1), (21.2, 11.1)]");

        maths::COrderings::simultaneousSort(keys, values1, values2, values3);
        LOG_DEBUG(<< "keys = " << core::CContainerPrinter::print(keys));
        LOG_DEBUG(<< "values1 = " << core::CContainerPrinter::print(values1));
        LOG_DEBUG(<< "values2 = " << core::CContainerPrinter::print(values2));
        LOG_DEBUG(<< "# copies = " << values3.front().ms_Copies);
        BOOST_REQUIRE_EQUAL(expectedKeys, core::CContainerPrinter::print(keys));
        BOOST_REQUIRE_EQUAL(expectedValues1, core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(expectedValues2, core::CContainerPrinter::print(values2));
        for (const auto& value : values3) {
            BOOST_REQUIRE_EQUAL(std::size_t(0), value.ms_Copies);
        }

        maths::COrderings::simultaneousSort(keys, values1, values2, values3);
        LOG_DEBUG(<< "keys = " << core::CContainerPrinter::print(keys));
        LOG_DEBUG(<< "values1 = " << core::CContainerPrinter::print(values1));
        LOG_DEBUG(<< "values2 = " << core::CContainerPrinter::print(values2));
        LOG_DEBUG(<< "# copies = " << values3.front().ms_Copies);
        BOOST_REQUIRE_EQUAL(expectedKeys, core::CContainerPrinter::print(keys));
        BOOST_REQUIRE_EQUAL(expectedValues1, core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(expectedValues2, core::CContainerPrinter::print(values2));
        for (const auto& value : values3) {
            BOOST_REQUIRE_EQUAL(std::size_t(0), value.ms_Copies);
        }
    }
    {
        TDoubleVec values1{5.0, 4.0, 3.0, 2.0, 1.0};
        TDoubleVec values2{1.0, 3.0, 2.0, 5.0, 4.0};
        TDoubleVec values3{4.0, 2.0, 3.0, 3.0, 5.0};
        TDoubleVec values4{2.0, 1.0, 5.0, 4.0, 1.0};
        TDoubleRangeVec range1{values1, 1, 4};
        TDoubleRangeVec range2{values2, 1, 4};
        TDoubleRangeVec range3{values3, 1, 4};
        TDoubleRangeVec range4{values4, 1, 4};

        maths::COrderings::simultaneousSort(range1, range2);

        LOG_DEBUG(<< "values1 = " << core::CContainerPrinter::print(values1));
        LOG_DEBUG(<< "values2 = " << core::CContainerPrinter::print(values2));
        BOOST_REQUIRE_EQUAL(std::string("[5, 2, 3, 4, 1]"),
                            core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(std::string("[1, 5, 2, 3, 4]"),
                            core::CContainerPrinter::print(values2));

        maths::COrderings::simultaneousSort(range2, range1, range3);

        LOG_DEBUG(<< "values1 = " << core::CContainerPrinter::print(values1));
        LOG_DEBUG(<< "values2 = " << core::CContainerPrinter::print(values2));
        LOG_DEBUG(<< "values3 = " << core::CContainerPrinter::print(values3));
        BOOST_REQUIRE_EQUAL(std::string("[5, 3, 4, 2, 1]"),
                            core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(std::string("[1, 2, 3, 5, 4]"),
                            core::CContainerPrinter::print(values2));
        BOOST_REQUIRE_EQUAL(std::string("[4, 3, 3, 2, 5]"),
                            core::CContainerPrinter::print(values3));

        maths::COrderings::simultaneousSort(range4, range1, range2, range3);

        LOG_DEBUG(<< "values1 = " << core::CContainerPrinter::print(values1));
        LOG_DEBUG(<< "values2 = " << core::CContainerPrinter::print(values2));
        LOG_DEBUG(<< "values3 = " << core::CContainerPrinter::print(values3));
        LOG_DEBUG(<< "values4 = " << core::CContainerPrinter::print(values4));
        BOOST_REQUIRE_EQUAL(std::string("[5, 3, 2, 4, 1]"),
                            core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(std::string("[1, 2, 5, 3, 4]"),
                            core::CContainerPrinter::print(values2));
        BOOST_REQUIRE_EQUAL(std::string("[4, 3, 2, 3, 5]"),
                            core::CContainerPrinter::print(values3));
        BOOST_REQUIRE_EQUAL(std::string("[2, 1, 4, 5, 1]"),
                            core::CContainerPrinter::print(values4));
    }
    {
        for (std::size_t i = 0u; i < 50; ++i) {
            TDoubleVec raw;
            rng.generateUniformSamples(0.0, 10.0, 50, raw);

            TDoubleVec keys(raw.begin(), raw.begin() + 10);
            TDoubleVec values1(raw.begin() + 10, raw.begin() + 20);
            TDoubleVec values2(raw.begin() + 20, raw.begin() + 30);
            TDoubleVec values3(raw.begin() + 30, raw.begin() + 40);
            TDoubleVec values4(raw.begin() + 40, raw.begin() + 50);

            TDoubleDoubleTupleMap expected;
            for (std::size_t j = 0u; j < 10; ++j) {
                expected[keys[j]] =
                    TDoubleTuple(values1[j], values2[j], values3[j], values4[j]);
            }

            maths::COrderings::simultaneousSort(keys, values1, values2, values3, values4);
            LOG_DEBUG(<< "keys = " << core::CContainerPrinter::print(keys));

            auto itr = expected.begin();
            for (std::size_t j = 0u; j < keys.size(); ++j, ++itr) {
                BOOST_REQUIRE_EQUAL(itr->first, keys[j]);
                BOOST_REQUIRE_EQUAL(itr->second.get<0>(), values1[j]);
                BOOST_REQUIRE_EQUAL(itr->second.get<1>(), values2[j]);
                BOOST_REQUIRE_EQUAL(itr->second.get<2>(), values3[j]);
                BOOST_REQUIRE_EQUAL(itr->second.get<3>(), values4[j]);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
