/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CVectorRange.h>

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <vector>

BOOST_AUTO_TEST_SUITE(CVectorRangeTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleRng = core::CVectorRange<TDoubleVec>;
using TDoubleCRng = core::CVectorRange<const TDoubleVec>;

BOOST_AUTO_TEST_CASE(testCreation) {
    {
        TDoubleVec values1{1.0, 0.1, 0.7, 9.8};
        TDoubleVec values2{3.1, 1.4, 5.7, 1.2};

        TDoubleRng range13{values1, 1, 3};
        range13 = core::make_range(values1, 0, 3);
        BOOST_REQUIRE_EQUAL(std::string("[1, 1, 0.1, 0.7, 9.8]"),
                            core::CContainerPrinter::print(values1));

        range13 = core::make_range(values2, 1, 4);
        BOOST_REQUIRE_EQUAL(std::string("[1, 1.4, 5.7, 1.2, 9.8]"),
                            core::CContainerPrinter::print(values1));

        range13.assign(2, 2.0);
        BOOST_REQUIRE_EQUAL(std::string("[1, 2, 2, 9.8]"),
                            core::CContainerPrinter::print(values1));

        range13.assign(values2.begin(), values2.end());
        BOOST_REQUIRE_EQUAL(std::string("[1, 3.1, 1.4, 5.7, 1.2, 9.8]"),
                            core::CContainerPrinter::print(values1));
    }
    {
        TDoubleVec values1{1.0, 0.1, 0.7, 9.8};
        TDoubleVec values2{3.1, 1.4, 5.7, 1.2};

        TDoubleCRng range1{values1, 1, 3};
        range1 = TDoubleCRng(values2, 0, 3);
        BOOST_REQUIRE_EQUAL(std::string("[1, 0.1, 0.7, 9.8]"),
                            core::CContainerPrinter::print(values1));
        BOOST_REQUIRE_EQUAL(std::string("[3.1, 1.4, 5.7]"),
                            core::CContainerPrinter::print(range1));
    }
}

BOOST_AUTO_TEST_CASE(testAccessors) {
    TDoubleVec values{1.0, 0.1, 0.7, 9.8, 8.0};

    TDoubleRng range14{values, 1, 4};
    const TDoubleRng crange14{values, 1, 4};

    BOOST_REQUIRE_EQUAL(0.1, range14.at(0));
    BOOST_REQUIRE_EQUAL(0.7, range14.at(1));
    BOOST_REQUIRE_EQUAL(9.8, range14.at(2));
    BOOST_REQUIRE_THROW(range14.at(3), std::out_of_range);
    BOOST_REQUIRE_EQUAL(0.1, crange14.at(0));
    BOOST_REQUIRE_EQUAL(0.7, crange14.at(1));
    BOOST_REQUIRE_EQUAL(9.8, crange14.at(2));
    BOOST_REQUIRE_THROW(crange14.at(4), std::out_of_range);

    BOOST_REQUIRE_EQUAL(0.1, range14[0]);
    BOOST_REQUIRE_EQUAL(0.7, range14[1]);
    BOOST_REQUIRE_EQUAL(9.8, range14[2]);
    BOOST_REQUIRE_EQUAL(0.1, crange14[0]);
    BOOST_REQUIRE_EQUAL(0.7, crange14[1]);
    BOOST_REQUIRE_EQUAL(9.8, crange14[2]);

    BOOST_REQUIRE_EQUAL(0.1, range14.front());
    BOOST_REQUIRE_EQUAL(0.1, crange14.front());

    BOOST_REQUIRE_EQUAL(9.8, range14.back());
    BOOST_REQUIRE_EQUAL(9.8, crange14.back());
}

BOOST_AUTO_TEST_CASE(testIterators) {
    TDoubleVec values{1.0, 0.1, 0.7, 9.8, 8.0};

    TDoubleRng range14{values, 1, 4};
    const TDoubleRng crange14{values, 1, 4};

    for (auto i = range14.begin(), j = values.begin() + 1; i != range14.end(); ++i, ++j) {
        BOOST_REQUIRE_EQUAL(*j, *i);
    }
    BOOST_REQUIRE_EQUAL(std::ptrdiff_t(3), range14.end() - range14.begin());

    for (auto i = range14.cbegin(), j = values.cbegin() + 1;
         i != range14.cend(); ++i, ++j) {
        BOOST_REQUIRE_EQUAL(*j, *i);
    }
    BOOST_REQUIRE_EQUAL(std::ptrdiff_t(3), range14.end() - range14.begin());

    for (auto i = crange14.begin(), j = values.cbegin() + 1;
         i != crange14.end(); ++i, ++j) {
        BOOST_REQUIRE_EQUAL(*j, *i);
    }
    BOOST_REQUIRE_EQUAL(std::ptrdiff_t(3), crange14.end() - crange14.begin());
}

BOOST_AUTO_TEST_CASE(testSizing) {
    TDoubleVec values{1.0, 0.1, 0.7, 9.8, 8.0};

    TDoubleRng range11{values, 1, 1};
    BOOST_TEST_REQUIRE(range11.empty());
    BOOST_REQUIRE_EQUAL(std::size_t(0), range11.size());

    const TDoubleRng crange23{values, 1, 2};
    BOOST_TEST_REQUIRE(!crange23.empty());
    BOOST_REQUIRE_EQUAL(std::size_t(1), crange23.size());

    BOOST_REQUIRE_EQUAL(values.max_size(), range11.max_size());

    LOG_DEBUG(<< "range capacity = " << range11.capacity());
    BOOST_REQUIRE_EQUAL(values.capacity() - values.size(), range11.capacity());

    range11.reserve(10);
    LOG_DEBUG(<< "new range capacity = " << range11.capacity());
    BOOST_TEST_REQUIRE(range11.capacity() >= 10);
    LOG_DEBUG(<< "new values capacity = " << values.capacity());
    BOOST_TEST_REQUIRE(values.capacity() >= 15);
}

BOOST_AUTO_TEST_CASE(testModifiers) {
    TDoubleVec values1{1.0, 0.1, 0.7, 9.8, 8.0};
    TDoubleVec values2{2.0, 3.5, 8.1, 1.8};

    TDoubleRng range111{values1, 1, 1};
    range111.clear();
    BOOST_TEST_REQUIRE(range111.empty());
    range111.push_back(1.2);
    range111.push_back(2.2);
    BOOST_REQUIRE_EQUAL(std::size_t(2), range111.size());
    range111.clear();
    BOOST_TEST_REQUIRE(range111.empty());
    BOOST_REQUIRE_EQUAL(std::string("[1, 0.1, 0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(values1));

    TDoubleRng range125{values1, 2, 5};
    range125.insert(range125.begin(), values2.begin(), values2.end());
    BOOST_REQUIRE_EQUAL(std::string("[1, 0.1, 2, 3.5, 8.1, 1.8, 0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(values1));
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 1.8, 0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(range125));
    BOOST_REQUIRE_EQUAL(std::size_t(7), range125.size());
    range125.erase(range125.begin(), range125.begin() + 4);
    BOOST_REQUIRE_EQUAL(std::string("[1, 0.1, 0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(values1));
    BOOST_REQUIRE_EQUAL(std::string("[0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(range125));

    TDoubleRng range203{values2, 0, 3};
    range203.push_back(5.0);
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 5, 1.8]"),
                        core::CContainerPrinter::print(values2));
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 5]"),
                        core::CContainerPrinter::print(range203));
    BOOST_REQUIRE_EQUAL(std::size_t(4), range203.size());
    range203.push_back(3.2);
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 5, 3.2, 1.8]"),
                        core::CContainerPrinter::print(values2));
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 5, 3.2]"),
                        core::CContainerPrinter::print(range203));
    BOOST_REQUIRE_EQUAL(std::size_t(5), range203.size());
    range203.pop_back();
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 5, 1.8]"),
                        core::CContainerPrinter::print(values2));
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 5]"),
                        core::CContainerPrinter::print(range203));
    BOOST_REQUIRE_EQUAL(std::size_t(4), range203.size());
    range203.pop_back();
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1, 1.8]"),
                        core::CContainerPrinter::print(values2));
    BOOST_REQUIRE_EQUAL(std::string("[2, 3.5, 8.1]"),
                        core::CContainerPrinter::print(range203));
    BOOST_REQUIRE_EQUAL(std::size_t(3), range203.size());

    TDoubleRng range102{values1, 0, 2};
    range102.resize(3, 5.0);
    BOOST_REQUIRE_EQUAL(std::string("[1, 0.1, 5, 0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(values1));
    BOOST_REQUIRE_EQUAL(std::string("[1, 0.1, 5]"), core::CContainerPrinter::print(range102));
    range102.resize(1);
    BOOST_REQUIRE_EQUAL(std::string("[1, 0.7, 9.8, 8]"),
                        core::CContainerPrinter::print(values1));
    BOOST_REQUIRE_EQUAL(std::string("[1]"), core::CContainerPrinter::print(range102));

    TDoubleRng range113{values1, 1, 3};
    TDoubleRng range223{values2, 2, 3};
    std::string s1 = core::CContainerPrinter::print(values1);
    std::string s2 = core::CContainerPrinter::print(values2);
    std::string s113 = core::CContainerPrinter::print(range113);
    std::string s223 = core::CContainerPrinter::print(range223);

    range113.swap(range223);

    BOOST_REQUIRE_EQUAL(s1, core::CContainerPrinter::print(values1));
    BOOST_REQUIRE_EQUAL(s2, core::CContainerPrinter::print(values2));
    BOOST_REQUIRE_EQUAL(s113, core::CContainerPrinter::print(range223));
    BOOST_REQUIRE_EQUAL(s223, core::CContainerPrinter::print(range113));
}

BOOST_AUTO_TEST_CASE(testComparisons) {
    TDoubleVec values1{1.0, 0.1, 0.7, 9.8, 8.0};
    TDoubleVec values2{1.2, 0.1, 0.7, 9.8, 18.0};

    TDoubleRng range103{values1, 0, 3};
    TDoubleRng range202{values2, 0, 2};
    LOG_DEBUG(<< "range103 = " << core::CContainerPrinter::print(range103));
    LOG_DEBUG(<< "range202 = " << core::CContainerPrinter::print(range202));

    BOOST_TEST_REQUIRE(range103 != range202);
    BOOST_TEST_REQUIRE(!(range103 == range202));

    TDoubleRng range114{values1, 1, 4};
    TDoubleRng range214{values2, 1, 4};
    LOG_DEBUG(<< "range114 = " << core::CContainerPrinter::print(range114));
    LOG_DEBUG(<< "range214 = " << core::CContainerPrinter::print(range214));

    BOOST_TEST_REQUIRE(range114 == range214);
    BOOST_TEST_REQUIRE(!(range114 < range214));
    BOOST_TEST_REQUIRE(!(range214 < range114));
    BOOST_TEST_REQUIRE(range114 <= range214);
    BOOST_TEST_REQUIRE(range114 >= range214);
    BOOST_TEST_REQUIRE(!(range114 != range214));

    BOOST_TEST_REQUIRE(range103 < range202);
    BOOST_TEST_REQUIRE(range202 > range103);
    BOOST_TEST_REQUIRE(range202 >= range103);
    BOOST_TEST_REQUIRE(range103 <= range202);
    BOOST_TEST_REQUIRE(!(range202 < range103));
    BOOST_TEST_REQUIRE(!(range103 >= range202));
}

BOOST_AUTO_TEST_SUITE_END()
