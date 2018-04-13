/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CVectorRangeTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CVectorRange.h>

#include <cstddef>
#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleRng = core::CVectorRange<TDoubleVec>;
using TDoubleCRng = core::CVectorRange<const TDoubleVec>;

void CVectorRangeTest::testCreation() {
    LOG_DEBUG("*** CVectorRangeTest::testCreation ***");

    {
        TDoubleVec values1{1.0, 0.1, 0.7, 9.8};
        TDoubleVec values2{3.1, 1.4, 5.7, 1.2};

        TDoubleRng range13{values1, 1, 3};
        range13 = core::make_range(values1, 0, 3);
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 1, 0.1, 0.7, 9.8]"), core::CContainerPrinter::print(values1));

        range13 = core::make_range(values2, 1, 4);
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 1.4, 5.7, 1.2, 9.8]"), core::CContainerPrinter::print(values1));

        range13.assign(2, 2.0);
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 2, 2, 9.8]"), core::CContainerPrinter::print(values1));

        range13.assign(values2.begin(), values2.end());
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 3.1, 1.4, 5.7, 1.2, 9.8]"), core::CContainerPrinter::print(values1));
    }
    {
        TDoubleVec values1{1.0, 0.1, 0.7, 9.8};
        TDoubleVec values2{3.1, 1.4, 5.7, 1.2};

        TDoubleCRng range1{values1, 1, 3};
        range1 = TDoubleCRng(values2, 0, 3);
        CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.1, 0.7, 9.8]"), core::CContainerPrinter::print(values1));
        CPPUNIT_ASSERT_EQUAL(std::string("[3.1, 1.4, 5.7]"), core::CContainerPrinter::print(range1));
    }
}

void CVectorRangeTest::testAccessors() {
    LOG_DEBUG("*** CVectorRangeTest::testAccessors ***");

    TDoubleVec values{1.0, 0.1, 0.7, 9.8, 8.0};

    TDoubleRng range14{values, 1, 4};
    const TDoubleRng crange14{values, 1, 4};

    CPPUNIT_ASSERT_EQUAL(0.1, range14.at(0));
    CPPUNIT_ASSERT_EQUAL(0.7, range14.at(1));
    CPPUNIT_ASSERT_EQUAL(9.8, range14.at(2));
    CPPUNIT_ASSERT_THROW_MESSAGE(std::string("out of range: 3 >= 3"), range14.at(3), std::out_of_range);
    CPPUNIT_ASSERT_EQUAL(0.1, crange14.at(0));
    CPPUNIT_ASSERT_EQUAL(0.7, crange14.at(1));
    CPPUNIT_ASSERT_EQUAL(9.8, crange14.at(2));
    CPPUNIT_ASSERT_THROW_MESSAGE(std::string("out of range: 4 >= 3"), crange14.at(4), std::out_of_range);

    CPPUNIT_ASSERT_EQUAL(0.1, range14[0]);
    CPPUNIT_ASSERT_EQUAL(0.7, range14[1]);
    CPPUNIT_ASSERT_EQUAL(9.8, range14[2]);
    CPPUNIT_ASSERT_EQUAL(0.1, crange14[0]);
    CPPUNIT_ASSERT_EQUAL(0.7, crange14[1]);
    CPPUNIT_ASSERT_EQUAL(9.8, crange14[2]);

    CPPUNIT_ASSERT_EQUAL(0.1, range14.front());
    CPPUNIT_ASSERT_EQUAL(0.1, crange14.front());

    CPPUNIT_ASSERT_EQUAL(9.8, range14.back());
    CPPUNIT_ASSERT_EQUAL(9.8, crange14.back());
}

void CVectorRangeTest::testIterators() {
    LOG_DEBUG("*** CVectorRangeTest::testIterators ***");

    TDoubleVec values{1.0, 0.1, 0.7, 9.8, 8.0};

    TDoubleRng range14{values, 1, 4};
    const TDoubleRng crange14{values, 1, 4};

    for (auto i = range14.begin(), j = values.begin() + 1; i != range14.end(); ++i, ++j) {
        CPPUNIT_ASSERT_EQUAL(*j, *i);
    }
    CPPUNIT_ASSERT_EQUAL(std::ptrdiff_t(3), range14.end() - range14.begin());

    for (auto i = range14.cbegin(), j = values.cbegin() + 1; i != range14.cend(); ++i, ++j) {
        CPPUNIT_ASSERT_EQUAL(*j, *i);
    }
    CPPUNIT_ASSERT_EQUAL(std::ptrdiff_t(3), range14.end() - range14.begin());

    for (auto i = crange14.begin(), j = values.cbegin() + 1; i != crange14.end(); ++i, ++j) {
        CPPUNIT_ASSERT_EQUAL(*j, *i);
    }
    CPPUNIT_ASSERT_EQUAL(std::ptrdiff_t(3), crange14.end() - crange14.begin());
}

void CVectorRangeTest::testSizing() {
    LOG_DEBUG("*** CVectorRangeTest::testSizing ***");

    TDoubleVec values{1.0, 0.1, 0.7, 9.8, 8.0};

    TDoubleRng range11{values, 1, 1};
    CPPUNIT_ASSERT(range11.empty());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), range11.size());

    const TDoubleRng crange23{values, 1, 2};
    CPPUNIT_ASSERT(!crange23.empty());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), crange23.size());

    CPPUNIT_ASSERT_EQUAL(values.max_size(), range11.max_size());

    LOG_DEBUG("range capacity = " << range11.capacity());
    CPPUNIT_ASSERT_EQUAL(values.capacity() - values.size(), range11.capacity());

    range11.reserve(10);
    LOG_DEBUG("new range capacity = " << range11.capacity());
    CPPUNIT_ASSERT(range11.capacity() >= 10);
    LOG_DEBUG("new values capacity = " << values.capacity());
    CPPUNIT_ASSERT(values.capacity() >= 15);
}

void CVectorRangeTest::testModifiers() {
    LOG_DEBUG("*** CVectorRangeTest::testModifiers ***");

    TDoubleVec values1{1.0, 0.1, 0.7, 9.8, 8.0};
    TDoubleVec values2{2.0, 3.5, 8.1, 1.8};

    TDoubleRng range111{values1, 1, 1};
    range111.clear();
    CPPUNIT_ASSERT(range111.empty());
    range111.push_back(1.2);
    range111.push_back(2.2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), range111.size());
    range111.clear();
    CPPUNIT_ASSERT(range111.empty());
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.1, 0.7, 9.8, 8]"), core::CContainerPrinter::print(values1));

    TDoubleRng range125{values1, 2, 5};
    range125.insert(range125.begin(), values2.begin(), values2.end());
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.1, 2, 3.5, 8.1, 1.8, 0.7, 9.8, 8]"), core::CContainerPrinter::print(values1));
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 1.8, 0.7, 9.8, 8]"), core::CContainerPrinter::print(range125));
    CPPUNIT_ASSERT_EQUAL(std::size_t(7), range125.size());
    range125.erase(range125.begin(), range125.begin() + 4);
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.1, 0.7, 9.8, 8]"), core::CContainerPrinter::print(values1));
    CPPUNIT_ASSERT_EQUAL(std::string("[0.7, 9.8, 8]"), core::CContainerPrinter::print(range125));

    TDoubleRng range203{values2, 0, 3};
    range203.push_back(5.0);
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 5, 1.8]"), core::CContainerPrinter::print(values2));
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 5]"), core::CContainerPrinter::print(range203));
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), range203.size());
    range203.push_back(3.2);
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 5, 3.2, 1.8]"), core::CContainerPrinter::print(values2));
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 5, 3.2]"), core::CContainerPrinter::print(range203));
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), range203.size());
    range203.pop_back();
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 5, 1.8]"), core::CContainerPrinter::print(values2));
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 5]"), core::CContainerPrinter::print(range203));
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), range203.size());
    range203.pop_back();
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1, 1.8]"), core::CContainerPrinter::print(values2));
    CPPUNIT_ASSERT_EQUAL(std::string("[2, 3.5, 8.1]"), core::CContainerPrinter::print(range203));
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), range203.size());

    TDoubleRng range102{values1, 0, 2};
    range102.resize(3, 5.0);
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.1, 5, 0.7, 9.8, 8]"), core::CContainerPrinter::print(values1));
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.1, 5]"), core::CContainerPrinter::print(range102));
    range102.resize(1);
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 0.7, 9.8, 8]"), core::CContainerPrinter::print(values1));
    CPPUNIT_ASSERT_EQUAL(std::string("[1]"), core::CContainerPrinter::print(range102));

    TDoubleRng range113{values1, 1, 3};
    TDoubleRng range223{values2, 2, 3};
    std::string s1 = core::CContainerPrinter::print(values1);
    std::string s2 = core::CContainerPrinter::print(values2);
    std::string s113 = core::CContainerPrinter::print(range113);
    std::string s223 = core::CContainerPrinter::print(range223);

    range113.swap(range223);

    CPPUNIT_ASSERT_EQUAL(s1, core::CContainerPrinter::print(values1));
    CPPUNIT_ASSERT_EQUAL(s2, core::CContainerPrinter::print(values2));
    CPPUNIT_ASSERT_EQUAL(s113, core::CContainerPrinter::print(range223));
    CPPUNIT_ASSERT_EQUAL(s223, core::CContainerPrinter::print(range113));
}

void CVectorRangeTest::testComparisons() {
    LOG_DEBUG("*** CVectorRangeTest::testComparisons ***");

    TDoubleVec values1{1.0, 0.1, 0.7, 9.8, 8.0};
    TDoubleVec values2{1.2, 0.1, 0.7, 9.8, 18.0};

    TDoubleRng range103{values1, 0, 3};
    TDoubleRng range202{values2, 0, 2};
    LOG_DEBUG("range103 = " << core::CContainerPrinter::print(range103));
    LOG_DEBUG("range202 = " << core::CContainerPrinter::print(range202));

    CPPUNIT_ASSERT(range103 != range202);
    CPPUNIT_ASSERT(!(range103 == range202));

    TDoubleRng range114{values1, 1, 4};
    TDoubleRng range214{values2, 1, 4};
    LOG_DEBUG("range114 = " << core::CContainerPrinter::print(range114));
    LOG_DEBUG("range214 = " << core::CContainerPrinter::print(range214));

    CPPUNIT_ASSERT(range114 == range214);
    CPPUNIT_ASSERT(!(range114 < range214));
    CPPUNIT_ASSERT(!(range214 < range114));
    CPPUNIT_ASSERT(range114 <= range214);
    CPPUNIT_ASSERT(range114 >= range214);
    CPPUNIT_ASSERT(!(range114 != range214));

    CPPUNIT_ASSERT(range103 < range202);
    CPPUNIT_ASSERT(range202 > range103);
    CPPUNIT_ASSERT(range202 >= range103);
    CPPUNIT_ASSERT(range103 <= range202);
    CPPUNIT_ASSERT(!(range202 < range103));
    CPPUNIT_ASSERT(!(range103 >= range202));
}

CppUnit::Test* CVectorRangeTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CVectorRangeTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CVectorRangeTest>("CVectorRangeTest::testCreation", &CVectorRangeTest::testCreation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CVectorRangeTest>("CVectorRangeTest::testAccessors", &CVectorRangeTest::testAccessors));
    suiteOfTests->addTest(new CppUnit::TestCaller<CVectorRangeTest>("CVectorRangeTest::testIterators", &CVectorRangeTest::testIterators));
    suiteOfTests->addTest(new CppUnit::TestCaller<CVectorRangeTest>("CVectorRangeTest::testSizing", &CVectorRangeTest::testSizing));
    suiteOfTests->addTest(new CppUnit::TestCaller<CVectorRangeTest>("CVectorRangeTest::testModifiers", &CVectorRangeTest::testModifiers));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CVectorRangeTest>("CVectorRangeTest::testComparisons", &CVectorRangeTest::testComparisons));

    return suiteOfTests;
}
