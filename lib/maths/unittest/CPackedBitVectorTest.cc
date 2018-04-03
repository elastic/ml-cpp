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

#include "CPackedBitVectorTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CPackedBitVector.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

#include <bitset>
#include <vector>

using namespace ml;

typedef std::vector<bool>                    TBoolVec;
typedef std::vector<std::size_t>             TSizeVec;
typedef std::vector<maths::CPackedBitVector> TPackedBitVectorVec;

namespace {

std::string print(const maths::CPackedBitVector &v) {
    std::ostringstream result;
    result << v;
    return result.str();
}

std::string toBitString(const TBoolVec &v) {
    std::ostringstream result;
    for (std::size_t i = 0u; i < v.size(); ++i) {
        result << static_cast<int>(v[i]);
    }
    return result.str();
}

}

void CPackedBitVectorTest::testCreation(void) {
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testCreation  |");
    LOG_DEBUG("+--------------------------------------+");

    maths::CPackedBitVector test1(3, true);
    LOG_DEBUG("test1 = " << test1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), test1.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(TBoolVec(3, true)),
                         core::CContainerPrinter::print(test1.toBitVector()));

    maths::CPackedBitVector test2(5, false);
    LOG_DEBUG("test2 = " << test2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), test2.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(TBoolVec(5, false)),
                         core::CContainerPrinter::print(test2.toBitVector()));

    maths::CPackedBitVector test3(255, true);
    LOG_DEBUG("test3 = " << test3);
    CPPUNIT_ASSERT_EQUAL(std::size_t(255), test3.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(TBoolVec(255, true)),
                         core::CContainerPrinter::print(test3.toBitVector()));

    maths::CPackedBitVector test4(279, true);
    LOG_DEBUG("test4 = " << test4);
    CPPUNIT_ASSERT_EQUAL(std::size_t(279), test4.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(TBoolVec(279, true)),
                         core::CContainerPrinter::print(test4.toBitVector()));

    maths::CPackedBitVector test5(512, false);
    LOG_DEBUG("test5 = " << test5);
    CPPUNIT_ASSERT_EQUAL(std::size_t(512), test5.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(TBoolVec(512, false)),
                         core::CContainerPrinter::print(test5.toBitVector()));

    maths::CPackedBitVector test6((TBoolVec()));
    LOG_DEBUG("test6 = " << test6);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), test6.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print((TBoolVec())),
                         core::CContainerPrinter::print(test6.toBitVector()));

    bool bits1_[] =
    {
        true,
        true
    };
    TBoolVec bits1(boost::begin(bits1_), boost::end(bits1_));
    maths::CPackedBitVector test7(bits1);
    LOG_DEBUG("test7 = " << test7);
    CPPUNIT_ASSERT_EQUAL(bits1.size(), test7.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits1),
                         core::CContainerPrinter::print(test7.toBitVector()));

    bool bits2_[] =
    {
        true, false, false, true, true, false, false, false, false, true, true, true, true, false
    };
    TBoolVec bits2(boost::begin(bits2_), boost::end(bits2_));
    maths::CPackedBitVector test8(bits2);
    LOG_DEBUG("test8 = " << test8);
    CPPUNIT_ASSERT_EQUAL(bits2.size(), test8.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits2),
                         core::CContainerPrinter::print(test8.toBitVector()));

    test::CRandomNumbers rng;

    TSizeVec components;
    for (std::size_t t = 0u; t < 100; ++t) {
        rng.generateUniformSamples(0, 2, 30, components);
        TBoolVec bits3(components.begin(), components.end());
        maths::CPackedBitVector test9(bits3);
        if ((t + 1) % 10 == 0) {
            LOG_DEBUG("test9 = " << test9);
        }
        CPPUNIT_ASSERT_EQUAL(bits3.size(), test9.dimension());
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits3),
                             core::CContainerPrinter::print(test9.toBitVector()));
    }
}

void CPackedBitVectorTest::testExtend(void) {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testExtend  |");
    LOG_DEBUG("+------------------------------------+");

    maths::CPackedBitVector test1;
    test1.extend(true);
    LOG_DEBUG("test1 = " << test1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), test1.dimension());
    CPPUNIT_ASSERT_EQUAL(std::string("[1]"), print(test1));

    test1.extend(true);
    LOG_DEBUG("test1 = " << test1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), test1.dimension());
    CPPUNIT_ASSERT_EQUAL(std::string("[1 1]"), print(test1));

    test1.extend(false);
    LOG_DEBUG("test1 = " << test1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), test1.dimension());
    CPPUNIT_ASSERT_EQUAL(std::string("[1 1 0]"), print(test1));

    test1.extend(false);
    LOG_DEBUG("test1 = " << test1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), test1.dimension());
    CPPUNIT_ASSERT_EQUAL(std::string("[1 1 0 0]"), print(test1));

    test1.extend(true);
    LOG_DEBUG("test1 = " << test1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), test1.dimension());
    CPPUNIT_ASSERT_EQUAL(std::string("[1 1 0 0 1]"), print(test1));

    maths::CPackedBitVector test2(254, true);
    test2.extend(true);
    LOG_DEBUG("test2 = " << test2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(255), test2.dimension());
    TBoolVec bits1(255, true);
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits1),
                         core::CContainerPrinter::print(test2.toBitVector()));
    test2.extend(false);
    bits1.push_back(false);
    LOG_DEBUG("test2 = " << test2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(256), test2.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits1),
                         core::CContainerPrinter::print(test2.toBitVector()));

    maths::CPackedBitVector test3(255, true);
    test3.extend(false);
    LOG_DEBUG("test3 = " << test2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(256), test3.dimension());
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits1),
                         core::CContainerPrinter::print(test3.toBitVector()));

    test::CRandomNumbers rng;

    TSizeVec components;
    rng.generateUniformSamples(0, 2, 1012, components);

    TBoolVec bits2;
    maths::CPackedBitVector test4;

    for (std::size_t i = 0u; i < components.size(); ++i) {
        bits2.push_back(components[i] > 0);
        test4.extend(components[i] > 0);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits2),
                             core::CContainerPrinter::print(test4.toBitVector()));
    }
}

void CPackedBitVectorTest::testContract(void) {
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testContract  |");
    LOG_DEBUG("+--------------------------------------+");

    maths::CPackedBitVector test1;
    test1.extend(true);
    test1.extend(true);
    test1.extend(false);
    test1.extend(true);
    std::string expected[] =
    {
        "[1 1 0 1]",
        "[1 0 1]",
        "[0 1]",
        "[1]"
    };
    for (const std::string *e = expected; test1.dimension() > 0; ++e) {
        LOG_DEBUG("test1 = " << test1);
        CPPUNIT_ASSERT_EQUAL(*e, print(test1));
        test1.contract();
    }

    TBoolVec bits1(256, true);
    bits1.push_back(false);
    bits1.push_back(true);
    bits1.push_back(false);
    maths::CPackedBitVector test2(bits1);
    for (std::size_t i = 0u; i < 10; ++i) {
        bits1.erase(bits1.begin());
        test2.contract();
        LOG_DEBUG("test2 = " << test2);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits1),
                             core::CContainerPrinter::print(test2.toBitVector()));
    }

    TBoolVec bits2(1024, true);
    bits2.push_back(false);
    maths::CPackedBitVector test3(1024, true);
    test3.extend(false);
    for (std::size_t i = 0u; i < 10; ++i) {
        bits2.erase(bits2.begin());
        test3.contract();
        LOG_DEBUG("test3 = " << test3);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(bits2),
                             core::CContainerPrinter::print(test3.toBitVector()));
    }
}

void CPackedBitVectorTest::testOperators(void) {
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testOperators  |");
    LOG_DEBUG("+---------------------------------------+");

    test::CRandomNumbers rng;

    TPackedBitVectorVec test;

    TSizeVec components;
    for (std::size_t t = 0u; t < 20; ++t) {
        rng.generateUniformSamples(0, 2, 20, components);
        TBoolVec bits(components.begin(), components.end());
        test.push_back(maths::CPackedBitVector(bits));
    }

    for (std::size_t i = 0u; i < test.size(); ++i) {
        for (std::size_t j = i; j < test.size(); ++j) {
            CPPUNIT_ASSERT(test[i] < test[j] || test[i] > test[j] || test[i] == test[j]);
        }
    }
}

void CPackedBitVectorTest::testInner(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testInner  |");
    LOG_DEBUG("+-----------------------------------+");

    typedef maths::CVector<double> TVector;
    typedef std::vector<TVector>   TVectorVec;

    maths::CPackedBitVector test1(10, true);
    maths::CPackedBitVector test2(10, false);
    bool bits1[] =
    {
        true, true, false, false, true, false, false, false, true, true
    };
    maths::CPackedBitVector test3(TBoolVec(boost::begin(bits1),
                                           boost::end(bits1)));
    bool bits2[] =
    {
        false, false, true, false, true, false, false, false, false, false
    };
    maths::CPackedBitVector test4(TBoolVec(boost::begin(bits2),
                                           boost::end(bits2)));

    CPPUNIT_ASSERT_EQUAL(10.0, test1.inner(test1));
    CPPUNIT_ASSERT_EQUAL(0.0,  test1.inner(test2));
    CPPUNIT_ASSERT_EQUAL(5.0,  test1.inner(test3));
    CPPUNIT_ASSERT_EQUAL(2.0,  test1.inner(test4));
    CPPUNIT_ASSERT_EQUAL(0.0,  test2.inner(test2));
    CPPUNIT_ASSERT_EQUAL(0.0,  test2.inner(test3));
    CPPUNIT_ASSERT_EQUAL(0.0,  test2.inner(test4));
    CPPUNIT_ASSERT_EQUAL(5.0,  test3.inner(test3));
    CPPUNIT_ASSERT_EQUAL(1.0,  test3.inner(test4));
    CPPUNIT_ASSERT_EQUAL(2.0,  test4.inner(test4));

    maths::CPackedBitVector test5(570, true);
    test5.extend(false);
    test5.extend(false);
    test5.extend(true);
    maths::CPackedBitVector test6(565, true);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(true);

    CPPUNIT_ASSERT_EQUAL(566.0, test5.inner(test6));

    test::CRandomNumbers rng;

    TPackedBitVectorVec test7;
    TVectorVec comparison;

    TSizeVec components;
    for (std::size_t t = 0u; t < 50; ++t) {
        rng.generateUniformSamples(0, 2, 50, components);
        TBoolVec bits3(components.begin(), components.end());
        test7.push_back(maths::CPackedBitVector(bits3));
        comparison.push_back(TVector(bits3.begin(), bits3.end()));
    }

    for (std::size_t i = 0u; i < test7.size(); ++i) {
        LOG_DEBUG("Testing " << test7[i]);
        for (std::size_t j = 0u; j < test7.size(); ++j) {
            CPPUNIT_ASSERT_EQUAL(comparison[i].inner(comparison[j]),
                                 test7[i].inner(test7[j]));
        }
    }
}

void CPackedBitVectorTest::testBitwiseOr(void) {
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testBitwiseOr  |");
    LOG_DEBUG("+---------------------------------------+");

    typedef std::vector<std::bitset<50> > TBitSetVec;

    test::CRandomNumbers rng;

    TPackedBitVectorVec test;
    TBitSetVec comparison;

    TSizeVec components;
    for (std::size_t t = 0u; t < 50; ++t) {
        rng.generateUniformSamples(0, 2, 50, components);
        TBoolVec bits(components.begin(), components.end());
        test.push_back(maths::CPackedBitVector(bits));
        comparison.push_back(std::bitset<50>(toBitString(bits)));
    }

    for (std::size_t i = 0u; i < test.size(); ++i) {
        LOG_DEBUG("Testing " << test[i]);
        for (std::size_t j = 0u; j < test.size(); ++j) {
            {
                double expected = 0.0;
                std::bitset<50> bitwiseOr = comparison[i] | comparison[j];
                for (std::size_t k = 0u; k < bitwiseOr.size(); ++k) {
                    expected += bitwiseOr[k] ? 1.0 : 0.0;
                }
                if (j % 10 == 0) {
                    LOG_DEBUG("or  = " << expected);
                }
                CPPUNIT_ASSERT_EQUAL(expected, test[i].inner(test[j],
                                                             maths::CPackedBitVector::E_OR));
            }
            {
                double expected = 0.0;
                std::bitset<50> bitwiseXor = comparison[i] ^ comparison[j];
                for (std::size_t k = 0u; k < bitwiseXor.size(); ++k) {
                    expected += bitwiseXor[k] ? 1.0 : 0.0;
                }
                if (j % 10 == 0) {
                    LOG_DEBUG("xor = " << expected);
                }
                CPPUNIT_ASSERT_EQUAL(expected, test[i].inner(test[j],
                                                             maths::CPackedBitVector::E_XOR));
            }
        }
    }
}

void CPackedBitVectorTest::testPersist(void) {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CPackedBitVectorTest::testPersist  |");
    LOG_DEBUG("+-------------------------------------+");

    bool bits[] =
    {
        true, true, false, false, true, false, false, false, true, true
    };

    for (std::size_t t = 0u; t < boost::size(bits); ++t) {
        maths::CPackedBitVector origVector(TBoolVec(boost::begin(bits),
                                                    boost::begin(bits) + t));

        std::string origXml = origVector.toDelimited();
        LOG_DEBUG("xml = " << origXml);

        maths::CPackedBitVector restoredVector;
        restoredVector.fromDelimited(origXml);

        CPPUNIT_ASSERT_EQUAL(origVector.checksum(), restoredVector.checksum());
    }
}

CppUnit::Test *CPackedBitVectorTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CPackedBitVectorTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testCreation",
                               &CPackedBitVectorTest::testCreation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testExtend",
                               &CPackedBitVectorTest::testExtend) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testContract",
                               &CPackedBitVectorTest::testContract) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testOperators",
                               &CPackedBitVectorTest::testOperators) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testInner",
                               &CPackedBitVectorTest::testInner) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testBitwiseOr",
                               &CPackedBitVectorTest::testBitwiseOr) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CPackedBitVectorTest>(
                               "CPackedBitVectorTest::testPersist",
                               &CPackedBitVectorTest::testPersist) );

    return suiteOfTests;
}
