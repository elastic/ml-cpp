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

#include "CPolymorphicStackObjectCPtrTest.h"

#include <core/CPolymorphicStackObjectCPtr.h>

#include <string>

using namespace ml;

namespace {

class CBase {
    public:
        virtual ~CBase(void) {}
        virtual std::string iam(void) const = 0;
};

class CDerived1 : public CBase {
    public:
        virtual std::string iam(void) const { return "d1"; }
};

class CDerived2 : public CBase {
    public:
        virtual std::string iam(void) const { return "d2"; }
};

class CDerived3 : public CBase {
    public:
        virtual std::string iam(void) const { return "d3"; }
};

class CDerived4 : public CBase {
    public:
        virtual std::string iam(void) const { return "d4"; }
};

}

void CPolymorphicStackObjectCPtrTest::testAll(void) {
    typedef core::CPolymorphicStackObjectCPtr<CBase, CDerived1, CDerived2> TStackPtr12;
    typedef core::CPolymorphicStackObjectCPtr<CBase, CDerived1, CDerived2, CDerived3, CDerived4> TStackPtr1234;

    TStackPtr12 test1((CDerived1()));
    CPPUNIT_ASSERT_EQUAL(std::string("d1"), test1->iam());
    CPPUNIT_ASSERT_EQUAL(std::string("d1"), (*test1).iam());

    TStackPtr12 test2((CDerived2()));
    CPPUNIT_ASSERT_EQUAL(std::string("d2"), test2->iam());
    CPPUNIT_ASSERT_EQUAL(std::string("d2"), (*test2).iam());

    test1 = test2;
    CPPUNIT_ASSERT_EQUAL(std::string("d2"), test1->iam());

    TStackPtr1234 test3((CDerived3()));
    CPPUNIT_ASSERT_EQUAL(std::string("d3"), test3->iam());

    TStackPtr1234 test4((CDerived4()));
    CPPUNIT_ASSERT_EQUAL(std::string("d4"), test4->iam());

    test3 = test4;
    CPPUNIT_ASSERT_EQUAL(std::string("d4"), test3->iam());

    test4 = test2;
    CPPUNIT_ASSERT_EQUAL(std::string("d2"), test4->iam());

    CPPUNIT_ASSERT(test1 && test2 && test3 && test4);

    TStackPtr1234 null;
    CPPUNIT_ASSERT(!null);
}

CppUnit::Test *CPolymorphicStackObjectCPtrTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CPolymorphicStackObjectCPtrTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CPolymorphicStackObjectCPtrTest>(
                               "CPolymorphicStackObjectCPtrTest::testAll",
                               &CPolymorphicStackObjectCPtrTest::testAll) );

    return suiteOfTests;
}
