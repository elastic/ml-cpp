/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CPolymorphicStackObjectCPtrTest.h"

#include <core/CPolymorphicStackObjectCPtr.h>

#include <string>

using namespace ml;

namespace
{

class CBase
{
    public:
        virtual ~CBase() {}
        virtual std::string iam() const = 0;
};

class CDerived1 : public CBase
{
    public:
        virtual std::string iam() const { return "d1"; }
};

class CDerived2 : public CBase
{
    public:
        virtual std::string iam() const { return "d2"; }
};

class CDerived3 : public CBase
{
    public:
        virtual std::string iam() const { return "d3"; }
};

class CDerived4 : public CBase
{
    public:
        virtual std::string iam() const { return "d4"; }
};

}

void CPolymorphicStackObjectCPtrTest::testAll()
{
    using TStackPtr12 = core::CPolymorphicStackObjectCPtr<CBase, CDerived1, CDerived2>;
    using TStackPtr1234 = core::CPolymorphicStackObjectCPtr<CBase, CDerived1, CDerived2, CDerived3, CDerived4>;

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

CppUnit::Test *CPolymorphicStackObjectCPtrTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CPolymorphicStackObjectCPtrTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CPolymorphicStackObjectCPtrTest>(
                                   "CPolymorphicStackObjectCPtrTest::testAll",
                                   &CPolymorphicStackObjectCPtrTest::testAll) );

    return suiteOfTests;
}
