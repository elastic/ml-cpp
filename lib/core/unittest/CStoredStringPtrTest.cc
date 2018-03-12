/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
#include "CStoredStringPtrTest.h"

#include <core/CMemory.h>
#include <core/CStoredStringPtr.h>

#include <boost/unordered_set.hpp>

#include <string>
#include <utility>


CppUnit::Test *CStoredStringPtrTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStoredStringPtrTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CStoredStringPtrTest>(
                               "CStoredStringPtrTest::testPointerSemantics",
                               &CStoredStringPtrTest::testPointerSemantics) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStoredStringPtrTest>(
                               "CStoredStringPtrTest::testMemoryUsage",
                               &CStoredStringPtrTest::testMemoryUsage) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStoredStringPtrTest>(
                               "CStoredStringPtrTest::testHash",
                               &CStoredStringPtrTest::testHash) );

    return suiteOfTests;
}

void CStoredStringPtrTest::testPointerSemantics() {
    {
        ml::core::CStoredStringPtr null;

        if (null) {
            CPPUNIT_FAIL("Should not return true in boolean context");
        }

        if (!null) {
            CPPUNIT_ASSERT(null != ml::core::CStoredStringPtr::makeStoredString("not null"));
            CPPUNIT_ASSERT(null == nullptr);
            CPPUNIT_ASSERT(null.get() == nullptr);
        } else {
            CPPUNIT_FAIL("Should not return false in negated boolean context");
        }
    }
    {
        std::string str1("my first string");

        ml::core::CStoredStringPtr ptr1 = ml::core::CStoredStringPtr::makeStoredString(str1);

        if (ptr1) {
            CPPUNIT_ASSERT(ptr1 == ptr1);
            CPPUNIT_ASSERT(ptr1 != nullptr);
            CPPUNIT_ASSERT(ptr1.get() != nullptr);
        } else {
            CPPUNIT_FAIL("Should not return false in boolean context");
        }

        if (!ptr1) {
            CPPUNIT_FAIL("Should not return true in negated boolean context");
        } else {
            CPPUNIT_ASSERT_EQUAL(0, ptr1->compare(str1));
        }
    }
    {
        // This is long because the most efficient way to move a small string
        // would be to leave the original value in the moved-from string
        std::string str2("my second string - long enough to not use the small string optimisation");

        ml::core::CStoredStringPtr ptr2 = ml::core::CStoredStringPtr::makeStoredString(std::move(str2));

        if (ptr2) {
            CPPUNIT_ASSERT(ptr2 == ptr2);
            CPPUNIT_ASSERT(ptr2 != nullptr);
            CPPUNIT_ASSERT(ptr2.get() != nullptr);
        } else {
            CPPUNIT_FAIL("Should not return false in boolean context");
        }

        if (!ptr2) {
            CPPUNIT_FAIL("Should not return true in negated boolean context");
        } else {
            // str2 should no longer contain its original value, as it should
            // have been moved to the stored string
            CPPUNIT_ASSERT(ptr2->compare(str2) != 0);
        }
    }
}

void CStoredStringPtrTest::testMemoryUsage() {
    {
        ml::core::CStoredStringPtr null;

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), ml::core::CMemory::dynamicSize(null));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), null.actualMemoryUsage());
    }
    {
        std::string str1("short");

        ml::core::CStoredStringPtr ptr1 = ml::core::CStoredStringPtr::makeStoredString(str1);

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), ml::core::CMemory::dynamicSize(ptr1));
        CPPUNIT_ASSERT_EQUAL(ml::core::CMemory::dynamicSize(&str1), ptr1.actualMemoryUsage());
    }
    {
        std::string str2("much longer - YUGE in fact!");

        ml::core::CStoredStringPtr ptr2 = ml::core::CStoredStringPtr::makeStoredString(str2);

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), ml::core::CMemory::dynamicSize(ptr2));
        CPPUNIT_ASSERT_EQUAL(ml::core::CMemory::dynamicSize(&str2), ptr2.actualMemoryUsage());
    }
}

void CStoredStringPtrTest::testHash() {
    using TStoredStringPtrUSet = boost::unordered_set<ml::core::CStoredStringPtr>;

    ml::core::CStoredStringPtr key = ml::core::CStoredStringPtr::makeStoredString("key");

    TStoredStringPtrUSet s;
    CPPUNIT_ASSERT(s.insert(key).second);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), s.count(key));
}

