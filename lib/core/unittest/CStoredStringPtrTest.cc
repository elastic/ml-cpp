/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMemory.h>
#include <core/CStoredStringPtr.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_set.hpp>

#include <string>
#include <utility>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::core::CStoredStringPtr)

BOOST_AUTO_TEST_SUITE(CStoredStringPtrTest)

BOOST_AUTO_TEST_CASE(testPointerSemantics) {
    {
        ml::core::CStoredStringPtr null;

        if (null) {
            BOOST_FAIL("Should not return true in boolean context");
        }

        if (!null) {
            BOOST_TEST(null != ml::core::CStoredStringPtr::makeStoredString("not null"));
            BOOST_TEST(null == nullptr);
            BOOST_TEST(null.get() == nullptr);
        } else {
            BOOST_FAIL("Should not return false in negated boolean context");
        }
    }
    {
        std::string str1("my first string");

        ml::core::CStoredStringPtr ptr1 = ml::core::CStoredStringPtr::makeStoredString(str1);

        if (ptr1) {
            BOOST_TEST(ptr1 == ptr1);
            BOOST_TEST(ptr1 != nullptr);
            BOOST_TEST(ptr1.get() != nullptr);
        } else {
            BOOST_FAIL("Should not return false in boolean context");
        }

        if (!ptr1) {
            BOOST_FAIL("Should not return true in negated boolean context");
        } else {
            BOOST_CHECK_EQUAL(0, ptr1->compare(str1));
        }
    }
    {
        // This is long because the most efficient way to move a small string
        // would be to leave the original value in the moved-from string
        std::string str2("my second string - long enough to not use the small string optimisation");

        ml::core::CStoredStringPtr ptr2 =
            ml::core::CStoredStringPtr::makeStoredString(std::move(str2));

        if (ptr2) {
            BOOST_TEST(ptr2 == ptr2);
            BOOST_TEST(ptr2 != nullptr);
            BOOST_TEST(ptr2.get() != nullptr);
        } else {
            BOOST_FAIL("Should not return false in boolean context");
        }

        if (!ptr2) {
            BOOST_FAIL("Should not return true in negated boolean context");
        } else {
            // str2 should no longer contain its original value, as it should
            // have been moved to the stored string
            BOOST_TEST(ptr2->compare(str2) != 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(testMemoryUsage) {
    {
        ml::core::CStoredStringPtr null;

        BOOST_CHECK_EQUAL(std::size_t(0), ml::core::CMemory::dynamicSize(null));
        BOOST_CHECK_EQUAL(std::size_t(0), null.actualMemoryUsage());
    }
    {
        std::string str1("short");

        ml::core::CStoredStringPtr ptr1 = ml::core::CStoredStringPtr::makeStoredString(str1);

        BOOST_CHECK_EQUAL(std::size_t(0), ml::core::CMemory::dynamicSize(ptr1));
        BOOST_CHECK_EQUAL(ml::core::CMemory::dynamicSize(&str1), ptr1.actualMemoryUsage());
    }
    {
        std::string str2("much longer - YUGE in fact!");

        ml::core::CStoredStringPtr ptr2 = ml::core::CStoredStringPtr::makeStoredString(str2);

        BOOST_CHECK_EQUAL(std::size_t(0), ml::core::CMemory::dynamicSize(ptr2));
        BOOST_CHECK_EQUAL(ml::core::CMemory::dynamicSize(&str2), ptr2.actualMemoryUsage());
    }
}

BOOST_AUTO_TEST_CASE(testHash) {
    using TStoredStringPtrUSet = boost::unordered_set<ml::core::CStoredStringPtr>;

    ml::core::CStoredStringPtr key = ml::core::CStoredStringPtr::makeStoredString("key");

    TStoredStringPtrUSet s;
    BOOST_TEST(s.insert(key).second);

    BOOST_CHECK_EQUAL(std::size_t(1), s.count(key));
}

BOOST_AUTO_TEST_SUITE_END()
