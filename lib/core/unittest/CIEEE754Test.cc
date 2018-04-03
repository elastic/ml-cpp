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

#include "CIEEE754Test.h"

#include <core/CLogger.h>
#include <core/CIEEE754.h>

#include <cmath>
#include <iomanip>
#include <sstream>

using namespace ml;
using namespace core;

void CIEEE754Test::testRound()
{
    {
        // Check it matches float precision.
        double test1 = 0.049999998;
        std::ostringstream o1;
        o1 << std::setprecision(10) << static_cast<float>(test1);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test1, CIEEE754::E_SinglePrecision);
        LOG_DEBUG("test1 " << o1.str() << " " << o2.str());
        CPPUNIT_ASSERT_EQUAL(o1.str(), o2.str());
    }
    {
        // Check it matches float precision.
        double test2 = 0.0499999998;
        std::ostringstream o1;
        o1 << std::setprecision(10) << static_cast<float>(test2);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test2, CIEEE754::E_SinglePrecision);
        LOG_DEBUG("test2 " << o1.str() << " " << o2.str());
        CPPUNIT_ASSERT_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding away from zero.
        double test3 = 0.5 - 0.5 / static_cast<double>(1 << 25);
        std::ostringstream o1;
        o1 << std::setprecision(10) << 0.5;
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test3, CIEEE754::E_SinglePrecision);
        LOG_DEBUG("test3 " << o1.str() << " " << o2.str());
        CPPUNIT_ASSERT_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding away from zero.
        double test4 = -(0.5 - 0.5 / static_cast<double>(1 << 25));
        std::ostringstream o1;
        o1 << std::setprecision(10) << -0.5;
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test4, CIEEE754::E_SinglePrecision);
        LOG_DEBUG("test4 " << o1.str() << " " << o2.str());
        CPPUNIT_ASSERT_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding for very large numbers.
        double test5 = 0.49999998 * std::pow(2.0, 1023.0);
        std::ostringstream o1;
        o1 << std::setprecision(10) << 0.4999999702 * std::pow(2, 1023.0);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test5, CIEEE754::E_SinglePrecision);
        LOG_DEBUG("test5 " << o1.str() << " " << o2.str());
        CPPUNIT_ASSERT_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding for very large numbers.
        double test6 = 0.499999998 * std::pow(2.0, 1023.0);
        std::ostringstream o1;
        o1 << std::setprecision(10) << std::pow(2.0, 1022.0);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test6, CIEEE754::E_SinglePrecision);
        LOG_DEBUG("test6 " << o1.str() << " " << o2.str());
        CPPUNIT_ASSERT_EQUAL(o1.str(), o2.str());
    }
}

CppUnit::Test *CIEEE754Test::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CIEEE754Test");

    suiteOfTests->addTest( new CppUnit::TestCaller<CIEEE754Test>(
                                   "CIEEE754Test::testRound",
                                   &CIEEE754Test::testRound) );

    return suiteOfTests;

}

