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

#include "CToolsTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <model/CModelTools.h>

#include <boost/range.hpp>

using namespace ml;
using namespace model;

void CToolsTest::testDataGatherers() {
    // TODO
}

void CToolsTest::testProbabilityAggregator() {
    // Test a variety of min aggregations.

    {
        LOG_DEBUG(<< "joint");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Min);
        CPPUNIT_ASSERT(actual.empty());
        actual.add(maths::CJointProbabilityOfLessLikelySamples());
        CPPUNIT_ASSERT(actual.empty());

        double p0;
        CPPUNIT_ASSERT(actual.calculate(p0));
        CPPUNIT_ASSERT_EQUAL(1.0, p0);

        maths::CJointProbabilityOfLessLikelySamples expected;

        double p[] = {0.01, 0.2, 0.001, 0.3, 0.456, 0.1};

        for (std::size_t i = 0u; i < boost::size(p); ++i) {
            actual.add(p[0]);
            expected.add(p[0]);
            CPPUNIT_ASSERT(!actual.empty());

            double pi;
            CPPUNIT_ASSERT(actual.calculate(pi));
            double pe;
            expected.calculate(pe);
            LOG_DEBUG(<< "pe = " << pe << " pi = " << pi);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(pe, pi, 1e-10);
        }
    }
    {
        LOG_DEBUG(<< "extreme");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Min);
        CPPUNIT_ASSERT(actual.empty());
        actual.add(maths::CProbabilityOfExtremeSample());
        CPPUNIT_ASSERT(actual.empty());

        double p0;
        CPPUNIT_ASSERT(actual.calculate(p0));
        CPPUNIT_ASSERT_EQUAL(1.0, p0);

        maths::CProbabilityOfExtremeSample expected;

        double p[] = {0.01, 0.2, 0.001, 0.3, 0.456, 0.1};

        for (std::size_t i = 0u; i < boost::size(p); ++i) {
            actual.add(p[0]);
            expected.add(p[0]);
            CPPUNIT_ASSERT(!actual.empty());

            double pi;
            CPPUNIT_ASSERT(actual.calculate(pi));
            double pe;
            expected.calculate(pe);
            LOG_DEBUG(<< "pe = " << pe << " pi = " << pi);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(pe, pi, 1e-10);
        }
    }
    {
        LOG_DEBUG(<< "minimum");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Min);
        CPPUNIT_ASSERT(actual.empty());
        actual.add(maths::CJointProbabilityOfLessLikelySamples());
        actual.add(maths::CProbabilityOfExtremeSample());
        CPPUNIT_ASSERT(actual.empty());

        double p0;
        CPPUNIT_ASSERT(actual.calculate(p0));
        CPPUNIT_ASSERT_EQUAL(1.0, p0);

        maths::CJointProbabilityOfLessLikelySamples joint;
        maths::CProbabilityOfExtremeSample extreme;

        double p[] = {0.01, 0.2, 0.001, 0.3, 0.456, 0.1};

        for (std::size_t i = 0u; i < boost::size(p); ++i) {
            actual.add(p[0]);
            joint.add(p[0]);
            extreme.add(p[0]);
            CPPUNIT_ASSERT(!actual.empty());

            double pi;
            CPPUNIT_ASSERT(actual.calculate(pi));
            double pj, pe;
            joint.calculate(pj);
            extreme.calculate(pe);
            LOG_DEBUG(<< "pj = " << pj << " pe = " << pe << " pi = " << pi);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(std::min(pj, pe), pi, 1e-10);
        }
    }
    {
        LOG_DEBUG(<< "sum");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Sum);
        CPPUNIT_ASSERT(actual.empty());
        actual.add(maths::CJointProbabilityOfLessLikelySamples(), 0.5);
        actual.add(maths::CProbabilityOfExtremeSample(), 0.5);
        CPPUNIT_ASSERT(actual.empty());

        double p0;
        CPPUNIT_ASSERT(actual.calculate(p0));
        CPPUNIT_ASSERT_EQUAL(1.0, p0);

        maths::CJointProbabilityOfLessLikelySamples joint;
        maths::CProbabilityOfExtremeSample extreme;

        double p[] = {0.01, 0.2, 0.001, 0.3, 0.456, 0.1};

        for (std::size_t i = 0u; i < boost::size(p); ++i) {
            actual.add(p[0]);
            joint.add(p[0]);
            extreme.add(p[0]);
            CPPUNIT_ASSERT(!actual.empty());

            double pi;
            CPPUNIT_ASSERT(actual.calculate(pi));
            double pj, pe;
            joint.calculate(pj);
            extreme.calculate(pe);
            LOG_DEBUG(<< "pj = " << pj << " pe = " << pe << " pi = " << pi);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(std::sqrt(pj) * std::sqrt(pe), pi, 1e-10);
        }
    }
}

CppUnit::Test* CToolsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CToolsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CToolsTest>(
        "CToolsTest::testProbabilityAggregator", &CToolsTest::testProbabilityAggregator));

    return suiteOfTests;
}
