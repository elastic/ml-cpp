/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <model/CModelTools.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CToolsTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testDataGatherers, *boost::unit_test::disabled()) {
    // TODO
}

BOOST_AUTO_TEST_CASE(testProbabilityAggregator) {
    // Test a variety of min aggregations.

    {
        LOG_DEBUG(<< "joint");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Min);
        BOOST_TEST_REQUIRE(actual.empty());
        actual.add(maths::CJointProbabilityOfLessLikelySamples());
        BOOST_TEST_REQUIRE(actual.empty());

        double p0;
        BOOST_TEST_REQUIRE(actual.calculate(p0));
        BOOST_REQUIRE_EQUAL(1.0, p0);

        maths::CJointProbabilityOfLessLikelySamples expected;

        for (double p : {0.01, 0.2, 0.001, 0.3, 0.456, 0.1}) {
            actual.add(p);
            expected.add(p);
            BOOST_TEST_REQUIRE(!actual.empty());

            double pi;
            BOOST_TEST_REQUIRE(actual.calculate(pi));
            double pe;
            expected.calculate(pe);
            LOG_DEBUG(<< "pe = " << pe << " pi = " << pi);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(pe, pi, 1e-10);
        }
    }
    {
        LOG_DEBUG(<< "extreme");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Min);
        BOOST_TEST_REQUIRE(actual.empty());
        actual.add(maths::CProbabilityOfExtremeSample());
        BOOST_TEST_REQUIRE(actual.empty());

        double p0;
        BOOST_TEST_REQUIRE(actual.calculate(p0));
        BOOST_REQUIRE_EQUAL(1.0, p0);

        maths::CProbabilityOfExtremeSample expected;

        for (double p : {0.01, 0.2, 0.001, 0.3, 0.456, 0.1}) {
            actual.add(p);
            expected.add(p);
            BOOST_TEST_REQUIRE(!actual.empty());

            double pi;
            BOOST_TEST_REQUIRE(actual.calculate(pi));
            double pe;
            expected.calculate(pe);
            LOG_DEBUG(<< "pe = " << pe << " pi = " << pi);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(pe, pi, 1e-10);
        }
    }
    {
        LOG_DEBUG(<< "minimum");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Min);
        BOOST_TEST_REQUIRE(actual.empty());
        actual.add(maths::CJointProbabilityOfLessLikelySamples());
        actual.add(maths::CProbabilityOfExtremeSample());
        BOOST_TEST_REQUIRE(actual.empty());

        double p0;
        BOOST_TEST_REQUIRE(actual.calculate(p0));
        BOOST_REQUIRE_EQUAL(1.0, p0);

        maths::CJointProbabilityOfLessLikelySamples joint;
        maths::CProbabilityOfExtremeSample extreme;

        for (double p : {0.01, 0.2, 0.001, 0.3, 0.456, 0.1}) {
            actual.add(p);
            joint.add(p);
            extreme.add(p);
            BOOST_TEST_REQUIRE(!actual.empty());

            double pi;
            BOOST_TEST_REQUIRE(actual.calculate(pi));
            double pj, pe;
            joint.calculate(pj);
            extreme.calculate(pe);
            LOG_DEBUG(<< "pj = " << pj << " pe = " << pe << " pi = " << pi);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(std::min(pj, pe), pi, 1e-10);
        }
    }
    {
        LOG_DEBUG(<< "sum");
        CModelTools::CProbabilityAggregator actual(CModelTools::CProbabilityAggregator::E_Sum);
        BOOST_TEST_REQUIRE(actual.empty());
        actual.add(maths::CJointProbabilityOfLessLikelySamples(), 0.5);
        actual.add(maths::CProbabilityOfExtremeSample(), 0.5);
        BOOST_TEST_REQUIRE(actual.empty());

        double p0;
        BOOST_TEST_REQUIRE(actual.calculate(p0));
        BOOST_REQUIRE_EQUAL(1.0, p0);

        maths::CJointProbabilityOfLessLikelySamples joint;
        maths::CProbabilityOfExtremeSample extreme;

        for (double p : {0.01, 0.2, 0.001, 0.3, 0.456, 0.1}) {
            actual.add(p);
            joint.add(p);
            extreme.add(p);
            BOOST_TEST_REQUIRE(!actual.empty());

            double pi;
            BOOST_TEST_REQUIRE(actual.calculate(pi));
            double pj, pe;
            joint.calculate(pj);
            extreme.calculate(pe);
            LOG_DEBUG(<< "pj = " << pj << " pe = " << pe << " pi = " << pi);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(std::sqrt(pj) * std::sqrt(pe), pi, 1e-10);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
