/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CRandomizedPeriodicityTest.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CRandomizedPeriodicityTestTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;

const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
}

BOOST_AUTO_TEST_CASE(testAccuracy) {
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMaxAccumulator =
        maths::CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>>;
    using TFunction = double (*)(core_t::TTime);

    test::CRandomNumbers rng;

    TMeanAccumulator typeI;
    TMeanAccumulator typeII;
    for (std::size_t t = 0u; t < 5; ++t) {
        LOG_DEBUG(<< "*** test = " << t << " ***");

        core_t::TTime time = 0;
        core_t::TTime day = 0;

        TDoubleVec samples;
        rng.generateLogNormalSamples(1.0, 4.0, 84000, samples);

        maths::CRandomizedPeriodicityTest::reset();

        maths::CRandomizedPeriodicityTest rtests[8];
        double falsePositives[3] = {0.0, 0.0, 0.0};
        double trueNegatives[3] = {0.0, 0.0, 0.0};
        double truePositives[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        double falseNegatives[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        TMeanVarAccumulator timeToDetectionMoments[5];
        TMaxAccumulator timeToDetectionMax[5];
        core_t::TTime lastTruePositive[5] = {time, time, time, time, time};
        TFunction functions[] = {&constant,     &ramp,         &markov,
                                 &smoothDaily,  &smoothWeekly, &spikeyDaily,
                                 &spikeyWeekly, &weekends};

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            for (std::size_t j = 0u; j < boost::size(functions); ++j) {
                rtests[j].add(time, 600.0 * (functions[j])(time) + samples[i]);
            }
            if (time >= day + DAY) {
                for (std::size_t j = 0u; j < boost::size(rtests); ++j) {
                    if (j < 3) {
                        (rtests[j].test() ? falsePositives[j] : trueNegatives[j]) += 1.0;
                    } else {
                        (rtests[j].test() ? truePositives[j - 3]
                                          : falseNegatives[j - 3]) += 1.0;
                        if (rtests[j].test()) {
                            timeToDetectionMoments[j - 3].add(
                                time - lastTruePositive[j - 3]);
                            timeToDetectionMax[j - 3].add(
                                static_cast<double>(time - lastTruePositive[j - 3]));
                            lastTruePositive[j - 3] = time;
                        }
                    }
                }
                day += DAY;
            }
            time += HALF_HOUR;
        }

        LOG_DEBUG(<< "falsePositives = " << core::CContainerPrinter::print(falsePositives));
        LOG_DEBUG(<< "trueNegatives = " << core::CContainerPrinter::print(trueNegatives));
        for (std::size_t i = 0u; i < boost::size(falsePositives); ++i) {
            BOOST_TEST_REQUIRE(falsePositives[i] / trueNegatives[i] < 0.1);
            typeI.add(falsePositives[i] / trueNegatives[i]);
        }
        LOG_DEBUG(<< "truePositives = " << core::CContainerPrinter::print(truePositives));
        LOG_DEBUG(<< "falseNegatives = " << core::CContainerPrinter::print(falseNegatives));
        for (std::size_t i = 0u; i < boost::size(falsePositives); ++i) {
            BOOST_TEST_REQUIRE(falseNegatives[i] / truePositives[i] < 0.2);
            typeII.add(falseNegatives[i] / truePositives[i]);
        }

        for (std::size_t i = 0u; i < boost::size(timeToDetectionMoments); ++i) {
            LOG_DEBUG(<< "time to detect moments = " << timeToDetectionMoments[i]);
            LOG_DEBUG(<< "maximum time to detect = " << timeToDetectionMax[i][0]);
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(timeToDetectionMoments[i]) < 1.5 * DAY);
            BOOST_TEST_REQUIRE(std::sqrt(maths::CBasicStatistics::variance(
                           timeToDetectionMoments[i])) < 5 * DAY);
            BOOST_TEST_REQUIRE(timeToDetectionMax[i][0] <= 27 * WEEK);
        }
    }
    LOG_DEBUG(<< "type I  = " << maths::CBasicStatistics::mean(typeI));
    LOG_DEBUG(<< "type II = " << maths::CBasicStatistics::mean(typeII));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(typeI) < 0.015);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(typeII) < 0.05);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that persistence is idempotent.

    maths::CRandomizedPeriodicityTest test;
    for (core_t::TTime t = 1400000000; t < 1400050000; t += 5000) {
        test.add(t, 0.2);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        test.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    std::string origStaticsXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        test.staticsAcceptPersistInserter(inserter);
        inserter.toXml(origStaticsXml);
    }

    // Check that the static state is also preserved
    uint64_t origNextRandom = test.ms_Rng();

    LOG_DEBUG(<< "XML representation:\n" << origXml);

    // Restore the XML into a new test
    maths::CRandomizedPeriodicityTest test2;
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(
            std::bind(&maths::CRandomizedPeriodicityTest::acceptRestoreTraverser,
                      &test2, std::placeholders::_1)));
    }
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        test2.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
    BOOST_REQUIRE_EQUAL(test.checksum(), test2.checksum());

    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origStaticsXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(
            &maths::CRandomizedPeriodicityTest::staticsAcceptRestoreTraverser));
    }
    std::string newStaticsXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        test2.staticsAcceptPersistInserter(inserter);
        inserter.toXml(newStaticsXml);
    }
    BOOST_REQUIRE_EQUAL(origStaticsXml, newStaticsXml);

    uint64_t newNextRandom = test2.ms_Rng();
    BOOST_REQUIRE_EQUAL(origNextRandom, newNextRandom);
}

BOOST_AUTO_TEST_SUITE_END()
