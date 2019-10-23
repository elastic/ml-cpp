/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CLimits.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CLimitsTest)

BOOST_AUTO_TEST_CASE(testTrivial) {
    ml::model::CLimits config;

    BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_AUTOCONFIG_EVENTS,
                        config.autoConfigEvents());
    BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                        config.anomalyMaxTimeBuckets());
    BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_RESULTS_MAX_EXAMPLES,
                        config.maxExamples());
    BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD / 100.0,
                        config.unusualProbabilityThreshold());
    BOOST_REQUIRE_EQUAL(ml::model::CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB,
                        config.memoryLimitMB());
}

BOOST_AUTO_TEST_CASE(testValid) {
    ml::model::CLimits config;
    BOOST_TEST_REQUIRE(config.init("testfiles/mllimits.conf"));

    // This one isn't present in the config file so should be defaulted
    BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                        config.anomalyMaxTimeBuckets());

    BOOST_REQUIRE_EQUAL(size_t(8), config.maxExamples());

    BOOST_REQUIRE_EQUAL(0.005, config.unusualProbabilityThreshold());

    BOOST_REQUIRE_EQUAL(size_t(4567), config.memoryLimitMB());
}

BOOST_AUTO_TEST_CASE(testInvalid) {
    ml::model::CLimits config;
    BOOST_TEST_REQUIRE(!config.init("testfiles/invalidmllimits.conf"));
}

BOOST_AUTO_TEST_SUITE_END()
