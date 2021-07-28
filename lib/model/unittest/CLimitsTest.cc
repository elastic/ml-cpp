/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CLimits.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CLimitsTest)

BOOST_AUTO_TEST_CASE(testTrivial) {
    ml::model::CLimits config;

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
    {
        ml::model::CLimits config;
        BOOST_TEST_REQUIRE(config.init("testfiles/mllimits.conf"));

        // This one isn't present in the config file so should be defaulted
        BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                            config.anomalyMaxTimeBuckets());

        BOOST_REQUIRE_EQUAL(8, config.maxExamples());

        BOOST_REQUIRE_EQUAL(0.005, config.unusualProbabilityThreshold());

        BOOST_REQUIRE_EQUAL(4567, config.memoryLimitMB());
    }
    {
        ml::model::CLimits config;

        // initialise from given values
        config.init(2, 4096);

        // This one should always be defaulted when there is no config file
        BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                            config.anomalyMaxTimeBuckets());

        // This also should be defaulted (as a percentage)
        BOOST_REQUIRE_EQUAL(ml::model::CLimits::DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD / 100,
                            config.unusualProbabilityThreshold());

        // These two should be as specified in the constructor.
        BOOST_REQUIRE_EQUAL(2, config.maxExamples());
        BOOST_REQUIRE_EQUAL(4096, config.memoryLimitMB());
    }
}

BOOST_AUTO_TEST_CASE(testInvalid) {
    ml::model::CLimits config;
    BOOST_TEST_REQUIRE(!config.init("testfiles/invalidmllimits.conf"));
}

BOOST_AUTO_TEST_SUITE_END()
