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

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CLimits.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CHierarchicalResultsNormalizerTest)
using namespace ml;

namespace {
constexpr std::size_t BUCKET_SIZE{3600};
}

BOOST_AUTO_TEST_CASE(testHierarchicalResultsNormalizerShouldUpdateMemoryUsage) {
    model::CLimits limits;
    auto modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
    model::CHierarchicalResultsNormalizer normalizer(limits, modelConfig);

    std::size_t initialMemoryUsage = normalizer.memoryUsage();
    normalizer.setJob(model::CHierarchicalResultsNormalizer::E_UpdateQuantiles);
    limits.resourceMonitor().forceRefreshAll();
    BOOST_TEST_REQUIRE(normalizer.memoryUsage() > initialMemoryUsage);
}

BOOST_AUTO_TEST_SUITE_END()
