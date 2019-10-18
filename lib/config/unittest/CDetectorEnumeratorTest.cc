/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDetectorEnumerator.h>
#include <config/CDetectorSpecification.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CDetectorEnumeratorTest)

using namespace ml;

namespace {
std::string print(const config::CDetectorEnumerator::TDetectorSpecificationVec& spec,
                  const std::string& indent = std::string()) {
    std::ostringstream result;
    for (std::size_t i = 0u; i < spec.size(); ++i) {
        result << indent << spec[i].description() << "\n";
    }
    return result.str();
}
}

BOOST_AUTO_TEST_CASE(testAll) {
    std::string empty;
    config::CAutoconfigurerParams params(empty, empty, false, false);
    config::CDetectorEnumerator enumerator(params);

    enumerator.addFunction(config_t::E_Count);
    enumerator.addFunction(config_t::E_DistinctCount);
    enumerator.addFunction(config_t::E_Mean);

    config::CDetectorEnumerator::TDetectorSpecificationVec spec;

    {
        enumerator.generate(spec);
        LOG_DEBUG(<< "1) detectors =\n" << print(spec, " "));
        std::string expected = "[low_|high_][non_zero_]count\n";
        BOOST_CHECK_EQUAL(expected, print(spec));
    }

    {
        enumerator.addCategoricalFunctionArgument("port");
        enumerator.addMetricFunctionArgument("bytes");
        enumerator.generate(spec);
        LOG_DEBUG(<< "2) detectors =\n" << print(spec, " "));
        std::string expected = "[low_|high_][non_zero_]count\n"
                               "[low_|high_]distinct_count(port)\n"
                               "[low_|high_]mean(bytes)\n";
        BOOST_CHECK_EQUAL(expected, print(spec));
    }

    {
        enumerator.addByField("process");
        enumerator.addByField("parent_process");
        enumerator.generate(spec);
        LOG_DEBUG(<< "3) detectors =\n" << print(spec, " "));
        std::string expected = "[low_|high_][non_zero_]count\n"
                               "[low_|high_]distinct_count(port)\n"
                               "[low_|high_]mean(bytes)\n"
                               "[low_|high_][non_zero_]count by 'process'\n"
                               "[low_|high_][non_zero_]count by 'parent_process'\n"
                               "[low_|high_]distinct_count(port) by 'process'\n"
                               "[low_|high_]distinct_count(port) by 'parent_process'\n"
                               "[low_|high_]mean(bytes) by 'process'\n"
                               "[low_|high_]mean(bytes) by 'parent_process'\n";
        BOOST_CHECK_EQUAL(expected, print(spec));
    }

    {
        enumerator.addOverField("machine");
        enumerator.addOverField("person");
        enumerator.generate(spec);
        LOG_DEBUG(<< "4) detectors =\n" << print(spec, " "));
        std::string expected =
            "[low_|high_][non_zero_]count\n"
            "[low_|high_]distinct_count(port)\n"
            "[low_|high_]mean(bytes)\n"
            "[low_|high_][non_zero_]count by 'process'\n"
            "[low_|high_][non_zero_]count by 'parent_process'\n"
            "[low_|high_]distinct_count(port) by 'process'\n"
            "[low_|high_]distinct_count(port) by 'parent_process'\n"
            "[low_|high_]mean(bytes) by 'process'\n"
            "[low_|high_]mean(bytes) by 'parent_process'\n"
            "[low_|high_]count over 'machine'\n"
            "[low_|high_]count over 'person'\n"
            "[low_|high_]distinct_count(port) over 'machine'\n"
            "[low_|high_]distinct_count(port) over 'person'\n"
            "[low_|high_]mean(bytes) over 'machine'\n"
            "[low_|high_]mean(bytes) over 'person'\n"
            "[low_|high_]count by 'process' over 'machine'\n"
            "[low_|high_]count by 'process' over 'person'\n"
            "[low_|high_]count by 'parent_process' over 'machine'\n"
            "[low_|high_]count by 'parent_process' over 'person'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'machine'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'person'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'machine'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'person'\n"
            "[low_|high_]mean(bytes) by 'process' over 'machine'\n"
            "[low_|high_]mean(bytes) by 'process' over 'person'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'machine'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'person'\n";
        BOOST_CHECK_EQUAL(expected, print(spec));
    }

    {
        enumerator.addPartitionField("process");
        enumerator.addPartitionField("machine");
        enumerator.addPartitionField("data_centre");
        enumerator.generate(spec);
        std::string expected =
            "[low_|high_][non_zero_]count\n"
            "[low_|high_]distinct_count(port)\n"
            "[low_|high_]mean(bytes)\n"
            "[low_|high_][non_zero_]count by 'process'\n"
            "[low_|high_][non_zero_]count by 'parent_process'\n"
            "[low_|high_]distinct_count(port) by 'process'\n"
            "[low_|high_]distinct_count(port) by 'parent_process'\n"
            "[low_|high_]mean(bytes) by 'process'\n"
            "[low_|high_]mean(bytes) by 'parent_process'\n"
            "[low_|high_]count over 'machine'\n"
            "[low_|high_]count over 'person'\n"
            "[low_|high_]distinct_count(port) over 'machine'\n"
            "[low_|high_]distinct_count(port) over 'person'\n"
            "[low_|high_]mean(bytes) over 'machine'\n"
            "[low_|high_]mean(bytes) over 'person'\n"
            "[low_|high_][non_zero_]count partition 'process'\n"
            "[low_|high_][non_zero_]count partition 'machine'\n"
            "[low_|high_][non_zero_]count partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) partition 'process'\n"
            "[low_|high_]distinct_count(port) partition 'machine'\n"
            "[low_|high_]distinct_count(port) partition 'data_centre'\n"
            "[low_|high_]mean(bytes) partition 'process'\n"
            "[low_|high_]mean(bytes) partition 'machine'\n"
            "[low_|high_]mean(bytes) partition 'data_centre'\n"
            "[low_|high_]count by 'process' over 'machine'\n"
            "[low_|high_]count by 'process' over 'person'\n"
            "[low_|high_]count by 'parent_process' over 'machine'\n"
            "[low_|high_]count by 'parent_process' over 'person'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'machine'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'person'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'machine'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'person'\n"
            "[low_|high_]mean(bytes) by 'process' over 'machine'\n"
            "[low_|high_]mean(bytes) by 'process' over 'person'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'machine'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'person'\n"
            "[low_|high_][non_zero_]count by 'process' partition 'machine'\n"
            "[low_|high_][non_zero_]count by 'process' partition 'data_centre'\n"
            "[low_|high_][non_zero_]count by 'parent_process' partition 'process'\n"
            "[low_|high_][non_zero_]count by 'parent_process' partition 'machine'\n"
            "[low_|high_][non_zero_]count by 'parent_process' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) by 'process' partition 'machine'\n"
            "[low_|high_]distinct_count(port) by 'process' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' partition 'process'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' partition 'machine'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) by 'process' partition 'machine'\n"
            "[low_|high_]mean(bytes) by 'process' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) by 'parent_process' partition 'process'\n"
            "[low_|high_]mean(bytes) by 'parent_process' partition 'machine'\n"
            "[low_|high_]mean(bytes) by 'parent_process' partition 'data_centre'\n"
            "[low_|high_]count over 'machine' partition 'process'\n"
            "[low_|high_]count over 'machine' partition 'data_centre'\n"
            "[low_|high_]count over 'person' partition 'process'\n"
            "[low_|high_]count over 'person' partition 'machine'\n"
            "[low_|high_]count over 'person' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) over 'machine' partition 'process'\n"
            "[low_|high_]distinct_count(port) over 'machine' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) over 'person' partition 'process'\n"
            "[low_|high_]distinct_count(port) over 'person' partition 'machine'\n"
            "[low_|high_]distinct_count(port) over 'person' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) over 'machine' partition 'process'\n"
            "[low_|high_]mean(bytes) over 'machine' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) over 'person' partition 'process'\n"
            "[low_|high_]mean(bytes) over 'person' partition 'machine'\n"
            "[low_|high_]mean(bytes) over 'person' partition 'data_centre'\n"
            "[low_|high_]count by 'process' over 'machine' partition 'data_centre'\n"
            "[low_|high_]count by 'process' over 'person' partition 'machine'\n"
            "[low_|high_]count by 'process' over 'person' partition 'data_centre'\n"
            "[low_|high_]count by 'parent_process' over 'machine' partition 'process'\n"
            "[low_|high_]count by 'parent_process' over 'machine' partition 'data_centre'\n"
            "[low_|high_]count by 'parent_process' over 'person' partition 'process'\n"
            "[low_|high_]count by 'parent_process' over 'person' partition 'machine'\n"
            "[low_|high_]count by 'parent_process' over 'person' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'machine' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'person' partition 'machine'\n"
            "[low_|high_]distinct_count(port) by 'process' over 'person' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'machine' partition 'process'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'machine' partition 'data_centre'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'person' partition 'process'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'person' partition 'machine'\n"
            "[low_|high_]distinct_count(port) by 'parent_process' over 'person' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) by 'process' over 'machine' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) by 'process' over 'person' partition 'machine'\n"
            "[low_|high_]mean(bytes) by 'process' over 'person' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'machine' partition 'process'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'machine' partition 'data_centre'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'person' partition 'process'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'person' partition 'machine'\n"
            "[low_|high_]mean(bytes) by 'parent_process' over 'person' partition 'data_centre'\n";
        LOG_DEBUG(<< "5) detectors =\n" << print(spec, " "));
        BOOST_CHECK_EQUAL(expected, print(spec));
    }
}


BOOST_AUTO_TEST_SUITE_END()
