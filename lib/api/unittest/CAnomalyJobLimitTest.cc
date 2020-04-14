/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <maths/CIntegerTools.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CJsonOutputWriter.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "CMockDataProcessor.h"

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>

BOOST_TEST_DONT_PRINT_LOG_VALUE(rapidjson::Value::MemberIterator)

BOOST_AUTO_TEST_SUITE(CAnomalyJobLimitTest)

using namespace ml;

std::set<std::string> getUniqueValues(const std::string& key, const std::string& output) {
    std::set<std::string> values;
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(output);
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    BOOST_TEST_REQUIRE(doc.IsArray());

    size_t i = 0;

    while (true) {
        rapidjson::Value* p1 = rapidjson::Pointer("/" + std::to_string(i)).Get(doc);
        if (p1 != nullptr) {
            size_t j = 0;
            while (true) {
                rapidjson::Value* p2 = rapidjson::Pointer("/" + std::to_string(i) + "/records/" +
                                                          std::to_string(j))
                                           .Get(doc);
                if (p2 != nullptr) {
                    size_t k = 0;
                    while (true) {
                        rapidjson::Value* p3 =
                            rapidjson::Pointer("/" + std::to_string(i) + "/records/" +
                                               std::to_string(j) + "/causes/" +
                                               std::to_string(k) + "/" + key)
                                .Get(doc);

                        if (p3 != nullptr) {
                            values.insert(p3->GetString());
                        } else {
                            break;
                        }
                        ++k;
                    }
                } else {
                    break;
                }
                ++j;
            }
        } else {
            break;
        }
        ++i;
    }

    return values;
}

BOOST_AUTO_TEST_CASE(testAccuracy) {
    // Check that the amount of memory used when we go over the
    // resource limit is close enough to the limit that we specified

    std::size_t nonLimitedUsage{0};
    std::size_t limitedUsage{0};

    {
        // Without limits, this data set should make the models around
        // 1230000 bytes
        // Run the data once to find out what the current platform uses
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clause;
        clause.push_back("value");
        clause.push_back("by");
        clause.push_back("colour");
        clause.push_back("over");
        clause.push_back("species");
        clause.push_back("partitionfield=greenhouse");

        BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        model::CLimits limits;
        //limits.resourceMonitor().m_ByteLimitHigh = 100000;
        //limits.resourceMonitor().m_ByteLimitLow = 90000;

        {
            LOG_TRACE(<< "Setting up job");
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                                 wrappedOutputStream, nullptr);

            std::ifstream inputStrm("testfiles/resource_accuracy.csv");
            BOOST_TEST_REQUIRE(inputStrm.is_open());
            api::CCsvInputParser parser(inputStrm);

            LOG_TRACE(<< "Reading file");
            BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::bind(
                &api::CAnomalyJob::handleRecord, &job, std::placeholders::_1)));

            LOG_TRACE(<< "Checking results");

            BOOST_REQUIRE_EQUAL(uint64_t(18630), job.numRecordsHandled());

            nonLimitedUsage = limits.resourceMonitor().totalMemory();
        }
    }
    {
        // Now run the data with limiting
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clause;
        clause.push_back("value");
        clause.push_back("by");
        clause.push_back("colour");
        clause.push_back("over");
        clause.push_back("species");
        clause.push_back("partitionfield=greenhouse");

        BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);
        model::CLimits limits;

        std::stringstream outputStrm;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

            limits.resourceMonitor().m_ByteLimitHigh = nonLimitedUsage / 10;
            limits.resourceMonitor().m_ByteLimitLow =
                limits.resourceMonitor().m_ByteLimitHigh - 1024;

            LOG_TRACE(<< "Setting up job");
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                                 wrappedOutputStream, nullptr);

            std::ifstream inputStrm("testfiles/resource_accuracy.csv");
            BOOST_TEST_REQUIRE(inputStrm.is_open());
            api::CCsvInputParser parser(inputStrm);

            LOG_TRACE(<< "Reading file");
            BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::bind(
                &api::CAnomalyJob::handleRecord, &job, std::placeholders::_1)));

            LOG_TRACE(<< "Checking results");

            BOOST_REQUIRE_EQUAL(uint64_t(18630), job.numRecordsHandled());

            // TODO this limit must be tightened once there is more granular
            // control over the model memory creation
            limitedUsage = limits.resourceMonitor().totalMemory();
        }
        LOG_TRACE(<< outputStrm.str());

        LOG_DEBUG(<< "Non-limited usage: " << nonLimitedUsage << "; limited: " << limitedUsage);
        BOOST_TEST_REQUIRE(limitedUsage < nonLimitedUsage);
    }
}

BOOST_AUTO_TEST_CASE(testLimit) {
    using TStrSet = std::set<std::string>;

    std::stringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        // Run the data without any resource limits and check that
        // all the expected fields are in the results set
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clause;
        clause.push_back("value");
        clause.push_back("by");
        clause.push_back("colour");
        clause.push_back("over");
        clause.push_back("species");
        clause.push_back("partitionfield=greenhouse");

        BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);

        LOG_TRACE(<< "Setting up job");
        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                             wrappedOutputStream, nullptr);

        std::ifstream inputStrm("testfiles/resource_limits_3_2over_3partition.csv");
        BOOST_TEST_REQUIRE(inputStrm.is_open());
        api::CCsvInputParser parser(inputStrm);

        LOG_TRACE(<< "Reading file");
        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::bind(
            &api::CAnomalyJob::handleRecord, &job, std::placeholders::_1)));
        LOG_TRACE(<< "Checking results");
        BOOST_REQUIRE_EQUAL(uint64_t(1176), job.numRecordsHandled());
    }

    std::string out = outputStrm.str();

    TStrSet partitions = getUniqueValues("partition_field_value", out);
    TStrSet people = getUniqueValues("over_field_value", out);
    TStrSet attributes = getUniqueValues("by_field_value", out);
    BOOST_REQUIRE_EQUAL(std::size_t(3), partitions.size());
    BOOST_REQUIRE_EQUAL(std::size_t(2), people.size());
    BOOST_REQUIRE_EQUAL(std::size_t(2), attributes.size());

    outputStrm.str("");
    outputStrm.clear();
    {
        // Run the data with some resource limits after the first 4 records and
        // check that we get only anomalies from the first 2 partitions
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clause;
        clause.push_back("value");
        clause.push_back("by");
        clause.push_back("colour");
        clause.push_back("over");
        clause.push_back("species");
        clause.push_back("partitionfield=greenhouse");

        BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);

        //::CMockOutputWriter resultsHandler;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        LOG_TRACE(<< "Setting up job");
        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                             wrappedOutputStream, nullptr);

        std::ifstream inputStrm("testfiles/resource_limits_3_2over_3partition_first8.csv");
        BOOST_TEST_REQUIRE(inputStrm.is_open());
        api::CCsvInputParser parser(inputStrm);

        LOG_TRACE(<< "Reading file");
        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::bind(
            &api::CAnomalyJob::handleRecord, &job, std::placeholders::_1)));
        // Now turn on the resource limiting
        limits.resourceMonitor().m_ByteLimitHigh = 0;
        limits.resourceMonitor().m_ByteLimitLow = 0;
        limits.resourceMonitor().m_AllowAllocations = false;

        std::ifstream inputStrm2("testfiles/resource_limits_3_2over_3partition_last1169.csv");
        BOOST_TEST_REQUIRE(inputStrm2.is_open());
        api::CCsvInputParser parser2(inputStrm2);

        LOG_TRACE(<< "Reading second file");
        BOOST_TEST_REQUIRE(parser2.readStreamIntoMaps(std::bind(
            &api::CAnomalyJob::handleRecord, &job, std::placeholders::_1)));
        LOG_TRACE(<< "Checking results");
        BOOST_REQUIRE_EQUAL(uint64_t(1180), job.numRecordsHandled());
    }

    out = outputStrm.str();

    partitions = getUniqueValues("partition_field_value", out);
    people = getUniqueValues("over_field_value", out);
    attributes = getUniqueValues("by_field_value", out);
    BOOST_REQUIRE_EQUAL(std::size_t(1), partitions.size());
    BOOST_REQUIRE_EQUAL(std::size_t(2), people.size());
    BOOST_REQUIRE_EQUAL(std::size_t(1), attributes.size());
}

BOOST_AUTO_TEST_CASE(testModelledEntityCountForFixedMemoryLimit) {
    using TOptionalDouble = boost::optional<double>;
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TGenerator = std::function<TOptionalDouble(core_t::TTime)>;
    using TGeneratorVec = std::vector<TGenerator>;

    test::CRandomNumbers rng;

    // Generators for a variety of data characteristics.

    TGenerator periodic = [&rng](core_t::TTime time) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 3.0, 1, noise);
        return TOptionalDouble{20.0 * std::sin(2.0 * boost::math::double_constants::pi *
                                               static_cast<double>(time) /
                                               static_cast<double>(core::constants::DAY)) +
                               noise[0]};
    };
    TGenerator tradingDays = [&periodic](core_t::TTime time) {
        double amplitude[]{1.0, 1.0, 0.7, 0.8, 1.0, 0.1, 0.1};
        return TOptionalDouble{amplitude[(time % core::constants::WEEK) / core::constants::DAY] *
                               *periodic(time)};
    };
    TGenerator level = [&rng](core_t::TTime) {
        TDoubleVec noise;
        rng.generateNormalSamples(10.0, 5.0, 1, noise);
        return TOptionalDouble{noise[0]};
    };
    TGenerator ramp = [&rng](core_t::TTime time) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 1.0, 1, noise);
        return TOptionalDouble{static_cast<double>(time) /
                                   static_cast<double>(core::constants::DAY) +
                               noise[0]};
    };
    TGenerator sparse = [&rng, &level](core_t::TTime time) {
        TDoubleVec uniform01;
        rng.generateUniformSamples(0.0, 1.0, 1, uniform01);
        return uniform01[0] < 0.1 ? level(time) : boost::none;
    };

    // We assert on the number of by, partition and over fields we can
    // create for a small(ish) memory limit to catch large changes in
    // the memory used per partition of the data.

    struct STestParams {
        core_t::TTime s_BucketLength;
        std::size_t s_ExpectedByFields;
        std::size_t s_ExpectedOverFields;
        std::size_t s_ExpectedPartitionFields;
        std::size_t s_ExpectedByMemoryUsageRelativeErrorDivisor;
        std::size_t s_ExpectedPartitionUsageRelativeErrorDivisor;
        std::size_t s_ExpectedOverUsageRelativeErrorDivisor;
    } testParams[]{{600, 550, 6000, 300, 33, 40, 40},
                   {3600, 550, 5500, 300, 27, 25, 20},
                   {172800, 150, 850, 110, 6, 6, 3}};

    for (const auto& testParam : testParams) {
        TGeneratorVec generators{periodic, tradingDays, level, ramp, sparse};
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        api::CAnomalyJob::TStrStrUMap dataRows;
        TSizeVec generator;

        LOG_DEBUG(<< "**** Test by with bucketLength = " << testParam.s_BucketLength << " ****");
        {
            std::size_t memoryLimit{10 /*MB*/};
            model::CLimits limits;
            limits.resourceMonitor().memoryLimit(memoryLimit);
            api::CFieldConfig fieldConfig;
            api::CFieldConfig::TStrVec clauses{"mean(foo)", "by", "bar"};
            fieldConfig.initFromClause(clauses);
            model::CAnomalyDetectorModelConfig modelConfig =
                model::CAnomalyDetectorModelConfig::defaultConfig(testParam.s_BucketLength);
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                                 wrappedOutputStream, nullptr);

            core_t::TTime startTime{1495110323};
            core_t::TTime endTime{1495260323};
            core_t::TTime time{startTime};
            double reportProgress{0.0};
            for (/**/; time < endTime; time += testParam.s_BucketLength) {
                double progress{static_cast<double>(time - startTime) /
                                static_cast<double>(endTime - startTime)};
                if (progress >= reportProgress) {
                    LOG_DEBUG(<< "Processed " << std::floor(100.0 * progress) << "%");
                    reportProgress += 0.1;
                }
                for (std::size_t i = 0; i < 900; ++i) {
                    rng.generateUniformSamples(0, generators.size(), 1, generator);
                    TOptionalDouble value{generators[generator[0]](time)};
                    if (value) {
                        dataRows["time"] = core::CStringUtils::typeToString(time);
                        dataRows["foo"] = core::CStringUtils::typeToString(*value);
                        dataRows["bar"] = "b" + core::CStringUtils::typeToString(i + 1);
                        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    }
                }
            }
            core_t::TTime startOfBucket{
                maths::CIntegerTools::floor(time, testParam.s_BucketLength)};
            auto used = limits.resourceMonitor().createMemoryUsageReport(startOfBucket);
            LOG_DEBUG(<< "# by = " << used.s_ByFields);
            LOG_DEBUG(<< "# partition = " << used.s_PartitionFields);
            LOG_DEBUG(<< "Memory status = " << used.s_MemoryStatus);
            LOG_DEBUG(<< "Memory usage bytes = " << used.s_Usage);
            LOG_DEBUG(<< "Memory limit bytes = " << memoryLimit * 1024 * 1024);
            BOOST_TEST_REQUIRE(used.s_ByFields > testParam.s_ExpectedByFields);
            BOOST_TEST_REQUIRE(used.s_ByFields < 800);
            BOOST_REQUIRE_EQUAL(std::size_t(2), used.s_PartitionFields);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                memoryLimit * 1024 * 1024 / 2, used.s_Usage,
                memoryLimit * 1024 * 1024 / testParam.s_ExpectedByMemoryUsageRelativeErrorDivisor);
        }

        LOG_DEBUG(<< "**** Test partition with bucketLength = " << testParam.s_BucketLength
                  << " ****");
        {
            std::size_t memoryLimit{10 /*MB*/};
            model::CLimits limits;
            limits.resourceMonitor().memoryLimit(memoryLimit);
            api::CFieldConfig fieldConfig;
            api::CFieldConfig::TStrVec clauses{"mean(foo)", "partitionfield=bar"};
            fieldConfig.initFromClause(clauses);
            model::CAnomalyDetectorModelConfig modelConfig =
                model::CAnomalyDetectorModelConfig::defaultConfig(testParam.s_BucketLength);
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                                 wrappedOutputStream, nullptr);

            core_t::TTime startTime{1495110323};
            core_t::TTime endTime{1495260323};
            core_t::TTime time{startTime};
            double reportProgress{0.0};
            for (/**/; time < endTime; time += testParam.s_BucketLength) {
                double progress{static_cast<double>(time - startTime) /
                                static_cast<double>(endTime - startTime)};
                if (progress >= reportProgress) {
                    LOG_DEBUG(<< "Processed " << std::floor(100.0 * progress) << "%");
                    reportProgress += 0.1;
                }
                for (std::size_t i = 0; i < 500; ++i) {
                    rng.generateUniformSamples(0, generators.size(), 1, generator);
                    TOptionalDouble value{generators[generator[0]](time)};
                    if (value) {
                        dataRows["time"] = core::CStringUtils::typeToString(time);
                        dataRows["foo"] = core::CStringUtils::typeToString(*value);
                        dataRows["bar"] = "b" + core::CStringUtils::typeToString(i + 1);
                        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    }
                }
            }
            core_t::TTime startOfBucket{
                maths::CIntegerTools::floor(time, testParam.s_BucketLength)};
            auto used = limits.resourceMonitor().createMemoryUsageReport(startOfBucket);
            LOG_DEBUG(<< "# by = " << used.s_ByFields);
            LOG_DEBUG(<< "# partition = " << used.s_PartitionFields);
            LOG_DEBUG(<< "Memory status = " << used.s_MemoryStatus);
            LOG_DEBUG(<< "Memory usage = " << used.s_Usage);
            BOOST_TEST_REQUIRE(used.s_PartitionFields > testParam.s_ExpectedPartitionFields);
            BOOST_TEST_REQUIRE(used.s_PartitionFields < 450);
            BOOST_TEST_REQUIRE(static_cast<double>(used.s_ByFields) >
                               0.96 * static_cast<double>(used.s_PartitionFields));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                memoryLimit * 1024 * 1024 / 2, used.s_Usage,
                memoryLimit * 1024 * 1024 / testParam.s_ExpectedPartitionUsageRelativeErrorDivisor);
        }

        LOG_DEBUG(<< "**** Test over with bucketLength = " << testParam.s_BucketLength
                  << " ****");
        {
            std::size_t memoryLimit{5 /*MB*/};
            model::CLimits limits;
            limits.resourceMonitor().memoryLimit(memoryLimit);
            api::CFieldConfig fieldConfig;
            api::CFieldConfig::TStrVec clauses{"mean(foo)", "over", "bar"};
            fieldConfig.initFromClause(clauses);
            model::CAnomalyDetectorModelConfig modelConfig =
                model::CAnomalyDetectorModelConfig::defaultConfig(testParam.s_BucketLength);
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig,
                                 wrappedOutputStream, nullptr);

            core_t::TTime startTime{1495110323};
            core_t::TTime endTime{1495230323};
            core_t::TTime time{startTime};
            double reportProgress{0.0};
            for (/**/; time < endTime; time += testParam.s_BucketLength) {
                double progress{static_cast<double>(time - startTime) /
                                static_cast<double>(endTime - startTime)};
                if (progress >= reportProgress) {
                    LOG_DEBUG(<< "Processed " << std::floor(100.0 * progress) << "%");
                    reportProgress += 0.1;
                }
                for (std::size_t i = 0; i < 9000; ++i) {
                    TOptionalDouble value{sparse(time)};
                    if (value) {
                        dataRows["time"] = core::CStringUtils::typeToString(time);
                        dataRows["foo"] = core::CStringUtils::typeToString(*value);
                        dataRows["bar"] = "b" + core::CStringUtils::typeToString(i + 1);
                        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    }
                }
            }
            core_t::TTime startOfBucket{
                maths::CIntegerTools::floor(time, testParam.s_BucketLength)};
            auto used = limits.resourceMonitor().createMemoryUsageReport(startOfBucket);
            LOG_DEBUG(<< "# over = " << used.s_OverFields);
            LOG_DEBUG(<< "Memory status = " << used.s_MemoryStatus);
            LOG_DEBUG(<< "Memory usage = " << used.s_Usage);
            BOOST_TEST_REQUIRE(used.s_OverFields > testParam.s_ExpectedOverFields);
            BOOST_TEST_REQUIRE(used.s_OverFields < 7000);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                memoryLimit * 1024 * 1024 / 2, used.s_Usage,
                memoryLimit * 1024 * 1024 / testParam.s_ExpectedOverUsageRelativeErrorDivisor);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
