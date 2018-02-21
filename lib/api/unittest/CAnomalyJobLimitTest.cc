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
#include "CAnomalyJobLimitTest.h"
#include "CMockDataProcessor.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/CHierarchicalResultsWriter.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include <boost/tuple/tuple.hpp>

#include <set>
#include <sstream>
#include <string>
#include <fstream>

using namespace ml;

std::set<std::string> getUniqueValues(const std::string &key, const std::string &output)
{
    std::set<std::string> values;
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(output);
    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT(doc.IsArray());

    size_t i = 0;

    while(true)
    {
        rapidjson::Value *p1 = rapidjson::Pointer("/" + std::to_string(i)).Get(doc);
        if (p1 != nullptr)
        {
            size_t j = 0;
            while(true)
            {
                rapidjson::Value *p2 = rapidjson::Pointer("/" + std::to_string(i)
                    + "/records/" + std::to_string(j)).Get(doc);
                if (p2 != nullptr)
                {
                    size_t k = 0;
                    while (true)
                    {
                        rapidjson::Value *p3 = rapidjson::Pointer("/" + std::to_string(i)
                            + "/records/" + std::to_string(j)
                            + "/causes/" + std::to_string(k)
                            + "/" + key).Get(doc);

                        if (p3 != nullptr)
                        {
                            values.insert(p3->GetString());
                        }
                        else
                        {
                            break;
                        }
                        ++k;
                    }
                }
                else
                {
                    break;
                }
                ++j;
            }
        }
        else
        {
            break;
        }
        ++i;
    }

    return values;
}

CppUnit::Test* CAnomalyJobLimitTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CAnomalyJobLimitTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyJobLimitTest>(
                                   "CAnomalyJobLimitTest::testLimit",
                                   &CAnomalyJobLimitTest::testLimit) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAnomalyJobLimitTest>(
                                   "CAnomalyJobLimitTest::testAccuracy",
                                   &CAnomalyJobLimitTest::testAccuracy) );
    return suiteOfTests;
}

void CAnomalyJobLimitTest::testAccuracy(void)
{
    // Check that the amount of memory used when we go over the
    // resource limit is close enough to the limit that we specified

    std::size_t nonLimitedUsage{0};

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

        CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        model::CLimits limits;
        //limits.resourceMonitor().m_ByteLimitHigh = 100000;
        //limits.resourceMonitor().m_ByteLimitLow = 90000;

        {
            LOG_TRACE("Setting up job");
            api::CAnomalyJob job("job",
                                   limits,
                                   fieldConfig,
                                   modelConfig,
                                   wrappedOutputStream);

            std::ifstream inputStrm("testfiles/resource_accuracy.csv");
            CPPUNIT_ASSERT(inputStrm.is_open());
            api::CCsvInputParser parser(inputStrm);

            LOG_TRACE("Reading file");
            CPPUNIT_ASSERT(parser.readStream(boost::bind(&api::CAnomalyJob::handleRecord,
                                                         &job,
                                                         _1)));

            LOG_TRACE("Checking results");

            CPPUNIT_ASSERT_EQUAL(uint64_t(18630), job.numRecordsHandled());
        }

        nonLimitedUsage = limits.resourceMonitor().totalMemory();
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

        CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);
        model::CLimits limits;

        std::stringstream outputStrm;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

            limits.resourceMonitor().m_ByteLimitHigh = nonLimitedUsage / 10;
            limits.resourceMonitor().m_ByteLimitLow =
                limits.resourceMonitor().m_ByteLimitHigh - 1024;

            LOG_TRACE("Setting up job");
            api::CAnomalyJob job("job",
                                   limits,
                                   fieldConfig,
                                   modelConfig,
                                   wrappedOutputStream);

            std::ifstream inputStrm("testfiles/resource_accuracy.csv");
            CPPUNIT_ASSERT(inputStrm.is_open());
            api::CCsvInputParser parser(inputStrm);

            LOG_TRACE("Reading file");
            CPPUNIT_ASSERT(parser.readStream(boost::bind(&api::CAnomalyJob::handleRecord,
                                                         &job,
                                                         _1)));

            LOG_TRACE("Checking results");

            CPPUNIT_ASSERT_EQUAL(uint64_t(18630), job.numRecordsHandled());
        }
        LOG_TRACE(outputStrm.str());

        // TODO this limit must be tightened once there is more granular control
        // over the model memory creation
        std::size_t limitedUsage = limits.resourceMonitor().totalMemory();
        LOG_DEBUG("Non-limited usage: " << nonLimitedUsage << "; limited: " << limitedUsage);
        CPPUNIT_ASSERT(limitedUsage < nonLimitedUsage);
    }
}

void CAnomalyJobLimitTest::testLimit(void)
{
    typedef std::set<std::string> TStrSet;

    std::stringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);
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

        CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);

        LOG_TRACE("Setting up job");
        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream);

        std::ifstream inputStrm("testfiles/resource_limits_3_2over_3partition.csv");
        CPPUNIT_ASSERT(inputStrm.is_open());
        api::CCsvInputParser parser(inputStrm);

        LOG_TRACE("Reading file");
        CPPUNIT_ASSERT(parser.readStream(boost::bind(&api::CAnomalyJob::handleRecord,
                                                     &job,
                                                     _1)));
        LOG_TRACE("Checking results");
        CPPUNIT_ASSERT_EQUAL(uint64_t(1176), job.numRecordsHandled());
    }

    std::string out = outputStrm.str();

    TStrSet partitions = getUniqueValues("partition_field_value", out);
    TStrSet people = getUniqueValues("over_field_value", out);
    TStrSet attributes = getUniqueValues("by_field_value", out);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), partitions.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), people.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), attributes.size());

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

        CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(3600);

        //::CMockOutputWriter resultsHandler;
        core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        LOG_TRACE("Setting up job");
        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream);

        std::ifstream inputStrm("testfiles/resource_limits_3_2over_3partition_first8.csv");
        CPPUNIT_ASSERT(inputStrm.is_open());
        api::CCsvInputParser parser(inputStrm);

        LOG_TRACE("Reading file");
        CPPUNIT_ASSERT(parser.readStream(boost::bind(&api::CAnomalyJob::handleRecord,
                                                     &job,
                                                     _1)));
        // Now turn on the resource limiting
        limits.resourceMonitor().m_ByteLimitHigh = 0;
        limits.resourceMonitor().m_ByteLimitLow = 0;
        limits.resourceMonitor().m_AllowAllocations = false;

        std::ifstream inputStrm2("testfiles/resource_limits_3_2over_3partition_last1169.csv");
        CPPUNIT_ASSERT(inputStrm2.is_open());
        api::CCsvInputParser parser2(inputStrm2);

        LOG_TRACE("Reading second file");
        CPPUNIT_ASSERT(parser2.readStream(boost::bind(&api::CAnomalyJob::handleRecord,
                                                     &job,
                                                     _1)));
        LOG_TRACE("Checking results");
        CPPUNIT_ASSERT_EQUAL(uint64_t(1180), job.numRecordsHandled());
    }

    out = outputStrm.str();

    partitions = getUniqueValues("partition_field_value", out);
    people = getUniqueValues("over_field_value", out);
    attributes = getUniqueValues("by_field_value", out);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), partitions.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), people.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), attributes.size());
}




