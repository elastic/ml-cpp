/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <model/FunctionTypes.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>

BOOST_AUTO_TEST_SUITE(CFieldConfigTest)

BOOST_AUTO_TEST_CASE(testTrivial) {
    ml::api::CFieldConfig config("count", "mlcategory");

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    BOOST_TEST_REQUIRE(iter != fields.end());
    BOOST_TEST_REQUIRE(iter->fieldName().empty());
    BOOST_REQUIRE_EQUAL(std::string("mlcategory"), iter->byFieldName());
    BOOST_TEST_REQUIRE(iter->overFieldName().empty());
    BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
    BOOST_REQUIRE_EQUAL(false, iter->useNull());
    BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
    BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));

    const ml::api::CFieldConfig::TStrSet& superset = config.fieldNameSuperset();
    BOOST_REQUIRE_EQUAL(size_t(1), superset.size());
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("mlcategory"));
}

BOOST_AUTO_TEST_CASE(testValid) {
    this->testValidFile(std::bind(&ml::api::CFieldConfig::initFromFile,
                                  std::placeholders::_1, std::placeholders::_2),
                        "testfiles/new_mlfields.conf");
}

BOOST_AUTO_TEST_CASE(testInvalid) {
    this->testInvalidFile(std::bind(&ml::api::CFieldConfig::initFromFile,
                                    std::placeholders::_1, std::placeholders::_2),
                          "testfiles/new_invalidmlfields.conf");
}

BOOST_AUTO_TEST_CASE(testValidSummaryCountFieldName) {
    this->testValidSummaryCountFieldNameFile(
        std::bind(&ml::api::CFieldConfig::initFromFile, std::placeholders::_1,
                  std::placeholders::_2),
        "testfiles/new_mlfields_summarycount.conf");
}

BOOST_AUTO_TEST_CASE(testValidClauses) {
    ml::api::CFieldConfig config;

    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("metric(ResponseTime)");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("Airline");
        clause.push_back("influencerfield=nationality");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());
        BOOST_REQUIRE_EQUAL(size_t(1), config.influencerFieldNames().size());
        BOOST_REQUIRE_EQUAL(std::string("nationality"),
                            config.influencerFieldNames().front());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("influencerfield=nationality");
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("Airline");
        clause.push_back("influencerfield=MarketCap");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());
        BOOST_REQUIRE_EQUAL(size_t(2), config.influencerFieldNames().size());
        BOOST_REQUIRE_EQUAL(std::string("MarketCap"),
                            config.influencerFieldNames().front());
        BOOST_REQUIRE_EQUAL(std::string("nationality"),
                            config.influencerFieldNames().back());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("min(ResponseTime),");
        clause.push_back("count");
        clause.push_back("By");
        clause.push_back("Airline");
        clause.push_back("partitionfield=host");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("min"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("min"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count,sum(ResponseTime)");
        clause.push_back("By");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back(",max(ResponseTime)");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("partitionField=host");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back("ResponseTime");
        clause.push_back("bY");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("ResponseTime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("low_count");
        clause.push_back("high_count");
        clause.push_back("bY");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("low_c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("low_count"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("high_c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("high_count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("avg(avg_responsetime)");
        clause.push_back("max(max_responsetime)");
        clause.push_back("median(median_responsetime)");
        clause.push_back("bY");
        clause.push_back("Airline");
        clause.push_back("summarycountfield=mycount");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_REQUIRE_EQUAL(std::string("mycount"), config.summaryCountFieldName());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(3), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("avg_responsetime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("avg"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("avg"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("max_responsetime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("median_responsetime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualMetricMedian,
                            iter->function());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("median"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("median"), iter->verboseFunctionName());
    }
}

BOOST_AUTO_TEST_CASE(testInvalidClauses) {
    ml::api::CFieldConfig config;

    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("by");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime()");
        clause.push_back("BY");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("Over");
        clause.push_back("By");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("BY");
        clause.push_back("ResponseTime");
        clause.push_back("over");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("OVER");
        clause.push_back("ResponseTime");
        clause.push_back("by");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("over");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("by");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime,");
        clause.push_back("By");
        clause.push_back("count");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count,ResponseTime");
        clause.push_back("By");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back(",ResponseTime");
        clause.push_back("count");
        clause.push_back("by");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("bY");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("metrc(ResponseTime)");
        clause.push_back("BY");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("partitionfield=Airline");

        // Invalid because the "by" field is the same as the partition field
        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("low_count(wrong)");
        clause.push_back("by");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("metric(responsetime)");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("summarycountfield=mycount");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("responsetime");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("summarycountfield=mycount");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
}

BOOST_AUTO_TEST_CASE(testFieldOptions) {
    {
        ml::api::CFieldConfig::CFieldOptions opt("count", 42);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount, opt.function());
        BOOST_REQUIRE_EQUAL(std::string("count"), opt.fieldName());
        BOOST_REQUIRE_EQUAL(42, opt.configKey());
        BOOST_TEST_REQUIRE(opt.byFieldName().empty());
        BOOST_TEST_REQUIRE(opt.overFieldName().empty());
        BOOST_TEST_REQUIRE(opt.partitionFieldName().empty());
        BOOST_TEST_REQUIRE(opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(opt.function()));
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, true, true, "c", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "byField",
                                                 "overField", "partitionField",
                                                 false, false, true);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationCount, opt.function());
        BOOST_TEST_REQUIRE(opt.fieldName().empty());
        BOOST_REQUIRE_EQUAL(1, opt.configKey());
        BOOST_REQUIRE_EQUAL(std::string("byField"), opt.byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("overField"), opt.overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("partitionField"), opt.partitionFieldName());
        BOOST_TEST_REQUIRE(opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, false, false, "count()", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 3, "", "",
                                                 "", false, false, false);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualRareCount, opt.function());
        BOOST_TEST_REQUIRE(opt.fieldName().empty());
        BOOST_REQUIRE_EQUAL(3, opt.configKey());
        BOOST_TEST_REQUIRE(opt.byFieldName().empty());
        BOOST_TEST_REQUIRE(opt.overFieldName().empty());
        BOOST_TEST_REQUIRE(opt.partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), opt.verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::CFieldOptions opt("bytes", 4);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_IndividualMetric, opt.function());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), opt.fieldName());
        BOOST_REQUIRE_EQUAL(4, opt.configKey());
        BOOST_TEST_REQUIRE(opt.byFieldName().empty());
        BOOST_TEST_REQUIRE(opt.overFieldName().empty());
        BOOST_TEST_REQUIRE(opt.partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, true, false, "dc(category)", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(
            function, fieldName, 5, "", "overField", "", false, false, false);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationDistinctCount,
                            opt.function());
        BOOST_TEST_REQUIRE(opt.byFieldName().empty());
        BOOST_REQUIRE_EQUAL(5, opt.configKey());
        BOOST_REQUIRE_EQUAL(std::string("category"), opt.fieldName());
        BOOST_REQUIRE_EQUAL(std::string("overField"), opt.overFieldName());
        BOOST_TEST_REQUIRE(opt.partitionFieldName().empty());
        BOOST_TEST_REQUIRE(!opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("dc"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("distinct_count"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, true, false, "info_content(mlsub)", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 6, "",
                                                 "mlhrd", "", false, false, false);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationInfoContent, opt.function());
        BOOST_TEST_REQUIRE(opt.byFieldName().empty());
        BOOST_REQUIRE_EQUAL(6, opt.configKey());
        BOOST_REQUIRE_EQUAL(std::string("mlsub"), opt.fieldName());
        BOOST_REQUIRE_EQUAL(std::string("mlhrd"), opt.overFieldName());
        BOOST_TEST_REQUIRE(opt.partitionFieldName().empty());
        BOOST_TEST_REQUIRE(!opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("info_content"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("info_content"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, true, false, "high_info_content(mlsub)", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "", "mlhrd",
                                                 "datacenter", false, false, false);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationHighInfoContent,
                            opt.function());
        BOOST_TEST_REQUIRE(opt.byFieldName().empty());
        BOOST_REQUIRE_EQUAL(1, opt.configKey());
        BOOST_REQUIRE_EQUAL(std::string("mlsub"), opt.fieldName());
        BOOST_REQUIRE_EQUAL(std::string("mlhrd"), opt.overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("datacenter"), opt.partitionFieldName());
        BOOST_TEST_REQUIRE(!opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("high_info_content"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("high_info_content"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, true, true, "rare()", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "byField",
                                                 "overField", "", false, false, false);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationRare, opt.function());
        BOOST_TEST_REQUIRE(opt.fieldName().empty());
        BOOST_REQUIRE_EQUAL(1, opt.configKey());
        BOOST_REQUIRE_EQUAL(std::string("byField"), opt.byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("overField"), opt.overFieldName());
        BOOST_TEST_REQUIRE(opt.partitionFieldName().empty());
        BOOST_TEST_REQUIRE(!opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("rare"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("rare"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        BOOST_TEST_REQUIRE(ml::api::CFieldConfig::parseFieldString(
            false, true, true, "rare_count", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "byField",
                                                 "overField", "partitionField",
                                                 false, false, true);

        BOOST_REQUIRE_EQUAL(ml::model::function_t::E_PopulationRareCount, opt.function());
        BOOST_TEST_REQUIRE(opt.fieldName().empty());
        BOOST_REQUIRE_EQUAL(1, opt.configKey());
        BOOST_REQUIRE_EQUAL(std::string("byField"), opt.byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("overField"), opt.overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("partitionField"), opt.partitionFieldName());
        BOOST_TEST_REQUIRE(opt.useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        BOOST_REQUIRE_EQUAL(std::string("rare_count"), opt.terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("rare_count"), opt.verboseFunctionName());
    }
}

BOOST_AUTO_TEST_CASE(testValidPopulationClauses) {
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back("OVER");
        clause.push_back("Airline");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("Airline"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("c");
        clause.push_back("over");
        clause.push_back("SRC");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("SRC"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("c"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("high_dc(DPT)");
        clause.push_back("over");
        clause.push_back("SRC");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("DPT"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("SRC"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("high_dc"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("high_distinct_count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("info_content(mlsub)");
        clause.push_back("over");
        clause.push_back("mlhrd");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("mlsub"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("mlhrd"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("info_content"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("info_content"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("rare");
        clause.push_back("fr");
        clause.push_back("By");
        clause.push_back("uri_path");
        clause.push_back("Over");
        clause.push_back("clientip");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("rare"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("rare"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("fr"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("freq_rare"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("bytes");
        clause.push_back("Over");
        clause.push_back("pid");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("pid"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("sum(bytes)");
        clause.push_back("Over");
        clause.push_back("pid");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("pid"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("fr");
        clause.push_back("min(bytes)");
        clause.push_back("by");
        clause.push_back("uri_path");
        clause.push_back("over");
        clause.push_back("clientip");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("fr"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("freq_rare"), iter->verboseFunctionName());
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("min"), iter->terseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("min"), iter->verboseFunctionName());
    }
}

BOOST_AUTO_TEST_CASE(testValidPopulation) {
    this->testValidPopulationFile(std::bind(&ml::api::CFieldConfig::initFromFile,
                                            std::placeholders::_1, std::placeholders::_2),
                                  "testfiles/new_populationmlfields.conf");
}

BOOST_AUTO_TEST_CASE(testDefaultCategorizationField) {
    this->testDefaultCategorizationFieldFile(
        std::bind(&ml::api::CFieldConfig::initFromFile, std::placeholders::_1,
                  std::placeholders::_2),
        "testfiles/new_mlfields_sos_message_cat.conf");
}

BOOST_AUTO_TEST_CASE(testCategorizationFieldWithFilters) {
    std::string fileName("testfiles/new_mlfields_categorization_filters.conf");

    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(config.initFromFile(fileName));
    BOOST_TEST_REQUIRE(!config.havePartitionFields());
    BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

    LOG_DEBUG(<< config.debug());

    const std::string& categorizationFieldName = config.categorizationFieldName();
    BOOST_REQUIRE_EQUAL(std::string("message"), categorizationFieldName);
    const ml::api::CFieldConfig::TStrVec& filters = config.categorizationFilters();
    BOOST_TEST_REQUIRE(filters.empty() == false);
    BOOST_REQUIRE_EQUAL(std::size_t(2), filters.size());
    BOOST_REQUIRE_EQUAL(std::string("foo"), config.categorizationFilters()[0]);
    BOOST_REQUIRE_EQUAL(std::string(" "), config.categorizationFilters()[1]);
}

BOOST_AUTO_TEST_CASE(testExcludeFrequentClauses) {
    {
        // Basic case with no excludefrequent
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("partitionfield=host");
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");
        clause.push_back("usenull=true");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_TRACE(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
    }
    {
        // "by" excludefrequent
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("PartitionField=host");
        clause.push_back("excludeFrequent=by");
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");
        clause.push_back("Usenull=true");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_TRACE(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
    }
    {
        // "over" excludefrequent
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("partitionfield=host");
        clause.push_back("excludefrequent=OVER");
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");
        clause.push_back("usenull=true");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_TRACE(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Over, iter->excludeFrequent());
    }
    {
        // "by" and "over" excludefrequent
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("partitionfield=host");
        clause.push_back("excludefrequent=All");
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");
        clause.push_back("usenull=true");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_TRACE(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("uri_path"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Both, iter->excludeFrequent());
    }
    {
        // Invalid partition excludefrequent
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("partitionfield=host");
        clause.push_back("excludefrequent=partition");
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");
        clause.push_back("usenull=true");

        BOOST_TEST_REQUIRE(!config.initFromClause(clause));
    }
    {
        // "by" excludefrequent with no "by" field
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("partitionfield=host");
        clause.push_back("excludefrequent=By");
        clause.push_back("max(bytes)");
        clause.push_back("OVER");
        clause.push_back("clientip");
        clause.push_back("usenull=true");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_TRACE(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->overFieldName());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
    }
    {
        // "over" excludefrequent with no "over" field
        ml::api::CFieldConfig config;
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("partitionfield=host");
        clause.push_back("excludefrequent=over");
        clause.push_back("max(bytes)");
        clause.push_back("by");
        clause.push_back("clientip");
        clause.push_back("usenull=true");

        BOOST_TEST_REQUIRE(config.initFromClause(clause));
        BOOST_TEST_REQUIRE(config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_TRACE(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("clientip"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        BOOST_REQUIRE_EQUAL(std::string("max"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
    }
}

BOOST_AUTO_TEST_CASE(testExcludeFrequent) {
    this->testExcludeFrequentFile(std::bind(&ml::api::CFieldConfig::initFromFile,
                                            std::placeholders::_1, std::placeholders::_2),
                                  "testfiles/new_mlfields_excludefrequent.conf");
}

BOOST_AUTO_TEST_CASE(testSlashes) {
    this->testSlashesFile(std::bind(&ml::api::CFieldConfig::initFromFile,
                                    std::placeholders::_1, std::placeholders::_2),
                          "testfiles/new_mlfields_slashes.conf");
}

BOOST_AUTO_TEST_CASE(testBracketPercent) {
    this->testBracketPercentFile(std::bind(&ml::api::CFieldConfig::initFromFile,
                                           std::placeholders::_1, std::placeholders::_2),
                                 "testfiles/new_mlfields_bracket_percent.conf");
}

BOOST_AUTO_TEST_CASE(testClauseTokenise) {
    ml::api::CFieldConfig config;

    {
        std::string clause;
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_TEST_REQUIRE(tokens.empty());
    }
    {
        std::string clause("responsetime by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(3), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("responsetime"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"responsetime\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(3), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("responsetime"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"funny field\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(3), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("funny field"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"field with escaped \\\" quotes\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(3), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("field with escaped \" quotes"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"field with nested , comma\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(3), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("field with nested , comma"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"field with escaped escape\\\\\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(3), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("field with escaped escape\\"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("one,two,three  by  airline");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(5), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("one"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("two"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("three"), tokens[2]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[3]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("one, two ,three by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(5), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("one"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("two"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("three"), tokens[2]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[3]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("one\t two ,\tthree by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(5), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("one"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("two"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("three"), tokens[2]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[3]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("\"one,\",\",two \"\t\" three,\" by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(5), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("one,"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string(",two "), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string(" three,"), tokens[2]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[3]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("responsetime by airline partitionfield=host");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(4), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("responsetime"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
        BOOST_REQUIRE_EQUAL(std::string("partitionfield=host"), tokens[3]);
    }
    {
        std::string clause("responsetime by airline partitionfield=\"funny field\"");
        ml::api::CFieldConfig::TStrVec tokens;

        BOOST_TEST_REQUIRE(config.tokenise(clause, tokens));

        BOOST_REQUIRE_EQUAL(size_t(4), tokens.size());
        BOOST_REQUIRE_EQUAL(std::string("responsetime"), tokens[0]);
        BOOST_REQUIRE_EQUAL(std::string("by"), tokens[1]);
        BOOST_REQUIRE_EQUAL(std::string("airline"), tokens[2]);
        BOOST_REQUIRE_EQUAL(std::string("partitionfield=funny field"), tokens[3]);
    }
}

BOOST_AUTO_TEST_CASE(testUtf8Bom) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_mlfields_with_utf8_bom.conf"));
}

BOOST_AUTO_TEST_CASE(testAddByOverPartitionInfluencers) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_mlfields_excludefrequent.conf"));

    BOOST_TEST_REQUIRE(config.influencerFieldNames().empty());

    config.addInfluencerFieldsFromByOverPartitionFields();

    ml::api::CFieldConfig::TStrVec copyInfluencers(config.influencerFieldNames());
    std::sort(copyInfluencers.begin(), copyInfluencers.end());

    BOOST_REQUIRE_EQUAL(size_t(6), copyInfluencers.size());
    BOOST_REQUIRE_EQUAL(std::string("airline"), copyInfluencers[0]);
    BOOST_REQUIRE_EQUAL(std::string("client"), copyInfluencers[1]);
    BOOST_REQUIRE_EQUAL(std::string("dest_ip"), copyInfluencers[2]);
    BOOST_REQUIRE_EQUAL(std::string("host"), copyInfluencers[3]);
    BOOST_REQUIRE_EQUAL(std::string("process"), copyInfluencers[4]);
    BOOST_REQUIRE_EQUAL(std::string("src_ip"), copyInfluencers[5]);
}

BOOST_AUTO_TEST_CASE(testAddOptions) {
    ml::api::CFieldConfig configFromFile;
    ml::api::CFieldConfig configFromScratch;

    BOOST_TEST_REQUIRE(configFromFile.initFromFile("testfiles/new_populationmlfields.conf"));

    ml::api::CFieldConfig::CFieldOptions options1("count", 1, "SRC", false, false);
    BOOST_TEST_REQUIRE(configFromScratch.addOptions(options1));

    ml::api::CFieldConfig::CFieldOptions options2(ml::model::function_t::E_PopulationCount,
                                                  "", 2, "DPT", "SRC", "",
                                                  false, false, true);
    BOOST_TEST_REQUIRE(configFromScratch.addOptions(options2));

    BOOST_REQUIRE_EQUAL(configFromFile.debug(), configFromScratch.debug());
}

BOOST_AUTO_TEST_CASE(testValidFileTInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(initFunc(&config, fileName));
    BOOST_TEST_REQUIRE(!config.havePartitionFields());
    BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());
    BOOST_TEST_REQUIRE(config.categorizationFilters().empty());

    LOG_DEBUG(<< config.debug());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    BOOST_REQUIRE_EQUAL(size_t(7), fields.size());

    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    {
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("mlcategory"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("remote_ip"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("remote_user"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("request"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("response"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("referrer"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("agent"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }

    const ml::api::CFieldConfig::TStrSet& superset = config.fieldNameSuperset();
    BOOST_REQUIRE_EQUAL(size_t(8), superset.size());
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("agent"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("bytes"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("mlcategory"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("referrer"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("remote_ip"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("remote_user"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("request"));
    BOOST_REQUIRE_EQUAL(size_t(1), superset.count("response"));
}

BOOST_AUTO_TEST_CASE(testInvalidFileTInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(!initFunc(&config, fileName));
}

BOOST_AUTO_TEST_CASE(testValidSummaryCountFieldNameFileTInitFromFileFunc initFunc,
                     const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(initFunc(&config, fileName));
    BOOST_TEST_REQUIRE(!config.havePartitionFields());
    BOOST_REQUIRE_EQUAL(std::string("count"), config.summaryCountFieldName());
}

BOOST_AUTO_TEST_CASE(testValidPopulationFileTInitFromFileFunc initFunc,
                     const std::string& fileName) {
    {
        ml::api::CFieldConfig config;
        BOOST_TEST_REQUIRE(initFunc(&config, fileName));
        BOOST_TEST_REQUIRE(!config.havePartitionFields());
        BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

        LOG_DEBUG(<< config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        BOOST_REQUIRE_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("SRC"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(false, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        ++iter;
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("DPT"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("SRC"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        BOOST_REQUIRE_EQUAL(true, iter->useNull());
        BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        BOOST_REQUIRE_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
    }
}

BOOST_AUTO_TEST_CASE(testDefaultCategorizationFieldFileTInitFromFileFunc initFunc,
                     const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(initFunc(&config, fileName));
    BOOST_TEST_REQUIRE(!config.havePartitionFields());
    BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

    LOG_DEBUG(<< config.debug());

    const std::string& categorizationFieldName = config.categorizationFieldName();
    BOOST_REQUIRE_EQUAL(std::string("message"), categorizationFieldName);
    BOOST_TEST_REQUIRE(config.categorizationFilters().empty());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    BOOST_REQUIRE_EQUAL(size_t(1), fields.size());
    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    BOOST_TEST_REQUIRE(iter != fields.end());
    BOOST_TEST_REQUIRE(iter->fieldName().empty());
    BOOST_REQUIRE_EQUAL(std::string("mlcategory"), iter->byFieldName());
    BOOST_TEST_REQUIRE(iter->overFieldName().empty());
    BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
    BOOST_REQUIRE_EQUAL(false, iter->useNull());
    BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
    BOOST_REQUIRE_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
}

BOOST_AUTO_TEST_CASE(testExcludeFrequentFileTInitFromFileFunc initFunc,
                     const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(initFunc(&config, fileName));
    BOOST_TEST_REQUIRE(config.havePartitionFields());
    BOOST_TEST_REQUIRE(config.summaryCountFieldName().empty());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    BOOST_REQUIRE_EQUAL(size_t(8), fields.size());
    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();

    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Both, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("dest_ip"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("src_ip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("metric"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("responsetime"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("airline"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("dest_ip"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("src_ip"), iter->byFieldName());
        BOOST_REQUIRE_EQUAL(std::string("dest_ip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_REQUIRE_EQUAL(std::string("src_ip"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_Over, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("sum"), iter->verboseFunctionName());
        BOOST_REQUIRE_EQUAL(std::string("bytes"), iter->fieldName());
        BOOST_TEST_REQUIRE(iter->byFieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("dest_ip"), iter->overFieldName());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("rare"), iter->verboseFunctionName());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("process"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
    {
        BOOST_TEST_REQUIRE(iter != fields.end());
        BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
        BOOST_REQUIRE_EQUAL(std::string("rare"), iter->verboseFunctionName());
        BOOST_TEST_REQUIRE(iter->fieldName().empty());
        BOOST_REQUIRE_EQUAL(std::string("client"), iter->byFieldName());
        BOOST_TEST_REQUIRE(iter->overFieldName().empty());
        BOOST_TEST_REQUIRE(iter->partitionFieldName().empty());
        iter++;
    }
}

BOOST_AUTO_TEST_CASE(testSlashesFileTInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(initFunc(&config, fileName));

    LOG_DEBUG(<< config.debug());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();

    for (ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
         iter != fields.end(); ++iter) {
        BOOST_REQUIRE_EQUAL(std::string("host"), iter->partitionFieldName());
    }
}

BOOST_AUTO_TEST_CASE(testBracketPercentFileTInitFromFileFunc initFunc,
                     const std::string& fileName) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(initFunc(&config, fileName));

    LOG_DEBUG(<< config.debug());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();

    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    BOOST_TEST_REQUIRE(iter != fields.end());
    BOOST_REQUIRE_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
    BOOST_REQUIRE_EQUAL(std::string("max"), iter->terseFunctionName());
    BOOST_REQUIRE_EQUAL(std::string("Level 1 (Urgent)"), iter->fieldName());
    BOOST_REQUIRE_EQUAL(std::string("10%"), iter->byFieldName());
    BOOST_REQUIRE_EQUAL(std::string("%10"), iter->overFieldName());
    BOOST_REQUIRE_EQUAL(std::string("Percentage (%)"), iter->partitionFieldName());
    BOOST_REQUIRE_EQUAL(std::string("This string should have quotes removed"),
                        config.categorizationFieldName());
}

BOOST_AUTO_TEST_CASE(testScheduledEvents) {
    ml::api::CFieldConfig config;

    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/scheduled_events.conf"));

    ml::api::CFieldConfig::TStrDetectionRulePrVec events = config.scheduledEvents();
    BOOST_REQUIRE_EQUAL(std::size_t{2}, events.size());
    BOOST_REQUIRE_EQUAL(std::string("May Bank Holiday"), events[0].first);
    BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1525132800.000000 AND TIME < 1525219200.000000"),
                        events[0].second.print());
    BOOST_REQUIRE_EQUAL(std::string("New Years Day"), events[1].first);
    BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1514764800.000000 AND TIME < 1514851200.000000"),
                        events[1].second.print());
}

BOOST_AUTO_TEST_SUITE_END()
