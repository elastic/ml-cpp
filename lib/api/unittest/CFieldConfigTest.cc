/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CFieldConfigTest.h"

#include <core/CLogger.h>

#include <model/FunctionTypes.h>

#include <boost/bind.hpp>

#include <algorithm>

CppUnit::Test* CFieldConfigTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CFieldConfigTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testTrivial", &CFieldConfigTest::testTrivial));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testValid", &CFieldConfigTest::testValid));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testInvalid", &CFieldConfigTest::testInvalid));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testValidSummaryCountFieldName",
                                                                    &CFieldConfigTest::testValidSummaryCountFieldName));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testValidClauses", &CFieldConfigTest::testValidClauses));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testInvalidClauses", &CFieldConfigTest::testInvalidClauses));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testFieldOptions", &CFieldConfigTest::testFieldOptions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testValidPopulationClauses",
                                                                    &CFieldConfigTest::testValidPopulationClauses));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testValidPopulation", &CFieldConfigTest::testValidPopulation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testDefaultCategorizationField",
                                                                    &CFieldConfigTest::testDefaultCategorizationField));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testCategorizationFieldWithFilters",
                                                                    &CFieldConfigTest::testCategorizationFieldWithFilters));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testExcludeFrequentClauses",
                                                                    &CFieldConfigTest::testExcludeFrequentClauses));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testExcludeFrequent", &CFieldConfigTest::testExcludeFrequent));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testSlashes", &CFieldConfigTest::testSlashes));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testBracketPercent", &CFieldConfigTest::testBracketPercent));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testClauseTokenise", &CFieldConfigTest::testClauseTokenise));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testUtf8Bom", &CFieldConfigTest::testUtf8Bom));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testAddByOverPartitionInfluencers",
                                                                    &CFieldConfigTest::testAddByOverPartitionInfluencers));
    suiteOfTests->addTest(new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testAddOptions", &CFieldConfigTest::testAddOptions));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CFieldConfigTest>("CFieldConfigTest::testScheduledEvents", &CFieldConfigTest::testScheduledEvents));
    return suiteOfTests;
}

void CFieldConfigTest::testTrivial() {
    ml::api::CFieldConfig config("count", "mlcategory");

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    CPPUNIT_ASSERT(iter != fields.end());
    CPPUNIT_ASSERT(iter->fieldName().empty());
    CPPUNIT_ASSERT_EQUAL(std::string("mlcategory"), iter->byFieldName());
    CPPUNIT_ASSERT(iter->overFieldName().empty());
    CPPUNIT_ASSERT(iter->partitionFieldName().empty());
    CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
    CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
    CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));

    const ml::api::CFieldConfig::TStrSet& superset = config.fieldNameSuperset();
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.size());
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("mlcategory"));
}

void CFieldConfigTest::testValid() {
    this->testValidFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2), "testfiles/new_mlfields.conf");
}

void CFieldConfigTest::testInvalid() {
    this->testInvalidFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2), "testfiles/new_invalidmlfields.conf");
}

void CFieldConfigTest::testValidSummaryCountFieldName() {
    this->testValidSummaryCountFieldNameFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2),
                                             "testfiles/new_mlfields_summarycount.conf");
}

void CFieldConfigTest::testValidClauses() {
    ml::api::CFieldConfig config;

    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("metric(ResponseTime)");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("Airline");
        clause.push_back("influencerfield=nationality");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(size_t(1), config.influencerFieldNames().size());
        CPPUNIT_ASSERT_EQUAL(std::string("nationality"), config.influencerFieldNames().front());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("influencerfield=nationality");
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("Airline");
        clause.push_back("influencerfield=MarketCap");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(size_t(2), config.influencerFieldNames().size());
        CPPUNIT_ASSERT_EQUAL(std::string("MarketCap"), config.influencerFieldNames().front());
        CPPUNIT_ASSERT_EQUAL(std::string("nationality"), config.influencerFieldNames().back());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("min(ResponseTime),");
        clause.push_back("count");
        clause.push_back("By");
        clause.push_back("Airline");
        clause.push_back("partitionfield=host");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("min"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("min"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count,sum(ResponseTime)");
        clause.push_back("By");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back(",max(ResponseTime)");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("partitionField=host");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back("ResponseTime");
        clause.push_back("bY");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("ResponseTime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("low_count");
        clause.push_back("high_count");
        clause.push_back("bY");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("low_c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("low_count"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("high_count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("avg(avg_responsetime)");
        clause.push_back("max(max_responsetime)");
        clause.push_back("median(median_responsetime)");
        clause.push_back("bY");
        clause.push_back("Airline");
        clause.push_back("summarycountfield=mycount");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT_EQUAL(std::string("mycount"), config.summaryCountFieldName());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(3), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("avg_responsetime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("avg"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("avg"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("max_responsetime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("median_responsetime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_IndividualMetricMedian, iter->function());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("median"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("median"), iter->verboseFunctionName());
    }
}

void CFieldConfigTest::testInvalidClauses() {
    ml::api::CFieldConfig config;

    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("by");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime()");
        clause.push_back("BY");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("Over");
        clause.push_back("By");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("BY");
        clause.push_back("ResponseTime");
        clause.push_back("over");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("OVER");
        clause.push_back("ResponseTime");
        clause.push_back("by");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("over");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime");
        clause.push_back("BY");
        clause.push_back("by");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("ResponseTime,");
        clause.push_back("By");
        clause.push_back("count");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count,ResponseTime");
        clause.push_back("By");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back(",ResponseTime");
        clause.push_back("count");
        clause.push_back("by");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("bY");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("metrc(ResponseTime)");
        clause.push_back("BY");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("partitionfield=Airline");

        // Invalid because the "by" field is the same as the partition field
        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("low_count(wrong)");
        clause.push_back("by");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("metric(responsetime)");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("summarycountfield=mycount");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
    {
        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("responsetime");
        clause.push_back("by");
        clause.push_back("Airline");
        clause.push_back("summarycountfield=mycount");

        CPPUNIT_ASSERT(!config.initFromClause(clause));
    }
}

void CFieldConfigTest::testFieldOptions() {
    {
        ml::api::CFieldConfig::CFieldOptions opt("count", 42);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_IndividualRareCount, opt.function());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), opt.fieldName());
        CPPUNIT_ASSERT_EQUAL(42, opt.configKey());
        CPPUNIT_ASSERT(opt.byFieldName().empty());
        CPPUNIT_ASSERT(opt.overFieldName().empty());
        CPPUNIT_ASSERT(opt.partitionFieldName().empty());
        CPPUNIT_ASSERT(opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(opt.function()));
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, true, true, "c", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "byField", "overField", "partitionField", false, false, true);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_PopulationCount, opt.function());
        CPPUNIT_ASSERT(opt.fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(1, opt.configKey());
        CPPUNIT_ASSERT_EQUAL(std::string("byField"), opt.byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("overField"), opt.overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("partitionField"), opt.partitionFieldName());
        CPPUNIT_ASSERT(opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, false, false, "count()", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 3, "", "", "", false, false, false);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_IndividualRareCount, opt.function());
        CPPUNIT_ASSERT(opt.fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(3, opt.configKey());
        CPPUNIT_ASSERT(opt.byFieldName().empty());
        CPPUNIT_ASSERT(opt.overFieldName().empty());
        CPPUNIT_ASSERT(opt.partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), opt.verboseFunctionName());
    }
    {
        ml::api::CFieldConfig::CFieldOptions opt("bytes", 4);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_IndividualMetric, opt.function());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), opt.fieldName());
        CPPUNIT_ASSERT_EQUAL(4, opt.configKey());
        CPPUNIT_ASSERT(opt.byFieldName().empty());
        CPPUNIT_ASSERT(opt.overFieldName().empty());
        CPPUNIT_ASSERT(opt.partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, true, false, "dc(category)", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 5, "", "overField", "", false, false, false);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_PopulationDistinctCount, opt.function());
        CPPUNIT_ASSERT(opt.byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(5, opt.configKey());
        CPPUNIT_ASSERT_EQUAL(std::string("category"), opt.fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("overField"), opt.overFieldName());
        CPPUNIT_ASSERT(opt.partitionFieldName().empty());
        CPPUNIT_ASSERT(!opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("dc"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, true, false, "info_content(mlsub)", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 6, "", "mlhrd", "", false, false, false);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_PopulationInfoContent, opt.function());
        CPPUNIT_ASSERT(opt.byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(6, opt.configKey());
        CPPUNIT_ASSERT_EQUAL(std::string("mlsub"), opt.fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("mlhrd"), opt.overFieldName());
        CPPUNIT_ASSERT(opt.partitionFieldName().empty());
        CPPUNIT_ASSERT(!opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, true, false, "high_info_content(mlsub)", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "", "mlhrd", "datacenter", false, false, false);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_PopulationHighInfoContent, opt.function());
        CPPUNIT_ASSERT(opt.byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(1, opt.configKey());
        CPPUNIT_ASSERT_EQUAL(std::string("mlsub"), opt.fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("mlhrd"), opt.overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("datacenter"), opt.partitionFieldName());
        CPPUNIT_ASSERT(!opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_info_content"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("high_info_content"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, true, true, "rare()", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "byField", "overField", "", false, false, false);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_PopulationRare, opt.function());
        CPPUNIT_ASSERT(opt.fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(1, opt.configKey());
        CPPUNIT_ASSERT_EQUAL(std::string("byField"), opt.byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("overField"), opt.overFieldName());
        CPPUNIT_ASSERT(opt.partitionFieldName().empty());
        CPPUNIT_ASSERT(!opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), opt.verboseFunctionName());
    }
    {
        ml::model::function_t::EFunction function;
        std::string fieldName;
        CPPUNIT_ASSERT(ml::api::CFieldConfig::parseFieldString(false, true, true, "rare_count", function, fieldName));

        ml::api::CFieldConfig::CFieldOptions opt(function, fieldName, 1, "byField", "overField", "partitionField", false, false, true);

        CPPUNIT_ASSERT_EQUAL(ml::model::function_t::E_PopulationRareCount, opt.function());
        CPPUNIT_ASSERT(opt.fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(1, opt.configKey());
        CPPUNIT_ASSERT_EQUAL(std::string("byField"), opt.byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("overField"), opt.overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("partitionField"), opt.partitionFieldName());
        CPPUNIT_ASSERT(opt.useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(opt.function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(opt.function()));
        CPPUNIT_ASSERT_EQUAL(std::string("rare_count"), opt.terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("rare_count"), opt.verboseFunctionName());
    }
}

void CFieldConfigTest::testValidPopulationClauses() {
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("count");
        clause.push_back("OVER");
        clause.push_back("Airline");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("Airline"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("c");
        clause.push_back("over");
        clause.push_back("SRC");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("SRC"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("high_dc(DPT)");
        clause.push_back("over");
        clause.push_back("SRC");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("DPT"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("SRC"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_dc"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("high_distinct_count"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("info_content(mlsub)");
        clause.push_back("over");
        clause.push_back("mlhrd");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("mlsub"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("mlhrd"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"), iter->verboseFunctionName());
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("fr"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("freq_rare"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("bytes");
        clause.push_back("Over");
        clause.push_back("pid");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("pid"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("sum(bytes)");
        clause.push_back("Over");
        clause.push_back("pid");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("pid"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
    }
    {
        ml::api::CFieldConfig config;

        ml::api::CFieldConfig::TStrVec clause;
        clause.push_back("max(bytes)");
        clause.push_back("BY");
        clause.push_back("uri_path");
        clause.push_back("OVER");
        clause.push_back("clientip");

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("fr"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("freq_rare"), iter->verboseFunctionName());
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("min"), iter->terseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("min"), iter->verboseFunctionName());
    }
}

void CFieldConfigTest::testValidPopulation() {
    this->testValidPopulationFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2), "testfiles/new_populationmlfields.conf");
}

void CFieldConfigTest::testDefaultCategorizationField() {
    this->testDefaultCategorizationFieldFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2),
                                             "testfiles/new_mlfields_sos_message_cat.conf");
}

void CFieldConfigTest::testCategorizationFieldWithFilters() {
    std::string fileName("testfiles/new_mlfields_categorization_filters.conf");

    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(config.initFromFile(fileName));
    CPPUNIT_ASSERT(!config.havePartitionFields());
    CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

    LOG_DEBUG(config.debug());

    const std::string& categorizationFieldName = config.categorizationFieldName();
    CPPUNIT_ASSERT_EQUAL(std::string("message"), categorizationFieldName);
    const ml::api::CFieldConfig::TStrVec& filters = config.categorizationFilters();
    CPPUNIT_ASSERT(filters.empty() == false);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), filters.size());
    CPPUNIT_ASSERT_EQUAL(std::string("foo"), config.categorizationFilters()[0]);
    CPPUNIT_ASSERT_EQUAL(std::string(" "), config.categorizationFilters()[1]);
}

void CFieldConfigTest::testExcludeFrequentClauses() {
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_TRACE(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_TRACE(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_TRACE(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_Over, iter->excludeFrequent());
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_TRACE(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("uri_path"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_Both, iter->excludeFrequent());
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

        CPPUNIT_ASSERT(!config.initFromClause(clause));
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_TRACE(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->overFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
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

        CPPUNIT_ASSERT(config.initFromClause(clause));
        CPPUNIT_ASSERT(config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_TRACE(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("clientip"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
    }
}

void CFieldConfigTest::testExcludeFrequent() {
    this->testExcludeFrequentFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2), "testfiles/new_mlfields_excludefrequent.conf");
}

void CFieldConfigTest::testSlashes() {
    this->testSlashesFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2), "testfiles/new_mlfields_slashes.conf");
}

void CFieldConfigTest::testBracketPercent() {
    this->testBracketPercentFile(boost::bind(&ml::api::CFieldConfig::initFromFile, _1, _2), "testfiles/new_mlfields_bracket_percent.conf");
}

void CFieldConfigTest::testClauseTokenise() {
    ml::api::CFieldConfig config;

    {
        std::string clause;
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT(tokens.empty());
    }
    {
        std::string clause("responsetime by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(3), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"responsetime\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(3), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"funny field\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(3), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("funny field"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"field with escaped \\\" quotes\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(3), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("field with escaped \" quotes"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"field with nested , comma\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(3), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("field with nested , comma"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("\"field with escaped escape\\\\\" by \"airline\"");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(3), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("field with escaped escape\\"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
    }
    {
        std::string clause("one,two,three  by  airline");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(5), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("one"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("two"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("three"), tokens[2]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[3]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("one, two ,three by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(5), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("one"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("two"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("three"), tokens[2]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[3]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("one\t two ,\tthree by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(5), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("one"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("two"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("three"), tokens[2]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[3]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("\"one,\",\",two \"\t\" three,\" by airline");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(5), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("one,"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string(",two "), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string(" three,"), tokens[2]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[3]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[4]);
    }
    {
        std::string clause("responsetime by airline partitionfield=host");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(4), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
        CPPUNIT_ASSERT_EQUAL(std::string("partitionfield=host"), tokens[3]);
    }
    {
        std::string clause("responsetime by airline partitionfield=\"funny field\"");
        ml::api::CFieldConfig::TStrVec tokens;

        CPPUNIT_ASSERT(config.tokenise(clause, tokens));

        CPPUNIT_ASSERT_EQUAL(size_t(4), tokens.size());
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), tokens[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("by"), tokens[1]);
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), tokens[2]);
        CPPUNIT_ASSERT_EQUAL(std::string("partitionfield=funny field"), tokens[3]);
    }
}

void CFieldConfigTest::testUtf8Bom() {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_mlfields_with_utf8_bom.conf"));
}

void CFieldConfigTest::testAddByOverPartitionInfluencers() {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_mlfields_excludefrequent.conf"));

    CPPUNIT_ASSERT(config.influencerFieldNames().empty());

    config.addInfluencerFieldsFromByOverPartitionFields();

    ml::api::CFieldConfig::TStrVec copyInfluencers(config.influencerFieldNames());
    std::sort(copyInfluencers.begin(), copyInfluencers.end());

    CPPUNIT_ASSERT_EQUAL(size_t(6), copyInfluencers.size());
    CPPUNIT_ASSERT_EQUAL(std::string("airline"), copyInfluencers[0]);
    CPPUNIT_ASSERT_EQUAL(std::string("client"), copyInfluencers[1]);
    CPPUNIT_ASSERT_EQUAL(std::string("dest_ip"), copyInfluencers[2]);
    CPPUNIT_ASSERT_EQUAL(std::string("host"), copyInfluencers[3]);
    CPPUNIT_ASSERT_EQUAL(std::string("process"), copyInfluencers[4]);
    CPPUNIT_ASSERT_EQUAL(std::string("src_ip"), copyInfluencers[5]);
}

void CFieldConfigTest::testAddOptions() {
    ml::api::CFieldConfig configFromFile;
    ml::api::CFieldConfig configFromScratch;

    CPPUNIT_ASSERT(configFromFile.initFromFile("testfiles/new_populationmlfields.conf"));

    ml::api::CFieldConfig::CFieldOptions options1("count", 1, "SRC", false, false);
    CPPUNIT_ASSERT(configFromScratch.addOptions(options1));

    ml::api::CFieldConfig::CFieldOptions options2(ml::model::function_t::E_PopulationCount, "", 2, "DPT", "SRC", "", false, false, true);
    CPPUNIT_ASSERT(configFromScratch.addOptions(options2));

    CPPUNIT_ASSERT_EQUAL(configFromFile.debug(), configFromScratch.debug());
}

void CFieldConfigTest::testValidFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(initFunc(&config, fileName));
    CPPUNIT_ASSERT(!config.havePartitionFields());
    CPPUNIT_ASSERT(config.summaryCountFieldName().empty());
    CPPUNIT_ASSERT(config.categorizationFilters().empty());

    LOG_DEBUG(config.debug());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    CPPUNIT_ASSERT_EQUAL(size_t(7), fields.size());

    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    {
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("mlcategory"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("remote_ip"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("remote_user"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("request"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("response"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("referrer"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("agent"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        iter++;
    }

    const ml::api::CFieldConfig::TStrSet& superset = config.fieldNameSuperset();
    CPPUNIT_ASSERT_EQUAL(size_t(8), superset.size());
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("agent"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("bytes"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("mlcategory"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("referrer"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("remote_ip"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("remote_user"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("request"));
    CPPUNIT_ASSERT_EQUAL(size_t(1), superset.count("response"));
}

void CFieldConfigTest::testInvalidFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(!initFunc(&config, fileName));
}

void CFieldConfigTest::testValidSummaryCountFieldNameFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(initFunc(&config, fileName));
    CPPUNIT_ASSERT(!config.havePartitionFields());
    CPPUNIT_ASSERT_EQUAL(std::string("count"), config.summaryCountFieldName());
}

void CFieldConfigTest::testValidPopulationFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    {
        ml::api::CFieldConfig config;
        CPPUNIT_ASSERT(initFunc(&config, fileName));
        CPPUNIT_ASSERT(!config.havePartitionFields());
        CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

        LOG_DEBUG(config.debug());

        const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
        CPPUNIT_ASSERT_EQUAL(size_t(2), fields.size());
        ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("SRC"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
        ++iter;
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("DPT"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("SRC"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(true, iter->useNull());
        CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
        CPPUNIT_ASSERT_EQUAL(true, ml::model::function_t::isPopulation(iter->function()));
    }
}

void CFieldConfigTest::testDefaultCategorizationFieldFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(initFunc(&config, fileName));
    CPPUNIT_ASSERT(!config.havePartitionFields());
    CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

    LOG_DEBUG(config.debug());

    const std::string& categorizationFieldName = config.categorizationFieldName();
    CPPUNIT_ASSERT_EQUAL(std::string("message"), categorizationFieldName);
    CPPUNIT_ASSERT(config.categorizationFilters().empty());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    CPPUNIT_ASSERT_EQUAL(size_t(1), fields.size());
    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    CPPUNIT_ASSERT(iter != fields.end());
    CPPUNIT_ASSERT(iter->fieldName().empty());
    CPPUNIT_ASSERT_EQUAL(std::string("mlcategory"), iter->byFieldName());
    CPPUNIT_ASSERT(iter->overFieldName().empty());
    CPPUNIT_ASSERT(iter->partitionFieldName().empty());
    CPPUNIT_ASSERT_EQUAL(false, iter->useNull());
    CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isMetric(iter->function()));
    CPPUNIT_ASSERT_EQUAL(false, ml::model::function_t::isPopulation(iter->function()));
}

void CFieldConfigTest::testExcludeFrequentFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(initFunc(&config, fileName));
    CPPUNIT_ASSERT(config.havePartitionFields());
    CPPUNIT_ASSERT(config.summaryCountFieldName().empty());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();
    CPPUNIT_ASSERT_EQUAL(size_t(8), fields.size());
    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();

    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_Both, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("dest_ip"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("src_ip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("airline"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("dest_ip"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("src_ip"), iter->byFieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("dest_ip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT_EQUAL(std::string("src_ip"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_Over, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), iter->verboseFunctionName());
        CPPUNIT_ASSERT_EQUAL(std::string("bytes"), iter->fieldName());
        CPPUNIT_ASSERT(iter->byFieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("dest_ip"), iter->overFieldName());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_By, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), iter->verboseFunctionName());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("process"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
    {
        CPPUNIT_ASSERT(iter != fields.end());
        CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), iter->verboseFunctionName());
        CPPUNIT_ASSERT(iter->fieldName().empty());
        CPPUNIT_ASSERT_EQUAL(std::string("client"), iter->byFieldName());
        CPPUNIT_ASSERT(iter->overFieldName().empty());
        CPPUNIT_ASSERT(iter->partitionFieldName().empty());
        iter++;
    }
}

void CFieldConfigTest::testSlashesFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(initFunc(&config, fileName));

    LOG_DEBUG(config.debug());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();

    for (ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin(); iter != fields.end(); ++iter) {
        CPPUNIT_ASSERT_EQUAL(std::string("host"), iter->partitionFieldName());
    }
}

void CFieldConfigTest::testBracketPercentFile(TInitFromFileFunc initFunc, const std::string& fileName) {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(initFunc(&config, fileName));

    LOG_DEBUG(config.debug());

    const ml::api::CFieldConfig::TFieldOptionsMIndex& fields = config.fieldOptions();

    ml::api::CFieldConfig::TFieldOptionsMIndexCItr iter = fields.begin();
    CPPUNIT_ASSERT(iter != fields.end());
    CPPUNIT_ASSERT_EQUAL(ml::model_t::E_XF_None, iter->excludeFrequent());
    CPPUNIT_ASSERT_EQUAL(std::string("max"), iter->terseFunctionName());
    CPPUNIT_ASSERT_EQUAL(std::string("Level 1 (Urgent)"), iter->fieldName());
    CPPUNIT_ASSERT_EQUAL(std::string("10%"), iter->byFieldName());
    CPPUNIT_ASSERT_EQUAL(std::string("%10"), iter->overFieldName());
    CPPUNIT_ASSERT_EQUAL(std::string("Percentage (%)"), iter->partitionFieldName());
    CPPUNIT_ASSERT_EQUAL(std::string("This string should have quotes removed"), config.categorizationFieldName());
}

void CFieldConfigTest::testScheduledEvents() {
    ml::api::CFieldConfig config;

    CPPUNIT_ASSERT(config.initFromFile("testfiles/scheduled_events.conf"));

    ml::api::CFieldConfig::TStrDetectionRulePrVec events = config.scheduledEvents();
    CPPUNIT_ASSERT_EQUAL(std::size_t{2}, events.size());
    CPPUNIT_ASSERT_EQUAL(std::string("May Bank Holiday"), events[0].first);
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF TIME >= 1525132800.000000 AND TIME < 1525219200.000000"),
                         events[0].second.print());
    CPPUNIT_ASSERT_EQUAL(std::string("New Years Day"), events[1].first);
    CPPUNIT_ASSERT_EQUAL(std::string("FILTER_RESULTS AND SKIP_SAMPLING IF TIME >= 1514764800.000000 AND TIME < 1514851200.000000"),
                         events[1].second.print());
}
