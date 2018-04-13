/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CFunctionTypesTest.h"

#include <core/CLogger.h>
#include <core/CContainerPrinter.h>

#include <model/FunctionTypes.h>

using namespace ml;
using namespace model;

void CFunctionTypesTest::testFeaturesToFunction()
{
    model_t::TFeatureVec features;

    {
        // Count.
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("count"), function_t::name(function_t::function(features)));
    }
    {
        // (Rare) Count.
        features.clear();
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("count"), function_t::name(function_t::function(features)));
    }
    {
        // Non-Zero Count.
        features.clear();
        features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("non_zero_count"), function_t::name(function_t::function(features)));
    }
    {
        // Non-Zero Rare Count.
        features.clear();
        features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
        features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("rare_non_zero_count"), function_t::name(function_t::function(features)));
    }
    {
        // Low Count.
        features.clear();
        features.push_back(model_t::E_IndividualLowCountsByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("low_count"), function_t::name(function_t::function(features)));
    }
    {
        // High Count.
        features.clear();
        features.push_back(model_t::E_IndividualHighCountsByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("high_count"), function_t::name(function_t::function(features)));
    }
    {
        // Rare Count.
        features.clear();
        features.push_back(model_t::E_IndividualIndicatorOfBucketPerson);
        features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), function_t::name(function_t::function(features)));
    }
    {
        // Min.
        features.clear();
        features.push_back(model_t::E_IndividualMinByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("min"), function_t::name(function_t::function(features)));
    }
    {
        // Mean.
        features.clear();
        features.push_back(model_t::E_IndividualMeanByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), function_t::name(function_t::function(features)));
        features.clear();
        features.push_back(model_t::E_IndividualLowMeanByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("low_mean"), function_t::name(function_t::function(features)));
        features.clear();
        features.push_back(model_t::E_IndividualHighMeanByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"), function_t::name(function_t::function(features)));
    }
    {
        // Median.
        features.clear();
        features.push_back(model_t::E_IndividualMedianByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("median"), function_t::name(function_t::function(features)));
        features.clear();
    }
    {
        // Max.
        features.clear();
        features.push_back(model_t::E_IndividualMaxByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("max"), function_t::name(function_t::function(features)));
    }
    {
        // Sum.
        features.clear();
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), function_t::name(function_t::function(features)));
        features.clear();
        features.push_back(model_t::E_IndividualLowSumByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("low_sum"), function_t::name(function_t::function(features)));
        features.clear();
        features.push_back(model_t::E_IndividualHighSumByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("high_sum"), function_t::name(function_t::function(features)));
    }
    {
        // Non-Zero Sum.
        features.clear();
        features.push_back(model_t::E_IndividualNonNullSumByBucketAndPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("non_null_sum"), function_t::name(function_t::function(features)));
    }
    {
        // Metric.
        features.clear();
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), function_t::name(function_t::function(features)));
    }
    {
        // Metric.
        features.clear();
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), function_t::name(function_t::function(features)));
    }
    {
        // Lat-long.
        features.clear();
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("lat_long"), function_t::name(function_t::function(features)));
    }
    {
        // Count.
        features.clear();
        features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
        features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("rare_count"), function_t::name(function_t::function(features)));
    }
    {
        // Low Count.
        features.clear();
        features.push_back(model_t::E_PopulationLowCountsByBucketPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("low_count"), function_t::name(function_t::function(features)));
    }
    {
        // High Count.
        features.clear();
        features.push_back(model_t::E_PopulationHighCountsByBucketPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("high_count"), function_t::name(function_t::function(features)));
    }
    {
        // Distinct count.
        features.clear();
        features.push_back(model_t::E_PopulationUniqueCountByBucketPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"), function_t::name(function_t::function(features)));
    }
    {
        // Min.
        features.clear();
        features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("min"), function_t::name(function_t::function(features)));
    }
    {
        // Mean.
        features.clear();
        features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), function_t::name(function_t::function(features)));
    }
    {
        // Median.
        features.clear();
        features.push_back(model_t::E_PopulationMedianByPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("median"), function_t::name(function_t::function(features)));
    }
    {
        // Max.
        features.clear();
        features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("max"), function_t::name(function_t::function(features)));
    }
    {
        // Sum.
        features.clear();
        features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), function_t::name(function_t::function(features)));
    }
    {
        // Metric.
        features.clear();
        features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
        features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), function_t::name(function_t::function(features)));
    }
    {
        // Metric.
        features.clear();
        features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
        features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
        features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
        LOG_DEBUG("function = '" << function_t::name(function_t::function(features)) << "'");
        CPPUNIT_ASSERT_EQUAL(std::string("metric"), function_t::name(function_t::function(features)));
    }
}

CppUnit::Test* CFunctionTypesTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CFunctionTypesTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CFunctionTypesTest>(
                                   "CFunctionTypesTest::testFeaturesToFunction",
                                   &CFunctionTypesTest::testFeaturesToFunction) );

    return suiteOfTests;
}
