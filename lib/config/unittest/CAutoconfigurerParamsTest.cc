/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CAutoconfigurerParamsTest.h"

#include <core/CLogger.h>

#include <config/CAutoconfigurerParams.h>

using namespace ml;

void CAutoconfigurerParamsTest::testDefaults()
{
    LOG_DEBUG("");
    LOG_DEBUG("+-------------------------------------------+");
    LOG_DEBUG("|  CAutoconfigurerParamsTest::testDefaults  |");
    LOG_DEBUG("+-------------------------------------------+");

    config::CAutoconfigurerParams params("time", "", false, false);
    std::string actual = params.print();
    std::string expected = "  TimeFieldName = time\n"
                           "  TimeFieldFormat = \n"
                           "  FieldsOfInterest = \"null\"\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::ARGUMENT_INDEX] = \"null\"\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::BY_INDEX] = \"null\"\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::OVER_INDEX] = \"null\"\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::PARTITION_INDEX] = \"null\"\n"
                           "  FunctionCategoriesToConfigure = [count, rare, distinct_count, info_content, mean, min, max, sum, varp, median]\n"
                           "  FieldDataType = []\n"
                           "  MinimumExamplesToClassify = 1000\n"
                           "  NumberOfMostFrequentFieldsCounts = 10\n"
                           "  MinimumRecordsToAttemptConfig = 10000\n"
                           "  HighNumberByFieldValues = 500\n"
                           "  MaximumNumberByFieldValues = 1000\n"
                           "  HighNumberRareByFieldValues = 50000\n"
                           "  MaximumNumberRareByFieldValues = 500000\n"
                           "  HighNumberPartitionFieldValues = 500000\n"
                           "  MaximumNumberPartitionFieldValues = 5000000\n"
                           "  LowNumberOverFieldValues = 50\n"
                           "  MinimumNumberOverFieldValues = 5\n"
                           "  HighCardinalityInTailFactor = 1.100000\n"
                           "  HighCardinalityInTailIncrement = 10\n"
                           "  HighCardinalityHighTailFraction = 0.005000\n"
                           "  HighCardinalityMaximumTailFraction = 0.050000\n"
                           "  LowPopulatedBucketFractions = [0.3333333, 0.02]\n"
                           "  MinimumPopulatedBucketFractions = [0.02, 0.002]\n"
                           "  HighPopulatedBucketFractions[1] = 0.100000\n"
                           "  MaximumPopulatedBucketFractions[1] = 0.500000\n"
                           "  CandidateBucketLengths = [60, 300, 600, 1800, 3600, 7200, 14400, 86400]\n"
                           "  LowNumberOfBucketsForConfig = 500.000000\n"
                           "  MinimumNumberOfBucketsForConfig = 50.000000\n"
                           "  PolledDataMinimumMassAtInterval = 0.990000\n"
                           "  PolledDataJitter = 0.010000\n"
                           "  LowCoefficientOfVariation = 0.001000\n"
                           "  MinimumCoefficientOfVariation = 0.000001\n"
                           "  LowLengthRangeForInfoContent = 10.000000\n"
                           "  MinimumLengthRangeForInfoContent = 1.000000\n"
                           "  LowMaximumLengthForInfoContent = 25.000000\n"
                           "  MinimumMaximumLengthForInfoContent = 5.000000\n"
                           "  LowEntropyForInfoContent = 0.010000\n"
                           "  MinimumEntropyForInfoContent = 0.000001\n"
                           "  LowDistinctCountForInfoContent = 500000.000000\n"
                           "  MinimumDistinctCountForInfoContent = 5000.000000\n";
    LOG_DEBUG("parameters =\n" << actual);
    CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void CAutoconfigurerParamsTest::testInit()
{
    LOG_DEBUG("");
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CAutoconfigurerParamsTest::testInit  |");
    LOG_DEBUG("+---------------------------------------+");

    config::CAutoconfigurerParams params("time", "", false, false);

    params.init("testfiles/parameters.conf");
    std::string actual = params.print();
    std::string expected = "  TimeFieldName = time\n"
                           "  TimeFieldFormat = \n"
                           "  FieldsOfInterest = [performance_metric, performance_metric_name, machine, user, region, program]\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::ARGUMENT_INDEX] = \"null\"\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::BY_INDEX] = [performance_metric_name, program]\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::OVER_INDEX] = [user]\n"
                           "  FieldsToUseInAutoconfigureByRole[constants::PARTITION_INDEX] = [user, machine, region]\n"
                           "  FunctionCategoriesToConfigure = [count, min, max, median]\n"
                           "  FieldDataType = [(machine,categorical), (performance_metric,numeric)]\n"
                           "  MinimumExamplesToClassify = 50\n"
                           "  NumberOfMostFrequentFieldsCounts = 20\n"
                           "  MinimumRecordsToAttemptConfig = 200\n"
                           "  HighNumberByFieldValues = 50\n"
                           "  MaximumNumberByFieldValues = 5000\n"
                           "  HighNumberRareByFieldValues = 10000\n"
                           "  MaximumNumberRareByFieldValues = 100000\n"
                           "  HighNumberPartitionFieldValues = 1000\n"
                           "  MaximumNumberPartitionFieldValues = 100000\n"
                           "  LowNumberOverFieldValues = 80\n"
                           "  MinimumNumberOverFieldValues = 20\n"
                           "  HighCardinalityInTailFactor = 1.030000\n"
                           "  HighCardinalityInTailIncrement = 2\n"
                           "  HighCardinalityHighTailFraction = 0.310000\n"
                           "  HighCardinalityMaximumTailFraction = 0.620000\n"
                           "  LowPopulatedBucketFractions = [0.35, 0.12]\n"
                           "  MinimumPopulatedBucketFractions = [0.11, 0.042]\n"
                           "  HighPopulatedBucketFractions[1] = 0.100000\n"
                           "  MaximumPopulatedBucketFractions[1] = 0.500000\n"
                           "  CandidateBucketLengths = [1, 60, 600, 1800, 7200]\n"
                           "  LowNumberOfBucketsForConfig = 30.000000\n"
                           "  MinimumNumberOfBucketsForConfig = 8.000000\n"
                           "  PolledDataMinimumMassAtInterval = 0.890000\n"
                           "  PolledDataJitter = 0.030000\n"
                           "  LowCoefficientOfVariation = 0.003000\n"
                           "  MinimumCoefficientOfVariation = 0.000200\n"
                           "  LowLengthRangeForInfoContent = 10.000000\n"
                           "  MinimumLengthRangeForInfoContent = 1.000000\n"
                           "  LowMaximumLengthForInfoContent = 25.000000\n"
                           "  MinimumMaximumLengthForInfoContent = 5.000000\n"
                           "  LowEntropyForInfoContent = 0.010000\n"
                           "  MinimumEntropyForInfoContent = 0.000001\n"
                           "  LowDistinctCountForInfoContent = 500000.000000\n"
                           "  MinimumDistinctCountForInfoContent = 5000.000000\n";
    LOG_DEBUG("parameters =\n" << actual);
    CPPUNIT_ASSERT_EQUAL(expected, actual);

    params.init("testfiles/badparameters.conf");
    actual = params.print();
    LOG_DEBUG("parameters =\n" << actual);
    CPPUNIT_ASSERT_EQUAL(expected, actual);
}

CppUnit::Test *CAutoconfigurerParamsTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CAutoconfigurerParamsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CAutoconfigurerParamsTest>(
                                   "CAutoconfigurerParamsTest::testDefaults",
                                   &CAutoconfigurerParamsTest::testDefaults) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CAutoconfigurerParamsTest>(
                                   "CAutoconfigurerParamsTest::testInit",
                                   &CAutoconfigurerParamsTest::testInit) );

    return suiteOfTests;
}
