/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CModelTypesTest.h"
#include <model/ModelTypes.h>

using namespace ml;
using namespace model;

void CModelTypesTest::testAll() {
    {
        // test print categories
        CPPUNIT_ASSERT_EQUAL(std::string("'counting'"), model_t::print(model_t::E_Counting));
        CPPUNIT_ASSERT_EQUAL(std::string("'event rate'"),
                             model_t::print(model_t::E_EventRate));
        CPPUNIT_ASSERT_EQUAL(std::string("'metric'"), model_t::print(model_t::E_Metric));
    }

    {
        // test functions for each feature
        core_t::TTime bucketStartTime = 10000;
        core_t::TTime bucketLength = 100;
        core_t::TTime time = 0;

        // Individual event rate features
        model_t::EFeature feature = model_t::E_IndividualCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualNonZeroCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'non-zero count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualTotalBucketCountByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'bucket count by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualIndicatorOfBucketPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'indicator per bucket of person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowCountsByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low values of count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighCountsByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high values of count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualArrivalTimesByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'mean arrival time by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLongArrivalTimesByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'long mean arrival time by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualShortArrivalTimesByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'short mean arrival time by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowNonZeroCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low non-zero count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighNonZeroCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high non-zero count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualUniqueCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'unique count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowUniqueCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low unique count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighUniqueCountByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high unique count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualInfoContentByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'information content of value per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowInfoContentByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low information content of value "
                                         "per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighInfoContentByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high information content of value "
                                         "per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualTimeOfDayByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'time-of-day per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualTimeOfWeekByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'time-of-week per bucket by person'"),
                             model_t::print(feature));

        // Individual metric features

        feature = model_t::E_IndividualMeanByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'arithmetic mean value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMedianByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("median"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'median value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMinByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("min"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'minimum value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMaxByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'maximum value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualSumByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'bucket sum by person'"), model_t::print(feature));

        feature = model_t::E_IndividualLowMeanByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low mean value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighMeanByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high mean value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowSumByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low bucket sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighSumByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high bucket sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualNonNullSumByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'bucket non-null sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowNonNullSumByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low bucket non-null sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighNonNullSumByBucketAndPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high bucket non-null sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMeanLatLongByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("lat_long"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'mean lat/long by person'"),
                             model_t::print(feature));

        // Population event rate features

        feature = model_t::E_PopulationAttributeTotalCountByPerson;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("'attribute counts by person'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationCountByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'non-zero count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationIndicatorOfBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("rare"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'indicator per bucket of person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationUniquePersonCountByAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("'unique person count by attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationUniqueCountByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'unique count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowCountsByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low values of non-zero count per "
                                         "bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighCountsByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high values of non-zero count per "
                                         "bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationInfoContentByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'information content of value per "
                                         "bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowInfoContentByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low information content of value "
                                         "per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighInfoContentByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high information content of value "
                                         "per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowUniqueCountByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low unique count per bucket by "
                                         "person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighUniqueCountByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high unique count per bucket by "
                                         "person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'time-of-day per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'time-of-week per bucket by person and attribute'"),
                             model_t::print(feature));

        // Population metric features

        feature = model_t::E_PopulationMeanByPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'mean value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMedianByPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("median"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'median value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMinByPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("min"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'minimum value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMaxByPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("max"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'maximum value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationSumByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'bucket sum by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowMeanByPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low mean by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighMeanByPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high mean by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowSumByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'low bucket sum by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighSumByBucketPersonAndAttribute;
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), model_t::dimension(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isCategorical(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isDiurnal(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isConstant(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMeanFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMedianFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMinFeature(feature));
        CPPUNIT_ASSERT_EQUAL(false, model_t::isMaxFeature(feature));
        CPPUNIT_ASSERT_EQUAL(true, model_t::isSumFeature(feature));
        CPPUNIT_ASSERT_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        CPPUNIT_ASSERT_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        CPPUNIT_ASSERT_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        CPPUNIT_ASSERT_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        CPPUNIT_ASSERT_EQUAL(std::string("'high bucket sum by person and attribute'"),
                             model_t::print(feature));
    }
}

CppUnit::Test* CModelTypesTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CModelTypesTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CModelTypesTest>(
        "CModelTypesTest::testAll", &CModelTypesTest::testAll));

    return suiteOfTests;
}
