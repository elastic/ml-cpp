/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/ModelTypes.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CModelTypesTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testAll) {
    {
        // test print categories
        BOOST_CHECK_EQUAL(std::string("'counting'"), model_t::print(model_t::E_Counting));
        BOOST_CHECK_EQUAL(std::string("'event rate'"),
                             model_t::print(model_t::E_EventRate));
        BOOST_CHECK_EQUAL(std::string("'metric'"), model_t::print(model_t::E_Metric));
    }

    {
        // test functions for each feature
        core_t::TTime bucketStartTime = 10000;
        core_t::TTime bucketLength = 100;
        core_t::TTime time = 0;

        // Individual event rate features
        model_t::EFeature feature = model_t::E_IndividualCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualNonZeroCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'non-zero count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualTotalBucketCountByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(true, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'bucket count by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualIndicatorOfBucketPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(true, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("rare"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'indicator per bucket of person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowCountsByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low values of count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighCountsByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high values of count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualArrivalTimesByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'mean arrival time by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLongArrivalTimesByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'long mean arrival time by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualShortArrivalTimesByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'short mean arrival time by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowNonZeroCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low non-zero count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighNonZeroCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high non-zero count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualUniqueCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'unique count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowUniqueCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low unique count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighUniqueCountByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high unique count per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualInfoContentByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'information content of value per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowInfoContentByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low information content of value per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighInfoContentByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high information content of value per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualTimeOfDayByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(true, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'time-of-day per bucket by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualTimeOfWeekByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(true, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'time-of-week per bucket by person'"),
                             model_t::print(feature));

        // Individual metric features

        feature = model_t::E_IndividualMeanByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'arithmetic mean value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMedianByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("median"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'median value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMinByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("min"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'minimum value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMaxByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("max"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'maximum value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualSumByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'bucket sum by person'"), model_t::print(feature));

        feature = model_t::E_IndividualLowMeanByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low mean value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighMeanByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high mean value by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowSumByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low bucket sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighSumByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high bucket sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualNonNullSumByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'bucket non-null sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualLowNonNullSumByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low bucket non-null sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualHighNonNullSumByBucketAndPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high bucket non-null sum by person'"),
                             model_t::print(feature));

        feature = model_t::E_IndividualMeanLatLongByPerson;
        BOOST_CHECK_EQUAL(std::size_t(2), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("lat_long"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'mean lat/long by person'"),
                             model_t::print(feature));

        // Population event rate features

        feature = model_t::E_PopulationAttributeTotalCountByPerson;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(true, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("'attribute counts by person'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationCountByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'non-zero count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationIndicatorOfBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(true, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("rare"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'indicator per bucket of person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationUniquePersonCountByAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(true, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("'unique person count by attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationUniqueCountByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'unique count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowCountsByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low values of non-zero count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighCountsByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("count"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high values of non-zero count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationInfoContentByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'information content of value per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowInfoContentByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low information content of value per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighInfoContentByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("info_content"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high information content of value per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowUniqueCountByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low unique count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighUniqueCountByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("distinct_count"),
                             model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high unique count per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(true, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'time-of-day per bucket by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(true, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(1.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(3.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("time"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'time-of-week per bucket by person and attribute'"),
                             model_t::print(feature));

        // Population metric features

        feature = model_t::E_PopulationMeanByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'mean value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMedianByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("median"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'median value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMinByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("min"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'minimum value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMaxByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("max"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'maximum value by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationSumByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'bucket sum by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowMeanByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low mean by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighMeanByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("mean"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high mean by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationLowSumByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedBelow,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'low bucket sum by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationHighSumByBucketPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(1), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(1.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_OneSidedAbove,
                             model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("sum"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'high bucket sum by person and attribute'"),
                             model_t::print(feature));

        feature = model_t::E_PopulationMeanLatLongByPersonAndAttribute;
        BOOST_CHECK_EQUAL(std::size_t(2), model_t::dimension(feature));
        BOOST_CHECK_EQUAL(false, model_t::isCategorical(feature));
        BOOST_CHECK_EQUAL(false, model_t::isDiurnal(feature));
        BOOST_CHECK_EQUAL(false, model_t::isConstant(feature));
        BOOST_CHECK_EQUAL(true, model_t::isMeanFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMedianFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMinFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isMaxFeature(feature));
        BOOST_CHECK_EQUAL(false, model_t::isSumFeature(feature));
        BOOST_CHECK_EQUAL(true, model_t::isLatLong(feature));
        BOOST_CHECK_EQUAL(2.0, model_t::varianceScale(feature, 4.0, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::offsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(2.0, model_t::inverseOffsetCountToZero(feature, 2.0));
        BOOST_CHECK_EQUAL(maths_t::E_TwoSided, model_t::probabilityCalculation(feature));
        BOOST_CHECK_EQUAL(
            core_t::TTime(10050),
            model_t::sampleTime(feature, bucketStartTime, bucketLength, time));
        BOOST_CHECK_EQUAL(std::string("lat_long"), model_t::outputFunctionName(feature));
        BOOST_CHECK_EQUAL(std::string("'mean lat/long by person and attribute'"),
                             model_t::print(feature));
    }
}


BOOST_AUTO_TEST_SUITE_END()
