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

#include <model/ModelTypes.h>

#include <core/Constants.h>

#include <maths/CMultivariatePrior.h>
#include <maths/CPrior.h>
#include <maths/CTimeSeriesDecomposition.h>

#include <model/CAnomalyDetector.h>
#include <model/CModelParams.h>
#include <model/CProbabilityAndInfluenceCalculator.h>

#include <boost/make_shared.hpp>

namespace ml {
namespace model_t {

namespace {

//! Compute x^4.
double pow4(double x) {
    double x2 = x * x;
    return x2 * x2;
}

}

std::string print(EModelType type) {
    switch (type) {
        case E_Counting:
            return "'counting'";
        case E_EventRateOnline:
            return "'online event rate'";
        case E_MetricOnline:
            return "'online metric'";
    }
    return "-";
}

MODEL_EXPORT
std::size_t dimension(EFeature feature) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return 1;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualSumVelocityByPerson:
            return 1;
        case E_IndividualMeanLatLongByPerson:
            return 2;

CASE_POPULATION_COUNT:
            return 1;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return 1;
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return 2;

CASE_PEERS_COUNT:
CASE_PEERS_METRIC:
            return 1;
    }

    return 1;
}

TDouble2Vec1Vec stripExtraStatistics(EFeature feature,
                                     const TDouble2Vec1Vec &values) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return values;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualSumVelocityByPerson:
        case E_IndividualMeanLatLongByPerson:
            return values;
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson: {
            TDouble2Vec1Vec result;
            result.reserve(values.size());
            for (const auto &value : values) {
                result.push_back(TDouble2Vec(value.begin(), value.begin() + value.size() / 2));
            }
            return result;
        }

CASE_POPULATION_COUNT:
        return values;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return values;
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute: {
            TDouble2Vec1Vec result;
            result.reserve(values.size());
            for (const auto &value : values) {
                result.push_back(TDouble2Vec(value.begin(), value.begin() + value.size() / 2));
            }
            return result;
        }

CASE_PEERS_COUNT:
CASE_PEERS_METRIC:
        return values;
    }

    return values;
}

bool isCategorical(EFeature feature) {
    switch (feature) {
        case E_IndividualTotalBucketCountByPerson:
            return true;
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return false;

CASE_INDIVIDUAL_METRIC:
            return false;

        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationUniquePersonCountByAttribute:
            return true;
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return false;

CASE_POPULATION_METRIC:
CASE_PEERS_COUNT:
CASE_PEERS_METRIC:
            return false;
    }

    return false;
}

bool isDiurnal(EFeature feature) {
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
            return false;
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return true;

CASE_INDIVIDUAL_METRIC:
            return false;

        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
        case E_PopulationUniquePersonCountByAttribute:
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
            return false;
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return true;

CASE_POPULATION_METRIC:
            return false;

        case E_PeersAttributeTotalCountByPerson:
        case E_PeersCountByBucketPersonAndAttribute:
        case E_PeersUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowCountsByBucketPersonAndAttribute:
        case E_PeersHighCountsByBucketPersonAndAttribute:
        case E_PeersInfoContentByBucketPersonAndAttribute:
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
            return false;
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            return true;

CASE_PEERS_METRIC:
            return false;
    }

    return false;
}

bool isLatLong(EFeature feature) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return false;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualSumVelocityByPerson:
            return false;
        case E_IndividualMeanLatLongByPerson:
            return true;

CASE_POPULATION_COUNT:
            return false;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return false;
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return true;

CASE_PEERS_COUNT:
CASE_PEERS_METRIC:
            return false;
    }

    return false;
}

bool isConstant(EFeature feature) {
    switch (feature) {
        case E_IndividualIndicatorOfBucketPerson:
            return true;
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return false;

CASE_INDIVIDUAL_METRIC:
            return false;

        case E_PopulationIndicatorOfBucketPersonAndAttribute:
            return true;
        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationUniquePersonCountByAttribute:
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return false;

CASE_POPULATION_METRIC:
CASE_PEERS_COUNT:
CASE_PEERS_METRIC:
            return false;
    }

    return false;
}

bool isMeanFeature(EFeature feature) {
    EMetricCategory category;
    return    metricCategory(feature, category) &&
              (category == E_Mean || category == E_MultivariateMean);
}

bool isMedianFeature(EFeature feature) {
    EMetricCategory category;
    return    metricCategory(feature, category) &&
              (category == E_Median);
}

bool isMinFeature(EFeature feature) {
    EMetricCategory category;
    return    metricCategory(feature, category) &&
              (category == E_Min || category == E_MultivariateMin);
}

bool isMaxFeature(EFeature feature) {
    EMetricCategory category;
    return    metricCategory(feature, category) &&
              (category == E_Max || category == E_MultivariateMax);
}

bool isVarianceFeature(EFeature feature) {
    EMetricCategory category;
    return     metricCategory(feature, category) &&
               (category == E_Variance);
}

bool isSumFeature(EFeature feature) {
    EMetricCategory category;
    return metricCategory(feature, category) && category == E_Sum;
}

double varianceScale(EFeature feature,
                     double sampleCount,
                     double count) {
    return    isMeanFeature(feature) ||
              isMedianFeature(feature) ||
              isVarianceFeature(feature) ?
              (sampleCount > 0.0 && count > 0.0 ? sampleCount / count : 1.0) : 1.0;
}

bool isSampled(EFeature feature) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return false;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMeanLatLongByPerson:
            return true;
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            return false;

CASE_POPULATION_COUNT:
            return false;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return true;
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return false;

CASE_PEERS_COUNT:
            return false;

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
        case E_PeersMedianByPersonAndAttribute:
        case E_PeersMinByPersonAndAttribute:
        case E_PeersMaxByPersonAndAttribute:
            return true;

        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return false;
    }
    return false;
}

unsigned minimumSampleCount(EFeature feature) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return 1;

        case E_IndividualMeanByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualMeanLatLongByPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualSumVelocityByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
            return 1;

        // Population variance needs a minimum population size
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
            return 3;

CASE_POPULATION_COUNT:
            return 1;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return 1;

        // Population variance needs a minimum population size
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
            return 3;

CASE_PEERS_COUNT:
            return 1;

CASE_PEERS_METRIC:
            return 1;
    }
    return 1;
}

double offsetCountToZero(EFeature feature, double count) {
    switch (feature) {
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
            return count - 1;
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return count;

CASE_INDIVIDUAL_METRIC:
            return count;

CASE_POPULATION_COUNT:
            return count - 1;

CASE_POPULATION_METRIC:
            return count;

CASE_PEERS_COUNT:
            return count - 1;

CASE_PEERS_METRIC:
            return count;
    }
    return count;
}

void offsetCountToZero(EFeature feature, TDouble1Vec &count) {
    for (std::size_t i = 0u; i < count.size(); ++i) {
        count[i] = offsetCountToZero(feature, count[i]);
    }
}

double inverseOffsetCountToZero(EFeature feature,  double count) {
    switch (feature) {
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
            return count + 1.0;
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return count;

CASE_INDIVIDUAL_METRIC:
            return count;

CASE_POPULATION_COUNT:
            return count + 1.0;

CASE_POPULATION_METRIC:
            return count;

CASE_PEERS_COUNT:
            return count + 1.0;

CASE_PEERS_METRIC:
            return count;
    }
    return count;
}

void inverseOffsetCountToZero(EFeature feature, TDouble1Vec &count) {
    for (std::size_t i = 0u; i < count.size(); ++i) {
        count[i] = inverseOffsetCountToZero(feature, count[i]);
    }
}

bool countsEmptyBuckets(EFeature feature) {
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
            return true;
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return false;

        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            return true;
        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualMeanLatLongByPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
            return false;

CASE_POPULATION_COUNT:
CASE_POPULATION_METRIC:
CASE_PEERS_COUNT:
CASE_PEERS_METRIC:
            return false;
    }
    return false;
}

double emptyBucketCountWeight(EFeature feature, double frequency, double cutoff) {
    if (countsEmptyBuckets(feature) && cutoff > 0.0) {
        static const double M = 1.001;
        static const double C = 0.025;
        static const double K = ::log((M + 1.0) / (M - 1.0)) / C;
        double df = frequency - std::min(cutoff + C, 1.0);
        if (df < -C) {
            return 0.0;
        } else if (df < C) {
            double fa = ::exp(K * df);
            return 0.5 * (1.0 + M * (fa - 1.0) / (fa + 1.0));
        }
    }
    return 1.0;
}

double learnRate(EFeature feature, const model::SModelParams &params) {
    return isDiurnal(feature) || isLatLong(feature) ? 1.0 : params.s_LearnRate;
}

maths_t::EProbabilityCalculation probabilityCalculation(EFeature feature) {
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return maths_t::E_TwoSided;
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
            return maths_t::E_OneSidedBelow;
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
            return maths_t::E_OneSidedAbove;

        case E_IndividualMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualMeanLatLongByPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualSumVelocityByPerson:
            return maths_t::E_TwoSided;
        case E_IndividualMinByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
            return maths_t::E_OneSidedBelow;
        case E_IndividualMaxByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
            return maths_t::E_OneSidedAbove;

        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
        case E_PopulationUniquePersonCountByAttribute:
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return maths_t::E_TwoSided;
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
            return maths_t::E_OneSidedBelow;
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
            return maths_t::E_OneSidedAbove;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return maths_t::E_TwoSided;
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
            return maths_t::E_OneSidedBelow;
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
            return maths_t::E_OneSidedAbove;

        case E_PeersAttributeTotalCountByPerson:
        case E_PeersCountByBucketPersonAndAttribute:
        case E_PeersUniqueCountByBucketPersonAndAttribute:
        case E_PeersInfoContentByBucketPersonAndAttribute:
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            return maths_t::E_TwoSided;
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowCountsByBucketPersonAndAttribute:
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
            return maths_t::E_OneSidedBelow;
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
        case E_PeersHighCountsByBucketPersonAndAttribute:
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
            return maths_t::E_OneSidedAbove;

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersMedianByPersonAndAttribute:
        case E_PeersSumByBucketPersonAndAttribute:
            return maths_t::E_TwoSided;
        case E_PeersMinByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
            return maths_t::E_OneSidedBelow;
        case E_PeersMaxByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return maths_t::E_OneSidedAbove;
    }

    return maths_t::E_TwoSided;
}

core_t::TTime sampleTime(EFeature feature,
                         core_t::TTime bucketStartTime,
                         core_t::TTime bucketLength,
                         core_t::TTime time) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return bucketStartTime + bucketLength / 2;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualMeanLatLongByPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
            return time == 0 ? bucketStartTime + bucketLength / 2 : time;
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            return bucketStartTime + bucketLength / 2;

CASE_POPULATION_COUNT:
            return bucketStartTime + bucketLength / 2;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
            return time == 0 ? bucketStartTime + bucketLength / 2 : time;
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return bucketStartTime + bucketLength / 2;

CASE_PEERS_COUNT:
            return bucketStartTime + bucketLength / 2;

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersMedianByPersonAndAttribute:
        case E_PeersMinByPersonAndAttribute:
        case E_PeersMaxByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
            return time == 0 ? bucketStartTime + bucketLength / 2 : time;
        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return bucketStartTime + bucketLength / 2;
    }

    return bucketStartTime + bucketLength / 2;
}

TDouble1VecDouble1VecPr support(EFeature feature) {
    static const double MIN_DOUBLE = -std::numeric_limits<double>::max();
    static const double MAX_DOUBLE =  std::numeric_limits<double>::max();

    std::size_t d = dimension(feature);

    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, MAX_DOUBLE)};
        case E_IndividualTimeOfDayByBucketAndPerson:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, static_cast<double>(core::constants::DAY))};
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, static_cast<double>(core::constants::WEEK))};

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualSumVelocityByPerson:
            return {TDouble1Vec(d, MIN_DOUBLE), TDouble1Vec(d, MAX_DOUBLE)};
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, MAX_DOUBLE)};
        case E_IndividualMeanLatLongByPerson:
            return {TDouble1Vec(d, -180.0), TDouble1Vec(d,  180.0)};

        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationUniquePersonCountByAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, MAX_DOUBLE)};
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
            return {TDouble1Vec(d, 0.0),
                    TDouble1Vec(d, static_cast<double>(core::constants::DAY))};
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, static_cast<double>(core::constants::WEEK))};

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return {TDouble1Vec(d, MIN_DOUBLE), TDouble1Vec(d, MAX_DOUBLE)};
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, MAX_DOUBLE)};
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return {TDouble1Vec(d, -180.0), TDouble1Vec(d,  180.0)};

CASE_PEERS_COUNT:
            return {TDouble1Vec(d, 0.0), TDouble1Vec(d, MAX_DOUBLE)};

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersMedianByPersonAndAttribute:
        case E_PeersMinByPersonAndAttribute:
        case E_PeersMaxByPersonAndAttribute:
        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return {TDouble1Vec(d, MIN_DOUBLE), TDouble1Vec(d, MAX_DOUBLE)};
    }

    return {TDouble1Vec(d, MIN_DOUBLE), TDouble1Vec(d, MAX_DOUBLE)};
}

double adjustProbability(EFeature feature,
                         core_t::TTime elapsedTime,
                         double probability) {
    // For the time of week calculation we assume that the
    // probability of less likely samples depends on whether
    // the value belongs to a cluster that hasn't yet been
    // identified. Specifically,
    //   P(less likely) =  P(less likely | new cluster) * P(new cluster)
    //                   + P(less likely | existing cluster) * (1 - P(new cluster))
    //
    // We estimate P(less likely | new cluster) = 1, i.e. we
    // assume that the density at the point will be high if
    // it belongs to a new cluster and assume that P(new cluster)
    // decays exponentially over time.

    double pNewCluster = 0.0;
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
            break;
        case E_IndividualTimeOfDayByBucketAndPerson:
            pNewCluster = ::exp(-pow4(  static_cast<double>(elapsedTime)
                                        / static_cast<double>(core::constants::DAY)));
            break;
        case E_IndividualTimeOfWeekByBucketAndPerson:
            pNewCluster = ::exp(-pow4(  static_cast<double>(elapsedTime)
                                        / static_cast<double>(core::constants::WEEK)));
            break;

CASE_INDIVIDUAL_METRIC:
            break;

        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
        case E_PopulationUniquePersonCountByAttribute:
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
            break;
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
            pNewCluster = ::exp(-pow4(  static_cast<double>(elapsedTime)
                                        / static_cast<double>(core::constants::DAY)));
            break;
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            pNewCluster = ::exp(-pow4(  static_cast<double>(elapsedTime)
                                        / static_cast<double>(core::constants::WEEK)));
            break;

CASE_POPULATION_METRIC:
            break;

        case E_PeersAttributeTotalCountByPerson:
        case E_PeersCountByBucketPersonAndAttribute:
        case E_PeersUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowCountsByBucketPersonAndAttribute:
        case E_PeersHighCountsByBucketPersonAndAttribute:
        case E_PeersInfoContentByBucketPersonAndAttribute:
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
            break;
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
            pNewCluster = ::exp(-pow4(  static_cast<double>(elapsedTime)
                                        / static_cast<double>(core::constants::DAY)));
            break;
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            pNewCluster = ::exp(-pow4(  static_cast<double>(elapsedTime)
                                        / static_cast<double>(core::constants::WEEK)));
            break;

CASE_PEERS_METRIC:
            break;
    }

    return pNewCluster + probability * (1.0 - pNewCluster);
}

TInfluenceCalculatorCPtr influenceCalculator(EFeature feature) {
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
            return boost::make_shared<model::CLogProbabilityComplementInfluenceCalculator>();
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return boost::make_shared<model::CIndicatorInfluenceCalculator>();
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
            return boost::make_shared<model::CMeanInfluenceCalculator>();
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
            return boost::make_shared<model::CLogProbabilityInfluenceCalculator>();
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
            return boost::make_shared<model::CInfluenceUnavailableCalculator>();

        case E_IndividualMeanByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
            return boost::make_shared<model::CMeanInfluenceCalculator>();
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
            return boost::make_shared<model::CVarianceInfluenceCalculator>();
        case E_IndividualMinByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualMaxVelocityByPerson:
        case E_IndividualMinVelocityByPerson:
            return boost::make_shared<model::CLogProbabilityInfluenceCalculator>();
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            return boost::make_shared<model::CLogProbabilityComplementInfluenceCalculator>();
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
            return boost::make_shared<model::CInfluenceUnavailableCalculator>();
        case E_IndividualMeanLatLongByPerson:
        case E_IndividualMeanVelocityByPerson:
            return boost::make_shared<model::CIndicatorInfluenceCalculator>();

        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityComplementInfluenceCalculator>();
        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
        case E_PopulationUniquePersonCountByAttribute:
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return boost::make_shared<model::CIndicatorInfluenceCalculator>();
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityInfluenceCalculator>();
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
            return boost::make_shared<model::CInfluenceUnavailableCalculator>();

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
            return boost::make_shared<model::CMeanInfluenceCalculator>();
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
            return boost::make_shared<model::CVarianceInfluenceCalculator>();
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityInfluenceCalculator>();
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityComplementInfluenceCalculator>();
        case E_PopulationLowSumByBucketPersonAndAttribute:
            return boost::make_shared<model::CInfluenceUnavailableCalculator>();
        case E_PopulationMeanLatLongByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
            return boost::make_shared<model::CIndicatorInfluenceCalculator>();

        case E_PeersCountByBucketPersonAndAttribute:
        case E_PeersHighCountsByBucketPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityComplementInfluenceCalculator>();
        case E_PeersAttributeTotalCountByPerson:
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            return boost::make_shared<model::CIndicatorInfluenceCalculator>();
        case E_PeersUniqueCountByBucketPersonAndAttribute:
        case E_PeersInfoContentByBucketPersonAndAttribute:
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityInfluenceCalculator>();
        case E_PeersLowCountsByBucketPersonAndAttribute:
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
            return boost::make_shared<model::CInfluenceUnavailableCalculator>();

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersMedianByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
            return boost::make_shared<model::CMeanInfluenceCalculator>();
        case E_PeersMinByPersonAndAttribute:
        case E_PeersMaxByPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityInfluenceCalculator>();
        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return boost::make_shared<model::CLogProbabilityComplementInfluenceCalculator>();
        case E_PeersLowSumByBucketPersonAndAttribute:
            return boost::make_shared<model::CInfluenceUnavailableCalculator>();
    }

    return TInfluenceCalculatorCPtr();
}

bool requiresInterimResultAdjustment(EFeature feature) {
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
            return true;
        case E_IndividualIndicatorOfBucketPerson:
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
            return false;
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
            return true;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMeanVelocityByPerson:
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
        case E_IndividualMinByPerson:
        case E_IndividualMinVelocityByPerson:
        case E_IndividualMaxByPerson:
        case E_IndividualMaxVelocityByPerson:
            return false;
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            return true;
        case E_IndividualMeanLatLongByPerson:
            return false;

        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
        case E_PopulationAttributeTotalCountByPerson:
            return true;
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
            return false;
        case E_PopulationUniquePersonCountByAttribute:
            return true;
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return false;
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
            return true;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return false;
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return true;

        case E_PeersCountByBucketPersonAndAttribute:
        case E_PeersHighCountsByBucketPersonAndAttribute:
        case E_PeersAttributeTotalCountByPerson:
            return true;
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            return false;
        case E_PeersUniqueCountByBucketPersonAndAttribute:
        case E_PeersInfoContentByBucketPersonAndAttribute:
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowCountsByBucketPersonAndAttribute:
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
            return true;

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
        case E_PeersMedianByPersonAndAttribute:
        case E_PeersMinByPersonAndAttribute:
        case E_PeersMaxByPersonAndAttribute:
            return false;
        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return true;
    }

    return false;
}

const std::string &outputFunctionName(EFeature feature) {
    switch (feature) {
        // Individual event rate features
        case E_IndividualCountByBucketAndPerson:
        case E_IndividualNonZeroCountByBucketAndPerson:
        case E_IndividualTotalBucketCountByPerson:
        case E_IndividualLowCountsByBucketAndPerson:
        case E_IndividualHighCountsByBucketAndPerson:
        case E_IndividualArrivalTimesByPerson:
        case E_IndividualLongArrivalTimesByPerson:
        case E_IndividualShortArrivalTimesByPerson:
        case E_IndividualLowNonZeroCountByBucketAndPerson:
        case E_IndividualHighNonZeroCountByBucketAndPerson:
            return model::CAnomalyDetector::COUNT_NAME;
        case E_IndividualTimeOfDayByBucketAndPerson:
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return model::CAnomalyDetector::TIME_NAME;
        case E_IndividualIndicatorOfBucketPerson:
            return model::CAnomalyDetector::RARE_NAME;
        case E_IndividualUniqueCountByBucketAndPerson:
        case E_IndividualLowUniqueCountByBucketAndPerson:
        case E_IndividualHighUniqueCountByBucketAndPerson:
            return model::CAnomalyDetector::DISTINCT_COUNT_NAME;
        case E_IndividualInfoContentByBucketAndPerson:
        case E_IndividualLowInfoContentByBucketAndPerson:
        case E_IndividualHighInfoContentByBucketAndPerson:
            return model::CAnomalyDetector::INFO_CONTENT_NAME;

        // Individual metric features
        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMeanVelocityByPerson:
            return model::CAnomalyDetector::MEAN_NAME;
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
            return model::CAnomalyDetector::MEDIAN_NAME;
        case E_IndividualMinByPerson:
        case E_IndividualMinVelocityByPerson:
            return model::CAnomalyDetector::MIN_NAME;
        case E_IndividualMaxByPerson:
        case E_IndividualMaxVelocityByPerson:
            return model::CAnomalyDetector::MAX_NAME;
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
            return model::CAnomalyDetector::VARIANCE_NAME;
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            return model::CAnomalyDetector::SUM_NAME;
        case E_IndividualMeanLatLongByPerson:
            return model::CAnomalyDetector::LAT_LONG_NAME;

        // Population event rate features
        case E_PopulationCountByBucketPersonAndAttribute:
        case E_PopulationLowCountsByBucketPersonAndAttribute:
        case E_PopulationHighCountsByBucketPersonAndAttribute:
            return model::CAnomalyDetector::COUNT_NAME;
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return model::CAnomalyDetector::TIME_NAME;
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
            return model::CAnomalyDetector::RARE_NAME;
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
            return model::CAnomalyDetector::DISTINCT_COUNT_NAME;
        case E_PopulationInfoContentByBucketPersonAndAttribute:
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
            return model::CAnomalyDetector::INFO_CONTENT_NAME;
        case E_PopulationAttributeTotalCountByPerson:
        case E_PopulationUniquePersonCountByAttribute:
            LOG_ERROR("Unexpected feature " << print(feature));
            break;

        // Population metric features
        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMeanLatLongByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
            return model::CAnomalyDetector::MEAN_NAME;
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
            return model::CAnomalyDetector::MEDIAN_NAME;
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
            return model::CAnomalyDetector::MIN_NAME;
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
            return model::CAnomalyDetector::MAX_NAME;
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
            return model::CAnomalyDetector::VARIANCE_NAME;
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            return model::CAnomalyDetector::SUM_NAME;

        // Peers event rate features
        case E_PeersCountByBucketPersonAndAttribute:
        case E_PeersLowCountsByBucketPersonAndAttribute:
        case E_PeersHighCountsByBucketPersonAndAttribute:
            return model::CAnomalyDetector::COUNT_NAME;
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            return model::CAnomalyDetector::TIME_NAME;
        case E_PeersUniqueCountByBucketPersonAndAttribute:
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
            return model::CAnomalyDetector::DISTINCT_COUNT_NAME;
        case E_PeersInfoContentByBucketPersonAndAttribute:
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
            return model::CAnomalyDetector::INFO_CONTENT_NAME;
        case E_PeersAttributeTotalCountByPerson:
            LOG_ERROR("Unexpected feature " << print(feature));
            break;

        // Peers metric features
        case E_PeersMeanByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
            return model::CAnomalyDetector::MEAN_NAME;
        case E_PeersMedianByPersonAndAttribute:
            return model::CAnomalyDetector::MEDIAN_NAME;
        case E_PeersMinByPersonAndAttribute:
            return model::CAnomalyDetector::MIN_NAME;
        case E_PeersMaxByPersonAndAttribute:
            return model::CAnomalyDetector::MAX_NAME;
        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            return model::CAnomalyDetector::SUM_NAME;
    }

    return model::CAnomalyDetector::COUNT_NAME;
}

std::string print(EFeature feature) {
    switch (feature) {
        case E_IndividualCountByBucketAndPerson:
            return "'count per bucket by person'";
        case E_IndividualNonZeroCountByBucketAndPerson:
            return "'non-zero count per bucket by person'";
        case E_IndividualTotalBucketCountByPerson:
            return "'bucket count by person'";
        case E_IndividualIndicatorOfBucketPerson:
            return "'indicator per bucket of person'";
        case E_IndividualLowCountsByBucketAndPerson:
            return "'low values of count per bucket by person'";
        case E_IndividualHighCountsByBucketAndPerson:
            return "'high values of count per bucket by person'";
        case E_IndividualArrivalTimesByPerson:
            return "'mean arrival time by person'";
        case E_IndividualLongArrivalTimesByPerson:
            return "'long mean arrival time by person'";
        case E_IndividualShortArrivalTimesByPerson:
            return "'short mean arrival time by person'";
        case E_IndividualLowNonZeroCountByBucketAndPerson:
            return "'low non-zero count per bucket by person'";
        case E_IndividualHighNonZeroCountByBucketAndPerson:
            return "'high non-zero count per bucket by person'";
        case E_IndividualUniqueCountByBucketAndPerson:
            return "'unique count per bucket by person'";
        case E_IndividualLowUniqueCountByBucketAndPerson:
            return "'low unique count per bucket by person'";
        case E_IndividualHighUniqueCountByBucketAndPerson:
            return "'high unique count per bucket by person'";
        case E_IndividualInfoContentByBucketAndPerson:
            return "'information content of value per bucket by person'";
        case E_IndividualHighInfoContentByBucketAndPerson:
            return "'high information content of value per bucket by person'";
        case E_IndividualLowInfoContentByBucketAndPerson:
            return "'low information content of value per bucket by person'";
        case E_IndividualTimeOfDayByBucketAndPerson:
            return "'time-of-day per bucket by person'";
        case E_IndividualTimeOfWeekByBucketAndPerson:
            return "'time-of-week per bucket by person'";
        case E_IndividualMeanByPerson:
            return "'arithmetic mean value by person'";
        case E_IndividualLowMeanByPerson:
            return "'low mean value by person'";
        case E_IndividualHighMeanByPerson:
            return "'high mean value by person'";
        case E_IndividualMedianByPerson:
            return "'median value by person'";
        case E_IndividualLowMedianByPerson:
            return "'low median value by person'";
        case E_IndividualHighMedianByPerson:
            return "'high median value by person'";
        case E_IndividualMinByPerson:
            return "'minimum value by person'";
        case E_IndividualMaxByPerson:
            return "'maximum value by person'";
        case E_IndividualVarianceByPerson:
            return "'variance of values by person'";
        case E_IndividualLowVarianceByPerson:
            return "'low variance of values by person'";
        case E_IndividualHighVarianceByPerson:
            return "'high variance of values by person'";
        case E_IndividualSumByBucketAndPerson:
            return "'bucket sum by person'";
        case E_IndividualLowSumByBucketAndPerson:
            return "'low bucket sum by person'";
        case E_IndividualHighSumByBucketAndPerson:
            return "'high bucket sum by person'";
        case E_IndividualNonNullSumByBucketAndPerson:
            return "'bucket non-null sum by person'";
        case E_IndividualLowNonNullSumByBucketAndPerson:
            return "'low bucket non-null sum by person'";
        case E_IndividualHighNonNullSumByBucketAndPerson:
            return "'high bucket non-null sum by person'";
        case E_IndividualMaxVelocityByPerson:
            return "'max velocity by person'";
        case E_IndividualMinVelocityByPerson:
            return "'min velocity by person'";
        case E_IndividualMeanVelocityByPerson:
            return "'mean velocity by person'";
        case E_IndividualSumVelocityByPerson:
            return "'sum velocity by person'";
        case E_IndividualMeanLatLongByPerson:
            return "'mean lat/long by person'";
        case E_PopulationAttributeTotalCountByPerson:
            return "'attribute counts by person'";
        case E_PopulationCountByBucketPersonAndAttribute:
            return "'non-zero count per bucket by person and attribute'";
        case E_PopulationIndicatorOfBucketPersonAndAttribute:
            return "'indicator per bucket of person and attribute'";
        case E_PopulationUniquePersonCountByAttribute:
            return "'unique person count by attribute'";
        case E_PopulationUniqueCountByBucketPersonAndAttribute:
            return "'unique count per bucket by person and attribute'";
        case E_PopulationLowUniqueCountByBucketPersonAndAttribute:
            return "'low unique count per bucket by person and attribute'";
        case E_PopulationHighUniqueCountByBucketPersonAndAttribute:
            return "'high unique count per bucket by person and attribute'";
        case E_PopulationLowCountsByBucketPersonAndAttribute:
            return "'low values of non-zero count per bucket by person and attribute'";
        case E_PopulationHighCountsByBucketPersonAndAttribute:
            return "'high values of non-zero count per bucket by person and attribute'";
        case E_PopulationInfoContentByBucketPersonAndAttribute:
            return "'information content of value per bucket by person and attribute'";
        case E_PopulationLowInfoContentByBucketPersonAndAttribute:
            return "'low information content of value per bucket by person and attribute'";
        case E_PopulationHighInfoContentByBucketPersonAndAttribute:
            return "'high information content of value per bucket by person and attribute'";
        case E_PopulationTimeOfDayByBucketPersonAndAttribute:
            return "'time-of-day per bucket by person and attribute'";
        case E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            return "'time-of-week per bucket by person and attribute'";
        case E_PopulationMeanByPersonAndAttribute:
            return "'mean value by person and attribute'";
        case E_PopulationLowMeanByPersonAndAttribute:
            return "'low mean by person and attribute'";
        case E_PopulationHighMeanByPersonAndAttribute:
            return "'high mean by person and attribute'";
        case E_PopulationMedianByPersonAndAttribute:
            return "'median value by person and attribute'";
        case E_PopulationLowMedianByPersonAndAttribute:
            return "'low median value by person and attribute'";
        case E_PopulationHighMedianByPersonAndAttribute:
            return "'high median value by person and attribute'";
        case E_PopulationMinByPersonAndAttribute:
            return "'minimum value by person and attribute'";
        case E_PopulationMaxByPersonAndAttribute:
            return "'maximum value by person and attribute'";
        case E_PopulationVarianceByPersonAndAttribute:
            return "'variance of values by person and attribute'";
        case E_PopulationLowVarianceByPersonAndAttribute:
            return "'low variance of values by person and attribute'";
        case E_PopulationHighVarianceByPersonAndAttribute:
            return "'high variance of values by person and attribute'";
        case E_PopulationSumByBucketPersonAndAttribute:
            return "'bucket sum by person and attribute'";
        case E_PopulationLowSumByBucketPersonAndAttribute:
            return "'low bucket sum by person and attribute'";
        case E_PopulationHighSumByBucketPersonAndAttribute:
            return "'high bucket sum by person and attribute'";
        case E_PopulationMeanLatLongByPersonAndAttribute:
            return "'mean lat/long by person and attribute'";
        case E_PopulationMaxVelocityByPersonAndAttribute:
            return "'max velocity by person and attribute'";
        case E_PopulationMinVelocityByPersonAndAttribute:
            return "'min velocity by person and attribute'";
        case E_PopulationMeanVelocityByPersonAndAttribute:
            return "'mean velocity by person and attribute'";
        case E_PopulationSumVelocityByPersonAndAttribute:
            return "'sum velocity by person and attribute'";
        case E_PeersAttributeTotalCountByPerson:
            return "'attribute counts by person'";
        case E_PeersCountByBucketPersonAndAttribute:
            return "'non-zero count per bucket by peers of person and attribute'";
        case E_PeersUniqueCountByBucketPersonAndAttribute:
            return "'unique count per bucket by peers of person and attribute'";
        case E_PeersLowUniqueCountByBucketPersonAndAttribute:
            return "'low unique count per bucket by peers of person and attribute'";
        case E_PeersHighUniqueCountByBucketPersonAndAttribute:
            return "'high unique count per bucket by peers of person and attribute'";
        case E_PeersLowCountsByBucketPersonAndAttribute:
            return "'low values of non-zero count per bucket by peers of person and attribute'";
        case E_PeersHighCountsByBucketPersonAndAttribute:
            return "'high values of non-zero count per bucket by peers of person and attribute'";
        case E_PeersInfoContentByBucketPersonAndAttribute:
            return "'information content of value per bucket by peers of person and attribute'";
        case E_PeersLowInfoContentByBucketPersonAndAttribute:
            return "'low information content of value per bucket by peers of person and attribute'";
        case E_PeersHighInfoContentByBucketPersonAndAttribute:
            return "'high information content of value per bucket by peers of person and attribute'";
        case E_PeersTimeOfDayByBucketPersonAndAttribute:
            return "'time-of-day per bucket by peers of person and attribute'";
        case E_PeersTimeOfWeekByBucketPersonAndAttribute:
            return "'time-of-week per bucket by peers of person and attribute'";
        case E_PeersMeanByPersonAndAttribute:
            return "'mean value by peers of person and attribute'";
        case E_PeersMedianByPersonAndAttribute:
            return "'median value by peers of person and attribute'";
        case E_PeersMinByPersonAndAttribute:
            return "'minimum value by peers of person and attribute'";
        case E_PeersMaxByPersonAndAttribute:
            return "'maximum value by peers of person and attribute'";
        case E_PeersSumByBucketPersonAndAttribute:
            return "'bucket sum by peers of person and attribute'";
        case E_PeersLowMeanByPersonAndAttribute:
            return "'low mean by peers of person and attribute'";
        case E_PeersHighMeanByPersonAndAttribute:
            return "'high mean by peers of person and attribute'";
        case E_PeersLowSumByBucketPersonAndAttribute:
            return "'low bucket sum by peers of person and attribute'";
        case E_PeersHighSumByBucketPersonAndAttribute:
            return "'high bucket sum by peers of person and attribute'";
    }

    return "-";
}

bool metricCategory(EFeature feature, EMetricCategory &result) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return false;

        case E_IndividualMeanByPerson:
        case E_IndividualLowMeanByPerson:
        case E_IndividualHighMeanByPerson:
        case E_IndividualMeanVelocityByPerson:
            result = E_Mean;
            return true;
        case E_IndividualMinByPerson:
        case E_IndividualMinVelocityByPerson:
            result = E_Min;
            return true;
        case E_IndividualMaxByPerson:
        case E_IndividualMaxVelocityByPerson:
            result = E_Max;
            return true;
        case E_IndividualSumByBucketAndPerson:
        case E_IndividualLowSumByBucketAndPerson:
        case E_IndividualHighSumByBucketAndPerson:
        case E_IndividualNonNullSumByBucketAndPerson:
        case E_IndividualLowNonNullSumByBucketAndPerson:
        case E_IndividualHighNonNullSumByBucketAndPerson:
        case E_IndividualSumVelocityByPerson:
            result = E_Sum;
            return true;
        case E_IndividualMeanLatLongByPerson:
            result = E_MultivariateMean;
            return true;
        case E_IndividualMedianByPerson:
        case E_IndividualLowMedianByPerson:
        case E_IndividualHighMedianByPerson:
            result = E_Median;
            return true;
        case E_IndividualVarianceByPerson:
        case E_IndividualLowVarianceByPerson:
        case E_IndividualHighVarianceByPerson:
            result = E_Variance;
            return true;

CASE_POPULATION_COUNT:
            return false;

        case E_PopulationMeanByPersonAndAttribute:
        case E_PopulationLowMeanByPersonAndAttribute:
        case E_PopulationHighMeanByPersonAndAttribute:
        case E_PopulationMeanVelocityByPersonAndAttribute:
            result = E_Mean;
            return true;
        case E_PopulationMinByPersonAndAttribute:
        case E_PopulationMinVelocityByPersonAndAttribute:
            result = E_Min;
            return true;
        case E_PopulationMaxByPersonAndAttribute:
        case E_PopulationMaxVelocityByPersonAndAttribute:
            result = E_Max;
            return true;
        case E_PopulationMeanLatLongByPersonAndAttribute:
            result = E_MultivariateMean;
            return true;
        case E_PopulationSumByBucketPersonAndAttribute:
        case E_PopulationLowSumByBucketPersonAndAttribute:
        case E_PopulationHighSumByBucketPersonAndAttribute:
        case E_PopulationSumVelocityByPersonAndAttribute:
            result = E_Sum;
            return true;
        case E_PopulationMedianByPersonAndAttribute:
        case E_PopulationLowMedianByPersonAndAttribute:
        case E_PopulationHighMedianByPersonAndAttribute:
            result = E_Median;
            return true;
        case E_PopulationVarianceByPersonAndAttribute:
        case E_PopulationLowVarianceByPersonAndAttribute:
        case E_PopulationHighVarianceByPersonAndAttribute:
            result = E_Variance;
            return true;

CASE_PEERS_COUNT:
            return false;

        case E_PeersMeanByPersonAndAttribute:
        case E_PeersLowMeanByPersonAndAttribute:
        case E_PeersHighMeanByPersonAndAttribute:
            result = E_Mean;
            return true;
        case E_PeersMinByPersonAndAttribute:
            result = E_Min;
            return true;
        case E_PeersMaxByPersonAndAttribute:
            result = E_Max;
            return true;
        case E_PeersSumByBucketPersonAndAttribute:
        case E_PeersLowSumByBucketPersonAndAttribute:
        case E_PeersHighSumByBucketPersonAndAttribute:
            result = E_Sum;
            return true;
        case E_PeersMedianByPersonAndAttribute:
            result = E_Median;
            return true;
    }

    return false;
}

std::string print(EMetricCategory category) {
    switch (category) {
        case E_Mean:
            return "'mean'";
        case E_Min:
            return "'minimum'";
        case E_Max:
            return "'maximum'";
        case E_Sum:
            return "'sum'";
        case E_MultivariateMean:
            return "'multivariate mean'";
        case E_MultivariateMin:
            return "'multivariate minimum'";
        case E_MultivariateMax:
            return "'multivariate maximum'";
        case E_Median:
            return "'median'";
        case E_Variance:
            return "'variance'";
    }
    return "-";
}

std::string print(EEventRateCategory category) {
    switch (category) {
        case E_MeanArrivalTimes:
            return "'mean arrival times'";
        case E_AttributePeople:
            return "'attributes' people'";
        case E_UniqueValues:
            return "'unique values'";
        case E_DiurnalTimes:
            return "'time-of-day values'";
    }
    return "-";
}

EAnalysisCategory analysisCategory(EFeature feature) {
    switch (feature) {
CASE_INDIVIDUAL_COUNT:
            return E_EventRate;

CASE_INDIVIDUAL_METRIC:
            return E_Metric;

CASE_POPULATION_COUNT:
            return E_PopulationEventRate;

CASE_POPULATION_METRIC:
            return E_PopulationMetric;

CASE_PEERS_COUNT:
            return E_PeersEventRate;

CASE_PEERS_METRIC:
            return E_PeersMetric;
    }

    return E_EventRate;
}

std::string print(EAnalysisCategory category) {
    switch (category) {
        case E_EventRate:
            return "'event rate'";
        case E_Metric:
            return "'metric'";
        case E_PopulationEventRate:
            return "'population event rate'";
        case E_PopulationMetric:
            return "'population metric'";
        case E_PeersEventRate:
            return "'peers event rate'";
        case E_PeersMetric:
            return "'peers metric'";
    }
    return "-";
}

std::string print(EMemoryStatus memoryStatus) {
    switch (memoryStatus) {
        case E_MemoryStatusOk:
            return "ok";
        case E_MemoryStatusSoftLimit:
            return "soft_limit";
        case E_MemoryStatusHardLimit:
            return "hard_limit";
    }
    return "-";
}

}
}

