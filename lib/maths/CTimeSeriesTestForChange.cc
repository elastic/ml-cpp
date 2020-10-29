
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesTestForChange.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCalendarComponent.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/CTrendComponent.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSegmentation = CTimeSeriesSegmentation;

double rightTailFTest(double v0, double v1, double df0, double df1) {
    if (df1 <= 0.0) {
        return 1.0;
    }
    double F{v0 == v1 ? 1.0 : (v0 / df0) / (v1 / df1)};
    return CStatisticalTests::rightTailFTest(F, df0, df1);
}

const std::size_t H1{1};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const std::string NO_CHANGE{"no change"};
}

bool CLevelShift::apply(CTrendComponent& component) const {
    component.shiftLevel(this->time(), m_ValueAtShift, m_Shift);
    return true;
}

const std::string& CLevelShift::type() const {
    return TYPE;
}

std::string CLevelShift::print() const {
    return "level shift by " + core::CStringUtils::typeToString(m_Shift);
}

const std::string CLevelShift::TYPE{"level shift"};

bool CScale::apply(CTrendComponent& component) const {
    component.linearScale(m_Scale);
    return true;
}

bool CScale::apply(CSeasonalComponent& component) const {
    component.linearScale(this->time(), m_Scale);
    return true;
}

bool CScale::apply(CCalendarComponent& component) const {
    component.linearScale(this->time(), m_Scale);
    return true;
}

const std::string& CScale::type() const {
    return TYPE;
}

std::string CScale::print() const {
    return "linear scale by " + core::CStringUtils::typeToString(m_Scale);
}

const std::string CScale::TYPE{"scale"};

bool CTimeShift::apply(CTimeSeriesDecomposition& decomposition) const {
    decomposition.shiftTime(m_Shift);
    return true;
}

const std::string& CTimeShift::type() const {
    return TYPE;
}

std::string CTimeShift::print() const {
    return "time shift by " + core::CStringUtils::typeToString(m_Shift) + "s";
}

const std::string CTimeShift::TYPE{"time shift"};

CTimeSeriesTestForChange::CTimeSeriesTestForChange(core_t::TTime valuesStartTime,
                                                   core_t::TTime bucketsStartTime,
                                                   core_t::TTime bucketLength,
                                                   core_t::TTime predictionInterval,
                                                   TPredictor predictor,
                                                   TFloatMeanAccumulatorVec values,
                                                   double minimumVariance,
                                                   double outlierFraction)
    : m_ValuesStartTime{valuesStartTime}, m_BucketsStartTime{bucketsStartTime},
      m_BucketLength{bucketLength}, m_PredictionInterval{predictionInterval},
      m_MinimumVariance{minimumVariance}, m_OutlierFraction{outlierFraction},
      m_Predictor{std::move(predictor)}, m_Values{std::move(values)},
      m_Outliers{static_cast<std::size_t>(std::max(
          outlierFraction * static_cast<double>(CSignal::countNotMissing(m_Values)) + 0.5,
          1.0))} {

    TMeanVarAccumulator moments{this->truncatedMoments(m_OutlierFraction, m_Values)};
    TMeanVarAccumulator meanAbs{this->truncatedMoments(
        m_OutlierFraction, m_Values, [](const TFloatMeanAccumulator& value) {
            return std::fabs(CBasicStatistics::mean(value));
        })};

    // Note we don't bother modelling changes whose size is too small compared
    // to the absolute values. We won't raise anomalies for differences from our
    // predictions which are smaller than this anyway.
    m_EpsVariance = std::max(
        CTools::pow2(1000.0 * std::numeric_limits<double>::epsilon()) *
            CBasicStatistics::maximumLikelihoodVariance(moments),
        CTools::pow2(MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)));
    LOG_TRACE(<< "eps variance = " << m_EpsVariance);
}

CTimeSeriesTestForChange::TChangePointUPtr CTimeSeriesTestForChange::test() const {

    using TChangePointVec = std::vector<SChangePoint>;

    double variance;
    double truncatedVariance;
    double parameters;
    std::tie(variance, truncatedVariance, parameters) = this->quadraticTrend();

    TChangePointVec shocks;
    shocks.reserve(3);
    shocks.push_back(this->levelShift(variance, truncatedVariance, parameters));
    shocks.push_back(this->scale(variance, truncatedVariance, parameters));
    shocks.push_back(this->timeShift(variance, truncatedVariance, parameters));

    shocks.erase(std::remove_if(shocks.begin(), shocks.end(),
                                [](const auto& change) {
                                    return change.s_Type == E_NoChangePoint;
                                }),
                 shocks.end());
    LOG_TRACE(<< "# shocks = " << shocks.size());

    if (shocks.size() > 0) {
        std::stable_sort(shocks.begin(), shocks.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.s_NumberParameters < rhs.s_NumberParameters;
        });

        // If there is strong evidence for a more complex explanation select that
        // otherwise fallback to AIC.

        double selectedEvidence{aic(shocks[0])};
        std::size_t selected{0};
        LOG_TRACE(<< print(shocks[0].s_Type) << " evidence = " << selectedEvidence);

        double n{static_cast<double>(CSignal::countNotMissing(m_Values))};
        for (std::size_t candidate = 1; candidate < shocks.size(); ++candidate) {
            double pValue{std::min(
                rightTailFTest(shocks[selected].s_TruncatedResidualVariance,
                               shocks[candidate].s_TruncatedResidualVariance,
                               (1.0 - m_OutlierFraction) * n - shocks[selected].s_NumberParameters,
                               (1.0 - m_OutlierFraction) * n - shocks[candidate].s_NumberParameters),
                rightTailFTest(shocks[selected].s_ResidualVariance,
                               shocks[candidate].s_ResidualVariance,
                               n - shocks[selected].s_NumberParameters,
                               n - shocks[candidate].s_NumberParameters))};
            double evidence{aic(shocks[H1])};
            LOG_TRACE(<< print(shocks[H1].s_Type) << " p-value = " << pValue
                      << ", evidence = " << evidence);
            if (pValue < m_SignificantPValue || evidence < selectedEvidence) {
                std::tie(selectedEvidence, selected) = std::make_pair(evidence, candidate);
            }
        }

        switch (shocks[selected].s_Type) {
        case E_LevelShift:
            return std::make_unique<CLevelShift>(
                shocks[selected].s_Time, shocks[selected].s_ValueAtChange,
                shocks[selected].s_LevelShift,
                std::move(shocks[selected].s_InitialValues));
        case E_Scale:
            return std::make_unique<CScale>(
                shocks[selected].s_Time, shocks[selected].s_Scale,
                std::move(shocks[selected].s_InitialValues));
        case E_TimeShift:
            return std::make_unique<CTimeShift>(
                shocks[selected].s_Time, shocks[selected].s_TimeShift,
                std::move(shocks[selected].s_InitialValues));
        case E_NoChangePoint:
            LOG_ERROR(<< "Unexpected type");
            break;
        }
    }

    return {};
}

CTimeSeriesTestForChange::TDoubleDoubleDoubleTr CTimeSeriesTestForChange::quadraticTrend() const {

    using TRegression = CLeastSquaresOnlineRegression<2, double>;

    m_ValuesMinusPredictions = this->removePredictions(this->bucketPredictor(), m_Values);

    TRegression trend;
    TRegression::TArray parameters;
    parameters.fill(0.0);
    auto predictor = [&](std::size_t i) {
        return trend.predict(parameters, static_cast<double>(i));
    };
    for (std::size_t i = 0; i < 2; ++i) {
        CSignal::reweightOutliers(predictor, m_OutlierFraction, m_ValuesMinusPredictions);
        for (std::size_t j = 0; j < m_ValuesMinusPredictions.size(); ++j) {
            trend.add(static_cast<double>(j),
                      CBasicStatistics::mean(m_ValuesMinusPredictions[j]),
                      CBasicStatistics::count(m_ValuesMinusPredictions[j]));
        }
        trend.parameters(parameters);
    }
    m_ValuesMinusPredictions =
        this->removePredictions(predictor, std::move(m_ValuesMinusPredictions));

    double variance;
    double truncatedVariance;
    std::tie(variance, truncatedVariance) = this->variances(m_ValuesMinusPredictions);
    LOG_TRACE(<< "variance = " << variance << ", truncated variance = " << truncatedVariance);

    return {variance, truncatedVariance, 3.0};
}

CTimeSeriesTestForChange::SChangePoint
CTimeSeriesTestForChange::levelShift(double varianceH0,
                                     double truncatedVarianceH0,
                                     double parametersH0) const {

    // Test for piecewise linear shift. We use a hypothesis test against a null
    // hypothesis that there is a quadratic trend.

    m_ValuesMinusPredictions = this->removePredictions(this->bucketPredictor(), m_Values);

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(
        m_ValuesMinusPredictions, m_SignificantPValue, m_OutlierFraction, 2)};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(trendSegments));

    if (trendSegments.size() > 2) {
        TDoubleVec shifts;
        auto residuals = TSegmentation::removePiecewiseLinear(
            m_ValuesMinusPredictions, trendSegments, m_OutlierFraction, &shifts);
        double variance;
        double truncatedVariance;
        std::tie(variance, truncatedVariance) = this->variances(residuals);
        LOG_TRACE(<< "shifts = " << core::CContainerPrinter::print(shifts));
        LOG_TRACE(<< "variance = " << variance << ", truncated variance = " << truncatedVariance
                  << ", minimum variance = " << m_MinimumVariance);
        LOG_TRACE(<< "change index = " << trendSegments[1]);

        double n{static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double parameters{2.0 * static_cast<double>(trendSegments.size() - 1)};
        double pValue{this->pValue(varianceH0, truncatedVarianceH0, parametersH0,
                                   variance, truncatedVariance, parameters, n)};
        LOG_TRACE(<< "shift p-value = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            SChangePoint change{E_LevelShift,
                                this->changeTime(trendSegments[1]),
                                this->changeValue(trendSegments[1]),
                                variance,
                                truncatedVariance,
                                2.0 * static_cast<double>(trendSegments.size() - 1),
                                std::move(residuals)};
            change.s_LevelShift = shifts.back();
            return change;
        }
    }

    return {};
}

CTimeSeriesTestForChange::SChangePoint
CTimeSeriesTestForChange::scale(double varianceH0, double truncatedVarianceH0, double parametersH0) const {

    // Test for linear scales of the base predictor. We use a hypothesis test
    // against a null hypothesis that there is a quadratic trend.

    auto predictor = this->bucketPredictor();

    TSizeVec scaleSegments{TSegmentation::piecewiseLinearScaledSeasonal(
        m_Values, predictor, m_SignificantPValue, 2)};
    LOG_TRACE(<< "scale segments = " << core::CContainerPrinter::print(scaleSegments));

    if (scaleSegments.size() > 2) {
        TDoubleVec scales;
        auto residuals = TSegmentation::removePiecewiseLinearScaledSeasonal(
            m_Values, predictor, scaleSegments, m_OutlierFraction, &scales);
        double variance;
        double truncatedVariance;
        std::tie(variance, truncatedVariance) = this->variances(residuals);
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
        LOG_TRACE(<< "variance = " << variance << ", truncated variance = " << truncatedVariance
                  << ", minimum variance = " << m_MinimumVariance);
        LOG_TRACE(<< "change index = " << scaleSegments[1]);

        double n{static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double parameters{static_cast<double>(scaleSegments.size() - 1)};
        double pValue{this->pValue(varianceH0, truncatedVarianceH0, parametersH0,
                                   variance, truncatedVariance, parameters, n)};
        LOG_TRACE(<< "scale p-value = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            SChangePoint change{E_Scale,
                                this->changeTime(scaleSegments[1]),
                                this->changeValue(scaleSegments[1]),
                                variance,
                                truncatedVariance,
                                static_cast<double>(scaleSegments.size() - 1),
                                std::move(residuals)};
            change.s_Scale = scales.back();
            return change;
        }
    }

    return {};
}

CTimeSeriesTestForChange::SChangePoint
CTimeSeriesTestForChange::timeShift(double varianceH0,
                                    double truncatedVarianceH0,
                                    double parametersH0) const {

    // Test for time shifts of the base predictor. We use a hypothesis test
    // against a null hypothesis that there is a quadratic trend.

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    auto predictor = [this](core_t::TTime time) {
        TMeanAccumulator result;
        for (core_t::TTime offset = 0; offset < m_BucketLength; offset += m_PredictionInterval) {
            result.add(m_Predictor(m_BucketsStartTime + time + offset));
        }
        return CBasicStatistics::mean(result);
    };

    TSegmentation::TTimeVec candidateShifts;
    for (core_t::TTime shift = -6 * HOUR; shift < 0; shift += HALF_HOUR) {
        candidateShifts.push_back(shift);
    }
    for (core_t::TTime shift = HALF_HOUR; shift <= 6 * HOUR; shift += HALF_HOUR) {
        candidateShifts.push_back(shift);
    }

    TSegmentation::TTimeVec shifts;
    TSizeVec shiftSegments{TSegmentation::piecewiseTimeShifted(
        m_Values, m_BucketLength, candidateShifts, predictor,
        m_SignificantPValue, 2, &shifts)};
    LOG_TRACE(<< "shift segments = " << core::CContainerPrinter::print(shiftSegments));

    if (shiftSegments.size() > 2) {
        auto shiftedPredictor = [&](std::size_t i) {
            return m_Predictor(m_ValuesStartTime +
                               m_BucketLength * static_cast<core_t::TTime>(i) +
                               TSegmentation::shiftAt(i, shiftSegments, shifts));
        };
        auto residuals = removePredictions(shiftedPredictor, m_Values);
        double variance;
        double truncatedVariance;
        std::tie(variance, truncatedVariance) = this->variances(residuals);
        LOG_TRACE(<< "shifts = " << core::CContainerPrinter::print(shifts));
        LOG_TRACE(<< "variance = " << variance << ", truncated variance = " << truncatedVariance
                  << ", minimum variance = " << m_MinimumVariance);
        LOG_TRACE(<< "change index = " << shiftSegments[1]);

        double n{static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double parameters{static_cast<double>(shiftSegments.size() - 1)};
        double pValue{this->pValue(varianceH0, truncatedVarianceH0, parametersH0,
                                   variance, truncatedVariance, parameters, n)};
        LOG_TRACE(<< "time shift p-value = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            SChangePoint change{E_TimeShift,
                                this->changeTime(shiftSegments[1]),
                                this->changeValue(shiftSegments[1]),
                                variance,
                                truncatedVariance,
                                static_cast<double>(shiftSegments.size() - 1),
                                std::move(residuals)};
            change.s_TimeShift = std::accumulate(shifts.begin(), shifts.end(), 0);
            return change;
        }
    }

    return {};
}

CTimeSeriesTestForChange::TBucketPredictor CTimeSeriesTestForChange::bucketPredictor() const {
    return [this](std::size_t i) {
        return m_Predictor(m_ValuesStartTime +
                           m_BucketLength * static_cast<core_t::TTime>(i));
    };
}

CTimeSeriesTestForChange::TMeanVarAccumulator
CTimeSeriesTestForChange::truncatedMoments(double outlierFraction,
                                           const TFloatMeanAccumulatorVec& residuals,
                                           const TTransform& transform) const {
    double cutoff{std::numeric_limits<double>::max()};
    std::size_t count{CSignal::countNotMissing(residuals)};
    std::size_t numberOutliers{
        static_cast<std::size_t>(outlierFraction * static_cast<double>(count) + 0.5)};
    if (numberOutliers > 0) {
        m_Outliers.clear();
        m_Outliers.resize(numberOutliers);
        for (const auto& value : residuals) {
            if (CBasicStatistics::count(value) > 0.0) {
                m_Outliers.add(std::fabs(transform(value)));
            }
        }
        cutoff = m_Outliers.biggest();
        count -= m_Outliers.count();
    }
    LOG_TRACE(<< "cutoff = " << cutoff << ", count = " << count);

    TMeanVarAccumulator moments;
    for (const auto& value : residuals) {
        if (CBasicStatistics::count(value) > 0.0 && std::fabs(transform(value)) < cutoff) {
            moments.add(transform(value));
        }
    }
    if (numberOutliers > 0) {
        moments.add(cutoff, static_cast<double>(count) - CBasicStatistics::count(moments));
    }
    CBasicStatistics::moment<1>(moments) += m_EpsVariance;

    return moments;
}

core_t::TTime CTimeSeriesTestForChange::changeTime(std::size_t changeIndex) const {
    return m_ValuesStartTime + m_BucketLength * static_cast<core_t::TTime>(changeIndex);
}

double CTimeSeriesTestForChange::changeValue(std::size_t changeIndex) const {
    return CBasicStatistics::mean(m_Values[changeIndex - 1]);
}

CTimeSeriesTestForChange::TDoubleDoublePr
CTimeSeriesTestForChange::variances(const TFloatMeanAccumulatorVec& residuals) const {
    return {CBasicStatistics::maximumLikelihoodVariance(this->truncatedMoments(0.0, residuals)),
            CBasicStatistics::maximumLikelihoodVariance(
                this->truncatedMoments(m_OutlierFraction, residuals))};
}

double CTimeSeriesTestForChange::pValue(double varianceH0,
                                        double truncatedVarianceH0,
                                        double parametersH0,
                                        double varianceH1,
                                        double truncatedVarianceH1,
                                        double parametersH1,
                                        double n) const {
    return std::min(rightTailFTest(varianceH0 + m_MinimumVariance, varianceH1 + m_MinimumVariance,
                                   n - parametersH0, n - parametersH1),
                    rightTailFTest(truncatedVarianceH0 + m_MinimumVariance,
                                   truncatedVarianceH1 + m_MinimumVariance,
                                   (1.0 - m_OutlierFraction) * n - parametersH0,
                                   (1.0 - m_OutlierFraction) * n - parametersH1));
}

CTimeSeriesTestForChange::TFloatMeanAccumulatorVec
CTimeSeriesTestForChange::removePredictions(const TBucketPredictor& predictor,
                                            TFloatMeanAccumulatorVec values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predictor(i);
        }
    }
    return values;
}

std::size_t CTimeSeriesTestForChange::buckets(core_t::TTime bucketLength,
                                              core_t::TTime interval) {
    return static_cast<std::size_t>((interval + bucketLength / 2) / bucketLength);
}

double CTimeSeriesTestForChange::aic(const SChangePoint& change) {
    // This is max_{\theta}{ -2 log(P(y | \theta)) + 2 * # parameters }
    //
    // We assume that the data are normally distributed.
    return -std::log(std::exp(-2.0) / boost::math::double_constants::two_pi /
                     change.s_ResidualVariance) +
           -std::log(std::exp(-2.0) / boost::math::double_constants::two_pi /
                     change.s_TruncatedResidualVariance) +
           4.0 * change.s_NumberParameters;
}

const std::string& CTimeSeriesTestForChange::print(EType type) {
    switch (type) {
    case E_LevelShift:
        return CLevelShift::TYPE;
    case E_Scale:
        return CScale::TYPE;
    case E_TimeShift:
        return CTimeShift::TYPE;
    case E_NoChangePoint:
        return NO_CHANGE;
    }
}
}
}
