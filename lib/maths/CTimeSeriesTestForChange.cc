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
#include <maths/CInformationCriteria.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CLinearAlgebra.h>
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
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

constexpr double EPS{0.1};

double rightTailFTest(double v0, double v1, double df0, double df1) {
    if (df1 <= 0.0) {
        return 1.0;
    }
    double F{v0 == v1 ? 1.0 : (v0 / df0) / (v1 / df1)};
    return CStatisticalTests::rightTailFTest(F, df0, df1);
}

std::size_t largestShift(const TDoubleVec& shifts) {
    std::size_t result{0};
    double largest{0.0};
    for (std::size_t i = 1; i < shifts.size(); ++i) {
        double shift{std::fabs(shifts[i] - shifts[i - 1])};
        // We prefer the earliest shift which is within 10% of the maximum.
        if (shift > (1.0 + EPS) * largest) {
            largest = shift;
            result = i;
        }
    }
    return result;
}

std::size_t largestScale(const TDoubleVec& scales) {
    std::size_t result{0};
    double largest{0.0};
    for (std::size_t i = 1; i < scales.size(); ++i) {
        double scale{std::fabs(scales[i] - scales[i - 1])};
        // We prefer the earliest scale which is within 10% of the maximum.
        if (scale > (1.0 + EPS) * largest) {
            largest = scale;
            result = i;
        }
    }
    return result;
}

const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
}

CChangePoint::~CChangePoint() = default;

CLevelShift::CLevelShift(bool reversion,
                         std::size_t changeIndex,
                         core_t::TTime time,
                         double shift,
                         core_t::TTime valuesStartTime,
                         core_t::TTime bucketLength,
                         TFloatMeanAccumulatorVec values,
                         TSizeVec segments,
                         TDoubleVec shifts,
                         TFloatMeanAccumulatorVec residuals)
    : CChangePoint{reversion, changeIndex, time, std::move(residuals)}, m_Shift{shift},
      m_ValuesStartTime{valuesStartTime}, m_BucketLength{bucketLength}, m_Values{std::move(values)},
      m_Segments{std::move(segments)}, m_Shifts{std::move(shifts)} {
}

bool CLevelShift::apply(CTrendComponent& component) const {
    component.shiftLevel(m_Shift, m_ValuesStartTime, m_BucketLength, m_Values,
                         m_Segments, m_Shifts);
    return true;
}

const std::string& CLevelShift::type() const {
    return TYPE;
}

std::string CLevelShift::print() const {
    return "level shift by " + core::CStringUtils::typeToString(m_Shift);
}

const std::string CLevelShift::TYPE{"level shift"};

CScale::CScale(bool reversion, std::size_t index, core_t::TTime time, double scale, double magnitude, TFloatMeanAccumulatorVec residuals)
    : CChangePoint{reversion, index, time, std::move(residuals)}, m_Scale{scale}, m_Magnitude{magnitude} {
}

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

CTimeShift::CTimeShift(bool reversion,
                       std::size_t index,
                       core_t::TTime time,
                       core_t::TTime shift,
                       TFloatMeanAccumulatorVec residuals)
    : CChangePoint{reversion, index, time, std::move(residuals)}, m_Shift{shift} {
}

bool CTimeShift::apply(CTimeSeriesDecomposition& decomposition) const {
    decomposition.shiftTime(this->time(), m_Shift);
    return true;
}

const std::string& CTimeShift::type() const {
    return TYPE;
}

std::string CTimeShift::print() const {
    return "time shift by " + core::CStringUtils::typeToString(m_Shift) + "s";
}

const std::string CTimeShift::TYPE{"time shift"};

CTimeSeriesTestForChange::CTimeSeriesTestForChange(int testFor,
                                                   core_t::TTime valuesStartTime,
                                                   core_t::TTime bucketsStartTime,
                                                   core_t::TTime bucketLength,
                                                   core_t::TTime sampleInterval,
                                                   TPredictor predictor,
                                                   TFloatMeanAccumulatorVec values,
                                                   double sampleVariance,
                                                   double outlierFraction)
    : m_TestFor{testFor}, m_ValuesStartTime{valuesStartTime}, m_BucketsStartTime{bucketsStartTime},
      m_BucketLength{bucketLength}, m_SampleInterval{sampleInterval},
      m_SampleVariance{sampleVariance}, m_OutlierFraction{outlierFraction},
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

    TChangePointVec changes;
    changes.reserve(3);
    if (m_TestFor & E_LevelShift) {
        changes.push_back(this->levelShift(variance, truncatedVariance, parameters));
    }
    if (m_TestFor & E_LinearScale) {
        changes.push_back(this->scale(variance, truncatedVariance, parameters));
    }
    if (m_TestFor & E_TimeShift) {
        changes.push_back(this->timeShift(variance, truncatedVariance, parameters));
    }

    changes.erase(std::remove_if(changes.begin(), changes.end(),
                                 [](const auto& change) {
                                     return change.s_ChangePoint == nullptr;
                                 }),
                  changes.end());
    LOG_TRACE(<< "# changes = " << changes.size());

    if (changes.size() > 0) {
        std::stable_sort(changes.begin(), changes.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.s_NumberParameters < rhs.s_NumberParameters;
        });

        // If the simpler hypothesis is strongly selected by the raw variance then
        // prefer it. Otherwise, if there is strong evidence for a more complex
        // explanation select that otherwise fallback to AIC.

        double selectedEvidence{aic(changes[0])};
        std::size_t selected{0};
        LOG_TRACE(<< changes[0].s_ChangePoint->print() << " evidence = " << selectedEvidence);

        double n{static_cast<double>(CSignal::countNotMissing(m_Values))};
        for (std::size_t candidate = 1; candidate < changes.size(); ++candidate) {
            double pValue{this->pValue(changes[candidate].s_ResidualVariance,
                                       changes[candidate].s_NumberParameters,
                                       changes[selected].s_ResidualVariance,
                                       changes[selected].s_NumberParameters, n)};
            if (pValue < m_SignificantPValue) {
                continue;
            }
            pValue = this->pValue(changes[selected].s_ResidualVariance,
                                  changes[selected].s_TruncatedResidualVariance,
                                  changes[selected].s_NumberParameters,
                                  changes[candidate].s_ResidualVariance,
                                  changes[candidate].s_TruncatedResidualVariance,
                                  changes[candidate].s_NumberParameters, n);
            double evidence{aic(changes[candidate])};
            LOG_TRACE(<< changes[candidate].s_ChangePoint->print()
                      << " p-value = " << pValue << ", evidence = " << evidence);
            if (pValue < m_SignificantPValue || evidence < selectedEvidence) {
                std::tie(selectedEvidence, selected) = std::make_pair(evidence, candidate);
            }
        }
        return std::move(changes[selected].s_ChangePoint);
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

    TSizeVec segments{TSegmentation::piecewiseLinear(
        m_ValuesMinusPredictions, m_SignificantPValue, m_OutlierFraction, 3)};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(segments));

    if (segments.size() > 2) {
        TDoubleVec shifts;
        auto residuals = TSegmentation::removePiecewiseLinear(
            m_ValuesMinusPredictions, segments, m_OutlierFraction, shifts);
        double varianceH1;
        double truncatedVarianceH1;
        std::tie(varianceH1, truncatedVarianceH1) = this->variances(residuals);
        std::size_t shiftIndex{largestShift(shifts)};
        std::size_t changeIndex{segments[shiftIndex]};
        std::size_t lastChangeIndex{segments[segments.size() - 2]};
        LOG_TRACE(<< "shifts = " << core::CContainerPrinter::print(shifts)
                  << ", shift index = " << shiftIndex);
        LOG_TRACE(<< "variance = " << varianceH1 << ", truncated variance = " << truncatedVarianceH1
                  << ", sample variance = " << m_SampleVariance);
        LOG_TRACE(<< "change index = " << changeIndex);

        double n{static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double parametersH1{2.0 * static_cast<double>(segments.size() - 1)};
        double pValue{this->pValue(varianceH0, truncatedVarianceH0, parametersH0,
                                   varianceH1, truncatedVarianceH1, parametersH1, n)};
        LOG_TRACE(<< "shift p-value = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            TMeanAccumulator shift;
            double weight{1.0};
            for (std::size_t i = residuals.size(); i > lastChangeIndex; --i, weight *= 0.9) {
                shift.add(CBasicStatistics::mean(m_ValuesMinusPredictions[i - 1]),
                          weight * CBasicStatistics::count(residuals[i - 1]));
            }

            // Check if the change is a reversion, i.e. the level is around the
            // same in the start and end intervals.
            bool reversion{std::fabs(shifts.back() - shifts.front()) <
                           EPS * std::fabs(shifts[shiftIndex] - shifts[shiftIndex - 1])};
            LOG_TRACE(<< "reversion = " << reversion);

            auto changePoint = std::make_unique<CLevelShift>(
                reversion, changeIndex, this->changeTime(changeIndex),
                CBasicStatistics::mean(shift), m_ValuesStartTime, m_BucketLength, m_Values,
                std::move(segments), std::move(shifts), std::move(residuals));

            return {varianceH1, truncatedVarianceH1, parametersH1, std::move(changePoint)};
        }
    }

    return {};
}

CTimeSeriesTestForChange::SChangePoint
CTimeSeriesTestForChange::scale(double varianceH0, double truncatedVarianceH0, double parametersH0) const {

    // Test for linear scales of the base predictor. We use a hypothesis test
    // against a null hypothesis that there is a quadratic trend.

    auto predictor = this->bucketPredictor();

    TSizeVec segments{TSegmentation::piecewiseLinearScaledSeasonal(
        m_Values, predictor, m_SignificantPValue, 3)};
    LOG_TRACE(<< "scale segments = " << core::CContainerPrinter::print(segments));

    if (segments.size() > 2) {
        TDoubleVec scales;
        auto residuals = TSegmentation::removePiecewiseLinearScaledSeasonal(
            m_Values, predictor, segments, m_OutlierFraction, scales);
        double varianceH1;
        double truncatedVarianceH1;
        std::tie(varianceH1, truncatedVarianceH1) = this->variances(residuals);
        std::size_t scaleIndex{largestScale(scales)};
        std::size_t changeIndex{segments[scaleIndex]};
        std::size_t lastChangeIndex{segments[segments.size() - 2]};
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales)
                  << ", scale index = " << scaleIndex);
        LOG_TRACE(<< "variance = " << varianceH1 << ", truncated variance = " << truncatedVarianceH1
                  << ", sample variance = " << m_SampleVariance);
        LOG_TRACE(<< "change index = " << changeIndex);

        double n{static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double parametersH1{static_cast<double>(segments.size() - 1)};
        double pValue{this->pValue(varianceH0, truncatedVarianceH0, parametersH0,
                                   varianceH1, truncatedVarianceH1, parametersH1, n)};
        LOG_TRACE(<< "scale p-value = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            TMeanAccumulator projection;
            TMeanAccumulator Z;
            double weight{1.0};
            for (std::size_t i = residuals.size(); i > lastChangeIndex; --i, weight *= 0.9) {
                double x{CBasicStatistics::mean(m_Values[i - 1])};
                double p{predictor(i - 1)};
                double w{weight * CBasicStatistics::count(residuals[i - 1]) * std::fabs(p)};
                if (w > 0.0) {
                    projection.add(x * p, w);
                    Z.add(p * p, w);
                }
            }
            double scale{CBasicStatistics::mean(Z) == 0.0
                             ? 1.0
                             : CBasicStatistics::mean(projection) /
                                   CBasicStatistics::mean(Z)};
            LOG_TRACE(<< "scale = " << scale);

            // Check if the change is a reversion, i.e. the scale is around the
            // same in the start and end intervals.
            bool reversion{std::fabs(scales.back() - scales.front()) <
                           EPS * std::fabs(scales[scaleIndex] - scales[scaleIndex - 1])};
            LOG_TRACE(<< "reversion = " << reversion);

            auto changePoint = std::make_unique<CScale>(
                reversion, changeIndex, this->changeTime(changeIndex), scale,
                std::fabs(scale - 1.0) * std::sqrt(CBasicStatistics::mean(Z)),
                std::move(residuals));

            return {varianceH1, truncatedVarianceH1, parametersH1, std::move(changePoint)};
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

    auto predictor = this->bucketTimePredictor();

    TSegmentation::TTimeVec candidateShifts;
    for (core_t::TTime shift = -6 * HOUR; shift < 0; shift += HALF_HOUR) {
        candidateShifts.push_back(shift);
    }
    for (core_t::TTime shift = HALF_HOUR; shift <= 6 * HOUR; shift += HALF_HOUR) {
        candidateShifts.push_back(shift);
    }

    TSegmentation::TTimeVec shifts;
    TSizeVec segments{TSegmentation::piecewiseTimeShifted(
        m_Values, m_BucketLength, candidateShifts, predictor,
        m_SignificantPValue, 2, &shifts)};
    LOG_TRACE(<< "shift segments = " << core::CContainerPrinter::print(segments));

    if (segments.size() > 2) {
        auto shiftedPredictor = [&](std::size_t i) {
            return predictor(static_cast<core_t::TTime>(m_BucketLength * i) +
                             TSegmentation::shiftAt(i, segments, shifts));
        };
        auto residuals = removePredictions(shiftedPredictor, m_Values);
        double varianceH1;
        double truncatedVarianceH1;
        std::tie(varianceH1, truncatedVarianceH1) = this->variances(residuals);
        std::size_t changeIndex{segments[segments.size() - 2]};
        LOG_TRACE(<< "shifts = " << core::CContainerPrinter::print(shifts));
        LOG_TRACE(<< "variance = " << varianceH1 << ", truncated variance = " << truncatedVarianceH1
                  << ", sample variance = " << m_SampleVariance);
        LOG_TRACE(<< "change index = " << changeIndex);

        double n{static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double parametersH1{static_cast<double>(segments.size() - 1)};
        double pValue{this->pValue(varianceH0, truncatedVarianceH0, parametersH0,
                                   varianceH1, truncatedVarianceH1, parametersH1, n)};
        LOG_TRACE(<< "time shift p-value = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            auto changePoint = std::make_unique<CTimeShift>(
                false, changeIndex, this->changeTime(changeIndex),
                shifts.back(), std::move(residuals));
            return {varianceH1, truncatedVarianceH1, parametersH1, std::move(changePoint)};
        }
    }

    return {};
}

CTimeSeriesTestForChange::TBucketPredictor CTimeSeriesTestForChange::bucketPredictor() const {
    // The following code is reverse engineering the average of the predictions
    // which fall in each bucket of the time window. We have samples at some
    // approximately fixed interval l and a bucket length L >= l. We also know
    //
    //   "values start - buckets start" = l / L sum_{0<=i<n}{ t0 + i l }    (1)
    //
    // where n = L / l. This summation is n t0 + n(n - 1)/2 l and we can use this
    // to solve (1) for t0. The expected prediction is then
    //
    //   1 / n sum_{0<=i<n}{ p(t + t0 + i * l) }

    double n{static_cast<double>(m_BucketLength) / static_cast<double>(m_SampleInterval)};
    core_t::TTime offset{static_cast<core_t::TTime>(
        std::max(static_cast<double>(m_ValuesStartTime - m_BucketsStartTime) -
                     0.5 * (n - 1.0) * static_cast<double>(m_SampleInterval),
                 0.0))};
    return [offset, this](std::size_t i) {
        TMeanAccumulator result;
        core_t::TTime time{m_BucketsStartTime +
                           static_cast<core_t::TTime>(m_BucketLength * i)};
        for (core_t::TTime dt = offset; dt < m_BucketLength; dt += m_SampleInterval) {
            result.add(m_Predictor(time + dt));
        }
        return CBasicStatistics::mean(result);
    };
}

CTimeSeriesTestForChange::TPredictor CTimeSeriesTestForChange::bucketTimePredictor() const {
    // See above.

    double n{static_cast<double>(m_BucketLength) / static_cast<double>(m_SampleInterval)};
    core_t::TTime offset{static_cast<core_t::TTime>(
        std::max(static_cast<double>(m_ValuesStartTime - m_BucketsStartTime) -
                     0.5 * (n - 1.0) * static_cast<double>(m_SampleInterval),
                 0.0))};
    return [offset, this](core_t::TTime time) {
        TMeanAccumulator result;
        time += m_BucketsStartTime;
        for (core_t::TTime dt = offset; dt < m_BucketLength; dt += m_SampleInterval) {
            result.add(m_Predictor(time + dt));
        }
        return CBasicStatistics::mean(result);
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

CTimeSeriesTestForChange::TDoubleDoublePr
CTimeSeriesTestForChange::variances(const TFloatMeanAccumulatorVec& residuals) const {
    return {CBasicStatistics::maximumLikelihoodVariance(
                this->truncatedMoments(0.0 /*all residuals*/, residuals)),
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
    return std::min(this->pValue(varianceH0, parametersH0,          // H0
                                 varianceH1, parametersH1,          // H1
                                 n),                                // # values
                    this->pValue(truncatedVarianceH0, parametersH0, // H0
                                 truncatedVarianceH1, parametersH1, // H1
                                 (1.0 - m_OutlierFraction) * n));   // # values
}

double CTimeSeriesTestForChange::pValue(double varianceH0,
                                        double parametersH0,
                                        double varianceH1,
                                        double parametersH1,
                                        double n) const {
    return rightTailFTest(varianceH0 + m_SampleVariance, varianceH1 + m_SampleVariance,
                          n - parametersH0, n - parametersH1);
}

double CTimeSeriesTestForChange::aic(const SChangePoint& change) const {
    using TVector1x1 = CVectorNx1<double, 1>;

    auto akaike = [&](const TMeanVarAccumulator& moments) {
        CSphericalGaussianInfoCriterion<TVector1x1, E_AICc> result;
        result.add(CBasicStatistics::momentsAccumulator(
            CBasicStatistics::count(moments), TVector1x1{CBasicStatistics::mean(moments)},
            TVector1x1{CBasicStatistics::maximumLikelihoodVariance(moments)}));
        return result.calculate(change.s_NumberParameters);
    };

    TMeanVarAccumulator moments{
        this->truncatedMoments(0.0, change.s_ChangePoint->residuals())};
    TMeanVarAccumulator truncatedMoments{this->truncatedMoments(
        m_OutlierFraction, change.s_ChangePoint->residuals())};

    return akaike(moments) + akaike(truncatedMoments);
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
}
}
