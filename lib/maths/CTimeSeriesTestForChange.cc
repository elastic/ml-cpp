/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesTestForChange.h>

#include <core/CContainerPrinter.h>
#include <core/CIEEE754.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCalendarComponent.h>
#include <maths/CChecksum.h>
#include <maths/CInformationCriteria.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CMathsFuncs.h>
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
#include <memory>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSegmentation = CTimeSeriesSegmentation;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

constexpr double EPS{0.1};
constexpr core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
constexpr core_t::TTime HOUR{core::constants::HOUR};
const double LOG0p95{std::log(0.95)};

double rightTailFTest(double v0, double v1, double df0, double df1) {
    // If there is insufficient data for either hypothesis treat we are conservative
    // and say the alternative hypothesis is not provable.
    if (df0 <= 0.0 || df1 <= 0.0) {
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

const core::TPersistenceTag TYPE_TAG{"a", "change_type"};
const core::TPersistenceTag TIME_TAG{"b", "change_time"};
const core::TPersistenceTag VALUE_TAG{"c", "change_value"};
const core::TPersistenceTag SIGNIFICANT_P_VALUE_TAG{"d", "significant_p_value"};
}

CChangePoint::CChangePoint(core_t::TTime time, TFloatMeanAccumulatorVec residuals, double significantPValue)
    : m_Time{time}, m_SignificantPValue{significantPValue}, m_Residuals{std::move(residuals)} {
}

CChangePoint::~CChangePoint() = default;

std::uint64_t CChangePoint::checksum(std::uint64_t seed) const {
    seed = CChecksum::calculate(seed, this->type());
    seed = CChecksum::calculate(seed, m_Time);
    seed = CChecksum::calculate(seed, m_SignificantPValue);
    seed = CChecksum::calculate(seed, m_Residuals);
    seed = CChecksum::calculate(seed, m_Mse);
    return CChecksum::calculate(seed, m_UndoneMse);
}

void CChangePoint::add(core_t::TTime time,
                       core_t::TTime lastTime,
                       double value,
                       double weight,
                       const TPredictor& predictor) {
    double factor{std::exp(3.0 * LOG0p95 * static_cast<double>(time - lastTime) /
                           static_cast<double>(HOUR))};
    m_Mse.add(CTools::pow2(value - predictor(time)), weight);
    m_Mse.age(factor);
    m_UndoneMse.add(CTools::pow2(value - this->undonePredict(predictor, time)), weight);
    m_UndoneMse.age(factor);
}

bool CChangePoint::shouldUndo() const {
    return rightTailFTest(CBasicStatistics::mean(m_Mse), CBasicStatistics::mean(m_UndoneMse),
                          CBasicStatistics::count(m_Mse),
                          CBasicStatistics::count(m_UndoneMse)) < m_SignificantPValue;
}

CLevelShift::CLevelShift(core_t::TTime time,
                         double shift,
                         core_t::TTime valuesStartTime,
                         core_t::TTime bucketLength,
                         TFloatMeanAccumulatorVec values,
                         TSizeVec segments,
                         TDoubleVec shifts,
                         TFloatMeanAccumulatorVec residuals,
                         double significantPValue)
    : CChangePoint{time, std::move(residuals), significantPValue}, m_Shift{shift},
      m_ValuesStartTime{valuesStartTime}, m_BucketLength{bucketLength}, m_Values{std::move(values)},
      m_Segments{std::move(segments)}, m_Shifts{std::move(shifts)} {
}

CLevelShift::TChangePointUPtr CLevelShift::undoable() const {
    return {};
}

bool CLevelShift::largeEnough(double threshold) const {
    return std::fabs(m_Shift) > threshold;
}

bool CLevelShift::longEnough(core_t::TTime time, core_t::TTime minimumDuration) const {
    return time >= this->time() + minimumDuration;
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

std::uint64_t CLevelShift::checksum(std::uint64_t seed) const {
    seed = this->CChangePoint::checksum(seed);
    seed = CChecksum::calculate(seed, m_Shift);
    seed = CChecksum::calculate(seed, m_ValuesStartTime);
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_Values);
    seed = CChecksum::calculate(seed, m_Segments);
    return CChecksum::calculate(seed, m_Shifts);
}

const std::string CLevelShift::TYPE{"level shift"};

CScale::CScale(core_t::TTime time,
               double scale,
               double magnitude,
               double minimumDurationScale,
               TFloatMeanAccumulatorVec residuals,
               double significantPValue)
    : CChangePoint{time, std::move(residuals), significantPValue}, m_Scale{scale},
      m_Magnitude{magnitude}, m_MinimumDurationScale{minimumDurationScale} {
}

CScale::TChangePointUPtr CScale::undoable() const {
    return {};
}

bool CScale::largeEnough(double threshold) const {
    return m_Magnitude > threshold;
}

bool CScale::longEnough(core_t::TTime time, core_t::TTime minimumDuration) const {
    minimumDuration = static_cast<core_t::TTime>(
        static_cast<double>(minimumDuration) / m_MinimumDurationScale + 0.5);
    return time >= this->time() + minimumDuration;
}

bool CScale::apply(CTrendComponent& component) const {
    component.linearScale(this->time(), m_Scale);
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

std::uint64_t CScale::checksum(std::uint64_t seed) const {
    seed = this->CChangePoint::checksum(seed);
    seed = CChecksum::calculate(seed, m_Scale);
    return CChecksum::calculate(seed, m_Magnitude);
}

const std::string CScale::TYPE{"scale"};

CTimeShift::CTimeShift(core_t::TTime time,
                       core_t::TTime shift,
                       TFloatMeanAccumulatorVec residuals,
                       double significantPValue)
    : CChangePoint{time, std::move(residuals), significantPValue}, m_Shift{shift} {
}

CTimeShift::CTimeShift(core_t::TTime time, core_t::TTime shift, double significantPValue)
    : CChangePoint{time, {}, significantPValue}, m_Shift{shift} {
}

CTimeShift::TChangePointUPtr CTimeShift::undoable() const {
    return std::make_unique<CTimeShift>(this->time(), -m_Shift, this->significantPValue());
}

bool CTimeShift::longEnough(core_t::TTime time, core_t::TTime minimumDuration) const {
    return time >= this->time() + minimumDuration;
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

std::uint64_t CTimeShift::checksum(std::uint64_t seed) const {
    seed = this->CChangePoint::checksum(seed);
    return CChecksum::calculate(seed, m_Shift);
}

double CTimeShift::undonePredict(const TPredictor& predictor, core_t::TTime time) const {
    return predictor(time + m_Shift);
}

const std::string CTimeShift::TYPE{"time shift"};

bool CUndoableChangePointStateSerializer::
operator()(TChangePointUPtr& result, core::CStateRestoreTraverser& traverser) const {
    std::string type;
    core_t::TTime time{std::numeric_limits<core_t::TTime>::min()};
    double value{std::numeric_limits<double>::quiet_NaN()};
    double significantPValue{std::numeric_limits<double>::quiet_NaN()};

    do {
        const std::string& name{traverser.name()};
        RESTORE_NO_ERROR(TYPE_TAG, type = traverser.value())
        RESTORE_BUILT_IN(TIME_TAG, time)
        RESTORE_BUILT_IN(VALUE_TAG, value)
        RESTORE_BUILT_IN(SIGNIFICANT_P_VALUE_TAG, significantPValue)
    } while (traverser.next());

    if (time == std::numeric_limits<core_t::TTime>::min()) {
        LOG_ERROR(<< "Missing '" << TIME_TAG << "'");
        return false;
    }
    if (CMathsFuncs::isFinite(value) == false) {
        LOG_ERROR(<< "Missing '" << VALUE_TAG << "'");
        return false;
    }
    if (CMathsFuncs::isFinite(significantPValue) == false) {
        LOG_ERROR(<< "Missing '" << SIGNIFICANT_P_VALUE_TAG << "'");
        return false;
    }

    if (type == CLevelShift::TYPE || type == CScale::TYPE) {
        LOG_ERROR(<< "Unexpected type '" << type << "'");
        return false;
    }
    if (type == CTimeShift::TYPE) {
        result = std::make_unique<CTimeShift>(
            time, static_cast<core_t::TTime>(value), significantPValue);
        return true;
    }
    LOG_ERROR(<< "Missing '" << TYPE_TAG << "'");
    return false;
}

void CUndoableChangePointStateSerializer::
operator()(const CChangePoint& changePoint, core::CStatePersistInserter& inserter) const {
    inserter.insertValue(TYPE_TAG, changePoint.type());
    inserter.insertValue(TIME_TAG, changePoint.time());
    inserter.insertValue(VALUE_TAG, changePoint.value(), core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(SIGNIFICANT_P_VALUE_TAG, changePoint.significantPValue(),
                         core::CIEEE754::E_DoublePrecision);
}

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

    m_ValuesMinusPredictions = removePredictions(this->bucketIndexPredictor(), m_Values);

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
        removePredictions(predictor, std::move(m_ValuesMinusPredictions));

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

    m_ValuesMinusPredictions = removePredictions(this->bucketIndexPredictor(), m_Values);

    TSizeVec segments{TSegmentation::piecewiseLinear(
        m_ValuesMinusPredictions, m_SignificantPValue, m_OutlierFraction, 3)};

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
        LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(segments));
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
            for (std::size_t i = residuals.size(); i > lastChangeIndex; --i, weight *= 0.85) {
                shift.add(CBasicStatistics::mean(m_ValuesMinusPredictions[i - 1]),
                          weight * CBasicStatistics::count(residuals[i - 1]));
            }

            auto changePoint = std::make_unique<CLevelShift>(
                this->changeTime(changeIndex), CBasicStatistics::mean(shift),
                m_ValuesStartTime, m_BucketLength, m_Values, std::move(segments),
                std::move(shifts), std::move(residuals), m_SignificantPValue);

            return {varianceH1, truncatedVarianceH1, parametersH1, std::move(changePoint)};
        }
    }

    return {};
}

CTimeSeriesTestForChange::SChangePoint
CTimeSeriesTestForChange::scale(double varianceH0, double truncatedVarianceH0, double parametersH0) const {

    // Test for linear scales of the base predictor. We use a hypothesis test
    // against a null hypothesis that there is a quadratic trend.

    auto predictor = this->bucketIndexPredictor();

    TSizeVec segments{TSegmentation::piecewiseLinearScaledSeasonal(
        m_Values, predictor, m_SignificantPValue, 3)};

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
        LOG_TRACE(<< "scale segments = " << core::CContainerPrinter::print(segments));
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
            for (std::size_t i = residuals.size(); i > lastChangeIndex; --i, weight *= 0.85) {
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
                             : std::max(CBasicStatistics::mean(projection) /
                                            CBasicStatistics::mean(Z),
                                        0.0)};
            LOG_TRACE(<< "scale = " << scale);

            // The impact of applying a scale is less clear for small values.
            // We therefore wait to see more data if the predicted absolute
            // values we've observed to change are relatively small.
            TMeanAccumulator averagePredictionBeforeChange;
            TMeanAccumulator averagePredictionAfterChange;
            for (std::size_t i = 0; i < changeIndex; ++i) {
                averagePredictionBeforeChange.add(std::fabs(predictor(i)));
            }
            for (std::size_t i = changeIndex; i < m_Values.size(); ++i) {
                averagePredictionAfterChange.add(std::fabs(predictor(i)));
            }
            double minimumDurationScale{
                std::min(CBasicStatistics::mean(averagePredictionAfterChange) /
                             CBasicStatistics::mean(averagePredictionBeforeChange),
                         1.0)};
            LOG_TRACE(<< "minimum duration scale = " << minimumDurationScale);

            auto changePoint = std::make_unique<CScale>(
                this->changeTime(changeIndex), scale,
                std::fabs(scale - 1.0) * std::sqrt(CBasicStatistics::mean(Z)),
                minimumDurationScale, std::move(residuals), m_SignificantPValue);

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

    auto predictor = this->bucketPredictor();

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
        m_SignificantPValue, 3, &shifts)};

    if (segments.size() > 2) {
        auto shiftedPredictor = [&](std::size_t i) {
            return predictor(m_BucketLength * static_cast<core_t::TTime>(i) +
                             TSegmentation::shiftAt(i, segments, shifts));
        };
        auto residuals = removePredictions(shiftedPredictor, m_Values);
        double varianceH1;
        double truncatedVarianceH1;
        std::tie(varianceH1, truncatedVarianceH1) = this->variances(residuals);
        std::size_t changeIndex{segments[segments.size() - 2]};
        LOG_TRACE(<< "shift segments = " << core::CContainerPrinter::print(segments));
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
                this->changeTime(changeIndex), shifts.back(),
                std::move(residuals), m_SignificantPValue);
            return {varianceH1, truncatedVarianceH1, parametersH1, std::move(changePoint)};
        }
    }

    return {};
}

CTimeSeriesTestForChange::TBucketIndexPredictor
CTimeSeriesTestForChange::bucketIndexPredictor() const {
    auto bucketPredictor = this->bucketPredictor();
    return [ predictor = std::move(bucketPredictor), this ](std::size_t i) {
        return predictor(m_BucketLength * static_cast<core_t::TTime>(i));
    };
}

CTimeSeriesTestForChange::TPredictor CTimeSeriesTestForChange::bucketPredictor() const {
    return CSignal::bucketPredictor(m_Predictor, m_BucketsStartTime, m_BucketLength,
                                    m_ValuesStartTime - m_BucketsStartTime, m_SampleInterval);
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
CTimeSeriesTestForChange::removePredictions(const TBucketIndexPredictor& predictor,
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
