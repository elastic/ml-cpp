/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesChangeDetector.h>

#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CPrior.h>
#include <maths/CPriorDetail.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRegressionDetail.h>
#include <maths/CRestoreParams.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/CTimeSeriesModel.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>
#include <boost/utility/in_place_factory.hpp>

namespace ml {
namespace maths {
using namespace time_series_change_detector_detail;

namespace {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TOptionalChangeDescription = CUnivariateTimeSeriesChangeDetector::TOptionalChangeDescription;
const std::string MINIMUM_TIME_TO_DETECT_TAG{"a"};
const std::string MAXIMUM_TIME_TO_DETECT_TAG{"b"};
const std::string MINIMUM_DELTA_BIC_TO_DETECT_TAG{"c"};
const std::string RESIDUAL_MODEL_MODE_TAG{"d"};
const std::string SAMPLE_COUNT_TAG{"e"};
const std::string DECISION_FUNCTION_TAG{"f"};
const std::string MIN_TIME_TAG{"g"};
const std::string MAX_TIME_TAG{"h"};
const std::string CHANGE_MODEL_TAG{"i"};
const std::string LOG_LIKELIHOOD_TAG{"j"};
const std::string EXPECTED_LOG_LIKELIHOOD_TAG{"k"};
const std::string SHIFT_TAG{"l"};
const std::string SCALE_TAG{"m"};
const std::string RESIDUAL_MODEL_TAG{"n"};
const std::string LOG_INVERSE_DECISION_FUNCTION_TREND_TAG{"p"};
const std::size_t EXPECTED_LOG_LIKELIHOOD_NUMBER_INTERVALS{4u};
const double EXPECTED_EVIDENCE_THRESHOLD_MULTIPLIER{0.9};
const std::size_t COUNT_TO_INITIALIZE{3u};
const double MINIMUM_SCALE{0.1};
const double MAXIMUM_SCALE{10.0};
const double WINSORISATION_DERATE{1.0};
const double MAXIMUM_DECISION_FUNCTION{32.0};
const double LOG_INV_MAXIMUM_DECISION_FUNCTION{-CTools::fastLog(MAXIMUM_DECISION_FUNCTION)};
}

SChangeDescription::SChangeDescription(EDescription description, double value, const TPriorPtr& residualModel)
    : s_Description{description}, s_Value{value}, s_ResidualModel{residualModel} {
}

std::string SChangeDescription::print() const {
    std::string result;
    switch (s_Description) {
    case E_LevelShift:
        result += "level shift by ";
        break;
    case E_LinearScale:
        result += "linear scale by ";
        break;
    case E_TimeShift:
        result += "time shift by ";
        break;
    }
    return result + core::CStringUtils::typeToString(s_Value[0]);
}

CUnivariateTimeSeriesChangeDetector::CUnivariateTimeSeriesChangeDetector(
    const TDecompositionPtr& trendModel,
    const TPriorPtr& residualModel,
    core_t::TTime minimumTimeToDetect,
    core_t::TTime maximumTimeToDetect,
    double minimumDeltaBicToDetect)
    : m_MinimumTimeToDetect{minimumTimeToDetect}, m_MaximumTimeToDetect{maximumTimeToDetect},
      m_MinimumDeltaBicToDetect{minimumDeltaBicToDetect}, m_SampleCount{0}, m_DecisionFunction{0.0},
      m_ChangeModels{
          std::make_shared<CUnivariateNoChangeModel>(trendModel, residualModel),
          std::make_shared<CUnivariateLevelShiftModel>(trendModel, residualModel),
          std::make_shared<CUnivariateTimeShiftModel>(trendModel, residualModel, -core::constants::HOUR),
          std::make_shared<CUnivariateTimeShiftModel>(trendModel,
                                                      residualModel,
                                                      +core::constants::HOUR)} {
    if (trendModel->seasonalComponents().size() > 0) {
        m_ChangeModels.push_back(std::make_shared<CUnivariateLinearScaleModel>(
            trendModel, residualModel));
    }
}

bool CUnivariateTimeSeriesChangeDetector::acceptRestoreTraverser(
    const SModelRestoreParams& params,
    core::CStateRestoreTraverser& traverser) {
    auto model = m_ChangeModels.begin();
    do {
        const std::string name{traverser.name()};
        RESTORE_BUILT_IN(MINIMUM_TIME_TO_DETECT_TAG, m_MinimumTimeToDetect)
        RESTORE_BUILT_IN(MAXIMUM_TIME_TO_DETECT_TAG, m_MaximumTimeToDetect)
        RESTORE_BUILT_IN(MINIMUM_DELTA_BIC_TO_DETECT_TAG, m_MinimumDeltaBicToDetect)
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE_BUILT_IN(DECISION_FUNCTION_TAG, m_DecisionFunction)
        RESTORE(LOG_INVERSE_DECISION_FUNCTION_TREND_TAG,
                traverser.traverseSubLevel(boost::bind(&TRegression::acceptRestoreTraverser,
                                                       &m_LogInvDecisionFunctionTrend, _1)))
        RESTORE_SETUP_TEARDOWN(MIN_TIME_TAG, core_t::TTime time,
                               core::CStringUtils::stringToType(traverser.value(), time),
                               m_TimeRange.add(time))
        RESTORE_SETUP_TEARDOWN(MAX_TIME_TAG, core_t::TTime time,
                               core::CStringUtils::stringToType(traverser.value(), time),
                               m_TimeRange.add(time))
        RESTORE(CHANGE_MODEL_TAG, traverser.traverseSubLevel(boost::bind(
                                      &CUnivariateChangeModel::acceptRestoreTraverser,
                                      (model++)->get(), boost::cref(params), _1)))
    } while (traverser.next());
    return true;
}

void CUnivariateTimeSeriesChangeDetector::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(MINIMUM_TIME_TO_DETECT_TAG, m_MinimumTimeToDetect);
    inserter.insertValue(MAXIMUM_TIME_TO_DETECT_TAG, m_MaximumTimeToDetect);
    inserter.insertValue(MINIMUM_DELTA_BIC_TO_DETECT_TAG, m_MinimumDeltaBicToDetect,
                         core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertValue(DECISION_FUNCTION_TAG, m_DecisionFunction,
                         core::CIEEE754::E_SinglePrecision);
    inserter.insertLevel(LOG_INVERSE_DECISION_FUNCTION_TREND_TAG,
                         boost::bind(&TRegression::acceptPersistInserter,
                                     &m_LogInvDecisionFunctionTrend, _1));
    if (m_TimeRange.initialized()) {
        inserter.insertValue(MIN_TIME_TAG, m_TimeRange.min());
        inserter.insertValue(MAX_TIME_TAG, m_TimeRange.max());
    }
    for (const auto& model : m_ChangeModels) {
        inserter.insertLevel(CHANGE_MODEL_TAG, boost::bind(&CUnivariateChangeModel::acceptPersistInserter,
                                                           model.get(), _1));
    }
}

TOptionalChangeDescription CUnivariateTimeSeriesChangeDetector::change() {
    std::size_t best{};
    m_DecisionFunction = this->decisionFunction(best);
    if (m_DecisionFunction > 0.0) {
        double x{static_cast<double>(m_TimeRange.range()) /
                 static_cast<double>(m_MaximumTimeToDetect)};
        double y{CTools::fastLog(1.0 / m_DecisionFunction)};
        m_LogInvDecisionFunctionTrend.add(x, y);
    }
    if (m_TimeRange.range() > m_MinimumTimeToDetect && m_DecisionFunction > 1.0) {
        return m_ChangeModels[best]->change();
    }
    return TOptionalChangeDescription{};
}

double CUnivariateTimeSeriesChangeDetector::probabilityWillAccept() const {
    double prediction{std::exp(-std::max(m_LogInvDecisionFunctionTrend.predict(1.0),
                                         LOG_INV_MAXIMUM_DECISION_FUNCTION))};
    return CTools::logisticFunction(std::max(m_DecisionFunction, prediction),
                                    0.1, 1.0, -1.0);
}

double CUnivariateTimeSeriesChangeDetector::decisionFunction(std::size_t& change) const {
    using TChangeModelPtr5VecCItr = TChangeModelPtr5Vec::const_iterator;
    using TDoubleChangeModelPtr5VecCItrPr = std::pair<double, TChangeModelPtr5VecCItr>;
    using TMinAccumulator =
        CBasicStatistics::COrderStatisticsStack<TDoubleChangeModelPtr5VecCItrPr, 2>;

    if (m_SampleCount <= COUNT_TO_INITIALIZE) {
        return 0.0;
    }

    double noChangeBic{m_ChangeModels[0]->bic()};
    TMinAccumulator candidates;
    for (auto i = m_ChangeModels.begin() + 1; i != m_ChangeModels.end(); ++i) {
        candidates.add({(*i)->bic(), i});
    }
    candidates.sort();

    double evidences[]{noChangeBic - candidates[0].first,
                       noChangeBic - candidates[1].first};
    double expectedEvidence{noChangeBic - (*candidates[0].second)->expectedBic()};

    double x[]{
        evidences[0] / m_MinimumDeltaBicToDetect,
        2.0 * (evidences[0] - evidences[1]) / m_MinimumDeltaBicToDetect,
        evidences[0] / EXPECTED_EVIDENCE_THRESHOLD_MULTIPLIER / expectedEvidence,
        std::max(static_cast<double>(m_TimeRange.range() - m_MinimumTimeToDetect), 0.0) /
            static_cast<double>(m_MaximumTimeToDetect - m_MinimumTimeToDetect)};
    double p{CTools::logisticFunction(x[0], 0.05, 1.0) *
             CTools::logisticFunction(x[1], 0.1, 1.0) *
             (x[2] < 0.0 ? 1.0 : CTools::logisticFunction(x[2], 0.2, 1.0)) *
             CTools::logisticFunction(x[3], 0.2, 0.5)};
    LOG_TRACE(<< "df(" << (*candidates[0].second)->change()->print()
              << ") = " << MAXIMUM_DECISION_FUNCTION * p
              << " | x = " << core::CContainerPrinter::print(x));

    change = candidates[0].second - m_ChangeModels.begin();

    // Note the maximum decision function value is chosen so that
    // this function is equal to one when each of the decision
    // criteria are at the centre of the sigmoid functions and
    // the time range is equal to "minimum time to detect". This
    // means we'll (just) accept the change if each "hard" decision
    // criterion is individually satisfied.

    return MAXIMUM_DECISION_FUNCTION * p;
}

bool CUnivariateTimeSeriesChangeDetector::stopTesting() const {
    core_t::TTime range{m_TimeRange.range()};
    return (range > m_MinimumTimeToDetect) &&
           (range > m_MaximumTimeToDetect || m_LogInvDecisionFunctionTrend.count() == 0.0 ||
            m_LogInvDecisionFunctionTrend.predict(1.0) > 2.0);
}

void CUnivariateTimeSeriesChangeDetector::addSamples(const TTimeDoublePr1Vec& samples,
                                                     const TDoubleWeightsAry1Vec& weights) {
    for (const auto& sample : samples) {
        m_TimeRange.add(sample.first);
    }

    ++m_SampleCount;

    for (auto& model : m_ChangeModels) {
        model->addSamples(m_SampleCount, samples, weights);
    }
}

void CUnivariateTimeSeriesChangeDetector::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    core::CMemoryDebug::dynamicSize("m_ChangeModels", m_ChangeModels, mem);
}

std::size_t CUnivariateTimeSeriesChangeDetector::memoryUsage() const {
    return core::CMemory::dynamicSize(m_ChangeModels);
}

uint64_t CUnivariateTimeSeriesChangeDetector::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MinimumTimeToDetect);
    seed = CChecksum::calculate(seed, m_MaximumTimeToDetect);
    seed = CChecksum::calculate(seed, m_MinimumDeltaBicToDetect);
    seed = CChecksum::calculate(seed, m_TimeRange);
    seed = CChecksum::calculate(seed, m_SampleCount);
    seed = CChecksum::calculate(seed, m_DecisionFunction);
    seed = CChecksum::calculate(seed, m_LogInvDecisionFunctionTrend);
    return CChecksum::calculate(seed, m_ChangeModels);
}

namespace time_series_change_detector_detail {

CUnivariateChangeModel::CUnivariateChangeModel(const TDecompositionPtr& trendModel,
                                               const TPriorPtr& residualModel)
    : m_LogLikelihood{0.0}, m_ExpectedLogLikelihood{0.0},
      m_TrendModel{trendModel}, m_ResidualModel{residualModel} {
}

bool CUnivariateChangeModel::acceptRestoreTraverser(const SModelRestoreParams& /*params*/,
                                                    core::CStateRestoreTraverser& traverser) {
    do {
        const std::string name{traverser.name()};
        RESTORE_BUILT_IN(LOG_LIKELIHOOD_TAG, m_LogLikelihood);
        RESTORE_BUILT_IN(EXPECTED_LOG_LIKELIHOOD_TAG, m_ExpectedLogLikelihood);
        return true;
    } while (traverser.next());
    return true;
}

void CUnivariateChangeModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(LOG_LIKELIHOOD_TAG, m_LogLikelihood, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(EXPECTED_LOG_LIKELIHOOD_TAG, m_ExpectedLogLikelihood,
                         core::CIEEE754::E_SinglePrecision);
}

void CUnivariateChangeModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    // Note if the trend and residual models are shallow copied their
    // reference count will be updated so core::CMemory::dynamicSize
    // will give the correct contribution for these reference.
    core::CMemoryDebug::dynamicSize("m_TrendModel", m_TrendModel, mem);
    core::CMemoryDebug::dynamicSize("m_ResidualModel", m_ResidualModel, mem);
}

std::size_t CUnivariateChangeModel::memoryUsage() const {
    // See above.
    return core::CMemory::dynamicSize(m_TrendModel) +
           core::CMemory::dynamicSize(m_ResidualModel);
}

uint64_t CUnivariateChangeModel::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_LogLikelihood);
    seed = CChecksum::calculate(seed, m_ExpectedLogLikelihood);
    seed = CChecksum::calculate(seed, m_TrendModel);
    return CChecksum::calculate(seed, m_ResidualModel);
}

bool CUnivariateChangeModel::restoreResidualModel(const SDistributionRestoreParams& params,
                                                  core::CStateRestoreTraverser& traverser) {
    return traverser.traverseSubLevel(boost::bind<bool>(
        CPriorStateSerialiser(), boost::cref(params), boost::ref(m_ResidualModel), _1));
}

double CUnivariateChangeModel::logLikelihood() const {
    return m_LogLikelihood;
}

double CUnivariateChangeModel::expectedLogLikelihood() const {
    return m_ExpectedLogLikelihood;
}

void CUnivariateChangeModel::updateLogLikelihood(const TDouble1Vec& samples,
                                                 const TDoubleWeightsAry1Vec& weights) {
    double logLikelihood{};
    if (m_ResidualModel->jointLogMarginalLikelihood(samples, weights, logLikelihood) ==
        maths_t::E_FpNoErrors) {
        m_LogLikelihood += logLikelihood;
    }
}

void CUnivariateChangeModel::updateExpectedLogLikelihood(const TDoubleWeightsAry1Vec& weights) {
    for (const auto& weight : weights) {
        double expectedLogLikelihood{};
        if (m_ResidualModel->expectation(
                maths::CPrior::CLogMarginalLikelihood{*m_ResidualModel, {weight}},
                EXPECTED_LOG_LIKELIHOOD_NUMBER_INTERVALS, expectedLogLikelihood, weight)) {
            m_ExpectedLogLikelihood += expectedLogLikelihood;
        }
    }
}

const CTimeSeriesDecompositionInterface& CUnivariateChangeModel::trendModel() const {
    return *m_TrendModel;
}

const CPrior& CUnivariateChangeModel::residualModel() const {
    return *m_ResidualModel;
}

CPrior& CUnivariateChangeModel::residualModel() {
    return *m_ResidualModel;
}

CUnivariateChangeModel::TPriorPtr CUnivariateChangeModel::residualModelPtr() const {
    return m_ResidualModel;
}

CUnivariateNoChangeModel::CUnivariateNoChangeModel(const TDecompositionPtr& trendModel,
                                                   const TPriorPtr& residualModel)
    : CUnivariateChangeModel{trendModel, residualModel} {
}

bool CUnivariateNoChangeModel::acceptRestoreTraverser(const SModelRestoreParams& params,
                                                      core::CStateRestoreTraverser& traverser) {
    return this->CUnivariateChangeModel::acceptRestoreTraverser(params, traverser);
}

void CUnivariateNoChangeModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    this->CUnivariateChangeModel::acceptPersistInserter(inserter);
}

double CUnivariateNoChangeModel::bic() const {
    return -2.0 * this->logLikelihood();
}

double CUnivariateNoChangeModel::expectedBic() const {
    // This is irrelevant since this is only used for deciding
    // whether to accept a change.
    return this->bic();
}

TOptionalChangeDescription CUnivariateNoChangeModel::change() const {
    return TOptionalChangeDescription();
}

void CUnivariateNoChangeModel::addSamples(const std::size_t count,
                                          const TTimeDoublePr1Vec& samples_,
                                          TDoubleWeightsAry1Vec weights) {
    // See, for example, CUnivariateLevelShiftModel::addSamples
    // for an explanation of the delay updating the log-likelihood.

    if (count >= COUNT_TO_INITIALIZE) {
        TDouble1Vec samples;
        samples.reserve(samples_.size());
        for (std::size_t i = 0u; i < samples_.size(); ++i) {
            core_t::TTime time{samples_[i].first};
            double value{samples_[i].second};
            double sample{this->trendModel().detrend(time, value, 0.0)};
            samples.push_back(sample);
        }
        for (auto& weight : weights) {
            maths_t::setWinsorisationWeight(1.0, weight);
        }
        this->updateLogLikelihood(samples, weights);
    }
}

std::size_t CUnivariateNoChangeModel::staticSize() const {
    return sizeof(*this);
}

uint64_t CUnivariateNoChangeModel::checksum(uint64_t seed) const {
    return this->CUnivariateChangeModel::checksum(seed);
}

CUnivariateLevelShiftModel::CUnivariateLevelShiftModel(const TDecompositionPtr& trendModel,
                                                       const TPriorPtr& residualModel)
    : CUnivariateChangeModel{trendModel, TPriorPtr{residualModel->clone()}},
      m_ResidualModelMode{residualModel->marginalLikelihoodMode()}, m_SampleCount{0.0} {
}

bool CUnivariateLevelShiftModel::acceptRestoreTraverser(const SModelRestoreParams& params,
                                                        core::CStateRestoreTraverser& traverser) {
    if (this->CUnivariateChangeModel::acceptRestoreTraverser(params, traverser) == false) {
        return false;
    }
    do {
        const std::string name{traverser.name()};
        RESTORE(SHIFT_TAG, m_Shift.fromDelimited(traverser.value()))
        RESTORE_BUILT_IN(RESIDUAL_MODEL_MODE_TAG, m_ResidualModelMode)
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE(RESIDUAL_MODEL_TAG,
                this->restoreResidualModel(params.s_DistributionParams, traverser))
    } while (traverser.next());
    return true;
}

void CUnivariateLevelShiftModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    this->CUnivariateChangeModel::acceptPersistInserter(inserter);
    inserter.insertValue(SHIFT_TAG, m_Shift.toDelimited());
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertLevel(RESIDUAL_MODEL_TAG,
                         boost::bind<void>(CPriorStateSerialiser(),
                                           boost::cref(this->residualModel()), _1));
}

double CUnivariateLevelShiftModel::bic() const {
    return -2.0 * this->logLikelihood() + CTools::fastLog(m_SampleCount);
}

double CUnivariateLevelShiftModel::expectedBic() const {
    return -2.0 * this->expectedLogLikelihood() + CTools::fastLog(m_SampleCount);
}

TOptionalChangeDescription CUnivariateLevelShiftModel::change() const {
    return SChangeDescription{SChangeDescription::E_LevelShift,
                              CBasicStatistics::mean(m_Shift), this->residualModelPtr()};
}

void CUnivariateLevelShiftModel::addSamples(const std::size_t count,
                                            const TTimeDoublePr1Vec& samples_,
                                            TDoubleWeightsAry1Vec weights) {
    const CTimeSeriesDecompositionInterface& trendModel{this->trendModel()};

    // We delay updating the log-likelihood because early on the
    // level can change giving us a better apparent fit to the
    // data than a fixed step. Five updates was found to be the
    // minimum to get empirically similar sum log-likelihood if
    // there is no change in the data.

    if (count >= COUNT_TO_INITIALIZE) {
        CPrior& residualModel{this->residualModel()};

        TDouble1Vec samples;
        samples.reserve(samples_.size());
        double shift{CBasicStatistics::mean(m_Shift)};
        for (std::size_t i = 0u; i < samples_.size(); ++i) {
            core_t::TTime time{samples_[i].first};
            double value{samples_[i].second};
            double seasonalScale{maths_t::seasonalVarianceScale(weights[i])};
            double sample{trendModel.detrend(time, value, 0.0) - shift};
            double weight{winsorisation::tailWeight(
                residualModel, WINSORISATION_DERATE, seasonalScale, sample)};
            samples.push_back(sample);
            maths_t::setWinsorisationWeight(weight, weights[i]);
            m_SampleCount += maths_t::count(weights[i]);
        }

        residualModel.addSamples(samples, weights);
        residualModel.propagateForwardsByTime(1.0);

        for (auto& weight : weights) {
            maths_t::setWinsorisationWeight(1.0, weight);
        }
        this->updateLogLikelihood(samples, weights);
        this->updateExpectedLogLikelihood(weights);
    }

    for (std::size_t i = 0u; i < samples_.size(); ++i) {
        core_t::TTime time{samples_[i].first};
        double value{samples_[i].second};
        double shift{trendModel.detrend(time, value, 0.0) - m_ResidualModelMode};
        m_Shift.add(shift);
    }
}

std::size_t CUnivariateLevelShiftModel::staticSize() const {
    return sizeof(*this);
}

uint64_t CUnivariateLevelShiftModel::checksum(uint64_t seed) const {
    seed = this->CUnivariateChangeModel::checksum(seed);
    seed = CChecksum::calculate(seed, m_Shift);
    return CChecksum::calculate(seed, m_SampleCount);
}

CUnivariateLinearScaleModel::CUnivariateLinearScaleModel(const TDecompositionPtr& trendModel,
                                                         const TPriorPtr& residualModel)
    : CUnivariateChangeModel{trendModel, TPriorPtr{residualModel->clone()}},
      m_ResidualModelMode{residualModel->marginalLikelihoodMode()}, m_SampleCount{0.0} {
}

bool CUnivariateLinearScaleModel::acceptRestoreTraverser(const SModelRestoreParams& params,
                                                         core::CStateRestoreTraverser& traverser) {
    if (this->CUnivariateChangeModel::acceptRestoreTraverser(params, traverser) == false) {
        return false;
    }
    do {
        const std::string name{traverser.name()};
        RESTORE(SCALE_TAG, m_Scale.fromDelimited(traverser.value()))
        RESTORE_BUILT_IN(RESIDUAL_MODEL_MODE_TAG, m_ResidualModelMode)
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE(RESIDUAL_MODEL_TAG,
                this->restoreResidualModel(params.s_DistributionParams, traverser))
    } while (traverser.next());
    return true;
}

void CUnivariateLinearScaleModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    this->CUnivariateChangeModel::acceptPersistInserter(inserter);
    inserter.insertValue(SCALE_TAG, m_Scale.toDelimited());
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertLevel(RESIDUAL_MODEL_TAG,
                         boost::bind<void>(CPriorStateSerialiser(),
                                           boost::cref(this->residualModel()), _1));
}

double CUnivariateLinearScaleModel::bic() const {
    return -2.0 * this->logLikelihood() + CTools::fastLog(m_SampleCount);
}

double CUnivariateLinearScaleModel::expectedBic() const {
    return -2.0 * this->expectedLogLikelihood() + CTools::fastLog(m_SampleCount);
}

CUnivariateLinearScaleModel::TOptionalChangeDescription
CUnivariateLinearScaleModel::change() const {
    return SChangeDescription{SChangeDescription::E_LinearScale,
                              CBasicStatistics::mean(m_Scale), this->residualModelPtr()};
}

void CUnivariateLinearScaleModel::addSamples(const std::size_t count,
                                             const TTimeDoublePr1Vec& samples_,
                                             TDoubleWeightsAry1Vec weights) {
    const CTimeSeriesDecompositionInterface& trendModel{this->trendModel()};

    // We delay updating the log-likelihood because early on the
    // scale can change giving us a better apparent fit to the
    // data than a fixed scale. Five updates was found to be the
    // minimum to get empirically similar sum log-likelihood if
    // there is no change in the data.

    for (std::size_t i = 0u; i < samples_.size(); ++i) {
        core_t::TTime time{samples_[i].first};
        double value{samples_[i].second - m_ResidualModelMode};
        double prediction{CBasicStatistics::mean(trendModel.value(time, 0.0))};
        double scale{std::fabs(value) / std::fabs(prediction)};
        m_Scale.add(value * prediction < 0.0
                        ? MINIMUM_SCALE
                        : CTools::truncate(scale, MINIMUM_SCALE, MAXIMUM_SCALE),
                    std::fabs(prediction));
    }

    if (count >= COUNT_TO_INITIALIZE) {
        CPrior& residualModel{this->residualModel()};

        TDouble1Vec samples;
        samples.reserve(samples_.size());
        double scale{CBasicStatistics::mean(m_Scale)};
        for (std::size_t i = 0u; i < samples_.size(); ++i) {
            core_t::TTime time{samples_[i].first};
            double value{samples_[i].second};
            double seasonalScale{maths_t::seasonalVarianceScale(weights[i])};
            double prediction{CBasicStatistics::mean(trendModel.value(time, 0.0))};
            double sample{value - scale * prediction};
            double weight{winsorisation::tailWeight(
                residualModel, WINSORISATION_DERATE, seasonalScale, sample)};
            samples.push_back(sample);
            maths_t::setWinsorisationWeight(weight, weights[i]);
            m_SampleCount += maths_t::count(weights[i]);
        }

        residualModel.addSamples(samples, weights);
        residualModel.propagateForwardsByTime(1.0);

        for (auto& weight : weights) {
            maths_t::setWinsorisationWeight(1.0, weight);
        }
        this->updateLogLikelihood(samples, weights);
        this->updateExpectedLogLikelihood(weights);
    }
}

std::size_t CUnivariateLinearScaleModel::staticSize() const {
    return sizeof(*this);
}

uint64_t CUnivariateLinearScaleModel::checksum(uint64_t seed) const {
    seed = this->CUnivariateChangeModel::checksum(seed);
    seed = CChecksum::calculate(seed, m_Scale);
    return CChecksum::calculate(seed, m_SampleCount);
}

CUnivariateTimeShiftModel::CUnivariateTimeShiftModel(const TDecompositionPtr& trendModel,
                                                     const TPriorPtr& residualModel,
                                                     core_t::TTime shift)
    : CUnivariateChangeModel{trendModel, TPriorPtr{residualModel->clone()}}, m_Shift{shift} {
}

bool CUnivariateTimeShiftModel::acceptRestoreTraverser(const SModelRestoreParams& params,
                                                       core::CStateRestoreTraverser& traverser) {
    if (this->CUnivariateChangeModel::acceptRestoreTraverser(params, traverser) == false) {
        return false;
    }
    do {
        const std::string name{traverser.name()};
        RESTORE(RESIDUAL_MODEL_TAG,
                this->restoreResidualModel(params.s_DistributionParams, traverser))
    } while (traverser.next());
    return true;
}

void CUnivariateTimeShiftModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    this->CUnivariateChangeModel::acceptPersistInserter(inserter);
    inserter.insertLevel(RESIDUAL_MODEL_TAG,
                         boost::bind<void>(CPriorStateSerialiser(),
                                           boost::cref(this->residualModel()), _1));
}

double CUnivariateTimeShiftModel::bic() const {
    return -2.0 * this->logLikelihood();
}

double CUnivariateTimeShiftModel::expectedBic() const {
    return -2.0 * this->expectedLogLikelihood();
}

TOptionalChangeDescription CUnivariateTimeShiftModel::change() const {
    return SChangeDescription{SChangeDescription::E_TimeShift,
                              static_cast<double>(m_Shift), this->residualModelPtr()};
}

void CUnivariateTimeShiftModel::addSamples(const std::size_t count,
                                           const TTimeDoublePr1Vec& samples_,
                                           TDoubleWeightsAry1Vec weights) {
    // See, for example, CUnivariateLevelShiftModel::addSamples
    // for an explanation of the delay updating the log-likelihood.

    if (count >= COUNT_TO_INITIALIZE) {
        CPrior& residualModel{this->residualModel()};

        TDouble1Vec samples;
        samples.reserve(samples_.size());
        for (std::size_t i = 0u; i < samples_.size(); ++i) {
            core_t::TTime time{samples_[i].first};
            double value{samples_[i].second};
            double sample{this->trendModel().detrend(time + m_Shift, value, 0.0)};
            samples.push_back(sample);
        }

        residualModel.addSamples(samples, weights);
        residualModel.propagateForwardsByTime(1.0);

        for (auto& weight : weights) {
            maths_t::setWinsorisationWeight(1.0, weight);
        }
        this->updateLogLikelihood(samples, weights);
        this->updateExpectedLogLikelihood(weights);
    }
}

std::size_t CUnivariateTimeShiftModel::staticSize() const {
    return sizeof(*this);
}

uint64_t CUnivariateTimeShiftModel::checksum(uint64_t seed) const {
    seed = this->CUnivariateChangeModel::checksum(seed);
    return CChecksum::calculate(seed, m_Shift);
}
}
}
}
