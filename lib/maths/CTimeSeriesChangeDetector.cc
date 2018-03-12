/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CTimeSeriesChangeDetector.h>

#include <core/Constants.h>
#include <core/CoreTypes.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CSmallVector.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CPrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CTimeSeriesModel.h>
#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>
#include <boost/utility/in_place_factory.hpp>

namespace ml
{
namespace maths
{
using namespace time_series_change_detector_detail;

namespace
{
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TOptionalChangeDescription = CUnivariateTimeSeriesChangeDetector::TOptionalChangeDescription;

const std::string MINIMUM_TIME_TO_DETECT{"a"};
const std::string MAXIMUM_TIME_TO_DETECT{"b"};
const std::string MINIMUM_DELTA_BIC_TO_DETECT{"c"};
const std::string RESIDUAL_MODEL_MODE_TAG{"d"};
const std::string SAMPLE_COUNT_TAG{"e"};
const std::string CURRENT_EVIDENCE_OF_CHANGE{"f"};
const std::string MIN_TIME_TAG{"g"};
const std::string MAX_TIME_TAG{"h"};
const std::string CHANGE_MODEL_TAG{"i"};
const std::string LOG_LIKELIHOOD_TAG{"j"};
const std::string SHIFT_TAG{"k"};
const std::string TREND_MODEL_TAG{"l"};
const std::string RESIDUAL_MODEL_TAG{"m"};
}

SChangeDescription::SChangeDescription(EDescription description,
                                       double value,
                                       const TPriorPtr &residualModel) :
        s_Description{description},
        s_Value{value},
        s_ResidualModel{residualModel}
{}

std::string SChangeDescription::print() const
{
    std::string result;
    switch (s_Description)
    {
    case E_LevelShift: result += "level shift by "; break;
    case E_TimeShift:  result += "time shift by ";  break;
    }
    return result + core::CStringUtils::typeToString(s_Value[0]);
}

CUnivariateTimeSeriesChangeDetector::CUnivariateTimeSeriesChangeDetector(const TDecompositionPtr &trendModel,
                                                                         const TPriorPtr &residualModel,
                                                                         core_t::TTime minimumTimeToDetect,
                                                                         core_t::TTime maximumTimeToDetect,
                                                                         double minimumDeltaBicToDetect) :
        m_MinimumTimeToDetect{minimumTimeToDetect},
        m_MaximumTimeToDetect{maximumTimeToDetect},
        m_MinimumDeltaBicToDetect{minimumDeltaBicToDetect},
        m_SampleCount{0},
        m_CurrentEvidenceOfChange{0.0},
        m_ChangeModels{boost::make_shared<CUnivariateNoChangeModel>(trendModel, residualModel),
                       boost::make_shared<CUnivariateLevelShiftModel>(trendModel, residualModel),
                       boost::make_shared<CUnivariateTimeShiftModel>(trendModel, residualModel, -core::constants::HOUR),
                       boost::make_shared<CUnivariateTimeShiftModel>(trendModel, residualModel, +core::constants::HOUR)}
{}

bool CUnivariateTimeSeriesChangeDetector::acceptRestoreTraverser(const SModelRestoreParams &params,
                                                                 core::CStateRestoreTraverser &traverser)
{
    auto model = m_ChangeModels.begin();
    do
    {
        const std::string name{traverser.name()};
        RESTORE_BUILT_IN(MINIMUM_TIME_TO_DETECT, m_MinimumTimeToDetect)
        RESTORE_BUILT_IN(MAXIMUM_TIME_TO_DETECT, m_MaximumTimeToDetect)
        RESTORE_BUILT_IN(MINIMUM_DELTA_BIC_TO_DETECT, m_MinimumDeltaBicToDetect)
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE_BUILT_IN(CURRENT_EVIDENCE_OF_CHANGE, m_CurrentEvidenceOfChange)
        RESTORE_SETUP_TEARDOWN(MIN_TIME_TAG,
                               core_t::TTime time,
                               core::CStringUtils::stringToType(traverser.value(), time),
                               m_TimeRange.add(time))
        RESTORE_SETUP_TEARDOWN(MAX_TIME_TAG,
                               core_t::TTime time,
                               core::CStringUtils::stringToType(traverser.value(), time),
                               m_TimeRange.add(time))
        RESTORE(CHANGE_MODEL_TAG, traverser.traverseSubLevel(boost::bind(
                                          &CUnivariateChangeModel::acceptRestoreTraverser,
                                          (model++)->get(), boost::cref(params), _1)))
    }
    while (traverser.next());
    return true;
}

void CUnivariateTimeSeriesChangeDetector::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(MINIMUM_TIME_TO_DETECT, m_MinimumTimeToDetect);
    inserter.insertValue(MAXIMUM_TIME_TO_DETECT, m_MaximumTimeToDetect);
    inserter.insertValue(MINIMUM_DELTA_BIC_TO_DETECT, m_MinimumDeltaBicToDetect,
                                                      core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertValue(CURRENT_EVIDENCE_OF_CHANGE, m_CurrentEvidenceOfChange,
                                                     core::CIEEE754::E_SinglePrecision);
    if (m_TimeRange.initialized())
    {
        inserter.insertValue(MIN_TIME_TAG, m_TimeRange.min());
        inserter.insertValue(MAX_TIME_TAG, m_TimeRange.max());
    }
    for (const auto &model : m_ChangeModels)
    {
        inserter.insertLevel(CHANGE_MODEL_TAG,
                             boost::bind(&CUnivariateChangeModel::acceptPersistInserter,
                                         model.get(), _1));
    }
}

TOptionalChangeDescription CUnivariateTimeSeriesChangeDetector::change()
{
    using TChangeModelPtr4VecCItr = TChangeModelPtr4Vec::const_iterator;
    using TDoubleChangeModelPtr4VecCItrPr = std::pair<double, TChangeModelPtr4VecCItr>;
    using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleChangeModelPtr4VecCItrPr, 2>;

    if (m_TimeRange.range() > m_MinimumTimeToDetect)
    {
        double noChangeBic{m_ChangeModels[0]->bic()};
        TMinAccumulator candidates;
        for (auto i = m_ChangeModels.begin() + 1; i != m_ChangeModels.end(); ++i)
        {
            candidates.add({(*i)->bic(), i});
        }
        candidates.sort();

        double evidences[]{noChangeBic - candidates[0].first,
                           noChangeBic - candidates[1].first};
        m_CurrentEvidenceOfChange = evidences[0];
        if (   evidences[0] > m_MinimumDeltaBicToDetect
            && evidences[0] > evidences[1] + m_MinimumDeltaBicToDetect / 2.0)
        {
            return (*candidates[0].second)->change();
        }
    }
    return TOptionalChangeDescription();
}

bool CUnivariateTimeSeriesChangeDetector::stopTesting() const
{
    core_t::TTime range{m_TimeRange.range()};
    if (range > m_MinimumTimeToDetect)
    {
        double scale{0.5 + CTools::smoothHeaviside(2.0 * m_CurrentEvidenceOfChange
                                                       / m_MinimumDeltaBicToDetect, 0.2, 1.0)};
        return  static_cast<double>(range)
              > m_MinimumTimeToDetect + scale * static_cast<double>(
                                                    m_MaximumTimeToDetect - m_MinimumTimeToDetect);
    }
    return false;
}

void CUnivariateTimeSeriesChangeDetector::addSamples(const TWeightStyleVec &weightStyles,
                                                     const TTimeDoublePr1Vec &samples,
                                                     const TDouble4Vec1Vec &weights)
{
    for (const auto &sample : samples)
    {
        m_TimeRange.add(sample.first);
    }

    ++m_SampleCount;

    for (auto &model : m_ChangeModels)
    {
        model->addSamples(m_SampleCount, weightStyles, samples, weights);
    }
}

void CUnivariateTimeSeriesChangeDetector::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    core::CMemoryDebug::dynamicSize("m_ChangeModels", m_ChangeModels, mem);
}

std::size_t CUnivariateTimeSeriesChangeDetector::memoryUsage() const
{
    return core::CMemory::dynamicSize(m_ChangeModels);
}

uint64_t CUnivariateTimeSeriesChangeDetector::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_MinimumTimeToDetect);
    seed = CChecksum::calculate(seed, m_MaximumTimeToDetect);
    seed = CChecksum::calculate(seed, m_MinimumDeltaBicToDetect);
    seed = CChecksum::calculate(seed, m_TimeRange);
    seed = CChecksum::calculate(seed, m_SampleCount);
    seed = CChecksum::calculate(seed, m_CurrentEvidenceOfChange);
    return CChecksum::calculate(seed, m_ChangeModels);
}

namespace time_series_change_detector_detail
{

CUnivariateChangeModel::CUnivariateChangeModel(const TDecompositionPtr &trendModel,
                                               const TPriorPtr &residualModel) :
        m_LogLikelihood{0.0}, m_TrendModel{trendModel}, m_ResidualModel{residualModel}
{}

void CUnivariateChangeModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    // Note if the trend and residual models are shallow copied their
    // reference count will be updated so core::CMemory::dynamicSize
    // will give the correct contribution for these reference.
    core::CMemoryDebug::dynamicSize("m_TrendModel", m_TrendModel, mem);
    core::CMemoryDebug::dynamicSize("m_ResidualModel", m_ResidualModel, mem);
}

std::size_t CUnivariateChangeModel::memoryUsage() const
{
    // See above.
    return  core::CMemory::dynamicSize(m_TrendModel)
          + core::CMemory::dynamicSize(m_ResidualModel);
}

uint64_t CUnivariateChangeModel::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_LogLikelihood);
    seed = CChecksum::calculate(seed, m_TrendModel);
    return CChecksum::calculate(seed, m_ResidualModel);
}

bool CUnivariateChangeModel::restoreResidualModel(const SDistributionRestoreParams &params,
                                                  core::CStateRestoreTraverser &traverser)
{
    return traverser.traverseSubLevel(boost::bind<bool>(CPriorStateSerialiser(),
                                                        boost::cref(params),
                                                        boost::ref(m_ResidualModel), _1));
}

double CUnivariateChangeModel::logLikelihood() const
{
    return m_LogLikelihood;
}

void CUnivariateChangeModel::addLogLikelihood(double logLikelihood)
{
    m_LogLikelihood += logLikelihood;
}

const CTimeSeriesDecompositionInterface &CUnivariateChangeModel::trendModel() const
{
    return *m_TrendModel;
}

CTimeSeriesDecompositionInterface &CUnivariateChangeModel::trendModel()
{
    return *m_TrendModel;
}

const CPrior &CUnivariateChangeModel::residualModel() const
{
    return *m_ResidualModel;
}

CPrior &CUnivariateChangeModel::residualModel()
{
    return *m_ResidualModel;
}

CUnivariateChangeModel::TPriorPtr CUnivariateChangeModel::residualModelPtr() const
{
    return m_ResidualModel;
}

CUnivariateNoChangeModel::CUnivariateNoChangeModel(const TDecompositionPtr &trendModel,
                                                   const TPriorPtr &residualModel) :
        CUnivariateChangeModel{trendModel, residualModel}
{}

bool CUnivariateNoChangeModel::acceptRestoreTraverser(const SModelRestoreParams &/*params*/,
                                                      core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(LOG_LIKELIHOOD_TAG,
                               double logLikelihood,
                               core::CStringUtils::stringToType(traverser.value(), logLikelihood),
                               this->addLogLikelihood(logLikelihood))
    }
    while (traverser.next());
    return true;
}

void CUnivariateNoChangeModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LOG_LIKELIHOOD_TAG, this->logLikelihood());
}

double CUnivariateNoChangeModel::bic() const
{
    return -2.0 * this->logLikelihood();
}

TOptionalChangeDescription CUnivariateNoChangeModel::change() const
{
    return TOptionalChangeDescription();
}

void CUnivariateNoChangeModel::addSamples(std::size_t count,
                                          const TWeightStyleVec &weightStyles,
                                          const TTimeDoublePr1Vec &samples_,
                                          const TDouble4Vec1Vec &weights)
{
    // See CUnivariateTimeSeriesLevelShiftModel for an explanation
    // of the delay updating the log-likelihood.

    if (count >= COUNT_TO_INITIALIZE)
    {
        TDouble1Vec samples;
        samples.reserve(samples_.size());
        for (const auto &sample : samples_)
        {
            samples.push_back(this->trendModel().detrend(sample.first, sample.second, 0.0));
        }

        double logLikelihood;
        if (this->residualModel().jointLogMarginalLikelihood(weightStyles, samples, weights,
                                                             logLikelihood) == maths_t::E_FpNoErrors)
        {
            this->addLogLikelihood(logLikelihood);
        }
    }
}

std::size_t CUnivariateNoChangeModel::staticSize() const
{
    return sizeof(*this);
}

uint64_t CUnivariateNoChangeModel::checksum(uint64_t seed) const
{
    return this->CUnivariateChangeModel::checksum(seed);
}

CUnivariateLevelShiftModel::CUnivariateLevelShiftModel(const TDecompositionPtr &trendModel,
                                                       const TPriorPtr &residualModel) :
        CUnivariateChangeModel{trendModel, TPriorPtr{residualModel->clone()}},
        m_ResidualModelMode{residualModel->marginalLikelihoodMode()},
        m_SampleCount{0.0}
{}

bool CUnivariateLevelShiftModel::acceptRestoreTraverser(const SModelRestoreParams &params,
                                                        core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(LOG_LIKELIHOOD_TAG,
                               double logLikelihood,
                               core::CStringUtils::stringToType(traverser.value(), logLikelihood),
                               this->addLogLikelihood(logLikelihood))
        RESTORE(SHIFT_TAG, m_Shift.fromDelimited(traverser.value()))
        RESTORE_BUILT_IN(RESIDUAL_MODEL_MODE_TAG, m_ResidualModelMode)
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE(RESIDUAL_MODEL_TAG, this->restoreResidualModel(params.s_DistributionParams, traverser))
    }
    while (traverser.next());
    return true;
}

void CUnivariateLevelShiftModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LOG_LIKELIHOOD_TAG, this->logLikelihood());
    inserter.insertValue(SHIFT_TAG, m_Shift.toDelimited());
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertLevel(RESIDUAL_MODEL_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                               boost::cref(this->residualModel()), _1));
}

double CUnivariateLevelShiftModel::bic() const
{
    return -2.0 * this->logLikelihood() + std::log(m_SampleCount);
}

TOptionalChangeDescription CUnivariateLevelShiftModel::change() const
{
    // The "magic" 0.9 is due to the fact that the trend is updated
    // with new values during change detection. As a result, the
    // estimate is biased (by early values) and too large. This was
    // an empirical estimate of the degree of bias across a range of
    // step changes.
    return SChangeDescription{SChangeDescription::E_LevelShift,
                              0.9 * CBasicStatistics::mean(m_Shift),
                              this->residualModelPtr()};
}

void CUnivariateLevelShiftModel::addSamples(std::size_t count,
                                            const TWeightStyleVec &weightStyles,
                                            const TTimeDoublePr1Vec &samples_,
                                            const TDouble4Vec1Vec &weights)
{
    const CTimeSeriesDecompositionInterface &trendModel{this->trendModel()};

    for (const auto &sample : samples_)
    {
        double x{trendModel.detrend(sample.first, sample.second, 0.0) - m_ResidualModelMode};
        m_Shift.add(x);
    }

    // We delay updating the log-likelihood because early on the
    // level can change giving us a better apparent fit to the
    // data than a fixed step. Five updates was found to be the
    // minimum to get empirically similar sum log-likelihood if
    // there is no shift in the data.

    if (count >= COUNT_TO_INITIALIZE)
    {
        TDouble1Vec samples;
        samples.reserve(samples_.size());
        for (const auto &sample : samples_)
        {
            double shift{CBasicStatistics::mean(m_Shift)};
            samples.push_back(trendModel.detrend(sample.first, sample.second, 0.0) - shift);
        }
        for (const auto &weight : weights)
        {
            m_SampleCount += maths_t::count(weightStyles, weight);
        }

        CPrior &residualModel{this->residualModel()};
        residualModel.addSamples(weightStyles, samples, weights);
        residualModel.propagateForwardsByTime(1.0);

        double logLikelihood;
        if (residualModel.jointLogMarginalLikelihood(weightStyles, samples, weights,
                                                     logLikelihood) == maths_t::E_FpNoErrors)
        {
            this->addLogLikelihood(logLikelihood);
        }
    }
}

std::size_t CUnivariateLevelShiftModel::staticSize() const
{
    return sizeof(*this);
}

uint64_t CUnivariateLevelShiftModel::checksum(uint64_t seed) const
{
    seed = this->CUnivariateChangeModel::checksum(seed);
    seed = CChecksum::calculate(seed, m_Shift);
    return CChecksum::calculate(seed, m_SampleCount);
}

CUnivariateTimeShiftModel::CUnivariateTimeShiftModel(const TDecompositionPtr &trendModel,
                                                     const TPriorPtr &residualModel,
                                                     core_t::TTime shift) :
        CUnivariateChangeModel{trendModel, TPriorPtr{residualModel->clone()}},
        m_Shift{shift}
{}

bool CUnivariateTimeShiftModel::acceptRestoreTraverser(const SModelRestoreParams &params,
                                                       core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(LOG_LIKELIHOOD_TAG,
                               double logLikelihood,
                               core::CStringUtils::stringToType(traverser.value(), logLikelihood),
                               this->addLogLikelihood(logLikelihood))
        RESTORE(RESIDUAL_MODEL_TAG, this->restoreResidualModel(params.s_DistributionParams, traverser))
    }
    while (traverser.next());
    return true;
}

void CUnivariateTimeShiftModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LOG_LIKELIHOOD_TAG, this->logLikelihood());
    inserter.insertLevel(RESIDUAL_MODEL_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                               boost::cref(this->residualModel()), _1));
}

double CUnivariateTimeShiftModel::bic() const
{
    return -2.0 * this->logLikelihood();
}

TOptionalChangeDescription CUnivariateTimeShiftModel::change() const
{
    return SChangeDescription{SChangeDescription::E_TimeShift,
                              static_cast<double>(m_Shift),
                              this->residualModelPtr()};
}

void CUnivariateTimeShiftModel::addSamples(std::size_t count,
                                           const TWeightStyleVec &weightStyles,
                                           const TTimeDoublePr1Vec &samples_,
                                           const TDouble4Vec1Vec &weights)
{
    // See CUnivariateTimeSeriesLevelShiftModel for an explanation
    // of the delay updating the log-likelihood.

    if (count >= COUNT_TO_INITIALIZE)
    {
        TDouble1Vec samples;
        samples.reserve(samples_.size());
        for (const auto &sample : samples_)
        {
            samples.push_back(this->trendModel().detrend(sample.first + m_Shift, sample.second, 0.0));
        }

        CPrior &residualModel{this->residualModel()};
        residualModel.addSamples(weightStyles, samples, weights);
        residualModel.propagateForwardsByTime(1.0);

        double logLikelihood;
        if (residualModel.jointLogMarginalLikelihood(weightStyles, samples, weights,
                                                     logLikelihood) == maths_t::E_FpNoErrors)
        {
            this->addLogLikelihood(logLikelihood);
        }
    }
}

std::size_t CUnivariateTimeShiftModel::staticSize() const
{
    return sizeof(*this);
}

uint64_t CUnivariateTimeShiftModel::checksum(uint64_t seed) const
{
    seed = this->CUnivariateChangeModel::checksum(seed);
    return CChecksum::calculate(seed, m_Shift);
}

}

}
}
