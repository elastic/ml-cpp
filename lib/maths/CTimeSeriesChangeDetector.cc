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

const std::string SAMPLE_COUNT_TAG{"a"};
const std::string MIN_TIME_TAG{"b"};
const std::string MAX_TIME_TAG{"c"};
const std::string CHANGE_MODEL_TAG{"d"};
const std::string LOG_LIKELIHOOD_TAG{"e"};
const std::string SHIFT_TAG{"f"};
const std::string RESIDUAL_MODEL_TAG{"g"};
}

SChangeDescription::SChangeDescription(EDescription description, double value) :
        s_Description{description}, s_Value{value}
{}

CUnivariateTimeSeriesChangeDetector::CUnivariateTimeSeriesChangeDetector(const CTimeSeriesDecompositionInterface &trendModel,
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
                       boost::make_shared<CUnivariateTimeSeriesLevelShiftModel>(trendModel, residualModel),
                       boost::make_shared<CUnivariateTimeSeriesTimeShiftModel>(trendModel, residualModel, -core::constants::HOUR),
                       boost::make_shared<CUnivariateTimeSeriesTimeShiftModel>(trendModel, residualModel, +core::constants::HOUR)}
{}

bool CUnivariateTimeSeriesChangeDetector::acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                                                 core::CStateRestoreTraverser &traverser)
{
    auto model = m_ChangeModels.begin();
    do
    {
        const std::string name{traverser.name()};
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE_SETUP_TEARDOWN(MIN_TIME_TAG,
                               core_t::TTime time,
                               core::CStringUtils::stringToType(traverser.value(), time),
                               m_TimeRange.add(time))
        RESTORE_SETUP_TEARDOWN(MAX_TIME_TAG,
                               core_t::TTime time,
                               core::CStringUtils::stringToType(traverser.value(), time),
                               m_TimeRange.add(time))
        RESTORE(CHANGE_MODEL_TAG, traverser.traverseSubLevel(boost::bind(
                                      &CUnivariateTimeSeriesChangeModel::acceptRestoreTraverser,
                                      (model++)->get(), boost::cref(params), _1)))
    }
    while (traverser.next());
    return true;
}

void CUnivariateTimeSeriesChangeDetector::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertValue(MIN_TIME_TAG, m_TimeRange.min());
    inserter.insertValue(MAX_TIME_TAG, m_TimeRange.max());
    for (const auto &model : m_ChangeModels)
    {
        inserter.insertLevel(CHANGE_MODEL_TAG,
                             boost::bind(&CUnivariateTimeSeriesChangeModel::acceptPersistInserter,
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
                                                       / m_MinimumDeltaBicToDetect, 0.2)};
        return  static_cast<double>(range)
              > m_MinimumTimeToDetect + scale * static_cast<double>(
                                                    m_MaximumTimeToDetect - m_MinimumTimeToDetect);
    }
    return false;
}
void CUnivariateTimeSeriesChangeDetector::addSamples(maths_t::EDataType dataType,
                                                     const TWeightStyleVec &weightStyles,
                                                     const TTimeDoublePr1Vec &samples,
                                                     const TDouble4Vec1Vec &weights,
                                                     double propagationInterval)
{
    for (const auto &sample : samples)
    {
        m_TimeRange.add(sample.first);
    }

    ++m_SampleCount;

    for (auto &model : m_ChangeModels)
    {
        model->addSamples(m_SampleCount, dataType,
                          weightStyles, samples, weights,
                          propagationInterval);
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
    seed = CChecksum::calculate(seed, m_TimeRange);
    seed = CChecksum::calculate(seed, m_SampleCount);
    return CChecksum::calculate(seed, m_ChangeModels);
}

namespace time_series_change_detector_detail
{

CUnivariateTimeSeriesChangeModel::CUnivariateTimeSeriesChangeModel(const CTimeSeriesDecompositionInterface &trendModel) :
        m_LogLikelihood{0.0}, m_TrendModel{trendModel}
{}

double CUnivariateTimeSeriesChangeModel::logLikelihood() const
{
    return m_LogLikelihood;
}

void CUnivariateTimeSeriesChangeModel::addLogLikelihood(double logLikelihood)
{
    m_LogLikelihood += logLikelihood;
}

const CTimeSeriesDecompositionInterface &CUnivariateTimeSeriesChangeModel::trendModel() const
{
    return m_TrendModel;
}

CUnivariateNoChangeModel::CUnivariateNoChangeModel(const CTimeSeriesDecompositionInterface &trendModel,
                                                   const TPriorPtr &residualModel) :
        CUnivariateTimeSeriesChangeModel{trendModel},
        m_ResidualModel{residualModel}
{}

bool CUnivariateNoChangeModel::acceptRestoreTraverser(const SDistributionRestoreParams &/*params*/,
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
                                          maths_t::EDataType /*dataType*/,
                                          const TWeightStyleVec &weightStyles,
                                          const TTimeDoublePr1Vec &samples_,
                                          const TDouble4Vec1Vec &weights,
                                          double /*propagationInterval*/)
{
    TDouble1Vec samples;
    samples.reserve(samples_.size());
    for (const auto &sample : samples_)
    {
        samples.push_back(this->trendModel().detrend(sample.first, sample.second, 0.0));
    }

    // See CUnivariateTimeSeriesLevelShiftModel for an explanation
    // of the delay updating the log-likelihood.

    double logLikelihood;
    if (count >= 5 && m_ResidualModel->jointLogMarginalLikelihood(
                          weightStyles, samples, weights,
                          logLikelihood) == maths_t::E_FpNoErrors)
    {
        this->addLogLikelihood(logLikelihood);
    }
}

void CUnivariateNoChangeModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr /*mem*/) const
{
}

std::size_t CUnivariateNoChangeModel::staticSize() const
{
    return sizeof(*this);
}

std::size_t CUnivariateNoChangeModel::memoryUsage() const
{
    return 0;
}

uint64_t CUnivariateNoChangeModel::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, this->logLikelihood());
    seed = CChecksum::calculate(seed, this->trendModel());
    return CChecksum::calculate(seed, m_ResidualModel);
}

CUnivariateTimeSeriesLevelShiftModel::CUnivariateTimeSeriesLevelShiftModel(const CTimeSeriesDecompositionInterface &trendModel,
                                                                           const TPriorPtr &residualModel) :
        CUnivariateTimeSeriesChangeModel{trendModel},
        m_SampleCount{0.0},
        m_ResidualModel{residualModel->clone()},
        m_ResidualModelMode{residualModel->marginalLikelihoodMode()}
{}

bool CUnivariateTimeSeriesLevelShiftModel::acceptRestoreTraverser(const SDistributionRestoreParams &params,
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
        RESTORE_BUILT_IN(SAMPLE_COUNT_TAG, m_SampleCount)
        RESTORE(RESIDUAL_MODEL_TAG, traverser.traverseSubLevel(
                                        boost::bind<bool>(CPriorStateSerialiser(),
                                                          boost::cref(params),
                                                          boost::ref(m_ResidualModel), _1)))
    }
    while (traverser.next());
    return true;
}

void CUnivariateTimeSeriesLevelShiftModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LOG_LIKELIHOOD_TAG, this->logLikelihood());
    inserter.insertValue(SHIFT_TAG, m_Shift.toDelimited());
    inserter.insertValue(SAMPLE_COUNT_TAG, m_SampleCount);
    inserter.insertLevel(RESIDUAL_MODEL_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                               boost::cref(*m_ResidualModel), _1));
}

double CUnivariateTimeSeriesLevelShiftModel::bic() const
{
    return -2.0 * this->logLikelihood() + std::log(m_SampleCount);
}

TOptionalChangeDescription CUnivariateTimeSeriesLevelShiftModel::change() const
{
    return SChangeDescription{SChangeDescription::E_LevelShift, CBasicStatistics::mean(m_Shift)};
}

void CUnivariateTimeSeriesLevelShiftModel::addSamples(std::size_t count,
                                                      maths_t::EDataType dataType,
                                                      const TWeightStyleVec &weightStyles,
                                                      const TTimeDoublePr1Vec &samples_,
                                                      const TDouble4Vec1Vec &weights,
                                                      double propagationInterval)
{
    TDouble1Vec samples;
    samples.reserve(samples_.size());
    for (const auto &sample : samples_)
    {
        double x{this->trendModel().detrend(sample.first, sample.second, 0.0)};
        samples.push_back(x);
        m_Shift.add(x - m_ResidualModelMode);
    }
    for (auto &sample : samples)
    {
        sample -= CBasicStatistics::mean(m_Shift);
    }
    for (const auto &weight : weights)
    {
        m_SampleCount += maths_t::count(weightStyles, weight);
    }

    m_ResidualModel->dataType(dataType);
    m_ResidualModel->addSamples(weightStyles, samples, weights);
    m_ResidualModel->propagateForwardsByTime(propagationInterval);

    // We delay updating the log-likelihood because early on the
    // level can change giving us a better apparent fit to the
    // data than a fixed step. Five updates was found to be the
    // minimum to get empirically similar sum log-likelihood if
    // there is no shift in the data.

    double logLikelihood;
    if (count >= 5 && m_ResidualModel->jointLogMarginalLikelihood(
                          weightStyles, samples, weights,
                          logLikelihood) == maths_t::E_FpNoErrors)
    {
        this->addLogLikelihood(logLikelihood);
    }
}

void CUnivariateTimeSeriesLevelShiftModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    core::CMemoryDebug::dynamicSize("m_ResidualModel", m_ResidualModel, mem);
}

std::size_t CUnivariateTimeSeriesLevelShiftModel::staticSize() const
{
    return sizeof(*this);
}

std::size_t CUnivariateTimeSeriesLevelShiftModel::memoryUsage() const
{
    return core::CMemory::dynamicSize(m_ResidualModel);
}

uint64_t CUnivariateTimeSeriesLevelShiftModel::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, this->logLikelihood());
    seed = CChecksum::calculate(seed, this->trendModel());
    seed = CChecksum::calculate(seed, m_Shift);
    seed = CChecksum::calculate(seed, m_SampleCount);
    return CChecksum::calculate(seed, m_ResidualModel);
}

CUnivariateTimeSeriesTimeShiftModel::CUnivariateTimeSeriesTimeShiftModel(const CTimeSeriesDecompositionInterface &trendModel,
                                                                         const TPriorPtr &residualModel,
                                                                         core_t::TTime shift) :
        CUnivariateTimeSeriesChangeModel{trendModel},
        m_Shift{shift},
        m_ResidualModel{residualModel->clone()}
{}

bool CUnivariateTimeSeriesTimeShiftModel::acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                                                  core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(LOG_LIKELIHOOD_TAG,
                               double logLikelihood,
                               core::CStringUtils::stringToType(traverser.value(), logLikelihood),
                               this->addLogLikelihood(logLikelihood))
        RESTORE(RESIDUAL_MODEL_TAG, traverser.traverseSubLevel(
                                        boost::bind<bool>(CPriorStateSerialiser(),
                                                          boost::cref(params),
                                                          boost::ref(m_ResidualModel), _1)))
    }
    while (traverser.next());
    return true;
}

void CUnivariateTimeSeriesTimeShiftModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LOG_LIKELIHOOD_TAG, this->logLikelihood());
    inserter.insertLevel(RESIDUAL_MODEL_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                               boost::cref(*m_ResidualModel), _1));
}

double CUnivariateTimeSeriesTimeShiftModel::bic() const
{
    return -2.0 * this->logLikelihood();
}

TOptionalChangeDescription CUnivariateTimeSeriesTimeShiftModel::change() const
{
    return SChangeDescription{SChangeDescription::E_TimeShift, static_cast<double>(m_Shift)};
}

void CUnivariateTimeSeriesTimeShiftModel::addSamples(std::size_t count,
                                                     maths_t::EDataType dataType,
                                                     const TWeightStyleVec &weightStyles,
                                                     const TTimeDoublePr1Vec &samples_,
                                                     const TDouble4Vec1Vec &weights,
                                                     double propagationInterval)
{
    TDouble1Vec samples;
    samples.reserve(samples_.size());
    for (const auto &sample : samples_)
    {
        samples.push_back(this->trendModel().detrend(sample.first + m_Shift, sample.second, 0.0));
    }

    m_ResidualModel->dataType(dataType);
    m_ResidualModel->addSamples(weightStyles, samples, weights);
    m_ResidualModel->propagateForwardsByTime(propagationInterval);

    // See CUnivariateTimeSeriesLevelShiftModel for an explanation
    // of the delay updating the log-likelihood.

    double logLikelihood;
    if (count >= 5 && m_ResidualModel->jointLogMarginalLikelihood(
                          weightStyles, samples, weights,
                          logLikelihood) == maths_t::E_FpNoErrors)
    {
        this->addLogLikelihood(logLikelihood);
    }
}

void CUnivariateTimeSeriesTimeShiftModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    core::CMemoryDebug::dynamicSize("m_ResidualModel", m_ResidualModel, mem);
}

std::size_t CUnivariateTimeSeriesTimeShiftModel::staticSize() const
{
    return sizeof(*this);
}

std::size_t CUnivariateTimeSeriesTimeShiftModel::memoryUsage() const
{
    return core::CMemory::dynamicSize(m_ResidualModel);
}

uint64_t CUnivariateTimeSeriesTimeShiftModel::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, this->logLikelihood());
    seed = CChecksum::calculate(seed, this->trendModel());
    seed = CChecksum::calculate(seed, m_Shift);
    return CChecksum::calculate(seed, m_ResidualModel);
}

}

}
}
