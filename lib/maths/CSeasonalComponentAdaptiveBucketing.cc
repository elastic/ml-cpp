/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CSeasonalComponentAdaptiveBucketing.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CRegression.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace ml
{
namespace maths
{
namespace
{

using TRegression = CSeasonalComponentAdaptiveBucketing::TRegression;

const double SUFFICIENT_HISTORY_TO_PREDICT{2.5};

//! Clear a vector and recover its memory.
template<typename T>
void clearAndShrink(std::vector<T> &vector)
{
    std::vector<T> empty;
    empty.swap(vector);
}

//! Get the predicted value of \p r at \p t.
double predict_(const TRegression &r, double t, double age)
{
    return age < SUFFICIENT_HISTORY_TO_PREDICT ? r.mean() : CRegression::predict(r, t);
}

const std::string ADAPTIVE_BUCKETING_TAG{"a"};
const std::string TIME_TAG{"b"};
const std::string INITIAL_TIME_TAG{"c"};
const std::string REGRESSION_TAG{"d"};
const std::string VARIANCES_TAG{"e"};
const std::string LAST_UPDATES_TAG{"f"};
const std::string PARAMETER_PROCESS_TAG{"g"};
const std::string EMPTY_STRING;
const core_t::TTime UNSET_LAST_UPDATE{0};

}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(void) :
        CAdaptiveBucketing{0.0, 0.0},
        m_InitialTime{boost::numeric::bounds<core_t::TTime>::lowest()}
{}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(const CSeasonalTime &time,
                                                                         double decayRate,
                                                                         double minimumBucketLength) :
        CAdaptiveBucketing{decayRate, minimumBucketLength},
        m_Time{time.clone()},
        m_InitialTime{boost::numeric::bounds<core_t::TTime>::lowest()}
{}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(const CSeasonalComponentAdaptiveBucketing &other) :
        CAdaptiveBucketing(other),
        m_Time{other.m_Time->clone()},
        m_InitialTime{other.m_InitialTime},
        m_Regressions(other.m_Regressions),
        m_Variances(other.m_Variances),
        m_LastUpdates(other.m_LastUpdates),
        m_ParameterProcess(other.m_ParameterProcess)
{}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(double decayRate,
                                                                         double minimumBucketLength,
                                                                         core::CStateRestoreTraverser &traverser) :
        CAdaptiveBucketing{decayRate, minimumBucketLength}
{
    traverser.traverseSubLevel(boost::bind(&CSeasonalComponentAdaptiveBucketing::acceptRestoreTraverser, this, _1));
}

const CSeasonalComponentAdaptiveBucketing &
CSeasonalComponentAdaptiveBucketing::operator=(const CSeasonalComponentAdaptiveBucketing &rhs)
{
    if (&rhs != this)
    {
        CSeasonalComponentAdaptiveBucketing tmp(rhs);
        this->swap(tmp);
    }
    return *this;
}

void CSeasonalComponentAdaptiveBucketing::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(ADAPTIVE_BUCKETING_TAG,
                         boost::bind(&CAdaptiveBucketing::acceptPersistInserter,
                                     static_cast<const CAdaptiveBucketing*>(this), _1));
    inserter.insertLevel(TIME_TAG, boost::bind(&CSeasonalTimeStateSerializer::acceptPersistInserter,
                                               boost::cref(*m_Time), _1));
    inserter.insertValue(INITIAL_TIME_TAG, m_InitialTime);
    for (const auto &regression : m_Regressions)
    {
        inserter.insertLevel(REGRESSION_TAG, boost::bind(&TRegression::acceptPersistInserter,
                                                         &regression, _1));
    }
    inserter.insertValue(VARIANCES_TAG, core::CPersistUtils::toString(m_Variances));
    inserter.insertValue(LAST_UPDATES_TAG, core::CPersistUtils::toString(m_LastUpdates));
    inserter.insertLevel(PARAMETER_PROCESS_TAG, boost::bind(&TRegressionParameterProcess::acceptPersistInserter,
                                                            &m_ParameterProcess, _1));
}

void CSeasonalComponentAdaptiveBucketing::swap(CSeasonalComponentAdaptiveBucketing &other)
{
    this->CAdaptiveBucketing::swap(other);
    m_Time.swap(other.m_Time);
    std::swap(m_InitialTime, other.m_InitialTime);
    m_Regressions.swap(other.m_Regressions);
    m_Variances.swap(other.m_Variances);
    m_LastUpdates.swap(other.m_LastUpdates);
    std::swap(m_ParameterProcess, other.m_ParameterProcess);
}

bool CSeasonalComponentAdaptiveBucketing::initialized(void) const
{
    return this->CAdaptiveBucketing::initialized();
}

bool CSeasonalComponentAdaptiveBucketing::initialize(std::size_t n)
{
    double a{0.0};
    double b{static_cast<double>(std::min(this->time().windowLength(),
                                          this->time().period()))};

    if (this->CAdaptiveBucketing::initialize(a, b, n))
    {
        n = this->size();
        m_Regressions.clear();
        m_Regressions.resize(n);
        m_Variances.clear();
        m_Variances.resize(n);
        m_LastUpdates.clear();
        m_LastUpdates.resize(n, UNSET_LAST_UPDATE);
        return true;
    }
    return false;
}

void CSeasonalComponentAdaptiveBucketing::initialValues(core_t::TTime startTime,
                                                        core_t::TTime endTime,
                                                        const TTimeTimePrMeanVarPrVec &values)
{
    if (this->initialized())
    {
        this->shiftOrigin(startTime);
        m_InitialTime = m_Time->startOfWindowRepeat(endTime);
        this->CAdaptiveBucketing::initialValues(startTime, values);
        m_LastUpdates.assign(this->size(), endTime);
    }
}

std::size_t CSeasonalComponentAdaptiveBucketing::size(void) const
{
    return this->CAdaptiveBucketing::size();
}

void CSeasonalComponentAdaptiveBucketing::clear(void)
{
    this->CAdaptiveBucketing::clear();
    clearAndShrink(m_Regressions);
    clearAndShrink(m_Variances);
    clearAndShrink(m_LastUpdates);
    m_ParameterProcess = TRegressionParameterProcess();
}

void CSeasonalComponentAdaptiveBucketing::shiftOrigin(core_t::TTime time)
{
    time = CIntegerTools::floor(time, core::constants::WEEK);
    double shift{m_Time->regression(time)};
    if (shift > 0.0)
    {
        for (auto &&regression : m_Regressions)
        {
            regression.shiftAbscissa(-shift);
        }
        m_Time->regressionOrigin(time);
    }
}

void CSeasonalComponentAdaptiveBucketing::shiftLevel(double shift)
{
    for (auto &&regression : m_Regressions)
    {
        regression.shiftOrdinate(shift);
    }
}

void CSeasonalComponentAdaptiveBucketing::shiftSlope(double shift)
{
    for (auto &&regression : m_Regressions)
    {
        regression.shiftGradient(shift);
    }
}

void CSeasonalComponentAdaptiveBucketing::add(core_t::TTime time,
                                              double value,
                                              double prediction,
                                              double weight)
{
    std::size_t bucket{0};
    if (!this->initialized() || !this->bucket(time, bucket))
    {
        return;
    }

    using TVector = CVectorNx1<double, 2>;

    this->CAdaptiveBucketing::add(bucket, time, weight);

    double t{m_Time->regression(time)};
    TRegression &regression{m_Regressions[bucket]};

    TDoubleMeanVarAccumulator moments =
            CBasicStatistics::momentsAccumulator(regression.count(),
                                                 prediction,
                                                 static_cast<double>(m_Variances[bucket]));
    moments.add(value, weight * weight);

    // Note this condition can change as a result adding the new
    // value we need to check before as well.
    bool sufficientHistoryBeforeUpdate{regression.sufficientHistoryToPredict()};
    TVector paramsDrift(regression.parameters(t));

    regression.add(t, value, weight);
    m_Variances[bucket] = CBasicStatistics::maximumLikelihoodVariance(moments);

    paramsDrift -= TVector(regression.parameters(t));

    if (   sufficientHistoryBeforeUpdate
        && regression.sufficientHistoryToPredict()
        && m_LastUpdates[bucket] != UNSET_LAST_UPDATE)
    {
        double interval{m_Time->regressionInterval(m_LastUpdates[bucket], time)};
        m_ParameterProcess.add(interval, paramsDrift, TVector(weight * interval));
    }
    m_LastUpdates[bucket] = time;
}

const CSeasonalTime &CSeasonalComponentAdaptiveBucketing::time(void) const
{
    return *m_Time;
}

void CSeasonalComponentAdaptiveBucketing::decayRate(double value)
{
    this->CAdaptiveBucketing::decayRate(value);
}

double CSeasonalComponentAdaptiveBucketing::decayRate(void) const
{
    return this->CAdaptiveBucketing::decayRate();
}

void CSeasonalComponentAdaptiveBucketing::propagateForwardsByTime(double time, bool meanRevert)
{
    if (time < 0.0)
    {
        LOG_ERROR("Can't propagate bucketing backwards in time");
    }
    else if (this->initialized())
    {
        double factor{std::exp(-this->CAdaptiveBucketing::decayRate() * time)};
        this->CAdaptiveBucketing::age(factor);
        for (auto &&regression : m_Regressions)
        {
            regression.age(factor, meanRevert);
        }
        m_ParameterProcess.age(factor);
    }
}

double CSeasonalComponentAdaptiveBucketing::minimumBucketLength(void) const
{
    return this->CAdaptiveBucketing::minimumBucketLength();
}

void CSeasonalComponentAdaptiveBucketing::refine(core_t::TTime time)
{
    this->CAdaptiveBucketing::refine(time);
}

double CSeasonalComponentAdaptiveBucketing::count(core_t::TTime time) const
{
    const TRegression *regression = this->regression(time);
    return regression ? regression->count() : 0.0;
}

const TRegression *CSeasonalComponentAdaptiveBucketing::regression(core_t::TTime time) const
{
    const TRegression *result{0};
    if (this->initialized())
    {
        std::size_t bucket{0};
        this->bucket(time, bucket);
        bucket = CTools::truncate(bucket, std::size_t(0), m_Regressions.size() - 1);
        result = &m_Regressions[bucket];
    }
    return result;
}

bool CSeasonalComponentAdaptiveBucketing::knots(core_t::TTime time,
                                                CSplineTypes::EBoundaryCondition boundary,
                                                TDoubleVec &knots,
                                                TDoubleVec &values,
                                                TDoubleVec &variances) const
{
    return this->CAdaptiveBucketing::knots(time, boundary, knots, values, variances);
}

double CSeasonalComponentAdaptiveBucketing::slope(void) const
{
    CBasicStatistics::CMinMax<double> minmax;
    for (const auto &regression : m_Regressions)
    {
        if (regression.count() > 0.0)
        {
            TRegression::TArray params;
            regression.parameters(params);
            minmax.add(params[1]);
        }
    }
    return minmax.initialized() ? minmax.signMargin() : 0.0;
}

bool CSeasonalComponentAdaptiveBucketing::sufficientHistoryToPredict(core_t::TTime time) const
{
    return this->bucketingAgeAt(time) >= SUFFICIENT_HISTORY_TO_PREDICT;
}

double CSeasonalComponentAdaptiveBucketing::varianceDueToParameterDrift(core_t::TTime time) const
{
    double result{0.0};
    if (this->initialized())
    {
        core_t::TTime last{*std::max_element(m_LastUpdates.begin(), m_LastUpdates.end())};
        core_t::TTime current{std::max(time - m_Time->period(), last)};
        if (current > last)
        {
            double interval{m_Time->regressionInterval(last, current)};
            result = m_ParameterProcess.predictionVariance(interval);
        }
    }
    return result;
}

uint64_t CSeasonalComponentAdaptiveBucketing::checksum(uint64_t seed) const
{
    seed = this->CAdaptiveBucketing::checksum(seed);
    seed = CChecksum::calculate(seed, m_Time);
    seed = CChecksum::calculate(seed, m_InitialTime);
    seed = CChecksum::calculate(seed, m_Regressions);
    seed = CChecksum::calculate(seed, m_Variances);
    seed = CChecksum::calculate(seed, m_LastUpdates);
    return CChecksum::calculate(seed, m_ParameterProcess);
}

void CSeasonalComponentAdaptiveBucketing::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CSeasonalComponentAdaptiveBucketing");
    core::CMemoryDebug::dynamicSize("m_Endpoints", this->CAdaptiveBucketing::endpoints(), mem);
    core::CMemoryDebug::dynamicSize("m_Centres", this->CAdaptiveBucketing::centres(), mem);
    core::CMemoryDebug::dynamicSize("m_Regressions", m_Regressions, mem);
    core::CMemoryDebug::dynamicSize("m_Variances", m_Variances, mem);
    core::CMemoryDebug::dynamicSize("m_LastUpdates", m_LastUpdates, mem);
}

std::size_t CSeasonalComponentAdaptiveBucketing::memoryUsage(void) const
{
    std::size_t mem{this->CAdaptiveBucketing::memoryUsage()};
    mem += core::CMemory::dynamicSize(m_Regressions);
    mem += core::CMemory::dynamicSize(m_Variances);
    mem += core::CMemory::dynamicSize(m_LastUpdates);
    return mem;
}

const CSeasonalComponentAdaptiveBucketing::TFloatVec &CSeasonalComponentAdaptiveBucketing::endpoints(void) const
{
    return this->CAdaptiveBucketing::endpoints();
}

double CSeasonalComponentAdaptiveBucketing::count(void) const
{
    return this->CAdaptiveBucketing::count();
}

CSeasonalComponentAdaptiveBucketing::TDoubleVec CSeasonalComponentAdaptiveBucketing::values(core_t::TTime time) const
{
    return this->CAdaptiveBucketing::values(time);
}

CSeasonalComponentAdaptiveBucketing::TDoubleVec CSeasonalComponentAdaptiveBucketing::variances(void) const
{
    return this->CAdaptiveBucketing::variances();
}

bool CSeasonalComponentAdaptiveBucketing::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(ADAPTIVE_BUCKETING_TAG, traverser.traverseSubLevel(
                    boost::bind(&CAdaptiveBucketing::acceptRestoreTraverser,
                                static_cast<CAdaptiveBucketing*>(this), _1)));
        RESTORE(TIME_TAG, traverser.traverseSubLevel(
                    boost::bind(&CSeasonalTimeStateSerializer::acceptRestoreTraverser, boost::ref(m_Time), _1)))
        RESTORE_BUILT_IN(INITIAL_TIME_TAG, m_InitialTime)
        RESTORE_SETUP_TEARDOWN(REGRESSION_TAG,
                               TRegression regression,
                               traverser.traverseSubLevel(boost::bind(&TRegression::acceptRestoreTraverser,
                                                                      &regression, _1)),
                               m_Regressions.push_back(regression))
        RESTORE(VARIANCES_TAG, core::CPersistUtils::fromString(traverser.value(), m_Variances))
        RESTORE(LAST_UPDATES_TAG, core::CPersistUtils::fromString(traverser.value(), m_LastUpdates))
        RESTORE(PARAMETER_PROCESS_TAG, traverser.traverseSubLevel(
                    boost::bind(&TRegressionParameterProcess::acceptRestoreTraverser, &m_ParameterProcess, _1)))
    }
    while (traverser.next());

    TRegressionVec(m_Regressions).swap(m_Regressions);
    if (m_LastUpdates.empty())
    {
        m_LastUpdates.resize(this->size(), UNSET_LAST_UPDATE);
    }

    return true;
}

void CSeasonalComponentAdaptiveBucketing::refresh(const TFloatVec &endpoints)
{
    // Values are assigned based on their intersection with each
    // bucket in the previous configuration. The regression and
    // variance are computed using the appropriate combination
    // rules. Note that the count is weighted by the square of
    // the fractional intersection between the old and new buckets.
    // This means that the effective weight of buckets whose end
    // points change significantly is reduced. This is reasonable
    // because the periodic trend is assumed to be unchanging
    // throughout the interval, when of course it is varying, so
    // adjusting the end points introduces error in the bucket
    // value, which we handle by reducing its significance in the
    // new bucket values.
    //
    // A better approximation is to assume that it the trend is
    // continuous. In fact, this can be done by using a spline
    // with the constraint that the mean of the spline in each
    // interval is equal to the mean value. We can additionally
    // decompose the variance into a contribution from noise and
    // a contribution from the trend. Under these assumptions it
    // is then possible (but not trivial) to update the bucket
    // means and variances based on the new end point positions.
    // This might be worth considering at some point.

    using TDoubleMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    std::size_t m{m_Regressions.size()};
    std::size_t n{endpoints.size()};
    if (m+1 != n)
    {
        LOG_ERROR("Inconsistent end points and regressions");
        return;
    }

    TFloatVec &m_Endpoints{this->CAdaptiveBucketing::endpoints()};
    TFloatVec &m_Centres{this->CAdaptiveBucketing::centres()};

    TRegressionVec regressions;
    TFloatVec centres;
    TFloatVec variances;
    regressions.reserve(m);
    centres.reserve(m);
    variances.reserve(m);

    for (std::size_t i = 1u; i < n; ++i)
    {
        double yl{m_Endpoints[i-1]};
        double yr{m_Endpoints[i]};
        std::size_t r = std::lower_bound(endpoints.begin(),
                                         endpoints.end(), yr) - endpoints.begin();
        r = CTools::truncate(r, std::size_t(1), n - 1);

        std::size_t l = std::upper_bound(endpoints.begin(),
                                         endpoints.end(), yl) - endpoints.begin();
        l = CTools::truncate(l, std::size_t(1), r);

        LOG_TRACE("interval = [" << yl << "," << yr << "]");
        LOG_TRACE("l = " << l << ", r = " << r);
        LOG_TRACE("[x(l), x(r)] = [" << endpoints[l-1] << "," << endpoints[r] << "]");

        double xl{endpoints[l-1]};
        double xr{endpoints[l]};
        if (l == r)
        {
            double interval{m_Endpoints[i] - m_Endpoints[i-1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            regressions.push_back(m_Regressions[l-1].scaled(w * w));
            centres.push_back(CTools::truncate(static_cast<double>(m_Centres[l-1]), yl, yr));
            variances.push_back(m_Variances[l-1]);
        }
        else
        {
            double interval{xr - m_Endpoints[i-1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            TDoubleRegression regression{m_Regressions[l-1].scaled(w)};
            TDoubleMeanAccumulator centre{
                    CBasicStatistics::momentsAccumulator(w * m_Regressions[l-1].count(),
                                                         static_cast<double>(m_Centres[l-1]))};
            TDoubleMeanVarAccumulator variance{
                    CBasicStatistics::momentsAccumulator(w * m_Regressions[l-1].count(),
                                                         m_Regressions[l-1].mean(),
                                                         static_cast<double>(m_Variances[l-1]))};
            double count{w * w * m_Regressions[l-1].count()};
            while (++l < r)
            {
                regression += m_Regressions[l-1];
                centre += CBasicStatistics::momentsAccumulator(m_Regressions[l-1].count(),
                                                               static_cast<double>(m_Centres[l-1]));
                variance += CBasicStatistics::momentsAccumulator(m_Regressions[l-1].count(),
                                                                 m_Regressions[l-1].mean(),
                                                                 static_cast<double>(m_Variances[l-1]));
                count += m_Regressions[l-1].count();
            }
            xl = endpoints[l-1];
            xr = endpoints[l];
            interval = m_Endpoints[i] - xl;
            w = CTools::truncate(interval / (xr - xl), 0.0, 1.0);
            regression += m_Regressions[l-1].scaled(w);
            centre += CBasicStatistics::momentsAccumulator(w * m_Regressions[l-1].count(),
                                                           static_cast<double>(m_Centres[l-1]));
            variance += CBasicStatistics::momentsAccumulator(w * m_Regressions[l-1].count(),
                                                             m_Regressions[l-1].mean(),
                                                             static_cast<double>(m_Variances[l-1]));
            count += w * w * m_Regressions[l-1].count();
            double scale{count == regression.count() ? 1.0 : count / regression.count()};
            regressions.push_back(regression.scaled(scale));
            centres.push_back(CTools::truncate(CBasicStatistics::mean(centre), yl, yr));
            variances.push_back(CBasicStatistics::maximumLikelihoodVariance(variance));
        }
    }

    // We want all regressions to respond at the same rate to changes
    // in the trend. To achieve this we should assign them a weight
    // that is equal to the number of points they will receive in one
    // period.
    double count{0.0};
    for (const auto &regression : regressions)
    {
        count += regression.count();
    }
    count /= (endpoints[m] - endpoints[0]);
    for (std::size_t i = 0u; i < m; ++i)
    {
        double c{regressions[i].count()};
        if (c > 0.0)
        {
            regressions[i].scale(count * (endpoints[i+1] - endpoints[i]) / c);
        }
    }

    LOG_TRACE("old endpoints   = " << core::CContainerPrinter::print(endpoints));
    LOG_TRACE("old regressions = " << core::CContainerPrinter::print(m_Regressions));
    LOG_TRACE("old centres     = " << core::CContainerPrinter::print(m_Centres));
    LOG_TRACE("old variances   = " << core::CContainerPrinter::print(m_Variances));
    LOG_TRACE("new endpoints   = " << core::CContainerPrinter::print(m_Endpoints));
    LOG_TRACE("new regressions = " << core::CContainerPrinter::print(regressions));
    LOG_TRACE("new centres     = " << core::CContainerPrinter::print(centres));
    LOG_TRACE("new variances   = " << core::CContainerPrinter::print(variances));
    m_Regressions.swap(regressions);
    m_Centres.swap(centres);
    m_Variances.swap(variances);
}

void CSeasonalComponentAdaptiveBucketing::add(std::size_t bucket,
                                              core_t::TTime time,
                                              double offset,
                                              const TDoubleMeanVarAccumulator &value)
{
    TRegression &regression{m_Regressions[bucket]};
    CFloatStorage &variance{m_Variances[bucket]};

    core_t::TTime tk{time +  (m_Time->windowStart() + static_cast<core_t::TTime>(offset + 0.5))
                            % m_Time->windowRepeat()};
    TDoubleMeanVarAccumulator variance_{
            CBasicStatistics::momentsAccumulator(regression.count(),
                                                 regression.mean(),
                                                 static_cast<double>(variance)) + value};

    regression.add(m_Time->regression(tk),
                   CBasicStatistics::mean(value),
                   CBasicStatistics::count(value));
    variance = CBasicStatistics::maximumLikelihoodVariance(variance_);
}

double CSeasonalComponentAdaptiveBucketing::offset(core_t::TTime time) const
{
    return m_Time->periodic(time);
}

double CSeasonalComponentAdaptiveBucketing::count(std::size_t bucket) const
{
    return m_Regressions[bucket].count();
}

double CSeasonalComponentAdaptiveBucketing::predict(std::size_t bucket, core_t::TTime time, double offset) const
{
    double t{m_Time->regression(time + static_cast<core_t::TTime>(offset + 0.5))};
    return predict_(m_Regressions[bucket], t, this->bucketingAgeAt(time));
}

double CSeasonalComponentAdaptiveBucketing::variance(std::size_t bucket) const
{
    return m_Variances[bucket];
}

double CSeasonalComponentAdaptiveBucketing::bucketingAgeAt(core_t::TTime time) const
{
    return static_cast<double>(time - m_InitialTime) / static_cast<double>(core::constants::WEEK);
}

}
}
