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

using TDoubleMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TRegression = CSeasonalComponentAdaptiveBucketing::TRegression;

//! Clear a vector and recover its memory.
template<typename T>
void clearAndShrink(std::vector<T> &vector)
{
    std::vector<T> empty;
    empty.swap(vector);
}

//! Get the gradient of \p r.
double gradient(const TRegression &r)
{
    TRegression::TArray params;
    r.parameters(params);
    return params[1];
}

// Version 6.3
const std::string VERSION_6_3_TAG("6.3");
const std::string ADAPTIVE_BUCKETING_6_3_TAG{"a"};
const std::string TIME_6_3_TAG{"b"};
const std::string BUCKETS_6_3_TAG{"e"};
const std::string REGRESSION_6_3_TAG{"e"};
const std::string VARIANCE_6_3_TAG{"f"};
const std::string FIRST_UPDATE_6_3_TAG{"g"};
const std::string LAST_UPDATE_6_3_TAG{"h"};
// Version < 6.3
const std::string ADAPTIVE_BUCKETING_OLD_TAG{"a"};
const std::string TIME_OLD_TAG{"b"};
const std::string INITIAL_TIME_OLD_TAG{"c"};
const std::string REGRESSION_OLD_TAG{"d"};
const std::string VARIANCES_OLD_TAG{"e"};
const std::string LAST_UPDATES_OLD_TAG{"f"};

const std::string EMPTY_STRING;
const core_t::TTime UNSET_TIME{0};
const double SUFFICIENT_INTERVAL_TO_ESTIMATE_SLOPE{2.5};

}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(void) :
        CAdaptiveBucketing{0.0, 0.0}
{}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(const CSeasonalTime &time,
                                                                         double decayRate,
                                                                         double minimumBucketLength) :
        CAdaptiveBucketing{decayRate, minimumBucketLength},
        m_Time{time.clone()}
{}

CSeasonalComponentAdaptiveBucketing::CSeasonalComponentAdaptiveBucketing(const CSeasonalComponentAdaptiveBucketing &other) :
        CAdaptiveBucketing(other),
        m_Time{other.m_Time->clone()},
        m_Buckets(other.m_Buckets)
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
    inserter.insertValue(VERSION_6_3_TAG, "");
    inserter.insertLevel(ADAPTIVE_BUCKETING_6_3_TAG,
                         boost::bind(&CAdaptiveBucketing::acceptPersistInserter,
                                     static_cast<const CAdaptiveBucketing*>(this), _1));
    inserter.insertLevel(TIME_6_3_TAG, boost::bind(&CSeasonalTimeStateSerializer::acceptPersistInserter,
                                                   boost::cref(*m_Time), _1));
    core::CPersistUtils::persist(BUCKETS_6_3_TAG, m_Buckets, inserter);
}

void CSeasonalComponentAdaptiveBucketing::swap(CSeasonalComponentAdaptiveBucketing &other)
{
    this->CAdaptiveBucketing::swap(other);
    m_Time.swap(other.m_Time);
    m_Buckets.swap(other.m_Buckets);
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
        m_Buckets.assign(n, SBucket());
        return true;
    }
    return false;
}

void CSeasonalComponentAdaptiveBucketing::initialValues(core_t::TTime startTime,
                                                        core_t::TTime endTime,
                                                        const TFloatMeanAccumulatorVec &values)
{
    if (this->initialized())
    {
        this->shiftOrigin(startTime);
        if (!values.empty())
        {
            this->CAdaptiveBucketing::initialValues(startTime, endTime, values);
            this->shiftSlope(-this->slope());
        }
    }
}

std::size_t CSeasonalComponentAdaptiveBucketing::size(void) const
{
    return this->CAdaptiveBucketing::size();
}

void CSeasonalComponentAdaptiveBucketing::clear(void)
{
    this->CAdaptiveBucketing::clear();
    clearAndShrink(m_Buckets);
}

void CSeasonalComponentAdaptiveBucketing::shiftOrigin(core_t::TTime time)
{
    time = CIntegerTools::floor(time, core::constants::WEEK);
    double shift{m_Time->regression(time)};
    if (shift > 0.0)
    {
        for (auto &bucket : m_Buckets)
        {
            bucket.s_Regression.shiftAbscissa(-shift);
        }
        m_Time->regressionOrigin(time);
    }
}

void CSeasonalComponentAdaptiveBucketing::shiftLevel(double shift)
{
    for (auto &bucket : m_Buckets)
    {
        bucket.s_Regression.shiftOrdinate(shift);
    }
}

void CSeasonalComponentAdaptiveBucketing::shiftSlope(double shift)
{
    for (auto &bucket : m_Buckets)
    {
        bucket.s_Regression.shiftGradient(shift);
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

    this->CAdaptiveBucketing::add(bucket, time, weight);

    SBucket &bucket_{m_Buckets[bucket]};
    double t{m_Time->regression(time)};
    TRegression &regression{bucket_.s_Regression};

    TDoubleMeanVarAccumulator moments =
            CBasicStatistics::accumulator(regression.count(),
                                          prediction,
                                          static_cast<double>(bucket_.s_Variance));
    moments.add(value, weight * weight);

    regression.add(t, value, weight);
    bucket_.s_Variance = CBasicStatistics::maximumLikelihoodVariance(moments);

    if (m_Time->regressionInterval(bucket_.s_FirstUpdate,
                                   bucket_.s_LastUpdate) < SUFFICIENT_INTERVAL_TO_ESTIMATE_SLOPE)
    {
        double delta{regression.predict(t)};
        regression.shiftGradient(-gradient(regression));
        delta -= regression.predict(t);
        regression.shiftOrdinate(delta);
    }

    bucket_.s_FirstUpdate = bucket_.s_FirstUpdate == UNSET_TIME ?
                            time : std::min(bucket_.s_FirstUpdate, time);
    bucket_.s_LastUpdate  = bucket_.s_LastUpdate == UNSET_TIME ?
                            time : std::max(bucket_.s_LastUpdate, time);
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
        for (auto &bucket : m_Buckets)
        {
            bucket.s_Regression.age(factor, meanRevert);
        }
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
        bucket = CTools::truncate(bucket, std::size_t(0), m_Buckets.size() - 1);
        result = &m_Buckets[bucket].s_Regression;
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
    for (const auto &bucket : m_Buckets)
    {
        if (bucket.s_Regression.count() > 0.0)
        {
            minmax.add(gradient(bucket.s_Regression));
        }
    }
    return minmax.initialized() ? minmax.signMargin() : 0.0;
}

bool CSeasonalComponentAdaptiveBucketing::slopeAccurate(core_t::TTime time) const
{
    return this->observedInterval(time) >= SUFFICIENT_INTERVAL_TO_ESTIMATE_SLOPE;
}

uint64_t CSeasonalComponentAdaptiveBucketing::checksum(uint64_t seed) const
{
    seed = this->CAdaptiveBucketing::checksum(seed);
    seed = CChecksum::calculate(seed, m_Time);
    return CChecksum::calculate(seed, m_Buckets);
}

void CSeasonalComponentAdaptiveBucketing::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CSeasonalComponentAdaptiveBucketing");
    core::CMemoryDebug::dynamicSize("m_Endpoints", this->CAdaptiveBucketing::endpoints(), mem);
    core::CMemoryDebug::dynamicSize("m_Centres", this->CAdaptiveBucketing::centres(), mem);
    core::CMemoryDebug::dynamicSize("m_Buckets", m_Buckets, mem);
}

std::size_t CSeasonalComponentAdaptiveBucketing::memoryUsage(void) const
{
    return  this->CAdaptiveBucketing::memoryUsage()
          + core::CMemory::dynamicSize(m_Buckets);
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
    if (traverser.name() == VERSION_6_3_TAG)
    {
        while (traverser.next())
        {
            const std::string &name{traverser.name()};
            RESTORE(ADAPTIVE_BUCKETING_6_3_TAG, traverser.traverseSubLevel(
                        boost::bind(&CAdaptiveBucketing::acceptRestoreTraverser,
                                    static_cast<CAdaptiveBucketing*>(this), _1)));
            RESTORE(TIME_6_3_TAG, traverser.traverseSubLevel(
                        boost::bind(&CSeasonalTimeStateSerializer::acceptRestoreTraverser, boost::ref(m_Time), _1)))
            RESTORE(BUCKETS_6_3_TAG, core::CPersistUtils::restore(BUCKETS_6_3_TAG, m_Buckets, traverser))
        }
    }
    else
    {
        // There is no version string this is historic state.

        using TTimeVec = std::vector<core_t::TTime>;
        using TRegressionVec = std::vector<TRegression>;

        core_t::TTime initialTime;
        TRegressionVec regressions;
        TFloatVec variances;
        TTimeVec lastUpdates;
        do
        {
            const std::string &name{traverser.name()};
            RESTORE(ADAPTIVE_BUCKETING_OLD_TAG, traverser.traverseSubLevel(
                        boost::bind(&CAdaptiveBucketing::acceptRestoreTraverser,
                                    static_cast<CAdaptiveBucketing*>(this), _1)));
            RESTORE(TIME_OLD_TAG, traverser.traverseSubLevel(
                        boost::bind(&CSeasonalTimeStateSerializer::acceptRestoreTraverser, boost::ref(m_Time), _1)))
            RESTORE_BUILT_IN(INITIAL_TIME_OLD_TAG, initialTime)
            RESTORE_SETUP_TEARDOWN(REGRESSION_OLD_TAG,
                                   TRegression regression,
                                   traverser.traverseSubLevel(boost::bind(&TRegression::acceptRestoreTraverser,
                                                                          &regression, _1)),
                                   regressions.push_back(regression))
            RESTORE(VARIANCES_OLD_TAG, core::CPersistUtils::fromString(traverser.value(), variances))
            RESTORE(LAST_UPDATES_OLD_TAG, core::CPersistUtils::fromString(traverser.value(), lastUpdates))
        }
        while (traverser.next());

        m_Buckets.clear();
        m_Buckets.reserve(regressions.size());
        for (std::size_t i = 0u; i < regressions.size(); ++i)
        {
            m_Buckets.emplace_back(regressions[i], variances[i], initialTime, lastUpdates[i]);
        }
    }

    m_Buckets.shrink_to_fit();

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
    using TMinAccumulator = CBasicStatistics::SMin<core_t::TTime>::TAccumulator;

    std::size_t m{m_Buckets.size()};
    std::size_t n{endpoints.size()};
    if (m+1 != n)
    {
        LOG_ERROR("Inconsistent end points and regressions");
        return;
    }

    TFloatVec &m_Endpoints{this->CAdaptiveBucketing::endpoints()};
    TFloatVec &m_Centres{this->CAdaptiveBucketing::centres()};

    TBucketVec buckets;
    TFloatVec centres;
    buckets.reserve(m);
    centres.reserve(m);

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
            const SBucket &bucket{m_Buckets[l-1]};
            buckets.emplace_back(bucket.s_Regression.scaled(w * w),
                                 bucket.s_Variance,
                                 bucket.s_FirstUpdate,
                                 bucket.s_LastUpdate);
            centres.push_back(CTools::truncate(static_cast<double>(m_Centres[l-1]), yl, yr));
        }
        else
        {
            double interval{xr - m_Endpoints[i-1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            const SBucket *bucket{&m_Buckets[l-1]};
            TMinAccumulator firstUpdate;
            TMinAccumulator lastUpdate;
            TDoubleRegression regression{bucket->s_Regression.scaled(w)};
            TDoubleMeanVarAccumulator variance{
                    CBasicStatistics::accumulator(w * bucket->s_Regression.count(),
                                                  bucket->s_Regression.mean(),
                                                  static_cast<double>(bucket->s_Variance))};
            firstUpdate.add(bucket->s_FirstUpdate);
            lastUpdate.add(bucket->s_LastUpdate);
            TDoubleMeanAccumulator centre{
                    CBasicStatistics::accumulator(w * bucket->s_Regression.count(),
                                                  static_cast<double>(m_Centres[l-1]))};
            double count{w * w * bucket->s_Regression.count()};
            while (++l < r)
            {
                bucket = &m_Buckets[l-1];
                regression += bucket->s_Regression;
                variance += CBasicStatistics::accumulator(bucket->s_Regression.count(),
                                                          bucket->s_Regression.mean(),
                                                          static_cast<double>(bucket->s_Variance));
                firstUpdate.add(bucket->s_FirstUpdate);
                lastUpdate.add(bucket->s_LastUpdate);
                centre += CBasicStatistics::accumulator(bucket->s_Regression.count(),
                                                        static_cast<double>(m_Centres[l-1]));
                count += bucket->s_Regression.count();
            }
            xl = endpoints[l-1];
            xr = endpoints[l];
            bucket = &m_Buckets[l-1];
            interval = m_Endpoints[i] - xl;
            w = CTools::truncate(interval / (xr - xl), 0.0, 1.0);
            regression += bucket->s_Regression.scaled(w);
            variance += CBasicStatistics::accumulator(w * bucket->s_Regression.count(),
                                                      bucket->s_Regression.mean(),
                                                      static_cast<double>(bucket->s_Variance));
            firstUpdate.add(bucket->s_FirstUpdate);
            lastUpdate.add(bucket->s_LastUpdate);
            centre += CBasicStatistics::accumulator(w * bucket->s_Regression.count(),
                                                    static_cast<double>(m_Centres[l-1]));
            count += w * w * bucket->s_Regression.count();
            double scale{count == regression.count() ? 1.0 : count / regression.count()};
            buckets.emplace_back(regression.scaled(scale),
                                 CBasicStatistics::maximumLikelihoodVariance(variance),
                                 firstUpdate[0], lastUpdate[0]);
            centres.push_back(CTools::truncate(CBasicStatistics::mean(centre), yl, yr));
        }
    }

    // We want all regressions to respond at the same rate to changes
    // in the trend. To achieve this we should assign them a weight
    // that is equal to the number of points they will receive in one
    // period.
    double count{0.0};
    for (const auto &bucket : buckets)
    {
        count += bucket.s_Regression.count();
    }
    count /= (endpoints[m] - endpoints[0]);
    for (std::size_t i = 0u; i < m; ++i)
    {
        double c{buckets[i].s_Regression.count()};
        if (c > 0.0)
        {
            buckets[i].s_Regression.scale(count * (endpoints[i+1] - endpoints[i]) / c);
        }
    }

    LOG_TRACE("old endpoints   = " << core::CContainerPrinter::print(endpoints));
    LOG_TRACE("old centres     = " << core::CContainerPrinter::print(m_Centres));
    LOG_TRACE("new endpoints   = " << core::CContainerPrinter::print(m_Endpoints));
    LOG_TRACE("new centres     = " << core::CContainerPrinter::print(centres));
    m_Buckets.swap(buckets);
    m_Centres.swap(centres);
}

bool CSeasonalComponentAdaptiveBucketing::inWindow(core_t::TTime time) const
{
    return m_Time->inWindow(time);
}

void CSeasonalComponentAdaptiveBucketing::add(std::size_t bucket, core_t::TTime time, double value, double weight)
{
    SBucket &bucket_{m_Buckets[bucket]};
    TRegression &regression{bucket_.s_Regression};
    CFloatStorage &variance{bucket_.s_Variance};
    TDoubleMeanVarAccumulator variance_{
            CBasicStatistics::accumulator(regression.count(),
                                          regression.mean(),
                                          static_cast<double>(variance))};
    variance_.add(value, weight);
    regression.add(m_Time->regression(time), value, weight);
    variance = CBasicStatistics::maximumLikelihoodVariance(variance_);
}

double CSeasonalComponentAdaptiveBucketing::offset(core_t::TTime time) const
{
    return m_Time->periodic(time);
}

double CSeasonalComponentAdaptiveBucketing::count(std::size_t bucket) const
{
    return m_Buckets[bucket].s_Regression.count();
}

double CSeasonalComponentAdaptiveBucketing::predict(std::size_t bucket, core_t::TTime time, double offset) const
{
    const SBucket &bucket_{m_Buckets[bucket]};
    core_t::TTime firstUpdate{bucket_.s_FirstUpdate};
    core_t::TTime lastUpdate{bucket_.s_LastUpdate};
    const TRegression &regression{bucket_.s_Regression};

    double interval{static_cast<double>(lastUpdate - firstUpdate)};
    if (interval == 0)
    {
        return regression.mean();
    }

    double t{m_Time->regression(time + static_cast<core_t::TTime>(offset + 0.5))};

    double extrapolateInterval{static_cast<double>(
        CBasicStatistics::max(time - lastUpdate, firstUpdate - time, core_t::TTime(0)))};
    if (extrapolateInterval == 0.0)
    {
        return regression.predict(t);
    }

    // We mean revert our predictions if trying to predict much further
    // ahead than the observed interval for the data.
    double alpha{CTools::smoothHeaviside(extrapolateInterval / interval, 1.0 / 12.0, -1.0)};
    double beta{1.0 - alpha};
    return alpha * regression.predict(t) + beta * regression.mean();
}

double CSeasonalComponentAdaptiveBucketing::variance(std::size_t bucket) const
{
    return m_Buckets[bucket].s_Variance;
}

double CSeasonalComponentAdaptiveBucketing::observedInterval(core_t::TTime time) const
{
    return m_Time->regressionInterval(std::min_element(
               m_Buckets.begin(), m_Buckets.end(),
               [](const SBucket &lhs, const SBucket &rhs)
               { return lhs.s_FirstUpdate < rhs.s_FirstUpdate; })->s_FirstUpdate, time);
}

CSeasonalComponentAdaptiveBucketing::SBucket::SBucket(void) :
        s_Variance{0.0},
        s_FirstUpdate{UNSET_TIME},
        s_LastUpdate{UNSET_TIME}
{}

CSeasonalComponentAdaptiveBucketing::SBucket::SBucket(const TRegression &regression,
                                                      double variance,
                                                      core_t::TTime firstUpdate,
                                                      core_t::TTime lastUpdate) :
        s_Regression{regression},
        s_Variance{variance},
        s_FirstUpdate{firstUpdate},
        s_LastUpdate{lastUpdate}
{}

bool CSeasonalComponentAdaptiveBucketing::SBucket::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(REGRESSION_6_3_TAG, traverser.traverseSubLevel(boost::bind(&TRegression::acceptRestoreTraverser,
                                                                           &s_Regression, _1)))
        RESTORE(VARIANCE_6_3_TAG, s_Variance.fromString(traverser.value()))
        RESTORE_BUILT_IN(FIRST_UPDATE_6_3_TAG, s_FirstUpdate)
        RESTORE_BUILT_IN(LAST_UPDATE_6_3_TAG, s_LastUpdate)
    }
    while (traverser.next());
    return true;
}

void CSeasonalComponentAdaptiveBucketing::SBucket::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(REGRESSION_6_3_TAG, boost::bind(&TRegression::acceptPersistInserter,
                                                         &s_Regression, _1));
    inserter.insertValue(VARIANCE_6_3_TAG, s_Variance.toString());
    inserter.insertValue(FIRST_UPDATE_6_3_TAG, s_FirstUpdate);
    inserter.insertValue(LAST_UPDATE_6_3_TAG, s_LastUpdate);
}

uint64_t CSeasonalComponentAdaptiveBucketing::SBucket::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, s_Regression);
    seed = CChecksum::calculate(seed, s_Variance);
    seed = CChecksum::calculate(seed, s_FirstUpdate);
    return CChecksum::calculate(seed, s_LastUpdate);
}

}
}
