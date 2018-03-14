/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CCalendarComponentAdaptiveBucketing.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TFloatMeanVarAccumulator = CCalendarComponentAdaptiveBucketing::TFloatMeanVarAccumulator;

//! Clear a vector and recover its memory.
template <typename T>
void clearAndShrink(std::vector<T>& vector) {
    std::vector<T> empty;
    empty.swap(vector);
}

const std::string ADAPTIVE_BUCKETING_TAG{"a"};
const std::string FEATURE_TAG{"b"};
const std::string VALUES_TAG{"c"};
const std::string EMPTY_STRING;
}

CCalendarComponentAdaptiveBucketing::CCalendarComponentAdaptiveBucketing(void)
    : CAdaptiveBucketing{0.0, 0.0} {}

CCalendarComponentAdaptiveBucketing::CCalendarComponentAdaptiveBucketing(CCalendarFeature feature,
                                                                         double decayRate,
                                                                         double minimumBucketLength)
    : CAdaptiveBucketing{decayRate, minimumBucketLength}, m_Feature{feature} {}

CCalendarComponentAdaptiveBucketing::CCalendarComponentAdaptiveBucketing(
    double decayRate,
    double minimumBucketLength,
    core::CStateRestoreTraverser& traverser)
    : CAdaptiveBucketing{decayRate, minimumBucketLength} {
    traverser.traverseSubLevel(
        boost::bind(&CCalendarComponentAdaptiveBucketing::acceptRestoreTraverser, this, _1));
}

void CCalendarComponentAdaptiveBucketing::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(ADAPTIVE_BUCKETING_TAG,
                         boost::bind(&CAdaptiveBucketing::acceptPersistInserter,
                                     static_cast<const CAdaptiveBucketing*>(this),
                                     _1));
    inserter.insertValue(FEATURE_TAG, m_Feature.toDelimited());
    core::CPersistUtils::persist(VALUES_TAG, m_Values, inserter);
}

void CCalendarComponentAdaptiveBucketing::swap(CCalendarComponentAdaptiveBucketing& other) {
    this->CAdaptiveBucketing::swap(other);
    std::swap(m_Feature, other.m_Feature);
    m_Values.swap(other.m_Values);
}

bool CCalendarComponentAdaptiveBucketing::initialized(void) const {
    return this->CAdaptiveBucketing::initialized();
}

bool CCalendarComponentAdaptiveBucketing::initialize(std::size_t n) {
    double a{0.0};
    double b{static_cast<double>(m_Feature.window())};

    if (this->CAdaptiveBucketing::initialize(a, b, n)) {
        m_Values.clear();
        m_Values.resize(this->size());
        return true;
    }
    return false;
}

std::size_t CCalendarComponentAdaptiveBucketing::size(void) const {
    return this->CAdaptiveBucketing::size();
}

void CCalendarComponentAdaptiveBucketing::clear(void) {
    this->CAdaptiveBucketing::clear();
    clearAndShrink(m_Values);
}

void CCalendarComponentAdaptiveBucketing::add(core_t::TTime time, double value, double weight) {
    std::size_t bucket{0};
    if (this->initialized() && this->bucket(time, bucket)) {
        this->CAdaptiveBucketing::add(bucket, time, weight);
        TFloatMeanVarAccumulator variance{m_Values[bucket]};
        variance.add(value, weight * weight);
        m_Values[bucket].add(value, weight);
        CBasicStatistics::moment<1>(m_Values[bucket]) =
            CBasicStatistics::maximumLikelihoodVariance(variance);
    }
}

CCalendarFeature CCalendarComponentAdaptiveBucketing::feature(void) const {
    return m_Feature;
}

void CCalendarComponentAdaptiveBucketing::decayRate(double value) {
    this->CAdaptiveBucketing::decayRate(value);
}

double CCalendarComponentAdaptiveBucketing::decayRate(void) const {
    return this->CAdaptiveBucketing::decayRate();
}

void CCalendarComponentAdaptiveBucketing::propagateForwardsByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR("Can't propagate bucketing backwards in time");
    } else if (this->initialized()) {
        double factor{::exp(-this->CAdaptiveBucketing::decayRate() * time)};
        this->CAdaptiveBucketing::age(factor);
        for (auto&& value : m_Values) {
            value.age(factor);
        }
    }
}

double CCalendarComponentAdaptiveBucketing::minimumBucketLength(void) const {
    return this->CAdaptiveBucketing::minimumBucketLength();
}

void CCalendarComponentAdaptiveBucketing::refine(core_t::TTime time) {
    this->CAdaptiveBucketing::refine(time);
}

double CCalendarComponentAdaptiveBucketing::count(core_t::TTime time) const {
    const TFloatMeanVarAccumulator* value = this->value(time);
    return value ? static_cast<double>(CBasicStatistics::count(*value)) : 0.0;
}

const TFloatMeanVarAccumulator*
CCalendarComponentAdaptiveBucketing::value(core_t::TTime time) const {
    const TFloatMeanVarAccumulator* result{0};
    if (this->initialized()) {
        std::size_t bucket{0};
        this->bucket(time, bucket);
        bucket = CTools::truncate(bucket, std::size_t(0), m_Values.size() - 1);
        result = &m_Values[bucket];
    }
    return result;
}

bool CCalendarComponentAdaptiveBucketing::knots(core_t::TTime time,
                                                CSplineTypes::EBoundaryCondition boundary,
                                                TDoubleVec& knots,
                                                TDoubleVec& values,
                                                TDoubleVec& variances) const {
    return this->CAdaptiveBucketing::knots(time, boundary, knots, values, variances);
}

uint64_t CCalendarComponentAdaptiveBucketing::checksum(uint64_t seed) const {
    seed = this->CAdaptiveBucketing::checksum(seed);
    seed = CChecksum::calculate(seed, m_Feature);
    return CChecksum::calculate(seed, m_Values);
}

void CCalendarComponentAdaptiveBucketing::debugMemoryUsage(
    core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCalendarComponentAdaptiveBucketing");
    core::CMemoryDebug::dynamicSize("m_Endpoints", this->CAdaptiveBucketing::endpoints(), mem);
    core::CMemoryDebug::dynamicSize("m_Centres", this->CAdaptiveBucketing::centres(), mem);
    core::CMemoryDebug::dynamicSize("m_Values", m_Values, mem);
}

std::size_t CCalendarComponentAdaptiveBucketing::memoryUsage(void) const {
    return this->CAdaptiveBucketing::memoryUsage() + core::CMemory::dynamicSize(m_Values);
}

const CCalendarComponentAdaptiveBucketing::TFloatVec&
CCalendarComponentAdaptiveBucketing::endpoints(void) const {
    return this->CAdaptiveBucketing::endpoints();
}

double CCalendarComponentAdaptiveBucketing::count(void) const {
    return this->CAdaptiveBucketing::count();
}

CCalendarComponentAdaptiveBucketing::TDoubleVec
CCalendarComponentAdaptiveBucketing::values(core_t::TTime time) const {
    return this->CAdaptiveBucketing::values(time);
}

CCalendarComponentAdaptiveBucketing::TDoubleVec
CCalendarComponentAdaptiveBucketing::variances(void) const {
    return this->CAdaptiveBucketing::variances();
}

bool CCalendarComponentAdaptiveBucketing::acceptRestoreTraverser(
    core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(ADAPTIVE_BUCKETING_TAG,
                traverser.traverseSubLevel(boost::bind(&CAdaptiveBucketing::acceptRestoreTraverser,
                                                       static_cast<CAdaptiveBucketing*>(this),
                                                       _1)));
        RESTORE(FEATURE_TAG, m_Feature.fromDelimited(traverser.value()));
        RESTORE(VALUES_TAG, core::CPersistUtils::restore(VALUES_TAG, m_Values, traverser))
    } while (traverser.next());

    return true;
}

void CCalendarComponentAdaptiveBucketing::refresh(const TFloatVec& endpoints) {
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
    using TDoubleMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    std::size_t m{m_Values.size()};
    std::size_t n{endpoints.size()};
    if (m + 1 != n) {
        LOG_ERROR("Inconsistent end points and regressions");
        return;
    }

    TFloatVec& m_Endpoints{this->CAdaptiveBucketing::endpoints()};
    TFloatVec& m_Centres{this->CAdaptiveBucketing::centres()};

    TFloatMeanVarVec values;
    TFloatVec centres;
    values.reserve(m);
    centres.reserve(m);

    for (std::size_t i = 1u; i < n; ++i) {
        double yl{m_Endpoints[i - 1]};
        double yr{m_Endpoints[i]};
        std::size_t r =
            std::lower_bound(endpoints.begin(), endpoints.end(), yr) - endpoints.begin();
        r = CTools::truncate(r, std::size_t(1), n - 1);

        std::size_t l =
            std::upper_bound(endpoints.begin(), endpoints.end(), yl) - endpoints.begin();
        l = CTools::truncate(l, std::size_t(1), r);

        LOG_TRACE("interval = [" << yl << "," << yr << "]");
        LOG_TRACE("l = " << l << ", r = " << r);
        LOG_TRACE("[x(l), x(r)] = [" << endpoints[l - 1] << "," << endpoints[r] << "]");

        double xl{endpoints[l - 1]};
        double xr{endpoints[l]};
        if (l == r) {
            double interval{m_Endpoints[i] - m_Endpoints[i - 1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            values.push_back(CBasicStatistics::scaled(m_Values[l - 1], w * w));
            centres.push_back(CTools::truncate(static_cast<double>(m_Centres[l - 1]), yl, yr));
        } else {
            double interval{xr - m_Endpoints[i - 1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            TDoubleMeanVarAccumulator value{CBasicStatistics::scaled(m_Values[l - 1], w)};
            TDoubleMeanAccumulator centre{
                CBasicStatistics::accumulator(w * CBasicStatistics::count(m_Values[l - 1]),
                                              static_cast<double>(m_Centres[l - 1]))};
            double count{w * w * CBasicStatistics::count(m_Values[l - 1])};
            while (++l < r) {
                value += m_Values[l - 1];
                centre += CBasicStatistics::accumulator(CBasicStatistics::count(m_Values[l - 1]),
                                                        static_cast<double>(m_Centres[l - 1]));
                count += CBasicStatistics::count(m_Values[l - 1]);
            }
            xl = endpoints[l - 1];
            xr = endpoints[l];
            interval = m_Endpoints[i] - xl;
            w = CTools::truncate(interval / (xr - xl), 0.0, 1.0);
            value += CBasicStatistics::scaled(m_Values[l - 1], w);
            centre += CBasicStatistics::accumulator(w * CBasicStatistics::count(m_Values[l - 1]),
                                                    static_cast<double>(m_Centres[l - 1]));
            count += w * w * CBasicStatistics::count(m_Values[l - 1]);
            double scale{count / CBasicStatistics::count(value)};
            values.push_back(CBasicStatistics::scaled(value, scale));
            centres.push_back(CTools::truncate(CBasicStatistics::mean(centre), yl, yr));
        }
    }

    // We want all values to respond at the same rate to changes
    // in the trend. To achieve this we should assign them a weight
    // that is equal to the number of points they will receive in one
    // period.
    double count{0.0};
    for (const auto& value : values) {
        count += CBasicStatistics::count(value);
    }
    count /= (endpoints[m] - endpoints[0]);
    for (std::size_t i = 0u; i < m; ++i) {
        double ci{CBasicStatistics::count(values[i])};
        if (ci > 0.0) {
            CBasicStatistics::scale(count * (endpoints[i + 1] - endpoints[i]) / ci, values[i]);
        }
    }

    LOG_TRACE("old endpoints = " << core::CContainerPrinter::print(endpoints));
    LOG_TRACE("old values    = " << core::CContainerPrinter::print(m_Values));
    LOG_TRACE("old centres   = " << core::CContainerPrinter::print(m_Centres));
    LOG_TRACE("new endpoints = " << core::CContainerPrinter::print(m_Endpoints));
    LOG_TRACE("new value     = " << core::CContainerPrinter::print(values));
    LOG_TRACE("new centres   = " << core::CContainerPrinter::print(centres));
    m_Values.swap(values);
    m_Centres.swap(centres);
}

bool CCalendarComponentAdaptiveBucketing::inWindow(core_t::TTime time) const {
    return m_Feature.inWindow(time);
}

void CCalendarComponentAdaptiveBucketing::add(std::size_t bucket,
                                              core_t::TTime /*time*/,
                                              double value,
                                              double weight) {
    m_Values[bucket].add(value, weight);
}

double CCalendarComponentAdaptiveBucketing::offset(core_t::TTime time) const {
    return static_cast<double>(m_Feature.offset(time));
}

double CCalendarComponentAdaptiveBucketing::count(std::size_t bucket) const {
    return CBasicStatistics::count(m_Values[bucket]);
}

double CCalendarComponentAdaptiveBucketing::predict(std::size_t bucket,
                                                    core_t::TTime /*time*/,
                                                    double /*offset*/) const {
    return CBasicStatistics::mean(m_Values[bucket]);
}

double CCalendarComponentAdaptiveBucketing::variance(std::size_t bucket) const {
    return CBasicStatistics::maximumLikelihoodVariance(m_Values[bucket]);
}
}
}
