/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
#include <maths/CMathsFuncs.h>
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
template<typename T>
void clearAndShrink(std::vector<T>& vector) {
    std::vector<T> empty;
    empty.swap(vector);
}

const std::string ADAPTIVE_BUCKETING_TAG{"a"};
const std::string FEATURE_TAG{"b"};
const std::string VALUES_TAG{"c"};
const std::string EMPTY_STRING;
}

CCalendarComponentAdaptiveBucketing::CCalendarComponentAdaptiveBucketing()
    : CAdaptiveBucketing{0.0, 0.0} {
}

CCalendarComponentAdaptiveBucketing::CCalendarComponentAdaptiveBucketing(CCalendarFeature feature,
                                                                         double decayRate,
                                                                         double minimumBucketLength)
    : CAdaptiveBucketing{decayRate, minimumBucketLength}, m_Feature{feature} {
}

CCalendarComponentAdaptiveBucketing::CCalendarComponentAdaptiveBucketing(
    double decayRate,
    double minimumBucketLength,
    core::CStateRestoreTraverser& traverser)
    : CAdaptiveBucketing{decayRate, minimumBucketLength} {
    traverser.traverseSubLevel(boost::bind(
        &CCalendarComponentAdaptiveBucketing::acceptRestoreTraverser, this, _1));
}

void CCalendarComponentAdaptiveBucketing::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(ADAPTIVE_BUCKETING_TAG, this->getAcceptPersistInserter());
    inserter.insertValue(FEATURE_TAG, m_Feature.toDelimited());
    core::CPersistUtils::persist(VALUES_TAG, m_Values, inserter);
}

void CCalendarComponentAdaptiveBucketing::swap(CCalendarComponentAdaptiveBucketing& other) {
    this->CAdaptiveBucketing::swap(other);
    std::swap(m_Feature, other.m_Feature);
    m_Values.swap(other.m_Values);
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

void CCalendarComponentAdaptiveBucketing::clear() {
    this->CAdaptiveBucketing::clear();
    clearAndShrink(m_Values);
}

void CCalendarComponentAdaptiveBucketing::linearScale(double scale) {
    for (auto& value : m_Values) {
        CBasicStatistics::moment<0>(value) *= scale;
    }
}

void CCalendarComponentAdaptiveBucketing::add(core_t::TTime time, double value, double weight) {
    std::size_t bucket{0};
    if (this->initialized() && this->bucket(time, bucket)) {
        this->CAdaptiveBucketing::add(bucket, time, weight);

        TFloatMeanVarAccumulator moments{m_Values[bucket]};
        double prediction{CBasicStatistics::mean(moments)};
        moments.add(value, weight * weight);

        m_Values[bucket].add(value, weight);
        CBasicStatistics::moment<1>(m_Values[bucket]) =
            CBasicStatistics::maximumLikelihoodVariance(moments);
        if (std::fabs(value - prediction) >
            LARGE_ERROR_STANDARD_DEVIATIONS *
                std::sqrt(CBasicStatistics::maximumLikelihoodVariance(moments))) {
            this->addLargeError(bucket, time);
        }
    }
}

CCalendarFeature CCalendarComponentAdaptiveBucketing::feature() const {
    return m_Feature;
}

void CCalendarComponentAdaptiveBucketing::propagateForwardsByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR(<< "Can't propagate bucketing backwards in time");
    } else if (this->initialized()) {
        double factor{std::exp(-this->decayRate() * time)};
        this->age(factor);
        for (auto& value : m_Values) {
            value.age(factor);
        }
    }
}

double CCalendarComponentAdaptiveBucketing::count(core_t::TTime time) const {
    const TFloatMeanVarAccumulator* value = this->value(time);
    return value ? static_cast<double>(CBasicStatistics::count(*value)) : 0.0;
}

const TFloatMeanVarAccumulator*
CCalendarComponentAdaptiveBucketing::value(core_t::TTime time) const {
    const TFloatMeanVarAccumulator* result{nullptr};
    if (this->initialized()) {
        std::size_t bucket{0};
        this->bucket(time, bucket);
        bucket = CTools::truncate(bucket, std::size_t(0), m_Values.size() - 1);
        result = &m_Values[bucket];
    }
    return result;
}

uint64_t CCalendarComponentAdaptiveBucketing::checksum(uint64_t seed) const {
    seed = this->CAdaptiveBucketing::checksum(seed);
    seed = CChecksum::calculate(seed, m_Feature);
    return CChecksum::calculate(seed, m_Values);
}

void CCalendarComponentAdaptiveBucketing::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCalendarComponentAdaptiveBucketing");
    core::CMemoryDebug::dynamicSize("m_Endpoints", this->endpoints(), mem);
    core::CMemoryDebug::dynamicSize("m_Centres", this->centres(), mem);
    core::CMemoryDebug::dynamicSize("m_LargeErrorCounts", this->largeErrorCounts(), mem);
    core::CMemoryDebug::dynamicSize("m_Values", m_Values, mem);
}

std::size_t CCalendarComponentAdaptiveBucketing::memoryUsage() const {
    return this->CAdaptiveBucketing::memoryUsage() + core::CMemory::dynamicSize(m_Values);
}

bool CCalendarComponentAdaptiveBucketing::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(ADAPTIVE_BUCKETING_TAG,
                traverser.traverseSubLevel(this->getAcceptRestoreTraverser()));
        RESTORE(FEATURE_TAG, m_Feature.fromDelimited(traverser.value()));
        RESTORE(VALUES_TAG, core::CPersistUtils::restore(VALUES_TAG, m_Values, traverser))
    } while (traverser.next());

    return true;
}

void CCalendarComponentAdaptiveBucketing::refresh(const TFloatVec& oldEndpoints) {
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
    std::size_t n{oldEndpoints.size()};
    if (m + 1 != n) {
        LOG_ERROR(<< "Inconsistent end points and regressions");
        return;
    }

    const TFloatVec& newEndpoints{this->endpoints()};
    const TFloatVec& oldCentres{this->centres()};
    const TFloatVec& oldLargeErrorCounts{this->largeErrorCounts()};

    TFloatMeanVarVec newValues;
    TFloatVec newCentres;
    TFloatVec newLargeErrorCounts;
    newValues.reserve(m);
    newCentres.reserve(m);
    newLargeErrorCounts.reserve(m);

    for (std::size_t i = 1u; i < n; ++i) {
        double yl{newEndpoints[i - 1]};
        double yr{newEndpoints[i]};
        std::size_t r = std::lower_bound(oldEndpoints.begin(), oldEndpoints.end(), yr) -
                        oldEndpoints.begin();
        r = CTools::truncate(r, std::size_t(1), n - 1);

        std::size_t l = std::upper_bound(oldEndpoints.begin(), oldEndpoints.end(), yl) -
                        oldEndpoints.begin();
        l = CTools::truncate(l, std::size_t(1), r);

        LOG_TRACE(<< "interval = [" << yl << "," << yr << "]");
        LOG_TRACE(<< "l = " << l << ", r = " << r);
        LOG_TRACE(<< "[x(l), x(r)] = [" << oldEndpoints[l - 1] << ","
                  << oldEndpoints[r] << "]");

        double xl{oldEndpoints[l - 1]};
        double xr{oldEndpoints[l]};
        if (l == r) {
            double interval{newEndpoints[i] - newEndpoints[i - 1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            newValues.push_back(CBasicStatistics::scaled(m_Values[l - 1], w * w));
            newCentres.push_back(
                CTools::truncate(static_cast<double>(oldCentres[l - 1]), yl, yr));
            newLargeErrorCounts.push_back(w * oldLargeErrorCounts[l - 1]);
        } else {
            double interval{xr - newEndpoints[i - 1]};
            double w{CTools::truncate(interval / (xr - xl), 0.0, 1.0)};
            TDoubleMeanVarAccumulator value{CBasicStatistics::scaled(m_Values[l - 1], w)};
            TDoubleMeanAccumulator centre{CBasicStatistics::momentsAccumulator(
                w * CBasicStatistics::count(m_Values[l - 1]),
                static_cast<double>(oldCentres[l - 1]))};
            double largeErrorCount{w * oldLargeErrorCounts[l - 1]};
            double count{w * w * CBasicStatistics::count(m_Values[l - 1])};
            while (++l < r) {
                value += m_Values[l - 1];
                centre += CBasicStatistics::momentsAccumulator(
                    CBasicStatistics::count(m_Values[l - 1]),
                    static_cast<double>(oldCentres[l - 1]));
                largeErrorCount += oldLargeErrorCounts[l - 1];
                count += CBasicStatistics::count(m_Values[l - 1]);
            }
            xl = oldEndpoints[l - 1];
            xr = oldEndpoints[l];
            interval = newEndpoints[i] - xl;
            w = CTools::truncate(interval / (xr - xl), 0.0, 1.0);
            value += CBasicStatistics::scaled(m_Values[l - 1], w);
            centre += CBasicStatistics::momentsAccumulator(
                w * CBasicStatistics::count(m_Values[l - 1]),
                static_cast<double>(oldCentres[l - 1]));
            largeErrorCount += w * oldLargeErrorCounts[l - 1];
            count += w * w * CBasicStatistics::count(m_Values[l - 1]);
            double scale{count / CBasicStatistics::count(value)};
            newValues.push_back(CBasicStatistics::scaled(value, scale));
            newCentres.push_back(CTools::truncate(CBasicStatistics::mean(centre), yl, yr));
            newLargeErrorCounts.push_back(largeErrorCount);
        }
    }

    // We want all values to respond at the same rate to changes
    // in the trend. To achieve this we should assign them a weight
    // that is equal to the number of points they will receive in one
    // period.
    double count{0.0};
    for (const auto& value : newValues) {
        count += CBasicStatistics::count(value);
    }
    count /= (oldEndpoints[m] - oldEndpoints[0]);
    for (std::size_t i = 0u; i < m; ++i) {
        double ci{CBasicStatistics::count(newValues[i])};
        if (ci > 0.0) {
            CBasicStatistics::scale(
                count * (oldEndpoints[i + 1] - oldEndpoints[i]) / ci, newValues[i]);
        }
    }

    LOG_TRACE(<< "old endpoints = " << core::CContainerPrinter::print(oldEndpoints));
    LOG_TRACE(<< "old centres   = " << core::CContainerPrinter::print(oldCentres));
    LOG_TRACE(<< "new endpoints = " << core::CContainerPrinter::print(newEndpoints));
    LOG_TRACE(<< "new centres   = " << core::CContainerPrinter::print(newCentres));
    m_Values.swap(newValues);
    this->centres().swap(newCentres);
    this->largeErrorCounts().swap(newLargeErrorCounts);
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

double CCalendarComponentAdaptiveBucketing::bucketCount(std::size_t bucket) const {
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

void CCalendarComponentAdaptiveBucketing::split(std::size_t bucket) {
    // We don't know the fraction of values' (weights) which would
    // have fallen in each half of the split bucket. However, some
    // fraction of them would ideally not be included in these
    // statistics, i.e. the values in the other half of the split.
    // If we assume an equal split but assign a weight of 0.0 to the
    // samples included in error we arrive at a multiplier of 0.25.
    // In practice this simply means we increase the significance
    // of new samples for some time which is reasonable.
    CBasicStatistics::scale(0.25, m_Values[bucket]);
    m_Values.insert(m_Values.begin() + bucket, m_Values[bucket]);
}

std::string CCalendarComponentAdaptiveBucketing::name() const {
    return "Calendar[" + std::to_string(this->decayRate()) + "," +
           std::to_string(this->minimumBucketLength()) + "]";
}

bool CCalendarComponentAdaptiveBucketing::isBad() const {
    // check for bad values in both the means and the variances
    return std::any_of(m_Values.begin(), m_Values.end(), [](const auto& value) {
        return ((CMathsFuncs::isFinite(CBasicStatistics::mean(value)) == false) ||
                (CMathsFuncs::isFinite(CBasicStatistics::variance(value))) == false);
    });
}
}
}
