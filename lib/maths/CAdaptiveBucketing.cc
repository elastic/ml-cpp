/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CAdaptiveBucketing.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>

#include <boost/bind.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TDoubleMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

//! Clear a vector and recover its memory.
template<typename T>
void clearAndShrink(std::vector<T>& vector) {
    std::vector<T> empty;
    empty.swap(vector);
}

const std::string DECAY_RATE_TAG{"a"};
const std::string ENDPOINT_TAG{"b"};
const std::string CENTRES_TAG{"c"};
const std::string LP_FORCE_TAG{"d"};
const std::string FORCE_TAG{"e"};
const std::string EMPTY_STRING;

const double SMOOTHING_FUNCTION[]{0.25, 0.5, 0.25};
const std::size_t WIDTH{boost::size(SMOOTHING_FUNCTION) / 2};
const double ALPHA{0.25};
const double EPS{std::numeric_limits<double>::epsilon()};
const double WEIGHTS[]{1.0, 1.0, 1.0, 0.75, 0.5};
const double MINIMUM_DECAY_RATE{0.001};
}

bool CAdaptiveBucketing::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(DECAY_RATE_TAG, m_DecayRate)
        RESTORE(ENDPOINT_TAG, core::CPersistUtils::fromString(traverser.value(), m_Endpoints))
        RESTORE(CENTRES_TAG, core::CPersistUtils::fromString(traverser.value(), m_Centres))
        RESTORE(LP_FORCE_TAG, m_LpForce.fromDelimited(traverser.value()))
        RESTORE(FORCE_TAG, m_Force.fromDelimited(traverser.value()))
    } while (traverser.next());
    return true;
}

void CAdaptiveBucketing::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, m_DecayRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(ENDPOINT_TAG, core::CPersistUtils::toString(m_Endpoints));
    inserter.insertValue(CENTRES_TAG, core::CPersistUtils::toString(m_Centres));
    inserter.insertValue(LP_FORCE_TAG, m_LpForce.toDelimited());
    inserter.insertValue(FORCE_TAG, m_Force.toDelimited());
}

CAdaptiveBucketing::CAdaptiveBucketing(double decayRate, double minimumBucketLength)
    : m_DecayRate{std::max(decayRate, MINIMUM_DECAY_RATE)}, m_MinimumBucketLength{minimumBucketLength} {
}

CAdaptiveBucketing::CAdaptiveBucketing(double decayRate,
                                       double minimumBucketLength,
                                       core::CStateRestoreTraverser& traverser)
    : m_DecayRate{std::max(decayRate, MINIMUM_DECAY_RATE)}, m_MinimumBucketLength{minimumBucketLength} {
    traverser.traverseSubLevel(
        boost::bind(&CAdaptiveBucketing::acceptRestoreTraverser, this, _1));
}

void CAdaptiveBucketing::swap(CAdaptiveBucketing& other) {
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_MinimumBucketLength, other.m_MinimumBucketLength);
    m_Endpoints.swap(other.m_Endpoints);
    m_Centres.swap(other.m_Centres);
    std::swap(m_LpForce, other.m_LpForce);
    std::swap(m_Force, other.m_Force);
}

bool CAdaptiveBucketing::initialized() const {
    return m_Endpoints.size() > 0;
}

bool CAdaptiveBucketing::initialize(double a, double b, std::size_t n) {
    if (n == 0) {
        LOG_ERROR(<< "Must have at least one bucket");
        return false;
    }

    if (m_MinimumBucketLength > 0.0) {
        // Handle the case that the minimum bucket length is
        // longer than the period.
        m_MinimumBucketLength = std::min(m_MinimumBucketLength, b - a);
        n = std::min(n, static_cast<std::size_t>((b - a) / m_MinimumBucketLength));
    }

    m_Endpoints.clear();
    m_Endpoints.reserve(n + 1);
    double width{(b - a) / static_cast<double>(n)};
    for (std::size_t i = 0u; i < n + 1; ++i) {
        m_Endpoints.push_back(a + static_cast<double>(i) * width);
    }
    m_Centres.clear();
    m_Centres.resize(n);

    return true;
}

void CAdaptiveBucketing::initialValues(core_t::TTime start,
                                       core_t::TTime end,
                                       const TFloatMeanAccumulatorVec& values) {
    if (!this->initialized()) {
        return;
    }

    core_t::TTime size{static_cast<core_t::TTime>(values.size())};
    core_t::TTime dT{(end - start) / size};
    core_t::TTime dt{static_cast<core_t::TTime>(
        CTools::truncate(m_MinimumBucketLength, 1.0, static_cast<double>(dT)))};

    double scale{std::pow(static_cast<double>(dt) / static_cast<double>(dT), 2.0)};

    for (core_t::TTime time = start + dt / 2; time < end; time += dt) {
        if (this->inWindow(time)) {
            core_t::TTime i{(time - start) / dT};
            double value{CBasicStatistics::mean(values[i])};
            double weight{scale * CBasicStatistics::count(values[i])};
            if (weight > 0.0) {
                std::size_t bucket;
                if (this->bucket(time, bucket)) {
                    this->add(bucket, time, weight);
                    this->add(bucket, time, value, weight);
                }
            }
        }
    }
}

std::size_t CAdaptiveBucketing::size() const {
    return m_Centres.size();
}

void CAdaptiveBucketing::clear() {
    clearAndShrink(m_Endpoints);
    clearAndShrink(m_Centres);
}

void CAdaptiveBucketing::add(std::size_t bucket, core_t::TTime time, double weight) {
    TDoubleMeanAccumulator centre{CBasicStatistics::accumulator(
        this->count(bucket), static_cast<double>(m_Centres[bucket]))};
    centre.add(this->offset(time), weight);
    m_Centres[bucket] = CBasicStatistics::mean(centre);
}

void CAdaptiveBucketing::decayRate(double value) {
    m_DecayRate = std::max(value, MINIMUM_DECAY_RATE);
}

double CAdaptiveBucketing::decayRate() const {
    return m_DecayRate;
}

void CAdaptiveBucketing::age(double factor) {
    factor = factor * factor;
    m_LpForce.age(factor);
    m_Force.age(factor);
}

double CAdaptiveBucketing::minimumBucketLength() const {
    return m_MinimumBucketLength;
}

void CAdaptiveBucketing::refine(core_t::TTime time) {
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMinAccumulator = CBasicStatistics::SMin<TDoubleSizePr>::TAccumulator;
    using TMaxAccumulator = CBasicStatistics::SMax<TDoubleSizePr>::TAccumulator;

    LOG_TRACE(<< "refining at " << time);

    std::size_t n{m_Endpoints.size()};
    if (n < 2) {
        return;
    }
    --n;

    double a{m_Endpoints[0]};
    double b{m_Endpoints[n]};

    // Extract the bucket means.
    TDoubleDoublePrVec values;
    values.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        values.emplace_back(this->count(i), this->predict(i, time, m_Centres[i]));
    }
    LOG_TRACE(<< "values = " << core::CContainerPrinter::print(values));

    // Compute the function range in each bucket, imposing periodic
    // boundary conditions at the start and end of the interval.
    TDoubleVec ranges;
    ranges.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        TDoubleDoublePr v[]{values[(n + i - 2) % n], values[(n + i - 1) % n],
                            values[(n + i + 0) % n], values[(n + i + 1) % n],
                            values[(n + i + 2) % n]};

        TMinAccumulator min;
        TMaxAccumulator max;
        for (std::size_t j = 0u; j < sizeof(v) / sizeof(v[0]); ++j) {
            if (v[j].first > 0.0) {
                min.add({v[j].second, j});
                max.add({v[j].second, j});
            }
        }

        if (min.count() > 0) {
            ranges.push_back(
                WEIGHTS[max[0].second > min[0].second ? max[0].second - min[0].second
                                                      : min[0].second - max[0].second] *
                std::pow(max[0].first - min[0].first, 0.75));
        } else {
            ranges.push_back(0.0);
        }
    }

    // Smooth the ranges by convolving with a smoothing function.
    // We do this in the "time" domain because the smoothing
    // function is narrow. Estimate the averaging error in each
    // bucket by multiplying the smoothed range by the bucket width.
    double totalAveragingError{0.0};
    TDoubleVec averagingErrors;
    averagingErrors.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        double ai{m_Endpoints[i]};
        double bi{m_Endpoints[i + 1]};

        double error{0.0};
        for (std::size_t j = 0u; j < boost::size(SMOOTHING_FUNCTION); ++j) {
            error += SMOOTHING_FUNCTION[j] * ranges[(n + i + j - WIDTH) % n];
        }

        double h{bi - ai};
        error *= h / (b - a);

        averagingErrors.push_back(error);
        totalAveragingError += error;
    }
    LOG_TRACE(<< "averagingErrors = " << core::CContainerPrinter::print(averagingErrors));
    LOG_TRACE(<< "totalAveragingError = " << totalAveragingError);

    double n_{static_cast<double>(n)};
    double step{(1 - n_ * EPS) * totalAveragingError / n_};
    TFloatVec endpoints{m_Endpoints};
    LOG_TRACE(<< "step = " << step);

    // If all the function values are identical then the end points
    // should be equidistant. We check step in case of underflow.
    if (step == 0.0) {
        m_Endpoints[0] = a;
        for (std::size_t i = 0u; i < n; ++i) {
            m_Endpoints[i] = (b - a) * static_cast<double>(i) / n_;
        }
        m_Endpoints[n] = b;
    } else {
        // Noise in the bucket mean values creates a "high"
        // frequency mean zero driving force on the buckets'
        // end points desired positions. Once they have stabilized
        // on their desired location for the trend, we are able
        // to detect this by comparing an IIR low pass filtered
        // force and the total force. The lower the ratio the
        // smaller the force we actually apply. Note we want to
        // damp the noise out because the process of adjusting
        // the buckets values loses a small amount of information,
        // see the comments at the start of refresh for more
        // details.
        double alpha{ALPHA * (CBasicStatistics::mean(m_Force) == 0.0
                                  ? 1.0
                                  : std::fabs(CBasicStatistics::mean(m_LpForce)) /
                                        CBasicStatistics::mean(m_Force))};
        double force{0.0};

        // Linearly interpolate between the current end points
        // and points separated by equal total averaging error.
        // Interpolating is equivalent to adding a drag term in
        // the differential equation governing the end point
        // dynamics and damps any oscillatory behavior which
        // might otherwise occur.
        double error{0.0};
        for (std::size_t i = 0u, j = 1u; i < n && j < n + 1; ++i) {
            double ai{endpoints[i]};
            double bi{endpoints[i + 1]};
            double h{bi - ai};
            double e{averagingErrors[i]};
            error += e;
            for (double e_ = step - (error - e); error >= step; e_ += step, error -= step) {
                double x{h * e_ / averagingErrors[i]};
                m_Endpoints[j] = endpoints[j] + alpha * (ai + x - endpoints[j]);
                force += (ai + x) - endpoints[j];
                LOG_TRACE(<< "interval averaging error = " << e
                          << ", a(i) = " << ai << ", x = " << x << ", endpoint "
                          << endpoints[j] << " -> " << ai + x);
                ++j;
            }
        }
        if (m_MinimumBucketLength > 0.0) {
            CTools::spread(a, b, m_MinimumBucketLength, m_Endpoints);
        }

        // By construction, the first and last end point should be
        // close "a" and "b", respectively, but we snap them to "a"
        // and "b" so that the total interval is unchanged.
        m_Endpoints[0] = a;
        m_Endpoints[n] = b;
        LOG_TRACE(<< "refinedEndpoints = " << core::CContainerPrinter::print(m_Endpoints));

        m_LpForce.add(force);
        m_Force.add(std::fabs(force));
    }

    this->refresh(endpoints);
}

bool CAdaptiveBucketing::knots(core_t::TTime time,
                               CSplineTypes::EBoundaryCondition boundary,
                               TDoubleVec& knots,
                               TDoubleVec& values,
                               TDoubleVec& variances) const {
    knots.clear();
    values.clear();
    variances.clear();

    std::size_t n{m_Centres.size()};
    for (std::size_t i = 0u; i < n; ++i) {
        if (this->count(i) > 0.0) {
            double wide{3.0 * (m_Endpoints[n] - m_Endpoints[0]) / static_cast<double>(n)};
            LOG_TRACE(<< "period " << m_Endpoints[n] - m_Endpoints[0]
                      << ", # buckets = " << n << ", wide = " << wide);

            // We get two points for each wide bucket but at most
            // one third of the buckets can be wide. In this case
            // we have 2 * n/3 + 2*n/3 knot points.
            knots.reserve(4 * n / 3);
            values.reserve(4 * n / 3);
            variances.reserve(4 * n / 3);

            double a{m_Endpoints[i]};
            double b{m_Endpoints[i + 1]};
            double c{m_Centres[i]};
            knots.push_back(m_Endpoints[0]);
            values.push_back(this->predict(i, time, c));
            variances.push_back(this->variance(i));
            for (/**/; i < n; ++i) {
                if (this->count(i) > 0.0) {
                    a = m_Endpoints[i];
                    b = m_Endpoints[i + 1];
                    c = m_Centres[i];
                    double m{this->predict(i, time, c)};
                    double v{this->variance(i)};
                    if (b - a > wide) {
                        knots.push_back(std::max(c - (b - a) / 4.0, a));
                        values.push_back(m);
                        variances.push_back(v);
                        knots.push_back(std::min(c + (b - a) / 4.0, b));
                        values.push_back(m);
                        variances.push_back(v);
                    } else {
                        knots.push_back(c);
                        values.push_back(m);
                        variances.push_back(v);
                    }
                }
            }

            switch (boundary) {
            case CSplineTypes::E_Natural:
            case CSplineTypes::E_ParabolicRunout:
                knots.push_back(m_Endpoints[n]);
                values.push_back(values.back());
                variances.push_back(variances.back());
                break;

            case CSplineTypes::E_Periodic:
                values[0] = (values[0] + values.back()) / 2.0;
                variances[0] = (variances[0] + variances.back()) / 2.0;
                knots.push_back(m_Endpoints[n]);
                values.push_back(values[0]);
                variances.push_back(variances[0]);
                break;
            }
        }
    }

    return knots.size() >= 2;
}

const CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::endpoints() const {
    return m_Endpoints;
}

CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::endpoints() {
    return m_Endpoints;
}

const CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::centres() const {
    return m_Centres;
}

CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::centres() {
    return m_Centres;
}

double CAdaptiveBucketing::count() const {
    double result = 0.0;
    for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
        result += this->count(i);
    }
    return result;
}

CAdaptiveBucketing::TDoubleVec CAdaptiveBucketing::values(core_t::TTime time) const {
    TDoubleVec result;
    result.reserve(m_Centres.size());
    for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
        result.push_back(this->predict(i, time, m_Centres[i]));
    }
    return result;
}

CAdaptiveBucketing::TDoubleVec CAdaptiveBucketing::variances() const {
    TDoubleVec result;
    result.reserve(m_Centres.size());
    for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
        result.push_back(this->variance(i));
    }
    return result;
}

bool CAdaptiveBucketing::bucket(core_t::TTime time, std::size_t& result) const {
    double t{this->offset(time)};

    std::size_t i(std::upper_bound(m_Endpoints.begin(), m_Endpoints.end(), t) -
                  m_Endpoints.begin());
    std::size_t n{m_Endpoints.size()};
    if (t < m_Endpoints[0] || i == n) {
        LOG_ERROR(<< "t = " << t << " out of range [" << m_Endpoints[0] << ","
                  << m_Endpoints[n - 1] << ")");
        return false;
    }

    result = i - 1;
    return true;
}

uint64_t CAdaptiveBucketing::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_MinimumBucketLength);
    seed = CChecksum::calculate(seed, m_Endpoints);
    return CChecksum::calculate(seed, m_Centres);
}

std::size_t CAdaptiveBucketing::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Endpoints)};
    mem += core::CMemory::dynamicSize(m_Centres);
    return mem;
}
}
}
