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

#include <boost/math/distributions/binomial.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TSizeVec = std::vector<std::size_t>;
using TFloatUInt32Pr = std::pair<CFloatStorage, std::uint32_t>;

//! \brief Used to keep track of continguous points after spreading.
class MATHS_EXPORT CContinugousPoints {
public:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

public:
    explicit CContinugousPoints(double point) { m_Centre.add(point); }

    std::size_t size() const {
        return static_cast<std::size_t>(CBasicStatistics::count(m_Centre));
    }

    double centre() const { return CBasicStatistics::mean(m_Centre); }

    double location(std::size_t j, double separation) const {
        return CBasicStatistics::mean(m_Centre) +
               static_cast<double>(j) * separation - this->radius(separation);
    }

    bool overlap(const CContinugousPoints& other, double separation) const {
        double left{CBasicStatistics::mean(m_Centre)};
        double right{CBasicStatistics::mean(other.m_Centre)};
        return std::fabs(right - left) <
               (1.0 + 1e-6) * (this->radius(separation) + other.radius(separation) + separation);
    }

    void merge(const CContinugousPoints& other, double a, double b, double separation) {
        m_Centre += other.m_Centre;
        CBasicStatistics::moment<0>(m_Centre) =
            this->truncate(CBasicStatistics::mean(m_Centre), a, b, separation);
    }

private:
    double truncate(double point, double a, double b, double separation) const {
        return CTools::truncate(point, a + this->radius(separation),
                                b - this->radius(separation));
    }
    double radius(double separation) const {
        return (CBasicStatistics::count(m_Centre) - 1.0) * separation / 2.0;
    }

private:
    TMeanAccumulator m_Centre;
};

//! Convert to a delimited string.
std::string significanceToDelimited(const TFloatUInt32Pr& value) {
    return value.first.toString() + CBasicStatistics::EXTERNAL_DELIMITER +
           core::CStringUtils::typeToString(value.second);
}

//! Initialize from a delimited string.
bool significanceFromDelimited(const std::string& delimited, TFloatUInt32Pr& value) {
    std::size_t pos{delimited.find(CBasicStatistics::EXTERNAL_DELIMITER)};
    if (pos == std::string::npos) {
        LOG_ERROR(<< "Failed to delimiter in '" << delimited << "'");
        return false;
    }
    unsigned int count{};
    if (value.first.fromString(delimited.substr(0, pos)) == false ||
        core::CStringUtils::stringToType(delimited.substr(pos + 1), count) == false) {
        LOG_ERROR(<< "Failed to extract value from '" << delimited << "'");
        return false;
    }
    value.second = count;
    return true;
}

//! Clear a vector and recover its memory.
template<typename T>
void clearAndShrink(std::vector<T>& vector) {
    std::vector<T> empty;
    empty.swap(vector);
}

const core::TPersistenceTag DECAY_RATE_TAG{"a", "decay_rate"};
const core::TPersistenceTag ENDPOINT_TAG{"b", "endpoint"};
const core::TPersistenceTag CENTRES_TAG{"c", "centres"};
const core::TPersistenceTag MEAN_DESIRED_DISPLACEMENT_TAG{"d", "mean_desired_displacement"};
const core::TPersistenceTag MEAN_ABS_DESIRED_DISPLACEMENT_TAG{"e", "mean_abs_desired_displacement"};
const core::TPersistenceTag LARGE_ERROR_COUNTS_TAG{"f", "large_error_counts"};
const core::TPersistenceTag TARGET_SIZE_TAG{"g", "target size"};
const core::TPersistenceTag LAST_LARGE_ERROR_BUCKET_TAG{"h", "last_large_error_bucket"};
const core::TPersistenceTag LAST_LARGE_ERROR_PERIOD_TAG{"i", "last_large_error_period"};
const core::TPersistenceTag LARGE_ERROR_COUNT_SIGNIFICANCES_TAG{"j", "large_error_counts_significance"};
const core::TPersistenceTag MEAN_WEIGHT_TAG{"k", "mean weight"};
const std::string EMPTY_STRING;

const double SMOOTHING_FUNCTION[]{0.25, 0.5, 0.25};
const std::size_t WIDTH{boost::size(SMOOTHING_FUNCTION) / 2};
const double ALPHA{0.25};
const double EPS{std::numeric_limits<double>::epsilon()};
const double WEIGHTS[]{1.0, 1.0, 1.0, 0.75, 0.5};
const double MINIMUM_DECAY_RATE{0.001};
const double MINIMUM_LARGE_ERROR_COUNT_TO_SPLIT{10.0};
const double MODERATE_SIGNIFICANCE{1e-2};
const double HIGH_SIGNIFICANCE{1e-3};
const double LOG_MODERATE_SIGNIFICANCE{std::log(MODERATE_SIGNIFICANCE)};
const double LOG_HIGH_SIGNIFICANCE{std::log(HIGH_SIGNIFICANCE)};
}

CAdaptiveBucketing::CAdaptiveBucketing(double decayRate, double minimumBucketLength)
    : m_DecayRate{std::max(decayRate, MINIMUM_DECAY_RATE)}, m_MinimumBucketLength{minimumBucketLength} {
}

CAdaptiveBucketing::TRestoreFunc CAdaptiveBucketing::getAcceptRestoreTraverser() {
    return std::bind(&CAdaptiveBucketing::acceptRestoreTraverser, this,
                     std::placeholders::_1);
}

CAdaptiveBucketing::TPersistFunc CAdaptiveBucketing::getAcceptPersistInserter() const {
    return std::bind(&CAdaptiveBucketing::acceptPersistInserter, this,
                     std::placeholders::_1);
}

bool CAdaptiveBucketing::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(DECAY_RATE_TAG, m_DecayRate)
        RESTORE_BUILT_IN(TARGET_SIZE_TAG, m_TargetSize)
        RESTORE_BUILT_IN(LAST_LARGE_ERROR_BUCKET_TAG, m_LastLargeErrorBucket)
        RESTORE_BUILT_IN(LAST_LARGE_ERROR_PERIOD_TAG, m_LastLargeErrorPeriod)
        RESTORE(LARGE_ERROR_COUNT_SIGNIFICANCES_TAG,
                m_LargeErrorCountSignificances.fromDelimited(traverser.value(), significanceFromDelimited))
        RESTORE(MEAN_WEIGHT_TAG, m_MeanWeight.fromDelimited(traverser.value()))
        RESTORE(ENDPOINT_TAG, core::CPersistUtils::fromString(traverser.value(), m_Endpoints))
        RESTORE(CENTRES_TAG, core::CPersistUtils::fromString(traverser.value(), m_Centres))
        RESTORE(LARGE_ERROR_COUNTS_TAG,
                core::CPersistUtils::fromString(traverser.value(), m_LargeErrorCounts))
        RESTORE(MEAN_DESIRED_DISPLACEMENT_TAG,
                m_MeanDesiredDisplacement.fromDelimited(traverser.value()))
        RESTORE(MEAN_ABS_DESIRED_DISPLACEMENT_TAG,
                m_MeanAbsDesiredDisplacement.fromDelimited(traverser.value()))
    } while (traverser.next());
    if (m_TargetSize == 0) {
        m_TargetSize = this->size();
    }
    if (m_LargeErrorCounts.empty()) {
        m_LargeErrorCounts.resize(m_Centres.size(), 0.0);
    }
    return true;
}

void CAdaptiveBucketing::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, m_DecayRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(TARGET_SIZE_TAG, m_TargetSize);
    inserter.insertValue(LAST_LARGE_ERROR_BUCKET_TAG, m_LastLargeErrorBucket);
    inserter.insertValue(LAST_LARGE_ERROR_PERIOD_TAG, m_LastLargeErrorPeriod);
    inserter.insertValue(LARGE_ERROR_COUNT_SIGNIFICANCES_TAG,
                         m_LargeErrorCountSignificances.toDelimited(significanceToDelimited));
    inserter.insertValue(MEAN_WEIGHT_TAG, m_MeanWeight.toDelimited());
    inserter.insertValue(ENDPOINT_TAG, core::CPersistUtils::toString(m_Endpoints));
    inserter.insertValue(CENTRES_TAG, core::CPersistUtils::toString(m_Centres));
    inserter.insertValue(LARGE_ERROR_COUNTS_TAG,
                         core::CPersistUtils::toString(m_LargeErrorCounts));
    inserter.insertValue(MEAN_DESIRED_DISPLACEMENT_TAG,
                         m_MeanDesiredDisplacement.toDelimited());
    inserter.insertValue(MEAN_ABS_DESIRED_DISPLACEMENT_TAG,
                         m_MeanAbsDesiredDisplacement.toDelimited());
}

void CAdaptiveBucketing::swap(CAdaptiveBucketing& other) {
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_MinimumBucketLength, other.m_MinimumBucketLength);
    std::swap(m_TargetSize, other.m_TargetSize);
    std::swap(m_LastLargeErrorBucket, other.m_LastLargeErrorBucket);
    std::swap(m_LastLargeErrorPeriod, other.m_LastLargeErrorPeriod);
    std::swap(m_LargeErrorCountSignificances, other.m_LargeErrorCountSignificances);
    std::swap(m_MeanWeight, other.m_MeanWeight);
    m_Endpoints.swap(other.m_Endpoints);
    m_Centres.swap(other.m_Centres);
    m_LargeErrorCounts.swap(other.m_LargeErrorCounts);
    std::swap(m_MeanDesiredDisplacement, other.m_MeanDesiredDisplacement);
    std::swap(m_MeanAbsDesiredDisplacement, other.m_MeanAbsDesiredDisplacement);
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

    m_TargetSize = n;
    m_Endpoints.clear();
    m_Endpoints.reserve(n + 1);
    double width{(b - a) / static_cast<double>(n)};
    for (std::size_t i = 0u; i < n + 1; ++i) {
        m_Endpoints.push_back(a + static_cast<double>(i) * width);
    }
    m_Centres.clear();
    m_Centres.resize(n);
    m_LargeErrorCounts.clear();
    m_LargeErrorCounts.resize(n, 0.0);

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
    clearAndShrink(m_LargeErrorCounts);
}

void CAdaptiveBucketing::add(std::size_t bucket, core_t::TTime time, double weight) {
    auto centre = CBasicStatistics::momentsAccumulator(
        this->bucketCount(bucket), static_cast<double>(m_Centres[bucket]));
    centre.add(this->offset(time), weight);
    m_Centres[bucket] = CBasicStatistics::mean(centre);
    m_MeanWeight.add(weight);
}

void CAdaptiveBucketing::addLargeError(std::size_t bucket, core_t::TTime time) {
    core_t::TTime period{static_cast<core_t::TTime>(
        m_Endpoints[m_Endpoints.size() - 1] - m_Endpoints[0])};
    time = CIntegerTools::floor(time, period);
    if (bucket != m_LastLargeErrorBucket || time != m_LastLargeErrorPeriod) {
        m_LargeErrorCounts[bucket] += 1.0;
    }
    m_LastLargeErrorBucket = bucket;
    m_LastLargeErrorPeriod = time;
}

void CAdaptiveBucketing::decayRate(double value) {
    m_DecayRate = std::max(value, MINIMUM_DECAY_RATE);
}

double CAdaptiveBucketing::decayRate() const {
    return m_DecayRate;
}

void CAdaptiveBucketing::age(double factor) {
    for (auto& count : m_LargeErrorCounts) {
        count *= factor;
    }
    m_MeanDesiredDisplacement.age(factor);
    m_MeanAbsDesiredDisplacement.age(factor);
    m_MeanWeight.age(factor);
}

double CAdaptiveBucketing::minimumBucketLength() const {
    return m_MinimumBucketLength;
}

void CAdaptiveBucketing::refine(core_t::TTime time) {
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMinMaxAccumulator = CBasicStatistics::CMinMax<TDoubleSizePr>;

    LOG_TRACE(<< "refining at " << time);

    if (m_Endpoints.size() < 2) {
        return;
    }

    // Check if any buckets should be split based on the large error counts.
    this->maybeSplitBucket();

    std::size_t n{m_Endpoints.size() - 1};
    double a{m_Endpoints[0]};
    double b{m_Endpoints[n]};

    // Extract the bucket means.
    TDoubleDoublePrVec values;
    values.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        values.emplace_back(this->bucketCount(i), this->predict(i, time, m_Centres[i]));
    }
    LOG_TRACE(<< "values = " << core::CContainerPrinter::print(values));

    // Compute the function range in each bucket, imposing periodic
    // boundary conditions at the start and end of the interval.
    TDoubleVec ranges;
    ranges.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        TDoubleDoublePr v[]{values[(n + i - 2) % n], values[(n + i - 1) % n],
                            values[(n + i + 0) % n], // centre
                            values[(n + i + 1) % n], values[(n + i + 2) % n]};

        TMinMaxAccumulator minmax;
        for (std::size_t j = 0u; j < sizeof(v) / sizeof(v[0]); ++j) {
            if (v[j].first > 0.0) {
                minmax.add({v[j].second, j});
            }
        }

        if (minmax.initialized() > 0) {
            ranges.push_back(WEIGHTS[minmax.max().second > minmax.min().second
                                         ? minmax.max().second - minmax.min().second
                                         : minmax.min().second - minmax.max().second] *
                             std::pow(minmax.max().first - minmax.min().first, 0.75));
        } else {
            ranges.push_back(0.0);
        }
    }

    // Smooth the ranges by convolving with a smoothing function.
    // We do this in the "time" domain because the smoothing
    // function is narrow. Estimate the averaging error in each
    // bucket by multiplying the smoothed range by the bucket width.
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
    }
    double maxAveragingError{
        *std::max_element(averagingErrors.begin(), averagingErrors.end())};
    for (const auto& significance : m_LargeErrorCountSignificances) {
        if (significance.first < MODERATE_SIGNIFICANCE) {
            averagingErrors[significance.second] = maxAveragingError;
        }
    }
    double totalAveragingError{
        std::accumulate(averagingErrors.begin(), averagingErrors.end(), 0.0)};
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
        // Noise in the bucket mean values creates a "high" frequency
        // mean zero driving force on the buckets' end points desired
        // positions. Once they have stabilized on their desired location
        // for the trend, we are able to detect this by comparing the
        // time averaged desired displacement and the absolute desired
        // displacement. The lower the ratio the more smoothing we apply.
        // Note we want to damp the noise out because the process of
        // adjusting the buckets end points loses a small amount of
        // information, see the comments at the start of refresh for
        // more details.
        double alpha{
            ALPHA * (CBasicStatistics::mean(m_MeanAbsDesiredDisplacement) == 0.0
                         ? 1.0
                         : std::fabs(CBasicStatistics::mean(m_MeanDesiredDisplacement)) /
                               CBasicStatistics::mean(m_MeanAbsDesiredDisplacement))};
        LOG_TRACE(<< "alpha = " << alpha);
        double displacement{0.0};

        // Linearly interpolate between the current end points and points
        // separated by equal total averaging error. Interpolating is
        // equivalent to adding drag to the end point dynamics and damps
        // any oscillations.
        double unassignedAveragingError{0.0};
        for (std::size_t i = 0, j = 1; i < n && j < n + 1; ++i) {
            double ai{endpoints[i]};
            double bi{endpoints[i + 1]};
            double hi{bi - ai};
            double ei{averagingErrors[i]};
            unassignedAveragingError += ei;
            for (double ej = step - (unassignedAveragingError - ei);
                 unassignedAveragingError >= step;
                 ej += step, unassignedAveragingError -= step, ++j) {
                double xj{hi * ej / ei};
                m_Endpoints[j] = std::max(
                    m_Endpoints[j - 1] + 1e-6 * std::fabs(m_Endpoints[j - 1]),
                    endpoints[j] + alpha * (ai + xj - endpoints[j]));
                displacement += (ai + xj) - endpoints[j];
                LOG_TRACE(<< "interval = [" << ai << "," << bi << "]"
                          << " averaging error / unit length = " << ei / hi << ", desired translation "
                          << endpoints[j] << " -> " << ai + xj);
            }
        }
        spread(a, b, m_MinimumBucketLength, m_Endpoints);

        // By construction, the first and last end point should be
        // close "a" and "b", respectively, but we snap them to "a"
        // and "b" so that the total interval is unchanged.
        m_Endpoints[0] = a;
        m_Endpoints[n] = b;
        LOG_TRACE(<< "refinedEndpoints = " << core::CContainerPrinter::print(m_Endpoints));

        m_MeanDesiredDisplacement.add(displacement);
        m_MeanAbsDesiredDisplacement.add(std::fabs(displacement));
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
        if (this->bucketCount(i) > 0.0) {
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
            double c0{c};
            knots.push_back(m_Endpoints[0]);
            values.push_back(this->predict(i, time, c));
            variances.push_back(this->variance(i));
            for (/**/; i < n; ++i) {
                if (this->bucketCount(i) > 0.0) {
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
                // We search for the value in the last and next period which
                // are adjacent. Note that values need not be the same at the
                // start and end of the period because the gradient can vary,
                // but we expect them to be continuous.
                for (std::size_t j = n - 1; j > 0; --j) {
                    if (this->bucketCount(j) > 0.0) {
                        double alpha{m_Endpoints[n] - m_Centres[j]};
                        double beta{c0};
                        double Z{alpha + beta};
                        double lastPeriodValue{
                            this->predict(j, time, m_Centres[j] - m_Endpoints[n])};
                        double lastPeriodVariance{this->variance(j)};
                        knots[0] = m_Endpoints[0];
                        values[0] = (alpha * values[0] + beta * lastPeriodValue) / Z;
                        variances[0] = (alpha * variances[0] + beta * lastPeriodVariance) / Z;
                        break;
                    }
                }
                for (std::size_t j = 0u; j < n; ++j) {
                    if (this->bucketCount(j) > 0.0) {
                        double alpha{m_Centres[j]};
                        double beta{m_Endpoints[n] - knots.back()};
                        double Z{alpha + beta};
                        double nextPeriodValue{
                            this->predict(j, time, m_Endpoints[n] + m_Centres[j])};
                        double nextPeriodVariance{this->variance(j)};
                        values.push_back((alpha * values.back() + beta * nextPeriodValue) / Z);
                        variances.push_back(
                            (alpha * variances.back() + beta * nextPeriodVariance) / Z);
                        knots.push_back(m_Endpoints[n]);
                        break;
                    }
                }
                break;
            }
        }
    }

    if (knots.size() > 2) {
        // If the distance between knot points becomes too small the
        // spline can become poorly conditioned. We can safely discard
        // knot points which are very close to one another.
        TSizeVec indices(knots.size());
        std::iota(indices.begin(), indices.end(), 0);
        indices.erase(std::remove_if(indices.begin() + 1, indices.end() - 1,
                                     [&knots](std::size_t i) {
                                         return knots[i] - knots[i - 1] < 1.0 ||
                                                knots[i + 1] - knots[i] < 1.0;
                                     }),
                      indices.end() - 1);
        if (indices.size() < knots.size()) {
            for (std::size_t i = 0u; i < indices.size(); ++i) {
                knots[i] = knots[indices[i]];
                values[i] = values[indices[i]];
                variances[i] = variances[indices[i]];
            }
            knots.resize(indices.size());
            values.resize(indices.size());
            variances.resize(indices.size());
        }
    }

    return knots.size() >= 2;
}

void CAdaptiveBucketing::spread(double a, double b, double separation, TFloatVec& points) {

    if (separation <= 0.0 || points.size() < 2) {
        return;
    }
    if (b <= a) {
        LOG_ERROR(<< "Bad interval [" << a << "," << b << "]");
        return;
    }

    // Check if we just need to space the points uniformly.
    std::size_t n{points.size() - 1};
    if (b - a <= separation * static_cast<double>(n + 1)) {
        for (std::size_t i = 0; i <= n; ++i) {
            points[i] = a + (b - a) * static_cast<double>(i) / static_cast<double>(n);
        }
        return;
    }

    // Check if there's nothing to do.
    double minSeparation{points[1] - points[0]};
    for (std::size_t i = 2; i < points.size(); ++i) {
        minSeparation = std::min(minSeparation, points[i] - points[i - 1]);
    }
    if (minSeparation > separation) {
        return;
    }

    // We can do this in n * log(n) complexity with at most log(n)
    // passes through the points. Provided the minimum separation
    // is at least "interval" / "# centres" the problem is feasible.
    //
    // We want to find the solution which minimizes the sum of the
    // distances the points move. This is possible by repeatedly
    // merging clusters of contiguous points and then placing them
    // at the mean of the points they contain. The process repeats
    // until no clusters merge.

    std::sort(points.begin(), points.end(),
              [](double lhs, double rhs) { return lhs < rhs; });

    std::vector<CContinugousPoints> contiguousPoints;
    contiguousPoints.reserve(points.size());
    for (const auto& point : points) {
        contiguousPoints.emplace_back(point);
    }

    for (std::size_t previousSize{0}; contiguousPoints.size() != previousSize;
         /**/) {
        previousSize = contiguousPoints.size();
        std::size_t last{0};
        for (std::size_t i = 1; i < contiguousPoints.size(); ++i) {
            if (contiguousPoints[last].overlap(contiguousPoints[i], separation)) {
                contiguousPoints[last].merge(contiguousPoints[i], a, b, separation);
            } else {
                std::swap(contiguousPoints[++last], contiguousPoints[i]);
            }
        }
        contiguousPoints.erase(contiguousPoints.begin() + last + 1,
                               contiguousPoints.end());
    }

    double last{-std::numeric_limits<double>::max()};
    for (std::size_t i = 0, j = 0; i < contiguousPoints.size(); ++i) {
        for (std::size_t k = 0; k < contiguousPoints[i].size(); ++j, ++k) {
            points[j] = std::max(last + 1e-6 * std::fabs(last),
                                 contiguousPoints[i].location(k, separation));
            last = points[j];
        }
    }
}

const CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::endpoints() const {
    return m_Endpoints;
}

const CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::centres() const {
    return m_Centres;
}

CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::centres() {
    return m_Centres;
}

const CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::largeErrorCounts() const {
    return m_LargeErrorCounts;
}

CAdaptiveBucketing::TFloatVec& CAdaptiveBucketing::largeErrorCounts() {
    return m_LargeErrorCounts;
}

double CAdaptiveBucketing::adjustedWeight(std::size_t bucket, double weight) const {
    for (const auto& significance : m_LargeErrorCountSignificances) {
        if (bucket == significance.second) {
            double maxWeight{CBasicStatistics::mean(m_MeanWeight)};
            double logSignificance{CTools::fastLog(significance.first)};
            return CTools::linearlyInterpolate(LOG_HIGH_SIGNIFICANCE, LOG_MODERATE_SIGNIFICANCE,
                                               maxWeight, weight, logSignificance);
        }
    }
    return weight;
}

double CAdaptiveBucketing::count() const {
    double result = 0.0;
    for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
        result += this->bucketCount(i);
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
    seed = CChecksum::calculate(seed, m_TargetSize);
    seed = CChecksum::calculate(seed, m_LastLargeErrorBucket);
    seed = CChecksum::calculate(seed, m_LastLargeErrorPeriod);
    seed = CChecksum::calculate(
        seed, m_LargeErrorCountSignificances.toDelimited(significanceToDelimited));
    seed = CChecksum::calculate(seed, m_MeanWeight);
    seed = CChecksum::calculate(seed, m_Endpoints);
    seed = CChecksum::calculate(seed, m_Centres);
    seed = CChecksum::calculate(seed, m_LargeErrorCounts);
    seed = CChecksum::calculate(seed, m_MeanDesiredDisplacement);
    return CChecksum::calculate(seed, m_MeanAbsDesiredDisplacement);
}

std::size_t CAdaptiveBucketing::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Endpoints)};
    mem += core::CMemory::dynamicSize(m_Centres);
    mem += core::CMemory::dynamicSize(m_LargeErrorCounts);
    return mem;
}

void CAdaptiveBucketing::maybeSplitBucket() {
    double largeErrorCount{std::accumulate(m_LargeErrorCounts.begin(),
                                           m_LargeErrorCounts.end(), 0.0)};
    double period{m_Endpoints[m_Endpoints.size() - 1] - m_Endpoints[0]};

    if (static_cast<double>(this->size() + 1) * m_MinimumBucketLength <= period &&
        largeErrorCount >= MINIMUM_LARGE_ERROR_COUNT_TO_SPLIT) {

        m_LargeErrorCountSignificances = TFloatUInt32PrMinAccumulator{};

        // We compute the right tail p-value of the count of large errors
        // in a bucket for the null hypothesis that they are uniformly
        // distributed on the total bucketed period and split if this is
        // less than a specified threshold.
        for (std::size_t i = 1; i < m_Endpoints.size(); ++i) {
            double interval{m_Endpoints[i] - m_Endpoints[i - 1]};
            try {
                boost::math::binomial binomial{largeErrorCount, interval / period};
                double oneMinusCdf{
                    CTools::safeCdfComplement(binomial, m_LargeErrorCounts[i - 1])};
                m_LargeErrorCountSignificances.add({oneMinusCdf, i - 1});
            } catch (const std::exception& e) {
                LOG_ERROR(<< "Failed to calculate splitting significance: '"
                          << e.what() << "' interval = " << interval << " period = " << period
                          << " buckets = " << core::CContainerPrinter::print(m_Endpoints)
                          << " type = " << this->name());
            }
        }
        if (m_LargeErrorCountSignificances.count() > 0) {
            // We're choosing the minimum p-value of number of buckets
            // independent statistics so the significance is one minus
            // the chance that all of them are greater than the observation.
            for (auto& significance : m_LargeErrorCountSignificances) {
                significance.first = CTools::oneMinusPowOneMinusX(
                    significance.first, static_cast<double>(this->size()));
                LOG_TRACE(<< "bucket [" << m_Endpoints[significance.second]
                          << "," << m_Endpoints[significance.second + 1]
                          << ") split significance = " << significance.first);
            }
            m_LargeErrorCountSignificances.sort();
        }

        if (2 * this->size() < 3 * m_TargetSize &&
            largeErrorCount > MINIMUM_LARGE_ERROR_COUNT_TO_SPLIT &&
            m_LargeErrorCountSignificances.count() > 0 &&
            m_LargeErrorCountSignificances[0].first < HIGH_SIGNIFICANCE) {
            this->splitBucket(m_LargeErrorCountSignificances[0].second);
        }
    }
}

void CAdaptiveBucketing::splitBucket(std::size_t bucket) {
    double leftEnd{m_Endpoints[bucket]};
    double rightEnd{m_Endpoints[bucket + 1]};
    LOG_TRACE(<< "splitting [" << leftEnd << "," << rightEnd << ")");
    double midpoint{(leftEnd + rightEnd) / 2.0};
    double centre{m_Centres[bucket]};
    double offset{std::min(centre - leftEnd, rightEnd - centre) / 2.0};
    m_Endpoints.insert(m_Endpoints.begin() + bucket + 1, midpoint);
    m_Centres[bucket] = std::max(centre + offset, midpoint);
    m_Centres.insert(m_Centres.begin() + bucket, std::min(centre - offset, midpoint));
    m_LargeErrorCounts[bucket] /= 1.75;
    m_LargeErrorCounts.insert(m_LargeErrorCounts.begin() + bucket,
                              m_LargeErrorCounts[bucket]);
    this->split(bucket);
}

const double CAdaptiveBucketing::LARGE_ERROR_STANDARD_DEVIATIONS{3.0};
}
}
