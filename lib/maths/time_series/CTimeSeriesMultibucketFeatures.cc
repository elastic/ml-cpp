/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/time_series/CTimeSeriesMultibucketFeatures.h>

#include <core/CPersistUtils.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraTools.h>
#include <maths/common/CSolvers.h>
#include <maths/common/CTypeTraits.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/CSeasonalComponent.h>

#include <boost/circular_buffer.hpp>

#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ml {
namespace maths {
namespace time_series {
namespace {

common::CFloatStorage minWeight(double weight) {
    return weight;
}
template<std::size_t N>
common::CFloatStorage minWeight(const core::CSmallVector<double, N>& weight) {
    return *std::min_element(weight.begin(), weight.end());
}

template<typename U, typename V>
V conformable(const U& /*x*/, V value) {
    return value;
}
template<typename U, typename V>
common::CVector<V> conformable(const common::CVector<U>& x, V value) {
    return common::CVector<V>(x.dimension(), value);
}

const std::string CAPACITY_TAG{"a"};
const std::string SLIDING_WINDOW_TAG{"b"};
const std::string VALUE_AND_WEIGHT_TAG{"c"};
}

template<typename T, typename STORAGE>
class CTimeSeriesMultibucketMeanImpl {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using Type = T;
    using TFeature = CTimeSeriesMultibucketFeature<T>;
    using TType1Vec = typename TFeature::TType1Vec;
    using TWeightsAry1Vec = typename TFeature::TWeightsAry1Vec;
    using TType1VecTWeightAry1VecPr = typename TFeature::TType1VecTWeightAry1VecPr;
    using TFloatMeanAccumulator =
        typename common::CBasicStatistics::SSampleMean<STORAGE>::TAccumulator;
    using TValueFunc = std::function<STORAGE(const TFloatMeanAccumulator&)>;
    using TWeightFunc = std::function<double(const TFloatMeanAccumulator&)>;

public:
    explicit CTimeSeriesMultibucketMeanImpl(std::size_t length = 0)
        : m_SlidingWindow(length) {}

    const TType1VecTWeightAry1VecPr& value() const { return m_ValueAndWeight; }

    double correlationWithBucketValue() const {
        // This follows from the weighting applied to values in the window and
        // linearity of expectation.
        double r{WINDOW_GEOMETRIC_WEIGHT * WINDOW_GEOMETRIC_WEIGHT};
        double length{static_cast<double>(m_SlidingWindow.size())};
        return length == 0.0 ? 0.0 : std::sqrt((1.0 - r) / (1.0 - std::pow(r, length)));
    }

    void clear() {
        m_SlidingWindow.clear();
        m_ValueAndWeight = TType1VecTWeightAry1VecPr{};
    }

    void add(core_t::TTime time,
             core_t::TTime bucketLength,
             const TType1Vec& values,
             const TWeightsAry1Vec& weights) {

        // Remove any old samples.
        core_t::TTime cutoff{time - this->windowInterval(bucketLength)};
        while (m_SlidingWindow.size() > 0 && m_SlidingWindow.front().first < cutoff) {
            m_SlidingWindow.pop_front();
        }

        if (values.size() > 0) {
            using TStorage1Vec = core::CSmallVector<STORAGE, 1>;

            // Get the scales to apply to each value.
            TStorage1Vec scales;
            scales.reserve(weights.size());
            for (const auto& weight : weights) {
                using std::sqrt;
                scales.push_back(sqrt(STORAGE{maths_t::countVarianceScale(weight)} *
                                      STORAGE{maths_t::seasonalVarianceScale(weight)}));
            }

            // Add the next sample.
            TFloatMeanAccumulator next{conformable(STORAGE(values[0]), 0.0F)};
            for (std::size_t i = 0; i < values.size(); ++i) {
                auto weight = maths_t::countForUpdate(weights[i]);
                next.add(STORAGE(values[i]) / scales[i], minWeight(weight));
            }
            m_SlidingWindow.push_back({time, next});
        }

        if (this->sufficientData()) {
            m_ValueAndWeight.first.assign({this->windowValue()});
            m_ValueAndWeight.second.assign({maths_t::countWeight(this->windowCount())});
        } else {
            m_ValueAndWeight = TType1VecTWeightAry1VecPr{};
        }
    }

    std::uint64_t checksum(std::uint64_t seed = 0) const {
        seed = common::CChecksum::calculate(seed, m_SlidingWindow.capacity());
        seed = common::CChecksum::calculate(seed, m_SlidingWindow);
        return common::CChecksum::calculate(seed, m_ValueAndWeight);
    }

    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        core::CMemoryDebug::dynamicSize("m_SlidingWindow", m_SlidingWindow, mem);
        core::CMemoryDebug::dynamicSize("m_ValueAndWeight", m_ValueAndWeight, mem);
    }

    std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(m_SlidingWindow) +
               core::CMemory::dynamicSize(m_ValueAndWeight);
    }

    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name{traverser.name()};
            RESTORE_SETUP_TEARDOWN(CAPACITY_TAG, std::size_t capacity,
                                   core::CStringUtils::stringToType(traverser.value(), capacity),
                                   m_SlidingWindow.set_capacity(capacity))
            RESTORE(SLIDING_WINDOW_TAG,
                    core::CPersistUtils::restore(SLIDING_WINDOW_TAG, m_SlidingWindow, traverser))
            RESTORE(VALUE_AND_WEIGHT_TAG,
                    core::CPersistUtils::restore(VALUE_AND_WEIGHT_TAG, m_ValueAndWeight, traverser))
        } while (traverser.next());
        return true;
    }

    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(CAPACITY_TAG, m_SlidingWindow.capacity());
        core::CPersistUtils::persist(SLIDING_WINDOW_TAG, m_SlidingWindow, inserter);
        core::CPersistUtils::persist(VALUE_AND_WEIGHT_TAG, m_ValueAndWeight, inserter);
    }

private:
    using TDoubleMeanAccumulator = typename common::SPromoted<TFloatMeanAccumulator>::Type;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;
    using TTimeFloatMeanAccumulatorPrCBuf = boost::circular_buffer<TTimeFloatMeanAccumulatorPr>;

private:
    //! The geometric weight applied to the window.
    constexpr static const double WINDOW_GEOMETRIC_WEIGHT{0.9};

private:
    bool sufficientData() const {
        return 4 * m_SlidingWindow.size() >= 3 * m_SlidingWindow.capacity();
    }

    //! Get the length of the window given \p bucketLength.
    core_t::TTime windowInterval(core_t::TTime bucketLength) const {
        return static_cast<core_t::TTime>(m_SlidingWindow.capacity()) * bucketLength;
    }

    //! Compute the window value using \p value to extract individual values.
    Type windowValue() const {
        auto valueFunc = [&](const TFloatMeanAccumulator& x) {
            return common::CBasicStatistics::mean(x);
        };
        auto weightFunc = [](const TFloatMeanAccumulator& x) {
            return static_cast<double>(common::CBasicStatistics::count(x));
        };
        return this->evaluateOnWindow(valueFunc, weightFunc);
    }

    //! Compute the window count.
    Type windowCount() const {
        auto countFunc = [&](const TFloatMeanAccumulator& x) {
            return conformable(common::CBasicStatistics::mean(x),
                               common::CBasicStatistics::count(x));
        };
        auto weightFunc = [](const TFloatMeanAccumulator&) { return 1.0; };
        return this->evaluateOnWindow(countFunc, weightFunc);
    }

    //! Compute the weighted mean of \p value on the sliding window.
    Type evaluateOnWindow(const TValueFunc& valueFunc, const TWeightFunc& weightFunc) const {
        double latest;
        double earliest;
        std::tie(earliest, latest) = this->windowTimeRange();
        double n{static_cast<double>(m_SlidingWindow.size())};
        double scale{(n - 1.0) * (latest == earliest ? 1.0 : 1.0 / (latest - earliest))};
        auto i = m_SlidingWindow.begin();
        TDoubleMeanAccumulator mean{conformable(valueFunc(i->second), 0.0)};
        for (double last{earliest}; i != m_SlidingWindow.end(); ++i) {
            double dt{static_cast<double>(i->first) - last};
            last = static_cast<double>(i->first);
            mean.age(std::pow(WINDOW_GEOMETRIC_WEIGHT, scale * dt));
            mean.add(valueFunc(i->second), weightFunc(i->second));
        }
        return this->toVector(common::CBasicStatistics::mean(mean));
    }

    //! Compute the time range of window.
    TDoubleDoublePr windowTimeRange() const {
        auto range =
            std::accumulate(m_SlidingWindow.begin(), m_SlidingWindow.end(),
                            common::CBasicStatistics::CMinMax<double>(),
                            [](common::CBasicStatistics::CMinMax<double> partial,
                               const TTimeFloatMeanAccumulatorPr& value) {
                                partial.add(static_cast<double>(value.first));
                                return partial;
                            });
        return {range.min(), range.max()};
    }

    template<typename U>
    static double toVector(const U& x) {
        return x;
    }
    template<typename U>
    static T toVector(const common::CVector<U>& x) {
        return x.template toVector<T>();
    }

private:
    TTimeFloatMeanAccumulatorPrCBuf m_SlidingWindow;
    TType1VecTWeightAry1VecPr m_ValueAndWeight;
};

CTimeSeriesMultibucketScalarMean::CTimeSeriesMultibucketScalarMean(std::size_t length)
    : m_Impl{std::make_unique<TImpl>(length)} {
}

CTimeSeriesMultibucketScalarMean::CTimeSeriesMultibucketScalarMean(const CTimeSeriesMultibucketScalarMean& other)
    : m_Impl{std::make_unique<TImpl>(*other.m_Impl)} {
}

CTimeSeriesMultibucketScalarMean::~CTimeSeriesMultibucketScalarMean() = default;
CTimeSeriesMultibucketScalarMean::CTimeSeriesMultibucketScalarMean(
    CTimeSeriesMultibucketScalarMean&&) noexcept = default;

CTimeSeriesMultibucketScalarMean& CTimeSeriesMultibucketScalarMean::
operator=(CTimeSeriesMultibucketScalarMean&&) noexcept = default;
CTimeSeriesMultibucketScalarMean& CTimeSeriesMultibucketScalarMean::
operator=(const CTimeSeriesMultibucketScalarMean& other) {
    if (this != &other) {
        CTimeSeriesMultibucketScalarMean tmp{*this};
        *this = std::move(tmp);
    }
    return *this;
}

CTimeSeriesMultibucketScalarMean::TPtr CTimeSeriesMultibucketScalarMean::clone() const {
    return std::make_unique<CTimeSeriesMultibucketScalarMean>(*this);
}

const CTimeSeriesMultibucketScalarMean::TType1VecTWeightAry1VecPr&
CTimeSeriesMultibucketScalarMean::value() const {
    return m_Impl->value();
}

double CTimeSeriesMultibucketScalarMean::correlationWithBucketValue() const {
    return m_Impl->correlationWithBucketValue();
}

void CTimeSeriesMultibucketScalarMean::clear() {
    m_Impl->clear();
}

void CTimeSeriesMultibucketScalarMean::add(core_t::TTime time,
                                           core_t::TTime bucketLength,
                                           const TType1Vec& values,
                                           const TWeightsAry1Vec& weights) {
    m_Impl->add(time, bucketLength, values, weights);
}

std::uint64_t CTimeSeriesMultibucketScalarMean::checksum(std::uint64_t seed) const {
    return m_Impl->checksum(seed);
}

void CTimeSeriesMultibucketScalarMean::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesMultibucketScalarMean");
    core::CMemoryDebug::dynamicSize("m_Impl", m_Impl, mem);
}

std::size_t CTimeSeriesMultibucketScalarMean::staticSize() const {
    return sizeof(*this);
}

std::size_t CTimeSeriesMultibucketScalarMean::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Impl);
}

bool CTimeSeriesMultibucketScalarMean::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CTimeSeriesMultibucketScalarMean::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}

CTimeSeriesMultibucketVectorMean::CTimeSeriesMultibucketVectorMean(std::size_t length)
    : m_Impl{std::make_unique<TImpl>(length)} {
}

CTimeSeriesMultibucketVectorMean::CTimeSeriesMultibucketVectorMean(const CTimeSeriesMultibucketVectorMean& other)
    : m_Impl{std::make_unique<TImpl>(*other.m_Impl)} {
}

CTimeSeriesMultibucketVectorMean::~CTimeSeriesMultibucketVectorMean() = default;
CTimeSeriesMultibucketVectorMean::CTimeSeriesMultibucketVectorMean(
    CTimeSeriesMultibucketVectorMean&&) noexcept = default;

CTimeSeriesMultibucketVectorMean& CTimeSeriesMultibucketVectorMean::
operator=(CTimeSeriesMultibucketVectorMean&&) noexcept = default;
CTimeSeriesMultibucketVectorMean& CTimeSeriesMultibucketVectorMean::
operator=(const CTimeSeriesMultibucketVectorMean& other) {
    if (this != &other) {
        CTimeSeriesMultibucketVectorMean tmp{*this};
        *this = std::move(tmp);
    }
    return *this;
}

CTimeSeriesMultibucketVectorMean::TPtr CTimeSeriesMultibucketVectorMean::clone() const {
    return std::make_unique<CTimeSeriesMultibucketVectorMean>(*this);
}

const CTimeSeriesMultibucketVectorMean::TType1VecTWeightAry1VecPr&
CTimeSeriesMultibucketVectorMean::value() const {
    return m_Impl->value();
}

double CTimeSeriesMultibucketVectorMean::correlationWithBucketValue() const {
    return m_Impl->correlationWithBucketValue();
}

void CTimeSeriesMultibucketVectorMean::clear() {
    m_Impl->clear();
}

void CTimeSeriesMultibucketVectorMean::add(core_t::TTime time,
                                           core_t::TTime bucketLength,
                                           const TType1Vec& values,
                                           const TWeightsAry1Vec& weights) {
    m_Impl->add(time, bucketLength, values, weights);
}

std::uint64_t CTimeSeriesMultibucketVectorMean::checksum(std::uint64_t seed) const {
    return m_Impl->checksum(seed);
}

void CTimeSeriesMultibucketVectorMean::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesMultibucketVectorMean");
    core::CMemoryDebug::dynamicSize("m_Impl", m_Impl, mem);
}

std::size_t CTimeSeriesMultibucketVectorMean::staticSize() const {
    return sizeof(*this);
}

std::size_t CTimeSeriesMultibucketVectorMean::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Impl);
}

bool CTimeSeriesMultibucketVectorMean::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CTimeSeriesMultibucketVectorMean::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}
}
}
}
