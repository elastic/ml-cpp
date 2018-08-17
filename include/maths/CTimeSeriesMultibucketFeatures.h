/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesMultibucketFeatures_h
#define INCLUDED_ml_maths_CTimeSeriesMultibucketFeatures_h

#include <core/CPersistUtils.h>
#include <core/CSmallVector.h>
#include <core/CTriple.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CTypeConversions.h>
#include <maths/MathsTypes.h>

#include <boost/circular_buffer.hpp>
#include <boost/make_unique.hpp>

#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ml {
namespace maths {

//! \brief Defines features on collections of time series values.
//!
//! DESCRIPTION:\n
//! The intention of these is to provide useful features for performing
//! anomaly detection. Specifically, unusual values of certain properties
//! of extended intervals of a time series are often the most interesting
//! events in a time series from a user's perspective.
template<typename T>
class CTimeSeriesMultibucketFeature {
public:
    using TT1Vec = core::CSmallVector<T, 1>;
    using TWeightsAry1Vec = core::CSmallVector<maths_t::TWeightsAry<T>, 1>;
    using TT1VecTWeightAry1VecPr = std::pair<TT1Vec, TWeightsAry1Vec>;
    using TPtr = std::unique_ptr<CTimeSeriesMultibucketFeature>;

public:
    virtual ~CTimeSeriesMultibucketFeature() = default;

    //! Clone this feature.
    virtual TPtr clone() const = 0;

    //! Get the feature value.
    virtual TT1VecTWeightAry1VecPr value() const = 0;

    //! Get the correlation of this feature with the bucket value.
    virtual double correlationWithBucketValue() const = 0;

    //! Clear the feature state.
    virtual void clear() = 0;

    //! Update the window with \p values at \p time.
    virtual void add(core_t::TTime time,
                     core_t::TTime bucketLength,
                     const TT1Vec& values,
                     const TWeightsAry1Vec& weights) = 0;

    //! Compute a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const = 0;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the static size of object.
    virtual std::size_t staticSize() const = 0;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const = 0;

    //! Initialize reading state from \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

protected:
    //! Return weight.
    static CFloatStorage minweight(double weight) { return weight; }
    //! Return the minimum weight.
    template<std::size_t N>
    static CFloatStorage minweight(const core::CSmallVector<double, N>& weight) {
        return *std::min_element(weight.begin(), weight.end());
    }

    //! Univariate implementation returns \p value.
    template<typename U, typename V>
    static V conformable(const U& /*x*/, V value) {
        return value;
    }
    //! Multivariate implementation returns the \p value scalar multiple
    //! of the one vector which is conformable with \p x.
    template<typename U, typename V>
    static CVector<V> conformable(const CVector<U>& x, V value) {
        return CVector<V>(x.dimension(), value);
    }

    //! Univariate implementation returns 1.
    template<typename U>
    static std::size_t dimension(const U& /*x*/) {
        return 1;
    }
    //! Multivariate implementation returns the dimension of \p x.
    template<typename U>
    static std::size_t dimension(const CVector<U>& x) {
        return x.dimension();
    }

    //! Univariate implementation returns \p x.
    template<typename U>
    static double toVector(const U& x) {
        return x;
    }
    //! Multivariate implementation returns \p x as the VECTOR type.
    template<typename U>
    static T toVector(const CVector<U>& x) {
        return x.template toVector<T>();
    }
};

//! \brief A weighted mean of a sliding window of time series values.
template<typename T>
class CTimeSeriesMultibucketMean final : public CTimeSeriesMultibucketFeature<T> {
public:
    using TT1Vec = typename CTimeSeriesMultibucketFeature<T>::TT1Vec;
    using TWeightsAry1Vec = typename CTimeSeriesMultibucketFeature<T>::TWeightsAry1Vec;
    using TT1VecTWeightAry1VecPr = typename CTimeSeriesMultibucketFeature<T>::TT1VecTWeightAry1VecPr;
    using TPtr = typename CTimeSeriesMultibucketFeature<T>::TPtr;

public:
    explicit CTimeSeriesMultibucketMean(std::size_t length = 0)
        : m_SlidingWindow(length) {}
    virtual ~CTimeSeriesMultibucketMean() = default;

    //! Clone this feature.
    virtual TPtr clone() const {
        return boost::make_unique<CTimeSeriesMultibucketMean>(*this);
    }

    //! Get the feature value and weight.
    virtual TT1VecTWeightAry1VecPr value() const {
        if (4 * m_SlidingWindow.size() >= 3 * m_SlidingWindow.capacity()) {
            return {{this->mean()}, {maths_t::countWeight(this->count())}};
        }
        return {{}, {}};
    }

    //! Get the correlation of this feature with the bucket value.
    virtual double correlationWithBucketValue() const {
        // This follows from the weighting applied to values in the
        // window and linearity of expectation.
        double r{WINDOW_GEOMETRIC_WEIGHT * WINDOW_GEOMETRIC_WEIGHT};
        double length{static_cast<double>(m_SlidingWindow.size())};
        return length == 0.0 ? 0.0 : std::sqrt((1.0 - r) / (1.0 - std::pow(r, length)));
    }

    //! Clear the feature state.
    virtual void clear() { m_SlidingWindow.clear(); }

    //! Add \p values and \p time.
    virtual void add(core_t::TTime time,
                     core_t::TTime bucketLength,
                     const TT1Vec& values,
                     const TWeightsAry1Vec& weights) {
        // Remove any old samples.
        core_t::TTime cutoff{time - this->windowInterval(bucketLength)};
        while (m_SlidingWindow.size() > 0 && m_SlidingWindow.front().first < cutoff) {
            m_SlidingWindow.pop_front();
        }
        if (values.size() > 0) {
            using TStorage1Vec = core::CSmallVector<TStorage, 1>;

            // Get the scales to apply to each value.
            TStorage1Vec scales;
            scales.reserve(weights.size());
            for (const auto& weight : weights) {
                using std::sqrt;
                scales.push_back(sqrt(TStorage(maths_t::countVarianceScale(weight)) *
                                      TStorage(maths_t::seasonalVarianceScale(weight))));
            }

            // Add the next sample.
            TFloatMeanAccumulator next{this->conformable(TStorage(values[0]), 0.0f)};
            for (std::size_t i = 0; i < values.size(); ++i) {
                auto weight = maths_t::countForUpdate(weights[i]);
                next.add(TStorage(values[i]) / scales[i], this->minweight(weight));
            }
            m_SlidingWindow.push_back({time, next});
        }
    }

    //! Compute a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const {
        return CChecksum::calculate(seed, m_SlidingWindow);
    }

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CTimeSeriesMultibucketMean");
        core::CMemoryDebug::dynamicSize("m_SlidingWindow", m_SlidingWindow, mem);
    }

    //! Get the static size of object.
    virtual std::size_t staticSize() const { return sizeof(*this); }

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(m_SlidingWindow);
    }

    //! Initialize reading state from \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name{traverser.name()};
            RESTORE_SETUP_TEARDOWN(CAPACITY_TAG, std::size_t capacity,
                                   core::CStringUtils::stringToType(traverser.value(), capacity),
                                   m_SlidingWindow.set_capacity(capacity))
            RESTORE(SLIDING_WINDOW_TAG,
                    core::CPersistUtils::restore(SLIDING_WINDOW_TAG, m_SlidingWindow, traverser))
        } while (traverser.next());
        return true;
    }

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(CAPACITY_TAG, m_SlidingWindow.size());
        core::CPersistUtils::persist(SLIDING_WINDOW_TAG, m_SlidingWindow, inserter);
    }

private:
    using TDoubleDoublePr = std::pair<double, double>;
    using TStorage =
        typename std::conditional<std::is_arithmetic<T>::value, CFloatStorage, CVector<CFloatStorage>>::type;
    using TFloatMeanAccumulator = typename CBasicStatistics::SSampleMean<TStorage>::TAccumulator;
    using TDoubleMeanAccumulator = typename SPromoted<TFloatMeanAccumulator>::Type;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;
    using TTimeFloatMeanAccumulatorPrCBuf = boost::circular_buffer<TTimeFloatMeanAccumulatorPr>;

private:
    //! The geometric weight applied to the window.
    constexpr static const double WINDOW_GEOMETRIC_WEIGHT = 0.9;
    //! The state tags.
    static const std::string CAPACITY_TAG;
    static const std::string SLIDING_WINDOW_TAG;

private:
    //! Get the length of the window given \p bucketLength.
    core_t::TTime windowInterval(core_t::TTime bucketLength) const {
        return static_cast<core_t::TTime>(m_SlidingWindow.capacity()) * bucketLength;
    }

    //! Compute the weighted mean of the values in the window.
    T mean() const {
        return evaluateOnWindow(
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::mean(value);
            },
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::count(value);
            });
    }

    //! Compute the weighted count of the values in the window.
    T count() const {
        return evaluateOnWindow(
            [&](const TFloatMeanAccumulator& value) {
                return this->conformable(CBasicStatistics::mean(value),
                                         static_cast<double>(CBasicStatistics::count(value)));
            },
            [](const TFloatMeanAccumulator&) { return 1.0; });
    }

    //! Compute the weighted mean of \p function on the sliding window.
    template<typename VALUE, typename WEIGHT>
    T evaluateOnWindow(VALUE value, WEIGHT weight) const {
        double latest, earliest;
        std::tie(earliest, latest) = this->range();
        double n{static_cast<double>(m_SlidingWindow.size())};
        double scale{(n - 1.0) * (latest == earliest ? 1.0 : 1.0 / (latest - earliest))};
        auto i = m_SlidingWindow.begin();
        TDoubleMeanAccumulator mean{this->conformable(value(i->second), 0.0)};
        for (double last{earliest}; i != m_SlidingWindow.end(); ++i) {
            double dt{static_cast<double>(i->first) - last};
            last = static_cast<double>(i->first);
            mean.age(std::pow(WINDOW_GEOMETRIC_WEIGHT, scale * dt));
            mean.add(value(i->second), weight(i->second));
        }
        return this->toVector(CBasicStatistics::mean(mean));
    }

    //! Compute the time range of window.
    TDoubleDoublePr range() const {
        auto range = std::accumulate(m_SlidingWindow.begin(), m_SlidingWindow.end(),
                                     CBasicStatistics::CMinMax<double>(),
                                     [](CBasicStatistics::CMinMax<double> partial,
                                        const TTimeFloatMeanAccumulatorPr& value) {
                                         partial.add(static_cast<double>(value.first));
                                         return partial;
                                     });
        return {range.min(), range.max()};
    }

private:
    //! The window values.
    TTimeFloatMeanAccumulatorPrCBuf m_SlidingWindow;
};

template<typename T>
const std::string CTimeSeriesMultibucketMean<T>::CAPACITY_TAG{"a"};
template<typename T>
const std::string CTimeSeriesMultibucketMean<T>::SLIDING_WINDOW_TAG{"b"};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesMultibucketFeatures_h
