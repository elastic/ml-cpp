/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CModelTools.h>

#include <maths/CBasicStatistics.h>
#include <maths/CIntegerTools.h>
#include <maths/CModel.h>
#include <maths/CMultinomialConjugate.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <model/CSample.h>

#include <boost/bind.hpp>

#include <algorithm>
#include <functional>
#include <numeric>

namespace ml {
namespace model {

namespace {

using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1>;

//! \brief Visitor to add a probability to variant of possible
//! aggregation styles.
struct SAddProbability : public boost::static_visitor<void> {
    void operator()(double probability, double weight, maths::CJointProbabilityOfLessLikelySamples& aggregator) const {
        aggregator.add(probability, weight);
    }
    void operator()(double probability, double /*weight*/, maths::CProbabilityOfExtremeSample& aggregator) const {
        aggregator.add(probability);
    }
};

//! \brief Visitor to read aggregate probability from a variant
//! of possible aggregation styles.
struct SReadProbability : public boost::static_visitor<bool> {
    template<typename T>
    bool operator()(double weight, double& result, const T& aggregator) const {
        double probability;
        if (!aggregator.calculate(probability)) {
            LOG_ERROR("Failed to compute probability");
            return false;
        }
        result *= weight < 1.0 ? std::pow(probability, std::max(weight, 0.0)) : probability;
        return true;
    }
    template<typename T>
    bool operator()(TMinAccumulator& result, const T& aggregator) const {
        double probability;
        if (!aggregator.calculate(probability)) {
            LOG_ERROR("Failed to compute probability");
            return false;
        }
        result.add(probability);
        return true;
    }
};
}

void CModelTools::CFuzzyDeduplicate::add(TDouble2Vec value) {
    // We need a very fast way to compute an approximate percentiles
    // for a large collection of samples. It is good enough to simply
    // take a small random sample and compute percentiles on this. We
    // would ideally like to sample each value with equal probability.
    // However, since we can't visit the values in random order we
    // need an approximately uniform online sampling scheme. In fact,
    // we can achieve very nearly uniform sampling by using a modest
    // size sample set, adding each value with probability equal to
    // the count of preceding values and if a value is selected evicting
    // from the sample set uniformly at random. To see this, consider
    // a stream of N values and choose the sample set size to be 100.
    // Note that the chance of the k'th value being in the final sample
    // is given by
    //
    //   P = P(k is sampled) * P(k is not evicted)
    //
    // Also,
    //
    //   P(k is sampled) = 100 / k
    //   P(k is not evicted) ~ (1 - 1/100)^E[# evicted in k+1 to N]
    //
    // It is easy to show by considering a sum of the indicator random
    // variables that the expected number of samples evicted in the
    // range k+1 to N is
    //
    //   100 / (k+1) + 100 / (k+2) + ... + 100 / N
    //
    // This is proportional to the difference of the harmonic numbers
    // H(N) - H(k+1). To leading order this is log(N / (k+1)). Setting
    // k = f N for f in [0,1] we get for moderate N
    //
    //   P ~ 100 / f / N x (1 - 1/100) ^ (100 x log(N / f / N))
    //     = 100 / N / f x exp(log(1 - 1/100) x 100 x log(1 / f))
    //     = 100 / N / f x exp(-(1/100 + O(1/100^2)) x 100 x log(1 / f))
    //     = 100 / N / f x f x f^O(1/100)
    //
    // For even moderate f this very quickly converges to the required
    // constant 100 / N. For example, for f >= 0.05 we have that
    //
    //   98.5 / N < P(f N) <= 100 / N

    ++m_Count;
    if (m_RandomSample.size() < 100) {
        m_RandomSample.push_back(std::move(value));
    } else if (maths::CSampling::uniformSample(m_Rng, 0.0, 1.0) < 100.0 / static_cast<double>(m_Count)) {
        std::size_t evict{maths::CSampling::uniformSample(m_Rng, 0, m_RandomSample.size())};
        m_RandomSample[evict].swap(value);
    }
}

void CModelTools::CFuzzyDeduplicate::computeEpsilons(core_t::TTime bucketLength, std::size_t desiredNumberSamples) {
    m_Quantize = m_Count > 0;
    if (m_Quantize) {
        m_QuantizedValues.reserve(std::min(m_Count, desiredNumberSamples));
        m_TimeEps = std::max(bucketLength / 60, core_t::TTime(1));
        m_ValueEps.assign(m_RandomSample[0].size(), 0.0);
        if (m_RandomSample.size() > 1) {
            TDoubleVec values(m_RandomSample.size());
            for (std::size_t i = 0u; i < m_ValueEps.size(); ++i) {
                for (std::size_t j = 0u; j < m_RandomSample.size(); ++j) {
                    values[j] = m_RandomSample[j][i];
                }
                std::size_t p10{values.size() / 10};
                std::size_t p90{(9 * values.size()) / 10};
                std::nth_element(values.begin(), values.begin() + p10, values.end());
                std::nth_element(values.begin() + p10 + 1, values.begin() + p90, values.end());
                m_ValueEps[i] = (values[p90] - values[p10]) / static_cast<double>(desiredNumberSamples);
            }
        }
        m_Count = 0;
    }
}

std::size_t CModelTools::CFuzzyDeduplicate::duplicate(core_t::TTime time, TDouble2Vec value) {
    return !m_Quantize ? m_Count++
                       : m_QuantizedValues
                             .emplace(boost::unordered::piecewise_construct,
                                      std::forward_as_tuple(this->quantize(time), this->quantize(value)),
                                      std::forward_as_tuple(m_QuantizedValues.size()))
                             .first->second;
}

CModelTools::TDouble2Vec CModelTools::CFuzzyDeduplicate::quantize(TDouble2Vec value) const {
    for (std::size_t i = 0u; i < value.size(); ++i) {
        value[i] = m_ValueEps[i] > 0.0 ? m_ValueEps[i] * std::floor(value[i] / m_ValueEps[i]) : value[i];
    }
    return value;
}

core_t::TTime CModelTools::CFuzzyDeduplicate::quantize(core_t::TTime time) const {
    return maths::CIntegerTools::floor(time, m_TimeEps);
}

std::size_t CModelTools::CFuzzyDeduplicate::SDuplicateValueHash::operator()(const TTimeDouble2VecPr& value) const {
    return static_cast<std::size_t>(
        std::accumulate(value.second.begin(), value.second.end(), static_cast<uint64_t>(value.first), [](uint64_t seed, double v) {
            return core::CHashing::hashCombine(seed, static_cast<uint64_t>(v));
        }));
}

CModelTools::CProbabilityAggregator::CProbabilityAggregator(EStyle style) : m_Style(style), m_TotalWeight(0.0) {
}

bool CModelTools::CProbabilityAggregator::empty() const {
    return m_TotalWeight == 0.0;
}

void CModelTools::CProbabilityAggregator::add(const TAggregator& aggregator, double weight) {
    switch (m_Style) {
    case E_Sum:
        if (weight > 0.0) {
            m_Aggregators.emplace_back(aggregator, weight);
        }
        break;

    case E_Min:
        m_Aggregators.emplace_back(aggregator, 1.0);
        break;
    }
}

void CModelTools::CProbabilityAggregator::add(double probability, double weight) {
    m_TotalWeight += weight;
    for (auto& aggregator : m_Aggregators) {
        boost::apply_visitor(boost::bind<void>(SAddProbability(), probability, weight, _1), aggregator.first);
    }
}

bool CModelTools::CProbabilityAggregator::calculate(double& result) const {
    result = 1.0;

    if (m_TotalWeight == 0.0) {
        LOG_TRACE("No samples");
        return true;
    }

    if (m_Aggregators.empty()) {
        LOG_ERROR("No probability aggregators specified");
        return false;
    }

    double p{1.0};

    switch (m_Style) {
    case E_Sum: {
        double n{0.0};
        for (const auto& aggregator : m_Aggregators) {
            n += aggregator.second;
        }
        for (const auto& aggregator : m_Aggregators) {
            if (!boost::apply_visitor(boost::bind<bool>(SReadProbability(), aggregator.second / n, boost::ref(p), _1), aggregator.first)) {
                return false;
            }
        }
        break;
    }
    case E_Min: {
        TMinAccumulator p_;
        for (const auto& aggregator : m_Aggregators) {
            if (!boost::apply_visitor(boost::bind<bool>(SReadProbability(), boost::ref(p_), _1), aggregator.first)) {
                return false;
            }
        }
        if (p_.count() > 0) {
            p = p_[0];
        }
        break;
    }
    }

    if (p < 0.0 || p > 1.001) {
        LOG_ERROR("Unexpected probability = " << p);
    }
    result = maths::CTools::truncate(p, maths::CTools::smallestProbability(), 1.0);

    return true;
}

CModelTools::CCategoryProbabilityCache::CCategoryProbabilityCache() : m_Prior(nullptr), m_SmallestProbability(1.0) {
}

CModelTools::CCategoryProbabilityCache::CCategoryProbabilityCache(const maths::CMultinomialConjugate& prior)
    : m_Prior(&prior), m_SmallestProbability(1.0) {
}

bool CModelTools::CCategoryProbabilityCache::lookup(std::size_t attribute, double& result) const {
    result = 1.0;
    if (!m_Prior || m_Prior->isNonInformative()) {
        return false;
    }

    if (m_Cache.empty()) {
        TDoubleVec lb;
        TDoubleVec ub;
        m_Prior->probabilitiesOfLessLikelyCategories(maths_t::E_TwoSided, lb, ub);
        LOG_TRACE("P({c}) >= " << core::CContainerPrinter::print(lb));
        LOG_TRACE("P({c}) <= " << core::CContainerPrinter::print(ub));
        m_Cache.swap(lb);
        m_SmallestProbability = 1.0;
        for (std::size_t i = 0u; i < ub.size(); ++i) {
            m_Cache[i] = (m_Cache[i] + ub[i]) / 2.0;
            m_SmallestProbability = std::min(m_SmallestProbability, m_Cache[i]);
        }
    }

    std::size_t index;
    result = (!m_Prior->index(static_cast<double>(attribute), index) || index >= m_Cache.size()) ? m_SmallestProbability : m_Cache[index];
    return true;
}

void CModelTools::CCategoryProbabilityCache::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CTools::CLessLikelyProbability");
    core::CMemoryDebug::dynamicSize("m_Cache", m_Cache, mem->addChild());
    if (m_Prior) {
        m_Prior->debugMemoryUsage(mem->addChild());
    }
}

std::size_t CModelTools::CCategoryProbabilityCache::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Cache)};
    if (m_Prior) {
        mem += m_Prior->memoryUsage();
    }
    return mem;
}

CModelTools::CProbabilityCache::CProbabilityCache(double maximumError) : m_MaximumError(maximumError) {
}

void CModelTools::CProbabilityCache::clear() {
    m_Caches.clear();
}

void CModelTools::CProbabilityCache::addModes(model_t::EFeature feature, std::size_t id, const maths::CModel& model) {
    if (model_t::dimension(feature) == 1) {
        TDouble1Vec& modes{m_Caches[{feature, id}].s_Modes};
        if (modes.empty()) {
            TDouble2Vec1Vec modes_(
                model.residualModes(maths::CConstantWeights::COUNT_VARIANCE, maths::CConstantWeights::unit<TDouble2Vec>(1)));
            for (const auto& mode : modes_) {
                modes.push_back(mode[0]);
            }
            std::sort(modes.begin(), modes.end());
        }
    }
}

void CModelTools::CProbabilityCache::addProbability(model_t::EFeature feature,
                                                    std::size_t id,
                                                    const TDouble2Vec1Vec& value,
                                                    double probability,
                                                    const TTail2Vec& tail,
                                                    bool conditional,
                                                    const TSize1Vec& mostAnomalousCorrelate) {
    if (m_MaximumError > 0.0 && value.size() == 1 && value[0].size() == 1) {
        m_Caches[{feature, id}].s_Probabilities.emplace(value[0][0], SProbability{probability, tail, conditional, mostAnomalousCorrelate});
    }
}

bool CModelTools::CProbabilityCache::lookup(model_t::EFeature feature,
                                            std::size_t id,
                                            const TDouble2Vec1Vec& value,
                                            double& probability,
                                            TTail2Vec& tail,
                                            bool& conditional,
                                            TSize1Vec& mostAnomalousCorrelate) const {
    // The idea of this cache is to:
    //   1. Check that the requested value x is in a region where the
    //      probability as a function of value is monotonic
    //   2. Check that the difference in the probability at the end
    //      points of the interval [a, b] including x is less than the
    //      required tolerance, hence by 1 that using the interpolated
    //      value won't introduce an error greater than the tolerance.
    //
    // To achieve 1 we note that the function is monotonic on an interval
    // [a, b] if we can verify it doesn't contain more than one stationary
    // points and the gradients satisfy P'(a) * P'(b) > 0.

    if (m_MaximumError > 0.0 && value.size() == 1 && value[0].size() == 1) {
        auto pos = m_Caches.find({feature, id});
        if (pos != m_Caches.end()) {
            double x{value[0][0]};
            const TDouble1Vec& modes{pos->second.s_Modes};
            const TDoubleProbabilityFMap& probabilities{pos->second.s_Probabilities};
            auto right = probabilities.lower_bound(x);

            if (right != probabilities.end() && right->first == x) {
                probability = right->second.s_Probability;
                tail = right->second.s_Tail;
                conditional = right->second.s_Conditional;
                mostAnomalousCorrelate = right->second.s_MostAnomalousCorrelate;
                return true;
            } else if (right != probabilities.end() && right + 1 != probabilities.end() && right != probabilities.begin() &&
                       right - 1 != probabilities.begin() && right - 2 != probabilities.begin()) {
                auto left = right - 1;
                double v[]{(left - 1)->first, left->first, right->first, (right + 1)->first};
                auto beginModes = std::lower_bound(modes.begin(), modes.end(), v[0]);
                auto endModes = std::lower_bound(modes.begin(), modes.end(), v[3]);
                LOG_TRACE("v = " << core::CContainerPrinter::print(v));

                if (beginModes == endModes && left->second.s_Tail == right->second.s_Tail) {
                    double p[]{(left - 1)->second.s_Probability,
                               (left)->second.s_Probability,
                               (right)->second.s_Probability,
                               (right + 1)->second.s_Probability};
                    LOG_TRACE("p(v) = " << core::CContainerPrinter::print(p));

                    if (std::is_sorted(p, p + 4, std::less<double>()) || std::is_sorted(p, p + 4, std::greater<double>())) {
                        auto nearest = x - v[1] < v[2] - x ? left : right;
                        probability = (p[2] * (x - v[1]) + p[1] * (v[2] - x)) / (v[2] - v[1]);
                        tail = nearest->second.s_Tail;
                        conditional = nearest->second.s_Conditional;
                        mostAnomalousCorrelate = nearest->second.s_MostAnomalousCorrelate;
                        return std::fabs(p[2] - p[1]) <= m_MaximumError * std::min(p[1], p[2]);
                    }
                }
            }
        }
    }

    return false;
}
}
}
