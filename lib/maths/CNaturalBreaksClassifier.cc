/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CNaturalBreaksClassifier.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CMathsFuncs.h>
#include <maths/CRestoreParams.h>
#include <maths/CSampling.h>

#include <boost/algorithm/cxx11/is_sorted.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <cmath>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {

namespace {

//! Orders two tuples by their mean.
struct SMeanLess {
    bool operator()(const CNaturalBreaksClassifier::TTuple& lhs,
                    const CNaturalBreaksClassifier::TTuple& rhs) const {
        return CBasicStatistics::mean(lhs) < CBasicStatistics::mean(rhs);
    }
};

//! Checks if a tuple count is less than a specified value.
class CCountLessThan {
public:
    CCountLessThan(double count) : m_Count(count) {}

    bool operator()(const CNaturalBreaksClassifier::TTuple& tuple) const {
        return CBasicStatistics::count(tuple) < m_Count;
    }

private:
    double m_Count;
};
const core::TPersistenceTag SPACE_TAG("a", "space");
const core::TPersistenceTag CATEGORY_TAG("b", "category");
const core::TPersistenceTag POINTS_TAG("c", "points");
const core::TPersistenceTag DECAY_RATE_TAG("d", "decay_rate");

const std::string EMPTY_STRING;
}

CNaturalBreaksClassifier::CNaturalBreaksClassifier(std::size_t space,
                                                   double decayRate,
                                                   double minimumCategoryCount)
    : m_Space(std::max(space, MINIMUM_SPACE)), m_DecayRate(decayRate),
      m_MinimumCategoryCount(minimumCategoryCount) {
    m_Categories.reserve(m_Space + MAXIMUM_BUFFER_SIZE + 1u);
    m_PointsBuffer.reserve(MAXIMUM_BUFFER_SIZE);
}

bool CNaturalBreaksClassifier::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                                      core::CStateRestoreTraverser& traverser) {
    m_DecayRate = params.s_DecayRate;
    m_MinimumCategoryCount = params.s_MinimumCategoryCount;

    do {
        const std::string& name = traverser.name();
        RESTORE(DECAY_RATE_TAG, m_DecayRate.fromString(traverser.value()))
        RESTORE_BUILT_IN(SPACE_TAG, m_Space)
        RESTORE(CATEGORY_TAG, core::CPersistUtils::restore(CATEGORY_TAG, m_Categories, traverser))
        RESTORE(POINTS_TAG, core::CPersistUtils::fromString(traverser.value(), m_PointsBuffer))
    } while (traverser.next());

    return true;
}

void CNaturalBreaksClassifier::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, m_DecayRate.toString());
    inserter.insertValue(SPACE_TAG, m_Space);
    core::CPersistUtils::persist(CATEGORY_TAG, m_Categories, inserter);
    inserter.insertValue(POINTS_TAG, core::CPersistUtils::toString(m_PointsBuffer));
}

double CNaturalBreaksClassifier::percentile(double p) const {
    LOG_TRACE(<< "percentile = " << p);
    p /= 100.0;

    double percentileCount = 0.0;
    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        percentileCount += CBasicStatistics::count(m_Categories[i]);
    }
    percentileCount *= p;
    LOG_TRACE(<< "percentileCount = " << percentileCount);

    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        double count = CBasicStatistics::count(m_Categories[i]);
        if (percentileCount < count) {
            double mean = CBasicStatistics::mean(m_Categories[i]);
            double deviation = std::sqrt(
                CBasicStatistics::maximumLikelihoodVariance(m_Categories[i]));
            if (deviation == 0.0) {
                return mean;
            }

            boost::math::normal_distribution<> normal(mean, deviation);
            double q = (count - percentileCount) / count;
            double x = q > 0.0 && q < 1.0
                           ? boost::math::quantile(normal, q)
                           : (2.0 * q - 1.0) * boost::numeric::bounds<double>::highest();
            LOG_TRACE(<< "N(" << mean << "," << deviation << ")"
                      << ", q = " << q << ", x = " << x);

            if (i > 0) {
                // Left truncate by the assignment boundary between
                // this and the left category. See deviation for
                // details.
                double n1 = std::sqrt(CBasicStatistics::count(m_Categories[i - 1]));
                double m1 = CBasicStatistics::mean(m_Categories[i - 1]);
                double d1 = std::sqrt(
                    CBasicStatistics::maximumLikelihoodVariance(m_Categories[i - 1]));
                double n2 = count;
                double m2 = mean;
                double d2 = deviation;
                double w1 = std::sqrt(n2 * d2);
                double w2 = std::sqrt(n1 * d1);
                double xl = (w1 * m1 + w2 * m2) / (w1 + w2);
                LOG_TRACE(<< "Left truncate to " << xl);
                x = std::max(x, xl);
            }
            if (i + 1 < m_Categories.size()) {
                // Right truncate by the assignment boundary between
                // this and the right category. See deviation for
                // details.
                double n1 = count;
                double m1 = mean;
                double d1 = deviation;
                double n2 = std::sqrt(CBasicStatistics::count(m_Categories[i + 1]));
                double m2 = CBasicStatistics::mean(m_Categories[i + 1]);
                double d2 = std::sqrt(
                    CBasicStatistics::maximumLikelihoodVariance(m_Categories[i + 1]));
                double w1 = std::sqrt(n2 * d2);
                double w2 = std::sqrt(n1 * d1);
                double xr = (w1 * m1 + w2 * m2) / (w1 + w2);
                LOG_TRACE(<< "Right truncate to " << xr);
                x = std::min(x, xr);
            }

            return x;
        }
        percentileCount -= count;
    }

    return (p <= 0.5 ? -1.0 : 1.0) * boost::numeric::bounds<double>::highest();
}

std::size_t CNaturalBreaksClassifier::size() const {
    return std::min(m_Categories.size() + m_PointsBuffer.size(), m_Space);
}

bool CNaturalBreaksClassifier::split(std::size_t n, std::size_t p, TClassifierVec& result) {
    LOG_TRACE(<< "split");

    result.clear();

    if (n == 0) {
        LOG_ERROR(<< "Bad request for zero categories");
        return false;
    }

    this->reduce();
    LOG_TRACE(<< "raw categories = " << this->print());

    if (n >= m_Categories.size()) {
        double p_ = static_cast<double>(p);
        for (std::size_t i = 0u; p_ > 0.0 && i < m_Categories.size(); ++i) {
            if (CBasicStatistics::count(m_Categories[i]) < p_) {
                return false;
            }
        }
        result.reserve(m_Categories.size());
        TTupleVec category(1);
        for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
            category[0] = m_Categories[i];
            result.push_back(CNaturalBreaksClassifier(
                m_Space, m_DecayRate, m_MinimumCategoryCount, category));
        }
        return true;
    } else if (n == 1) {
        double p_ = static_cast<double>(p);
        double count = 0.0;
        for (std::size_t i = 0u; p_ > 0.0 && i < m_Categories.size(); ++i) {
            count += CBasicStatistics::count(m_Categories[i]);
        }
        if (count < p_) {
            return false;
        }
        result.push_back(*this);
        return true;
    }

    TSizeVec split;
    if (!this->naturalBreaks(n, p, split)) {
        return false;
    }

    result.reserve(n);
    for (std::size_t i = 0u, j = 0u; i < split.size(); ++i) {
        TTupleVec categories;
        for (/**/; j < split[i]; ++j) {
            categories.push_back(m_Categories[j]);
        }
        result.push_back(CNaturalBreaksClassifier(
            m_Space, m_DecayRate, m_MinimumCategoryCount, categories));
    }

    return true;
}

bool CNaturalBreaksClassifier::split(const TSizeVec& split, TClassifierVec& result) {
    result.clear();

    this->reduce();

    // Sanity checks.
    if (split.empty() || split[split.size() - 1] != m_Categories.size() ||
        !std::is_sorted(split.begin(), split.end())) {
        LOG_ERROR(<< "Bad split = " << core::CContainerPrinter::print(split));
        return false;
    }

    result.reserve(split.size());
    TTupleVec categories;
    for (std::size_t i = 0u, j = 0u; i < split.size(); ++i) {
        categories.clear();
        categories.reserve(split[i] - j);
        for (/**/; j < split[i]; ++j) {
            categories.push_back(m_Categories[j]);
        }
        result.push_back(CNaturalBreaksClassifier(
            m_Space, m_DecayRate, m_MinimumCategoryCount, categories));
    }

    return true;
}

bool CNaturalBreaksClassifier::naturalBreaks(std::size_t n, std::size_t p, TSizeVec& result) {
    return naturalBreaksImpl(m_Categories, n, p, E_TargetDeviation, result);
}

bool CNaturalBreaksClassifier::categories(std::size_t n, std::size_t p, TTupleVec& result, bool append) {
    LOG_TRACE(<< "categories");

    if (!append) {
        result.clear();
    }

    if (n == 0) {
        LOG_ERROR(<< "Bad request for zero categories");
        return false;
    }

    this->reduce();
    LOG_TRACE(<< "raw categories = " << this->print());

    if (n >= m_Categories.size()) {
        double p_ = static_cast<double>(p);
        for (std::size_t i = 0u; p_ > 0.0 && i < m_Categories.size(); ++i) {
            if (CBasicStatistics::count(m_Categories[i]) < p_) {
                return false;
            }
        }
        if (!append) {
            result = m_Categories;
        } else {
            result.insert(result.end(), m_Categories.begin(), m_Categories.end());
        }
        return true;
    } else if (n == 1) {
        double p_ = static_cast<double>(p);
        TTuple category =
            std::accumulate(m_Categories.begin(), m_Categories.end(), TTuple());
        if (CBasicStatistics::count(category) < p_) {
            return false;
        }
        result.push_back(category);
        return true;
    }

    TSizeVec split;
    if (!this->naturalBreaks(n, p, split)) {
        return false;
    }

    result.reserve(result.size() + n);
    for (std::size_t i = 0u, j = 0u; i < split.size(); ++i) {
        TTuple category;
        for (/**/; j < split[i]; ++j) {
            category += m_Categories[j];
        }
        result.push_back(category);
    }

    return true;
}

bool CNaturalBreaksClassifier::categories(const TSizeVec& split, TTupleVec& result) {
    result.clear();

    // Sanity checks.
    if (split.empty() || split[split.size() - 1] != m_Categories.size() ||
        !std::is_sorted(split.begin(), split.end())) {
        LOG_ERROR(<< "Bad split = " << core::CContainerPrinter::print(split));
        return false;
    }

    result.reserve(split.size());
    for (std::size_t i = 0u, j = 0u; i < split.size(); ++i) {
        TTuple category;
        for (/**/; j < split[i]; ++j) {
            category += m_Categories[j];
        }
        result.push_back(category);
    }

    return true;
}

void CNaturalBreaksClassifier::add(double x, double count) {
    LOG_TRACE(<< "Adding " << x);

    if (m_PointsBuffer.size() < MAXIMUM_BUFFER_SIZE) {
        m_PointsBuffer.emplace_back(x, count);
    } else {
        m_Categories.push_back(TTuple());
        m_Categories.back().add(x, count);
        this->reduce();
    }
}

void CNaturalBreaksClassifier::merge(const CNaturalBreaksClassifier& other) {
    LOG_TRACE(<< "Merge");

    for (std::size_t i = 0u; i < other.m_PointsBuffer.size(); ++i) {
        m_Categories.push_back(TTuple());
        m_Categories.back().add(other.m_PointsBuffer[i].first,
                                other.m_PointsBuffer[i].second);
    }
    m_Categories.insert(m_Categories.end(), other.m_Categories.begin(),
                        other.m_Categories.end());

    this->reduce();

    // Reclaim memory from the vector buffer.
    TTupleVec categories(m_Categories);
    m_Categories.swap(categories);
}

void CNaturalBreaksClassifier::decayRate(double decayRate) {
    m_DecayRate = decayRate;
}

void CNaturalBreaksClassifier::propagateForwardsByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR(<< "Can't propagate backwards in time");
        return;
    }

    double alpha = std::exp(-m_DecayRate * time);
    LOG_TRACE(<< "alpha = " << alpha);
    LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(m_Categories));

    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        m_Categories[i].age(alpha);
    }

    // Prune any dead categories: we're not interested in maintaining
    // categories with low counts.
    m_Categories.erase(std::remove_if(m_Categories.begin(), m_Categories.end(),
                                      CCountLessThan(m_MinimumCategoryCount)),
                       m_Categories.end());

    LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(m_Categories));
}

bool CNaturalBreaksClassifier::buffering() const {
    return m_PointsBuffer.size() > 0;
}

void CNaturalBreaksClassifier::sample(std::size_t numberSamples,
                                      double smallest,
                                      double /*largest*/,
                                      TDoubleVec& result) const {
    result.clear();
    if (numberSamples == 0) {
        return;
    }

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    static const double ALMOST_ONE = 0.99999;

    // See, for example, Effective C++ item 3.
    const_cast<CNaturalBreaksClassifier*>(this)->reduce();
    LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(m_Categories));

    TDoubleVec weights;
    weights.reserve(m_Categories.size());
    double weightSum = 0.0;

    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        double nCategory = CBasicStatistics::count(m_Categories[i]);
        weights.push_back(nCategory);
        weightSum += nCategory;
    }
    for (std::size_t i = 0u; i < weights.size(); ++i) {
        weights[i] /= weightSum;
    }

    numberSamples = std::min(numberSamples, static_cast<std::size_t>(weightSum));
    LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(weights)
              << ", weightSum = " << weightSum << ", n = " << numberSamples);

    result.reserve(numberSamples);

    TMeanAccumulator sample;
    TDoubleVec categorySamples;
    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        double ni = static_cast<double>(numberSamples) * weights[i];
        std::size_t ni_ = static_cast<std::size_t>(std::ceil(ni));

        double m = CBasicStatistics::mean(m_Categories[i]);
        double v = CBasicStatistics::maximumLikelihoodVariance(m_Categories[i]);

        CSampling::normalSampleQuantiles(m, v, ni_, categorySamples);
        for (std::size_t j = 0u; j < categorySamples.size(); ++j) {
            if (categorySamples[j] < smallest) {
                if (v == 0.0) {
                    categorySamples.assign(ni_, smallest);
                } else {
                    m -= std::min(smallest, 0.0);
                    double shape = m * m / v;
                    double rate = m / v;
                    CSampling::gammaSampleQuantiles(shape, rate, ni_, categorySamples);
                    for (std::size_t k = 0u; k < categorySamples.size(); ++k) {
                        categorySamples[k] += std::min(smallest, 0.0);
                    }
                }
                break;
            }
        }

        if (!categorySamples.empty()) {
            ni /= static_cast<double>(categorySamples.size());
            for (std::size_t j = 0u; j < categorySamples.size(); ++j) {
                double nij = std::min(1.0 - CBasicStatistics::count(sample), ni);
                sample.add(categorySamples[j], nij);
                if (CBasicStatistics::count(sample) > ALMOST_ONE) {
                    result.push_back(CBasicStatistics::mean(sample));
                    sample = nij < ni ? CBasicStatistics::momentsAccumulator(
                                            ni - nij, categorySamples[j])
                                      : TMeanAccumulator{};
                }
            }
        }
    }

    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(result));
}

std::string CNaturalBreaksClassifier::print() const {
    return core::CContainerPrinter::print(m_Categories);
}

uint64_t CNaturalBreaksClassifier::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Space);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_Categories);
    return CChecksum::calculate(seed, m_PointsBuffer);
}

void CNaturalBreaksClassifier::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CNaturalBreaksClassifier");
    core::CMemoryDebug::dynamicSize("m_Categories", m_Categories, mem);
    core::CMemoryDebug::dynamicSize("m_PointsBuffer", m_PointsBuffer, mem);
}

std::size_t CNaturalBreaksClassifier::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_Categories);
    mem += core::CMemory::dynamicSize(m_PointsBuffer);
    return mem;
}

bool CNaturalBreaksClassifier::naturalBreaks(const TTupleVec& categories,
                                             std::size_t n,
                                             std::size_t p,
                                             EObjective target,
                                             TSizeVec& result) {
    return naturalBreaksImpl(categories, n, p, target, result);
}

bool CNaturalBreaksClassifier::naturalBreaks(const TDoubleTupleVec& categories,
                                             std::size_t n,
                                             std::size_t p,
                                             EObjective target,
                                             TSizeVec& result) {
    return naturalBreaksImpl(categories, n, p, target, result);
}

template<typename TUPLE>
bool CNaturalBreaksClassifier::naturalBreaksImpl(const std::vector<TUPLE>& categories,
                                                 std::size_t n,
                                                 std::size_t p,
                                                 EObjective target,
                                                 TSizeVec& result) {
    result.clear();

    if (categories.empty()) {
        return true;
    }

    if (n == 0) {
        LOG_ERROR(<< "Bad request for zero categories");
        return false;
    }

    if (n >= categories.size()) {
        result.reserve(categories.size());
        for (std::size_t i = 1u; i < categories.size(); ++i) {
            result.push_back(i);
        }
        result.push_back(categories.size());
        return true;
    } else if (n == 1) {
        result.push_back(categories.size());
        return true;
    }

    // See Wang and Song for details. Their argument goes
    // through essentially unmodified to work on the tuples
    // we store and the deviation objective. In the following
    // I've adopted the notation from their paper. Note that
    // there is an error in the initialization of the dynamic
    // program they give. This can be fixed by using N x n
    // matrix for D where the first column are the one cluster
    // deviations and all other entries are 0.
    //
    // In order to impose a constraint on the minimum cluster
    // size we simply need to set any matrix entries that
    // violate the constraint to some large value (which will
    // mean it is never chosen).

    // A proxy for infinity since operations on infinity are
    // slow and this will be well out of range of any real
    // values.

    result.clear();

    static const double INF = boost::numeric::bounds<double>::highest();

    double pp = static_cast<double>(p);

    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVecVec = std::vector<TSizeVec>;

    std::size_t N = categories.size();

    TSizeVecVec B(N, TSizeVec(n, 0));
    TDoubleVecVec D(N, TDoubleVec(n, 0.0));
    {
        TTuple t;
        for (std::size_t i = 0u; i < N; ++i) {
            t += categories[i];
            D[i][0] = CBasicStatistics::count(t) < pp ? INF : objective(target, t);
        }
    }

    LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(categories));

    for (std::size_t i = 1u; i < N; ++i) {
        for (std::size_t m = 1u; m <= std::min(i, n - 1); ++m) {
            std::size_t b = m + 1;
            double d = INF;

            TTuple t;
            for (std::size_t j = i; j >= m; --j) {
                t += categories[j];
                double c = (D[j - 1][m - 1] == INF || CBasicStatistics::count(t) < pp)
                               ? INF
                               : D[j - 1][m - 1] + objective(target, t);
                if (c <= d) {
                    b = j;
                    d = c;
                }
            }

            B[i][m] = b;
            D[i][m] = d;
        }
    }

    if (D[N - 1][n - 1] == INF) {
        return false;
    }

    LOG_TRACE(<< "D = " << core::CContainerPrinter::print(D));
    LOG_TRACE(<< "B = " << core::CContainerPrinter::print(B));

    // Find the solution by back tracking. Note that we return
    // the end points of the half open intervals comprising
    // the solution. In particular, if the intervals comprise
    // the set {[0, j(1)), [j(1), j(2)), ... , [j(n-1), N)}
    // we return the set {j(1), j(2), ... N}.

    result.resize(n, 0);
    result[n - 1] = N;
    result[n - 2] = B[N - 1][n - 1];
    for (std::size_t i = 3u; i <= n; ++i) {
        result[n - i] = B[result[n - i + 1] - 1][n - i + 1];
    }

    LOG_TRACE(<< "result = " << core::CContainerPrinter::print(result));

    return true;
}

CNaturalBreaksClassifier::CNaturalBreaksClassifier(std::size_t space,
                                                   double decayRate,
                                                   double minimumCategoryCount,
                                                   TTupleVec& categories)
    : m_Space(space), m_DecayRate(decayRate),
      m_MinimumCategoryCount(minimumCategoryCount) {
    m_Categories.swap(categories);
    m_Categories.reserve(m_Space + MAXIMUM_BUFFER_SIZE + 1u);
    m_PointsBuffer.reserve(MAXIMUM_BUFFER_SIZE);
}

void CNaturalBreaksClassifier::reduce() {
    LOG_TRACE(<< "Reduce");

    // Experimenting with using the optimal reduction gives no
    // significant QoR improvements, but results in much longer
    // run times. It varies with m_Space, but is typically about
    // an order of magnitude slower.

    // Add all the points as new categories tuples and reduce.
    for (std::size_t i = 0u; i < m_PointsBuffer.size(); ++i) {
        m_Categories.push_back(TTuple());
        m_Categories.back().add(m_PointsBuffer[i].first, m_PointsBuffer[i].second);
    }
    m_PointsBuffer.clear();

    std::sort(m_Categories.begin(), m_Categories.end(), SMeanLess());
    LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(m_Categories));

    while (m_Categories.size() > m_Space) {
        // Find the tuples to merge.
        TSizeSizePr toMerge = this->closestPair();

        // Merge and tidy up.
        m_Categories[toMerge.first] += m_Categories[toMerge.second];
        m_Categories.erase(m_Categories.begin() + toMerge.second);
    }

    LOG_TRACE(<< "reduced categories = " << core::CContainerPrinter::print(m_Categories));
}

CNaturalBreaksClassifier::TSizeSizePr CNaturalBreaksClassifier::closestPair() const {
    LOG_TRACE(<< "Closest pair");

    TSizeSizePr result;

    double dDeviationMin = boost::numeric::bounds<double>::highest();
    for (std::size_t i = 1u; i < m_Categories.size(); ++i) {
        double dDeviation = deviation(m_Categories[i] + m_Categories[i - 1]) -
                            deviation(m_Categories[i]) - deviation(m_Categories[i - 1]);

        LOG_TRACE(<< "mean[" << i - 1
                  << "] = " << CBasicStatistics::mean(m_Categories[i - 1]) << ", mean["
                  << i << "] = " << CBasicStatistics::mean(m_Categories[i])
                  << ", dDeviation = " << dDeviation);

        if (dDeviation < dDeviationMin) {
            result = TSizeSizePr(i - 1, i);
            dDeviationMin = dDeviation;
        }
    }

    LOG_TRACE(<< "Closest pair = " << core::CContainerPrinter::print(result));

    return result;
}

double CNaturalBreaksClassifier::deviation(const TTuple& category) {
    // The deviation objective is in some senses more natural
    // than the variation in one dimension. In particular, the
    // distances of the class boundaries from the class centers
    // are weighted by their standard deviations. To see this
    // consider the following set up. Assume two classes have
    // means m1 and m2 and standard deviations d1 and d2. Further
    // suppose that the number of points in each class is high
    // so that the impact of an extra point on the class mean
    // and standard deviation is small. Consider, where the
    // decision boundary will be between class 1 and 2 for the
    // objectives:
    //   V(.) = min(  Sum_{x(i) in C1}{ (x(i) - m1)^2 }
    //              + Sum_{x(i) in C2}{ (x(i) - m2)^2 } )
    //
    // and
    //   D(.) = min(  (Sum_{x(i) in C1}{ (x(i) - m1)^2 })^(1/p)
    //              + (Sum_{x(i) in C2}{ (x(i) - m2)^2 })^(1/p) )
    //
    // The decision boundary occurs when a point x*-e will be
    // assigned to class C1 and x*+e will be assigned to C2 for
    // small e. Clearly x* is in [m1, m2] and also if a point
    // is added at m1 to C1 to then neither V nor D are modified
    // likewise for m2 and C2. We are looking for a point x* s.t.
    // the change in V (or D) adding the point to C1 equals the
    // change adding it to C2. Using our boundary condition and
    // ignoring the change in m1 and m2 (as per our assumption).
    // This happens when:
    //   c + (x* - m1)^2 = c + (x* - m2)^2                  (1)
    //
    // for some constant c and the variation objective and
    //   (c1^2 + (x* - m1)^2)^(1/p) + c2^(2/p)
    //             = c1^(2/p) + (c2^2 + (x* - m2)^2)^(1/p)  (2)
    //
    // for constants c1 and c2 and the deviation objective. Note
    // that c1^2 = n1 * d1^2 and c2^2 = n2 * d2^2. Solving (1)
    // gives
    //   x*(V) = (m1 + m2) / 2
    //
    // i.e. the mid point between C1 and C2. To approximately
    // solve (2) we note that we can write it as
    //   Integral_{[m1, x*]}{ d((c1^2 + (y - m1)^2)^(1/p) + c2^(2/p))/dy }dy
    //     = Integral_{[x*,m2]}{ d(c1^(2/p) + (c2^2 + (m2 - y)^2)^(1/p)/dy }dy
    //
    // Using our simplifying assumption that the deviations of
    // the two classes are unaffected by adding the point and
    // integrating, gives that x*(D) satisfies:
    //   ((x*(D) - m1) / c1^((p-1)/p))^2 = ((x*(D) - m2) / c2^((p-1)/p))^2
    //
    // as required. The boundary is at
    //   x*(D) = (  (n2^(1/2) * d2)^((p-1)/p) * m1
    //            + (n1^(1/2) * d1)^((p-1)/p) * m2 )
    //           / ((n1^(1/2) * d1)^((p-1)/p) + (n2^(1/2) * d2)^((p-1)/p))
    //
    // Note that for large p this suffers very badly from
    // cancellation errors so we just use p = 2 in practice.

    double count = CBasicStatistics::count(category);
    double variance = CBasicStatistics::maximumLikelihoodVariance(category);
    return std::sqrt(count * variance);
}

double CNaturalBreaksClassifier::variation(const TTuple& category) {
    double count = CBasicStatistics::count(category);
    double variance = CBasicStatistics::maximumLikelihoodVariance(category);
    return count * variance;
}

const std::size_t CNaturalBreaksClassifier::MINIMUM_SPACE(2u);
const std::size_t CNaturalBreaksClassifier::MAXIMUM_BUFFER_SIZE(2u);
}
}
