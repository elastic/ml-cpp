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

#include <config/CDataSummaryStatistics.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>

#include <maths/CMixtureDistribution.h>
#include <maths/COrderings.h>
#include <maths/CSampling.h>

#include <config/CTools.h>

#include <boost/math/distributions/normal.hpp>

namespace ml {
namespace config {
namespace {

typedef core::CFunctional::SDereference<maths::COrderings::SSecondLess> TDerefSecondLess;
typedef core::CFunctional::SDereference<maths::COrderings::SSecondGreater> TDerefSecondGreater;

std::size_t topNSize(std::size_t n) {
    return static_cast<std::size_t>(::ceil(1.5 * static_cast<double>(n)));
}

const std::size_t DS_NUMBER_HASHES = 7;
const std::size_t DS_MAX_SIZE = 1000;
const std::size_t CS_ROWS = 7;
const std::size_t CS_COLUMNS = 5000;
const std::size_t QS_SIZE = 500;
const std::size_t ES_K = 20;
const double CLUSTER_MINIMUM_FRACTION = 0.005;
const double CLUSTER_MINIMUM_COUNT = 10.0;
core::CHashing::CMurmurHash2String HASHER;
double PROBABILITY_TO_SAMPLE_ENTROPY = 0.5;
double PROBABILITY_TO_SAMPLE_N_GRAMS = 0.02;
}

CDataSummaryStatistics::CDataSummaryStatistics(void) : m_Count(0) {
}

uint64_t CDataSummaryStatistics::count(void) const {
    return m_Count;
}

core_t::TTime CDataSummaryStatistics::earliest(void) const {
    return m_Earliest[0];
}

core_t::TTime CDataSummaryStatistics::latest(void) const {
    return m_Latest[0];
}

double CDataSummaryStatistics::meanRate(void) const {
    return static_cast<double>(m_Count) / static_cast<double>(m_Latest[0] - m_Earliest[0]);
}

void CDataSummaryStatistics::add(core_t::TTime time) {
    m_Earliest.add(time);
    m_Latest.add(time);
    ++m_Count;
}

CCategoricalDataSummaryStatistics::CCategoricalDataSummaryStatistics(std::size_t n, std::size_t toApproximate)
    : m_ToApproximate(toApproximate),
      m_Approximating(toApproximate == 0),
      m_DistinctValues(DS_NUMBER_HASHES, DS_MAX_SIZE),
      m_CountSketch(CS_ROWS, CS_COLUMNS),
      m_N(std::max(n, std::size_t(1))),
      m_TopN(topNSize(m_N)), // This is important to stop invalidation of
                             // the lowest top-n iterator by an insertion.
      m_LowestTopN(m_TopN.end()),
      m_EmpiricalEntropy(ES_K),
      m_DistinctNGrams(NUMBER_N_GRAMS, maths::CBjkstUniqueValues(DS_NUMBER_HASHES, DS_MAX_SIZE)),
      m_NGramEmpricalEntropy(NUMBER_N_GRAMS, maths::CEntropySketch(ES_K)) {
}

CCategoricalDataSummaryStatistics::CCategoricalDataSummaryStatistics(const CDataSummaryStatistics& other,
                                                                     std::size_t n,
                                                                     std::size_t toApproximate)
    : CDataSummaryStatistics(other),
      m_ToApproximate(toApproximate),
      m_Approximating(toApproximate == 0),
      m_DistinctValues(DS_NUMBER_HASHES, DS_MAX_SIZE),
      m_CountSketch(CS_ROWS, CS_COLUMNS),
      m_N(std::max(n, std::size_t(1))),
      m_TopN(topNSize(m_N)), // This is important to stop invalidation of
                             // the lowest top-n iterator by an insertion.
      m_LowestTopN(m_TopN.end()),
      m_EmpiricalEntropy(ES_K),
      m_DistinctNGrams(NUMBER_N_GRAMS, maths::CBjkstUniqueValues(DS_NUMBER_HASHES, DS_MAX_SIZE)),
      m_NGramEmpricalEntropy(NUMBER_N_GRAMS, maths::CEntropySketch(ES_K)) {
}

void CCategoricalDataSummaryStatistics::add(core_t::TTime time, const std::string& example) {
    this->CDataSummaryStatistics::add(time);

    std::size_t category;
    if (!m_Approximating) {
        category = CTools::category64(example);
        ++m_ValueCounts[category];
    } else {
        category = CTools::category32(example);
        m_DistinctValues.add(static_cast<uint32_t>(category));
        m_CountSketch.add(static_cast<uint32_t>(category), 1.0);
    }
    m_MinLength.add(example.length());
    m_MaxLength.add(example.length());
    if (maths::CSampling::uniformSample(m_Rng, 0.0, 1.0) < PROBABILITY_TO_SAMPLE_ENTROPY) {
        m_EmpiricalEntropy.add(HASHER(example));
    }
    if (maths::CSampling::uniformSample(m_Rng, 0.0, 1.0) < PROBABILITY_TO_SAMPLE_N_GRAMS) {
        for (std::size_t n = 1u; n <= 5; ++n) {
            this->addNGrams(n, example);
        }
    }

    this->updateCalibrators(category);

    TStrUInt64UMapItr i = m_TopN.find(example);
    if (i == m_TopN.end()) {
        if (m_TopN.size() > topNSize(m_N)) {
            double estimate_ = this->calibratedCount(category);
            if (estimate_ > 0.0) {
                std::size_t estimate = static_cast<std::size_t>(estimate_ + 0.5);
                if (m_LowestTopN == m_TopN.end()) {
                    this->findLowestTopN();
                }
                if (estimate > m_LowestTopN->second) {
                    m_TopN.erase(m_LowestTopN);
                    m_TopN.insert(std::make_pair(example, estimate));
                    this->findLowestTopN();
                }
            }
        } else {
            i = m_TopN.insert(std::make_pair(example, std::size_t(1))).first;
        }
    } else {
        ++i->second;
        if (i == m_LowestTopN) {
            this->findLowestTopN();
        }
    }

    this->approximateIfCardinalityTooHigh();
}

std::size_t CCategoricalDataSummaryStatistics::distinctCount(void) const {
    return !m_Approximating ? m_ValueCounts.size() : m_DistinctValues.number();
}

std::size_t CCategoricalDataSummaryStatistics::minimumLength(void) const {
    return m_MinLength[0];
}

std::size_t CCategoricalDataSummaryStatistics::maximumLength(void) const {
    return m_MaxLength[0];
}

double CCategoricalDataSummaryStatistics::entropy(void) const {
    return m_EmpiricalEntropy.calculate();
}

void CCategoricalDataSummaryStatistics::topN(TStrSizePrVec& result) const {
    result.clear();
    result.reserve(m_N);

    TStrUInt64UMapCItrVec topN;
    this->topN(topN);

    for (std::size_t i = 0u; i < topN.size(); ++i) {
        result.push_back(*topN[i]);
    }
}

double CCategoricalDataSummaryStatistics::meanCountInRemainders(void) const {
    TStrUInt64UMapCItrVec topN;
    this->topN(topN);

    uint64_t total = 0;
    for (std::size_t i = 0u; i < topN.size(); ++i) {
        total += topN[i]->second;
    }

    return static_cast<double>(this->count() - std::min(total, this->count())) /
           static_cast<double>(std::max(static_cast<std::size_t>(m_DistinctValues.number()), m_TopN.size()));
}

void CCategoricalDataSummaryStatistics::addNGrams(std::size_t n, const std::string& example) {
    for (std::size_t i = n; i < example.length(); ++i) {
        std::size_t hash = HASHER(example.substr(i - n, n));
        m_DistinctNGrams[n - 1].add(CTools::category32(hash));
        m_NGramEmpricalEntropy[n - 1].add(hash);
    }
}

void CCategoricalDataSummaryStatistics::approximateIfCardinalityTooHigh(void) {
    typedef TSizeUInt64UMap::const_iterator TSizeUInt64UMapCItr;

    if (m_ValueCounts.size() >= m_ToApproximate) {
        for (TSizeUInt64UMapCItr i = m_ValueCounts.begin(); i != m_ValueCounts.end(); ++i) {
            uint32_t category = CTools::category32(i->first);
            double count = static_cast<double>(i->second);
            m_DistinctValues.add(category);
            m_CountSketch.add(category, count);
        }
    }
}

void CCategoricalDataSummaryStatistics::updateCalibrators(std::size_t category_) {
    uint32_t category = m_Approximating ? static_cast<uint32_t>(category_) : CTools::category32(category_);
    std::size_t i =
        std::lower_bound(m_Calibrators.begin(), m_Calibrators.end(), category, maths::COrderings::SFirstLess()) - m_Calibrators.begin();
    if (i == m_Calibrators.size() || m_Calibrators[i].first != category) {
        if (m_Calibrators.size() < 5) {
            m_Calibrators.insert(m_Calibrators.begin() + i, std::make_pair(category, 1));
        }
    } else {
        ++m_Calibrators[i].second;
    }
}

double CCategoricalDataSummaryStatistics::calibratedCount(std::size_t category) const {
    if (!m_Approximating) {
        return static_cast<double>(m_ValueCounts.find(category)->second);
    }

    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

    TMeanAccumulator error;
    if (m_CountSketch.sketched()) {
        for (std::size_t j = 0u; j < m_Calibrators.size(); ++j) {
            error.add(m_CountSketch.count(m_Calibrators[j].first) - static_cast<double>(m_Calibrators[j].second));
        }
    }
    return m_CountSketch.count(static_cast<uint32_t>(category)) - maths::CBasicStatistics::mean(error);
}

void CCategoricalDataSummaryStatistics::findLowestTopN(void) {
    typedef maths::CBasicStatistics::COrderStatisticsStack<TStrUInt64UMapItr, 1, TDerefSecondLess> TMinAccumulator;
    TMinAccumulator lowest;
    for (TStrUInt64UMapItr i = m_TopN.begin(); i != m_TopN.end(); ++i) {
        lowest.add(i);
    }
    m_LowestTopN = lowest[0];
}

void CCategoricalDataSummaryStatistics::topN(TStrUInt64UMapCItrVec& result) const {
    typedef maths::CBasicStatistics::COrderStatisticsHeap<TStrUInt64UMapCItr, TDerefSecondGreater> TMaxAccumulator;
    TMaxAccumulator topN(m_N);
    for (TStrUInt64UMapCItr i = m_TopN.begin(); i != m_TopN.end(); ++i) {
        topN.add(i);
    }
    topN.sort();
    result.assign(topN.begin(), topN.end());
}

CNumericDataSummaryStatistics::CNumericDataSummaryStatistics(bool integer)
    : m_NonNumericCount(0),
      m_QuantileSketch(maths::CQuantileSketch::E_Linear, QS_SIZE),
      m_Clusters(integer ? maths_t::E_IntegerData : maths_t::E_ContinuousData,
                 maths::CAvailableModeDistributions::NORMAL,
                 maths_t::E_ClustersFractionWeight,
                 0.0,                      // No decay
                 CLUSTER_MINIMUM_FRACTION, // We're only interested in clusters which
                                           // comprise at least 0.5% of the data.
                 CLUSTER_MINIMUM_COUNT)    // We need a few points to get a reasonable
                                           // variance estimate.
{
}

CNumericDataSummaryStatistics::CNumericDataSummaryStatistics(const CDataSummaryStatistics& other, bool integer)
    : CDataSummaryStatistics(other),
      m_NonNumericCount(0),
      m_QuantileSketch(maths::CQuantileSketch::E_Linear, QS_SIZE),
      m_Clusters(integer ? maths_t::E_IntegerData : maths_t::E_ContinuousData,
                 maths::CAvailableModeDistributions::NORMAL,
                 maths_t::E_ClustersFractionWeight,
                 0.0,                      // No decay
                 CLUSTER_MINIMUM_FRACTION, // We're only interested in clusters which
                                           // comprise at least 0.5% of the data.
                 CLUSTER_MINIMUM_COUNT)    // Need a few points to get a reasonable
                                           // variance estimate.
{
}

void CNumericDataSummaryStatistics::add(core_t::TTime time, const std::string& example) {
    std::string trimmed = example;
    core::CStringUtils::trimWhitespace(trimmed);

    this->CDataSummaryStatistics::add(time);

    double value;
    if (!core::CStringUtils::stringToTypeSilent(trimmed, value)) {
        ++m_NonNumericCount;
        return;
    }

    m_QuantileSketch.add(value);
    m_Clusters.add(value);
}

double CNumericDataSummaryStatistics::minimum(void) const {
    double result;
    m_QuantileSketch.minimum(result);
    return result;
}

double CNumericDataSummaryStatistics::median(void) const {
    double result;
    m_QuantileSketch.quantile(50.0, result);
    return result;
}

double CNumericDataSummaryStatistics::maximum(void) const {
    double result;
    m_QuantileSketch.maximum(result);
    return result;
}

bool CNumericDataSummaryStatistics::densityChart(TDoubleDoublePrVec& result) const {
    result.clear();

    if (m_Clusters.clusters().empty()) {
        return true;
    }

    typedef std::vector<double> TDoubleVec;
    typedef std::vector<boost::math::normal_distribution<>> TNormalVec;
    typedef maths::CMixtureDistribution<boost::math::normal_distribution<>> TGMM;

    const maths::CXMeansOnline1d::TClusterVec& clusters = m_Clusters.clusters();
    std::size_t n = clusters.size();

    try {
        TDoubleVec weights;
        TNormalVec modes;
        weights.reserve(n);
        modes.reserve(n);
        for (std::size_t i = 0u; i < n; ++i) {
            LOG_TRACE("weight = " << clusters[i].count() << ", mean = " << clusters[i].centre() << ", sd = " << clusters[i].spread());
            weights.push_back(clusters[i].count());
            modes.push_back(boost::math::normal_distribution<>(clusters[i].centre(), clusters[i].spread()));
        }

        TGMM gmm(weights, modes);

        static const double QUANTILES[] = {0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 0.999};

        TDoubleVec pillars;
        pillars.reserve(boost::size(QUANTILES));
        for (std::size_t i = 0u; i < boost::size(QUANTILES); ++i) {
            pillars.push_back(maths::quantile(gmm, QUANTILES[i]));
        }
        LOG_TRACE("pillars = " << core::CContainerPrinter::print(pillars));

        result.reserve(10 * boost::size(QUANTILES));
        for (std::size_t i = 1u; i < pillars.size(); ++i) {
            double x = pillars[i - 1];
            double b = pillars[i];
            double dx = (b - x) / 10.0;
            for (std::size_t j = 0u; j < 10; ++j, x += dx) {
                result.push_back(std::make_pair(x, maths::pdf(gmm, x)));
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to compute density chart: " << e.what());
        return false;
    }
    return true;
}
}
}
