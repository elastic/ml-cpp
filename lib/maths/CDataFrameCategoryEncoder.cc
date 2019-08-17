/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameCategoryEncoder.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CTriple.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>

#include <algorithm>
#include <numeric>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;
using TSizeUSet = boost::unordered_set<std::size_t>;

enum EEncoding {
    E_OneHot = 0,
    E_Frequency,
    E_TargetMean,
    E_CandidateEncodings
};

const std::size_t CATEGORY_FOR_METRICS{std::numeric_limits<std::size_t>::max()};
const std::size_t CATEGORY_FOR_FREQUANCY_ENCODING{CATEGORY_FOR_METRICS - 1};
const std::size_t CATEGORY_FOR_TARGET_MEAN_ENCODING{CATEGORY_FOR_FREQUANCY_ENCODING - 1};
const std::size_t CATEGORY_FOR_DEPENDENT_VARIABLE{CATEGORY_FOR_TARGET_MEAN_ENCODING - 1};

bool isMetric(std::size_t category) {
    return category == CATEGORY_FOR_METRICS;
}
bool isFrequency(std::size_t category) {
    return category == CATEGORY_FOR_FREQUANCY_ENCODING;
}
bool isTargetMean(std::size_t category) {
    return category == CATEGORY_FOR_TARGET_MEAN_ENCODING;
}
bool isCategory(std::size_t category) {
    return (isMetric(category) || isFrequency(category) || isTargetMean(category)) == false;
}
std::string print(std::size_t category) {
    if (isMetric(category)) {
        return "metric";
    }
    if (isFrequency(category)) {
        return "frequency";
    }
    if (isTargetMean(category)) {
        return "target mean";
    }
    return std::to_string(category);
}

//! \brief Maintains the state for a single feature in a greedy search for the
//! minimum redundancy maximum relevance feature selection.
class CFeatureRelevanceMinusRedundancy {
public:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

public:
    CFeatureRelevanceMinusRedundancy(std::size_t feature, std::size_t category, double micWithDependentVariable)
        : m_Feature{feature}, m_Category{category}, m_MicWithDependentVariable{
                                                        micWithDependentVariable} {}

    bool isMetric() const { return maths::isMetric(m_Category); }
    bool isFrequency() const { return maths::isFrequency(m_Category); }
    bool isTargetMean() const { return maths::isTargetMean(m_Category); }
    bool isCategory() const { return maths::isCategory(m_Category); }
    std::size_t feature() const { return m_Feature; }
    std::size_t category() const { return m_Category; }

    double micWithDependentVariable() const {
        return m_MicWithDependentVariable;
    }

    double relevanceMinusRedundancy(double redundancyWeight) const {
        return m_MicWithDependentVariable -
               redundancyWeight * this->micWithSelectedVariables();
    }

    std::unique_ptr<CDataFrameUtils::CColumnValue>
    columnValue(const TSizeUSet& rareCategories,
                const TDoubleVec& frequencies,
                const TDoubleVec& targetMeanValues) const {
        if (this->isMetric()) {
            return std::make_unique<CDataFrameUtils::CMetricColumnValue>(m_Feature);
        }
        if (this->isFrequency()) {
            return std::make_unique<CDataFrameUtils::CFrequencyCategoricalColumnValue>(
                m_Feature, frequencies);
        }
        if (this->isTargetMean()) {
            return std::make_unique<CDataFrameUtils::CTargetMeanCategoricalColumnValue>(
                m_Feature, rareCategories, targetMeanValues);
        }
        return std::make_unique<CDataFrameUtils::COneHotCategoricalColumnValue>(
            m_Feature, m_Category);
    }

    void update(const TSizeDoublePrVecVec& mics) {
        auto i = std::find_if(
            mics[m_Feature].begin(), mics[m_Feature].end(),
            [this](const TSizeDoublePr& mic) { return mic.first == m_Category; });
        if (i != mics[m_Feature].end()) {
            m_MicWithSelectedVariables.add(i->second);
        }
    }

private:
    double micWithSelectedVariables() const {
        return CBasicStatistics::mean(m_MicWithSelectedVariables);
    }

private:
    std::size_t m_Feature = 0;
    std::size_t m_Category = 0;
    double m_MicWithDependentVariable = 0.0;
    TMeanAccumulator m_MicWithSelectedVariables;
};

//! \brief Manages a greedy search for the minimum redundancy maximum relevancy
//! feature set.
//!
//! DESCRIPTION:\n
//! Implements a greedy search to approximately solve the optimization problem
//!
//! \f$arg\max_S \frac{1}{|S|} \sum_{f\in S}{MIC(f,t)} - \frac{1}{|S|^2} \sum_{f\in S, g\in S} {MIC(f,g)}\f$
//!
//! This trades redundancy of information in the feature set as a whole and
//! relevance of each individual feature when deciding which to include. We
//! extend the basic measure by including a non-negative redundancy weight
//! which controls the priority of minimizing redundancy vs maximizing relevancy.
//! For more information see
//! https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_(mRMR)_feature_selection
//! and references therein.
class CMinRedundancyMaxRelevancyGreedySearch {
public:
    CMinRedundancyMaxRelevancyGreedySearch(double redundancyWeight,
                                           const TSizeDoublePrVecVec& mics)
        : m_RedundancyWeight{redundancyWeight} {
        for (std::size_t i = 0; i < mics.size(); ++i) {
            for (std::size_t j = 0; j < mics[i].size(); ++j) {
                std::size_t category;
                double mic;
                std::tie(category, mic) = mics[i][j];
                if (mic > 0.0) {
                    m_Features.emplace_back(i, category, mic);
                }
            }
        }
    }

    CFeatureRelevanceMinusRedundancy selectNext() {
        auto selected = std::max_element(
            m_Features.begin(), m_Features.end(),
            [this](const CFeatureRelevanceMinusRedundancy& lhs,
                   const CFeatureRelevanceMinusRedundancy& rhs) {
                return lhs.relevanceMinusRedundancy(m_RedundancyWeight) <
                       rhs.relevanceMinusRedundancy(m_RedundancyWeight);
            });
        CFeatureRelevanceMinusRedundancy result{*selected};
        m_Features.erase(selected);
        return result;
    }

    void update(const TSizeDoublePrVecVec& mics) {
        for (auto& feature : m_Features) {
            feature.update(mics);
        }
    }

private:
    using TFeatureRelevanceMinusRedundancyList = std::list<CFeatureRelevanceMinusRedundancy>;

private:
    double m_RedundancyWeight;
    TFeatureRelevanceMinusRedundancyList m_Features;
};

const std::string MINIMUM_ROWS_PER_FEATURE_TAG{"minimum_rows_per_feature"};
const std::string MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE_TAG{"minimum_frequency_to_one_hot_encode"};
const std::string REDUNDANCY_WEIGHT_TAG{"redundancy_weight"};
const std::string COLUMN_IS_CATEGORICAL_TAG{"is_categorical"};
const std::string COLUMN_USES_FREQUENCY_ENCODING_TAG{"uses_frequency_encoding"};
const std::string ONE_HOT_ENCODED_CATEGORIES_TAG{"one_hot_encoded_categories"};
const std::string RARE_CATEGORIES_TAG{"rare_categories"};
const std::string CATEGORY_FREQUENCIES_TAG{"category_frequencies"};
const std::string TARGET_MEAN_VALUES_TAG{"target_mean_values"};
const std::string FEATURE_VECTOR_MICS_TAG{"feature_vector_mics"};
const std::string FEATURE_VECTOR_COLUMN_MAP_TAG{"feature_vector_column_map"};
const std::string FEATURE_VECTOR_ENCODING_MAP_TAG{"feature_vector_encoding_map"};
}

CEncodedDataFrameRowRef::CEncodedDataFrameRowRef(TRowRef row, const CDataFrameCategoryEncoder& encoder)
    : m_Row{std::move(row)}, m_Encoder{&encoder} {
}

CFloatStorage CEncodedDataFrameRowRef::operator[](std::size_t i) const {

    std::size_t feature{m_Encoder->column(i)};

    CFloatStorage value{m_Row[feature]};

    if (m_Encoder->columnIsCategorical(feature) == false) {
        return value;
    }

    std::size_t encoding{m_Encoder->encoding(i)};
    std::size_t category{static_cast<std::size_t>(value)};

    std::size_t numberOneHotEncodedCategories{
        m_Encoder->numberOneHotEncodedCategories(feature)};

    if (encoding < numberOneHotEncodedCategories) {
        return m_Encoder->isHot(encoding, feature, category) ? 1.0 : 0.0;
    }

    if (encoding == numberOneHotEncodedCategories &&
        m_Encoder->usesFrequencyEncoding(feature)) {
        return m_Encoder->frequency(feature, category);
    }

    return m_Encoder->targetMeanValue(feature, category);
}

std::size_t CEncodedDataFrameRowRef::index() const {
    return m_Row.index();
}

std::size_t CEncodedDataFrameRowRef::numberColumns() const {
    return m_Encoder->numberFeatures();
}

const CEncodedDataFrameRowRef::TRowRef& CEncodedDataFrameRowRef::unencodedRow() const {
    return m_Row;
}

CDataFrameCategoryEncoder::CDataFrameCategoryEncoder(std::size_t numberThreads,
                                                     const core::CDataFrame& frame,
                                                     const core::CPackedBitVector& rowMask,
                                                     const TSizeVec& columnMask,
                                                     std::size_t targetColumn,
                                                     std::size_t minimumRowsPerFeature,
                                                     double minimumFrequencyToOneHotEncode,
                                                     double redundancyWeight)
    : m_MinimumRowsPerFeature{minimumRowsPerFeature},
      m_MinimumFrequencyToOneHotEncode{minimumFrequencyToOneHotEncode}, m_RedundancyWeight{redundancyWeight},
      m_ColumnIsCategorical(frame.columnIsCategorical()) {

    TSizeVec metricColumnMask(columnMask);
    metricColumnMask.erase(
        std::remove_if(metricColumnMask.begin(), metricColumnMask.end(),
                       [targetColumn, this](std::size_t i) {
                           return i == targetColumn || m_ColumnIsCategorical[i];
                       }),
        metricColumnMask.end());
    LOG_TRACE(<< "metric column mask = " << core::CContainerPrinter::print(metricColumnMask));

    TSizeVec categoricalColumnMask(columnMask);
    categoricalColumnMask.erase(
        std::remove_if(categoricalColumnMask.begin(), categoricalColumnMask.end(),
                       [targetColumn, this](std::size_t i) {
                           return i == targetColumn || m_ColumnIsCategorical[i] == false;
                       }),
        categoricalColumnMask.end());
    LOG_TRACE(<< "categorical column mask = "
              << core::CContainerPrinter::print(categoricalColumnMask));

    // The top-level strategy is as follows:
    //
    // We one-hot encode the frequent categories with the highest non-zero MICe up
    // to the permitted overall feature count.
    // We mean value encode the remaining features where we have a representative
    // sample size.
    // We frequency encode all rare categories of a categorical feature.

    this->setupFrequencyEncoding(numberThreads, frame, rowMask, categoricalColumnMask);

    this->setupTargetMeanValueEncoding(numberThreads, frame, rowMask,
                                       categoricalColumnMask, targetColumn);

    this->finishEncoding(
        targetColumn, this->selectFeatures(numberThreads, frame, rowMask, metricColumnMask,
                                           categoricalColumnMask, targetColumn));
}

CDataFrameCategoryEncoder::CDataFrameCategoryEncoder(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel(std::bind(&CDataFrameCategoryEncoder::acceptRestoreTraverser,
                                             this, std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore category encoder"};
    }
}

CEncodedDataFrameRowRef CDataFrameCategoryEncoder::encode(TRowRef row) const {
    return {std::move(row), *this};
}

bool CDataFrameCategoryEncoder::columnIsCategorical(std::size_t feature) const {
    return m_ColumnIsCategorical[feature];
}

const CDataFrameCategoryEncoder::TDoubleVec& CDataFrameCategoryEncoder::featureMics() const {
    return m_FeatureVectorMics;
}

std::size_t CDataFrameCategoryEncoder::numberFeatures() const {
    return m_FeatureVectorMics.size();
}

std::size_t CDataFrameCategoryEncoder::encoding(std::size_t index) const {
    return m_FeatureVectorEncodingMap[index];
}

std::size_t CDataFrameCategoryEncoder::column(std::size_t index) const {
    return m_FeatureVectorColumnMap[index];
}

bool CDataFrameCategoryEncoder::isBinary(std::size_t index) const {
    std::size_t encoding{this->encoding(index)};
    std::size_t feature{this->column(index)};
    return encoding < m_OneHotEncodedCategories[feature].size();
}

std::size_t CDataFrameCategoryEncoder::numberOneHotEncodedCategories(std::size_t feature) const {
    return m_OneHotEncodedCategories[feature].size();
}

bool CDataFrameCategoryEncoder::isHot(std::size_t encoding,
                                      std::size_t feature,
                                      std::size_t category) const {

    // The most important categories are one-hot encode. In the encoded row the
    // layout of the encoding dimensions, for each categorical feature, is as
    // follows:
    //   (...| one-hot | mean target | frequency |...)
    //
    // The ones are in the order the categories appear in m_OneHotEncodedCategories.
    // For example, if m_OneHotEncodedCategories[feature] = (2, 5, 7) for any other
    // category the encoded row will contain (...| 0 0 0 | ...). For 2, 5 and 7 it
    // will contain (...| 1 0 0 |...), (...| 0 1 0 |...) and (...| 0 0 1 |...),
    // respectively.
    //
    // In the following we therefore 1) check to see if the category is being
    // one-hot encoded, 2) check if the encoding of the dimension, i.e. its offset
    // relative to the start of the encoding dimensions for the feature, is equal
    // to the position of the one for the category.

    auto one = std::lower_bound(m_OneHotEncodedCategories[feature].begin(),
                                m_OneHotEncodedCategories[feature].end(), category);
    return one != m_OneHotEncodedCategories[feature].end() && category == *one &&
           static_cast<std::ptrdiff_t>(encoding) ==
               one - m_OneHotEncodedCategories[feature].begin();
}

bool CDataFrameCategoryEncoder::usesFrequencyEncoding(std::size_t feature) const {
    return m_ColumnUsesFrequencyEncoding[feature];
}

double CDataFrameCategoryEncoder::frequency(std::size_t feature, std::size_t category) const {
    return m_CategoryFrequencies[feature][category];
}

bool CDataFrameCategoryEncoder::isRareCategory(std::size_t feature, std::size_t category) const {
    return m_RareCategories[feature].find(category) != m_RareCategories[feature].end();
}

double CDataFrameCategoryEncoder::targetMeanValue(std::size_t feature,
                                                  std::size_t category) const {
    // TODO combine rare categories and use one mapping for collections.
    return this->isRareCategory(feature, category)
               ? 0.0
               : m_TargetMeanValues[feature][category];
}

std::uint64_t CDataFrameCategoryEncoder::checksum(std::uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MinimumRowsPerFeature);
    seed = CChecksum::calculate(seed, m_MinimumFrequencyToOneHotEncode);
    seed = CChecksum::calculate(seed, m_RedundancyWeight);
    seed = CChecksum::calculate(seed, m_ColumnIsCategorical);
    seed = CChecksum::calculate(seed, m_ColumnUsesFrequencyEncoding);
    seed = CChecksum::calculate(seed, m_OneHotEncodedCategories);
    seed = CChecksum::calculate(seed, m_RareCategories);
    seed = CChecksum::calculate(seed, m_CategoryFrequencies);
    seed = CChecksum::calculate(seed, m_TargetMeanValues);
    seed = CChecksum::calculate(seed, m_FeatureVectorMics);
    seed = CChecksum::calculate(seed, m_FeatureVectorColumnMap);
    return CChecksum::calculate(seed, m_FeatureVectorEncodingMap);
}

void CDataFrameCategoryEncoder::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(MINIMUM_ROWS_PER_FEATURE_TAG, m_MinimumRowsPerFeature);
    inserter.insertValue(MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE_TAG, m_MinimumFrequencyToOneHotEncode,
                         core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(REDUNDANCY_WEIGHT_TAG, m_RedundancyWeight,
                         core::CIEEE754::E_DoublePrecision);
    core::CPersistUtils::persist(COLUMN_IS_CATEGORICAL_TAG, m_ColumnIsCategorical, inserter);
    core::CPersistUtils::persist(COLUMN_USES_FREQUENCY_ENCODING_TAG,
                                 m_ColumnUsesFrequencyEncoding, inserter);
    core::CPersistUtils::persist(ONE_HOT_ENCODED_CATEGORIES_TAG,
                                 m_OneHotEncodedCategories, inserter);
    core::CPersistUtils::persist(RARE_CATEGORIES_TAG, m_RareCategories, inserter);
    core::CPersistUtils::persist(CATEGORY_FREQUENCIES_TAG, m_CategoryFrequencies, inserter);
    core::CPersistUtils::persist(TARGET_MEAN_VALUES_TAG, m_TargetMeanValues, inserter);
    core::CPersistUtils::persist(FEATURE_VECTOR_MICS_TAG, m_FeatureVectorMics, inserter);
    core::CPersistUtils::persist(FEATURE_VECTOR_COLUMN_MAP_TAG,
                                 m_FeatureVectorColumnMap, inserter);
    core::CPersistUtils::persist(FEATURE_VECTOR_ENCODING_MAP_TAG,
                                 m_FeatureVectorEncodingMap, inserter);
}

bool CDataFrameCategoryEncoder::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(MINIMUM_ROWS_PER_FEATURE_TAG, m_MinimumRowsPerFeature)
        RESTORE_BUILT_IN(MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE_TAG, m_MinimumFrequencyToOneHotEncode)
        RESTORE_BUILT_IN(REDUNDANCY_WEIGHT_TAG, m_RedundancyWeight)
        RESTORE(COLUMN_IS_CATEGORICAL_TAG,
                core::CPersistUtils::restore(COLUMN_IS_CATEGORICAL_TAG,
                                             m_ColumnIsCategorical, traverser))
        RESTORE(COLUMN_USES_FREQUENCY_ENCODING_TAG,
                core::CPersistUtils::restore(COLUMN_USES_FREQUENCY_ENCODING_TAG,
                                             m_ColumnUsesFrequencyEncoding, traverser))
        RESTORE(ONE_HOT_ENCODED_CATEGORIES_TAG,
                core::CPersistUtils::restore(ONE_HOT_ENCODED_CATEGORIES_TAG,
                                             m_OneHotEncodedCategories, traverser))
        RESTORE(RARE_CATEGORIES_TAG,
                core::CPersistUtils::restore(RARE_CATEGORIES_TAG, m_RareCategories, traverser))
        RESTORE(CATEGORY_FREQUENCIES_TAG,
                core::CPersistUtils::restore(CATEGORY_FREQUENCIES_TAG,
                                             m_CategoryFrequencies, traverser))
        RESTORE(TARGET_MEAN_VALUES_TAG,
                core::CPersistUtils::restore(TARGET_MEAN_VALUES_TAG, m_TargetMeanValues, traverser))
        RESTORE(FEATURE_VECTOR_MICS_TAG,
                core::CPersistUtils::restore(FEATURE_VECTOR_MICS_TAG,
                                             m_FeatureVectorMics, traverser))
        RESTORE(FEATURE_VECTOR_COLUMN_MAP_TAG,
                core::CPersistUtils::restore(FEATURE_VECTOR_COLUMN_MAP_TAG,
                                             m_FeatureVectorColumnMap, traverser))
        RESTORE(FEATURE_VECTOR_ENCODING_MAP_TAG,
                core::CPersistUtils::restore(FEATURE_VECTOR_ENCODING_MAP_TAG,
                                             m_FeatureVectorEncodingMap, traverser))
    } while (traverser.next());
    return true;
}

CDataFrameCategoryEncoder::TSizeDoublePrVecVec
CDataFrameCategoryEncoder::mics(std::size_t numberThreads,
                                const core::CDataFrame& frame,
                                const CDataFrameUtils::CColumnValue& target,
                                const core::CPackedBitVector& rowMask,
                                const TSizeVec& metricColumnMask,
                                const TSizeVec& categoricalColumnMask) const {

    CDataFrameUtils::TEncoderFactoryVec encoderFactories(E_CandidateEncodings);
    encoderFactories[E_OneHot] = std::make_pair(
        [](std::size_t, std::size_t sampleColumn, std::size_t category) {
            return std::make_unique<CDataFrameUtils::COneHotCategoricalColumnValue>(
                sampleColumn, category);
        },
        m_MinimumFrequencyToOneHotEncode);
    encoderFactories[E_TargetMean] = std::make_pair(
        [this](std::size_t column, std::size_t sampleColumn, std::size_t) {
            return std::make_unique<CDataFrameUtils::CTargetMeanCategoricalColumnValue>(
                sampleColumn, m_RareCategories[column], m_TargetMeanValues[column]);
        },
        0.0);
    encoderFactories[E_Frequency] = std::make_pair(
        [this](std::size_t column, std::size_t sampleColumn, std::size_t) {
            return std::make_unique<CDataFrameUtils::CFrequencyCategoricalColumnValue>(
                sampleColumn, m_CategoryFrequencies[column]);
        },
        0.0);

    auto metricMics = CDataFrameUtils::metricMicWithColumn(target, frame, rowMask,
                                                           metricColumnMask);
    auto categoricalMics = CDataFrameUtils::categoricalMicWithColumn(
        target, numberThreads, frame, rowMask, categoricalColumnMask, encoderFactories);

    TSizeDoublePrVecVec mics(std::move(categoricalMics[E_OneHot]));
    for (std::size_t i = 0; i < categoricalMics[E_TargetMean].size(); ++i) {
        if (categoricalMics[E_TargetMean][i].size() > 0) {
            mics[i].emplace_back(CATEGORY_FOR_TARGET_MEAN_ENCODING,
                                 categoricalMics[E_TargetMean][i][0].second);
        }
    }
    for (std::size_t i = 0; i < categoricalMics[E_Frequency].size(); ++i) {
        if (categoricalMics[E_Frequency][i].size() > 0) {
            mics[i].emplace_back(CATEGORY_FOR_FREQUANCY_ENCODING,
                                 categoricalMics[E_Frequency][i][0].second);
        }
    }
    for (std::size_t i = 0; i < metricMics.size(); ++i) {
        if (metricMics[i] > 0.0) {
            mics[i].emplace_back(CATEGORY_FOR_METRICS, metricMics[i]);
        }
    }
    LOG_TRACE(<< "MICe = " << core::CContainerPrinter::print(mics));

    return mics;
}

void CDataFrameCategoryEncoder::setupFrequencyEncoding(std::size_t numberThreads,
                                                       const core::CDataFrame& frame,
                                                       const core::CPackedBitVector& rowMask,
                                                       const TSizeVec& categoricalColumnMask) {

    m_CategoryFrequencies = CDataFrameUtils::categoryFrequencies(
        numberThreads, frame, rowMask, categoricalColumnMask);
    LOG_TRACE(<< "category frequencies = "
              << core::CContainerPrinter::print(m_CategoryFrequencies));

    m_RareCategories.resize(frame.numberColumns());
    for (std::size_t i = 0; i < m_CategoryFrequencies.size(); ++i) {
        for (std::size_t j = 0; j < m_CategoryFrequencies[i].size(); ++j) {
            std::size_t count{static_cast<std::size_t>(
                m_CategoryFrequencies[i][j] * static_cast<double>(frame.numberRows()) + 0.5)};
            if (count < m_MinimumRowsPerFeature) {
                m_RareCategories[i].insert(j);
            }
        }
    }
    LOG_TRACE(<< "rare categories = " << core::CContainerPrinter::print(m_RareCategories));
}

void CDataFrameCategoryEncoder::setupTargetMeanValueEncoding(std::size_t numberThreads,
                                                             const core::CDataFrame& frame,
                                                             const core::CPackedBitVector& rowMask,
                                                             const TSizeVec& categoricalColumnMask,
                                                             std::size_t targetColumn) {

    m_TargetMeanValues = CDataFrameUtils::meanValueOfTargetForCategories(
        CDataFrameUtils::CMetricColumnValue{targetColumn}, numberThreads, frame,
        rowMask, categoricalColumnMask);
    LOG_TRACE(<< "target mean values = "
              << core::CContainerPrinter::print(m_TargetMeanValues));
}

CDataFrameCategoryEncoder::TSizeSizePrDoubleMap
CDataFrameCategoryEncoder::selectAllFeatures(const TSizeDoublePrVecVec& mics) {

    TSizeSizePrDoubleMap selectedFeatureMics;

    for (std::size_t feature = 0; feature < mics.size(); ++feature) {
        for (std::size_t i = 0; i < mics[feature].size(); ++i) {
            std::size_t category;
            double mic;
            std::tie(category, mic) = mics[feature][i];
            if (mic == 0.0) {
                continue;
            }
            LOG_TRACE(<< "Selected feature = " << feature << ", category = "
                      << print(category) << ", mic with target = " << mic);

            selectedFeatureMics[{feature, category}] = mic;

            if (isCategory(category)) {
                m_OneHotEncodedCategories[feature].push_back(category);
            } else if (isFrequency(category)) {
                m_ColumnUsesFrequencyEncoding[feature] = true;
            } else if (isTargetMean(category)) {
                // Nothing to do
            }
        }
    }

    LOG_TRACE(<< "one-hot encoded = "
              << core::CContainerPrinter::print(m_OneHotEncodedCategories));

    return selectedFeatureMics;
}

CDataFrameCategoryEncoder::TSizeSizePrDoubleMap
CDataFrameCategoryEncoder::selectFeatures(std::size_t numberThreads,
                                          const core::CDataFrame& frame,
                                          const core::CPackedBitVector& rowMask,
                                          TSizeVec metricColumnMask,
                                          TSizeVec categoricalColumnMask,
                                          std::size_t targetColumn) {

    // We want to choose features which provide independent information about the
    // target variable. Ideally, we'd recompute MICe w.r.t. target - f(x) with x
    // the features selected so far. This would be very computationally expensive
    // since it requires training a model f(.) on a subset of the features after
    // each decision. Instead, we use the average MICe between the unselected and
    // selected features as a useful proxy. This is essentially the mRMR approach
    // of Peng et al. albeit with MICe rather than mutual information. Except, it
    // also supports a redundancy weight, which should be non-negative and is used
    // to control the relative weight of MICe with the target vs the selected
    // variables. A value of zero means exclusively maximise MICe with the target
    // and as redundancy weight -> infinity means exclusively minimise MICe with
    // the selected variables.

    TSizeDoublePrVecVec mics(this->mics(
        numberThreads, frame, CDataFrameUtils::CMetricColumnValue{targetColumn},
        rowMask, metricColumnMask, categoricalColumnMask));
    LOG_TRACE(<< "features MICe = " << core::CContainerPrinter::print(mics));

    std::size_t numberAvailableFeatures{this->numberAvailableFeatures(mics)};
    std::size_t maximumNumberFeatures{
        (frame.numberRows() + m_MinimumRowsPerFeature / 2) / m_MinimumRowsPerFeature};
    LOG_TRACE(<< "number possible features = " << numberAvailableFeatures
              << " maximum permitted features = " << maximumNumberFeatures);

    m_ColumnUsesFrequencyEncoding.resize(frame.numberColumns(), false);
    m_OneHotEncodedCategories.resize(frame.numberColumns());

    TSizeSizePrDoubleMap selectedFeatureMics;

    if (maximumNumberFeatures >= numberAvailableFeatures) {

        selectedFeatureMics = this->selectAllFeatures(mics);
    } else {

        CMinRedundancyMaxRelevancyGreedySearch search{m_RedundancyWeight, mics};

        for (std::size_t i = 0; i < maximumNumberFeatures; ++i) {

            CFeatureRelevanceMinusRedundancy selected{search.selectNext()};

            double mic{selected.micWithDependentVariable()};
            std::size_t feature{selected.feature()};
            std::size_t category{selected.category()};
            LOG_TRACE(<< "Selected feature = " << feature << ", category = "
                      << print(category) << ", mic with target = " << mic);

            selectedFeatureMics[{feature, category}] = mic;

            if (selected.isCategory()) {
                m_OneHotEncodedCategories[feature].push_back(category);
            } else if (selected.isFrequency()) {
                m_ColumnUsesFrequencyEncoding[feature] = true;
            } else if (selected.isTargetMean()) {
                // Nothing to do
            } else if (selected.isMetric()) {
                metricColumnMask.erase(std::find(metricColumnMask.begin(),
                                                 metricColumnMask.end(), feature));
            }

            auto columnValue = selected.columnValue(m_RareCategories[feature],
                                                    m_CategoryFrequencies[feature],
                                                    m_TargetMeanValues[feature]);
            mics = this->mics(numberThreads, frame, *columnValue, rowMask,
                              metricColumnMask, categoricalColumnMask);
            search.update(mics);
        }
    }

    for (auto& categories : m_OneHotEncodedCategories) {
        categories.shrink_to_fit();
        std::sort(categories.begin(), categories.end());
    }

    LOG_TRACE(<< "one-hot encoded = "
              << core::CContainerPrinter::print(m_OneHotEncodedCategories));
    LOG_TRACE(<< "selected features MICe = "
              << core::CContainerPrinter::print(selectedFeatureMics));

    return selectedFeatureMics;
}

void CDataFrameCategoryEncoder::finishEncoding(std::size_t targetColumn,
                                               TSizeSizePrDoubleMap selectedFeatureMics) {

    // Fill in a mapping from encoded column indices to raw column indices.

    selectedFeatureMics[{targetColumn, CATEGORY_FOR_DEPENDENT_VARIABLE}] = 0.0;

    m_FeatureVectorMics.reserve(selectedFeatureMics.size());
    m_FeatureVectorColumnMap.reserve(selectedFeatureMics.size());
    m_FeatureVectorEncodingMap.reserve(selectedFeatureMics.size());

    auto i = selectedFeatureMics.begin();
    auto end = selectedFeatureMics.end();
    std::size_t encoding{0};
    for (;;) {
        std::size_t feature{i->first.first};
        double mic{i->second};
        m_FeatureVectorMics.push_back(mic);
        m_FeatureVectorColumnMap.push_back(feature);
        m_FeatureVectorEncodingMap.push_back(encoding);
        if (++i == end) {
            break;
        }
        encoding = i->first.first == feature ? encoding + 1 : 0;
    }

    LOG_TRACE(<< "feature vector MICe = "
              << core::CContainerPrinter::print(m_FeatureVectorMics));
    LOG_TRACE(<< "feature vector index to column map = "
              << core::CContainerPrinter::print(m_FeatureVectorColumnMap));
    LOG_TRACE(<< "feature vector index to encoding map = "
              << core::CContainerPrinter::print(m_FeatureVectorEncodingMap));
}

std::size_t CDataFrameCategoryEncoder::numberAvailableFeatures(const TSizeDoublePrVecVec& mics) const {
    std::size_t count{0};
    for (const auto& featureMics : mics) {
        count += std::count_if(featureMics.begin(), featureMics.end(),
                               [](const auto& mic) { return mic.second > 0.0; });
    }
    return count;
}
}
}
