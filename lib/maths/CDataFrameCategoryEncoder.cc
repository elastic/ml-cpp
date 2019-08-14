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

//! \brief Maintains the state for a single feature in a greedy search for the
//! minimum redundancy maximum relevance feature selection.
class CFeatureRelevanceMinusRedundancy {
public:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

public:
    CFeatureRelevanceMinusRedundancy(std::size_t feature,
                                     std::size_t category,
                                     bool isCategorical,
                                     double micWithDependentVariable)
        : m_Feature{feature}, m_Category{category}, m_IsCategorical{isCategorical},
          m_MicWithDependentVariable{micWithDependentVariable} {}

    bool isCategorical() const { return m_IsCategorical; }
    std::size_t feature() const { return m_Feature; }
    std::size_t category() const { return m_Category; }

    double micWithDependentVariable() const {
        return m_MicWithDependentVariable;
    }

    double relevanceMinusRedundancy(double redundancyWeight) const {
        return m_MicWithDependentVariable -
               redundancyWeight * this->micWithSelectedVariables();
    }

    void update(const TDoubleVec& metricMics, const TSizeDoublePrVecVec& categoricalMics) {
        if (m_IsCategorical) {
            auto i = std::find_if(categoricalMics[m_Feature].begin(),
                                  categoricalMics[m_Feature].end(),
                                  [this](const TSizeDoublePr& categoryMic) {
                                      return categoryMic.first == m_Category;
                                  });
            if (i != categoricalMics[m_Feature].end()) {
                m_MicWithSelectedVariables.add(i->second);
            }
        } else {
            m_MicWithSelectedVariables.add(metricMics[m_Feature]);
        }
    }

private:
    double micWithSelectedVariables() const {
        return CBasicStatistics::mean(m_MicWithSelectedVariables);
    }

private:
    std::size_t m_Feature = 0;
    std::size_t m_Category = 0;
    bool m_IsCategorical;
    double m_MicWithDependentVariable = 0.0;
    TMeanAccumulator m_MicWithSelectedVariables;
};

//! \brief Manages a greedy search for the minimum redundancy maximum relevancy
//! feature set.
//!
//! DESCRIPTION:\n
//! As with Peng et al, we use a greedy search to approximately solve the
//! optimization problem
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
                                           const TDoubleVec& metricMics,
                                           const TSizeDoublePrVecVec& categoricalMics)
        : m_RedundancyWeight{redundancyWeight} {
        for (std::size_t i = 0; i < metricMics.size(); ++i) {
            if (metricMics[i] > 0.0) {
                m_Features.emplace_back(i, 0, false, metricMics[i]);
            }
        }
        for (std::size_t i = 0; i < categoricalMics.size(); ++i) {
            for (std::size_t j = 0; j < categoricalMics[i].size(); ++j) {
                std::size_t category;
                double mic;
                std::tie(category, mic) = categoricalMics[i][j];
                if (mic > 0.0) {
                    m_Features.emplace_back(i, category, true, mic);
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

    void update(const TDoubleVec& metricMics, const TSizeDoublePrVecVec& categoricalMics) {
        for (auto& feature : m_Features) {
            feature.update(metricMics, categoricalMics);
        }
    }

private:
    using TFeatureRelevanceMinusRedundancyList = std::list<CFeatureRelevanceMinusRedundancy>;

private:
    double m_RedundancyWeight;
    TFeatureRelevanceMinusRedundancyList m_Features;
};
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
        return m_Encoder->isOne(encoding, feature, category) ? 1.0 : 0.0;
    }

    if (encoding == numberOneHotEncodedCategories && m_Encoder->hasRareCategories(feature)) {
        return m_Encoder->isRareCategory(feature, category) ? 1.0 : 0.0;
    }

    return m_Encoder->targetMeanValue(feature, category);
}

std::size_t CEncodedDataFrameRowRef::index() const {
    return m_Row.index();
}

std::size_t CEncodedDataFrameRowRef::numberColumns() const {
    return m_Encoder->numberFeatures();
}

CDataFrameCategoryEncoder::CDataFrameCategoryEncoder(std::size_t numberThreads,
                                                     const core::CDataFrame& frame,
                                                     const TSizeVec& columnMask,
                                                     std::size_t targetColumn,
                                                     std::size_t minimumRowsPerFeature,
                                                     double minimumFrequencyToOneHotEncode,
                                                     double redundancyWeight)
    : m_RedundancyWeight{redundancyWeight},
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
    // We use one indicator dimension in the feature vector for encoding all rare
    // categories of a categorical column.

    this->isRareEncode(numberThreads, frame, categoricalColumnMask, minimumRowsPerFeature);

    this->targetMeanValueEncode(numberThreads, frame, categoricalColumnMask, targetColumn);

    this->oneHotEncode(numberThreads, frame, metricColumnMask,
                       categoricalColumnMask, targetColumn,
                       minimumRowsPerFeature, minimumFrequencyToOneHotEncode);

    this->setupEncodingMaps(frame);
}

CEncodedDataFrameRowRef CDataFrameCategoryEncoder::encode(TRowRef row) {
    return {std::move(row), *this};
}

bool CDataFrameCategoryEncoder::columnIsCategorical(std::size_t feature) const {
    return m_ColumnIsCategorical[feature];
}

const CDataFrameCategoryEncoder::TSizeVec&
CDataFrameCategoryEncoder::selectedMetricFeatures() const {
    return m_SelectedMetricFeatures;
}

const CDataFrameCategoryEncoder::TDoubleVec&
CDataFrameCategoryEncoder::selectedMetricFeatureMics() const {
    return m_SelectedMetricFeatureMics;
}

const CDataFrameCategoryEncoder::TSizeVec&
CDataFrameCategoryEncoder::selectedCategoricalFeatures() const {
    return m_SelectedCategoricalFeatures;
}

std::size_t CDataFrameCategoryEncoder::numberFeatures() const {
    return m_FeatureVectorColumnMap.size();
}

std::size_t CDataFrameCategoryEncoder::encoding(std::size_t index) const {
    return m_FeatureVectorEncodingMap[index];
}

std::size_t CDataFrameCategoryEncoder::column(std::size_t index) const {
    return m_FeatureVectorColumnMap[index];
}

std::size_t CDataFrameCategoryEncoder::numberOneHotEncodedCategories(std::size_t feature) const {
    return m_OneHotEncodedCategories[feature].size();
}

bool CDataFrameCategoryEncoder::isOne(std::size_t encoding,
                                      std::size_t feature,
                                      std::size_t category) const {

    // The most important categories are one-hot encode. In the encoded row the
    // layout of the encoding dimensions, for each categorical feature, is as
    // follows:
    //   (...| one-hot | mean target | is rare |...)
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

bool CDataFrameCategoryEncoder::hasRareCategories(std::size_t feature) const {
    return m_RareCategories[feature].size() > 0;
}

bool CDataFrameCategoryEncoder::isRareCategory(std::size_t feature, std::size_t category) const {
    return std::binary_search(m_RareCategories[feature].begin(),
                              m_RareCategories[feature].end(), category);
}

double CDataFrameCategoryEncoder::targetMeanValue(std::size_t feature,
                                                  std::size_t category) const {
    return m_TargetMeanValues[feature][category];
}

std::pair<TDoubleVec, TSizeDoublePrVecVec>
CDataFrameCategoryEncoder::mics(std::size_t numberThreads,
                                const core::CDataFrame& frame,
                                std::size_t feature,
                                std::size_t category,
                                const TSizeVec& metricColumnMask,
                                const TSizeVec& categoricalColumnMask,
                                double minimumFrequencyToOneHotEncode) const {
    if (m_ColumnIsCategorical[feature]) {
        return {CDataFrameUtils::micWithColumn(
                    CDataFrameUtils::COneHotCategoricalColumnValue{feature, category},
                    frame, metricColumnMask),
                CDataFrameUtils::categoryMicWithColumn(
                    CDataFrameUtils::COneHotCategoricalColumnValue{feature, category}, numberThreads,
                    frame, categoricalColumnMask, minimumFrequencyToOneHotEncode)};
    }
    return {CDataFrameUtils::micWithColumn(CDataFrameUtils::CMetricColumnValue{feature},
                                           frame, metricColumnMask),
            CDataFrameUtils::categoryMicWithColumn(
                CDataFrameUtils::CMetricColumnValue{feature}, numberThreads,
                frame, categoricalColumnMask, minimumFrequencyToOneHotEncode)};
}

void CDataFrameCategoryEncoder::isRareEncode(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             const TSizeVec& categoricalColumnMask,
                                             std::size_t minimumRowsPerFeature) {

    TDoubleVecVec categoryFrequencies(CDataFrameUtils::categoryFrequencies(
        numberThreads, frame, categoricalColumnMask));
    LOG_TRACE(<< "category frequencies = "
              << core::CContainerPrinter::print(categoryFrequencies));

    m_RareCategories.resize(frame.numberColumns());
    for (std::size_t i = 0; i < categoryFrequencies.size(); ++i) {
        for (std::size_t j = 0; j < categoryFrequencies[i].size(); ++j) {
            std::size_t count{static_cast<std::size_t>(
                categoryFrequencies[i][j] * static_cast<double>(frame.numberRows()) + 0.5)};
            if (count < minimumRowsPerFeature) {
                m_RareCategories[i].push_back(j);
            }
        }
        m_RareCategories[i].shrink_to_fit();
    }
    LOG_TRACE(<< "rare categories = " << core::CContainerPrinter::print(m_RareCategories));
}

void CDataFrameCategoryEncoder::oneHotEncode(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             TSizeVec metricColumnMask,
                                             TSizeVec categoricalColumnMask,
                                             std::size_t targetColumn,
                                             std::size_t minimumRowsPerFeature,
                                             double minimumFrequencyToOneHotEncode) {

    // We want to choose features which provide independent information about the
    // target variable. Ideally, we'd recompute MICe w.r.t. target - f(x) with x
    // the features selected so far. This would be very computationally expensive
    // since it requires training a model f(.) on a subset of the features after
    // each decision. Instead, we use the average MICe between the unselected and
    // selected features as a useful proxy. This is essentially the mRMR approach
    // of Peng et al. albeit with MICe rather than MI. We also support a parameter
    // redundancy weight, which should be non-negative, controls the relative weight
    // of MICe with the target vs the selected variables. A value of zero means
    // exclusively maximise MICe with the target and as redundancy weight -> infinity
    // means exclusively minimise MICe with the selected variables.

    TDoubleVec metricMics;
    TSizeDoublePrVecVec categoricalMics;
    std::tie(metricMics, categoricalMics) =
        this->mics(numberThreads, frame, targetColumn, 0, metricColumnMask,
                   categoricalColumnMask, minimumFrequencyToOneHotEncode);
    LOG_TRACE(<< "metric MICe = " << core::CContainerPrinter::print(metricMics));
    LOG_TRACE(<< "categorical MICe = " << core::CContainerPrinter::print(categoricalMics));

    std::size_t numberAvailableFeatures{
        this->numberAvailableFeatures(metricMics, categoricalMics)};
    std::size_t maximumNumberFeatures{
        (frame.numberRows() + minimumRowsPerFeature / 2) / minimumRowsPerFeature};
    LOG_TRACE(<< "number possible features = " << numberAvailableFeatures
              << " maximum permitted features = " << maximumNumberFeatures);

    m_OneHotEncodedCategories.resize(frame.numberColumns());

    if (maximumNumberFeatures >= numberAvailableFeatures) {
        this->oneHotEncodeAll(metricMics, categoricalMics);

    } else {
        CMinRedundancyMaxRelevancyGreedySearch search{m_RedundancyWeight,
                                                      metricMics, categoricalMics};

        for (std::size_t i = 0; i < maximumNumberFeatures; ++i) {

            CFeatureRelevanceMinusRedundancy selected{search.selectNext()};

            double mic{selected.micWithDependentVariable()};
            std::size_t feature{selected.feature()};
            std::size_t category{selected.category()};

            if (selected.isCategorical()) {
                m_SelectedCategoricalFeatures.push_back(feature);
                m_OneHotEncodedCategories[feature].push_back(category);

                if (m_OneHotEncodedCategories[feature].size() == 1) {
                    i += 1 + (this->hasRareCategories(feature) ? 1 : 0);
                }
                if (m_OneHotEncodedCategories[feature].size() ==
                    categoricalMics[feature].size()) {
                    categoricalColumnMask.erase(std::find(
                        categoricalColumnMask.begin(), categoricalColumnMask.end(), feature));
                }
            } else {
                m_SelectedMetricFeatures.push_back(feature);
                m_SelectedMetricFeatureMics.push_back(mic);
                metricColumnMask.erase(std::find(metricColumnMask.begin(),
                                                 metricColumnMask.end(), feature));
            }

            std::tie(metricMics, categoricalMics) =
                this->mics(numberThreads, frame, feature, category, metricColumnMask,
                           categoricalColumnMask, minimumFrequencyToOneHotEncode);
            search.update(metricMics, categoricalMics);
        }
    }

    COrderings::simultaneousSort(m_SelectedMetricFeatures, m_SelectedMetricFeatureMics);
    std::sort(m_SelectedCategoricalFeatures.begin(),
              m_SelectedCategoricalFeatures.end());
    m_SelectedCategoricalFeatures.erase(
        std::unique(m_SelectedCategoricalFeatures.begin(),
                    m_SelectedCategoricalFeatures.end()),
        m_SelectedCategoricalFeatures.end());
    m_SelectedCategoricalFeatures.shrink_to_fit();

    for (auto& categories : m_OneHotEncodedCategories) {
        categories.shrink_to_fit();
        std::sort(categories.begin(), categories.end());
    }

    LOG_TRACE(<< "selected metrics = "
              << core::CContainerPrinter::print(m_SelectedMetricFeatures));
    LOG_TRACE(<< "one-hot encoded = "
              << core::CContainerPrinter::print(m_OneHotEncodedCategories));
}

void CDataFrameCategoryEncoder::oneHotEncodeAll(const TDoubleVec& metricMics,
                                                const TSizeDoublePrVecVec& categoricalMics) {

    for (std::size_t i = 0; i < metricMics.size(); ++i) {
        if (metricMics[i] > 0.0) {
            m_SelectedMetricFeatures.push_back(i);
            m_SelectedMetricFeatureMics.push_back(metricMics[i]);
        }
    }

    for (std::size_t i = 0; i < categoricalMics.size(); ++i) {
        for (std::size_t j = 0; j < categoricalMics[i].size(); ++j) {
            std::size_t category;
            double mic;
            std::tie(category, mic) = categoricalMics[i][j];
            if (mic > 0.0) {
                m_SelectedCategoricalFeatures.push_back(i);
                m_OneHotEncodedCategories[i].push_back(category);
            }
        }
    }
}

void CDataFrameCategoryEncoder::targetMeanValueEncode(std::size_t numberThreads,
                                                      const core::CDataFrame& frame,
                                                      const TSizeVec& categoricalColumnMask,
                                                      std::size_t targetColumn) {

    m_TargetMeanValues = CDataFrameUtils::meanValueOfTargetForCategories(
        CDataFrameUtils::CMetricColumnValue{targetColumn}, numberThreads, frame,
        categoricalColumnMask);
    LOG_TRACE(<< "target mean values = "
              << core::CContainerPrinter::print(m_TargetMeanValues));
}

void CDataFrameCategoryEncoder::setupEncodingMaps(const core::CDataFrame& frame) {

    // Fill in a mapping from encoded column indices to raw column indices.

    for (std::size_t i = 0; i < frame.numberColumns(); ++i) {

        if (m_ColumnIsCategorical[i]) {
            std::size_t numberOneHot{m_OneHotEncodedCategories[i].size()};
            std::size_t numberRare{m_RareCategories[i].size()};
            std::size_t numberCategories{m_TargetMeanValues[i].size()};

            std::size_t offset{m_FeatureVectorColumnMap.size()};
            std::size_t extra{numberOneHot + (numberRare > 0 ? 1 : 0) +
                              (numberCategories > numberOneHot + numberRare ? 1 : 0)};

            m_FeatureVectorColumnMap.resize(offset + extra, i);
            m_FeatureVectorEncodingMap.resize(offset + extra);
            std::iota(m_FeatureVectorEncodingMap.begin() + offset,
                      m_FeatureVectorEncodingMap.end(), 0);

        } else {
            m_FeatureVectorColumnMap.push_back(i);
            m_FeatureVectorEncodingMap.push_back(0);
        }
    }
    m_FeatureVectorColumnMap.shrink_to_fit();
    m_FeatureVectorEncodingMap.shrink_to_fit();
    LOG_TRACE(<< "feature vector index to column map = "
              << core::CContainerPrinter::print(m_FeatureVectorColumnMap));
    LOG_TRACE(<< "feature vector index to encoding map = "
              << core::CContainerPrinter::print(m_FeatureVectorEncodingMap));
}

std::size_t
CDataFrameCategoryEncoder::numberAvailableFeatures(const TDoubleVec& metricMics,
                                                   const TSizeDoublePrVecVec& categoricalMics) const {
    std::size_t numberFeatures(std::count_if(metricMics.begin(), metricMics.end(),
                                             [](double mic) { return mic > 0.0; }));
    for (std::size_t i = 0; i < categoricalMics.size(); ++i) {
        numberFeatures += std::count_if(categoricalMics[i].begin(),
                                        categoricalMics[i].end(),
                                        [&](auto categoryMic) {
                                            std::size_t category;
                                            double mic;
                                            std::tie(category, mic) = categoryMic;
                                            return mic > 0.0;
                                        }) +
                          (this->hasRareCategories(i) ? 1 : 0);
    }
    return numberFeatures;
}
}
}
