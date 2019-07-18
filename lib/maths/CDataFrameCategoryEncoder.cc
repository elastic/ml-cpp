/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameCategoryEncoder.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CTriple.h>

#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>

#include <algorithm>
#include <numeric>

namespace ml {
namespace maths {

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
                                                     double lambda)
    : m_Lambda{lambda}, m_ColumnIsCategorical(frame.columnIsCategorical()) {

    TSizeVec metricColumnMask(columnMask);
    std::iota(metricColumnMask.begin(), metricColumnMask.end(), 0);
    metricColumnMask.erase(
        std::remove_if(metricColumnMask.begin(), metricColumnMask.end(),
                       [targetColumn, this](std::size_t i) {
                           return i == targetColumn || m_ColumnIsCategorical[i];
                       }),
        metricColumnMask.end());
    LOG_TRACE(<< "metric column mask = " << core::CContainerPrinter::print(metricColumnMask));

    TSizeVec categoricalColumnMask(columnMask);
    std::iota(categoricalColumnMask.begin(), categoricalColumnMask.end(), 0);
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

    this->hotOneEncode(numberThreads, frame, metricColumnMask,
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

bool CDataFrameCategoryEncoder::isOne(std::size_t index,
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
    // will contain (...| 1 0 0 | ...), (...| 0 1 0 | ...) and (...| 0 0 1 | ...),
    // respectively.
    //
    // In the following we therefore 1) check to see if the category is being
    // one-hot encoded, 2) check if the index of the dimension, i.e. its offset
    // relative to the start of the encoding dimensions for the feature, is equal
    // to the position of the one for the category.

    auto one = std::lower_bound(m_OneHotEncodedCategories[feature].begin(),
                                m_OneHotEncodedCategories[feature].end(), category);
    return one != m_OneHotEncodedCategories[feature].end() && category == *one &&
           static_cast<std::ptrdiff_t>(index) ==
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

void CDataFrameCategoryEncoder::hotOneEncode(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             const TSizeVec& metricColumnMask,
                                             const TSizeVec& categoricalColumnMask,
                                             std::size_t targetColumn,
                                             std::size_t minimumRowsPerFeature,
                                             double minimumFrequencyToOneHotEncode) {

    // We expect information carried by distinct features to be less correlated
    // so want to prefer choosing distinct features if the decision is marginal.
    // Ideally we'd recompute MICe w.r.t. target - prediction given the features
    // selected so far. This would be very expensive since it requires training
    // a model on a subset of the features for each decision. Instead, we introduce
    // a single non-negative parameter lambda and the category MICe for each feature
    // is scaled by exp(-lambda x "count of categories one-hot encoded for feature").
    // In the limit lambda is zero this amounts to simply choosing the metrics
    // and categories to one-hot encode that carry the most information about, i.e.
    // have the highest MICe with, the target variable. In the limit lambda is large
    // this amounts to choosing all the metrics and the same count of categories to
    // one-hot encode for each categorical feature. In this case the MICe with the
    // target is maximised independently for each categorical feature.

    using TDoubleSizeSizeTr = core::CTriple<double, std::size_t, std::size_t>;
    using TDoubleSizeSizeTrList = std::list<TDoubleSizeSizeTr>;

    LOG_TRACE(<< "target column = " << targetColumn);

    auto metricMics = CDataFrameUtils::micWithColumn(frame, metricColumnMask, targetColumn);
    auto categoricalMics = CDataFrameUtils::categoryMicWithColumn(
        numberThreads, frame, categoricalColumnMask, targetColumn, minimumFrequencyToOneHotEncode);
    LOG_TRACE(<< "metric MICe = " << core::CContainerPrinter::print(metricMics));
    LOG_TRACE(<< "categorical MICe = " << core::CContainerPrinter::print(categoricalMics));

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

    std::size_t maximumNumberFeatures{
        (frame.numberRows() + minimumRowsPerFeature / 2) / minimumRowsPerFeature};
    LOG_TRACE(<< "number possible features = " << numberFeatures
              << " maximum permitted features = " << maximumNumberFeatures);

    m_OneHotEncodedCategories.resize(frame.numberColumns());

    if (maximumNumberFeatures >= numberFeatures) {
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
                    m_OneHotEncodedCategories[i].push_back(category);
                }
            }
        }

    } else {
        TDoubleSizeSizeTrList mics;
        for (std::size_t i = 0; i < metricMics.size(); ++i) {
            if (metricMics[i] > 0.0) {
                mics.emplace_back(metricMics[i], i, 0);
            }
        }
        for (std::size_t i = 0; i < categoricalMics.size(); ++i) {
            for (std::size_t j = 0; j < categoricalMics[i].size(); ++j) {
                std::size_t category;
                double mic;
                std::tie(category, mic) = categoricalMics[i][j];
                if (mic > 0.0) {
                    mics.emplace_back(mic, i, category);
                }
            }
        }
        LOG_TRACE(<< "MICe = " << core::CContainerPrinter::print(mics));

        TDoubleVec hotOneEncodedCounts(frame.numberColumns(), 0.0);

        for (std::size_t i = 0; i < maximumNumberFeatures; ++i) {

            auto next = std::max_element(
                mics.begin(), mics.end(),
                [&](const TDoubleSizeSizeTr& lhs, const TDoubleSizeSizeTr& rhs) {
                    return std::exp(-m_Lambda * hotOneEncodedCounts[lhs.second]) * lhs.first <
                           std::exp(-m_Lambda * hotOneEncodedCounts[rhs.second]) * rhs.first;
                });

            double mic{next->first};
            std::size_t feature{next->second};
            std::size_t category{next->third};

            if (m_ColumnIsCategorical[feature]) {
                m_SelectedCategoricalFeatures.push_back(feature);
                m_OneHotEncodedCategories[feature].push_back(category);
                hotOneEncodedCounts[feature] += 1.0;
                if (hotOneEncodedCounts[feature] == 1.0) {
                    i += 1 + (this->hasRareCategories(feature) ? 1 : 0);
                }
            } else {
                m_SelectedMetricFeatures.push_back(feature);
                m_SelectedMetricFeatureMics.push_back(mic);
            }

            mics.erase(next);
        }
    }

    for (auto& categories : m_OneHotEncodedCategories) {
        categories.shrink_to_fit();
        std::sort(categories.begin(), categories.end());
    }
    COrderings::simultaneousSort(m_SelectedMetricFeatures, m_SelectedMetricFeatureMics);
    LOG_TRACE(<< "selected metrics = "
              << core::CContainerPrinter::print(m_SelectedMetricFeatures));
    LOG_TRACE(<< "one-hot encoded = "
              << core::CContainerPrinter::print(m_OneHotEncodedCategories));
}

void CDataFrameCategoryEncoder::targetMeanValueEncode(std::size_t numberThreads,
                                                      const core::CDataFrame& frame,
                                                      const TSizeVec& categoricalColumnMask,
                                                      std::size_t targetColumn) {

    m_TargetMeanValues = CDataFrameUtils::meanValueOfTargetForCategories(
        numberThreads, frame, categoricalColumnMask, targetColumn);
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
}
}
