/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameCategoryEncoder_h
#define INCLUDED_ml_maths_CDataFrameCategoryEncoder_h

#include <core/CDataFrame.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <utility>
#include <vector>

namespace ml {
namespace maths {
class CDataFrameCategoryEncoder;

//! \brief A wrapper of a data frame row reference which handles encoding categories.
//!
//! DESCRIPTION:\n
//! This implements a subset of the core::CDataFrame::TRowRef interface dealing with
//! the extra dimensions introduced by hot-one encoded categories, extracting the mean
//! target variable value for other categories and so on. The intention is it provides
//! a drop in replacement for core::CDataFrame::TRowRef when a data frame contains
//! categorical columns.
class MATHS_EXPORT CEncodedDataFrameRowRef final {
public:
    using TRowRef = core::CDataFrame::TRowRef;

public:
    CEncodedDataFrameRowRef(TRowRef row, const CDataFrameCategoryEncoder& encoder);

    //! Get column \p i value.
    CFloatStorage operator[](std::size_t i) const;

    //! Get the row's index.
    std::size_t index() const;

    //! Get the number of columns.
    std::size_t numberColumns() const;

    //! Copy the range to \p output iterator.
    //!
    //! \warning The output iterator that must be able to receive numberColumns
    //! values.
    template<typename ITR>
    void copyTo(ITR output) const {
        for (std::size_t i = 0; i < this->numberColumns(); ++i, ++output) {
            *output = this->operator[](i);
        }
    }

private:
    TRowRef m_Row;
    const CDataFrameCategoryEncoder* m_Encoder;
};

//! \brief Performs encoding of the categorical columns in a data frame.
//!
//! DESCRIPTION:\n
//! We select appropriate encodings for categorical values based on their cardinality,
//! the information they carry about the regression target variable and so on. We use
//! a mixture of hot-one encoding for the categories which carry the most information
//! (because this is lossless), mean target encoding for categories where we have a
//! reasonable sample set and a catch all indicator feature for all rare categories.
//!
//! This also selects the independent metric features to use at the same time controlling
//! the total number of features we use in total based on the quantity of training data.
class MATHS_EXPORT CDataFrameCategoryEncoder final {
public:
    using TBoolVec = std::vector<bool>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TRowRef = core::CDataFrame::TRowRef;

public:
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the encoding.
    //! \param[in] columnMask A mask of the columns to include.
    //! \param[in] targetColumn The column whose mean values are computed.
    //! \param[in] minimumRowsPerFeature The minimum number of rows needed per dimension
    //! of the feature vector.
    //! \param[in] minimumFrequencyToHotOneEncode The minimum relative frequency of a
    //! category in \p frame to consider hot-one encoding it.
    //! \param[in] lambda Controls the weight between category MICe and the count of
    //! categories already hot-one encoded for a feature when selecting features. This
    //! should be positive and the higher the value the more it will prefer to choose
    //! all the metrics and equal numbers from each category to hot-one encode.
    CDataFrameCategoryEncoder(std::size_t numberThreads,
                              const core::CDataFrame& frame,
                              const TSizeVec& columnMask,
                              std::size_t targetColumn,
                              std::size_t minimumRowsPerFeature,
                              double minimumFrequencyToHotOneEncode = 0.01,
                              double lambda = 0.2 * std::log(2.0));

    //! Get a row reference which encodes the categories in \p row.
    CEncodedDataFrameRowRef encode(TRowRef row);

    //! Get an indicator vector of categorical columns.
    bool columnIsCategorical(std::size_t feature) const;

    //! Get the selected metric features.
    const TSizeVec& selectedMetricFeatures() const;

    //! Get the selected metric features' MICs.
    const TDoubleVec& selectedMetricFeatureMics() const;

    //! Get the selected categorical features.
    const TSizeVec& selectedCategoricalFeatures() const;

    //! Get the number of dimensions in the feature vector.
    std::size_t numberFeatures() const;

    //! Get the encoding offset in feature vector of \p index.
    std::size_t encoding(std::size_t index) const;

    //! Get the data frame column of the feature vector \p index.
    std::size_t column(std::size_t index) const;

    //! Get the number of hot-one encoded categories for \p feature.
    std::size_t numberHotOneEncodedCategories(std::size_t feature) const;

    //! Check if \p index is the hot-one for category \p category.
    bool isHotOne(std::size_t index, std::size_t feature, std::size_t category) const;

    //! Check if \p feature has rare categories.
    bool hasRareCategories(std::size_t feature) const;

    //! Check if \p category is rare.
    bool isRareCategory(std::size_t feature, std::size_t category) const;

    //! Get the mean value of the target variable for \p category.
    double targetMeanValue(std::size_t feature, std::size_t category) const;

private:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
    using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;

private:
    void isRareEncode(std::size_t numberThreads,
                      const core::CDataFrame& frame,
                      const TSizeVec& categoricalColumnMask,
                      std::size_t minimumRowsPerFeature);
    void hotOneEncode(std::size_t numberThreads,
                      const core::CDataFrame& frame,
                      const TSizeVec& metricColumnMask,
                      const TSizeVec& categoricalColumnMask,
                      std::size_t targetColumn,
                      std::size_t minimumRowsPerFeature,
                      double minimumFrequencyToHotOneEncode);
    void targetMeanValueEncode(std::size_t numberThreads,
                               const core::CDataFrame& frame,
                               const TSizeVec& categoricalColumnMask,
                               std::size_t targetColumn);
    void setupEncodingMaps(const core::CDataFrame& frame);

private:
    double m_Lambda;
    TBoolVec m_ColumnIsCategorical;
    TSizeVec m_SelectedMetricFeatures;
    TDoubleVec m_SelectedMetricFeatureMics;
    TSizeVec m_SelectedCategoricalFeatures;
    TSizeVec m_FeatureVectorColumnMap;
    TSizeVec m_FeatureVectorEncodingMap;
    TDoubleVecVec m_TargetMeanValues;
    TSizeVecVec m_RareCategories;
    TSizeVecVec m_HotOneEncodedCategories;
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameCategoryEncoder_h
