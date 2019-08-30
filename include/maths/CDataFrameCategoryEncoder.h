/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameCategoryEncoder_h
#define INCLUDED_ml_maths_CDataFrameCategoryEncoder_h

#include <core/CDataFrame.h>

#include <maths/CDataFrameUtils.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/unordered_set.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CPackedBitVector;
}
namespace maths {
class CDataFrameCategoryEncoder;

//! \brief A wrapper of a data frame row reference which handles encoding categories.
//!
//! DESCRIPTION:\n
//! This implements a subset of the core::CDataFrame::TRowRef interface dealing with
//! the extra dimensions introduced by one-hot encoded categories, extracting the mean
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
    //! \warning The output iterator that must be able to receive numberColumns values.
    template<typename ITR>
    void copyTo(ITR output) const {
        for (std::size_t i = 0; i < this->numberColumns(); ++i, ++output) {
            *output = this->operator[](i);
        }
    }

    //! Get the underlying row reference.
    const TRowRef& unencodedRow() const { return m_Row; }

private:
    TRowRef m_Row;
    const CDataFrameCategoryEncoder* m_Encoder;
};

//! \brief Performs encoding of the categorical columns in a data frame.
//!
//! DESCRIPTION:\n
//! We select appropriate encodings for categorical values based on their cardinality,
//! the information they carry about the regression target variable and so on. We use
//! a mixture of one-hot encoding for the categories which carry the most information
//! (because this is lossless), mean target encoding for categories where we have a
//! reasonable sample set and a catch all indicator feature for all rare categories.
//!
//! This also selects the independent metric features to use at the same time controlling
//! the number of features we use in total based on the quantity of training data.
class MATHS_EXPORT CDataFrameCategoryEncoder final {
public:
    using TBoolVec = std::vector<bool>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TRowRef = core::CDataFrame::TRowRef;

public:
    //! This ensures we don't try and one-hot encode too many categories and thus
    //! overfit. This represents a reasonable default, but this is likely better
    //! estimated from the data characteristics.
    static constexpr double MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE{0.05};

    //! By default assign roughly twice the importance to maximising relevance vs
    //! minimising redundancy when choosing the encoding and selecting features.
    static constexpr double REDUNDANCY_WEIGHT{0.5};

public:
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the encoding.
    //! \param[in] rowMask A mask of the rows to use to determine the encoding.
    //! \param[in] columnMask A mask of the columns to include.
    //! \param[in] targetColumn The regression target variable.
    //! \param[in] minimumRowsPerFeature The minimum number of rows needed per dimension
    //! of the feature vector.
    //! \param[in] minimumFrequencyToOneHotEncode The minimum relative frequency of a
    //! category in \p frame to consider one-hot encoding it.
    //! \param[in] redundancyWeight Controls the weight between feature MIC with the
    //! target and with the features already selected. This should be non-negative and
    //! the higher the value the more the encoder will prefer to minimise MIC with the
    //! features already selected.
    CDataFrameCategoryEncoder(std::size_t numberThreads,
                              const core::CDataFrame& frame,
                              const core::CPackedBitVector& rowMask,
                              const TSizeVec& columnMask,
                              std::size_t targetColumn,
                              std::size_t minimumRowsPerFeature,
                              double minimumFrequencyToOneHotEncode = MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE,
                              double redundancyWeight = REDUNDANCY_WEIGHT);

    //! Initialize from serialized data.
    CDataFrameCategoryEncoder(core::CStateRestoreTraverser& traverser);

    //! Get a row reference which encodes the categories in \p row.
    CEncodedDataFrameRowRef encode(TRowRef row) const;

    //! Check if \p feature is categorical.
    bool columnIsCategorical(std::size_t feature) const;

    //! Get the MICs of the selected features.
    const TDoubleVec& featureMics() const;

    //! Get the total number of dimensions in the feature vector.
    std::size_t numberFeatures() const;

    //! Get the encoding offset in feature vector of \p index.
    std::size_t encoding(std::size_t index) const;

    //! Get the data frame column of \p index into the feature vector.
    std::size_t column(std::size_t index) const;

    //! Check if \p index is a binary encoded feature.
    bool isBinary(std::size_t index) const;

    //! Get the number of one-hot encoded categories for \p feature.
    std::size_t numberOneHotEncodedCategories(std::size_t feature) const;

    //! Check if \p category of \p feature uses one-hot encoding.
    bool usesOneHotEncoding(std::size_t feature, std::size_t category) const;

    //! Check if feature with encoding \p encoding is one for \p category of \p feature.
    bool isHot(std::size_t encoding, std::size_t feature, std::size_t category) const;

    //! Check if \p feature uses frequency encoding.
    bool usesFrequencyEncoding(std::size_t feature) const;

    //! Check if \p category of \p feature is a rare category.
    bool isRareCategory(std::size_t feature, std::size_t category) const;

    //! Get the frequency of \p category of \p feature.
    double frequency(std::size_t feature, std::size_t category) const;

    //! Get the mean value of the target variable for \p category of \p feature.
    double targetMeanValue(std::size_t feature, std::size_t category) const;

    //! Get a checksum of the state of this object seeded with \p seed.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

private:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
    using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeSizePrDoubleMap = std::map<TSizeSizePr, double>;
    using TSizeUSet = boost::unordered_set<std::size_t>;
    using TSizeUSetVec = std::vector<TSizeUSet>;

private:
    TSizeDoublePrVecVec mics(std::size_t numberThreads,
                             const core::CDataFrame& frame,
                             const CDataFrameUtils::CColumnValue& target,
                             const core::CPackedBitVector& rowMask,
                             const TSizeVec& metricColumnMask,
                             const TSizeVec& categoricalColumnMask) const;
    void setupFrequencyEncoding(std::size_t numberThreads,
                                const core::CDataFrame& frame,
                                const core::CPackedBitVector& rowMask,
                                const TSizeVec& categoricalColumnMask);
    void setupTargetMeanValueEncoding(std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      const core::CPackedBitVector& rowMask,
                                      const TSizeVec& categoricalColumnMask,
                                      std::size_t targetColumn);
    TSizeSizePrDoubleMap selectFeatures(std::size_t numberThreads,
                                        const core::CDataFrame& frame,
                                        const core::CPackedBitVector& rowMask,
                                        TSizeVec metricColumnMask,
                                        TSizeVec categoricalColumnMask,
                                        std::size_t targetColumn);
    TSizeSizePrDoubleMap selectAllFeatures(const TSizeDoublePrVecVec& mics);
    void finishEncoding(std::size_t targetColumn, TSizeSizePrDoubleMap selectedFeatureMics);
    std::size_t numberAvailableFeatures(const TSizeDoublePrVecVec& mics) const;

private:
    std::size_t m_MinimumRowsPerFeature;
    double m_MinimumFrequencyToOneHotEncode;
    double m_RedundancyWeight;
    TBoolVec m_ColumnIsCategorical;
    TBoolVec m_ColumnUsesFrequencyEncoding;
    TSizeVecVec m_OneHotEncodedCategories;
    TSizeUSetVec m_RareCategories;
    TDoubleVecVec m_CategoryFrequencies;
    TDoubleVec m_MeanCategoryFrequencies;
    TDoubleVecVec m_CategoryTargetMeanValues;
    TDoubleVec m_MeanCategoryTargetMeanValues;
    TDoubleVec m_FeatureVectorMics;
    TSizeVec m_FeatureVectorColumnMap;
    TSizeVec m_FeatureVectorEncodingMap;
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameCategoryEncoder_h
