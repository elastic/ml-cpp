/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameCategoryEncoder_h
#define INCLUDED_ml_maths_CDataFrameCategoryEncoder_h

#include <core/CDataFrame.h>
#include <core/CPackedBitVector.h>

#include <maths/CDataFrameUtils.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/optional.hpp>
#include <boost/unordered_set.hpp>

#include <cstdint>
#include <memory>
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
    CFloatStorage operator[](std::size_t encodedColumnIndex) const;

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

class CMakeDataFrameCategoryEncoder;

//! Encoding styles considered.
enum EEncoding {
    E_OneHot = 0,
    E_Frequency,
    E_TargetMean,
    E_IdentityEncoding // This must stay at the end
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
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TRowRef = core::CDataFrame::TRowRef;

    //! \brief Base type of category encodings.
    class MATHS_EXPORT CEncoding {
    public:
        CEncoding(std::size_t inputColumnIndex, double mic);
        virtual ~CEncoding() = default;
        virtual EEncoding type() const = 0;
        virtual double encode(double value) const = 0;
        virtual std::uint64_t checksum() const = 0;
        virtual bool isBinary() const = 0;
        //! return encoding type as string
        virtual std::string typeString() const = 0;

        std::size_t inputColumnIndex() const;
        double encode(const TRowRef& row) const;
        double mic() const;
        //! Persist by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
        //! Populate the object from serialized data.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

        CEncoding(const CEncoding&) = delete;

    protected:
        std::size_t m_InputColumnIndex;
        double m_Mic;

    private:
        virtual void
        acceptPersistInserterForDerivedTypeState(core::CStatePersistInserter& inserter) const = 0;
        virtual bool
        acceptRestoreTraverserForDerivedTypeState(core::CStateRestoreTraverser& traverser) = 0;
    };

    using TEncodingUPtr = std::unique_ptr<CEncoding>;
    using TEncodingUPtrVec = std::vector<TEncodingUPtr>;

    //! \brief Returns the supplied value.
    class MATHS_EXPORT CIdentityEncoding : public CEncoding {
    public:
        CIdentityEncoding(std::size_t inputColumnIndex, double mic);
        EEncoding type() const override;
        double encode(double value) const override;
        bool isBinary() const override;
        std::uint64_t checksum() const override;
        std::string typeString() const override;

    private:
        void acceptPersistInserterForDerivedTypeState(core::CStatePersistInserter& inserter) const override;
        bool acceptRestoreTraverserForDerivedTypeState(core::CStateRestoreTraverser& traverser) override;
    };

    //! \brief One-hot encoding.
    class MATHS_EXPORT COneHotEncoding : public CEncoding {
    public:
        COneHotEncoding(std::size_t inputColumnIndex, double mic, std::size_t hotCategory);
        EEncoding type() const override;
        double encode(double value) const override;
        bool isBinary() const override;
        std::uint64_t checksum() const override;
        std::string typeString() const override;

    private:
        void acceptPersistInserterForDerivedTypeState(core::CStatePersistInserter& inserter) const override;
        bool acceptRestoreTraverserForDerivedTypeState(core::CStateRestoreTraverser& traverser) override;

    private:
        std::size_t m_HotCategory;
    };

    //! \brief Looks up the encoding in a map.
    class MATHS_EXPORT CMappedEncoding : public CEncoding {
    public:
        CMappedEncoding(std::size_t inputColumnIndex,
                        double mic,
                        EEncoding encoding,
                        const TDoubleVec& map,
                        double fallback);
        EEncoding type() const override;
        double encode(double value) const override;
        bool isBinary() const override;
        std::uint64_t checksum() const override;
        std::string typeString() const override;

    private:
        void acceptPersistInserterForDerivedTypeState(core::CStatePersistInserter& inserter) const override;
        bool acceptRestoreTraverserForDerivedTypeState(core::CStateRestoreTraverser& traverser) override;

    private:
        EEncoding m_Encoding;
        TDoubleVec m_Map;
        double m_Fallback;
        bool m_Binary;
    };

public:
    CDataFrameCategoryEncoder(CMakeDataFrameCategoryEncoder parameters);

    //! Initialize from serialized data.
    CDataFrameCategoryEncoder(core::CStateRestoreTraverser& traverser);

    //! Get a row reference which encodes the categories in \p row.
    CEncodedDataFrameRowRef encode(TRowRef row) const;

    //! Get the MICs of the selected features.
    TDoubleVec encodedColumnMics() const;

    //! Get the total number of dimensions in the feature vector.
    std::size_t numberEncodedColumns() const;

    //! Get the encoded feature at position \p encodedColumnIndex.
    const CEncoding& encoding(std::size_t encodedColumnIndex) const;

    //! Check if \p encodedColumnIndex is a binary encoded feature.
    bool isBinary(std::size_t encodedColumnIndex) const;

    //! Get a checksum of the state of this object seeded with \p seed.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

private:
    bool restoreEncodings(core::CStateRestoreTraverser& traverser);
    template<typename T, typename... Args>
    bool forwardRestoreEncodings(core::CStateRestoreTraverser& traverser, Args&&... args);

private:
    TEncodingUPtrVec m_Encodings;
};

//! \brief Implements the named parameter idiom for CDataFrameCategoryEncoder.
class MATHS_EXPORT CMakeDataFrameCategoryEncoder {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalDouble = boost::optional<double>;
    using TEncodingUPtrVec = CDataFrameCategoryEncoder::TEncodingUPtrVec;

public:
    //! The minimum number of training rows needed per feature used.
    static constexpr std::size_t MINIMUM_ROWS_PER_FEATURE{50};

    //! This ensures we don't try and one-hot encode too many categories and thus
    //! overfit. This represents a reasonable default, but this is likely better
    //! estimated from the data characteristics.
    static constexpr double MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE{0.05};

    //! The minimum relative MIC of a feature to select it for training.
    static constexpr double MINIMUM_RELATIVE_MIC_TO_SELECT_FEATURE{1e-3};

    //! By default assign roughly twice the importance to maximising relevance vs
    //! minimising redundancy when choosing the encoding and selecting features.
    static constexpr double REDUNDANCY_WEIGHT{0.5};

public:
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the encoding.
    //! \param[in] targetColumn The regression target variable.
    CMakeDataFrameCategoryEncoder(std::size_t numberThreads,
                                  const core::CDataFrame& frame,
                                  std::size_t targetColumn);

    //! Set the minimum number of training rows needed per feature used.
    CMakeDataFrameCategoryEncoder& minimumRowsPerFeature(std::size_t minimumRowsPerFeature);

    //! Set the minimum relative frequency of a category in \p frame to consider one-hot
    //! encoding it.
    CMakeDataFrameCategoryEncoder&
    minimumFrequencyToOneHotEncode(TOptionalDouble minimumFrequencyToOneHotEncode);

    //! Set the minimum relative MIC, i.e. one vs all, of a feature to consider using it.
    CMakeDataFrameCategoryEncoder&
    minimumRelativeMicToSelectFeature(TOptionalDouble minimumRelativeMicToSelectFeature);

    //! Set the weight between feature MIC with the target and with the features already
    //! selected when choosing the order in which to select features. This should be non-
    //! negative and the higher the value the more the encoder will prefer to minimise
    //! MIC with the features already selected.
    CMakeDataFrameCategoryEncoder& redundancyWeight(TOptionalDouble redundancyWeight);

    //! Set a mask of the rows to use to determine the encoding.
    CMakeDataFrameCategoryEncoder& rowMask(core::CPackedBitVector rowMask);

    //! Set a mask of the columns to include.
    CMakeDataFrameCategoryEncoder& columnMask(TSizeVec columnMask);

    //! Make the encoding.
    TEncodingUPtrVec makeEncodings();

    //! \name Test Methods
    //@{
    //! Get the encoding offset in feature vector of \p index.
    std::size_t encoding(std::size_t index) const;

    //! Check if \p category of \p inputColumnIndex uses one-hot encoding.
    bool usesOneHotEncoding(std::size_t inputColumnIndex, std::size_t category) const;

    //! Check if \p category of \p inputColumnIndex is a rare category.
    bool isRareCategory(std::size_t inputColumnIndex, std::size_t category) const;
    //@}

private:
    using TBoolVec = std::vector<bool>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
    using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeSizePrDoubleMap = std::map<TSizeSizePr, double>;
    using TSizeUSet = boost::unordered_set<std::size_t>;
    using TSizeUSetVec = std::vector<TSizeUSet>;

private:
    TSizeDoublePrVecVec mics(const CDataFrameUtils::CColumnValue& target,
                             const TSizeVec& metricColumnMask,
                             const TSizeVec& categoricalColumnMask) const;
    void setupFrequencyEncoding(const TSizeVec& categoricalColumnMask);
    void setupTargetMeanValueEncoding(const TSizeVec& categoricalColumnMask);
    TSizeSizePrDoubleMap selectFeatures(TSizeVec metricColumnMask,
                                        const TSizeVec& categoricalColumnMask);
    TSizeSizePrDoubleMap selectAllFeatures(const TSizeDoublePrVecVec& mics);
    void finishEncoding(TSizeSizePrDoubleMap selectedFeatureMics);
    void discardNuisanceFeatures(TSizeDoublePrVecVec& mics) const;
    std::size_t numberAvailableFeatures(const TSizeDoublePrVecVec& mics) const;

private:
    // Begin parameters
    std::size_t m_MinimumRowsPerFeature = MINIMUM_ROWS_PER_FEATURE;
    double m_MinimumFrequencyToOneHotEncode = MINIMUM_FREQUENCY_TO_ONE_HOT_ENCODE;
    double m_MinimumRelativeMicToSelectFeature = MINIMUM_RELATIVE_MIC_TO_SELECT_FEATURE;
    double m_RedundancyWeight = REDUNDANCY_WEIGHT;
    std::size_t m_NumberThreads;
    const core::CDataFrame* m_Frame;
    core::CPackedBitVector m_RowMask;
    TSizeVec m_ColumnMask;
    std::size_t m_TargetColumn;
    // End parameters

    TBoolVec m_InputColumnUsesFrequencyEncoding;
    TSizeVecVec m_OneHotEncodedCategories;
    TSizeUSetVec m_RareCategories;
    TDoubleVecVec m_CategoryFrequencies;
    TDoubleVec m_MeanCategoryFrequencies;
    TDoubleVecVec m_CategoryTargetMeanValues;
    TDoubleVec m_MeanCategoryTargetMeanValues;
    TDoubleVec m_EncodedColumnMics;
    TSizeVec m_EncodedColumnInputColumnMap;
    TSizeVec m_EncodedColumnEncodingMap;
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameCategoryEncoder_h
