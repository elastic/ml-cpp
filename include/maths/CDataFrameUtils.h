/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameUtils_h
#define INCLUDED_ml_maths_CDataFrameUtils_h

#include <core/CDataFrame.h>
#include <core/CNonInstantiatable.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CPackedBitVector;
}
namespace maths {
class CDataFrameCategoryEncoder;
class CQuantileSketch;

namespace data_frame_utils_detail {
template<typename T>
struct SRowTo {
    static_assert(sizeof(T) < 0, "Vector type not supported");
};

template<typename T, Eigen::AlignmentType ALIGNMENT>
struct SRowTo<CMemoryMappedDenseVector<T, ALIGNMENT>> {
    static CMemoryMappedDenseVector<T, ALIGNMENT>
    dispatch(const core::CDataFrame::TRowRef& row) {
        return {row.data(), static_cast<long>(row.numberColumns())};
    }
};

template<typename T>
struct SRowTo<CDenseVector<T>> {
    static CDenseVector<T> dispatch(const core::CDataFrame::TRowRef& row) {
        std::size_t n{row.numberColumns()};
        CDenseVector<T> result{SConstant<CDenseVector<T>>::get(n, 0)};
        for (std::size_t i = 0; i < n; ++i) {
            result(i) = row[i];
        }
        return result;
    }
};
}

//! \brief A collection of basic utilities for analyses on a data frame.
class MATHS_EXPORT CDataFrameUtils : private core::CNonInstantiatable {
public:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TFloatVec = std::vector<CFloatStorage>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
    using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;
    using TSizeDoublePrVecVecVec = std::vector<TSizeDoublePrVecVec>;
    using TRowRef = core::CDataFrame::TRowRef;
    using TWeightFunc = std::function<double(const TRowRef&)>;
    using TDoubleVector = CDenseVector<double>;
    using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;
    using TReadPredictionFunc = std::function<TMemoryMappedFloatVector(const TRowRef&)>;
    using TQuantileSketchVec = std::vector<CQuantileSketch>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;

    //! \brief Defines the data type of a collection of numbers.
    struct MATHS_EXPORT SDataType {
        static const char EXTERNAL_DELIMITER;
        static const char INTERNAL_DELIMITER;

        std::string toDelimited() const;
        bool fromDelimited(const std::string& delimited);

        bool s_IsInteger;
        double s_Min;
        double s_Max;
    };

    using TDataTypeVec = std::vector<SDataType>;

    //! \brief Used to extract the value from a specific column of the data frame.
    class MATHS_EXPORT CColumnValue {
    public:
        CColumnValue(std::size_t column) : m_Column{column} {}
        virtual ~CColumnValue() = default;
        virtual double operator()(const TRowRef& row) const = 0;
        virtual double operator()(const TFloatVec& row) const = 0;
        virtual std::size_t hash() const = 0;

    protected:
        std::size_t column() const { return m_Column; }

    private:
        std::size_t m_Column;
    };

    //! \brief Used to extract the value from a metric column of the data frame.
    class MATHS_EXPORT CMetricColumnValue final : public CColumnValue {
    public:
        CMetricColumnValue(std::size_t column) : CColumnValue{column} {}
        double operator()(const TRowRef& row) const override {
            return row[this->column()];
        }
        double operator()(const TFloatVec& row) const override {
            return row[this->column()];
        }
        std::size_t hash() const override { return 0; }
    };

    //! \brief Used to extract the value from a one-hot encoded categorical column
    //! of the data frame.
    class MATHS_EXPORT COneHotCategoricalColumnValue final : public CColumnValue {
    public:
        COneHotCategoricalColumnValue(std::size_t column, std::size_t category)
            : CColumnValue{column}, m_Category{category} {}
        double operator()(const TRowRef& row) const override {
            if (isMissing(row[this->column()])) {
                return core::CDataFrame::valueOfMissing();
            }
            return static_cast<std::size_t>(row[this->column()]) == m_Category ? 1.0 : 0.0;
        }
        double operator()(const TFloatVec& row) const override {
            if (isMissing(row[this->column()])) {
                return core::CDataFrame::valueOfMissing();
            }
            return static_cast<std::size_t>(row[this->column()]) == m_Category ? 1.0 : 0.0;
        }
        std::size_t hash() const override { return m_Category; }

    private:
        std::size_t m_Category;
    };

    //! \brief Used to extract the value from a "is rare" encoded categorical column
    //! of the data frame.
    class MATHS_EXPORT CFrequencyCategoricalColumnValue final : public CColumnValue {
    public:
        CFrequencyCategoricalColumnValue(std::size_t column, const TDoubleVec& frequencies)
            : CColumnValue{column}, m_Frequencies{&frequencies} {}
        double operator()(const TRowRef& row) const override {
            if (isMissing(row[this->column()])) {
                return core::CDataFrame::valueOfMissing();
            }
            std::size_t category{static_cast<std::size_t>(row[this->column()])};
            return (*m_Frequencies)[category];
        }
        double operator()(const TFloatVec& row) const override {
            if (isMissing(row[this->column()])) {
                return core::CDataFrame::valueOfMissing();
            }
            std::size_t category{static_cast<std::size_t>(row[this->column()])};
            return (*m_Frequencies)[category];
        }
        std::size_t hash() const override { return 0; }

    private:
        const TDoubleVec* m_Frequencies;
    };

    //! \brief Used to extract the value from a mean target encoded categorical
    //! column of the data frame.
    class MATHS_EXPORT CTargetMeanCategoricalColumnValue final : public CColumnValue {
    public:
        using TSizeUSet = boost::unordered_set<std::size_t>;

    public:
        CTargetMeanCategoricalColumnValue(std::size_t column,
                                          const TSizeUSet& rareCategories,
                                          const TDoubleVec& targetMeanValues)
            : CColumnValue{column}, m_RareCategories{&rareCategories}, m_TargetMeanValues{&targetMeanValues} {
        }
        double operator()(const TRowRef& row) const override {
            if (isMissing(row[this->column()])) {
                return core::CDataFrame::valueOfMissing();
            }
            std::size_t category{static_cast<std::size_t>(row[this->column()])};
            return this->isRare(category) ? 0.0 : (*m_TargetMeanValues)[category];
        }
        double operator()(const TFloatVec& row) const override {
            if (isMissing(row[this->column()])) {
                return core::CDataFrame::valueOfMissing();
            }
            std::size_t category{static_cast<std::size_t>(row[this->column()])};
            return this->isRare(category) ? 0.0 : (*m_TargetMeanValues)[category];
        }
        std::size_t hash() const override { return 0; }

    private:
        bool isRare(std::size_t category) const {
            return m_RareCategories->find(category) != m_RareCategories->end();
        }

    private:
        const TSizeUSet* m_RareCategories;
        const TDoubleVec* m_TargetMeanValues;
    };

    using TEncoderFactory =
        std::function<std::unique_ptr<CColumnValue>(std::size_t, std::size_t, std::size_t)>;
    using TEncoderFactoryVec = std::vector<std::pair<TEncoderFactory, double>>;

public:
    //! Convert a row of the data frame to a specified vector type.
    template<typename VECTOR>
    static VECTOR rowTo(const core::CDataFrame::TRowRef& row) {
        return data_frame_utils_detail::SRowTo<VECTOR>::dispatch(row);
    }

    //! Subtract the mean and divide each column value by its standard deviation.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in,out] frame The data frame whose columns are to be standardized.
    static bool standardizeColumns(std::size_t numberThreads, core::CDataFrame& frame);

    //! Extract column data types.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame whose columns are to be standardized.
    //! \param[in] rowMask A mask of the rows from which to compute data types.
    //! \param[in] columnMask A mask of the columns for which to compute data types.
    //! \param[in] encoder If non-null used to encode the rows for which to compute
    //! data types.
    static TDataTypeVec columnDataTypes(std::size_t numberThreads,
                                        const core::CDataFrame& frame,
                                        const core::CPackedBitVector& rowMask,
                                        const TSizeVec& columnMask,
                                        const CDataFrameCategoryEncoder* encoder = nullptr);

    //! Get a quantile sketch of each column's values.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the column quantiles.
    //! \param[in] rowMask A mask of the rows from which to compute quantiles.
    //! \param[in] columnMask A mask of the columns for which to compute quantiles.
    //! \param[in] quantileEstimator Estimates the quantiles of a collection of
    //! weighted samples.
    //! \param[in] encoder If non-null used to encode the rows for which to compute
    //! quantiles.
    //! \param[in] weight The weight to assign each row. The default is unity for
    //! all rows.
    static std::pair<TQuantileSketchVec, bool>
    columnQuantiles(std::size_t numberThreads,
                    const core::CDataFrame& frame,
                    const core::CPackedBitVector& rowMask,
                    const TSizeVec& columnMask,
                    CQuantileSketch quantileEstimator,
                    const CDataFrameCategoryEncoder* encoder = nullptr,
                    const TWeightFunc& weight = unitWeight);

    //! \brief Compute disjoint stratified random train/test row masks suitable
    //! for cross-validation.
    //!
    //! This works for both classification and regression.
    //!
    //! For classification we assign uniformly at random equal proportions of
    //! each class as they appear in the full training set. For example, if we
    //! have \f$n_A\f$ and \f$n_B\f$ counts of class A and B, respectively, and
    //! we're doing \f$k\f$-fold cross validation we'd choose random subsets of
    //! sizes \f$\frac{n_A}{k}\f$ and \f$\frac{n_B}{k}\f$ of A and B examples,
    //! respectively, for each test set.
    //!
    //! For regression we do stratified fractional cross-validation on \p numberBins
    //! data quantiles. This is equivalent to stratification for classification
    //! where each class accounts for 1/\p numberBins of the data and the comprises
    //! the examples in each inter quantile range.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the row masks.
    //! \param[in] targetColumn The index of the column to predict.
    //! \param[in] rng The random number generator to use.
    //! \param[in] numberFolds The number of folds to use.
    //! \param[in] numberBuckets The number of buckets to use when stratifying by
    //! target quantiles for regression.
    //! \param[in] allTrainingRowsMask A mask of the candidate training rows.
    //! \warning This fails if the target is not categorical.
    static std::tuple<TPackedBitVectorVec, TPackedBitVectorVec, TDoubleVec>
    stratifiedCrossValidationRowMasks(std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      std::size_t targetColumn,
                                      CPRNG::CXorOShiro128Plus rng,
                                      std::size_t numberFolds,
                                      std::size_t numberBuckets,
                                      const core::CPackedBitVector& allTrainingRowsMask);

    //! Get the relative frequency of each category in \p frame.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute category frequencies.
    //! \param[in] rowMask A mask of the rows from which to compute category frequencies.
    //! \param[in] columnMask A mask of the columns to include.
    //! \return The frequency of each category. The collection is indexed by column
    //! and then category identifier.
    static TDoubleVecVec categoryFrequencies(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             const core::CPackedBitVector& rowMask,
                                             TSizeVec columnMask);

    //! Compute the mean value of \p target on the restriction to the rows labelled
    //! by each distinct category of the categorical columns.
    //!
    //! \param[in] target The column value for which to compute the mean value.
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute mean values.
    //! \param[in] rowMask A mask of the rows from which to compute mean values.
    //! \param[in] columnMask A mask of the columns to include.
    //! \return The mean values of \p target for each category. The collection
    //! is indexed by column and then category identifier.
    static TDoubleVecVec meanValueOfTargetForCategories(const CColumnValue& target,
                                                        std::size_t numberThreads,
                                                        const core::CDataFrame& frame,
                                                        const core::CPackedBitVector& rowMask,
                                                        TSizeVec columnMask);

    //! Assess the strength of the relationship for each distinct category with
    //! \p target by computing the maximum information coefficient (MIC).
    //!
    //! \param[in] target Extracts the column value with which to compute MIC.
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the category MICs.
    //! \param[in] rowMask A mask of the rows from which to compute MIC.
    //! \param[in] columnMask A mask of the columns to include.
    //! \param[in] encoderFactories A collection of (factory, minimum frequency)
    //! pairs, with the factory making the encoder for distinct category values
    //! and the minimum frequency being the minimum frequency for a category in
    //! the frame to use the corresponding encoding.
    //! \return A collection containing (category, MIC with \p target) pairs
    //! for each category whose frequency in \p frame is greater than \p minimumFrequency
    //! and each categorical column. The collection is indexed by column.
    static TSizeDoublePrVecVecVec
    categoricalMicWithColumn(const CColumnValue& target,
                             std::size_t numberThreads,
                             const core::CDataFrame& frame,
                             const core::CPackedBitVector& rowMask,
                             TSizeVec columnMask,
                             const TEncoderFactoryVec& encoderFactories);

    //! Assess the strength of the relationship for each metric valued column with
    //! \p target by computing the maximum information coefficient (MIC).
    //!
    //! \param[in] target Extracts the column value with which to compute MIC.
    //! \param[in] frame The data frame for which to compute the column MICs.
    //! \param[in] rowMask A mask of the rows from which to compute MIC.
    //! \param[in] columnMask A mask of the columns to include.
    //! \return A collection containing the MIC of each metric valued column with
    //! \p target. The collectionis indexed by column.
    static TDoubleVec metricMicWithColumn(const CColumnValue& target,
                                          const core::CDataFrame& frame,
                                          const core::CPackedBitVector& rowMask,
                                          TSizeVec columnMask);

    //! Compute the multiplicative weights to apply to each class probability such
    //! that choosing the maximum "weighted" probability class for each example
    //! maximizes the minimum per class recall.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the threshold.
    //! \param[in] rowMask A mask of the rows from which to compute the threshold.
    //! \param[in] numberClasses The number of possible classes.
    //! \param[in] targetColumn The index of the column to predict.
    //! \param[in] readPrediction Callback to read the prediction from a row.
    static TDoubleVector
    maximumMinimumRecallClassWeights(std::size_t numberThreads,
                                     const core::CDataFrame& frame,
                                     const core::CPackedBitVector& rowMask,
                                     std::size_t numberClasses,
                                     std::size_t targetColumn,
                                     const TReadPredictionFunc& readPrediction);

    //! Check if a data frame value is missing.
    static bool isMissing(double value);

private:
    static TSizeDoublePrVecVecVec
    categoricalMicWithColumnDataFrameInMemory(const CColumnValue& target,
                                              const core::CDataFrame& frame,
                                              const core::CPackedBitVector& rowMask,
                                              const TSizeVec& columnMask,
                                              const TEncoderFactoryVec& encoderFactories,
                                              const TDoubleVecVec& frequencies,
                                              std::size_t numberSamples);
    static TSizeDoublePrVecVecVec
    categoricalMicWithColumnDataFrameOnDisk(const CColumnValue& target,
                                            const core::CDataFrame& frame,
                                            const core::CPackedBitVector& rowMask,
                                            const TSizeVec& columnMask,
                                            const TEncoderFactoryVec& encoderFactories,
                                            const TDoubleVecVec& frequencies,
                                            std::size_t numberSamples);
    static TDoubleVec
    metricMicWithColumnDataFrameInMemory(const CColumnValue& target,
                                         const core::CDataFrame& frame,
                                         const core::CPackedBitVector& rowMask,
                                         const TSizeVec& columnMask,
                                         std::size_t numberSamples);
    static TDoubleVec
    metricMicWithColumnDataFrameOnDisk(const CColumnValue& target,
                                       const core::CDataFrame& frame,
                                       const core::CPackedBitVector& rowMask,
                                       const TSizeVec& columnMask,
                                       std::size_t numberSamples);
    static TDoubleVector
    maximizeMinimumRecallForBinary(std::size_t numberThreads,
                                   const core::CDataFrame& frame,
                                   const core::CPackedBitVector& rowMask,
                                   std::size_t targetColumn,
                                   const TReadPredictionFunc& readPrediction);
    static TDoubleVector
    maximizeMinimumRecallForMulticlass(std::size_t numberThreads,
                                       const core::CDataFrame& frame,
                                       const core::CPackedBitVector& rowMask,
                                       std::size_t numberClasses,
                                       std::size_t targetColumn,
                                       const TReadPredictionFunc& readPrediction);
    static void removeMetricColumns(const core::CDataFrame& frame, TSizeVec& columnMask);
    static void removeCategoricalColumns(const core::CDataFrame& frame, TSizeVec& columnMask);
    static double unitWeight(const TRowRef&);
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameUtils_h
