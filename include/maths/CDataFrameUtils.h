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
#include <maths/ImportExport.h>

#include <functional>
#include <vector>

namespace ml {
namespace core {
class CPackedBitVector;
}
namespace maths {
class CQuantileSketch;

namespace data_frame_utils_detail {
template<typename T>
struct SRowTo {
    static_assert(sizeof(T) < 0, "Vector type not supported");
};

template<typename T>
struct SRowTo<CMemoryMappedDenseVector<T>> {
    static CMemoryMappedDenseVector<T> dispatch(const core::CDataFrame::TRowRef& row) {
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
    using TRowRef = core::CDataFrame::TRowRef;
    using TWeightFunction = std::function<double(TRowRef)>;
    using TQuantileSketchVec = std::vector<CQuantileSketch>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;

    //! \brief Used to extract the value from a specific column of the data frame.
    class CColumnValue {
    public:
        virtual ~CColumnValue() = default;
        virtual double operator()(const TRowRef& row) const = 0;
        virtual double operator()(const TFloatVec& row) const = 0;
    };

    //! \brief Used to extract the value from a metric column of the data frame.
    class CMetricColumnValue final : public CColumnValue {
    public:
        CMetricColumnValue(std::size_t column) : m_Column{column} {}
        double operator()(const TRowRef& row) const override {
            return row[m_Column];
        }
        double operator()(const TFloatVec& row) const override {
            return row[m_Column];
        }

    private:
        std::size_t m_Column;
    };

    //! \brief Used to extract the value from a one-hot encoded categorical column
    //! of the data frame.
    class COneHotCategoricalColumnValue final : public CColumnValue {
    public:
        COneHotCategoricalColumnValue(std::size_t column, CFloatStorage category)
            : m_Column{column}, m_Category{static_cast<std::size_t>(category)} {}
        double operator()(const TRowRef& row) const override {
            return static_cast<std::size_t>(row[m_Column]) == m_Category ? 1.0 : 0.0;
        }
        double operator()(const TFloatVec& row) const override {
            return static_cast<std::size_t>(row[m_Column]) == m_Category ? 1.0 : 0.0;
        }

    private:
        std::size_t m_Column;
        std::size_t m_Category;
    };

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

    //! Get a quantile sketch of each column's values.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the column quantiles.
    //! \param[in] rowMask A mask of the rows from which to compute quantiles.
    //! \param[in] columnMask A mask of the columns for which to compute quantiles.
    //! \param[in] sketch The sketch to be used to estimate column quantiles.
    //! \param[out] result Filled in with the column quantile estimates.
    //! \param[in] weight The weight to assign each row. The default is unity for
    //! all rows.
    static bool columnQuantiles(std::size_t numberThreads,
                                const core::CDataFrame& frame,
                                const core::CPackedBitVector& rowMask,
                                const TSizeVec& columnMask,
                                const CQuantileSketch& sketch,
                                TQuantileSketchVec& result,
                                TWeightFunction weight = unitWeight);

    //! Get the relative frequency of each category in \p frame.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute category frequencies.
    //! \param[in] columnMask A mask of the columns to include.
    //! \return The frequency of each category. The collection is indexed by column
    //! and then category identifier.
    static TDoubleVecVec categoryFrequencies(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             TSizeVec columnMask);

    //! Compute the mean value of \p target on the restriction to the rows labelled
    //! by each distinct category of the categorical columns.
    //!
    //! \param[in] target Extracts the column value with which to compute MIC.
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute mean values.
    //! \param[in] columnMask A mask of the columns to include.
    //! \return The mean values of \p target for each category. The collection
    //! is indexed by column and then category identifier.
    static TDoubleVecVec meanValueOfTargetForCategories(const CColumnValue& target,
                                                        std::size_t numberThreads,
                                                        const core::CDataFrame& frame,
                                                        TSizeVec columnMask);

    //! Assess the strength of the relationship for each distinct category with
    //! \p target by computing the maximum information coefficient (MIC).
    //!
    //! \param[in] target Extracts the column value with which to compute MIC.
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame for which to compute the category MICs.
    //! \param[in] columnMask A mask of the columns to include.
    //! \param[in] minimumFrequency The minimum frequency of the category in the
    //! data set for which to compute MIC.
    //! \return A collection containing (category, MIC with \p target) pairs
    //! for each category whose frequency in \p frame is greater than \p minimumFrequency
    //! and each categorical column. The collection is indexed by column.
    static TSizeDoublePrVecVec categoryMicWithColumn(const CColumnValue& target,
                                                     std::size_t numberThreads,
                                                     const core::CDataFrame& frame,
                                                     TSizeVec columnMask,
                                                     double minimumFrequency = 0.01);

    //! Assess the strength of the relationship for each metric valued column with
    //! \p target by computing the maximum information coefficient (MIC).
    //!
    //! \param[in] target Extracts the column value with which to compute MIC.
    //! \param[in] frame The data frame for which to compute the column MICs.
    //! \param[in] columnMask A mask of the columns to include.
    //! \return A collection containing the MIC of each metric valued column with
    //! \p target. The collectionis indexed by column.
    static TDoubleVec micWithColumn(const CColumnValue& target,
                                    const core::CDataFrame& frame,
                                    TSizeVec columnMask);

    //! Check if a data frame value is missing.
    static bool isMissing(double value);

private:
    static TSizeDoublePrVecVec
    categoryMicWithColumnDataFrameInMemory(const CColumnValue& target,
                                           std::size_t numberThreads,
                                           const core::CDataFrame& frame,
                                           const TSizeVec& columnMask,
                                           std::size_t numberSamples,
                                           double minimumFrequency);
    static TSizeDoublePrVecVec
    categoryMicWithColumnDataFrameOnDisk(const CColumnValue& target,
                                         std::size_t numberThreads,
                                         const core::CDataFrame& frame,
                                         const TSizeVec& columnMask,
                                         std::size_t numberSamples,
                                         double minimumFrequency);
    static TDoubleVec micWithColumnDataFrameInMemory(const CColumnValue& target,
                                                     const core::CDataFrame& frame,
                                                     const TSizeVec& columnMask,
                                                     std::size_t numberSamples);
    static TDoubleVec micWithColumnDataFrameOnDisk(const CColumnValue& target,
                                                   const core::CDataFrame& frame,
                                                   const TSizeVec& columnMask,
                                                   std::size_t numberSamples);
    static void removeMetricColumns(const core::CDataFrame& frame, TSizeVec& columnMask);
    static void removeCategoricalColumns(const core::CDataFrame& frame, TSizeVec& columnMask);
    static double unitWeight(const TRowRef&);
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameUtils_h
