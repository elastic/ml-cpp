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
    using TSizeVec = std::vector<std::size_t>;
    using TRowRef = core::CDataFrame::TRowRef;
    using TWeightFunction = std::function<double(TRowRef)>;
    using TQuantileSketchVec = std::vector<CQuantileSketch>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;

public:
    //! Convert a row of the data frame to a specified vector type.
    template<typename VECTOR>
    static VECTOR rowTo(const core::CDataFrame::TRowRef& row) {
        return data_frame_utils_detail::SRowTo<VECTOR>::dispatch(row);
    }

    //! Subtract the mean and divide each column value by its standard deviation.
    //!
    //! \param[in] numberThreads The number of threads available.
    //! \param[in] frame The data frame whose columns are to be standardized.
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

    //! Check if a data frame value is missing.
    static bool isMissing(double value);

private:
    static double unitWeight(const TRowRef&);
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameUtils_h
