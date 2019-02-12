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

namespace ml {
namespace maths {
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

//! \brief A collection of basic utilities applied to a data frame.
class MATHS_EXPORT CDataFrameUtils : private core::CNonInstantiatable {
public:
    //! Convert a row of the data frame to a specified vector type.
    template<typename VECTOR>
    static VECTOR rowTo(const core::CDataFrame::TRowRef& row) {
        return data_frame_utils_detail::SRowTo<VECTOR>::dispatch(row);
    }

    //! Subtract the mean and divide each column value by its standard deviation.
    static bool standardizeColumns(std::size_t numberThreads, core::CDataFrame& frame);
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameUtils_h
