/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CFeatureData_h
#define INCLUDED_ml_model_CFeatureData_h

#include <core/CMemoryUsage.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <model/CSample.h>
#include <model/ImportExport.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {

//! \brief Manages the indexing for the feature values in the statistics
//! vectors passed from data gatherers to the model classes.
class MODEL_EXPORT CFeatureDataIndexing {
public:
    using TSizeVec = std::vector<std::size_t>;

public:
    //! Get the indices of the actual feature value(s) in the feature
    //! data vector.
    static const TSizeVec& valueIndices(std::size_t dimension);
};

//! \brief The data for an event rate series feature.
struct MODEL_EXPORT SEventRateFeatureData {
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TStrCRef = std::reference_wrapper<const std::string>;
    using TDouble1VecDoublePr = std::pair<TDouble1Vec, double>;
    using TStrCRefDouble1VecDoublePrPr = std::pair<TStrCRef, TDouble1VecDoublePr>;
    using TStrCRefDouble1VecDoublePrPrVec = std::vector<TStrCRefDouble1VecDoublePrPr>;
    using TStrCRefDouble1VecDoublePrPrVecVec = std::vector<TStrCRefDouble1VecDoublePrPrVec>;

    SEventRateFeatureData(uint64_t count);

    //! Efficiently swap the contents of this and \p other.
    void swap(SEventRateFeatureData& other);

    //! Print the data for debug.
    std::string print() const;

    //! Get the memory usage of this component.
    std::size_t memoryUsage() const;

    //! Get the memory usage of this component in a tree structure.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    uint64_t s_Count;
    TStrCRefDouble1VecDoublePrPrVecVec s_InfluenceValues;
};

//! \brief The data for a metric series feature.
struct MODEL_EXPORT SMetricFeatureData {
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TOptionalSample = boost::optional<CSample>;
    using TSampleVec = std::vector<CSample>;
    using TStrCRef = std::reference_wrapper<const std::string>;
    using TDouble1VecDoublePr = std::pair<TDouble1Vec, double>;
    using TStrCRefDouble1VecDoublePrPr = std::pair<TStrCRef, TDouble1VecDoublePr>;
    using TStrCRefDouble1VecDoublePrPrVec = std::vector<TStrCRefDouble1VecDoublePrPr>;
    using TStrCRefDouble1VecDoublePrPrVecVec = std::vector<TStrCRefDouble1VecDoublePrPrVec>;

    SMetricFeatureData(core_t::TTime bucketTime,
                       const TDouble1Vec& bucketValue,
                       double bucketVarianceScale,
                       double bucketCount,
                       TStrCRefDouble1VecDoublePrPrVecVec& influenceValues,
                       bool isInteger,
                       bool isNonNegative,
                       const TSampleVec& samples);

    SMetricFeatureData(bool isInteger, bool isNonNegative, const TSampleVec& samples);

    //! Print the data for debug.
    std::string print() const;

    //! Get the memory usage of this component.
    std::size_t memoryUsage() const;

    //! Get the memory usage of this component in a tree structure.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! The bucket value.
    TOptionalSample s_BucketValue;
    //! The influencing values and counts.
    TStrCRefDouble1VecDoublePrPrVecVec s_InfluenceValues;
    //! True if all the samples are integer.
    bool s_IsInteger;
    //! True if all the samples are non-negative.
    bool s_IsNonNegative;
    //! The samples.
    TSampleVec s_Samples;
};
}
}

#endif // INCLUDED_ml_model_CFeatureData_h
