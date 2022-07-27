/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CFeatureData.h>

#include <core/CLogger.h>
#include <core/CMemoryDef.h>

#include <boost/iterator/counting_iterator.hpp>

#include <vector>

namespace ml {
namespace model {

namespace {

using TSizeVec = std::vector<std::size_t>;

//! Get the sequence [0, N).
template<std::size_t N>
const TSizeVec& sequence() {
    static const TSizeVec result(boost::counting_iterator<std::size_t>(0),
                                 boost::counting_iterator<std::size_t>(N));
    return result;
}
}

////// CFeatureDataIndexing //////

const TSizeVec& CFeatureDataIndexing::valueIndices(std::size_t dimension) {
    switch (dimension) {
    case 1:
        return sequence<1>();
    case 2:
        return sequence<2>();
    case 3:
        return sequence<3>();
    case 4:
        return sequence<4>();
    case 5:
        return sequence<5>();
    case 6:
        return sequence<6>();
    case 7:
        return sequence<7>();
    case 8:
        return sequence<8>();
    case 9:
        return sequence<9>();
    default:
        LOG_ERROR(<< "Unsupported dimension " << dimension);
        break;
    }
    static const TSizeVec EMPTY;
    return EMPTY;
}

////// SEventRateFeatureData //////

SEventRateFeatureData::SEventRateFeatureData(std::uint64_t count)
    : s_Count(count) {
}

SEventRateFeatureData::~SEventRateFeatureData() = default;

void SEventRateFeatureData::swap(SEventRateFeatureData& other) {
    std::swap(s_Count, other.s_Count);
    s_InfluenceValues.swap(other.s_InfluenceValues);
}

std::string SEventRateFeatureData::print() const {
    std::ostringstream result;
    result << s_Count;
    if (!s_InfluenceValues.empty()) {
        result << ", " << core::CContainerPrinter::print(s_InfluenceValues);
    }
    return result.str();
}

std::size_t SEventRateFeatureData::memoryUsage() const {
    std::size_t mem = sizeof(*this);
    mem += core::CMemory::dynamicSize(s_InfluenceValues);
    return mem;
}

void SEventRateFeatureData::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("SMetricFeatureData", sizeof(*this));
    core::CMemoryDebug::dynamicSize("s_InfluenceValues", s_InfluenceValues, mem);
}

////// SMetricFeatureData //////

SMetricFeatureData::SMetricFeatureData(core_t::TTime bucketTime,
                                       const TDouble1Vec& bucketValue,
                                       double bucketVarianceScale,
                                       double bucketCount,
                                       TStrCRefDouble1VecDoublePrPrVecVec& influenceValues,
                                       bool isInteger,
                                       bool isNonNegative,
                                       const TSampleVec& samples)
    : s_BucketValue(std::in_place, bucketTime, bucketValue, bucketVarianceScale, bucketCount),
      s_IsInteger(isInteger), s_IsNonNegative(isNonNegative), s_Samples(samples) {
    s_InfluenceValues.swap(influenceValues);
}

SMetricFeatureData::SMetricFeatureData(bool isInteger, bool isNonNegative, const TSampleVec& samples)
    : s_IsInteger(isInteger), s_IsNonNegative(isNonNegative), s_Samples(samples) {
}

SMetricFeatureData::~SMetricFeatureData() = default;

std::string SMetricFeatureData::print() const {
    std::ostringstream result;
    result << "value = " << core::CContainerPrinter::print(s_BucketValue)
           << ", is integer " << s_IsInteger << ", is non-negative " << s_IsNonNegative
           << ", samples = " << core::CContainerPrinter::print(s_Samples);
    return result.str();
}

std::size_t SMetricFeatureData::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(s_BucketValue);
    mem += core::CMemory::dynamicSize(s_InfluenceValues);
    mem += core::CMemory::dynamicSize(s_Samples);
    return mem;
}

void SMetricFeatureData::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("SMetricFeatureData");
    core::CMemoryDebug::dynamicSize("s_BucketValue", s_BucketValue, mem);
    core::CMemoryDebug::dynamicSize("s_InfluenceValues", s_InfluenceValues, mem);
    core::CMemoryDebug::dynamicSize("s_Samples", s_Samples, mem);
}
}
}
