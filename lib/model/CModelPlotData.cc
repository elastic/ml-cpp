/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CModelPlotData.h>

namespace ml {
namespace model {

CModelPlotData::CModelPlotData() : m_Time(0) {
}

CModelPlotData::CModelPlotData(core_t::TTime time,
                               const std::string& partitionFieldName,
                               const std::string& partitionFieldValue,
                               const std::string& overFieldName,
                               const std::string& byFieldName,
                               core_t::TTime bucketSpan,
                               int detectorIndex)
    : m_Time(time), m_PartitionFieldName(partitionFieldName),
      m_PartitionFieldValue(partitionFieldValue),
      m_OverFieldName(overFieldName), m_ByFieldName(byFieldName),
      m_BucketSpan(bucketSpan), m_DetectorIndex(detectorIndex) {
}

CModelPlotData::SByFieldData::SByFieldData()
    : s_LowerBound(0.0), s_UpperBound(0.0), s_Median(0.0), s_ValuesPerOverField() {
}

CModelPlotData::SByFieldData::SByFieldData(double lowerBound, double upperBound, double median)
    : s_LowerBound(lowerBound), s_UpperBound(upperBound), s_Median(median),
      s_ValuesPerOverField() {
}

const std::string& CModelPlotData::partitionFieldName() const {
    return m_PartitionFieldName;
}

const std::string& CModelPlotData::partitionFieldValue() const {
    return m_PartitionFieldValue;
}

const std::string& CModelPlotData::overFieldName() const {
    return m_OverFieldName;
}

const std::string& CModelPlotData::byFieldName() const {
    return m_ByFieldName;
}

core_t::TTime CModelPlotData::time() const {
    return m_Time;
}

core_t::TTime CModelPlotData::bucketSpan() const {
    return m_BucketSpan;
}

int CModelPlotData::detectorIndex() const {
    return m_DetectorIndex;
}

void CModelPlotData::SByFieldData::addValue(const std::string& personName, double value) {
    s_ValuesPerOverField.emplace_back(personName, value);
}

CModelPlotData::TFeatureStrByFieldDataUMapUMapCItr CModelPlotData::begin() const {
    return m_DataPerFeature.begin();
}

CModelPlotData::TFeatureStrByFieldDataUMapUMapCItr CModelPlotData::end() const {
    return m_DataPerFeature.end();
}

CModelPlotData::SByFieldData& CModelPlotData::get(const model_t::EFeature& feature,
                                                  const std::string& byFieldValue) {
    // note: This creates/inserts! elements and returns a reference for writing
    // data insert happens here
    return m_DataPerFeature[feature][byFieldValue];
}

std::string CModelPlotData::print() const {
    return "nothing";
}
}
}
