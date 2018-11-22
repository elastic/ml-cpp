/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CModelPlotData_h
#define INCLUDED_ml_model_CModelPlotData_h

#include <model/ImportExport.h>

#include <model/ModelTypes.h>

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

namespace ml {
namespace model {

//! \brief Data necessary to create a model plot
class MODEL_EXPORT CModelPlotData {
public:
    using TStrDoublePr = std::pair<std::string, double>;
    using TStrDoublePrVec = std::vector<TStrDoublePr>;

public:
    struct MODEL_EXPORT SByFieldData {
        SByFieldData();
        SByFieldData(double lowerBound, double upperBound, double median);

        void addValue(const std::string& personName, double value);

        double s_LowerBound;
        double s_UpperBound;
        double s_Median;

        // Change to vector of pair str->double
        TStrDoublePrVec s_ValuesPerOverField;
    };

public:
    using TStrByFieldDataUMap = boost::unordered_map<std::string, SByFieldData>;
    using TFeatureStrByFieldDataUMapPr = std::pair<model_t::EFeature, TStrByFieldDataUMap>;
    using TFeatureStrByFieldDataUMapUMap =
        boost::unordered_map<model_t::EFeature, TStrByFieldDataUMap>;
    using TIntStrByFieldDataUMapUMap = boost::unordered_map<int, TStrByFieldDataUMap>;
    using TFeatureStrByFieldDataUMapUMapCItr = TFeatureStrByFieldDataUMapUMap::const_iterator;

public:
    CModelPlotData();
    CModelPlotData(core_t::TTime time,
                   const std::string& partitionFieldName,
                   const std::string& partitionFieldValue,
                   const std::string& overFieldName,
                   const std::string& byFieldName,
                   core_t::TTime bucketSpan,
                   int detectorIndex);
    TFeatureStrByFieldDataUMapUMapCItr begin() const;
    TFeatureStrByFieldDataUMapUMapCItr end() const;
    SByFieldData& get(const model_t::EFeature& feature, const std::string& byFieldValue);
    const std::string& partitionFieldName() const;
    const std::string& partitionFieldValue() const;
    const std::string& overFieldName() const;
    const std::string& byFieldName() const;
    core_t::TTime time() const;
    core_t::TTime bucketSpan() const;
    int detectorIndex() const;
    std::string print() const;

private:
    TFeatureStrByFieldDataUMapUMap m_DataPerFeature;
    core_t::TTime m_Time;
    std::string m_PartitionFieldName;
    std::string m_PartitionFieldValue;
    std::string m_OverFieldName;
    std::string m_ByFieldName;
    core_t::TTime m_BucketSpan;
    int m_DetectorIndex;
};
}
}

#endif // INCLUDED_ml_model_CModelPlotData_h
