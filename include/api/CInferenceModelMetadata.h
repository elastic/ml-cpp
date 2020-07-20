/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelMetadata_h
#define INCLUDED_ml_api_CInferenceModelMetadata_h

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraEigen.h>

#include <api/CInferenceModelDefinition.h>
#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

class API_EXPORT CInferenceModelMetadata {
public:
    using TVector = maths::CDenseVector<double>;
    using TStrVec = std::vector<std::string>;
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    void write(TRapidJsonWriter& writer) const;
    void columnNames(const TStrVec& columnNames);
    void classValues(const TStrVec& classValues);

    const std::string& typeString() const;
    void addToFeatureImportance(std::size_t i, const TVector& values);

private:
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<TVector>::TAccumulator;
    using TMinMaxAccumulator = maths::CBasicStatistics::CMinMax<TVector>;
    using TTotalShapValuesMeanVar = std::unordered_map<std::size_t, TMeanVarAccumulator>;
    using TTotalShapValuesMinMax = std::unordered_map<std::size_t, TMinMaxAccumulator>;

private:
    void writeTotalFeatureImportance(TRapidJsonWriter& writer) const;

private:
    TTotalShapValuesMeanVar m_TotalShapValuesMeanVar;
    TTotalShapValuesMinMax m_TotalShapValuesMinMax;
    TStrVec m_ColumnNames;
    TStrVec m_ClassValues;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelMetadata_h
