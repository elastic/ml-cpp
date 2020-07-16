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

class API_EXPORT CInferenceModelMetadata : public CSerializableToJsonDocument {
public:
    using TVector = maths::CDenseVector<double>;

public:
    CInferenceModelMetadata() : m_TotalShapValues(){};
    void addToJsonDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    void columnNames(const std::vector<std::string>& columnNames);
    const std::string& typeString() const;
    void addToFeatureImportance(std::size_t i, const TVector& values);

private:
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<TVector>::TAccumulator;
    using TTotalShapValues = std::unordered_map<std::size_t, TMeanAccumulator>;

private:
    TTotalShapValues m_TotalShapValues;
    std::vector<std::string> m_ColumnNames;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelMetadata_h
