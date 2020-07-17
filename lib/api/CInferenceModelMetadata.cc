/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelMetadata.h>
#include <rapidjson/document.h>

namespace ml {
namespace api {

namespace {
const std::string JSON_MODEL_METADATA_TAG{"model_metadata"};
const std::string JSON_FIELD_NAME_TAG{"field_name"};
const std::string JSON_IMPORTANCE_TAG{"importance"};
const std::string JSON_TOTAL_FEATURE_IMPORTANCE_TAG{"total_feature_importance"};
const std::string JSON_FOOBAR_TAG{"foobar"};
}

void CInferenceModelMetadata::addToJsonDocument(rapidjson::Value& parentObject,
                                                TRapidJsonWriter& writer) const {
    auto array = writer.makeArray();
    for (const auto& item : m_TotalShapValues) {
        auto jsonItem = writer.makeObject();
        rapidjson::Value s;
        s = rapidjson::StringRef(m_ColumnNames[item.first].c_str(),
                                 m_ColumnNames[item.first].size());
        writer.addMember(JSON_FIELD_NAME_TAG, s, jsonItem);
        writer.addMember(
            JSON_IMPORTANCE_TAG,
            rapidjson::Value(maths::CBasicStatistics::mean(item.second)[0]).Move(), jsonItem),
            array.PushBack(jsonItem, writer.getRawAllocator());

        // writer.StartObject();
        // writer.Key(FEATURE_NAME_FIELD_NAME);
        // writer.String(featureImportance->columnNames()[item.first]);
        // writer.Key(IMPORTANCE_FIELD_NAME);
        // writer.Double(maths::CBasicStatistics::mean(item.second)[0]);
        // writer.EndObject();
        LOG_DEBUG(<< "Count: " << maths::CBasicStatistics::count(item.second));
    }
    writer.addMember(JSON_TOTAL_FEATURE_IMPORTANCE_TAG, array, parentObject);
}

const std::string& CInferenceModelMetadata::typeString() const {
    return JSON_MODEL_METADATA_TAG;
}

void CInferenceModelMetadata::columnNames(const std::vector<std::string>& columnNames) {
    m_ColumnNames = columnNames;
}

void CInferenceModelMetadata::addToFeatureImportance(std::size_t i, const TVector& values) {
    m_TotalShapValues.emplace(std::make_pair(i, TVector::Zero(values.size())))
        .first->second.add(values.cwiseAbs());
}
}
}