/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelMetadata.h>

#include <cmath>

namespace ml {
namespace api {

void CInferenceModelMetadata::write(TRapidJsonWriter& writer) const {
    this->writeTotalFeatureImportance(writer);
    this->writeFeatureImportanceBaseline(writer);
}

void CInferenceModelMetadata::writeTotalFeatureImportance(TRapidJsonWriter& writer) const {
    writer.Key(JSON_TOTAL_FEATURE_IMPORTANCE_TAG);
    writer.StartArray();
    for (const auto& item : m_TotalShapValuesMean) {
        writer.StartObject();
        writer.Key(JSON_FEATURE_NAME_TAG);
        writer.String(m_ColumnNames[item.first]);
        auto meanFeatureImportance = maths::CBasicStatistics::mean(item.second);
        const auto& minMaxFeatureImportance = m_TotalShapValuesMinMax.at(item.first);
        if (meanFeatureImportance.size() == 1 && m_ClassValues.empty()) {
            // Regression
            writer.Key(JSON_IMPORTANCE_TAG);
            writer.StartObject();
            writer.Key(JSON_MEAN_MAGNITUDE_TAG);
            writer.Double(meanFeatureImportance[0]);
            writer.Key(JSON_MIN_TAG);
            writer.Double(minMaxFeatureImportance[0].min());
            writer.Key(JSON_MAX_TAG);
            writer.Double(minMaxFeatureImportance[0].max());
            writer.EndObject();
        } else if (meanFeatureImportance.size() == 1 && m_ClassValues.empty() == false) {
            // Binary classification
            // since we track the min/max only for one class, this will make the range more robust
            double minimum{std::min(minMaxFeatureImportance[0].min(),
                                    -minMaxFeatureImportance[0].max())};
            double maximum{-minimum};
            writer.Key(JSON_CLASSES_TAG);
            writer.StartArray();
            for (std::size_t j = 0; j < m_ClassValues.size(); ++j) {
                writer.StartObject();
                writer.Key(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.Key(JSON_IMPORTANCE_TAG);
                writer.StartObject();
                writer.Key(JSON_MEAN_MAGNITUDE_TAG);
                // mean magnitude is the same for both classes
                writer.Double(meanFeatureImportance[0]);
                writer.Key(JSON_MIN_TAG);
                writer.Double(minimum);
                writer.Key(JSON_MAX_TAG);
                writer.Double(maximum);
                writer.EndObject();
                writer.EndObject();
            }
            writer.EndArray();
        } else {
            // Multiclass classification
            writer.Key(JSON_CLASSES_TAG);
            writer.StartArray();
            for (std::size_t j = 0;
                 j < static_cast<std::size_t>(meanFeatureImportance.size()) &&
                 j < m_ClassValues.size();
                 ++j) {
                writer.StartObject();
                writer.Key(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.Key(JSON_IMPORTANCE_TAG);
                writer.StartObject();
                writer.Key(JSON_MEAN_MAGNITUDE_TAG);
                writer.Double(meanFeatureImportance[j]);
                writer.Key(JSON_MIN_TAG);
                writer.Double(minMaxFeatureImportance[j].min());
                writer.Key(JSON_MAX_TAG);
                writer.Double(minMaxFeatureImportance[j].max());
                writer.EndObject();
                writer.EndObject();
            }
            writer.EndArray();
        }
        writer.EndObject();
    }
    writer.EndArray();
}

void CInferenceModelMetadata::writeFeatureImportanceBaseline(TRapidJsonWriter& writer) const {
    if (m_ShapBaseline) {
        writer.Key(JSON_FEATURE_IMPORTANCE_BASELINE_TAG);
        writer.StartObject();
        if (m_ShapBaseline->size() == 1 && m_ClassValues.empty()) {
            // Regression
            writer.Key(JSON_BASELINE_TAG);
            writer.Double(m_ShapBaseline.get()(0));
        } else if (m_ShapBaseline->size() == 1 && m_ClassValues.empty() == false) {
            // Binary classification
            writer.Key(JSON_CLASSES_TAG);
            writer.StartArray();
            for (std::size_t j = 0; j < m_ClassValues.size(); ++j) {
                writer.StartObject();
                writer.Key(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.Key(JSON_BASELINE_TAG);
                if (j == 1) {
                    writer.Double(m_ShapBaseline.get()(0));
                } else {
                    writer.Double(-m_ShapBaseline.get()(0));
                }
                writer.EndObject();
            }

            writer.EndArray();

        } else {
            // Multiclass classification
            writer.Key(JSON_CLASSES_TAG);
            writer.StartArray();
            for (std::size_t j = 0; j < static_cast<std::size_t>(m_ShapBaseline->size()) &&
                                    j < m_ClassValues.size();
                 ++j) {
                writer.StartObject();
                writer.Key(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.Key(JSON_BASELINE_TAG);
                writer.Double(m_ShapBaseline.get()(j));
                writer.EndObject();
            }
            writer.EndArray();
        }
        writer.EndObject();
    }
}

const std::string& CInferenceModelMetadata::typeString() const {
    return JSON_MODEL_METADATA_TAG;
}

void CInferenceModelMetadata::columnNames(const TStrVec& columnNames) {
    m_ColumnNames = columnNames;
}

void CInferenceModelMetadata::classValues(const TStrVec& classValues) {
    m_ClassValues = classValues;
}

void CInferenceModelMetadata::predictionFieldTypeResolverWriter(
    const TPredictionFieldTypeResolverWriter& resolverWriter) {
    m_PredictionFieldTypeResolverWriter = resolverWriter;
}

void CInferenceModelMetadata::addToFeatureImportance(std::size_t i, const TVector& values) {
    auto& meanVector = m_TotalShapValuesMean
                           .emplace(std::make_pair(i, TMeanAccumulator(values.size())))
                           .first->second;
    auto& minMaxVector =
        m_TotalShapValuesMinMax
            .emplace(std::make_pair(i, TMinMaxAccumulator(values.size())))
            .first->second;
    for (std::size_t j = 0; j < minMaxVector.size(); ++j) {
        meanVector[j].add(std::fabs(values[j]));
        minMaxVector[j].add(values[j]);
    }
}

void CInferenceModelMetadata::featureImportanceBaseline(TVector&& baseline) {
    m_ShapBaseline = baseline;
}

// clang-format off
const std::string CInferenceModelMetadata::JSON_BASELINE_TAG{"baseline"};
const std::string CInferenceModelMetadata::JSON_FEATURE_IMPORTANCE_BASELINE_TAG{"feature_importance_baseline"};
const std::string CInferenceModelMetadata::JSON_CLASS_NAME_TAG{"class_name"};
const std::string CInferenceModelMetadata::JSON_CLASSES_TAG{"classes"};
const std::string CInferenceModelMetadata::JSON_FEATURE_NAME_TAG{"feature_name"};
const std::string CInferenceModelMetadata::JSON_IMPORTANCE_TAG{"importance"};
const std::string CInferenceModelMetadata::JSON_MAX_TAG{"max"};
const std::string CInferenceModelMetadata::JSON_MEAN_MAGNITUDE_TAG{"mean_magnitude"};
const std::string CInferenceModelMetadata::JSON_MIN_TAG{"min"};
const std::string CInferenceModelMetadata::JSON_MODEL_METADATA_TAG{"model_metadata"};
const std::string CInferenceModelMetadata::JSON_TOTAL_FEATURE_IMPORTANCE_TAG{"total_feature_importance"};
// clang-format on
}
}
