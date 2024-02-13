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
#include <api/CInferenceModelMetadata.h>

#include <core/CLogger.h>

#include <api/ApiTypes.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <maths/analytics/CBoostedTreeHyperparameters.h>

#include <cmath>
#include <cstdint>

namespace ml {
namespace api {

void CInferenceModelMetadata::write(TBoostJsonWriter& writer) const {
    this->writeTotalFeatureImportance(writer);
    this->writeFeatureImportanceBaseline(writer);
    this->writeHyperparameterImportance(writer);
    LOG_TRACE(<< "Number data summarization rows " << m_NumDataSummarizationRows);
    if (m_NumDataSummarizationRows > 0) {
        // Only output if data summarization fraction was specified.
        this->writeTrainProperties(writer);
        this->writeDataSummarization(writer);
    }
}

void CInferenceModelMetadata::writeTotalFeatureImportance(TBoostJsonWriter& writer) const {
    writer.onKey(JSON_TOTAL_FEATURE_IMPORTANCE_TAG);
    writer.onArrayBegin();
    for (const auto& item : m_TotalShapValuesMean) {
        writer.onObjectBegin();
        writer.onKey(JSON_FEATURE_NAME_TAG);
        writer.onString(m_ColumnNames[item.first]);
        auto meanFeatureImportance = maths::common::CBasicStatistics::mean(item.second);
        const auto& minMaxFeatureImportance = m_TotalShapValuesMinMax.at(item.first);
        if (meanFeatureImportance.size() == 1 && m_ClassValues.empty()) {
            // Regression.
            writer.onKey(JSON_IMPORTANCE_TAG);
            writer.onObjectBegin();
            writer.onKey(JSON_MEAN_MAGNITUDE_TAG);
            writer.onDouble(meanFeatureImportance[0]);
            writer.onKey(JSON_MIN_TAG);
            writer.onDouble(minMaxFeatureImportance[0].min());
            writer.onKey(JSON_MAX_TAG);
            writer.onDouble(minMaxFeatureImportance[0].max());
            writer.onObjectEnd();
        } else if (meanFeatureImportance.size() == 1 && m_ClassValues.empty() == false) {
            // Binary classification.
            // Since we track the min/max only for one class, this will make the range more robust.
            double minimum{std::min(minMaxFeatureImportance[0].min(),
                                    -minMaxFeatureImportance[0].max())};
            double maximum{-minimum};
            writer.onKey(JSON_CLASSES_TAG);
            writer.onArrayBegin();
            for (const auto& classValue : m_ClassValues) {
                writer.onObjectBegin();
                writer.onKey(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(classValue, writer);
                writer.onKey(JSON_IMPORTANCE_TAG);
                writer.onObjectBegin();
                writer.onKey(JSON_MEAN_MAGNITUDE_TAG);
                // mean magnitude is the same for both classes
                writer.onDouble(meanFeatureImportance[0]);
                writer.onKey(JSON_MIN_TAG);
                writer.onDouble(minimum);
                writer.onKey(JSON_MAX_TAG);
                writer.onDouble(maximum);
                writer.onObjectEnd();
                writer.onObjectEnd();
            }
            writer.onArrayEnd();
        } else {
            // Multiclass classification.
            writer.onKey(JSON_CLASSES_TAG);
            writer.onArrayBegin();
            for (std::size_t j = 0;
                 j < static_cast<std::size_t>(meanFeatureImportance.size()) &&
                 j < m_ClassValues.size();
                 ++j) {
                writer.onObjectBegin();
                writer.onKey(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.onKey(JSON_IMPORTANCE_TAG);
                writer.onObjectBegin();
                writer.onKey(JSON_MEAN_MAGNITUDE_TAG);
                writer.onDouble(meanFeatureImportance[j]);
                writer.onKey(JSON_MIN_TAG);
                writer.onDouble(minMaxFeatureImportance[j].min());
                writer.onKey(JSON_MAX_TAG);
                writer.onDouble(minMaxFeatureImportance[j].max());
                writer.onObjectEnd();
                writer.onObjectEnd();
            }
            writer.onArrayEnd();
        }
        writer.onObjectEnd();
    }
    writer.onArrayEnd();
}

void CInferenceModelMetadata::writeFeatureImportanceBaseline(TBoostJsonWriter& writer) const {
    if (m_ShapBaseline) {
        writer.onKey(JSON_FEATURE_IMPORTANCE_BASELINE_TAG);
        writer.onObjectBegin();
        if (m_ShapBaseline->size() == 1 && m_ClassValues.empty()) {
            // Regression.
            writer.onKey(JSON_BASELINE_TAG);
            writer.onDouble((*m_ShapBaseline)(0));
        } else if (m_ShapBaseline->size() == 1 && m_ClassValues.empty() == false) {
            // Binary classification.
            writer.onKey(JSON_CLASSES_TAG);
            writer.onArrayBegin();
            for (std::size_t j = 0; j < m_ClassValues.size(); ++j) {
                writer.onObjectBegin();
                writer.onKey(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.onKey(JSON_BASELINE_TAG);
                if (j == 1) {
                    writer.onDouble((*m_ShapBaseline)(0));
                } else {
                    writer.onDouble(-(*m_ShapBaseline)(0));
                }
                writer.onObjectEnd();
            }
            writer.onArrayEnd();
        } else {
            // Multiclass classification.
            writer.onKey(JSON_CLASSES_TAG);
            writer.onArrayBegin();
            for (std::size_t j = 0; j < static_cast<std::size_t>(m_ShapBaseline->size()) &&
                                    j < m_ClassValues.size();
                 ++j) {
                writer.onObjectBegin();
                writer.onKey(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(m_ClassValues[j], writer);
                writer.onKey(JSON_BASELINE_TAG);
                writer.onDouble((*m_ShapBaseline)(j));
                writer.onObjectEnd();
            }
            writer.onArrayEnd();
        }
        writer.onObjectEnd();
    }
}

void CInferenceModelMetadata::writeHyperparameterImportance(TBoostJsonWriter& writer) const {
    writer.onKey(JSON_HYPERPARAMETERS_TAG);
    writer.onArrayBegin();
    for (const auto& item : m_HyperparameterImportance) {
        writer.onObjectBegin();
        writer.onKey(JSON_HYPERPARAMETER_NAME_TAG);
        writer.onString(item.s_HyperparameterName);
        writer.onKey(JSON_HYPERPARAMETER_VALUE_TAG);
        switch (item.s_Type) {
        case SHyperparameterImportance::E_Double:
            writer.onDouble(item.s_Value);
            break;
        case SHyperparameterImportance::E_Uint64:
            writer.onUint64(static_cast<std::uint64_t>(item.s_Value));
            break;
        }
        if (item.s_Supplied == false) {
            writer.onKey(JSON_ABSOLUTE_IMPORTANCE_TAG);
            writer.onDouble(item.s_AbsoluteImportance);
            writer.onKey(JSON_RELATIVE_IMPORTANCE_TAG);
            writer.onDouble(item.s_RelativeImportance);
        }
        writer.onKey(JSON_HYPERPARAMETER_SUPPLIED_TAG);
        writer.onBool(item.s_Supplied);
        writer.onObjectEnd();
    }
    writer.onArrayEnd();
}

void CInferenceModelMetadata::writeDataSummarization(TBoostJsonWriter& writer) const {
    if (m_NumDataSummarizationRows > 0) {
        // Only output if the data summarization fraction was specified.
        writer.onKey(JSON_DATA_SUMMARIZATION_TAG);
        writer.onObjectBegin();
        writer.onKey(JSON_NUM_DATA_SUMMARIZATION_ROWS_TAG);
        writer.onUint64(m_NumDataSummarizationRows);
        writer.onObjectEnd();
    }
}

void CInferenceModelMetadata::writeTrainProperties(TBoostJsonWriter& writer) const {
    if (m_NumTrainRows > 0) {
        writer.onKey(JSON_TRAIN_PROPERTIES_TAG);
        writer.onObjectBegin();
        writer.onKey(JSON_NUM_TRAIN_ROWS_TAG);
        writer.onUint64(m_NumTrainRows);
        writer.onKey(JSON_LOSS_GAP_TAG);
        writer.onDouble(m_LossGap);
        writer.onKey(JSON_TRAINED_MODEL_MEMORY_USAGE_TAG);
        writer.onUint64(m_TrainedModelMemoryUsage);
        writer.onObjectEnd();
    }
}

const std::string& CInferenceModelMetadata::typeString() {
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

void CInferenceModelMetadata::hyperparameterImportance(
    const maths::analytics::CBoostedTree::THyperparameterImportanceVec& hyperparameterImportance) {
    m_HyperparameterImportance.clear();
    m_HyperparameterImportance.reserve(hyperparameterImportance.size());
    for (const auto& item : hyperparameterImportance) {
        std::string hyperparameterName;
        switch (item.s_Hyperparameter) {
        // Train + (maybe incremental train) hyperparameters.
        case maths::analytics::E_Alpha:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::ALPHA;
            break;
        case maths::analytics::E_DownsampleFactor:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_FACTOR;
            break;
        case maths::analytics::E_Eta:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::ETA;
            break;
        case maths::analytics::E_EtaGrowthRatePerTree:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::ETA_GROWTH_RATE_PER_TREE;
            break;
        case maths::analytics::E_FeatureBagFraction:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION;
            break;
        case maths::analytics::E_Gamma:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::GAMMA;
            break;
        case maths::analytics::E_Lambda:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::LAMBDA;
            break;
        case maths::analytics::E_SoftTreeDepthLimit:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT;
            break;
        case maths::analytics::E_SoftTreeDepthTolerance:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE;
            break;

        // Not tuned via Bayesian Optimisation.
        case maths::analytics::E_MaximumNumberTrees:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::MAX_TREES;
            break;

        // Incremental train hyperparameters.
        case maths::analytics::E_PredictionChangeCost:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::PREDICTION_CHANGE_COST;
            break;
        case maths::analytics::E_RetrainedTreeEta:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::RETRAINED_TREE_ETA;
            break;
        case maths::analytics::E_TreeTopologyChangePenalty:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::TREE_TOPOLOGY_CHANGE_PENALTY;
            break;
        }
        double absoluteImportance{(std::fabs(item.s_AbsoluteImportance) < 1e-8)
                                      ? 0.0
                                      : item.s_AbsoluteImportance};
        double relativeImportance{(std::fabs(item.s_RelativeImportance) < 1e-8)
                                      ? 0.0
                                      : item.s_RelativeImportance};
        m_HyperparameterImportance.push_back(
            {hyperparameterName, item.s_Value, absoluteImportance, relativeImportance,
             item.s_Supplied, static_cast<SHyperparameterImportance::EType>(item.s_Type)});
    }
    std::sort(m_HyperparameterImportance.begin(),
              m_HyperparameterImportance.end(), [](const auto& a, const auto& b) {
                  return a.s_AbsoluteImportance > b.s_AbsoluteImportance;
              });
}

void CInferenceModelMetadata::numTrainRows(std::size_t numRows) {
    m_NumTrainRows = numRows;
}

void CInferenceModelMetadata::lossGap(double lossGap) {
    m_LossGap = lossGap;
}

void CInferenceModelMetadata::numDataSummarizationRows(std::size_t numRows) {
    m_NumDataSummarizationRows = numRows;
}

void CInferenceModelMetadata::trainedModelMemoryUsage(std::size_t memoryUsage) {
    m_TrainedModelMemoryUsage = memoryUsage;
}

// clang-format off
const std::string CInferenceModelMetadata::JSON_ABSOLUTE_IMPORTANCE_TAG{"absolute_importance"};
const std::string CInferenceModelMetadata::JSON_BASELINE_TAG{"baseline"};
const std::string CInferenceModelMetadata::JSON_CLASS_NAME_TAG{"class_name"};
const std::string CInferenceModelMetadata::JSON_CLASSES_TAG{"classes"};
const std::string CInferenceModelMetadata::JSON_DATA_SUMMARIZATION_TAG{"data_summarization"};
const std::string CInferenceModelMetadata::JSON_NUM_DATA_SUMMARIZATION_ROWS_TAG{"num_rows"};
const std::string CInferenceModelMetadata::JSON_FEATURE_IMPORTANCE_BASELINE_TAG{"feature_importance_baseline"};
const std::string CInferenceModelMetadata::JSON_FEATURE_NAME_TAG{"feature_name"};
const std::string CInferenceModelMetadata::JSON_HYPERPARAMETERS_TAG{"hyperparameters"};
const std::string CInferenceModelMetadata::JSON_HYPERPARAMETER_NAME_TAG{"name"};
const std::string CInferenceModelMetadata::JSON_HYPERPARAMETER_VALUE_TAG{"value"};
const std::string CInferenceModelMetadata::JSON_HYPERPARAMETER_SUPPLIED_TAG{"supplied"};
const std::string CInferenceModelMetadata::JSON_IMPORTANCE_TAG{"importance"};
const std::string CInferenceModelMetadata::JSON_LOSS_GAP_TAG{"loss_gap"};
const std::string CInferenceModelMetadata::JSON_MAX_TAG{"max"};
const std::string CInferenceModelMetadata::JSON_MEAN_MAGNITUDE_TAG{"mean_magnitude"};
const std::string CInferenceModelMetadata::JSON_MIN_TAG{"min"};
const std::string CInferenceModelMetadata::JSON_MODEL_METADATA_TAG{"model_metadata"};
const std::string CInferenceModelMetadata::JSON_NUM_TRAIN_ROWS_TAG{"num_train_rows"};
const std::string CInferenceModelMetadata::JSON_RELATIVE_IMPORTANCE_TAG{"relative_importance"};
const std::string CInferenceModelMetadata::JSON_TOTAL_FEATURE_IMPORTANCE_TAG{"total_feature_importance"};
const std::string CInferenceModelMetadata::JSON_TRAIN_PROPERTIES_TAG{"train_properties"};
const std::string CInferenceModelMetadata::JSON_TRAINED_MODEL_MEMORY_USAGE_TAG{"trained_model_memory_usage"};
// clang-format on
}
}
