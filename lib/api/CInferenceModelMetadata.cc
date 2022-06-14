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

#include <api/ApiTypes.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <maths/analytics/CBoostedTreeUtils.h>

#include <cmath>
#include <cstdint>

namespace ml {
namespace api {

void CInferenceModelMetadata::write(TRapidJsonWriter& writer) const {
    this->writeTotalFeatureImportance(writer);
    this->writeFeatureImportanceBaseline(writer);
    this->writeHyperparameterImportance(writer);
    LOG_DEBUG(<< "Number data summarization rows " << m_NumDataSummarizationRows);
    if (m_NumDataSummarizationRows > 0) {
        // Only output if data summarization fraction was specified.
        this->writeTrainProperties(writer);
        this->writeDataSummarization(writer);
    }
}

void CInferenceModelMetadata::writeTotalFeatureImportance(TRapidJsonWriter& writer) const {
    writer.Key(JSON_TOTAL_FEATURE_IMPORTANCE_TAG);
    writer.StartArray();
    for (const auto& item : m_TotalShapValuesMean) {
        writer.StartObject();
        writer.Key(JSON_FEATURE_NAME_TAG);
        writer.String(m_ColumnNames[item.first]);
        auto meanFeatureImportance = maths::common::CBasicStatistics::mean(item.second);
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
            for (const auto& classValue : m_ClassValues) {
                writer.StartObject();
                writer.Key(JSON_CLASS_NAME_TAG);
                m_PredictionFieldTypeResolverWriter(classValue, writer);
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

void CInferenceModelMetadata::writeHyperparameterImportance(TRapidJsonWriter& writer) const {
    writer.Key(JSON_HYPERPARAMETERS_TAG);
    writer.StartArray();
    for (const auto& item : m_HyperparameterImportance) {
        writer.StartObject();
        writer.Key(JSON_HYPERPARAMETER_NAME_TAG);
        writer.String(item.s_HyperparameterName);
        writer.Key(JSON_HYPERPARAMETER_VALUE_TAG);
        switch (item.s_Type) {
        case SHyperparameterImportance::E_Double:
            writer.Double(item.s_Value);
            break;
        case SHyperparameterImportance::E_Uint64:
            writer.Uint64(static_cast<std::uint64_t>(item.s_Value));
            break;
        }
        if (item.s_Supplied == false) {
            writer.Key(JSON_ABSOLUTE_IMPORTANCE_TAG);
            writer.Double(item.s_AbsoluteImportance);
            writer.Key(JSON_RELATIVE_IMPORTANCE_TAG);
            writer.Double(item.s_RelativeImportance);
        }
        writer.Key(JSON_HYPERPARAMETER_SUPPLIED_TAG);
        writer.Bool(item.s_Supplied);
        writer.EndObject();
    }
    writer.EndArray();
}

void CInferenceModelMetadata::writeDataSummarization(TRapidJsonWriter& writer) const {
    // only write out if data summarization exists
    if (m_NumDataSummarizationRows > 0) {
        writer.Key(JSON_DATA_SUMMARIZATION_TAG);
        writer.StartObject();
        writer.Key(JSON_NUM_DATA_SUMMARIZATION_ROWS_TAG);
        writer.Uint64(m_NumDataSummarizationRows);
        writer.EndObject();
    }
}

void CInferenceModelMetadata::writeTrainProperties(TRapidJsonWriter& writer) const {
    if (m_NumTrainRows > 0) {
        writer.Key(JSON_TRAIN_PROPERTIES_TAG);
        writer.StartObject();
        writer.Key(JSON_NUM_TRAIN_ROWS_TAG);
        writer.Uint64(m_NumTrainRows);
        writer.Key(JSON_LOSS_GAP_TAG);
        writer.Double(m_LossGap);
        writer.EndObject();
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
        // Train hyperparameters.
        case maths::analytics::boosted_tree_detail::E_Alpha:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::ALPHA;
            break;
        case maths::analytics::boosted_tree_detail::E_DownsampleFactor:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_FACTOR;
            break;
        case maths::analytics::boosted_tree_detail::E_Eta:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::ETA;
            break;
        case maths::analytics::boosted_tree_detail::E_EtaGrowthRatePerTree:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::ETA_GROWTH_RATE_PER_TREE;
            break;
        case maths::analytics::boosted_tree_detail::E_FeatureBagFraction:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION;
            break;
        case maths::analytics::boosted_tree_detail::E_Gamma:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::GAMMA;
            break;
        case maths::analytics::boosted_tree_detail::E_Lambda:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::LAMBDA;
            break;
        case maths::analytics::boosted_tree_detail::E_SoftTreeDepthLimit:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT;
            break;
        case maths::analytics::boosted_tree_detail::E_SoftTreeDepthTolerance:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE;
            break;
        // Not tuned directly.
        case maths::analytics::boosted_tree_detail::E_MaximumNumberTrees:
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::MAX_TREES;
            break;

        // Incremental train hyperparameters.
        case maths::analytics::boosted_tree_detail::E_PredictionChangeCost:
            if (m_Task != api_t::E_Update) {
                continue;
            }
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::PREDICTION_CHANGE_COST;
            break;
        case maths::analytics::boosted_tree_detail::E_RetrainedTreeEta:
            if (m_Task != api_t::E_Update) {
                continue;
            }
            hyperparameterName = CDataFrameTrainBoostedTreeRunner::RETRAINED_TREE_ETA;
            break;
        case maths::analytics::boosted_tree_detail::E_TreeTopologyChangePenalty:
            if (m_Task != api_t::E_Update) {
                continue;
            }
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

void CInferenceModelMetadata::task(api_t::EDataFrameTrainBoostedTreeTask task) {
    m_Task = task;
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
// clang-format on
}
}
