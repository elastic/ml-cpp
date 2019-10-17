/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CBoostedTreeRegressionInferenceModelBuilder.h>

#include <core/LogMacros.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/writer.h>

#include <algorithm>

namespace ml {
namespace api {

namespace {
const std::string INFERENCE_MODEL{"inference_model"};
}

void CBoostedTreeRegressionInferenceModelBuilder::addTree() {
    auto ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    ensemble->trainedModels().emplace_back(std::make_unique<CTree>());
}

void CBoostedTreeRegressionInferenceModelBuilder::addIdentityEncoding(std::size_t inputColumnIndex) {
    if (inputColumnIndex < m_FieldNames.size()) {
        // The target column is excluded from m_FieldNames.
        m_FeatureNames.push_back(m_FieldNames[inputColumnIndex]);
    }
}

void CBoostedTreeRegressionInferenceModelBuilder::addOneHotEncoding(std::size_t inputColumnIndex,
                                                                    std::size_t hotCategory) {
    std::string fieldName{m_Definition.input().fieldNames()[inputColumnIndex]};
    std::string category = m_CategoryNames[inputColumnIndex][hotCategory];
    std::string featureName = fieldName + "_" + category;
    if (m_OneHotEncodingMaps.find(fieldName) == m_OneHotEncodingMaps.end()) {
        auto apiEncoding = std::make_unique<COneHotEncoding>(
            fieldName, COneHotEncoding::TStringStringUMap());
        m_OneHotEncodingMaps.emplace(fieldName, std::move(apiEncoding));
    }
    m_OneHotEncodingMaps[fieldName]->hotMap().emplace(category, featureName);
    m_FeatureNames.push_back(featureName);
}

void CBoostedTreeRegressionInferenceModelBuilder::addTargetMeanEncoding(std::size_t inputColumnIndex,
                                                                        const TDoubleVec& map,
                                                                        double fallback) {
    std::string fieldName{m_Definition.input().fieldNames()[inputColumnIndex]};
    std::string featureName{fieldName + "_targetmean"};
    auto stringMap = this->encodingMap(inputColumnIndex, map);
    m_Definition.preprocessors().push_back(std::make_unique<CTargetMeanEncoding>(
        fieldName, fallback, featureName, std::move(stringMap)));
    m_FeatureNames.push_back(featureName);
}

void CBoostedTreeRegressionInferenceModelBuilder::addFrequencyEncoding(std::size_t inputColumnIndex,
                                                                       const TDoubleVec& map) {
    std::string fieldName{m_Definition.input().fieldNames()[inputColumnIndex]};
    std::string featureName{fieldName + "_frequency"};
    auto stringMap = this->encodingMap(inputColumnIndex, map);
    m_Definition.preprocessors().push_back(std::make_unique<CFrequencyEncoding>(
        fieldName, featureName, std::move(stringMap)));
    m_FeatureNames.push_back(featureName);
}

CInferenceModelDefinition&& CBoostedTreeRegressionInferenceModelBuilder::build() {

    // Finalize OneHotEncoding Mappings
    for (auto& oneHotEncodingMapping : m_OneHotEncodingMaps) {
        m_Definition.preprocessors().emplace_back(
            std::move(oneHotEncodingMapping.second));
    }

    // Add aggregated output after the number of trees is known
    auto ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
    ensemble->aggregateOutput(std::make_unique<CWeightedSum>(ensemble->size(), 1.0));

    ensemble->targetType(CTrainedModel::E_Regression);
    ensemble->featureNames(m_FeatureNames);
    ensemble->adjustFeatureNames();

    return std::move(m_Definition);
}

void CBoostedTreeRegressionInferenceModelBuilder::addNode(
    std::size_t splitFeature,
    double splitValue,
    bool assignMissingToLeft,
    double nodeValue,
    double gain,
    ml::maths::CBoostedTreeNode::TOptionalNodeIndex leftChild,
    ml::maths::CBoostedTreeNode::TOptionalNodeIndex rightChild) {
    auto ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
    // use dynamic cast to prevent using wrong type of trained models
    auto tree = dynamic_cast<CTree*>(ensemble->trainedModels().back().get());
    if (tree == nullptr) {
        HANDLE_FATAL(<< "Internal error. Tree points to a nullptr.")
    }
    tree->treeStructure().emplace_back(tree->size(), splitValue, assignMissingToLeft, nodeValue,
                                       splitFeature, leftChild, rightChild, gain);
}

CBoostedTreeRegressionInferenceModelBuilder::CBoostedTreeRegressionInferenceModelBuilder(
    TStrVec fieldNames,
    std::size_t dependentVariableColumnIndex,
    TStrVecVec categoryNames)
    : m_DependentVariableColumnIndex(dependentVariableColumnIndex) {
    // filter filed names containing empty string
    fieldNames.erase(std::remove(fieldNames.begin(), fieldNames.end(), ""),
                     fieldNames.end());
    m_FieldNames = fieldNames;

    this->categoryNames(categoryNames);
    m_Definition.fieldNames(fieldNames);
    m_Definition.trainedModel(std::make_unique<CEnsemble>());
    m_Definition.typeString(INFERENCE_MODEL);
}

CBoostedTreeRegressionInferenceModelBuilder::TStringDoubleUMap
CBoostedTreeRegressionInferenceModelBuilder::encodingMap(std::size_t inputColumnIndex,
                                                         const TDoubleVec& map_) {
    TStringDoubleUMap map;
    for (std::size_t categoryUInt = 0; categoryUInt < map_.size(); ++categoryUInt) {
        std::string category{m_CategoryNames[inputColumnIndex][categoryUInt]};
        map.emplace(category, map_[categoryUInt]);
    }
    return map;
}

void CBoostedTreeRegressionInferenceModelBuilder::categoryNames(const TStrVecVec& categoryNames) {
    m_CategoryNames = categoryNames;
}
}
}
