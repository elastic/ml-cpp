/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CBoostedTreeInferenceModelBuilder.h>

#include <core/LogMacros.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/writer.h>

#include <algorithm>

namespace ml {
namespace api {

namespace {
const std::string INFERENCE_MODEL{"inference_model"};
}

void CBoostedTreeInferenceModelBuilder::addTree() {
    auto ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    ensemble->trainedModels().emplace_back(std::make_unique<CTree>());
}

void CBoostedTreeInferenceModelBuilder::addIdentityEncoding(std::size_t inputColumnIndex) {
    if (inputColumnIndex < m_FieldNames.size()) {
        // The target column is excluded from m_FieldNames.
        m_FeatureNames.push_back(m_FieldNames[inputColumnIndex]);
    }
}

void CBoostedTreeInferenceModelBuilder::addOneHotEncoding(std::size_t inputColumnIndex,
                                                          std::size_t hotCategory) {
    if (inputColumnIndex < m_Definition.input().fieldNames().size()) {
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
}

void CBoostedTreeInferenceModelBuilder::addTargetMeanEncoding(std::size_t inputColumnIndex,
                                                              const TDoubleVec& map,
                                                              double fallback) {
    if (inputColumnIndex < m_Definition.input().fieldNames().size()) {
        std::string fieldName{m_Definition.input().fieldNames()[inputColumnIndex]};
        std::string featureName{fieldName + "_targetmean"};
        auto stringMap = this->encodingMap(inputColumnIndex, map);
        m_Definition.preprocessors().push_back(std::make_unique<CTargetMeanEncoding>(
            fieldName, fallback, featureName, std::move(stringMap)));
        m_FeatureNames.push_back(featureName);
    }
}

void CBoostedTreeInferenceModelBuilder::addFrequencyEncoding(std::size_t inputColumnIndex,
                                                             const TDoubleVec& map) {
    if (inputColumnIndex < m_Definition.input().fieldNames().size()) {
        std::string fieldName{m_Definition.input().fieldNames()[inputColumnIndex]};
        std::string featureName{fieldName + "_frequency"};
        auto stringMap = this->encodingMap(inputColumnIndex, map);
        m_Definition.preprocessors().push_back(std::make_unique<CFrequencyEncoding>(
            fieldName, featureName, std::move(stringMap)));
        m_FeatureNames.push_back(featureName);
    }
}

CInferenceModelDefinition&& CBoostedTreeInferenceModelBuilder::build() {

    // Finalize OneHotEncoding Mappings
    for (auto& oneHotEncodingMapping : m_OneHotEncodingMaps) {
        m_Definition.preprocessors().emplace_back(
            std::move(oneHotEncodingMapping.second));
    }

    // Add aggregated output after the number of trees is known
    auto ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
    this->setAggregateOutput(ensemble);

    this->setTargetType();
    ensemble->featureNames(m_FeatureNames);
    ensemble->adjustFeatureNames();

    return std::move(m_Definition);
}

void CBoostedTreeInferenceModelBuilder::addNode(
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

CBoostedTreeInferenceModelBuilder::CBoostedTreeInferenceModelBuilder(
    TStrVec fieldNames,
    std::size_t dependentVariableColumnIndex,
    TStrVecVec categoryNames)
    : m_DependentVariableColumnIndex(dependentVariableColumnIndex) {
    // filter filed names containing empty string
    fieldNames.erase(std::remove(fieldNames.begin(), fieldNames.end(), ""),
                     fieldNames.end());
    fieldNames.erase(std::remove(fieldNames.begin(), fieldNames.end(), "."),
                     fieldNames.end());
    m_FieldNames = fieldNames;

    this->categoryNames(categoryNames);
    fieldNames.erase(fieldNames.begin() +
                     static_cast<std::ptrdiff_t>(dependentVariableColumnIndex));
    m_Definition.fieldNames(fieldNames);
    m_Definition.trainedModel(std::make_unique<CEnsemble>());
    m_Definition.typeString(INFERENCE_MODEL);
}

CBoostedTreeInferenceModelBuilder::TStringDoubleUMap
CBoostedTreeInferenceModelBuilder::encodingMap(std::size_t inputColumnIndex,
                                               const TDoubleVec& map_) {
    TStringDoubleUMap map;
    for (std::size_t categoryUInt = 0; categoryUInt < map_.size(); ++categoryUInt) {
        std::string category{m_CategoryNames[inputColumnIndex][categoryUInt]};
        map.emplace(category, map_[categoryUInt]);
    }
    return map;
}

void CBoostedTreeInferenceModelBuilder::categoryNames(const TStrVecVec& categoryNames) {
    m_CategoryNames = categoryNames;
}

CInferenceModelDefinition& CBoostedTreeInferenceModelBuilder::definition() {
    return m_Definition;
}

void CRegressionInferenceModelBuilder::setTargetType() {
    this->definition().trainedModel()->targetType(CTrainedModel::ETargetType::E_Regression);
}

CRegressionInferenceModelBuilder::CRegressionInferenceModelBuilder(TStrVec fieldNames,
                                                                   std::size_t dependentVariableColumnIndex,
                                                                   const TStrVecVec& categoryNames)
    : CBoostedTreeInferenceModelBuilder(fieldNames, dependentVariableColumnIndex, categoryNames) {
}

void CRegressionInferenceModelBuilder::setAggregateOutput(CEnsemble* ensemble) const {
    ensemble->aggregateOutput(std::make_unique<CWeightedSum>(ensemble->size(), 1.0));
}

void CClassificationInferenceModelBuilder::setTargetType() {
    this->definition().trainedModel()->targetType(CTrainedModel::ETargetType::E_Classification);
}

CClassificationInferenceModelBuilder::CClassificationInferenceModelBuilder(
    TStrVec fieldNames,
    std::size_t dependentVariableColumnIndex,
    const TStrVecVec& categoryNames)
    : CBoostedTreeInferenceModelBuilder(fieldNames, dependentVariableColumnIndex, categoryNames) {
}

void CClassificationInferenceModelBuilder::setAggregateOutput(CEnsemble* ensemble) const {
    ensemble->aggregateOutput(std::make_unique<CLogisticRegression>(ensemble->size(), 1.0));
}
}
}
