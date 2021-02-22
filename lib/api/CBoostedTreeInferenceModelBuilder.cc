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

void CBoostedTreeInferenceModelBuilder::addTree() {
    auto ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    ensemble->trainedModels().emplace_back(std::make_unique<CTree>());
}

void CBoostedTreeInferenceModelBuilder::addIdentityEncoding(std::size_t inputColumnIndex) {
    m_FeatureNames.push_back(m_FieldNames[inputColumnIndex]);
}

void CBoostedTreeInferenceModelBuilder::addOneHotEncoding(std::size_t inputColumnIndex,
                                                          std::size_t hotCategory) {
    std::string fieldName{m_Definition.fieldNames()[inputColumnIndex]};
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

void CBoostedTreeInferenceModelBuilder::addTargetMeanEncoding(std::size_t inputColumnIndex,
                                                              const TDoubleVec& map,
                                                              double fallback) {
    const std::string& fieldName{m_Definition.fieldNames()[inputColumnIndex]};
    std::string featureName{fieldName + "_targetmean"};
    auto stringMap = this->encodingMap(inputColumnIndex, map);
    m_Definition.preprocessors().push_back(std::make_unique<CTargetMeanEncoding>(
        fieldName, fallback, featureName, std::move(stringMap)));
    m_FeatureNames.push_back(featureName);
}

void CBoostedTreeInferenceModelBuilder::addFrequencyEncoding(std::size_t inputColumnIndex,
                                                             const TDoubleVec& map) {
    const std::string& fieldName{m_Definition.fieldNames()[inputColumnIndex]};
    std::string featureName{fieldName + "_frequency"};
    auto stringMap = this->encodingMap(inputColumnIndex, map);
    m_Definition.preprocessors().push_back(std::make_unique<CFrequencyEncoding>(
        fieldName, featureName, std::move(stringMap)));
    m_FeatureNames.push_back(featureName);
}

void CBoostedTreeInferenceModelBuilder::addCustomProcessor(TApiCustomEncodingUPtr value) {
    m_CustomProcessors.emplace_back(std::move(value));
}

CInferenceModelDefinition&& CBoostedTreeInferenceModelBuilder::build() {

    // Finalize OneHotEncoding Mappings
    for (auto& oneHotEncodingMapping : m_OneHotEncodingMaps) {
        m_Definition.preprocessors().emplace_back(
            std::move(oneHotEncodingMapping.second));
    }

    for (auto& customProcessor : m_CustomProcessors) {
        m_Definition.customPreprocessors().emplace_back(std::move(customProcessor));
    }

    // Add aggregated output after the number of trees is known
    auto ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
    this->setAggregateOutput(ensemble);

    this->setTargetType();
    ensemble->featureNames(m_FeatureNames);
    ensemble->removeUnusedFeatures();

    return std::move(m_Definition);
}

void CBoostedTreeInferenceModelBuilder::addNode(std::size_t splitFeature,
                                                double splitValue,
                                                bool assignMissingToLeft,
                                                const TVector& nodeValue,
                                                double gain,
                                                std::size_t numberSamples,
                                                maths::CBoostedTreeNode::TOptionalNodeIndex leftChild,
                                                maths::CBoostedTreeNode::TOptionalNodeIndex rightChild) {
    auto ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
    // use dynamic cast to prevent using wrong type of trained models
    auto tree = dynamic_cast<CTree*>(ensemble->trainedModels().back().get());
    if (tree == nullptr) {
        HANDLE_FATAL(<< "Internal error. Tree points to a nullptr.");
    }
    tree->treeStructure().emplace_back(tree->size(), splitValue, assignMissingToLeft,
                                       nodeValue.to<TDoubleVec>(), splitFeature,
                                       numberSamples, leftChild, rightChild, gain);
}

CBoostedTreeInferenceModelBuilder::CBoostedTreeInferenceModelBuilder(TStrVec fieldNames,
                                                                     std::size_t dependentVariableColumnIndex,
                                                                     const TStrVecVec& categoryNames)
    : m_CategoryNames{categoryNames} {

    // filter filed names containing empty string
    fieldNames.erase(std::remove(fieldNames.begin(), fieldNames.end(), ""),
                     fieldNames.end());
    fieldNames.erase(std::remove(fieldNames.begin(), fieldNames.end(), "."),
                     fieldNames.end());
    m_FieldNames = fieldNames;

    m_Definition.dependentVariableColumnIndex(dependentVariableColumnIndex);
    m_Definition.fieldNames(std::move(fieldNames));
    m_Definition.trainedModel(std::make_unique<CEnsemble>());
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

CInferenceModelDefinition& CBoostedTreeInferenceModelBuilder::definition() {
    return m_Definition;
}

CRegressionInferenceModelBuilder::CRegressionInferenceModelBuilder(const TStrVec& fieldNames,
                                                                   std::size_t dependentVariableColumnIndex,
                                                                   const TStrVecVec& categoryNames)
    : CBoostedTreeInferenceModelBuilder{fieldNames, dependentVariableColumnIndex, categoryNames} {
}

void CRegressionInferenceModelBuilder::addClassificationWeights(TDoubleVec /*weights*/) {
}

void CRegressionInferenceModelBuilder::addLossFunction(const maths::CBoostedTree::TLossFunction& lossFunction) {
    m_LossType = lossFunction.type();
}

void CRegressionInferenceModelBuilder::setTargetType() {
    this->definition().trainedModel()->targetType(CTrainedModel::ETargetType::E_Regression);
}

void CRegressionInferenceModelBuilder::setAggregateOutput(CEnsemble* ensemble) const {
    switch (m_LossType) {
    case TLossType::E_MsleRegression:
        ensemble->aggregateOutput(std::make_unique<CExponent>(ensemble->size(), 1.0));
        break;
    case TLossType::E_MseRegression:
    case TLossType::E_HuberRegression:
        ensemble->aggregateOutput(std::make_unique<CWeightedSum>(ensemble->size(), 1.0));
        break;
    case TLossType::E_BinaryClassification:
    case TLossType::E_MulticlassClassification:
        LOG_ERROR(<< "Input error: classification objective function received where regression objective expected.");
        break;
    }
}

CClassificationInferenceModelBuilder::CClassificationInferenceModelBuilder(
    const TStrVec& fieldNames,
    std::size_t dependentVariableColumnIndex,
    const TStrVecVec& categoryNames)
    : CBoostedTreeInferenceModelBuilder{fieldNames, dependentVariableColumnIndex, categoryNames} {
    this->definition().trainedModel()->classificationLabels(
        categoryNames[dependentVariableColumnIndex]);
}

void CClassificationInferenceModelBuilder::addClassificationWeights(TDoubleVec weights) {
    this->definition().trainedModel()->classificationWeights(std::move(weights));
}

void CClassificationInferenceModelBuilder::addLossFunction(
    const maths::CBoostedTree::TLossFunction& /*lossFunction*/) {
}

void CClassificationInferenceModelBuilder::setTargetType() {
    this->definition().trainedModel()->targetType(CTrainedModel::ETargetType::E_Classification);
}

void CClassificationInferenceModelBuilder::setAggregateOutput(CEnsemble* ensemble) const {
    ensemble->aggregateOutput(std::make_unique<CLogisticRegression>(ensemble->size(), 1.0));
}
}
}
