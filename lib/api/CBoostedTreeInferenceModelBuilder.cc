/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CBoostedTreeInferenceModelBuilder.h>

#include <core/LogMacros.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CInferenceModelDefinition.h>

#include <algorithm>

namespace ml {
namespace api {

void CBoostedTreeInferenceModelBuilder::addTree() {
    auto* ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    ensemble->trainedModels().emplace_back(std::make_unique<CTree>());
}

void CBoostedTreeInferenceModelBuilder::addIdentityEncoding(std::size_t inputColumnIndex) {
    m_FeatureNames.push_back(m_FeatureNameProvider.identityEncodingName(inputColumnIndex));
}

void CBoostedTreeInferenceModelBuilder::addOneHotEncoding(std::size_t inputColumnIndex,
                                                          std::size_t hotCategory) {
    const std::string& fieldName{m_FeatureNameProvider.fieldName(inputColumnIndex)};
    const std::string& category{m_FeatureNameProvider.category(inputColumnIndex, hotCategory)};
    std::string featureName{
        m_FeatureNameProvider.oneHotEncodingName(inputColumnIndex, hotCategory)};
    if (m_OneHotEncodingMaps.find(fieldName) == m_OneHotEncodingMaps.end()) {
        m_OneHotEncodingMaps.emplace(
            fieldName, std::make_unique<COneHotEncoding>(
                           fieldName, COneHotEncoding::TStringStringUMap()));
    }
    m_OneHotEncodingMaps[fieldName]->hotMap().emplace(category, featureName);
    m_FeatureNames.push_back(std::move(featureName));
}

void CBoostedTreeInferenceModelBuilder::addTargetMeanEncoding(std::size_t inputColumnIndex,
                                                              const TDoubleVec& map,
                                                              double fallback) {
    const std::string& fieldName{m_FeatureNameProvider.fieldName(inputColumnIndex)};
    std::string featureName{m_FeatureNameProvider.targetMeanEncodingName(inputColumnIndex)};
    m_Definition.preprocessors().push_back(std::make_unique<CTargetMeanEncoding>(
        fieldName, fallback, featureName, this->encodingMap(inputColumnIndex, map)));
    m_FeatureNames.push_back(std::move(featureName));
}

void CBoostedTreeInferenceModelBuilder::addFrequencyEncoding(std::size_t inputColumnIndex,
                                                             const TDoubleVec& map) {
    const std::string& fieldName{m_FeatureNameProvider.fieldName(inputColumnIndex)};
    std::string featureName{m_FeatureNameProvider.frequencyEncodingName(inputColumnIndex)};
    m_Definition.preprocessors().push_back(std::make_unique<CFrequencyEncoding>(
        fieldName, featureName, this->encodingMap(inputColumnIndex, map)));
    m_FeatureNames.push_back(std::move(featureName));
}

void CBoostedTreeInferenceModelBuilder::addCustomProcessor(TApiCustomEncodingUPtr value) {
    m_CustomProcessors.emplace_back(std::move(value));
}

CInferenceModelDefinition&& CBoostedTreeInferenceModelBuilder::build() {

    // Finalize one-hot encoding mappings
    for (auto& oneHotEncodingMapping : m_OneHotEncodingMaps) {
        m_Definition.preprocessors().emplace_back(
            std::move(oneHotEncodingMapping.second));
    }

    // Copy the custom preprocessors.
    for (auto& customProcessor : m_CustomProcessors) {
        m_Definition.customPreprocessors().emplace_back(std::move(customProcessor));
    }

    // Add aggregated output after the number of trees is known.
    auto* ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
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
    auto* ensemble{static_cast<CEnsemble*>(m_Definition.trainedModel().get())};
    // use dynamic cast to prevent using wrong type of trained models
    auto* tree = dynamic_cast<CTree*>(ensemble->trainedModels().back().get());
    if (tree == nullptr) {
        HANDLE_FATAL(<< "Internal error. Tree is null.");
        return;
    }
    tree->treeStructure().emplace_back(tree->size(), splitValue, assignMissingToLeft,
                                       nodeValue.to<TDoubleVec>(), splitFeature,
                                       numberSamples, leftChild, rightChild, gain);
}

CBoostedTreeInferenceModelBuilder::CBoostedTreeInferenceModelBuilder(TStrVec fieldNames,
                                                                     std::size_t dependentVariableColumnIndex,
                                                                     TStrVecVec categoryNames)
    : m_FeatureNameProvider{fieldNames, std::move(categoryNames)} {

    fieldNames.erase(std::remove_if(fieldNames.begin(), fieldNames.end(),
                                    [](const auto& name) {
                                        return name.empty() ||
                                               name == CDataFrameAnalyzer::CONTROL_MESSAGE_FIELD_NAME;
                                    }),
                     fieldNames.end());

    m_Definition.dependentVariableColumnIndex(dependentVariableColumnIndex);
    m_Definition.fieldNames(std::move(fieldNames));
    m_Definition.trainedModel(std::make_unique<CEnsemble>());
}

CBoostedTreeInferenceModelBuilder::TStrDoubleUMap
CBoostedTreeInferenceModelBuilder::encodingMap(std::size_t inputColumnIndex,
                                               const TDoubleVec& map) {
    TStrDoubleUMap result;
    for (std::size_t i = 0; i < map.size(); ++i) {
        result.emplace(m_FeatureNameProvider.category(inputColumnIndex, i), map[i]);
    }
    return result;
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
