#include <api/CBoostedTreeRegressionInferenceModelFormatter.h>

#include <core/CJsonStateRestoreTraverser.h>
#include <core/LogMacros.h>
#include <core/RestoreMacros.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/writer.h>

namespace {
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string ENCODER_TAG{"encoder_tag"};

const std::string REGRESSION_INFERENCE_MODEL{"regression_inference_model"};
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::visit(const ml::maths::CBoostedTree* tree) {
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::visit(const ml::maths::CBoostedTreeImpl* impl) {
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::visit(const ml::maths::CBoostedTreeNode* node) {
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::addTree() {
    auto ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    ensemble->trainedModels().emplace_back();
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::addOneHotEncoding(std::size_t inputColumnIndex,
                                                                             double mic,
                                                                             std::size_t hotCategory) {
    std::string fieldName{m_Definition.input().columns()[inputColumnIndex]};
    std::string category = m_ReverseCategoryNameMap[inputColumnIndex][hotCategory];
    std::string encodedFieldName = fieldName + "_" + category;
    if (m_OneHotEncodingMaps.find(fieldName) == m_OneHotEncodingMaps.end()) {
        auto apiEncoding = std::make_unique<api::COneHotEncoding>(
            fieldName, api::COneHotEncoding::TStringStringUMap());
        m_OneHotEncodingMaps.emplace(fieldName, std::move(apiEncoding));
    }
    m_OneHotEncodingMaps[fieldName]->hotMap().emplace(category, encodedFieldName);
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::addTargetMeanEncoding(
    std::size_t inputColumnIndex,
    double mic,
    const TDoubleVec& map,
    double fallback) {
    std::string fieldName{m_Definition.input().columns()[inputColumnIndex]};
    std::string featureName{fieldName + "_targetmean"};
    std::map<std::string, double> stringMap = this->encodingMap(inputColumnIndex, map);
    m_Definition.preprocessing().push_back(std::make_unique<CTargetMeanEncoding>(
        fieldName, fallback, featureName, std::move(stringMap)));
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::addFrequencyEncoding(
    std::size_t inputColumnIndex,
    double mic,
    const TDoubleVec& map,
    double fallback) {
    std::string fieldName{m_Definition.input().columns()[inputColumnIndex]};
    std::string featureName{fieldName + "_frequency"};
    std::map<std::string, double> stringMap = this->encodingMap(inputColumnIndex, map);
    m_Definition.preprocessing().push_back(std::make_unique<CFrequencyEncoding>(
        fieldName, featureName, std::move(stringMap)));
}

ml::api::CInferenceModelDefinition&&
ml::api::CBoostedTreeRegressionInferenceModelBuilder::build() {

    // Finalize OneHotEncoding Mappings
    for (auto& oneHotEncodingMapping : m_OneHotEncodingMaps) {
        m_Definition.preprocessing().emplace_back(
            std::move(oneHotEncodingMapping.second));
    }

    // Add aggregated output after the number of trees is known
    auto ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    ensemble->aggregateOutput(std::make_unique<CWeightedSum>(ensemble->size(), 1.0));

    ensemble->targetType(CTrainedModel::E_Regression);

    return std::move(m_Definition);
}

void ml::api::CBoostedTreeRegressionInferenceModelBuilder::addNode(
    std::size_t splitFeature,
    double splitValue,
    bool assignMissingToLeft,
    double nodeValue,
    double gain,
    ml::maths::CBoostedTreeNode::TOptionalSize leftChild,
    ml::maths::CBoostedTreeNode::TOptionalSize rightChild) {
    auto ensemble = static_cast<CEnsemble*>(m_Definition.trainedModel().get());
    CTree& tree{ensemble->trainedModels().back()};
    tree.treeStructure().emplace_back(tree.size(), splitValue, assignMissingToLeft, nodeValue,
                                      splitFeature, leftChild, rightChild, gain);
}

ml::api::CBoostedTreeRegressionInferenceModelBuilder::CBoostedTreeRegressionInferenceModelBuilder(
    const TStringVec& fieldNames,
    const TStringSizeUMapVec& categoryNameMap)
    : m_Definition(fieldNames, categoryNameMap) {
    this->categoryNameMap(categoryNameMap);
    m_Definition.trainedModel(std::make_unique<CEnsemble>());
    m_Definition.typeString(REGRESSION_INFERENCE_MODEL);
}
