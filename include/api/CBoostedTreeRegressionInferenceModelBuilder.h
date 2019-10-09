/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CBoostedTreeRegressionInferenceModelBuilder_h
#define INCLUDED_ml_api_CBoostedTreeRegressionInferenceModelBuilder_h

#include <maths/CBoostedTree.h>

#include <api/CInferenceModelDefinition.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <string>

namespace ml {
namespace api {

//! \brief Builds a a serialisable trained model object by visiting a maths::CBoostedTree object.
class API_EXPORT CBoostedTreeRegressionInferenceModelBuilder
    : public maths::CBoostedTree::CVisitor {
public:
    using TDoubleVec = std::vector<double>;
    using TStringVec = std::vector<std::string>;
    using TStringSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStringSizeUMapVec = std::vector<TStringSizeUMap>;
    using TSizeStringUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;

public:
    CBoostedTreeRegressionInferenceModelBuilder(TStringVec fieldNames,
                                                std::size_t dependentVariableColumnIndex,
                                                const TStringSizeUMapVec& categoryNameMap);
    ~CBoostedTreeRegressionInferenceModelBuilder() override = default;
    void addTree() override;
    void addNode(std::size_t splitFeature,
                 double splitValue,
                 bool assignMissingToLeft,
                 double nodeValue,
                 double gain,
                 maths::CBoostedTreeNode::TOptionalNodeIndex leftChild,
                 maths::CBoostedTreeNode::TOptionalNodeIndex rightChild) override;
    void addIdentityEncoding(std::size_t inputColumnIndex) override;
    void addOneHotEncoding(std::size_t inputColumnIndex, std::size_t hotCategory) override;
    void addTargetMeanEncoding(std::size_t inputColumnIndex,
                               const TDoubleVec& map,
                               double fallback) override;
    void addFrequencyEncoding(std::size_t inputColumnIndex, const TDoubleVec& map) override;
    CInferenceModelDefinition&& build();

private:
    using TOneHotEncodingUPtr = std::unique_ptr<COneHotEncoding>;
    using TOneHotEncodingUMap = std::unordered_map<std::string, TOneHotEncodingUPtr>;
    using TStringDoubleUMap = std::unordered_map<std::string, double>;

private:
    TStringDoubleUMap encodingMap(std::size_t inputColumnIndex, const TDoubleVec& map_);

    void categoryNameMap(const CInferenceModelDefinition::TStringSizeUMapVec& categoryNameMap);

private:
    CInferenceModelDefinition m_Definition;
    TSizeStringUMapVec m_ReverseCategoryNameMap;
    TOneHotEncodingUMap m_OneHotEncodingMaps;
    TStringVec m_FieldNames;
    TStringVec m_FeatureNames;
};
}
}

#endif // INCLUDED_ml_api_CBoostedTreeRegressionInferenceModelBuilder_h
