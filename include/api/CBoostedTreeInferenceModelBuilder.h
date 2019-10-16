/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h
#define INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h

#include <maths/CBoostedTree.h>

#include <api/CInferenceModelDefinition.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <string>

namespace ml {
namespace api {

//! \brief Builds a a serialisable trained model object by visiting a maths::CBoostedTree object.
class API_EXPORT CBoostedTreeInferenceModelBuilder : public maths::CBoostedTree::CVisitor {
public:
    using TDoubleVec = std::vector<double>;
    using TStringVec = std::vector<std::string>;
    using TStringSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStringSizeUMapVec = std::vector<TStringSizeUMap>;
    using TSizeStringUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;

public:
    CBoostedTreeInferenceModelBuilder(TStringVec fieldNames,
                                      std::size_t dependentVariableColumnIndex,
                                      const TStringSizeUMapVec& categoryNameMap);
    ~CBoostedTreeInferenceModelBuilder() override = default;
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

protected:
    CInferenceModelDefinition& definition();
    virtual void setTargetType() = 0;
    virtual void setAggregateOutput(CEnsemble* ensemble) const = 0;

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

class API_EXPORT CRegressionInferenceModelBuilder : public CBoostedTreeInferenceModelBuilder {
protected:
public:
    CRegressionInferenceModelBuilder(const TStringVec& fieldNames,
                                     size_t dependentVariableColumnIndex,
                                     const TStringSizeUMapVec& categoryNameMap);

protected:
    void setTargetType() override;
    void setAggregateOutput(CEnsemble* ensemble) const override;
};

class API_EXPORT CClassificationInferenceModelBuilder : public CBoostedTreeInferenceModelBuilder {
public:
    CClassificationInferenceModelBuilder(const TStringVec& fieldNames,
                                         size_t dependentVariableColumnIndex,
                                         const TStringSizeUMapVec& categoryNameMap);

protected:
    void setTargetType() override;
    void setAggregateOutput(CEnsemble* ensemble) const override;
};
}
}

#endif // INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h
