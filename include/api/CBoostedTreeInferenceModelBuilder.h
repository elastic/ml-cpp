/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h
#define INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeLoss.h>

#include <api/CInferenceModelDefinition.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <boost/unordered_map.hpp>

#include <string>

namespace ml {
namespace api {

//! \brief Builds a serialisable trained model object by visiting a maths::CBoostedTree object.
class API_EXPORT CBoostedTreeInferenceModelBuilder : public maths::CBoostedTree::CVisitor {
public:
    using TDoubleVec = std::vector<double>;
    using TStrVec = std::vector<std::string>;
    using TStrVecVec = std::vector<TStrVec>;
    using TVector = maths::CBoostedTreeNode::TVector;
    using TApiCustomEncodingUPtr = std::unique_ptr<api::CCustomEncoding>;
    using TApiCustomEncodingUPtrVec = std::vector<TApiCustomEncodingUPtr>;

public:
    CBoostedTreeInferenceModelBuilder(TStrVec fieldNames,
                                      std::size_t dependentVariableColumnIndex,
                                      TStrVecVec categoryNames);
    ~CBoostedTreeInferenceModelBuilder() override = default;
    void addTree() override;
    void addNode(std::size_t splitFeature,
                 double splitValue,
                 bool assignMissingToLeft,
                 const TVector& nodeValue,
                 double gain,
                 std::size_t numberSamples,
                 maths::CBoostedTreeNode::TOptionalNodeIndex leftChild,
                 maths::CBoostedTreeNode::TOptionalNodeIndex rightChild) override;
    void addIdentityEncoding(std::size_t inputColumnIndex) override;
    void addOneHotEncoding(std::size_t inputColumnIndex, std::size_t hotCategory) override;
    void addTargetMeanEncoding(std::size_t inputColumnIndex,
                               const TDoubleVec& map,
                               double fallback) override;
    void addFrequencyEncoding(std::size_t inputColumnIndex, const TDoubleVec& map) override;
    void addCustomProcessor(TApiCustomEncodingUPtr value);
    virtual CInferenceModelDefinition&& build();

protected:
    CInferenceModelDefinition& definition();

private:
    using TOneHotEncodingUPtr = std::unique_ptr<COneHotEncoding>;
    using TStrOneHotEncodingUPtrUMap = boost::unordered_map<std::string, TOneHotEncodingUPtr>;
    using TStrDoubleUMap = boost::unordered_map<std::string, double>;

private:
    virtual void setTargetType() = 0;
    virtual void setAggregateOutput(CEnsemble* ensemble) const = 0;
    TStrDoubleUMap encodingMap(std::size_t inputColumnIndex, const TDoubleVec& map);

private:
    CInferenceModelDefinition m_Definition;
    CTrainedModel::CFeatureNameProvider m_FeatureNameProvider;
    TStrVec m_FeatureNames;
    TStrOneHotEncodingUPtrUMap m_OneHotEncodingMaps;
    TApiCustomEncodingUPtrVec m_CustomProcessors;
};

class API_EXPORT CRegressionInferenceModelBuilder final : public CBoostedTreeInferenceModelBuilder {
public:
    CRegressionInferenceModelBuilder(const TStrVec& fieldNames,
                                     std::size_t dependentVariableColumnIndex,
                                     const TStrVecVec& categoryNames);
    void addClassificationWeights(TDoubleVec weights) override;
    void addLossFunction(const maths::CBoostedTree::TLossFunction& lossFunction) override;

private:
    using TLossType = maths::boosted_tree::ELossType;

private:
    void setTargetType() override;
    void setAggregateOutput(CEnsemble* ensemble) const override;

private:
    TLossType m_LossType;
};

class API_EXPORT CClassificationInferenceModelBuilder final
    : public CBoostedTreeInferenceModelBuilder {
public:
    CClassificationInferenceModelBuilder(const TStrVec& fieldNames,
                                         std::size_t dependentVariableColumnIndex,
                                         const TStrVecVec& categoryNames);
    ~CClassificationInferenceModelBuilder() override = default;
    void addClassificationWeights(TDoubleVec weights) override;
    void addLossFunction(const maths::CBoostedTree::TLossFunction& lossFunction) override;

private:
    void setTargetType() override;
    void setAggregateOutput(CEnsemble* ensemble) const override;
};
}
}

#endif // INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h
