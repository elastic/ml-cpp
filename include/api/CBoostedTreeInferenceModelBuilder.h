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
    using TStrVec = std::vector<std::string>;
    using TStrVecVec = std::vector<TStrVec>;
    using TSizeStringUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;
    using TVector = maths::CBoostedTreeNode::TVector;

public:
    CBoostedTreeInferenceModelBuilder(TStrVec fieldNames,
                                      std::size_t dependentVariableColumnIndex,
                                      const TStrVecVec& categoryNames);
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
    virtual CInferenceModelDefinition&& build();

protected:
    CInferenceModelDefinition& definition();

private:
    using TOneHotEncodingUPtr = std::unique_ptr<COneHotEncoding>;
    using TOneHotEncodingUMap = std::unordered_map<std::string, TOneHotEncodingUPtr>;
    using TStringDoubleUMap = std::unordered_map<std::string, double>;

private:
    virtual void setTargetType() = 0;
    virtual void setAggregateOutput(CEnsemble* ensemble) const = 0;
    TStringDoubleUMap encodingMap(std::size_t inputColumnIndex, const TDoubleVec& map_);

private:
    CInferenceModelDefinition m_Definition;
    TStrVecVec m_CategoryNames;
    TOneHotEncodingUMap m_OneHotEncodingMaps;
    TStrVec m_FieldNames;
    TStrVec m_FeatureNames;
};

class API_EXPORT CRegressionInferenceModelBuilder final : public CBoostedTreeInferenceModelBuilder {
public:
    CRegressionInferenceModelBuilder(const TStrVec& fieldNames,
                                     std::size_t dependentVariableColumnIndex,
                                     const TStrVecVec& categoryNames);
    void addProbabilityAtWhichToAssignClassOne(double probability) override;

private:
    void setTargetType() override;
    void setAggregateOutput(CEnsemble* ensemble) const override;
};

class API_EXPORT CClassificationInferenceModelBuilder final
    : public CBoostedTreeInferenceModelBuilder {
public:
    CClassificationInferenceModelBuilder(const TStrVec& fieldNames,
                                         std::size_t dependentVariableColumnIndex,
                                         const TStrVecVec& categoryNames);
    ~CClassificationInferenceModelBuilder() override = default;
    void addProbabilityAtWhichToAssignClassOne(double probability) override;

private:
    void setTargetType() override;
    void setAggregateOutput(CEnsemble* ensemble) const override;
};
}
}

#endif // INCLUDED_ml_api_CBoostedTreeInferenceModelBuilder_h
