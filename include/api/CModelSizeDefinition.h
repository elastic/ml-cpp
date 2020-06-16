/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CModelSizeDefinition_h
#define INCLUDED_ml_api_CModelSizeDefinition_h

#include <api/CInferenceModelDefinition.h>
#include <api/ImportExport.h>
#include <bits/c++config.h>
#include <memory>
#include <vector>

namespace ml {
namespace api {

//! TODO.
class API_EXPORT CNumOutputProcessorWeights : public CSerializableToJson {
public:
    using TDoubleVec = std::vector<double>;

public:
    //! Construct with the \p weights vector.
    explicit CNumOutputProcessorWeights(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CNumOutputProcessorWeights(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    ~CNumOutputProcessorWeights() override = default;
};

class API_EXPORT CTrainedModelSize : public CSerializableToJson {
public:
    ~CTrainedModelSize() override = default;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    virtual void numClasses(std::size_t numClasses);
    virtual std::size_t numClasses() const;
    virtual void numClassificationWeights(std::size_t numClassificationWeights);
    virtual std::size_t classificationWeights() const;

private:
    std::size_t m_numClasses;
    std::size_t m_numClassificationWeights;
    std::size_t m_numOutputProcessorWeights;
    std::size_t m_numOperations;
};

//!
class API_EXPORT CTreeSize final : public CSerializableToJson {
public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    void numNodes(std::size_t numNodes) const;
    std::size_t numNodes() const;
    void numLeaves(std::size_t numLeaves) const;
    std::size_t numLeaves() const;

private:
    std::size_t m_NumNodes;
    std::size_t m_NumLeaves;
};

//!
class API_EXPORT CEnsembleModelSize final : public CTrainedModelSize {
public:
    using TTreeSizeUPtr = std::unique_ptr<CTreeSize>;
    using TTreeSizeUPtrVec = std::vector<TTreeSizeUPtr>;

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;

    TTreeSizeUPtrVec& trainedModels();

    std::size_t size() const;

private:
    TTreeSizeUPtrVec m_TrainedModels;
};

class API_EXPORT CEncodingSize : public CSerializableToJson {
public:
    ~CEncodingSize() override = default;
    explicit CEncodingSize(std::string field);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;

private:
    std::size_t m_FieldLength;
    std::size_t m_FieldValueLengths;
};

//! \brief
class API_EXPORT CFrequencyEncodingSize final : public CEncodingSize {
public:
    ~CFrequencyEncodingSize() override = default;
    CFrequencyEncodingSize();
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;

private:
    std::size_t m_FeatureNameLength;
};

//! \brief
class API_EXPORT COneHotEncodingSize final : public CEncodingSize {
public:
    using TStringStringUMap = std::map<std::string, std::string>;

public:
    ~COneHotEncodingSize() override = default;
    COneHotEncodingSize();
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;

private:
    std::size_t m_FeatureNameLengths;
};

//! \brief
class API_EXPORT CTargetMeanEncodingSize final : public CEncodingSize {
public:
    ~CTargetMeanEncodingSize() override = default;
    CTargetMeanEncodingSize();
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;

private:
    std::string m_FeatureNameLength;
};

//! \brief
class API_EXPORT CModelSizeDefinition : public CSerializableToJson {
public:
    explicit CModelSizeDefinition(const CInferenceModelDefinition& inferenceModel);
    ~CModelSizeDefinition() override = default;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    std::string jsonString();

private:
    using TEncodingSizeUPtr = std::unique_ptr<CEncoding::CSizeInfo>;
    using TEncodingSizeUPtrVec = std::vector<TEncodingSizeUPtr>;
    using TTrainedModelSizeUPtr = std::unique_ptr<CTrainedModel::CSizeInfo>;

private:
    TEncodingSizeUPtrVec m_EncodingSizeItems;
    TTrainedModelSizeUPtr m_TrainedModelSize;
};
}
}

#endif //INCLUDED_ml_api_CModelSizeDefinition_h
