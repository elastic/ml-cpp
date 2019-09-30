/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_SInferenceModelDefinition_h
#define INCLUDED_ml_api_SInferenceModelDefinition_h

#include <core/CRapidJsonLineWriter.h>
#include <core/CStateRestoreTraverser.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <map>
#include <string>
#include <vector>

namespace ml {
namespace api {

class CSerializableToJson {
public:
    using TJsonArray = rapidjson::Document::Array;
    using TJsonObject = rapidjson::Document::Object;
    using TJsonAllocator = rapidjson::Document::AllocatorType;
    using TJsonValue = rapidjson::Document::GenericValue;
    using TRapidJsonWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;
    virtual void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) = 0;
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
};

class CAggregateOutput {};

/**
 * Allows to used (weighted) majority vote for classification.
 */
class SWeightedMode : public CAggregateOutput {
    std::vector<double> weights;
};

class SWeightedSum : public CAggregateOutput {
    std::vector<double> weights;
};

enum ENumericRelationship { E_LTE };

class CTreeNode : public CSerializableToJson {
    ENumericRelationship m_DecisionType = E_LTE;
    bool m_DefaultLeft;
    boost::optional<std::size_t> m_LeftChild;
    double m_LeafValue;
    boost::optional<std::size_t> m_RightChild;
    std::size_t m_SplitFeature;
    boost::optional<double> m_SplitGain;
    std::size_t m_NodeIndex;
    double m_Threshold;

    //! Populate the object from serialized data
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
};

/**
 * Details of the model evaluation step.
 */
class CBasicEvaluator : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;

public:
    //! Populate the object from serialized data
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    virtual void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer);

private:
    TStringVec m_FeatureNames;

public:
    const TStringVec& featureNames() const;

    virtual void featureNames(const TStringVec& featureNames);
};

class CTree : public CBasicEvaluator {
    std::vector<CTreeNode> m_TreeStructure;
    //! Populate the object from serialized data
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
};

class CEnsemble : public CBasicEvaluator {
public:
    using TAggregateOutputUPtr = std::unique_ptr<CAggregateOutput>;

public:
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    void featureNames(const TStringVec& featureNames) override;

private:
    std::vector<CTree> m_TrainedModels;

    TAggregateOutputUPtr m_AggregateOutput;
};

/**
 * Information related to the input.
 */
class CInput : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;
    using TStringVecOptional = boost::optional<TStringVec>;

public:
    const TStringVecOptional& columns() const;
    void columns(const TStringVec& columns);
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    TStringVecOptional m_Columns;
};

class CEncoding {
    /**
     * Input field name
     */
    std::string field;
};

/**
 * Mapping from categorical columns to numerical values related to categorical value
 * distribution
 */
class CFrequencyEncoding : public CEncoding {
    /**
     * Feature name after pre-processing
     */
    std::string featureName;

    /**
     * Map from the category names to the frequency values.
     */
    std::map<std::string, double> frequencyMap;
};

/**
 * Application of the one-hot encoding function on a single column.
 */
class COneHotEncoding : public CEncoding {
    /**
     * Map from the category names of the original field to the new field names.
     */
    std::map<std::string, std::string> hotMap;
};

/**
 * Mapping from categorical columns to numerical values related to the target value
 */
class CTargetMeanEncoding : public CEncoding {
private:
    /**
     * Value for categories that have not been seen before
     */
    double defaultValue;
    /**
     * Feature name after pre-processing
     */
    std::string featureName;
    /**
     * Map from the category names to the target values.
     */
    std::map<std::string, double> targetMap;
};

/**
 * Technical details required for model evaluation.
 */
class CInferenceModelDefinition {

public:
    using TJsonDocument = rapidjson::Document;
    using TStringVec = std::vector<std::string>;

public:
    std::string jsonString();

    void fieldNames(const std::vector<std::string>& fieldNames);

private:
    /**
 * Information related to the input.
 */
    CInput m_Input;
    /**
     * Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
     */
    boost::optional<std::vector<CEncoding*>> m_Preprocessing;
    /**
     * Details of the model evaluation step.
     */
    std::unique_ptr<CBasicEvaluator> m_TrainedModel;
};
}
}

#endif //INCLUDED_ml_api_SInferenceModelDefinition_h
