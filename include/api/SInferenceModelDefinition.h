/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_SInferenceModelDefinition_h
#define INCLUDED_ml_api_SInferenceModelDefinition_h


#include <core/CStateRestoreTraverser.h>
#include <core/CRapidJsonLineWriter.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <map>
#include <string>
#include <vector>

namespace ml {
namespace api {

struct SSerializableToJson {
    using TJsonArray = rapidjson::Document::Array;
    using TJsonObject = rapidjson::Document::Object;
    using TJsonAllocator = rapidjson::Document::AllocatorType;
    using TJsonValue = rapidjson::Document::GenericValue;
    using TRapidJsonWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;
    virtual void addToDocument(rapidjson::Value &parentObject, TRapidJsonWriter &writer) = 0;
};

struct SAggregateOutput {};

/**
 * Allows to used (weighted) majority vote for classification.
 */
struct SWeightedMode : public SAggregateOutput {
    std::vector<double> weights;
};

struct SWeightedSum : public SAggregateOutput {
    std::vector<double> weights;
};

enum ENumericRelationship { E_LTE };

struct STreeNode : public SSerializableToJson{
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
    void addToDocument(rapidjson::Value &parentObject, TRapidJsonWriter &writer) override;
};

/**
 * Details of the model evaluation step.
 */
struct SBasicEvaluator : public SSerializableToJson{
    std::vector<std::string> m_FeatureNames;

    //! Populate the object from serialized data
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    virtual void addToDocument(rapidjson::Value &parentObject, TRapidJsonWriter &writer);
};

struct STree : public SBasicEvaluator {
    std::vector<STreeNode> m_TreeStructure;
    //! Populate the object from serialized data
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value &parentObject, TRapidJsonWriter &writer) override;
};

struct SEnsemble : public SBasicEvaluator  {
    std::vector<STree> m_TrainedModels;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    std::unique_ptr<SAggregateOutput> m_AggregateOutput;
    void addToDocument(rapidjson::Value &parentObject, TRapidJsonWriter &writer) override;
};

/**
 * Information related to the input.
 */
struct SInput {
    /**
     * List of the column names.
     */
    std::shared_ptr<std::vector<std::string>> columns;
};

struct SEncoding {
    /**
     * Input field name
     */
    std::string field;
};

/**
 * Mapping from categorical columns to numerical values related to categorical value
 * distribution
 */
struct SFrequencyEncoding : public SEncoding {
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
struct SOneHotEncoding : public SEncoding {
    /**
     * Map from the category names of the original field to the new field names.
     */
    std::map<std::string, std::string> hotMap;
};

/**
 * Mapping from categorical columns to numerical values related to the target value
 */
struct STargetMeanEncoding : public SEncoding {
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
struct SInferenceModelDefinition {

    using TJsonDocument = rapidjson::Document;

    /**
     * Information related to the input.
     */
    SInput m_Input;
    /**
     * Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
     */
    boost::optional<std::vector<SEncoding*>> m_Preprocessing;
    /**
     * Details of the model evaluation step.
     */
    std::unique_ptr<SBasicEvaluator> m_TrainedModel;

    std::string jsonString();
};
}
}

#endif //INCLUDED_ml_api_SInferenceModelDefinition_h
