/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_SInferenceModelDefinition_h
#define INCLUDED_ml_api_SInferenceModelDefinition_h

#include <core/CRapidJsonLineWriter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CDataFrameCategoryEncoder.h>

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
public:
    void field(const std::string& field);

private:
    /**
     * Input field name
     */
    std::string m_Field;
};

/**
 * Mapping from categorical columns to numerical values related to categorical value
 * distribution
 */
class CFrequencyEncoding : public CEncoding {
public:
    void featureName(const std::string& featureName);

    void frequencyMap(const std::map<std::string, double>& frequencyMap);

private:
    /**
     * Feature name after pre-processing
     */
    std::string m_FeatureName;
    /**
     * Map from the category names to the frequency values.
     */
    std::map<std::string, double> m_FrequencyMap;
};

/**
 * Application of the one-hot encoding function on a single column.
 */
class COneHotEncoding : public CEncoding {
public:
    void hotMap(const std::map<std::string, std::string>& hotMap);

private:
    /**
     * Map from the category names of the original field to the new field names.
     */
    std::map<std::string, std::string> m_HotMap;
};

/**
 * Mapping from categorical columns to numerical values related to the target value
 */
class CTargetMeanEncoding : public CEncoding {
public:
    void defaultValue(double defaultValue);

    void featureName(const std::string& featureName);

    void targetMap(const std::map<std::string, double>& targetMap);

private:
    /**
     * Value for categories that have not been seen before
     */
    double m_DefaultValue;
    /**
     * Feature name after pre-processing
     */
    std::string m_FeatureName;
    /**
     * Map from the category names to the target values.
     */
    std::map<std::string, double> m_TargetMap;
};

/**
 * Technical details required for model evaluation.
 */
class CInferenceModelDefinition {

public:
    using TJsonDocument = rapidjson::Document;
    using TStringVec = std::vector<std::string>;
    using TEncodingUPtr = std::unique_ptr<maths::CDataFrameCategoryEncoder::CEncoding>;
    using TEncodingUPtrVec = std::vector<TEncodingUPtr>;
    using TApiEncodingUPtr = std::unique_ptr<api::CEncoding>;
    using TApiEncodingUPtrVec = std::vector<TApiEncodingUPtr>;

public:
    std::string jsonString();

    void fieldNames(const TStringVec& fieldNames);
    void encodings(const TEncodingUPtrVec& encodings);

private:
    /**
 * Information related to the input.
 */
    CInput m_Input;
    /**
     * Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
     */
    TApiEncodingUPtrVec m_Preprocessing;
    /**
     * Details of the model evaluation step.
     */
    std::unique_ptr<CBasicEvaluator> m_TrainedModel;

    TStringVec m_FieldNames;
};
}
}

#endif //INCLUDED_ml_api_SInferenceModelDefinition_h
