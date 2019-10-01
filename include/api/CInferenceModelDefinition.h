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
#include <unordered_map>
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
public:
    //! Populate the object from serialized data
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    ENumericRelationship m_DecisionType = E_LTE;
    bool m_DefaultLeft;
    boost::optional<std::size_t> m_LeftChild;
    double m_LeafValue;
    boost::optional<std::size_t> m_RightChild;
    std::size_t m_SplitFeature;
    boost::optional<double> m_SplitGain;
    std::size_t m_NodeIndex;
    double m_Threshold;
};

/**
 * Details of the model evaluation step.
 */
class CBasicEvaluator : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;
    using TStringVecOptional = boost::optional<TStringVec>;

    enum ETargetType { E_Classification, E_Regression };

public:
    const TStringVec& featureNames() const;

    virtual void featureNames(const TStringVec& featureNames);
    //! Populate the object from serialized data
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    void classificationLabels(const TStringVec& classificationLabels);

    void targetType(ETargetType targetType);

private:
    TStringVec m_FeatureNames;
    TStringVecOptional m_ClassificationLabels;
    ETargetType m_TargetType;
};

class CTree : public CBasicEvaluator {
public:
    using TTreeNodeVec = std::vector<CTreeNode>;

public:
    //! Populate the object from serialized data
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    TTreeNodeVec m_TreeStructure;
};

class CEnsemble : public CBasicEvaluator {
public:
    using TAggregateOutputUPtr = std::unique_ptr<CAggregateOutput>;
    using TTreeVec = std::vector<CTree>;

public:
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    void featureNames(const TStringVec& featureNames) override;

private:
    TTreeVec m_TrainedModels;
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
    const TStringVec& columns() const;
    void columns(const TStringVec& columns);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    TStringVec m_Columns;
};

class CEncoding : public CSerializableToJson {
public:
    void field(const std::string& field);

    CEncoding(const std::string& field);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    virtual const std::string& typeString() const = 0;

private:
    std::string m_Field;
};

/**
 * Mapping from categorical columns to numerical values related to categorical value
 * distribution
 */
class CFrequencyEncoding : public CEncoding {
public:
    CFrequencyEncoding(const std::string& field,
                       const std::string& featureName,
                       const std::map<std::string, double>& frequencyMap);

    void featureName(const std::string& featureName);

    void frequencyMap(const std::map<std::string, double>& frequencyMap);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

    const std::string& typeString() const override;

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
    using TStringStringUMap = std::map<std::string, std::string>;

public:
    COneHotEncoding(const std::string& field, const TStringStringUMap& hotMap);

    TStringStringUMap& hotMap();
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

    const std::string& typeString() const override;

private:
    /**
     * Map from the category names of the original field to the new field names.
     */
    TStringStringUMap m_HotMap;
};

/**
 * Mapping from categorical columns to numerical values related to the target value
 */
class CTargetMeanEncoding : public CEncoding {
public:
    CTargetMeanEncoding(const std::string& field,
                        double defaultValue,
                        const std::string& featureName,
                        const std::map<std::string, double>& targetMap);

    void defaultValue(double defaultValue);
    void featureName(const std::string& featureName);

    const std::string& typeString() const override;

    void targetMap(const std::map<std::string, double>& targetMap);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

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
    using TStrSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapVec = std::vector<TStrSizeUMap>;
    using TSizeStrUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStrUMapVec = std::vector<TSizeStrUMap>;

public:
    std::string jsonString();
    rapidjson::Value&& jsonObject();

    void fieldNames(const TStringVec& fieldNames);
    void encodings(const TEncodingUPtrVec& encodings);
    void trainedModel(std::unique_ptr<CBasicEvaluator>&& trainedModel);

    const TStrSizeUMapVec& categoryNameMap() const;
    void categoryNameMap(const TStrSizeUMapVec& categoryNameMap);

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
    TStrSizeUMapVec m_CategoryNameMap;
    TSizeStrUMapVec m_ReverseCategoryNameMap;
};
}
}

#endif //INCLUDED_ml_api_SInferenceModelDefinition_h
