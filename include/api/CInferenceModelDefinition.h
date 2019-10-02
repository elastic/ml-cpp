/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelDefinition_h
#define INCLUDED_ml_api_CInferenceModelDefinition_h

#include <core/CRapidJsonLineWriter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CDataFrameCategoryEncoder.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// TODO add documentation

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

class CAggregateOutput : public CSerializableToJson {
    virtual const std::string& stringType() = 0;
};

class CWeightedMode : public CAggregateOutput {
public:
    explicit CWeightedMode(const std::vector<double>& weights);
    CWeightedMode(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    const std::string& stringType() override;
    std::vector<double> m_Weights;
};

class CWeightedSum : public CAggregateOutput {
public:
    explicit CWeightedSum(const std::vector<double>& weights);
    CWeightedSum(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    const std::string& stringType() override;
    std::vector<double> m_Weights;
};

enum ENumericRelationship { E_LTE };

class CTreeNode : public CSerializableToJson {
public:
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

class CBasicEvaluator : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;
    using TStringVecOptional = boost::optional<TStringVec>;

    enum ETargetType { E_Classification, E_Regression };

public:
    const TStringVec& featureNames() const;

    virtual void featureNames(const TStringVec& featureNames);
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

    std::size_t size() const;

private:
    TTreeVec m_TrainedModels;

private:
    TAggregateOutputUPtr m_AggregateOutput;

public:
    void aggregateOutput(TAggregateOutputUPtr&& aggregateOutput);
};

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

class CFrequencyEncoding : public CEncoding {
public:
    CFrequencyEncoding(const std::string& field,
                       const std::string& featureName,
                       const std::map<std::string, double>& frequencyMap);

    void featureName(const std::string& featureName);

    void frequencyMap(const std::map<std::string, double>& frequencyMap);

    const std::string& featureName() const;

    const std::map<std::string, double>& frequencyMap() const;

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

    const std::string& typeString() const override;

private:
    std::string m_FeatureName;
    std::map<std::string, double> m_FrequencyMap;
};

class COneHotEncoding : public CEncoding {
public:
    using TStringStringUMap = std::map<std::string, std::string>;

public:
    COneHotEncoding(const std::string& field, const TStringStringUMap& hotMap);

    TStringStringUMap& hotMap();
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

    const std::string& typeString() const override;

private:
    TStringStringUMap m_HotMap;
};

class CTargetMeanEncoding : public CEncoding {
public:
    CTargetMeanEncoding(const std::string& field,
                        double defaultValue,
                        const std::string& featureName,
                        const std::map<std::string, double>& targetMap);

    void defaultValue(double defaultValue);

    double defaultValue() const;

    const std::string& featureName() const;

    const std::map<std::string, double>& targetMap() const;

    void featureName(const std::string& featureName);

    const std::string& typeString() const override;

    void targetMap(const std::map<std::string, double>& targetMap);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

private:
    double m_DefaultValue;
    std::string m_FeatureName;
    std::map<std::string, double> m_TargetMap;
};

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

    const CInput& input() const;

    const TApiEncodingUPtrVec& preprocessing() const;

    CInferenceModelDefinition(const TStringVec& fieldNames,
                              const TStrSizeUMapVec& categoryNameMap);

    void categoryNameMap(const TStrSizeUMapVec& categoryNameMap);
    std::unique_ptr<CBasicEvaluator>& trainedModel();

private:
    CInput m_Input;
    TApiEncodingUPtrVec m_Preprocessing;
    std::unique_ptr<CBasicEvaluator> m_TrainedModel;
    TStringVec m_FieldNames;
    TStrSizeUMapVec m_CategoryNameMap;
    TSizeStrUMapVec m_ReverseCategoryNameMap;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelDefinition_h
