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

#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// TODO add documentation

namespace ml {
namespace api {

//! \brief Abstract class for all elements the the inference definition
//! that can will be serialized into JSON.
class API_EXPORT CSerializableToJson {
public:
    using TRapidJsonWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

public:
    //! Serialize the object as JSON items under the \p parentObject using the specified \p writer.
    virtual void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) = 0;
};

//! \brief Abstract class for all elements that initialize their member variables
//! from JSON string of the AnalysisRunner.
class API_EXPORT CDeserializableFromJson {
public:
    //! Initialize member variable using \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;
};

//! Abstract class for output aggregation.
class API_EXPORT CAggregateOutput : public CSerializableToJson {
public:
    //! Aggregation type as a string.
    virtual const std::string& stringType() = 0;
};

//! Allows to use (weighted) majority vote for classification.
class API_EXPORT CWeightedMode : public CAggregateOutput {
public:
    explicit CWeightedMode(const std::vector<double>& weights);
    CWeightedMode(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    const std::string& stringType() override;

private:
    std::vector<double> m_Weights;
};

//! Allows to use (weighted) sum for regression.
class API_EXPORT CWeightedSum : public CAggregateOutput {
public:
    explicit CWeightedSum(const std::vector<double>& weights);
    CWeightedSum(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    const std::string& stringType() override;

private:
    std::vector<double> m_Weights;
};

//! List of support numeric relationships. It's only "<=" at the moment.
enum ENumericRelationship { E_LTE };

class API_EXPORT CTrainedModel : public CSerializableToJson, public CDeserializableFromJson {
public:
    using TStringVec = std::vector<std::string>;
    using TStringVecOptional = boost::optional<TStringVec>;

    enum ETargetType { E_Classification, E_Regression };

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    const TStringVec& featureNames() const;
    //! Names of the features used by the model.
    virtual void featureNames(const TStringVec& featureNames);
    void targetType(ETargetType targetType);
    ETargetType targetType() const;

private:
    TStringVecOptional m_ClassificationLabels;
    TStringVec m_FeatureNames;
    ETargetType m_TargetType;
};

//! Classification and regression trees.
class API_EXPORT CTree : public CTrainedModel {
public:
    class CTreeNode : public CSerializableToJson {
    public:
        using TOptionalSize = boost::optional<std::size_t>;
        using TOptionalDouble = boost::optional<double>;

    public:
        CTreeNode(size_t nodeIndex,
                  double threshold,
                  bool defaultLeft,
                  double leafValue,
                  size_t splitFeature,
                  const TOptionalSize& leftChild,
                  const TOptionalSize& rightChild,
                  const TOptionalDouble& splitGain);

        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
        void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;

    private:
        std::size_t m_NodeIndex;
        double m_Threshold;
        ENumericRelationship m_DecisionType = E_LTE;
        bool m_DefaultLeft;
        double m_LeafValue;
        std::size_t m_SplitFeature;
        TOptionalSize m_LeftChild;
        TOptionalSize m_RightChild;
        TOptionalDouble m_SplitGain;
    };
    using TTreeNodeVec = std::vector<CTreeNode>;

public:
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Total number of tree nodes.
    std::size_t size() const;

    TTreeNodeVec& treeStructure();

private:
    TTreeNodeVec m_TreeStructure;
};

//! Ensemble of a collection of trained models
// TODO this should be a list of basic evaluators, not the list of trees.
class API_EXPORT CEnsemble : public CTrainedModel {
public:
    using TAggregateOutputUPtr = std::unique_ptr<CAggregateOutput>;
    using TTreeVec = std::vector<CTree>;

public:
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Aggregation mechanism for the output from individual models.
    void aggregateOutput(TAggregateOutputUPtr&& aggregateOutput);
    const TAggregateOutputUPtr& aggregateOutput() const;
    void featureNames(const TStringVec& featureNames) override;
    //! List of trained models withing this ensemble.
    TTreeVec& trainedModels();
    //! Number of models in the ensemble.
    std::size_t size() const;

private:
    TTreeVec m_TrainedModels;
    TAggregateOutputUPtr m_AggregateOutput;
};

//!\brief Information related to the input.
class API_EXPORT CInput : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;
    using TStringVecOptional = boost::optional<TStringVec>;

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! List of the column names.
    const TStringVec& columns() const;
    //! List of the column names.
    void columns(const TStringVec& columns);

private:
    //! List of the column names.
    TStringVec m_Columns;
};

class API_EXPORT CEncoding : public CSerializableToJson {
public:
    explicit CEncoding(const std::string& field);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Input field name. Must be defined in the input section.
    void field(const std::string& field);
    //! Encoding type as string.
    virtual const std::string& typeString() const = 0;

private:
    //! Input field name. Must be defined in the input section.
    std::string m_Field;
};

//! \brief Mapping from categorical columns to numerical values related to categorical value distribution.
class API_EXPORT CFrequencyEncoding : public CEncoding {
public:
    CFrequencyEncoding(const std::string& field,
                       const std::string& featureName,
                       const std::map<std::string, double>& frequencyMap);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Feature name after pre-processing.
    const std::string& featureName() const;
    //! Map from the category names to the frequency values.
    const std::map<std::string, double>& frequencyMap() const;
    const std::string& typeString() const override;

private:
    std::string m_FeatureName;
    std::map<std::string, double> m_FrequencyMap;
};

//! \brief Application of the one-hot encoding function on a single column.
class API_EXPORT COneHotEncoding : public CEncoding {
public:
    using TStringStringUMap = std::map<std::string, std::string>;

public:
    COneHotEncoding(const std::string& field, const TStringStringUMap& hotMap);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Map from the category names of the original field to the new field names.
    TStringStringUMap& hotMap();
    const std::string& typeString() const override;

private:
    TStringStringUMap m_HotMap;
};

//! \brief Mapping from categorical columns to numerical values related to the target value.
class API_EXPORT CTargetMeanEncoding : public CEncoding {
public:
    CTargetMeanEncoding(const std::string& field,
                        double defaultValue,
                        const std::string& featureName,
                        std::map<std::string, double>&& targetMap);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Value for categories that have not been seen before.
    double defaultValue() const;
    //! Feature name after pre-processing.
    const std::string& featureName() const;
    //! Map from the category names to the target values.
    const std::map<std::string, double>& targetMap() const;
    const std::string& typeString() const override;

private:
    double m_DefaultValue;
    std::string m_FeatureName;
    std::map<std::string, double> m_TargetMap;
};

//! \brief Technical details required for model evaluation.
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
    CInferenceModelDefinition(const TStringVec& fieldNames,
                              const TStrSizeUMapVec& categoryNameMap);
    std::string jsonString();
    rapidjson::Value&& jsonObject();
    void fieldNames(const TStringVec& fieldNames);
    void encodings(const TEncodingUPtrVec& encodings);
    void trainedModel(std::unique_ptr<CTrainedModel>&& trainedModel);
    std::unique_ptr<CTrainedModel>& trainedModel();
    const std::unique_ptr<CTrainedModel>& trainedModel() const;
    const TStrSizeUMapVec& categoryNameMap() const;
    const CInput& input() const;
    TApiEncodingUPtrVec& preprocessing();
    void categoryNameMap(const TStrSizeUMapVec& categoryNameMap);
    const std::string& typeString() const;
    void typeString(const std::string& typeString);

private:
    //! Information related to the input.
    CInput m_Input;
    //! Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
    TApiEncodingUPtrVec m_Preprocessing;
    //! Details of the model evaluation step with a trained_model.
    std::unique_ptr<CTrainedModel> m_TrainedModel;
    TStringVec m_FieldNames;
    TStrSizeUMapVec m_CategoryNameMap;
    TSizeStrUMapVec m_ReverseCategoryNameMap;
    std::string m_TypeString;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelDefinition_h
