/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelDefinition_h
#define INCLUDED_ml_api_CInferenceModelDefinition_h

#include <core/CRapidJsonLineWriter.h>

#include <maths/CDataFrameCategoryEncoder.h>

#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <string>
#include <unordered_map>
#include <vector>

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

//! Abstract class for output aggregation.
class API_EXPORT CAggregateOutput : public CSerializableToJson {
public:
    //! Aggregation type as a string.
    virtual const std::string& stringType() = 0;
};

//! Allows to use (weighted) majority vote for classification.
class API_EXPORT CWeightedMode : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

public:
    //! Construct with the \param weight vector.
    explicit CWeightedMode(TDoubleVec&& weights);
    //! Construct with a weight vector of \param size with all entries equal to \param weight.
    CWeightedMode(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    const std::string& stringType() override;

private:
    std::vector<double> m_Weights;
};

//! Allows to use (weighted) sum for regression.
class API_EXPORT CWeightedSum : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

public:
    //! Construct with the \param weight vector.
    explicit CWeightedSum(TDoubleVec&& weights);
    //! Construct with a weight vector of \param size with all entries equal to \param weight.
    CWeightedSum(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    const std::string& stringType() override;

private:
    std::vector<double> m_Weights;
};

//! List of support numeric relationships. It's only "<=" at the moment.
enum ENumericRelationship { E_LTE };

class API_EXPORT CTrainedModel : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;
    using TStringVecOptional = boost::optional<TStringVec>;

    enum ETargetType { E_Classification, E_Regression };

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    const TStringVec& featureNames() const;
    //! Names of the features used by the model.
    virtual void featureNames(const TStringVec& featureNames);
    //! Sets target type (regression or classification)
    virtual void targetType(ETargetType targetType);
    //! Returns target type (regression or classification)
    virtual ETargetType targetType() const;

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
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Aggregation mechanism for the output from individual models.
    void aggregateOutput(TAggregateOutputUPtr&& aggregateOutput);
    const TAggregateOutputUPtr& aggregateOutput() const;
    void featureNames(const TStringVec& featureNames) override;
    //! List of trained models withing this ensemble.
    TTreeVec& trainedModels();
    //! Number of models in the ensemble.
    std::size_t size() const;

    void targetType(ETargetType targetType) override;

    ETargetType targetType() const override;

private:
    TTreeVec m_TrainedModels;
    TAggregateOutputUPtr m_AggregateOutput;
};

//!\brief Information related to the input.
class API_EXPORT CInput : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! List of the column names.
    const TStringVec& columns() const;
    //! List of the column names.
    void fieldNames(const TStringVec& columns);

private:
    //! List of the column names.
    TStringVec m_FieldNames;
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
    using TStringDoubleUMap = const std::unordered_map<std::string, double>;

public:
    CFrequencyEncoding(const std::string& field,
                       const std::string& featureName,
                       const TStringDoubleUMap& frequencyMap);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Feature name after pre-processing.
    const std::string& featureName() const;
    //! Map from the category names to the frequency values.
    const TStringDoubleUMap& frequencyMap() const;
    const std::string& typeString() const override;

private:
    std::string m_FeatureName;
    TStringDoubleUMap m_FrequencyMap;
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
    using TStringDoubleUMap = std::unordered_map<std::string, double>;

public:
    CTargetMeanEncoding(const std::string& field,
                        double defaultValue,
                        const std::string& featureName,
                        TStringDoubleUMap&& targetMap);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) override;
    //! Value for categories that have not been seen before.
    double defaultValue() const;
    //! Feature name after pre-processing.
    const std::string& featureName() const;
    //! Map from the category names to the target values.
    const TStringDoubleUMap& targetMap() const;
    const std::string& typeString() const override;

private:
    double m_DefaultValue;
    std::string m_FeatureName;
    TStringDoubleUMap m_TargetMap;
};

//! \brief Technical details required for model evaluation.
class CInferenceModelDefinition {

public:
    using TStringVec = std::vector<std::string>;
    using TApiEncodingUPtr = std::unique_ptr<api::CEncoding>;
    using TApiEncodingUPtrVec = std::vector<TApiEncodingUPtr>;
    using TStringSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStringSizeUMapVec = std::vector<TStringSizeUMap>;
    using TSizeStringUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;

public:
    CInferenceModelDefinition(const TStringVec& fieldNames,
                              const TStringSizeUMapVec& categoryNameMap);
    std::string jsonString();

    void fieldNames(const TStringVec& fieldNames);

    void trainedModel(std::unique_ptr<CTrainedModel>&& trainedModel);
    std::unique_ptr<CTrainedModel>& trainedModel();
    const std::unique_ptr<CTrainedModel>& trainedModel() const;
    const TStringSizeUMapVec& categoryNameMap() const;
    const CInput& input() const;
    TApiEncodingUPtrVec& preprocessors();
    void categoryNameMap(const TStringSizeUMapVec& categoryNameMap);
    const std::string& typeString() const;
    void typeString(const std::string& typeString);

private:
    //! Information related to the input.
    CInput m_Input;
    //! Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
    TApiEncodingUPtrVec m_Preprocessors;
    //! Details of the model evaluation step with a trained_model.
    std::unique_ptr<CTrainedModel> m_TrainedModel;
    TStringVec m_FieldNames;
    TStringSizeUMapVec m_CategoryNameMap;
    TSizeStringUMapVec m_ReverseCategoryNameMap;
    std::string m_TypeString;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelDefinition_h
