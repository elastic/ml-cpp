/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelDefinition_h
#define INCLUDED_ml_api_CInferenceModelDefinition_h

#include <core/CRapidJsonConcurrentLineWriter.h>

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
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    virtual ~CSerializableToJson() = default;
    //! Serialize the object as JSON items under the \p parentObject using the specified \p writer.
    virtual void addToDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const = 0;
};

//! Abstract class for output aggregation.
class API_EXPORT CAggregateOutput : public CSerializableToJson {
public:
    //! Aggregation type as a string.
    virtual const std::string& stringType() const = 0;
    ~CAggregateOutput() override = default;
};

//! Allows to use (weighted) majority vote for classification.
class API_EXPORT CWeightedMode final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

public:
    ~CWeightedMode() override = default;
    //! Construct with the \p weights vector.
    explicit CWeightedMode(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CWeightedMode(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! Allows to use (weighted) sum for regression.
class API_EXPORT CWeightedSum final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

public:
    ~CWeightedSum() override = default;
    //! Construct with the \p weights vector.
    explicit CWeightedSum(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CWeightedSum(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! Allows to use logistic regression aggregation.
//!
//! Given a weights vector $\vec{w}$ as a parameter and an output vector from the ensemble $\vec{x}$,
//! it computes the logistic regression function \f$1/(1 + \exp(-\vec{w}^T \vec{x}))\f$.
class API_EXPORT CLogisticRegression final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

public:
    ~CLogisticRegression() override = default;
    //! Construct with the \p weights vector.
    explicit CLogisticRegression(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CLogisticRegression(std::size_t size, double weight);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! List of support numeric relationships. It's only "<" at the moment.
enum ENumericRelationship { E_LT };

class API_EXPORT CTrainedModel : public CSerializableToJson {
public:
    using TDoubleVec = std::vector<double>;
    using TStringVec = std::vector<std::string>;
    using TOptionalDoubleVec = boost::optional<TDoubleVec>;
    using TOptionalStringVec = boost::optional<TStringVec>;

    enum ETargetType { E_Classification, E_Regression };

public:
    ~CTrainedModel() override = default;
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    //! Names of the features used by the model.
    virtual const TStringVec& featureNames() const;
    virtual TStringVec& featureNames();
    //! Names of the features used by the model.
    virtual void featureNames(const TStringVec& featureNames);
    virtual void featureNames(TStringVec&& featureNames);
    //! Sets target type (regression or classification).
    virtual void targetType(ETargetType targetType);
    //! Returns target type (regression or classification).
    virtual ETargetType targetType() const;
    //! Adjust the feature names, e.g. to exclude not used feature names like the target column.
    virtual TStringVec removeUnusedFeatures() = 0;
    //! Set the labels to use for each class.
    virtual void classificationLabels(const TStringVec& classificationLabels);
    //! Get the labels to use for each class.
    virtual const TOptionalStringVec& classificationLabels() const;
    //! Set weights by which to multiply classes when doing label assignment.
    virtual void classificationWeights(const TDoubleVec& classificationWeights);
    //! Get weights by which to multiply classes when doing label assignment.
    virtual const TOptionalDoubleVec& classificationWeights() const;

private:
    TStringVec m_FeatureNames;
    ETargetType m_TargetType;
    TOptionalStringVec m_ClassificationLabels;
    TOptionalDoubleVec m_ClassificationWeights;
};

//! Classification and regression trees.
class API_EXPORT CTree final : public CTrainedModel {
public:
    class CTreeNode : public CSerializableToJson {
    public:
        using TNodeIndex = std::uint32_t;
        using TOptionalNodeIndex = boost::optional<TNodeIndex>;
        using TOptionalDouble = boost::optional<double>;

    public:
        CTreeNode(TNodeIndex nodeIndex,
                  double threshold,
                  bool defaultLeft,
                  double leafValue,
                  size_t splitFeature,
                  const TOptionalNodeIndex& leftChild,
                  const TOptionalNodeIndex& rightChild,
                  const TOptionalDouble& splitGain);

        void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
        size_t splitFeature() const;
        void splitFeature(size_t splitFeature);
        bool leaf() const;

    private:
        bool m_DefaultLeft;
        ENumericRelationship m_DecisionType = E_LT;
        TNodeIndex m_NodeIndex;
        TOptionalNodeIndex m_LeftChild;
        TOptionalNodeIndex m_RightChild;
        std::size_t m_SplitFeature;
        double m_Threshold;
        double m_LeafValue;
        TOptionalDouble m_SplitGain;
    };
    using TTreeNodeVec = std::vector<CTreeNode>;

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    //! Total number of tree nodes.
    std::size_t size() const;
    TStringVec removeUnusedFeatures() override;
    TTreeNodeVec& treeStructure();

private:
    TTreeNodeVec m_TreeStructure;
};

//! Ensemble of a collection of trained models
class API_EXPORT CEnsemble final : public CTrainedModel {
public:
    using TAggregateOutputUPtr = std::unique_ptr<CAggregateOutput>;
    using TTrainedModelUPtr = std::unique_ptr<CTrainedModel>;
    using TTrainedModelUPtrVec = std::vector<TTrainedModelUPtr>;

public:
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    //! Aggregation mechanism for the output from individual models.
    void aggregateOutput(TAggregateOutputUPtr&& aggregateOutput);
    const TAggregateOutputUPtr& aggregateOutput() const;
    const TStringVec& featureNames() const override;
    void featureNames(const TStringVec& featureNames) override;
    //! List of trained models withing this ensemble.
    TTrainedModelUPtrVec& trainedModels();
    //! Number of models in the ensemble.
    std::size_t size() const;
    TStringVec removeUnusedFeatures() override;
    void targetType(ETargetType targetType) override;
    //! Set the labels to use for each class.
    void classificationLabels(const TStringVec& classificationLabels) override;
    //! Set weights by which to multiply classes when doing label assignment.
    void classificationWeights(const TDoubleVec& classificationWeights) override;
    using CTrainedModel::classificationLabels;
    using CTrainedModel::classificationWeights;
    using CTrainedModel::targetType;

private:
    TTrainedModelUPtrVec m_TrainedModels;
    TAggregateOutputUPtr m_AggregateOutput;
};

class API_EXPORT CEncoding : public CSerializableToJson {
public:
    ~CEncoding() override = default;
    explicit CEncoding(std::string field);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    //! Input field name. Must be defined in the input section.
    void field(const std::string& field);
    const std::string& field() const;
    //! Encoding type as string.
    virtual const std::string& typeString() const = 0;

private:
    //! Input field name. Must be defined in the input section.
    std::string m_Field;
};

//! \brief Mapping from categorical columns to numerical values related to categorical value distribution.
class API_EXPORT CFrequencyEncoding final : public CEncoding {
public:
    using TStringDoubleUMap = const std::unordered_map<std::string, double>;

public:
    ~CFrequencyEncoding() override = default;
    CFrequencyEncoding(const std::string& field, std::string featureName, TStringDoubleUMap frequencyMap);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
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
class API_EXPORT COneHotEncoding final : public CEncoding {
public:
    using TStringStringUMap = std::map<std::string, std::string>;

public:
    ~COneHotEncoding() override = default;
    COneHotEncoding(const std::string& field, TStringStringUMap hotMap);
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    //! Map from the category names of the original field to the new field names.
    TStringStringUMap& hotMap();
    const std::string& typeString() const override;

private:
    TStringStringUMap m_HotMap;
};

//! \brief Mapping from categorical columns to numerical values related to the target value.
class API_EXPORT CTargetMeanEncoding final : public CEncoding {
public:
    using TStringDoubleUMap = std::unordered_map<std::string, double>;

public:
    ~CTargetMeanEncoding() override = default;
    CTargetMeanEncoding(const std::string& field,
                        double defaultValue,
                        std::string featureName,
                        TStringDoubleUMap&& targetMap);

    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
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
class API_EXPORT CInferenceModelDefinition : public CSerializableToJson {
public:
    using TStringVec = std::vector<std::string>;
    using TApiEncodingUPtr = std::unique_ptr<api::CEncoding>;
    using TApiEncodingUPtrVec = std::vector<TApiEncodingUPtr>;
    using TStringSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStringSizeUMapVec = std::vector<TStringSizeUMap>;
    using TSizeStringUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;

public:
    TApiEncodingUPtrVec& preprocessors();
    void trainedModel(std::unique_ptr<CTrainedModel>&& trainedModel);
    std::unique_ptr<CTrainedModel>& trainedModel();
    void addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const override;
    std::string jsonString();
    void fieldNames(TStringVec&& fieldNames);
    const TStringVec& fieldNames() const;
    const std::string& typeString() const;
    void typeString(const std::string& typeString);
    std::size_t dependentVariableColumnIndex() const;
    void dependentVariableColumnIndex(size_t dependentVariableColumnIndex);

private:
    //! Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
    TApiEncodingUPtrVec m_Preprocessors;
    //! Details of the model evaluation step with a trained_model.
    std::unique_ptr<CTrainedModel> m_TrainedModel;
    TStringVec m_FieldNames;
    std::string m_TypeString;
    std::size_t m_DependentVariableColumnIndex;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelDefinition_h
