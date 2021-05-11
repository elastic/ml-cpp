/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelDefinition_h
#define INCLUDED_ml_api_CInferenceModelDefinition_h

#include <maths/CDataFrameCategoryEncoder.h>

#include <api/CSerializableToJson.h>
#include <api/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>

#include <sstream>
#include <string>
#include <vector>

namespace ml {
namespace api {

//! Abstract class for output aggregation.
class API_EXPORT CAggregateOutput : public CSerializableToJsonStream {
public:
    static const std::string JSON_WEIGHTS_TAG;

public:
    //! Aggregation type as a string.
    virtual const std::string& stringType() const = 0;
    ~CAggregateOutput() override = default;
};

//! Allows to use (weighted) majority vote for classification.
class API_EXPORT CWeightedMode final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

    static const std::string JSON_WEIGHTED_MODE_TAG;

public:
    ~CWeightedMode() override = default;
    //! Construct with the \p weights vector.
    explicit CWeightedMode(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CWeightedMode(std::size_t size, double weight);
    void addToJsonStream(TGenericLineWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! Allows to use (weighted) sum for regression.
class API_EXPORT CWeightedSum final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

    static const std::string JSON_WEIGHTED_SUM_TAG;

public:
    ~CWeightedSum() override = default;
    //! Construct with the \p weights vector.
    explicit CWeightedSum(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CWeightedSum(std::size_t size, double weight);
    void addToJsonStream(TGenericLineWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! \brief Logistic regression aggregation.
//!
//! DESCRIPTION:\n
//! Given a weights vector $\vec{w}$ as a parameter and an output vector from
//! the ensemble $\vec{x}$, it computes the logistic regression function
//! \f$1/(1 + \exp(-\vec{w}^T \vec{x}))\f$.
class API_EXPORT CLogisticRegression final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

    static const std::string JSON_LOGISTIC_REGRESSION_TAG;

public:
    ~CLogisticRegression() override = default;
    //! Construct with the \p weights vector.
    explicit CLogisticRegression(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CLogisticRegression(std::size_t size, double weight);
    void addToJsonStream(TGenericLineWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! \brief Exponent aggregation.
//!
//! DESCRIPTION:\n
//! Given a weights vector $\vec{w}$ as a parameter and an output vector from the
//! ensemble $\vec{x}$, it computes the exponent function \f$\exp(\vec{w}^T \vec{x})\f$.
class API_EXPORT CExponent final : public CAggregateOutput {
public:
    using TDoubleVec = std::vector<double>;

    static const std::string JSON_EXPONENT_TAG;

public:
    ~CExponent() override = default;
    //! Construct with the \p weights vector.
    explicit CExponent(TDoubleVec&& weights);
    //! Construct with a weight vector of \p size with all entries equal to \p weight.
    CExponent(std::size_t size, double weight);
    void addToJsonStream(TGenericLineWriter& writer) const override;
    const std::string& stringType() const override;

private:
    TDoubleVec m_Weights;
};

//! List of support numeric relationships. It's only "<" at the moment.
enum ENumericRelationship { E_LT };

class API_EXPORT CTrainedModel : public CSerializableToJsonStream {
public:
    using TDoubleVec = std::vector<double>;
    using TStringVec = std::vector<std::string>;
    using TOptionalDoubleVec = boost::optional<TDoubleVec>;
    using TOptionalStringVec = boost::optional<TStringVec>;

    enum ETargetType { E_Classification, E_Regression };

    //! \brief Provides feature names.
    //!
    //! DESCRIPTION:\n
    //! Trained model features include any input feature and any synthetic features
    //! which training adds, which include, for example, category encodings. Any code
    //! which references trained model features by name needs to use consistent naming.
    //! We standardise by using this class to encapsulate naming trained model features
    //! from the input feature names, the category names and the type of operation used
    //! to generate the feature.
    class CFeatureNameProvider {
    public:
        using TStrVec = std::vector<std::string>;
        using TStrVecVec = std::vector<TStrVec>;

    public:
        CFeatureNameProvider(TStrVec fieldNames, TStrVecVec categoryNames);

        const std::string& fieldName(std::size_t inputColumnIndex) const;
        const std::string& category(std::size_t inputColumnIndex, std::size_t hotCategory) const;
        std::string identityEncodingName(std::size_t inputColumnIndex) const;
        std::string oneHotEncodingName(std::size_t inputColumnIndex,
                                       std::size_t hotCategory) const;
        std::string targetMeanEncodingName(std::size_t inputColumnIndex) const;
        std::string frequencyEncodingName(std::size_t inputColumnIndex) const;

    private:
        TStrVec m_FieldNames;
        TStrVecVec m_CategoryNames;
    };

    //! \brief A measure of the model complexity.
    class CSizeInfo : public CSerializableToJsonDocument {
    public:
        static const std::string JSON_NUM_CLASSES_TAG;
        static const std::string JSON_NUM_CLASSIFICATION_WEIGHTS_TAG;

    public:
        explicit CSizeInfo(const CTrainedModel& trainedModel);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        //! \return Expected number of operation for the model evaluation.
        virtual std::size_t numOperations() const = 0;

    private:
        const CTrainedModel& m_TrainedModel;
    };
    using TSizeInfoUPtr = std::unique_ptr<CSizeInfo>;

    static const std::string JSON_CLASSIFICATION_LABELS_TAG;
    static const std::string JSON_CLASSIFICATION_WEIGHTS_TAG;
    static const std::string JSON_FEATURE_NAMES_TAG;
    static const std::string JSON_TARGET_TYPE_CLASSIFICATION;
    static const std::string JSON_TARGET_TYPE_REGRESSION;
    static const std::string JSON_TARGET_TYPE_TAG;

public:
    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Names of the features used by the model.
    virtual const TStringVec& featureNames() const;
    virtual TStringVec& featureNames();
    //! Names of the features used by the model.
    virtual void featureNames(TStringVec featureNames);
    //! Sets target type (regression or classification).
    virtual void targetType(ETargetType targetType);
    //! Returns target type (regression or classification).
    virtual ETargetType targetType() const;
    //! Adjust the feature names, e.g. to exclude not used feature names like the target column.
    virtual const TStringVec& removeUnusedFeatures() = 0;
    //! Set the labels to use for each class.
    virtual void classificationLabels(const TStringVec& classificationLabels);
    //! Get the labels to use for each class.
    virtual const TOptionalStringVec& classificationLabels() const;
    //! Set weights by which to multiply classes when doing label assignment.
    virtual void classificationWeights(TDoubleVec classificationWeights);
    //! Get weights by which to multiply classes when doing label assignment.
    virtual const TOptionalDoubleVec& classificationWeights() const;
    //! Get the object for model size with information for estimation.
    virtual TSizeInfoUPtr sizeInfo() const = 0;

private:
    TStringVec m_FeatureNames;
    ETargetType m_TargetType;
    TOptionalStringVec m_ClassificationLabels;
    TOptionalDoubleVec m_ClassificationWeights;
};

//! Classification and regression trees.
class API_EXPORT CTree final : public CTrainedModel {
public:
    class CTreeNode : public CSerializableToJsonStream {
    public:
        using TDoubleVec = std::vector<double>;
        using TNodeIndex = std::uint32_t;
        using TOptionalNodeIndex = boost::optional<TNodeIndex>;
        using TOptionalDouble = boost::optional<double>;

        static const std::string JSON_DECISION_TYPE_TAG;
        static const std::string JSON_DEFAULT_LEFT_TAG;
        static const std::string JSON_LEAF_VALUE_TAG;
        static const std::string JSON_LEFT_CHILD_TAG;
        static const std::string JSON_LT;
        static const std::string JSON_NODE_INDEX_TAG;
        static const std::string JSON_NUMBER_SAMPLES_TAG;
        static const std::string JSON_RIGHT_CHILD_TAG;
        static const std::string JSON_SPLIT_FEATURE_TAG;
        static const std::string JSON_SPLIT_GAIN_TAG;
        static const std::string JSON_THRESHOLD_TAG;

    public:
        CTreeNode(TNodeIndex nodeIndex,
                  double threshold,
                  bool defaultLeft,
                  TDoubleVec leafValue,
                  std::size_t splitFeature,
                  std::size_t numberSamples,
                  const TOptionalNodeIndex& leftChild,
                  const TOptionalNodeIndex& rightChild,
                  const TOptionalDouble& splitGain);

        void addToJsonStream(TGenericLineWriter& writer) const override;
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
        std::size_t m_NumberSamples;
        double m_Threshold;
        TDoubleVec m_LeafValue;
        TOptionalDouble m_SplitGain;
    };

    class CSizeInfo : public CTrainedModel::CSizeInfo {
    public:
        static const std::string JSON_NUM_NODES_TAG;
        static const std::string JSON_NUM_LEAVES_TAG;

    public:
        explicit CSizeInfo(const CTree& tree);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        std::size_t numOperations() const override;

    private:
        const CTree& m_Tree;
    };

    using TTreeNodeVec = std::vector<CTreeNode>;

    static const std::string JSON_TREE_TAG;
    static const std::string JSON_TREE_STRUCTURE_TAG;

public:
    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Total number of tree nodes.
    std::size_t size() const;
    const TStringVec& removeUnusedFeatures() override;
    TTreeNodeVec& treeStructure();
    //! Get the object for model size with information for estimation.
    TSizeInfoUPtr sizeInfo() const override;

private:
    TTreeNodeVec m_TreeStructure;
};

//! Ensemble of a collection of trained models
class API_EXPORT CEnsemble final : public CTrainedModel {
public:
    using TAggregateOutputUPtr = std::unique_ptr<CAggregateOutput>;
    using TTrainedModelUPtr = std::unique_ptr<CTrainedModel>;
    using TTrainedModelUPtrVec = std::vector<TTrainedModelUPtr>;

    class CSizeInfo : public CTrainedModel::CSizeInfo {
    public:
        static const std::string JSON_FEATURE_NAME_LENGTHS_TAG;
        static const std::string JSON_NUM_OPERATIONS_TAG;
        static const std::string JSON_NUM_OUTPUT_PROCESSOR_WEIGHTS_TAG;
        static const std::string JSON_TREE_SIZES_TAG;

    public:
        explicit CSizeInfo(const CEnsemble& ensemble);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        std::size_t numOperations() const override;

    private:
        const CEnsemble* m_Ensemble;
    };

    static const std::string JSON_AGGREGATE_OUTPUT_TAG;
    static const std::string JSON_ENSEMBLE_TAG;
    static const std::string JSON_TRAINED_MODELS_TAG;

public:
    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Aggregation mechanism for the output from individual models.
    void aggregateOutput(TAggregateOutputUPtr&& aggregateOutput);
    const TAggregateOutputUPtr& aggregateOutput() const;
    void featureNames(TStringVec featureNames) override;
    //! List of trained models withing this ensemble.
    TTrainedModelUPtrVec& trainedModels();
    //! Number of models in the ensemble.
    std::size_t size() const;
    const TStringVec& removeUnusedFeatures() override;
    void targetType(ETargetType targetType) override;
    //! Set the labels to use for each class.
    void classificationLabels(const TStringVec& classificationLabels) override;
    //! Set weights by which to multiply classes when doing label assignment.
    void classificationWeights(TDoubleVec classificationWeights) override;
    //! Get the object for model size with information for estimation.
    TSizeInfoUPtr sizeInfo() const override;
    using CTrainedModel::classificationLabels;
    using CTrainedModel::classificationWeights;
    using CTrainedModel::featureNames;
    using CTrainedModel::targetType;

private:
    TTrainedModelUPtrVec m_TrainedModels;
    TAggregateOutputUPtr m_AggregateOutput;
};

class API_EXPORT CEncoding : public CSerializableToJsonStream {
public:
    class CSizeInfo : public CSerializableToJsonDocument {
    public:
        static const std::string JSON_FEATURE_NAME_LENGTH_TAG;
        static const std::string JSON_FIELD_VALUE_LENGTHS_TAG;
        static const std::string JSON_FIELD_LENGTH_TAG;

    public:
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        virtual const std::string& typeString() const = 0;
        const CEncoding* encoding() const;

    protected:
        using TSizeVec = std::vector<std::size_t>;

    protected:
        explicit CSizeInfo(const CEncoding* encoding);

    private:
        const CEncoding* m_Encoding;
    };
    using TSizeInfoUPtr = std::unique_ptr<CSizeInfo>;

    static const std::string JSON_FIELD_TAG;
    static const std::string JSON_FEATURE_NAME_TAG;

public:
    ~CEncoding() override = default;
    explicit CEncoding(std::string field);
    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Input field name. Must be defined in the input section.
    void field(const std::string& field);
    const std::string& field() const;
    //! Encoding type as string.
    virtual const std::string& typeString() const = 0;
    //! Get the object for model size with information for estimation.
    virtual TSizeInfoUPtr sizeInfo() const = 0;

private:
    //! Input field name. Must be defined in the input section.
    std::string m_Field;
};

class API_EXPORT CCustomEncoding : public CSerializableToJsonStream {};

//! \brief Mapping from categorical columns to numerical values related to categorical value distribution.
class API_EXPORT CFrequencyEncoding final : public CEncoding {
public:
    class CSizeInfo final : public CEncoding::CSizeInfo {
    public:
        explicit CSizeInfo(const CFrequencyEncoding& encoding);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        const std::string& typeString() const override;

    private:
        const CFrequencyEncoding& m_Encoding;
    };
    using TStringDoubleUMap = const boost::unordered_map<std::string, double>;

    static const std::string JSON_FREQUENCY_MAP_TAG;
    static const std::string JSON_FREQUENCY_ENCODING_TAG;

public:
    ~CFrequencyEncoding() override = default;
    CFrequencyEncoding(const std::string& field, std::string featureName, TStringDoubleUMap frequencyMap);

    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Feature name after pre-processing.
    const std::string& featureName() const;
    //! Map from the category names to the frequency values.
    const TStringDoubleUMap& frequencyMap() const;
    const std::string& typeString() const override;
    //! Get the object for model size with information for estimation.
    TSizeInfoUPtr sizeInfo() const override;

private:
    std::string m_FeatureName;
    TStringDoubleUMap m_FrequencyMap;
};

//! \brief Application of the one-hot encoding function on a single column.
class API_EXPORT COneHotEncoding final : public CEncoding {
public:
    class CSizeInfo final : public CEncoding::CSizeInfo {
    public:
        static const std::string JSON_FEATURE_NAME_LENGTHS_TAG;

    public:
        explicit CSizeInfo(const COneHotEncoding& encoding);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        const std::string& typeString() const override;

    private:
        const COneHotEncoding& m_Encoding;
    };
    using TStringStringUMap = std::map<std::string, std::string>;

    static const std::string JSON_HOT_MAP_TAG;
    static const std::string JSON_ONE_HOT_ENCODING_TAG;

public:
    ~COneHotEncoding() override = default;
    COneHotEncoding(const std::string& field, TStringStringUMap hotMap);
    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Map from the category names of the original field to the new field names.
    const TStringStringUMap& hotMap() const;
    TStringStringUMap& hotMap();
    const std::string& typeString() const override;
    //! Get the object for model size with information for estimation.
    TSizeInfoUPtr sizeInfo() const override;

private:
    TStringStringUMap m_HotMap;
};

//! \brief Mapping from categorical columns to numerical values related to the target value.
class API_EXPORT CTargetMeanEncoding final : public CEncoding {
public:
    class CSizeInfo final : public CEncoding::CSizeInfo {
    public:
        explicit CSizeInfo(const CTargetMeanEncoding& encoding);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        const std::string& typeString() const override;

    private:
        const CTargetMeanEncoding& m_Encoding;
    };
    using TStringDoubleUMap = boost::unordered_map<std::string, double>;

    static const std::string JSON_TARGET_MAP_TAG;
    static const std::string JSON_TARGET_MEAN_ENCODING_TAG;
    static const std::string JSON_DEFAULT_VALUE_TAG;

public:
    ~CTargetMeanEncoding() override = default;
    CTargetMeanEncoding(const std::string& field,
                        double defaultValue,
                        std::string featureName,
                        TStringDoubleUMap&& targetMap);

    void addToJsonStream(TGenericLineWriter& writer) const override;
    //! Value for categories that have not been seen before.
    double defaultValue() const;
    //! Feature name after pre-processing.
    const std::string& featureName() const;
    //! Map from the category names to the target values.
    const TStringDoubleUMap& targetMap() const;
    const std::string& typeString() const override;
    //! Get the object for model size with information for estimation.
    TSizeInfoUPtr sizeInfo() const override;

private:
    double m_DefaultValue;
    std::string m_FeatureName;
    TStringDoubleUMap m_TargetMap;
};

//! \brief A JSON blob defining a custom encoding or an array of custom encodings.
class API_EXPORT COpaqueEncoding final : public CCustomEncoding {
public:
    explicit COpaqueEncoding(const rapidjson::Document& object);

    void addToJsonStream(TGenericLineWriter& writer) const override;

private:
    rapidjson::Document m_Object;
};

//! \brief Technical details required for model evaluation.
class API_EXPORT CInferenceModelDefinition : public CSerializableToJsonDocumentCompressed {
public:
    using TApiEncodingUPtr = std::unique_ptr<api::CEncoding>;
    using TApiEncodingUPtrVec = std::vector<TApiEncodingUPtr>;
    using TApiCustomEncodingUPtr = std::unique_ptr<api::CCustomEncoding>;
    using TApiCustomEncodingUPtrVec = std::vector<TApiCustomEncodingUPtr>;
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;
    using TSizeStringUMap = boost::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;
    using TStringSizeUMap = boost::unordered_map<std::string, std::size_t>;
    using TStringSizeUMapVec = std::vector<TStringSizeUMap>;
    using TStringVec = std::vector<std::string>;
    using TTrainedModelUPtr = CEnsemble::TTrainedModelUPtr;

    class API_EXPORT CSizeInfo final : public CSerializableToJsonDocument {
    public:
        static const std::string JSON_ENSEMBLE_MODEL_SIZE_TAG;
        static const std::string JSON_MODEL_SIZE_INFO_TAG;
        static const std::string JSON_TRAINED_MODEL_SIZE_TAG;

    public:
        explicit CSizeInfo(const CInferenceModelDefinition& definition);
        void addToJsonDocument(rapidjson::Value& parentObject,
                               TRapidJsonWriter& writer) const override;
        const std::string& typeString() const;
        std::string jsonString();

    private:
        const CInferenceModelDefinition& m_Definition;
    };

    using TSizeInfoUPtr = std::unique_ptr<CSizeInfo>;

    static const std::string JSON_COMPRESSED_INFERENCE_MODEL_TAG;
    static const std::string JSON_DEFINITION_TAG;
    static const std::string JSON_PREPROCESSORS_TAG;
    static const std::string JSON_TRAINED_MODEL_TAG;

public:
    TApiEncodingUPtrVec& preprocessors();
    const TApiEncodingUPtrVec& preprocessors() const { return m_Preprocessors; }
    TApiCustomEncodingUPtrVec& customPreprocessors();
    const TApiCustomEncodingUPtrVec& customPreprocessors() const {
        return m_CustomPreprocessors;
    }
    void trainedModel(TTrainedModelUPtr&& trainedModel);
    TTrainedModelUPtr& trainedModel();
    const TTrainedModelUPtr& trainedModel() const;
    void addToJsonStream(TGenericLineWriter& writer) const final;
    void addToDocumentCompressed(TRapidJsonWriter& writer) const final;
    std::string jsonString() const;
    void fieldNames(TStringVec&& fieldNames);
    const TStringVec& fieldNames() const;
    const std::string& typeString() const;
    std::size_t dependentVariableColumnIndex() const;
    void dependentVariableColumnIndex(size_t dependentVariableColumnIndex);
    //! Get the object for model size with information for estimation.
    TSizeInfoUPtr sizeInfo() const;

private:
    //! Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
    TApiEncodingUPtrVec m_Preprocessors;
    //! Optional step for custom pre-processing data, supplied from analytics config
    TApiCustomEncodingUPtrVec m_CustomPreprocessors;
    //! Details of the model evaluation step with a trained_model.
    TTrainedModelUPtr m_TrainedModel;
    TStringVec m_FieldNames;
    std::string m_TypeString;
    std::size_t m_DependentVariableColumnIndex;
};
}
}

#endif //INCLUDED_ml_api_CInferenceModelDefinition_h
