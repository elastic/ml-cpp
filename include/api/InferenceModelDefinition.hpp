//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     InferenceModelDefinition data = nlohmann::json::parse(jsonString);

#pragma once

#include <json/json.hpp>

#ifndef NLOHMANN_OPT_HELPER
#define NLOHMANN_OPT_HELPER
namespace nlohmann {
template <typename T>
struct adl_serializer<std::shared_ptr<T>> {
    static void to_json(json & j, const std::shared_ptr<T> & opt) {
        if (!opt) j = nullptr; else j = *opt;
    }

    static std::shared_ptr<T> from_json(const json & j) {
        if (j.is_null()) return std::unique_ptr<T>(); else return std::unique_ptr<T>(new T(j.get<T>()));
    }
};
}
#endif

namespace ml {
namespace api {
namespace inference_model {

using nlohmann::json;

inline json get_untyped(const json &j, const char *property) {
    if (j.find(property) != j.end()) {
        return j.at(property).get<json>();
    }
    return json();
}

inline json get_untyped(const json &j, std::string property) {
    return get_untyped(j, property.data());
}

template<typename T>
inline std::shared_ptr<T> get_optional(const json &j, const char *property) {
    if (j.find(property) != j.end()) {
        return j.at(property).get<std::shared_ptr<T>>();
    }
    return std::shared_ptr<T>();
}

template<typename T>
inline std::shared_ptr<T> get_optional(const json &j, std::string property) {
    return get_optional<T>(j, property.data());
}

/**
 * Allows to used (weighted) majority vote for classification.
 */
struct WeightedMode {
    std::vector<double> weights;
};

struct WeightedSum {
    std::vector<double> weights;
};

struct AggregateOutput {
    std::shared_ptr<WeightedSum> weightedSum;
    /**
     * Allows to used (weighted) majority vote for classification.
     */
    std::shared_ptr<WeightedMode> weightedMode;
};

enum class NumericRelationship : int {
    Empty
};

struct TreeNode {
    NumericRelationship decisionType;
    bool defaultLeft;
    std::shared_ptr<TreeNode> leftChild;
    std::shared_ptr<TreeNode> rightChild;
    int64_t splitFeature;
    std::shared_ptr<double> splitGain;
    int64_t splitIndex;
    double threshold;
};

struct Tree {
    std::shared_ptr<std::vector<std::string>> featureNames;
    TreeNode treeStructure;
};

/**
 * Details of the model evaluation step.
 */
struct Evaluation {
    std::shared_ptr<std::vector<std::string>> featureNames;
    std::shared_ptr<TreeNode> treeStructure;
    std::shared_ptr<AggregateOutput> aggregateOutput;
    std::shared_ptr<std::vector<Tree>> models;
};

/**
 * Information related to the input.
 *
 * Information related to the input
 */
struct Input {
    /**
     * List of the column names.
     */
    std::shared_ptr<std::vector<std::string>> columns;
};

/**
 * Mapping from categorical columns to numerical values related to categorical value
 * distribution
 */
struct FrequencyEncoding {
    /**
     * Feature name after pre-processing
     */
    std::string featureName;
    /**
     * Input field name
     */
    std::string field;
    /**
     * Map from the category names to the frequency values.
     */
    std::map<std::string, double> frequencyMap;
};

/**
 * Application of the one-hot encoding function on a single column.
 */
struct OneHotEncoding {
    /**
     * Input field name. Must be defined in the input section.
     */
    std::string field;
    /**
     * Map from the category names of the original field to the new field names.
     */
    std::map<std::string, std::string> hotMap;
};

/**
 * Mapping from categorical columns to numerical values related to the target value
 */
struct TargetMeanEncoding {
    /**
     * Feature name after pre-processing
     */
    std::string featureName;
    /**
     * Input field name
     */
    std::string field;
    /**
     * Map from the category names to the target values.
     */
    std::map<std::string, double> targetMap;
};

/**
 * Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
 *
 * Steps for data pre-processing.
 */
struct Preprocessing {
    /**
     * Application of the one-hot encoding function on a single column.
     */
    std::shared_ptr<OneHotEncoding> oneHotEncoding;
    /**
     * Mapping from categorical columns to numerical values related to the target value
     */
    std::shared_ptr<TargetMeanEncoding> targetMeanEncoding;
    /**
     * Mapping from categorical columns to numerical values related to categorical value
     * distribution
     */
    std::shared_ptr<FrequencyEncoding> frequencyEncoding;
};

/**
 * Technical details required for model evaluation.
 */
struct InferenceModelDefinition {
    /**
     * Details of the model evaluation step.
     */
    Evaluation evaluation;
    /**
     * Information related to the input.
     */
    Input input;
    /**
     * Optional step for pre-processing data, e.g. vector embedding, one-hot-encoding, etc.
     */
    std::shared_ptr<std::vector<Preprocessing>> preprocessing;
};
}
}
}

namespace nlohmann {
void from_json(const json & j, ml::api::inference_model::WeightedMode & x);
void to_json(json & j, const ml::api::inference_model::WeightedMode & x);

void from_json(const json & j, ml::api::inference_model::WeightedSum & x);
void to_json(json & j, const ml::api::inference_model::WeightedSum & x);

void from_json(const json & j, ml::api::inference_model::AggregateOutput & x);
void to_json(json & j, const ml::api::inference_model::AggregateOutput & x);

void from_json(const json & j, ml::api::inference_model::TreeNode & x);
void to_json(json & j, const ml::api::inference_model::TreeNode & x);

void from_json(const json & j, ml::api::inference_model::Tree & x);
void to_json(json & j, const ml::api::inference_model::Tree & x);

void from_json(const json & j, ml::api::inference_model::Evaluation & x);
void to_json(json & j, const ml::api::inference_model::Evaluation & x);

void from_json(const json & j, ml::api::inference_model::Input & x);
void to_json(json & j, const ml::api::inference_model::Input & x);

void from_json(const json & j, ml::api::inference_model::FrequencyEncoding & x);
void to_json(json & j, const ml::api::inference_model::FrequencyEncoding & x);

void from_json(const json & j, ml::api::inference_model::OneHotEncoding & x);
void to_json(json & j, const ml::api::inference_model::OneHotEncoding & x);

void from_json(const json & j, ml::api::inference_model::TargetMeanEncoding & x);
void to_json(json & j, const ml::api::inference_model::TargetMeanEncoding & x);

void from_json(const json & j, ml::api::inference_model::Preprocessing & x);
void to_json(json & j, const ml::api::inference_model::Preprocessing & x);

void from_json(const json & j, ml::api::inference_model::InferenceModelDefinition & x);
void to_json(json & j, const ml::api::inference_model::InferenceModelDefinition & x);

void from_json(const json & j, ml::api::inference_model::NumericRelationship & x);
void to_json(json & j, const ml::api::inference_model::NumericRelationship & x);

inline void from_json(const json & j, ml::api::inference_model::WeightedMode& x) {
    x.weights = j.at("weights").get<std::vector<double>>();
}

inline void to_json(json & j, const ml::api::inference_model::WeightedMode & x) {
    j = json::object();
    j["weights"] = x.weights;
}

inline void from_json(const json & j, ml::api::inference_model::WeightedSum& x) {
    x.weights = j.at("weights").get<std::vector<double>>();
}

inline void to_json(json & j, const ml::api::inference_model::WeightedSum & x) {
    j = json::object();
    j["weights"] = x.weights;
}

inline void from_json(const json & j, ml::api::inference_model::AggregateOutput& x) {
    x.weightedSum = ml::api::inference_model::get_optional<ml::api::inference_model::WeightedSum>(j, "weighted_sum");
    x.weightedMode = ml::api::inference_model::get_optional<ml::api::inference_model::WeightedMode>(j, "weighted_mode");
}

inline void to_json(json & j, const ml::api::inference_model::AggregateOutput & x) {
    j = json::object();
    j["weighted_sum"] = x.weightedSum;
    j["weighted_mode"] = x.weightedMode;
}

inline void from_json(const json & j, ml::api::inference_model::TreeNode& x) {
    x.decisionType = j.at("decision_type").get<ml::api::inference_model::NumericRelationship>();
    x.defaultLeft = j.at("default_left").get<bool>();
    x.leftChild = ml::api::inference_model::get_optional<ml::api::inference_model::TreeNode>(j, "left_child");
    x.rightChild = ml::api::inference_model::get_optional<ml::api::inference_model::TreeNode>(j, "right_child");
    x.splitFeature = j.at("split_feature").get<int64_t>();
    x.splitGain = ml::api::inference_model::get_optional<double>(j, "split_gain");
    x.splitIndex = j.at("split_index").get<int64_t>();
    x.threshold = j.at("threshold").get<double>();
}

inline void to_json(json & j, const ml::api::inference_model::TreeNode & x) {
    j = json::object();
    j["decision_type"] = x.decisionType;
    j["default_left"] = x.defaultLeft;
    j["left_child"] = x.leftChild;
    j["right_child"] = x.rightChild;
    j["split_feature"] = x.splitFeature;
    j["split_gain"] = x.splitGain;
    j["split_index"] = x.splitIndex;
    j["threshold"] = x.threshold;
}

inline void from_json(const json & j, ml::api::inference_model::Tree& x) {
    x.featureNames = ml::api::inference_model::get_optional<std::vector<std::string>>(j, "feature_names");
    x.treeStructure = j.at("tree_structure").get<ml::api::inference_model::TreeNode>();
}

inline void to_json(json & j, const ml::api::inference_model::Tree & x) {
    j = json::object();
    j["feature_names"] = x.featureNames;
    j["tree_structure"] = x.treeStructure;
}

inline void from_json(const json & j, ml::api::inference_model::Evaluation& x) {
    x.featureNames = ml::api::inference_model::get_optional<std::vector<std::string>>(j, "feature_names");
    x.treeStructure = ml::api::inference_model::get_optional<ml::api::inference_model::TreeNode>(j, "tree_structure");
    x.aggregateOutput = ml::api::inference_model::get_optional<ml::api::inference_model::AggregateOutput>(j, "aggregate_output");
    x.models = ml::api::inference_model::get_optional<std::vector<ml::api::inference_model::Tree>>(j, "models");
}

inline void to_json(json & j, const ml::api::inference_model::Evaluation & x) {
    j = json::object();
    j["feature_names"] = x.featureNames;
    j["tree_structure"] = x.treeStructure;
    j["aggregate_output"] = x.aggregateOutput;
    j["models"] = x.models;
}

inline void from_json(const json & j, ml::api::inference_model::Input& x) {
    x.columns = ml::api::inference_model::get_optional<std::vector<std::string>>(j, "columns");
}

inline void to_json(json & j, const ml::api::inference_model::Input & x) {
    j = json::object();
    j["columns"] = x.columns;
}

inline void from_json(const json & j, ml::api::inference_model::FrequencyEncoding& x) {
    x.featureName = j.at("feature_name").get<std::string>();
    x.field = j.at("field").get<std::string>();
    x.frequencyMap = j.at("frequency_map").get<std::map<std::string, double>>();
}

inline void to_json(json & j, const ml::api::inference_model::FrequencyEncoding & x) {
    j = json::object();
    j["feature_name"] = x.featureName;
    j["field"] = x.field;
    j["frequency_map"] = x.frequencyMap;
}

inline void from_json(const json & j, ml::api::inference_model::OneHotEncoding& x) {
    x.field = j.at("field").get<std::string>();
    x.hotMap = j.at("hot_map").get<std::map<std::string, std::string>>();
}

inline void to_json(json & j, const ml::api::inference_model::OneHotEncoding & x) {
    j = json::object();
    j["field"] = x.field;
    j["hot_map"] = x.hotMap;
}

inline void from_json(const json & j, ml::api::inference_model::TargetMeanEncoding& x) {
    x.featureName = j.at("feature_name").get<std::string>();
    x.field = j.at("field").get<std::string>();
    x.targetMap = j.at("target_map").get<std::map<std::string, double>>();
}

inline void to_json(json & j, const ml::api::inference_model::TargetMeanEncoding & x) {
    j = json::object();
    j["feature_name"] = x.featureName;
    j["field"] = x.field;
    j["target_map"] = x.targetMap;
}

inline void from_json(const json & j, ml::api::inference_model::Preprocessing& x) {
    x.oneHotEncoding = ml::api::inference_model::get_optional<ml::api::inference_model::OneHotEncoding>(j, "one_hot_encoding");
    x.targetMeanEncoding = ml::api::inference_model::get_optional<ml::api::inference_model::TargetMeanEncoding>(j, "target_mean_encoding");
    x.frequencyEncoding = ml::api::inference_model::get_optional<ml::api::inference_model::FrequencyEncoding>(j, "frequency_encoding");
}

inline void to_json(json & j, const ml::api::inference_model::Preprocessing & x) {
    j = json::object();
    j["one_hot_encoding"] = x.oneHotEncoding;
    j["target_mean_encoding"] = x.targetMeanEncoding;
    j["frequency_encoding"] = x.frequencyEncoding;
}

inline void from_json(const json & j, ml::api::inference_model::InferenceModelDefinition& x) {
    x.evaluation = j.at("evaluation").get<ml::api::inference_model::Evaluation>();
    x.input = j.at("input").get<ml::api::inference_model::Input>();
    x.preprocessing = ml::api::inference_model::get_optional<std::vector<ml::api::inference_model::Preprocessing>>(j, "preprocessing");
}

inline void to_json(json & j, const ml::api::inference_model::InferenceModelDefinition & x) {
    j = json::object();
    j["evaluation"] = x.evaluation;
    j["input"] = x.input;
    j["preprocessing"] = x.preprocessing;
}

inline void from_json(const json & j, ml::api::inference_model::NumericRelationship & x) {
    if (j == "<=") x = ml::api::inference_model::NumericRelationship::Empty;
    else throw "Input JSON does not conform to schema";
}

inline void to_json(json & j, const ml::api::inference_model::NumericRelationship & x) {
    switch (x) {
        case ml::api::inference_model::NumericRelationship::Empty: j = "<="; break;
        default: throw "This should not happen";
    }
}
}
